"""Throughput evaluation across the routed tensor-core backends.

Reports cyc*batch/s for the best GPU backend vs the numpy interpreter (CPU baseline;
not an optimized simulator — see scripts/bench_verilator.sh for Verilator).
"""
import argparse
import time
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from rtlgemm.frontend import synth, parse_netlist
from rtlgemm.ir import compile_design
from rtlgemm.reference import simulate_netlist, state_from_port
from rtlgemm.runtime.simulate import CompiledSim

DESIGNS = [
    ("benchmarks/lfsr16_free.v", "lfsr16_free", "state"),
    ("benchmarks/crc8_serial.v", "crc8_serial", "crc"),
    ("benchmarks/nfsr16.v",      "nfsr16",      "s"),
    ("benchmarks/lfsr8.v",       "lfsr8",       "state"),
    ("benchmarks/counter8.v",    "counter8",    "cnt"),
]


def time_runs(sim, x0, u, runs):
    sim.run(x0, u); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        sim.run(x0, u)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1 << 16)
    ap.add_argument("--cycles", type=int, default=512)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--cpu-batch", type=int, default=256)
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "triton", "gemm", "gather"],
                    help="CompiledSim backend selector")
    ap.add_argument("--designs", default=",".join(top for _, top, _ in DESIGNS),
                    help="Comma-separated design tops to run")
    args = ap.parse_args()
    B, C, RUNS = args.batch, args.cycles, args.runs
    selected = {x.strip() for x in args.designs.split(",") if x.strip()}
    print(f"{'design':<14}{'mode':<12}{'best cyc*b/s':>15}{'CPU cyc*b/s':>14}{'vs CPU':>9}")
    for src, top, port in DESIGNS:
        if top not in selected:
            continue
        plan = compile_design(src, top)
        rng = np.random.default_rng(0)
        seeds = rng.integers(1, 1 << plan.nl.n_state, size=B)
        x0 = torch.as_tensor(state_from_port(plan.nl, port, seeds), dtype=torch.int8, device="cuda")
        u = (torch.randint(0, 2, (C, B, plan.nl.n_input), dtype=torch.int8, device="cuda")
             if plan.nl.n_input > 0 else torch.zeros((C, B, 0), dtype=torch.int8, device="cuda"))
        sim = CompiledSim.build(plan, B, C, "cuda", backend=args.backend)
        gpu_tp = B * C / time_runs(sim, x0, u, RUNS)

        # CPU baseline: numpy interpreter on the LUT netlist of the same design
        nl = parse_netlist(synth(src, top).json_path)
        Bc = args.cpu_batch
        x0c = state_from_port(nl, port, seeds[:Bc])
        uc = (np.random.default_rng(1).integers(0, 2, (C, Bc, nl.n_input)).astype(np.uint8)
              if nl.n_input > 0 else np.zeros((C, Bc, 0), np.uint8))
        t0 = time.perf_counter(); simulate_netlist(nl, x0c, uc, C); cpu_s = time.perf_counter() - t0
        cpu_tp = Bc * C / cpu_s

        print(f"{top:<14}{plan.mode:<12}{gpu_tp:>15.3e}{cpu_tp:>14.3e}{gpu_tp/cpu_tp:>9.0f}")


if __name__ == "__main__":
    main()
