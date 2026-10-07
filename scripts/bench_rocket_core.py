#!/usr/bin/env python3
"""Run TensorLUT throughput on the RTeAAL Rocket core LUT/DFF netlist."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtlgemm.frontend import parse_netlist
from rtlgemm.ir import build_plan
from rtlgemm.runtime.simulate import CompiledSim


def parse_batches(text: str) -> list[int]:
    return [int(x, 0) for x in text.split(",") if x]


def run_batch(plan, nl, batch: int, cycles: int, runs: int, device: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, (batch, nl.n_state), dtype=np.uint8)
    u = rng.integers(0, 2, (cycles, batch, nl.n_input), dtype=np.uint8)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    sim = CompiledSim.build(plan, batch, cycles, device, use_cuda_graph=True, backend="auto")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    build_s = time.perf_counter() - t0

    rows = []
    for i in range(runs):
        t0 = time.perf_counter()
        sim.run(x0, u)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        rows.append(
            {
                "run": i + 1,
                "wall_s": wall_s,
                "cycle_stimuli_per_s": batch * cycles / wall_s,
            }
        )

    rates = [r["cycle_stimuli_per_s"] for r in rows]
    mean_rate = statistics.mean(rates)
    return {
        "batch": batch,
        "cycles": cycles,
        "build_s": build_s,
        "runs": rows,
        "cycle_stimuli_per_s_mean": mean_rate,
        "cycle_stimuli_per_s_stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "per_seed_hz_mean": mean_rate / batch,
        "cuda_memory_allocated": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
        "cuda_memory_allocated_gb": (
            torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0.0
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-netlist", default="build/large_rtl/rocket_Rocket_lut.json")
    ap.add_argument("--top", default="Rocket")
    ap.add_argument("--batches", default="4096,16384,65536,131072")
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    nl = parse_netlist(args.json_netlist, args.top)
    plan = build_plan(nl)
    results = [
        run_batch(plan, nl, batch, args.cycles, args.runs, args.device, batch)
        for batch in parse_batches(args.batches)
    ]
    print(
        json.dumps(
            {
                "mode": plan.mode,
                "top": args.top,
                "n_state": nl.n_state,
                "n_input": nl.n_input,
                "n_output_bits": sum(len(bits) for _, bits in nl.outputs),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
