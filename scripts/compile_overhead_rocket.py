#!/usr/bin/env python3
"""Per-stage compile overhead for the Rocket core: TensorLUT flow vs. Verilator build.

TensorLUT stages: Yosys/ABC (RTL -> LUT/DFF JSON), netlist parse, levelized plan, ANF tensor
construction (B-independent), CUDA-graph capture for one (batch, cycles) shape, and stimulus upload.
Verilator stages: LUT JSON -> Verilog round-trip and `verilator --build -O3`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtlgemm.frontend import parse_netlist
from rtlgemm.ir import build_plan
from rtlgemm.runtime.b1_sim import B1LayerPacked8GraphSim, B1LayerPacked8Sim
from scripts import bench_rocket_verilator as brv
from scripts import rocket_lut_stats as rls


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=rls.DEFAULT_SRC)
    ap.add_argument("--top", default="Rocket")
    ap.add_argument("--suite", type=Path, default=brv.DEFAULT_SUITE)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=1 << 18)
    ap.add_argument("--cycles", type=int, default=64)
    ap.add_argument("--chunk-outputs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--skip-verilator", action="store_true")
    ap.add_argument("--output-json", type=Path)
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    yosys = brv._tool("yosys", args.suite)
    json_path, yosys_s = timed(lambda: rls.yosys_lut(args.src, args.top, args.workdir, yosys, 6, True))
    nl, parse_s = timed(lambda: parse_netlist(str(json_path), args.top))
    plan, plan_s = timed(lambda: build_plan(nl))

    def tensors():
        sim = B1LayerPacked8Sim(plan, "cuda", args.chunk_outputs)
        torch.cuda.synchronize()
        return sim
    _, tensor_s = timed(tensors)

    def graph_sim():
        sim = B1LayerPacked8GraphSim(plan, args.batch, args.cycles, "cuda",
                                     chunk_outputs=args.chunk_outputs, use_cuda_graph=True)
        torch.cuda.synchronize()
        return sim
    sim, graph_total_s = timed(graph_sim)
    capture_s = max(graph_total_s - tensor_s, 0.0)

    rng = np.random.default_rng(args.seed)
    u = rng.integers(0, 2, (args.cycles, args.batch, nl.n_input), dtype=np.uint8)

    def load():
        sim.load_inputs(u)
        torch.cuda.synchronize()
    _, load_s = timed(load)

    result = {
        "top": args.top,
        "batch": args.batch,
        "cycles": args.cycles,
        "host_logical_cpus": os.cpu_count(),
        "tensorlut": {
            "yosys_abc_lut6_s": yosys_s,
            "parse_netlist_s": parse_s,
            "build_plan_s": plan_s,
            "anf_tensor_build_s": tensor_s,
            "cuda_graph_capture_s": capture_s,
            "stimulus_upload_s": load_s,
            "n_lut": len(nl.luts),
            "n_dff": nl.n_state,
            "layers": len(plan.layers),
        },
    }

    if not args.skip_verilator:
        vdir = args.workdir / "verilator"
        shutil.rmtree(vdir, ignore_errors=True)
        vdir.mkdir(parents=True)
        inputs, outputs, _, _ = brv._load_ports(json_path, args.top)
        env = os.environ.copy()
        brv._suite_env(args.suite, env)
        vlog = vdir / f"{args.top}_lut.v"
        tb = vdir / f"tb_{args.top}_lut.cpp"
        _, rt_s = timed(lambda: brv._run_yosys(json_path, vlog, yosys))
        tb.write_text(brv._harness(args.top, inputs, outputs, 2, 0x20260708))
        verilator = brv._tool("verilator", args.suite)
        _, vbuild_s = timed(lambda: brv._build_verilator(vlog, tb, args.top, vdir, env, verilator))
        result["verilator"] = {"lut_json_to_verilog_s": rt_s, "verilator_build_O3_s": vbuild_s}

    text = json.dumps(result, indent=2)
    if args.output_json:
        args.output_json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
