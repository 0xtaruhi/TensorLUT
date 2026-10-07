#!/usr/bin/env python3
"""Rocket throughput with uniform random stimuli generated on the GPU inside the timed loop.

Packed uint8 words drawn uniformly from [0, 256) are eight independent uniform stimulus bits, so
this matches the in-process random-PI generation of the Verilator harness.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtlgemm.frontend import parse_netlist
from rtlgemm.ir import build_plan
from rtlgemm.runtime.b1_sim import B1LayerPacked8GraphSim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-netlist", default="build/large_rtl/rocket_Rocket_lut.json")
    ap.add_argument("--top", default="Rocket")
    ap.add_argument("--batches", default="1024,16384,262144")
    ap.add_argument("--cycles", type=int, default=64)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--output-json", type=Path)
    args = ap.parse_args()

    nl = parse_netlist(args.json_netlist, args.top)
    plan = build_plan(nl)
    rows = []
    for batch in [int(x) for x in args.batches.split(",") if x]:
        sim = B1LayerPacked8GraphSim(plan, batch, args.cycles, "cuda", chunk_outputs=24,
                                     capture_po=False)
        gen = torch.Generator(device="cuda")
        gen.manual_seed(1)

        def generate():
            sim.u_buf.random_(0, 256, generator=gen)

        generate()
        sim.replay()
        row = {"batch": batch, "cycles": args.cycles}
        for name, fn in [("replay", sim.replay), ("gen+replay", lambda: (generate(), sim.replay()))]:
            walls = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                walls.append(time.perf_counter() - t0)
            row[f"{name}_cycle_stimuli_per_s"] = batch * args.cycles * len(walls) / sum(walls)
        rows.append(row)
        print(row, flush=True)
        del sim
        torch.cuda.empty_cache()
    if args.output_json:
        args.output_json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
