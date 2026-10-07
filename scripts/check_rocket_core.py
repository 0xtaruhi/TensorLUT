#!/usr/bin/env python3
"""Differential check for the RTeAAL Rocket core TensorLUT backend.

This is a backend-level large-netlist check: all three executions consume the same
Yosys/ABC LUT/DFF netlist.  The CPU NumPy interpreter is the golden for the
synthesized netlist; the GPU tensor-core ANF backend and the GPU gather backend
must match both state and primary-output traces bit-for-bit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtlgemm.frontend import parse_netlist
from rtlgemm.ir import build_plan
from rtlgemm.reference import simulate_netlist
from rtlgemm.runtime.simulate import CompiledSim


def _mismatch(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a.astype(np.uint8) ^ b.astype(np.uint8)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-netlist", default="build/large_rtl/rocket_Rocket_lut.json")
    ap.add_argument("--top", default="Rocket")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-cpu", action="store_true",
                    help="Only compare GPU Tensor Core and gather backends; useful for larger batches.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    nl = parse_netlist(args.json_netlist, args.top)
    plan = build_plan(nl)
    rng = np.random.default_rng(args.seed)
    x0 = rng.integers(0, 2, (args.batch, nl.n_state), dtype=np.uint8)
    u = rng.integers(0, 2, (args.cycles, args.batch, nl.n_input), dtype=np.uint8)

    cpu_state = cpu_po = None
    cpu_s = None
    if not args.skip_cpu:
        t0 = time.perf_counter()
        cpu_state, cpu_po = simulate_netlist(nl, x0, u, args.cycles, capture_po=True)
        cpu_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    tc = CompiledSim.build(plan, args.batch, args.cycles, args.device,
                           use_cuda_graph=False, backend="auto")
    tc_state = tc.run(x0, u)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    tc_s = time.perf_counter() - t0
    tc_state_np = tc_state.cpu().numpy().astype(np.uint8)
    tc_po_np = tc.po_out.cpu().numpy().astype(np.uint8)

    t0 = time.perf_counter()
    gather = CompiledSim.build(plan, args.batch, args.cycles, args.device,
                               use_cuda_graph=False, backend="gather")
    gather_state = gather.run(x0, u)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    gather_s = time.perf_counter() - t0
    gather_state_np = gather_state.cpu().numpy().astype(np.uint8)
    gather_po_np = gather.po_out.cpu().numpy().astype(np.uint8)

    result = {
        "top": args.top,
        "batch": args.batch,
        "cycles": args.cycles,
        "seed": args.seed,
        "mode": plan.mode,
        "n_state": nl.n_state,
        "n_input": nl.n_input,
        "n_output_bits": sum(len(bits) for _, bits in nl.outputs),
        "n_lut": len(nl.luts),
        "cpu_reference_s": cpu_s,
        "tensor_core_s": tc_s,
        "gather_s": gather_s,
        "state_mismatches_tensor_core_vs_gather": _mismatch(tc_state_np, gather_state_np),
        "po_mismatches_tensor_core_vs_gather": _mismatch(tc_po_np, gather_po_np),
    }
    if not args.skip_cpu:
        result.update({
            "state_mismatches_tensor_core_vs_cpu": _mismatch(tc_state_np, cpu_state),
            "po_mismatches_tensor_core_vs_cpu": _mismatch(tc_po_np, cpu_po),
            "state_mismatches_gather_vs_cpu": _mismatch(gather_state_np, cpu_state),
            "po_mismatches_gather_vs_cpu": _mismatch(gather_po_np, cpu_po),
        })
    result["passed"] = all(
        result[k] == 0 for k in result if k.startswith(("state_mismatches", "po_mismatches"))
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
