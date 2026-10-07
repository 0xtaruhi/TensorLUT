#!/usr/bin/env python3
"""Tensor-core work cost for a synthesized LUT/DFF JSON netlist.

The metric mirrors the packed-v8 b1 layer backend: each chunk first builds ANF
monomial features with b1 BMMA and then combines them with a second b1 BMMA.
It is intentionally static, so synthesis recipes can be ranked before launching
GPU benchmarks.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtlgemm.frontend import parse_netlist
from rtlgemm.ir import build_plan
from rtlgemm.ir.anf import lut_anf
from rtlgemm.ir.chunking import chunk_layer_luts

WM, WN, WK, KW = 8, 8, 128, 4
SMEM_MAX = 98304


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def layer_chunk_rows(plan, chunk_outputs: int = 24, order: str = "input",
                     greedy_window: int | None = None,
                     literal_bypass: bool = True) -> list[dict]:
    nl = plan.nl
    col = {}
    for bit in nl.state_bits:
        col.setdefault(bit, len(col))
    for bit in nl.input_bits:
        col.setdefault(bit, len(col))
    for lut in nl.luts:
        col.setdefault(lut.out, len(col))

    rows = []
    for li, layer in enumerate(plan.layers):
        chunks = chunk_layer_luts(layer, col, chunk_outputs, order, greedy_window)
        for ci, chunk in enumerate(chunks):
            monos = [lut_anf(lut) for lut in chunk]
            mset = {}
            for ms in monos:
                for mono in ms:
                    mset.setdefault(mono, len(mset))
            in_nets = sorted({v for mono in mset for v in mono})
            degrees = [len(mono) for mono in mset]
            F, NIN, G = len(mset), len(in_nets), len(chunk)
            n_literal = sum(d <= 1 for d in degrees) if literal_bypass else 0
            Ftc = F - n_literal
            n_mt = ceildiv(Ftc, WN)
            n_kt = ceildiv(max(NIN, 1), WK)
            n_ot = ceildiv(G, WN)
            n_ft = ceildiv(F, WK)
            stage1 = n_mt * n_kt
            stage2 = n_ot * n_ft
            phi_w = ceildiv(F, 32)
            phi_pad_w = ceildiv(phi_w, KW) * KW
            scratch_words = n_kt * WM * KW + WM * phi_pad_w
            wpb = 8
            while wpb > 1 and wpb * scratch_words * 4 > SMEM_MAX:
                wpb //= 2
            rows.append({
                "layer": li,
                "chunk": ci,
                "outputs": G,
                "NIN": NIN,
                "F": F,
                "Ftc": Ftc,
                "literal_bypass": n_literal,
                "degree0": sum(d == 0 for d in degrees),
                "degree1": sum(d == 1 for d in degrees),
                "n_mt": n_mt,
                "n_kt": n_kt,
                "n_ot": n_ot,
                "n_ft": n_ft,
                "stage1_bmma": stage1,
                "stage2_bmma": stage2,
                "bmma_per_tile": stage1 + stage2,
                "scratch_words": scratch_words,
                "smem_bytes": wpb * scratch_words * 4,
                "warps_per_block": wpb,
            })
    return rows


def summarize_cost(json_netlist: str | Path, top: str, chunk_outputs: int = 24,
                   order: str = "input", greedy_window: int | None = None,
                   literal_bypass: bool = True) -> dict:
    nl = parse_netlist(str(json_netlist), top)
    plan = build_plan(nl)
    rows = layer_chunk_rows(plan, chunk_outputs=chunk_outputs, order=order,
                            greedy_window=greedy_window,
                            literal_bypass=literal_bypass)
    bmma_vals = [r["bmma_per_tile"] for r in rows]
    stage1 = sum(r["stage1_bmma"] for r in rows)
    stage2 = sum(r["stage2_bmma"] for r in rows)
    useful_stage1 = sum(WM * r["Ftc"] * max(r["NIN"], 1) for r in rows)
    cap_stage1 = sum(r["stage1_bmma"] * WM * WN * WK for r in rows)
    useful_stage2 = sum(WM * r["F"] * r["outputs"] for r in rows)
    cap_stage2 = sum(r["stage2_bmma"] * WM * WN * WK for r in rows)
    return {
        "top": top,
        "json_netlist": str(json_netlist),
        "chunk_outputs": chunk_outputs,
        "order": order,
        "greedy_window": greedy_window,
        "literal_bypass": literal_bypass,
        "n_state": nl.n_state,
        "n_input": nl.n_input,
        "n_output_bits": sum(len(bits) for _, bits in nl.outputs),
        "n_lut": len(nl.luts),
        "n_layers": len(plan.layers),
        "n_chunks": len(rows),
        "n_literal_bypass": sum(r["literal_bypass"] for r in rows),
        "n_degree0": sum(r["degree0"] for r in rows),
        "n_degree1": sum(r["degree1"] for r in rows),
        "stage1_bmma_per_8stim_cycle": stage1,
        "stage2_bmma_per_8stim_cycle": stage2,
        "bmma_per_8stim_cycle": stage1 + stage2,
        "bmma_per_chunk_min": min(bmma_vals) if bmma_vals else 0,
        "bmma_per_chunk_median": statistics.median(bmma_vals) if bmma_vals else 0,
        "bmma_per_chunk_max": max(bmma_vals) if bmma_vals else 0,
        "smem_bytes_max": max((r["smem_bytes"] for r in rows), default=0),
        "tc_tile_useful_ratio": (
            (useful_stage1 + useful_stage2) / max(cap_stage1 + cap_stage2, 1)
        ),
        "stage1_useful_ratio": useful_stage1 / max(cap_stage1, 1),
        "stage2_useful_ratio": useful_stage2 / max(cap_stage2, 1),
        "largest_chunks": sorted(rows, key=lambda r: r["bmma_per_tile"], reverse=True)[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-netlist", default="build/large_rtl/rocket_Rocket_lut.json")
    ap.add_argument("--top", default="Rocket")
    ap.add_argument("--chunk-outputs", type=int, default=24)
    ap.add_argument("--order", choices=["input", "none", "original", "greedy", "tc_greedy"],
                    default="input")
    ap.add_argument("--greedy-window", type=int,
                    help="Candidate window for greedy/tc_greedy chunk ordering.")
    ap.add_argument("--literal-bypass", dest="literal_bypass", action="store_true",
                    default=True,
                    help="Model direct phi fill for constant/single-input monomials.")
    ap.add_argument("--no-literal-bypass", dest="literal_bypass", action="store_false",
                    help="Model the older path where all monomials use stage-1 BMMA.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out = summarize_cost(args.json_netlist, args.top, args.chunk_outputs, args.order,
                         args.greedy_window, args.literal_bypass)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"{out['top']}: LUT={out['n_lut']} layers={out['n_layers']} "
          f"chunks={out['n_chunks']}")
    print(f"BMMA/8stim/cycle={out['bmma_per_8stim_cycle']} "
          f"(stage1={out['stage1_bmma_per_8stim_cycle']}, "
          f"stage2={out['stage2_bmma_per_8stim_cycle']})")
    print(f"useful_ratio={out['tc_tile_useful_ratio']:.3f} "
          f"chunk min/med/max={out['bmma_per_chunk_min']}/"
          f"{out['bmma_per_chunk_median']}/{out['bmma_per_chunk_max']}")


if __name__ == "__main__":
    main()
