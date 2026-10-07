#!/usr/bin/env python3
"""Static scheduling/work analysis for Rocket b1 Tensor-Core ANF chunks."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtlgemm.frontend import parse_netlist
from rtlgemm.ir import build_plan
from rtlgemm.ir.anf import lut_anf

WM, WN, WK, KW = 8, 8, 128, 4
SMEM_MAX = 98304


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def layer_chunk_stats(plan, chunk_outputs: int):
    rows = []
    for li, layer in enumerate(plan.layers):
        for c0 in range(0, len(layer), chunk_outputs):
            chunk = layer[c0:c0 + chunk_outputs]
            monos = [lut_anf(lut) for lut in chunk]
            mset = {}
            for ms in monos:
                for m in ms:
                    mset.setdefault(m, len(mset))
            in_nets = sorted({v for m in mset for v in m})
            F, NIN, G = len(mset), len(in_nets), len(chunk)
            n_mt = ceildiv(F, WN)
            n_kt = ceildiv(max(NIN, 1), WK)
            n_ot = ceildiv(G, WN)
            n_ft = ceildiv(F, WK)
            phi_w = ceildiv(F, 32)
            phi_pad_w = ceildiv(phi_w, KW) * KW
            scratch_words = n_kt * WM * KW + WM * phi_pad_w
            wpb = 8
            while wpb > 1 and wpb * scratch_words * 4 > SMEM_MAX:
                wpb //= 2
            rows.append({
                "layer": li,
                "chunk": c0 // chunk_outputs,
                "outputs": G,
                "NIN": NIN,
                "F": F,
                "n_kt": n_kt,
                "n_mt": n_mt,
                "n_ft": n_ft,
                "n_ot": n_ot,
                "bmma_per_tile": n_mt * n_kt + n_ot * n_ft,
                "scratch_words": scratch_words,
                "smem_bytes": wpb * scratch_words * 4,
                "warps_per_block": wpb,
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-netlist", default="build/large_rtl/rocket_Rocket_lut.json")
    ap.add_argument("--top", default="Rocket")
    ap.add_argument("--batch", type=int, default=131072)
    ap.add_argument("--cycles", type=int, default=64)
    ap.add_argument("--chunk-outputs", type=int, default=256)
    ap.add_argument("--sms", type=int, default=128)
    ap.add_argument("--throughput-json", type=Path,
                    help="Optional benchmark JSON used to compute implied BMMA/s.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    nl = parse_netlist(args.json_netlist, args.top)
    plan = build_plan(nl)
    rows = layer_chunk_stats(plan, args.chunk_outputs)
    n_tiles = ceildiv(args.batch, WM)
    bmma_per_cycle_tile = sum(r["bmma_per_tile"] for r in rows)
    total_bmma = bmma_per_cycle_tile * n_tiles * args.cycles
    blocks = [min(ceildiv(n_tiles, r["warps_per_block"]), 1024) for r in rows]
    bmma_vals = [r["bmma_per_tile"] for r in rows]
    small = sorted(rows, key=lambda r: r["bmma_per_tile"])[:10]
    large = sorted(rows, key=lambda r: r["bmma_per_tile"], reverse=True)[:10]

    out = {
        "top": args.top,
        "batch": args.batch,
        "cycles": args.cycles,
        "chunk_outputs": args.chunk_outputs,
        "n_layers": len(plan.layers),
        "n_chunks": len(rows),
        "n_tiles": n_tiles,
        "sms": args.sms,
        "min_grid_blocks": min(blocks),
        "max_grid_blocks": max(blocks),
        "chunks_with_grid_lt_sms": sum(1 for b in blocks if b < args.sms),
        "bmma_per_cycle_per_8_stimuli": bmma_per_cycle_tile,
        "total_bmma": total_bmma,
        "bmma_per_chunk_min": min(bmma_vals),
        "bmma_per_chunk_median": statistics.median(bmma_vals),
        "bmma_per_chunk_max": max(bmma_vals),
        "smem_bytes_max": max(r["smem_bytes"] for r in rows),
        "warps_per_block_values": sorted(set(r["warps_per_block"] for r in rows)),
        "smallest_chunks": small,
        "largest_chunks": large,
    }

    if args.throughput_json:
        data = json.loads(args.throughput_json.read_text())
        rates = []
        for r in data.get("results", []):
            if r.get("batch") == args.batch and r.get("cycles") == args.cycles:
                wall = r["wall_s_mean"]
                rates.append({
                    "backend": r["backend"],
                    "implied_bmma_per_s": total_bmma / wall,
                    "cycle_stimuli_per_s": r["cycle_stimuli_per_s_mean"],
                    "wall_s_mean": wall,
                })
        out["benchmark_rates"] = rates

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"chunks={out['n_chunks']} layers={out['n_layers']} tiles={n_tiles}")
        print(f"grid blocks min/max={out['min_grid_blocks']}/{out['max_grid_blocks']} "
              f"chunks grid<SM={out['chunks_with_grid_lt_sms']}")
        print(f"BMMA per cycle per 8 stimuli={bmma_per_cycle_tile:,}")
        print(f"BMMA/chunk min/median/max="
              f"{out['bmma_per_chunk_min']}/{out['bmma_per_chunk_median']}/{out['bmma_per_chunk_max']}")
        print(f"warps/block={out['warps_per_block_values']} smem max={out['smem_bytes_max']} B")
        for r in out.get("benchmark_rates", []):
            print(f"{r['backend']}: {r['implied_bmma_per_s']:.3e} BMMA/s "
                  f"({r['cycle_stimuli_per_s']:.3e} cycle*stimuli/s)")


if __name__ == "__main__":
    main()
