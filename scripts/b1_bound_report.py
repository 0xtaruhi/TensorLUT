#!/usr/bin/env python3
"""Combine b1 static cost and runtime JSON into effective Tensor-Core work rates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tc_lut_cost import summarize_cost


def load_results(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data.get("results", [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-netlist", default="build/large_rtl/rocket_Rocket_lut.json")
    ap.add_argument("--top", default="Rocket")
    ap.add_argument("--chunk-outputs", type=int, default=24)
    ap.add_argument("--order", default="input")
    ap.add_argument("--literal-bypass", dest="literal_bypass", action="store_true",
                    default=True,
                    help="Use the literal-bypass static BMMA model.")
    ap.add_argument("--no-literal-bypass", dest="literal_bypass", action="store_false",
                    help="Use the older static model where all monomials use stage-1 BMMA.")
    ap.add_argument("--benchmark-json", type=Path, required=True)
    ap.add_argument("--backend", default="b1_layerreplayv824")
    ap.add_argument("--baseline-json", type=Path,
                    help="Optional benchmark JSON to report speedup against.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cost = summarize_cost(args.json_netlist, args.top, args.chunk_outputs, args.order,
                          literal_bypass=args.literal_bypass)
    bmma_per_8stim_cycle = cost["bmma_per_8stim_cycle"]

    baseline = {}
    if args.baseline_json:
        for row in load_results(args.baseline_json):
            if row.get("backend") == args.backend:
                baseline[row.get("batch")] = row

    rows = []
    for row in load_results(args.benchmark_json):
        if row.get("backend") != args.backend:
            continue
        cycle_stimuli_s = row["cycle_stimuli_per_s_mean"]
        effective_bmma_s = cycle_stimuli_s / 8.0 * bmma_per_8stim_cycle
        out = {
            "backend": row["backend"],
            "batch": row["batch"],
            "cycles": row["cycles"],
            "chunk_outputs": args.chunk_outputs,
            "bmma_per_8stim_cycle": bmma_per_8stim_cycle,
            "cycle_stimuli_per_s": cycle_stimuli_s,
            "effective_bmma_per_s": effective_bmma_s,
            "wall_s_mean": row["wall_s_mean"],
            "capture_po": row.get("capture_po"),
        }
        base = baseline.get(row.get("batch"))
        if base:
            out["baseline_cycle_stimuli_per_s"] = base["cycle_stimuli_per_s_mean"]
            out["speedup_vs_baseline"] = cycle_stimuli_s / base["cycle_stimuli_per_s_mean"]
        rows.append(out)

    report = {
        "cost": {
            k: cost[k]
            for k in [
                "n_lut",
                "n_layers",
                "n_chunks",
                "stage1_bmma_per_8stim_cycle",
                "stage2_bmma_per_8stim_cycle",
                "bmma_per_8stim_cycle",
                "n_literal_bypass",
                "tc_tile_useful_ratio",
                "bmma_per_chunk_max",
            ]
        },
        "rows": rows,
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(
        "batch\tcycle_stim/s\teff_BMMA/s\tBMMA/8stim/cyc\t"
        "speedup_vs_baseline"
    )
    for row in rows:
        speed = row.get("speedup_vs_baseline")
        speed_text = "-" if speed is None else f"{speed:.3f}x"
        print(
            f"{row['batch']}\t{row['cycle_stimuli_per_s']:.3e}\t"
            f"{row['effective_bmma_per_s']:.3e}\t"
            f"{row['bmma_per_8stim_cycle']}\t{speed_text}"
        )


if __name__ == "__main__":
    main()
