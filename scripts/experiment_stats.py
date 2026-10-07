#!/usr/bin/env python3
"""CPU-only experiment statistics for the ASP-DAC paper.

The script synthesizes ISCAS'89 benchmarks to LUT/DFF netlists, builds the
levelized LUT-ANF tensor plan, and reports the quantities used by the paper:
DFF/LUT count, layer count, total and maximum distinct monomials, monomial reuse,
and how many tensor chunks are needed under a monomial tiling threshold.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtlgemm.frontend import parse_netlist, synth
from rtlgemm.ir import build_plan
from rtlgemm.ir.anf import lut_anf


DEFAULT_DESIGNS = [
    "s27", "s298", "s344", "s349", "s382", "s386", "s400", "s444",
    "s510", "s526", "s641", "s713", "s820", "s832", "s953", "s1196",
    "s1238", "s1423", "s1488", "s1494", "s5378", "s9234_1", "s13207",
    "s15850",
]


def stats_for(name: str, tau: int, outdir: str):
    src = ROOT / "benchmarks" / "iscas89" / f"{name}.v"
    top = f"{name}_bench"
    nl = parse_netlist(synth(str(src), top, outdir=outdir).json_path, top)
    plan = build_plan(nl)
    total_terms = 0
    distinct_terms = 0
    max_terms = 0
    max_degree = 0
    chunks = 0
    chunked_layers = 0
    for layer in plan.layers:
        layer_terms = 0
        monos = set()
        for lut in layer:
            for mono in lut_anf(lut):
                layer_terms += 1
                monos.add(mono)
                max_degree = max(max_degree, len(mono))
        f = len(monos)
        total_terms += layer_terms
        distinct_terms += f
        max_terms = max(max_terms, f)
        q = max(1, math.ceil(f / tau))
        chunks += q
        chunked_layers += int(q > 1)
    reuse = total_terms / max(distinct_terms, 1)
    return {
        "design": name.replace("_1", ""),
        "dff": nl.n_state,
        "pi": nl.n_input,
        "po": sum(len(bits) for _, bits in nl.outputs),
        "lut": len(nl.luts),
        "layers": len(plan.layers),
        "mono_total": distinct_terms,
        "max_mono_layer": max_terms,
        "max_degree": max_degree,
        "reuse": reuse,
        "chunks": chunks,
        "chunked_layers": chunked_layers,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("designs", nargs="*", default=DEFAULT_DESIGNS)
    ap.add_argument("--tau", type=int, default=512)
    ap.add_argument("--outdir", default="build/experiment_stats")
    args = ap.parse_args()

    header = [
        "design", "dff", "pi", "po", "lut", "layers", "mono_total",
        "max_mono_layer", "max_degree", "reuse", "chunks", "chunked_layers",
    ]
    print("\t".join(header))
    rows = [stats_for(name, args.tau, args.outdir) for name in args.designs]
    for row in rows:
        vals = []
        for key in header:
            val = row[key]
            vals.append(f"{val:.2f}" if isinstance(val, float) else str(val))
        print("\t".join(vals))


if __name__ == "__main__":
    main()
