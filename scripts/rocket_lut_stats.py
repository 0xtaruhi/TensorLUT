#!/usr/bin/env python3
"""Compile RocketChip modules from the RTeAAL artifact and report TensorLUT stats."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtlgemm.frontend import parse_netlist
from rtlgemm.ir import build_plan
from rtlgemm.ir.anf import lut_anf

DEFAULT_SRC = (
    ROOT
    / "third_party"
    / "simulators"
    / "RTeAAL-Sim"
    / "verilator"
    / "verilogs"
    / "freechips.rocketchip.system.OctaCoreConfig.v"
)
DEFAULT_TOPS = ["Rocket", "ALU", "MulDiv", "CSRFile"]


def yosys_lut(src: Path, top: str, outdir: Path, yosys: str, k: int, rebuild: bool) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"rocket_{top}_lut.json"
    stat_path = outdir / f"rocket_{top}_lut_stat.json"
    if json_path.exists() and stat_path.exists() and not rebuild:
        return json_path
    script = f"""\
read_verilog -sv {src}
hierarchy -top {top}
proc
flatten
opt
memory_map
opt
techmap
opt
async2sync
dfflegalize -cell $_DFF_P_ x
techmap
abc -lut {k}
opt_clean
tee -o {stat_path} stat -json
write_json {json_path}
"""
    proc = subprocess.run([yosys, "-q", "-p", script], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Yosys failed for top={top}\n{proc.stdout}\n{proc.stderr}")
    return json_path


def stats_for(json_path: Path, top: str, tau: int) -> dict:
    nl = parse_netlist(str(json_path), top)
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
            lut_terms = lut_anf(lut)
            layer_terms += len(lut_terms)
            monos.update(lut_terms)
            for mono in lut_terms:
                max_degree = max(max_degree, len(mono))
        f = len(monos)
        total_terms += layer_terms
        distinct_terms += f
        max_terms = max(max_terms, f)
        q = max(1, math.ceil(f / tau))
        chunks += q
        chunked_layers += int(q > 1)
    return {
        "module": top,
        "pi": nl.n_input,
        "po": sum(len(bits) for _, bits in nl.outputs),
        "dff": nl.n_state,
        "lut": len(nl.luts),
        "layers": len(plan.layers),
        "mono_total": distinct_terms,
        "max_mono_layer": max_terms,
        "max_degree": max_degree,
        "chunks_tau512": chunks,
        "chunked_layers": chunked_layers,
        "term_reuse": round(total_terms / max(distinct_terms, 1), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tops", nargs="*", default=DEFAULT_TOPS)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--outdir", type=Path, default=ROOT / "build" / "large_rtl")
    ap.add_argument("--yosys", default="yosys")
    ap.add_argument("--lut-k", type=int, default=6)
    ap.add_argument("--tau", type=int, default=512)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = ap.parse_args()

    rows = []
    for top in args.tops:
        json_path = yosys_lut(args.src, top, args.outdir, args.yosys, args.lut_k, args.rebuild)
        rows.append(stats_for(json_path, top, args.tau))

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    header = [
        "module",
        "pi",
        "po",
        "dff",
        "lut",
        "layers",
        "mono_total",
        "max_mono_layer",
        "chunks_tau512",
        "chunked_layers",
    ]
    print("\t".join(header))
    for row in rows:
        print("\t".join(str(row[key]) for key in header))


if __name__ == "__main__":
    main()
