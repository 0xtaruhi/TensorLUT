#!/usr/bin/env python3
"""Sweep Yosys/ABC LUT mapping recipes and rank them by b1 tensor-core cost."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tc_lut_cost import summarize_cost

DEFAULT_SRC = (
    ROOT
    / "third_party"
    / "simulators"
    / "RTeAAL-Sim"
    / "verilator"
    / "verilogs"
    / "freechips.rocketchip.system.OctaCoreConfig.v"
)


@dataclass(frozen=True)
class Recipe:
    name: str
    abc_cmd: str


RECIPES = [
    Recipe("lut6_default", "abc -lut 6"),
    Recipe("lut6_fast", "abc -fast -lut 6"),
    Recipe("lut6_share2", "abc -S 2 -lut 6"),
    Recipe("lut6_share4", "abc -S 4 -lut 6"),
    Recipe("lut6_share8", "abc -S 8 -lut 6"),
    Recipe("lut4to6", "abc -lut 4:6"),
    Recipe("lut3to6", "abc -lut 3:6"),
    Recipe("cost_linear", "abc -luts 1,2,3,4,5,6"),
    Recipe("cost_exp", "abc -luts 1,1,2,4,8,16"),
    Recipe("cost_anf_bias", "abc -luts 1,1,2,3,6,12"),
    Recipe("abc9_lut6", "abc9 -lut 6"),
    Recipe("abc9_cost_exp", "abc9 -luts 1,1,2,4,8,16"),
]


def recipe_by_name(name: str) -> Recipe:
    for r in RECIPES:
        if r.name == name:
            return r
    raise KeyError(name)


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def synth_recipe(src: Path, top: str, outdir: Path, yosys: str, recipe: Recipe,
                 rebuild: bool) -> Path:
    tag = safe_name(recipe.name)
    rdir = outdir / tag
    rdir.mkdir(parents=True, exist_ok=True)
    json_path = rdir / f"rocket_{top}_lut.json"
    stat_path = rdir / f"rocket_{top}_lut_stat.json"
    meta_path = rdir / "recipe.json"
    script_path = rdir / "run.ys"
    digest = hashlib.sha256(recipe.abc_cmd.encode()).hexdigest()[:12]
    if json_path.exists() and stat_path.exists() and meta_path.exists() and not rebuild:
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("abc_cmd_sha256") == digest:
                return json_path
        except Exception:
            pass

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
{recipe.abc_cmd}
opt_clean
tee -o {stat_path} stat -json
write_json {json_path}
"""
    script_path.write_text(script)
    proc = subprocess.run([yosys, "-q", "-s", str(script_path)],
                          capture_output=True, text=True)
    meta_path.write_text(json.dumps({
        "name": recipe.name,
        "abc_cmd": recipe.abc_cmd,
        "abc_cmd_sha256": digest,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }, indent=2))
    if proc.returncode != 0:
        raise RuntimeError(
            f"Yosys failed for recipe={recipe.name}\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    return json_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--top", default="Rocket")
    ap.add_argument("--outdir", type=Path, default=ROOT / "build" / "tc_mapping_sweep")
    ap.add_argument("--yosys", default="yosys")
    ap.add_argument("--recipes", default=",".join(r.name for r in RECIPES))
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
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="Stop on the first synthesis/cost failure.")
    ap.add_argument("--output-json", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = []
    for name in [x.strip() for x in args.recipes.split(",") if x.strip()]:
        recipe = recipe_by_name(name)
        try:
            json_path = synth_recipe(args.src, args.top, args.outdir, args.yosys,
                                     recipe, args.rebuild)
            cost = summarize_cost(json_path, args.top, args.chunk_outputs, args.order,
                                  args.greedy_window, args.literal_bypass)
            cost.update({
                "recipe": recipe.name,
                "abc_cmd": recipe.abc_cmd,
            })
            rows.append(cost)
        except Exception as exc:
            if args.strict:
                raise
            rows.append({
                "recipe": recipe.name,
                "abc_cmd": recipe.abc_cmd,
                "error": str(exc)[-4000:],
                "bmma_per_8stim_cycle": 10**18,
                "n_lut": 10**18,
                "n_layers": 10**18,
            })

    rows.sort(key=lambda r: (r["bmma_per_8stim_cycle"], r["n_lut"], r["n_layers"]))
    out = {
        "src": str(args.src),
        "top": args.top,
        "chunk_outputs": args.chunk_outputs,
        "order": args.order,
        "greedy_window": args.greedy_window,
        "literal_bypass": args.literal_bypass,
        "rows": rows,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2))
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print("recipe\tbmma\tstage1\tstage2\tlut\tlayers\tchunks\tuse")
    for r in rows:
        if "error" in r:
            print(f"{r['recipe']}\tERROR\t{r['error'].splitlines()[0]}")
            continue
        print(
            f"{r['recipe']}\t{r['bmma_per_8stim_cycle']}\t"
            f"{r['stage1_bmma_per_8stim_cycle']}\t"
            f"{r['stage2_bmma_per_8stim_cycle']}\t"
            f"{r['n_lut']}\t{r['n_layers']}\t{r['n_chunks']}\t"
            f"{r['tc_tile_useful_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
