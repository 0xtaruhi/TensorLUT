#!/usr/bin/env python3
"""Multi-process Verilator scaling for the same-boundary Rocket LUT netlist."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import bench_rocket_verilator as brv


OUT_RE = re.compile(
    r"acc=(\d+)\s+cycle_stimuli_per_s=([0-9.eE+-]+)\s+wall_s=([0-9.eE+-]+)"
)


def parse_counts(text: str) -> list[int]:
    return [int(x, 0) for x in text.split(",") if x]


def ensure_exe(args) -> Path:
    inputs, outputs, _, _ = brv._load_ports(args.json_netlist, args.top)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    verilog_path = args.build_dir / f"{args.top}_lut.v"
    tb_path = args.build_dir / f"tb_{args.top}_lut.cpp"

    env = os.environ.copy()
    brv._suite_env(args.suite, env)
    yosys = brv._tool("yosys", args.suite)
    verilator = brv._tool("verilator", args.suite)

    exe = args.build_dir / f"obj_{args.top}_lut" / f"vsim_{args.top}_lut"
    if not args.skip_build or not exe.exists():
        brv._run_yosys(args.json_netlist, verilog_path, yosys)
        tb_path.write_text(brv._harness(args.top, inputs, outputs, args.reset_cycles, args.seed))
        exe = brv._build_verilator(verilog_path, tb_path, args.top, args.build_dir, env, verilator)
    return exe


def run_one_sweep(exe: Path, procs: int, batch: int, cycles: int, run_idx: int) -> dict:
    t0 = time.perf_counter()
    children = [
        subprocess.Popen(
            [str(exe), str(batch), str(cycles), str(run_idx * procs + shard)],
            cwd=exe.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for shard in range(procs)
    ]
    samples = []
    for shard, child in enumerate(children):
        out, err = child.communicate()
        if child.returncode != 0:
            raise RuntimeError(
                f"worker {shard} exited {child.returncode}\nstdout:\n{out}\nstderr:\n{err}"
            )
        m = OUT_RE.search(err)
        if not m:
            raise RuntimeError(f"could not parse worker {shard} output:\n{err}")
        samples.append(
            {
                "shard": shard,
                "acc": int(m.group(1)),
                "self_cycle_stimuli_per_s": float(m.group(2)),
                "self_wall_s": float(m.group(3)),
            }
        )
    wall_s = time.perf_counter() - t0
    total = procs * batch * cycles
    acc_xor = 0
    for sample in samples:
        acc_xor ^= sample["acc"]
    return {
        "run": run_idx + 1,
        "wall_s": wall_s,
        "cycle_stimuli_per_s": total / wall_s,
        "sum_worker_cycle_stimuli_per_s": sum(s["self_cycle_stimuli_per_s"] for s in samples),
        "max_worker_wall_s": max(s["self_wall_s"] for s in samples),
        "worker_acc_xor": acc_xor,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-netlist", type=Path, default=brv.DEFAULT_NETLIST)
    ap.add_argument("--top", default="Rocket")
    ap.add_argument("--build-dir", type=Path, default=ROOT / "build" / "rocket_verilator")
    ap.add_argument("--suite", type=Path, default=brv.DEFAULT_SUITE)
    ap.add_argument("--processes", default="1,8,16,32,64,80")
    ap.add_argument("--batch-per-process", type=int, default=4096)
    ap.add_argument("--cycles", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--reset-cycles", type=int, default=2)
    ap.add_argument("--seed", type=lambda x: int(x, 0), default=0x20260708)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--output-json", type=Path)
    args = ap.parse_args()

    exe = ensure_exe(args)
    inputs, outputs, n_lut, n_dff = brv._load_ports(args.json_netlist, args.top)
    rows = []
    for procs in parse_counts(args.processes):
        runs = [run_one_sweep(exe, procs, args.batch_per_process, args.cycles, i)
                for i in range(args.runs)]
        rates = [r["cycle_stimuli_per_s"] for r in runs]
        rows.append(
            {
                "processes": procs,
                "batch_per_process": args.batch_per_process,
                "total_batch": procs * args.batch_per_process,
                "cycles": args.cycles,
                "runs": runs,
                "cycle_stimuli_per_s_mean": statistics.mean(rates),
                "cycle_stimuli_per_s_stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
                "parallel_efficiency_vs_1_process": None,
            }
        )
    if rows:
        base = rows[0]["cycle_stimuli_per_s_mean"] / rows[0]["processes"]
        for row in rows:
            row["parallel_efficiency_vs_1_process"] = (
                row["cycle_stimuli_per_s_mean"] / row["processes"] / base
            )

    result = {
        "top": args.top,
        "json_netlist": str(args.json_netlist),
        "executable": str(exe),
        "host_logical_cpus": os.cpu_count(),
        "n_input_ports": len(inputs),
        "n_output_ports": len(outputs),
        "n_input_bits_including_clock": sum(p["width"] for p in inputs),
        "n_output_bits": sum(p["width"] for p in outputs),
        "n_lut": n_lut,
        "n_dff": n_dff,
        "results": rows,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for row in rows:
            print(
                f"P={row['processes']:>2} "
                f"{row['cycle_stimuli_per_s_mean']:.3e} cycle*stimuli/s "
                f"eff={row['parallel_efficiency_vs_1_process']:.2f}"
            )


if __name__ == "__main__":
    main()
