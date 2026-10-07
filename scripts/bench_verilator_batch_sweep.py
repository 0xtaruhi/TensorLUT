#!/usr/bin/env python3
"""Best-effort multi-process Verilator throughput as a function of total stimulus batch.

For a total batch B, the seeds are sharded over P = min(max_procs, B) independent processes
(B/P seeds each). Wall time spans process launch to the last exit, so small-B points include
process startup, the same way TensorLUT small-B points include launch overhead. Each executable
must accept `<batch> <cycles>` on its command line.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def run_once(exe: Path, procs: int, per_proc: int, cycles: int) -> float:
    t0 = time.perf_counter()
    children = [
        subprocess.Popen([str(exe), str(per_proc), str(cycles)], cwd=exe.parent,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        for _ in range(procs)
    ]
    for child in children:
        _, err = child.communicate()
        if child.returncode != 0:
            raise RuntimeError(f"{exe} exited {child.returncode}: {err}")
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", action="append", required=True, help="name=path/to/vsim")
    ap.add_argument("--batches", default="64,256,1024,4096,16384,65536")
    ap.add_argument("--cycles", type=int, default=512)
    ap.add_argument("--max-procs", type=int, default=os.cpu_count())
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--output-json", type=Path)
    args = ap.parse_args()

    rows = []
    for spec in args.exe:
        name, path = spec.split("=", 1)
        exe = Path(path).resolve()
        for batch in [int(x) for x in args.batches.split(",") if x]:
            procs = min(args.max_procs, batch)
            per_proc = batch // procs
            walls = [run_once(exe, procs, per_proc, args.cycles) for _ in range(args.runs)]
            rates = [procs * per_proc * args.cycles / w for w in walls]
            row = {
                "design": name,
                "batch": procs * per_proc,
                "processes": procs,
                "batch_per_process": per_proc,
                "cycles": args.cycles,
                "wall_s": walls,
                "cycle_stimuli_per_s_mean": statistics.mean(rates),
                "cycle_stimuli_per_s_stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
            }
            rows.append(row)
            print(f"{name:<12} B={row['batch']:>7} P={procs:>2} "
                  f"{row['cycle_stimuli_per_s_mean']:.3e} cycle*stimuli/s", flush=True)

    result = {"host_logical_cpus": os.cpu_count(), "max_procs": args.max_procs, "results": rows}
    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
