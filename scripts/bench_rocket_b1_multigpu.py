#!/usr/bin/env python3
"""Steady-state multi-GPU sharding for Rocket b1 Tensor Core simulation."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_int_csv(text: str) -> list[int]:
    return [int(x, 0) for x in text.split(",") if x]


def worker(rank: int, gpu: int, args, barrier, queue) -> None:
    import torch

    from rtlgemm.frontend import parse_netlist
    from rtlgemm.ir import build_plan
    from rtlgemm.runtime.b1_sim import B1AnfSim, B1ChunkedSim, B1LayerPacked8GraphSim

    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    nl = parse_netlist(args.json_netlist, args.top)
    plan = build_plan(nl)
    build_t0 = time.perf_counter()
    if args.backend == "layer":
        # Final fast path: packed-v8 layer replay; stimuli are drawn on the GPU (uniform random
        # packed bytes are uniform random stimulus bits), so no host array is materialized.
        sim = B1LayerPacked8GraphSim(plan, args.batch_per_gpu, args.cycles, device,
                                     chunk_outputs=args.chunk_outputs, capture_po=False)
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + rank)

        def step():
            sim.u_buf.random_(0, 256, generator=gen)
            sim.replay()
    else:
        rng = np.random.default_rng(args.seed + rank)
        u = rng.integers(0, 2, (args.cycles, args.batch_per_gpu, nl.n_input), dtype=np.uint8)
    if args.backend == "layer":
        pass
    elif args.backend == "chunk":
        sim = B1ChunkedSim(plan, device, chunk_outputs=args.chunk_outputs)
    else:
        sim = B1AnfSim(plan, device)
    if args.backend != "layer":
        def step():
            sim.run(u, args.cycles, capture_po=False)
    step()
    torch.cuda.synchronize(gpu)
    build_warmup_s = time.perf_counter() - build_t0

    rows = []
    for run in range(args.runs):
        barrier.wait()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize(gpu)
        wall_s = time.perf_counter() - t0
        rows.append({"run": run + 1, "wall_s": wall_s})
        barrier.wait()
    queue.put({
        "rank": rank,
        "gpu": gpu,
        "batch": args.batch_per_gpu,
        "cycles": args.cycles,
        "build_warmup_s": build_warmup_s,
        "runs": rows,
        "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated(gpu) / 1e9,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-netlist", default="build/large_rtl/rocket_Rocket_lut.json")
    ap.add_argument("--top", default="Rocket")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--batch-per-gpu", type=int, default=65536)
    ap.add_argument("--cycles", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--backend", choices=["dense", "chunk", "layer"], default="dense")
    ap.add_argument("--chunk-outputs", type=int, default=256)
    ap.add_argument("--output-json", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    gpus = parse_int_csv(args.gpus)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(gpus))
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=worker, args=(rank, gpu, args, barrier, queue))
        for rank, gpu in enumerate(gpus)
    ]
    for proc in procs:
        proc.start()
    workers = [queue.get() for _ in procs]
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise SystemExit(f"worker pid={proc.pid} exited {proc.exitcode}")

    by_rank = {w["rank"]: w for w in workers}
    aggregate_runs = []
    for run in range(args.runs):
        walls = [by_rank[r]["runs"][run]["wall_s"] for r in range(len(gpus))]
        wall_s = max(walls)
        aggregate_runs.append({
            "run": run + 1,
            "wall_s_max": wall_s,
            "worker_wall_s": walls,
            "cycle_stimuli_per_s": len(gpus) * args.batch_per_gpu * args.cycles / wall_s,
        })
    rates = [r["cycle_stimuli_per_s"] for r in aggregate_runs]
    result = {
        "top": args.top,
        "gpus": gpus,
        "batch_per_gpu": args.batch_per_gpu,
        "total_batch": len(gpus) * args.batch_per_gpu,
        "cycles": args.cycles,
        "runs": args.runs,
        "backend": args.backend,
        "chunk_outputs": args.chunk_outputs if args.backend in ("chunk", "layer") else None,
        "workers": sorted(workers, key=lambda w: w["rank"]),
        "aggregate_runs": aggregate_runs,
        "cycle_stimuli_per_s_mean": statistics.mean(rates),
        "cycle_stimuli_per_s_stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"{len(gpus)} GPUs: {result['cycle_stimuli_per_s_mean']:.3e} "
            f"cycle*stimuli/s (stdev {result['cycle_stimuli_per_s_stdev']:.3e})"
        )


if __name__ == "__main__":
    main()
