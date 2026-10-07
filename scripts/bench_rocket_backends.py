#!/usr/bin/env python3
"""Rocket same-plan GPU backend comparison.

This keeps the Rocket LUT/DFF boundary fixed and compares GPU execution paths over
the same random primary-input stream.  All backends start from zero state and report
primary-output mismatches against the vectorized gather backend.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtlgemm.frontend import parse_netlist
from rtlgemm.ir import build_plan
from rtlgemm.runtime.b1_sim import (
    B1AnfBufferedGraphSim,
    B1AnfGraphSim,
    B1AnfSim,
    B1ChunkedDirectVSim,
    B1ChunkedSim,
    B1ChunkedNoScatterGraphSim,
    B1ChunkedNoScatterSim,
    B1ChunkedNoScatterStreamSim,
    B1ChunkedStreamSim,
    B1Packed8GraphSim,
    B1Packed8Sim,
    B1Packed8StreamSim,
    B1LayerPacked8Sim,
    B1LayerPacked8GraphSim,
    B1ProgramPacked8Sim,
    B1BlockProgramPacked8Sim,
    B1CoopProgramPacked8Sim,
    B1SharedInputChunkedSim,
)
from rtlgemm.runtime.lut_cuda_sim import CudaLutGraphSim
from rtlgemm.runtime.simulate import CompiledSim


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x, 0) for x in parse_csv(text)]


def time_backend(fn, warmup: int, runs: int) -> tuple[list[dict], float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    rows = []
    for i in range(runs):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        rows.append({"run": i + 1, "wall_s": wall_s})
    vals = [r["wall_s"] for r in rows]
    return rows, statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


def tensor_po(sim: CompiledSim, x0, u):
    sim.run(x0, u)
    torch.cuda.synchronize()
    return sim.po_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-netlist", default="build/large_rtl/rocket_Rocket_lut.json")
    ap.add_argument("--top", default="Rocket")
    ap.add_argument("--batches", default="4096,16384,65536")
    ap.add_argument("--cycles", type=int, default=64)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backends", default="gather,torch_tc,triton_tc,b1_and_popc,b1_vgraph,b1_buffered_graph,b1_chunk64,b1_shared256,b1_chunk256s4,b1_noscatter256,b1_noscatter256s8")
    ap.add_argument("--skip-reference", action="store_true",
                    help="Do not build the gather oracle; mismatch fields become null.")
    ap.add_argument("--capture-po", action="store_true",
                    help="Record primary-output traces even when --skip-reference is set.")
    ap.add_argument("--output-json", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.device.startswith("cuda"):
        raise SystemExit("backend comparison expects a CUDA device")

    nl = parse_netlist(args.json_netlist, args.top)
    plan = build_plan(nl)
    backends = parse_csv(args.backends)
    results = []
    rng = np.random.default_rng(args.seed)

    for batch in parse_int_csv(args.batches):
        u = rng.integers(0, 2, (args.cycles, batch, nl.n_input), dtype=np.uint8)
        x0 = np.zeros((batch, nl.n_state), dtype=np.uint8)
        ref_po = None
        batch_rows = []

        need_reference = (not args.skip_reference) and any(b != "gather" for b in backends)
        if "gather" in backends or need_reference:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            gather = CompiledSim.build(plan, batch, args.cycles, args.device,
                                       use_cuda_graph=True, backend="gather")
            ref_po = tensor_po(gather, x0, u)
            build_s = time.perf_counter() - t0
            rows, wall_mean, wall_stdev = time_backend(
                lambda: tensor_po(gather, x0, u), args.warmup, args.runs
            )
            if "gather" in backends:
                batch_rows.append({
                    "backend": "gather",
                    "batch": batch,
                    "cycles": args.cycles,
                    "build_s": build_s,
                    "runs": rows,
                    "wall_s_mean": wall_mean,
                    "wall_s_stdev": wall_stdev,
                    "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                    "po_mismatches_vs_gather": 0,
                    "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                })

        if "torch_tc" in backends:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = CompiledSim.build(plan, batch, args.cycles, args.device,
                                    use_cuda_graph=True, backend="auto")
            po = tensor_po(sim, x0, u)
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(
                torch.count_nonzero((po ^ ref_po).to(torch.int8)).item()
            )
            rows, wall_mean, wall_stdev = time_backend(
                lambda: tensor_po(sim, x0, u), args.warmup, args.runs
            )
            batch_rows.append({
                "backend": "torch_tc",
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        if "triton_tc" in backends:
            from rtlgemm.kernels.triton_lut import HAVE_TRITON, TritonLutSim
            if HAVE_TRITON:
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                sim = TritonLutSim(plan, batch, args.cycles, args.device, use_cuda_graph=True)
                po = sim.run(u)
                torch.cuda.synchronize()
                build_s = time.perf_counter() - t0
                mismatches = None if ref_po is None else int(
                    torch.count_nonzero((po ^ ref_po).to(torch.int8)).item()
                )
                rows, wall_mean, wall_stdev = time_backend(
                    lambda: sim.run(u), args.warmup, args.runs
                )
                batch_rows.append({
                    "backend": "triton_tc",
                    "batch": batch,
                    "cycles": args.cycles,
                    "build_s": build_s,
                    "runs": rows,
                    "wall_s_mean": wall_mean,
                    "wall_s_stdev": wall_stdev,
                    "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                    "po_mismatches_vs_gather": mismatches,
                    "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                })

        if "b1_and_popc" in backends:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1AnfSim(plan, args.device)
            po = sim.run(u, args.cycles, capture_po=not args.skip_reference)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u, args.cycles, capture_po=not args.skip_reference),
                args.warmup, args.runs
            )
            batch_rows.append({
                "backend": "b1_and_popc",
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        if "b1_vgraph" in backends:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1AnfGraphSim(plan, batch, args.cycles, args.device,
                                capture_po=not args.skip_reference, use_cuda_graph=True)
            po = sim.run(u)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u), args.warmup, args.runs
            )
            batch_rows.append({
                "backend": "b1_vgraph",
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "cuda_graph": sim.has_graph,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        if "b1_buffered_graph" in backends:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1AnfBufferedGraphSim(plan, batch, args.cycles, args.device,
                                        capture_po=not args.skip_reference,
                                        use_cuda_graph=True)
            po = sim.run(u)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u), args.warmup, args.runs
            )
            batch_rows.append({
                "backend": "b1_buffered_graph",
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "cuda_graph": sim.has_graph,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends if b.startswith("b1_chunk") and "s" not in b]:
            chunk = int(backend.removeprefix("b1_chunk"))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1ChunkedSim(plan, args.device, chunk_outputs=chunk)
            po = sim.run(u, args.cycles, capture_po=not args.skip_reference)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u, args.cycles, capture_po=not args.skip_reference),
                args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends if b.startswith("b1_shared")]:
            chunk = int(backend.removeprefix("b1_shared"))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1SharedInputChunkedSim(plan, args.device, chunk_outputs=chunk)
            po = sim.run(u, args.cycles, capture_po=not args.skip_reference)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u, args.cycles, capture_po=not args.skip_reference),
                args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends if b.startswith("b1_chunk") and "s" in b]:
            chunk_text, stream_text = backend.removeprefix("b1_chunk").split("s", 1)
            chunk = int(chunk_text)
            num_streams = int(stream_text)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1ChunkedStreamSim(plan, args.device, chunk_outputs=chunk,
                                     num_streams=num_streams)
            po = sim.run(u, args.cycles, capture_po=not args.skip_reference)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u, args.cycles, capture_po=not args.skip_reference),
                args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "num_streams": num_streams,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends
                        if b.startswith("b1_noscatter")
                        and "s" not in b.removeprefix("b1_noscatter")]:
            chunk = int(backend.removeprefix("b1_noscatter"))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1ChunkedNoScatterSim(plan, args.device, chunk_outputs=chunk)
            po = sim.run(u, args.cycles, capture_po=not args.skip_reference)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u, args.cycles, capture_po=not args.skip_reference),
                args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends
                        if b.startswith("b1_noscatter")
                        and "s" in b.removeprefix("b1_noscatter")]:
            chunk_text, stream_text = backend.removeprefix("b1_noscatter").split("s", 1)
            chunk = int(chunk_text)
            num_streams = int(stream_text)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1ChunkedNoScatterStreamSim(plan, args.device, chunk_outputs=chunk,
                                              num_streams=num_streams)
            po = sim.run(u, args.cycles, capture_po=not args.skip_reference)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u, args.cycles, capture_po=not args.skip_reference),
                args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "num_streams": num_streams,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends if b.startswith("b1_graph")]:
            chunk = int(backend.removeprefix("b1_graph"))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1ChunkedNoScatterGraphSim(plan, batch, args.cycles, args.device,
                                             chunk_outputs=chunk,
                                             capture_po=not args.skip_reference,
                                             use_cuda_graph=True)
            po = sim.run(u)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u), args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "cuda_graph": sim.has_graph,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends if b.startswith("b1_v8graph")]:
            chunk = int(backend.removeprefix("b1_v8graph"))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1Packed8GraphSim(plan, batch, args.cycles, args.device,
                                    chunk_outputs=chunk,
                                    capture_po=not args.skip_reference,
                                    use_cuda_graph=True)
            po = sim.run(u)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u), args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "cuda_graph": sim.has_graph,
                "resident_input": False,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends if b.startswith("b1_v8replay")]:
            chunk = int(backend.removeprefix("b1_v8replay"))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1Packed8GraphSim(plan, batch, args.cycles, args.device,
                                    chunk_outputs=chunk,
                                    capture_po=not args.skip_reference,
                                    use_cuda_graph=True)
            po = sim.run(u)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.replay(), args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "cuda_graph": sim.has_graph,
                "resident_input": True,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends if b.startswith("b1_layerv8")]:
            chunk = int(backend.removeprefix("b1_layerv8"))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1LayerPacked8Sim(plan, args.device, chunk_outputs=chunk)
            po = sim.run(u, args.cycles, capture_po=not args.skip_reference)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u, args.cycles, capture_po=not args.skip_reference),
                args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends if b.startswith("b1_layergraphv8")]:
            chunk = int(backend.removeprefix("b1_layergraphv8"))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1LayerPacked8GraphSim(plan, batch, args.cycles, args.device,
                                         chunk_outputs=chunk,
                                         capture_po=not args.skip_reference,
                                         use_cuda_graph=True)
            po = sim.run(u)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u), args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "cuda_graph": sim.has_graph,
                "resident_input": False,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends if b.startswith("b1_layerreplayv8")]:
            chunk = int(backend.removeprefix("b1_layerreplayv8"))
            capture_po = args.capture_po or not args.skip_reference
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1LayerPacked8GraphSim(plan, batch, args.cycles, args.device,
                                         chunk_outputs=chunk,
                                         capture_po=capture_po,
                                         use_cuda_graph=True)
            if ref_po is None:
                sim.load_inputs(u)
                po = sim.replay()
            else:
                po = sim.run(u)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.replay(), args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "cuda_graph": sim.has_graph,
                "resident_input": True,
                "capture_po": capture_po,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        if "cuda_lutreplay" in backends:
            capture_po = args.capture_po or not args.skip_reference
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = CudaLutGraphSim(plan, batch, args.cycles, args.device,
                                  capture_po=capture_po,
                                  use_cuda_graph=True)
            po = sim.run(u)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.replay(), args.warmup, args.runs
            )
            batch_rows.append({
                "backend": "cuda_lutreplay",
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "cuda_graph": sim.has_graph,
                "resident_input": True,
                "capture_po": capture_po,
                "uses_tensor_cores": False,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends if b.startswith("b1_programv8")]:
            chunk = int(backend.removeprefix("b1_programv8"))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1ProgramPacked8Sim(plan, batch, args.cycles, args.device,
                                      chunk_outputs=chunk,
                                      capture_po=not args.skip_reference)
            po = sim.run(u)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u), args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "resident_input": False,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends if b.startswith("b1_programreplayv8")]:
            chunk = int(backend.removeprefix("b1_programreplayv8"))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1ProgramPacked8Sim(plan, batch, args.cycles, args.device,
                                      chunk_outputs=chunk,
                                      capture_po=not args.skip_reference)
            po = sim.run(u)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.replay(), args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "resident_input": True,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends if b.startswith("b1_blockprogramreplayv8")]:
            chunk = int(backend.removeprefix("b1_blockprogramreplayv8"))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1BlockProgramPacked8Sim(plan, batch, args.cycles, args.device,
                                           chunk_outputs=chunk,
                                           capture_po=not args.skip_reference)
            po = sim.run(u)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.replay(), args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "resident_input": True,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends if b.startswith("b1_coopreplayv8")]:
            chunk = int(backend.removeprefix("b1_coopreplayv8"))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1CoopProgramPacked8Sim(plan, batch, args.cycles, args.device,
                                          chunk_outputs=chunk,
                                          capture_po=not args.skip_reference)
            po = sim.run(u)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.replay(), args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "resident_input": True,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends
                        if b.startswith("b1_v8")
                        and b.removeprefix("b1_v8").isdigit()]:
            chunk = int(backend.removeprefix("b1_v8"))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1Packed8Sim(plan, args.device, chunk_outputs=chunk)
            po = sim.run(u, args.cycles, capture_po=not args.skip_reference)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u, args.cycles, capture_po=not args.skip_reference),
                args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends
                        if b.startswith("b1_v8")
                        and "s" in b.removeprefix("b1_v8")
                        and all(x.isdigit() for x in b.removeprefix("b1_v8").split("s", 1))]:
            chunk_text, stream_text = backend.removeprefix("b1_v8").split("s", 1)
            chunk = int(chunk_text)
            num_streams = int(stream_text)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1Packed8StreamSim(plan, args.device, chunk_outputs=chunk,
                                     num_streams=num_streams)
            po = sim.run(u, args.cycles, capture_po=not args.skip_reference)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u, args.cycles, capture_po=not args.skip_reference),
                args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "num_streams": num_streams,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        for backend in [b for b in backends if b.startswith("b1_direct")]:
            chunk = int(backend.removeprefix("b1_direct"))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sim = B1ChunkedDirectVSim(plan, args.device, chunk_outputs=chunk)
            po = sim.run(u, args.cycles, capture_po=not args.skip_reference)
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            mismatches = None if ref_po is None else int(torch.count_nonzero((po ^ ref_po).to(torch.int8)).item())
            rows, wall_mean, wall_stdev = time_backend(
                lambda: sim.run(u, args.cycles, capture_po=not args.skip_reference),
                args.warmup, args.runs
            )
            batch_rows.append({
                "backend": backend,
                "batch": batch,
                "cycles": args.cycles,
                "build_s": build_s,
                "runs": rows,
                "wall_s_mean": wall_mean,
                "wall_s_stdev": wall_stdev,
                "cycle_stimuli_per_s_mean": batch * args.cycles / wall_mean,
                "po_mismatches_vs_gather": mismatches,
                "chunk_outputs": chunk,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

        results.extend(batch_rows)

    out = {
        "top": args.top,
        "mode": plan.mode,
        "n_state": nl.n_state,
        "n_input": nl.n_input,
        "n_output_bits": sum(len(bits) for _, bits in nl.outputs),
        "n_lut": len(nl.luts),
        "cycles": args.cycles,
        "runs": args.runs,
        "warmup": args.warmup,
        "results": results,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2) + "\n")
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for row in results:
            print(
                f"B={row['batch']:<7} {row['backend']:<12} "
                f"{row['cycle_stimuli_per_s_mean']:.3e} cycle*stimuli/s "
                f"mismatch={row['po_mismatches_vs_gather']}"
            )


if __name__ == "__main__":
    main()
