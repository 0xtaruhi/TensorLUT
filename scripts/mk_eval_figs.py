"""Generate the evaluation figures for the ASP-DAC paper.

figs/batch_scaling.pdf -- Rocket throughput and GPU memory vs. batch, and TensorLUT speedup over
                          measured multi-process Verilator across workloads.
figs/eval_summary.pdf  -- ESOP ablation, RocketChip module scale, and compile-time breakdown.

Every series is distinguished by marker and line style as well as color, and bars by hatching,
so the figures read in grayscale print and for color-vision-deficient readers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
FIGS = ROOT / "paper" / "figs"

# One style per simulator, reused in every panel: (label, color, marker, filled, line style).
STYLE = {
    "tensorlut": ("TensorLUT", "#B2182B", "o", True, "-"),
    "rtlflow": ("RTLflow", "#2166AC", "s", False, (0, (5, 2))),
    "cuda": ("CUDA LUT (no TC)", "#1B7837", "^", False, (0, (5, 1.5, 1.5, 1.5))),
    "verilator": ("Verilator, 80 proc.", "black", "D", False, (0, (1.5, 1.5))),
}
# Workload curves in the speedup panel.
WORKLOAD = [
    ("#B2182B", "o", True, "-"),
    ("#2166AC", "s", False, (0, (5, 2))),
    ("#1B7837", "^", False, (0, (5, 1.5, 1.5, 1.5))),
    ("black", "D", False, (0, (1.5, 1.5))),
    ("#D95F02", "v", True, (0, (8, 2))),
]
GRAY = "#7F7F7F"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "black",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#BDBDBD",
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 6.5,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
    "legend.borderpad": 0.3,
    "legend.handlelength": 2.4,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.6,
    "hatch.linewidth": 0.5,
    "savefig.dpi": 300,
})


def _load(name: str):
    with (BENCH / name).open() as f:
        return json.load(f)


def _plot(ax, xs, ys, color, marker, filled, ls, label=None):
    ax.plot(xs, ys, color=color, marker=marker, linestyle=ls, label=label,
            markerfacecolor=color if filled else "white", markeredgecolor=color,
            markeredgewidth=0.8, zorder=3)


def _caption(ax, xlabel, caption):
    # Subfigure caption under the axis label, as in two-column EDA proceedings.
    ax.set_xlabel(f"{xlabel}\n\n{caption}", linespacing=1.0)


def _pow2_axis(ax, lo: int, hi: int, step: int = 2):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator([2**e for e in range(lo, hi + 1, step)]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: rf"$2^{{{int(round(np.log2(v)))}}}$"))
    ax.set_xlim(2**lo / 1.3, 2**hi * 1.3)


def _pow10(v, _):
    e = int(np.floor(np.log10(v) + 1e-9))
    return rf"$10^{{{e}}}$" if abs(v / 10**e - 1) < 1e-6 else ""


def fig_batch_scaling() -> None:
    b1 = sorted(_load("rocket_batch_sweep_b1.json")["results"], key=lambda r: r["batch"])
    cu = sorted(_load("rocket_batch_sweep_cuda.json")["results"], key=lambda r: r["batch"])
    vr = _load("verilator_batch_sweep_rocket.json")["results"]
    v1 = _load("rocket_verilator_scale.json")["results"][0]["cycle_stimuli_per_s_mean"]
    rf = _load("rocket_rtlflow_sweep.json")["results"]
    gem = _load("rocket_gem.json")["cycles_per_s_single_stimulus"]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(7.1, 2.25), constrained_layout=True,
                                        gridspec_kw={"width_ratios": [1.15, 0.85, 1.15]})

    # (a) Rocket throughput vs batch.
    rocket_v = {r["batch"]: r["cycle_stimuli_per_s_mean"] for r in vr}
    series = [
        ("tensorlut", [r["batch"] for r in b1 if r["batch"] >= 64],
         [r["cycle_stimuli_per_s_mean"] for r in b1 if r["batch"] >= 64]),
        ("rtlflow", [r["batch"] for r in rf], [r["median_cycle_stimuli_per_s"] for r in rf]),
        ("cuda", [r["batch"] for r in cu], [r["cycle_stimuli_per_s_mean"] for r in cu]),
        ("verilator", sorted(rocket_v), [rocket_v[b] for b in sorted(rocket_v)]),
    ]
    for key, xs, ys in series:
        label, color, marker, filled, ls = STYLE[key]
        _plot(ax0, xs, ys, color, marker, filled, ls, label)
    ax0.axhline(v1, color=GRAY, linewidth=0.7, linestyle="--", zorder=2)
    ax0.text(2**19, v1 * 0.82, "Verilator, 1 proc.", fontsize=6.3, va="top", ha="right")
    ax0.axhline(gem, color=GRAY, linewidth=0.7, linestyle="-.", zorder=2)
    ax0.text(2**19, gem * 1.18, "GEM, 1 stimulus", fontsize=6.3, va="bottom", ha="right")
    ax0.set_yscale("log")
    ax0.yaxis.set_major_formatter(FuncFormatter(_pow10))
    ax0.set_ylim(2e4, 6e8)
    _pow2_axis(ax0, 6, 19, 2)
    ax0.set_ylabel(r"Throughput (cycle$\cdot$stimuli/s)")
    _caption(ax0, "Stimuli per batch $B$", "(a) Rocket core throughput")
    ax0.legend(loc="upper left", ncol=2, columnspacing=0.8, handlelength=2.6)

    # (b) GPU memory vs batch (its own panel: one y-axis per measure).
    for key, rows in (("tensorlut", b1), ("cuda", cu)):
        label, color, marker, filled, ls = STYLE[key]
        xs = [r["batch"] for r in rows if r["batch"] >= 1024]
        ys = [r["cuda_memory_allocated_gb"] for r in rows if r["batch"] >= 1024]
        _plot(ax1, xs, ys, color, marker, filled, ls, label)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.set_ylim(0.005, 300)
    _pow2_axis(ax1, 10, 19, 3)
    ax1.set_ylabel("Peak GPU memory (GB)")
    _caption(ax1, "Stimuli per batch $B$", "(b) Rocket GPU memory")
    ax1.legend(loc="upper left", handlelength=2.6)

    # (c) Speedup over measured multi-process Verilator.
    micro = _load("micro_batch_sweep.json")["results"]
    iscas = _load("iscas_batch_sweep.json")["results"]
    vsmall = _load("verilator_batch_sweep_small.json")["results"]

    def vrate(design, batch):
        rows = [r for r in vsmall if r["design"] == design]
        return min(rows, key=lambda r: abs(np.log2(r["batch"]) - np.log2(batch)))[
            "cycle_stimuli_per_s_mean"]

    curves = []
    b1map = {r["batch"]: r["cycle_stimuli_per_s_mean"] for r in b1}
    # Verilator shards B over 80 processes, so its batches are multiples of 80 (e.g. 4080);
    # pair each with the nearest power-of-two GPU batch.
    rb = sorted(rocket_v)
    rx = [2 ** int(round(np.log2(b))) for b in rb]
    curves.append(("Rocket (LUT-ANF)", rx, [b1map[x] / rocket_v[b] for x, b in zip(rx, rb)]))
    for design, label in (("lfsr16_free", "LFSR16 (affine)"), ("counter8", "Counter8 (LUT-ANF)")):
        rows = sorted([r for r in micro if r["design"] == design], key=lambda r: r["batch"])
        curves.append((label, [r["batch"] for r in rows],
                       [r["tensorlut_cycle_stimuli_per_s"] / vrate(design, r["batch"]) for r in rows]))
    for design, label in (("s1488_bench", "s1488 (ESOP)"), ("s349_bench", "s349 (ESOP)")):
        rows = sorted([r for r in iscas if r["design"] == design], key=lambda r: r["batch"])
        curves.append((label, [r["batch"] for r in rows],
                       [r["esop_cycle_stimuli_per_s"] / vrate(design, r["batch"]) for r in rows]))
    for (label, xs, ys), (color, marker, filled, ls) in zip(curves, WORKLOAD):
        _plot(ax2, xs, ys, color, marker, filled, ls, label)
    ax2.axhline(1.0, color=GRAY, linewidth=0.7, linestyle="--", zorder=2)
    ax2.set_yscale("log")
    ax2.yaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" + r"$\times$"))
    ax2.set_ylim(0.5, 900)
    _pow2_axis(ax2, 6, 18, 2)
    ax2.set_ylabel("Speedup over 80-proc. Verilator")
    _caption(ax2, "Stimuli per batch $B$", "(c) Speedup over the CPU baseline")
    ax2.legend(loc="upper center", ncol=2, fontsize=6.0, columnspacing=0.8, handlelength=2.4)

    fig.savefig(FIGS / "batch_scaling.pdf")


def fig_eval_summary() -> None:
    data = _load("performance_results.json")
    comp = _load("rocket_compile_overhead.json")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(7.1, 2.1), constrained_layout=True,
                                        gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]})
    bar = dict(edgecolor="black", linewidth=0.6, zorder=3)

    # (a) ESOP speedup over levelized LUT-ANF at two batch sizes.
    esop = data["esop_vs_lut_anf"]
    designs = [("nfsr16", "NFSR16"), ("s349_bench", "s349"), ("s386_bench", "s386"),
               ("s510_bench", "s510"), ("s1488_bench", "s1488")]
    x = np.arange(len(designs))
    width = 0.36
    for i, (batch, fill, hatch) in enumerate([(65536, "white", "////"), (262144, "#8C8C8C", "")]):
        vals = [next(r for r in esop if r["workload"] == d and r["batch"] == batch)["speedup"]
                for d, _ in designs]
        ax0.bar(x + (i - 0.5) * width, vals, width, color=fill, hatch=hatch,
                label=rf"$B=2^{{{int(np.log2(batch))}}}$", **bar)
    ax0.axhline(1.0, color="black", linewidth=0.6, linestyle="--", zorder=2)
    ax0.set_xticks(x)
    ax0.set_xticklabels([n for _, n in designs])
    ax0.tick_params(axis="x", top=False)
    ax0.grid(False, axis="x")
    ax0.set_ylim(0, 2.6)
    ax0.set_ylabel("Speedup over LUT-ANF")
    _caption(ax0, "Design", "(a) ESOP vs. levelized LUT-ANF")
    ax0.legend(loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.6)

    # (b) RocketChip module scale after abc -lut 6.
    rows = {r["module"]: r for r in data["large_rtl_tensorlut_compile"]}
    modules = ["ALU", "CSRFile", "MulDiv", "Rocket"]
    y = np.arange(len(modules))
    h = 0.36
    luts = [rows[m]["lut"] / 1e3 for m in modules]
    monos = [rows[m]["mono_total"] / 1e3 for m in modules]
    ax1.barh(y + h / 2, luts, h, color="white", hatch="////", label="LUTs", **bar)
    ax1.barh(y - h / 2, monos, h, color="#8C8C8C", label="Distinct monomials", **bar)
    for yi, m in enumerate(modules):
        ax1.text(monos[yi] + 1.5, yi - h / 2, f"{rows[m]['chunks_tau512']} chunks",
                 va="center", fontsize=6.3)
    ax1.set_yticks(y)
    ax1.set_yticklabels(modules)
    ax1.tick_params(axis="y", right=False)
    ax1.grid(False, axis="y")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.7, 3.7)
    _caption(ax1, "Count (thousands)", "(b) RocketChip lowering scale")
    ax1.legend(loc="lower right", handlelength=1.6)

    # (c) Compile time: shared Yosys/ABC plus each simulator's own build.
    t = comp["tensorlut"]
    v = comp["verilator"]
    rfc = _load("rocket_rtlflow_sweep.json")["compile_s"]
    gmc = _load("rocket_gem.json")["compile_s"]
    shared = t["yosys_abc_lut6_s"]
    own = {
        "TensorLUT": t["parse_netlist_s"] + t["build_plan_s"] + t["anf_tensor_build_s"]
                     + t["cuda_graph_capture_s"],
        "GEM": gmc["yosys_aigpdk"] + gmc["cut_map_interactive"],
        "Verilator": v["lut_json_to_verilog_s"] + v["verilator_build_O3_s"],
        "RTLflow": v["lut_json_to_verilog_s"] + rfc["transpile"] + rfc["nvcc_model"],
    }
    names = ["RTLflow", "Verilator", "GEM", "TensorLUT"]
    yy = np.arange(len(names))
    ax2.barh(yy, [shared] * len(names), 0.55, color="white", hatch="////",
             label="Yosys/ABC (shared)", **bar)
    ax2.barh(yy, [own[n] for n in names], 0.55, left=[shared] * len(names), color="#8C8C8C",
             label="Simulator build", **bar)
    for yi, n in enumerate(names):
        ax2.text(shared + own[n] + 8, yi, f"+{own[n]:.0f} s", va="center", fontsize=6.3)
    ax2.set_yticks(yy)
    ax2.set_yticklabels(names)
    ax2.tick_params(axis="y", right=False)
    ax2.grid(False, axis="y")
    ax2.set_xlim(0, 760)
    ax2.set_ylim(-0.6, 4.3)
    _caption(ax2, "Wall time (s)", "(c) Rocket compile time")
    ax2.legend(loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.6, fontsize=6.0)

    fig.savefig(FIGS / "eval_summary.pdf")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_batch_scaling()
    fig_eval_summary()
    print(f"wrote {FIGS / 'batch_scaling.pdf'} and {FIGS / 'eval_summary.pdf'}")


if __name__ == "__main__":
    main()
