#!/usr/bin/env python3
"""Verilator baseline for the synthesized Rocket LUT/DFF netlist.

The benchmark consumes the same Yosys ``write_json`` netlist used by TensorLUT,
round-trips it back to Verilog, and runs a deterministic scalar Verilator harness.
This keeps the comparison boundary at the Rocket core after LUT lowering instead
of comparing against a full SoC software benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Optional oss-cad-suite install (yosys, verilator); tools are taken from PATH when unset.
DEFAULT_SUITE = Path(os.environ.get("OSS_CAD_SUITE", ""))
DEFAULT_NETLIST = ROOT / "build" / "large_rtl" / "rocket_Rocket_lut.json"


def _suite_env(suite: Path, env: dict) -> dict:
    """Prefer an oss-cad-suite install, if one is given, over tools on PATH."""
    if str(suite) and (suite / "bin").is_dir():
        env["PATH"] = f"{suite / 'bin'}:{env.get('PATH', '')}"
        env["VERILATOR_ROOT"] = str(suite / "share" / "verilator")
    return env


def _tool(name: str, suite: Path) -> str:
    candidate = suite / "bin" / name
    return str(candidate) if candidate.exists() else name


def _load_ports(json_path: Path, top: str) -> tuple[list[dict], list[dict], int, int]:
    design = json.loads(json_path.read_text())
    module = design["modules"][top]
    inputs = []
    outputs = []
    for name, port in module["ports"].items():
        row = {"name": name, "width": len(port["bits"])}
        if port["direction"] == "input":
            inputs.append(row)
        else:
            outputs.append(row)
    n_lut = sum(1 for cell in module.get("cells", {}).values() if cell["type"] == "$lut")
    n_dff = sum(1 for cell in module.get("cells", {}).values() if cell["type"] == "$_DFF_P_")
    return inputs, outputs, n_lut, n_dff


def _c_mask(width: int) -> str:
    if width >= 64:
        return "~0ULL"
    return f"((1ULL << {width}) - 1ULL)"


def _set_input_stmt(port: dict, idx: int) -> str:
    name = port["name"]
    width = port["width"]
    if width == 1:
        return f"      top->{name} = (CData)((next_rng(rng) >> {idx % 19}) & 1ULL);"
    return f"      top->{name} = (next_rng(rng) & {_c_mask(width)});"


def _acc_output_stmt(port: dict, idx: int) -> str:
    name = port["name"]
    salt = f"0x{(0x9E3779B97F4A7C15 ^ (idx * 0xBF58476D1CE4E5B9)) & ((1 << 64) - 1):016x}ULL"
    return textwrap.dedent(
        f"""\
              {{
                uint64_t y = (uint64_t)top->{name};
                acc ^= y + {salt} + (acc << 6) + (acc >> 2);
              }}"""
    ).rstrip()


def _harness(top: str, inputs: list[dict], outputs: list[dict], reset_cycles: int, seed: int) -> str:
    clk = next((p["name"] for p in inputs if p["name"] == "clock"), None)
    rst = next((p["name"] for p in inputs if p["name"] == "reset"), None)
    data_inputs = [p for p in inputs if p["name"] not in {clk, rst}]
    set_inputs = "\n".join(_set_input_stmt(p, i) for i, p in enumerate(data_inputs)) or "      (void)rng;"
    acc_outputs = "\n".join(_acc_output_stmt(p, i) for i, p in enumerate(outputs)) or "      acc ^= 0;"
    clk_low = f"top->{clk} = 0;" if clk else ""
    clk_high = f"top->{clk} = 1;" if clk else ""
    rst_on = f"top->{rst} = 1;" if rst else ""
    rst_off = f"top->{rst} = 0;" if rst else ""
    return textwrap.dedent(
        f"""\
        #include "V{top}.h"
        #include "verilated.h"
        #include <cstdint>
        #include <cstdio>
        #include <cstdlib>
        #include <ctime>

        static inline uint64_t next_rng(uint64_t& x) {{
          x ^= x << 13;
          x ^= x >> 7;
          x ^= x << 17;
          return x;
        }}

        int main(int argc, char** argv) {{
          const long B = argc > 1 ? std::atol(argv[1]) : 4096;
          const long C = argc > 2 ? std::atol(argv[2]) : 256;
          const uint64_t shard = argc > 3 ? std::strtoull(argv[3], nullptr, 0) : 0ULL;
          V{top}* top = new V{top};
          uint64_t acc = 0;
          timespec t0, t1;
          clock_gettime(CLOCK_MONOTONIC, &t0);
          for (long b = 0; b < B; ++b) {{
            const uint64_t global_b = (uint64_t)b + shard * (uint64_t)B;
            uint64_t rng = 0x{seed:016x}ULL ^ global_b * 0xd1342543de82ef95ULL;
            {clk_low}
            {rst_on}
            top->eval();
            for (int r = 0; r < {reset_cycles}; ++r) {{
        {set_inputs}
              {clk_high}
              top->eval();
              {clk_low}
              top->eval();
            }}
            {rst_off}
            for (long c = 0; c < C; ++c) {{
        {set_inputs}
              {clk_high}
              top->eval();
        {acc_outputs}
              {clk_low}
              top->eval();
            }}
          }}
          clock_gettime(CLOCK_MONOTONIC, &t1);
          const double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
          const double cps = (double)B * (double)C / s;
          std::fprintf(stderr, "{top} acc=%llu cycle_stimuli_per_s=%.6e wall_s=%.6f\\n",
                       (unsigned long long)acc, cps, s);
          delete top;
          return 0;
        }}
        """
    )


def _run_yosys(json_path: Path, verilog_path: Path, yosys: str) -> None:
    verilog_path.parent.mkdir(parents=True, exist_ok=True)
    script = f"read_json {json_path}; write_verilog -noattr {verilog_path}"
    subprocess.run([yosys, "-q", "-p", script], cwd=ROOT, check=True)


def _build_verilator(verilog_path: Path, tb_path: Path, top: str, build_dir: Path, env: dict, verilator: str) -> Path:
    cmd = [
        verilator,
        "--cc",
        "--exe",
        "--build",
        "-O3",
        "--Wno-fatal",
        "-Wno-DECLFILENAME",
        "-Wno-WIDTH",
        "-Wno-UNOPTFLAT",
        "-CFLAGS",
        "-O3 -march=native -std=c++17",
        "--top-module",
        top,
        "--Mdir",
        f"obj_{top}_lut",
        str(verilog_path),
        str(tb_path.relative_to(build_dir)),
        "-o",
        f"vsim_{top}_lut",
    ]
    subprocess.run(cmd, cwd=build_dir, env=env, check=True)
    return build_dir / f"obj_{top}_lut" / f"vsim_{top}_lut"


def _run_exe(exe: Path, batch: int, cycles: int) -> tuple[float, float, int]:
    proc = subprocess.run(
        [str(exe), str(batch), str(cycles)],
        cwd=exe.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    text = proc.stderr.strip()
    m = re.search(r"acc=(\d+)\s+cycle_stimuli_per_s=([0-9.eE+-]+)\s+wall_s=([0-9.eE+-]+)", text)
    if not m:
        raise RuntimeError(f"could not parse benchmark output:\n{text}")
    return float(m.group(2)), float(m.group(3)), int(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-netlist", type=Path, default=DEFAULT_NETLIST)
    ap.add_argument("--top", default="Rocket")
    ap.add_argument("--build-dir", type=Path, default=ROOT / "build" / "rocket_verilator")
    ap.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--cycles", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--reset-cycles", type=int, default=2)
    ap.add_argument("--seed", type=lambda x: int(x, 0), default=0x20260708)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    inputs, outputs, n_lut, n_dff = _load_ports(args.json_netlist, args.top)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    verilog_path = args.build_dir / f"{args.top}_lut.v"
    tb_path = args.build_dir / f"tb_{args.top}_lut.cpp"

    env = os.environ.copy()
    _suite_env(args.suite, env)
    yosys = _tool("yosys", args.suite)
    verilator = _tool("verilator", args.suite)

    if not args.skip_build:
        _run_yosys(args.json_netlist, verilog_path, yosys)
        tb_path.write_text(_harness(args.top, inputs, outputs, args.reset_cycles, args.seed))
        exe = _build_verilator(verilog_path, tb_path, args.top, args.build_dir, env, verilator)
    else:
        exe = args.build_dir / f"obj_{args.top}_lut" / f"vsim_{args.top}_lut"

    samples = []
    for _ in range(args.runs):
        cps, wall_s, acc = _run_exe(exe, args.batch, args.cycles)
        samples.append({"cycle_stimuli_per_s": cps, "wall_s": wall_s, "acc": acc})

    rates = [s["cycle_stimuli_per_s"] for s in samples]
    result = {
        "top": args.top,
        "json_netlist": str(args.json_netlist.relative_to(ROOT) if args.json_netlist.is_relative_to(ROOT) else args.json_netlist),
        "verilog": str(verilog_path.relative_to(ROOT) if verilog_path.is_relative_to(ROOT) else verilog_path),
        "batch": args.batch,
        "cycles": args.cycles,
        "runs": args.runs,
        "reset_cycles": args.reset_cycles,
        "seed": args.seed,
        "n_input_ports": len(inputs),
        "n_output_ports": len(outputs),
        "n_input_bits_including_clock": sum(p["width"] for p in inputs),
        "n_output_bits": sum(p["width"] for p in outputs),
        "n_lut": n_lut,
        "n_dff": n_dff,
        "samples": samples,
        "cycle_stimuli_per_s_mean": statistics.mean(rates),
        "cycle_stimuli_per_s_stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "per_seed_hz_mean": statistics.mean(rates) / args.batch,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"{args.top}: {result['cycle_stimuli_per_s_mean']:.3e} cycle*stimuli/s "
            f"(stdev {result['cycle_stimuli_per_s_stdev']:.3e}, B={args.batch}, C={args.cycles})"
        )


if __name__ == "__main__":
    sys.exit(main())
