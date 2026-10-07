#!/usr/bin/env python3
"""Single-thread Verilator baseline for scalar ISCAS'89 benchmark modules.

The generated harness drives reset at cycle boundaries, randomizes all other primary
inputs with a deterministic xorshift stream, samples all primary outputs into an
accumulator, and reports cycle*stimuli/s. This is a CPU baseline only; independent
stimulus shards can be run as separate processes to estimate a multicore ceiling.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITE = Path(os.environ.get("OSS_CAD_SUITE", ""))  # optional; tools come from PATH when unset


def _ports(src: Path):
    text = src.read_text()
    top = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)", text).group(1)
    ins = re.findall(r"^input\s+([A-Za-z_][A-Za-z0-9_$]*)\s*;", text, re.M)
    outs = re.findall(r"^output\s+([A-Za-z_][A-Za-z0-9_$]*)\s*;", text, re.M)
    return top, ins, outs


def _tb(top: str, inputs: list[str], outputs: list[str], reset_cycles: int):
    clk = "blif_clk_net" if "blif_clk_net" in inputs else None
    rst = "blif_reset_net" if "blif_reset_net" in inputs else None
    data_inputs = [p for p in inputs if p not in {clk, rst}]
    set_inputs = "\n".join(
        f"      top->{p} = (next_rng(rng) >> {i % 17}) & 1;" for i, p in enumerate(data_inputs)
    ) or "      (void)rng;"
    acc_outputs = "\n".join(
        f"      acc += ((uint64_t)(top->{p} & 1u) << ({i % 31}));" for i, p in enumerate(outputs)
    ) or "      acc += 0;"
    clk_low = f"top->{clk}=0;" if clk else ""
    clk_high = f"top->{clk}=1;" if clk else ""
    rst_on = f"top->{rst}=1;" if rst else ""
    rst_off = f"top->{rst}=0;" if rst else ""
    return textwrap.dedent(f"""\
    #include "V{top}.h"
    #include "verilated.h"
    #include <cstdio>
    #include <cstdlib>
    #include <cstdint>
    #include <ctime>

    static inline uint64_t next_rng(uint64_t& x) {{
      x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x;
    }}

    int main(int argc, char** argv) {{
      long B = argc > 1 ? atol(argv[1]) : 4096;
      long C = argc > 2 ? atol(argv[2]) : 256;
      V{top}* top = new V{top};
      uint64_t acc = 0;
      struct timespec t0, t1;
      clock_gettime(CLOCK_MONOTONIC, &t0);
      for (long b = 0; b < B; ++b) {{
        uint64_t rng = 0x9e3779b97f4a7c15ULL ^ (uint64_t)b;
        {clk_low} {rst_on}
        top->eval();
        for (int r = 0; r < {reset_cycles}; ++r) {{
          {set_inputs}
          {clk_high} top->eval();
          {clk_low} top->eval();
        }}
        {rst_off}
        for (long c = 0; c < C; ++c) {{
          {set_inputs}
          {clk_high} top->eval();
          {acc_outputs}
          {clk_low} top->eval();
        }}
      }}
      clock_gettime(CLOCK_MONOTONIC, &t1);
      double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
      std::fprintf(stderr, "{top} acc=%llu cyc*stim/s=%.3e time=%.3fs\\n",
                   (unsigned long long)acc, (double)B * C / s, s);
      delete top;
      return 0;
    }}
    """)


def run_design(name: str, batch: int, cycles: int, reset_cycles: int):
    src = ROOT / "benchmarks" / "iscas89" / f"{name}.v"
    top, inputs, outputs = _ports(src)
    build = ROOT / "build" / "verilator_iscas"
    build.mkdir(parents=True, exist_ok=True)
    tb = build / f"tb_{name}.cpp"
    tb.write_text(_tb(top, inputs, outputs, reset_cycles))
    env = os.environ.copy()
    if str(SUITE) and (SUITE / "bin").is_dir():
        env["PATH"] = f"{SUITE / 'bin'}:{env.get('PATH', '')}"
        env["VERILATOR_ROOT"] = str(SUITE / "share" / "verilator")
    cmd = [
        "verilator", "--cc", "--exe", "--build", "-O3",
        "-CFLAGS", "-O3 -march=native",
        "--top-module", top,
        "--Mdir", f"obj_{name}",
        str(src), str(tb.relative_to(build)),
        "-o", f"vsim_{name}",
    ]
    subprocess.run(cmd, cwd=build, env=env, check=True)
    exe = build / f"obj_{name}" / f"vsim_{name}"
    subprocess.run([str(exe), str(batch), str(cycles)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("designs", nargs="*", default=["s1488", "s5378", "s13207", "s15850"])
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--cycles", type=int, default=256)
    ap.add_argument("--reset-cycles", type=int, default=2)
    args = ap.parse_args()
    for name in args.designs:
        run_design(name, args.batch, args.cycles, args.reset_cycles)


if __name__ == "__main__":
    main()
