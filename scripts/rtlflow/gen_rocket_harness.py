#!/usr/bin/env python3
"""Generate an RTLflow testbench for the Rocket LUT/DFF netlist.

Inputs are drawn on the GPU every cycle (a counter-based hash of seed, testbench, cycle, port), and the
clock is toggled by a kernel, so the timed loop does not migrate managed memory to the host.
Usage: gen_rocket_harness.py obj_dir/VRocket.h > main.cu
"""
import re
import sys

hdr = open(sys.argv[1]).read()
ports = []
for kind, name, msb, lsb, loc in re.findall(
        r"RF_IN(8|16|64|)\((\w+),(\d+),(\d+)\)\{(\d+) \* THREADS", hdr):
    if name == "clock":
        continue
    arr = {"8": "_csignals", "16": "_ssignals", "": "_isignals", "64": "_qsignals"}[kind]
    ctype = {"8": "CData", "16": "SData", "": "IData", "64": "QData"}[kind]
    ports.append((arr, ctype, name, int(msb) - int(lsb) + 1, int(loc)))
outs = []
for kind, name, msb, lsb, loc in re.findall(
        r"RF_OUT(8|16|64|)\((\w+),(\d+),(\d+)\)\{(\d+) \* THREADS", hdr):
    arr = {"8": "_csignals", "16": "_ssignals", "": "_isignals", "64": "_qsignals"}[kind]
    outs.append((arr, name, int(loc)))
dump = "\n".join(
    f'      printf("%zu {name} %llx\\n", t, (unsigned long long)rtlflow.{arr}[{loc}ull * THREADS + t]);'
    for arr, name, loc in outs)
clock = re.search(r"RF_IN8\(clock,0,0\)\{(\d+) \* THREADS", hdr).group(1)

lines = []
for i, (arr, ctype, name, width, loc) in enumerate(ports):
    mask = "0xFFFFFFFFFFFFFFFFull" if width == 64 else f"((1ull << {width}) - 1)"
    lines.append(f"    {arr}[{loc}ull * THREADS + t] = ({ctype})(mix(seed, t, cyc, {i}) & {mask});")
body = "\n".join(lines)

print(f"""#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "VRocket.h"

using namespace RF;

RF::RTLflow rtlflow(GPU_THREADS);
RF::RTLflow& RF::VRocket::_rtlflow = rtlflow;

__device__ __forceinline__ unsigned long long mix(unsigned long long s, unsigned long long t,
                                                  unsigned long long c, unsigned long long p) {{
  unsigned long long x = s ^ (t * 0x9E3779B97F4A7C15ull) ^ (c * 0xC2B2AE3D27D4EB4Full) ^ (p * 0x165667B19E3779F9ull);
  x ^= x >> 33; x *= 0xFF51AFD7ED558CCDull; x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ull; x ^= x >> 33;
  return x;
}}

__global__ void drive(CData* _csignals, SData* _ssignals, IData* _isignals, QData* _qsignals,
                      unsigned long long seed, unsigned long long cyc) {{
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= THREADS) return;
{body}
}}

__global__ void set_clock(CData* _csignals, CData v) {{
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t < THREADS) _csignals[{clock}ull * THREADS + t] = v;
}}

int main(int argc, char** argv) {{
  int cycles = argc > 1 ? atoi(argv[1]) : 64;
  int warm = argc > 2 ? atoi(argv[2]) : 4;
  int dump_tbs = argc > 3 ? atoi(argv[3]) : 0;  // >0: validation mode, print POs of the first N testbenches
  RF::VRocket* tb = new RF::VRocket;
  unsigned blocks = (THREADS + 255) / 256;
  auto step = [&](int cyc) {{
    drive<<<blocks, 256>>>(rtlflow._csignals, rtlflow._ssignals, rtlflow._isignals, rtlflow._qsignals,
                           0x20260708ull, (unsigned long long)cyc);
    set_clock<<<blocks, 256>>>(rtlflow._csignals, 0);
    tb->eval();
    set_clock<<<blocks, 256>>>(rtlflow._csignals, 1);
    tb->eval();
  }};
  tb->eval();
  if (dump_tbs > 0) {{
    // Validation: run `cycles` full cycles, then drive cycle `cycles` and evaluate the negedge only, so the
    // printed outputs are comb(state_C, u_C) -- the same sampling point as TensorLUT's PO[C].
    for (int i = 0; i < cycles; ++i) step(i);
    drive<<<blocks, 256>>>(rtlflow._csignals, rtlflow._ssignals, rtlflow._isignals, rtlflow._qsignals,
                           0x20260708ull, (unsigned long long)cycles);
    set_clock<<<blocks, 256>>>(rtlflow._csignals, 0);
    tb->eval();
    cudaDeviceSynchronize();
    for (size_t t = 0; t < (size_t)dump_tbs; ++t) {{
{dump}
    }}
    delete tb;
    return 0;
  }}
  for (int i = 0; i < warm; ++i) step(-1 - i);
  cudaDeviceSynchronize();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < cycles; ++i) step(i);
  cudaDeviceSynchronize();
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  unsigned long long acc = 0;
  for (size_t t = 0; t < THREADS; ++t) acc ^= (unsigned long long)rtlflow._csignals[t] << (t % 56);
  fprintf(stderr, "batch=%zu cycles=%d wall_s=%.6f cycle_stimuli_per_s=%.6e acc=%llu\\n",
          (size_t)THREADS, cycles, s, (double)THREADS * cycles / s, acc);
  delete tb;
  return 0;
}}""")
