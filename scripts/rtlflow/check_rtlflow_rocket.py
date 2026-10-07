#!/usr/bin/env python3
"""Check RTLflow's Rocket outputs against TensorLUT's verified gather backend on identical stimuli.

Reproduces the harness's counter-based input hash, simulates with CompiledSim(backend="gather"), and
compares every primary-output bit of the first N testbenches at cycle C (sampled before the commit).
Usage: check_rtlflow_rocket.py VRocket.h dump.txt C [--json-netlist ...]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rtlgemm.frontend import parse_netlist
from rtlgemm.ir import build_plan
from rtlgemm.runtime.simulate import CompiledSim

M64 = (1 << 64) - 1


def mix(s, t, c, p):
    x = (s ^ (t * 0x9E3779B97F4A7C15) ^ (c * 0xC2B2AE3D27D4EB4F) ^ (p * 0x165667B19E3779F9)) & M64
    x ^= x >> 33; x = (x * 0xFF51AFD7ED558CCD) & M64
    x ^= x >> 33; x = (x * 0xC4CEB9FE1A85EC53) & M64
    x ^= x >> 33
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("header")
    ap.add_argument("dump")
    ap.add_argument("cycles", type=int)
    ap.add_argument("--json-netlist", default="build/large_rtl/rocket_Rocket_lut.json")
    ap.add_argument("--seed", type=lambda x: int(x, 0), default=0x20260708)
    args = ap.parse_args()

    hdr = open(args.header).read()
    ports = [(n, int(m) - int(l) + 1) for _, n, m, l, _ in re.findall(
        r"RF_IN(8|16|64|)\((\w+),(\d+),(\d+)\)\{(\d+) \* THREADS", hdr) if n != "clock"]
    dump = {}
    for line in open(args.dump):
        t, name, val = line.split()
        dump[(int(t), name)] = int(val, 16)
    n_tb = 1 + max(t for t, _ in dump)

    nl = parse_netlist(args.json_netlist, "Rocket")
    pos = {b: j for j, b in enumerate(nl.input_bits)}  # input columns exclude clock nets
    col = {(name, k): pos[b] for name, bits in nl.inputs for k, b in enumerate(bits) if b in pos}
    C = args.cycles + 1
    u = np.zeros((C, n_tb, nl.n_input), dtype=np.uint8)
    for c in range(C):
        for t in range(n_tb):
            for i, (name, width) in enumerate(ports):
                v = mix(args.seed, t, c, i) & ((1 << width) - 1)
                for k in range(width):
                    if (name, k) in col:
                        u[c, t, col[(name, k)]] = (v >> k) & 1
    plan = build_plan(nl)
    sim = CompiledSim.build(plan, n_tb, C, "cuda", use_cuda_graph=False, backend="gather")
    sim.run(np.zeros((n_tb, nl.n_state), dtype=np.uint8), u)
    torch.cuda.synchronize()
    po = sim.po_out[C - 1].cpu().numpy() if torch.is_tensor(sim.po_out) else np.asarray(sim.po_out)[C - 1]

    mism = checked = 0
    off = 0
    for name, bits in nl.outputs:
        for t in range(n_tb):
            ref = sum(int(po[t, off + k]) << k for k in range(len(bits)))
            got = dump[(t, name)]
            checked += len(bits)
            mism += bin(ref ^ got).count("1")
        off += len(bits)
    print(f"testbenches={n_tb} cycle={args.cycles} output_bits_checked={checked} mismatched_bits={mism}")


if __name__ == "__main__":
    main()
