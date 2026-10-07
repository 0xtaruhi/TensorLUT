#!/usr/bin/env python3
"""Check GEM's Rocket output VCD against a scalar evaluation of the same LUT/DFF netlist.

GEM evaluates once per clock edge and stamps the result at the previous active timestamp, so its
output at time 2c+1 is comb(state_{c+1}, u_{c+1}): TensorLUT's PO[c+1] (sampled before that cycle's
commit). We replay the input VCD on the netlist and compare every output bit for c = 0..cycles-2.
Input VCD values must be zero-padded to the port width (make_random_vcd.py does this).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rtlgemm.frontend import parse_netlist


def parse_vcd(path):
    ids, changes, t = {}, {}, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("$var"):
                p = line.split()
                ids[p[3]] = p[4]
            elif line.startswith("#"):
                t = int(line[1:])
            elif line and line[0] in "01xz" and not line.startswith("$"):
                changes.setdefault(t, []).append((ids.get(line[1:]), line[0]))
            elif line.startswith("b"):
                v, i = line[1:].split()
                changes.setdefault(t, []).append((ids.get(i), v))
    return changes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-netlist", required=True)
    ap.add_argument("--input-vcd", required=True)
    ap.add_argument("--output-vcd", required=True)
    ap.add_argument("--cycles", type=int, default=200)
    args = ap.parse_args()

    nl = parse_netlist(args.json_netlist, "Rocket")
    in_bits = {name: bits for name, bits in nl.inputs}
    cin = parse_vcd(args.input_vcd)
    cout = parse_vcd(args.output_vcd)

    def val(nv, b):
        if isinstance(b, str):
            return int(b)
        return nv.get(b, 0)

    U, inputs = [], {}
    for c in range(args.cycles):
        for name, v in cin.get(2 * c, []):
            if name == "clock" or name not in in_bits:
                continue
            x = int(v.replace("x", "0").replace("z", "0"), 2)
            for k, b in enumerate(in_bits[name]):
                if not isinstance(b, str):
                    inputs[b] = (x >> k) & 1
        U.append(dict(inputs))
    state = {q: 0 for q in nl.state_bits}
    cur_out = {}
    checked = mism = 0
    for c in range(args.cycles - 1):
        state = nl.eval_next(state, U[c])          # commit edge c -> state_{c+1}
        nv = dict(state); nv.update(U[c + 1])
        for lut in nl._lut_topo:                   # PO[c+1] = comb(state_{c+1}, u_{c+1})
            nv[lut.out] = lut.eval_scalar(nv)
        for name, v in cout.get(2 * c + 1, []):
            cur_out[name] = v
        for name, bits in nl.outputs:
            for k, b in enumerate(bits):
                key = f"{name}[{k}]" if len(bits) > 1 else name
                if key not in cur_out:
                    continue
                checked += 1
                mism += int(cur_out[key] in "01" and int(cur_out[key]) != (val(nv, b) & 1))
    print(f"cycles={args.cycles} output_bits_checked={checked} mismatched_bits={mism}")


if __name__ == "__main__":
    main()
