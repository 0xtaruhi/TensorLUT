#!/usr/bin/env python3
"""Write a random-stimulus VCD for GEM's `cuda_test`, from a Yosys JSON netlist's port list.

Every non-clock input gets a new uniform random value each cycle (reset is held for the first
`--reset-cycles` cycles, as in the Verilator harness); the clock toggles twice per cycle.
"""
from __future__ import annotations

import argparse
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-netlist", required=True)
    ap.add_argument("--top", default="Rocket")
    ap.add_argument("--cycles", type=int, default=10000)
    ap.add_argument("--reset-cycles", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--scope", default="Rocket")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bit-blast", action="store_true",
                    help="declare one 1-bit var per port bit (GEM does not read this form for inputs)")
    args = ap.parse_args()

    mod = json.load(open(args.json_netlist))["modules"][args.top]
    ports = [(n, len(p["bits"])) for n, p in mod["ports"].items() if p["direction"] == "input"]
    rng = random.Random(args.seed)
    ids = {}
    with open(args.out, "w") as f:
        f.write("$timescale 1ns $end\n")
        f.write(f"$scope module {args.scope} $end\n")
        n = 0
        for name, width in ports:
            if not args.bit_blast or width == 1:
                ids[name] = f"!{n:x}"; n += 1
                rng_ = f" [{width - 1}:0]" if width > 1 else ""
                f.write(f"$var wire {width} {ids[name]} {name}{rng_} $end\n")
            else:
                ids[name] = []
                for k in range(width):
                    ids[name].append(f"!{n:x}"); n += 1
                    f.write(f"$var wire 1 {ids[name][-1]} {name}[{k}] $end\n")
        f.write("$upscope $end\n$enddefinitions $end\n")

        def emit(name, width, value):
            if width == 1:
                f.write(f"{value}{ids[name]}\n")
            elif isinstance(ids[name], list):
                for k, vid in enumerate(ids[name]):
                    f.write(f"{(value >> k) & 1}{vid}\n")
            else:
                # zero-padded: GEM maps value characters to bits MSB-first without left-extension
                f.write(f"b{value:0{width}b} {ids[name]}\n")

        for cyc in range(args.cycles):
            f.write(f"#{2 * cyc}\n")
            emit("clock", 1, 0)
            for name, width in ports:
                if name == "clock":
                    continue
                if name == "reset":
                    emit(name, 1, 1 if cyc < args.reset_cycles else rng.getrandbits(1))
                else:
                    emit(name, width, rng.getrandbits(width))
            f.write(f"#{2 * cyc + 1}\n")
            emit("clock", 1, 1)
        f.write(f"#{2 * args.cycles}\n")


if __name__ == "__main__":
    main()
