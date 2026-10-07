"""Drive Yosys 0.52 to lower a Verilog design into (a) a ``$lut``/``$_DFF_P_``
JSON netlist for the tensor path and (b) a CXXRTL C++ golden reference.

MVP scope: single clock domain, synchronous reset, 2-state, no large memories.
Reset/enable are folded into combinational LUT logic (``dfflegalize -cell $_DFF_P_``),
so all flops become plain positive-edge DFFs and every input except the clock is a
regular primary input of the transition function.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass
class SynthResult:
    top: str
    json_path: str
    cxxrtl_path: str


# NOTE: no '#' comment lines here — yosys -p parses each newline as a command
# and mishandles inline comments. The two-stage intent:
#   1) write_cxxrtl on the behavioral netlist  -> independent golden reference
#   2) dfflegalize to plain $_DFF_P_ + abc -lut -> $lut/$_DFF_P_ tensor netlist
_YS_TEMPLATE = """\
read_verilog {src}
hierarchy -top {top}
proc
flatten
memory_map
opt
write_cxxrtl {cxxrtl}
techmap
opt
async2sync
dfflegalize -cell $_DFF_P_ x
techmap
abc -lut {k}
opt_clean
write_json {json}
"""


def synth(src: str, top: str, outdir: str = "build", k: int = 6,
          yosys: str = "yosys") -> SynthResult:
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, f"{top}.json")
    cxxrtl_path = os.path.join(outdir, f"{top}_cxxrtl.cc")
    script = _YS_TEMPLATE.format(src=src, top=top, k=k,
                                 json=json_path, cxxrtl=cxxrtl_path)
    proc = subprocess.run([yosys, "-p", script],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"yosys failed for {src} (top={top}):\n{proc.stdout}\n{proc.stderr}")
    return SynthResult(top=top, json_path=json_path, cxxrtl_path=cxxrtl_path)
