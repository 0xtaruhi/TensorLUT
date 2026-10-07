"""AIG frontend: lower a design to an AND-inverter graph (`aigmap`) so the
next-state functions can be analysed algebraically (ANF / GF(2) monomials) rather
than as truth tables. Cells: ``$_AND_`` (A,B,Y), ``$_NOT_`` (A,Y), ``$_DFF_P_`` (C,D,Q).

Exposes the same state/input/output/dff structure as :class:`~rtlgemm.frontend.netlist.Netlist`
so the reference harnesses (``state_from_port``, ``read_ports``, ``golden_iverilog``) work
unchanged.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field

from .netlist import Dff, _const_val  # reuse Dff + constant handling

_YS_AIG = """\
read_verilog {src}
hierarchy -top {top}
proc
flatten
memory_map
opt
techmap
opt
async2sync
dfflegalize -cell $_DFF_P_ x
aigmap
opt_clean
write_json {json}
"""


@dataclass
class AigNetlist:
    name: str
    inputs: list = field(default_factory=list)       # (port, [bits])
    outputs: list = field(default_factory=list)      # (port, [bits])
    clock_bits: set = field(default_factory=set)
    dffs: list = field(default_factory=list)          # list[Dff]
    ands: dict = field(default_factory=dict)          # Y -> (A, B)
    nots: dict = field(default_factory=dict)          # Y -> A
    state_bits: list = field(default_factory=list)
    input_bits: list = field(default_factory=list)

    @property
    def n_state(self):
        return len(self.state_bits)

    @property
    def n_input(self):
        return len(self.input_bits)


def synth_aig(src: str, top: str, outdir: str = "build", yosys: str = "yosys") -> str:
    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, f"{top}_aig.json")
    proc = subprocess.run([yosys, "-p", _YS_AIG.format(src=src, top=top, json=jpath)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"yosys aig flow failed:\n{proc.stdout}\n{proc.stderr}")
    return jpath


def parse_aig(json_path: str, top: str | None = None) -> AigNetlist:
    design = json.load(open(json_path))
    modules = design["modules"]
    if top is None:
        top = next(iter(modules))
    m = modules[top]
    nl = AigNetlist(name=top)
    for pname, p in m["ports"].items():
        (nl.inputs if p["direction"] == "input" else nl.outputs).append((pname, list(p["bits"])))
    for cn, c in m["cells"].items():
        t, k = c["type"], c["connections"]
        if t == "$_AND_":
            nl.ands[k["Y"][0]] = (k["A"][0], k["B"][0])
        elif t == "$_NOT_":
            nl.nots[k["Y"][0]] = k["A"][0]
        elif t == "$_DFF_P_":
            nl.dffs.append(Dff(name=cn, d=k["D"][0], q=k["Q"][0], clk=k["C"][0]))
        else:
            raise ValueError(f"unexpected AIG cell {t!r} in {cn!r}")
    nl.clock_bits = {ff.clk for ff in nl.dffs if _const_val(ff.clk) is None}
    for pname, bits in nl.inputs:
        for b in bits:
            if b not in nl.clock_bits:
                nl.input_bits.append(b)
    nl.state_bits = [ff.q for ff in nl.dffs]
    return nl
