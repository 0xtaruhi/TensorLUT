"""Parse a Yosys ``write_json`` LUT netlist into a simulatable :class:`Netlist`.

The MVP frontend flow lowers a design to exactly two cell kinds:

* ``$lut``     — combinational logic; ``LUT`` is an MSB-first truth-table string of
                 length ``2**WIDTH``; port ``A`` lists input bits (``A[0]`` = LSB of the
                 truth-table index), port ``Y`` is the single output bit.
* ``$_DFF_P_`` — positive-edge D flip-flop; ports ``C`` (clock), ``D``, ``Q``.
* ``$scopeinfo`` — non-semantic source/debug metadata emitted by Yosys; ignored.

Net "bits" in the JSON are integer net ids, or the literal strings ``"0"``/``"1"``
(and ``"x"``/``"z"``, which the MVP treats as 0) for constant-driven bits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

# A "bit" is either an int net id or a constant string ("0"/"1"/"x"/"z").
Bit = int


def _const_val(b) -> int | None:
    """Return 0/1 if ``b`` is a constant bit, else None (it is a net id)."""
    if isinstance(b, str):
        return 1 if b == "1" else 0  # x/z treated as 0 in 2-state MVP
    return None


@dataclass
class Lut:
    name: str
    inputs: list  # list of Bit (net ids or const strings)
    out: int
    table: str    # MSB-first truth table, len == 2**width
    width: int

    def eval_scalar(self, netvals: dict) -> int:
        idx = 0
        for i, b in enumerate(self.inputs):
            c = _const_val(b)
            v = c if c is not None else (netvals[b] & 1)
            idx |= (v & 1) << i
        # Yosys parameter bitstrings are MSB-first: char at position len-1-idx.
        return int(self.table[len(self.table) - 1 - idx])


@dataclass
class Dff:
    name: str
    d: Bit
    q: int
    clk: Bit


@dataclass
class Netlist:
    name: str
    # primary inputs (excluding clock): list of (port_name, [bits msb..lsb? -> as given])
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)   # (port_name, [bits])
    clock_bits: set = field(default_factory=set)
    luts: list = field(default_factory=list)       # list[Lut]
    dffs: list = field(default_factory=list)        # list[Dff]

    # derived
    input_bits: list = field(default_factory=list)  # flat list of PI net bits (order = self.inputs flattened)
    state_bits: list = field(default_factory=list)   # DFF Q net ids, canonical order
    _lut_topo: list = field(default_factory=list)     # luts in topological order

    @property
    def n_state(self) -> int:
        return len(self.state_bits)

    @property
    def n_input(self) -> int:
        return len(self.input_bits)

    def eval_next(self, state_vals: dict, input_vals: dict) -> dict:
        """Given current register net values and PI net values, evaluate all
        combinational nets and return the next register net values (D inputs)."""
        nv = dict(state_vals)
        nv.update(input_vals)
        for lut in self._lut_topo:
            nv[lut.out] = lut.eval_scalar(nv)
        nxt = {}
        for ff in self.dffs:
            c = _const_val(ff.d)
            nxt[ff.q] = c if c is not None else (nv[ff.d] & 1)
        return nxt


def _bits_of(conn) -> list:
    return list(conn)


def parse_netlist(json_path: str, top: str | None = None) -> Netlist:
    with open(json_path) as f:
        design = json.load(f)
    modules = design["modules"]
    if top is None:
        top = next(iter(modules))
    m = modules[top]

    nl = Netlist(name=top)

    # ports
    for pname, p in m["ports"].items():
        bits = _bits_of(p["bits"])
        if p["direction"] == "input":
            nl.inputs.append((pname, bits))
        else:
            nl.outputs.append((pname, bits))

    # cells
    for cname, c in m["cells"].items():
        ctype = c["type"]
        conn = c["connections"]
        if ctype == "$lut":
            width = int(c["parameters"]["WIDTH"], 2)
            table = c["parameters"]["LUT"]
            nl.luts.append(Lut(name=cname, inputs=_bits_of(conn["A"]),
                               out=conn["Y"][0], table=table, width=width))
        elif ctype == "$_DFF_P_":
            nl.dffs.append(Dff(name=cname, d=conn["D"][0], q=conn["Q"][0],
                               clk=conn["C"][0]))
        elif ctype == "$scopeinfo":
            continue
        else:
            raise ValueError(
                f"unsupported cell type {ctype!r} in {cname!r}; MVP frontend "
                f"expects only $lut and $_DFF_P_ (check the synthesis flow)")

    # clock nets = DFF clock bits; PIs = input ports minus clock nets
    nl.clock_bits = {ff.clk for ff in nl.dffs if _const_val(ff.clk) is None}
    for pname, bits in nl.inputs:
        for b in bits:
            if b not in nl.clock_bits:
                nl.input_bits.append(b)
    # canonical state order = order of DFF Q bits as declared
    nl.state_bits = [ff.q for ff in nl.dffs]

    nl._lut_topo = _toposort_luts(nl)
    return nl


def _toposort_luts(nl: Netlist) -> list:
    """Order LUTs so each LUT's inputs are produced before it. Inputs that are
    PIs or register outputs are 'ready' from the start; only LUT outputs create
    dependencies (combinational logic in a synchronous netlist is acyclic)."""
    driver = {}  # net -> Lut producing it
    for lut in nl.luts:
        driver[lut.out] = lut
    ready = set(nl.input_bits) | set(nl.state_bits)
    order = []
    visited = set()

    def visit(lut, stack):
        if lut.name in visited:
            return
        if lut.name in stack:
            raise ValueError(f"combinational cycle through {lut.name}")
        stack.add(lut.name)
        for b in lut.inputs:
            if _const_val(b) is not None or b in ready:
                continue
            dl = driver.get(b)
            if dl is None:
                # undriven net (e.g. dangling) -> treat as ready(0)
                continue
            visit(dl, stack)
        stack.discard(lut.name)
        visited.add(lut.name)
        order.append(lut)

    for lut in nl.luts:
        visit(lut, set())
    return order
