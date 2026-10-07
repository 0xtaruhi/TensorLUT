"""Batched pure-numpy interpreter over the parsed LUT netlist.

This is the MVP's *second* reference (the independent golden is CXXRTL). Because
it consumes the same ``$lut``/``$_DFF_P_`` netlist the tensor path does, it isolates
bugs in the GPU tensorization from bugs in the frontend/semantics.
"""
from __future__ import annotations

import numpy as np

from ..frontend.netlist import Netlist, _const_val


def _lut_table_array(lut) -> np.ndarray:
    """Truth table as uint8 array indexed by integer input value (LSB = A[0])."""
    n = 1 << lut.width
    s = lut.table
    return np.array([int(s[len(s) - 1 - i]) for i in range(n)], dtype=np.uint8)


def po_bits(nl: Netlist):
    """Flatten primary outputs to a list of (port, bit_pos, net_or_const)."""
    out = []
    for pname, bits in nl.outputs:
        for pos, b in enumerate(bits):
            out.append((pname, pos, b))
    return out


def simulate_netlist(nl: Netlist, x0: np.ndarray, u_seq: np.ndarray,
                     cycles: int, capture_po: bool = False):
    """Simulate ``cycles`` clock edges for a batch of stimuli.

    x0:    (batch, n_state) uint8   initial register values (order = nl.state_bits)
    u_seq: (cycles, batch, n_input) uint8   PI values per cycle (order = nl.input_bits)
    returns states: (cycles+1, batch, n_state) uint8, including the initial state.
    If ``capture_po``, also returns po_trace (cycles, batch, n_po_bits): each primary
    output sampled from the combinational logic of (state_t, u_t), before the edge.
    """
    batch = x0.shape[0]
    assert x0.shape[1] == nl.n_state, (x0.shape, nl.n_state)
    assert u_seq.shape == (cycles, batch, nl.n_input), u_seq.shape

    tables = {lut.name: _lut_table_array(lut) for lut in nl.luts}
    zero = np.zeros(batch, dtype=np.uint8)
    one = np.ones(batch, dtype=np.uint8)

    state = x0.astype(np.uint8).copy()
    out = np.empty((cycles + 1, batch, nl.n_state), dtype=np.uint8)
    out[0] = state
    pobits = po_bits(nl) if capture_po else []
    po = np.zeros((cycles, batch, len(pobits)), dtype=np.uint8) if capture_po else None

    for t in range(cycles):
        nv = {}
        for i, b in enumerate(nl.state_bits):
            nv[b] = state[:, i]
        for i, b in enumerate(nl.input_bits):
            nv[b] = u_seq[t, :, i]
        for lut in nl._lut_topo:
            idx = np.zeros(batch, dtype=np.int64)
            for j, b in enumerate(lut.inputs):
                c = _const_val(b)
                v = (one if c == 1 else zero) if c is not None else nv[b]
                idx |= (v.astype(np.int64) & 1) << j
            nv[lut.out] = tables[lut.name][idx]
        if capture_po:
            for k, (_, _, b) in enumerate(pobits):
                c = _const_val(b)
                po[t, :, k] = (one if c == 1 else zero) if c is not None else nv.get(b, zero)
        nxt = np.empty_like(state)
        for i, ff in enumerate(nl.dffs):
            c = _const_val(ff.d)
            nxt[:, i] = (one if c == 1 else zero) if c is not None else nv[ff.d]
        state = nxt
        out[t + 1] = state
    return (out, po) if capture_po else out


def state_from_port(nl: Netlist, port_name: str, values: np.ndarray) -> np.ndarray:
    """Build an initial register vector (batch, n_state) in ``nl.state_bits`` order
    from an integer value of an output port that is driven directly by registers.
    Inverse of the register->port mapping used by :func:`read_ports`."""
    values = np.asarray(values).astype(np.int64)
    batch = values.shape[0]
    q_index = {q: i for i, q in enumerate(nl.state_bits)}
    x0 = np.zeros((batch, nl.n_state), dtype=np.uint8)
    bits = dict(nl.outputs)[port_name]
    for pos, b in enumerate(bits):
        if b in q_index:
            x0[:, q_index[b]] = (values >> pos) & 1
    return x0


def read_ports(nl: Netlist, states: np.ndarray, out_ports: np.ndarray = None):
    """Extract output-port bit-vectors from a register-state trace.

    Only supports output ports driven directly by registers (the MVP benchmark
    outputs). Returns dict port_name -> (cycles+1, batch) int array (bit0 = bits[0]).
    """
    q_index = {q: i for i, q in enumerate(nl.state_bits)}
    result = {}
    for pname, bits in nl.outputs:
        vals = np.zeros(states.shape[:2], dtype=np.int64)
        for pos, b in enumerate(bits):
            c = _const_val(b)
            if c is not None:
                bit = c
            elif b in q_index:
                bit = states[:, :, q_index[b]]
            else:
                raise ValueError(
                    f"output port {pname!r} bit {pos} (net {b}) is not a register; "
                    f"read_ports only supports register-driven outputs in the MVP")
            vals |= (np.asarray(bit).astype(np.int64) & 1) << pos
        result[pname] = vals
    return result
