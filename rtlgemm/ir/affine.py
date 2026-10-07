"""GF(2)-affine detection and global composition.

If every LUT in the design computes an affine Boolean function
``f(a) = a0 ⊕ Σ a_i·x_i`` over GF(2), then the whole next-state function composes
into a single matrix ``M`` over GF(2):

    next_state = M · v (mod 2),   v = [state_0..state_{n-1}, u_0..u_{m-1}, 1]

so one clock edge == one GF(2) GEMM. The trailing constant-1 feature folds in the
affine offset ``c``. This is the headline "single global A matrix" result.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frontend.netlist import Netlist, _const_val
from ..reference.interp import _lut_table_array


def is_affine_lut(lut):
    """Return (is_affine, a0, coeffs) where coeffs[i] is the GF(2) coefficient of
    input i. ``lut`` is affine iff its truth table equals a0 ⊕ Σ coeffs[i]·x_i."""
    w = lut.width
    tab = _lut_table_array(lut)  # index by integer input value, LSB = input 0
    a0 = int(tab[0])
    coeffs = [int(tab[1 << i]) ^ a0 for i in range(w)]
    # verify over the whole table
    idx = np.arange(1 << w)
    pred = np.full(1 << w, a0, dtype=np.int64)
    for i in range(w):
        biti = (idx >> i) & 1
        if coeffs[i]:
            pred ^= biti
    if not np.array_equal(pred.astype(np.uint8), tab):
        return False, None, None
    return True, a0, coeffs


@dataclass
class AffinePlan:
    """next_state = (M @ v) mod 2, v = [state | inputs | 1], M is (n_state, F) uint8."""
    M: np.ndarray            # (n_state, F) uint8 over GF(2)
    n_state: int
    n_input: int
    F: int                   # n_state + n_input + 1


def try_build_affine(nl: Netlist) -> AffinePlan | None:
    """Symbolically propagate GF(2)-affine forms through the netlist. Returns an
    AffinePlan if the entire next-state cone is affine, else None."""
    n, m = nl.n_state, nl.n_input
    F = n + m + 1
    CONST_COL = F - 1

    # affine form of a net = uint8 row of length F (coeffs over v, incl const col)
    form: dict[int, np.ndarray] = {}
    for i, b in enumerate(nl.state_bits):
        row = np.zeros(F, dtype=np.uint8); row[i] = 1; form[b] = row
    for i, b in enumerate(nl.input_bits):
        row = np.zeros(F, dtype=np.uint8); row[n + i] = 1; form[b] = row

    def form_of(bit):
        c = _const_val(bit)
        if c is not None:
            row = np.zeros(F, dtype=np.uint8)
            if c == 1:
                row[CONST_COL] = 1
            return row
        return form.get(bit)

    for lut in nl._lut_topo:
        ok, a0, coeffs = is_affine_lut(lut)
        if not ok:
            return None
        acc = np.zeros(F, dtype=np.uint8)
        if a0:
            acc[CONST_COL] ^= 1
        for i, b in enumerate(lut.inputs):
            if not coeffs[i]:
                continue
            fin = form_of(b)
            if fin is None:
                return None  # driven by unknown/non-affine net
            acc ^= fin
        form[lut.out] = acc

    M = np.zeros((n, F), dtype=np.uint8)
    for j, ff in enumerate(nl.dffs):
        f = form_of(ff.d)
        if f is None:
            return None
        M[j] = f
    return AffinePlan(M=M, n_state=n, n_input=m, F=F)
