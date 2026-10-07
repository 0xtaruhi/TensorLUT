"""Algebraic Normal Form (ANF / Zhegalkin) propagation over an AIG.

Each net's function is represented as a GF(2) polynomial: a set of monomials, each a
frozenset of support-variable indices (state bits 0..n-1, then input bits n..n+m-1;
the empty set is the constant 1). AND multiplies (Cartesian product of monomials, XOR
of overlapping terms), NOT toggles the constant term, XOR is symmetric difference.

If the whole next-state cone stays within a monomial budget, the transition is
``next_state = M · phi (mod 2)`` where phi is the shared monomial feature vector — one
GF(2) GEMM, no truth tables. Degree-1 reduces to the affine matrix; each AND adds a
higher-degree product feature.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from ..frontend.aig import AigNetlist
from ..frontend.netlist import _const_val


def lut_anf(lut):
    """ANF of a single LUT as a list of monomials (frozensets of input-net ids; the
    empty set is the constant 1). Constant inputs are folded in by restricting the
    truth table, so ``x & 0`` correctly drops the whole term and ``x & 1`` drops the
    factor. Output value = XOR of the listed monomials' products."""
    import numpy as np
    from ..reference.interp import _lut_table_array

    w = lut.width
    nonconst = [(j, b) for j, b in enumerate(lut.inputs) if _const_val(b) is None]
    constv = {j: _const_val(b) for j, b in enumerate(lut.inputs) if _const_val(b) is not None}
    wp = len(nonconst)
    tab = _lut_table_array(lut)
    red = np.zeros(1 << wp, dtype=np.uint8)
    for xr in range(1 << wp):
        full = 0
        for k, (j, _) in enumerate(nonconst):
            if (xr >> k) & 1:
                full |= 1 << j
        for j, v in constv.items():
            if v:
                full |= 1 << j
        red[xr] = tab[full]
    a = red.copy()
    for i in range(wp):                       # Möbius transform -> ANF coefficients
        st = 1 << i
        for x in range(1 << wp):
            if x & st:
                a[x] ^= a[x ^ st]
    monos = []
    for S in range(1 << wp):
        if a[S]:
            monos.append(frozenset(nonconst[k][1] for k in range(wp) if (S >> k) & 1))
    return monos


@dataclass
class AnfPlan:
    monos: list          # list[tuple[int]] over support indices; () == const 1
    M: np.ndarray        # (n_state, F) uint8 over GF(2)
    n_state: int
    n_input: int
    degree: int

    @property
    def F(self):
        return len(self.monos)


def _xor(a: set, b: set) -> set:
    return a ^ b


def _and(a: set, b: set, max_terms: int) -> set:
    r: set = set()
    for x in a:
        for y in b:
            m = x | y
            if m in r:
                r.discard(m)
            else:
                r.add(m)
                if len(r) > max_terms:
                    raise _Budget()
    return r


class _Budget(Exception):
    pass


def try_build_anf(nl: AigNetlist, max_terms: int = 4096,
                  max_degree: int = 6) -> AnfPlan | None:
    """Build an AnfPlan if the next-state cone stays within the monomial/degree budget,
    else None (caller falls back to the LUT path)."""
    n, m = nl.n_state, nl.n_input
    var = {b: i for i, b in enumerate(nl.state_bits)}
    for i, b in enumerate(nl.input_bits):
        var[b] = n + i

    anf: dict = {b: {frozenset([i])} for b, i in var.items()}
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))

    def get(bit, stack):
        c = _const_val(bit)
        if c is not None:
            return {frozenset()} if c == 1 else set()
        if bit in anf:
            return anf[bit]
        if bit in stack:
            raise ValueError("combinational cycle in AIG")
        stack.add(bit)
        if bit in nl.nots:
            r = _xor(get(nl.nots[bit], stack), {frozenset()})
        elif bit in nl.ands:
            a, b = nl.ands[bit]
            r = _and(get(a, stack), get(b, stack), max_terms)
        else:
            r = set()  # undriven -> 0
        stack.discard(bit)
        anf[bit] = r
        return r

    try:
        dnf = {ff.q: get(ff.d, set()) for ff in nl.dffs}
    except (_Budget, RecursionError):
        return None

    monos_set: set = set()
    for r in dnf.values():
        monos_set |= r
    degree = max((len(x) for x in monos_set), default=0)
    if degree > max_degree or len(monos_set) > max_terms:
        return None
    monos = sorted(monos_set, key=lambda s: (len(s), sorted(s)))
    midx = {mm: i for i, mm in enumerate(monos)}

    M = np.zeros((n, len(monos)), dtype=np.uint8)
    for j, ff in enumerate(nl.dffs):
        for mm in dnf[ff.q]:
            M[j, midx[mm]] = 1
    return AnfPlan(monos=[tuple(sorted(mm)) for mm in monos], M=M,
                   n_state=n, n_input=m, degree=degree)
