"""Classify a netlist into a simulation plan: a single global GF(2) affine matrix
when the whole next-state cone is affine, otherwise a topologically-layered LUT
schedule (each layer's LUTs depend only on earlier layers / PIs / registers)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..frontend.netlist import Netlist, _const_val
from .affine import AffinePlan, try_build_affine
from .anf import AnfPlan


@dataclass
class SimPlan:
    mode: str                       # 'GF2_AFFINE' | 'ANF' | 'LUT_TENSOR'
    nl: object                      # Netlist (LUT) or AigNetlist (affine/ANF)
    affine: AffinePlan | None = None
    anf: AnfPlan | None = None
    layers: list = field(default_factory=list)   # list[list[Lut]] for LUT_TENSOR

    def summary(self) -> str:
        if self.mode == "GF2_AFFINE":
            a = self.affine
            return (f"GF2_AFFINE  n_state={a.n_state} n_input={a.n_input} "
                    f"F={a.F} matrix={a.M.shape}")
        if self.mode == "ANF":
            a = self.anf
            return (f"ANF  n_state={a.n_state} n_input={a.n_input} "
                    f"monomials={a.F} degree={a.degree}")
        widths = sorted({l.width for layer in self.layers for l in layer})
        return (f"LUT_TENSOR  n_state={self.nl.n_state} n_input={self.nl.n_input} "
                f"luts={sum(len(l) for l in self.layers)} layers={len(self.layers)} "
                f"lut_widths={widths}")


def anf_to_affine(anf: AnfPlan) -> AffinePlan:
    """Convert a degree<=1 ANF into the affine matrix layout [state | inputs | const]."""
    n, m = anf.n_state, anf.n_input
    F = n + m + 1
    M = np.zeros((n, F), dtype=np.uint8)
    for k, mono in enumerate(anf.monos):
        col = F - 1 if len(mono) == 0 else mono[0]   # () -> const col; (i,) -> var i
        M[:, col] |= anf.M[:, k]
    return AffinePlan(M=M, n_state=n, n_input=m, F=F)


def _layer_luts(nl: Netlist) -> list:
    """Group LUTs into dependency layers over the already topo-sorted list."""
    level = {}          # net -> level at which it becomes available
    for b in nl.input_bits:
        level[b] = 0
    for b in nl.state_bits:
        level[b] = 0
    layers: list = []
    for lut in nl._lut_topo:
        lvl = 0
        for b in lut.inputs:
            if _const_val(b) is not None:
                continue
            lvl = max(lvl, level.get(b, 0))
        my = lvl + 1
        level[lut.out] = my
        while len(layers) < my:
            layers.append([])
        layers[my - 1].append(lut)
    return layers


def build_plan(nl: Netlist) -> SimPlan:
    aff = try_build_affine(nl)
    if aff is not None:
        return SimPlan(mode="GF2_AFFINE", nl=nl, affine=aff)
    return SimPlan(mode="LUT_TENSOR", nl=nl, layers=_layer_luts(nl))


def compile_design(src: str, top: str, *, max_terms: int = 4096, max_degree: int = 6,
                   outdir: str = "build") -> SimPlan:
    """Router: try the algebraic (AIG -> ANF) path first — degree<=1 becomes the fast
    affine matrix, low-degree becomes the ANF monomial GEMM. If the monomial/degree
    budget is exceeded, repair to the levelized LUT tensor path (also on Tensor Cores)."""
    from ..frontend.aig import synth_aig, parse_aig
    from ..frontend.synth import synth
    from ..frontend.netlist import parse_netlist
    from .anf import try_build_anf

    aig = parse_aig(synth_aig(src, top, outdir=outdir), top)
    anf = try_build_anf(aig, max_terms=max_terms, max_degree=max_degree)
    if anf is not None:
        if anf.degree <= 1:
            return SimPlan(mode="GF2_AFFINE", nl=aig, affine=anf_to_affine(anf))
        return SimPlan(mode="ANF", nl=aig, anf=anf)
    nl = parse_netlist(synth(src, top, outdir=outdir).json_path, top)
    return build_plan(nl)
