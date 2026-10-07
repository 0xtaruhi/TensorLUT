"""Fused tensor-core LUT-layer evaluation in Triton.

Each layer's LUTs are represented by their ANF (per-layer monomials). One Triton kernel
per layer fuses: gather the layer's input signals from the resident net-value buffer V,
build the monomial features with a tensor-core ``tl.dot`` (incidence matmul + support
compare), GF(2)-combine each LUT output with a second tensor-core ``tl.dot`` (accumulated
over F-tiles), and scatter the results back to V — all with intermediates kept in SRAM,
so the only global traffic is V (which stays resident across cycles). ``tl.dot`` maps to
INT8 tensor-core MMA. This is the fused, tensor-core version of the per-layer ANF path.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    _CONFIGS = [   # small set (autotune benchmarks all per distinct tile shape)
        triton.Config({"BT": 64, "FT": 64}, num_warps=4, num_stages=2),
        triton.Config({"BT": 128, "FT": 64}, num_warps=8, num_stages=2),
        triton.Config({"BT": 64, "FT": 128}, num_warps=8, num_stages=2),
    ]

    # NOTE: key excludes `batch` — the best block config doesn't depend on the exact
    # (large) batch, so tuning is shared across batch sizes and not re-run per batch.
    @triton.autotune(configs=_CONFIGS, key=["NINP", "FPAD", "GP"])
    @triton.jit
    def _anf_layer_kernel(V, batch, in_rows, At, sizes, Ct, out_rows, n_out,
                          NIN, NINP: tl.constexpr, FPAD: tl.constexpr, GP: tl.constexpr,
                          FT: tl.constexpr, BT: tl.constexpr):
        pid = tl.program_id(0)
        rb = pid * BT + tl.arange(0, BT)                 # batch rows (stimuli)
        mb = rb < batch
        kk = tl.arange(0, NINP)
        rin = tl.load(in_rows + kk, mask=kk < NIN, other=0)
        # gather cv (BT, NINP) from V[in_row, stimulus]; pad cols (kk>=NIN) = 0
        cv = tl.load(V + rin[None, :].to(tl.int64) * batch + rb[:, None],
                     mask=mb[:, None] & (kk[None, :] < NIN), other=0)
        acc = tl.zeros((BT, GP), dtype=tl.int32)
        for f0 in range(0, FPAD, FT):
            ff = f0 + tl.arange(0, FT)
            At_t = tl.load(At + kk[:, None] * FPAD + ff[None, :],
                           mask=kk[:, None] < NIN, other=0)      # (NINP, FT) int8
            cnt = tl.dot(cv, At_t, out_dtype=tl.int32)           # (BT, FT)  tensor core
            sz = tl.load(sizes + ff)
            phi = (cnt == sz[None, :]).to(tl.int8)               # monomial features
            Ct_t = tl.load(Ct + ff[:, None] * GP + tl.arange(0, GP)[None, :])  # (FT, GP) int8
            acc += tl.dot(phi, Ct_t, out_dtype=tl.int32)         # (BT, GP)  tensor core
        outv = (acc & 1).to(tl.int8)
        gg = tl.arange(0, GP)
        rout = tl.load(out_rows + gg, mask=gg < n_out, other=0)
        tl.store(V + rout[None, :].to(tl.int64) * batch + rb[:, None], outv,
                 mask=mb[:, None] & (gg[None, :] < n_out))


def _ceil(n, m):
    return (n + m - 1) // m * m


def _pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


class TritonLutSim:
    """Compile-once/run-many fused tensor-core LUT simulator (ANF per layer)."""

    _MAXFT = 128        # FPAD is a multiple of this so any autotuned FT divides it

    def __init__(self, plan, batch, cycles, device="cuda", use_cuda_graph=True):
        from ..ir.anf import lut_anf
        from ..reference.interp import po_bits
        from ..frontend.netlist import _const_val

        nl = plan.nl
        self.nl, self.device = nl, device
        self.batch, self.cycles = batch, cycles
        row_of = {}
        for b in nl.state_bits: row_of.setdefault(b, len(row_of))
        for b in nl.input_bits: row_of.setdefault(b, len(row_of))
        for lut in nl.luts: row_of.setdefault(lut.out, len(row_of))
        self.ZERO = len(row_of); self.ONE = len(row_of) + 1
        self.n_nets = len(row_of) + 2

        def rrow(b):
            c = _const_val(b)
            if c is None: return row_of[b]
            return self.ONE if c == 1 else self.ZERO

        # 2D output tiling: split each layer into chunks of <=GT LUT outputs, and give
        # each chunk its OWN local monomial subset (strategy A). This bounds both F
        # (only monomials the chunk's LUTs need) and GP (<=GT), so acc[BT,GP] stays small
        # instead of blowing up registers on wide layers.
        GT = 1 << 30                     # per-layer (2D output tiling measured slower:
        self.layers = []                 # bottleneck is V memory traffic, not GP registers)
        for layer in plan.layers:
            for c0 in range(0, len(layer), GT):
                chunk = layer[c0:c0 + GT]
                lut_monos = [lut_anf(lut) for lut in chunk]
                mono_set = {}
                for monos in lut_monos:
                    for mm in monos: mono_set.setdefault(mm, len(mono_set))
                in_nets = sorted({v for mm in mono_set for v in mm})
                col = {net: i for i, net in enumerate(in_nets)}
                F, NIN, G = len(mono_set), len(in_nets), len(chunk)
                # tl.arange bounds must be powers of 2; FPAD is a multiple of _MAXFT so
                # any autotuned FT (<=_MAXFT) evenly tiles the F-loop.
                NINP, FPAD, GP = _pow2(max(NIN, 32)), _ceil(F, self._MAXFT), _pow2(max(G, 16))
                At = np.zeros((NINP, FPAD), np.int8)
                sizes = np.full(FPAD, 127, np.int32)
                for mm, k in mono_set.items():
                    sizes[k] = len(mm)
                    for v in mm: At[col[v], k] = 1
                Ct = np.zeros((FPAD, GP), np.int8)
                for g, monos in enumerate(lut_monos):
                    for mm in monos: Ct[mono_set[mm], g] = 1
                d = lambda a: torch.as_tensor(a, device=device)
                self.layers.append(dict(
                    in_rows=d(np.array([rrow(n) for n in in_nets] or [0], np.int32)),
                    At=d(At.reshape(-1)), sizes=d(sizes), Ct=d(Ct.reshape(-1)),
                    out_rows=d(np.array([row_of[l.out] for l in chunk], np.int32)),
                    NIN=NIN, NINP=NINP, FPAD=FPAD, GP=GP, G=G))

        d = lambda a: torch.as_tensor(np.asarray(a, np.int32), device=device)
        self.state_rows = d([row_of[b] for b in nl.state_bits])
        self.input_rows = d([row_of[b] for b in nl.input_bits])
        self.dff_src = d([rrow(ff.d) for ff in nl.dffs])
        self.pb = po_bits(nl)
        self.po_rows = d([rrow(b) for _, _, b in self.pb])

        # preallocated resident buffers (net-value buffer V stays resident across cycles);
        # input is stored transposed (n_input, batch) so the hot loop needs no transpose.
        P = len(self.pb)
        self.V = torch.zeros((self.n_nets, batch), dtype=torch.int8, device=device)
        self.u_buf = torch.zeros((cycles, nl.n_input, batch), dtype=torch.int8, device=device)
        self.po_out = torch.zeros((cycles, P, batch), dtype=torch.int8, device=device)
        self._graph = None
        if use_cuda_graph:
            self._capture()

    def _body(self):
        nl = self.nl
        batch = self.batch
        grid = lambda meta: ((batch + meta["BT"] - 1) // meta["BT"],)  # BT is autotuned
        self.V.zero_()                                # x0 = 0 (reset driven via inputs)
        self.V[self.ONE] = 1
        for t in range(self.cycles):
            if nl.n_input:
                self.V[self.input_rows] = self.u_buf[t]           # inputs for cycle t
            for L in self.layers:                                 # combinational eval
                _anf_layer_kernel[grid](
                    self.V, batch, L["in_rows"], L["At"], L["sizes"], L["Ct"],
                    L["out_rows"], L["G"], L["NIN"], NINP=L["NINP"], FPAD=L["FPAD"],
                    GP=L["GP"])                                    # BT/FT/warps/stages: autotuned
            self.po_out[t] = self.V[self.po_rows] & 1              # sample PO (state_t, u_t)
            self.V[self.state_rows] = self.V[self.dff_src] & 1     # commit next state

    def _capture(self):
        try:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._body()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._body()
            self._graph = g
        except Exception:
            self._graph = None

    def run(self, u_seq):
        """u_seq: (cycles, batch, n_input). Returns po (cycles, batch, n_po) int8."""
        u = torch.as_tensor(u_seq, dtype=torch.int8, device=self.device)
        if self.nl.n_input:
            self.u_buf.copy_(u.permute(0, 2, 1))       # -> (cycles, n_input, batch)
        if self._graph is not None:
            self._graph.replay()
        else:
            self._body()
        return self.po_out.permute(0, 2, 1).contiguous()   # -> (cycles, batch, n_po)
