"""Per-cycle GPU simulation driver.

Structure (transition matrix / truth tables) stays resident on the device; only the
state/input batch flows each cycle. To avoid being CPU-launch bound, the whole
multi-cycle loop is built once over *preallocated* buffers (no per-cycle allocation,
``cat`` or re-padding) and captured into a **CUDA graph**, so every
``cycles * kernels`` launch replays from a single CPU call.

Use :class:`CompiledSim` to compile once and run many stimulus batches (the
regression-simulation use case, where graph replay pays off). :func:`simulate_gpu`
is the one-shot convenience wrapper.

* GF2_AFFINE: one INT8 tensor-core GEMM + parity advances a whole clock edge.
* LUT_TENSOR: evaluate LUT layers in topological order (gather), then commit D->Q.
"""
from __future__ import annotations

import torch

from ..frontend.netlist import _const_val
from ..ir.plan import SimPlan
from ..kernels.gf2 import _pad8, _INT_MM_MAX_ROWS, _INT_MM_MIN_ROWS, int8_gemm_i32
from ..kernels.lut import lut_table_tensor
from ..kernels.triton_affine import TritonAffine, HAVE_TRITON


class CompiledSim:
    """A plan compiled to a fixed (batch, cycles) shape, optionally CUDA-graph
    captured. Call :meth:`run` repeatedly with new stimuli — replay is one launch."""

    def __init__(self, plan, batch, cycles, device, use_cuda_graph, out, u_buf, body):
        self.plan = plan
        self.batch = batch
        self.cycles = cycles
        self.device = device
        self.out = out
        self.po_out = None          # (cycles, batch, n_po_bits) for the LUT backend
        self._u_buf = u_buf
        self._body = body
        self._graph = None
        if use_cuda_graph and out.is_cuda:
            self._try_capture()

    def _try_capture(self):
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._body()                    # warmup (allocator / cuBLAS)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._body()                    # record only
            self._graph = g
        except Exception:
            self._graph = None                  # fall back to eager

    def run(self, x0, u_seq) -> torch.Tensor:
        x0 = torch.as_tensor(x0, dtype=torch.int8, device=self.device)
        u_seq = torch.as_tensor(u_seq, dtype=torch.int8, device=self.device)
        assert x0.shape == (self.batch, self.plan.nl.n_state)
        assert u_seq.shape == (self.cycles, self.batch, self.plan.nl.n_input)
        self.out[0].copy_(x0)
        if self._u_buf is not None:
            self._u_buf.copy_(u_seq)
        if self._graph is not None:
            self._graph.replay()
        else:
            self._body()
        return self.out

    # ---- builders -------------------------------------------------------

    @staticmethod
    def build(plan: SimPlan, batch: int, cycles: int, device: str = "cuda",
              use_cuda_graph: bool = True, backend: str = "auto"):
        """backend: 'auto' | 'triton' | 'gemm'. For GF2_AFFINE, 'auto'/'triton' use the
        fused Triton recurrence when available (F<=64, CUDA); else the INT8-GEMM graph
        path. LUT_TENSOR always uses the layered-gather graph path."""
        if plan.mode == "GF2_AFFINE":
            triton_ok = (HAVE_TRITON and plan.affine.F <= 64
                         and str(device).startswith("cuda"))
            if backend in ("auto", "triton") and triton_ok:
                return TritonAffine(plan, batch, cycles, device)
            if backend == "triton" and not triton_ok:
                raise RuntimeError("triton backend unavailable (need Triton, CUDA, F<=64)")
            return _build_affine(plan, batch, cycles, device, use_cuda_graph)
        if plan.mode == "ANF":
            return _build_anf(plan, batch, cycles, device, use_cuda_graph)
        # LUT_TENSOR: default to the tensor-core per-layer ANF GEMM; 'gather' forces a
        # non-TC vectorized reference backend for comparison.
        if backend == "gather":
            return _build_lut(plan, batch, cycles, device, use_cuda_graph)
        return _build_lut_tc(plan, batch, cycles, device, use_cuda_graph)


def _build_affine(plan, batch, cycles, device, use_cuda_graph):
    aff = plan.affine
    n, m, F = aff.n_state, aff.n_input, aff.F
    Kp, Np = _pad8(F), _pad8(n)

    M = torch.as_tensor(aff.M, dtype=torch.int8, device=device)   # (n, F)
    Wp = torch.zeros((Kp, Np), dtype=torch.int8, device=device)
    Wp[:F, :n] = M.t()                                            # padded, transposed
    feats = torch.zeros((batch, Kp), dtype=torch.int8, device=device)
    feats[:, F - 1] = 1                                           # constant-1 feature (col n+m)

    out = torch.empty((cycles + 1, batch, n), dtype=torch.int8, device=device)
    u_buf = (torch.zeros((cycles, batch, m), dtype=torch.int8, device=device)
             if m else None)
    small = batch < _INT_MM_MIN_ROWS

    def body():
        for t in range(cycles):
            feats[:, :n] = out[t]
            if m:
                feats[:, n:n + m] = u_buf[t]
            for s0 in range(0, batch, _INT_MM_MAX_ROWS):
                e0 = min(s0 + _INT_MM_MAX_ROWS, batch)
                fp = feats[s0:e0]
                if small:
                    acc = (fp.float() @ Wp.float()).round().to(torch.int32)
                else:
                    acc = torch._int_mm(fp, Wp)                   # (e0-s0, Np) tensor core
                out[t + 1, s0:e0] = (acc[:, :n] & 1).to(torch.int8)

    return CompiledSim(plan, batch, cycles, device, use_cuda_graph, out, u_buf, body)


def _build_anf(plan, batch, cycles, device, use_cuda_graph):
    """ANF path: two INT8 tensor-core GEMMs per cycle. (1) monomial features via an
    incidence matmul — phi[k] = 1 iff all support bits of monomial k are set, i.e.
    (cv @ A^T)[k] == |support_k|; (2) GF(2) transition GEMM next = (phi @ M^T) & 1."""
    anf = plan.anf
    n, m, F = anf.n_state, anf.n_input, anf.F
    nv = n + m
    A = torch.zeros((F, nv), dtype=torch.int8, device=device)     # monomial incidence
    sizes = torch.zeros(F, dtype=torch.int32, device=device)
    for k, mono in enumerate(anf.monos):
        for v in mono:
            A[k, v] = 1
        sizes[k] = len(mono)
    At = A.t().contiguous()                                       # (nv, F)
    Mt = torch.as_tensor(anf.M, dtype=torch.int8, device=device).t().contiguous()  # (F, n)

    out = torch.empty((cycles + 1, batch, n), dtype=torch.int8, device=device)
    cv = torch.zeros((batch, nv), dtype=torch.int8, device=device)
    u_buf = (torch.zeros((cycles, batch, m), dtype=torch.int8, device=device)
             if m else None)

    def body():
        for t in range(cycles):
            cv[:, :n] = out[t]
            if m:
                cv[:, n:] = u_buf[t]
            cnt = int8_gemm_i32(cv, At)                           # (batch, F) int32
            phi = (cnt == sizes).to(torch.int8)                   # monomial features
            out[t + 1] = (int8_gemm_i32(phi, Mt) & 1).to(torch.int8)

    return CompiledSim(plan, batch, cycles, device, use_cuda_graph, out, u_buf, body)


def _po_and_dff_meta(nl, row_of, device, zero_row=None, one_row=None):
    """Shared PO / next-state readout metadata for the LUT backends."""
    from ..reference.interp import po_bits as _po_bits
    def row_or_const(bit):
        c = _const_val(bit)
        if c is not None:
            if zero_row is None or one_row is None:
                return 0, c == 1
            return (one_row if c == 1 else zero_row), False
        if bit not in row_of:
            if zero_row is not None:
                return zero_row, False
            return 0, False
        return row_of[bit], False

    dff_src, dff_const_cols = [], []
    for i, ff in enumerate(nl.dffs):
        row, is_one = row_or_const(ff.d)
        dff_src.append(row)
        if is_one:
            dff_const_cols.append(i)
    pb = _po_bits(nl)
    po_src, po_const = [], []
    for _, _, b in pb:
        row, is_one = row_or_const(b)
        po_src.append(row)
        if is_one:
            po_const.append(len(po_src) - 1)
    T = lambda x: torch.tensor(x, dtype=torch.long, device=device)
    return (pb, T(dff_src), T(dff_const_cols), len(dff_const_cols) > 0,
            T(po_src), T(po_const), len(po_const) > 0)


def _build_lut_tc(plan, batch, cycles, device, use_cuda_graph, max_monomials=8192):
    """Tensor-core LUT evaluation via per-layer ANF: each layer is two INT8 tensor-core
    GEMMs — (1) build monomial features phi[k] = (Σ support bits == |support|) with an
    incidence matmul, (2) GF(2)-combine per LUT output = (phi @ Cᵀ) & 1. Falls back to
    the gather backend if any layer exceeds the monomial budget."""
    from ..ir.anf import lut_anf

    nl = plan.nl
    n, m = nl.n_state, nl.n_input
    row_of = {}
    for b in nl.state_bits:
        row_of.setdefault(b, len(row_of))
    for b in nl.input_bits:
        row_of.setdefault(b, len(row_of))
    for lut in nl.luts:
        row_of.setdefault(lut.out, len(row_of))
    ZERO_ROW = len(row_of)
    ONE_ROW = ZERO_ROW + 1

    layers_meta = []
    for layer in plan.layers:
        lut_monos = [lut_anf(lut) for lut in layer]
        mono_set = {}
        for monos in lut_monos:
            for mm in monos:
                mono_set.setdefault(mm, len(mono_set))
        if len(mono_set) > max_monomials:
            return _build_lut(plan, batch, cycles, device, use_cuda_graph)  # reference path
        in_nets = sorted({v for mm in mono_set for v in mm})
        col = {net: i for i, net in enumerate(in_nets)}
        F, nin, G = len(mono_set), len(in_nets), len(layer)
        A = torch.zeros((F, max(nin, 1)), dtype=torch.int8, device=device)
        sizes = torch.zeros(F, dtype=torch.int32, device=device)
        for mm, k in mono_set.items():
            sizes[k] = len(mm)
            for v in mm:
                A[k, col[v]] = 1
        C = torch.zeros((G, F), dtype=torch.int8, device=device)
        for g, monos in enumerate(lut_monos):
            for mm in monos:
                C[g, mono_set[mm]] = 1
        in_rows = torch.tensor([row_of[net] for net in in_nets] or [0],
                               dtype=torch.long, device=device)
        out_rows = torch.tensor([row_of[lut.out] for lut in layer], dtype=torch.long, device=device)
        layers_meta.append((in_rows, A.t().contiguous(), sizes, C.t().contiguous(),
                            out_rows, nin))

    V = torch.zeros((len(row_of) + 2, batch), dtype=torch.int8, device=device)
    state_rows = torch.tensor([row_of[b] for b in nl.state_bits], dtype=torch.long, device=device)
    input_rows = torch.tensor([row_of[b] for b in nl.input_bits], dtype=torch.long, device=device)
    pb, dff_src_rows, const_cols, has_const, po_rows, po_const_cols, has_po_const = \
        _po_and_dff_meta(nl, row_of, device, ZERO_ROW, ONE_ROW)

    out = torch.empty((cycles + 1, batch, n), dtype=torch.int8, device=device)
    po_out = torch.empty((cycles, batch, len(pb)), dtype=torch.int8, device=device)
    u_buf = (torch.zeros((cycles, batch, m), dtype=torch.int8, device=device) if m else None)

    def body():
        V[ZERO_ROW] = 0
        V[ONE_ROW] = 1
        for t in range(cycles):
            V[state_rows] = out[t].t()
            if m:
                V[input_rows] = u_buf[t].t()
            for in_rows, At, sizes, Ct, out_rows, nin in layers_meta:
                cv = V[in_rows].t().contiguous()                    # (batch, nin) int8
                cnt = int8_gemm_i32(cv, At)                         # (batch, F) monomial support counts
                phi = (cnt == sizes).to(torch.int8)                 # monomial features
                outv = (int8_gemm_i32(phi, Ct) & 1).to(torch.int8)  # (batch, G) GF(2)-combine
                V[out_rows] = outv.t()
            if len(pb):
                p = (V[po_rows].t() & 1).to(torch.int8)
                if has_po_const:
                    p[:, po_const_cols] = 1
                po_out[t] = p
            nxt = (V[dff_src_rows].t() & 1).to(torch.int8)
            if has_const:
                nxt[:, const_cols] = 1
            out[t + 1] = nxt

    sim = CompiledSim(plan, batch, cycles, device, use_cuda_graph, out, u_buf, body)
    sim.po_out = po_out
    return sim


def _build_lut(plan, batch, cycles, device, use_cuda_graph):
    """Vectorized layered LUT evaluation: within each layer, LUTs are grouped by input
    width and evaluated with a single batched ``gather`` per group (not one op per LUT),
    so hundreds of LUTs cost a handful of kernels — needed to scale to real netlists."""
    nl = plan.nl
    n, m = nl.n_state, nl.n_input

    row_of = {}
    for b in nl.state_bits:
        row_of.setdefault(b, len(row_of))
    for b in nl.input_bits:
        row_of.setdefault(b, len(row_of))
    for lut in nl.luts:
        row_of.setdefault(lut.out, len(row_of))
    ZERO_ROW = len(row_of); ONE_ROW = ZERO_ROW + 1        # constant sentinel rows
    V = torch.zeros((len(row_of) + 2, batch), dtype=torch.int64, device=device)

    def _row(b):
        c = _const_val(b)
        if c is None:
            return row_of[b]
        return ONE_ROW if c == 1 else ZERO_ROW

    state_rows = torch.tensor([row_of[b] for b in nl.state_bits], dtype=torch.long, device=device)
    input_rows = torch.tensor([row_of[b] for b in nl.input_bits], dtype=torch.long, device=device)
    def _read_row(b):
        c = _const_val(b)
        if c is not None:
            return ONE_ROW if c == 1 else ZERO_ROW
        return row_of.get(b, ZERO_ROW)

    # group LUTs by (layer, width): one batched gather per group
    groups = []   # (in_rows(G,w) long, shifts(w,), tables(G,2^w) int64, out_rows(G,) long)
    for layer in plan.layers:
        bywidth = {}
        for lut in layer:
            bywidth.setdefault(lut.width, []).append(lut)
        for w, luts in bywidth.items():
            G = len(luts)
            in_rows = torch.tensor([[_row(b) for b in lut.inputs] for lut in luts],
                                   dtype=torch.long, device=device)          # (G, w)
            shifts = torch.arange(w, device=device, dtype=torch.int64).view(1, w, 1)
            tables = torch.stack([lut_table_tensor(lut, device).to(torch.int64)
                                  for lut in luts])                          # (G, 2^w)
            out_rows = torch.tensor([row_of[lut.out] for lut in luts],
                                    dtype=torch.long, device=device)          # (G,)
            groups.append((in_rows, shifts, tables, out_rows))

    dff_src, dff_const_cols = [], []
    for i, ff in enumerate(nl.dffs):
        c = _const_val(ff.d)
        dff_src.append(_read_row(ff.d))
        if c == 1:
            dff_const_cols.append(i)
    dff_src_rows = torch.tensor(dff_src, dtype=torch.long, device=device)
    const_cols = torch.tensor(dff_const_cols, dtype=torch.long, device=device)
    has_const = len(dff_const_cols) > 0

    # primary-output bits: read each PO net from V after the layer eval
    from ..reference.interp import po_bits as _po_bits
    pb = _po_bits(nl)
    po_src, po_const = [], []
    for k, (_, _, b) in enumerate(pb):
        c = _const_val(b)
        po_src.append(_read_row(b))
        if c == 1:
            po_const.append(k)
    po_rows = torch.tensor(po_src, dtype=torch.long, device=device)
    po_const_cols = torch.tensor(po_const, dtype=torch.long, device=device)
    has_po_const = len(po_const) > 0

    out = torch.empty((cycles + 1, batch, n), dtype=torch.int8, device=device)
    po_out = torch.empty((cycles, batch, len(pb)), dtype=torch.int8, device=device)
    u_buf = (torch.zeros((cycles, batch, m), dtype=torch.int8, device=device)
             if m else None)

    def body():
        V[ONE_ROW] = 1                                              # constant sentinels
        for t in range(cycles):
            V[state_rows] = out[t].t().to(torch.int64)
            if m:
                V[input_rows] = u_buf[t].t().to(torch.int64)
            for in_rows, shifts, tables, out_rows in groups:
                idx = ((V[in_rows] & 1) << shifts).sum(dim=1)       # (G, batch) in [0,2^w)
                V[out_rows] = torch.gather(tables, 1, idx)          # per-LUT table lookup
            if len(pb):                                             # sample POs (state_t, u_t)
                p = (V[po_rows].t() & 1).to(torch.int8)
                if has_po_const:
                    p[:, po_const_cols] = 1
                po_out[t] = p
            nxt = (V[dff_src_rows].t() & 1).to(torch.int8)          # (batch, n)
            if has_const:
                nxt[:, const_cols] = 1
            out[t + 1] = nxt

    sim = CompiledSim(plan, batch, cycles, device, use_cuda_graph, out, u_buf, body)
    sim.po_out = po_out
    return sim


def simulate_gpu(plan: SimPlan, x0, u_seq, cycles: int, device: str = "cuda",
                 use_cuda_graph: bool = True, backend: str = "auto") -> torch.Tensor:
    """One-shot convenience: compile the plan and run a single stimulus batch.
    x0: (batch, n_state); u_seq: (cycles, batch, n_input); values in {0,1}.
    Returns states (cycles+1, batch, n_state) int8 on ``device`` (incl. initial)."""
    batch = x0.shape[0]
    sim = CompiledSim.build(plan, batch, cycles, device, use_cuda_graph, backend)
    return sim.run(x0, u_seq)
