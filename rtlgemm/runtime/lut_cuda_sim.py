"""CUDA-core LUT simulator baseline.

This backend evaluates the same synthesized LUT/DFF netlist as TensorLUT, but
uses ordinary CUDA cores and truth-table lookup rather than Tensor Cores or ANF.
"""
from __future__ import annotations

import numpy as np
import torch

from ..frontend.netlist import _const_val
from ..reference.interp import _lut_table_array, po_bits
from ..kernels import lut_cuda


class CudaLutGraphSim:
    """Graph-captured levelized LUT simulator using ordinary CUDA cores."""

    def __init__(self, plan, batch: int, cycles: int, device="cuda",
                 capture_po=False, use_cuda_graph=True):
        nl = plan.nl
        self.nl = nl
        self.batch = batch
        self.cycles = cycles
        self.device = device
        self.capture_po = capture_po

        col = {}
        for b in nl.state_bits:
            col.setdefault(b, len(col))
        for b in nl.input_bits:
            col.setdefault(b, len(col))
        for lut in nl.luts:
            col.setdefault(lut.out, len(col))
        self.ZERO = len(col)
        self.ONE = len(col) + 1
        self.n_cols = len(col) + 2

        def ccol(b):
            c = _const_val(b)
            if c is None:
                return col[b]
            return self.ONE if c == 1 else self.ZERO

        self.layers = []
        for layer in plan.layers:
            n_lut = len(layer)
            in_cols = np.zeros((n_lut, 6), np.int32)
            out_cols = np.zeros(n_lut, np.int32)
            widths = np.zeros(n_lut, np.uint8)
            tables = np.zeros(n_lut, np.uint64)
            for i, lut in enumerate(layer):
                widths[i] = lut.width
                out_cols[i] = col[lut.out]
                for j, b in enumerate(lut.inputs):
                    in_cols[i, j] = ccol(b)
                tab = _lut_table_array(lut)
                tables[i] = sum(int(v) << e for e, v in enumerate(tab))
            self.layers.append((
                torch.as_tensor(np.ascontiguousarray(in_cols.reshape(-1)), dtype=torch.int32, device=device),
                torch.as_tensor(out_cols, dtype=torch.int32, device=device),
                torch.as_tensor(widths, dtype=torch.uint8, device=device),
                torch.as_tensor(tables.view(np.int64), dtype=torch.int64, device=device),
            ))

        self.state_cols_i32 = torch.tensor([col[b] for b in nl.state_bits],
                                           dtype=torch.int32, device=device)
        self.input_cols_i32 = torch.tensor([col[b] for b in nl.input_bits],
                                           dtype=torch.int32, device=device)
        self.dff_cols_i32 = torch.tensor([ccol(ff.d) for ff in nl.dffs],
                                         dtype=torch.int32, device=device)
        state_set = set(self.state_cols_i32.cpu().numpy().tolist())
        dff_set = set(self.dff_cols_i32.cpu().numpy().tolist())
        if state_set & dff_set:
            raise NotImplementedError("CudaLutGraphSim currently assumes non-overlapping DFF/state columns")

        self.pb = po_bits(nl)
        self.po_cols_i32 = torch.tensor([ccol(b) for _, _, b in self.pb],
                                        dtype=torch.int32, device=device)
        self.V = torch.empty((batch, self.n_cols), dtype=torch.uint8, device=device)
        self.u_buf = torch.zeros((cycles, batch, nl.n_input), dtype=torch.uint8, device=device)
        self.po_out = (torch.empty((cycles, batch, len(self.pb)), dtype=torch.uint8, device=device)
                       if capture_po else None)
        self.final_state = torch.empty((batch, nl.n_state), dtype=torch.uint8, device=device)
        self._graph = None
        if use_cuda_graph:
            self._capture()

    def _body(self):
        lut_cuda.init_state(self.V, self.ZERO, self.ONE, self.state_cols_i32)
        for t in range(self.cycles):
            lut_cuda.scatter_inputs(self.V, self.input_cols_i32, self.u_buf[t])
            for in_cols, out_cols, widths, tables in self.layers:
                lut_cuda.eval_layer(self.V, in_cols, out_cols, widths, tables)
            if self.capture_po:
                lut_cuda.gather_cols(self.V, self.po_cols_i32, self.po_out[t])
            lut_cuda.commit_state(self.V, self.state_cols_i32, self.dff_cols_i32)
        if not self.capture_po:
            lut_cuda.gather_cols(self.V, self.state_cols_i32, self.final_state)

    def _capture(self):
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._body()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._body()
            self._graph = g
        except Exception:
            self._graph = None

    @property
    def has_graph(self):
        return self._graph is not None

    def load_inputs(self, u_seq):
        u = torch.as_tensor(np.asarray(u_seq, dtype=np.uint8), dtype=torch.uint8,
                            device=self.device)
        if self.nl.n_input:
            self.u_buf.copy_(u)

    def replay(self):
        if self._graph is not None:
            self._graph.replay()
        else:
            self._body()
        torch.cuda.synchronize()
        return self.po_out if self.capture_po else self.final_state

    def run(self, u_seq):
        self.load_inputs(u_seq)
        return self.replay()
