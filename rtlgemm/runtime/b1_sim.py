"""Full-machine cycle-accurate simulator built on the b1 Tensor-Core ANF kernel.

Each combinational layer of the LUT netlist is compiled to its ANF and evaluated by the
b1 `and.popc` kernel (B1Anf); a resident V buffer (batch, n_nets) routes signals between
layers, and clock edges commit next-state. Primary outputs are read from V for the
differential check against Verilator/iverilog.

This wires the per-layer tensor-core primitive into a full multi-layer, multi-cycle sim.
"""
from __future__ import annotations

import numpy as np
import torch

from ..frontend.netlist import _const_val
from ..ir.anf import lut_anf
from ..ir.chunking import chunk_layer_luts, ordered_layer_luts
from ..kernels.b1_anf import (
    B1Anf,
    B1LayerV8,
    B1ProgramV8,
    B1BlockProgramV8,
    B1CoopProgramV8,
    v8_commit_direct,
    v8_scatter_cols,
)
from ..reference.interp import po_bits


def _ordered_layer_luts(layer, col):
    return ordered_layer_luts(layer, col)


def _layer_lut_chunks(layer, col, chunk_outputs):
    return chunk_layer_luts(layer, col, chunk_outputs)


class B1AnfSim:
    def __init__(self, plan, device="cuda"):
        nl = plan.nl
        self.nl = nl
        self.device = device
        # net -> column in V; two extra columns for constants 0 and 1
        col = {}
        for b in nl.state_bits: col.setdefault(b, len(col))
        for b in nl.input_bits: col.setdefault(b, len(col))
        for lut in nl.luts: col.setdefault(lut.out, len(col))
        self.ZERO = len(col); self.ONE = len(col) + 1
        self.n_cols = len(col) + 2

        def ccol(b):
            c = _const_val(b)
            if c is None:
                return col[b]
            return self.ONE if c == 1 else self.ZERO

        # per-layer ANF: monomials over the layer's input nets; C = which monomials each LUT XORs
        self.layers = []
        for layer in plan.layers:
            monos = [lut_anf(lut) for lut in layer]
            mset = {}
            for ms in monos:
                for m in ms:
                    mset.setdefault(m, len(mset))
            in_nets = sorted({v for m in mset for v in m})
            lc = {net: i for i, net in enumerate(in_nets)}
            F, NIN, G = len(mset), len(in_nets), len(layer)
            A = np.zeros((F, max(NIN, 1)), np.uint8)
            for m, k in mset.items():
                for v in m:
                    A[k, lc[v]] = 1
            C = np.zeros((G, F), np.uint8)
            for g, ms in enumerate(monos):
                for m in ms:
                    C[g, mset[m]] = 1
            self.layers.append(dict(
                b1=B1Anf(A, C, device=device),
                in_cols=torch.tensor([col[n] for n in in_nets] or [0], dtype=torch.long, device=device),
                in_cols_i32=torch.tensor([col[n] for n in in_nets] or [0], dtype=torch.int32, device=device),
                out_cols_i32=torch.tensor([col[lut.out] for lut in layer], dtype=torch.int32, device=device),
                out_cols=torch.tensor([col[lut.out] for lut in layer], dtype=torch.long, device=device)))

        self.state_cols = torch.tensor([col[b] for b in nl.state_bits], dtype=torch.long, device=device)
        self.input_cols = torch.tensor([col[b] for b in nl.input_bits], dtype=torch.long, device=device)
        self.dff_cols = torch.tensor([ccol(ff.d) for ff in nl.dffs], dtype=torch.long, device=device)
        self.pb = po_bits(nl)
        self.po_cols = torch.tensor([ccol(b) for _, _, b in self.pb], dtype=torch.long, device=device)

    def run(self, u_seq, cycles, capture_po=True):
        """u_seq: (cycles, batch, n_input) uint8.

        If capture_po is true, returns po (cycles, batch, n_po) uint8.  Otherwise
        returns only the final state, avoiding waveform-sized output traffic for
        throughput runs.
        x0 = 0 (reset driven via inputs)."""
        nl = self.nl
        u = torch.as_tensor(u_seq, dtype=torch.int8, device=self.device)
        batch = u.shape[1]
        V = torch.zeros((batch, self.n_cols), dtype=torch.int8, device=self.device)
        V[:, self.ONE] = 1
        po = (torch.empty((cycles, batch, len(self.pb)), dtype=torch.int8, device=self.device)
              if capture_po else None)
        for t in range(cycles):
            if nl.n_input:
                V[:, self.input_cols] = u[t]
            for L in self.layers:
                X = V[:, L["in_cols"]].contiguous()
                V[:, L["out_cols"]] = L["b1"].run_into(X)          # tensor-core ANF, no sync
            if capture_po:
                po[t] = V[:, self.po_cols] & 1
            V[:, self.state_cols] = V[:, self.dff_cols] & 1        # commit next state
        torch.cuda.synchronize()
        return po if capture_po else (V[:, self.state_cols] & 1)


class B1AnfGraphSim(B1AnfSim):
    """Compile-once b1 Tensor-Core simulator with resident V and CUDA Graph replay."""

    def __init__(self, plan, batch, cycles, device="cuda", capture_po=False, use_cuda_graph=True):
        super().__init__(plan, device)
        self.batch = batch
        self.cycles = cycles
        self.capture_po = capture_po
        nl = self.nl
        self.V = torch.zeros((batch, self.n_cols), dtype=torch.int8, device=device)
        self.u_buf = torch.zeros((cycles, nl.n_input, batch), dtype=torch.int8, device=device)
        self.po_out = (torch.empty((cycles, batch, len(self.pb)), dtype=torch.int8, device=device)
                       if capture_po else None)
        self.final_state = torch.empty((batch, nl.n_state), dtype=torch.int8, device=device)
        self._graph = None
        if use_cuda_graph:
            self._capture()

    def _body(self):
        nl = self.nl
        self.V.zero_()
        self.V[:, self.ONE] = 1
        for t in range(self.cycles):
            if nl.n_input:
                self.V[:, self.input_cols] = self.u_buf[t].t()
            for L in self.layers:
                L["b1"].run_v(self.V, L["in_cols_i32"], L["out_cols_i32"])
            if self.capture_po:
                self.po_out[t] = self.V[:, self.po_cols] & 1
            self.final_state.copy_(self.V[:, self.dff_cols] & 1)
            self.V[:, self.state_cols] = self.final_state

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

    def run(self, u_seq):
        u = torch.as_tensor(u_seq, dtype=torch.int8, device=self.device)
        if self.nl.n_input:
            self.u_buf.copy_(u.permute(0, 2, 1))
        if self._graph is not None:
            self._graph.replay()
        else:
            self._body()
        torch.cuda.synchronize()
        return self.po_out if self.capture_po else self.final_state


class B1AnfBufferedGraphSim(B1AnfSim):
    """CUDA Graph b1 simulator with fixed coalesced X/Y buffers per layer."""

    def __init__(self, plan, batch, cycles, device="cuda", capture_po=False, use_cuda_graph=True):
        super().__init__(plan, device)
        self.batch = batch
        self.cycles = cycles
        self.capture_po = capture_po
        nl = self.nl
        self.V = torch.zeros((batch, self.n_cols), dtype=torch.int8, device=device)
        self.u_buf = torch.zeros((cycles, batch, nl.n_input), dtype=torch.int8, device=device)
        self.x_bufs = [
            torch.empty((batch, L["b1"].NIN), dtype=torch.int8, device=device)
            for L in self.layers
        ]
        self.y_bufs = [
            torch.empty((batch, L["b1"].G), dtype=torch.int8, device=device)
            for L in self.layers
        ]
        self.po_out = (torch.empty((cycles, batch, len(self.pb)), dtype=torch.int8, device=device)
                       if capture_po else None)
        self.final_state = torch.empty((batch, nl.n_state), dtype=torch.int8, device=device)
        self._graph = None
        if use_cuda_graph:
            self._capture()

    def _body(self):
        nl = self.nl
        self.V.zero_()
        self.V[:, self.ONE] = 1
        for t in range(self.cycles):
            if nl.n_input:
                self.V.index_copy_(1, self.input_cols, self.u_buf[t])
            for i, L in enumerate(self.layers):
                torch.index_select(self.V, 1, L["in_cols"], out=self.x_bufs[i])
                L["b1"].run_into(self.x_bufs[i], self.y_bufs[i])
                self.V.index_copy_(1, L["out_cols"], self.y_bufs[i])
            if self.capture_po:
                torch.index_select(self.V, 1, self.po_cols, out=self.po_out[t])
            torch.index_select(self.V, 1, self.dff_cols, out=self.final_state)
            self.V.index_copy_(1, self.state_cols, self.final_state)

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

    def run(self, u_seq):
        u = torch.as_tensor(u_seq, dtype=torch.int8, device=self.device)
        if self.nl.n_input:
            self.u_buf.copy_(u)
        if self._graph is not None:
            self._graph.replay()
        else:
            self._body()
        torch.cuda.synchronize()
        return self.po_out if self.capture_po else self.final_state


class B1ChunkedSim(B1AnfSim):
    """b1 Tensor-Core simulator with local monomial matrices per output chunk."""

    def __init__(self, plan, device="cuda", chunk_outputs=64):
        nl = plan.nl
        self.nl = nl
        self.device = device
        self.chunk_outputs = chunk_outputs
        col = {}
        for b in nl.state_bits: col.setdefault(b, len(col))
        for b in nl.input_bits: col.setdefault(b, len(col))
        for lut in nl.luts: col.setdefault(lut.out, len(col))
        self.ZERO = len(col); self.ONE = len(col) + 1
        self.n_cols = len(col) + 2

        def ccol(b):
            c = _const_val(b)
            if c is None:
                return col[b]
            return self.ONE if c == 1 else self.ZERO

        self.layers = []
        for layer in plan.layers:
            chunks = []
            for chunk in _layer_lut_chunks(layer, col, chunk_outputs):
                monos = [lut_anf(lut) for lut in chunk]
                mset = {}
                for ms in monos:
                    for m in ms:
                        mset.setdefault(m, len(mset))
                in_nets = sorted({v for m in mset for v in m})
                lc = {net: i for i, net in enumerate(in_nets)}
                F, NIN, G = len(mset), len(in_nets), len(chunk)
                A = np.zeros((F, max(NIN, 1)), np.uint8)
                for m, k in mset.items():
                    for v in m:
                        A[k, lc[v]] = 1
                C = np.zeros((G, F), np.uint8)
                for g, ms in enumerate(monos):
                    for m in ms:
                        C[g, mset[m]] = 1
                chunks.append(dict(
                    b1=B1Anf(A, C, device=device),
                    in_cols=torch.tensor([col[n] for n in in_nets] or [0], dtype=torch.long, device=device),
                    in_cols_i32=torch.tensor([col[n] for n in in_nets] or [0], dtype=torch.int32, device=device),
                    out_cols=torch.tensor([col[lut.out] for lut in chunk], dtype=torch.long, device=device),
                    out_cols_i32=torch.tensor([col[lut.out] for lut in chunk], dtype=torch.int32, device=device)))
            self.layers.append(chunks)

        self.state_cols = torch.tensor([col[b] for b in nl.state_bits], dtype=torch.long, device=device)
        self.input_cols = torch.tensor([col[b] for b in nl.input_bits], dtype=torch.long, device=device)
        self.dff_cols = torch.tensor([ccol(ff.d) for ff in nl.dffs], dtype=torch.long, device=device)
        self.pb = po_bits(nl)
        self.po_cols = torch.tensor([ccol(b) for _, _, b in self.pb], dtype=torch.long, device=device)

    def run(self, u_seq, cycles, capture_po=True):
        nl = self.nl
        u = torch.as_tensor(u_seq, dtype=torch.int8, device=self.device)
        batch = u.shape[1]
        V = torch.zeros((batch, self.n_cols), dtype=torch.int8, device=self.device)
        V[:, self.ONE] = 1
        po = (torch.empty((cycles, batch, len(self.pb)), dtype=torch.int8, device=self.device)
              if capture_po else None)
        for t in range(cycles):
            if nl.n_input:
                V[:, self.input_cols] = u[t]
            for chunks in self.layers:
                for L in chunks:
                    X = V[:, L["in_cols"]].contiguous()
                    V[:, L["out_cols"]] = L["b1"].run_into(X)
            if capture_po:
                po[t] = V[:, self.po_cols] & 1
            V[:, self.state_cols] = V[:, self.dff_cols] & 1
        torch.cuda.synchronize()
        return po if capture_po else (V[:, self.state_cols] & 1)


class B1SharedInputChunkedSim(B1ChunkedSim):
    """Chunked b1 simulator that gathers each layer input once and reuses it."""

    def __init__(self, plan, device="cuda", chunk_outputs=256):
        nl = plan.nl
        self.nl = nl
        self.device = device
        self.chunk_outputs = chunk_outputs
        col = {}
        for b in nl.state_bits: col.setdefault(b, len(col))
        for b in nl.input_bits: col.setdefault(b, len(col))
        for lut in nl.luts: col.setdefault(lut.out, len(col))
        self.ZERO = len(col); self.ONE = len(col) + 1
        self.n_cols = len(col) + 2

        def ccol(b):
            c = _const_val(b)
            if c is None:
                return col[b]
            return self.ONE if c == 1 else self.ZERO

        self.layers = []
        for layer in plan.layers:
            layer_chunks = _layer_lut_chunks(layer, col, chunk_outputs)
            mono_by_id = {id(lut): lut_anf(lut) for chunk in layer_chunks for lut in chunk}
            layer_in_nets = sorted({
                v
                for ms in mono_by_id.values()
                for mm in ms
                for v in mm
            })
            layer_pos = {net: i for i, net in enumerate(layer_in_nets)}
            chunks = []
            for chunk in layer_chunks:
                monos = [mono_by_id[id(lut)] for lut in chunk]
                mset = {}
                for ms in monos:
                    for m in ms:
                        mset.setdefault(m, len(mset))
                in_nets = sorted({v for m in mset for v in m})
                lc = {net: i for i, net in enumerate(in_nets)}
                F, NIN, G = len(mset), len(in_nets), len(chunk)
                A = np.zeros((F, max(NIN, 1)), np.uint8)
                for m, k in mset.items():
                    for v in m:
                        A[k, lc[v]] = 1
                C = np.zeros((G, F), np.uint8)
                for g, ms in enumerate(monos):
                    for m in ms:
                        C[g, mset[m]] = 1
                chunks.append(dict(
                    b1=B1Anf(A, C, device=device),
                    x_cols=torch.tensor([layer_pos[n] for n in in_nets] or [0],
                                        dtype=torch.int32, device=device),
                    out_cols=torch.tensor([col[lut.out] for lut in chunk],
                                          dtype=torch.long, device=device)))
            self.layers.append(dict(
                in_cols=torch.tensor([col[n] for n in layer_in_nets] or [0],
                                     dtype=torch.long, device=device),
                chunks=chunks))

        self.state_cols = torch.tensor([col[b] for b in nl.state_bits], dtype=torch.long, device=device)
        self.input_cols = torch.tensor([col[b] for b in nl.input_bits], dtype=torch.long, device=device)
        self.dff_cols = torch.tensor([ccol(ff.d) for ff in nl.dffs], dtype=torch.long, device=device)
        self.pb = po_bits(nl)
        self.po_cols = torch.tensor([ccol(b) for _, _, b in self.pb], dtype=torch.long, device=device)

    def run(self, u_seq, cycles, capture_po=True):
        nl = self.nl
        u = torch.as_tensor(u_seq, dtype=torch.int8, device=self.device)
        batch = u.shape[1]
        V = torch.zeros((batch, self.n_cols), dtype=torch.int8, device=self.device)
        V[:, self.ONE] = 1
        po = (torch.empty((cycles, batch, len(self.pb)), dtype=torch.int8, device=self.device)
              if capture_po else None)
        for t in range(cycles):
            if nl.n_input:
                V[:, self.input_cols] = u[t]
            for L in self.layers:
                X_layer = V[:, L["in_cols"]].contiguous()
                for chunk in L["chunks"]:
                    Y = chunk["b1"].run_xcols_into(X_layer, chunk["x_cols"])
                    V[:, chunk["out_cols"]] = Y
            if capture_po:
                po[t] = V[:, self.po_cols] & 1
            V[:, self.state_cols] = V[:, self.dff_cols] & 1
        torch.cuda.synchronize()
        return po if capture_po else (V[:, self.state_cols] & 1)


class B1ChunkedStreamSim(B1ChunkedSim):
    """Chunked b1 simulator that runs independent chunks of each layer on streams."""

    def __init__(self, plan, device="cuda", chunk_outputs=256, num_streams=4):
        super().__init__(plan, device, chunk_outputs)
        self.streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]

    def run(self, u_seq, cycles, capture_po=True):
        nl = self.nl
        u = torch.as_tensor(u_seq, dtype=torch.int8, device=self.device)
        batch = u.shape[1]
        V = torch.zeros((batch, self.n_cols), dtype=torch.int8, device=self.device)
        V[:, self.ONE] = 1
        po = (torch.empty((cycles, batch, len(self.pb)), dtype=torch.int8, device=self.device)
              if capture_po else None)
        default = torch.cuda.current_stream(V.device)
        for t in range(cycles):
            if nl.n_input:
                V[:, self.input_cols] = u[t]
            for chunks in self.layers:
                for stream in self.streams:
                    stream.wait_stream(default)
                for ci, L in enumerate(chunks):
                    stream = self.streams[ci % len(self.streams)]
                    with torch.cuda.stream(stream):
                        X = V[:, L["in_cols"]].contiguous()
                        Y = L["b1"].run_into(X)
                        V[:, L["out_cols"]] = Y
                for stream in self.streams:
                    default.wait_stream(stream)
            if capture_po:
                po[t] = V[:, self.po_cols] & 1
            V[:, self.state_cols] = V[:, self.dff_cols] & 1
        torch.cuda.synchronize()
        return po if capture_po else (V[:, self.state_cols] & 1)


class B1ChunkedNoScatterSim(B1ChunkedSim):
    """Chunked b1 simulator whose kernels write outputs directly back to V."""

    def run(self, u_seq, cycles, capture_po=True):
        nl = self.nl
        u = torch.as_tensor(u_seq, dtype=torch.int8, device=self.device)
        batch = u.shape[1]
        V = torch.zeros((batch, self.n_cols), dtype=torch.int8, device=self.device)
        V[:, self.ONE] = 1
        po = (torch.empty((cycles, batch, len(self.pb)), dtype=torch.int8, device=self.device)
              if capture_po else None)
        for t in range(cycles):
            if nl.n_input:
                V[:, self.input_cols] = u[t]
            for chunks in self.layers:
                for L in chunks:
                    X = V[:, L["in_cols"]].contiguous()
                    L["b1"].run_into_v(X, V, L["out_cols_i32"])
            if capture_po:
                po[t] = V[:, self.po_cols] & 1
            V[:, self.state_cols] = V[:, self.dff_cols] & 1
        torch.cuda.synchronize()
        return po if capture_po else (V[:, self.state_cols] & 1)


class B1ChunkedNoScatterGraphSim(B1ChunkedSim):
    """Graph-captured chunked b1 simulator with fixed gather buffers."""

    def __init__(self, plan, batch, cycles, device="cuda", chunk_outputs=256,
                 capture_po=False, use_cuda_graph=True):
        super().__init__(plan, device, chunk_outputs)
        self.batch = batch
        self.cycles = cycles
        self.capture_po = capture_po
        nl = self.nl
        self.V = torch.zeros((batch, self.n_cols), dtype=torch.int8, device=device)
        self.u_buf = torch.zeros((cycles, batch, nl.n_input), dtype=torch.int8, device=device)
        self.x_bufs = [
            [
                torch.empty((batch, L["b1"].NIN), dtype=torch.int8, device=device)
                for L in chunks
            ]
            for chunks in self.layers
        ]
        self.po_out = (torch.empty((cycles, batch, len(self.pb)), dtype=torch.int8, device=device)
                       if capture_po else None)
        self.final_state = torch.empty((batch, nl.n_state), dtype=torch.int8, device=device)
        self._graph = None
        if use_cuda_graph:
            self._capture()

    def _body(self):
        nl = self.nl
        self.V.zero_()
        self.V.select(1, self.ONE).fill_(1)
        for t in range(self.cycles):
            if nl.n_input:
                self.V.index_copy_(1, self.input_cols, self.u_buf[t])
            for li, chunks in enumerate(self.layers):
                for ci, L in enumerate(chunks):
                    torch.index_select(self.V, 1, L["in_cols"], out=self.x_bufs[li][ci])
                    L["b1"].run_into_v(self.x_bufs[li][ci], self.V, L["out_cols_i32"])
            if self.capture_po:
                torch.index_select(self.V, 1, self.po_cols, out=self.po_out[t])
            torch.index_select(self.V, 1, self.dff_cols, out=self.final_state)
            self.V.index_copy_(1, self.state_cols, self.final_state)

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

    def run(self, u_seq):
        u = torch.as_tensor(u_seq, dtype=torch.int8, device=self.device)
        if self.nl.n_input:
            self.u_buf.copy_(u)
        if self._graph is not None:
            self._graph.replay()
        else:
            self._body()
        torch.cuda.synchronize()
        return self.po_out if self.capture_po else self.final_state


class B1ChunkedDirectVSim(B1ChunkedSim):
    """Chunked b1 simulator that reads and writes resident V inside the kernel."""

    def run(self, u_seq, cycles, capture_po=True):
        nl = self.nl
        u = torch.as_tensor(u_seq, dtype=torch.int8, device=self.device)
        batch = u.shape[1]
        V = torch.zeros((batch, self.n_cols), dtype=torch.int8, device=self.device)
        V[:, self.ONE] = 1
        po = (torch.empty((cycles, batch, len(self.pb)), dtype=torch.int8, device=self.device)
              if capture_po else None)
        for t in range(cycles):
            if nl.n_input:
                V[:, self.input_cols] = u[t]
            for chunks in self.layers:
                for L in chunks:
                    L["b1"].run_v(V, L["in_cols_i32"], L["out_cols_i32"])
            if capture_po:
                po[t] = V[:, self.po_cols] & 1
            V[:, self.state_cols] = V[:, self.dff_cols] & 1
        torch.cuda.synchronize()
        return po if capture_po else (V[:, self.state_cols] & 1)


class B1Packed8Sim(B1ChunkedSim):
    """Chunked b1 simulator with an 8-stimulus packed resident state.

    The packed byte is aligned with the b1 Tensor-Core tile height (WM=8), so the
    kernel reads one byte per input net per stimulus tile and writes one byte per
    output net per tile.
    """

    @staticmethod
    def _pack_u8(u_seq):
        u = np.asarray(u_seq, dtype=np.uint8)
        cycles, batch, n_input = u.shape
        n_tiles = (batch + 7) // 8
        if n_tiles * 8 != batch:
            u = np.pad(u, ((0, 0), (0, n_tiles * 8 - batch), (0, 0)))
        bits = (np.uint8(1) << np.arange(8, dtype=np.uint8))
        packed = (u.reshape(cycles, n_tiles, 8, n_input) *
                  bits[None, None, :, None]).sum(axis=2).astype(np.uint8)
        return np.ascontiguousarray(packed)

    @staticmethod
    def _unpack_tiles(packed, batch):
        bits = torch.arange(8, dtype=torch.uint8, device=packed.device)
        u = (packed.unsqueeze(2) >> bits.view(1, 1, 8, 1)) & 1
        return u.reshape(packed.shape[0], packed.shape[1] * 8, packed.shape[2])[:, :batch].to(torch.int8)

    def run(self, u_seq, cycles, capture_po=True):
        nl = self.nl
        batch = u_seq.shape[1]
        n_tiles = (batch + 7) // 8
        up = torch.as_tensor(self._pack_u8(u_seq), dtype=torch.uint8, device=self.device)
        V = torch.zeros((n_tiles, self.n_cols), dtype=torch.uint8, device=self.device)
        V[:, self.ONE] = 0xff
        po_pack = (torch.empty((cycles, n_tiles, len(self.pb)), dtype=torch.uint8, device=self.device)
                   if capture_po else None)
        for t in range(cycles):
            if nl.n_input:
                V[:, self.input_cols] = up[t]
            for chunks in self.layers:
                for L in chunks:
                    L["b1"].run_v8(V, L["in_cols_i32"], L["out_cols_i32"])
            if capture_po:
                po_pack[t] = V[:, self.po_cols]
            V[:, self.state_cols] = V[:, self.dff_cols]
        torch.cuda.synchronize()
        return self._unpack_tiles(po_pack, batch) if capture_po else V[:, self.state_cols]


class B1LayerPacked8Sim:
    """Packed-v8 simulator that fuses all chunks of each layer into one kernel."""

    _pack_u8 = staticmethod(B1Packed8Sim._pack_u8)
    _unpack_tiles = staticmethod(B1Packed8Sim._unpack_tiles)

    def __init__(self, plan, device="cuda", chunk_outputs=64):
        nl = plan.nl
        self.nl = nl
        self.device = device
        self.chunk_outputs = chunk_outputs
        col = {}
        for b in nl.state_bits: col.setdefault(b, len(col))
        for b in nl.input_bits: col.setdefault(b, len(col))
        for lut in nl.luts: col.setdefault(lut.out, len(col))
        self.ZERO = len(col); self.ONE = len(col) + 1
        self.n_cols = len(col) + 2

        def ccol(b):
            c = _const_val(b)
            if c is None:
                return col[b]
            return self.ONE if c == 1 else self.ZERO

        self.layers = []
        for layer in plan.layers:
            chunks = []
            for chunk in _layer_lut_chunks(layer, col, chunk_outputs):
                monos = [lut_anf(lut) for lut in chunk]
                mset = {}
                for ms in monos:
                    for m in ms:
                        mset.setdefault(m, len(mset))
                in_nets = sorted({v for m in mset for v in m})
                lc = {net: i for i, net in enumerate(in_nets)}
                F, NIN, G = len(mset), len(in_nets), len(chunk)
                A = np.zeros((F, max(NIN, 1)), np.uint8)
                for m, k in mset.items():
                    for v in m:
                        A[k, lc[v]] = 1
                C = np.zeros((G, F), np.uint8)
                for g, ms in enumerate(monos):
                    for m in ms:
                        C[g, mset[m]] = 1
                chunks.append((
                    A,
                    C,
                    np.array([col[n] for n in in_nets] or [0], np.int32),
                    np.array([col[lut.out] for lut in chunk], np.int32),
                ))
            self.layers.append(B1LayerV8(chunks, device=device))

        self.state_cols = torch.tensor([col[b] for b in nl.state_bits], dtype=torch.long, device=device)
        self.input_cols = torch.tensor([col[b] for b in nl.input_bits], dtype=torch.long, device=device)
        self.dff_cols = torch.tensor([ccol(ff.d) for ff in nl.dffs], dtype=torch.long, device=device)
        self.pb = po_bits(nl)
        self.po_cols = torch.tensor([ccol(b) for _, _, b in self.pb], dtype=torch.long, device=device)

    def run(self, u_seq, cycles, capture_po=True):
        nl = self.nl
        batch = u_seq.shape[1]
        n_tiles = (batch + 7) // 8
        up = torch.as_tensor(self._pack_u8(u_seq), dtype=torch.uint8, device=self.device)
        V = torch.zeros((n_tiles, self.n_cols), dtype=torch.uint8, device=self.device)
        V[:, self.ONE] = 0xff
        po_pack = (torch.empty((cycles, n_tiles, len(self.pb)), dtype=torch.uint8, device=self.device)
                   if capture_po else None)
        for t in range(cycles):
            if nl.n_input:
                V[:, self.input_cols] = up[t]
            for L in self.layers:
                L.run_v8(V)
            if capture_po:
                po_pack[t] = V[:, self.po_cols]
            V[:, self.state_cols] = V[:, self.dff_cols]
        torch.cuda.synchronize()
        return self._unpack_tiles(po_pack, batch) if capture_po else V[:, self.state_cols]


class B1LayerPacked8GraphSim(B1LayerPacked8Sim):
    """Graph-captured layer-fused packed-v8 simulator."""

    def __init__(self, plan, batch, cycles, device="cuda", chunk_outputs=64,
                 capture_po=False, use_cuda_graph=True):
        super().__init__(plan, device, chunk_outputs)
        self.batch = batch
        self.cycles = cycles
        self.n_tiles = (batch + 7) // 8
        self.capture_po = capture_po
        nl = self.nl
        self.V = torch.zeros((self.n_tiles, self.n_cols), dtype=torch.uint8, device=device)
        self.u_buf = torch.zeros((cycles, self.n_tiles, nl.n_input), dtype=torch.uint8, device=device)
        self.po_pack = (torch.empty((cycles, self.n_tiles, len(self.pb)), dtype=torch.uint8, device=device)
                        if capture_po else None)
        self.final_state = torch.empty((self.n_tiles, nl.n_state), dtype=torch.uint8, device=device)
        self.input_cols_i32 = self.input_cols.to(torch.int32)
        self.state_cols_i32 = self.state_cols.to(torch.int32)
        self.dff_cols_i32 = self.dff_cols.to(torch.int32)
        state_set = set(self.state_cols_i32.cpu().numpy().tolist())
        dff_set = set(self.dff_cols_i32.cpu().numpy().tolist())
        self.direct_commit = not bool(state_set & dff_set)
        self._graph = None
        if use_cuda_graph:
            self._capture()

    def _body(self):
        nl = self.nl
        self.V.zero_()
        self.V.select(1, self.ONE).fill_(0xff)
        for t in range(self.cycles):
            if nl.n_input:
                v8_scatter_cols(self.V, self.input_cols_i32, self.u_buf[t])
            for L in self.layers:
                L.run_v8(self.V)
            if self.capture_po:
                torch.index_select(self.V, 1, self.po_cols, out=self.po_pack[t])
            if self.direct_commit:
                v8_commit_direct(self.V, self.state_cols_i32, self.dff_cols_i32)
            else:
                torch.index_select(self.V, 1, self.dff_cols, out=self.final_state)
                self.V.index_copy_(1, self.state_cols, self.final_state)
        if self.direct_commit and not self.capture_po:
            torch.index_select(self.V, 1, self.state_cols, out=self.final_state)

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
        # Pack one cycle at a time: large batches would otherwise materialize several
        # multi-GB host temporaries.
        if self.nl.n_input:
            for t in range(u_seq.shape[0]):
                self.u_buf[t].copy_(torch.from_numpy(self._pack_u8(u_seq[t:t + 1])[0]))

    def replay(self):
        if self._graph is not None:
            self._graph.replay()
        else:
            self._body()
        torch.cuda.synchronize()
        return self.po_pack if self.capture_po else self.final_state

    def run(self, u_seq):
        self.load_inputs(u_seq)
        out = self.replay()
        return self._unpack_tiles(out, self.batch) if self.capture_po else out


class B1ProgramPacked8Sim:
    """Single-kernel packed-v8 simulator for the whole layer/cycle program."""

    _pack_u8 = staticmethod(B1Packed8Sim._pack_u8)
    _unpack_tiles = staticmethod(B1Packed8Sim._unpack_tiles)

    def __init__(self, plan, batch, cycles, device="cuda", chunk_outputs=24,
                 capture_po=False):
        nl = plan.nl
        self.nl = nl
        self.device = device
        self.batch = batch
        self.cycles = cycles
        self.n_tiles = (batch + 7) // 8
        self.capture_po = capture_po
        col = {}
        for b in nl.state_bits: col.setdefault(b, len(col))
        for b in nl.input_bits: col.setdefault(b, len(col))
        for lut in nl.luts: col.setdefault(lut.out, len(col))
        self.ZERO = len(col); self.ONE = len(col) + 1
        self.n_cols = len(col) + 2

        def ccol(b):
            c = _const_val(b)
            if c is None:
                return col[b]
            return self.ONE if c == 1 else self.ZERO

        layer_chunks = []
        for layer in plan.layers:
            chunks = []
            for chunk in _layer_lut_chunks(layer, col, chunk_outputs):
                monos = [lut_anf(lut) for lut in chunk]
                mset = {}
                for ms in monos:
                    for m in ms:
                        mset.setdefault(m, len(mset))
                in_nets = sorted({v for m in mset for v in m})
                lc = {net: i for i, net in enumerate(in_nets)}
                F, NIN, G = len(mset), len(in_nets), len(chunk)
                A = np.zeros((F, max(NIN, 1)), np.uint8)
                for m, k in mset.items():
                    for v in m:
                        A[k, lc[v]] = 1
                C = np.zeros((G, F), np.uint8)
                for g, ms in enumerate(monos):
                    for m in ms:
                        C[g, mset[m]] = 1
                chunks.append((
                    A,
                    C,
                    np.array([col[n] for n in in_nets] or [0], np.int32),
                    np.array([col[lut.out] for lut in chunk], np.int32),
                ))
            layer_chunks.append(chunks)

        self.prog_layer_chunks = layer_chunks
        self.state_cols_np = np.array([col[b] for b in nl.state_bits], np.int32)
        self.dff_cols_np = np.array([ccol(ff.d) for ff in nl.dffs], np.int32)
        self.input_cols_np = np.array([col[b] for b in nl.input_bits], np.int32)
        self.pb = po_bits(nl)
        self.po_cols_np = np.array([ccol(b) for _, _, b in self.pb], np.int32)
        self.prog = B1ProgramV8(
            layer_chunks, self.n_cols, self.ONE, self.state_cols_np, self.dff_cols_np,
            self.input_cols_np, self.po_cols_np, device=device)
        self.u_buf = torch.zeros((cycles, self.n_tiles, nl.n_input), dtype=torch.uint8, device=device)
        self.V = torch.empty((self.n_tiles, self.n_cols), dtype=torch.uint8, device=device)
        self.po_pack = (torch.empty((cycles, self.n_tiles, len(self.pb)), dtype=torch.uint8, device=device)
                        if capture_po else torch.empty((1,), dtype=torch.uint8, device=device))
        self.final_state = torch.empty((self.n_tiles, nl.n_state), dtype=torch.uint8, device=device)

    def load_inputs(self, u_seq):
        # Pack one cycle at a time: large batches would otherwise materialize several
        # multi-GB host temporaries.
        if self.nl.n_input:
            for t in range(u_seq.shape[0]):
                self.u_buf[t].copy_(torch.from_numpy(self._pack_u8(u_seq[t:t + 1])[0]))

    def replay(self):
        self.prog.run(self.V, self.u_buf, self.po_pack, self.final_state,
                      self.cycles, self.capture_po)
        torch.cuda.synchronize()
        return self.po_pack if self.capture_po else self.final_state

    def run(self, u_seq):
        self.load_inputs(u_seq)
        out = self.replay()
        return self._unpack_tiles(out, self.batch) if self.capture_po else out


class B1BlockProgramPacked8Sim(B1ProgramPacked8Sim):
    """Single-kernel simulator with one cooperative block per packed stimulus tile."""

    def __init__(self, plan, batch, cycles, device="cuda", chunk_outputs=24,
                 capture_po=False, warps_per_block=8):
        super().__init__(plan, batch, cycles, device, chunk_outputs, capture_po)
        self.prog = B1BlockProgramV8(
            self.prog_layer_chunks, self.n_cols, self.ONE,
            self.state_cols_np, self.dff_cols_np, self.input_cols_np, self.po_cols_np,
            device=device, warps_per_block=warps_per_block)

    def replay(self):
        self.prog.run_block(self.u_buf, self.po_pack, self.final_state,
                            self.cycles, self.capture_po)
        torch.cuda.synchronize()
        return self.po_pack if self.capture_po else self.final_state


class B1CoopProgramPacked8Sim(B1ProgramPacked8Sim):
    """Cooperative-grid single-kernel simulator with global sync between layers."""

    def __init__(self, plan, batch, cycles, device="cuda", chunk_outputs=24,
                 capture_po=False, warps_per_block=8, grid_blocks=128):
        super().__init__(plan, batch, cycles, device, chunk_outputs, capture_po)
        self.prog = B1CoopProgramV8(
            self.prog_layer_chunks, self.n_cols, self.ONE,
            self.state_cols_np, self.dff_cols_np, self.input_cols_np, self.po_cols_np,
            device=device, warps_per_block=warps_per_block, grid_blocks=grid_blocks)

    def replay(self):
        self.prog.run_coop(self.V, self.u_buf, self.po_pack, self.final_state,
                           self.cycles, self.capture_po)
        torch.cuda.synchronize()
        return self.po_pack if self.capture_po else self.final_state


class B1Packed8StreamSim(B1Packed8Sim):
    """Packed-v8 simulator with per-layer multi-stream chunk parallelism."""

    def __init__(self, plan, device="cuda", chunk_outputs=256, num_streams=4):
        super().__init__(plan, device, chunk_outputs)
        self.streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]

    def run(self, u_seq, cycles, capture_po=True):
        nl = self.nl
        batch = u_seq.shape[1]
        n_tiles = (batch + 7) // 8
        up = torch.as_tensor(self._pack_u8(u_seq), dtype=torch.uint8, device=self.device)
        V = torch.zeros((n_tiles, self.n_cols), dtype=torch.uint8, device=self.device)
        V[:, self.ONE] = 0xff
        po_pack = (torch.empty((cycles, n_tiles, len(self.pb)), dtype=torch.uint8, device=self.device)
                   if capture_po else None)
        default = torch.cuda.current_stream(V.device)
        for t in range(cycles):
            if nl.n_input:
                V[:, self.input_cols] = up[t]
            for chunks in self.layers:
                for stream in self.streams:
                    stream.wait_stream(default)
                for ci, L in enumerate(chunks):
                    stream = self.streams[ci % len(self.streams)]
                    with torch.cuda.stream(stream):
                        L["b1"].run_v8(V, L["in_cols_i32"], L["out_cols_i32"])
                for stream in self.streams:
                    default.wait_stream(stream)
            if capture_po:
                po_pack[t] = V[:, self.po_cols]
            V[:, self.state_cols] = V[:, self.dff_cols]
        torch.cuda.synchronize()
        return self._unpack_tiles(po_pack, batch) if capture_po else V[:, self.state_cols]


class B1Packed8GraphSim(B1Packed8Sim):
    """Graph-captured packed-v8 simulator with resident packed input/state buffers."""

    def __init__(self, plan, batch, cycles, device="cuda", chunk_outputs=256,
                 capture_po=False, use_cuda_graph=True):
        super().__init__(plan, device, chunk_outputs)
        self.batch = batch
        self.cycles = cycles
        self.n_tiles = (batch + 7) // 8
        self.capture_po = capture_po
        nl = self.nl
        self.V = torch.zeros((self.n_tiles, self.n_cols), dtype=torch.uint8, device=device)
        self.u_buf = torch.zeros((cycles, self.n_tiles, nl.n_input), dtype=torch.uint8, device=device)
        self.po_pack = (torch.empty((cycles, self.n_tiles, len(self.pb)), dtype=torch.uint8, device=device)
                        if capture_po else None)
        self.final_state = torch.empty((self.n_tiles, nl.n_state), dtype=torch.uint8, device=device)
        self._graph = None
        if use_cuda_graph:
            self._capture()

    def _body(self):
        nl = self.nl
        self.V.zero_()
        self.V.select(1, self.ONE).fill_(0xff)
        for t in range(self.cycles):
            if nl.n_input:
                self.V.index_copy_(1, self.input_cols, self.u_buf[t])
            for chunks in self.layers:
                for L in chunks:
                    L["b1"].run_v8(self.V, L["in_cols_i32"], L["out_cols_i32"])
            if self.capture_po:
                torch.index_select(self.V, 1, self.po_cols, out=self.po_pack[t])
            torch.index_select(self.V, 1, self.dff_cols, out=self.final_state)
            self.V.index_copy_(1, self.state_cols, self.final_state)

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
        # Pack one cycle at a time: large batches would otherwise materialize several
        # multi-GB host temporaries.
        if self.nl.n_input:
            for t in range(u_seq.shape[0]):
                self.u_buf[t].copy_(torch.from_numpy(self._pack_u8(u_seq[t:t + 1])[0]))

    def replay(self):
        if self._graph is not None:
            self._graph.replay()
        else:
            self._body()
        torch.cuda.synchronize()
        return self.po_pack if self.capture_po else self.final_state

    def run(self, u_seq):
        self.load_inputs(u_seq)
        out = self.replay()
        return self._unpack_tiles(out, self.batch) if self.capture_po else out


class B1ChunkedNoScatterStreamSim(B1ChunkedStreamSim):
    """No-scatter chunked simulator with per-layer multi-stream chunk parallelism."""

    def run(self, u_seq, cycles, capture_po=True):
        nl = self.nl
        u = torch.as_tensor(u_seq, dtype=torch.int8, device=self.device)
        batch = u.shape[1]
        V = torch.zeros((batch, self.n_cols), dtype=torch.int8, device=self.device)
        V[:, self.ONE] = 1
        po = (torch.empty((cycles, batch, len(self.pb)), dtype=torch.int8, device=self.device)
              if capture_po else None)
        default = torch.cuda.current_stream(V.device)
        for t in range(cycles):
            if nl.n_input:
                V[:, self.input_cols] = u[t]
            for chunks in self.layers:
                for stream in self.streams:
                    stream.wait_stream(default)
                for ci, L in enumerate(chunks):
                    stream = self.streams[ci % len(self.streams)]
                    with torch.cuda.stream(stream):
                        X = V[:, L["in_cols"]].contiguous()
                        L["b1"].run_into_v(X, V, L["out_cols_i32"])
                for stream in self.streams:
                    default.wait_stream(stream)
            if capture_po:
                po[t] = V[:, self.po_cols] & 1
            V[:, self.state_cols] = V[:, self.dff_cols] & 1
        torch.cuda.synchronize()
        return po if capture_po else (V[:, self.state_cols] & 1)
