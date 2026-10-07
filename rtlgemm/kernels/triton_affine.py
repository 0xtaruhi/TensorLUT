"""Fused GF(2) affine recurrence as a single Triton kernel.

Instead of one INT8 GEMM per cycle (state round-tripped through global memory), pack
the whole feature vector [state | inputs | const] into one 64-bit word, keep the state
in registers, and iterate *all* cycles inside one kernel. Each next-state bit is
``parity(feat & rowmask[i])`` computed by shift-XOR folding. One launch, no matmul,
no per-cycle global state traffic (only the packed trace is written).

Requires F = n_state + n_input + 1 <= 64 (single-word packing).
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _affine_recur_packed(x0_ptr, u_ptr, rmask_ptr, wts_ptr, out_ptr,
                             batch, cycles, CONST, NPOW2: tl.constexpr, BLOCK_B: tl.constexpr):
        """Write the state trace in packed int64 form (fastest; consumer unpacks)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        mb = offs < batch
        rows = tl.arange(0, NPOW2)
        rmask = tl.load(rmask_ptr + rows)
        wts = tl.load(wts_ptr + rows)
        state = tl.load(x0_ptr + offs, mask=mb, other=0)
        tl.store(out_ptr + offs, state, mask=mb)
        for t in range(cycles):
            uf = tl.load(u_ptr + t * batch + offs, mask=mb, other=0)
            feat = state | uf | CONST
            x = feat[:, None] & rmask[None, :]
            x = x ^ (x >> 32); x = x ^ (x >> 16); x = x ^ (x >> 8)
            x = x ^ (x >> 4); x = x ^ (x >> 2); x = x ^ (x >> 1)
            bits = x & 1
            state = tl.sum(bits * wts[None, :], axis=1)
            tl.store(out_ptr + (t + 1) * batch + offs, state, mask=mb)

    @triton.jit
    def _affine_recur_i8(x0_ptr, u_ptr, rmask_ptr, wts_ptr, out_ptr,
                         batch, cycles, CONST, N, NPOW2: tl.constexpr, BLOCK_B: tl.constexpr):
        """State stays packed in registers for the recurrence, but the trace is written
        as unpacked int8 (cycles+1, batch, N) directly — no separate unpack pass."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)          # (BLOCK_B,)
        mb = offs < batch
        rows = tl.arange(0, NPOW2)                            # (NPOW2,)
        rmask = tl.load(rmask_ptr + rows)
        wts = tl.load(wts_ptr + rows)
        smask = mb[:, None] & (rows[None, :] < N)             # valid (elem, bit) lanes
        rowbase = offs[:, None] * N + rows[None, :]           # (BLOCK_B, NPOW2)

        state = tl.load(x0_ptr + offs, mask=mb, other=0)      # (BLOCK_B,) packed
        ibits = ((state[:, None] >> rows[None, :]) & 1).to(tl.int8)
        tl.store(out_ptr + rowbase, ibits, mask=smask)        # trace row t=0
        for t in range(cycles):
            uf = tl.load(u_ptr + t * batch + offs, mask=mb, other=0)
            feat = state | uf | CONST
            x = feat[:, None] & rmask[None, :]                # (BLOCK_B, NPOW2)
            x = x ^ (x >> 32); x = x ^ (x >> 16); x = x ^ (x >> 8)
            x = x ^ (x >> 4); x = x ^ (x >> 2); x = x ^ (x >> 1)
            bits = x & 1
            state = tl.sum(bits * wts[None, :], axis=1)       # repack (register)
            tl.store(out_ptr + (t + 1) * batch * N + rowbase, bits.to(tl.int8), mask=smask)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class TritonAffine:
    """Compiled fused affine simulator (packed int64 state trace)."""

    def __init__(self, plan, batch, cycles, device="cuda", block_b=256):
        aff = plan.affine
        self.n, self.m, self.F = aff.n_state, aff.n_input, aff.F
        assert self.F <= 64, f"F={self.F} exceeds single-word packing (<=64)"
        self.batch, self.cycles, self.device = batch, cycles, device
        self.block_b = block_b
        self.const = 1 << (self.n + self.m)
        self.npow2 = _next_pow2(self.n)

        M = torch.as_tensor(aff.M, dtype=torch.int64, device=device)   # (n, F) 0/1
        fbits = (torch.arange(self.F, device=device, dtype=torch.int64))
        rmask = (M << fbits).sum(dim=1)                                # (n,) row bitmasks
        self.rmask = torch.zeros(self.npow2, dtype=torch.int64, device=device)
        self.rmask[:self.n] = rmask
        self.wts = (torch.arange(self.npow2, device=device, dtype=torch.int64))
        self.wts = (torch.ones_like(self.wts) << self.wts)             # 1<<i
        self.out_packed = torch.empty((cycles + 1, batch), dtype=torch.int64, device=device)
        self.out_i8 = torch.empty((cycles + 1, batch, self.n), dtype=torch.int8, device=device)

    def _pack_x0(self, x0):
        x0 = torch.as_tensor(x0, dtype=torch.int64, device=self.device)  # (batch, n)
        bits = torch.arange(self.n, device=self.device, dtype=torch.int64)
        return (x0 << bits).sum(dim=1)                                   # (batch,)

    def _pack_u(self, u_seq):
        if self.m == 0:
            return torch.zeros((self.cycles, self.batch), dtype=torch.int64, device=self.device)
        u = torch.as_tensor(u_seq, dtype=torch.int64, device=self.device)  # (C,batch,m)
        bits = torch.arange(self.m, device=self.device, dtype=torch.int64)
        return (u << (self.n + bits)).sum(dim=2)                           # (C,batch) shifted into feature slot

    def _grid(self):
        return ((self.batch + self.block_b - 1) // self.block_b,)

    def run_packed(self, x0, u_seq):
        """Return packed state trace (cycles+1, batch) int64 (fastest path)."""
        _affine_recur_packed[self._grid()](
            self._pack_x0(x0), self._pack_u(u_seq), self.rmask, self.wts,
            self.out_packed, self.batch, self.cycles, self.const,
            NPOW2=self.npow2, BLOCK_B=self.block_b)
        return self.out_packed

    def run(self, x0, u_seq):
        """Return unpacked states (cycles+1, batch, n_state) int8 (int8 trace written
        directly in-kernel — no separate unpack pass)."""
        _affine_recur_i8[self._grid()](
            self._pack_x0(x0), self._pack_u(u_seq), self.rmask, self.wts,
            self.out_i8, self.batch, self.cycles, self.const, self.n,
            NPOW2=self.npow2, BLOCK_B=self.block_b)
        return self.out_i8
