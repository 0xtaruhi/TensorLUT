"""k-LUT evaluation kernels.

Two equivalent primitives:

* ``lut_gather`` — exact table lookup (the correctness path used by the runtime):
  form the integer input index per stimulus and index the truth table.
* ``lut_onehot_gemm`` — the Tensor-Core framing used for the throughput story:
  a group of LUTs sharing one truth table becomes  out = onehot(idx) @ table,
  i.e. an INT8 GEMM. Validated against ``lut_gather``.
"""
from __future__ import annotations

import torch


def lut_table_tensor(lut, device) -> torch.Tensor:
    """Truth table as uint8 (2**width,) indexed by integer input value (LSB=A[0])."""
    s = lut.table
    n = 1 << lut.width
    vals = [int(s[len(s) - 1 - i]) for i in range(n)]
    return torch.tensor(vals, dtype=torch.uint8, device=device)


def lut_gather(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """table: (2**w,) uint8; idx: (batch,) int64 in [0,2**w). Returns (batch,) uint8."""
    return table[idx]


def lut_onehot_gemm(idx: torch.Tensor, tables: torch.Tensor) -> torch.Tensor:
    """Tensor-core LUT eval for LUTs sharing a truth table.

    idx:    (batch,) int64 input values in [0, 2**w)
    tables: (2**w, G) int8 — G LUTs (columns) that share this input group
    returns (batch, G) int8 = onehot(idx) @ tables  (each column = that LUT's output)
    """
    two_w, G = tables.shape
    batch = idx.shape[0]
    onehot = torch.zeros((batch, two_w), dtype=torch.int8, device=idx.device)
    onehot[torch.arange(batch, device=idx.device), idx] = 1
    if batch >= 17:
        Gp = (G + 7) // 8 * 8                      # _int_mm needs N a multiple of 8
        if Gp != G:
            tp = tables.new_zeros((two_w, Gp)); tp[:, :G] = tables
        else:
            tp = tables
        return torch._int_mm(onehot, tp.contiguous())[:, :G].to(torch.int8)
    return (onehot.to(torch.float32) @ tables.to(torch.float32)).to(torch.int8)
