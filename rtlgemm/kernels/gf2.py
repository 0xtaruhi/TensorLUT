"""GF(2) GEMM on the Tensor Core via INT8 matmul + parity (mod 2).

For 0/1 operands, ``torch._int_mm`` (INT8 in, INT32 accumulate, runs on the Ada
tensor cores) computes the integer dot product Σ_k A[i,k]·B[k,j]; taking ``& 1``
recovers the GF(2) inner product (XOR-accumulate). This sidesteps the fragile
1-bit MMA path entirely and is bit-exact.

``_int_mm`` requires the first operand to have > 16 rows, so we keep the batch
dimension first: out(batch, N) = feats(batch, K) @ W(K, N). Small batches fall
back to an exact FP32 matmul (still on GPU) to preserve generality.
"""
from __future__ import annotations

import torch

_INT_MM_MIN_ROWS = 17    # _int_mm requires the first operand to have > 16 rows
_INT_MM_MULT = 8         # _int_mm requires K (and N) to be multiples of 8
_INT_MM_MAX_ROWS = 32768  # cuBLASLt int8 caps the row dim below 65536; chunk under it


def _pad8(n: int) -> int:
    return (n + _INT_MM_MULT - 1) // _INT_MM_MULT * _INT_MM_MULT


def gf2_matmul_mod2(feats: torch.Tensor, w_kn: torch.Tensor) -> torch.Tensor:
    """Return (feats @ w_kn) mod 2 as int8.

    feats: (batch, K) int8 on cuda, entries in {0,1}
    w_kn:  (K, N)     int8 on cuda, entries in {0,1}  (already transposed for the GEMM)

    ``_int_mm`` needs batch > 16 and K, N multiples of 8; we zero-pad (harmless over
    GF(2)) and slice the result back. Small batches use an exact FP32 GPU fallback.
    """
    return (int8_gemm_i32(feats, w_kn) & 1).to(torch.int8)


def int8_gemm_i32(feats: torch.Tensor, w_kn: torch.Tensor) -> torch.Tensor:
    """INT8 tensor-core GEMM returning the raw INT32 accumulation (feats @ w_kn).

    feats: (batch, K) int8, w_kn: (K, N) int8. Zero-pads K,N to multiples of 8 and
    chunks the batch under the cuBLASLt row cap. Small batches use an exact FP32 GPU
    fallback. Used both for GF(2) (caller takes ``& 1``) and for ANF monomial counting
    (caller compares to support sizes)."""
    assert feats.dtype == torch.int8 and w_kn.dtype == torch.int8
    batch, K = feats.shape
    K2, N = w_kn.shape
    assert K == K2, (K, K2)
    if batch < _INT_MM_MIN_ROWS:
        return (feats.to(torch.float32) @ w_kn.to(torch.float32)).round().to(torch.int32)

    Kp, Np = _pad8(K), _pad8(N)
    wp = w_kn
    if Kp != K or Np != N:
        wp = w_kn.new_zeros((Kp, Np)); wp[:K, :N] = w_kn
    wp = wp.contiguous()

    out = torch.empty((batch, N), dtype=torch.int32, device=feats.device)
    for s in range(0, batch, _INT_MM_MAX_ROWS):     # chunk under the cuBLASLt row cap
        e = min(s + _INT_MM_MAX_ROWS, batch)
        fp = feats[s:e]
        if Kp != K:
            fp = feats.new_zeros((e - s, Kp)); fp[:, :K] = feats[s:e]
        out[s:e] = torch._int_mm(fp, wp)[:, :N]      # tensor core INT8 GEMM
    return out
