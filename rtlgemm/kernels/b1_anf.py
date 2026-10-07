"""Host driver for the b1 (single-bit) Tensor-Core ANF kernel (b1_anf.cu).

Packs the monomial-incidence A and coefficient C into b1 (bit) matrices, launches the
two-stage b1 `and.popc` MMA kernel, and returns the layer outputs. This is the
tensor-core realization of ANF evaluation (the core contribution)."""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

import numpy as np
import torch

_HERE = os.path.dirname(__file__)
_CU = os.path.join(_HERE, "b1_anf.cu")
_CACHE = os.path.expanduser("~/.cache/rtlgemm")
WM, WN, WK, KW = 8, 8, 128, 4
_SMEM_MAX = 98304


def _cu13():
    return os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "nvidia", "cu13")


def _build():
    os.makedirs(_CACHE, exist_ok=True)
    cu = _cu13()
    h = hashlib.md5(open(_CU, "rb").read()).hexdigest()[:12]
    so = os.path.join(_CACHE, f"b1_anf_{h}.so")
    if not os.path.exists(so):
        r = subprocess.run(
            [os.path.join(cu, "bin", "nvcc"), "-arch=sm_89", "-ccbin", "g++", "-O3",
             "--shared", "-Xcompiler", "-fPIC", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
             f"-I{cu}/include", _CU, "-o", so],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc build failed:\n{r.stderr[-2000:]}")
    return so


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(_build())
        _LIB.launch_b1_anf.restype = None
        _LIB.launch_b1_anf_xcols.restype = None
        _LIB.launch_b1_anf_x_to_v.restype = None
        _LIB.launch_b1_anf_v.restype = None
        _LIB.launch_b1_anf_v8.restype = None
        _LIB.launch_b1_layer_v8.restype = None
        _LIB.launch_b1_layer_v8_k1.restype = None
        _LIB.launch_b1_program_v8.restype = None
        _LIB.launch_b1_block_program_v8.restype = None
        _LIB.launch_b1_coop_program_v8.restype = None
        _LIB.launch_v8_scatter_cols.restype = None
        _LIB.launch_v8_commit_direct.restype = None
    return _LIB


def _pack_B(inc, N, K):
    """inc: (N, K) 0/1 -> b1 col-major fragment tiles [n_nt, n_kt, WN, KW] uint32."""
    n_nt, n_kt = (N + WN - 1) // WN, (K + WK - 1) // WK
    out = np.zeros((n_nt, n_kt, WN, KW), np.uint32)
    for n in range(N):
        nt, nn = n // WN, n % WN
        row = inc[n]
        for k in range(K):
            if row[k]:
                kt, w, b = k // WK, (k % WK) // 32, k % 32
                out[nt, kt, nn, w] |= np.uint32(1) << np.uint32(b)
    return out, n_nt, n_kt


def v8_scatter_cols(V, cols_i32, X):
    n_tiles, n_cols = V.shape
    n_set = cols_i32.numel()
    grid = min(1024, max(1, (n_tiles * max(n_set, 1) + 255) // 256))
    _lib().launch_v8_scatter_cols(
        ctypes.c_void_p(V.data_ptr()), ctypes.c_int(n_tiles), ctypes.c_int(n_cols),
        ctypes.c_void_p(cols_i32.data_ptr()), ctypes.c_int(n_set),
        ctypes.c_void_p(X.data_ptr()), ctypes.c_int(grid),
        ctypes.c_void_p(torch.cuda.current_stream(V.device).cuda_stream))


def v8_commit_direct(V, state_cols_i32, dff_cols_i32):
    n_tiles, n_cols = V.shape
    n_state = state_cols_i32.numel()
    grid = min(1024, max(1, (n_tiles * max(n_state, 1) + 255) // 256))
    _lib().launch_v8_commit_direct(
        ctypes.c_void_p(V.data_ptr()), ctypes.c_int(n_tiles), ctypes.c_int(n_cols),
        ctypes.c_void_p(state_cols_i32.data_ptr()),
        ctypes.c_void_p(dff_cols_i32.data_ptr()), ctypes.c_int(n_state),
        ctypes.c_int(grid),
        ctypes.c_void_p(torch.cuda.current_stream(V.device).cuda_stream))


class B1Anf:
    """Compiled b1 ANF evaluator for one layer (A incidence, C coeff)."""

    def __init__(self, A_inc, C_coeff, device="cuda"):
        self.F, self.NIN = A_inc.shape
        self.G = C_coeff.shape[0]
        self.device = device
        deg = A_inc.sum(1).astype(np.int32)
        Ap, self.n_mt, self.n_kt = _pack_B(A_inc, self.F, self.NIN)    # B-frag [input,mono]
        Cp, self.n_ot, self.n_ft = _pack_B(C_coeff, self.G, self.F)    # B-frag [mono,out]
        d = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
        self.A = d(Ap.view(np.int32)); self.C = d(Cp.view(np.int32)); self.deg = d(deg)

        phiW = (self.F + 31) // 32
        # A/C/deg stay in GLOBAL (hot in L2); only the per-warp scratch is in shared, so
        # many blocks fit per SM (occupancy was the ncu-identified bottleneck).
        phiPadW = (phiW + KW - 1) // KW * KW
        self.scratch = self.n_kt * WM * KW + WM * phiPadW
        self.wpb = 8
        while self.wpb > 1 and self.wpb * self.scratch * 4 > _SMEM_MAX:
            self.wpb //= 2
        self.smem_bytes = self.wpb * self.scratch * 4
        self.grid_cap = 1024

    def run(self, X):
        """X: (batch, NIN) int8/uint8 0/1 -> Y (batch, G) int8 (synchronizes)."""
        Y = self.run_into(torch.as_tensor(X, dtype=torch.int8, device=self.device).contiguous())
        torch.cuda.synchronize()
        return Y

    def run_into(self, X, Y=None):
        """Launch on device tensor X (batch,NIN) int8; no synchronize. Returns Y."""
        batch = X.shape[0]
        if Y is None:
            Y = torch.empty((batch, self.G), dtype=torch.int8, device=self.device)
        n_mtiles = (batch + WM - 1) // WM
        grid = min((n_mtiles + self.wpb - 1) // self.wpb, self.grid_cap)
        _lib().launch_b1_anf(
            ctypes.c_void_p(X.data_ptr()), ctypes.c_int(batch), ctypes.c_int(self.NIN),
            ctypes.c_void_p(self.A.data_ptr()), ctypes.c_int(self.F),
            ctypes.c_int(self.n_mt), ctypes.c_int(self.n_kt),
            ctypes.c_void_p(self.deg.data_ptr()),
            ctypes.c_void_p(self.C.data_ptr()), ctypes.c_int(self.G),
            ctypes.c_int(self.n_ot), ctypes.c_int(self.n_ft),
            ctypes.c_void_p(Y.data_ptr()),
            ctypes.c_int(self.wpb), ctypes.c_int(grid), ctypes.c_int(self.smem_bytes),
            ctypes.c_void_p(torch.cuda.current_stream(X.device).cuda_stream))
        return Y

    def run_v(self, V, in_cols, out_cols):
        """Evaluate this layer in-place over a resident row-major V[batch,n_cols]."""
        batch, n_cols = V.shape
        n_mtiles = (batch + WM - 1) // WM
        grid = min((n_mtiles + self.wpb - 1) // self.wpb, self.grid_cap)
        _lib().launch_b1_anf_v(
            ctypes.c_void_p(V.data_ptr()), ctypes.c_int(batch), ctypes.c_int(n_cols),
            ctypes.c_void_p(in_cols.data_ptr()), ctypes.c_int(self.NIN),
            ctypes.c_void_p(self.A.data_ptr()), ctypes.c_int(self.F),
            ctypes.c_int(self.n_mt), ctypes.c_int(self.n_kt),
            ctypes.c_void_p(self.deg.data_ptr()),
            ctypes.c_void_p(self.C.data_ptr()), ctypes.c_int(self.G),
            ctypes.c_int(self.n_ot), ctypes.c_int(self.n_ft),
            ctypes.c_void_p(out_cols.data_ptr()),
            ctypes.c_int(self.wpb), ctypes.c_int(grid), ctypes.c_int(self.smem_bytes),
            ctypes.c_void_p(torch.cuda.current_stream(V.device).cuda_stream))

    def run_v8(self, V, in_cols, out_cols):
        """Evaluate in-place over packed resident V8[ceil(batch/8),n_cols] uint8."""
        n_tiles, n_cols = V.shape
        grid = min((n_tiles + self.wpb - 1) // self.wpb, self.grid_cap)
        _lib().launch_b1_anf_v8(
            ctypes.c_void_p(V.data_ptr()), ctypes.c_int(n_tiles), ctypes.c_int(n_cols),
            ctypes.c_void_p(in_cols.data_ptr()), ctypes.c_int(self.NIN),
            ctypes.c_void_p(self.A.data_ptr()), ctypes.c_int(self.F),
            ctypes.c_int(self.n_mt), ctypes.c_int(self.n_kt),
            ctypes.c_void_p(self.deg.data_ptr()),
            ctypes.c_void_p(self.C.data_ptr()), ctypes.c_int(self.G),
            ctypes.c_int(self.n_ot), ctypes.c_int(self.n_ft),
            ctypes.c_void_p(out_cols.data_ptr()),
            ctypes.c_int(self.wpb), ctypes.c_int(grid), ctypes.c_int(self.smem_bytes),
            ctypes.c_void_p(torch.cuda.current_stream(V.device).cuda_stream))

    def run_xcols_into(self, X, x_cols, Y=None):
        """Evaluate using columns selected from a shared row-major X[batch,XNIN]."""
        batch, xnin = X.shape
        if Y is None:
            Y = torch.empty((batch, self.G), dtype=torch.int8, device=self.device)
        n_mtiles = (batch + WM - 1) // WM
        grid = min((n_mtiles + self.wpb - 1) // self.wpb, self.grid_cap)
        _lib().launch_b1_anf_xcols(
            ctypes.c_void_p(X.data_ptr()), ctypes.c_int(batch), ctypes.c_int(xnin),
            ctypes.c_void_p(x_cols.data_ptr()), ctypes.c_int(self.NIN),
            ctypes.c_void_p(self.A.data_ptr()), ctypes.c_int(self.F),
            ctypes.c_int(self.n_mt), ctypes.c_int(self.n_kt),
            ctypes.c_void_p(self.deg.data_ptr()),
            ctypes.c_void_p(self.C.data_ptr()), ctypes.c_int(self.G),
            ctypes.c_int(self.n_ot), ctypes.c_int(self.n_ft),
            ctypes.c_void_p(Y.data_ptr()),
            ctypes.c_int(self.wpb), ctypes.c_int(grid), ctypes.c_int(self.smem_bytes),
            ctypes.c_void_p(torch.cuda.current_stream(X.device).cuda_stream))
        return Y

    def run_into_v(self, X, V, out_cols):
        """Evaluate compact row-major X and scatter outputs directly into V."""
        batch = X.shape[0]
        n_cols = V.shape[1]
        n_mtiles = (batch + WM - 1) // WM
        grid = min((n_mtiles + self.wpb - 1) // self.wpb, self.grid_cap)
        _lib().launch_b1_anf_x_to_v(
            ctypes.c_void_p(X.data_ptr()), ctypes.c_int(batch), ctypes.c_int(self.NIN),
            ctypes.c_void_p(self.A.data_ptr()), ctypes.c_int(self.F),
            ctypes.c_int(self.n_mt), ctypes.c_int(self.n_kt),
            ctypes.c_void_p(self.deg.data_ptr()),
            ctypes.c_void_p(self.C.data_ptr()), ctypes.c_int(self.G),
            ctypes.c_int(self.n_ot), ctypes.c_int(self.n_ft),
            ctypes.c_void_p(V.data_ptr()), ctypes.c_int(n_cols),
            ctypes.c_void_p(out_cols.data_ptr()),
            ctypes.c_int(self.wpb), ctypes.c_int(grid), ctypes.c_int(self.smem_bytes),
            ctypes.c_void_p(torch.cuda.current_stream(X.device).cuda_stream))

    @staticmethod
    def reference(X, A_inc, C_coeff):
        """numpy ANF reference: y = ((X@A^T == deg) @ C^T) & 1."""
        X = X.astype(np.int64); deg = A_inc.sum(1)
        phi = (X @ A_inc.T.astype(np.int64) == deg).astype(np.int64)
        return ((phi @ C_coeff.T.astype(np.int64)) & 1).astype(np.int8)


class B1LayerV8:
    """One-kernel packed-v8 evaluator for all independent chunks in one layer."""

    def __init__(self, chunks, device="cuda"):
        self.device = device
        self.n_chunks = len(chunks)
        meta, A_all, C_all, deg_all, in_all, out_all, bypass_all = [], [], [], [], [], [], []
        # Front-bypass packing for the multi-tile kernel: phi = [literals | pad to 32 | tc monos].
        meta_f, C_all_f, src_all_f = [], [], []
        self.maxNkt = 1
        self.maxPhiPadW = KW
        self.maxPhiPadW_f = KW
        literal_bypass = os.environ.get("RTLGEMM_B1_LITERAL_BYPASS", "1") != "0"
        for A_inc, C_coeff, in_cols, out_cols in chunks:
            F, NIN = A_inc.shape
            G = C_coeff.shape[0]
            deg = A_inc.sum(1).astype(np.int32)
            if literal_bypass:
                tc_idx = np.flatnonzero(deg > 1)
                bypass_idx = np.flatnonzero(deg <= 1)
            else:
                tc_idx = np.arange(F)
                bypass_idx = np.zeros(0, dtype=np.int64)
            order = np.concatenate([tc_idx, bypass_idx]).astype(np.int64, copy=False)
            Ftc = int(tc_idx.size)
            A_tc = A_inc[tc_idx] if Ftc else np.zeros((0, NIN), dtype=np.uint8)
            C_reordered = C_coeff[:, order] if order.size else C_coeff[:, :0]
            deg_tc = deg[tc_idx]
            Ap, n_mt, n_kt = _pack_B(A_tc, Ftc, NIN)
            Cp, n_ot, n_ft = _pack_B(C_reordered, G, F)
            phiW = (F + 31) // 32
            phiPadW = (phiW + KW - 1) // KW * KW
            self.maxNkt = max(self.maxNkt, n_kt)
            self.maxPhiPadW = max(self.maxPhiPadW, phiPadW)
            aoff = sum(x.size for x in A_all)
            coff = sum(x.size for x in C_all)
            doff = sum(x.size for x in deg_all)
            ioff = sum(x.size for x in in_all)
            ooff = sum(x.size for x in out_all)
            boff = sum(x.size for x in bypass_all)
            bypass = np.zeros((len(bypass_idx), 2), dtype=np.int32)
            for j, src_idx in enumerate(bypass_idx):
                phi_idx = Ftc + j
                nz = np.flatnonzero(A_inc[src_idx])
                src_col = -1 if len(nz) == 0 else int(in_cols[int(nz[0])])
                bypass[j, 0] = phi_idx
                bypass[j, 1] = src_col
            meta.append([F, NIN, G, n_mt, n_kt, n_ot, n_ft,
                         aoff, coff, doff, ioff, ooff, Ftc, boff, len(bypass_idx)])
            nb = len(bypass_idx)
            nbp = (nb + 31) // 32 * 32
            Ff = nbp + Ftc
            Cf = np.zeros((G, Ff), dtype=C_coeff.dtype)
            if nb:
                Cf[:, :nb] = C_coeff[:, bypass_idx]
            if Ftc:
                Cf[:, nbp:] = C_coeff[:, tc_idx]
            Cpf, n_ot_f, n_ft_f = _pack_B(Cf, G, Ff)
            phiPadW_f = ((Ff + 31) // 32 + KW - 1) // KW * KW
            self.maxPhiPadW_f = max(self.maxPhiPadW_f, phiPadW_f)
            coff_f = sum(x.size for x in C_all_f)
            soff_f = sum(x.size for x in src_all_f)
            meta_f.append([Ff, NIN, G, n_mt, n_kt, n_ot_f, n_ft_f,
                           aoff, coff_f, doff, ioff, ooff, Ftc, soff_f, nb])
            C_all_f.append(Cpf.reshape(-1).view(np.uint32))
            src_all_f.append(bypass[:, 1].copy() if nb else np.zeros(0, np.int32))
            A_all.append(Ap.reshape(-1).view(np.uint32))
            C_all.append(Cp.reshape(-1).view(np.uint32))
            deg_all.append(deg_tc)
            in_all.append(np.asarray(in_cols, np.int32))
            out_all.append(np.asarray(out_cols, np.int32))
            bypass_all.append(bypass.reshape(-1))

        def cat_or_zeros(vals, dtype):
            if vals and sum(x.size for x in vals):
                return np.concatenate(vals).astype(dtype, copy=False)
            return np.zeros(1, dtype)

        d = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
        self.meta = d(np.array(meta, np.int32).reshape(-1))
        self.A_all = d(cat_or_zeros(A_all, np.uint32).view(np.int32))
        self.C_all = d(cat_or_zeros(C_all, np.uint32).view(np.int32))
        self.deg_all = d(cat_or_zeros(deg_all, np.int32))
        self.in_all = d(cat_or_zeros(in_all, np.int32))
        self.out_all = d(cat_or_zeros(out_all, np.int32))
        self.bypass_all = d(cat_or_zeros(bypass_all, np.int32))
        self.meta_f = d(np.array(meta_f, np.int32).reshape(-1))
        self.C_all_f = d(cat_or_zeros(C_all_f, np.uint32).view(np.int32))
        self.src_all_f = d(cat_or_zeros(src_all_f, np.int32))

        scratch = self.maxNkt * WM * KW + WM * self.maxPhiPadW
        self.wpb = int(os.environ.get("RTLGEMM_B1_LAYER_WPB", "8"))
        self.wpb = max(1, min(32, self.wpb))
        while self.wpb > 1 and self.wpb * scratch * 4 > _SMEM_MAX:
            self.wpb //= 2
        self.smem_bytes = self.wpb * scratch * 4
        self.grid_cap = 1024
        self.use_k1 = self.maxNkt == 1 and os.environ.get("RTLGEMM_B1_LAYER_K1", "1") != "0"
        # Tiles of 8 stimuli per warp on the k1 path; T > 1 reuses each A/C fragment T times.
        # RTLGEMM_B1_TPW=auto (default) picks T from the batch: 1 below 2^10 stimuli, 2 below
        # 2^14, else 4 (measured on the Rocket core, RTX 4090).
        tpw = os.environ.get("RTLGEMM_B1_TPW", "auto")
        self.tpw = tpw if tpw == "auto" else int(tpw)
        if not self.use_k1 or self.tpw not in ("auto", 1, 2, 4, 8):
            self.tpw = 1
        self.per_tile_bytes = (WM * KW + WM * self.maxPhiPadW_f) * 4

    def _tiles_per_warp(self, n_tiles):
        if self.tpw != "auto":
            return self.tpw
        return 1 if n_tiles < 128 else 2 if n_tiles < 2048 else 4

    def run_v8(self, V):
        n_tiles, n_cols = V.shape
        tpw = self._tiles_per_warp(n_tiles)
        if tpw > 1:
            wpb = self.wpb
            while wpb > 1 and wpb * self.per_tile_bytes * tpw > _SMEM_MAX:
                wpb //= 2
            smem = wpb * self.per_tile_bytes * tpw
            groups = (n_tiles + tpw - 1) // tpw
            total_work = groups * self.n_chunks
            grid = min((total_work + wpb - 1) // wpb, self.grid_cap)
            _lib().launch_b1_layer_v8_k1_t(
                ctypes.c_void_p(V.data_ptr()), ctypes.c_int(n_tiles), ctypes.c_int(n_cols),
                ctypes.c_int(self.n_chunks),
                ctypes.c_void_p(self.meta_f.data_ptr()),
                ctypes.c_void_p(self.A_all.data_ptr()),
                ctypes.c_void_p(self.C_all_f.data_ptr()),
                ctypes.c_void_p(self.deg_all.data_ptr()),
                ctypes.c_void_p(self.in_all.data_ptr()),
                ctypes.c_void_p(self.out_all.data_ptr()),
                ctypes.c_void_p(self.src_all_f.data_ptr()),
                ctypes.c_int(self.maxPhiPadW_f), ctypes.c_int(tpw),
                ctypes.c_int(wpb), ctypes.c_int(grid), ctypes.c_int(smem),
                ctypes.c_void_p(torch.cuda.current_stream(V.device).cuda_stream))
            return
        total_work = n_tiles * self.n_chunks
        grid = min((total_work + self.wpb - 1) // self.wpb, self.grid_cap)
        launch = _lib().launch_b1_layer_v8_k1 if self.use_k1 else _lib().launch_b1_layer_v8
        launch(
            ctypes.c_void_p(V.data_ptr()), ctypes.c_int(n_tiles), ctypes.c_int(n_cols),
            ctypes.c_int(self.n_chunks),
            ctypes.c_void_p(self.meta.data_ptr()),
            ctypes.c_void_p(self.A_all.data_ptr()),
            ctypes.c_void_p(self.C_all.data_ptr()),
            ctypes.c_void_p(self.deg_all.data_ptr()),
            ctypes.c_void_p(self.in_all.data_ptr()),
            ctypes.c_void_p(self.out_all.data_ptr()),
            ctypes.c_void_p(self.bypass_all.data_ptr()),
            ctypes.c_int(self.maxNkt), ctypes.c_int(self.maxPhiPadW),
            ctypes.c_int(self.wpb), ctypes.c_int(grid), ctypes.c_int(self.smem_bytes),
            ctypes.c_void_p(torch.cuda.current_stream(V.device).cuda_stream))


class B1ProgramV8:
    """Single-kernel packed-v8 evaluator for all layers and cycles."""

    def __init__(self, layer_chunks, n_cols, one_col, state_cols, dff_cols, input_cols,
                 po_cols, device="cuda"):
        self.device = device
        self.n_cols = n_cols
        self.one_col = one_col
        self.n_layers = len(layer_chunks)
        meta, A_all, C_all, deg_all, in_all, out_all = [], [], [], [], [], []
        layer_off, layer_count = [], []
        self.maxNkt = 1
        self.maxPhiPadW = KW
        for chunks in layer_chunks:
            layer_off.append(len(meta))
            layer_count.append(len(chunks))
            for A_inc, C_coeff, in_cols, out_cols in chunks:
                F, NIN = A_inc.shape
                G = C_coeff.shape[0]
                deg = A_inc.sum(1).astype(np.int32)
                Ap, n_mt, n_kt = _pack_B(A_inc, F, NIN)
                Cp, n_ot, n_ft = _pack_B(C_coeff, G, F)
                phiW = (F + 31) // 32
                phiPadW = (phiW + KW - 1) // KW * KW
                self.maxNkt = max(self.maxNkt, n_kt)
                self.maxPhiPadW = max(self.maxPhiPadW, phiPadW)
                aoff = sum(x.size for x in A_all)
                coff = sum(x.size for x in C_all)
                doff = sum(x.size for x in deg_all)
                ioff = sum(x.size for x in in_all)
                ooff = sum(x.size for x in out_all)
                meta.append([F, NIN, G, n_mt, n_kt, n_ot, n_ft,
                             aoff, coff, doff, ioff, ooff])
                A_all.append(Ap.reshape(-1).view(np.uint32))
                C_all.append(Cp.reshape(-1).view(np.uint32))
                deg_all.append(deg)
                in_all.append(np.asarray(in_cols, np.int32))
                out_all.append(np.asarray(out_cols, np.int32))

        def cat_or_zeros(vals, dtype):
            if vals:
                return np.concatenate(vals).astype(dtype, copy=False)
            return np.zeros(1, dtype)

        d = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
        self.layer_off = d(np.array(layer_off, np.int32))
        self.layer_count = d(np.array(layer_count, np.int32))
        self.meta = d(np.array(meta, np.int32).reshape(-1))
        self.A_all = d(cat_or_zeros(A_all, np.uint32).view(np.int32))
        self.C_all = d(cat_or_zeros(C_all, np.uint32).view(np.int32))
        self.deg_all = d(cat_or_zeros(deg_all, np.int32))
        self.in_all = d(cat_or_zeros(in_all, np.int32))
        self.out_all = d(cat_or_zeros(out_all, np.int32))
        self.state_cols = d(np.asarray(state_cols, np.int32))
        self.dff_cols = d(np.asarray(dff_cols, np.int32))
        self.input_cols = d(np.asarray(input_cols, np.int32))
        self.po_cols = d(np.asarray(po_cols, np.int32))
        self.n_state = len(state_cols)
        self.n_input = len(input_cols)
        self.n_po = len(po_cols)

        state_pad = (self.n_state + 15) // 16 * 16
        scratch_words = self.maxNkt * WM * KW + WM * self.maxPhiPadW
        per_warp_bytes = state_pad + scratch_words * 4
        self.wpb = 8
        while self.wpb > 1 and self.wpb * per_warp_bytes > _SMEM_MAX:
            self.wpb //= 2
        self.smem_bytes = self.wpb * per_warp_bytes
        self.grid_cap = 1024

    def run(self, V, U, po_out, final_state, cycles, capture_po):
        n_tiles, n_cols = V.shape
        grid = min((n_tiles + self.wpb - 1) // self.wpb, self.grid_cap)
        _lib().launch_b1_program_v8(
            ctypes.c_void_p(V.data_ptr()), ctypes.c_int(n_tiles), ctypes.c_int(n_cols),
            ctypes.c_int(self.n_layers), ctypes.c_int(cycles),
            ctypes.c_void_p(self.layer_off.data_ptr()),
            ctypes.c_void_p(self.layer_count.data_ptr()),
            ctypes.c_void_p(self.meta.data_ptr()),
            ctypes.c_void_p(self.A_all.data_ptr()),
            ctypes.c_void_p(self.C_all.data_ptr()),
            ctypes.c_void_p(self.deg_all.data_ptr()),
            ctypes.c_void_p(self.in_all.data_ptr()),
            ctypes.c_void_p(self.out_all.data_ptr()),
            ctypes.c_int(self.n_state), ctypes.c_void_p(self.state_cols.data_ptr()),
            ctypes.c_void_p(self.dff_cols.data_ptr()),
            ctypes.c_int(self.n_input), ctypes.c_void_p(self.input_cols.data_ptr()),
            ctypes.c_int(self.one_col),
            ctypes.c_int(self.n_po), ctypes.c_void_p(self.po_cols.data_ptr()),
            ctypes.c_void_p(U.data_ptr()),
            ctypes.c_void_p(po_out.data_ptr()),
            ctypes.c_void_p(final_state.data_ptr()),
            ctypes.c_int(self.maxNkt), ctypes.c_int(self.maxPhiPadW),
            ctypes.c_int(1 if capture_po else 0),
            ctypes.c_int(self.wpb), ctypes.c_int(grid), ctypes.c_int(self.smem_bytes),
            ctypes.c_void_p(torch.cuda.current_stream(V.device).cuda_stream))


class B1BlockProgramV8(B1ProgramV8):
    """Single-kernel packed-v8 program with one cooperative block per stimulus tile."""

    def __init__(self, layer_chunks, n_cols, one_col, state_cols, dff_cols, input_cols,
                 po_cols, device="cuda", warps_per_block=8):
        super().__init__(layer_chunks, n_cols, one_col, state_cols, dff_cols,
                         input_cols, po_cols, device=device)
        state_pad = (self.n_state + 15) // 16 * 16
        v_pad = (self.n_cols + 15) // 16 * 16
        scratch_words = self.maxNkt * WM * KW + WM * self.maxPhiPadW
        scratch_bytes = scratch_words * 4
        self.wpb = int(os.environ.get("RTLGEMM_B1_COOP_WPB", str(warps_per_block)))
        while self.wpb > 1 and v_pad + state_pad + self.wpb * scratch_bytes > _SMEM_MAX:
            self.wpb //= 2
        self.smem_bytes = v_pad + state_pad + self.wpb * scratch_bytes
        self.grid_cap = 65535

    def run_block(self, U, po_out, final_state, cycles, capture_po):
        n_tiles = U.shape[1]
        grid = min(n_tiles, self.grid_cap)
        _lib().launch_b1_block_program_v8(
            ctypes.c_int(n_tiles), ctypes.c_int(self.n_cols),
            ctypes.c_int(self.n_layers), ctypes.c_int(cycles),
            ctypes.c_void_p(self.layer_off.data_ptr()),
            ctypes.c_void_p(self.layer_count.data_ptr()),
            ctypes.c_void_p(self.meta.data_ptr()),
            ctypes.c_void_p(self.A_all.data_ptr()),
            ctypes.c_void_p(self.C_all.data_ptr()),
            ctypes.c_void_p(self.deg_all.data_ptr()),
            ctypes.c_void_p(self.in_all.data_ptr()),
            ctypes.c_void_p(self.out_all.data_ptr()),
            ctypes.c_int(self.n_state), ctypes.c_void_p(self.state_cols.data_ptr()),
            ctypes.c_void_p(self.dff_cols.data_ptr()),
            ctypes.c_int(self.n_input), ctypes.c_void_p(self.input_cols.data_ptr()),
            ctypes.c_int(self.one_col),
            ctypes.c_int(self.n_po), ctypes.c_void_p(self.po_cols.data_ptr()),
            ctypes.c_void_p(U.data_ptr()),
            ctypes.c_void_p(po_out.data_ptr()),
            ctypes.c_void_p(final_state.data_ptr()),
            ctypes.c_int(self.maxNkt), ctypes.c_int(self.maxPhiPadW),
            ctypes.c_int(1 if capture_po else 0),
            ctypes.c_int(self.wpb), ctypes.c_int(grid), ctypes.c_int(self.smem_bytes),
            ctypes.c_void_p(torch.cuda.current_stream(U.device).cuda_stream))


class B1CoopProgramV8(B1ProgramV8):
    """Cooperative-grid packed-v8 program with grid-wide sync between layers."""

    def __init__(self, layer_chunks, n_cols, one_col, state_cols, dff_cols, input_cols,
                 po_cols, device="cuda", warps_per_block=8, grid_blocks=128):
        super().__init__(layer_chunks, n_cols, one_col, state_cols, dff_cols,
                         input_cols, po_cols, device=device)
        scratch = self.maxNkt * WM * KW + WM * self.maxPhiPadW
        self.wpb = warps_per_block
        while self.wpb > 1 and self.wpb * scratch * 4 > _SMEM_MAX:
            self.wpb //= 2
        self.smem_bytes = self.wpb * scratch * 4
        self.grid_blocks = int(os.environ.get("RTLGEMM_B1_COOP_GRID", str(grid_blocks)))

    def run_coop(self, V, U, po_out, final_state, cycles, capture_po):
        n_tiles, n_cols = V.shape
        grid = min(self.grid_blocks, n_tiles)
        _lib().launch_b1_coop_program_v8(
            ctypes.c_void_p(V.data_ptr()), ctypes.c_int(n_tiles), ctypes.c_int(n_cols),
            ctypes.c_int(self.n_layers), ctypes.c_int(cycles),
            ctypes.c_void_p(self.layer_off.data_ptr()),
            ctypes.c_void_p(self.layer_count.data_ptr()),
            ctypes.c_void_p(self.meta.data_ptr()),
            ctypes.c_void_p(self.A_all.data_ptr()),
            ctypes.c_void_p(self.C_all.data_ptr()),
            ctypes.c_void_p(self.deg_all.data_ptr()),
            ctypes.c_void_p(self.in_all.data_ptr()),
            ctypes.c_void_p(self.out_all.data_ptr()),
            ctypes.c_int(self.n_state), ctypes.c_void_p(self.state_cols.data_ptr()),
            ctypes.c_void_p(self.dff_cols.data_ptr()),
            ctypes.c_int(self.n_input), ctypes.c_void_p(self.input_cols.data_ptr()),
            ctypes.c_int(self.one_col),
            ctypes.c_int(self.n_po), ctypes.c_void_p(self.po_cols.data_ptr()),
            ctypes.c_void_p(U.data_ptr()),
            ctypes.c_void_p(po_out.data_ptr()),
            ctypes.c_void_p(final_state.data_ptr()),
            ctypes.c_int(self.maxNkt), ctypes.c_int(self.maxPhiPadW),
            ctypes.c_int(1 if capture_po else 0),
            ctypes.c_int(self.wpb), ctypes.c_int(grid), ctypes.c_int(self.smem_bytes),
            ctypes.c_void_p(torch.cuda.current_stream(U.device).cuda_stream))
