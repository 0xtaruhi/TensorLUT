"""Host driver for the fused multi-layer b1 ANF megakernel (b1_mega.cu).

Flattens every combinational layer's ANF (A incidence, C coeff, degrees, input/output net
columns) into flat blobs with per-layer offsets, and launches ONE kernel that simulates
all layers and all cycles with the net-state resident in shared memory."""
from __future__ import annotations

import ctypes, hashlib, os, subprocess
import numpy as np
import torch

from ..frontend.netlist import _const_val
from ..ir.anf import lut_anf
from ..reference.interp import po_bits
from .b1_anf import _pack_B, WM, WN, WK, KW

_CU = os.path.join(os.path.dirname(__file__), "b1_mega.cu")
_CACHE = os.path.expanduser("~/.cache/rtlgemm")
_SMEM_MAX = 98304


def _cu13():
    return os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "nvidia", "cu13")


def _build():
    os.makedirs(_CACHE, exist_ok=True); cu = _cu13()
    h = hashlib.md5(open(_CU, "rb").read()).hexdigest()[:12]
    so = os.path.join(_CACHE, f"b1_mega_{h}.so")
    if not os.path.exists(so):
        r = subprocess.run([os.path.join(cu, "bin", "nvcc"), "-arch=sm_89", "-ccbin", "g++",
            "-O3", "--shared", "-Xcompiler", "-fPIC", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
            f"-I{cu}/include", _CU, "-o", so], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc build failed:\n{r.stderr[-2000:]}")
    return so


_LIB = None
def _lib():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(_build()); _LIB.launch_b1_mega.restype = None
    return _LIB


class B1MegaSim:
    def __init__(self, plan, device="cuda"):
        nl = plan.nl; self.nl = nl; self.device = device
        col = {}
        for b in nl.state_bits: col.setdefault(b, len(col))
        for b in nl.input_bits: col.setdefault(b, len(col))
        for lut in nl.luts: col.setdefault(lut.out, len(col))
        self.ZERO = len(col); self.ONE = len(col) + 1; self.n_nets = len(col) + 2
        def ccol(b):
            c = _const_val(b)
            return (self.ONE if c == 1 else self.ZERO) if c is not None else col[b]

        A_all, C_all, deg_all, in_all, out_all, meta = [], [], [], [], [], []
        self.maxNkt = 1; self.maxPhiPadW = KW
        for layer in plan.layers:
            monos = [lut_anf(lut) for lut in layer]
            mset = {}
            for ms in monos:
                for m in ms: mset.setdefault(m, len(mset))
            in_nets = sorted({v for m in mset for v in m})
            lc = {net: i for i, net in enumerate(in_nets)}
            F, NIN, G = len(mset), len(in_nets), len(layer)
            A = np.zeros((F, max(NIN, 1)), np.uint8)
            deg = np.zeros(F, np.int32)
            for m, k in mset.items():
                deg[k] = len(m)
                for v in m: A[k, lc[v]] = 1
            C = np.zeros((G, F), np.uint8)
            for g, ms in enumerate(monos):
                for m in ms: C[g, mset[m]] = 1
            Ap, n_mt, n_kt = _pack_B(A, F, NIN)
            Cp, n_ot, n_ft = _pack_B(C, G, F)
            phiPadW = ((F + 31) // 32 + KW - 1) // KW * KW
            self.maxNkt = max(self.maxNkt, n_kt); self.maxPhiPadW = max(self.maxPhiPadW, phiPadW)
            meta.append([F, NIN, G, n_mt, n_kt, n_ot, n_ft,
                         len(A_all)//1, 0, 0, 0, 0])  # offsets patched below
            # record offsets (in element units)
            aoff = sum(x.size for x in A_all); coff = sum(x.size for x in C_all)
            doff = sum(x.size for x in deg_all); ioff = sum(len(x) for x in in_all)
            ooff = sum(len(x) for x in out_all)
            meta[-1][7:] = [aoff, coff, doff, ioff, ooff]
            A_all.append(Ap.reshape(-1).view(np.uint32))
            C_all.append(Cp.reshape(-1).view(np.uint32))
            deg_all.append(deg)
            in_all.append(np.array([col[n] for n in in_nets] or [0], np.int32))
            out_all.append(np.array([col[lut.out] for lut in layer], np.int32))

        d = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
        self.meta = d(np.array(meta, np.int32).reshape(-1))
        self.A_all = d(np.concatenate(A_all).view(np.int32))
        self.C_all = d(np.concatenate(C_all).view(np.int32))
        self.deg_all = d(np.concatenate(deg_all))
        self.in_all = d(np.concatenate(in_all)); self.out_all = d(np.concatenate(out_all))
        self.n_layers = len(plan.layers)
        self.state_nets = d(np.array([col[b] for b in nl.state_bits], np.int32))
        self.dff_nets = d(np.array([ccol(ff.d) for ff in nl.dffs], np.int32))
        self.input_nets = d(np.array([col[b] for b in nl.input_bits], np.int32))
        self.pb = po_bits(nl)
        self.po_nets = d(np.array([ccol(b) for _, _, b in self.pb], np.int32))
        self.po_const = [k for k, (_, _, b) in enumerate(self.pb) if _const_val(b) == 1]

        perWarp = ((8 * self.n_nets + 15) & ~15) + (self.maxNkt * WM * KW + WM * self.maxPhiPadW) * 4
        self.wpb = 8
        while self.wpb > 1 and self.wpb * perWarp > _SMEM_MAX:
            self.wpb //= 2
        self.smem_bytes = self.wpb * perWarp
        self.grid_cap = 1024

    def run(self, u_seq, cycles):
        nl = self.nl
        u = torch.as_tensor(u_seq, dtype=torch.int8, device=self.device)
        batch = u.shape[1]
        up = u.permute(0, 2, 1).contiguous() if nl.n_input else \
             torch.zeros((cycles, 0, batch), dtype=torch.int8, device=self.device)
        P = len(self.pb)
        po = torch.zeros((cycles, P, batch), dtype=torch.int8, device=self.device)
        n_mtiles = (batch + WM - 1) // WM
        grid = min((n_mtiles + self.wpb - 1) // self.wpb, self.grid_cap)
        v = lambda t: ctypes.c_void_p(t.data_ptr())
        _lib().launch_b1_mega(
            ctypes.c_int(batch), ctypes.c_int(self.n_nets), ctypes.c_int(self.n_layers),
            ctypes.c_int(cycles), v(self.meta), v(self.A_all), v(self.C_all), v(self.deg_all),
            v(self.in_all), v(self.out_all),
            ctypes.c_int(nl.n_state), v(self.state_nets), v(self.dff_nets),
            ctypes.c_int(nl.n_input), v(self.input_nets), ctypes.c_int(self.ONE),
            ctypes.c_int(P), v(self.po_nets), v(up), v(po),
            ctypes.c_int(self.maxNkt), ctypes.c_int(self.maxPhiPadW),
            ctypes.c_int(self.wpb), ctypes.c_int(grid), ctypes.c_int(self.smem_bytes))
        torch.cuda.synchronize()
        out = po.permute(0, 2, 1).contiguous().cpu().numpy()   # (cycles, batch, P)
        for k in self.po_const: out[:, :, k] = 1
        return out
