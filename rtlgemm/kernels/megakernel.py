"""Host driver for the one-launch bit-parallel simulation megakernel (megakernel.cu).

Flattens a LUT netlist to flat arrays, bit-packs 32 stimuli per uint32, compiles the
.cu with nvcc into a .so (cached), and launches it once via ctypes on torch device
pointers — the whole simulation (all layers, all cycles) runs in a single kernel.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

import numpy as np
import torch

from ..frontend.netlist import _const_val
from ..reference.interp import _lut_table_array, po_bits

_HERE = os.path.dirname(__file__)
_CU = os.path.join(_HERE, "megakernel.cu")
_MAX_NETS = 2048
_MAX_STATE = 512
_CACHE = os.path.expanduser("~/.cache/rtlgemm")


def _cuda_home():
    return os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "nvidia", "cu13")


def _build():
    os.makedirs(_CACHE, exist_ok=True)
    h = hashlib.md5(open(_CU, "rb").read()).hexdigest()[:12]
    so = os.path.join(_CACHE, f"megakernel_{h}_{_MAX_NETS}_{_MAX_STATE}.so")
    if not os.path.exists(so):
        nvcc = os.path.join(_cuda_home(), "bin", "nvcc")
        cmd = [nvcc, "-arch=sm_89", "-ccbin", "g++", "-O3", "--shared", "-Xcompiler",
               "-fPIC", f"-DMAX_NETS={_MAX_NETS}", f"-DMAX_STATE={_MAX_STATE}", _CU, "-o", so]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc build failed:\n{r.stderr}")
    return so


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(_build())
        _LIB.launch_mega.restype = None
    return _LIB


class MegaSim:
    """Compile-once/run-many one-launch bit-parallel simulator for a LUT plan."""

    def __init__(self, plan, device="cuda", threads=128):
        nl = plan.nl
        self.nl, self.device, self.threads = nl, device, threads
        luts = nl._lut_topo

        idx = {}
        def add(n):
            if n not in idx:
                idx[n] = len(idx)
        add("__zero__"); self.ZERO = idx["__zero__"]
        add("__one__"); self.ONE = idx["__one__"]
        for b in nl.state_bits: add(b)
        for b in nl.input_bits: add(b)
        for lut in luts: add(lut.out)
        self.n_nets = len(idx)
        if self.n_nets > _MAX_NETS:
            raise RuntimeError(f"n_nets={self.n_nets} > MAX_NETS={_MAX_NETS}")
        if nl.n_state > _MAX_STATE:
            raise RuntimeError(f"n_state={nl.n_state} > MAX_STATE={_MAX_STATE}")

        def rid(b):
            c = _const_val(b)
            if c is not None:
                return self.ONE if c == 1 else self.ZERO
            return idx[b]

        L = len(luts)
        self.n_luts = L
        lut_out = np.array([idx[l.out] for l in luts], np.int32)
        lut_w = np.array([l.width for l in luts], np.int32)
        lin = np.zeros((L, 6), np.int32)
        tab = np.zeros(L, np.uint64)
        for i, l in enumerate(luts):
            for j, b in enumerate(l.inputs):
                lin[i, j] = rid(b)
            t = _lut_table_array(l)
            tab[i] = int(sum(int(v) << e for e, v in enumerate(t)))
        self.pb = po_bits(nl)

        # resident device arrays for the (constant) structure
        d = lambda a: torch.as_tensor(np.asarray(a), device=device)
        self._d = dict(
            lut_out=d(lut_out), lut_w=d(lut_w), lut_in=d(lin.reshape(-1)),
            lut_tab=torch.as_tensor(tab.view(np.int64), device=device),
            state_net=d(np.array([idx[b] for b in nl.state_bits], np.int32)),
            dff_d=d(np.array([rid(ff.d) for ff in nl.dffs], np.int32)),
            input_net=d(np.array([idx[b] for b in nl.input_bits], np.int32)),
            po_net=d(np.array([rid(b) for _, _, b in self.pb], np.int32)),
        )
        self.po_const = [k for k, (_, _, b) in enumerate(self.pb) if _const_val(b) == 1]
        # shared-memory bytes for the structure
        self.smem = (L * 3 + L * 6 + nl.n_state * 2 + len(self.pb) + nl.n_input) * 4 + L * 8

    def run(self, u_seq_np, cycles):
        """u_seq_np: (cycles, batch, n_input) uint8. Returns po (cycles, batch, n_po) uint8."""
        nl = self.nl
        batch = u_seq_np.shape[1]
        ng = (batch + 31) // 32
        I, P = nl.n_input, len(self.pb)
        # vectorized bit-pack: 32 stimuli -> one uint32, layout (cycles, I, ng)
        wts = (np.uint32(1) << np.arange(32, dtype=np.uint32))
        if I:
            pad = ng * 32 - batch
            ur = u_seq_np.astype(np.uint32)
            if pad:
                ur = np.pad(ur, ((0, 0), (0, pad), (0, 0)))
            ur = ur.reshape(cycles, ng, 32, I)
            up = (ur * wts[None, None, :, None]).sum(axis=2).astype(np.uint32)  # (cycles,ng,I)
            up = np.ascontiguousarray(up.transpose(0, 2, 1))                    # (cycles,I,ng)
        else:
            up = np.zeros((cycles, 0, ng), np.uint32)
        u_dev = torch.as_tensor(up.view(np.int32), device=self.device).contiguous()
        po_dev = torch.zeros((cycles, P, ng), dtype=torch.int32, device=self.device)

        lib = _lib()
        cargs = [
            ctypes.c_int(self.n_luts),
            ctypes.c_void_p(self._d["lut_out"].data_ptr()),
            ctypes.c_void_p(self._d["lut_w"].data_ptr()),
            ctypes.c_void_p(self._d["lut_in"].data_ptr()),
            ctypes.c_void_p(self._d["lut_tab"].data_ptr()),
            ctypes.c_int(self.n_nets), ctypes.c_int(self.ONE),
            ctypes.c_int(nl.n_state),
            ctypes.c_void_p(self._d["state_net"].data_ptr()),
            ctypes.c_void_p(self._d["dff_d"].data_ptr()),
            ctypes.c_int(I), ctypes.c_void_p(self._d["input_net"].data_ptr()),
            ctypes.c_int(P), ctypes.c_void_p(self._d["po_net"].data_ptr()),
            ctypes.c_int(cycles), ctypes.c_int(ng),
            ctypes.c_void_p(u_dev.data_ptr()), ctypes.c_void_p(po_dev.data_ptr()),
            ctypes.c_int(self.threads), ctypes.c_int(self.smem),
        ]
        lib.launch_mega(*cargs)
        torch.cuda.synchronize()

        pop = po_dev.cpu().numpy().view(np.uint32)   # (cycles, P, ng)
        # vectorized unpack: (cycles,P,ng) -> (cycles, ng*32, P) -> crop to batch
        bit = np.arange(32, dtype=np.uint32)
        pu = pop.transpose(0, 2, 1)[:, :, None, :]   # (cycles, ng, 1, P)
        po = ((pu >> bit[None, None, :, None]) & 1).astype(np.uint8)  # (cycles,ng,32,P)
        po = po.reshape(cycles, ng * 32, P)[:, :batch, :]
        for k in self.po_const:
            po[:, :, k] = 1
        return po

    def run_kernel_only(self, u_dev, po_dev, cycles, ng):
        """Launch just the kernel on already-packed device buffers (for timing)."""
        nl = self.nl
        lib = _lib()
        lib.launch_mega(
            ctypes.c_int(self.n_luts),
            ctypes.c_void_p(self._d["lut_out"].data_ptr()),
            ctypes.c_void_p(self._d["lut_w"].data_ptr()),
            ctypes.c_void_p(self._d["lut_in"].data_ptr()),
            ctypes.c_void_p(self._d["lut_tab"].data_ptr()),
            ctypes.c_int(self.n_nets), ctypes.c_int(self.ONE), ctypes.c_int(nl.n_state),
            ctypes.c_void_p(self._d["state_net"].data_ptr()),
            ctypes.c_void_p(self._d["dff_d"].data_ptr()),
            ctypes.c_int(nl.n_input), ctypes.c_void_p(self._d["input_net"].data_ptr()),
            ctypes.c_int(len(self.pb)), ctypes.c_void_p(self._d["po_net"].data_ptr()),
            ctypes.c_int(cycles), ctypes.c_int(ng),
            ctypes.c_void_p(u_dev.data_ptr()), ctypes.c_void_p(po_dev.data_ptr()),
            ctypes.c_int(self.threads), ctypes.c_int(self.smem))
