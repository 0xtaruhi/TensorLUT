"""CUDA-core LUT truth-table kernels used as a no-Tensor-Core GPU baseline."""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

import torch

_HERE = os.path.dirname(__file__)
_CU = os.path.join(_HERE, "lut_cuda.cu")
_CACHE = os.path.expanduser("~/.cache/rtlgemm")


def _cu13():
    return os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "nvidia", "cu13")


def _build():
    os.makedirs(_CACHE, exist_ok=True)
    cu = _cu13()
    h = hashlib.md5(open(_CU, "rb").read()).hexdigest()[:12]
    so = os.path.join(_CACHE, f"lut_cuda_{h}.so")
    if not os.path.exists(so):
        r = subprocess.run(
            [os.path.join(cu, "bin", "nvcc"), "-arch=sm_89", "-ccbin", "g++", "-O3",
             "--shared", "-Xcompiler", "-fPIC", f"-I{cu}/include", _CU, "-o", so],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc build failed:\n{r.stderr[-2000:]}")
    return so


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(_build())
        _LIB.launch_lut_init_state.restype = None
        _LIB.launch_lut_scatter_inputs.restype = None
        _LIB.launch_lut_eval_layer.restype = None
        _LIB.launch_lut_commit_state.restype = None
        _LIB.launch_lut_gather_cols.restype = None
    return _LIB


def _grid(n: int) -> int:
    return min(65535, max(1, (int(n) + 255) // 256))


def _stream(t: torch.Tensor):
    return ctypes.c_void_p(torch.cuda.current_stream(t.device).cuda_stream)


def init_state(V, zero_col: int, one_col: int, state_cols_i32):
    batch, n_cols = V.shape
    _lib().launch_lut_init_state(
        ctypes.c_void_p(V.data_ptr()), ctypes.c_int(batch), ctypes.c_int(n_cols),
        ctypes.c_int(zero_col), ctypes.c_int(one_col),
        ctypes.c_int(state_cols_i32.numel()), ctypes.c_void_p(state_cols_i32.data_ptr()),
        ctypes.c_int(_grid(batch * (state_cols_i32.numel() + 2))), _stream(V))


def scatter_inputs(V, input_cols_i32, u_t):
    batch, n_cols = V.shape
    n_input = input_cols_i32.numel()
    if n_input == 0:
        return
    _lib().launch_lut_scatter_inputs(
        ctypes.c_void_p(V.data_ptr()), ctypes.c_int(batch), ctypes.c_int(n_cols),
        ctypes.c_int(n_input), ctypes.c_void_p(input_cols_i32.data_ptr()),
        ctypes.c_void_p(u_t.data_ptr()), ctypes.c_int(_grid(batch * n_input)), _stream(V))


def eval_layer(V, in_cols_i32, out_cols_i32, widths_u8, tables_u64):
    batch, n_cols = V.shape
    n_lut = out_cols_i32.numel()
    if n_lut == 0:
        return
    _lib().launch_lut_eval_layer(
        ctypes.c_void_p(V.data_ptr()), ctypes.c_int(batch), ctypes.c_int(n_cols),
        ctypes.c_int(n_lut), ctypes.c_void_p(in_cols_i32.data_ptr()),
        ctypes.c_void_p(out_cols_i32.data_ptr()), ctypes.c_void_p(widths_u8.data_ptr()),
        ctypes.c_void_p(tables_u64.data_ptr()), ctypes.c_int(_grid(batch * n_lut)),
        _stream(V))


def commit_state(V, state_cols_i32, dff_cols_i32):
    batch, n_cols = V.shape
    n_state = state_cols_i32.numel()
    _lib().launch_lut_commit_state(
        ctypes.c_void_p(V.data_ptr()), ctypes.c_int(batch), ctypes.c_int(n_cols),
        ctypes.c_int(n_state), ctypes.c_void_p(state_cols_i32.data_ptr()),
        ctypes.c_void_p(dff_cols_i32.data_ptr()), ctypes.c_int(_grid(batch * n_state)),
        _stream(V))


def gather_cols(V, cols_i32, out):
    batch, n_cols = V.shape
    n_out = cols_i32.numel()
    if n_out == 0:
        return
    _lib().launch_lut_gather_cols(
        ctypes.c_void_p(V.data_ptr()), ctypes.c_int(batch), ctypes.c_int(n_cols),
        ctypes.c_int(n_out), ctypes.c_void_p(cols_i32.data_ptr()),
        ctypes.c_void_p(out.data_ptr()), ctypes.c_int(_grid(batch * n_out)), _stream(V))
