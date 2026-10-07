#!/usr/bin/env bash
# Provision a Python 3.12 venv with CUDA 13 PyTorch and the pip-packaged nvcc that
# rtlgemm/kernels/b1_anf.py uses to build the b1 tensor-core kernels (no system CUDA toolkit).
# Measured configuration: torch 2.12.1+cu130, nvidia-cuda-nvcc 13.3.73, RTX 4090 (sm_89).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    torch numpy pytest "nvidia-cuda-nvcc==13.3.*" "nvidia-cuda-crt==13.3.*"

.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")
print("device", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
PY
echo "[setup] done."
