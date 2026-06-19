#!/usr/bin/env bash
# Build this fork's own SM70 (V100) wheels from source: vllm + flash_attn_v100.
#
# Produces self-contained wheels (vendored TurboMind SM70 AWQ GEMM + GGUF csrc
# fp16 clamps + flash_attn_v100 HDIM templates all compiled in), so there is no
# dependency on 1Cat's prebuilt wheel and no .py overlay drift.
#
# Requirements (on the build host -- a GPU is NOT needed to compile, only nvcc):
#   - CUDA toolkit with nvcc (CUDA 12.8 matches torch 2.9.1+cu128)
#   - the target torch already importable in the active Python/venv
#
# Usage:
#   tools/build_sm70_wheels.sh [output_dir]
#
# Env overrides:
#   CUDA_HOME   (default: /usr/local/cuda-12.8)
#   MAX_JOBS    (default: nproc, capped at 32)
#   NVCC_THREADS(default: 4)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${REPO_ROOT}/wheels}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="7.0"
export VLLM_TARGET_DEVICE="cuda"
export NVCC_THREADS="${NVCC_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-$(( $(nproc) < 32 ? $(nproc) : 32 ))}"

echo ">> CUDA_HOME=${CUDA_HOME}"
command -v nvcc >/dev/null || { echo "ERROR: nvcc not found on PATH"; exit 1; }
nvcc --version | tail -2
python -c "import torch; print('torch', torch.__version__)" \
    || { echo "ERROR: torch not importable in the active environment"; exit 1; }
echo ">> TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS} NVCC_THREADS=${NVCC_THREADS}"

mkdir -p "${OUT_DIR}"

echo ">> Building vllm wheel (this is the long one) ..."
python -m pip wheel "${REPO_ROOT}" --no-build-isolation --no-deps -w "${OUT_DIR}"

echo ">> Building flash_attn_v100 wheel ..."
python -m pip wheel "${REPO_ROOT}/flash-attention-v100" --no-build-isolation --no-deps -w "${OUT_DIR}"

echo ">> Done. Wheels in ${OUT_DIR}:"
ls -la "${OUT_DIR}"/*.whl
