#!/bin/bash
# Cai dat CUDA runtime + torch + vllm tren Colab.
# Tat ca gia tri co the override qua bien moi truong, mac dinh giu nguyen
# cau hinh da chay duoc: CUDA 13.0 runtime, torch 2.11.0 (cu130), vllm 0.24.0.
set -euo pipefail

CUDA_KEYRING_URL="${CUDA_KEYRING_URL:-https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb}"
CUDA_PACKAGE="${CUDA_PACKAGE:-cuda-cudart-13-0}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda-13.0/targets/x86_64-linux/lib}"
CUDA_LIB_NAME="${CUDA_LIB_NAME:-libcudart.so.13}"

TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
VLLM_VERSION="${VLLM_VERSION:-0.24.0}"

SKIP_CUDA="${SKIP_CUDA:-0}"
SKIP_PYTHON_DEPS="${SKIP_PYTHON_DEPS:-0}"

if [ "$SKIP_CUDA" != "1" ]; then
  echo "========================================"
  echo "1. DANG CAU HINH CUDA RUNTIME..."
  echo "========================================"
  rm -f cuda-keyring*.deb
  wget -q "$CUDA_KEYRING_URL"
  dpkg -i cuda-keyring_1.1-1_all.deb
  apt-get update -qq
  apt-get install -y "$CUDA_PACKAGE"
  ln -sf "${CUDA_LIB_DIR}/${CUDA_LIB_NAME}" "/usr/local/lib/${CUDA_LIB_NAME}"
  ldconfig
else
  echo "==> Bo qua buoc cai CUDA (SKIP_CUDA=1)"
fi

if [ "$SKIP_PYTHON_DEPS" != "1" ]; then
  echo "========================================"
  echo "2. DANG DONG BO PYTORCH ${TORCH_VERSION} VA VLLM ${VLLM_VERSION}..."
  echo "========================================"
  pip uninstall -y vllm torch torchvision torchaudio
  pip install "torch==${TORCH_VERSION}" torchvision torchaudio --extra-index-url "$TORCH_INDEX_URL"
  pip install "vllm==${VLLM_VERSION}" --no-cache-dir
else
  echo "==> Bo qua buoc cai torch/vllm (SKIP_PYTHON_DEPS=1)"
fi

echo "========================================"
echo "Setup hoan tat. Dung scripts/serve_vllm.sh de khoi dong server."
echo "========================================"
