#!/bin/bash
# Cai dat CUDA runtime + torch + vllm.
#
# Tat ca gia tri co the override qua bien moi truong. Diem quan trong nhat la
# CUDA_VERSION (dang 3 chu so: "128" = CUDA 12.8, "129" = CUDA 12.9,
# "130" = CUDA 13.0). Tu bien nay script suy ra apt package, duong dan lib,
# torch index va dung dung wheel vllm chinh thuc (vllm+cuXXX) thay vi de pip
# tu chon ban mac dinh (co the doi theo thoi gian).
#
# 3 kich ban dung:
#   A. Dev tren Colab (mac dinh):            bash setup_vllm.sh
#   B. Nop bai cho BTC (CUDA da co san):      SKIP_CUDA=1 CUDA_VERSION=128 bash setup_vllm.sh
#   C. Test tren Colab gia lap CUDA cua BTC:  CUDA_VERSION=128 bash setup_vllm.sh
#      (neu truoc do trong cung session da cai ban khac -> phai restart kernel
#      truoc khi chay serve_vllm.sh, xem notebooks/vllm_colab.ipynb)
set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-130}"          # 128 / 129 / 130 ...
CPU_ARCH="${CPU_ARCH:-x86_64}"
VLLM_VERSION="${VLLM_VERSION:-0.24.0}"

CUDA_KEYRING_URL="${CUDA_KEYRING_URL:-https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb}"

CUDA_MAJOR="${CUDA_VERSION:0:2}"
CUDA_MINOR="${CUDA_VERSION:2}"
CUDA_DOTTED="${CUDA_MAJOR}.${CUDA_MINOR}"
CUDA_DASHED="${CUDA_MAJOR}-${CUDA_MINOR}"

CUDA_PACKAGE="${CUDA_PACKAGE:-cuda-cudart-${CUDA_DASHED}}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda-${CUDA_DOTTED}/targets/x86_64-linux/lib}"
CUDA_LIB_NAME="${CUDA_LIB_NAME:-libcudart.so.${CUDA_MAJOR}}"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu${CUDA_VERSION}}"
VLLM_WHEEL_URL="${VLLM_WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_35_${CPU_ARCH}.whl}"

# SKIP_CUDA=1: dung khi CUDA/driver da duoc moi truong dich (vd may cham BTC)
# cung cap san - khong tu y apt install de lech voi cau hinh chuan cua ho.
SKIP_CUDA="${SKIP_CUDA:-0}"
SKIP_PYTHON_DEPS="${SKIP_PYTHON_DEPS:-0}"

echo "==> CUDA_VERSION=${CUDA_VERSION} (${CUDA_DOTTED}) | VLLM_VERSION=${VLLM_VERSION} | SKIP_CUDA=${SKIP_CUDA}"

if [ "$SKIP_CUDA" != "1" ]; then
  echo "========================================"
  echo "1. DANG CAU HINH CUDA ${CUDA_DOTTED} RUNTIME..."
  echo "========================================"
  rm -f cuda-keyring*.deb
  wget -q "$CUDA_KEYRING_URL"
  dpkg -i cuda-keyring_1.1-1_all.deb
  apt-get update -qq
  apt-get install -y "$CUDA_PACKAGE"
  ln -sf "${CUDA_LIB_DIR}/${CUDA_LIB_NAME}" "/usr/local/lib/${CUDA_LIB_NAME}"
  ldconfig
else
  echo "==> Bo qua buoc cai CUDA (SKIP_CUDA=1) - dung CUDA runtime co san cua moi truong"
fi

if [ "$SKIP_PYTHON_DEPS" != "1" ]; then
  echo "========================================"
  echo "2. DANG DONG BO TORCH VA VLLM ${VLLM_VERSION} (cu${CUDA_VERSION})..."
  echo "========================================"
  pip uninstall -y vllm torch torchvision torchaudio
  pip install "$VLLM_WHEEL_URL" --extra-index-url "$TORCH_INDEX_URL"
else
  echo "==> Bo qua buoc cai torch/vllm (SKIP_PYTHON_DEPS=1)"
fi

echo "========================================"
echo "Setup hoan tat. Dung scripts/serve_vllm.sh de khoi dong server."
echo "Neu ban vua doi CUDA_VERSION so voi lan chay truoc trong cung session"
echo "Python (da tung import torch/vllm), HAY RESTART SESSION truoc khi serve."
echo "========================================"
