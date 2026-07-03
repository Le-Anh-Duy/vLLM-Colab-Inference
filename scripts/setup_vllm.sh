#!/bin/bash
# Cai dat CUDA runtime + torch + vllm.
#
# Co 2 CACH CAI (chon qua PINNED_CUDA_WHEEL):
#
#   PINNED_CUDA_WHEEL=0 (mac dinh, dung cho dev tren Colab): cai torch qua
#   --extra-index-url roi `pip install vllm==VERSION` (khong suffix CUDA) -
#   day la cach da chay on dinh tren Colab tu truoc gio.
#
#   PINNED_CUDA_WHEEL=1 (dung khi can khop dung mot ban CUDA cu the, vd may
#   BTC): tai thang wheel `vllm-VERSION+cuXXX-...` tu GitHub Releases. CHU Y:
#   moi VLLM_VERSION chi publish MOT SO bien the CUDA nhat dinh (vd v0.24.0
#   chi co +cu129, +cpu, va ban mac dinh - KHONG co +cu128/+cu130). Script se
#   kiem tra URL ton tai truoc khi cai; neu khong co, xem danh sach that tai
#   https://github.com/vllm-project/vllm/releases/tag/v<VLLM_VERSION>
#
# 3 kich ban dung:
#   A. Dev tren Colab (mac dinh):
#        bash setup_vllm.sh
#   B. Nop bai cho BTC (CUDA da co san, khong tu apt install):
#        SKIP_CUDA=1 PINNED_CUDA_WHEEL=1 CUDA_VERSION=12.9 bash setup_vllm.sh
#   C. Test tren Colab gia lap CUDA cua BTC:
#        PINNED_CUDA_WHEEL=1 CUDA_VERSION=12.9 bash setup_vllm.sh
#      (neu truoc do trong cung session da cai ban khac -> phai restart kernel
#      truoc khi chay serve_vllm.sh, xem notebooks/vllm_colab.ipynb)
set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-13.0}"         # chap nhan "12.9", "12.9.1", hoac dang nen "129"
CPU_ARCH="${CPU_ARCH:-x86_64}"
VLLM_VERSION="${VLLM_VERSION:-0.24.0}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
VLLM_MANYLINUX_TAG="${VLLM_MANYLINUX_TAG:-manylinux_2_28}"
PINNED_CUDA_WHEEL="${PINNED_CUDA_WHEEL:-0}"

CUDA_KEYRING_URL="${CUDA_KEYRING_URL:-https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb}"

# Chuan hoa CUDA_VERSION ve major/minor, du nguoi dung truyen "12.9", "12.9.1"
# (bo qua patch) hay dang nen "129".
if [[ "$CUDA_VERSION" == *.* ]]; then
  IFS='.' read -r CUDA_MAJOR CUDA_MINOR _CUDA_PATCH <<< "$CUDA_VERSION"
else
  CUDA_MAJOR="${CUDA_VERSION:0:2}"
  CUDA_MINOR="${CUDA_VERSION:2}"
fi

if [[ -z "$CUDA_MAJOR" || -z "$CUDA_MINOR" || ! "$CUDA_MAJOR" =~ ^[0-9]+$ || ! "$CUDA_MINOR" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CUDA_VERSION='${CUDA_VERSION}' khong hop le. Dung dang 'MAJOR.MINOR' (vd '12.9') hoac nen 'MAJORMINOR' (vd '129')." >&2
  exit 1
fi

CUDA_DOTTED="${CUDA_MAJOR}.${CUDA_MINOR}"
CUDA_DASHED="${CUDA_MAJOR}-${CUDA_MINOR}"
CUDA_COMPACT="${CUDA_MAJOR}${CUDA_MINOR}"

CUDA_PACKAGE="${CUDA_PACKAGE:-cuda-cudart-${CUDA_DASHED}}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda-${CUDA_DOTTED}/targets/x86_64-linux/lib}"
CUDA_LIB_NAME="${CUDA_LIB_NAME:-libcudart.so.${CUDA_MAJOR}}"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu${CUDA_COMPACT}}"
VLLM_WHEEL_URL="${VLLM_WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_COMPACT}-cp38-abi3-${VLLM_MANYLINUX_TAG}_${CPU_ARCH}.whl}"

# SKIP_CUDA=1: dung khi CUDA/driver da duoc moi truong dich (vd may cham BTC)
# cung cap san - khong tu y apt install de lech voi cau hinh chuan cua ho.
SKIP_CUDA="${SKIP_CUDA:-0}"
SKIP_PYTHON_DEPS="${SKIP_PYTHON_DEPS:-0}"

echo "==> CUDA_VERSION=${CUDA_VERSION} (${CUDA_DOTTED}) | VLLM_VERSION=${VLLM_VERSION} | SKIP_CUDA=${SKIP_CUDA} | PINNED_CUDA_WHEEL=${PINNED_CUDA_WHEEL}"

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
  pip uninstall -y vllm torch torchvision torchaudio

  if [ "$PINNED_CUDA_WHEEL" = "1" ]; then
    echo "========================================"
    echo "2. DANG CAI VLLM ${VLLM_VERSION} (+cu${CUDA_COMPACT}) TU GITHUB RELEASE..."
    echo "========================================"
    http_code="$(curl -s -o /dev/null -w '%{http_code}' -L "$VLLM_WHEEL_URL")"
    if [ "$http_code" != "200" ]; then
      echo "ERROR: khong tim thay wheel cho vllm==${VLLM_VERSION}+cu${CUDA_COMPACT} (HTTP ${http_code})." >&2
      echo "       URL da thu: ${VLLM_WHEEL_URL}" >&2
      echo "       Kiem tra danh sach asset THAT cua version nay tai:" >&2
      echo "       https://github.com/vllm-project/vllm/releases/tag/v${VLLM_VERSION}" >&2
      echo "       roi chinh CUDA_VERSION / VLLM_MANYLINUX_TAG / VLLM_WHEEL_URL cho khop." >&2
      exit 1
    fi
    pip install "$VLLM_WHEEL_URL" --extra-index-url "$TORCH_INDEX_URL"
  else
    echo "========================================"
    echo "2. DANG DONG BO TORCH ${TORCH_VERSION} (cu${CUDA_COMPACT}) VA VLLM ${VLLM_VERSION}..."
    echo "========================================"
    pip install "torch==${TORCH_VERSION}" torchvision torchaudio --extra-index-url "$TORCH_INDEX_URL"
    pip install "vllm==${VLLM_VERSION}" --no-cache-dir
  fi
else
  echo "==> Bo qua buoc cai torch/vllm (SKIP_PYTHON_DEPS=1)"
fi

echo "========================================"
echo "Setup hoan tat. Dung scripts/serve_vllm.sh de khoi dong server."
echo "Neu ban vua doi CUDA_VERSION so voi lan chay truoc trong cung session"
echo "Python (da tung import torch/vllm), HAY RESTART SESSION truoc khi serve."
echo "========================================"
