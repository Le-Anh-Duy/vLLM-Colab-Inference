#!/bin/bash
# Khoi dong vLLM OpenAI-compatible API server.
# Tat ca gia tri co the override qua bien moi truong hoac truyen truc tiep
# them tham so vllm o cuoi lenh goi script (se duoc noi vao VLLM_EXTRA_ARGS).
#
# Mac dinh khop voi baseline-and-input/docker-compose-baseline.yml (vong 1
# so loai): model Qwen/Qwen3.5-2B, served-model-name=Qwen3.5-2B (khop field
# "model" trong trace-round1.jsonl), max-model-len=262144, prefix caching on.
#
# CHU Y VRAM: moi trong BTC cham la 1 lat MiG H200 chi 18GB VRAM. Context
# 262144 token o 18GB la rat cang - GPU_MEMORY_UTILIZATION mac dinh o day
# (0.85) la cho GPU dev tren Colab (thuong nhieu VRAM hon 18GB); khi test
# tren GPU co VRAM tuong duong 18-24GB, cân nhac dung 0.95 nhu baseline that
# va/hoac giam --max-model-len, va gan nhu chac chan can KV_CACHE_DTYPE=fp8
# de du cho KV cache.
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-2B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# Cac toggle toi uu (Statement.txt muc 3: KV Cache & Memory), khop voi
# docker/docker-compose.yml de hanh vi test tren Colab phan anh dung ban se
# nop. Xem giai thich chi tiet (vi sao bat/tat tung cai) trong comment cua
# docker/docker-compose.yml.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
SWAP_SPACE="${SWAP_SPACE:-1}"
DISABLE_LOG_REQUESTS="${DISABLE_LOG_REQUESTS:-1}"

PREFIX_CACHING_FLAG="--enable-prefix-caching"
if [ "$ENABLE_PREFIX_CACHING" = "0" ]; then
  PREFIX_CACHING_FLAG="--no-enable-prefix-caching"
fi

KV_CACHE_DTYPE_FLAG=()
if [ -n "$KV_CACHE_DTYPE" ]; then
  KV_CACHE_DTYPE_FLAG=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

DISABLE_LOG_REQUESTS_FLAG=()
if [ "$DISABLE_LOG_REQUESTS" = "1" ]; then
  DISABLE_LOG_REQUESTS_FLAG=(--disable-log-requests)
fi

echo "========================================"
echo "DANG KHOI DONG VLLM API SERVER..."
echo "  model             = ${MODEL_NAME}"
echo "  served-model-name = ${SERVED_MODEL_NAME}"
echo "  max-model-len     = ${MAX_MODEL_LEN}"
echo "  host              = ${HOST}"
echo "  port              = ${PORT}"
echo "  gpu-mem-util      = ${GPU_MEMORY_UTILIZATION}"
echo "  swap-space        = ${SWAP_SPACE} GiB"
echo "  prefix caching    = ${ENABLE_PREFIX_CACHING}"
echo "  kv cache dtype    = ${KV_CACHE_DTYPE:-auto}"
echo "========================================"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --swap-space "$SWAP_SPACE" \
  "$PREFIX_CACHING_FLAG" \
  "${KV_CACHE_DTYPE_FLAG[@]}" \
  "${DISABLE_LOG_REQUESTS_FLAG[@]}" \
  $VLLM_EXTRA_ARGS "$@"
