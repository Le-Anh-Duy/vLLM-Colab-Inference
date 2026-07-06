#!/bin/bash
# Khoi dong vLLM OpenAI-compatible API server.
# Tat ca gia tri co the override qua bien moi truong hoac truyen truc tiep
# them tham so vllm o cuoi lenh goi script (se duoc noi vao VLLM_EXTRA_ARGS).
#
# Mac dinh khop voi docker/docker-compose.yml (vong 1 so loai): model
# Qwen/Qwen3.5-2B, served-model-name=Qwen3.5-2B (khop field "model" trong
# trace-round1.jsonl), prefix caching on.
#
# MAX_MODEL_LEN: da doi tu 262144 (baseline BTC) xuong 65536. Da dem ky tu
# thuc te cua tat ca request trong trace-round1.jsonl: request dai nhat
# (12 message, hoi thoai nhieu luot) ~167.283 ky tu, uoc luong ~42.000-56.000
# token (~3-4 ky tu/token, chua dung tokenizer that). 65536 du margin ma
# khong lang phi VRAM nhu 262144 (qua du thua so voi trace thuc te).
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-2B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# Cac toggle toi uu (Statement.txt muc 3: KV Cache & Memory), khop voi
# docker/docker-compose.yml de hanh vi test tren Colab phan anh dung ban se
# nop. Xem giai thich chi tiet (vi sao bat/tat tung cai) trong comment cua
# docker/docker-compose.yml.
#
# CHU Y: --swap-space DA BI BO - khong con ton tai trong CLI cua ban vllm
# dang dung (kien truc engine V1 khong con swap KV cache ra CPU RAM kieu cu,
# dung recompute khi preempt thay vi swap). --disable-log-requests cung doi
# ten thanh --no-enable-log-requests (BooleanOptionalAction). Da tung dung
# ten flag cu va bi loi "unrecognized arguments" khi BTC chay that.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
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
  DISABLE_LOG_REQUESTS_FLAG=(--no-enable-log-requests)
fi

echo "========================================"
echo "DANG KHOI DONG VLLM API SERVER..."
echo "  model             = ${MODEL_NAME}"
echo "  served-model-name = ${SERVED_MODEL_NAME}"
echo "  max-model-len     = ${MAX_MODEL_LEN}"
echo "  host              = ${HOST}"
echo "  port              = ${PORT}"
echo "  gpu-mem-util      = ${GPU_MEMORY_UTILIZATION}"
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
  "$PREFIX_CACHING_FLAG" \
  "${KV_CACHE_DTYPE_FLAG[@]}" \
  "${DISABLE_LOG_REQUESTS_FLAG[@]}" \
  $VLLM_EXTRA_ARGS "$@"
