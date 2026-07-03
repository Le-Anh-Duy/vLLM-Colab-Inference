#!/bin/bash
# Khoi dong vLLM OpenAI-compatible API server.
# Tat ca gia tri co the override qua bien moi truong hoac truyen truc tiep
# them tham so vllm o cuoi lenh goi script (se duoc noi vao VLLM_EXTRA_ARGS).
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# Cac toggle toi uu (Statement.txt muc 5: KV Cache Optimization). Mac dinh
# giu nguyen hanh vi vllm (prefix caching thuong da bat san, KV_CACHE_DTYPE
# rong = "auto"). Bat KV_CACHE_DTYPE=fp8 de thu quantize KV cache - kiem tra
# accuracy bang scripts/eval_gpqa.py truoc khi coi day la optimization chinh
# thuc, vi day la thay doi co the anh huong chat luong dau ra.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"

PREFIX_CACHING_FLAG="--enable-prefix-caching"
if [ "$ENABLE_PREFIX_CACHING" = "0" ]; then
  PREFIX_CACHING_FLAG="--no-enable-prefix-caching"
fi

KV_CACHE_DTYPE_FLAG=()
if [ -n "$KV_CACHE_DTYPE" ]; then
  KV_CACHE_DTYPE_FLAG=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

echo "========================================"
echo "DANG KHOI DONG VLLM API SERVER..."
echo "  model             = ${MODEL_NAME}"
echo "  host              = ${HOST}"
echo "  port              = ${PORT}"
echo "  prefix caching    = ${ENABLE_PREFIX_CACHING}"
echo "  kv cache dtype    = ${KV_CACHE_DTYPE:-auto}"
echo "========================================"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "$PREFIX_CACHING_FLAG" \
  "${KV_CACHE_DTYPE_FLAG[@]}" \
  $VLLM_EXTRA_ARGS "$@"
