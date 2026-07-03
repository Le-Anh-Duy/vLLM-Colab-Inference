#!/bin/bash
# Khoi dong vLLM OpenAI-compatible API server.
# Tat ca gia tri co the override qua bien moi truong hoac truyen truc tiep
# them tham so vllm o cuoi lenh goi script (se duoc noi vao VLLM_EXTRA_ARGS).
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.3}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

echo "========================================"
echo "DANG KHOI DONG VLLM API SERVER..."
echo "  model  = ${MODEL_NAME}"
echo "  host   = ${HOST}"
echo "  port   = ${PORT}"
echo "========================================"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  $VLLM_EXTRA_ARGS "$@"
