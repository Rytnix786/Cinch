#!/usr/bin/env bash
# Run vLLM with Speculative Draft Decoding (Bash)
set -euo pipefail

TARGET_MODEL="${1:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
DRAFT_MODEL="${2:-Qwen/Qwen2.5-0.5B-Instruct}"
NUM_TOKENS="${3:-5}"
PORT="${4:-8000}"

echo "=== Launching vLLM with Speculative Decoding ($TARGET_MODEL + $DRAFT_MODEL) ==="

docker run --gpus all \
    --ipc=host \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -p "${PORT}:8000" \
    --env VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
    --name cinch-vllm-speculative \
    --rm -it \
    vllm/vllm-openai:v0.6.3.post1 \
    vllm serve "$TARGET_MODEL" \
    --quantization awq_marlin \
    --dtype auto \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --enforce-eager \
    --speculative-model "$DRAFT_MODEL" \
    --num-speculative-tokens "$NUM_TOKENS" \
    --port 8000
