#!/usr/bin/env bash
# Automated AutoAWQ Quantization Shell Script
set -euo pipefail

MODEL_PATH="${1:-Qwen/Qwen2.5-7B-Instruct}"
QUANT_PATH="${2:-models/Qwen2.5-7B-Instruct-AWQ}"
W_BIT="${3:-4}"
Q_GROUP_SIZE="${4:-128}"
VERSION="${5:-GEMM}"

echo "=== Running AutoAWQ Quantization: $MODEL_PATH -> $QUANT_PATH ==="

python scripts/quantize_awq.py \
    --model-path "$MODEL_PATH" \
    --quant-path "$QUANT_PATH" \
    --w-bit "$W_BIT" \
    --q-group-size "$Q_GROUP_SIZE" \
    --version "$VERSION"
