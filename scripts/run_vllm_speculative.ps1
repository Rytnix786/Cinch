# Run vLLM with Speculative Draft Decoding (PowerShell)
param(
    [string]$TargetModel = "Qwen/Qwen2.5-7B-Instruct-AWQ",
    [string]$DraftModel = "Qwen/Qwen2.5-0.5B-Instruct",
    [int]$NumSpeculativeTokens = 5,
    [int]$GpuPort = 8000
)

$ErrorActionPreference = "Stop"

Write-Host "=== Launching vLLM with Speculative Decoding ($TargetModel + Draft: $DraftModel) ===" -ForegroundColor Cyan

docker run --gpus all `
    --ipc=host `
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface `
    -p ${GpuPort}:8000 `
    --env VLLM_WSL2_ENABLE_PIN_MEMORY=1 `
    --name cinch-vllm-speculative `
    --rm -it `
    vllm/vllm-openai:v0.6.3.post1 `
    vllm serve $TargetModel `
    --quantization awq_marlin `
    --dtype auto `
    --gpu-memory-utilization 0.85 `
    --max-model-len 4096 `
    --enforce-eager `
    --speculative-model $DraftModel `
    --num-speculative-tokens $NumSpeculativeTokens `
    --port 8000
