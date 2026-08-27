# Milestone 1: Local vLLM Serving in Docker

## Hardware & Model Target
- **Target GPU**: NVIDIA GeForce RTX 3060 Ti (8192 MiB total VRAM, Ampere compute capability 8.6)
- **Model**: `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit Marlin AWQ quantized weights, FlashAttention-2)
- **Runtime Environment**: Docker Desktop WSL2 backend, CUDA 12.4 passthrough
- **Configuration**:
  - `--gpu-memory-utilization 0.85` (0.85 × 8192 MiB = 6963.2 MiB = 6.8 GiB allocation target)
  - `--max-model-len 4096`
  - `--enforce-eager` (eliminates CUDA graph compilation overhead on 8GB VRAM)
  - `--dtype half`
  - `VLLM_WSL2_ENABLE_PIN_MEMORY=1` (enables WSL2 UVA tensor staging)

---

## Live Verification & Benchmark Telemetry

### 1. Server Initialization
- **Safetensors Shards**: 2 shards loaded in 42.40s
- **Model Memory**: 5.29 GiB weights
- **KV Cache Allocation**: 0.97 GiB (18,096 tokens, 4.42x max concurrency at 4096 tokens)
- **Host VRAM Footprint (`nvidia-smi`)**: 7882 MiB / 8192 MiB under load (includes Windows host display manager and vLLM server)

### 2. Live Smoke Test Results
- **Endpoint**: `http://localhost:8000/v1/chat/completions`
- **Model Served**: `Qwen/Qwen2.5-7B-Instruct-AWQ`
- **Request Prompt Tokens**: 39 tokens
- **Generated Completion Tokens**: 25 tokens
- **Single-Request End-to-End Latency**: ~1.054s

---

## Prerequisites
1. **NVIDIA Driver**: CUDA-capable driver installed on host (tested on NVIDIA Driver 610.88 / CUDA 13.3/12.4).
2. **Docker Desktop**: With WSL2 backend and NVIDIA GPU passthrough enabled.

---

## Quickstart Instructions

### 1. Configure Environment
Copy `.env.example` to `.env` (optional; defaults in `docker-compose.vllm.yml` are preconfigured):
```powershell
cp .env.example .env
```

### 2. Start vLLM Serving Container
```powershell
# Using PowerShell helper:
.\scripts\start_vllm.ps1 -Action start

# Or directly with Docker Compose:
docker compose -f docker-compose.vllm.yml up -d
```

### 3. Monitor Initialization & Readiness
```powershell
# Wait for healthcheck to pass:
.\scripts\start_vllm.ps1 -Action wait

# Tail logs:
.\scripts\start_vllm.ps1 -Action logs
```

### 4. Run Verification Smoke Test
Once the server reports healthy on port 8000:
```powershell
python scripts/smoke_test.py --base-url http://localhost:8000
```

### 5. Run Local Unit Tests
```powershell
python -m pytest tests/ -v
python -m ruff check .
```

