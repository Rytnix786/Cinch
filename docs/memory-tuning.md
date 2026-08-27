# Memory Tuning: Decisions and Empirical Evidence

This document records the memory budgeting rationale and empirical test results for serving `Qwen2.5-7B-Instruct-AWQ` on a single NVIDIA GeForce RTX 3060 Ti (8192 MiB VRAM) under vLLM.

---

## 1. Hardware Constraints and Host Memory Topology

Serving on a local workstation differs from dedicated cloud infrastructure. The local GPU concurrently drives the host display server and Windows Desktop Window Manager (DWM).

| Memory Component | Allocation | Percentage of 8192 MiB |
|---|---|---|
| Total Physical VRAM | 8192.0 MiB | 100.0% |
| Host Windows DWM / Background Display Processes | ~1064.0 MiB | 13.0% |
| Available Free VRAM at vLLM Startup | 7128.0 MiB (~6.96 GiB) | 87.0% |

Because host display processes reserve approximately 1.04 GiB, vLLM cannot allocate the full 8192 MiB. Any configuration requesting more than 6.96 GiB crashes during initialization.

---

## 2. Mathematical KV Cache Derivation for Qwen2.5-7B

`Qwen/Qwen2.5-7B-Instruct-AWQ` uses Grouped-Query Attention (GQA) with the following architectural parameters:

- **Transformer Layers ($L$)**: 28
- **Key-Value Attention Heads ($H_{kv}$)**: 4
- **Attention Head Dimension ($D_h$)**: 128
- **Data Type**: FP16 / Half Precision ($B = 2$ bytes per element)

### Per-Token Memory Formula
For each token, the engine stores both Key and Value state tensors across all 28 layers:

$$\text{KV Bytes per Token} = 2 \times L \times H_{kv} \times D_h \times B$$
$$\text{KV Bytes per Token} = 2 \times 28 \times 4 \times 128 \times 2 = 57,344\text{ bytes} \approx 0.0546875\text{ MiB/token}$$

### Token Capacity and Concurrency
Given an available KV cache allocation $M_{kv}$ in MiB:

$$\text{Token Capacity} = \left\lfloor \frac{M_{kv} \times 1024 \times 1024}{57,344} \right\rfloor$$

$$\text{Maximum Concurrency at Context Length } C = \frac{\text{Token Capacity}}{C}$$

---

## 3. Empirical Configuration Matrix

We evaluated five configurations on the host hardware. The table below lists the measured outcomes and live telemetry.

| Configuration ID | `gpu_memory_utilization` | `max_model_len` | Execution Mode | Startup Status | Weights Memory | KV Cache Memory | Token Capacity | Max Concurrency (at `max_model_len`) | Host VRAM (`nvidia-smi`) |
|---|---|---|---|---|---|---|---|---|---|
| **A (Default vLLM)** | 0.90 (7.20 GiB) | 4096 | Torch Inductor + CUDA Graphs | **FAILED** (`ValueError`) | N/A | 0.00 GiB | 0 | 0.00x | 1064 MiB (pre-crash) |
| **B (Low Utilization Default)** | 0.80 (6.40 GiB) | 4096 | Torch Inductor + CUDA Graphs | **FAILED** (`ValueError`) | 5.29 GiB | 0.00 GiB | 0 | 0.00x | 6481 MiB (pre-crash) |
| **C (Conservative Eager)** | 0.80 (6.40 GiB) | 4096 | `--enforce-eager` | **PASSED** | 5.29 GiB | 0.56 GiB | 10,240 | 2.50x | 7472 MiB |
| **D (Production Target)** | 0.85 (6.80 GiB) | 4096 | `--enforce-eager` | **PASSED** | 5.29 GiB | 0.97 GiB | 18,096 | 4.42x | 7882 MiB |
| **E (Long Context Stress)** | 0.85 (6.80 GiB) | 8192 | `--enforce-eager` | **PASSED** | 5.29 GiB | 0.97 GiB | 18,096 | 2.21x | 7882 MiB |

---

## 4. Failure Analysis and Technical Root Cause

### Failure Case A: `gpu_memory_utilization = 0.90`
- **Observed Engine Output**:
  ```text
  ValueError: Free memory on device cuda:0 (6.96/8.0 GiB) on startup is less than desired GPU memory utilization (0.9, 7.2 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
  ```
- **Root Cause**: vLLM calculates the target pool as $0.90 \times 8.0\text{ GiB} = 7.20\text{ GiB}$. Because Windows reserves ~1.04 GiB for the desktop display pipeline, the host has only 6.96 GiB free on initialization. The startup check aborts before model loading.

### Failure Case B: `gpu_memory_utilization = 0.80` with PyTorch Inductor Compilation
- **Observed Engine Output**:
  ```text
  ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization when initializing the engine.
  ```
- **Root Cause**: Model weights consume 5.29 GiB. PyTorch Inductor compilation captures 48 CUDA graph sizes (1 to 512) and allocates scratch buffers during compilation. This warmup consumes an additional 1.11 GiB of VRAM workspace, pushing total consumption past the 6.40 GiB ceiling and leaving zero bytes for KV cache blocks.

---

## 5. Architectural Decision: `--enforce-eager` Mode

On GPUs with 16 GB or more VRAM, CUDA graph capture costs negligible percentage overhead. On an 8 GB consumer GPU, CUDA graph memory reservation creates severe contention with the model weights.

Passing `--enforce-eager`:
1. Disables PyTorch Dynamo bytecode transformations and Inductor graph workspace buffers.
2. Eliminates 48 static CUDA graph capture pools ($0.0\text{ GiB}$ CUDA graph memory vs ~$1.1\text{ GiB}$).
3. Frees $0.97\text{ GiB}$ exclusively for KV cache token allocation.

---

## 6. Context Length Decision: `max_model_len = 4096`

While Qwen2.5-7B supports up to 32,768 native context tokens, capping context length at 4096 tokens balances request length against concurrency.

- At **2048 tokens**, the engine supports up to **8.83x concurrency**, but truncates longer technical prompts.
- At **4096 tokens**, the engine maintains **4.42x concurrency** (18,096 tokens capacity), sufficient for multi-turn conversations, RAG contexts, and concurrent gateway load testing.
- At **8192 tokens**, maximum concurrency drops to **2.21x**, causing request queuing under minimal concurrency load.

---

## 7. Locked Configuration

The following parameters are locked for downstream gateway and benchmark milestones:

```yaml
MODEL_NAME: Qwen/Qwen2.5-7B-Instruct-AWQ
QUANTIZATION: awq
DTYPE: half
GPU_MEMORY_UTILIZATION: 0.85
MAX_MODEL_LEN: 4096
EXTRA_ARGS: --enforce-eager
VLLM_WSL2_ENABLE_PIN_MEMORY: 1
```

This configuration reserves:
- **5.29 GiB** for AWQ 4-bit model weights
- **0.27 GiB** for peak activation memory
- **0.97 GiB** for KV cache (18,096 tokens)
- **1.23 GiB** host headroom for Windows DWM and display drivers
