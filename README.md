# Cinch

Self-hosted, high-throughput LLM inference-serving platform featuring 4-bit quantized model serving (vLLM with AWQ Marlin kernels), a stateless FastAPI API Gateway with sliding-window rate limiting, multi-node Kubernetes orchestration on k3d, horizontal pod autoscaling under load, and an empirical benchmark suite comparing throughput, latency, and quality against a naive Hugging Face Transformers baseline.

---

## 1. Executive Summary & Key Results

All performance claims and benchmarks trace directly to raw JSON execution traces in [`benchmarks/results/`](benchmarks/results/).

| Metric | Naive HF Baseline | Cinch (vLLM + AWQ) | Delta / Speedup |
|---|---|---|---|
| **Peak Throughput ($C=16$)** | $30.5\text{ tok/s}$ | **$331.0\text{ tok/s}$** | **$10.86\times$ Throughput Gain** |
| **p95 Request Latency ($C=16$)** | $32.03\text{ s}$ | **$6.29\text{ s}$** | **$5.09\times$ Latency Reduction** |
| **p50 Request Latency ($C=16$)** | $26.85\text{ s}$ | **$3.88\text{ s}$** | **$6.92\times$ Latency Reduction** |
| **Quantization Quality Retention** | $100.0\%$ (Reference) | **$97.2\%$ Retention Index** | $100\%$ Code AST / $100\%$ Math Syntax |
| **Gateway Proxy Overhead** | N/A | **$+2.6\%$ p95 latency** | Negligible proxy cost ($0.16\text{s}$) |
| **Kubernetes HPA Scaling** | N/A | **$2 \to 6$ Replicas** | Scaled across 3 k3d nodes at $223\%$ CPU |

---

## 2. System Architecture

Cinch decouples stateless routing/rate-limiting from GPU-accelerated inference:

```
                                  [ Client Traffic ]
                                          |
                                          v (Port 8081)
+-----------------------------------------------------------------------------------------+
|                                  k3d Multi-Node Cluster                                 |
|                                                                                         |
|                         [ Ingress: Traefik LoadBalancer ]                               |
|                                         |                                               |
|                    +--------------------+--------------------+                          |
|                    |                                         |                          |
|                    v                                         v                          |
|  +-----------------------------------+     +-----------------------------------+        |
|  | Node: cinch-cluster-agent-0       |     | Node: cinch-cluster-agent-1       |        |
|  |  [Pod: cinch-gateway-pod-1]       |     |  [Pod: cinch-gateway-pod-2]       |        |
|  |   - Bearer / API Key Auth         |     |   - Bearer / API Key Auth         |        |
|  |   - Sliding-Window Rate Limiter   |     |   - Sliding-Window Rate Limiter   |        |
|  |   - SSE Streaming Proxy           |     |   - SSE Streaming Proxy           |        |
|  |   - HPA: 2 -> 6 Replicas (50% CPU)|     |   - HPA: 2 -> 6 Replicas (50% CPU)|        |
|  +-----------------------------------+     +-----------------------------------+        |
|                    |                                         |                          |
+--------------------|-----------------------------------------|--------------------------+
                     +--------------------+--------------------+
                                          | (Cluster DNS / host.k3d.internal:8000)
                                          v
                      +----------------------------------------+
                      | Host GPU Docker Daemon                 |
                      |  [ Container: cinch-vllm ]             |
                      |   - Engine: vLLM v0.6.3 (Marlin Kernel)|
                      |   - Model: Qwen2.5-7B-Instruct-AWQ     |
                      |   - Precision: W4A16 Quantization      |
                      |   - Hardware: RTX 3060 Ti (8GB VRAM)   |
                      +----------------------------------------+
```

---

## 3. Mathematical KV Cache Budgeting & Memory Profile

Target Hardware: **NVIDIA GeForce RTX 3060 Ti (8,192 MiB VRAM)**  
Serving Engine: **vLLM with Marlin AWQ W4A16 GEMM Kernels**

### KV Cache Memory Formula for Qwen2.5-7B (Grouped-Query Attention)

For Grouped-Query Attention (GQA) with $L=28$ layers, $H_{kv}=4$ key-value heads, $D_{head}=128$, and 16-bit FP16 precision ($2\text{ bytes/element}$):

$$\text{Bytes per Token} = 2 \times L \times H_{kv} \times D_{head} \times 2 = 2 \times 28 \times 4 \times 128 \times 2 = 57,344\text{ bytes/token}$$

### Production Allocation Decision Table

| Parameter | Selected Value | Engineering Rationale |
|---|---|---|
| `gpu_memory_utilization` | `0.85` ($6,963\text{ MiB}$) | Reserves $1,229\text{ MiB}$ headroom for PyTorch activation overhead and desktop OS display compositor. |
| `--max-model-len` | `4096` | Accommodates 16 concurrent requests at 256 tokens ($16 \times 256 \times 57,344\text{ B} = 224\text{ MiB}$) with zero OOM risk. |
| `--enforce-eager` | `Enabled` | Avoids PyTorch CUDA Graph capture allocations ($500\text{--}1,200\text{ MiB}$), maximizing usable KV cache on 8GB VRAM. |
| `VLLM_WSL2_ENABLE_PIN_MEMORY` | `1` | Prevents CUDA page-locked memory exhaustion across WSL2/Docker host boundary. |

---

## 4. Empirical Benchmark Dossier

Benchmarks were conducted across concurrency tiers $C \in \{1, 4, 8, 16\}$ using a fixed dataset of 16 multi-turn evaluation prompts ([`benchmarks/prompts.json`](benchmarks/prompts.json)).

### 4.1. vLLM+AWQ vs Naive Transformers Baseline

Data sources: [`benchmarks/results/baseline_hf.json`](benchmarks/results/baseline_hf.json), [`benchmarks/results/vllm_awq.json`](benchmarks/results/vllm_awq.json)

| Concurrency ($C$) | Naive HF Throughput | vLLM+AWQ Throughput | Speedup | Naive HF p95 Latency | vLLM+AWQ p95 Latency | Latency Reduction |
|---|---|---|---|---|---|---|
| **$C=1$** | $30.4\text{ tok/s}$ | $34.7\text{ tok/s}$ | **$1.14\times$** | $1.76\text{ s}$ | $1.74\text{ s}$ | **$1.01\times$** |
| **$C=4$** | $30.5\text{ tok/s}$ | $132.8\text{ tok/s}$ | **$4.35\times$** | $8.07\text{ s}$ | $2.14\text{ s}$ | **$3.77\times$** |
| **$C=8$** | $30.5\text{ tok/s}$ | $228.4\text{ tok/s}$ | **$7.48\times$** | $16.03\text{ s}$ | $3.57\text{ s}$ | **$4.49\times$** |
| **$C=16$** | $30.5\text{ tok/s}$ | **$331.0\text{ tok/s}$** | **$10.86\times$** | $32.03\text{ s}$ | **$6.29\text{ s}$** | **$5.09\times$** |

```
Throughput Scaling (tok/s vs Concurrency)
350 |                                                       * (331.0 tok/s)
300 |                                            * (228.4)
250 |
200 |
150 |                              * (132.8)
100 |
 50 |      * (34.7)
  0 +------+-----------------------+-------------+----------+
    HF:   30.4                    30.5          30.5       30.5  (Flatline)
          C=1                     C=4           C=8        C=16
```

### 4.2. API Gateway Proxy Overhead

Data source: [`benchmarks/results/gateway_vllm_awq.json`](benchmarks/results/gateway_vllm_awq.json)

| Concurrency ($C$) | Direct vLLM p95 | Gateway + vLLM p95 | Proxy Delta ($\Delta$) | Overhead % |
|---|---|---|---|---|
| **$C=1$** | $1.74\text{ s}$ | $1.79\text{ s}$ | $+0.05\text{ s}$ | $+2.8\%$ |
| **$C=4$** | $2.14\text{ s}$ | $2.18\text{ s}$ | $+0.04\text{ s}$ | $+1.8\%$ |
| **$C=8$** | $3.57\text{ s}$ | $3.68\text{ s}$ | $+0.11\text{ s}$ | $+3.0\%$ |
| **$C=16$** | $6.29\text{ s}$ | $6.45\text{ s}$ | $+0.16\text{ s}$ | $+2.6\%$ |

---

## 5. Quantization Quality Equivalence Evaluation

Data source: [`benchmarks/results/quality_eval.json`](benchmarks/results/quality_eval.json)

Quality retention was verified against 9 held-out evaluation tasks spanning 5 distinct domains:

| Category | Tasks Evaluated | Quality Score | Metric Evaluated |
|---|---|---|---|
| **Code Generation** | 2 | **$100.0\%$** | Syntax correctness validated via Python AST (`ast.parse()`). |
| **Mathematical Reasoning** | 2 | **$100.0\%$** | Multi-step numerical calculation accuracy. |
| **Factual QA / Extraction** | 2 | **$87.5\%$** | Key factual entity and constraint matching. |
| **Format Constraints** | 2 | **$100.0\%$** | Schema validation (`json.loads()`, sentence counts). |
| **Summarization** | 1 | **$100.0\%$** | Core theme and keyword density retention. |
| **Overall Retention Index** | **9** | **$97.2\%$** | **Quantized output matches reference output fidelity.** |

---

## 6. Kubernetes Horizontal Pod Autoscaler (HPA)

Data source: [`benchmarks/results/hpa_scaling.json`](benchmarks/results/hpa_scaling.json)  
Test Load: 24 concurrent workers, **26,296 requests** processed.

```
Replicas & CPU Utilization Over Time
  6 Replicas |                             +---------------------------+
             |                            / (223% CPU Peak)            |
  4 Replicas |                   +-------+                             |
             |                  / (88% CPU)                            |
  2 Replicas | +---------------+                                       +--------------+
  (Min)      | (1-4% CPU)                                              (1% CPU Cooldown)
             +-----------------+---------+-----------------------------+--------------+
             0s               50s       60s                           95s            135s
             [----- BASELINE -----] [----- LOAD INJECTION -----] [----- COOLDOWN -----]
```

- **Scale-Up Trigger (51.1s)**: CPU crossed 50% target to 88%, scaling gateway from 2 to 4 replicas.
- **Max Scale Trigger (66.5s)**: CPU peaked at 223%, scaling to max limit of 6 replicas across all 3 k3d nodes.
- **Cooldown (95.1s – 135s)**: Traffic stopped, CPU dropped to 1%, and 60-second stabilization window gradually scaled pods back to minReplicas (2).

---

## 7. Honest Scope Boundaries & Production Extension Path (PRD §6)

### What is In Scope (Tested & Verified)
- 4-bit quantized AWQ Marlin inference serving on local NVIDIA RTX 3060 Ti GPU.
- Full empirical comparison against Hugging Face Transformers baseline.
- Quality equivalence validation with AST parsing.
- Multi-node Kubernetes orchestration on local k3d cluster with Ingress routing.
- Real CPU-based Horizontal Pod Autoscaling of the stateless FastAPI gateway under load.

### What is Explicitly Out of Scope (Hardware Boundary)
- Live multi-GPU node scaling across physical cloud clusters (this workstation has one physical GPU).

### Production Cloud Extension Path (`k8s/production-extension/`)
For multi-node cloud deployments (GKE / EKS / AKS):
1. **GPU Worker Daemon (`k8s/production-extension/vllm-gpu-deployment.yaml`)**: Declares `nvidia.com/gpu: 1` limits, node affinity (`accelerator: nvidia-gpu`), `/dev/shm` shared memory mounts (4Gi), and persistent volume claims for Hugging Face model caching.
2. **KEDA Queue-Depth Autoscaler (`k8s/production-extension/keda-scaledobject.yaml`)**: Replaces CPU autoscaling with Prometheus-backed inference queue depth metrics (`vllm:num_requests_waiting > 4`) for reactive GPU pod provisioning.

---

## 8. Quickstart & Reproduction Guide

### Prerequisites
- Docker Desktop with NVIDIA GPU Container Toolkit enabled.
- Python 3.12+
- `kubectl` and `k3d` (`winget install k3d.k3d`)

### 1. Launch GPU Inference Backend (vLLM)
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_vllm.ps1
```

### 2. Deploy Multi-Node Kubernetes Stack
```powershell
powershell -ExecutionPolicy Bypass -File scripts/k8s_deploy.ps1
```

### 3. Run Automated Tests
```powershell
python -m pytest tests/ -v
python -m ruff check .
```

### 4. Run Benchmark Suite & Comparison
```powershell
# Run benchmark against Gateway
python benchmarks/runner.py --endpoint-type gateway --concurrency 1 4 8 16 --output benchmarks/results/gateway_vllm_awq.json

# Generate comparison summary table
python benchmarks/compare.py --baseline benchmarks/results/baseline_hf.json --vllm benchmarks/results/vllm_awq.json --gateway benchmarks/results/gateway_vllm_awq.json
```

### 5. Run Quality Evaluation & HPA Load Test
```powershell
# Run quality evaluation
python evals/runner.py --output benchmarks/results/quality_eval.json

# Run HPA load test and record scaling
python scripts/load_test_hpa.py --gateway-url http://localhost:8081 --concurrency 24 --duration 60 --cooldown 75
```
