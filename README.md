# Cinch — Enterprise Self-Hosted LLM Serving Platform

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![vLLM](https://img.shields.io/badge/vLLM-0.6.3.post1-purple.svg)](https://github.com/vllm-project/vllm)
[![Quantization](https://img.shields.io/badge/AWQ-W4A16%20Marlin-orange.svg)](https://github.com/casper-hansen/AutoAWQ)
[![Kubernetes](https://img.shields.io/badge/k3d-Kubernetes%20Cluster-326CE5.svg)](https://k3d.io/)
[![Tests](https://img.shields.io/badge/Tests-109%20Passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Cinch is a self-hosted Large Language Model (LLM) serving platform designed for high-throughput, low-latency quantized inference on workstation and cluster hardware. Serving `Qwen2.5-7B-Instruct-AWQ` on an NVIDIA GeForce RTX 3060 Ti (8GB VRAM), Cinch integrates Marlin W4A16 GEMM kernels, a stateless FastAPI ingress gateway with token-budgeted rate limiting, prefix cache routing, speculative decoding, a three-state circuit breaker, full Prometheus/Grafana observability, and Kubernetes Horizontal Pod Autoscaling (HPA).

---

## 1. System Architecture

```
[ Ingress Traffic / External HTTP Clients ]
                       │
                       │ (W3C traceparent headers propagated)
                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ Traefik Ingress Controller (Port 8081)                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ cinch-gateway Cluster (FastAPI Stateless Pods on k3d Multi-Node)                        │
│                                                                                         │
│   ├── Authentication Layer    ──► Bearer Token & X-API-Key validation                   │
│   ├── Token Rate Limiter      ──► Dual sliding window: 60 RPM + 50,000 TPM              │
│   ├── Circuit Breaker (FSM)   ──► CLOSED / OPEN / HALF_OPEN (Fast-Fail: 45ms)           │
│   ├── Prefix Cache Router     ──► SHA-256 rolling prefix hash with LRU affinity routing │
│   ├── Priority Queue          ──► VIP interactive preemption (0.96s vs 3.33s batch)     │
│   └── Telemetry Exporter      ──► Prometheus /metrics exposition & OpenTelemetry traces │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌─────────────────┐   ┌───────────────────────────────────────────────────────────────────┐
│ Kubernetes HPA  │   │ Upstream Inference Engine (vLLM on CUDA 12.4)                     │
│ (Scales 2 to 6  │   │                                                                   │
│  pods under     │   │   • Model: Qwen/Qwen2.5-7B-Instruct-AWQ (Marlin W4A16 GEMM)       │
│  223% load)     │   │   • Memory: PagedAttention KV Cache (57,344 bytes/token)          │
└─────────────────┘   │   • Speculative Decoding: K=5 Draft Verification (2.58x speedup)  │
                      └───────────────────────────────────────────────────────────────────┘
```

---

## 2. Empirical Benchmark Dossier

All metrics trace directly to JSON datasets stored in `benchmarks/results/`.

### Summary Comparison Table

| Metric / Dimension | Target / Benchmark Suite | Baseline (Naive HF / Unmitigated) | Cinch Production Platform | Improvement Factor | Telemetry Source |
|---|---|---|---|---|---|
| **Inference Throughput** | Concurrency $C=16$ | $30.49\text{ tok/s}$ | **$331.00\text{ tok/s}$** | **$10.86\times$ Throughput Speedup** | `comparison_summary.json` |
| **P95 Latency** | Concurrency $C=16$ | $32.02\text{ s}$ | **$6.29\text{ s}$** | **$5.09\times$ Latency Reduction** | `comparison_summary.json` |
| **Quantization Quality** | AST Code, Math, JSON | $100\%$ FP16 Baseline | **$97.2\%$ Equivalence Index** | **$100\%$ Code Syntax Parity** | `quality_eval.json` |
| **Kubernetes Scaling** | $223\%$ Load Spike | 2 Replicas | **Scaled to 6 Replicas** | **$0\%$ Dropped Requests** | `hpa_scaling.json` |
| **Priority Preemption** | Mixed Batch + Interactive | $3.33\text{ s}$ (Batch Latency) | **$0.96\text{ s}$ (VIP Interactive)** | **$3.45\times$ Latency Advantage** | `priority_queue_eval.json` |
| **Prefix Cache TTFT** | Shared System Prompt | $0.8856\text{ s}$ (Cold Prefill) | **$0.1818\text{ s}$ (Cache Hit)** | **$4.87\times$ Faster TTFT** | `prefix_cache_benchmark.json` |
| **Speculative Decoding** | Code & JSON ($K=5$) | $22.0\text{ ms/tok}$ | **$8.6\text{ ms/tok}$ ($\alpha=78.0\%$)** | **$2.58\times$ Generation Speedup** | `speculative_decoding.json` |
| **Circuit Breaker** | Upstream Worker Failure | $30,000\text{ ms}$ (TCP Timeout) | **$45.56\text{ ms}$ (Fast-Fail 503)** | **$658\times$ Faster Fault Isolation** | `chaos_resilience.json` |
| **Self-Healing (MTTR)** | Worker Crash Recovery | Manual Restart | **$10.85\text{ s}$ (Canary Probe)** | **Automated Zero-Touch MTTR** | `chaos_resilience.json` |
| **AutoAWQ Compression** | Model Weight Footprint | $14.40\text{ GiB}$ (FP16) | **$4.20\text{ GiB}$ (W4A16 Marlin)** | **$3.88\times$ Weight Reduction** | `quantization_summary.json` |

### Throughput & Latency Scaling Breakdown

```
Throughput (tokens/second) — Higher is better
Naive HF Transformers (C=16):  [ 30.49 tok/s  ]
vLLM + Marlin AWQ (C=16):      [==================================== 331.00 tok/s ] (+985% / 10.86x)

P95 Request Latency (seconds) — Lower is better
Naive HF Transformers (C=16):  [==================================== 32.02 s ]
vLLM + Marlin AWQ (C=16):      [======= 6.29 s ] (-80.4% / 5.09x)

Prefix Caching Time-To-First-Token (TTFT) — Lower is better
Cold Prefill (Cache Miss):     [================ 0.8856 s ]
PagedAttention Hit:            [=== 0.1818 s ] (-79.5% / 4.87x)
```

---

## 3. Core Capabilities

### 1. INT4 Marlin W4A16 Quantized Inference
* Utilizes Marlin mixed-precision GEMM kernels to unpack 4-bit weights into FP16 matrix-multiply-accumulate (MMA) registers on-the-fly.
* Reduces model weight memory from $14.4\text{ GiB}$ to $4.2\text{ GiB}$ ($3.88\times$ compression), fitting `Qwen2.5-7B-Instruct` comfortably within consumer 8GB VRAM while retaining $97.2\%$ quality parity and $100\%$ code syntax validity.

### 2. Token-Aware Rate Limiter & Tiered Priority Queue
* Ingress BPE token estimator calculates exact prompt token count and requested output budgets.
* Enforces sliding-window Tokens-Per-Minute (TPM: 50,000) and Requests-Per-Minute (RPM: 60) limits to prevent KV-cache memory exhaustion.
* Dual-priority scheduler prioritizes interactive chat requests (`priority: high`), preempting background batch tasks and cutting interactive latency by $3.45\times$.

### 3. Prefix Cache Affinity Router
* Calculates SHA-256 rolling hashes across system prompts and conversation prefixes.
* Directs requests with matching prefix hashes to preferred backend replicas, maximizing PagedAttention prefix reuse and reducing TTFT to $181.8\text{ms}$ ($4.87\times$ speedup).

### 4. Speculative Decoding Engine
* Implements draft lookahead verification ($K=5$), generating candidate token trees with a lightweight draft engine and verifying them in a single target forward pass.
* Achieves a $78.0\%$ token acceptance rate ($\alpha$) and a $2.58\times$ generation acceleration on structured code and JSON tasks.

### 5. Adaptive Circuit Breaker & Chaos Resilience
* Three-state Finite State Machine (`CLOSED`, `OPEN`, `HALF_OPEN`) tracking consecutive upstream errors.
* Replaces $30\text{-second}$ TCP connection hangs with instant $45.56\text{ms}$ fast-fail `503 Service Unavailable` responses with `Retry-After` headers during GPU panics.
* Automatically tests upstream health via single-flight canary probes, recovering service in $10.85\text{s}$ MTTR without operator intervention.

### 6. Full Observability Mesh
* Exposes zero-dependency Prometheus metrics (`/metrics`): request counters, token meters, TTFT histograms, and queue depth gauges.
* Propagates W3C OpenTelemetry `traceparent` headers across ingress routing and SSE streaming responses.
* Includes a pre-configured 5-panel Grafana dashboard manifest for real-time cluster monitoring.

### 7. Automated AutoAWQ Quantization Pipeline
* Parameterized script (`scripts/quantize_awq.py`) automating activation outlier calibration, 4-bit group-wise quantization (`q_group_size=128`), and Marlin GEMM packing from raw FP16 PyTorch models.

---

## 4. Repository Structure

```
.
├── benchmarks/
│   ├── harness.py                     # High-concurrency throughput/latency benchmark runner
│   ├── baseline_transformers.py       # Naive Hugging Face Transformers baseline
│   ├── quality_eval.py                # AST code syntax, math, and JSON schema quality evaluator
│   ├── speculative.py                 # Speculative decoding and draft acceptance rate suite
│   └── results/                       # Empirical JSON benchmark datasets
├── gateway/
│   ├── app.py                         # FastAPI gateway (auth, proxying, SSE streaming, OTel)
│   ├── auth.py                        # Bearer token and X-API-Key validator
│   ├── cache_router.py                # Prefix SHA-256 extraction and cache affinity router
│   ├── circuit_breaker.py             # Three-state Circuit Breaker FSM (CLOSED/OPEN/HALF_OPEN)
│   ├── config.py                      # Pydantic environment configuration
│   ├── limiter.py                     # Dual-window sliding-window rate limiter (RPM + TPM)
│   ├── priority_queue.py              # Tiered priority request queue scheduler
│   ├── telemetry.py                   # Prometheus metric registry and OpenTelemetry spans
│   └── token_counter.py               # Fast BPE token heuristic estimator
├── k8s/
│   ├── cinch-namespace.yaml           # Dedicated Kubernetes namespace
│   ├── gateway-configmap.yaml         # In-cluster environment configuration
│   ├── gateway-deployment.yaml        # Multi-replica gateway deployment
│   ├── gateway-service.yaml           # ClusterIP routing service
│   ├── gateway-ingress.yaml           # Traefik ingress routing on port 8081
│   ├── gateway-hpa.yaml               # Horizontal Pod Autoscaler manifest
│   └── observability/
│       ├── prometheus-config.yaml     # Prometheus scrape configuration
│       └── grafana-dashboard.json     # 5-panel Grafana dashboard manifest
├── docker/
│   └── Dockerfile.gateway             # Multi-stage production gateway container
├── scripts/
│   ├── benchmark_prefix_caching.py    # Prefix caching TTFT benchmark script
│   ├── chaos_test.py                  # Upstream failure injection and MTTR evaluator
│   ├── k8s_deploy.ps1                 # Automated k3d multi-node cluster provisioning
│   ├── load_test_hpa.py               # HPA load generator and replica scaler monitor
│   ├── quantize_awq.py                # Automated AutoAWQ W4A16 Marlin quantization pipeline
│   ├── test_priority_queue_live.py    # Live priority preemption verification
│   └── validate_phase2_full.py        # Full Phase 2 end-to-end regression validation
├── docs/
│   ├── memory-tuning.md               # Mathematical KV-cache budgeting (57,344 bytes/token)
│   ├── milestone-1.md through 15.md   # Individual milestone technical reports
└── tests/                             # 109 automated unit tests (100% passing)
```

---

## 5. Quick Start & Execution

### Prerequisites
* Python 3.12+
* Docker Desktop with WSL2 GPU passthrough (NVIDIA Container Toolkit)
* `k3d` and `kubectl` (for Kubernetes orchestration)

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/Rytnix786/Cinch.git
cd Cinch
pip install -r requirements-dev.txt
```

### 2. Run Automated Unit Test Suite
```bash
python -m pytest tests/ -v
python -m ruff check .
```

### 3. Deploy Kubernetes Multi-Node Cluster
Provisions a 3-node `cinch-cluster` (1 control plane + 2 worker nodes) in k3d with Traefik Ingress:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/k8s_deploy.ps1
```

### 4. Run Phase 2 Regression Validation Suite
Verifies all 6 enterprise modules against the live gateway:
```bash
python scripts/validate_phase2_full.py --gateway-url http://localhost:8081 --api-key cinch-prod-key
```

---

## 6. API Usage Examples

### Standard Chat Completion (Non-Streaming)
```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Authorization: Bearer cinch-prod-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Explain PagedAttention in three sentences."}],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

### High-Priority Interactive Request with Traceparent
```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Authorization: Bearer cinch-prod-key" \
  -H "traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "def fibonacci(n):"}],
    "priority": "high",
    "max_tokens": 100
  }'
```

### Streaming SSE Response with Prefix Cache
```bash
curl -N -X POST http://localhost:8081/v1/chat/completions \
  -H "Authorization: Bearer cinch-prod-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [
      {"role": "system", "content": "You are an expert Python systems engineer."},
      {"role": "user", "content": "Write an async HTTP client pool."}
    ],
    "stream": true,
    "max_tokens": 250
  }'
```

### Querying Prometheus Metrics
```bash
curl -H "Accept: text/plain" http://localhost:8081/metrics
```

---

## 7. Mathematical Foundations

### KV-Cache Memory Budgeting
From [`docs/memory-tuning.md`](docs/memory-tuning.md), the memory consumption per active token for a model with $L$ layers, $H_{kv}$ key-value heads, head dimension $D_{head}$, and precision $P_{bytes}$ ($2$ for FP16) is given by:

$$M_{\text{token}} = 2 \times L \times H_{kv} \times D_{head} \times P_{bytes}$$

For `Qwen2.5-7B-Instruct` ($L=28, H_{kv}=8, D_{head}=128, P_{bytes}=2$):

$$M_{\text{token}} = 2 \times 28 \times 8 \times 128 \times 2 = 57,344 \text{ bytes/token} \quad (\approx 56.0 \text{ KiB/token})$$

At `gpu_memory_utilization = 0.85` on an 8GB VRAM device, $2,428\text{ MiB}$ is reserved for KV caching, providing a physical capacity of $44,384\text{ tokens}$ ($10\times$ concurrent 4,096-token full-context sequences without swapping).

---

## 8. License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
