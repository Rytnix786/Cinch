<div align="center">

# ⚡ Cinch

### Enterprise Self-Hosted LLM Serving Platform, Kubernetes Autoscaler & 14-Stage Inference Gateway

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![vLLM](https://img.shields.io/badge/vLLM-0.6.3.post1-8B5CF6?style=for-the-badge&logo=vllm&logoColor=white)](https://github.com/vllm-project/vllm)
[![Quantization](https://img.shields.io/badge/AWQ-W4A16%20Marlin-F59E0B?style=for-the-badge)](https://github.com/casper-hansen/AutoAWQ)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-k3d%20Cluster-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://k3d.io/)
[![Tests](https://img.shields.io/badge/Tests-195%20Passing-10B981?style=for-the-badge&logo=pytest&logoColor=white)](./tests/)
[![License](https://img.shields.io/badge/License-Apache%202.0-0EA5E9?style=for-the-badge)](./LICENSE)

<p align="center">
  <a href="#1-logical-overview">Overview</a> •
  <a href="#2-core-features">Key Capabilities</a> •
  <a href="#3-system-architecture">Architecture</a> •
  <a href="#4-workspace-layout">Repository Layout</a> •
  <a href="#5-getting-started">Getting Started</a> •
  <a href="#6-api-reference">API Reference</a> •
  <a href="#7-empirical-benchmarks">Empirical Benchmarks</a> •
  <a href="#8-contributing--governance">Contributing</a>
</p>

</div>

---

## 1. Logical Overview

Self-hosting Large Language Models in production requires reconciling memory-bound GPU hardware constraints with strict enterprise requirements: predictable sub-second latency, multi-tenant cost attribution, prompt injection defense, and elastic cluster autoscaling. Standard Hugging Face inference pipelines consume excessive VRAM and lack continuous batching, prefix memoization, and fault-tolerant rate governance.

**Cinch** is a production-grade inference serving platform engineered for quantized, high-throughput LLM workloads on consumer and datacenter GPUs. Serving `Qwen2.5-7B-Instruct-AWQ` on an NVIDIA GeForce RTX 3060 Ti (8GB VRAM), Cinch integrates Marlin INT4 mixed-precision GEMM kernels with an asynchronous 14-stage FastAPI gateway. The platform delivers sub-5ms semantic caching, automated Kubernetes Horizontal Pod Autoscaling (HPA), token-budgeted rate limiting, multi-tenant FinOps spend controls, server-side sandboxed tool execution, and an interactive real-time serving console.

---

## 2. Core Features

- **Sub-5ms Semantic Vector Caching**: Cosine similarity memoization ($\ge 0.95$) bypasses GPU compute on semantically equivalent queries, returning responses in **$4.12\text{ ms}$** ($165\times$ speedup) at 0W GPU power consumption.
- **Marlin INT4 Quantized Inference**: Unpacks 4-bit weights into FP16 registers on-the-fly, compressing `Qwen2.5-7B` footprint from $14.4\text{ GiB}$ to **$4.2\text{ GiB}$** ($3.88\times$) and delivering **$331.0\text{ tok/s}$** ($10.86\times$ throughput vs. naive Transformers).
- **Radix Prefix Cache Affinity Routing**: Radix tree prefix matching routes requests sharing identical system prompts to warm KV-cache nodes, reducing Time-To-First-Token (TTFT) by **$4.87\times$** ($0.18\text{ s}$ vs. $0.88\text{ s}$).
- **Guided JSON Grammar Guard**: Outlines-driven EBNF grammar extraction and automated AST-guided repair guarantee **100% schema compliance** for structured JSON completions.
- **Multi-Tenant FinOps Cost Accounting**: Micro-dollar usage ledger ($0.15/1M prompt, $0.60/1M completion tokens) with per-tenant tracking and hard HTTP 402 budget cutoff enforcement.
- **Native Server-Side Tool Sandboxes**: Zero-roundtrip closed-loop tool executor running arithmetic (`calculator`), relational queries (`sql_runner`), and sandboxed code (`python_repl`) in isolated execution environments.
- **Production Shadow Traffic Replayer**: Asynchronous, non-blocking candidate backend replication ($0.0\text{ ms}$ primary serving penalty) with live Jaccard divergence scoring.
- **Interactive Real-Time Serving Console**: Dark-mode operations console at `/ui/` featuring live SSE token streaming, KV-cache heatmaps, security audit streams, and tenant spend management.

---

## 3. System Architecture

```
[ Ingress Traffic / External Clients / Web Console (/ui/) ]
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
│   1. Circuit Breaker Fast-Fail        ──► CLOSED / OPEN / HALF_OPEN (Fast-Fail: 45ms)   │
│   2. Multi-Tenant FinOps Pre-Flight   ──► Hard HTTP 402 budget cutoff & tenant ledger   │
│   3. Ingress Security & PII Redaction ──► DAN injection defense & PII token masking     │
│   4. Context & Prompt Compaction      ──► Lexical entropy filtering (23.1% token cut)   │
│   5. Guided Grammar Guard             ──► 100% deterministic JSON schema validation     │
│   6. Smart Model Cascading            ──► Heuristic complexity routing (0.5B vs 7B)     │
│   7. Multi-LoRA Multiplexer           ──► base:adapter resolution (<1ms virtual switch) │
│   8. Sliding Window Rate Limiter      ──► Dual 60 RPM + 50,000 TPM sliding window       │
│   9. Sub-5ms Semantic Vector Cache    ──► Cosine >= 0.95 similarity, 0W GPU power       │
│  10. Prefix Cache Affinity Router     ──► Radix prefix hashing & cluster affinity       │
│  11. Dual-Tier Priority Queue         ──► VIP preemption (0.96s vs 3.33s batch)         │
│  12. Upstream Inference Engine        ──► vLLM CUDA 12.4 forward pass                   │
│  13. Server-Side Tool Execution Loop  ──► Isolated sandboxes (calc, sql, python_repl)   │
│  14. Shadow Traffic Replayer          ──► Asynchronous candidate replication (0.0ms)    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
┌───────────────────────────────┐   ┌─────────────────────────────────────────────────────┐
│ Kubernetes HPA                │   │ Upstream vLLM Container (Marlin W4A16 GEMM)         │
│ (Autoscales 2 to 6 replicas)  │   │   • Model: Qwen/Qwen2.5-7B-Instruct-AWQ             │
│                               │   │   • Memory: PagedAttention KV Cache (57.3 KB/tok)   │
│                               │   │   • Speculative Decoding: K=5 Draft Verification    │
└───────────────────────────────┘   └─────────────────────────────────────────────────────┘
```

---

## 4. Workspace Layout

```
Cinch/
├── benchmarks/              # Empirical evaluation datasets and telemetry
│   └── results/             # Raw benchmark JSON traces (M1–M26)
├── docker/                  # Multi-stage production container definitions
│   ├── Dockerfile.gateway   # Stateless FastAPI gateway image
│   └── Dockerfile.vllm      # Upstream vLLM engine container
├── docs/                    # Technical milestone dossiers (M1 to M26)
├── gateway/                 # 14-stage asynchronous serving middleware
│   ├── app.py               # Core ASGI application router & lifecycle
│   ├── auth.py              # Bearer and API key authentication provider
│   ├── cache_router.py      # Radix prefix hash table and ring affinity router
│   ├── cascade_router.py    # Heuristic complexity model cascade classifier
│   ├── circuit_breaker.py   # Three-state fault-isolation state machine
│   ├── compressor.py        # Lexical entropy prompt compaction filter
│   ├── config.py            # Pydantic GatewaySettings configuration schema
│   ├── finops.py            # Micro-dollar cost accounting & budget enforcement
│   ├── grammar_guard.py     # EBNF grammar extraction and JSON repair engine
│   ├── guardrails.py        # Ingress prompt injection and PII redaction filter
│   ├── limiter.py           # Dual sliding-window RPM and TPM rate limiter
│   ├── lora_router.py       # Multi-LoRA compound model multiplexer
│   ├── priority_queue.py    # VIP interactive preemption request scheduler
│   ├── semantic_cache.py    # Sub-5ms cosine vector cache memoization
│   ├── shadow_replayer.py   # Asynchronous shadow traffic duplicator
│   ├── telemetry.py         # Prometheus metrics registry & OpenTelemetry spans
│   ├── token_counter.py     # Fast BPE prompt token estimator
│   └── tool_engine.py       # Isolated sandboxed agentic tool executor
├── k8s/                     # Kubernetes manifests and autoscaling rules
│   ├── gateway-configmap.yaml
│   ├── gateway-deployment.yaml
│   ├── gateway-hpa.yaml
│   ├── gateway-ingress.yaml
│   └── gateway-service.yaml
├── scripts/                 # Benchmarking and automated validation harnesses
│   ├── benchmark_harness.py
│   ├── validate_phase3_full.py
│   └── verify_pipeline.py
├── tests/                   # Automated pytest unit and integration test suite
│   ├── test_gateway.py
│   ├── test_phase3_capstone.py
│   └── test_ui_endpoints.py
├── ui/                      # Zero-dependency dark-mode serving console
│   ├── app.js               # SSE stream reader and telemetry dashboard poller
│   ├── index.html           # 5-tab serving console layout
│   └── style.css            # Obsidian design system tokens
└── README.md                # Master platform documentation
```

---

## 5. Getting Started

### Prerequisites

- **Host GPU**: NVIDIA GPU with CUDA 12.4+ and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **Orchestration**: Docker Engine & [k3d](https://k3d.io/) (or local Kubernetes cluster).
- **Client Tools**: `kubectl`, Python 3.12+, and `curl`.

---

### Step 1: Launch Upstream vLLM Engine

Run the quantized `Qwen2.5-7B-Instruct-AWQ` engine on the host with Marlin W4A16 GEMM kernels:

```bash
docker run -d --name cinch-vllm --gpus all -p 8000:8000 --ipc=host vllm/vllm-openai:v0.6.3.post1 --model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq_marlin --gpu-memory-utilization 0.90 --max-model-len 4096
```

---

### Step 2: Deploy Local Kubernetes Cluster & Gateway

Provision the multi-node `k3d` cluster and apply the declarative Kubernetes manifests:

```bash
# 1. Create multi-node k3d cluster with Traefik ingress mapped to port 8081
k3d cluster create cinch-cluster --agents 2 -p "8081:80@loadbalancer"

# 2. Build gateway container image
docker build -t cinch-gateway:latest -f docker/Dockerfile.gateway .

# 3. Import image into k3d runtime nodes
k3d image import cinch-gateway:latest -c cinch-cluster

# 4. Deploy Kubernetes resources
kubectl apply -f k8s/
```

---

### Step 3: Verify Cluster & Gateway Health

```bash
# Check pod rollout status
kubectl get pods -n cinch

# Probe gateway health endpoint
curl http://localhost:8081/health
```

---

### Step 4: Run Automated Verification Suite

Execute the full automated test suite (195 unit and integration tests) and the live Capstone validation harness:

```bash
# Run pytest regression suite
python -m pytest tests/ -v

# Run live Phase 3 Capstone benchmark across all 10 enterprise capabilities
python scripts/validate_phase3_full.py --gateway-url http://localhost:8081 --api-key cinch-prod-key
```

---

## 6. API Reference

All gateway routes require authentication via `Authorization: Bearer <API_KEY>` or `X-API-Key: <API_KEY>`.

| Endpoint | Method | Purpose | Key Headers / Parameters | Status Codes |
|---|---|---|---|---|
| `/v1/chat/completions` | `POST` | OpenAI-compatible chat inference | `model`, `messages`, `max_tokens`, `stream`, `response_format` | `200`, `400`, `402`, `429`, `503` |
| `/v1/models` | `GET` | List available models & LoRA adapters | None | `200` |
| `/v1/tenants/usage` | `GET` | Query multi-tenant FinOps spend ledger | `X-Tenant-ID` (optional) | `200` |
| `/v1/tenants/budget` | `POST` | Update dynamic tenant budget limit | `tenant_id`, `budget_limit_usd` | `200`, `400` |
| `/v1/shadow/metrics` | `GET` | Fetch shadow replication telemetry | None | `200` |
| `/v1/shadow/traces` | `GET` | Inspect live shadow divergence traces | None | `200` |
| `/v1/console/state` | `GET` | Aggregated telemetry state for WebUI | None | `200` |
| `/health` | `GET` | Readiness & subsystem diagnostic probe | None | `200`, `503` |
| `/metrics` | `GET` | Prometheus telemetry exposition | None | `200` |
| `/ui/` | `GET` | Interactive Real-Time Serving Console | Browser Access | `200` |

---

### Example API Invocations

#### Standard Chat Completion
```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Authorization: Bearer cinch-prod-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Explain prefix caching in one sentence."}],
    "max_tokens": 128
  }'
```

#### Real-Time Server-Sent Events (SSE) Streaming
```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Authorization: Bearer cinch-prod-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Write a Python binary search function."}],
    "stream": true
  }'
```

#### Multi-LoRA Dynamic Multiplexing (`base:adapter`)
```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Authorization: Bearer cinch-prod-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ:sql-copilot",
    "messages": [{"role": "user", "content": "SELECT id, name FROM users WHERE active = true;"}],
    "max_tokens": 128
  }'
```

#### Deterministic Guided JSON Grammar Output
```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Authorization: Bearer cinch-prod-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Extract: server prod-1 has 64GB RAM."}],
    "response_format": {
      "type": "json_object",
      "schema": {
        "type": "object",
        "properties": {
          "hostname": {"type": "string"},
          "ram_gb": {"type": "integer"}
        },
        "required": ["hostname", "ram_gb"]
      }
    }
  }'
```

---

## 7. Empirical Benchmarks

All metrics trace directly to JSON datasets stored in [benchmarks/results/](./benchmarks/results/).

| Metric / Dimension | Benchmark Scenario | Baseline (Naive HF / Unmitigated) | Cinch Production Platform | Improvement Factor | Telemetry Source |
|---|---|---|---|---|---|
| **Inference Throughput** | Concurrency $C=16$ | $30.49\text{ tok/s}$ | **$331.00\text{ tok/s}$** | **$10.86\times$ Throughput Speedup** | [`comparison_summary.json`](./benchmarks/results/comparison_summary.json) |
| **P95 Request Latency** | Concurrency $C=16$ | $32.02\text{ s}$ | **$6.29\text{ s}$** | **$5.09\times$ Latency Reduction** | [`comparison_summary.json`](./benchmarks/results/comparison_summary.json) |
| **Quantization Quality** | AST Code, Math, JSON | $100\%$ FP16 Baseline | **$97.2\%$ Equivalence Index** | **$100\%$ Code Syntax Parity** | [`quality_eval.json`](./benchmarks/results/quality_eval.json) |
| **Prefix Cache TTFT** | Shared System Prompt | $0.8856\text{ s}$ (Cold Prefill) | **$0.1818\text{ s}$ (Cache Hit)** | **$4.87\times$ Faster TTFT** | [`prefix_cache_benchmark.json`](./benchmarks/results/prefix_cache_benchmark.json) |
| **Speculative Decoding** | Code & JSON ($K=5$) | $22.0\text{ ms/tok}$ | **$8.6\text{ ms/tok}$ ($\alpha=78.0\%$)** | **$2.58\times$ Generation Speedup** | [`speculative_decoding.json`](./benchmarks/results/speculative_decoding.json) |
| **Semantic Cache Hit** | Duplicate Query | $680.0\text{ ms}$ (GPU Forward) | **$4.12\text{ ms}$ (Vector Lookup)** | **$165\times$ Latency Reduction** | [`semantic_cache_eval.json`](./benchmarks/results/semantic_cache_eval.json) |
| **Prompt Compaction** | Context Windows | $100\%$ Token Volume | **$76.9\%$ Token Volume** | **$23.1\%$ Token Savings** | [`prompt_compaction_eval.json`](./benchmarks/results/prompt_compaction_eval.json) |
| **Circuit Breaker** | Upstream Crash | $30,000\text{ ms}$ (Timeout) | **$45.56\text{ ms}$ (Fast-Fail 503)** | **$658\times$ Faster Fault Isolation** | [`chaos_resilience.json`](./benchmarks/results/chaos_resilience.json) |
| **Self-Healing MTTR** | Worker Recovery | Manual Intervention | **$10.85\text{ s}$ (Canary Probe)** | **Automated Zero-Touch MTTR** | [`chaos_resilience.json`](./benchmarks/results/chaos_resilience.json) |
| **AutoAWQ Compression** | Model Weight Size | $14.40\text{ GiB}$ (FP16) | **$4.20\text{ GiB}$ (W4A16 Marlin)** | **$3.88\times$ Weight Reduction** | [`quantization_summary.json`](./benchmarks/results/quantization_summary.json) |

---

## 8. Contributing & Governance

Contributions are welcome. Follow standard open-source conventions:

1. **Branching**: Create focused feature branches (`feat/feature-name`, `fix/issue-description`, `bench/benchmark-target`).
2. **Conventional Commits**: Format commit messages following the Conventional Commits specification:
   - `feat:` for new capabilities or pipeline stages
   - `fix:` for bug fixes or middleware patches
   - `bench:` for performance benchmark additions
   - `docs:` for documentation updates
   - `test:` for test suite expansion
3. **Testing Standard**: Every serving path modification must include automated tests verifying correctness (`python -m pytest tests/ -v`).
4. **Code Quality**: Run `ruff check gateway/ tests/ scripts/` before submitting pull requests.

```bash
git checkout -b feat/my-new-feature
git commit -m "feat(gateway): add custom response compression middleware"
git push origin feat/my-new-feature
```

---

## 9. License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.
