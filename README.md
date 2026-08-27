# Cinch: Enterprise Self-Hosted LLM Serving Platform

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![vLLM](https://img.shields.io/badge/vLLM-0.6.3.post1-purple.svg)](https://github.com/vllm-project/vllm)
[![Quantization](https://img.shields.io/badge/AWQ-W4A16%20Marlin-orange.svg)](https://github.com/casper-hansen/AutoAWQ)
[![Kubernetes](https://img.shields.io/badge/k3d-Kubernetes%20Cluster-326CE5.svg)](https://k3d.io/)
[![Tests](https://img.shields.io/badge/Tests-195%20Passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Cinch is an enterprise self-hosted Large Language Model (LLM) inference platform. It delivers low-latency, high-throughput quantized model serving on consumer workstations and multi-node Kubernetes clusters. Serving `Qwen2.5-7B-Instruct-AWQ` on an NVIDIA GeForce RTX 3060 Ti (8GB VRAM), Cinch integrates an asynchronous 14-stage middleware gateway with sub-5ms semantic caching, multi-tenant FinOps budget enforcement, prompt compaction, server-side sandboxed tool execution, and an interactive real-time serving console.

---

## 1. Empirical Benchmark Dossier

All metrics trace directly to JSON datasets stored in `benchmarks/results/`.

### Summary Performance Comparison

| Metric / Dimension | Benchmark Scenario | Baseline (Naive HF / Unmitigated) | Cinch Production Platform | Improvement Factor | Telemetry Source |
|---|---|---|---|---|---|
| **Inference Throughput** | Concurrency $C=16$ | $30.49\text{ tok/s}$ | **$331.00\text{ tok/s}$** | **$10.86\times$ Throughput Speedup** | `comparison_summary.json` |
| **P95 Request Latency** | Concurrency $C=16$ | $32.02\text{ s}$ | **$6.29\text{ s}$** | **$5.09\times$ Latency Reduction** | `comparison_summary.json` |
| **Quantization Quality** | AST Code, Math, JSON | $100\%$ FP16 Baseline | **$97.2\%$ Equivalence Index** | **$100\%$ Code Syntax Parity** | `quality_eval.json` |
| **Prefix Cache TTFT** | Shared System Prompt | $0.8856\text{ s}$ (Cold Prefill) | **$0.1818\text{ s}$ (Cache Hit)** | **$4.87\times$ Faster TTFT** | `prefix_cache_benchmark.json` |
| **Speculative Decoding** | Code & JSON ($K=5$) | $22.0\text{ ms/tok}$ | **$8.6\text{ ms/tok}$ ($\alpha=78.0\%$)** | **$2.58\times$ Generation Speedup** | `speculative_decoding.json` |
| **Semantic Cache Hit** | Duplicate Query | $680.0\text{ ms}$ (GPU Forward) | **$4.12\text{ ms}$ (Vector Lookup)** | **$165\times$ Latency Reduction** | `semantic_cache_eval.json` |
| **Prompt Compaction** | Context Windows | $100\%$ Token Volume | **$76.9\%$ Token Volume** | **$23.1\%$ Token Savings** | `prompt_compaction_eval.json` |
| **Circuit Breaker** | Worker Crash Failure | $30,000\text{ ms}$ (Timeout) | **$45.56\text{ ms}$ (Fast-Fail 503)** | **$658\times$ Faster Fault Isolation** | `chaos_resilience.json` |
| **Self-Healing MTTR** | Worker Crash Recovery | Manual Restart | **$10.85\text{ s}$ (Canary Probe)** | **Automated Zero-Touch MTTR** | `chaos_resilience.json` |
| **AutoAWQ Compression** | Model Weight Footprint | $14.40\text{ GiB}$ (FP16) | **$4.20\text{ GiB}$ (W4A16 Marlin)** | **$3.88\times$ Weight Reduction** | `quantization_summary.json` |

### Throughput & Latency Scaling

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

Semantic Vector Cache Latency (milliseconds) — Lower is better
GPU Forward Pass (Miss):       [==================================== 680.00 ms ]
Semantic Hit (0W GPU Power):   [= 4.12 ms ] (-99.4% / 165x)
```

---

## 2. System Architecture

```
[ Ingress Traffic / External Clients / Serving Console (/ui/) ]
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

## 3. Platform Capability Matrix (Milestones 1–26)

### Phase 1: High-Performance Engine Core & Benchmarking

| Milestone | Capability Name | Technical Implementation | Verified SLA |
|---|---|---|---|
| **M1** | Environment & CUDA 12.4 Setup | WSL2 / Docker GPU passthrough on RTX 3060 Ti | CUDA 12.4 runtime validated |
| **M2** | Naive Transformers Baseline | Hugging Face FP16 sequential generation | $30.49\text{ tok/s}$ baseline |
| **M3** | vLLM Engine Deployment | PagedAttention KV memory and continuous batching | $256.0\text{ tok/s}$ ($8.40\times$) |
| **M4** | INT4 AWQ Marlin Quantization | AutoAWQ W4A16 Marlin mixed-precision GEMM | **$331.0\text{ tok/s}$ ($10.86\times$), 4.2 GiB VRAM** |
| **M5** | Structured Evaluation Suite | Automated AST code parsing and math evaluation | **$97.2\%$ quality parity**, 100% syntax |
| **M6** | Locust Load Generation Engine | Concurrent user simulation with Poisson arrivals | 16-user load profile verified |
| **M7** | Speculative Decoding Engine | Draft verification using $K=5$ tokens | **$2.58\times$ generation speedup** ($\alpha=78.0\%$) |
| **M8** | Cost Analysis & Chaos Suite | FinOps dollar estimation and worker crash testing | **MTTR 10.85s**, 45ms fault isolation |

### Phase 2: Kubernetes Infrastructure, Routing & Autoscaling

| Milestone | Capability Name | Technical Implementation | Verified SLA |
|---|---|---|---|
| **M9** | Containerized Gateway Packaging | Multi-stage Docker packaging with health probes | Sub-100ms startup |
| **M10**| Kubernetes Multi-Node Cluster | Multi-node k3d cluster with Traefik ingress | 2 worker nodes + loadbalancer |
| **M11**| Horizontal Pod Autoscaling (HPA) | CPU and concurrency-driven pod scaling | **Scaled 2 to 6 pods** (0% dropped) |
| **M12**| Prefix Cache Affinity Routing | SHA-256 Radix prefix hash table | **$4.87\times$ faster TTFT** ($0.18\text{s}$) |
| **M13**| Tiered Priority Request Queue | VIP interactive preemption over batch tasks | **$3.45\times$ interactive latency edge** |
| **M14**| Token-Budgeted Rate Limiter | Sliding-window 60 RPM and 50,000 TPM controls | Zero KV-cache out-of-memory errors |
| **M15**| Distributed OpenTelemetry Spans | W3C traceparent context propagation | Full distributed trace visibility |

### Phase 3: Enterprise Gateway, FinOps & Developer Ergonomics

| Milestone | Capability Name | Technical Implementation | Verified SLA |
|---|---|---|---|
| **M16**| Sub-5ms Semantic Vector Cache | Cosine similarity memoization ($\ge 0.95$) | **$4.12\text{ ms}$ response** at 0W GPU power |
| **M17**| Multi-LoRA Dynamic Multiplexing| `base:adapter` compound identifier resolution | **$< 1\text{ms}$ virtual adapter switch** |
| **M18**| Guided JSON Grammar Guard | Outlines/EBNF schema extraction and JSON repair | **$100\%$ valid JSON schema output** |
| **M19**| Ingress Security & PII Redaction| Sub-millisecond CPU heuristic injection & PII filter | **$< 1\text{ms}$ scan**, HTTP 400 DAN block |
| **M20**| Smart Complexity Model Cascading| Heuristic complexity classifier ($0.5\text{B}$ vs $7\text{B}$) | **$50\%$ GPU energy savings** on simple tasks |
| **M21**| Context & Prompt Compaction | Lexical entropy token compaction | **$23.1\%$ token volume reduction** |
| **M22**| Server-Side Agentic Tool Sandboxes| In-process isolated `calculator`, `sql`, `python_repl` | **$100\%$ tool execution accuracy** |
| **M23**| Multi-Tenant FinOps Cost Accounting| Micro-dollar token accounting and budget enforcement | **Real-time spend ledger**, HTTP 402 cutoff |
| **M24**| Production Shadow Traffic Replayer| Asynchronous candidate backend replication | **$0.0\text{ ms}$ primary latency impact** |
| **M25**| Interactive Real-Time Serving Console| Dark-mode WebUI with SSE streaming client | **$3.64\text{ ms}$ asset delivery**, 50.5 tok/s |
| **M26**| Phase 3 Capstone Integration | Full 26-milestone automated regression suite | **$100\%$ Conformance (10/10)**, 195 tests pass |

---

## 4. Interactive Real-Time Serving Console

Access the dark-mode serving console at `http://localhost:8081/ui/` or `http://localhost:8081/console`.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ ⚡ CINCH Serving Console               [Playground] [Cache] [FinOps] [Security] [Shadow] │
├──────────────────────────────────────┬─────────────────────────────────────────────────┤
│ INFERENCE CONTROLS                   │ TELEMETRY: TTFT: 159ms | 50.5 tok/s | Cost: $0  │
│ Target: Qwen2.5-7B-Instruct-AWQ     │ ─────────────────────────────────────────────── │
│ Max Tokens: [====|===] 128           │ STREAM OUTPUT                                   │
│ Temperature: [==|====] 0.7           │ In vLLM, Radix prefix caching organizes KV      │
│ [x] SSE Stream  [x] Compaction       │ tensors into a tree structure, enabling shared  │
│ [x] Guardrails  [x] Tool Engine      │ system prompt reuse across concurrent requests. │
│ [▶ Dispatch Inference]              │                                                 │
└──────────────────────────────────────┴─────────────────────────────────────────────────┘
```

Features:
1. **Interactive Prompt Playground**: Live SSE token streaming, TTFT and TPS calculation, parameter sliders, model selector, grammar schema editor, and server-side tool execution.
2. **KV-Cache Memory Heatmap**: 64-block visual matrix of shared prefix and active context allocations, prefix hit ratios, and LRU cache statistics.
3. **FinOps Cost Center**: Live micro-dollar platform spend meters, team budget progress bars, dynamic budget limit configuration, and tenant usage ledgers.
4. **Security & Guardrails Audit Stream**: Real-time event log of verified clean queries, blocked prompt injections, and redacted PII entities.
5. **Shadow Replayer & Tools Inspector**: Production vs candidate model comparison diff viewer with latency deltas and sandboxed tool execution results.

---

## 5. API Quickstart & Examples

### 1. Standard Chat Completion
```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Authorization: Bearer cinch-prod-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Explain Kubernetes scheduling."}],
    "max_tokens": 60
  }'
```

### 2. Live SSE Token Streaming
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

### 3. Multi-LoRA Dynamic Multiplexing
```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Authorization: Bearer cinch-prod-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ:sql-copilot",
    "messages": [{"role": "user", "content": "SELECT id, name FROM users WHERE active = true;"}],
    "max_tokens": 40
  }'
```

### 4. Guided JSON Grammar Schema Enforcement
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

### 5. Native Server-Side Agentic Tool Execution
```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Authorization: Bearer cinch-prod-key" \
  -H "Content-Type: application/json" \
  -H "X-Server-Tool-Execution: true" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Calculate the compound interest on $10,000 at 5% for 3 years."}],
    "max_tokens": 100
  }'
```

### 6. Multi-Tenant FinOps Cost Accounting & Budget Query
```bash
# Query tenant usage ledger
curl -X GET http://localhost:8081/v1/tenants/usage \
  -H "Authorization: Bearer cinch-prod-key"

# Adjust tenant budget limit
curl -X POST http://localhost:8081/v1/tenants/budget \
  -H "Authorization: Bearer cinch-prod-key" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "data-science", "budget_limit_usd": 250.0}'
```

### 7. Asynchronous Production Shadow Replay Mirroring
```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Authorization: Bearer cinch-prod-key" \
  -H "Content-Type: application/json" \
  -H "X-Shadow-Replay: true" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Summarize microservices resilience."}],
    "max_tokens": 50
  }'
```

---

## 6. Production Deployment & Setup

### 1. Prerequisites
- NVIDIA GPU with CUDA 12.4+ and Container Toolkit
- Docker Engine & `k3d`
- Kubernetes CLI (`kubectl`)
- Python 3.12+

### 2. Start Host vLLM Engine
```powershell
docker run -d --name cinch-vllm --gpus all `
  -p 8000:8000 `
  --ipc=host `
  vllm/vllm-openai:v0.6.3.post1 `
  --model Qwen/Qwen2.5-7B-Instruct-AWQ `
  --quantization awq_marlin `
  --gpu-memory-utilization 0.90 `
  --max-model-len 4096
```

### 3. Deploy Kubernetes Multi-Node Cluster
```powershell
# Create multi-node k3d cluster with Traefik ingress mapped to port 8081
k3d cluster create cinch-cluster --agents 2 -p "8081:80@loadbalancer"

# Build gateway container image
docker build -t cinch-gateway:latest -f docker/Dockerfile.gateway .

# Import image into k3d nodes
k3d image import cinch-gateway:latest -c cinch-cluster

# Deploy Kubernetes manifests
kubectl apply -f k8s/
```

### 4. Verify Gateway Health
```powershell
curl http://localhost:8081/health
```

---

## 7. Configuration Reference

The gateway supports environment variables via Pydantic `GatewaySettings` (`gateway/config.py`):

| Variable Name | Default Value | Description |
|---|---|---|
| `VLLM_BASE_URL` | `http://host.k3d.internal:8000` | Upstream vLLM OpenAI-compatible endpoint |
| `API_KEYS` | `["cinch-prod-key"]` | Authorized bearer API keys |
| `RATE_LIMIT_RPM` | `60` | Maximum requests per minute per client |
| `RATE_LIMIT_TPM` | `50000` | Maximum tokens per minute per client |
| `CIRCUIT_BREAKER_ENABLED` | `True` | Fast-fail protection on upstream failure |
| `CIRCUIT_FAILURE_THRESHOLD` | `5` | Consecutive failures before opening circuit |
| `CIRCUIT_RECOVERY_TIMEOUT_SECONDS` | `10.0` | Half-open probe delay |
| `SEMANTIC_CACHE_CAPACITY` | `5000` | Vector cache size |
| `SEMANTIC_CACHE_SIMILARITY_THRESHOLD`| `0.95` | Cosine similarity threshold for zero-GPU hit |
| `LORA_ROUTING_ENABLED` | `True` | Multi-LoRA compound model resolution |
| `GRAMMAR_GUARD_ENABLED` | `True` | Deterministic structured JSON enforcement |
| `GUARDRAILS_ENABLED` | `True` | Heuristic prompt injection & PII masking |
| `CASCADE_ROUTING_ENABLED` | `True` | Small vs Large model complexity routing |
| `COMPRESSOR_ENABLED` | `True` | Lexical entropy context compaction |
| `TOOL_ENGINE_ENABLED` | `True` | Server-side agentic sandbox execution |
| `FINOPS_ENABLED` | `True` | Micro-dollar multi-tenant cost tracking |
| `FINOPS_ENFORCE_BUDGETS` | `True` | Hard HTTP 402 cutoff on budget breach |
| `SHADOW_REPLAYER_ENABLED` | `True` | Asynchronous candidate replication |
| `SHADOW_SAMPLE_RATE` | `0.10` | Sampling probability for shadow duplication |

---

## 8. Automated Testing & Verification

Cinch enforces automated regression testing across all serving paths:

```powershell
# Run full unit and integration test suite
python -m pytest tests/ -v

# Run code linter
python -m ruff check gateway/ tests/ scripts/

# Run live Capstone verification harness
python scripts/validate_phase3_full.py --gateway-url http://localhost:8081 --api-key cinch-prod-key
```

```
====================== 195 passed in 4.90s ======================
All checks passed!
```

---

## 9. Repository Layout

```
Cinch/
├── benchmarks/              # Empirical evaluation datasets & charts
│   └── results/             # Raw benchmark JSON traces (M1–M26)
├── docker/                  # Multi-stage Dockerfiles (Gateway & vLLM)
├── docs/                    # Technical milestone dossiers (milestone-01 to 26)
├── gateway/                 # 14-stage FastAPI serving gateway
│   ├── app.py               # Core ASGI application & middleware router
│   ├── auth.py              # API key authentication provider
│   ├── cache_router.py      # Radix prefix cache hash & ring router
│   ├── cascade_router.py    # Small vs Large complexity classifier
│   ├── circuit_breaker.py   # Three-state fault-isolation FSM
│   ├── compressor.py        # Lexical entropy prompt compaction
│   ├── config.py            # Pydantic configuration schemas
│   ├── finops.py            # Micro-dollar cost accounting & budget cutoff
│   ├── grammar_guard.py     # EBNF grammar extraction & JSON repair
│   ├── guardrails.py        # Ingress injection scanner & PII anonymizer
│   ├── limiter.py           # Dual sliding-window RPM/TPM limiter
│   ├── lora_router.py       # Multi-LoRA compound model multiplexer
│   ├── priority_queue.py    # VIP interactive preemption scheduler
│   ├── semantic_cache.py    # Cosine vector cache memoization
│   ├── shadow_replayer.py   # Asynchronous candidate traffic duplicator
│   ├── telemetry.py         # Prometheus registry & OpenTelemetry spans
│   ├── token_counter.py     # Fast BPE token estimation
│   └── tool_engine.py       # Sandboxed closed-loop tool executor
├── k8s/                     # Kubernetes production manifests & HPA
├── scripts/                 # Live benchmark and validation harnesses
├── tests/                   # Automated pytest unit & integration suite (195 tests)
├── ui/                      # Interactive dark-mode WebUI serving console
│   ├── app.js               # SSE stream reader & telemetry poller
│   ├── index.html           # 5-tab dashboard shell
│   └── style.css            # Obsidian design tokens
└── README.md                # Master platform documentation
```

---

## 10. License

Apache License 2.0. See [LICENSE](LICENSE) for details.
