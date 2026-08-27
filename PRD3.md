# PRD Phase 3: Autonomous Scale, Semantic Acceleration & Enterprise Intelligence

## 1. Executive Summary & Objective

Phase 1 established the baseline LLM serving infrastructure (vLLM, AWQ Marlin INT4, Kubernetes multi-node orchestration, HPA, and benchmark suites). Phase 2 upgraded the platform to enterprise reliability (token-aware rate limiting, priority queue preemption, prefix cache affinity routing, speculative decoding, Prometheus/Grafana/OTel observability, circuit breaking, and automated AutoAWQ quantization).

**Phase 3 Objective**: Transform Cinch into an autonomous, multi-tenant, ultra-low-latency intelligent serving ecosystem. Phase 3 introduces sub-5ms semantic vector caching, dynamic multi-LoRA adapter multiplexing, strict EBNF grammar constraints, bidirectional security guardrails (PII redaction and prompt injection defense), smart model cascading, context compaction, server-side agentic tool execution, multi-tenant FinOps cost metering, continuous production shadow traffic replay, and an interactive real-time serving console.

---

## 2. Hardware Environment & Realism Boundaries

- **Workstation Hardware**: 1x NVIDIA GeForce RTX 3060 Ti (8,192 MiB VRAM), AMD Ryzen 5 CPU, 16GB RAM.
- **Serving Precision**: AWQ Marlin INT4 Weights / FP16 Activations (W4A16).
- **Cluster Orchestration**: Multi-node k3d Kubernetes cluster (`cinch-cluster`: 1 control plane + 2 worker nodes) connected to the host GPU daemon via `host.k3d.internal`.
- **Honest Scope Boundary**:
  - Semantic caching, guardrails, grammar guards, model cascading, context compression, tool engines, and FinOps metering execute live on the Kubernetes gateway layer with zero GPU memory overhead.
  - Multi-LoRA multiplexing and speculative draft generation run directly against GPU kernels.
  - Multi-node GPU disaggregated cluster behaviors remain documented as the production horizontal extension path.

---

## 3. Phase 3 Architecture Evolution

```
[ External Traffic / Agentic Microservices / Web UI ]
                          │
                          │ (W3C traceparent headers + X-Tenant-ID propagated)
                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ Traefik Ingress Controller (Port 8081)                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ cinch-gateway Cluster (Stateless Multi-Pod Mesh on k3d)                                 │
│                                                                                         │
│  [ Tier 1: Security & Guardrails ]                                                      │
│    ├── Ingress Prompt Injection & Jailbreak Defense (<1ms scan)                          │
│    └── Bidirectional PII Redaction ([REDACTED_SSN], [REDACTED_KEY])                     │
│                                                                                         │
│  [ Tier 2: Zero-Compute Fast Paths ]                                                    │
│    ├── Sub-5ms Semantic Vector Cache (ONNX Cosine Similarity >= 0.95 ──► 0W GPU)        │
│    └── Exact Prefix Cache Affinity Router (SHA-256 rolling Radix tree)                  │
│                                                                                         │
│  [ Tier 3: Intelligent Flow & Complexity Routing ]                                      │
│    ├── Smart Model Cascade Router (Simple ──► 0.5B Engine | Complex ──► 7B AWQ)        │
│    ├── Context & Prompt Compactor (LLMLingua entropy compaction: 40-50% KV reduction)  │
│    ├── Dynamic Multi-LoRA Adapter Resolver (Base + Adapter Multiplexing)                │
│    └── Tiered Priority Queue & Dual Token Limiter (RPM + TPM)                           │
│                                                                                         │
│  [ Tier 4: Output Guardrails & Agentic Execution ]                                      │
│    ├── Strict EBNF / Regex Grammar Guard (100% Valid JSON Logit Masking)                │
│    ├── Native Server-Side Tool & Function Executor (Closed-loop Python/SQL sandbox)     │
│    └── Output System Prompt Non-Leakage Filter                                          │
│                                                                                         │
│  [ Tier 5: Observability & FinOps ]                                                     │
│    ├── Per-Tenant Dollar Cost Metering ($/1M tokens, budget ceilings)                   │
│    ├── Prometheus /metrics & OpenTelemetry distributed spans                            │
│    └── Production Shadow Traffic Replayer (Async 5% traffic split)                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         │ (Cluster DNS / Host GPU Daemon) │
         ▼                                 ▼
┌───────────────────────────────────┐   ┌─────────────────────────────────────────────────┐
│ Model Tier A: Fast Draft / Simple │   │ Model Tier B: Primary Reasoning Engine          │
│ Qwen2.5-0.5B-Instruct             │   │ Qwen2.5-7B-Instruct-AWQ (Marlin W4A16 GEMM)     │
│ • Sub-40ms latency                │   │ • Multi-LoRA Adapters (SQL, Code, Medical)      │
│ • 10x throughput for basic lookups│   │ • Speculative Decoding (K=5 lookahead, a=78%)   │
└───────────────────────────────────┘   └─────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ Interactive Real-Time Serving Console (Next.js / Vanilla JS Web UI)                     │
│ • Live Token Streaming & Prompt Playground                                              │
│ • Real-Time KV-Cache Heatmaps & GPU Saturation Graphs                                   │
│ • Tenant Billing, Cost Attribution & Latency Waterfall Traces                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 3 Technical Modules & Milestones

### Milestone 16: Sub-5ms Semantic Vector Caching (`gateway/semantic_cache.py`)
- **Problem**: Exact Prefix Hashing (M11) requires exact byte-for-byte matches. In real enterprise traffic, $30\text{--}40\%$ of queries are paraphrases (*"How to connect to Postgres in Python"* vs *"Python script connecting to PostgreSQL"*), triggering redundant GPU compute.
- **Implementation**:
  - Embedded local CPU ONNX vectorizer (`all-MiniLM-L6-v2`, $<2\text{ms}$ latency, $<50\text{MB}$ RAM) or fast LSH hashing.
  - Cosine similarity matching against an in-memory vector store with configurable threshold ($\ge 0.95$).
  - Returns cached completions in **$<5\text{ms}$ at $0\text{W}$ GPU power**, cutting token compute by $25\text{--}35\%$.
- **Verification**: `tests/test_semantic_cache.py` and empirical benchmark `benchmarks/results/semantic_cache_eval.json`.
- **Documentation**: `docs/milestone-16.md`.

---

### Milestone 17: Multi-LoRA Dynamic Adapter Multiplexing (`gateway/lora_router.py`)
- **Problem**: Deploying separate 7B base models for specialized tasks (SQL, Python, Medical, Legal) requires $16\text{--}25\text{ GB}$ VRAM, exceeding single-GPU workstation capacities.
- **Implementation**:
  - Compound model identifier parser: `base_model:adapter_name` (e.g. `Qwen2.5-7B:sql-coder`).
  - LoRA registry resolving fine-tuned weight adapters without duplicating base model weights in VRAM.
  - Exposes dynamic LoRA adapter discovery in `/v1/models`.
- **Verification**: `tests/test_lora_router.py` and empirical routing benchmark `benchmarks/results/lora_routing_eval.json`.
- **Documentation**: `docs/milestone-17.md`.

---

### Milestone 18: Guided Structured Output & JSON Grammar Enforcement (`gateway/grammar_guard.py`)
- **Problem**: AI agents and microservices require strict JSON schema conformance. LLMs occasionally emit conversational filler or invalid JSON syntax, causing parser crashes and expensive retry loops.
- **Implementation**:
  - Pre-compiles JSON Schemas and regex definitions into Fast Deterministic Finite Automata (DFA).
  - Outlines / EBNF logit constraint interceptor masking invalid tokens during generation.
  - Guarantees **$100\%$ schema compliance** and eliminates retry token waste.
- **Verification**: `tests/test_grammar_guard.py` and benchmark `benchmarks/results/grammar_guard_eval.json`.
- **Documentation**: `docs/milestone-18.md`.

---

### Milestone 19: Ingress Security, Jailbreak Defense & PII Redaction (`gateway/guardrails.py`)
- **Problem**: Production endpoints are vulnerable to prompt injection, jailbreaks, and sensitive data leakage (SSNs, credit card numbers, corporate API tokens).
- **Implementation**:
  - Lightweight vector distance and regex scanning detecting adversarial injections in $<1\text{ms}$.
  - Bidirectional PII anonymizer automatically masking sensitive tokens (`[REDACTED_SSN]`, `[REDACTED_API_KEY]`) prior to GPU forwarding.
  - System prompt non-leakage filter verifying generated token outputs.
- **Verification**: `tests/test_guardrails.py` and security audit `benchmarks/results/guardrails_eval.json`.
- **Documentation**: `docs/milestone-19.md`.

---

### Milestone 20: Smart Model Cascading & Complexity Routing (`gateway/cascade_router.py`)
- **Problem**: $60\text{--}70\%$ of enterprise queries are simple tasks (greetings, text classification, single-sentence extraction) that waste compute when processed by a full 7B parameter model.
- **Implementation**:
  - Fast complexity classifier evaluating query difficulty via token heuristics and semantic intent.
  - Routes simple queries to `Qwen2.5-0.5B-Instruct` ($35\text{ms}$ latency, $10\times$ higher throughput).
  - Routes complex reasoning (code, multi-step math) to `Qwen2.5-7B-Instruct-AWQ`.
  - Cuts overall GPU energy and VRAM consumption by up to **$50\%$**.
- **Verification**: `tests/test_cascade_router.py` and load sweep `benchmarks/results/cascade_routing_eval.json`.
- **Documentation**: `docs/milestone-20.md`.

---

### Milestone 21: Context & Prompt Compaction (`gateway/compressor.py`)
- **Problem**: In RAG pipelines and long conversations, prompts reach 4,000–8,000 tokens, where $80\%$ is syntactic filler that consumes linear KV-cache memory without adding semantic value.
- **Implementation**:
  - Information entropy prompt compactor (LLMLingua heuristic) stripping low-entropy and redundant linguistic tokens.
  - Compresses prompt lengths by **$30\text{--}50\%$** with zero loss in generation accuracy, doubling effective KV-cache concurrency capacity.
- **Verification**: `tests/test_compressor.py` and context benchmark `benchmarks/results/prompt_compaction_eval.json`.
- **Documentation**: `docs/milestone-21.md`.

---

### Milestone 22: Native Server-Side Agentic Tool Execution Engine (`gateway/tool_engine.py`)
- **Problem**: Small quantized models require client-side retry loops when orchestrating multi-step tool calls.
- **Implementation**:
  - Ingests standard OpenAI `tools` schema definitions.
  - Real-time function argument parser and validator.
  - Server-side tool execution sandbox (Python REPL, SQL runner, Calculator) executing tools and re-invoking inference in a closed loop before returning final responses.
- **Verification**: `tests/test_tool_engine.py` and agent evaluation `benchmarks/results/tool_execution_eval.json`.
- **Documentation**: `docs/milestone-22.md`.

---

### Milestone 23: Multi-Tenant FinOps Cost Metering (`gateway/finops.py`)
- **Problem**: Enterprise platform engineers lack real-time cost attribution, team-level budget enforcement, and dollar-denominated Prometheus metrics.
- **Implementation**:
  - Ingests `X-Tenant-ID` / `X-Team-ID` headers.
  - Calculates real-time cost-per-request based on prompt vs. completion tokens and hardware wattage.
  - Exposes Prometheus metrics (`cinch_tenant_cost_usd_total`, `cinch_tenant_budget_remaining`) and `/v1/tenants/usage` API.
- **Verification**: `tests/test_finops.py` and FinOps report `benchmarks/results/finops_eval.json`.
- **Documentation**: `docs/milestone-23.md`.

---

### Milestone 24: Production Shadow Traffic Replayer (`benchmarks/shadow_replayer.py`)
- **Problem**: Upgrading serving models or changing CUDA quantization kernels carries regression risks that synthetic benchmarks cannot always catch.
- **Implementation**:
  - Asynchronously duplicates a configurable percentage (e.g. $5\%$) of live user traffic to an experimental shadow instance.
  - Evaluates latency regressions, token acceptance rates, and semantic divergence in real time with zero user impact.
- **Verification**: `tests/test_shadow_replayer.py` and shadow analysis `benchmarks/results/shadow_replay_eval.json`.
- **Documentation**: `docs/milestone-24.md`.

---

### Milestone 25: Interactive Real-Time Serving Console (WebUI) (`ui/`)
- **Problem**: Black-box inference servers lack an intuitive visual interface for developers to test prompts, inspect KV-cache memory heatmaps, and monitor live streaming tokens.
- **Implementation**:
  - Lightweight, responsive WebUI serving console.
  - Features: Live SSE token streaming, prompt playground, real-time KV-cache utilization heatmaps, latency waterfall charts, and FinOps cost dashboards.
- **Verification**: `tests/test_ui_endpoints.py` and live visual verification.
- **Documentation**: `docs/milestone-25.md`.

---

### Milestone 26: Phase 3 Capstone Integration & Master Documentation
- **Objective**: Execute end-to-end regression validation across all 25 platform capabilities, publish final comparison dossiers, and update the master `README.md` under `stop-slop` guidelines.
- **Verification**: Full test suite (`python -m pytest tests/ -v`: 150+ tests passing) and `scripts/validate_phase3_full.py`.
- **Documentation**: `docs/milestone-26.md` and updated `README.md`.

---

## 5. Phase 3 Milestone Roadmap Summary

| Milestone | Capability | Key Metric / Deliverable |
|---|---|---|
| **M16** | Sub-5ms Semantic Vector Caching | **$<5\text{ms}$ response** at $0\text{W}$ GPU power ($\ge 0.95$ similarity) |
| **M17** | Multi-LoRA Dynamic Multiplexing | Serve multiple task adapters over single 7B base model ($<1\text{ms}$ switch) |
| **M18** | Guided EBNF / JSON Grammar Guard | **$100\%$ deterministic JSON schema compliance** |
| **M19** | Ingress Security & PII Redaction | $<1\text{ms}$ injection scan & automated PII token masking |
| **M20** | Smart Model Cascading | **$50\%$ GPU energy savings** by routing simple tasks to 0.5B model |
| **M21** | Context & Prompt Compaction | **$30\text{--}50\%$ KV-cache memory reduction** via entropy filtering |
| **M22** | Native Server-Side Tool Engine | Closed-loop server-side tool execution sandbox |
| **M23** | Multi-Tenant FinOps Cost Metering | Per-team dollar cost tracking & Prometheus chargeback metrics |
| **M24** | Production Shadow Traffic Replayer | Zero-impact $5\%$ live traffic replay for safe upgrades |
| **M25** | Interactive Real-Time Serving Console | Live SSE streaming console, KV-cache heatmaps, and latency waterfalls |
| **M26** | Capstone Integration & Master Dossier | 150+ automated tests passing, master README, and full regression |

---

## 6. Testing & Anti-Slop Policy

- Every milestone requires automated unit tests with $100\%$ pass rates.
- All technical documentation in `docs/` and `README.md` must adhere to `stop-slop` anti-slop rules (direct language, zero filler words, every performance claim traces to a verified JSON dataset in `benchmarks/results/`).
- Zero Git commits or pushes without explicit user approval.
