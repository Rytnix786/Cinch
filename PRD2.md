# PRD Phase 2: Cinch Enterprise Infrastructure & Inference Optimization

## 1. Executive Summary & Objective

Phase 1 established the baseline LLM serving platform: 4-bit AWQ Marlin serving on vLLM, a stateless FastAPI gateway, automated benchmarks (10.86x throughput speedup), quality evaluation (97.2% retention), multi-node Kubernetes orchestration on k3d, and CPU-based HPA scaling.

**Phase 2 Objective**: Upgrade Cinch from a baseline inference server to an enterprise-grade LLM serving platform. This phase adds token-budgeted multi-tenancy, prefix cache routing affinity, speculative decoding, end-to-end telemetry (Prometheus, Grafana, OpenTelemetry), circuit-breaking fault tolerance, and an automated model quantization pipeline.

---

## 2. Hardware Environment & Realism Guardrails

- **Workstation Hardware**: 1x NVIDIA GeForce RTX 3060 Ti (8,192 MiB VRAM), AMD Ryzen 5 CPU, 16GB RAM.
- **Serving Precision**: AWQ Marlin INT4 Weights / FP16 Activations (W4A16).
- **Cluster Environment**: Multi-node local k3d Kubernetes cluster (`cinch-cluster`: 1 control plane, 2 worker nodes) connected to the host GPU daemon via `host.k3d.internal`.
- **Honest Scope Boundary**:
  - All Gateway routing, token estimation, priority queues, telemetry pipelines, and circuit breakers execute live in the Kubernetes cluster.
  - Speculative decoding runs directly on hardware (n-gram or small draft model).
  - Multi-node GPU cluster behaviors remain documented as the production extension path where hardware is physically limited to a single GPU.

---

## 3. Architecture Evolution

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
|  |  [ Pod: cinch-gateway-pod-1 ]     |     |  [ Pod: cinch-gateway-pod-2 ]     |        |
|  |   - OpenTelemetry Trace Injection |     |   - OpenTelemetry Trace Injection |        |
|  |   - Token-Aware TPM Rate Limiter  |     |   - Token-Aware TPM Rate Limiter  |        |
|  |   - Dual-Priority Scheduling Queue|     |   - Dual-Priority Scheduling Queue|        |
|  |   - Prefix-Cache Hash Router      |     |   - Prefix-Cache Hash Router      |        |
|  |   - Adaptive Circuit Breaker      |     |   - Adaptive Circuit Breaker      |        |
|  +-----------------------------------+     +-----------------------------------+        |
|                    |                                         |                          |
|                    +--------------------+--------------------+                          |
|                                         |                                               |
|      +----------------------------------+----------------------------------+            |
|      | (Observability Mesh)                                                |            |
|      v                                                                     v            |
|  +-----------------------------------+                     +-------------------------+  |
|  | [Pod: prometheus-k8s]             |                     | [Pod: grafana-k8s]      |  |
|  |  - Scrapes Gateway :8080/metrics  |                     |  - Pre-built Dashboards |  |
|  |  - Scrapes vLLM :8000/metrics     |                     |  - TTFT, TPOT, KV-Cache |  |
|  +-----------------------------------+                     +-------------------------+  |
+-----------------------------------------|-----------------------------------------------+
                                          | (Cluster DNS / host.k3d.internal:8000)
                                          v
                      +----------------------------------------+
                      | Host GPU Docker Daemon                 |
                      |  [ Container: cinch-vllm ]             |
                      |   - vLLM v0.6.3 + Marlin AWQ Kernel    |
                      |   - Speculative Decoding Engine        |
                      |   - Automatic Prefix Caching (APC)     |
                      |   - Prometheus Metrics Exporter :8000  |
                      +----------------------------------------+
```

---

## 4. Phase 2 Technical Modules

### Module 1: Token-Aware Rate Limiting & Tiered Priority Queue
- **Problem**: RPM (Requests-per-minute) rate limiting treats a 4,000-token prompt the same as a 10-token prompt, causing GPU KV-cache starvation.
- **Implementation**:
  - Fast BPE Token Estimator at gateway ingress.
  - Token-Bucket algorithm budgeting Tokens-Per-Minute (TPM) in addition to RPM.
  - Dual-Priority Queue: `Interactive (High Priority)` with low queue latency vs `Batch (Low Priority)` drained during idle windows.

### Module 2: Prefix Cache Affinity & Hash Router
- **Problem**: Kubernetes round-robin ingress scatters identical system prompts across different workers, defeating vLLM's Automatic Prefix Caching (PagedAttention).
- **Implementation**:
  - Gateway computes rolling SHA-256 hash across system prompts and conversation prefixes.
  - Maintains affinity table routing identical prefix requests to preferred backend instances to maximize cache hit rates.

### Module 3: Speculative Decoding Optimization & Benchmark Suite
- **Problem**: Autoregressive decoding is memory-bandwidth bound ($O(1)$ token per memory read).
- **Implementation**:
  - Configure speculative decoding on vLLM (using draft models or n-gram speculative proposals).
  - Expand benchmark suite to measure draft acceptance rate ($\alpha$), single-stream latency speedup ($1.4\times\text{--}2.0\times$), and quality equivalence verification.

### Module 4: Production Observability Stack (Prometheus + Grafana + OpenTelemetry)
- **Problem**: Black-box inference serving lacks granular visibility into Time-To-First-Token (TTFT), Time-Per-Output-Token (TPOT), and KV cache saturation.
- **Implementation**:
  - Kubernetes manifests for Prometheus and Grafana scraping both Gateway (`/metrics`) and vLLM (`:8000/metrics`).
  - OpenTelemetry distributed tracing: `traceparent` propagation from Gateway down through SSE chunk streaming.
  - Custom Grafana dashboard JSON visualizing TTFT histograms, TPOT rates, and GPU memory saturation.

### Module 5: Adaptive Circuit Breaking & Chaos Engineering
- **Problem**: Inference backend saturation or transient GPU OOM cascades into dropped connections and gateway crashes.
- **Implementation**:
  - Gateway Circuit Breaker with sliding error window and p95 latency thresholds.
  - Graceful load shedding returning `503 Service Unavailable` with `Retry-After` headers during overload.
  - Automated chaos test suite (`scripts/chaos_test.py`) injecting container failures and verifying instant recovery.

### Module 6: Automated End-to-End Model Quantization Pipeline
- **Problem**: Relying solely on third-party pre-quantized weights limits model agility.
- **Implementation**:
  - Standalone pipeline script (`scripts/quantize_awq.py`) automating calibration dataset generation, AutoAWQ 4-bit quantization, Marlin kernel validation, and perplexity verification.

---

## 5. Phase 2 Milestones (Task List for Build Loop)

1. **Milestone 10: Token-Aware Rate Limiter & Tiered Priority Queue**
   - Implement token estimation, TPM token-bucket limiter, and dual-tier priority queue in `gateway/`.
   - Unit tests and live load verification demonstrating VIP request preemption under load.
   - Document in `docs/milestone-10.md`.

2. **Milestone 11: Prefix Cache Affinity & Hash Router**
   - Implement prefix hashing and cache affinity routing in `gateway/`.
   - Benchmark prefix cache hit rate and TTFT latency reduction under shared prompt sweeps.
   - Document in `docs/milestone-11.md`.

3. **Milestone 12: Speculative Decoding Integration & Benchmark Suite**
   - Configure speculative decoding in vLLM serving parameters.
   - Run concurrency and single-stream latency benchmark sweeps; compute acceptance rate ($\alpha$).
   - Document in `docs/milestone-12.md`.

4. **Milestone 13: Observability Stack (Prometheus, Grafana, OpenTelemetry)**
   - Implement Prometheus and Grafana manifests in `k8s/monitoring/`.
   - Integrate OpenTelemetry distributed tracing across Gateway proxy layers.
   - Export Grafana dashboard configuration and capture live metric visualization.
   - Document in `docs/milestone-13.md`.

5. **Milestone 14: Circuit Breaker & Chaos Resilience Engineering**
   - Implement adaptive circuit breaker and backoff load shedding in `gateway/`.
   - Build and execute chaos testing script (`scripts/chaos_test.py`) simulating engine failure/recovery.
   - Document in `docs/milestone-14.md`.

6. **Milestone 15: Automated End-to-End AWQ Quantization Pipeline**
   - Build `scripts/quantize_awq.py` for automated model calibration, AWQ INT4 export, and Marlin validation.
   - Run quality and perplexity verification on generated weights.
   - Update `README.md` and documentation with Phase 2 capabilities and architecture.

---

## 6. Testing Bar & Anti-Slop Policy

- **Testing Standard**: Every new gateway mechanism, routing policy, telemetry pipeline, and quantization script must ship with automated unit and integration tests (target: 90%+ coverage).
- **Anti-Slop Standard**: No unsubstantiated performance claims. Every metric (TTFT speedup, acceptance rate, TPM throughput) must trace directly to a JSON artifact in `benchmarks/results/`.
