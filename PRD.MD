# PRD.md — Self-Hosted LLM Inference-Serving Platform

## 1. Overview
A production-style LLM inference-serving stack, built and benchmarked
entirely on local hardware at zero cloud cost. The project demonstrates
the engineering discipline that separates "I called an API" from "I
understand what happens between a request and a token" — quantization
tradeoffs, memory budgeting under real VRAM constraints, batching
behavior, and orchestration design.

## 2. Hardware target (fixed constraint, not a placeholder)
- Single local NVIDIA RTX 3060 Ti, 8GB VRAM, Ampere (compute capability 8.6)
- No cloud GPU spend, ever, in the default path
- Everything in this PRD must run to completion on this card. If a task
  needs more VRAM than this card has, the task is wrong, not the card.

## 3. Model & quantization
- **Base model:** Qwen2.5-7B-Instruct (Apache 2.0 — no license friction
  for a public repo, unlike Llama's community license terms)
- **Quantization:** AWQ, 4-bit. Rationale to document in repo: AWQ
  preserves accuracy on salient weight channels better than naive RTN
  quantization at this bit width, and vLLM has native AWQ kernel support
  on Ampere.
- **Memory budgeting (explicit, not default vLLM settings):**
  - `gpu_memory_utilization` tuned down from vLLM's 0.9 default — leave
    real headroom for OS/display driver overhead on a dev machine
  - `max_model_len` capped (target ~4096, not the model's native context)
    to leave enough KV cache room for meaningful concurrent batching
  - Both numbers get tuned empirically and the tuning process itself is
    a benchmark artifact, not a one-line config choice

## 4. Architecture
```
client → FastAPI gateway (auth, rate limit, request routing)
              → vLLM server (AWQ-quantized Qwen2.5-7B, Docker + GPU passthrough)
benchmark harness → hits gateway → records latency/throughput/memory → results/
```
- **Serving layer:** vLLM in Docker (`nvidia-container-toolkit` GPU
  passthrough), single container, single GPU
- **Gateway layer:** FastAPI — stateless, this is the layer that actually
  gets scaled in the autoscaling demo (see §6)
- **Orchestration layer:** Kubernetes manifests (Deployment, Service, HPA)
  written for real, deployed to a local k3d/kind cluster for validation

## 5. Benchmark suite (this is the deliverable that proves the project, not a nice-to-have)
Compare against a naive HF `transformers` baseline on identical hardware:
- p50 / p95 latency per request at 1, 4, 8, 16 concurrent requests
- Throughput: tokens/sec at each concurrency level
- VRAM footprint over time under sustained load
- Quality check: a small held-out prompt set scored for output
  equivalence between quantized and full-precision outputs (quantization
  isn't free — the PRD requires measuring what it costs, not just what
  it saves)
- Results committed to `benchmarks/results/` as the source of truth for
  every performance claim made in the README

## 6. Autoscaling — scoped honestly
**In scope:** real HPA configuration, real load-testing, real scaling
behavior — of the stateless FastAPI gateway layer on a local multi-node
k3d cluster, using CPU-based scaling triggers under generated load.

**Explicitly out of scope, and stated as such in the README:** live
autoscaling of GPU-backed vLLM replicas across real cloud nodes. This
project has one GPU. The manifests and design for that (cluster
autoscaler, GPU node pools, KEDA/queue-depth-based scaling triggers) are
written and documented as the production extension path, not claimed as
tested.

## 7. Milestones (feed this to the build-loop workflow as the initial task list)
1. Local vLLM serving Qwen2.5-7B-AWQ in Docker, single request working
2. Memory tuning pass — document `gpu_memory_utilization`/`max_model_len`
   decisions with evidence
3. FastAPI gateway in front of vLLM, basic auth + rate limiting
4. Benchmark harness — baseline HF `transformers` run recorded first
5. Benchmark harness — vLLM+AWQ run, full comparison table generated
6. Quality-equivalence check between quantized and full-precision outputs
7. K8s manifests for the full stack, deployed to local k3d
8. HPA on the gateway layer, load test, scaling behavior recorded
9. README: architecture diagram, benchmark results, honest scope
   boundaries (§6), and the "production extension path" section

## 8. Non-goals
- Not a chatbot demo, not a RAG pipeline (already covered elsewhere in
  the portfolio) — this project's identity is the serving/infra layer
  only
- Not claiming multi-GPU or cloud-scale numbers this hardware can't
  produce