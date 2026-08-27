# Milestone 4: Benchmark Harness and Naive Hugging Face Baseline

This document records the design of the Cinch load-testing benchmark harness and the empirical performance baseline of naive Hugging Face `transformers` on the target NVIDIA GeForce RTX 3060 Ti hardware.

---

## 1. Benchmark Harness Design

The benchmark harness in `benchmarks/runner.py` provides automated load generation across standardized prompt tiers.

### Components
1. **Prompt Dataset (`benchmarks/prompts.json`)**: 12 curated prompts across short (32-64 tokens), medium (128-160 tokens), and long (256 tokens) generation targets.
2. **Metrics Collector (`benchmarks/metrics.py`)**: Computes exact percentiles (p50, p90, p95, p99, min, max, mean), token throughput (tokens/sec), request throughput (RPS), and status distributions.
3. **VRAM Sampler (`benchmarks/vram_sampler.py`)**: Background thread polling `nvidia-smi` at 100ms intervals to track peak and average memory footprint during active runs.
4. **Concurrency Pool (`benchmarks/runner.py`)**: Asynchronous worker pool driving concurrency tiers $C \in [1, 4, 8, 16]$ with warmup requests and JSON report exports.

---

## 2. Naive Hugging Face Baseline Architecture

The naive baseline in `benchmarks/baseline_hf.py` models standard unoptimized model serving without continuous batching or PagedAttention.

### Characteristics
- Single-stream sequential execution: Each incoming request processes one after another.
- Memory allocation: Fixed model weights in VRAM without dynamic KV cache memory paging.
- Concurrency behavior: Requests entering concurrently wait in queue while previous requests finish token generation.

---

## 3. Empirical Baseline Results

Source file: `benchmarks/results/baseline_hf.json`  
Hardware: NVIDIA GeForce RTX 3060 Ti (8192 MiB VRAM)  
Model: `Qwen/Qwen2.5-7B-Instruct-AWQ`  
Requests per concurrency tier: 16  

| Concurrency Level | Total Duration (s) | Throughput (tok/s) | Request Rate (RPS) | Latency p50 (s) | Latency p95 (s) | Latency Max (s) | Peak VRAM (MiB) |
|---|---|---|---|---|---|---|---|
| **1** | 33.67 | 30.41 | 0.48 | 2.105 | 2.110 | 2.110 | 7821.0 |
| **4** | 33.62 | 30.46 | 0.48 | 8.403 | 8.414 | 8.417 | 7747.0 |
| **8** | 33.60 | 30.48 | 0.48 | 16.787 | 16.798 | 16.800 | 7710.0 |
| **16** | 33.61 | 30.47 | 0.48 | 17.845 | 32.025 | 33.606 | 7709.0 |

---

## 4. Key Performance Observations

### Throughput Plateau
Aggregate generation throughput remains flat at **30.4-30.5 tokens/sec** across all concurrency levels. Without continuous batching, the GPU computes generation steps for only one sequence at a time. Increasing client concurrency yields zero throughput gain.

### Linear Latency Degradation
Because incoming requests queue behind active generations:
- At **Concurrency 1**, p95 latency is **2.11 seconds**.
- At **Concurrency 4**, p95 latency quadruples to **8.41 seconds**.
- At **Concurrency 8**, p95 latency reaches **16.80 seconds**.
- At **Concurrency 16**, p95 latency spikes to **32.03 seconds** (with max latency reaching 33.61 seconds).

This data establishes the empirical baseline against which vLLM continuous batching and PagedAttention will be compared in Milestone 5.
