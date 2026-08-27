"""Live benchmark for Semantic Vector Cache: measures cold GPU latency vs. cache HIT response time."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, List
import httpx


COLD_PROMPTS = [
    "How do I connect to a PostgreSQL database in Python?",
    "Write a function to compute Fibonacci numbers recursively.",
    "Explain the difference between TCP and UDP protocols.",
    "What is the time complexity of quicksort in the worst case?",
    "How does Kubernetes horizontal pod autoscaling work?",
    "Write a Python context manager for file locking.",
    "What is the difference between a mutex and a semaphore?",
    "Explain gradient descent optimization in machine learning.",
    "How does PagedAttention improve LLM KV-cache efficiency?",
    "What is the purpose of the softmax function in neural networks?",
]

# Paraphrase variants of the same semantic intent as cold prompts
PARAPHRASE_PROMPTS = [
    "Python script to establish a PostgreSQL connection",
    "Fibonacci sequence implementation using recursion in Python",
    "Differences between TCP and UDP networking protocols",
    "Worst-case time complexity analysis of quicksort algorithm",
    "Kubernetes HPA autoscaling explained",
    "Python file lock using context manager",
    "Mutex versus semaphore in concurrent programming",
    "Gradient descent for neural network training",
    "PagedAttention KV cache optimization for language models",
    "Softmax activation function explained",
]


async def run_benchmark(
    gateway_url: str,
    api_key: str,
    output_path: str,
) -> Dict[str, Any]:
    print("=" * 70)
    print("  CINCH SEMANTIC CACHE BENCHMARK — M16")
    print("=" * 70)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    model = "Qwen/Qwen2.5-7B-Instruct-AWQ"

    cold_latencies: List[float] = []
    hit_latencies: List[float] = []
    paraphrase_latencies: List[float] = []
    paraphrase_hits = 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        # Phase 1: Cold requests — all MISS, populates cache
        print(f"\nPhase 1: Cold Baseline ({len(COLD_PROMPTS)} unique queries -> MISS expected)...")
        for i, prompt in enumerate(COLD_PROMPTS):
            t0 = time.perf_counter()
            resp = await client.post(
                f"{gateway_url}/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 30},
                headers=headers,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            cold_latencies.append(lat_ms)
            cache_status = resp.headers.get("X-Semantic-Cache-Status", "unknown")
            print(f"  [{i+1:2d}] {lat_ms:7.1f}ms | {cache_status} | HTTP {resp.status_code}")

        # Phase 2: Exact repeat requests — all HIT
        print(f"\nPhase 2: Exact Repeat Queries ({len(COLD_PROMPTS)} identical requests -> HIT expected)...")
        for i, prompt in enumerate(COLD_PROMPTS):
            t0 = time.perf_counter()
            resp = await client.post(
                f"{gateway_url}/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 30},
                headers=headers,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            hit_latencies.append(lat_ms)
            cache_status = resp.headers.get("X-Semantic-Cache-Status", "unknown")
            similarity = resp.headers.get("X-Semantic-Cache-Similarity", "n/a")
            print(f"  [{i+1:2d}] {lat_ms:7.2f}ms | {cache_status} (sim={similarity}) | HTTP {resp.status_code}")

        # Phase 3: Paraphrase variants
        print(f"\nPhase 3: Paraphrase Variants ({len(PARAPHRASE_PROMPTS)} rephrasings -> HIT rate measured)...")
        for i, prompt in enumerate(PARAPHRASE_PROMPTS):
            t0 = time.perf_counter()
            resp = await client.post(
                f"{gateway_url}/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 30},
                headers=headers,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            paraphrase_latencies.append(lat_ms)
            cache_status = resp.headers.get("X-Semantic-Cache-Status", "unknown")
            similarity = resp.headers.get("X-Semantic-Cache-Similarity", "0")
            if cache_status == "HIT":
                paraphrase_hits += 1
            print(f"  [{i+1:2d}] {lat_ms:7.2f}ms | {cache_status} (sim={similarity}) | HTTP {resp.status_code}")

    avg_cold = sum(cold_latencies) / len(cold_latencies)
    avg_hit = sum(hit_latencies) / len(hit_latencies)
    speedup = avg_cold / max(avg_hit, 0.01)
    paraphrase_hit_rate = paraphrase_hits / len(PARAPHRASE_PROMPTS)

    print("\n" + "=" * 70)
    print("  SEMANTIC CACHE BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Cold GPU Path Average Latency:    {avg_cold:7.1f} ms")
    print(f"  Semantic Cache HIT Average:       {avg_hit:7.2f} ms")
    print(f"  Speedup Factor:                   {speedup:.1f}x faster on cache HITs")
    print(f"  Paraphrase Hit Rate:              {paraphrase_hit_rate:.0%} ({paraphrase_hits}/{len(PARAPHRASE_PROMPTS)} paraphrases hit cache)")
    print("=" * 70)

    payload = {
        "benchmark": "semantic_vector_cache_m16",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_url": gateway_url,
        "metrics": {
            "cold_gpu_avg_latency_ms": round(avg_cold, 2),
            "cache_hit_avg_latency_ms": round(avg_hit, 2),
            "speedup_factor": round(speedup, 1),
            "paraphrase_hit_rate": round(paraphrase_hit_rate, 3),
            "paraphrase_hits": paraphrase_hits,
            "paraphrase_total": len(PARAPHRASE_PROMPTS),
        },
        "cold_samples_ms": [round(x, 2) for x in cold_latencies],
        "hit_samples_ms": [round(x, 2) for x in hit_latencies],
        "paraphrase_samples_ms": [round(x, 2) for x in paraphrase_latencies],
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[SAVED] Benchmark results -> {output_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Cache Benchmark M16")
    parser.add_argument("--gateway-url", default="http://localhost:8081")
    parser.add_argument("--api-key", default="cinch-prod-key")
    parser.add_argument("--output", default="benchmarks/results/semantic_cache_eval.json")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.gateway_url, args.api_key, args.output))


if __name__ == "__main__":
    main()
