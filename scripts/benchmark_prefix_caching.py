"""Benchmark harness evaluating TTFT and latency speedup from vLLM Prefix Caching."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, List
import httpx


SYSTEM_PROMPT_SHARED = """You are an elite Staff Software Engineer and Distributed Systems Architect specializing in Kubernetes, asynchronous event-driven microservices, high-throughput streaming RPCs, and low-latency database sharding. You provide precise, rigorous architectural reviews with complete code examples, memory budgeting calculations, concurrency safety analysis, and failure mode mitigation strategies."""


async def measure_streaming_ttft(
    client: httpx.AsyncClient,
    gateway_url: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    iteration: int,
) -> Dict[str, Any]:
    """Execute streaming completion request and measure exact Time-To-First-Token (TTFT)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 32,
        "stream": True,
    }

    endpoint = f"{gateway_url.rstrip('/')}/v1/chat/completions"
    t_start = time.perf_counter()
    ttft = None
    first_chunk_received = False
    tokens_received = 0
    cache_status = "UNKNOWN"
    prefix_hash = "none"

    try:
        async with client.stream("POST", endpoint, json=payload, headers=headers, timeout=60.0) as resp:
            cache_status = resp.headers.get("X-Cache-Status", "UNKNOWN")
            prefix_hash = resp.headers.get("X-Cache-Prefix-Hash", "none")

            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                if not first_chunk_received:
                    ttft = time.perf_counter() - t_start
                    first_chunk_received = True
                tokens_received += 1

        total_latency = time.perf_counter() - t_start
        if ttft is None:
            ttft = total_latency

    except Exception as exc:
        total_latency = time.perf_counter() - t_start
        ttft = total_latency
        cache_status = f"ERROR: {exc}"

    return {
        "iteration": iteration,
        "cache_status": cache_status,
        "prefix_hash": prefix_hash,
        "ttft_seconds": round(ttft, 4),
        "total_latency_seconds": round(total_latency, 4),
        "tokens_generated": tokens_received,
    }


async def run_prefix_cache_benchmark(
    gateway_url: str = "http://localhost:8081",
    api_key: str = "cinch-prod-key",
    iterations: int = 5,
    output_path: str = "benchmarks/results/prefix_cache_benchmark.json",
) -> Dict[str, Any]:
    """Run comparative benchmark across shared warm prefix cache vs unique cold prefixes."""
    print(f"=== Starting Prefix Cache Benchmark on {gateway_url} ===")

    shared_results: List[Dict[str, Any]] = []
    unique_results: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        # 1. Warm Prefix Cache Sweep (Shared System Prompt)
        print("\n--- Running Shared System Prompt Sweep (Prefix Cache Hits) ---")
        for i in range(iterations):
            user_msg = f"Provide question #{i+1}: What is linearizability vs serializability in distributed databases?"
            res = await measure_streaming_ttft(
                client=client,
                gateway_url=gateway_url,
                api_key=api_key,
                system_prompt=SYSTEM_PROMPT_SHARED,
                user_prompt=user_msg,
                iteration=i + 1,
            )
            shared_results.append(res)
            print(
                f"[Shared Run #{res['iteration']:2d}] TTFT: {res['ttft_seconds']:6.3f}s | "
                f"Latency: {res['total_latency_seconds']:6.3f}s | Status: {res['cache_status']:4s} | Hash: {res['prefix_hash']}"
            )
            await asyncio.sleep(0.5)

        # 2. Unique Cold Prefix Sweep (Different system prompt each request)
        print("\n--- Running Unique Cold Prompt Sweep (Prefix Cache Misses) ---")
        for i in range(iterations):
            unique_sys = f"You are arbitrary persona #{i+1} with unique instructions: {os.urandom(32).hex()}."
            user_msg = "Provide a brief summary of consensus algorithms."
            res = await measure_streaming_ttft(
                client=client,
                gateway_url=gateway_url,
                api_key=api_key,
                system_prompt=unique_sys,
                user_prompt=user_msg,
                iteration=i + 1,
            )
            unique_results.append(res)
            print(
                f"[Unique Run #{res['iteration']:2d}] TTFT: {res['ttft_seconds']:6.3f}s | "
                f"Latency: {res['total_latency_seconds']:6.3f}s | Status: {res['cache_status']:4s} | Hash: {res['prefix_hash']}"
            )
            await asyncio.sleep(0.5)

    # 3. Calculate Performance Metrics
    cold_shared_ttft = shared_results[0]["ttft_seconds"]
    warm_shared_ttft = sum(r["ttft_seconds"] for r in shared_results[1:]) / max(1, len(shared_results) - 1)
    unique_avg_ttft = sum(r["ttft_seconds"] for r in unique_results) / len(unique_results)

    ttft_speedup = (cold_shared_ttft / warm_shared_ttft) if warm_shared_ttft > 0 else 1.0

    print("\n=== Benchmark Summary ===")
    print(f"Initial Cold TTFT (Shared):     {cold_shared_ttft:.4f}s")
    print(f"Average Warm TTFT (Shared):     {warm_shared_ttft:.4f}s")
    print(f"Average Unique TTFT (Cold):     {unique_avg_ttft:.4f}s")
    print(f"TTFT Reduction Speedup:         {ttft_speedup:.2f}x faster TTFT on prefix cache hits")

    payload = {
        "benchmark": "prefix_cache_affinity_evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_url": gateway_url,
        "iterations_per_suite": iterations,
        "metrics": {
            "initial_cold_ttft_seconds": round(cold_shared_ttft, 4),
            "average_warm_ttft_seconds": round(warm_shared_ttft, 4),
            "average_unique_ttft_seconds": round(unique_avg_ttft, 4),
            "ttft_speedup_factor": round(ttft_speedup, 2),
        },
        "shared_prefix_runs": shared_results,
        "unique_prefix_runs": unique_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[SAVED] Benchmark data exported to {output_path}")
    return payload


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Prefix Caching TTFT Benchmark")
    parser.add_argument("--gateway-url", type=str, default="http://localhost:8081", help="Gateway URL")
    parser.add_argument("--api-key", type=str, default="cinch-prod-key", help="Gateway API Key")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per sweep")
    parser.add_argument("--output", type=str, default="benchmarks/results/prefix_cache_benchmark.json", help="Output path")
    args = parser.parse_args()

    asyncio.run(
        run_prefix_cache_benchmark(
            gateway_url=args.gateway_url,
            api_key=args.api_key,
            iterations=args.iterations,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
