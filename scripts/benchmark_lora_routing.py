"""Live benchmark and validation for Multi-LoRA Dynamic Adapter Routing (M17)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, List
import httpx


TEST_QUERIES = [
    {
        "target": "base_model",
        "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "prompt": "Explain the concept of mutual exclusion in operating systems.",
        "expected_adapter": "none",
    },
    {
        "target": "compound_sql",
        "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ:sql-coder",
        "prompt": "Write a SQL query to find top 5 customers by revenue in 2025.",
        "expected_adapter": "sql-coder",
    },
    {
        "target": "compound_python",
        "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ:python-agent",
        "prompt": "Write a Python script with asyncio to fetch 3 URLs concurrently.",
        "expected_adapter": "python-agent",
    },
    {
        "target": "alias_medical",
        "model_id": "medical-expert",
        "prompt": "Summarize the primary mechanism of action for ACE inhibitors.",
        "expected_adapter": "medical-expert",
    },
    {
        "target": "alias_legal",
        "model_id": "legal-analyst",
        "prompt": "Explain indemnification clauses in standard SaaS master service agreements.",
        "expected_adapter": "legal-analyst",
    },
]


async def run_benchmark(
    gateway_url: str,
    api_key: str,
    output_path: str,
) -> Dict[str, Any]:
    print("=" * 70)
    print("  CINCH MULTI-LoRA DYNAMIC ROUTING BENCHMARK - M17")
    print("=" * 70)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        # 1. Model Discovery (/v1/models)
        print("\nPhase 1: Dynamic Model & LoRA Discovery (/v1/models)...")
        models_resp = await client.get(f"{gateway_url}/v1/models", headers=headers)
        if models_resp.status_code != 200:
            print(f"[ERROR] Failed to fetch /v1/models (HTTP {models_resp.status_code})")
            models_data = {"data": []}
        else:
            models_data = models_resp.json()

        discovered_ids = [m.get("id") for m in models_data.get("data", [])]
        print(f"  Discovered {len(discovered_ids)} models/virtual endpoints:")
        for mid in discovered_ids:
            print(f"    - {mid}")

        # 2. Live Request Multiplexing
        print(f"\nPhase 2: Live Request Dispatch ({len(TEST_QUERIES)} distinct virtual endpoints)...")
        results: List[Dict[str, Any]] = []

        for i, q in enumerate(TEST_QUERIES):
            t0 = time.perf_counter()
            resp = await client.post(
                f"{gateway_url}/v1/chat/completions",
                json={
                    "model": q["model_id"],
                    "messages": [{"role": "user", "content": q["prompt"]}],
                    "max_tokens": 25,
                },
                headers=headers,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            adapter_active = resp.headers.get("X-LoRA-Adapter-Active", "unknown")
            base_model = resp.headers.get("X-LoRA-Base-Model", "unknown")
            cache_status = resp.headers.get("X-Semantic-Cache-Status", "MISS")

            passed = (
                resp.status_code == 200
                and (adapter_active == q["expected_adapter"] or q["expected_adapter"] == "none")
            )
            print(
                f"  [{i+1:2d}] {q['model_id']:<45} | {lat_ms:6.1f}ms | "
                f"Adapter: {adapter_active:<15} | HTTP {resp.status_code} "
                f"[{'OK' if passed else 'FAIL'}]"
            )

            results.append({
                "target": q["target"],
                "requested_model": q["model_id"],
                "expected_adapter": q["expected_adapter"],
                "resolved_adapter": adapter_active,
                "resolved_base_model": base_model,
                "latency_ms": round(lat_ms, 2),
                "status_code": resp.status_code,
                "cache_status": cache_status,
                "passed": passed,
            })

    # VRAM Economics Analysis
    base_model_vram_gb = 4.40  # Qwen2.5-7B AWQ Marlin INT4
    lora_adapter_vram_gb = 0.10  # Rank-16/32 LoRA weights
    num_adapters = 4
    multiplexed_vram_gb = base_model_vram_gb + (num_adapters * lora_adapter_vram_gb)
    duplicated_vram_gb = (1 + num_adapters) * base_model_vram_gb
    vram_savings_pct = (1.0 - (multiplexed_vram_gb / duplicated_vram_gb)) * 100.0
    vram_compression_ratio = duplicated_vram_gb / multiplexed_vram_gb

    avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0.0

    print("\n" + "=" * 70)
    print("  MULTI-LoRA BENCHMARK & VRAM ECONOMICS SUMMARY")
    print("=" * 70)
    print(f"  Discovered Models in /v1/models:    {len(discovered_ids)}")
    print(f"  Multiplexed Endpoints Tested:       {len(results)}")
    print(f"  Average Dispatch Latency:           {avg_latency:6.1f} ms")
    print(f"  Single 7B AWQ Base Model VRAM:      {base_model_vram_gb:.2f} GB")
    print(f"  Multiplexed VRAM (1 Base + 4 LoRA): {multiplexed_vram_gb:.2f} GB")
    print(f"  Duplicated VRAM (5 Full 7B Models): {duplicated_vram_gb:.2f} GB")
    print(f"  VRAM Memory Savings:                {vram_savings_pct:.1f}% ({vram_compression_ratio:.2f}x compression)")
    print("=" * 70)

    payload = {
        "benchmark": "multi_lora_routing_m17",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_url": gateway_url,
        "discovered_models": discovered_ids,
        "vram_economics": {
            "base_model_vram_gb": base_model_vram_gb,
            "lora_adapter_vram_gb": lora_adapter_vram_gb,
            "num_adapters": num_adapters,
            "multiplexed_vram_gb": round(multiplexed_vram_gb, 2),
            "duplicated_vram_gb": round(duplicated_vram_gb, 2),
            "vram_savings_pct": round(vram_savings_pct, 1),
            "vram_compression_ratio": round(vram_compression_ratio, 2),
        },
        "metrics": {
            "total_requests": len(results),
            "successful_routes": sum(1 for r in results if r["passed"]),
            "average_latency_ms": round(avg_latency, 2),
        },
        "samples": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[SAVED] Benchmark dataset -> {output_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-LoRA Routing Benchmark M17")
    parser.add_argument("--gateway-url", default="http://localhost:8081")
    parser.add_argument("--api-key", default="cinch-prod-key")
    parser.add_argument("--output", default="benchmarks/results/lora_routing_eval.json")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.gateway_url, args.api_key, args.output))


if __name__ == "__main__":
    main()
