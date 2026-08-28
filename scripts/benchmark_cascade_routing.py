"""Live Smart Model Cascading & Complexity Routing Benchmark (M20)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, List
import httpx


CASCADE_SCENARIOS = [
    {
        "name": "simple_greeting",
        "category": "chit_chat",
        "prompt": "Hello! Good morning.",
        "expected_tier": "SMALL",
        "max_score": 0.40,
    },
    {
        "name": "sentiment_classification",
        "category": "text_classification",
        "prompt": "Classify sentiment of this review: 'The latency was exceptional and rock solid.'",
        "expected_tier": "SMALL",
        "max_score": 0.45,
    },
    {
        "name": "short_factoid_qa",
        "category": "factoid_qa",
        "prompt": "What is the capital of France?",
        "expected_tier": "SMALL",
        "max_score": 0.45,
    },
    {
        "name": "async_python_code_generator",
        "category": "software_engineering",
        "prompt": "Write a Python function using async def to fetch data concurrently with a connection pool.",
        "expected_tier": "LARGE",
        "min_score": 0.50,
    },
    {
        "name": "multi_table_sql_join",
        "category": "database_engineering",
        "prompt": "Write a SQL query: SELECT department_id, avg(salary) FROM employees GROUP BY department_id HAVING avg(salary) > 80000",
        "expected_tier": "LARGE",
        "min_score": 0.50,
    },
    {
        "name": "mathematical_proof_derivation",
        "category": "mathematical_reasoning",
        "prompt": "Derive the mathematical proof for gradient descent convergence on convex functions.",
        "expected_tier": "LARGE",
        "min_score": 0.50,
    },
]


async def run_benchmark(
    gateway_url: str,
    api_key: str,
    output_path: str,
) -> Dict[str, Any]:
    print("=" * 70)
    print("  CINCH SMART MODEL CASCADING & COMPLEXITY ROUTING BENCHMARK - M20")
    print("=" * 70)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    results: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        # First verify /v1/models exposes virtual auto model
        models_resp = await client.get(f"{gateway_url}/v1/models", headers=headers)
        models_list = models_resp.json().get("data", [])
        has_auto = any(m.get("id") == "auto" for m in models_list)
        print(f"\n[DISCOVERY] /v1/models: {len(models_list)} models discovered (auto-router active: {has_auto})")

        print(f"\nDispatching {len(CASCADE_SCENARIOS)} Cascading Query Scenarios (model='auto')...")

        for i, scenario in enumerate(CASCADE_SCENARIOS):
            t0 = time.perf_counter()
            resp = await client.post(
                f"{gateway_url}/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": scenario["prompt"]}],
                    "max_tokens": 30,
                },
                headers=headers,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0

            tier = resp.headers.get("X-Cascade-Routing-Tier", "UNKNOWN")
            score_str = resp.headers.get("X-Cascade-Complexity-Score", "0.0")
            score = float(score_str)
            selected_model = resp.headers.get("X-Cascade-Selected-Model", "unknown")
            reason = resp.headers.get("X-Cascade-Reason", "unknown")

            passed = tier == scenario["expected_tier"]

            print(
                f"  [{i + 1:2d}] {scenario['name']:<30} | {lat_ms:6.1f}ms | "
                f"Score: {score:.2f} | Tier: {tier:<5} -> {selected_model.split('/')[-1]:<25} "
                f"[{'PASS' if passed else 'FAIL'}]"
            )

            results.append(
                {
                    "scenario": scenario["name"],
                    "category": scenario["category"],
                    "prompt": scenario["prompt"],
                    "expected_tier": scenario["expected_tier"],
                    "assigned_tier": tier,
                    "complexity_score": score,
                    "selected_model": selected_model,
                    "reason": reason,
                    "latency_ms": round(lat_ms, 2),
                    "passed": passed,
                }
            )

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    accuracy = (passed_count / max(total, 1)) * 100.0
    small_count = sum(1 for r in results if r["assigned_tier"] == "SMALL")
    large_count = sum(1 for r in results if r["assigned_tier"] == "LARGE")
    avg_latency = sum(r["latency_ms"] for r in results) / max(total, 1)

    # Small tier compute savings: 0.5B model vs 7B AWQ (~93% reduction per small request)
    energy_saved_pct = round((small_count * 0.93 / max(total, 1)) * 100.0, 1)

    print("\n" + "=" * 70)
    print("  CASCADE ROUTER BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Total Scenarios Evaluated:         {total}")
    print(f"  Routing Classification Accuracy:   {passed_count} / {total} ({accuracy:.1f}%)")
    print(f"  Small Tier Invocations (0.5B):     {small_count} ({small_count / total * 100:.1f}%)")
    print(f"  Large Tier Invocations (7B AWQ):   {large_count} ({large_count / total * 100:.1f}%)")
    print(f"  Estimated GPU Energy Saved:        {energy_saved_pct}% compute reduction")
    print(f"  Average End-to-End Latency:        {avg_latency:6.1f} ms")
    print("=" * 70)

    payload = {
        "benchmark": "smart_model_cascading_m20",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_url": gateway_url,
        "metrics": {
            "total_scenarios": total,
            "routing_accuracy_pct": round(accuracy, 1),
            "small_tier_count": small_count,
            "large_tier_count": large_count,
            "estimated_gpu_energy_saved_pct": energy_saved_pct,
            "average_latency_ms": round(avg_latency, 2),
        },
        "scenarios": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[SAVED] Cascade routing dataset -> {output_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Cascade Router Benchmark M20")
    parser.add_argument("--gateway-url", default="http://localhost:8081")
    parser.add_argument("--api-key", default="cinch-prod-key")
    parser.add_argument("--output", default="benchmarks/results/cascade_routing_eval.json")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.gateway_url, args.api_key, args.output))


if __name__ == "__main__":
    main()
