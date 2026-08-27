"""Live validation and latency measurement for Gateway Priority Queue & TPM Limiting."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict
import httpx


async def send_priority_request(
    client: httpx.AsyncClient,
    gateway_url: str,
    api_key: str,
    req_index: int,
    priority: str,
    prompt: str,
    max_tokens: int,
    send_time_offset: float = 0.0,
) -> Dict[str, Any]:
    """Send a single request with designated priority header and measure timing."""
    if send_time_offset > 0:
        await asyncio.sleep(send_time_offset)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Priority": priority,
    }
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "priority": priority,
    }

    req_start = time.perf_counter()
    endpoint = f"{gateway_url.rstrip('/')}/v1/chat/completions"

    try:
        resp = await client.post(endpoint, json=payload, headers=headers, timeout=60.0)
        latency = time.perf_counter() - req_start
        status_code = resp.status_code
        res_json = resp.json() if status_code == 200 else {}
        text = res_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        req_id = resp.headers.get("X-Request-ID", f"req-{req_index}")
        est_tokens = resp.headers.get("X-Request-Estimated-Tokens", "unknown")
    except Exception as exc:
        latency = time.perf_counter() - req_start
        status_code = 500
        text = f"Error: {exc}"
        req_id = f"err-{req_index}"
        est_tokens = "0"

    return {
        "req_index": req_index,
        "priority": priority,
        "status_code": status_code,
        "latency_seconds": round(latency, 3),
        "request_id": req_id,
        "estimated_tokens": est_tokens,
        "response_preview": text[:60].replace("\n", " ") if text else "",
    }


async def run_priority_evaluation(
    gateway_url: str = "http://localhost:8081",
    api_key: str = "cinch-prod-key",
    output_path: str = "benchmarks/results/priority_queue_eval.json",
) -> Dict[str, Any]:
    """Dispatch mixed batch traffic and demonstrate priority preemption."""
    print(f"=== Starting Priority Queue Live Evaluation on {gateway_url} ===")

    # 1. Health check
    async with httpx.AsyncClient() as client:
        health_resp = await client.get(f"{gateway_url.rstrip('/')}/health")
        print(f"Gateway Health Check: Status {health_resp.status_code}")
        print(f"Queue Status: {health_resp.json().get('queue')}\n")

    # 2. Dispatch mixed burst:
    # 4 Low priority batch jobs submitted first
    # 2 High priority interactive requests submitted with 0.1s delay
    tasks = []
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=20, max_keepalive_connections=20)) as client:
        # Submit Low Priority Batch Requests
        for i in range(4):
            t = send_priority_request(
                client=client,
                gateway_url=gateway_url,
                api_key=api_key,
                req_index=i + 1,
                priority="low",
                prompt="Write a detailed history of distributed systems in three paragraphs.",
                max_tokens=64,
                send_time_offset=0.0,
            )
            tasks.append(t)

        # Submit High Priority VIP Requests slightly after (preempting low priority queue)
        for i in range(2):
            t = send_priority_request(
                client=client,
                gateway_url=gateway_url,
                api_key=api_key,
                req_index=10 + i + 1,
                priority="high",
                prompt="Quick answer: What is 42 * 2?",
                max_tokens=16,
                send_time_offset=0.1,
            )
            tasks.append(t)

        results = await asyncio.gather(*tasks)

    # 3. Analyze latency by priority
    high_latencies = [r["latency_seconds"] for r in results if r["priority"] == "high" and r["status_code"] == 200]
    low_latencies = [r["latency_seconds"] for r in results if r["priority"] == "low" and r["status_code"] == 200]

    avg_high = sum(high_latencies) / len(high_latencies) if high_latencies else 0.0
    avg_low = sum(low_latencies) / len(low_latencies) if low_latencies else 0.0

    print("\n=== Live Priority Dispatch Results ===")
    for r in sorted(results, key=lambda x: x["latency_seconds"]):
        print(
            f"[{r['priority'].upper():4s}] Req #{r['req_index']:2d} | ID: {r['request_id']} | "
            f"Latency: {r['latency_seconds']:5.2f}s | Est Tokens: {r['estimated_tokens']:3s} | Preview: {r['response_preview']}"
        )

    print(f"\nAverage High-Priority Latency: {avg_high:.3f}s")
    print(f"Average Low-Priority Latency:  {avg_low:.3f}s")

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_url": gateway_url,
        "total_requests": len(results),
        "avg_high_priority_latency_seconds": round(avg_high, 3),
        "avg_low_priority_latency_seconds": round(avg_low, 3),
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[SAVED] Results exported to {output_path}")
    return payload


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Live Priority Queue Evaluation")
    parser.add_argument("--gateway-url", type=str, default="http://localhost:8081", help="Gateway URL")
    parser.add_argument("--api-key", type=str, default="cinch-prod-key", help="Gateway API Key")
    parser.add_argument("--output", type=str, default="benchmarks/results/priority_queue_eval.json", help="Output path")
    args = parser.parse_args()

    asyncio.run(
        run_priority_evaluation(
            gateway_url=args.gateway_url,
            api_key=args.api_key,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
