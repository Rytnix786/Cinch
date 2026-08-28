"""Live verification script capturing Prometheus metrics and OTel trace headers from Cinch Gateway."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict
import httpx


async def test_observability_live(
    gateway_url: str = "http://localhost:8081",
    api_key: str = "cinch-prod-key",
    output_path: str = "benchmarks/results/observability_snapshot.json",
) -> Dict[str, Any]:
    """Send live test requests and query /metrics endpoint."""
    print(f"=== Testing Live Observability & Telemetry on {gateway_url} ===\n")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        # 1. Send High-Priority Request
        print("1. Sending High-Priority Interactive Request...")
        req1 = {
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "messages": [{"role": "user", "content": "Return 1 + 1 as integer."}],
            "max_tokens": 10,
            "priority": "high",
        }
        r1 = await client.post(f"{gateway_url}/v1/chat/completions", json=req1, headers=headers)
        print(f"   Response: {r1.status_code} | Traceparent Header: {r1.headers.get('traceparent')}")

        # 2. Send Streaming Request
        print("2. Sending Streaming Request with Shared Prefix...")
        req2 = {
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "messages": [
                {"role": "system", "content": "You are a concise mathematician assistant."},
                {"role": "user", "content": "Explain prime numbers in 1 sentence."},
            ],
            "max_tokens": 20,
            "stream": True,
        }
        async with client.stream("POST", f"{gateway_url}/v1/chat/completions", json=req2, headers=headers) as r2:
            async for _ in r2.aiter_lines():
                pass
        print("   Streaming completed.")

        # 3. Query /metrics endpoint (Prometheus format)
        print("3. Querying Prometheus /metrics endpoint...")
        m_resp = await client.get(f"{gateway_url}/metrics")
        prom_text = m_resp.text
        print("\n--- Raw Prometheus Output (Snippet) ---")
        lines = [line for line in prom_text.splitlines() if line and not line.startswith("#")]
        for line in lines[:12]:
            print(f"  {line}")

    snapshot = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_gateway": gateway_url,
        "prometheus_metrics_raw": prom_text,
        "parsed_active_metrics": lines,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"\n[SAVED] Prometheus telemetry snapshot saved to {output_path}")
    return snapshot


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Live Observability Test")
    parser.add_argument("--gateway-url", type=str, default="http://localhost:8081", help="Gateway URL")
    parser.add_argument("--api-key", type=str, default="cinch-prod-key", help="Gateway API Key")
    parser.add_argument(
        "--output", type=str, default="benchmarks/results/observability_snapshot.json", help="Output path"
    )
    args = parser.parse_args()

    asyncio.run(
        test_observability_live(
            gateway_url=args.gateway_url,
            api_key=args.api_key,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
