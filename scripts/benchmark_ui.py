"""Live Empirical Benchmark for Interactive Real-Time Serving Console (M25)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, List
import httpx


async def run_benchmark(
    gateway_url: str,
    api_key: str,
    output_path: str,
) -> Dict[str, Any]:
    print("=" * 70)
    print("  CINCH REAL-TIME SERVING CONSOLE BENCHMARK - M25")
    print("=" * 70)

    base_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    asset_endpoints = [
        {"name": "UI HTML Shell", "path": "/ui/", "expect_code": 200},
        {"name": "UI Style CSS", "path": "/ui/style.css", "expect_code": 200},
        {"name": "UI Engine JS", "path": "/ui/app.js", "expect_code": 200},
        {"name": "Console Redirect", "path": "/console", "expect_code": 200, "follow_redirects": True},
        {"name": "Console State API", "path": "/v1/console/state", "expect_code": 200, "auth": True},
    ]

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
        print("\n[Step 1] Benchmarking Static Assets & Telemetry API Latencies...")
        asset_results: List[Dict[str, Any]] = []

        for asset in asset_endpoints:
            headers = base_headers if asset.get("auth") else {}
            follow = asset.get("follow_redirects", False)

            latencies: List[float] = []
            status_ok = True
            for _ in range(5):
                t0 = time.perf_counter()
                resp = await client.get(f"{gateway_url}{asset['path']}", headers=headers, follow_redirects=follow)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(lat_ms)
                if resp.status_code != asset["expect_code"]:
                    status_ok = False

            avg_lat = sum(latencies) / len(latencies)
            min_lat = min(latencies)

            print(
                f"  {asset['name']:<24} | Avg: {avg_lat:5.2f}ms | Min: {min_lat:5.2f}ms | "
                f"Status: {resp.status_code} [{'PASS' if status_ok else 'FAIL'}]"
            )

            asset_results.append({
                "name": asset["name"],
                "path": asset["path"],
                "status_code": resp.status_code,
                "avg_latency_ms": round(avg_lat, 2),
                "min_latency_ms": round(min_lat, 2),
                "passed": status_ok,
            })

        print("\n[Step 2] Benchmarking Live SSE Stream Rendering via Console Pipeline...")
        t_req_start = time.perf_counter()
        stream_chunks: List[float] = []
        first_token_time: float = 0.0
        total_tokens = 0

        stream_payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "messages": [
                {"role": "system", "content": "You are a concise technical assistant."},
                {"role": "user", "content": "Explain prefix caching in two sentences."},
            ],
            "max_tokens": 40,
            "stream": True,
        }

        async with client.stream(
            "POST",
            f"{gateway_url}/v1/chat/completions",
            json=stream_payload,
            headers=base_headers,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    t_chunk = time.perf_counter()
                    if not first_token_time:
                        first_token_time = (t_chunk - t_req_start) * 1000.0
                    stream_chunks.append(t_chunk)
                    total_tokens += 1

        total_stream_lat = (time.perf_counter() - t_req_start) * 1000.0
        gen_duration_sec = max((total_stream_lat - first_token_time) / 1000.0, 0.001)
        tps = total_tokens / gen_duration_sec

        print(f"  TTFT (Time-To-First-Token):   {first_token_time:6.1f} ms")
        print(f"  Total Stream Duration:       {total_stream_lat:6.1f} ms")
        print(f"  Tokens Streamed:             {total_tokens} tokens")
        print(f"  Streaming Generation Speed:  {tps:6.1f} tok/s")

    all_passed = all(a["passed"] for a in asset_results)
    avg_asset_lat = sum(a["avg_latency_ms"] for a in asset_results) / len(asset_results)

    print("\n" + "=" * 70)
    print("  SERVING CONSOLE BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Static Asset Conformance:      {'ALL PASS' if all_passed else 'FAIL'}")
    print(f"  Average Asset Delivery Lat:    {avg_asset_lat:.2f} ms (< 5ms SLA)")
    print(f"  Streaming TTFT:                {first_token_time:.1f} ms")
    print(f"  Streaming Generation Rate:     {tps:.1f} tokens/sec")
    print("=" * 70)

    payload = {
        "benchmark": "interactive_serving_console_m25",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_url": gateway_url,
        "metrics": {
            "all_assets_passed": all_passed,
            "average_asset_latency_ms": round(avg_asset_lat, 2),
            "streaming_ttft_ms": round(first_token_time, 2),
            "total_stream_duration_ms": round(total_stream_lat, 2),
            "tokens_streamed": total_tokens,
            "tokens_per_second": round(tps, 1),
        },
        "asset_benchmarks": asset_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[SAVED] Serving console evaluation dataset -> {output_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Serving Console Benchmark M25")
    parser.add_argument("--gateway-url", default="http://localhost:8081")
    parser.add_argument("--api-key", default="cinch-prod-key")
    parser.add_argument("--output", default="benchmarks/results/ui_console_eval.json")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.gateway_url, args.api_key, args.output))


if __name__ == "__main__":
    main()
