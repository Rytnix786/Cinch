"""Load generation and time-series scaling telemetry collector for Kubernetes HPA."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional
import httpx


def query_k8s_hpa_status() -> Dict[str, Any]:
    """Query current HPA and Deployment status via kubectl."""
    status_info: Dict[str, Any] = {
        "replicas": 2,
        "ready_replicas": 2,
        "current_cpu_percent": 0,
        "desired_replicas": 2,
        "pod_metrics": [],
    }
    try:
        # 1. Query Deployment
        dep_out = subprocess.check_output(
            ["kubectl", "get", "deployment", "cinch-gateway", "-n", "cinch", "-o", "json"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        dep_data = json.loads(dep_out)
        status_info["replicas"] = dep_data.get("status", {}).get("replicas", 2)
        status_info["ready_replicas"] = dep_data.get("status", {}).get("readyReplicas", 2)

        # 2. Query HPA
        hpa_out = subprocess.check_output(
            ["kubectl", "get", "hpa", "cinch-gateway-hpa", "-n", "cinch", "-o", "json"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        hpa_data = json.loads(hpa_out)
        hpa_status = hpa_data.get("status", {})
        status_info["desired_replicas"] = hpa_status.get("desiredReplicas", 2)
        current_metrics = hpa_status.get("currentMetrics", [])
        for m in current_metrics:
            if m.get("type") == "Resource" and m.get("resource", {}).get("name") == "cpu":
                status_info["current_cpu_percent"] = m.get("resource", {}).get("current", {}).get("averageUtilization", 0)

        # 3. Query pod top metrics
        top_out = subprocess.check_output(
            ["kubectl", "top", "pods", "-n", "cinch", "--no-headers"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        pod_list = []
        for line in top_out.strip().splitlines():
            parts = line.split()
            if len(parts) >= 3 and "cinch-gateway" in parts[0]:
                pod_list.append({"pod": parts[0], "cpu": parts[1], "memory": parts[2]})
        status_info["pod_metrics"] = pod_list

    except Exception:
        pass

    return status_info


async def traffic_worker(
    client: httpx.AsyncClient,
    target_url: str,
    api_key: Optional[str],
    stop_event: asyncio.Event,
    request_counter: List[int],
) -> None:
    """Worker task sending requests to drive gateway CPU."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    endpoint = f"{target_url.rstrip('/')}/v1/models"
    endpoint_chat = f"{target_url.rstrip('/')}/v1/chat/completions"

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [{"role": "user", "content": "What is horizontal pod autoscaling?"}],
        "max_tokens": 32,
    }

    while not stop_event.is_set():
        try:
            # Alternate between fast models endpoint and chat completions for CPU load
            if request_counter[0] % 3 == 0:
                await client.post(endpoint_chat, json=payload, headers=headers, timeout=5.0)
            else:
                await client.get(endpoint, headers=headers, timeout=5.0)
            request_counter[0] += 1
        except Exception:
            await asyncio.sleep(0.05)


async def run_hpa_load_test(
    gateway_url: str = "http://localhost:8081",
    api_key: Optional[str] = "cinch-prod-key",
    concurrency: int = 24,
    load_duration_seconds: int = 60,
    cooldown_seconds: int = 75,
    sample_interval_seconds: float = 2.0,
    output_path: str = "benchmarks/results/hpa_scaling.json",
) -> Dict[str, Any]:
    """Execute sustained load test, monitor scaling transitions, and export time-series."""
    print(f"=== Starting HPA Scaling Load Test on {gateway_url} ===")
    print(f"Target Concurrency: {concurrency} workers | Load Duration: {load_duration_seconds}s | Cooldown: {cooldown_seconds}s\n")

    timeline: List[Dict[str, Any]] = []
    stop_traffic = asyncio.Event()
    request_counter = [0]
    start_time = time.perf_counter()

    # Launch load generator
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)) as client:
        workers = [
            asyncio.create_task(traffic_worker(client, gateway_url, api_key, stop_traffic, request_counter))
            for _ in range(concurrency)
        ]

        total_test_duration = load_duration_seconds + cooldown_seconds
        phase = "LOAD_INJECTION"

        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= load_duration_seconds and not stop_traffic.is_set():
                print("\n[PHASE CHANGE] Stopping traffic injection. Entering COOLDOWN phase...")
                stop_traffic.set()
                phase = "COOLDOWN"

            if elapsed >= total_test_duration:
                break

            k8s_state = query_k8s_hpa_status()
            current_rps = round(request_counter[0] / max(0.1, elapsed), 1)

            entry = {
                "elapsed_seconds": round(elapsed, 1),
                "phase": phase,
                "replicas": k8s_state["replicas"],
                "ready_replicas": k8s_state["ready_replicas"],
                "desired_replicas": k8s_state["desired_replicas"],
                "cpu_utilization_percent": k8s_state["current_cpu_percent"],
                "total_requests": request_counter[0],
                "cumulative_rps": current_rps,
                "pod_count": len(k8s_state["pod_metrics"]),
            }
            timeline.append(entry)

            print(
                f"[{elapsed:5.1f}s | {phase:<14}] Replicas: {entry['replicas']} (Ready: {entry['ready_replicas']}, Desired: {entry['desired_replicas']}) | "
                f"CPU: {entry['cpu_utilization_percent']:3.0f}% / 50% | Requests: {entry['total_requests']:4d}"
            )
            await asyncio.sleep(sample_interval_seconds)

        stop_traffic.set()
        await asyncio.gather(*workers, return_exceptions=True)

    summary_payload = {
        "test": "k8s_hpa_gateway_scaling",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_url": gateway_url,
        "concurrency": concurrency,
        "min_replicas": 2,
        "max_replicas": 6,
        "initial_replicas": timeline[0]["replicas"] if timeline else 2,
        "peak_replicas": max(t["replicas"] for t in timeline) if timeline else 2,
        "final_replicas": timeline[-1]["replicas"] if timeline else 2,
        "peak_cpu_percent": max(t["cpu_utilization_percent"] for t in timeline) if timeline else 0,
        "total_requests": request_counter[0],
        "timeline": timeline,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\n[SAVED] HPA scaling dataset saved to {output_path}")
    print(f"Summary: Replicas scaled from {summary_payload['initial_replicas']} -> {summary_payload['peak_replicas']} (Peak CPU: {summary_payload['peak_cpu_percent']}%)")

    return summary_payload


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Kubernetes Gateway HPA Load Test")
    parser.add_argument("--gateway-url", type=str, default="http://localhost:8081", help="Gateway URL")
    parser.add_argument("--api-key", type=str, default="cinch-prod-key", help="Gateway API Key")
    parser.add_argument("--concurrency", type=int, default=24, help="Concurrent workers")
    parser.add_argument("--duration", type=int, default=60, help="Load duration in seconds")
    parser.add_argument("--cooldown", type=int, default=75, help="Cooldown duration in seconds")
    parser.add_argument("--output", type=str, default="benchmarks/results/hpa_scaling.json", help="Output JSON path")
    args = parser.parse_args()

    asyncio.run(
        run_hpa_load_test(
            gateway_url=args.gateway_url,
            api_key=args.api_key,
            concurrency=args.concurrency,
            load_duration_seconds=args.duration,
            cooldown_seconds=args.cooldown,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
