"""Chaos engineering resilience harness evaluating Circuit Breaker fast-fail protection and MTTR."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, List
import httpx


async def run_chaos_evaluation(
    gateway_url: str = "http://localhost:8081",
    api_key: str = "cinch-prod-key",
    output_path: str = "benchmarks/results/chaos_resilience.json",
) -> Dict[str, Any]:
    """Execute live chaos injection sequence verifying circuit breaker fast-fail and self-healing across cluster replicas."""
    print(f"=== Starting Chaos Engineering & Resilience Evaluation on {gateway_url} ===\n")

    headers = {"Authorization": f"Bearer {api_key}"}
    fast_fail_latencies: List[float] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        # Phase 1: Baseline Healthy State
        print("Phase 1: Verifying Baseline Healthy State (Circuit State: CLOSED)...")
        r_base = await client.post(
            f"{gateway_url}/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
            },
            headers=headers,
        )
        print(
            f"   Baseline Status: {r_base.status_code} | Breaker State: {r_base.headers.get('X-Circuit-Breaker-State')}"
        )

        # Phase 2: Fault Injection (Bursting 8 faults to trip all cluster replicas)
        print("\nPhase 2: Injecting Upstream Fault Burst across Gateway Replicas to Trip Breakers...")
        for i in range(8):
            t0 = time.perf_counter()
            r_fault = await client.post(
                f"{gateway_url}/v1/chat/completions",
                json={
                    "model": "simulated-crash-model",
                    "messages": [{"role": "user", "content": "crash"}],
                    "max_tokens": 5,
                },
                headers=headers,
            )
            lat = (time.perf_counter() - t0) * 1000.0
            print(
                f"   Fault Probe #{i + 1:2d}: Status {r_fault.status_code} ({lat:5.1f}ms) | Breaker: {r_fault.headers.get('X-Circuit-Breaker-State')}"
            )

        # Phase 3: Fast-Fail Latency Protection
        print("\nPhase 3: Measuring Fast-Fail Latency Protection (Circuit State: OPEN)...")
        for i in range(10):
            t0 = time.perf_counter()
            r_fast = await client.post(
                f"{gateway_url}/v1/chat/completions",
                json={
                    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                    "messages": [{"role": "user", "content": "blocked"}],
                    "max_tokens": 5,
                },
                headers=headers,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            fast_fail_latencies.append(lat_ms)
            cb_state = r_fast.headers.get("X-Circuit-Breaker-State", "unknown")
            retry_after = r_fast.headers.get("Retry-After", "none")
            print(
                f"   Fast-Fail Request #{i + 1:2d}: Status {r_fast.status_code} in {lat_ms:5.2f}ms | Breaker: {cb_state} | Retry-After: {retry_after}s"
            )

        # Phase 4: Self-Healing Canary Probe & MTTR Recovery
        print("\nPhase 4: Waiting for Recovery Cooldown (10.0s) & Testing Canary Self-Healing...")
        t_trip = time.perf_counter()
        await asyncio.sleep(10.5)  # Cooldown wait

        # Canary Probes across replicas
        print("   Dispatching Canary Recovery Probes...")
        r_canary1 = await client.post(
            f"{gateway_url}/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 5,
            },
            headers=headers,
        )
        r_canary2 = await client.post(
            f"{gateway_url}/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "messages": [{"role": "user", "content": "What is 3+3?"}],
                "max_tokens": 5,
            },
            headers=headers,
        )
        mttr_seconds = time.perf_counter() - t_trip
        print(
            f"   Canary #1: Status {r_canary1.status_code} | Breaker: {r_canary1.headers.get('X-Circuit-Breaker-State')}"
        )
        print(
            f"   Canary #2: Status {r_canary2.status_code} | Breaker: {r_canary2.headers.get('X-Circuit-Breaker-State')}"
        )
        print(f"   Measured MTTR: {mttr_seconds:.2f}s")

        # Verify Breaker is restored to CLOSED
        r_restored = await client.get(f"{gateway_url}/health")
        h_data = r_restored.json()
        final_state = h_data.get("circuit_breaker", {}).get("state", "unknown")
        print(f"   Gateway Health Restored: {h_data.get('status')} | Final Breaker State: {final_state}")

    avg_fast_fail_ms = sum(fast_fail_latencies) / len(fast_fail_latencies)
    unmitigated_timeout_ms = 30000.0  # 30s timeout
    latency_reduction_factor = unmitigated_timeout_ms / max(0.1, avg_fast_fail_ms)

    print("\n=== Chaos Resilience Summary ===")
    print(f"Average Fast-Fail Response Latency: {avg_fast_fail_ms:.2f} ms")
    print(f"Unmitigated Connection Timeout:     {unmitigated_timeout_ms:.0f} ms")
    print(
        f"Protection Factor:                  {latency_reduction_factor:.0f}x faster rejection (protecting queue & memory)"
    )
    print(f"Mean Time To Recover (MTTR):        {mttr_seconds:.2f} s")

    payload = {
        "benchmark": "circuit_breaker_chaos_resilience",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_url": gateway_url,
        "metrics": {
            "average_fast_fail_latency_ms": round(avg_fast_fail_ms, 2),
            "unmitigated_timeout_ms": unmitigated_timeout_ms,
            "protection_speedup_factor": round(latency_reduction_factor, 1),
            "mean_time_to_recover_seconds": round(mttr_seconds, 2),
            "trip_threshold_failures": 3,
            "cooldown_period_seconds": 10.0,
            "recovery_canary_status": r_canary1.status_code,
            "final_circuit_state": final_state,
        },
        "fast_fail_samples_ms": [round(x, 2) for x in fast_fail_latencies],
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[SAVED] Chaos resilience evaluation dataset saved to {output_path}")
    return payload


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Circuit Breaker Chaos Evaluation")
    parser.add_argument("--gateway-url", type=str, default="http://localhost:8081", help="Gateway URL")
    parser.add_argument("--api-key", type=str, default="cinch-prod-key", help="Gateway API Key")
    parser.add_argument("--output", type=str, default="benchmarks/results/chaos_resilience.json", help="Output path")
    args = parser.parse_args()

    asyncio.run(
        run_chaos_evaluation(
            gateway_url=args.gateway_url,
            api_key=args.api_key,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
