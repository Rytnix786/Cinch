"""Asynchronous load generator and benchmark execution runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional
import httpx

from benchmarks.metrics import (
    ConcurrencyMetrics,
    RequestRecord,
    compute_concurrency_metrics,
)
from benchmarks.vram_sampler import VRAMSampler


def load_prompts(prompts_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load benchmark prompt dataset from JSON file."""
    if prompts_path is None:
        prompts_path = str(pathlib.Path(__file__).parent / "prompts.json")
    with open(prompts_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def send_single_request(
    client: httpx.AsyncClient,
    target_url: str,
    model_name: str,
    prompt_item: Dict[str, Any],
    concurrency: int,
    api_key: Optional[str] = None,
    stream: bool = False,
    timeout_seconds: float = 60.0,
) -> RequestRecord:
    """Execute a single inference request and capture detailed telemetry."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    endpoint = f"{target_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_item["prompt"]}],
        "max_tokens": prompt_item.get("target_max_tokens", 128),
        "temperature": 0.0,
        "stream": stream,
    }

    start_time = time.perf_counter()
    ttft: Optional[float] = None

    try:
        if stream:
            prompt_tokens = 0
            completion_tokens = 0
            first_chunk_received = False

            async with client.stream("POST", endpoint, json=payload, headers=headers, timeout=timeout_seconds) as resp:
                status_code = resp.status_code
                if status_code != 200:
                    raw_err = await resp.aread()
                    elapsed = time.perf_counter() - start_time
                    return RequestRecord(
                        prompt_id=prompt_item.get("id", "unknown"),
                        concurrency=concurrency,
                        latency_seconds=elapsed,
                        status_code=status_code,
                        is_success=False,
                        error_message=raw_err.decode("utf-8", errors="ignore"),
                    )

                async for line in resp.aiter_lines():
                    if not first_chunk_received and line.startswith("data: ") and not line.endswith("[DONE]"):
                        ttft = time.perf_counter() - start_time
                        first_chunk_received = True
                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                        completion_tokens += 1

            elapsed = time.perf_counter() - start_time
            # Rough prompt token estimate if not returned in SSE stream
            prompt_tokens = len(prompt_item["prompt"].split()) * 2
            return RequestRecord(
                prompt_id=prompt_item.get("id", "unknown"),
                concurrency=concurrency,
                latency_seconds=elapsed,
                ttft_seconds=ttft,
                prompt_tokens=prompt_tokens,
                completion_tokens=max(1, completion_tokens),
                status_code=200,
                is_success=True,
            )

        else:
            resp = await client.post(endpoint, json=payload, headers=headers, timeout=timeout_seconds)
            elapsed = time.perf_counter() - start_time
            status_code = resp.status_code

            if status_code != 200:
                return RequestRecord(
                    prompt_id=prompt_item.get("id", "unknown"),
                    concurrency=concurrency,
                    latency_seconds=elapsed,
                    status_code=status_code,
                    is_success=False,
                    error_message=resp.text,
                )

            data = resp.json()
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", len(prompt_item["prompt"].split()) * 2)
            completion_tokens = usage.get("completion_tokens", len(data.get("choices", [{}])[0].get("message", {}).get("content", "").split()))

            return RequestRecord(
                prompt_id=prompt_item.get("id", "unknown"),
                concurrency=concurrency,
                latency_seconds=elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=max(1, completion_tokens),
                status_code=200,
                is_success=True,
            )

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        return RequestRecord(
            prompt_id=prompt_item.get("id", "unknown"),
            concurrency=concurrency,
            latency_seconds=elapsed,
            status_code=0,
            is_success=False,
            error_message=str(exc),
        )


async def run_concurrency_tier(
    target_url: str,
    model_name: str,
    prompts: List[Dict[str, Any]],
    concurrency: int,
    total_requests: int,
    api_key: Optional[str] = None,
    stream: bool = False,
) -> ConcurrencyMetrics:
    """Execute load test for a single concurrency level using an asyncio worker pool."""
    sampler = VRAMSampler(sample_interval_seconds=0.1)
    sampler.start()

    records: List[RequestRecord] = []
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    # Populate request queue round-robin across prompts
    for i in range(total_requests):
        prompt = prompts[i % len(prompts)]
        queue.put_nowait(prompt)

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)) as client:
        async def worker() -> None:
            while not queue.empty():
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                record = await send_single_request(
                    client=client,
                    target_url=target_url,
                    model_name=model_name,
                    prompt_item=item,
                    concurrency=concurrency,
                    api_key=api_key,
                    stream=stream,
                )
                records.append(record)
                queue.task_done()

        start_time = time.perf_counter()
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*workers)
        total_duration = time.perf_counter() - start_time

    sampler.stop()
    peak_vram = sampler.get_peak_mib()
    avg_vram = sampler.get_average_mib()

    return compute_concurrency_metrics(
        records=records,
        concurrency=concurrency,
        total_duration_seconds=total_duration,
        peak_vram_mib=peak_vram,
        avg_vram_mib=avg_vram,
    )


async def run_benchmark_suite(
    target_url: str = "http://localhost:8000",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
    concurrency_levels: Optional[List[int]] = None,
    requests_per_tier: int = 16,
    api_key: Optional[str] = None,
    stream: bool = False,
    prompts_path: Optional[str] = None,
    output_path: Optional[str] = None,
    engine_label: str = "vLLM-AWQ",
) -> Dict[str, Any]:
    """Execute the full benchmark suite across all requested concurrency levels."""
    if concurrency_levels is None:
        concurrency_levels = [1, 4, 8, 16]

    prompts = load_prompts(prompts_path)
    print(f"=== Starting Benchmark Suite against {target_url} ({engine_label}) ===")
    print(f"Model: {model_name} | Concurrencies: {concurrency_levels} | Requests/tier: {requests_per_tier}\n")

    # Warmup
    print("Sending warmup request...")
    async with httpx.AsyncClient() as client:
        await send_single_request(
            client=client,
            target_url=target_url,
            model_name=model_name,
            prompt_item=prompts[0],
            concurrency=1,
            api_key=api_key,
            stream=stream,
        )
    print("Warmup complete. Starting benchmark sweeps...\n")

    tier_metrics: List[ConcurrencyMetrics] = []
    for c in concurrency_levels:
        print(f"--> Running Concurrency = {c} ({requests_per_tier} requests)...", end="", flush=True)
        metric = await run_concurrency_tier(
            target_url=target_url,
            model_name=model_name,
            prompts=prompts,
            concurrency=c,
            total_requests=requests_per_tier,
            api_key=api_key,
            stream=stream,
        )
        tier_metrics.append(metric)
        print(
            f" Done. Throughput: {metric.tokens_per_second:.1f} tok/s | "
            f"p50: {metric.latency_p50:.3f}s | p95: {metric.latency_p95:.3f}s | "
            f"VRAM Peak: {metric.peak_vram_mib or 0:.0f} MiB"
        )

    # Compile structured payload
    payload = {
        "engine": engine_label,
        "model": model_name,
        "target_url": target_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tiers": [m.to_dict() for m in tier_metrics],
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[SAVED] Benchmark results saved to {output_path}")

    return payload


def main() -> None:
    """CLI entrypoint for running benchmark suites."""
    parser = argparse.ArgumentParser(description="Cinch LLM Inference Benchmark Suite")
    parser.add_argument("--target-url", type=str, default="http://localhost:8000", help="Target API base URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ", help="Model name")
    parser.add_argument("--concurrency", type=str, default="1,4,8,16", help="Comma-separated concurrency levels")
    parser.add_argument("--requests", type=int, default=16, help="Requests per concurrency tier")
    parser.add_argument("--api-key", type=str, default=None, help="API key if required")
    parser.add_argument("--stream", action="store_true", help="Use streaming SSE requests")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--engine", type=str, default="vLLM-AWQ", help="Label for serving engine")
    args = parser.parse_args()

    levels = [int(c.strip()) for c in args.concurrency.split(",") if c.strip()]
    asyncio.run(
        run_benchmark_suite(
            target_url=args.target_url,
            model_name=args.model,
            concurrency_levels=levels,
            requests_per_tier=args.requests,
            api_key=args.api_key,
            stream=args.stream,
            output_path=args.output,
            engine_label=args.engine,
        )
    )


if __name__ == "__main__":
    main()
