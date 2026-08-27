"""Automated unit test suite for Cinch Benchmark Harness."""

from __future__ import annotations

import json
import pathlib
import tempfile
from typing import Any, AsyncGenerator
from unittest.mock import patch
import pytest
import httpx

from benchmarks.metrics import (
    RequestRecord,
    calculate_percentile,
    compute_concurrency_metrics,
)
from benchmarks.runner import (
    load_prompts,
    run_benchmark_suite,
    run_concurrency_tier,
    send_single_request,
)
from benchmarks.vram_sampler import VRAMSampler


def test_calculate_percentile_empty() -> None:
    """Verify calculate_percentile returns 0.0 for empty input."""
    assert calculate_percentile([], 50.0) == 0.0


def test_calculate_percentile_single() -> None:
    """Verify calculate_percentile returns element for single-item list."""
    assert calculate_percentile([42.0], 95.0) == 42.0


def test_calculate_percentile_interpolation() -> None:
    """Verify linear interpolation matches exact mathematical percentiles."""
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    # Min, median, max
    assert calculate_percentile(values, 0.0) == 10.0
    assert calculate_percentile(values, 50.0) == 30.0
    assert calculate_percentile(values, 100.0) == 50.0
    # Interpolated p25 = 20.0, p75 = 40.0
    assert calculate_percentile(values, 25.0) == 20.0
    assert calculate_percentile(values, 75.0) == 40.0


def test_compute_concurrency_metrics_empty() -> None:
    """Verify metrics computation on empty record set."""
    metrics = compute_concurrency_metrics(records=[], concurrency=4, total_duration_seconds=10.0)
    assert metrics.total_requests == 0
    assert metrics.successful_requests == 0
    assert metrics.tokens_per_second == 0.0
    assert metrics.latency_p50 == 0.0


def test_compute_concurrency_metrics_aggregation() -> None:
    """Verify aggregation of mixed successful and failed records."""
    records = [
        RequestRecord("p1", concurrency=2, latency_seconds=1.0, ttft_seconds=0.2, prompt_tokens=10, completion_tokens=20, is_success=True),
        RequestRecord("p2", concurrency=2, latency_seconds=2.0, ttft_seconds=0.4, prompt_tokens=15, completion_tokens=30, is_success=True),
        RequestRecord("p3", concurrency=2, latency_seconds=3.0, ttft_seconds=0.6, prompt_tokens=20, completion_tokens=50, is_success=True),
        RequestRecord("p4", concurrency=2, latency_seconds=0.5, status_code=500, is_success=False, error_message="Internal Error"),
    ]

    metrics = compute_concurrency_metrics(
        records=records,
        concurrency=2,
        total_duration_seconds=5.0,
        peak_vram_mib=7200.0,
        avg_vram_mib=7100.0,
    )

    assert metrics.total_requests == 4
    assert metrics.successful_requests == 3
    assert metrics.failed_requests == 1
    assert metrics.total_prompt_tokens == 45
    assert metrics.total_completion_tokens == 100
    assert metrics.tokens_per_second == 20.0  # 100 / 5.0
    assert metrics.requests_per_second == 0.6  # 3 / 5.0
    assert metrics.latency_p50 == 2.0
    assert metrics.latency_min == 1.0
    assert metrics.latency_max == 3.0
    assert metrics.ttft_p50 == 0.4
    assert metrics.peak_vram_mib == 7200.0


def test_vram_sampler_mock() -> None:
    """Verify VRAM sampler background tracking."""
    sampler = VRAMSampler(sample_interval_seconds=0.01)
    with patch.object(sampler, "_query_vram", side_effect=[7000.0, 7500.0, 7200.0]):
        sampler.start()
        sampler.stop()
        # Mock manually samples
        sampler._samples = [(0.0, 7000.0), (0.1, 7500.0), (0.2, 7200.0)]
        assert sampler.get_peak_mib() == 7500.0
        assert pytest.approx(sampler.get_average_mib(), 0.1) == 7233.33
        assert len(sampler.get_samples()) == 3


def test_load_prompts() -> None:
    """Verify prompts.json loading."""
    prompts = load_prompts()
    assert len(prompts) >= 10
    for p in prompts:
        assert "id" in p
        assert "prompt" in p
        assert "target_max_tokens" in p


@pytest.mark.asyncio
async def test_send_single_request_non_streaming() -> None:
    """Verify single request execution against mock server."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Test output"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8},
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        prompt_item = {"id": "test-1", "prompt": "Hello world", "target_max_tokens": 64}
        record = await send_single_request(
            client=client,
            target_url="http://mock-server:8000",
            model_name="Qwen/Qwen2.5-7B-Instruct-AWQ",
            prompt_item=prompt_item,
            concurrency=1,
            stream=False,
        )
        assert record.is_success is True
        assert record.status_code == 200
        assert record.prompt_tokens == 12
        assert record.completion_tokens == 8
        assert record.latency_seconds > 0.0


@pytest.mark.asyncio
async def test_send_single_request_streaming_sse() -> None:
    """Verify single streaming request execution and TTFT detection."""
    def handler(request: httpx.Request) -> httpx.Response:
        async def sse_gen() -> AsyncGenerator[bytes, None]:
            yield b"data: {\"choices\": [{\"delta\": {\"content\": \"First\"}}]}\n\n"
            yield b"data: {\"choices\": [{\"delta\": {\"content\": \" second\"}}]}\n\n"
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, content=sse_gen(), headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        prompt_item = {"id": "test-stream", "prompt": "Streaming prompt", "target_max_tokens": 64}
        record = await send_single_request(
            client=client,
            target_url="http://mock-server:8000",
            model_name="Qwen/Qwen2.5-7B-Instruct-AWQ",
            prompt_item=prompt_item,
            concurrency=1,
            stream=True,
        )
        assert record.is_success is True
        assert record.ttft_seconds is not None
        assert record.completion_tokens >= 2


@pytest.mark.asyncio
async def test_send_single_request_failure() -> None:
    """Verify handling of server error response."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        prompt_item = {"id": "test-err", "prompt": "Error test"}
        record = await send_single_request(
            client=client,
            target_url="http://mock-server:8000",
            model_name="Qwen/Qwen2.5-7B-Instruct-AWQ",
            prompt_item=prompt_item,
            concurrency=1,
        )
        assert record.is_success is False
        assert record.status_code == 500
        assert "Internal Server Error" in str(record.error_message)


_real_async_client = httpx.AsyncClient


@pytest.mark.asyncio
async def test_run_concurrency_tier_mock() -> None:
    """Verify concurrency worker distribution."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 5}})

    transport = httpx.MockTransport(handler)

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return _real_async_client(transport=transport, **kwargs)

    with patch("benchmarks.runner.httpx.AsyncClient", side_effect=client_factory):
        prompts = [{"id": f"p-{i}", "prompt": f"Test {i}", "target_max_tokens": 32} for i in range(4)]
        metrics = await run_concurrency_tier(
            target_url="http://mock-server:8000",
            model_name="test-model",
            prompts=prompts,
            concurrency=2,
            total_requests=4,
        )
        assert metrics.total_requests == 4
        assert metrics.successful_requests == 4
        assert metrics.concurrency == 2
        assert metrics.tokens_per_second > 0.0


@pytest.mark.asyncio
async def test_run_benchmark_suite_mock() -> None:
    """Verify full benchmark suite flow and JSON file export."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "resp"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 5}})

    transport = httpx.MockTransport(handler)

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return _real_async_client(transport=transport, **kwargs)

    with patch("benchmarks.runner.httpx.AsyncClient", side_effect=client_factory):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = str(pathlib.Path(tmpdir) / "test_results.json")
            res = await run_benchmark_suite(
                target_url="http://mock-server:8000",
                model_name="test-model",
                concurrency_levels=[1, 2],
                requests_per_tier=2,
                output_path=out_file,
                engine_label="test-engine",
            )
            assert res["engine"] == "test-engine"
            assert len(res["tiers"]) == 2
            assert pathlib.Path(out_file).exists()

            with open(out_file, "r", encoding="utf-8") as f:
                saved_json = json.load(f)
                assert saved_json["engine"] == "test-engine"
                assert len(saved_json["tiers"]) == 2
