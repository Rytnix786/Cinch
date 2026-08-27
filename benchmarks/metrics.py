"""Metrics calculation and aggregation models for Cinch benchmarks."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class RequestRecord:
    """Telemetry captured from a single benchmark request."""

    prompt_id: str
    concurrency: int
    latency_seconds: float
    ttft_seconds: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    status_code: int = 200
    is_success: bool = True
    error_message: Optional[str] = None


@dataclasses.dataclass
class ConcurrencyMetrics:
    """Aggregated performance metrics for a specific concurrency level."""

    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_duration_seconds: float
    tokens_per_second: float
    requests_per_second: float
    latency_p50: float
    latency_p90: float
    latency_p95: float
    latency_p99: float
    latency_mean: float
    latency_min: float
    latency_max: float
    ttft_p50: Optional[float] = None
    ttft_p95: Optional[float] = None
    peak_vram_mib: Optional[float] = None
    avg_vram_mib: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a JSON-serializable dictionary."""
        return dataclasses.asdict(self)


def calculate_percentile(sorted_values: List[float], percentile: float) -> float:
    """Compute percentile from sorted float array using nearest-rank / linear interpolation.

    Percentile argument in range [0.0, 100.0].
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    k = (len(sorted_values) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


def compute_concurrency_metrics(
    records: List[RequestRecord],
    concurrency: int,
    total_duration_seconds: float,
    peak_vram_mib: Optional[float] = None,
    avg_vram_mib: Optional[float] = None,
) -> ConcurrencyMetrics:
    """Aggregate a list of RequestRecords into a ConcurrencyMetrics summary."""
    if not records:
        return ConcurrencyMetrics(
            concurrency=concurrency,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_duration_seconds=total_duration_seconds,
            tokens_per_second=0.0,
            requests_per_second=0.0,
            latency_p50=0.0,
            latency_p90=0.0,
            latency_p95=0.0,
            latency_p99=0.0,
            latency_mean=0.0,
            latency_min=0.0,
            latency_max=0.0,
            peak_vram_mib=peak_vram_mib,
            avg_vram_mib=avg_vram_mib,
        )

    successful = [r for r in records if r.is_success]
    failed = [r for r in records if not r.is_success]

    total_prompt_tokens = sum(r.prompt_tokens for r in successful)
    total_completion_tokens = sum(r.completion_tokens for r in successful)

    effective_duration = max(0.001, total_duration_seconds)
    tokens_per_second = total_completion_tokens / effective_duration
    requests_per_second = len(successful) / effective_duration

    latencies = sorted([r.latency_seconds for r in successful])
    ttfts = sorted([r.ttft_seconds for r in successful if r.ttft_seconds is not None])

    if latencies:
        p50 = calculate_percentile(latencies, 50.0)
        p90 = calculate_percentile(latencies, 90.0)
        p95 = calculate_percentile(latencies, 95.0)
        p99 = calculate_percentile(latencies, 99.0)
        mean = sum(latencies) / len(latencies)
        min_lat = latencies[0]
        max_lat = latencies[-1]
    else:
        p50 = p90 = p95 = p99 = mean = min_lat = max_lat = 0.0

    ttft_p50 = calculate_percentile(ttfts, 50.0) if ttfts else None
    ttft_p95 = calculate_percentile(ttfts, 95.0) if ttfts else None

    return ConcurrencyMetrics(
        concurrency=concurrency,
        total_requests=len(records),
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_duration_seconds=round(total_duration_seconds, 4),
        tokens_per_second=round(tokens_per_second, 2),
        requests_per_second=round(requests_per_second, 2),
        latency_p50=round(p50, 4),
        latency_p90=round(p90, 4),
        latency_p95=round(p95, 4),
        latency_p99=round(p99, 4),
        latency_mean=round(mean, 4),
        latency_min=round(min_lat, 4),
        latency_max=round(max_lat, 4),
        ttft_p50=round(ttft_p50, 4) if ttft_p50 is not None else None,
        ttft_p95=round(ttft_p95, 4) if ttft_p95 is not None else None,
        peak_vram_mib=round(peak_vram_mib, 2) if peak_vram_mib is not None else None,
        avg_vram_mib=round(avg_vram_mib, 2) if avg_vram_mib is not None else None,
    )
