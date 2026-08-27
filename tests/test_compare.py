"""Unit tests for benchmark comparison generator."""

from __future__ import annotations

import tempfile
import pathlib
import json
from benchmarks.compare import (
    build_comparison_data,
    format_markdown_table,
    load_result_json,
)


def test_build_comparison_data_calculations() -> None:
    """Verify speedup and latency reduction mathematical accuracy."""
    baseline = {
        "engine": "HF-Naive",
        "tiers": [
            {"concurrency": 1, "tokens_per_second": 30.0, "latency_p50": 2.0, "latency_p95": 2.2, "peak_vram_mib": 7800.0},
            {"concurrency": 4, "tokens_per_second": 30.0, "latency_p50": 8.0, "latency_p95": 8.5, "peak_vram_mib": 7800.0},
        ],
    }
    optimized = {
        "engine": "vLLM-AWQ",
        "tiers": [
            {"concurrency": 1, "tokens_per_second": 45.0, "latency_p50": 2.0, "latency_p95": 5.0, "peak_vram_mib": 7875.0},
            {"concurrency": 4, "tokens_per_second": 180.0, "latency_p50": 2.0, "latency_p95": 5.5, "peak_vram_mib": 7875.0},
        ],
    }
    gateway = {
        "engine": "Gateway+vLLM",
        "tiers": [
            {"concurrency": 1, "tokens_per_second": 44.5, "latency_p50": 2.002, "latency_p95": 5.05},
            {"concurrency": 4, "tokens_per_second": 178.0, "latency_p50": 2.003, "latency_p95": 5.55},
        ],
    }

    rows = build_comparison_data(baseline, optimized, gateway)
    assert len(rows) == 2

    # Tier 1
    r1 = rows[0]
    assert r1["concurrency"] == 1
    assert r1["throughput_speedup"] == 1.5  # 45 / 30
    assert r1["p50_reduction"] == 1.0  # 2.0 / 2.0
    assert r1["gateway_overhead_ms"] == 2.0  # (2.002 - 2.0) * 1000

    # Tier 4
    r4 = rows[1]
    assert r4["concurrency"] == 4
    assert r4["throughput_speedup"] == 6.0  # 180 / 30
    assert r4["p50_reduction"] == 4.0  # 8.0 / 2.0


def test_format_markdown_table() -> None:
    """Verify markdown table formatting."""
    rows = [
        {
            "concurrency": 1,
            "baseline_tps": 30.4,
            "optimized_tps": 46.7,
            "throughput_speedup": 1.54,
            "baseline_p50": 2.105,
            "optimized_p50": 2.231,
            "p50_reduction": 0.94,
            "baseline_p95": 2.110,
            "optimized_p95": 5.515,
            "p95_reduction": 0.38,
            "optimized_vram_peak": 7875.0,
        }
    ]
    md = format_markdown_table(rows)
    assert "| Concurrency |" in md
    assert "46.7" in md
    assert "1.54x" in md


def test_load_result_json() -> None:
    """Verify JSON file loader."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = str(pathlib.Path(tmpdir) / "test.json")
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump({"key": "value"}, f)

        data = load_result_json(fpath)
        assert data == {"key": "value"}
