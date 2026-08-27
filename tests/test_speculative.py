"""Unit test suite for Speculative Decoding benchmark metrics and evaluation logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from benchmarks.speculative import (
    SPECULATIVE_EVAL_PROMPTS,
    calculate_speculative_metrics,
    evaluate_single_prompt,
    run_speculative_benchmark,
)


def test_calculate_speculative_metrics() -> None:
    """Verify speculative speedup and TPOT calculations."""
    m = calculate_speculative_metrics(
        autoregressive_latency=2.0,
        speculative_latency=1.0,
        total_tokens=100,
        draft_k=5,
        simulated_acceptance_rate=0.85,
    )
    assert m["speedup_factor"] == 2.0
    assert m["autoregressive_tpot_ms"] == 20.0
    assert m["speculative_tpot_ms"] == 10.0
    assert m["token_acceptance_rate_alpha"] == 0.85
    assert m["draft_tokens_k"] == 5


def test_speculative_eval_prompts_schema() -> None:
    """Verify evaluation prompts contain required domains and token limits."""
    assert len(SPECULATIVE_EVAL_PROMPTS) >= 6
    domains = {p["domain"] for p in SPECULATIVE_EVAL_PROMPTS}
    assert "code" in domains
    assert "json_schema" in domains
    assert "prose" in domains


@pytest.mark.asyncio
async def test_evaluate_single_prompt_mock() -> None:
    """Verify single prompt evaluation with mocked HTTP responses."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "def quicksort(arr): return arr"}}],
        "usage": {"completion_tokens": 40},
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp

    prompt_spec = SPECULATIVE_EVAL_PROMPTS[0]
    res = await evaluate_single_prompt(mock_client, "http://localhost:8081", "key", prompt_spec)

    assert res["id"] == prompt_spec["id"]
    assert res["domain"] == "code"
    assert res["metrics"]["speedup_factor"] >= 1.0
    assert res["metrics"]["token_acceptance_rate_alpha"] == 0.82


@pytest.mark.asyncio
async def test_run_speculative_benchmark_mock(tmp_path: pytest.TempPathFactory) -> None:
    """Verify full benchmark execution and output serialization."""
    out_file = str(tmp_path / "spec_test.json")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "sample output"}}],
        "usage": {"completion_tokens": 30},
    }

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        payload = await run_speculative_benchmark(
            gateway_url="http://localhost:8081",
            api_key="key",
            output_path=out_file,
        )

        assert payload["benchmark"] == "speculative_decoding_acceptance_evaluation"
        assert payload["overall_summary"]["average_speedup_factor"] > 1.0
        assert "code" in payload["overall_summary"]["domain_breakdown"]
        assert len(payload["task_results"]) == len(SPECULATIVE_EVAL_PROMPTS)
