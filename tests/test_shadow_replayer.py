"""Unit tests for Production Shadow Traffic Replayer (gateway/shadow_replayer.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest
from gateway.shadow_replayer import ShadowTrafficReplayer, compute_lexical_similarity


def test_sampling_logic() -> None:
    # Always sample
    replayer = ShadowTrafficReplayer(sample_rate=1.0)
    assert replayer.should_sample() is True

    # Never sample
    replayer = ShadowTrafficReplayer(sample_rate=0.0)
    assert replayer.should_sample() is False

    # Header overrides
    assert replayer.should_sample(force_header="true") is True
    assert replayer.should_sample(force_header="1") is True
    assert replayer.should_sample(force_header="false") is False

    # Disabled
    replayer = ShadowTrafficReplayer(enabled=False, sample_rate=1.0)
    assert replayer.should_sample() is False
    assert replayer.should_sample(force_header="true") is False


def test_lexical_similarity_and_divergence() -> None:
    # Identical strings
    score_identical = compute_lexical_similarity("Kubernetes pod readiness check", "Kubernetes pod readiness check")
    assert score_identical == 1.0

    # High similarity paraphrase
    score_similar = compute_lexical_similarity(
        "A database foreign key provides relational integrity constraints.",
        "A database foreign key ensures relational integrity and constraints across tables.",
    )
    assert score_similar > 0.50

    # Diverging outputs
    score_diverging = compute_lexical_similarity(
        "Python is a dynamic programming language.",
        "Quantum computing relies on qubits in superposition.",
    )
    assert score_diverging < 0.10


@pytest.mark.asyncio
async def test_replay_shadow_execution() -> None:
    replayer = ShadowTrafficReplayer(
        enabled=True,
        shadow_backend_url="http://mock-shadow:8000",
        sample_rate=1.0,
    )

    # Mock successful candidate response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Sample candidate output."}}],
        "usage": {"completion_tokens": 12},
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp

    req_body = {"model": "Qwen2.5-7B", "messages": [{"role": "user", "content": "Hello"}]}
    prod_json = {
        "choices": [{"message": {"role": "assistant", "content": "Sample candidate output."}}],
        "usage": {"completion_tokens": 12},
    }

    record = await replayer.replay_shadow(
        client=mock_client,
        request_body=req_body,
        prod_resp_json=prod_json,
        prod_latency_ms=25.0,
        prod_status=200,
    )

    assert record is not None
    assert record.prod_status == 200
    assert record.shadow_status == 200
    assert record.divergence_detected is False
    assert len(replayer.get_traces()) == 1


def test_ring_buffer_trace_eviction() -> None:
    replayer = ShadowTrafficReplayer(max_traces=3)

    # Directly insert records
    for i in range(5):
        from gateway.shadow_replayer import ShadowTraceRecord

        rec = ShadowTraceRecord(
            trace_id=f"t_{i}",
            model="Qwen",
            prompt="test",
            prod_status=200,
            shadow_status=200,
            prod_latency_ms=10.0,
            shadow_latency_ms=12.0,
            latency_delta_ms=2.0,
            prod_tokens=10,
            shadow_tokens=10,
            token_count_ratio=1.0,
            lexical_similarity_score=1.0,
            divergence_detected=False,
        )
        replayer._traces.append(rec)

    traces = replayer.get_traces()
    assert len(traces) == 3
    assert traces[0]["trace_id"] == "t_4"  # Most recent first
    assert traces[2]["trace_id"] == "t_2"


def test_dynamic_reconfiguration() -> None:
    replayer = ShadowTrafficReplayer(sample_rate=0.1, shadow_backend_url="http://a:8000")

    metrics = replayer.set_config(sample_rate=0.5, shadow_backend_url="http://b:9000", enabled=True)
    assert metrics["sample_rate"] == 0.5
    assert metrics["shadow_backend_url"] == "http://b:9000"
    assert metrics["enabled"] is True


def test_shadow_metrics_accuracy() -> None:
    replayer = ShadowTrafficReplayer()
    metrics = replayer.get_metrics()
    assert metrics["enabled"] is True
    assert metrics["total_sampled_requests"] == 0
    assert metrics["divergence_rate_pct"] == 0.0
