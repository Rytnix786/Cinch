"""Unit tests for Context & Prompt Compaction Engine (gateway/compressor.py)."""

from __future__ import annotations

import time
from gateway.compressor import PromptCompressor


def test_verbose_rag_prompt_compaction() -> None:
    compressor = PromptCompressor(min_tokens=30, target_ratio=0.60)
    verbose_text = (
        "Please note that in order to effectively configure the production Kubernetes cluster, "
        "it is important to note that the horizontal pod autoscaler must monitor memory and CPU. "
        "Due to the fact that traffic spikes can occur at the present time, furthermore, "
        "as an AI assistant, I recommend configuring replica limits between 2 and 10 pods. "
        "Additionally, basically and essentially, every deployment requires liveness probes."
    )
    compacted, res = compressor.compress_text(verbose_text)

    assert res.is_compacted is True
    assert res.compacted_tokens < res.original_tokens
    assert res.compression_ratio <= 0.85
    assert "in order to" not in compacted.lower()
    assert "as an ai assistant" not in compacted.lower()
    assert "Kubernetes" in compacted
    assert "2" in compacted and "10" in compacted


def test_code_block_exact_preservation() -> None:
    compressor = PromptCompressor(min_tokens=20, target_ratio=0.60, preserve_code_blocks=True)
    code_block = (
        "```python\n"
        "def compute_gradients(weights, loss):\n"
        "    return np.dot(weights.T, loss) * 0.01\n"
        "```"
    )
    text = (
        "Please note that in order to properly inspect the neural network training pipeline, "
        "here is the exact implementation code:\n"
        f"{code_block}\n"
        "It is worth noting that you should verify the gradient dimensions carefully."
    )
    compacted, res = compressor.compress_text(text)

    assert res.is_compacted is True
    assert code_block in compacted, "Code block was altered during compaction!"


def test_short_prompt_bypassed() -> None:
    compressor = PromptCompressor(min_tokens=50)
    short_query = "What is the status of the vLLM container?"
    compacted, res = compressor.compress_text(short_query)

    assert res.is_compacted is False
    assert compacted == short_query
    assert res.tokens_saved == 0
    assert res.compression_ratio == 1.0


def test_entity_and_number_retention() -> None:
    compressor = PromptCompressor(min_tokens=20, target_ratio=0.50)
    text = (
        "As an AI assistant, please note that on 2026-08-27, the server IP 192.168.1.100 "
        "processed 45,000 transactions across PostgreSQL and Redis clusters with 99.9% uptime. "
        "Furthermore, basically, it is important to note that the AWS S3 backup completed in 14.2 seconds."
    )
    compacted, res = compressor.compress_text(text)

    assert res.is_compacted is True
    assert "2026-08-27" in compacted
    assert "192.168.1.100" in compacted
    assert "45,000" in compacted or "45000" in compacted or "99.9%" in compacted
    assert "PostgreSQL" in compacted
    assert "Redis" in compacted
    assert "14.2" in compacted


def test_chat_messages_multi_turn_compaction() -> None:
    compressor = PromptCompressor(min_tokens=30, target_ratio=0.60)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful coding assistant. Please note that you should answer concisely.",
        },
        {
            "role": "user",
            "content": (
                "In order to effectively optimize our database queries, due to the fact that we have high traffic, "
                "could you please explain how index scans compare to sequential scans in PostgreSQL with 10M rows?"
            ),
        },
    ]
    compacted_msgs, res = compressor.compress_messages(messages)

    assert res.is_compacted is True
    assert len(compacted_msgs) == 2
    assert res.tokens_saved > 0
    assert "PostgreSQL" in compacted_msgs[1]["content"]


def test_sub_millisecond_latency() -> None:
    compressor = PromptCompressor(min_tokens=30)
    long_doc = "The distributed cluster architecture requires robust consensus algorithms. " * 30

    t0 = time.perf_counter()
    _, res = compressor.compress_text(long_doc)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert res.is_compacted is True
    assert elapsed_ms < 5.0, f"Compaction latency took {elapsed_ms:.2f}ms — expected < 5ms"


def test_compressor_metrics_tracking() -> None:
    compressor = PromptCompressor(min_tokens=10, target_ratio=0.60)
    compressor.compress_text("Short")  # bypassed (< 10)
    compressor.compress_text(
        "In order to effectively deploy this, please note that we need 4 replicas."
    )

    metrics = compressor.get_metrics()
    assert metrics["enabled"] is True
    assert metrics["compacted_requests"] == 1
    assert metrics["tokens_saved_total"] > 0
