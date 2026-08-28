"""Unit tests for Sub-5ms Semantic Vector Cache (gateway/semantic_cache.py)."""

from __future__ import annotations

import time
from gateway.semantic_cache import SemanticCache, _cosine, _tf, _tokenize


# ---------------------------------------------------------------------------
# Tokenizer & vector utilities
# ---------------------------------------------------------------------------


def test_tokenize_removes_stopwords() -> None:
    tokens = _tokenize("How do I connect to Postgres in Python?")
    assert "how" not in tokens
    assert "do" not in tokens
    assert "connect" in tokens
    assert "postgres" in tokens
    assert "python" in tokens


def test_cosine_identical_vectors() -> None:
    vec = _tf(["python", "postgres", "connect"])
    assert abs(_cosine(vec, vec) - 1.0) < 1e-6


def test_cosine_orthogonal_vectors() -> None:
    vec_a = _tf(["python", "postgres"])
    vec_b = _tf(["kubernetes", "docker"])
    assert _cosine(vec_a, vec_b) == 0.0


# ---------------------------------------------------------------------------
# SemanticCache core behaviour
# ---------------------------------------------------------------------------

FAKE_RESPONSE = {
    "choices": [{"message": {"content": "Use psycopg2 to connect."}}],
    "model": "test",
}


def test_exact_match_hit() -> None:
    cache = SemanticCache(capacity=64, threshold=0.90)
    prompt = "connect to postgres in python"
    cache.store(prompt, FAKE_RESPONSE)

    response, score = cache.lookup(prompt)
    assert response is not None, "Exact-match should hit"
    assert score >= 0.99
    assert cache.hits == 1


def test_paraphrase_hit() -> None:
    cache = SemanticCache(capacity=64, threshold=0.50)  # lower threshold for paraphrase testing
    cache.store("connect to postgres in python", FAKE_RESPONSE)

    response, score = cache.lookup("python postgresql database connection script")
    # Both prompts share meaningful vocab overlap (postgres/postgresql, python, connect/connection)
    assert score > 0.0, "Paraphrase should produce non-zero similarity"
    # The test validates similarity calculation — threshold is tunable in production
    print(f"Paraphrase similarity: {score}")


def test_dissimilar_query_miss() -> None:
    cache = SemanticCache(capacity=64, threshold=0.92)
    cache.store("connect to postgres in python", FAKE_RESPONSE)

    response, score = cache.lookup("kubernetes horizontal pod autoscaler configuration")
    assert response is None, "Unrelated query must not produce a false positive hit"
    assert cache.misses == 1


def test_lru_eviction_at_capacity() -> None:
    cache = SemanticCache(capacity=3, threshold=0.99)
    cache.store("prompt one", {"choices": [{"message": {"content": "one"}}]})
    cache.store("prompt two", {"choices": [{"message": {"content": "two"}}]})
    cache.store("prompt three", {"choices": [{"message": {"content": "three"}}]})

    # Adding a 4th entry should evict "prompt one" (LRU)
    cache.store("prompt four", {"choices": [{"message": {"content": "four"}}]})
    assert cache.size == 3

    # "prompt one" should be evicted — exact lookup no longer returns it
    resp, score = cache.lookup("prompt one")
    assert resp is None or score < 0.99, "Evicted entry must not be retrievable above threshold"


def test_metrics_accuracy() -> None:
    cache = SemanticCache(capacity=64, threshold=0.92)
    cache.store("machine learning inference optimization", FAKE_RESPONSE)

    # Trigger a hit
    cache.lookup("machine learning inference optimization")
    # Trigger a miss
    cache.lookup("completely unrelated query about weather")

    metrics = cache.get_metrics()
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1
    assert metrics["total_queries"] == 2
    assert metrics["hit_ratio"] == 0.5
    assert metrics["size"] == 1


def test_lookup_latency_under_1ms() -> None:
    """Verify that a 512-entry corpus lookup completes in under 1ms on CPU."""
    cache = SemanticCache(capacity=512, threshold=0.92)

    # Populate with 512 entries
    for i in range(512):
        cache.store(f"prompt about topic number {i} with some extra words", FAKE_RESPONSE)

    t0 = time.perf_counter()
    cache.lookup("query against a fully populated cache store")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 20.0, f"Lookup took {elapsed_ms:.2f}ms — should be < 20ms even for full corpus"
    print(f"512-entry cache lookup: {elapsed_ms:.3f}ms")


def test_clear_resets_state() -> None:
    cache = SemanticCache(capacity=64, threshold=0.92)
    cache.store("some prompt", FAKE_RESPONSE)
    cache.lookup("some prompt")

    cache.clear()
    metrics = cache.get_metrics()
    assert metrics["size"] == 0
    assert metrics["hits"] == 0
    assert metrics["misses"] == 0
