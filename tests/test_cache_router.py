"""Unit test suite for Prefix Hashing and Cache Affinity Router."""

from __future__ import annotations

from gateway.cache_router import PrefixCacheRouter, extract_prompt_prefix


def test_extract_prompt_prefix_system() -> None:
    """Verify prefix extraction from system message."""
    body = {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert autonomous code refactoring assistant with deep Python knowledge.",
            },
            {"role": "user", "content": "Hello!"},
        ]
    }
    prefix, p_hash = extract_prompt_prefix(body, min_chars=32)
    assert "expert autonomous code refactoring" in prefix
    assert len(p_hash) == 16


def test_extract_prompt_prefix_fallback_user() -> None:
    """Verify fallback to long first user message when no system message exists."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": "Please analyze this long multi-line document and extract all key technical entities and dates.",
            },
        ]
    }
    prefix, p_hash = extract_prompt_prefix(body, min_chars=32)
    assert "Please analyze this long multi-line document" in prefix
    assert len(p_hash) == 16


def test_extract_prompt_prefix_raw_prompt() -> None:
    """Verify prefix extraction from raw prompt string."""
    body = {"prompt": "The quick brown fox jumps over the lazy dog repeatedly until sunset."}
    prefix, p_hash = extract_prompt_prefix(body, min_chars=32)
    assert "The quick brown fox" in prefix
    assert len(p_hash) == 16


def test_extract_prompt_prefix_short_ignored() -> None:
    """Verify short prefixes below min_chars return empty strings."""
    body = {"messages": [{"role": "user", "content": "Hi!"}]}
    prefix, p_hash = extract_prompt_prefix(body, min_chars=32)
    assert prefix == ""
    assert p_hash == ""


def test_sha256_hash_stability() -> None:
    """Verify hash stability across extra spaces and newlines."""
    body1 = {
        "messages": [{"role": "system", "content": "You are a helpful coding assistant.\n\nAlways provide tests."}]
    }
    body2 = {
        "messages": [{"role": "system", "content": "You are a  helpful   coding assistant. Always provide tests."}]
    }

    _, hash1 = extract_prompt_prefix(body1, min_chars=20)
    _, hash2 = extract_prompt_prefix(body2, min_chars=20)
    assert hash1 == hash2


def test_cache_router_miss_and_hit() -> None:
    """Verify cache miss on first route and cache hit on subsequent route."""
    router = PrefixCacheRouter(capacity=10)
    targets = ["http://vllm-1:8000", "http://vllm-2:8000"]
    p_hash = "abcdef1234567890"

    # First access -> MISS
    target1, is_hit1 = router.route(p_hash, available_targets=targets)
    assert is_hit1 is False
    assert target1 in targets

    # Second access -> HIT
    target2, is_hit2 = router.route(p_hash, available_targets=targets)
    assert is_hit2 is True
    assert target2 == target1

    metrics = router.get_metrics()
    assert metrics["total_hits"] == 1
    assert metrics["total_misses"] == 1
    assert metrics["hit_ratio"] == 0.5


def test_cache_router_lru_eviction() -> None:
    """Verify LRU capacity eviction when capacity is exceeded."""
    router = PrefixCacheRouter(capacity=2)
    targets = ["http://vllm:8000"]

    router.route("hash1", targets)
    router.route("hash2", targets)
    assert router.get_metrics()["cached_prefixes"] == 2

    # Add 3rd item -> hash1 evicted
    router.route("hash3", targets)
    assert router.get_metrics()["cached_prefixes"] == 2

    # hash1 should now be a miss
    _, is_hit = router.route("hash1", targets)
    assert is_hit is False
