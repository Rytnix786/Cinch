"""Unit test suite for Token Counter and Dual RPM/TPM Rate Limiter."""

from __future__ import annotations

from gateway.limiter import SlidingWindowRateLimiter
from gateway.token_counter import estimate_chat_tokens, estimate_request_tokens, estimate_text_tokens


def test_estimate_text_tokens() -> None:
    """Verify text token estimation heuristics."""
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens(None) == 0
    assert estimate_text_tokens("hello") >= 1
    # 20 words should be ~26 tokens
    twenty_words = "word " * 20
    assert 20 <= estimate_text_tokens(twenty_words) <= 35


def test_estimate_chat_tokens() -> None:
    """Verify chat message array token estimation."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a python function to compute fibonacci."},
    ]
    tokens = estimate_chat_tokens(messages)
    assert tokens > 15  # Includes priming and framing overhead


def test_estimate_request_tokens() -> None:
    """Verify request token estimation combining prompt and max_tokens."""
    body = {
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 64,
    }
    tokens = estimate_request_tokens(body)
    assert tokens > 64  # Prompt tokens + 64 max tokens

    # Default max_tokens fallback
    body_no_max = {"prompt": "Write a short poem."}
    tokens_default = estimate_request_tokens(body_no_max, default_max_tokens=128)
    assert tokens_default >= 128


def test_sliding_window_rpm_enforcement() -> None:
    """Verify RPM rate limit enforcement."""
    limiter = SlidingWindowRateLimiter(window_seconds=10.0)
    now = 1000.0

    for _ in range(5):
        allowed, rem_req, _, _, _ = limiter.check("client_a", max_requests=5, max_tokens=10000, now=now)
        assert allowed is True

    # 6th request should fail due to RPM
    allowed, rem_req, _, retry_after, reason = limiter.check("client_a", max_requests=5, max_tokens=10000, now=now)
    assert allowed is False
    assert rem_req == 0
    assert retry_after == 10.0
    assert "RPM" in reason


def test_sliding_window_tpm_enforcement() -> None:
    """Verify TPM token-budget rate limit enforcement."""
    limiter = SlidingWindowRateLimiter(window_seconds=10.0)
    now = 1000.0

    # Request 400 tokens out of 1000 budget (allowed)
    allowed, _, rem_tok, _, _ = limiter.check("client_b", max_requests=10, max_tokens=1000, requested_tokens=400, now=now)
    assert allowed is True
    assert rem_tok == 600

    # Request 700 tokens (exceeds remaining 600 -> fails due to TPM)
    allowed, _, rem_tok, retry_after, reason = limiter.check(
        "client_b", max_requests=10, max_tokens=1000, requested_tokens=700, now=now
    )
    assert allowed is False
    assert "TPM" in reason
    assert retry_after == 10.0


def test_sliding_window_replenishment() -> None:
    """Verify quota replenishes after sliding window expires."""
    limiter = SlidingWindowRateLimiter(window_seconds=10.0)
    t0 = 1000.0

    # Exhaust quota
    limiter.check("client_c", max_requests=2, max_tokens=500, requested_tokens=500, now=t0)
    limiter.check("client_c", max_requests=2, max_tokens=500, requested_tokens=1, now=t0)

    # Recheck at t0 + 11s (window expired)
    t1 = t0 + 11.0
    allowed, rem_req, rem_tok, _, _ = limiter.check(
        "client_c", max_requests=2, max_tokens=500, requested_tokens=100, now=t1
    )
    assert allowed is True
    assert rem_req == 1
    assert rem_tok == 400
