"""Sliding-window dual-metric (RPM + TPM) rate limiter for Cinch FastAPI Gateway."""

from __future__ import annotations

import collections
import math
import time
from typing import Deque, Dict, Tuple

from fastapi import Depends, HTTPException, Request, status

from gateway.config import GatewaySettings, get_settings


class SlidingWindowRateLimiter:
    """In-memory sliding window rate limiter tracking both requests and token budgets."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window_seconds = window_seconds
        # _history maps key -> Deque of (timestamp, token_count)
        self._history: Dict[str, Deque[Tuple[float, int]]] = collections.defaultdict(collections.deque)

    def _cleanup_old_records(self, key: str, now: float) -> Deque[Tuple[float, int]]:
        """Prune records older than current sliding window."""
        window_start = now - self.window_seconds
        records = self._history[key]
        while records and records[0][0] <= window_start:
            records.popleft()
        if not records:
            self._history.pop(key, None)
            records = collections.deque()
            self._history[key] = records
        return records

    def check(
        self,
        key: str,
        max_requests: int,
        max_tokens: int = 100000,
        requested_tokens: int = 1,
        now: float | None = None,
    ) -> Tuple[bool, int, int, float, str]:
        """Check if request under key is allowed within RPM and TPM quotas.

        Returns:
            Tuple of (is_allowed, remaining_requests, remaining_tokens, retry_after_seconds, violation_reason)
        """
        if now is None:
            now = time.time()

        records = self._cleanup_old_records(key, now)

        current_requests = len(records)
        current_tokens = sum(r[1] for r in records)

        # Check RPM breach
        if current_requests >= max_requests:
            oldest = records[0][0]
            retry_after = max(1.0, math.ceil((oldest + self.window_seconds) - now))
            return False, 0, max(0, max_tokens - current_tokens), retry_after, "RPM limit exceeded"

        # Check TPM breach
        if current_tokens + requested_tokens > max_tokens:
            oldest = records[0][0]
            retry_after = max(1.0, math.ceil((oldest + self.window_seconds) - now))
            return (
                False,
                max(0, max_requests - current_requests),
                0,
                retry_after,
                f"TPM limit exceeded (requested {requested_tokens} tokens, budget {max_tokens})",
            )

        # Record new request
        records.append((now, requested_tokens))
        remaining_requests = max_requests - len(records)
        remaining_tokens = max_tokens - sum(r[1] for r in records)

        return True, remaining_requests, max(0, remaining_tokens), 0.0, ""

    def reset(self) -> None:
        """Clear all stored rate limit history."""
        self._history.clear()


# Global limiter instance
rate_limiter = SlidingWindowRateLimiter(window_seconds=60.0)


def enforce_rate_limit(
    request: Request,
    current_settings: GatewaySettings = Depends(get_settings),
) -> Dict[str, str]:
    """FastAPI dependency to enforce RPM and TPM rate limiting by client IP."""
    client_ip = request.client.host if request.client else "unknown"
    rpm = current_settings.rate_limit_rpm
    tpm = current_settings.rate_limit_tpm

    # Check request state if token estimation was attached
    estimated_tokens = getattr(request.state, "estimated_tokens", 1)

    allowed, rem_rpm, rem_tpm, retry_after, reason = rate_limiter.check(
        client_ip,
        max_requests=rpm,
        max_tokens=tpm,
        requested_tokens=estimated_tokens,
    )

    headers = {
        "X-RateLimit-Limit": str(rpm),
        "X-RateLimit-Remaining": str(rem_rpm),
        "X-RateLimit-Limit-Requests": str(rpm),
        "X-RateLimit-Remaining-Requests": str(rem_rpm),
        "X-RateLimit-Limit-Tokens": str(tpm),
        "X-RateLimit-Remaining-Tokens": str(rem_tpm),
        "X-RateLimit-Reset": str(int(time.time() + 60.0)),
    }

    if not allowed:
        headers["Retry-After"] = str(int(retry_after))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {reason}. Max {rpm} RPM / {tpm} TPM.",
            headers=headers,
        )

    return headers
