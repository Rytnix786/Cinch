"""Sliding-window rate limiter for Cinch FastAPI Gateway."""

from __future__ import annotations

import collections
import math
import time
from typing import Deque, Dict, Tuple

from fastapi import Depends, HTTPException, Request, status

from gateway.config import GatewaySettings, get_settings


class SlidingWindowRateLimiter:
    """In-memory sliding window rate limiter."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window_seconds = window_seconds
        self._history: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)

    def _cleanup_old_requests(self, key: str, now: float) -> Deque[float]:
        """Prune timestamps older than current sliding window."""
        window_start = now - self.window_seconds
        timestamps = self._history[key]
        while timestamps and timestamps[0] <= window_start:
            timestamps.popleft()
        if not timestamps:
            self._history.pop(key, None)
            timestamps = collections.deque()
            self._history[key] = timestamps
        return timestamps

    def check(self, key: str, max_requests: int, now: float | None = None) -> Tuple[bool, int, float]:
        """Check if request under key is allowed within max_requests.

        Returns:
            Tuple of (is_allowed, remaining_quota, retry_after_seconds)
        """
        if now is None:
            now = time.time()

        timestamps = self._cleanup_old_requests(key, now)

        if len(timestamps) >= max_requests:
            oldest = timestamps[0]
            retry_after = max(1.0, math.ceil((oldest + self.window_seconds) - now))
            return False, 0, retry_after

        timestamps.append(now)
        remaining = max_requests - len(timestamps)
        return True, remaining, 0.0

    def reset(self) -> None:
        """Clear all stored rate limit history."""
        self._history.clear()


# Global limiter instance
rate_limiter = SlidingWindowRateLimiter(window_seconds=60.0)


def enforce_rate_limit(
    request: Request,
    current_settings: GatewaySettings = Depends(get_settings),
) -> Dict[str, str]:
    """FastAPI dependency to enforce rate limiting by client IP."""
    client_ip = request.client.host if request.client else "unknown"
    rpm = current_settings.rate_limit_rpm

    allowed, remaining, retry_after = rate_limiter.check(client_ip, max_requests=rpm)

    headers = {
        "X-RateLimit-Limit": str(rpm),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(int(time.time() + 60.0)),
    }

    if not allowed:
        headers["Retry-After"] = str(int(retry_after))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {rpm} requests per minute allowed.",
            headers=headers,
        )

    return headers
