"""Three-state Circuit Breaker protecting Cinch Gateway from upstream GPU/worker panics."""

from __future__ import annotations

import enum
import time
from typing import Any, Dict, Optional, Tuple


class CircuitState(enum.Enum):
    """Circuit breaker operational states."""

    CLOSED = "closed"  # Normal operational state, requests pass through
    OPEN = "open"  # Tripped state, fast-fails immediately with 503
    HALF_OPEN = "half-open"  # Probing recovery state, permits single canary request


class CircuitBreaker:
    """Automated Circuit Breaker state machine with consecutive error detection and canary recovery."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 10.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time = 0.0
        self._last_state_change = time.time()
        self._total_trips = 0
        self._canary_in_flight = False

    @property
    def state(self) -> CircuitState:
        """Current state of the circuit breaker, evaluating cooldown transition to HALF_OPEN."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._last_state_change = time.time()
                self._canary_in_flight = False
        return self._state

    def can_execute(self) -> Tuple[bool, str, Optional[float]]:
        """Determine whether an incoming request may proceed to upstream backend.

        Returns:
            Tuple of (allowed, reason, retry_after_seconds).
        """
        current_state = self.state

        if current_state == CircuitState.CLOSED:
            return True, "healthy", None

        if current_state == CircuitState.HALF_OPEN:
            if not self._canary_in_flight:
                self._canary_in_flight = True
                return True, "canary_probe", None
            # Fast fail other traffic while canary probe is in-flight
            return False, "Canary probe in-flight, testing upstream recovery.", 2.0

        # OPEN state: Fast fail immediately
        elapsed = time.time() - self._last_failure_time
        remaining_cooldown = max(0.5, self.recovery_timeout - elapsed)
        return (
            False,
            f"Circuit breaker is OPEN. Upstream backend is degraded ({self._consecutive_failures} consecutive failures).",
            round(remaining_cooldown, 1),
        )

    def record_success(self) -> None:
        """Record successful upstream request completion (HTTP 200)."""
        if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            self._state = CircuitState.CLOSED
            self._last_state_change = time.time()
        self._consecutive_failures = 0
        self._canary_in_flight = False

    def record_failure(self) -> None:
        """Record upstream error (5xx, connection drop, timeout, or CUDA OOM panic)."""
        self._consecutive_failures += 1
        self._last_failure_time = time.time()
        self._canary_in_flight = False

        if self._state == CircuitState.HALF_OPEN or self._consecutive_failures >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                self._total_trips += 1
                self._last_state_change = time.time()
            self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset breaker to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time = 0.0
        self._canary_in_flight = False

    def get_metrics(self) -> Dict[str, Any]:
        """Return circuit breaker telemetry status."""
        return {
            "state": self.state.value,
            "consecutive_failures": self._consecutive_failures,
            "failure_threshold": self.failure_threshold,
            "total_trips": self._total_trips,
            "recovery_timeout_seconds": self.recovery_timeout,
            "last_failure_timestamp": round(self._last_failure_time, 2),
        }
