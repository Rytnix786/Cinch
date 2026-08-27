"""Unit test suite for Circuit Breaker state machine and fault protection."""

from __future__ import annotations

import time
from gateway.circuit_breaker import CircuitBreaker, CircuitState


def test_circuit_breaker_initial_closed() -> None:
    """Verify breaker starts in CLOSED state and allows execution."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=5.0)
    assert cb.state == CircuitState.CLOSED
    allowed, reason, retry_after = cb.can_execute()
    assert allowed is True
    assert reason == "healthy"
    assert retry_after is None


def test_circuit_breaker_trips_to_open() -> None:
    """Verify breaker trips to OPEN after reaching failure threshold."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=5.0)

    # 1st and 2nd failure -> still CLOSED
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED

    # 3rd failure -> trips to OPEN
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # Ingress denied with retry header
    allowed, reason, retry_after = cb.can_execute()
    assert allowed is False
    assert "Circuit breaker is OPEN" in reason
    assert retry_after is not None
    assert retry_after > 0


def test_circuit_breaker_cooldown_to_half_open() -> None:
    """Verify breaker transitions to HALF_OPEN after cooldown expiration."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.1)

    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # Sleep past cooldown
    time.sleep(0.15)
    assert cb.state == CircuitState.HALF_OPEN

    # 1st request is canary -> allowed
    allowed1, reason1, _ = cb.can_execute()
    assert allowed1 is True
    assert reason1 == "canary_probe"

    # Concurrent 2nd request while canary in-flight -> fast-failed
    allowed2, reason2, _ = cb.can_execute()
    assert allowed2 is False
    assert "Canary probe in-flight" in reason2


def test_circuit_breaker_canary_success_recovers() -> None:
    """Verify successful canary probe recovers breaker back to CLOSED."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.1)
    cb.record_failure()
    cb.record_failure()
    time.sleep(0.15)
    assert cb.state == CircuitState.HALF_OPEN

    # Canary succeeds
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.get_metrics()["consecutive_failures"] == 0


def test_circuit_breaker_canary_failure_reopens() -> None:
    """Verify failed canary probe immediately re-trips breaker back to OPEN."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.1)
    cb.record_failure()
    cb.record_failure()
    time.sleep(0.15)
    assert cb.state == CircuitState.HALF_OPEN

    # Canary fails
    cb.record_failure()
    assert cb.state == CircuitState.OPEN


def test_circuit_breaker_metrics() -> None:
    """Verify circuit breaker telemetry metrics."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=10.0)
    cb.record_failure()
    metrics = cb.get_metrics()
    assert metrics["state"] == "closed"
    assert metrics["consecutive_failures"] == 1
    assert metrics["failure_threshold"] == 3
    assert metrics["total_trips"] == 0
