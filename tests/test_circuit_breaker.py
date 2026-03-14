"""Tests for brain.circuit_breaker -- per-routine circuit breaker.

The breaker transitions CLOSED -> OPEN -> HALF_OPEN -> CLOSED based on
windowed failure counts and a recovery timer. Time is controlled via
constructor-injected clock.
"""

from __future__ import annotations

from brain.circuit_breaker import CircuitBreaker


class TestCircuitBreaker:
    def test_initial_state_closed(self) -> None:
        cb = CircuitBreaker("test")
        assert cb.state == "CLOSED"

    def test_allow_when_closed(self) -> None:
        cb = CircuitBreaker("test")
        assert cb.allow() is True

    def test_single_failure_stays_closed(self) -> None:
        now = 1000.0
        cb = CircuitBreaker("test", max_failures=5, clock=lambda: now)
        cb.record_failure()
        assert cb.state == "CLOSED"

    def test_failures_trip_to_open(self) -> None:
        now = 1000.0
        cb = CircuitBreaker("test", max_failures=3, clock=lambda: now)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "OPEN"

    def test_open_denies(self) -> None:
        now = 1000.0
        cb = CircuitBreaker("test", max_failures=2, clock=lambda: now)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.allow() is False

    def test_recovery_transitions_to_half_open(self) -> None:
        t = [1000.0]
        cb = CircuitBreaker("test", max_failures=2, recovery_seconds=10, clock=lambda: t[0])
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "OPEN"
        t[0] = 1011.0
        assert cb.allow() is True
        assert cb.state == "HALF_OPEN"

    def test_half_open_allows_probe(self) -> None:
        t = [1000.0]
        cb = CircuitBreaker("test", max_failures=2, recovery_seconds=10, clock=lambda: t[0])
        cb.record_failure()
        cb.record_failure()
        t[0] = 1011.0
        cb.allow()  # transitions to HALF_OPEN
        assert cb.state == "HALF_OPEN"
        # A second allow in HALF_OPEN still returns True (probe attempt)
        assert cb.allow() is True

    def test_probe_success_closes(self) -> None:
        t = [1000.0]
        cb = CircuitBreaker("test", max_failures=2, recovery_seconds=10, clock=lambda: t[0])
        cb.record_failure()
        cb.record_failure()
        t[0] = 1011.0
        cb.allow()  # -> HALF_OPEN
        cb.record_success()
        assert cb.state == "CLOSED"

    def test_probe_failure_reopens(self) -> None:
        t = [1000.0]
        cb = CircuitBreaker("test", max_failures=2, recovery_seconds=10, clock=lambda: t[0])
        cb.record_failure()
        cb.record_failure()
        t[0] = 1011.0
        cb.allow()  # -> HALF_OPEN
        cb.record_failure()
        assert cb.state == "OPEN"

    def test_trip_count_increments(self) -> None:
        t = [1000.0]
        cb = CircuitBreaker("test", max_failures=2, recovery_seconds=5, clock=lambda: t[0])
        assert cb.trip_count == 0
        cb.record_failure()
        cb.record_failure()
        assert cb.trip_count == 1
        # Recover, close, trip again
        t[0] = 1006.0
        cb.allow()  # -> HALF_OPEN
        cb.record_success()  # -> CLOSED
        cb.record_failure()
        cb.record_failure()
        assert cb.trip_count == 2

    def test_stale_failures_pruned(self) -> None:
        t = [1000.0]
        cb = CircuitBreaker("test", max_failures=3, window_seconds=60, clock=lambda: t[0])
        cb.record_failure()
        cb.record_failure()
        # Advance past the window so those failures become stale
        t[0] = 1100.0
        cb.record_failure()
        # Only 1 failure in the current window -- should still be CLOSED
        assert cb.state == "CLOSED"
