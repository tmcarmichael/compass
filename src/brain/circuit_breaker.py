"""Per-routine circuit breaker: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.

Prevents a repeatedly-failing routine from consuming brain ticks and
blocking lower-priority routines from running. Emergency rules (FLEE,
DEATH_RECOVERY, FEIGN_DEATH) are exempt -- the agent can always flee.

States:
  CLOSED    -- normal operation, failures are tracked
  OPEN      -- routine suppressed, waiting for recovery period
  HALF_OPEN -- one probe attempt allowed; success closes, failure reopens
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from util.structured_log import log_event

log = logging.getLogger(__name__)

__all__ = ["CircuitBreaker"]


@dataclass(slots=True)
class CircuitBreaker:
    """Per-routine circuit breaker with windowed failure tracking."""

    name: str
    max_failures: int = 5
    window_seconds: float = 300.0
    recovery_seconds: float = 120.0
    clock: Callable[[], float] = time.time
    # -- internal state --
    _failures: list[float] = field(default_factory=list)
    _state: str = "CLOSED"
    _opened_at: float = 0.0
    _trip_count: int = 0

    def record_failure(self) -> None:
        """Record a routine FAILURE. May trip the breaker."""
        now = self.clock()
        self._failures.append(now)
        # Trim failures outside window
        cutoff = now - self.window_seconds
        self._failures = [t for t in self._failures if t > cutoff]

        if self._state == "HALF_OPEN":
            self._state = "OPEN"
            self._opened_at = now
            log.info("[CIRCUIT] %s: HALF_OPEN -> OPEN (probe failed)", self.name)
            return

        if self._state == "CLOSED" and len(self._failures) >= self.max_failures:
            self._state = "OPEN"
            self._opened_at = now
            self._trip_count += 1
            log_event(
                log,
                "circuit_open",
                f"[CIRCUIT] {self.name}: TRIPPED after "
                f"{len(self._failures)} failures in "
                f"{self.window_seconds:.0f}s",
                level=logging.WARNING,
                routine=self.name,
                failures=len(self._failures),
                trip_count=self._trip_count,
            )

    def record_success(self) -> None:
        """Record a SUCCESS. Resets the breaker if half-open."""
        if self._state == "HALF_OPEN":
            self._state = "CLOSED"
            self._failures.clear()
            log.info("[CIRCUIT] %s: HALF_OPEN -> CLOSED (probe succeeded)", self.name)

    def allow(self) -> bool:
        """Returns True if the routine is allowed to run."""
        if self._state == "CLOSED":
            return True
        if self._state == "OPEN":
            if self.clock() - self._opened_at > self.recovery_seconds:
                self._state = "HALF_OPEN"
                log.info("[CIRCUIT] %s: OPEN -> HALF_OPEN (recovery elapsed)", self.name)
                return True  # allow one probe
            return False
        # HALF_OPEN: allow probe
        return True

    @property
    def state(self) -> str:
        """Current breaker state: CLOSED, OPEN, or HALF_OPEN."""
        return self._state

    @property
    def trip_count(self) -> int:
        """How many times this breaker has tripped this session."""
        return self._trip_count
