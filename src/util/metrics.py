"""Metrics engine: percentile tracking, success rates, session-level aggregation.

Provides rolling-window percentile tracking (p50/p95/p99) and per-action
success rate tracking. MetricsEngine aggregates all metrics for inclusion
in session reports and dashboard display.

All classes are designed for single-thread use (brain thread only).
"""

from __future__ import annotations

import bisect
from collections import deque


class PercentileTracker:
    """Rolling-window percentile tracker using a sorted list.

    Maintains last N values. O(N) insert via bisect, O(1) percentile lookup.
    Designed for tick_ms tracking where N=1000 and inserts happen at 10Hz.
    """

    def __init__(self, window: int = 1000) -> None:
        self._values: deque[float] = deque(maxlen=window)
        self._sorted: list[float] = []
        self._window = window

    def add(self, value: float) -> None:
        """Add a value. If window is full, remove oldest before inserting."""
        if len(self._values) >= self._window:
            old = self._values[0]
            idx = bisect.bisect_left(self._sorted, old)
            if idx < len(self._sorted) and self._sorted[idx] == old:
                self._sorted.pop(idx)
        self._values.append(value)
        bisect.insort(self._sorted, value)

    def percentile(self, pct: float) -> float:
        """Get the value at the given percentile (0.0-1.0)."""
        if not self._sorted:
            return 0.0
        idx = int(pct * (len(self._sorted) - 1))
        return self._sorted[idx]

    def p50(self) -> float:
        return self.percentile(0.5)

    def p95(self) -> float:
        return self.percentile(0.95)

    def p99(self) -> float:
        return self.percentile(0.99)

    @property
    def count(self) -> int:
        return len(self._values)

    def as_dict(self) -> dict[str, object]:
        """Compact summary for session report."""
        if not self._sorted:
            return {"p50": 0, "p95": 0, "p99": 0, "count": 0}
        return {
            "p50": round(self.p50(), 2),
            "p95": round(self.p95(), 2),
            "p99": round(self.p99(), 2),
            "count": self.count,
        }


class SuccessRateTracker:
    """Rolling success/failure rate over last N outcomes."""

    def __init__(self, window: int = 100) -> None:
        self._outcomes: deque[bool] = deque(maxlen=window)

    def record(self, success: bool) -> None:
        self._outcomes.append(success)

    def rate(self) -> float:
        """Success rate as 0.0-1.0. Returns 1.0 if no data."""
        if not self._outcomes:
            return 1.0
        return sum(self._outcomes) / len(self._outcomes)

    @property
    def count(self) -> int:
        return len(self._outcomes)

    def as_dict(self) -> dict[str, object]:
        return {
            "rate": round(self.rate(), 3),
            "count": self.count,
        }


class MetricsEngine:
    """Central metrics aggregator. One per session.

    Tracks:
    - tick_duration: p50/p95/p99 of brain tick wall time
    - per_routine_success: success rate by routine name
    - per_action_success: success rate for key actions (pull, acquire, loot, cast)
    """

    def __init__(self) -> None:
        self.tick_duration = PercentileTracker(window=1000)
        self._routine_success: dict[str, SuccessRateTracker] = {}
        self._action_success: dict[str, SuccessRateTracker] = {}

    def record_tick(self, tick_ms: float) -> None:
        """Record one brain tick duration in milliseconds."""
        self.tick_duration.add(tick_ms)

    def record_routine_outcome(self, name: str, success: bool) -> None:
        """Record a routine completion (success or failure)."""
        if name not in self._routine_success:
            self._routine_success[name] = SuccessRateTracker(window=100)
        self._routine_success[name].record(success)

    def record_action(self, action: str, success: bool) -> None:
        """Record a discrete action outcome (pull, acquire, loot, cast)."""
        if action not in self._action_success:
            self._action_success[action] = SuccessRateTracker(window=100)
        self._action_success[action].record(success)

    def summary(self) -> dict[str, object]:
        """Full metrics summary for session report JSON."""
        return {
            "tick_duration": self.tick_duration.as_dict(),
            "routine_success": {name: tracker.as_dict() for name, tracker in self._routine_success.items()},
            "action_success": {name: tracker.as_dict() for name, tracker in self._action_success.items()},
        }
