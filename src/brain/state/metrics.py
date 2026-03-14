"""Session metrics: counters, timing, XP tracking."""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field


@dataclass(slots=True, kw_only=True)
class SessionMetrics:
    """All session-level counters and timing stats."""

    session_start: float = field(default_factory=time.time)

    # Routine tracking
    routine_counts: Counter[str] = field(default_factory=Counter)
    routine_failures: Counter[str] = field(default_factory=Counter)
    routine_time: dict[str, float] = field(default_factory=dict)
    routine_start_time: float = 0.0
    _last_routine_name: str = ""
    _last_routine_start: float = 0.0

    # Combat stats
    total_casts: int = 0
    rest_count: int = 0
    flee_count: int = 0
    total_combat_time: float = 0.0
    total_cast_time: float = 0.0

    # Pull stats
    pull_distances: list[float] = field(default_factory=list)
    pull_engage_times: list[float] = field(default_factory=list)
    pull_dc_fizzles: int = 0
    pull_pet_only_count: int = 0
    cycle_start_time: float = 0.0
    total_cycle_times: list[float] = field(default_factory=list)

    # Acquire stats
    acquire_modes: dict[str, int] = field(default_factory=dict)
    acquire_tab_totals: list[int] = field(default_factory=list)
    acquire_invalid_tabs: int = 0
    acquire_approach_forced: int = 0
    consecutive_acquire_fails: int = 0

    # Wander stats
    wander_total_distance: float = 0.0
    wander_count: int = 0

    # Stuck tracking
    stuck_count: int = 0
    stuck_total_time: float = 0.0

    # XP rate tracking
    xp_history: list[tuple[float, int]] = field(default_factory=list)
    xp_last_raw: int = 0
    xp_gained_pct: float = 0.0

    # Positional diversity tracking (wander variation)
    last_kill_x: float = 0.0
    last_kill_y: float = 0.0
    stationary_kills: int = 0

    def record_xp_sample(self, timestamp: float, xp_raw: int) -> None:
        """Record XP sample for rate tracking."""
        if xp_raw <= 0:
            return
        self.xp_history.append((timestamp, xp_raw))
        if len(self.xp_history) > 200:
            cutoff = timestamp - 1800
            self.xp_history = [(t, x) for t, x in self.xp_history if t > cutoff]

    def xp_per_hour(self, window_seconds: float = 600) -> float:
        """XP percentage points per hour over a moving window."""
        from core.constants import XP_SCALE_MAX

        if len(self.xp_history) < 2:
            return 0.0
        now_t = self.xp_history[-1][0]
        cutoff = now_t - window_seconds
        first = None
        for t, x in self.xp_history:
            if t >= cutoff:
                first = (t, x)
                break
        if first is None or first[0] >= now_t - 1:
            return 0.0
        last = self.xp_history[-1]
        elapsed_hr = (last[0] - first[0]) / 3600
        if elapsed_hr < 0.001:
            return 0.0
        delta_pct: float = (last[1] - first[1]) / float(XP_SCALE_MAX) * 100
        if delta_pct < 0:
            xph: float = self.xp_gained_pct / elapsed_hr if self.xp_gained_pct > 0 else 0.0
            return xph
        rate: float = delta_pct / elapsed_hr
        return rate

    def time_to_level(self) -> float | None:
        """Estimated minutes to next level. None if insufficient data."""
        from core.constants import XP_SCALE_MAX

        rate = self.xp_per_hour(300)
        if rate <= 0 or not self.xp_history:
            return None
        current_raw = self.xp_history[-1][1]
        remaining_pct = (XP_SCALE_MAX - current_raw) / float(XP_SCALE_MAX) * 100
        if remaining_pct <= 0:
            return None
        minutes: float = (remaining_pct / rate) * 60
        return minutes

    def update_stationary_kills(self, pos_x: float, pos_y: float) -> None:
        """Track whether kills are happening in the same location."""
        if self.last_kill_x == 0 and self.last_kill_y == 0:
            self.last_kill_x = pos_x
            self.last_kill_y = pos_y
            self.stationary_kills = 1
            return
        dist = math.hypot(pos_x - self.last_kill_x, pos_y - self.last_kill_y)
        if dist < 30:
            self.stationary_kills += 1
        else:
            self.stationary_kills = 1
        self.last_kill_x = pos_x
        self.last_kill_y = pos_y

    def should_reposition(self) -> bool:
        """Probabilistic check: should the agent move to a new spot?"""
        if self.stationary_kills < 2:
            return False
        chance = 0.25 + 0.15 * (self.stationary_kills - 2)
        chance = min(chance, 0.90)
        if random.random() < chance:
            self.stationary_kills = 0
            return True
        return False

    def trim_lists(self) -> None:
        """Cap rolling lists to prevent unbounded growth in long sessions."""
        if len(self.pull_distances) > 500:
            self.pull_distances = self.pull_distances[-250:]
        if len(self.pull_engage_times) > 500:
            self.pull_engage_times = self.pull_engage_times[-250:]
        if len(self.total_cycle_times) > 500:
            self.total_cycle_times = self.total_cycle_times[-250:]
        if len(self.acquire_tab_totals) > 500:
            self.acquire_tab_totals = self.acquire_tab_totals[-250:]
        if len(self.xp_history) > 500:
            self.xp_history = self.xp_history[-250:]

    def snapshot_collections(self) -> dict:
        """Return shallow copies of all mutable collections for cross-thread reads.

        Call under ctx.lock to avoid RuntimeError from concurrent mutation.
        Returns a dict of collection copies safe to iterate outside the lock.
        """
        return {
            "routine_time": dict(self.routine_time),
            "routine_counts": Counter(self.routine_counts),
            "routine_failures": Counter(self.routine_failures),
            "pull_distances": list(self.pull_distances),
            "pull_engage_times": list(self.pull_engage_times),
            "total_cycle_times": list(self.total_cycle_times),
            "acquire_tab_totals": list(self.acquire_tab_totals),
            "acquire_modes": dict(self.acquire_modes),
            "xp_history": list(self.xp_history),
        }
