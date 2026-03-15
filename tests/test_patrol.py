"""Tests for brain/world/patrol.py: patrol detection and safety queries.

Covers PatrolMixin trace management, cycle detection (_detect_patrol),
patrol_safe_window threat prediction, and patrolling_threats filtering.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from brain.world.patrol import (
    PATROL_TRACE_WINDOW,
    PatrolMixin,
    patrol_safe_window,
    patrolling_threats,
)
from core.types import Point

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeMobHistory(PatrolMixin):
    """Minimal host for PatrolMixin, providing required attributes."""

    def __init__(self) -> None:
        self.patrol_trace: deque[tuple[float, float, float]] | None = None
        self.patrol_period: float = 0.0
        self._patrol_checked: float = 0.0


def _make_profile(
    *, is_patrolling: bool = False, is_threat: bool = True, x: float = 0.0, y: float = 0.0, spawn_id: int = 1
) -> SimpleNamespace:
    from core.types import Point

    spawn = SimpleNamespace(x=x, y=y, z=0.0, spawn_id=spawn_id, pos=Point(x, y, 0.0))
    return SimpleNamespace(is_patrolling=is_patrolling, is_threat=is_threat, spawn=spawn)


def _make_tracker(vx: float = 0.0, vy: float = 0.0) -> MagicMock:
    tracker = MagicMock()
    tracker.velocity.return_value = (vx, vy)
    return tracker


# ---------------------------------------------------------------------------
# PatrolMixin: trace management
# ---------------------------------------------------------------------------


class TestPatrolMixinTrace:
    def test_initializes_trace_on_first_update(self) -> None:
        mob = FakeMobHistory()
        assert mob.patrol_trace is None
        mob._update_patrol_trace(0.0, Point(100.0, 200.0, 0.0))
        assert mob.patrol_trace is not None
        assert len(mob.patrol_trace) == 1

    def test_sparse_sampling_2s(self) -> None:
        mob = FakeMobHistory()
        mob._update_patrol_trace(0.0, Point(0.0, 0.0, 0.0))
        mob._update_patrol_trace(1.0, Point(1.0, 1.0, 0.0))  # <2s gap, skipped
        mob._update_patrol_trace(2.0, Point(2.0, 2.0, 0.0))  # >=2s gap, added
        assert len(mob.patrol_trace) == 2

    def test_trims_to_60s_window(self) -> None:
        mob = FakeMobHistory()
        # Add samples over 70 seconds
        for t in range(0, 70, 2):
            mob._update_patrol_trace(float(t), Point(0.0, 0.0, 0.0))
        # All samples should be within PATROL_TRACE_WINDOW of the last time
        oldest_t = mob.patrol_trace[0][0]
        newest_t = mob.patrol_trace[-1][0]
        assert newest_t - oldest_t <= PATROL_TRACE_WINDOW


# ---------------------------------------------------------------------------
# PatrolMixin: patrol detection
# ---------------------------------------------------------------------------


class TestPatrolDetection:
    def test_not_enough_samples(self) -> None:
        mob = FakeMobHistory()
        for t in range(4):
            mob._update_patrol_trace(float(t * 3), Point(0.0, 0.0, 0.0))
        mob._detect_patrol()
        assert mob.patrol_period == 0.0

    def test_stationary_npc_no_patrol(self) -> None:
        """NPC that stays in one place should not be flagged as patrolling."""
        mob = FakeMobHistory()
        for t in range(0, 30, 2):
            mob._update_patrol_trace(float(t), Point(100.0, 100.0, 0.0))
        mob._detect_patrol()
        assert mob.patrol_period == 0.0

    def test_patrol_cycle_detected(self) -> None:
        """NPC walks 80u away and returns -- should detect patrol."""
        mob = FakeMobHistory()
        # Start at origin
        mob._update_patrol_trace(0.0, Point(0.0, 0.0, 0.0))
        # Walk to 80u away (exceeds 60u displacement threshold)
        mob._update_patrol_trace(4.0, Point(80.0, 0.0, 0.0))
        mob._update_patrol_trace(6.0, Point(80.0, 0.0, 0.0))
        mob._update_patrol_trace(8.0, Point(80.0, 0.0, 0.0))
        # Walk back to within PATROL_RETURN_DIST of start
        mob._update_patrol_trace(12.0, Point(10.0, 0.0, 0.0))
        mob._detect_patrol()
        assert mob.patrol_period > 0.0

    def test_no_cycle_when_not_returned(self) -> None:
        """NPC that walks away without returning should not be patrol."""
        mob = FakeMobHistory()
        mob._update_patrol_trace(0.0, Point(0.0, 0.0, 0.0))
        mob._update_patrol_trace(4.0, Point(80.0, 0.0, 0.0))
        mob._update_patrol_trace(8.0, Point(160.0, 0.0, 0.0))
        mob._update_patrol_trace(12.0, Point(240.0, 0.0, 0.0))
        mob._update_patrol_trace(16.0, Point(320.0, 0.0, 0.0))
        mob._detect_patrol()
        assert mob.patrol_period == 0.0

    def test_short_cycle_ignored(self) -> None:
        """Return within <10s should not count (too recent for a full cycle)."""
        mob = FakeMobHistory()
        mob._update_patrol_trace(0.0, Point(0.0, 0.0, 0.0))
        mob._update_patrol_trace(2.0, Point(80.0, 0.0, 0.0))
        mob._update_patrol_trace(4.0, Point(80.0, 0.0, 0.0))
        mob._update_patrol_trace(6.0, Point(0.0, 0.0, 0.0))
        mob._update_patrol_trace(8.0, Point(0.0, 0.0, 0.0))  # returned but dt=8 < 10
        mob._detect_patrol()
        assert mob.patrol_period == 0.0

    def test_update_triggers_detection_every_10s(self) -> None:
        """_detect_patrol is called every 10s via _update_patrol_trace."""
        mob = FakeMobHistory()
        mob._patrol_checked = 0.0
        # Build a patrol path
        mob._update_patrol_trace(0.0, Point(0.0, 0.0, 0.0))
        mob._update_patrol_trace(4.0, Point(80.0, 0.0, 0.0))
        mob._update_patrol_trace(6.0, Point(80.0, 0.0, 0.0))
        mob._update_patrol_trace(8.0, Point(80.0, 0.0, 0.0))
        # This update at t=12 triggers check (12 - 0 >= 10)
        mob._update_patrol_trace(12.0, Point(5.0, 0.0, 0.0))
        assert mob.patrol_period > 0.0


# ---------------------------------------------------------------------------
# patrol_safe_window
# ---------------------------------------------------------------------------


class TestPatrolSafeWindow:
    def test_no_patrols_returns_inf(self) -> None:
        profiles = [_make_profile(is_patrolling=False)]
        result = patrol_safe_window(profiles, {}, Point(0.0, 0.0, 0.0), 30.0)
        assert result == float("inf")

    def test_patrol_already_in_range(self) -> None:
        # Spawn at (50, 0), target at (0, 0) => distance = 50 < threat_radius=80
        profiles = [_make_profile(is_patrolling=True, is_threat=True, x=50.0, y=0.0)]
        trackers = {}
        result = patrol_safe_window(profiles, trackers, Point(0.0, 0.0, 0.0), 30.0, threat_radius=80.0)
        assert result == 0.0

    def test_approaching_patrol_returns_eta(self) -> None:
        # Spawn at (200, 0), target at (0, 0) => distance = 200
        profiles = [_make_profile(is_patrolling=True, is_threat=True, x=200.0, y=0.0, spawn_id=1)]
        tracker = _make_tracker(vx=-10.0, vy=0.0)  # speed = 10
        trackers = {1: tracker}
        result = patrol_safe_window(profiles, trackers, Point(0.0, 0.0, 0.0), 30.0, threat_radius=80.0)
        # approach_dist = 200 - 80 = 120, speed = 10, eta = 12
        assert result == pytest.approx(12.0)

    def test_stationary_patrol_ignored(self) -> None:
        # Spawn at (200, 0), target at (0, 0) => distance = 200 > threat_radius
        profiles = [_make_profile(is_patrolling=True, is_threat=True, x=200.0, y=0.0, spawn_id=1)]
        tracker = _make_tracker(vx=0.0, vy=0.0)  # stationary (speed < 0.5)
        trackers = {1: tracker}
        result = patrol_safe_window(profiles, trackers, Point(0.0, 0.0, 0.0), 30.0)
        assert result == float("inf")

    def test_no_tracker_skipped(self) -> None:
        # Spawn at (200, 0), target at (0, 0) but no tracker for spawn_id=99
        profiles = [_make_profile(is_patrolling=True, is_threat=True, x=200.0, y=0.0, spawn_id=99)]
        trackers = {}
        result = patrol_safe_window(profiles, trackers, Point(0.0, 0.0, 0.0), 30.0)
        assert result == float("inf")


# ---------------------------------------------------------------------------
# patrolling_threats
# ---------------------------------------------------------------------------


class TestPatrollingThreats:
    def test_filters_patrolling_threats(self) -> None:
        profiles = [
            _make_profile(is_patrolling=True, is_threat=True, spawn_id=1),
            _make_profile(is_patrolling=False, is_threat=True, spawn_id=2),
            _make_profile(is_patrolling=True, is_threat=False, spawn_id=3),
            _make_profile(is_patrolling=True, is_threat=True, spawn_id=4),
        ]
        result = patrolling_threats(profiles)
        ids = [p.spawn.spawn_id for p in result]
        assert ids == [1, 4]

    def test_empty_profiles(self) -> None:
        assert patrolling_threats([]) == []
