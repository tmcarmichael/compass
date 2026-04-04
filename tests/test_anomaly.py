"""Tests for brain.world.anomaly -- real-time anomaly detector.

Tests each detector (camp drift, acquire loop, heading lock, mana depletion,
defeat drought, fizzle streak, stuck events), debounce logic, issue severity,
and the check() orchestrator.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from brain.world.anomaly import (
    AnomalyDetector,
    Issue,
    IssueSeverity,
    IssueType,
)
from core.types import Point
from tests.factories import make_game_state

# ---------------------------------------------------------------------------
# Lightweight stub for AgentContext -- only the fields anomaly.py reads
# ---------------------------------------------------------------------------


def _make_camp(camp_x: float = 100.0, camp_y: float = 100.0, roam_radius: float = 100.0) -> Any:
    ns = SimpleNamespace(camp_pos=Point(camp_x, camp_y, 0.0), roam_radius=roam_radius)
    ns.distance_to_camp = lambda state: 50.0  # default, overridden per-test
    return ns


def _make_combat(engaged: bool = False) -> Any:
    return SimpleNamespace(engaged=engaged)


def _make_plan(active: str = "") -> Any:
    return SimpleNamespace(active=active, travel=SimpleNamespace(target_x=0, target_y=0))


def _make_metrics(
    consecutive_acquire_fails: int = 0,
    rest_count: int = 0,
    pull_dc_fizzles: int = 0,
    session_start: float | None = None,
) -> Any:
    return SimpleNamespace(
        consecutive_acquire_fails=consecutive_acquire_fails,
        rest_count=rest_count,
        pull_dc_fizzles=pull_dc_fizzles,
        session_start=session_start or time.time() - 300,
    )


def _make_ctx(
    camp_x: float = 100.0,
    camp_y: float = 100.0,
    dist_to_camp: float = 50.0,
    engaged: bool = False,
    acquire_fails: int = 0,
    rest_count: int = 0,
    fizzles: int = 0,
    last_kill_age_val: float = 10.0,
    roam_radius: float = 100.0,
) -> Any:
    camp = _make_camp(camp_x, camp_y, roam_radius)
    camp.distance_to_camp = lambda state: dist_to_camp
    defeat_tracker = SimpleNamespace()
    defeat_tracker.last_kill_age = lambda: last_kill_age_val
    ctx = SimpleNamespace(
        camp=camp,
        combat=_make_combat(engaged),
        plan=_make_plan(),
        metrics=_make_metrics(acquire_fails, rest_count, fizzles),
        defeat_tracker=defeat_tracker,
    )
    return ctx


# ---------------------------------------------------------------------------
# Issue dataclass
# ---------------------------------------------------------------------------


class TestIssue:
    def test_issue_creation(self) -> None:
        i = Issue(
            type=IssueType.CAMP_DRIFT,
            message="test",
            severity=IssueSeverity.WARNING,
            timestamp=1.0,
        )
        assert i.type == IssueType.CAMP_DRIFT
        assert i.severity == IssueSeverity.WARNING

    def test_issue_frozen(self) -> None:
        i: Any = Issue(type=IssueType.CAMP_DRIFT, message="x", severity=IssueSeverity.INFO, timestamp=0)
        with pytest.raises(AttributeError):
            i.message = "y"


# ---------------------------------------------------------------------------
# Camp drift
# ---------------------------------------------------------------------------


class TestCampDrift:
    def test_no_issue_within_range(self) -> None:
        ctx = _make_ctx(dist_to_camp=100)
        det = AnomalyDetector(ctx=ctx)
        state = make_game_state()
        issues = det.check(state)
        assert not any(i.type == IssueType.CAMP_DRIFT for i in issues)

    def test_warning_at_moderate_drift(self) -> None:
        ctx = _make_ctx(dist_to_camp=400)
        det = AnomalyDetector(ctx=ctx)
        state = make_game_state()
        issues = det.check(state)
        drift_issues = [i for i in issues if i.type == IssueType.CAMP_DRIFT]
        assert len(drift_issues) == 1
        assert drift_issues[0].severity == IssueSeverity.WARNING

    def test_critical_at_severe_drift(self) -> None:
        ctx = _make_ctx(dist_to_camp=700)
        det = AnomalyDetector(ctx=ctx)
        state = make_game_state()
        issues = det.check(state)
        drift_issues = [i for i in issues if i.type == IssueType.CAMP_DRIFT]
        assert len(drift_issues) == 1
        assert drift_issues[0].severity == IssueSeverity.CRITICAL

    def test_no_camp_no_issue(self) -> None:
        ctx = _make_ctx(camp_x=0, camp_y=0, dist_to_camp=999)
        det = AnomalyDetector(ctx=ctx)
        state = make_game_state()
        issues = det.check(state)
        assert not any(i.type == IssueType.CAMP_DRIFT for i in issues)


# ---------------------------------------------------------------------------
# Acquire loop
# ---------------------------------------------------------------------------


class TestAcquireLoop:
    def test_no_issue_below_threshold(self) -> None:
        ctx = _make_ctx(acquire_fails=3)
        det = AnomalyDetector(ctx=ctx)
        issues = det.check(make_game_state())
        assert not any(i.type == IssueType.ACQUIRE_LOOP for i in issues)

    def test_warning_at_5_fails(self) -> None:
        ctx = _make_ctx(acquire_fails=5)
        det = AnomalyDetector(ctx=ctx)
        issues = det.check(make_game_state())
        acq = [i for i in issues if i.type == IssueType.ACQUIRE_LOOP]
        assert len(acq) == 1
        assert acq[0].severity == IssueSeverity.WARNING

    def test_critical_at_10_fails(self) -> None:
        ctx = _make_ctx(acquire_fails=10)
        det = AnomalyDetector(ctx=ctx)
        issues = det.check(make_game_state())
        acq = [i for i in issues if i.type == IssueType.ACQUIRE_LOOP]
        assert len(acq) == 1
        assert acq[0].severity == IssueSeverity.CRITICAL


# ---------------------------------------------------------------------------
# Heading lock
# ---------------------------------------------------------------------------


class TestHeadingLock:
    def test_no_issue_with_changing_heading(self) -> None:
        ctx = _make_ctx()
        det = AnomalyDetector(ctx=ctx)
        for h in [10, 20, 30, 40]:
            issues = det.check(make_game_state(heading=float(h)))
        assert not any(i.type == IssueType.HEADING_LOCK for i in issues)

    def test_heading_lock_after_3_same(self) -> None:
        ctx = _make_ctx()
        det = AnomalyDetector(ctx=ctx)
        # Need 4 checks total: first sets baseline, then 3 same = stuck_count reaches 3
        for _ in range(4):
            issues = det.check(make_game_state(heading=100.0))
        assert any(i.type == IssueType.HEADING_LOCK for i in issues)


# ---------------------------------------------------------------------------
# Mana depletion
# ---------------------------------------------------------------------------


class TestManaDepletion:
    def test_no_issue_with_healthy_mana(self) -> None:
        ctx = _make_ctx()
        det = AnomalyDetector(ctx=ctx)
        issues = det.check(make_game_state(mana_current=400, mana_max=500))
        assert not any(i.type == IssueType.MANA_DEPLETION for i in issues)

    def test_issue_at_low_mana_no_rests(self) -> None:
        ctx = _make_ctx(rest_count=0)
        det = AnomalyDetector(ctx=ctx)
        # mana_pct = 50/500 = 0.10 < 0.15
        issues = det.check(make_game_state(mana_current=50, mana_max=500))
        assert any(i.type == IssueType.MANA_DEPLETION for i in issues)

    def test_no_issue_during_combat(self) -> None:
        ctx = _make_ctx(rest_count=0, engaged=True)
        det = AnomalyDetector(ctx=ctx)
        issues = det.check(make_game_state(mana_current=10, mana_max=500))
        assert not any(i.type == IssueType.MANA_DEPLETION for i in issues)


# ---------------------------------------------------------------------------
# Kill drought
# ---------------------------------------------------------------------------


class TestKillDrought:
    def test_no_issue_with_recent_kill(self) -> None:
        ctx = _make_ctx(last_kill_age_val=30.0)
        det = AnomalyDetector(ctx=ctx)
        issues = det.check(make_game_state())
        assert not any(i.type == IssueType.DEFEAT_DROUGHT for i in issues)

    def test_warning_at_extended_drought(self) -> None:
        ctx = _make_ctx(last_kill_age_val=200.0)
        det = AnomalyDetector(ctx=ctx)
        issues = det.check(make_game_state())
        drought = [i for i in issues if i.type == IssueType.DEFEAT_DROUGHT]
        assert len(drought) == 1
        assert drought[0].severity == IssueSeverity.WARNING

    def test_critical_at_severe_drought(self) -> None:
        ctx = _make_ctx(last_kill_age_val=400.0)
        det = AnomalyDetector(ctx=ctx)
        issues = det.check(make_game_state())
        drought = [i for i in issues if i.type == IssueType.DEFEAT_DROUGHT]
        assert len(drought) == 1
        assert drought[0].severity == IssueSeverity.CRITICAL


# ---------------------------------------------------------------------------
# Fizzle streak
# ---------------------------------------------------------------------------


class TestFizzleStreak:
    def test_no_issue_below_threshold(self) -> None:
        ctx = _make_ctx(fizzles=2)
        det = AnomalyDetector(ctx=ctx)
        issues = det.check(make_game_state())
        assert not any(i.type == IssueType.FIZZLE_STREAK for i in issues)

    def test_issue_at_threshold(self) -> None:
        ctx = _make_ctx(fizzles=4)
        det = AnomalyDetector(ctx=ctx)
        issues = det.check(make_game_state())
        assert any(i.type == IssueType.FIZZLE_STREAK for i in issues)


# ---------------------------------------------------------------------------
# Debounce
# ---------------------------------------------------------------------------


class TestDebounce:
    def test_same_issue_debounced(self) -> None:
        ctx = _make_ctx(acquire_fails=5)
        det = AnomalyDetector(ctx=ctx)
        issues1 = det.check(make_game_state())
        assert any(i.type == IssueType.ACQUIRE_LOOP for i in issues1)
        # Second check within debounce window should not re-fire
        issues2 = det.check(make_game_state())
        assert not any(i.type == IssueType.ACQUIRE_LOOP for i in issues2)

    def test_different_issues_not_debounced(self) -> None:
        ctx = _make_ctx(acquire_fails=5, fizzles=4)
        det = AnomalyDetector(ctx=ctx)
        issues = det.check(make_game_state())
        types = {i.type for i in issues}
        assert IssueType.ACQUIRE_LOOP in types
        assert IssueType.FIZZLE_STREAK in types


# ---------------------------------------------------------------------------
# Orchestration: active_issues and history
# ---------------------------------------------------------------------------


class TestCheckOrchestration:
    def test_active_issues_updated(self) -> None:
        ctx = _make_ctx(fizzles=4)
        det = AnomalyDetector(ctx=ctx)
        det.check(make_game_state())
        assert len(det.active_issues) >= 1

    def test_history_accumulates(self) -> None:
        ctx = _make_ctx(acquire_fails=5)
        det = AnomalyDetector(ctx=ctx)
        det.check(make_game_state())
        # Second call is debounced, but first added to history
        assert len(det._issue_history) >= 1

    def test_history_pruned_at_max(self) -> None:
        ctx = _make_ctx(fizzles=4)
        det = AnomalyDetector(ctx=ctx, _max_history=10, _debounce_seconds=0.0)
        for _ in range(15):
            det.check(make_game_state())
        assert len(det._issue_history) <= 10
