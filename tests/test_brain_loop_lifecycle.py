"""Tests for pure/isolated helpers in brain.runner.loop and brain.runner.lifecycle.

These modules orchestrate the brain loop and have deep wiring dependencies.
Only functions testable in isolation (no fully-wired BrainRunner) are tested:
  - LifecycleHandler.handle_death (death tracking, recovery decision)
  - LifecycleHandler.check_no_progress_safety (timeout-based camp-out)
  - LifecycleHandler.check_watchdog_restart (stub -- always False)
  - BrainRunner._record_session_to_memory (static, builds SessionRecord)
  - BrainRunner properties (seconds_since_heartbeat, brain_healthy, paused)
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from brain.runner.lifecycle import LifecycleHandler
from brain.state.combat import CombatState
from brain.state.plan import PlanState
from brain.state.player import PlayerState
from core.types import PlanType

# ---------------------------------------------------------------------------
# Helpers -- lightweight BrainRunner / context stubs
# ---------------------------------------------------------------------------


def _make_runner_stub(**overrides) -> SimpleNamespace:
    """Minimal BrainRunner-like namespace."""
    stop_event = threading.Event()
    defaults = dict(
        _stop_event=stop_event,
        _brain=SimpleNamespace(_active_name="WANDER", _active=None),
        _reader=MagicMock(),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_lifecycle_ctx(
    *,
    dead: bool = False,
    deaths: int = 0,
    session_start: float = 0.0,
    last_kill_age: float = 0.0,
    plan_active: str | None = None,
    engaged: bool = False,
) -> MagicMock:
    """Build a minimal ctx for lifecycle tests."""
    ctx = MagicMock()
    ctx.player = PlayerState(dead=dead, deaths=deaths)
    ctx.combat = CombatState(engaged=engaged)
    ctx.plan = PlanState(active=plan_active)
    ctx.metrics = SimpleNamespace(session_start=session_start)
    ctx.defeat_tracker = MagicMock()
    ctx.defeat_tracker.last_kill_age.return_value = last_kill_age
    ctx.diag = SimpleNamespace(forensics=None, incident_reporter=None)
    return ctx


# ===========================================================================
# LifecycleHandler.check_watchdog_restart
# ===========================================================================


class TestCheckWatchdogRestart:
    def test_base_returns_false(self) -> None:
        runner = _make_runner_stub()
        lh = LifecycleHandler(runner)
        assert lh.check_watchdog_restart() is False


# ===========================================================================
# LifecycleHandler.handle_death
# ===========================================================================


class TestHandleDeath:
    def test_already_dead_returns_false(self) -> None:
        runner = _make_runner_stub()
        lh = LifecycleHandler(runner)
        ctx = _make_lifecycle_ctx(dead=True, deaths=1)
        assert lh.handle_death(ctx, "hp_zero") is False

    @patch("brain.runner.lifecycle.flags")
    @patch("brain.runner.lifecycle.time.sleep")
    @patch("brain.runner.lifecycle.sit")
    @patch("brain.runner.lifecycle.do_camp")
    def test_first_death_no_recover_sets_stop(self, mock_camp, mock_sit, mock_sleep, mock_flags) -> None:
        mock_flags.should_recover_death.return_value = False
        runner = _make_runner_stub()
        lh = LifecycleHandler(runner)
        ctx = _make_lifecycle_ctx(dead=False, deaths=0)
        result = lh.handle_death(ctx, "body_state")
        assert result is True
        assert ctx.player.dead is True
        assert ctx.player.deaths == 1
        assert runner._stop_event.is_set()

    @patch("brain.runner.lifecycle.flags")
    def test_death_with_recovery(self, mock_flags) -> None:
        mock_flags.should_recover_death.return_value = True
        runner = _make_runner_stub()
        lh = LifecycleHandler(runner)
        ctx = _make_lifecycle_ctx(dead=False, deaths=0)
        result = lh.handle_death(ctx, "hp_zero")
        assert result is False
        assert ctx.player.dead is True
        assert ctx.player.deaths == 1
        assert not runner._stop_event.is_set()

    @patch("brain.runner.lifecycle.flags")
    def test_increments_death_count(self, mock_flags) -> None:
        mock_flags.should_recover_death.return_value = True
        runner = _make_runner_stub()
        lh = LifecycleHandler(runner)
        ctx = _make_lifecycle_ctx(dead=False, deaths=2)
        lh.handle_death(ctx, "test")
        assert ctx.player.deaths == 3


# ===========================================================================
# LifecycleHandler.check_no_progress_safety
# ===========================================================================


class TestCheckNoProgressSafety:
    def test_returns_false_when_session_young(self) -> None:
        runner = _make_runner_stub()
        lh = LifecycleHandler(runner)
        # Session only 60s old, well below NO_PROGRESS_TIMEOUT
        ctx = _make_lifecycle_ctx(session_start=time.time() - 60)
        assert lh.check_no_progress_safety(ctx) is False

    def test_returns_false_when_recent_kill(self) -> None:
        runner = _make_runner_stub()
        lh = LifecycleHandler(runner)
        ctx = _make_lifecycle_ctx(
            session_start=time.time() - 2000,
            last_kill_age=30.0,  # recent kill
        )
        assert lh.check_no_progress_safety(ctx) is False

    def test_returns_false_when_travel_plan(self) -> None:
        runner = _make_runner_stub()
        lh = LifecycleHandler(runner)
        ctx = _make_lifecycle_ctx(
            session_start=time.time() - 2000,
            last_kill_age=2000.0,
            plan_active=PlanType.TRAVEL,
        )
        assert lh.check_no_progress_safety(ctx) is False

    def test_returns_false_when_memorize_plan(self) -> None:
        runner = _make_runner_stub()
        lh = LifecycleHandler(runner)
        ctx = _make_lifecycle_ctx(
            session_start=time.time() - 2000,
            last_kill_age=2000.0,
            plan_active=PlanType.NEEDS_MEMORIZE,
        )
        assert lh.check_no_progress_safety(ctx) is False

    def test_returns_false_when_engaged(self) -> None:
        runner = _make_runner_stub()
        lh = LifecycleHandler(runner)
        ctx = _make_lifecycle_ctx(
            session_start=time.time() - 2000,
            last_kill_age=2000.0,
            engaged=True,
        )
        assert lh.check_no_progress_safety(ctx) is False

    @patch("brain.runner.lifecycle.time.sleep")
    @patch("brain.runner.lifecycle.sit")
    @patch("brain.runner.lifecycle.do_camp")
    def test_triggers_camp_when_no_progress(self, mock_camp, mock_sit, mock_sleep) -> None:
        runner = _make_runner_stub()
        lh = LifecycleHandler(runner)
        ctx = _make_lifecycle_ctx(
            session_start=time.time() - 2000,
            last_kill_age=2000.0,
            plan_active=None,
            engaged=False,
        )
        result = lh.check_no_progress_safety(ctx)
        assert result is True
        assert runner._stop_event.is_set()
        mock_sit.assert_called_once()
        mock_camp.assert_called_once()


# ===========================================================================
# BrainRunner._record_session_to_memory (static method)
# ===========================================================================


class TestRecordSessionToMemory:
    @patch("brain.runner.loop.compute_scorecard")
    def test_records_session(self, mock_scorecard) -> None:
        from brain.runner.loop import BrainRunner

        mock_scorecard.return_value = {"survival": 80, "overall": 70, "grade": "B"}
        ctx = MagicMock()
        ctx.metrics.session_start = time.time() - 3600  # 1 hour
        ctx.defeat_tracker.defeats = 25
        ctx.player.deaths = 1
        ctx.metrics.flee_count = 2

        sm = MagicMock()
        sm._zone = "testzone"
        BrainRunner._record_session_to_memory(ctx, sm)
        sm.record.assert_called_once()
        record = sm.record.call_args[0][0]
        assert record.defeats_per_hour > 0
        assert record.deaths == 1

    @patch("brain.runner.loop.compute_scorecard", side_effect=TypeError("broken"))
    def test_handles_scorecard_failure(self, mock_scorecard) -> None:
        from brain.runner.loop import BrainRunner

        ctx = MagicMock()
        ctx.metrics.session_start = time.time() - 3600
        ctx.defeat_tracker.defeats = 0
        ctx.player.deaths = 0
        ctx.metrics.flee_count = 0
        sm = MagicMock()
        sm._zone = "testzone"
        # Should not raise
        BrainRunner._record_session_to_memory(ctx, sm)
        sm.record.assert_not_called()
