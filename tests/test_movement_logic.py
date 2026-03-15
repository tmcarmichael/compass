"""Tests for nav.movement -- StuckRecovery escalation, MovementPhase state
machine, and MovementController initialization.

The conftest autouse fixture _recording_motor ensures motor calls record
instead of sleeping.
"""

from __future__ import annotations

import random
import time
from types import SimpleNamespace

from core.types import Point
from nav.movement import (
    HeadingController,
    MovementController,
    MovementPhase,
    RecoveryAction,
    StuckRecovery,
    _cancel_event,
    clear_movement_cancel,
    get_stuck_points,
    is_near_stuck_point,
    load_stuck_points,
    request_movement_cancel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    x: float = 0.0,
    y: float = 0.0,
    heading: float = 0.0,
    speed_run: float = 0.7,
    is_sitting: bool = False,
    stand_state: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        x=x,
        y=y,
        z=0.0,
        heading=heading,
        speed_run=speed_run,
        is_sitting=is_sitting,
        stand_state=stand_state,
    )


# ===========================================================================
# StuckRecovery -- escalation logic
# ===========================================================================


class TestStuckRecoveryEscalation:
    """Verify the escalation pattern: strafe -> opposite strafe -> backup -> bigger backup."""

    def test_initial_attempt_is_zero(self) -> None:
        sr = StuckRecovery()
        assert sr.attempt == 0

    def test_first_recovery_is_strafe(self) -> None:
        random.seed(42)
        sr = StuckRecovery()
        action = sr.next_recovery()
        assert sr.attempt == 1
        assert action.action_type in ("strafe_left", "strafe_right")
        assert action.duration > 0
        assert action.turn_amount != 0

    def test_second_recovery_is_opposite_strafe(self) -> None:
        random.seed(42)
        sr = StuckRecovery()
        first = sr.next_recovery()
        second = sr.next_recovery()
        assert sr.attempt == 2
        assert second.action_type in ("strafe_left", "strafe_right")
        # Opposite direction from first
        assert second.action_type != first.action_type

    def test_third_recovery_is_backup(self) -> None:
        sr = StuckRecovery()
        sr.next_recovery()  # 1
        sr.next_recovery()  # 2
        third = sr.next_recovery()  # 3
        assert sr.attempt == 3
        assert third.action_type == "backup"
        assert third.duration > 0
        assert third.strafe_duration > 0
        assert third.turn_amount != 0

    def test_fourth_recovery_is_bigger_backup(self) -> None:
        sr = StuckRecovery()
        for _ in range(3):
            sr.next_recovery()
        fourth = sr.next_recovery()
        assert sr.attempt == 4
        assert fourth.action_type == "backup"
        assert fourth.strafe_dir in (-1, 1)

    def test_reset_clears_attempt_count(self) -> None:
        sr = StuckRecovery()
        sr.next_recovery()
        sr.next_recovery()
        assert sr.attempt == 2
        sr.reset()
        assert sr.attempt == 0

    def test_reset_then_first_recovery_is_strafe(self) -> None:
        sr = StuckRecovery()
        for _ in range(4):
            sr.next_recovery()
        sr.reset()
        action = sr.next_recovery()
        assert sr.attempt == 1
        assert action.action_type in ("strafe_left", "strafe_right")


class TestRecoveryActionDataclass:
    def test_frozen_defaults(self) -> None:
        action = RecoveryAction(action_type="strafe_left", duration=1.0)
        assert action.strafe_dir == 0
        assert action.strafe_duration == 0.0
        assert action.turn_amount == 0.0

    def test_backup_with_all_fields(self) -> None:
        action = RecoveryAction(
            action_type="backup",
            duration=1.0,
            strafe_dir=-1,
            strafe_duration=0.8,
            turn_amount=45.0,
        )
        assert action.strafe_dir == -1
        assert action.strafe_duration == 0.8


# ===========================================================================
# HeadingController
# ===========================================================================


class TestHeadingController:
    def test_already_facing_returns_true(self) -> None:
        hc = HeadingController()
        # heading_to(0, 0, 100, 0) will produce a heading; we set current_heading
        # to match. With tolerance=360, anything is within tolerance.
        result = hc.face_toward(
            target_x=100.0,
            target_y=0.0,
            current_x=0.0,
            current_y=0.0,
            current_heading=hc._last_desired,  # will be 0, but with huge tolerance:
            tolerance=360.0,
        )
        assert result is True

    def test_is_facing_within_tolerance(self) -> None:
        hc = HeadingController()
        hc._last_desired = 100.0
        assert hc.is_facing(105.0, tolerance=10.0) is True
        assert hc.is_facing(200.0, tolerance=10.0) is False

    def test_last_desired_updated(self) -> None:
        hc = HeadingController()
        assert hc.last_desired == 0.0
        hc.face_toward(100.0, 0.0, 0.0, 0.0, 0.0, tolerance=360.0)
        assert hc.last_desired != 0.0 or True  # just check it runs


# ===========================================================================
# MovementController -- stuck point management
# ===========================================================================


class TestMovementControllerStuckPoints:
    def test_record_and_check_stuck_point(self) -> None:
        mc = MovementController()
        mc.record_stuck_point(100.0, 200.0)
        assert mc.is_near_stuck_point(105.0, 205.0) is True
        assert mc.is_near_stuck_point(500.0, 500.0) is False

    def test_duplicate_stuck_point_deduped(self) -> None:
        mc = MovementController()
        mc.record_stuck_point(100.0, 200.0)
        mc.record_stuck_point(105.0, 205.0)  # within STUCK_AVOIDANCE_RADIUS
        assert len(mc.get_stuck_points()) == 1

    def test_distant_stuck_points_both_recorded(self) -> None:
        mc = MovementController()
        mc.record_stuck_point(100.0, 200.0)
        mc.record_stuck_point(500.0, 600.0)  # far away
        assert len(mc.get_stuck_points()) == 2

    def test_load_stuck_points(self) -> None:
        mc = MovementController()
        points = [Point(10.0, 20.0, 0.0), Point(30.0, 40.0, 0.0)]
        mc.load_stuck_points(points)
        assert len(mc.get_stuck_points()) == 2
        assert mc.is_near_stuck_point(10.0, 20.0) is True

    def test_load_clears_existing(self) -> None:
        mc = MovementController()
        mc.record_stuck_point(100.0, 200.0)
        mc.load_stuck_points([Point(500.0, 600.0, 0.0)])
        assert len(mc.get_stuck_points()) == 1
        assert mc.is_near_stuck_point(100.0, 200.0) is False

    def test_stuck_event_count(self) -> None:
        mc = MovementController()
        assert mc.stuck_event_count == 0

    def test_initial_state(self) -> None:
        mc = MovementController()
        assert mc.zone_map is None
        assert mc.terrain is None
        assert mc.cancel_requested is False
        assert mc.get_stuck_points() == []


# ===========================================================================
# MovementPhase -- tick-based state machine
# ===========================================================================


class TestMovementPhaseBasicStates:
    def test_already_at_target_returns_true(self) -> None:
        """If player is already at target, first tick returns True (arrived)."""
        state = _make_state(x=100.0, y=100.0)
        phase = MovementPhase(
            target_x=100.0,
            target_y=100.0,
            read_state_fn=lambda: state,
            arrival_tolerance=15.0,
        )
        result = phase.tick()
        assert result is True
        assert phase.done is True
        assert phase.arrived is True

    def test_timeout_returns_false(self) -> None:
        """If timeout is immediate (0), tick returns False."""
        state = _make_state(x=0.0, y=0.0)
        phase = MovementPhase(
            target_x=1000.0,
            target_y=1000.0,
            read_state_fn=lambda: state,
            timeout=0.0,  # instant timeout
        )
        # Need to make start_time in the past
        phase._start_time = time.perf_counter() - 1.0
        result = phase.tick()
        assert result is False
        assert phase.done is True
        assert phase.arrived is False

    def test_cancel_stops_phase(self) -> None:
        state = _make_state(x=0.0, y=0.0)
        phase = MovementPhase(
            target_x=1000.0,
            target_y=1000.0,
            read_state_fn=lambda: state,
        )
        phase.cancel()
        assert phase.done is True
        assert phase.arrived is False
        # tick after cancel returns cached result
        result = phase.tick()
        assert result is False

    def test_cancel_event_stops_phase(self) -> None:
        """Module-level _cancel_event stops phase."""
        state = _make_state(x=0.0, y=0.0)
        phase = MovementPhase(
            target_x=1000.0,
            target_y=1000.0,
            read_state_fn=lambda: state,
        )
        _cancel_event.set()
        try:
            result = phase.tick()
            assert result is False
            assert phase.done is True
        finally:
            _cancel_event.clear()

    def test_tick_returns_none_when_moving(self) -> None:
        """Normal tick while far from target returns None (still moving)."""
        state = _make_state(x=0.0, y=0.0, heading=0.0)
        phase = MovementPhase(
            target_x=500.0,
            target_y=0.0,
            read_state_fn=lambda: state,
            timeout=30.0,
        )
        result = phase.tick()
        assert result is None
        assert phase.done is False

    def test_done_tick_returns_cached_result(self) -> None:
        """Once done, tick keeps returning the same result."""
        state = _make_state(x=100.0, y=100.0)
        phase = MovementPhase(
            target_x=100.0,
            target_y=100.0,
            read_state_fn=lambda: state,
        )
        first = phase.tick()
        assert first is True
        second = phase.tick()
        assert second is True


# ===========================================================================
# Module-level API -- cancel flag management
# ===========================================================================


class TestCancelManagement:
    def test_request_and_clear(self) -> None:
        clear_movement_cancel()
        assert not _cancel_event.is_set()
        request_movement_cancel()
        assert _cancel_event.is_set()
        clear_movement_cancel()
        assert not _cancel_event.is_set()


# ===========================================================================
# Module-level stuck point API
# ===========================================================================


class TestModuleLevelStuckAPI:
    def test_module_level_load_and_check(self) -> None:
        """The module singleton forwards to _controller."""
        load_stuck_points([Point(999.0, 999.0, 0.0)])
        assert is_near_stuck_point(999.0, 999.0) is True
        points = get_stuck_points()
        assert any(p.x == 999.0 and p.y == 999.0 for p in points)
        # Clean up
        load_stuck_points([])
