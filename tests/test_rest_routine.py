"""Tests for RestRoutine motor-coupled lifecycle.

Verifies enter/tick/exit produce the expected motor commands via
RecordingMotor. No sleeping, no input -- all actions recorded.
"""

from __future__ import annotations

from unittest.mock import patch

from brain.context import AgentContext
from motor.recording import RecordingMotor
from routines.base import RoutineStatus
from routines.rest import RestRoutine
from tests.factories import make_game_state


def _noop_sleep(base: float, interrupt_fn=None, poll_interval=0.1, sigma=0.3) -> bool:
    """No-op replacement for interruptible_sleep in tests."""
    return False


class TestRestEnter:
    """RestRoutine.enter() motor actions."""

    @patch("routines.rest.interruptible_sleep", _noop_sleep)
    def test_enter_sits_when_standing(self, _recording_motor: RecordingMotor) -> None:
        """enter() sends sit_stand when the player is standing."""
        ctx = AgentContext()
        state = make_game_state(hp_current=500, hp_max=1000, stand_state=0)
        routine = RestRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine.enter(state)
        assert "sit_stand" in _recording_motor.actions

    @patch("routines.rest.interruptible_sleep", _noop_sleep)
    def test_enter_skips_sit_when_already_sitting(self, _recording_motor: RecordingMotor) -> None:
        """enter() does not toggle sit_stand if player is already sitting."""
        ctx = AgentContext()
        state = make_game_state(hp_current=500, hp_max=1000, stand_state=1)
        routine = RestRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine.enter(state)
        # Should NOT send sit_stand -- player already sitting
        assert "sit_stand" not in _recording_motor.actions

    @patch("routines.rest.interruptible_sleep", _noop_sleep)
    def test_enter_increments_rest_count(self, _recording_motor: RecordingMotor) -> None:
        """enter() bumps ctx.metrics.rest_count."""
        ctx = AgentContext()
        state = make_game_state(hp_current=500, hp_max=1000, stand_state=0)
        routine = RestRoutine(ctx=ctx, read_state_fn=lambda: state)
        assert ctx.metrics.rest_count == 0
        routine.enter(state)
        assert ctx.metrics.rest_count == 1


class TestRestTick:
    """RestRoutine.tick() regen polling."""

    @patch("routines.rest.interruptible_sleep", _noop_sleep)
    def test_tick_running_while_hp_low(self, _recording_motor: RecordingMotor) -> None:
        """tick() returns RUNNING when HP is below threshold."""
        ctx = AgentContext()
        state = make_game_state(hp_current=500, hp_max=1000, mana_current=500, mana_max=500)
        routine = RestRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine.enter(state)
        status = routine.tick(state)
        assert status == RoutineStatus.RUNNING

    @patch("routines.rest.interruptible_sleep", _noop_sleep)
    def test_tick_running_while_mana_low(self, _recording_motor: RecordingMotor) -> None:
        """tick() returns RUNNING when mana is below threshold."""
        ctx = AgentContext()
        state = make_game_state(hp_current=1000, hp_max=1000, mana_current=200, mana_max=500)
        routine = RestRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine.enter(state)
        status = routine.tick(state)
        assert status == RoutineStatus.RUNNING

    @patch("routines.rest.interruptible_sleep", _noop_sleep)
    def test_tick_success_when_fully_recovered(self, _recording_motor: RecordingMotor) -> None:
        """tick() returns SUCCESS when HP and mana are above thresholds."""
        ctx = AgentContext()
        # Enter with low state, then tick with recovered state
        low_state = make_game_state(hp_current=500, hp_max=1000, mana_current=200, mana_max=500)
        routine = RestRoutine(ctx=ctx, read_state_fn=lambda: low_state)
        routine.enter(low_state)

        # Now tick with full HP/mana
        full_state = make_game_state(hp_current=1000, hp_max=1000, mana_current=500, mana_max=500)
        status = routine.tick(full_state)
        assert status == RoutineStatus.SUCCESS

    @patch("routines.rest.interruptible_sleep", _noop_sleep)
    def test_tick_no_motor_actions_during_normal_regen(self, _recording_motor: RecordingMotor) -> None:
        """tick() produces no motor commands when just waiting for regen."""
        ctx = AgentContext()
        state = make_game_state(hp_current=700, hp_max=1000, mana_current=200, mana_max=500)
        routine = RestRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine.enter(state)
        _recording_motor.clear()
        routine.tick(state)
        # No motor actions during passive regen tick (no combat, no heal needed)
        assert _recording_motor.actions == []


class TestRestExit:
    """RestRoutine.exit() motor actions."""

    @patch("routines.rest.interruptible_sleep", _noop_sleep)
    def test_exit_stands_when_sitting(self, _recording_motor: RecordingMotor) -> None:
        """exit() sends sit_stand to stand up when player is sitting."""
        ctx = AgentContext()
        sitting_state = make_game_state(
            hp_current=1000, hp_max=1000, mana_current=500, mana_max=500, stand_state=1
        )
        routine = RestRoutine(ctx=ctx, read_state_fn=lambda: sitting_state)
        # We need the stance tracker to think we're sitting
        # enter() with a standing state will call sit(), setting _stance.sitting = True
        standing_state = make_game_state(
            hp_current=500, hp_max=1000, mana_current=200, mana_max=500, stand_state=0
        )
        routine.enter(standing_state)
        _recording_motor.clear()
        # exit with is_sitting=True triggers stand()
        routine.exit(sitting_state)
        assert "sit_stand" in _recording_motor.actions

    @patch("routines.rest.interruptible_sleep", _noop_sleep)
    def test_exit_skips_stand_when_already_standing(self, _recording_motor: RecordingMotor) -> None:
        """exit() does not send stand if player already standing."""
        ctx = AgentContext()
        standing_state = make_game_state(
            hp_current=1000, hp_max=1000, mana_current=500, mana_max=500, stand_state=0
        )
        routine = RestRoutine(ctx=ctx, read_state_fn=lambda: standing_state)
        # Enter while already sitting so we do NOT call sit() (no stance toggle)
        sitting_enter = make_game_state(hp_current=500, hp_max=1000, stand_state=1)
        routine.enter(sitting_enter)
        _recording_motor.clear()
        # Exit with standing state -- is_sitting is False, so no stand needed
        routine.exit(standing_state)
        assert "sit_stand" not in _recording_motor.actions


class TestRestFullCycle:
    """Full enter -> tick -> exit lifecycle."""

    @patch("routines.rest.interruptible_sleep", _noop_sleep)
    def test_full_rest_cycle(self, _recording_motor: RecordingMotor) -> None:
        """Complete rest cycle: sit, regen, stand."""
        ctx = AgentContext()
        low_state = make_game_state(
            hp_current=500, hp_max=1000, mana_current=200, mana_max=500, stand_state=0
        )
        full_state = make_game_state(
            hp_current=1000, hp_max=1000, mana_current=500, mana_max=500, stand_state=1
        )
        routine = RestRoutine(ctx=ctx, read_state_fn=lambda: low_state)

        routine.enter(low_state)
        assert "sit_stand" in _recording_motor.actions  # sat down

        # Tick while resting
        status = routine.tick(low_state)
        assert status == RoutineStatus.RUNNING

        # Tick when recovered
        status = routine.tick(full_state)
        assert status == RoutineStatus.SUCCESS

        _recording_motor.clear()
        routine.exit(full_state)
        # Should stand up (is_sitting=True from enter, state says sitting)
        assert "sit_stand" in _recording_motor.actions
