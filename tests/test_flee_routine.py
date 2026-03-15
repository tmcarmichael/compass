"""Tests for FleeRoutine motor-coupled lifecycle.

Verifies enter/tick/exit produce the expected motor commands via
RecordingMotor. No sleeping, no input -- all actions recorded.
"""

from __future__ import annotations

from unittest.mock import patch

from brain.context import AgentContext
from core.types import Point
from motor.recording import RecordingMotor
from routines.base import RoutineStatus
from routines.flee import FleeRoutine
from tests.factories import make_game_state


def _noop_sleep(base: float, interrupt_fn=None, poll_interval=0.1, sigma=0.3) -> bool:
    """No-op replacement for interruptible_sleep in tests."""
    return False


class TestFleeEnter:
    """FleeRoutine.enter() motor actions."""

    @patch("routines.flee.interruptible_sleep", _noop_sleep)
    def test_enter_stops_forward_and_starts_flee(self, _recording_motor: RecordingMotor) -> None:
        """enter() stops movement, increments flee count, and locks the routine."""
        ctx = AgentContext()
        ctx.pet.alive = False  # no pet -- skip pet_back_off
        state = make_game_state(hp_current=100, hp_max=1000, mana_current=0, mana_max=500)
        routine = FleeRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine.enter(state)
        # Should stop forward movement
        assert "-forward" in _recording_motor.actions
        assert routine.locked is True

    @patch("routines.flee.interruptible_sleep", _noop_sleep)
    def test_enter_increments_flee_count(self, _recording_motor: RecordingMotor) -> None:
        """enter() bumps ctx.metrics.flee_count."""
        ctx = AgentContext()
        ctx.pet.alive = False
        state = make_game_state(hp_current=100, hp_max=1000, mana_current=0, mana_max=500)
        routine = FleeRoutine(ctx=ctx, read_state_fn=lambda: state)
        assert ctx.metrics.flee_count == 0
        routine.enter(state)
        assert ctx.metrics.flee_count == 1

    @patch("routines.flee.interruptible_sleep", _noop_sleep)
    def test_enter_clears_engagement(self, _recording_motor: RecordingMotor) -> None:
        """enter() clears combat.engaged and pull_target_id."""
        ctx = AgentContext()
        ctx.combat.engaged = True
        ctx.combat.pull_target_id = 42
        ctx.pet.alive = False
        state = make_game_state(hp_current=100, hp_max=1000, mana_current=0, mana_max=500)
        routine = FleeRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine.enter(state)
        assert ctx.combat.engaged is False
        assert ctx.combat.pull_target_id is None

    @patch("routines.flee.interruptible_sleep", _noop_sleep)
    def test_enter_pet_back_off_when_pet_alive(self, _recording_motor: RecordingMotor) -> None:
        """enter() sends pet back_off commands when pet is alive."""
        ctx = AgentContext()
        ctx.pet.alive = True
        state = make_game_state(hp_current=100, hp_max=1000, mana_current=0, mana_max=500)
        routine = FleeRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine.enter(state)
        # pet_back_off presses hotbar slot 2 multiple times
        assert any("hot1_2" in a for a in _recording_motor.actions)


class TestFleeTick:
    """FleeRoutine.tick() navigation."""

    @patch("routines.flee.interruptible_sleep", _noop_sleep)
    def test_tick_failure_without_context(self, _recording_motor: RecordingMotor) -> None:
        """tick() returns FAILURE when no context is set."""
        state = make_game_state(hp_current=100, hp_max=1000)
        routine = FleeRoutine(ctx=None, read_state_fn=None)
        routine._locked = True
        status = routine.tick(state)
        assert status == RoutineStatus.FAILURE
        assert routine.failure_reason == "no_context"

    @patch("routines.flee.interruptible_sleep", _noop_sleep)
    @patch("routines.flee.move_to_point", return_value=False)
    def test_tick_running_while_navigating(self, _mock_move, _recording_motor: RecordingMotor) -> None:
        """tick() returns RUNNING while navigating toward waypoints."""
        ctx = AgentContext()
        ctx.camp.flee_waypoints = [Point(100.0, 200.0, 0.0)]
        state = make_game_state(hp_current=100, hp_max=1000)
        routine = FleeRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._locked = True
        routine._flee_start = __import__("time").time()
        status = routine.tick(state)
        assert status == RoutineStatus.RUNNING

    @patch("routines.flee.interruptible_sleep", _noop_sleep)
    def test_tick_success_after_gate(self, _recording_motor: RecordingMotor) -> None:
        """tick() returns SUCCESS when gate teleport flag is set."""
        ctx = AgentContext()
        state = make_game_state(hp_current=100, hp_max=1000)
        routine = FleeRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._gated = True
        routine._locked = True
        status = routine.tick(state)
        assert status == RoutineStatus.SUCCESS


class TestFleeExit:
    """FleeRoutine.exit() state cleanup."""

    @patch("routines.flee.interruptible_sleep", _noop_sleep)
    def test_exit_unlocks_and_clears_combat(self, _recording_motor: RecordingMotor) -> None:
        """exit() unlocks routine and clears combat state."""
        ctx = AgentContext()
        ctx.pet.alive = False
        state = make_game_state(hp_current=500, hp_max=1000)
        routine = FleeRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._locked = True
        routine.exit(state)
        assert routine.locked is False
        assert ctx.combat.engaged is False
        assert ctx.combat.pull_target_id is None

    @patch("routines.flee.interruptible_sleep", _noop_sleep)
    def test_exit_records_last_flee_time(self, _recording_motor: RecordingMotor) -> None:
        """exit() sets ctx.player.last_flee_time."""
        ctx = AgentContext()
        ctx.pet.alive = False
        state = make_game_state(hp_current=500, hp_max=1000)
        routine = FleeRoutine(ctx=ctx, read_state_fn=lambda: state)
        assert ctx.player.last_flee_time == 0.0
        routine.exit(state)
        assert ctx.player.last_flee_time > 0.0
