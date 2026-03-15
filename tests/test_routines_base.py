"""Tests for routines.base  -- routine lifecycle contract.

The RoutineBase ABC defines the enter/tick/exit state machine contract
that every behavior in the system implements. These tests verify the
contract using minimal concrete subclasses.
"""

from __future__ import annotations

import pytest

from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus
from tests.factories import make_game_state

# ---------------------------------------------------------------------------
# A minimal concrete routine for testing
# ---------------------------------------------------------------------------


class _CountingRoutine(RoutineBase):
    """Runs for N ticks then succeeds."""

    def __init__(self, ticks_to_complete: int = 3) -> None:
        self._remaining = ticks_to_complete
        self.entered = False
        self.exited = False
        self.tick_count = 0

    def enter(self, state: GameState) -> None:
        self.entered = True

    def tick(self, state: GameState) -> RoutineStatus:
        self.tick_count += 1
        self._remaining -= 1
        if self._remaining <= 0:
            return RoutineStatus.SUCCESS
        return RoutineStatus.RUNNING

    def exit(self, state: GameState) -> None:
        self.exited = True


class _FailingRoutine(RoutineBase):
    """Fails immediately on first tick."""

    def enter(self, state: GameState) -> None:
        pass

    def tick(self, state: GameState) -> RoutineStatus:
        self.failure_reason = "test failure"
        return RoutineStatus.FAILURE

    def exit(self, state: GameState) -> None:
        pass


# ---------------------------------------------------------------------------
# RoutineStatus
# ---------------------------------------------------------------------------


class TestRoutineStatus:
    def test_three_members(self) -> None:
        assert set(RoutineStatus) == {
            RoutineStatus.RUNNING,
            RoutineStatus.SUCCESS,
            RoutineStatus.FAILURE,
        }

    def test_is_str(self) -> None:
        assert isinstance(RoutineStatus.RUNNING, str)
        assert RoutineStatus.SUCCESS == "success"


# ---------------------------------------------------------------------------
# RoutineBase ABC
# ---------------------------------------------------------------------------


class TestRoutineBase:
    def test_cannot_instantiate_abstract(self) -> None:
        cls: type = RoutineBase
        with pytest.raises(TypeError, match="abstract"):
            cls()

    def test_name_returns_classname(self) -> None:
        r = _CountingRoutine()
        assert r.name == "_CountingRoutine"

    def test_locked_default_false(self) -> None:
        r = _CountingRoutine()
        assert r.locked is False


# ---------------------------------------------------------------------------
# Lifecycle contract
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_enter_tick_exit_sequence(self) -> None:
        state = make_game_state()
        r = _CountingRoutine(ticks_to_complete=2)

        r.enter(state)
        assert r.entered is True

        status = r.tick(state)
        assert status == RoutineStatus.RUNNING
        assert r.tick_count == 1

        status = r.tick(state)
        assert status == RoutineStatus.SUCCESS
        assert r.tick_count == 2

        r.exit(state)
        assert r.exited is True

    def test_failure_sets_reason(self) -> None:
        state = make_game_state()
        r = _FailingRoutine()
        r.enter(state)
        status = r.tick(state)
        assert status == RoutineStatus.FAILURE
        assert r.failure_reason == "test failure"
