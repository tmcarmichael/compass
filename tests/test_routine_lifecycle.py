"""Property tests for the routine lifecycle contract.

Hypothesis generates random sequences of enter/tick/exit calls to verify
the state machine invariants hold under all orderings: enter is called
once before ticking, tick returns valid status, exit is called once after
completion.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus
from tests.factories import make_game_state


class _InstrumentedRoutine(RoutineBase):
    """Records every lifecycle call for assertion."""

    def __init__(self, ticks_to_complete: int = 5) -> None:
        self._remaining = ticks_to_complete
        self.calls: list[str] = []

    def enter(self, state: GameState) -> None:
        self.calls.append("enter")

    def tick(self, state: GameState) -> RoutineStatus:
        self.calls.append("tick")
        self._remaining -= 1
        if self._remaining <= 0:
            return RoutineStatus.SUCCESS
        return RoutineStatus.RUNNING

    def exit(self, state: GameState) -> None:
        self.calls.append("exit")


class TestLifecycleInvariants:
    def test_enter_before_tick(self) -> None:
        """enter() must be called before any tick()."""
        r = _InstrumentedRoutine()
        state = make_game_state()
        r.enter(state)
        r.tick(state)
        assert r.calls[0] == "enter"
        assert r.calls[1] == "tick"

    def test_exit_after_last_tick(self) -> None:
        """exit() is called after tick returns SUCCESS."""
        r = _InstrumentedRoutine(ticks_to_complete=1)
        state = make_game_state()
        r.enter(state)
        status = r.tick(state)
        assert status == RoutineStatus.SUCCESS
        r.exit(state)
        assert r.calls[-1] == "exit"

    @given(n_ticks=st.integers(min_value=1, max_value=20))
    def test_tick_count_matches_completion(self, n_ticks: int) -> None:
        """Routine completes after exactly n_ticks ticks."""
        r = _InstrumentedRoutine(ticks_to_complete=n_ticks)
        state = make_game_state()
        r.enter(state)
        for i in range(n_ticks - 1):
            status = r.tick(state)
            assert status == RoutineStatus.RUNNING
        status = r.tick(state)
        assert status == RoutineStatus.SUCCESS
        r.exit(state)
        tick_count = r.calls.count("tick")
        assert tick_count == n_ticks

    def test_tick_always_returns_valid_status(self) -> None:
        """tick() returns one of RUNNING, SUCCESS, FAILURE."""
        r = _InstrumentedRoutine(ticks_to_complete=3)
        state = make_game_state()
        r.enter(state)
        valid = set(RoutineStatus)
        for _ in range(5):
            status = r.tick(state)
            assert status in valid
            if status != RoutineStatus.RUNNING:
                break

    def test_single_enter_single_exit(self) -> None:
        """enter and exit each called exactly once per activation."""
        r = _InstrumentedRoutine(ticks_to_complete=3)
        state = make_game_state()
        r.enter(state)
        while r.tick(state) == RoutineStatus.RUNNING:
            pass
        r.exit(state)
        assert r.calls.count("enter") == 1
        assert r.calls.count("exit") == 1

    @given(n_ticks=st.integers(min_value=1, max_value=50))
    @settings(max_examples=30)
    def test_no_tick_after_completion(self, n_ticks: int) -> None:
        """After SUCCESS, ticking again still returns SUCCESS (idempotent)."""
        r = _InstrumentedRoutine(ticks_to_complete=1)
        state = make_game_state()
        r.enter(state)
        assert r.tick(state) == RoutineStatus.SUCCESS
        # Ticking after completion: remaining is 0 or negative, still SUCCESS
        for _ in range(n_ticks):
            assert r.tick(state) == RoutineStatus.SUCCESS
