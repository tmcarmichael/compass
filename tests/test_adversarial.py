"""Adversarial state sequence tests.

Verifies brain resilience under pathological, random, and edge-case input.
"""

from __future__ import annotations

import time

from hypothesis import given, settings
from hypothesis import strategies as st

from brain.decision import Brain
from routines.base import RoutineBase, RoutineStatus
from tests.factories import make_game_state, st_game_state

# ---------------------------------------------------------------------------
# Stub routine
# ---------------------------------------------------------------------------


class _StubRoutine(RoutineBase):
    def enter(self, state):
        pass

    def tick(self, state):
        return RoutineStatus.RUNNING

    def exit(self, state):
        pass


def _make_brain() -> Brain:
    brain = Brain(ctx=None, utility_phase=0)
    brain.add_rule("FLEE", lambda s: s.hp_pct < 0.30, _StubRoutine(), emergency=True)
    brain.add_rule("REST", lambda s: s.hp_pct < 0.85, _StubRoutine())
    brain.add_rule("WANDER", lambda s: True, _StubRoutine())
    return brain


# ---------------------------------------------------------------------------
# Arbitrary state sequences
# ---------------------------------------------------------------------------


@given(states=st.lists(st_game_state, min_size=5, max_size=50))
@settings(max_examples=50)
def test_brain_never_crashes_on_arbitrary_state_sequence(states) -> None:
    """Brain handles any sequence of valid GameState without exceptions."""
    brain = _make_brain()
    for state in states:
        brain.tick(state)
    # If we get here without raising, the brain is robust


@given(state=st_game_state)
@settings(max_examples=30)
def test_brain_tick_completes_under_100ms(state) -> None:
    """A single brain tick must complete within 100ms."""
    brain = _make_brain()
    t0 = time.perf_counter()
    brain.tick(state)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"Brain tick took {elapsed * 1000:.1f}ms (budget: 100ms)"


# ---------------------------------------------------------------------------
# Edge-case states
# ---------------------------------------------------------------------------


def test_brain_handles_zero_hp_max() -> None:
    """hp_max=0 must not cause division by zero."""
    brain = _make_brain()
    state = make_game_state(hp_current=0, hp_max=0, mana_current=0, mana_max=0)
    brain.tick(state)
    # hp_pct returns 1.0 when hp_max=0 (safe default)


def test_brain_handles_negative_hp() -> None:
    """Negative hp_current should not crash."""
    brain = _make_brain()
    state = make_game_state(hp_current=-1, hp_max=1000)
    brain.tick(state)


def test_brain_handles_hp_exceeds_max() -> None:
    """hp_current > hp_max (buff overflow) should not crash."""
    brain = _make_brain()
    state = make_game_state(hp_current=1500, hp_max=1000)
    brain.tick(state)


def test_brain_handles_empty_spawns() -> None:
    """Empty spawn list is the normal idle case."""
    brain = _make_brain()
    state = make_game_state(spawns=())
    brain.tick(state)


def test_brain_handles_massive_spawn_list() -> None:
    """Large spawn list should not blow up."""
    from tests.factories import make_spawn

    spawns = tuple(make_spawn(spawn_id=i, x=float(i * 10)) for i in range(200))
    state = make_game_state(spawns=spawns)
    brain = _make_brain()
    brain.tick(state)


def test_brain_consecutive_identical_states() -> None:
    """Same state fed 100 times should not cause drift or crash."""
    brain = _make_brain()
    state = make_game_state(hp_current=800, hp_max=1000, mana_current=300, mana_max=500)
    for _ in range(100):
        brain.tick(state)
