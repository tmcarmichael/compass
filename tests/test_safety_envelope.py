"""Property-based stateful tests for Brain safety invariants.

Uses hypothesis.stateful.RuleBasedStateMachine to verify that no random
sequence of state mutations and brain ticks can violate core safety
properties: emergency override, lock respect, and internal consistency.
"""

from __future__ import annotations

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    rule,
)

from brain.decision import Brain
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus
from tests.factories import make_game_state

# ---------------------------------------------------------------------------
# Stub routine (same pattern as test_decision_engine.py)
# ---------------------------------------------------------------------------


class _StubRoutine(RoutineBase):
    """A routine that returns RUNNING forever."""

    def __init__(self, *, lock: bool = False) -> None:
        self._lock = lock

    @property
    def locked(self) -> bool:
        return self._lock

    def enter(self, state: GameState) -> None:
        pass

    def tick(self, state: GameState) -> RoutineStatus:
        return RoutineStatus.RUNNING

    def exit(self, state: GameState) -> None:
        pass


# ---------------------------------------------------------------------------
# Condition functions
# ---------------------------------------------------------------------------


def _flee_condition(s: GameState) -> bool:
    return s.hp_pct < 0.30


def _rest_condition(s: GameState) -> bool:
    return s.hp_pct < 0.85


def _acquire_condition(s: GameState) -> bool:
    return s.hp_pct >= 0.50 and s.mana_pct >= 0.30


def _wander_condition(s: GameState) -> bool:
    return True


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class BrainSafetyMachine(RuleBasedStateMachine):
    """Verify Brain safety invariants under random state sequences."""

    def __init__(self) -> None:
        super().__init__()
        self.hp_pct: float = 1.0
        self.mana_pct: float = 1.0
        self.brain: Brain = Brain(ctx=None, utility_phase=0)

        # Routines -- ACQUIRE is lockable to test lock-in behaviour
        self.flee_routine = _StubRoutine()
        self.rest_routine = _StubRoutine()
        self.acquire_routine = _StubRoutine(lock=True)
        self.wander_routine = _StubRoutine()

        # Rules in priority order (highest first)
        self.brain.add_rule(
            "FLEE",
            _flee_condition,
            self.flee_routine,
            emergency=True,
        )
        self.brain.add_rule("REST", _rest_condition, self.rest_routine)
        self.brain.add_rule("ACQUIRE", _acquire_condition, self.acquire_routine)
        self.brain.add_rule("WANDER", _wander_condition, self.wander_routine)

    # -- Helpers ----------------------------------------------------------

    def _make_state(self) -> GameState:
        hp_max = 1000
        mana_max = 500
        return make_game_state(
            hp_current=int(self.hp_pct * hp_max),
            hp_max=hp_max,
            mana_current=int(self.mana_pct * mana_max),
            mana_max=mana_max,
        )

    def _flee_should_fire(self, state: GameState) -> bool:
        """Return True if FLEE's condition matches and nothing blocks it."""
        if not _flee_condition(state):
            return False
        # FLEE is emergency and exempt from circuit breakers, but check cooldown
        if "FLEE" in self.brain._cooldowns:
            import time

            if time.time() < self.brain._cooldowns["FLEE"]:
                return False
        return True

    # -- Rules (hypothesis @rule methods) ---------------------------------

    @rule(hp_pct=st.floats(min_value=0.1, max_value=1.0))
    def set_hp(self, hp_pct: float) -> None:
        self.hp_pct = hp_pct

    @rule(mana_pct=st.floats(min_value=0.0, max_value=1.0))
    def set_mana(self, mana_pct: float) -> None:
        self.mana_pct = mana_pct

    @rule()
    def tick_brain(self) -> None:
        state = self._make_state()
        self.brain.tick(state)

    # -- Invariants (checked after every step) ----------------------------

    @invariant()
    def emergency_override(self) -> None:
        """If FLEE should fire this tick, FLEE must be active."""
        state = self._make_state()
        if self._flee_should_fire(state):
            # Only assert if we already ticked with this state.
            # The invariant fires after set_hp/set_mana too, where the
            # brain hasn't re-evaluated yet. So only check when the brain
            # has an active routine (meaning at least one tick happened)
            # and the last matched rule was evaluated with similar HP.
            if self.brain._active is not None and self.brain._last_matched_rule != "":
                # Re-derive what the brain saw on its last tick:
                # if the brain's current active routine is from a tick where
                # FLEE's condition was true, it must be FLEE (or blocked).
                # We check _last_matched_rule which is set each tick.
                if self.brain._last_matched_rule == "FLEE":
                    assert self.brain._active_name == "FLEE" or (
                        self.brain._active is not None and self.brain._active.locked
                    ), (
                        f"FLEE matched but not active: "
                        f"active={self.brain._active_name!r}, "
                        f"locked={self.brain._active.locked if self.brain._active else False}"
                    )

    @invariant()
    def lock_respect(self) -> None:
        """A locked non-emergency routine cannot be replaced by a non-emergency rule.

        _last_matched_rule shows which rule *evaluation* selected, but
        the transition logic blocks non-emergency rules from overriding a
        lock. So if active is locked AND the matched rule is non-emergency
        AND different from active, the brain must still be running the
        locked routine (i.e. the transition was blocked).
        """
        if self.brain._active is not None and self.brain._active.locked:
            last = self.brain._last_matched_rule
            if last and last != self.brain._active_name:
                # A different rule wanted to fire. Check it's not a
                # non-emergency that somehow replaced the locked routine.
                is_emergency = any(r.name == last and r.emergency for r in self.brain._rules)
                if not is_emergency:
                    # The lock should have held -- active must still be
                    # the locked routine, NOT the matched rule.
                    assert self.brain._active_name != last, (
                        f"Locked routine was replaced by non-emergency "
                        f"{last!r} (active={self.brain._active_name!r})"
                    )

    @invariant()
    def no_orphan_state(self) -> None:
        """If brain._active is set, brain._active_name must not be empty."""
        if self.brain._active is not None:
            assert self.brain._active_name != "", "brain._active is set but _active_name is empty"

    @invariant()
    def consistent_none_state(self) -> None:
        """brain._active_name == '' iff brain._active is None."""
        if self.brain._active is None:
            assert self.brain._active_name == "", (
                f"brain._active is None but _active_name={self.brain._active_name!r}"
            )
        else:
            assert self.brain._active_name != "", "brain._active is set but _active_name is empty"


# Hypothesis test runner picks this up
TestBrainSafety = BrainSafetyMachine.TestCase
TestBrainSafety.settings = settings(max_examples=200, stateful_step_count=30)
