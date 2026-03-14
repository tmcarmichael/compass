"""Exhaustive state transition tests for Brain decision engine.

Verifies every valid transition between Brain states and confirms
invalid transitions are correctly blocked.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from brain.decision import Brain
from core.types import FailureCategory
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus
from tests.factories import make_game_state

# ---------------------------------------------------------------------------
# Stub routines with lifecycle tracking
# ---------------------------------------------------------------------------


class _StubRoutine(RoutineBase):
    """Returns RUNNING, not locked. Tracks enter/exit calls."""

    def __init__(
        self,
        on_enter: Callable[[GameState], None] | None = None,
        on_exit: Callable[[GameState], None] | None = None,
        on_tick: Callable[[GameState], RoutineStatus | None] | None = None,
    ) -> None:
        self.enters: list[str] = []
        self.exits: list[str] = []
        self.ticks: list[str] = []
        self._on_enter = on_enter
        self._on_exit = on_exit
        self._on_tick = on_tick

    @property
    def locked(self) -> bool:
        return False

    def enter(self, state: GameState) -> None:
        self.enters.append("enter")
        if self._on_enter is not None:
            self._on_enter(state)

    def tick(self, state: GameState) -> RoutineStatus:
        self.ticks.append("tick")
        if self._on_tick is not None:
            result = self._on_tick(state)
            if result is not None:
                return result
        return RoutineStatus.RUNNING

    def exit(self, state: GameState) -> None:
        self.exits.append("exit")
        if self._on_exit is not None:
            self._on_exit(state)


class _LockedRoutine(RoutineBase):
    """Returns RUNNING, locked=True. Tracks enter/exit calls."""

    def __init__(
        self,
        on_exit: Callable[[GameState], None] | None = None,
    ) -> None:
        self.enters: list[str] = []
        self.exits: list[str] = []
        self.ticks: list[str] = []
        self._on_exit = on_exit

    @property
    def locked(self) -> bool:
        return True

    def enter(self, state: GameState) -> None:
        self.enters.append("enter")

    def tick(self, state: GameState) -> RoutineStatus:
        self.ticks.append("tick")
        return RoutineStatus.RUNNING

    def exit(self, state: GameState) -> None:
        self.exits.append("exit")
        if self._on_exit is not None:
            self._on_exit(state)


class _FailRoutine(RoutineBase):
    """Returns FAILURE on first tick, then RUNNING."""

    def __init__(
        self,
        on_after_tick: Callable[[RoutineBase, RoutineStatus], None] | None = None,
    ) -> None:
        self._first = True
        self.enters: list[str] = []
        self.exits: list[str] = []
        self.ticks: list[str] = []
        self._on_after_tick = on_after_tick

    @property
    def locked(self) -> bool:
        return False

    def enter(self, state: GameState) -> None:
        self._first = True
        self.enters.append("enter")

    def tick(self, state: GameState) -> RoutineStatus:
        self.ticks.append("tick")
        if self._first:
            self._first = False
            self.failure_reason = "test_fail"
            self.failure_category = FailureCategory.EXECUTION
            result = RoutineStatus.FAILURE
        else:
            result = RoutineStatus.RUNNING
        if self._on_after_tick is not None:
            self._on_after_tick(self, result)
        return result

    def exit(self, state: GameState) -> None:
        self.exits.append("exit")


class _SuccessRoutine(RoutineBase):
    """Returns SUCCESS immediately. Tracks lifecycle."""

    def __init__(self) -> None:
        self.enters: list[str] = []
        self.exits: list[str] = []
        self.ticks: list[str] = []

    @property
    def locked(self) -> bool:
        return False

    def enter(self, state: GameState) -> None:
        self.enters.append("enter")

    def tick(self, state: GameState) -> RoutineStatus:
        self.ticks.append("tick")
        return RoutineStatus.SUCCESS

    def exit(self, state: GameState) -> None:
        self.exits.append("exit")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_brain() -> Brain:
    """Create a Brain with no ctx (test-only mode)."""
    return Brain(ctx=None, utility_phase=0)


# ---------------------------------------------------------------------------
# Transition tests
# ---------------------------------------------------------------------------


class TestBrainTransitions:
    """Exhaustive state transition tests for the Brain decision engine."""

    # 1. idle -> active
    def test_idle_to_active(self) -> None:
        """No routine active -> condition fires -> routine enters."""
        brain = _make_brain()
        r = _StubRoutine()
        brain.add_rule("A", condition=lambda s: True, routine=r)
        state = make_game_state()

        assert brain._active is None
        assert brain._active_name == ""

        brain.tick(state)

        assert brain._active is r
        assert brain._active_name == "A"
        assert r.enters == ["enter"]
        assert len(r.ticks) == 1
        assert r.exits == []

    # 2. active A -> different active B (higher priority preemption)
    def test_active_to_different_active(self) -> None:
        """Routine A active -> condition B fires (higher priority) -> A exits, B enters."""
        brain = _make_brain()
        a = _StubRoutine()
        b = _StubRoutine()
        fire_b = [False]

        brain.add_rule("B", condition=lambda s: fire_b[0], routine=b)
        brain.add_rule("A", condition=lambda s: True, routine=a)
        state = make_game_state()

        # Tick 1: only A's condition fires (B is higher priority but off)
        brain.tick(state)
        assert brain._active is a
        assert a.enters == ["enter"]
        assert b.enters == []

        # Tick 2: B fires (higher priority because added first)
        fire_b[0] = True
        brain.tick(state)
        assert brain._active is b
        assert brain._active_name == "B"
        assert a.exits == ["exit"]
        assert b.enters == ["enter"]

    # 3. active stays when same routine wins
    def test_active_stays_when_same_wins(self) -> None:
        """Routine A active -> A still wins -> no transition (enter not called again)."""
        brain = _make_brain()
        a = _StubRoutine()
        brain.add_rule("A", condition=lambda s: True, routine=a)
        state = make_game_state()

        brain.tick(state)
        assert a.enters == ["enter"]

        brain.tick(state)
        brain.tick(state)
        # enter should only be called once; no exit/re-enter cycle
        assert a.enters == ["enter"]
        assert a.exits == []
        assert len(a.ticks) == 3

    # 4. active -> locked (locked property becomes True on the routine)
    def test_active_to_locked(self) -> None:
        """Routine enters, locked=True -> verify non-emergency cannot interrupt."""
        brain = _make_brain()
        locked = _LockedRoutine()
        other = _StubRoutine()

        brain.add_rule("LOCKED", condition=lambda s: True, routine=locked)
        brain.add_rule("OTHER", condition=lambda s: True, routine=other)
        state = make_game_state()

        brain.tick(state)
        assert brain._active is locked
        assert locked.locked is True

        # OTHER cannot preempt because LOCKED is locked
        # We need OTHER to win evaluation -- but LOCKED wins by priority.
        # Let's restructure: OTHER is higher priority but locked blocks it.
        brain2 = _make_brain()
        locked2 = _LockedRoutine()
        other2 = _StubRoutine()
        fire_other = [False]

        brain2.add_rule("OTHER", condition=lambda s: fire_other[0], routine=other2)
        brain2.add_rule("LOCKED", condition=lambda s: True, routine=locked2)
        state2 = make_game_state()

        # Activate LOCKED first
        brain2.tick(state2)
        assert brain2._active is locked2

        # Now OTHER fires (higher priority) -- but LOCKED is locked
        fire_other[0] = True
        brain2.tick(state2)
        assert brain2._active is locked2  # still locked
        assert other2.enters == []

    # 5. locked blocks non-emergency
    def test_locked_blocks_non_emergency(self) -> None:
        """Locked routine active, lower priority rule fires -> locked routine stays."""
        brain = _make_brain()
        locked = _LockedRoutine()
        intruder = _StubRoutine()
        fire_intruder = [False]

        brain.add_rule("INTRUDER", condition=lambda s: fire_intruder[0], routine=intruder)
        brain.add_rule("LOCKED", condition=lambda s: True, routine=locked)
        state = make_game_state()

        # Activate LOCKED
        brain.tick(state)
        assert brain._active is locked
        assert locked.enters == ["enter"]

        # INTRUDER fires but is non-emergency
        fire_intruder[0] = True
        brain.tick(state)
        assert brain._active is locked
        assert intruder.enters == []
        assert locked.exits == []

    # 6. locked allows emergency
    def test_locked_allows_emergency(self) -> None:
        """Locked routine active, emergency rule fires -> lock broken, emergency enters."""
        brain = _make_brain()
        locked = _LockedRoutine()
        emergency = _StubRoutine()
        fire_emergency = [False]

        brain.add_rule(
            "EMERGENCY",
            condition=lambda s: fire_emergency[0],
            routine=emergency,
            emergency=True,
        )
        brain.add_rule("LOCKED", condition=lambda s: True, routine=locked)
        state = make_game_state()

        # Activate LOCKED
        brain.tick(state)
        assert brain._active is locked

        # Emergency fires -> overrides lock
        fire_emergency[0] = True
        brain.tick(state)
        assert brain._active is emergency
        assert brain._active_name == "EMERGENCY"
        assert locked.exits == ["exit"]
        assert emergency.enters == ["enter"]

    # 7. success clears active
    def test_success_clears_active(self) -> None:
        """Routine returns SUCCESS -> brain._active is None, brain._active_name is ''."""
        brain = _make_brain()
        r = _SuccessRoutine()
        # Condition fires once, then nothing matches
        fired = [True]
        brain.add_rule("S", condition=lambda s: fired[0], routine=r)
        state = make_game_state()

        brain.tick(state)
        # After SUCCESS, active should be cleared
        assert brain._active is None
        assert brain._active_name == ""
        assert r.enters == ["enter"]
        assert r.exits == ["exit"]
        assert r.ticks == ["tick"]

    # 8. failure clears active
    def test_failure_clears_active(self) -> None:
        """Routine returns FAILURE -> active cleared."""
        brain = _make_brain()
        r = _FailRoutine()
        fired = [True]
        brain.add_rule("F", condition=lambda s: fired[0], routine=r)
        state = make_game_state()

        brain.tick(state)
        # After FAILURE on first tick, active should be cleared
        assert brain._active is None
        assert brain._active_name == ""
        assert r.enters == ["enter"]
        assert r.exits == ["exit"]

    # 9. failure applies cooldown
    def test_failure_applies_cooldown(self) -> None:
        """Routine fails with failure_cooldown > 0 -> cooldown entry created."""
        brain = _make_brain()
        r = _FailRoutine()
        brain.add_rule("COOL", condition=lambda s: True, routine=r, failure_cooldown=30.0)
        state = make_game_state()

        brain.tick(state)
        # Routine failed -> cooldown should be set
        assert "COOL" in brain._cooldowns
        assert brain._cooldowns["COOL"] > time.time()

    # 10. cooldown blocks reactivation
    def test_cooldown_blocks_reactivation(self) -> None:
        """Rule on cooldown -> rule eval shows 'cooldown', rule skipped."""
        brain = _make_brain()
        r = _FailRoutine()
        brain.add_rule("COOL", condition=lambda s: True, routine=r, failure_cooldown=30.0)
        state = make_game_state()

        # First tick: routine enters and fails, cooldown applied
        brain.tick(state)
        assert brain._active is None

        # Second tick: rule should be on cooldown
        brain.tick(state)
        assert brain._active is None
        assert "cooldown" in brain.last_rule_eval.get("COOL", "").lower()

    # 11. cooldown expiry allows reactivation
    def test_cooldown_expiry_allows_reactivation(self) -> None:
        """Cooldown time passes -> rule fires again."""
        brain = _make_brain()
        r = _FailRoutine()
        brain.add_rule("COOL", condition=lambda s: True, routine=r, failure_cooldown=10.0)
        state = make_game_state()

        # Tick 1: fails, cooldown applied
        brain.tick(state)
        assert "COOL" in brain._cooldowns

        # Manually expire the cooldown
        brain._cooldowns["COOL"] = time.time() - 1.0

        # Tick 2: cooldown expired, rule fires again
        brain.tick(state)
        assert brain.last_rule_eval.get("COOL") == "YES"
        # The routine was entered again (FailRoutine resets _first on enter)
        assert len(r.enters) == 2

    # 12. circuit breaker trips after N failures
    def test_circuit_breaker_trips_after_n_failures(self) -> None:
        """5 failures in window -> breaker.allow() returns False -> rule blocked."""
        brain = _make_brain()
        r = _FailRoutine()
        brain.add_rule(
            "BREAK",
            condition=lambda s: True,
            routine=r,
            failure_cooldown=0.0,  # no cooldown so we can re-enter quickly
            breaker_max_failures=5,
            breaker_window=300.0,
            breaker_recovery=120.0,
        )
        state = make_game_state()

        # Fail 5 times
        for i in range(5):
            brain.tick(state)
            # After each failure, active is cleared; next tick will re-enter

        # Now the breaker should be open
        breaker = brain._breakers["BREAK"]
        assert breaker.state == "OPEN"
        assert breaker.allow() is False

        # Next tick: rule should be blocked by circuit breaker
        brain.tick(state)
        assert brain._active is None
        assert brain.last_rule_eval.get("BREAK") == "OPEN"

    # 13. circuit breaker exempt for emergency
    def test_circuit_breaker_exempt_for_emergency(self) -> None:
        """Emergency rules don't get circuit breakers."""
        brain = _make_brain()
        r = _FailRoutine()
        brain.add_rule(
            "FLEE",
            condition=lambda s: True,
            routine=r,
            emergency=True,
            breaker_max_failures=5,
        )

        # Emergency rules should not have a circuit breaker created
        assert "FLEE" not in brain._breakers

    # 14. no rule matches, no active
    def test_no_rule_matches_no_active(self) -> None:
        """No condition returns True -> no routine active."""
        brain = _make_brain()
        r = _StubRoutine()
        brain.add_rule("NEVER", condition=lambda s: False, routine=r)
        state = make_game_state()

        brain.tick(state)
        assert brain._active is None
        assert brain._active_name == ""
        assert r.enters == []
        assert r.ticks == []

    # 15. max lock timeout forces exit
    def test_max_lock_timeout_forces_exit(self) -> None:
        """Routine locked with max_lock_seconds -> after timeout, lock is force-broken.

        The lock timeout only triggers when a *different* rule wins evaluation.
        OTHER must be higher priority so that selected != active, entering the
        transition path where _maybe_force_lock_exit runs.
        """
        brain = _make_brain()
        locked = _LockedRoutine()
        other = _StubRoutine()
        fire_other = [False]

        # OTHER is higher priority (added first); fires only when enabled
        brain.add_rule("OTHER", condition=lambda s: fire_other[0], routine=other)
        brain.add_rule(
            "LOCKED",
            condition=lambda s: True,
            routine=locked,
            max_lock_seconds=10.0,
        )
        state = make_game_state()

        # Activate LOCKED (OTHER not firing yet)
        brain.tick(state)
        assert brain._active is locked

        # Simulate time passing beyond the lock timeout
        brain._active_start_time = time.time() - 15.0

        # Fire OTHER (higher priority) to trigger transition path
        fire_other[0] = True
        brain.tick(state)

        # Lock should have been force-broken: LOCKED exited, then OTHER entered
        assert locked.exits == ["exit"]
        assert brain._active is other
        assert other.enters == ["enter"]

    # ---------------------------------------------------------------------------
    # Additional edge-case transitions
    # ---------------------------------------------------------------------------

    def test_success_then_new_rule_fires_next_tick(self) -> None:
        """After SUCCESS clears active within a tick, the *next* tick re-evaluates
        and fires a matching rule. Rule evaluation + transition happens before
        routine tick, so SUCCESS is processed after the transition decision.
        The active is cleared within the same tick, then next tick re-enters.
        """
        brain = _make_brain()
        s_routine = _SuccessRoutine()
        fallback = _StubRoutine()

        fired_count = [0]

        def win_cond(s: GameState) -> bool:
            fired_count[0] += 1
            return fired_count[0] <= 2  # fire first two evaluations

        brain.add_rule("WIN", condition=win_cond, routine=s_routine)
        brain.add_rule("FALLBACK", condition=lambda s: True, routine=fallback)
        state = make_game_state()

        # Tick 1: WIN enters, succeeds, active cleared within tick
        brain.tick(state)
        assert brain._active is None  # cleared after SUCCESS
        assert s_routine.exits == ["exit"]
        assert s_routine.enters == ["enter"]

        # Tick 2: WIN fires again, enters again, succeeds again
        brain.tick(state)
        assert brain._active is None  # SUCCESS cleared again
        assert len(s_routine.enters) == 2
        assert len(s_routine.exits) == 2

        # Tick 3: WIN condition is now False; FALLBACK fires
        brain.tick(state)
        assert brain._active is fallback
        assert fallback.enters == ["enter"]

    def test_failure_with_zero_cooldown_allows_immediate_reentry(self) -> None:
        """Failure with failure_cooldown=0 allows immediate re-entry next tick."""
        brain = _make_brain()
        r = _FailRoutine()
        brain.add_rule("NO_CD", condition=lambda s: True, routine=r, failure_cooldown=0.0)
        state = make_game_state()

        brain.tick(state)  # enters, fails
        assert brain._active is None
        assert "NO_CD" not in brain._cooldowns

        brain.tick(state)  # should re-enter immediately
        assert r.enters == ["enter", "enter"]

    def test_locked_emergency_lifecycle_order(self) -> None:
        """Emergency override: locked.exit() before emergency.enter()."""
        brain = _make_brain()
        fire_emergency = [False]
        call_order: list[str] = []

        locked = _LockedRoutine(on_exit=lambda s: call_order.append("locked_exit"))
        emergency = _StubRoutine(on_enter=lambda s: call_order.append("emergency_enter"))

        brain.add_rule(
            "FLEE",
            condition=lambda s: fire_emergency[0],
            routine=emergency,
            emergency=True,
        )
        brain.add_rule("LOCKED", condition=lambda s: True, routine=locked)
        state = make_game_state()

        brain.tick(state)  # LOCKED enters
        fire_emergency[0] = True
        brain.tick(state)  # FLEE overrides

        assert call_order == ["locked_exit", "emergency_enter"]

    def test_multiple_cooldowns_independent(self) -> None:
        """Multiple rules can have independent cooldowns."""
        brain = _make_brain()
        r1 = _FailRoutine()
        r2 = _FailRoutine()

        brain.add_rule("RULE1", condition=lambda s: True, routine=r1, failure_cooldown=10.0)
        brain.add_rule("RULE2", condition=lambda s: True, routine=r2, failure_cooldown=20.0)
        state = make_game_state()

        # Tick 1: RULE1 fires (higher priority), fails, cooldown applied
        brain.tick(state)
        assert "RULE1" in brain._cooldowns

        # Tick 2: RULE1 on cooldown, RULE2 fires, fails, cooldown applied
        brain.tick(state)
        assert "RULE2" in brain._cooldowns

        # Both cooldowns exist independently
        assert brain._cooldowns["RULE1"] != brain._cooldowns["RULE2"]

    def test_lock_timeout_applies_failure_cooldown(self) -> None:
        """When lock timeout forces exit, failure_cooldown is applied.

        OTHER must be higher priority to trigger the transition path where
        _maybe_force_lock_exit runs.
        """
        brain = _make_brain()
        locked = _LockedRoutine()
        other = _StubRoutine()
        fire_other = [False]

        # OTHER is higher priority (added first)
        brain.add_rule("OTHER", condition=lambda s: fire_other[0], routine=other)
        brain.add_rule(
            "LOCKED",
            condition=lambda s: True,
            routine=locked,
            max_lock_seconds=10.0,
            failure_cooldown=30.0,
        )
        state = make_game_state()

        # Activate LOCKED (OTHER not firing yet)
        brain.tick(state)
        assert brain._active is locked

        # Expire the lock, then fire OTHER to trigger transition
        brain._active_start_time = time.time() - 15.0
        fire_other[0] = True
        brain.tick(state)

        # Cooldown should have been applied from the forced lock exit
        assert "LOCKED" in brain._cooldowns

    def test_circuit_breaker_not_tripped_by_precondition_failure(self) -> None:
        """Precondition failures should not trip the circuit breaker."""
        brain = _make_brain()

        def _set_precondition(routine: RoutineBase, result: RoutineStatus) -> None:
            if result == RoutineStatus.FAILURE:
                routine.failure_category = FailureCategory.PRECONDITION

        r = _FailRoutine(on_after_tick=_set_precondition)
        brain.add_rule(
            "PRECON",
            condition=lambda s: True,
            routine=r,
            breaker_max_failures=2,
            breaker_window=300.0,
        )
        state = make_game_state()

        # Fail twice with PRECONDITION category
        brain.tick(state)  # fail 1
        brain.tick(state)  # fail 2

        breaker = brain._breakers["PRECON"]
        # Should still be CLOSED because precondition failures don't trip
        assert breaker.state == "CLOSED"

    def test_idle_remains_idle_no_rules(self) -> None:
        """Brain with no rules stays idle."""
        brain = _make_brain()
        state = make_game_state()

        brain.tick(state)
        assert brain._active is None
        assert brain._active_name == ""

    def test_active_to_idle_when_condition_stops_firing(self) -> None:
        """When the winning rule stops firing and no other matches, active
        routine continues (Brain does not preempt a RUNNING routine
        just because its condition stopped matching -- 'selected is self._active'
        comparison handles this since the same object remains selected=None,
        and a None != active triggers transition)."""
        brain = _make_brain()
        r = _StubRoutine()
        should_fire = [True]
        brain.add_rule("TEMP", condition=lambda s: should_fire[0], routine=r)
        state = make_game_state()

        brain.tick(state)
        assert brain._active is r

        # Condition stops firing
        should_fire[0] = False
        brain.tick(state)
        # selected will be None, which != active, so transition fires: exit active
        assert brain._active is None
        assert r.exits == ["exit"]
