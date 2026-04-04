"""Deep tests for Brain decision engine (src/brain/decision.py).

Covers cooldown mechanics, circuit breakers, utility scoring phases 2-4,
profiling, and last-matched-rule tracking -- beyond what test_integration.py
covers.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from brain.decision import Brain
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus
from tests.factories import make_game_state, make_spawn

# ---------------------------------------------------------------------------
# Stub routines (same pattern as test_integration.py)
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


class _FailOnceRoutine(RoutineBase):
    """Returns FAILURE on first tick, then RUNNING."""

    def __init__(self) -> None:
        self._first = True

    def enter(self, state: GameState) -> None:
        self._first = True

    def tick(self, state: GameState) -> RoutineStatus:
        if self._first:
            self._first = False
            self.failure_reason = "test_fail"
            from core.types import FailureCategory

            self.failure_category = FailureCategory.EXECUTION
            return RoutineStatus.FAILURE
        return RoutineStatus.RUNNING

    def exit(self, state: GameState) -> None:
        pass


class _SuccessRoutine(RoutineBase):
    """Returns SUCCESS on first tick."""

    def enter(self, state: GameState) -> None:
        pass

    def tick(self, state: GameState) -> RoutineStatus:
        return RoutineStatus.SUCCESS

    def exit(self, state: GameState) -> None:
        pass


def _make_brain(utility_phase: int = 0) -> Brain:
    return Brain(ctx=None, utility_phase=utility_phase)


# ---------------------------------------------------------------------------
# Cooldown mechanics
# ---------------------------------------------------------------------------


class TestCooldownMechanics:
    def test_failure_applies_cooldown(self) -> None:
        """After FAILURE, the rule is suppressed for cooldown duration."""
        brain = _make_brain()
        fail_routine = _FailOnceRoutine()
        fallback = _StubRoutine()

        brain.add_rule("FAIL_RULE", lambda s: True, fail_routine, failure_cooldown=10.0)
        brain.add_rule("FALLBACK", lambda s: True, fallback)

        state = make_game_state()

        # Tick 1: FAIL_RULE activates, ticks, returns FAILURE in same cycle.
        # Brain applies cooldown. Next eval within same tick picks FALLBACK.
        brain.tick(state)
        # After the failure, FALLBACK should be active on the next tick
        brain.tick(state)
        assert brain._active_name == "FALLBACK"

        # Confirm FAIL_RULE is on cooldown
        assert "FAIL_RULE" in brain._cooldowns
        assert brain._cooldowns["FAIL_RULE"] > time.time()

    def test_cooldown_expires(self) -> None:
        """After cooldown expires, the rule can fire again."""
        brain = _make_brain()
        fail_routine = _FailOnceRoutine()
        fallback = _StubRoutine()

        brain.add_rule("FAIL_RULE", lambda s: True, fail_routine, failure_cooldown=0.1)
        brain.add_rule("FALLBACK", lambda s: True, fallback)

        state = make_game_state()

        # Tick 1+2: FAIL_RULE activates, fails, cooldown applied
        brain.tick(state)
        brain.tick(state)
        assert brain._active_name == "FALLBACK"

        # Fast-forward past cooldown
        brain._cooldowns["FAIL_RULE"] = time.time() - 1.0
        # Force re-evaluation by clearing active
        brain._active = None
        brain._active_name = ""

        # _FailOnceRoutine.enter() resets _first=True, so it will fail again
        # in the same tick. But the key test is that the rule WAS eligible
        # (cooldown expired) and DID fire: cooldown gets re-applied.
        brain.tick(state)
        assert "FAIL_RULE" in brain._cooldowns
        assert brain._cooldowns["FAIL_RULE"] > time.time()

    def test_no_cooldown_when_zero(self) -> None:
        """Rules with failure_cooldown=0.0 get no cooldown entry."""
        brain = _make_brain()
        fail_routine = _FailOnceRoutine()
        fallback = _StubRoutine()

        brain.add_rule("NO_CD", lambda s: True, fail_routine, failure_cooldown=0.0)
        brain.add_rule("FALLBACK", lambda s: True, fallback)

        state = make_game_state()

        brain.tick(state)  # activate NO_CD
        brain.tick(state)  # FAILURE -> no cooldown
        assert "NO_CD" not in brain._cooldowns


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_circuit_trips_after_n_failures(self) -> None:
        """After N failures in window, rule is circuit-broken (OPEN)."""
        brain = _make_brain()
        fallback = _StubRoutine()

        brain.add_rule(
            "FRAGILE",
            lambda s: True,
            _FailOnceRoutine(),
            failure_cooldown=0.0,
            breaker_max_failures=3,
            breaker_window=300.0,
            breaker_recovery=60.0,
        )
        brain.add_rule("FALLBACK", lambda s: True, fallback)

        state = make_game_state()

        # Each pair of ticks: enter -> FAILURE
        for _ in range(3):
            brain.tick(state)  # enter
            brain.tick(state)  # FAILURE

        # After 3 failures, breaker should be OPEN
        assert brain._breakers["FRAGILE"].state == "OPEN"

        # FRAGILE should not fire; FALLBACK should
        brain._active = None
        brain._active_name = ""
        brain.tick(state)
        assert brain._active_name == "FALLBACK"

    def test_emergency_rules_no_breaker(self) -> None:
        """Emergency rules are exempt from circuit breakers."""
        brain = _make_brain()

        brain.add_rule(
            "FLEE",
            lambda s: s.hp_pct < 0.3,
            _StubRoutine(),
            emergency=True,
            breaker_max_failures=3,
        )

        # Emergency rules should not have a breaker registered
        assert "FLEE" not in brain._breakers


# ---------------------------------------------------------------------------
# Utility Phase 2: tier-based scoring
# ---------------------------------------------------------------------------


class TestUtilityPhase2:
    def test_tier_based_highest_score_wins(self) -> None:
        """Within a tier, the highest-scoring rule wins in Phase 2."""
        brain = _make_brain(utility_phase=2)
        r_low = _StubRoutine()
        r_high = _StubRoutine()

        brain.add_rule("LOW", lambda s: True, r_low, score_fn=lambda s: 0.3, tier=0)
        brain.add_rule("HIGH", lambda s: True, r_high, score_fn=lambda s: 0.9, tier=0)

        state = make_game_state()
        brain.tick(state)

        assert brain._active_name == "HIGH"

    def test_higher_tier_wins_over_lower(self) -> None:
        """A scoring rule in a higher-priority tier (lower number) beats lower tier."""
        brain = _make_brain(utility_phase=2)
        r_priority = _StubRoutine()
        r_normal = _StubRoutine()

        brain.add_rule("PRIORITY", lambda s: True, r_priority, score_fn=lambda s: 0.1, tier=0)
        brain.add_rule("NORMAL", lambda s: True, r_normal, score_fn=lambda s: 0.9, tier=1)

        state = make_game_state()
        brain.tick(state)

        assert brain._active_name == "PRIORITY"

    def test_zero_score_skips_tier(self) -> None:
        """If all rules in a tier score 0, the next tier is evaluated."""
        brain = _make_brain(utility_phase=2)
        r_dead = _StubRoutine()
        r_alive = _StubRoutine()

        brain.add_rule("DEAD", lambda s: True, r_dead, score_fn=lambda s: 0.0, tier=0)
        brain.add_rule("ALIVE", lambda s: True, r_alive, score_fn=lambda s: 0.5, tier=1)

        state = make_game_state()
        brain.tick(state)

        assert brain._active_name == "ALIVE"

    def test_cooldown_skips_rule_in_tier(self) -> None:
        """A rule on cooldown is excluded from tier-based scoring."""
        brain = _make_brain(utility_phase=2)
        r_cd = _StubRoutine()
        r_alt = _StubRoutine()

        brain.add_rule("ON_CD", lambda s: True, r_cd, score_fn=lambda s: 0.9, tier=0)
        brain.add_rule("ALT", lambda s: True, r_alt, score_fn=lambda s: 0.5, tier=0)

        # Force cooldown
        brain._cooldowns["ON_CD"] = time.time() + 999

        state = make_game_state()
        brain.tick(state)

        assert brain._active_name == "ALT"


# ---------------------------------------------------------------------------
# Utility Phase 3: weighted cross-tier scoring
# ---------------------------------------------------------------------------


class TestUtilityPhase3:
    def test_weighted_scoring_selects_highest_weighted(self) -> None:
        """Phase 3: weight * score determines winner across tiers."""
        brain = _make_brain(utility_phase=3)
        r_heavy = _StubRoutine()
        r_light = _StubRoutine()

        # r_heavy: score=0.5, weight=3.0 => weighted=1.5
        brain.add_rule(
            "HEAVY",
            lambda s: True,
            r_heavy,
            score_fn=lambda s: 0.5,
            tier=1,
            weight=3.0,
        )
        # r_light: score=0.9, weight=1.0 => weighted=0.9
        brain.add_rule(
            "LIGHT",
            lambda s: True,
            r_light,
            score_fn=lambda s: 0.9,
            tier=1,
            weight=1.0,
        )

        state = make_game_state()
        brain.tick(state)

        assert brain._active_name == "HEAVY"

    def test_emergency_rules_take_priority_in_phase3(self) -> None:
        """Emergency rules win over non-emergency even with higher weighted score."""
        brain = _make_brain(utility_phase=3)
        r_emergency = _StubRoutine()
        r_normal = _StubRoutine()

        brain.add_rule(
            "FLEE",
            lambda s: True,
            r_emergency,
            score_fn=lambda s: 0.1,
            emergency=True,
            weight=1.0,
        )
        brain.add_rule(
            "ACQUIRE",
            lambda s: True,
            r_normal,
            score_fn=lambda s: 0.9,
            weight=10.0,
        )

        state = make_game_state()
        brain.tick(state)

        assert brain._active_name == "FLEE"

    def test_zero_score_excluded_from_weighted(self) -> None:
        """Rules scoring 0 are excluded from weighted selection."""
        brain = _make_brain(utility_phase=3)
        r_zero = _StubRoutine()
        r_positive = _StubRoutine()

        brain.add_rule(
            "ZERO",
            lambda s: True,
            r_zero,
            score_fn=lambda s: 0.0,
            weight=100.0,
        )
        brain.add_rule(
            "POS",
            lambda s: True,
            r_positive,
            score_fn=lambda s: 0.1,
            weight=1.0,
        )

        state = make_game_state()
        brain.tick(state)

        assert brain._active_name == "POS"


# ---------------------------------------------------------------------------
# Utility Phase 4: consideration-based scoring
# ---------------------------------------------------------------------------


class TestUtilityPhase4:
    def _make_brain_with_ctx(self) -> Brain:
        from brain.context import AgentContext

        ctx = AgentContext()
        return Brain(ctx=ctx, utility_phase=4)

    def test_consideration_based_selection(self) -> None:
        """Phase 4: rules with considerations use weighted geometric mean."""
        from brain.rule_def import Consideration

        brain = self._make_brain_with_ctx()
        r_high = _StubRoutine()
        r_low = _StubRoutine()

        brain.add_rule(
            "HIGH",
            lambda s: True,
            r_high,
            score_fn=lambda s: 0.1,  # low fallback
            considerations=[
                Consideration(name="always_high", input_fn=lambda s, ctx: 0.9, curve=lambda v: v),
            ],
            weight=1.0,
        )
        brain.add_rule(
            "LOW",
            lambda s: True,
            r_low,
            score_fn=lambda s: 0.8,  # high fallback
            considerations=[
                Consideration(name="always_low", input_fn=lambda s, ctx: 0.3, curve=lambda v: v),
            ],
            weight=1.0,
        )

        state = make_game_state()
        brain.tick(state)

        assert brain._active_name == "HIGH"

    def test_hard_gate_zeros_score(self) -> None:
        """A consideration returning 0 gates the entire rule."""
        from brain.rule_def import Consideration

        brain = self._make_brain_with_ctx()
        r_gated = _StubRoutine()
        r_open = _StubRoutine()

        brain.add_rule(
            "GATED",
            lambda s: True,
            r_gated,
            considerations=[
                Consideration(name="good", input_fn=lambda s, ctx: 0.9, curve=lambda v: v),
                Consideration(name="gate", input_fn=lambda s, ctx: 0.0, curve=lambda v: v),  # hard gate
            ],
            weight=10.0,
        )
        brain.add_rule(
            "OPEN",
            lambda s: True,
            r_open,
            score_fn=lambda s: 0.5,
            weight=1.0,
        )

        state = make_game_state()
        brain.tick(state)

        assert brain._active_name == "OPEN"

    def test_fallback_to_score_fn_without_considerations(self) -> None:
        """Rules without considerations fall back to score_fn in Phase 4."""
        brain = self._make_brain_with_ctx()
        r_scorefn = _StubRoutine()
        r_low = _StubRoutine()

        brain.add_rule(
            "SCORE_FN",
            lambda s: True,
            r_scorefn,
            score_fn=lambda s: 0.9,
            weight=2.0,
        )
        brain.add_rule(
            "LOW_SCORE",
            lambda s: True,
            r_low,
            score_fn=lambda s: 0.1,
            weight=1.0,
        )

        state = make_game_state()
        brain.tick(state)

        assert brain._active_name == "SCORE_FN"

    def test_emergency_overrides_considerations(self) -> None:
        """Emergency rules always win in Phase 4."""
        from brain.rule_def import Consideration

        brain = self._make_brain_with_ctx()
        r_emerg = _StubRoutine()
        r_normal = _StubRoutine()

        brain.add_rule(
            "FLEE",
            lambda s: True,
            r_emerg,
            score_fn=lambda s: 0.1,
            emergency=True,
            weight=1.0,
        )
        brain.add_rule(
            "ACQUIRE",
            lambda s: True,
            r_normal,
            considerations=[
                Consideration(name="perfect", input_fn=lambda s, ctx: 1.0, curve=lambda v: v),
            ],
            weight=100.0,
        )

        state = make_game_state()
        brain.tick(state)

        assert brain._active_name == "FLEE"


# ---------------------------------------------------------------------------
# Rule evaluation profiling
# ---------------------------------------------------------------------------


class TestProfiling:
    def test_rule_times_populated_after_tick(self) -> None:
        """brain.rule_times contains entries after a tick."""
        brain = _make_brain()
        brain.add_rule("A", lambda s: True, _StubRoutine())
        brain.add_rule("B", lambda s: False, _StubRoutine())

        brain.tick(make_game_state())

        assert "A" in brain.rule_times
        # B is skipped after A matches in Phase 0, so it gets 0.0
        assert "B" in brain.rule_times
        # All times should be non-negative
        for v in brain.rule_times.values():
            assert v >= 0.0

    def test_tick_total_ms_populated(self) -> None:
        brain = _make_brain()
        brain.add_rule("A", lambda s: True, _StubRoutine())
        brain.tick(make_game_state())
        assert brain.tick_total_ms >= 0.0

    def test_routine_tick_ms_populated(self) -> None:
        brain = _make_brain()
        brain.add_rule("A", lambda s: True, _StubRoutine())
        brain.tick(make_game_state())
        # First tick enters the routine; routine_tick_ms set on tick
        brain.tick(make_game_state())
        assert brain.routine_tick_ms >= 0.0

    def test_routine_tick_ms_zero_when_no_active(self) -> None:
        brain = _make_brain()
        brain.add_rule("A", lambda s: False, _StubRoutine())
        brain.tick(make_game_state())
        assert brain.routine_tick_ms == 0.0


# ---------------------------------------------------------------------------
# Last matched rule tracking
# ---------------------------------------------------------------------------


class TestLastMatchedRule:
    def test_set_after_tick(self) -> None:
        brain = _make_brain()
        brain.add_rule("REST", lambda s: True, _StubRoutine())
        brain.tick(make_game_state())
        assert brain._last_matched_rule == "REST"

    def test_empty_when_no_match(self) -> None:
        brain = _make_brain()
        brain.add_rule("REST", lambda s: False, _StubRoutine())
        brain.tick(make_game_state())
        assert brain._last_matched_rule == ""

    def test_changes_on_transition(self) -> None:
        brain = _make_brain()
        brain.add_rule("REST", lambda s: s.hp_pct < 0.5, _StubRoutine())
        brain.add_rule("ACQUIRE", lambda s: True, _StubRoutine())

        brain.tick(make_game_state(hp_current=300, hp_max=1000))
        assert brain._last_matched_rule == "REST"

        brain.tick(make_game_state(hp_current=900, hp_max=1000))
        assert brain._last_matched_rule == "ACQUIRE"


# ---------------------------------------------------------------------------
# Rule eval dict (last_rule_eval)
# ---------------------------------------------------------------------------


class TestRuleEvalDict:
    def test_last_rule_eval_populated(self) -> None:
        brain = _make_brain()
        brain.add_rule("A", lambda s: True, _StubRoutine())
        brain.add_rule("B", lambda s: False, _StubRoutine())

        brain.tick(make_game_state())

        assert "A" in brain.last_rule_eval
        assert brain.last_rule_eval["A"] == "YES"
        # B is skipped after A matches
        assert brain.last_rule_eval["B"] == "skip"

    def test_cooldown_shown_in_eval(self) -> None:
        brain = _make_brain()
        brain.add_rule("CD_RULE", lambda s: True, _StubRoutine())
        brain._cooldowns["CD_RULE"] = time.time() + 100

        brain.tick(make_game_state())

        assert "cooldown" in brain.last_rule_eval["CD_RULE"]


# ---------------------------------------------------------------------------
# Phase 2 scoring: parametrized scenarios
# ---------------------------------------------------------------------------


class TestPhase2Parametrized:
    @pytest.mark.parametrize(
        "scores,expected",
        [
            ([0.1, 0.5, 0.3], "B"),
            ([0.9, 0.1, 0.1], "A"),
            ([0.0, 0.0, 0.7], "C"),
        ],
    )
    def test_highest_score_wins_in_same_tier(self, scores: list[float], expected: str) -> None:
        brain = _make_brain(utility_phase=2)
        names = ["A", "B", "C"]
        for name, score in zip(names, scores, strict=True):
            brain.add_rule(
                name,
                lambda s: True,
                _StubRoutine(),
                score_fn=lambda s, sc=score: sc,
                tier=0,
            )

        brain.tick(make_game_state())
        assert brain._active_name == expected


# ---------------------------------------------------------------------------
# Helper: mock AgentContext for tests that exercise ctx-dependent paths
# ---------------------------------------------------------------------------


def _make_mock_ctx(
    *,
    engaged: bool = False,
    pull_target_id: int = 0,
    last_mob_hp_pct: float = 1.0,
    engagement_start: float = 0.0,
    last_fight_id: int = 0,
    last_fight_name: str = "",
    last_fight_x: float = 0.0,
    last_fight_y: float = 0.0,
    cycle_id: int = 0,
    pet_alive: bool = True,
    imminent_threat: bool = False,
) -> SimpleNamespace:
    """Build a lightweight mock AgentContext with the fields accessed by Brain."""
    ctx = SimpleNamespace(
        combat=SimpleNamespace(
            engaged=engaged,
            pull_target_id=pull_target_id,
            last_mob_hp_pct=last_mob_hp_pct,
        ),
        player=SimpleNamespace(engagement_start=engagement_start),
        defeat_tracker=SimpleNamespace(
            last_fight_id=last_fight_id,
            last_fight_name=last_fight_name,
            last_fight_x=last_fight_x,
            last_fight_y=last_fight_y,
            cycle_id=cycle_id,
        ),
        pet=SimpleNamespace(alive=pet_alive),
        threat=SimpleNamespace(imminent_threat=imminent_threat),
        diag=SimpleNamespace(
            last_rule_evaluation={},
            goap_suggestion="",
            phase_detector=None,
            metrics=None,  # None => _notify_cycle_tracker returns early
            cycle_tracker=None,  # None => _notify_cycle_tracker returns early
            forensics=None,
            breaker_states={},
            tick_overbudget_count=0,
            tick_overbudget_max_ms=0.0,
            tick_overbudget_last_routine="",
            rule_scores={},
        ),
        metrics=SimpleNamespace(
            routine_time={},
            routine_counts=defaultdict(int),
            routine_failures=defaultdict(int),
            routine_start_time=0.0,
            xp_last_raw=0,
        ),
        lock=threading.Lock(),
        record_kill=MagicMock(),
        clear_engagement=MagicMock(),
    )
    return ctx


def _make_brain_with_ctx(ctx: SimpleNamespace, utility_phase: int = 0) -> Brain:
    """Create a Brain wired to a mock AgentContext."""
    brain = Brain(ctx=ctx, utility_phase=utility_phase)
    return brain


# ---------------------------------------------------------------------------
# Tests for _maybe_clear_stale_engagement (lines ~393-474)
# ---------------------------------------------------------------------------


class TestMaybeClearStaleEngagement:
    """Exercises _maybe_clear_stale_engagement through Brain.tick().

    The method runs inside _handle_transition when ctx.combat.engaged is True.
    """

    def test_no_rule_matched_target_gone_after_grace_clears_engagement(self) -> None:
        """No rule matched + target gone after 10s grace period -> clears engagement."""
        ctx = _make_mock_ctx(
            engaged=True,
            # engagement_start 20s ago (well past 10s grace period)
            engagement_start=time.time() - 20.0,
        )
        brain = _make_brain_with_ctx(ctx)
        # No rules at all => selected=None, which triggers the stale check
        # GameState with no target (target=None)
        state = make_game_state()
        assert state.target is None

        brain.tick(state)

        ctx.clear_engagement.assert_called_once()

    def test_no_clear_within_grace_period(self) -> None:
        """No rule matched + target gone but within 10s grace -> no clear."""
        ctx = _make_mock_ctx(
            engaged=True,
            engagement_start=time.time() - 5.0,  # only 5s, within grace
        )
        brain = _make_brain_with_ctx(ctx)
        state = make_game_state()  # no target

        brain.tick(state)

        ctx.clear_engagement.assert_not_called()

    def test_zombie_engagement_clears(self) -> None:
        """Engaged 60s+ but not in combat/pull/flee -> clears engagement."""
        ctx = _make_mock_ctx(
            engaged=True,
            engagement_start=time.time() - 65.0,  # 65s, past 60s threshold
        )
        brain = _make_brain_with_ctx(ctx)
        # Add a rule that matches but is NOT combat/pull/flee
        brain.add_rule("REST", lambda s: True, _StubRoutine())

        state = make_game_state()
        brain.tick(state)

        # selected != None (REST matched), but _active_name is "REST" which
        # is not in ("IN_COMBAT", "PULL", "FLEE"), so zombie branch fires.
        # First tick activates REST. On second tick, REST is already active
        # and selected==active so _handle_transition returns early.
        # We need the zombie check to fire on a tick where _active_name
        # is already set to REST. Tick once to set it, then tick again.
        brain.tick(state)

        ctx.clear_engagement.assert_called()

    def test_defeat_recording_target_hp_zero(self) -> None:
        """Target HP=0 -> record_kill called with target info."""
        target = make_spawn(
            spawn_id=42,
            name="a_skeleton",
            hp_current=0,
            hp_max=100,
            x=10.0,
            y=20.0,
        )
        ctx = _make_mock_ctx(
            engaged=True,
            engagement_start=time.time() - 20.0,
            pull_target_id=42,
            last_fight_id=42,
        )
        brain = _make_brain_with_ctx(ctx)
        # No rules => selected=None => stale path fires
        state = make_game_state(target=target)

        brain.tick(state)

        ctx.record_kill.assert_called_once()
        ctx.clear_engagement.assert_called_once()

    def test_inferred_defeat_low_hp(self) -> None:
        """Target gone + last_mob_hp_pct < 0.20 -> inferred kill recorded."""
        ctx = _make_mock_ctx(
            engaged=True,
            engagement_start=time.time() - 20.0,
            last_mob_hp_pct=0.10,  # 10% -- below 20% threshold
            last_fight_id=99,
            last_fight_name="a_moss_snake",
            last_fight_x=30.0,
            last_fight_y=40.0,
        )
        brain = _make_brain_with_ctx(ctx)
        # No target, no rules => stale path fires, target gone branch
        state = make_game_state()

        brain.tick(state)

        ctx.record_kill.assert_called_once()
        ctx.clear_engagement.assert_called_once()

    def test_possible_evade_high_hp_not_recorded(self) -> None:
        """Target gone + last_mob_hp_pct >= 0.20 -> NOT recorded as kill."""
        ctx = _make_mock_ctx(
            engaged=True,
            engagement_start=time.time() - 20.0,
            last_mob_hp_pct=0.50,  # 50% -- above 20% threshold
            last_fight_id=88,
            last_fight_name="a_bat",
            last_fight_x=5.0,
            last_fight_y=6.0,
        )
        brain = _make_brain_with_ctx(ctx)
        state = make_game_state()

        brain.tick(state)

        ctx.record_kill.assert_not_called()
        # Engagement is still cleared (stale state)
        ctx.clear_engagement.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for _maybe_force_lock_exit (lines ~475-517)
# ---------------------------------------------------------------------------


class TestMaybeForceLockExit:
    """Exercises _maybe_force_lock_exit through Brain.tick().

    The method runs inside _handle_transition when a locked routine is active
    and a different rule wants to fire.
    """

    def test_locked_routine_exceeds_max_lock_seconds_force_exit(self) -> None:
        """Locked routine exceeds max_lock_seconds -> force exit, cooldown applied."""
        ctx = _make_mock_ctx(engaged=True, engagement_start=time.time() - 5.0)
        brain = _make_brain_with_ctx(ctx)

        locked_routine = _StubRoutine(lock=True)
        fallback = _StubRoutine()

        brain.add_rule(
            "PULL",
            lambda s: True,
            locked_routine,
            max_lock_seconds=10.0,
            failure_cooldown=5.0,
        )
        brain.add_rule("FALLBACK", lambda s: True, fallback)

        state = make_game_state()

        # Tick 1: PULL activates (highest priority, condition=True)
        brain.tick(state)
        assert brain._active_name == "PULL"

        # Simulate that the routine has been active for longer than max_lock_seconds
        brain._active_start_time = time.time() - 15.0  # 15s > 10s limit

        # Tick 2: PULL is still active and locked. _handle_transition sees
        # a different rule (FALLBACK or PULL re-evaluation) and checks lock timeout.
        # Since PULL condition is still True, selected == PULL routine == active,
        # so _handle_transition returns early (selected is self._active).
        # We need a scenario where selected != active. Change PULL condition
        # so it no longer matches, forcing FALLBACK to be selected.
        brain._rules[0].condition = lambda s: False

        brain.tick(state)

        # PULL should have been force-exited; cooldown should be applied
        assert "PULL" in brain._cooldowns
        assert brain._cooldowns["PULL"] > time.time()
        # After force-exit, _active is set to None. On next tick FALLBACK fires.
        brain.tick(state)
        assert brain._active_name == "FALLBACK"

    def test_below_timeout_no_action(self) -> None:
        """Locked routine below timeout -> no force exit."""
        ctx = _make_mock_ctx()
        brain = _make_brain_with_ctx(ctx)

        locked_routine = _StubRoutine(lock=True)
        fallback = _StubRoutine()

        brain.add_rule("PULL", lambda s: True, locked_routine, max_lock_seconds=30.0)
        brain.add_rule("FALLBACK", lambda s: True, fallback)

        state = make_game_state()

        # Activate PULL
        brain.tick(state)
        assert brain._active_name == "PULL"

        # Only 2s elapsed -- well below 30s limit
        brain._active_start_time = time.time() - 2.0

        # Make PULL no longer match so FALLBACK is selected (triggering lock path)
        brain._rules[0].condition = lambda s: False

        brain.tick(state)

        # PULL should still be active (locked, not timed out)
        assert brain._active_name == "PULL"
        assert "PULL" not in brain._cooldowns

    def test_max_lock_seconds_zero_disabled(self) -> None:
        """max_lock_seconds=0 -> lock timeout disabled, routine stays locked."""
        ctx = _make_mock_ctx()
        brain = _make_brain_with_ctx(ctx)

        locked_routine = _StubRoutine(lock=True)
        fallback = _StubRoutine()

        brain.add_rule("PULL", lambda s: True, locked_routine, max_lock_seconds=0.0)
        brain.add_rule("FALLBACK", lambda s: True, fallback)

        state = make_game_state()

        brain.tick(state)
        assert brain._active_name == "PULL"

        # Even with a very old start time, no force exit because max_lock=0
        brain._active_start_time = time.time() - 9999.0
        brain._rules[0].condition = lambda s: False

        brain.tick(state)

        # Still locked -- no timeout applied
        assert brain._active_name == "PULL"


# ---------------------------------------------------------------------------
# Tests for _hard_kill_routine (lines ~620-653)
# ---------------------------------------------------------------------------


class _TimeWarpRoutine(RoutineBase):
    """A routine whose tick() manipulates a fake perf_counter to simulate >5s.

    Only warps on the second tick onwards so the first Brain.tick() can
    activate the routine without immediately triggering the hard kill.
    """

    def __init__(self, fake_clock: list[float]) -> None:
        self._fake_clock = fake_clock
        self._tick_count = 0

    def enter(self, state: GameState) -> None:
        self._tick_count = 0

    def tick(self, state: GameState) -> RoutineStatus:
        self._tick_count += 1
        if self._tick_count >= 2:
            # Advance the fake clock by 6 seconds during this tick
            self._fake_clock[0] += 6.0
        return RoutineStatus.RUNNING

    def exit(self, state: GameState) -> None:
        pass


class TestHardKillRoutine:
    """Exercises _hard_kill_routine through Brain.tick().

    The hard kill triggers when routine.tick() returns RUNNING but the
    measured wall time exceeds 5000ms.
    """

    def test_routine_tick_over_5s_force_exit(self) -> None:
        """Routine tick takes >5s -> force exit with TIMEOUT failure_category."""

        ctx = _make_mock_ctx()
        # Use a fake clock that the routine can advance
        fake_clock = [0.0]
        brain = _make_brain_with_ctx(ctx)
        brain.perf_clock = lambda: fake_clock[0]

        warp_routine = _TimeWarpRoutine(fake_clock)
        brain.add_rule("SLOW", lambda s: True, warp_routine, failure_cooldown=8.0)

        state = make_game_state()

        # Tick 1: activates SLOW (enter is called)
        fake_clock[0] = 100.0
        brain.tick(state)
        assert brain._active_name == "SLOW"

        # Tick 2: SLOW.tick() advances clock by 6s -> routine_tick_ms > 5000
        # -> _hard_kill_routine fires
        fake_clock[0] = 200.0  # reset to a known baseline
        brain.tick(state)

        # After hard kill: routine should be cleared
        assert brain._active is None
        assert brain._active_name == ""

        # Failure recorded in metrics
        assert ctx.metrics.routine_failures["SLOW"] >= 1

        # Cooldown applied
        assert "SLOW" in brain._cooldowns


# ---------------------------------------------------------------------------
# Circuit breaker enforcement in Phases 2/3/4
# ---------------------------------------------------------------------------


class TestCircuitBreakerInScoringPhases:
    """Verify that circuit breakers block rules in Phases 2, 3, and 4."""

    @pytest.mark.parametrize("phase", [2, 3, 4])
    def test_circuit_breaker_blocks_rule(self, phase: int) -> None:
        """A tripped (OPEN) breaker prevents the rule from being selected."""
        brain = _make_brain(utility_phase=phase)
        fallback = _StubRoutine()

        brain.add_rule(
            "FRAGILE",
            lambda s: True,
            _FailOnceRoutine(),
            score_fn=lambda s: 0.9,
            failure_cooldown=0.0,
            breaker_max_failures=3,
            breaker_window=300.0,
            breaker_recovery=60.0,
        )
        brain.add_rule("FALLBACK", lambda s: True, fallback, score_fn=lambda s: 0.1)

        state = make_game_state()

        # Trip the breaker: 3 enter-FAILURE cycles
        for _ in range(3):
            brain.tick(state)  # enter FRAGILE
            brain.tick(state)  # FAILURE

        assert brain._breakers["FRAGILE"].state == "OPEN"

        # Clear active so next tick does fresh selection
        brain._active = None
        brain._active_name = ""
        brain.tick(state)

        # FRAGILE is blocked; FALLBACK should be selected
        assert brain._active_name == "FALLBACK"


# ---------------------------------------------------------------------------
# Score function exception handling in Phases 2/3/4
# ---------------------------------------------------------------------------


class TestScoreExceptionHandling:
    """Verify that score_fn exceptions are caught in all scoring phases."""

    @pytest.mark.parametrize("phase", [2, 3, 4])
    def test_score_fn_exception_returns_zero(self, phase: int) -> None:
        """A broken score_fn scores 0, allowing other rules to win."""
        brain = _make_brain(utility_phase=phase)

        def _broken_score(s):
            raise RuntimeError("score explosion")

        fallback = _StubRoutine()
        brain.add_rule("BROKEN", lambda s: True, _StubRoutine(), score_fn=_broken_score)
        brain.add_rule("FALLBACK", lambda s: True, fallback, score_fn=lambda s: 0.5)

        state = make_game_state()
        brain.tick(state)  # must NOT raise

        assert brain._active_name == "FALLBACK"


# ---------------------------------------------------------------------------
# Condition predicate enforcement in phases 2-4
# ---------------------------------------------------------------------------


class TestConditionEnforcedInScoringPhases:
    """Phases 2-4 must respect r.condition(state).

    A rule whose condition returns False must never win, even if its
    score_fn returns the highest value.  Regression test for a bug where
    phases 2-4 only checked cooldowns and circuit breakers but not the
    condition predicate.
    """

    @pytest.mark.parametrize("phase", [2, 3, 4])
    def test_false_condition_never_selected(self, phase: int) -> None:
        brain = _make_brain(utility_phase=phase)
        blocked = _StubRoutine()
        fallback = _StubRoutine()

        brain.add_rule("BLOCKED", lambda s: False, blocked, score_fn=lambda s: 10.0)
        brain.add_rule("FALLBACK_OK", lambda s: True, fallback, score_fn=lambda s: 0.5)

        state = make_game_state()
        brain.tick(state)

        assert brain._active_name == "FALLBACK_OK"

    @pytest.mark.parametrize("phase", [2, 3, 4])
    def test_all_false_conditions_yields_no_selection(self, phase: int) -> None:
        brain = _make_brain(utility_phase=phase)
        brain.add_rule("A", lambda s: False, _StubRoutine(), score_fn=lambda s: 5.0)
        brain.add_rule("B", lambda s: False, _StubRoutine(), score_fn=lambda s: 3.0)

        state = make_game_state()
        brain.tick(state)

        assert brain._active_name is None or brain._active_name == ""
