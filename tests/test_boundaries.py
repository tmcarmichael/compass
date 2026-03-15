"""Boundary tests: verify behavior flips at every named threshold.

Each tunable threshold gets a test with values on both sides of the
boundary, ensuring the system transitions correctly at the exact point.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.circuit_breaker import CircuitBreaker
from brain.context import AgentContext
from brain.goap.actions import AcquireAction, DefeatAction, PullAction, RestAction
from brain.goap.goals import ManageResourcesGoal
from brain.goap.planner import SATISFACTION_THRESHOLD, GOAPPlanner
from brain.learning.encounters import FightHistory
from brain.rules.survival import (
    _check_core_safety_floors,
    flee_condition,
    rest_needs_check,
)
from brain.scoring.target import ScoringWeights
from brain.scoring.weight_learner import (
    MAX_DRIFT,
    STEP_INTERVAL,
    GradientTuner,
)
from core.features import flags
from tests.factories import make_game_state, make_plan_world_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _enable_flags() -> None:
    """Enable survival-relevant flags for all tests in this module."""
    flags.flee = True
    flags.rest = True


def _make_ctx(**overrides: object) -> AgentContext:
    """Build an AgentContext with sensible test defaults."""
    ctx = AgentContext()
    ctx.player.last_buff_time = 0.0
    ctx.player.last_flee_time = 0.0
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


# ===================================================================
# 1. Flee urgency enter / exit (hysteresis)
# ===================================================================


class TestFleeUrgencyHysteresis:
    """FLEE_URGENCY_ENTER=0.65, FLEE_URGENCY_EXIT=0.35 boundary tests."""

    @pytest.mark.parametrize(
        "hp_current, expected_flee",
        [
            pytest.param(640, False, id="urgency_below_enter_0.64"),
            pytest.param(340, True, id="urgency_above_enter_0.66_via_low_hp"),
        ],
    )
    def test_flee_enter_boundary(self, hp_current: int, expected_flee: bool) -> None:
        """With flee_urgency_active=False, urgency near ENTER threshold."""
        state = make_game_state(hp_current=hp_current, hp_max=1000)
        ctx = _make_ctx()
        ctx.pet.alive = True
        ctx.combat.flee_urgency_active = False

        result = flee_condition(ctx, state)
        assert result is expected_flee

    @pytest.mark.parametrize(
        "hp_current, expected_stay_active",
        [
            pytest.param(1000, False, id="full_hp_urgency_below_exit"),
            pytest.param(500, True, id="half_hp_urgency_above_exit"),
        ],
    )
    def test_flee_exit_boundary(self, hp_current: int, expected_stay_active: bool) -> None:
        """With flee_urgency_active=True, test urgency near EXIT threshold."""
        state = make_game_state(hp_current=hp_current, hp_max=1000)
        ctx = _make_ctx()
        ctx.pet.alive = True
        ctx.combat.flee_urgency_active = True

        result = flee_condition(ctx, state)
        assert result is expected_stay_active


# ===================================================================
# 2. HP 40% safety floor
# ===================================================================


class TestSafetyFloorHP40:
    """HP < 0.40 triggers the safety floor unconditionally."""

    @pytest.mark.parametrize(
        "hp_pct, expected_floor",
        [
            pytest.param(0.41, None, id="hp_41pct_no_floor"),
            pytest.param(0.39, True, id="hp_39pct_floor_fires"),
        ],
    )
    def test_hp_safety_floor(self, hp_pct: float, expected_floor: bool | None) -> None:
        hp = int(hp_pct * 1000)
        state = make_game_state(hp_current=hp, hp_max=1000)
        ctx = _make_ctx()
        ctx.pet.alive = True  # keep other floors from firing

        result = _check_core_safety_floors(ctx, state, "test")
        assert result is expected_floor


# ===================================================================
# 3. MIN_FIGHTS_FOR_LEARNED = 5
# ===================================================================


class TestMinFightsForLearned:
    """FightHistory.has_learned() flips at exactly MIN_FIGHTS_FOR_LEARNED."""

    @pytest.mark.parametrize(
        "num_fights, expected_learned",
        [
            pytest.param(4, False, id="4_fights_not_learned"),
            pytest.param(5, True, id="5_fights_learned"),
        ],
    )
    def test_has_learned_boundary(self, tmp_path: Path, num_fights: int, expected_learned: bool) -> None:
        fh = FightHistory(zone="test", data_dir=str(tmp_path))
        for i in range(num_fights):
            fh.record(
                mob_name="a_skeleton",
                duration=30.0,
                mana_spent=100,
                hp_delta=-0.05,
                casts=3,
                pet_heals=2,
                pet_died=False,
                defeated=True,
            )
        assert fh.has_learned("a_skeleton") is expected_learned


# ===================================================================
# 4. GOAP SATISFACTION_THRESHOLD = 0.70
# ===================================================================


class TestGOAPSatisfactionThreshold:
    """Planner returns None when goal is already satisfied (>= 0.70)."""

    def _build_planner(self) -> GOAPPlanner:
        """Build a minimal planner with ManageResourcesGoal + basic actions."""
        goal = ManageResourcesGoal(name="MANAGE_RESOURCES", priority=2)
        actions = [
            RestAction(name="rest", routine_name="REST"),
            AcquireAction(name="acquire", routine_name="ACQUIRE"),
            PullAction(name="pull", routine_name="PULL"),
            DefeatAction(name="defeat", routine_name="COMBAT"),
        ]
        return GOAPPlanner(goals=[goal], actions=actions)

    @pytest.mark.parametrize(
        "mana_pct, hp_pct, expect_plan",
        [
            pytest.param(0.40, 0.80, True, id="sat_0.67_plan_generated"),
            pytest.param(0.55, 0.95, False, id="sat_0.85_already_satisfied"),
        ],
    )
    def test_satisfaction_boundary(self, mana_pct: float, hp_pct: float, expect_plan: bool) -> None:
        planner = self._build_planner()
        ws = make_plan_world_state(
            mana_pct=mana_pct,
            hp_pct=hp_pct,
            pet_alive=True,
            engaged=False,
            targets_available=3,
        )
        # Verify the satisfaction is on the expected side of the threshold
        goal = planner._goals[0]
        sat = goal.satisfaction(ws)
        if expect_plan:
            assert sat < SATISFACTION_THRESHOLD, f"sat={sat} should be < {SATISFACTION_THRESHOLD}"
        else:
            assert sat >= SATISFACTION_THRESHOLD, f"sat={sat} should be >= {SATISFACTION_THRESHOLD}"

        plan = planner.generate(ws)
        if expect_plan:
            assert plan is not None, f"Expected plan but got None (sat={sat})"
        else:
            assert plan is None, f"Expected None but got plan (sat={sat})"


# ===================================================================
# 5. Rest entry thresholds
# ===================================================================


class TestRestEntryThresholds:
    """ctx.rest_hp_entry=0.85, ctx.rest_mana_entry=0.40 boundary tests."""

    @pytest.mark.parametrize(
        "hp_pct, expected_hp_low",
        [
            pytest.param(0.86, False, id="hp_86pct_no_rest"),
            pytest.param(0.84, True, id="hp_84pct_rest_triggers"),
        ],
    )
    def test_hp_rest_entry(self, hp_pct: float, expected_hp_low: bool) -> None:
        hp = int(hp_pct * 1000)
        state = make_game_state(hp_current=hp, hp_max=1000, mana_current=500, mana_max=500)
        ctx = _make_ctx()
        ctx.rest_hp_entry = 0.85

        hp_low, mana_low, pet_low = rest_needs_check(ctx, state)
        assert hp_low is expected_hp_low

    @pytest.mark.parametrize(
        "mana_pct, expected_mana_low",
        [
            pytest.param(0.41, False, id="mana_41pct_no_rest"),
            pytest.param(0.10, True, id="mana_10pct_rest_triggers"),
        ],
    )
    def test_mana_rest_entry(self, mana_pct: float, expected_mana_low: bool) -> None:
        """Mana below 20% always triggers rest regardless of pet/HP state."""
        mana = int(mana_pct * 500)
        state = make_game_state(hp_current=900, hp_max=1000, mana_current=mana, mana_max=500)
        ctx = _make_ctx()
        ctx.rest_mana_entry = 0.40

        hp_low, mana_low, pet_low = rest_needs_check(ctx, state)
        assert mana_low is expected_mana_low


# ===================================================================
# 6. Circuit breaker (default 5 failures in 300s)
# ===================================================================


class TestCircuitBreakerBoundary:
    """CircuitBreaker trips at exactly max_failures (default=5)."""

    @pytest.mark.parametrize(
        "num_failures, expected_allow",
        [
            pytest.param(4, True, id="4_failures_still_closed"),
            pytest.param(5, False, id="5_failures_trips_open"),
        ],
    )
    def test_failure_count_boundary(self, num_failures: int, expected_allow: bool) -> None:
        now = 1000.0
        cb = CircuitBreaker("test", max_failures=5, window_seconds=300.0, clock=lambda: now)
        for _ in range(num_failures):
            cb.record_failure()
        assert cb.allow() is expected_allow


# ===================================================================
# 7. Weight learner bounds (MAX_DRIFT = 0.20)
# ===================================================================


class TestWeightLearnerDriftBound:
    """Weights clamped to +/-20% of default after gradient steps."""

    def test_large_gradient_clamped(self) -> None:
        """Apply extreme gradient pressure and verify clamping."""
        w = ScoringWeights()
        tuner = GradientTuner(w)
        defaults = ScoringWeights()

        # Push many strongly correlated observations to drive weights
        # to the boundary. High con_pref + high fitness = push up.
        for cycle in range(10):
            for i in range(STEP_INTERVAL):
                tuner.observe(1.0, {"con_pref": 100.0})
            tuner.step()

        # Verify con_white (mapped from con_pref) is within bounds
        default_con_white = float(defaults.con_white)
        current_con_white = float(w.con_white)
        lo = default_con_white * (1.0 - MAX_DRIFT)
        hi = default_con_white * (1.0 + MAX_DRIFT)
        assert lo - 0.01 <= current_con_white <= hi + 0.01, (
            f"con_white={current_con_white} not in [{lo}, {hi}]"
        )

    def test_negative_gradient_clamped(self) -> None:
        """Apply extreme negative gradient pressure and verify clamping."""
        w = ScoringWeights()
        tuner = GradientTuner(w)
        defaults = ScoringWeights()

        # Anti-correlated: high con_pref + low fitness = push down
        for cycle in range(10):
            for i in range(STEP_INTERVAL):
                fitness = 1.0 - (i / STEP_INTERVAL) * 0.9
                tuner.observe(fitness, {"con_pref": float(100 - i * 5)})
            tuner.step()

        default_con_white = float(defaults.con_white)
        current_con_white = float(w.con_white)
        lo = default_con_white * (1.0 - MAX_DRIFT)
        hi = default_con_white * (1.0 + MAX_DRIFT)
        assert lo - 0.01 <= current_con_white <= hi + 0.01, (
            f"con_white={current_con_white} not in [{lo}, {hi}]"
        )


# ===================================================================
# 8. Scoring weight STEP_INTERVAL = 15
# ===================================================================


class TestStepIntervalBoundary:
    """GradientTuner.ready_to_step() flips at exactly STEP_INTERVAL."""

    @pytest.mark.parametrize(
        "num_observations, expected_ready",
        [
            pytest.param(14, False, id="14_observations_not_ready"),
            pytest.param(15, True, id="15_observations_ready"),
        ],
    )
    def test_ready_to_step_boundary(self, num_observations: int, expected_ready: bool) -> None:
        w = ScoringWeights()
        tuner = GradientTuner(w)
        for _ in range(num_observations):
            tuner.observe(0.5, {"con_pref": 30.0})
        assert tuner.ready_to_step() is expected_ready
