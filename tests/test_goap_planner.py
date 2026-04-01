"""Tests for brain.goap.planner  -- A* plan generation over world state.

The planner searches from the current world state through action effects
to find the cheapest sequence that satisfies a goal. These tests verify
plan structure and validity without requiring a live AgentContext.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from brain.goap.actions import PlanAction, build_action_set
from brain.goap.goals import Goal, build_goal_set
from brain.goap.planner import GOAPPlanner, Plan, _Node
from brain.goap.world_state import PlanWorldState
from tests.factories import make_plan_world_state


@pytest.fixture
def planner() -> GOAPPlanner:
    """A planner with the standard goal and action sets."""
    return GOAPPlanner(goals=build_goal_set(), actions=build_action_set())


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------


class TestPlanGeneration:
    def test_low_resources_produces_plan(self, planner: GOAPPlanner) -> None:
        ws = PlanWorldState(hp_pct=0.3, mana_pct=0.2, pet_alive=True, targets_available=3)
        plan = planner.generate(ws)
        assert plan is not None
        assert len(plan.steps) > 0

    def test_ready_state_with_targets(self, planner: GOAPPlanner) -> None:
        ws = PlanWorldState(hp_pct=0.8, mana_pct=0.7, pet_alive=True, targets_available=5)
        plan = planner.generate(ws)
        # Planner may or may not find a plan depending on goal satisfaction
        if plan is not None:
            assert len(plan.steps) > 0

    def test_satisfied_state_may_return_none(self, planner: GOAPPlanner) -> None:
        ws = PlanWorldState(hp_pct=1.0, mana_pct=1.0, pet_alive=True, targets_available=0)
        plan = planner.generate(ws)
        # With no targets and full resources, planner may find nothing to do
        if plan is not None:
            assert len(plan.steps) > 0


# ---------------------------------------------------------------------------
# Plan structure
# ---------------------------------------------------------------------------


class TestPlanStructure:
    def test_steps_have_routine_names(self, planner: GOAPPlanner) -> None:
        ws = PlanWorldState(hp_pct=0.5, mana_pct=0.3, targets_available=3)
        plan = planner.generate(ws)
        assert plan is not None
        for step in plan.steps:
            assert step.routine_name, f"Step {step} has empty routine_name"

    def test_expected_cost_positive(self, planner: GOAPPlanner) -> None:
        ws = PlanWorldState(hp_pct=0.5, mana_pct=0.3, targets_available=3)
        plan = planner.generate(ws)
        assert plan is not None
        assert plan.expected_cost > 0

    def test_plan_length_within_bounds(self, planner: GOAPPlanner) -> None:
        ws = PlanWorldState(hp_pct=0.4, mana_pct=0.2, targets_available=5)
        plan = planner.generate(ws)
        assert plan is not None
        assert 1 <= len(plan.steps) <= 8


# ---------------------------------------------------------------------------
# Plan lifecycle
# ---------------------------------------------------------------------------


class TestPlanLifecycle:
    def test_has_plan_after_generate(self, planner: GOAPPlanner) -> None:
        ws = PlanWorldState(hp_pct=0.3, mana_pct=0.2, targets_available=3)
        planner.generate(ws)
        assert planner.has_plan() is True

    def test_invalidate_clears_plan(self, planner: GOAPPlanner) -> None:
        ws = PlanWorldState(hp_pct=0.3, mana_pct=0.2, targets_available=3)
        planner.generate(ws)
        planner.invalidate("emergency")
        assert planner.has_plan() is False

    def test_advance_increments_step(self, planner: GOAPPlanner) -> None:
        ws = PlanWorldState(hp_pct=0.3, mana_pct=0.2, targets_available=3)
        plan = planner.generate(ws)
        assert plan is not None
        initial_step = plan.step_index
        planner.advance(ws)
        assert plan.step_index == initial_step + 1

    def test_current_step_returns_first(self, planner: GOAPPlanner) -> None:
        ws = PlanWorldState(hp_pct=0.3, mana_pct=0.2, targets_available=3)
        plan = planner.generate(ws)
        assert plan is not None
        assert plan.current_step is plan.steps[0]

    def test_complete_finishes_plan(self, planner: GOAPPlanner) -> None:
        ws = PlanWorldState(hp_pct=0.3, mana_pct=0.2, targets_available=3)
        planner.generate(ws)
        planner.complete()
        assert planner.has_plan() is False


# ---------------------------------------------------------------------------
# Plan object
# ---------------------------------------------------------------------------


class TestPlanObject:
    def test_summary_format(self) -> None:
        goal = MagicMock(spec=Goal)
        goal.name = "survive"
        step1 = MagicMock(spec=PlanAction)
        step1.name = "rest"
        step2 = MagicMock(spec=PlanAction)
        step2.name = "acquire"
        plan = Plan(
            goal=goal,
            steps=[step1, step2],
            expected_cost=35.0,
            expected_satisfaction=0.85,
        )
        s = plan.summary()
        assert "survive" in s
        assert "rest" in s
        assert "acquire" in s
        assert "35.0s" in s

    def test_current_step_none_when_completed(self) -> None:
        goal = MagicMock(spec=Goal)
        plan = Plan(goal=goal, steps=[], expected_cost=0.0, expected_satisfaction=1.0)
        assert plan.current_step is None
        assert plan.completed is True

    def test_advance_moves_step_index(self) -> None:
        goal = MagicMock(spec=Goal)
        step = MagicMock(spec=PlanAction)
        plan = Plan(goal=goal, steps=[step], expected_cost=5.0, expected_satisfaction=0.8)
        assert plan.step_index == 0
        plan.advance()
        assert plan.step_index == 1
        assert plan.completed is True


# ---------------------------------------------------------------------------
# _Node comparison
# ---------------------------------------------------------------------------


class TestNodeComparison:
    def test_lt_by_f_cost(self) -> None:
        ws = PlanWorldState()
        n1 = _Node(state=ws, g_cost=1.0, h_cost=1.0, actions=[], depth=0)
        n2 = _Node(state=ws, g_cost=2.0, h_cost=2.0, actions=[], depth=0)
        assert n1 < n2
        assert not n2 < n1

    def test_f_cost_property(self) -> None:
        ws = PlanWorldState()
        n = _Node(state=ws, g_cost=3.0, h_cost=7.0, actions=[], depth=0)
        assert n.f_cost == 10.0


# ---------------------------------------------------------------------------
# Cost correction
# ---------------------------------------------------------------------------


class TestCostCorrection:
    def test_no_correction_under_3_observations(self, planner: GOAPPlanner) -> None:
        action = MagicMock(spec=PlanAction)
        action.name = "rest"
        action.estimate_cost.return_value = 30.0
        planner._cost_corrections["rest"] = 5.0
        planner._cost_correction_counts["rest"] = 2
        cost = planner.get_corrected_cost(action, None)
        assert cost == 30.0  # no correction applied

    def test_correction_applied_after_3_observations(self, planner: GOAPPlanner) -> None:
        action = MagicMock(spec=PlanAction)
        action.name = "rest"
        action.estimate_cost.return_value = 30.0
        planner._cost_corrections["rest"] = 5.0
        planner._cost_correction_counts["rest"] = 3
        cost = planner.get_corrected_cost(action, None)
        assert cost == 35.0

    def test_correction_floor_at_10_pct(self, planner: GOAPPlanner) -> None:
        action = MagicMock(spec=PlanAction)
        action.name = "rest"
        action.estimate_cost.return_value = 30.0
        planner._cost_corrections["rest"] = -100.0  # huge negative
        planner._cost_correction_counts["rest"] = 5
        cost = planner.get_corrected_cost(action, None)
        assert cost == 3.0  # 10% of base

    def test_update_cost_correction_ema(self, planner: GOAPPlanner) -> None:
        planner._update_cost_correction("defeat", 10.0)
        assert planner._cost_corrections["defeat"] == pytest.approx(3.0)
        assert planner._cost_correction_counts["defeat"] == 1
        planner._update_cost_correction("defeat", 10.0)
        # EMA: 3.0 * 0.7 + 10.0 * 0.3 = 5.1
        assert planner._cost_corrections["defeat"] == pytest.approx(5.1)
        assert planner._cost_correction_counts["defeat"] == 2

    def test_load_cost_corrections(self, planner: GOAPPlanner) -> None:
        planner.load_cost_corrections({"rest": 2.5, "pull": -1.0})
        assert planner._cost_corrections == {"rest": 2.5, "pull": -1.0}
        assert planner._cost_correction_counts["rest"] >= 3
        assert planner._cost_correction_counts["pull"] >= 3

    def test_cost_corrections_property(self, planner: GOAPPlanner) -> None:
        planner._cost_corrections["wander"] = 4.0
        d = planner.cost_corrections
        assert d == {"wander": 4.0}
        # Should be a copy
        d["wander"] = 99.0
        assert planner._cost_corrections["wander"] == 4.0


# ---------------------------------------------------------------------------
# stats_summary
# ---------------------------------------------------------------------------


class TestStatsSummary:
    def test_format(self, planner: GOAPPlanner) -> None:
        planner._plans_generated = 10
        planner._plans_completed = 7
        planner._plans_invalidated = 3
        planner._cost_errors = [1.0, -2.0, 3.0]
        s = planner.stats_summary()
        assert "10 generated" in s
        assert "7 completed" in s
        assert "3 invalidated" in s
        assert "70%" in s

    def test_zero_generated(self, planner: GOAPPlanner) -> None:
        s = planner.stats_summary()
        assert "0 generated" in s
        assert "0%" in s

    def test_no_cost_errors(self, planner: GOAPPlanner) -> None:
        planner._plans_generated = 1
        planner._plans_completed = 1
        s = planner.stats_summary()
        assert "+0.0s" in s


# ---------------------------------------------------------------------------
# Advance with cost tracking and early completion
# ---------------------------------------------------------------------------


class TestAdvanceWithTracking:
    def test_advance_tracks_cost_accuracy(self, planner: GOAPPlanner) -> None:
        ws = make_plan_world_state(hp_pct=0.3, mana_pct=0.2, targets_available=3)
        plan = planner.generate(ws)
        assert plan is not None

        # Simulate start_step
        planner.start_step()
        assert planner._step_start_time > 0

        # Advance with same world state (plan step completes)
        planner.advance(ws)
        # Cost error should have been recorded
        assert len(planner._cost_errors) >= 1

    def test_advance_early_satisfaction(self, planner: GOAPPlanner) -> None:
        ws_low = make_plan_world_state(hp_pct=0.3, mana_pct=0.2, targets_available=3)
        plan = planner.generate(ws_low)
        assert plan is not None

        # Advance with a fully satisfied world state
        ws_satisfied = make_plan_world_state(hp_pct=1.0, mana_pct=1.0, pet_alive=True)
        planner.advance(ws_satisfied)
        # Plan should be completed early
        assert planner.has_plan() is False

    def test_advance_no_plan_does_nothing(self, planner: GOAPPlanner) -> None:
        planner.advance(PlanWorldState())
        # Should not raise

    def test_advance_preconditions_invalidate(self, planner: GOAPPlanner) -> None:
        ws = make_plan_world_state(hp_pct=0.3, mana_pct=0.2, targets_available=3)
        plan = planner.generate(ws)
        assert plan is not None
        assert len(plan.steps) >= 1

        # Advance step 1, then present a world state where next step fails
        ws_bad = make_plan_world_state(
            hp_pct=0.3,
            mana_pct=0.2,
            engaged=True,
            pet_alive=False,
            targets_available=0,
            nearby_threats=5,
        )
        planner.advance(ws_bad)
        # Plan should have been invalidated or completed

    def test_start_step_records_estimate(self, planner: GOAPPlanner) -> None:
        ws = make_plan_world_state(hp_pct=0.3, mana_pct=0.2, targets_available=3)
        planner.generate(ws)
        planner.start_step()
        assert planner._step_estimated_cost > 0
        assert planner._step_start_time > 0


# ---------------------------------------------------------------------------
# Planner with no goals
# ---------------------------------------------------------------------------


class TestMCRobustnessGate:
    """Monte Carlo evaluation gates plan acceptance."""

    def test_robust_plan_accepted(self, planner: GOAPPlanner) -> None:
        """A plan from low resources should pass MC gate (robustness > 0.50)."""
        ws = PlanWorldState(hp_pct=0.3, mana_pct=0.2, pet_alive=True, targets_available=3)
        plan = planner.generate(ws)
        # Low-resource state reliably produces a plan with rest -> acquire steps
        assert plan is not None
        assert plan.expected_satisfaction >= 0.50

    def test_mc_sat_stored_on_plan(self, planner: GOAPPlanner) -> None:
        ws = PlanWorldState(hp_pct=0.4, mana_pct=0.3, pet_alive=True, targets_available=5)
        plan = planner.generate(ws)
        assert plan is not None
        # expected_satisfaction comes from MC evaluation, not deterministic check
        assert 0.0 <= plan.expected_satisfaction <= 1.0


class TestLearnedMCSigma:
    """MC noise sigma derived from encounter posterior variance."""

    def test_no_ctx_returns_default(self) -> None:
        from brain.goap.planner import MC_NOISE_SIGMA

        hp_sigma, mana_sigma = GOAPPlanner._learned_mc_sigma(None)
        assert hp_sigma == MC_NOISE_SIGMA
        assert mana_sigma == MC_NOISE_SIGMA

    def test_no_fight_history_returns_default(self) -> None:
        from types import SimpleNamespace

        from brain.goap.planner import MC_NOISE_SIGMA

        ctx = SimpleNamespace(fight_history=None)
        hp_sigma, mana_sigma = GOAPPlanner._learned_mc_sigma(ctx)
        assert hp_sigma == MC_NOISE_SIGMA
        assert mana_sigma == MC_NOISE_SIGMA

    def test_learned_variance_narrows_sigma(self) -> None:
        from types import SimpleNamespace

        # Tight posteriors (low variance) should produce small sigma
        stats = SimpleNamespace(danger_post_var=0.01, mana_post_var=0.01, fights=10)
        fh = SimpleNamespace(get_all_stats=lambda: {"skeleton": stats})
        ctx = SimpleNamespace(fight_history=fh)
        hp_sigma, mana_sigma = GOAPPlanner._learned_mc_sigma(ctx)
        assert hp_sigma < 0.15  # tighter than default
        assert mana_sigma < 0.15

    def test_wide_posteriors_widen_sigma(self) -> None:
        from types import SimpleNamespace

        # Wide posteriors (high variance, few observations) -> larger sigma
        stats = SimpleNamespace(danger_post_var=0.10, mana_post_var=0.15, fights=2)
        fh = SimpleNamespace(get_all_stats=lambda: {"unknown_mob": stats})
        ctx = SimpleNamespace(fight_history=fh)
        hp_sigma, mana_sigma = GOAPPlanner._learned_mc_sigma(ctx)
        assert hp_sigma >= 0.10
        assert mana_sigma >= 0.10

    def test_sigma_clamped_to_bounds(self) -> None:
        from types import SimpleNamespace

        # Extremely tight variance should clamp to 0.02 floor
        stats = SimpleNamespace(danger_post_var=0.0001, mana_post_var=0.0001, fights=100)
        fh = SimpleNamespace(get_all_stats=lambda: {"trivial": stats})
        ctx = SimpleNamespace(fight_history=fh)
        hp_sigma, mana_sigma = GOAPPlanner._learned_mc_sigma(ctx)
        assert hp_sigma >= 0.02
        assert mana_sigma >= 0.02


class TestPlannerEdgeCases:
    def test_no_goals_returns_none(self) -> None:
        p = GOAPPlanner(goals=[], actions=build_action_set())
        ws = make_plan_world_state(hp_pct=0.3, targets_available=5)
        assert p.generate(ws) is None

    def test_no_actions_returns_none(self) -> None:
        p = GOAPPlanner(goals=build_goal_set(), actions=[])
        ws = make_plan_world_state(hp_pct=0.3, targets_available=5)
        plan = p.generate(ws)
        # With no actions, planner can't find a plan
        assert plan is None

    def test_current_step_no_plan(self, planner: GOAPPlanner) -> None:
        assert planner.current_step is None

    def test_invalidate_without_plan(self, planner: GOAPPlanner) -> None:
        planner.invalidate("test")
        assert planner.has_plan() is False

    def test_complete_without_plan(self, planner: GOAPPlanner) -> None:
        planner.complete()
        assert planner.has_plan() is False
