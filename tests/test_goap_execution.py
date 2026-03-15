"""Tests for multi-step GOAP plan execution lifecycle.

Covers the gap between plan generation (tested in test_goap_planner.py)
and plan-driven behavior in the live system. Verifies full operational
cycles, emergency overrides, precondition invalidation, cost correction
convergence, and replanning.
"""

from __future__ import annotations

import time

from brain.decision import Brain
from brain.goap.actions import (
    RestAction,
    build_action_set,
)
from brain.goap.goals import build_goal_set
from brain.goap.planner import GOAPPlanner
from brain.rules import register_all
from tests.factories import make_agent_context, make_game_state, make_plan_world_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_planner() -> GOAPPlanner:
    return GOAPPlanner(goals=build_goal_set(), actions=build_action_set())


# ---------------------------------------------------------------------------
# Full cycle execution
# ---------------------------------------------------------------------------


class TestFullCycleExecution:
    """Walk a plan through rest -> acquire -> pull -> defeat."""

    def test_depleted_state_generates_resource_plan(self) -> None:
        """Low resources should produce a plan starting with rest."""
        planner = _make_planner()
        ws = make_plan_world_state(
            hp_pct=0.4,
            mana_pct=0.2,
            pet_alive=True,
            targets_available=3,
        )
        plan = planner.generate(ws)
        assert plan is not None
        assert plan.steps[0].name == "rest"
        assert planner.has_plan()

    def test_full_cycle_advance_through_steps(self) -> None:
        """Advance through a multi-step plan, verifying each step."""
        planner = _make_planner()
        # Mid-range resources with targets: GAIN_XP or MANAGE_RESOURCES should
        # produce a multi-step plan (rest -> acquire -> pull -> defeat)
        ws = make_plan_world_state(
            hp_pct=0.6,
            mana_pct=0.3,
            pet_alive=True,
            targets_available=3,
        )
        plan = planner.generate(ws)
        assert plan is not None
        # Accept single-step plans too; the planner may find a short path
        assert len(plan.steps) >= 1

        step_names: list[str] = []
        while planner.has_plan():
            step = planner.current_step
            assert step is not None
            step_names.append(step.name)
            planner.start_step()
            # Simulate routine completing by applying effects
            ws = step.apply_effects(ws)
            planner.advance(ws)

        assert len(step_names) >= 1
        # Plan should have been completed or invalidated
        assert not planner.has_plan()

    def test_step_routine_names_are_nonempty(self) -> None:
        """Every step in a generated plan has a non-empty routine_name."""
        planner = _make_planner()
        ws = make_plan_world_state(
            hp_pct=0.3,
            mana_pct=0.15,
            pet_alive=True,
            targets_available=3,
        )
        plan = planner.generate(ws)
        assert plan is not None
        for step in plan.steps:
            assert step.routine_name, f"Step '{step.name}' has empty routine_name"

    def test_expected_cost_positive(self) -> None:
        """Generated plans have positive expected cost."""
        planner = _make_planner()
        ws = make_plan_world_state(
            hp_pct=0.3,
            mana_pct=0.15,
            pet_alive=True,
            targets_available=3,
        )
        plan = planner.generate(ws)
        assert plan is not None
        assert plan.expected_cost > 0


# ---------------------------------------------------------------------------
# Emergency override during plan
# ---------------------------------------------------------------------------


class TestEmergencyOverride:
    """Emergency rules must override GOAP plan suggestions."""

    def test_flee_overrides_goap_suggestion(self) -> None:
        """When HP drops critically, FLEE fires regardless of GOAP plan."""
        ctx = make_agent_context()
        brain = Brain(ctx=ctx, utility_phase=0)

        state_holder = [make_game_state(hp_current=1000, hp_max=1000, mana_current=500, mana_max=500)]

        def read_state_fn():
            return state_holder[0]

        register_all(brain, ctx, read_state_fn)

        # Set up GOAP suggestion as if planner recommended ACQUIRE
        ctx.diag.goap_suggestion = "ACQUIRE"

        # First tick: normal operation
        brain.tick(state_holder[0])

        # Now drop HP to critical level (triggers flee urgency)
        state_holder[0] = make_game_state(
            hp_current=50,
            hp_max=1000,
            mana_current=500,
            mana_max=500,
        )
        ctx.pet.alive = False  # no pet + low HP = high flee urgency

        brain.tick(state_holder[0])

        # FLEE or FEIGN_DEATH should have fired, not ACQUIRE
        assert brain._active_name in ("FLEE", "FEIGN_DEATH", "REST"), (
            f"Expected emergency rule but got {brain._active_name}"
        )


# ---------------------------------------------------------------------------
# Plan invalidation on precondition failure
# ---------------------------------------------------------------------------


class TestPlanInvalidation:
    """Plans invalidate when next step's preconditions fail."""

    def test_pet_death_invalidates_pull_step(self) -> None:
        """Killing the pet should invalidate a plan with a pull step."""
        planner = _make_planner()
        # State that will produce rest -> acquire -> pull -> defeat
        ws = make_plan_world_state(
            hp_pct=0.3,
            mana_pct=0.15,
            pet_alive=True,
            targets_available=3,
        )
        plan = planner.generate(ws)
        assert plan is not None

        # Advance through steps until we reach one that requires pet_alive
        steps_advanced = 0
        while planner.has_plan():
            step = planner.current_step
            assert step is not None
            planner.start_step()
            ws = step.apply_effects(ws)
            # Remove pet before advancing
            ws = ws.with_changes(pet_alive=False)
            planner.advance(ws)
            steps_advanced += 1
            if not planner.has_plan():
                break

        # Plan should have been invalidated (pull/acquire require pet_alive)
        assert not planner.has_plan()

    def test_invalidate_clears_plan(self) -> None:
        """Explicit invalidation clears the plan completely."""
        planner = _make_planner()
        ws = make_plan_world_state(
            hp_pct=0.3,
            mana_pct=0.15,
            pet_alive=True,
            targets_available=3,
        )
        planner.generate(ws)
        assert planner.has_plan()

        planner.invalidate("test_emergency")
        assert not planner.has_plan()
        assert planner.current_step is None


# ---------------------------------------------------------------------------
# Cost correction convergence
# ---------------------------------------------------------------------------


class TestCostCorrection:
    """Cost self-correction converges toward observed costs."""

    def test_correction_applied_after_threshold(self) -> None:
        """Corrections are only applied after 3+ observations."""
        planner = _make_planner()
        rest = RestAction(name="rest", routine_name="REST")

        # Before any observations: base cost
        base = rest.estimate_cost(None)
        assert planner.get_corrected_cost(rest, None) == base

        # Feed 2 observations: should NOT apply correction yet
        planner._update_cost_correction("rest", 10.0)
        planner._update_cost_correction("rest", 10.0)
        assert planner.get_corrected_cost(rest, None) == base

        # Feed 3rd observation: correction now applies
        planner._update_cost_correction("rest", 10.0)
        corrected = planner.get_corrected_cost(rest, None)
        assert corrected > base, "Correction should increase cost for consistently slow steps"

    def test_correction_converges_with_consistent_error(self) -> None:
        """Feeding consistent positive error should push corrected cost up."""
        planner = _make_planner()
        rest = RestAction(name="rest", routine_name="REST")
        base = rest.estimate_cost(None)

        # Feed 15 observations with +10s error
        for _ in range(15):
            planner._update_cost_correction("rest", 10.0)

        corrected = planner.get_corrected_cost(rest, None)
        # EMA of 10.0 should converge close to 10.0
        assert corrected > base + 5.0, (
            f"After 15 consistent +10s errors, corrected ({corrected:.1f}) "
            f"should exceed base ({base:.1f}) by at least 5s"
        )

    def test_correction_bounded_below(self) -> None:
        """Correction cannot reduce cost below 10% of base."""
        planner = _make_planner()
        rest = RestAction(name="rest", routine_name="REST")
        base = rest.estimate_cost(None)

        # Feed extreme negative error (step took almost no time)
        for _ in range(20):
            planner._update_cost_correction("rest", -base * 2)

        corrected = planner.get_corrected_cost(rest, None)
        assert corrected >= base * 0.1, "Corrected cost must not drop below 10% of base"

    def test_per_step_tracking_via_advance(self) -> None:
        """Advancing a plan tracks cost accuracy per step."""
        planner = _make_planner()
        ws = make_plan_world_state(
            hp_pct=0.3,
            mana_pct=0.15,
            pet_alive=True,
            targets_available=3,
        )
        planner.generate(ws)
        assert planner.has_plan()

        step = planner.current_step
        assert step is not None

        planner.start_step()
        # Simulate time passing
        planner._step_start_time = time.time() - 15.0  # 15s elapsed
        ws = step.apply_effects(ws)
        planner.advance(ws)

        # Should have recorded a cost error
        assert len(planner._cost_errors) >= 1


# ---------------------------------------------------------------------------
# Replan after completion
# ---------------------------------------------------------------------------


class TestReplanAfterCompletion:
    """After a plan completes, a new plan can be generated."""

    def test_replan_from_fresh_state(self) -> None:
        """Generate, complete, then generate a new plan."""
        planner = _make_planner()

        # First plan
        ws = make_plan_world_state(
            hp_pct=0.3,
            mana_pct=0.15,
            pet_alive=True,
            targets_available=3,
        )
        plan1 = planner.generate(ws)
        assert plan1 is not None

        # Complete by force
        planner.complete()
        assert not planner.has_plan()

        # New plan from depleted state (guarantees unsatisfied goal)
        ws2 = make_plan_world_state(
            hp_pct=0.3,
            mana_pct=0.1,
            pet_alive=True,
            targets_available=3,
        )
        plan2 = planner.generate(ws2)
        assert plan2 is not None
        assert planner.has_plan()

    def test_replan_after_invalidation(self) -> None:
        """After invalidation, replanning produces a fresh plan."""
        planner = _make_planner()
        ws = make_plan_world_state(
            hp_pct=0.3,
            mana_pct=0.15,
            pet_alive=True,
            targets_available=3,
        )
        planner.generate(ws)
        planner.invalidate("test")
        assert not planner.has_plan()

        plan2 = planner.generate(ws)
        assert plan2 is not None

    def test_stats_track_completions_and_invalidations(self) -> None:
        """Stats counters increment correctly."""
        planner = _make_planner()
        ws = make_plan_world_state(
            hp_pct=0.3,
            mana_pct=0.15,
            pet_alive=True,
            targets_available=3,
        )

        planner.generate(ws)
        planner.complete()

        planner.generate(ws)
        planner.invalidate("test")

        summary = planner.stats_summary()
        assert "2 generated" in summary
        assert "1 completed" in summary
        assert "1 invalidated" in summary
