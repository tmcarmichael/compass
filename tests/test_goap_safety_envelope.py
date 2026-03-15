"""Property-based stateful tests for GOAP planner invariants.

Uses hypothesis.stateful.RuleBasedStateMachine to verify that no random
sequence of world state mutations, plan generations, advances, and
invalidations can violate core planner properties.

Supplements test_safety_envelope.py (Brain rule invariants) with
planner-specific guarantees.
"""

from __future__ import annotations

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    rule,
)

from brain.goap.actions import build_action_set
from brain.goap.goals import build_goal_set
from brain.goap.planner import GOAPPlanner
from brain.goap.world_state import PlanWorldState

# ---------------------------------------------------------------------------
# Stateful machine
# ---------------------------------------------------------------------------


class GOAPPlannerMachine(RuleBasedStateMachine):
    """Verify GOAP planner invariants under random state sequences."""

    def __init__(self) -> None:
        super().__init__()
        self.planner = GOAPPlanner(goals=build_goal_set(), actions=build_action_set())
        self.ws = PlanWorldState()

    # -- State mutations ---------------------------------------------------

    @rule(hp=st.floats(min_value=0.0, max_value=1.0))
    def set_hp(self, hp: float) -> None:
        self.ws = self.ws.with_changes(hp_pct=hp)

    @rule(mana=st.floats(min_value=0.0, max_value=1.0))
    def set_mana(self, mana: float) -> None:
        self.ws = self.ws.with_changes(mana_pct=mana)

    @rule(alive=st.booleans())
    def set_pet(self, alive: bool) -> None:
        self.ws = self.ws.with_changes(pet_alive=alive)

    @rule(n=st.integers(min_value=0, max_value=10))
    def set_targets(self, n: int) -> None:
        self.ws = self.ws.with_changes(targets_available=n)

    @rule(engaged=st.booleans())
    def set_engaged(self, engaged: bool) -> None:
        self.ws = self.ws.with_changes(engaged=engaged)

    @rule(n=st.integers(min_value=0, max_value=5))
    def set_threats(self, n: int) -> None:
        self.ws = self.ws.with_changes(nearby_threats=n)

    # -- Planner actions ---------------------------------------------------

    @rule()
    def generate_plan(self) -> None:
        self.planner.generate(self.ws)

    @rule()
    def advance_plan(self) -> None:
        if self.planner.has_plan():
            step = self.planner.current_step
            if step:
                self.planner.start_step()
                self.ws = step.apply_effects(self.ws)
            self.planner.advance(self.ws)

    @rule()
    def invalidate_plan(self) -> None:
        self.planner.invalidate("random_test")

    @rule()
    def complete_plan(self) -> None:
        self.planner.complete()

    # -- Invariants --------------------------------------------------------

    @invariant()
    def step_index_bounded(self) -> None:
        """step_index never exceeds the number of steps."""
        plan = self.planner.plan
        if plan is not None:
            assert plan.step_index <= len(plan.steps)

    @invariant()
    def steps_have_routine_names(self) -> None:
        """Every step in any active plan has a non-empty routine_name."""
        plan = self.planner.plan
        if plan is not None:
            for step in plan.steps:
                assert step.routine_name, f"Step '{step.name}' has empty routine_name"

    @invariant()
    def positive_expected_cost(self) -> None:
        """Active plans have positive expected cost."""
        plan = self.planner.plan
        if plan is not None and not plan.completed:
            assert plan.expected_cost > 0

    @invariant()
    def no_consecutive_duplicates(self) -> None:
        """No two consecutive steps in a plan have the same action name."""
        plan = self.planner.plan
        if plan is not None:
            for i in range(1, len(plan.steps)):
                assert plan.steps[i].name != plan.steps[i - 1].name, (
                    f"Consecutive duplicate action: {plan.steps[i].name} at positions {i - 1} and {i}"
                )

    @invariant()
    def completed_or_has_current_step(self) -> None:
        """An active plan either has a current step or is completed."""
        plan = self.planner.plan
        if plan is not None:
            if not plan.completed:
                assert plan.current_step is not None

    @invariant()
    def stats_consistent(self) -> None:
        """Generated count >= completed + invalidated."""
        assert self.planner._plans_generated >= (
            self.planner._plans_completed + self.planner._plans_invalidated
        )


TestGOAPPlannerSafety = GOAPPlannerMachine.TestCase
TestGOAPPlannerSafety.settings = settings(
    max_examples=200,
    stateful_step_count=40,
)
