"""Tests for brain.goap.goals  -- goal satisfaction and insistence invariants.

Goals drive the GOAP planner by expressing what the agent wants and how
urgently. These tests verify range invariants and known-value behavior
for the standard goal set.
"""

from __future__ import annotations

import pytest
from hypothesis import given

from brain.goap.goals import (
    GainXPGoal,
    ManageInventoryGoal,
    ManageResourcesGoal,
    SurviveGoal,
    build_goal_set,
)
from brain.goap.world_state import PlanWorldState
from tests.factories import st_plan_world_state


def _get_goal(goal_type: type) -> object:
    """Get a properly constructed goal instance from the standard set."""
    for g in build_goal_set():
        if isinstance(g, goal_type):
            return g
    raise ValueError(f"No {goal_type.__name__} in build_goal_set()")


class TestGoalSetStructure:
    def test_build_returns_goals(self) -> None:
        goals = build_goal_set()
        assert len(goals) >= 3
        assert all(hasattr(g, "satisfaction") for g in goals)
        assert all(hasattr(g, "insistence") for g in goals)


class TestSatisfactionRange:
    @given(ws=st_plan_world_state)
    def test_survive(self, ws: PlanWorldState) -> None:
        assert 0.0 <= _get_goal(SurviveGoal).satisfaction(ws) <= 1.0

    @given(ws=st_plan_world_state)
    def test_gain_xp(self, ws: PlanWorldState) -> None:
        assert 0.0 <= _get_goal(GainXPGoal).satisfaction(ws) <= 1.0

    @given(ws=st_plan_world_state)
    def test_manage_resources(self, ws: PlanWorldState) -> None:
        assert 0.0 <= _get_goal(ManageResourcesGoal).satisfaction(ws) <= 1.0

    @given(ws=st_plan_world_state)
    def test_manage_inventory(self, ws: PlanWorldState) -> None:
        assert 0.0 <= _get_goal(ManageInventoryGoal).satisfaction(ws) <= 1.0


class TestInsistenceRange:
    @given(ws=st_plan_world_state)
    def test_survive(self, ws: PlanWorldState) -> None:
        assert 0.0 <= _get_goal(SurviveGoal).insistence(ws) <= 1.0

    @given(ws=st_plan_world_state)
    def test_gain_xp(self, ws: PlanWorldState) -> None:
        assert 0.0 <= _get_goal(GainXPGoal).insistence(ws) <= 1.0


class TestKnownValues:
    def test_survive_full_hp_satisfied(self) -> None:
        ws = PlanWorldState(hp_pct=1.0, mana_pct=1.0, pet_alive=True)
        assert _get_goal(SurviveGoal).satisfaction(ws) > 0.8

    def test_survive_critical_hp_urgent(self) -> None:
        ws = PlanWorldState(hp_pct=0.1)
        assert _get_goal(SurviveGoal).insistence(ws) == pytest.approx(1.0)

    def test_inventory_below_threshold_satisfied(self) -> None:
        ws = PlanWorldState(inventory_pct=0.3)
        assert _get_goal(ManageInventoryGoal).satisfaction(ws) == pytest.approx(1.0)

    def test_inventory_above_threshold_drops(self) -> None:
        ws = PlanWorldState(inventory_pct=0.9)
        assert _get_goal(ManageInventoryGoal).satisfaction(ws) < 0.5


# ---------------------------------------------------------------------------
# Property-based tests: goal satisfaction/insistence invariants
# ---------------------------------------------------------------------------


class TestGoalSatisfactionProperties:
    @given(ws=st_plan_world_state)
    def test_satisfaction_in_unit_interval(self, ws: PlanWorldState) -> None:
        """All goal satisfaction values are in [0.0, 1.0]."""
        for goal in build_goal_set():
            sat = goal.satisfaction(ws)
            assert 0.0 <= sat <= 1.0, f"{goal.name} satisfaction={sat}"

    @given(ws=st_plan_world_state)
    def test_insistence_non_negative(self, ws: PlanWorldState) -> None:
        """Goal insistence is never negative."""
        for goal in build_goal_set():
            ins = goal.insistence(ws)
            assert ins >= 0.0, f"{goal.name} insistence={ins}"
