"""Tests for brain.goap.world_state  -- PlanWorldState immutable contract.

PlanWorldState is the planner's view of the world: a frozen dataclass
with with_changes() for immutable modification. Every field has a default,
so constructing test instances requires zero boilerplate.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from brain.goap.world_state import PlanWorldState


class TestDefaults:
    def test_default_values(self, plan_world_state: PlanWorldState) -> None:
        ws = plan_world_state
        assert ws.hp_pct == 1.0
        assert ws.mana_pct == 1.0
        assert ws.pet_alive is True
        assert ws.engaged is False
        assert ws.has_target is False
        assert ws.targets_available == 0


class TestFrozenContract:
    def test_immutable(self, plan_world_state: PlanWorldState) -> None:
        mutable: Any = plan_world_state
        with pytest.raises(dataclasses.FrozenInstanceError):
            mutable.hp_pct = 0.5

    def test_has_slots(self, plan_world_state: PlanWorldState) -> None:
        assert not hasattr(plan_world_state, "__dict__")

    def test_hashable(self, plan_world_state: PlanWorldState) -> None:
        s = {plan_world_state}
        assert plan_world_state in s

    def test_equality(self) -> None:
        a = PlanWorldState()
        b = PlanWorldState()
        assert a == b

    def test_inequality(self) -> None:
        a = PlanWorldState(hp_pct=1.0)
        b = PlanWorldState(hp_pct=0.5)
        assert a != b


class TestWithChanges:
    def test_returns_new_instance(self, plan_world_state: PlanWorldState) -> None:
        modified = plan_world_state.with_changes(hp_pct=0.5)
        assert modified is not plan_world_state

    def test_changes_specified_field(self) -> None:
        ws = PlanWorldState()
        modified = ws.with_changes(hp_pct=0.3)
        assert modified.hp_pct == 0.3

    def test_preserves_unchanged_fields(self) -> None:
        ws = PlanWorldState(mana_pct=0.8, pet_alive=True)
        modified = ws.with_changes(hp_pct=0.5)
        assert modified.mana_pct == 0.8
        assert modified.pet_alive is True

    def test_multiple_fields(self) -> None:
        ws = PlanWorldState()
        modified = ws.with_changes(hp_pct=0.5, mana_pct=0.3, engaged=True)
        assert modified.hp_pct == 0.5
        assert modified.mana_pct == 0.3
        assert modified.engaged is True

    def test_no_args_returns_equal_copy(self) -> None:
        ws = PlanWorldState(hp_pct=0.7)
        copy = ws.with_changes()
        assert copy == ws
        assert copy is not ws


# ---------------------------------------------------------------------------
# build_world_state -- extracts PlanWorldState from GameState + AgentContext
# ---------------------------------------------------------------------------


class TestBuildWorldState:
    """Tests for build_world_state factory function."""

    def _make_ctx(
        self,
        *,
        pet_alive: bool = True,
        engaged: bool = False,
        camp_x: float = 0.0,
        camp_y: float = 0.0,
        roam_radius: float = 200.0,
        weight_baseline: float = 0,
        weight_threshold: float = 100,
        has_unlootable_corpse: bool = False,
        world_targets: list | None = None,
        world_threats: list | None = None,
    ) -> object:
        from types import SimpleNamespace

        world = SimpleNamespace(
            _targets=world_targets or [],
            _threats=world_threats or [],
        )
        return SimpleNamespace(
            pet=SimpleNamespace(alive=pet_alive),
            combat=SimpleNamespace(engaged=engaged),
            camp=SimpleNamespace(camp_x=camp_x, camp_y=camp_y, roam_radius=roam_radius),
            inventory=SimpleNamespace(
                weight_baseline=weight_baseline,
                weight_threshold=weight_threshold,
            ),
            has_unlootable_corpse=lambda state, max_dist=100.0: has_unlootable_corpse,
            world=world,
        )

    def test_basic_build(self) -> None:
        from brain.goap.world_state import build_world_state
        from tests.factories import make_game_state

        state = make_game_state(hp_current=800, hp_max=1000, mana_current=300, mana_max=500)
        ctx = self._make_ctx()
        ws = build_world_state(state, ctx)
        assert ws.hp_pct == 0.8
        assert ws.mana_pct == 0.6
        assert ws.pet_alive is True
        assert ws.engaged is False

    def test_targets_and_threats(self) -> None:
        from types import SimpleNamespace

        from brain.goap.world_state import build_world_state
        from tests.factories import make_game_state

        targets = [SimpleNamespace(score=10), SimpleNamespace(score=0), SimpleNamespace(score=5)]
        threats = [SimpleNamespace(), SimpleNamespace()]
        state = make_game_state()
        ctx = self._make_ctx(world_targets=targets, world_threats=threats)
        ws = build_world_state(state, ctx)
        assert ws.targets_available == 2  # only score > 0
        assert ws.nearby_threats == 2

    def test_corpse_nearby(self) -> None:
        from brain.goap.world_state import build_world_state
        from tests.factories import make_game_state

        state = make_game_state()
        ctx = self._make_ctx(has_unlootable_corpse=True)
        ws = build_world_state(state, ctx)
        assert ws.corpse_nearby is True

    def test_at_camp_within_radius(self) -> None:
        from brain.goap.world_state import build_world_state
        from tests.factories import make_game_state

        state = make_game_state(x=10.0, y=10.0)
        ctx = self._make_ctx(camp_x=0.0, camp_y=0.0, roam_radius=200.0)
        ws = build_world_state(state, ctx)
        assert ws.at_camp is True

    def test_at_camp_outside_radius(self) -> None:
        from brain.goap.world_state import build_world_state
        from tests.factories import make_game_state

        state = make_game_state(x=1000.0, y=1000.0)
        ctx = self._make_ctx(camp_x=0.0, camp_y=0.0, roam_radius=50.0)
        ws = build_world_state(state, ctx)
        assert ws.at_camp is False

    def test_inventory_pct(self) -> None:
        from brain.goap.world_state import build_world_state
        from tests.factories import make_game_state

        state = make_game_state(weight=150)
        ctx = self._make_ctx(weight_baseline=100, weight_threshold=200)
        ws = build_world_state(state, ctx)
        assert ws.inventory_pct == 0.25  # (150-100)/200

    def test_inventory_pct_capped_at_1(self) -> None:
        from brain.goap.world_state import build_world_state
        from tests.factories import make_game_state

        state = make_game_state(weight=500)
        ctx = self._make_ctx(weight_baseline=100, weight_threshold=200)
        ws = build_world_state(state, ctx)
        assert ws.inventory_pct == 1.0

    def test_buffs_active_with_empty_buffs(self) -> None:
        from brain.goap.world_state import build_world_state
        from tests.factories import make_game_state

        state = make_game_state(buffs=[])
        ctx = self._make_ctx()
        ws = build_world_state(state, ctx)
        assert ws.buffs_active is False

    def test_buffs_active_with_buffs(self) -> None:
        from brain.goap.world_state import build_world_state
        from tests.factories import make_game_state

        state = make_game_state(buffs=[1, 2, 3])
        ctx = self._make_ctx()
        ws = build_world_state(state, ctx)
        assert ws.buffs_active is True

    def test_zero_roam_radius_is_at_camp(self) -> None:
        from brain.goap.world_state import build_world_state
        from tests.factories import make_game_state

        state = make_game_state(x=5000.0, y=5000.0)
        ctx = self._make_ctx(roam_radius=0.0)
        ws = build_world_state(state, ctx)
        assert ws.at_camp is True  # roam_radius=0 means no camp check

    def test_no_world_object(self) -> None:
        from types import SimpleNamespace

        from brain.goap.world_state import build_world_state
        from tests.factories import make_game_state

        state = make_game_state()
        ctx = SimpleNamespace(
            pet=SimpleNamespace(alive=True),
            combat=SimpleNamespace(engaged=False),
            camp=SimpleNamespace(camp_x=0.0, camp_y=0.0, roam_radius=200.0),
            inventory=SimpleNamespace(weight_baseline=0, weight_threshold=100),
            has_unlootable_corpse=lambda state, max_dist=100.0: False,
            world=None,
        )
        ws = build_world_state(state, ctx)
        assert ws.targets_available == 0
        assert ws.nearby_threats == 0
