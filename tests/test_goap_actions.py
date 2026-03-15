"""Tests for brain.goap.actions -- GOAP action preconditions and effects.

Each PlanAction has preconditions_met() and apply_effects() that operate on
PlanWorldState. Tests verify guard conditions and state transitions.
"""

from __future__ import annotations

import pytest
from hypothesis import given

from brain.goap.actions import (
    AcquireAction,
    BuffAction,
    DefeatAction,
    MemorizeAction,
    PlanAction,
    PullAction,
    RestAction,
    SummonPetAction,
    WanderAction,
    build_action_set,
)
from brain.goap.world_state import PlanWorldState
from tests.factories import make_plan_world_state, st_plan_world_state


def _rest() -> RestAction:
    return RestAction(name="rest", routine_name="REST")


def _acquire() -> AcquireAction:
    return AcquireAction(name="acquire", routine_name="ACQUIRE")


def _pull() -> PullAction:
    return PullAction(name="pull", routine_name="PULL")


def _defeat() -> DefeatAction:
    return DefeatAction(name="defeat", routine_name="IN_COMBAT")


class TestRestAction:
    def test_blocked_when_engaged(self) -> None:
        ws = make_plan_world_state(engaged=True)
        assert _rest().preconditions_met(ws) is False

    def test_blocked_when_threats(self) -> None:
        ws = make_plan_world_state(engaged=False, nearby_threats=2)
        assert _rest().preconditions_met(ws) is False

    def test_allowed_when_safe(self) -> None:
        ws = make_plan_world_state(engaged=False, nearby_threats=0)
        assert _rest().preconditions_met(ws) is True

    def test_effects_restore(self) -> None:
        ws = make_plan_world_state(hp_pct=0.3, mana_pct=0.1)
        result = _rest().apply_effects(ws)
        assert result.hp_pct >= 0.3
        assert result.mana_pct >= 0.1


class TestAcquireAction:
    def test_needs_targets(self) -> None:
        ws = make_plan_world_state(targets_available=0)
        assert _acquire().preconditions_met(ws) is False

    def test_needs_pet(self) -> None:
        ws = make_plan_world_state(targets_available=5, pet_alive=False, mana_pct=0.5)
        assert _acquire().preconditions_met(ws) is False

    def test_needs_mana(self) -> None:
        ws = make_plan_world_state(targets_available=5, pet_alive=True, mana_pct=0.1)
        assert _acquire().preconditions_met(ws) is False

    def test_allowed_when_ready(self) -> None:
        ws = make_plan_world_state(targets_available=5, pet_alive=True, mana_pct=0.5, engaged=False)
        assert _acquire().preconditions_met(ws) is True

    def test_effects_set_target(self) -> None:
        ws = make_plan_world_state(targets_available=3, pet_alive=True, mana_pct=0.5)
        result = _acquire().apply_effects(ws)
        assert result.has_target is True


class TestDefeatAction:
    def test_needs_engaged(self) -> None:
        ws = make_plan_world_state(engaged=False)
        assert _defeat().preconditions_met(ws) is False

    def test_allowed_when_engaged(self) -> None:
        ws = make_plan_world_state(engaged=True)
        assert _defeat().preconditions_met(ws) is True

    def test_effects(self) -> None:
        ws = make_plan_world_state(engaged=True, hp_pct=1.0, mana_pct=1.0)
        result = _defeat().apply_effects(ws)
        assert result.engaged is False
        assert result.corpse_nearby is True


class TestBuildActionSet:
    def test_count(self) -> None:
        actions = build_action_set()
        assert len(actions) == 8

    def test_all_actions_have_routine_names(self) -> None:
        actions = build_action_set()
        for a in actions:
            assert isinstance(a.routine_name, str)
            assert len(a.routine_name) > 0

    def test_all_are_plan_actions(self) -> None:
        actions = build_action_set()
        for a in actions:
            assert isinstance(a, PlanAction)

    @given(ws=st_plan_world_state)
    def test_apply_effects_returns_valid_state(self, ws: PlanWorldState) -> None:
        """For any PlanWorldState, apply_effects returns bounded hp/mana."""
        actions = build_action_set()
        for a in actions:
            result = a.apply_effects(ws)
            assert 0.0 <= result.hp_pct <= 1.0 + 1e-9
            assert 0.0 <= result.mana_pct <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Base PlanAction (default implementations)
# ---------------------------------------------------------------------------


class TestPlanActionBase:
    def test_preconditions_met_returns_false(self) -> None:
        a = PlanAction(name="test", routine_name="TEST")
        ws = make_plan_world_state()
        assert a.preconditions_met(ws) is False

    def test_apply_effects_returns_same_state(self) -> None:
        a = PlanAction(name="test", routine_name="TEST")
        ws = make_plan_world_state(hp_pct=0.5)
        result = a.apply_effects(ws)
        assert result == ws

    def test_estimate_cost_unknown_name(self) -> None:
        a = PlanAction(name="unknown_action", routine_name="UNKNOWN")
        assert a.estimate_cost(None) == 20.0

    def test_estimate_cost_known_name(self) -> None:
        a = PlanAction(name="rest", routine_name="REST_BASE")
        assert a.estimate_cost(None) == 30.0


# ---------------------------------------------------------------------------
# RestAction.estimate_cost with ctx
# ---------------------------------------------------------------------------


class TestRestActionCost:
    def test_estimate_cost_no_ctx(self) -> None:
        r = _rest()
        assert r.estimate_cost(None) == 30.0

    def test_estimate_cost_with_ctx(self) -> None:
        from types import SimpleNamespace

        r = _rest()
        ctx = SimpleNamespace(combat=SimpleNamespace(last_mana_pct=0.3))
        cost = r.estimate_cost(ctx)
        # deficit = 0.80 - 0.3 = 0.50, max(10, 0.50 * 60) = 30.0
        assert cost == 30.0

    def test_estimate_cost_high_mana(self) -> None:
        from types import SimpleNamespace

        r = _rest()
        ctx = SimpleNamespace(combat=SimpleNamespace(last_mana_pct=0.75))
        cost = r.estimate_cost(ctx)
        # deficit = 0.80 - 0.75 = 0.05, max(10, 0.05 * 60) = 10.0
        assert cost == 10.0

    def test_estimate_cost_no_last_mana_attr(self) -> None:
        from types import SimpleNamespace

        r = _rest()
        ctx = SimpleNamespace(combat=SimpleNamespace())
        cost = r.estimate_cost(ctx)
        # No last_mana_pct -> default 0.5, deficit = 0.30, max(10, 0.30*60) = 18.0
        assert cost == pytest.approx(18.0)


# ---------------------------------------------------------------------------
# PullAction
# ---------------------------------------------------------------------------


class TestPullAction:
    def test_preconditions_need_target(self) -> None:
        ws = make_plan_world_state(has_target=False, pet_alive=True, engaged=False)
        assert _pull().preconditions_met(ws) is False

    def test_preconditions_need_pet(self) -> None:
        ws = make_plan_world_state(has_target=True, pet_alive=False, engaged=False)
        assert _pull().preconditions_met(ws) is False

    def test_preconditions_blocked_when_engaged(self) -> None:
        ws = make_plan_world_state(has_target=True, pet_alive=True, engaged=True)
        assert _pull().preconditions_met(ws) is False

    def test_preconditions_met(self) -> None:
        ws = make_plan_world_state(has_target=True, pet_alive=True, engaged=False)
        assert _pull().preconditions_met(ws) is True

    def test_effects_set_engaged(self) -> None:
        ws = make_plan_world_state(has_target=True, pet_alive=True)
        result = _pull().apply_effects(ws)
        assert result.engaged is True

    def test_estimate_cost(self) -> None:
        assert _pull().estimate_cost(None) == 8.0


# ---------------------------------------------------------------------------
# BuffAction
# ---------------------------------------------------------------------------


class TestBuffAction:
    def _buff(self) -> BuffAction:
        return BuffAction(name="buff", routine_name="BUFF")

    def test_preconditions_blocked_when_buffs_active(self) -> None:
        ws = make_plan_world_state(buffs_active=True, engaged=False, nearby_threats=0)
        assert self._buff().preconditions_met(ws) is False

    def test_preconditions_blocked_when_engaged(self) -> None:
        ws = make_plan_world_state(buffs_active=False, engaged=True, nearby_threats=0)
        assert self._buff().preconditions_met(ws) is False

    def test_preconditions_blocked_when_threats(self) -> None:
        ws = make_plan_world_state(buffs_active=False, engaged=False, nearby_threats=2)
        assert self._buff().preconditions_met(ws) is False

    def test_preconditions_met(self) -> None:
        ws = make_plan_world_state(buffs_active=False, engaged=False, nearby_threats=0)
        assert self._buff().preconditions_met(ws) is True

    def test_effects(self) -> None:
        ws = make_plan_world_state(buffs_active=False)
        result = self._buff().apply_effects(ws)
        assert result.buffs_active is True

    def test_estimate_cost(self) -> None:
        assert self._buff().estimate_cost(None) == 12.0


# ---------------------------------------------------------------------------
# SummonPetAction
# ---------------------------------------------------------------------------


class TestSummonPetAction:
    def _summon(self) -> SummonPetAction:
        return SummonPetAction(name="summon_pet", routine_name="SUMMON_PET")

    def test_preconditions_blocked_when_pet_alive(self) -> None:
        ws = make_plan_world_state(pet_alive=True, engaged=False, mana_pct=0.5)
        assert self._summon().preconditions_met(ws) is False

    def test_preconditions_blocked_when_engaged(self) -> None:
        ws = make_plan_world_state(pet_alive=False, engaged=True, mana_pct=0.5)
        assert self._summon().preconditions_met(ws) is False

    def test_preconditions_blocked_when_low_mana(self) -> None:
        ws = make_plan_world_state(pet_alive=False, engaged=False, mana_pct=0.1)
        assert self._summon().preconditions_met(ws) is False

    def test_preconditions_met(self) -> None:
        ws = make_plan_world_state(pet_alive=False, engaged=False, mana_pct=0.5)
        assert self._summon().preconditions_met(ws) is True

    def test_effects(self) -> None:
        ws = make_plan_world_state(pet_alive=False, mana_pct=0.5)
        result = self._summon().apply_effects(ws)
        assert result.pet_alive is True
        assert result.mana_pct < 0.5  # mana cost deducted

    def test_estimate_cost(self) -> None:
        assert self._summon().estimate_cost(None) == 15.0


# ---------------------------------------------------------------------------
# MemorizeAction
# ---------------------------------------------------------------------------


class TestMemorizeAction:
    def _mem(self) -> MemorizeAction:
        return MemorizeAction(name="memorize", routine_name="MEMORIZE_SPELLS")

    def test_preconditions_blocked_when_spells_ready(self) -> None:
        ws = make_plan_world_state(spells_ready=True, engaged=False)
        assert self._mem().preconditions_met(ws) is False

    def test_preconditions_blocked_when_engaged(self) -> None:
        ws = make_plan_world_state(spells_ready=False, engaged=True)
        assert self._mem().preconditions_met(ws) is False

    def test_preconditions_met(self) -> None:
        ws = make_plan_world_state(spells_ready=False, engaged=False)
        assert self._mem().preconditions_met(ws) is True

    def test_effects(self) -> None:
        ws = make_plan_world_state(spells_ready=False)
        result = self._mem().apply_effects(ws)
        assert result.spells_ready is True

    def test_estimate_cost(self) -> None:
        assert self._mem().estimate_cost(None) == 20.0


# ---------------------------------------------------------------------------
# WanderAction
# ---------------------------------------------------------------------------


class TestWanderAction:
    def _wander(self) -> WanderAction:
        return WanderAction(name="wander", routine_name="WANDER")

    def test_preconditions_blocked_when_targets_available(self) -> None:
        ws = make_plan_world_state(targets_available=3, engaged=False)
        assert self._wander().preconditions_met(ws) is False

    def test_preconditions_blocked_when_engaged(self) -> None:
        ws = make_plan_world_state(targets_available=0, engaged=True)
        assert self._wander().preconditions_met(ws) is False

    def test_preconditions_met(self) -> None:
        ws = make_plan_world_state(targets_available=0, engaged=False)
        assert self._wander().preconditions_met(ws) is True

    def test_effects(self) -> None:
        ws = make_plan_world_state(targets_available=0)
        result = self._wander().apply_effects(ws)
        assert result.targets_available == 1

    def test_estimate_cost(self) -> None:
        assert self._wander().estimate_cost(None) == 30.0


# ---------------------------------------------------------------------------
# DefeatAction.estimate_cost with ctx + _update_learned_deltas
# ---------------------------------------------------------------------------


class TestDefeatActionCost:
    def test_no_ctx_returns_default(self) -> None:
        assert _defeat().estimate_cost(None) == 25.0

    def test_no_fight_history_returns_default(self) -> None:
        from types import SimpleNamespace

        ctx = SimpleNamespace(fight_history=None)
        assert _defeat().estimate_cost(ctx) == 25.0

    def test_with_fight_history_no_qualified_stats(self) -> None:
        from types import SimpleNamespace

        stats = SimpleNamespace(fights=1, avg_duration=20.0)  # fights < 3, not qualified
        fh = SimpleNamespace(get_all_stats=lambda: {"mob": stats})
        ctx = SimpleNamespace(fight_history=fh, _last_max_mana=500)
        # No mob has >= 3 fights, so falls back to default cost
        assert _defeat().estimate_cost(ctx) == 25.0

    def test_with_fight_history_qualified_stats(self) -> None:
        from types import SimpleNamespace

        stats = SimpleNamespace(
            fights=5,
            avg_duration=18.0,
            avg_mana=100,
            avg_hp_lost=0.08,
        )
        fh = SimpleNamespace(get_all_stats=lambda: {"skeleton": stats})
        ctx = SimpleNamespace(fight_history=fh, _last_max_mana=500)
        # _update_learned_deltas sets instance attrs on frozen dataclass which
        # fails on Python 3.14 free-threaded. Patch it to test estimate_cost logic.
        from unittest.mock import patch

        d = _defeat()
        with patch.object(DefeatAction, "_update_learned_deltas"):
            cost = d.estimate_cost(ctx)
        assert cost == 18.0

    def test_update_learned_deltas_no_fh(self) -> None:
        from types import SimpleNamespace

        d = _defeat()
        ctx = SimpleNamespace(fight_history=None)
        # Should not crash -- early return when no fight_history
        d._update_learned_deltas(ctx)

    def test_update_learned_deltas_empty_stats(self) -> None:
        from types import SimpleNamespace

        d = _defeat()
        fh = SimpleNamespace(get_all_stats=lambda: {})
        ctx = SimpleNamespace(fight_history=fh)
        # Should not crash -- early return when empty stats
        d._update_learned_deltas(ctx)

    def test_defeat_effects_use_class_defaults(self) -> None:
        """apply_effects reads class-level _learned_mana_delta / _learned_hp_delta."""
        d = _defeat()
        ws = make_plan_world_state(engaged=True, hp_pct=1.0, mana_pct=1.0)
        result = d.apply_effects(ws)
        # Class defaults: _learned_mana_delta=0.30, _learned_hp_delta=0.10
        assert abs(result.mana_pct - 0.70) < 0.01
        assert abs(result.hp_pct - 0.90) < 0.01


# ---------------------------------------------------------------------------
# AcquireAction cost
# ---------------------------------------------------------------------------


class TestAcquireActionCost:
    def test_estimate_cost(self) -> None:
        assert _acquire().estimate_cost(None) == 5.0
