"""Tests for brain.rules.maintenance -- memorize, summon pet, buff conditions.

Condition functions are called directly with GameState + AgentContext.
Spell lookups are injected via the get_spell parameter (no monkeypatching).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from brain.context import AgentContext
from brain.rules.maintenance import (
    _MaintenanceRuleState,
    _score_buff,
    _score_memorize,
    _score_summon_pet,
    _should_buff,
    _should_memorize_spells,
    _should_summon_pet,
    register,
)
from core.features import flags
from core.types import PlanType
from perception.state import GameState
from routines.base import RoutineStatus
from tests.factories import make_game_state


@dataclass
class FakeSpell:
    """Minimal spell object for testing maintenance conditions."""

    name: str = "Shield of Lava"
    mana_cost: int = 100
    gem: int = 1
    spell_id: int = 0


def _fake_provider(spell: object | None) -> Callable[..., object | None]:
    """Return a spell-provider function that always returns *spell*."""

    def _provider(role: object) -> object | None:
        return spell

    return _provider


# ---------------------------------------------------------------------------
# _should_memorize_spells
# ---------------------------------------------------------------------------


class TestShouldMemorizeSpells:
    """MEMORIZE_SPELLS condition: plan.active == NEEDS_MEMORIZE."""

    @pytest.mark.parametrize(
        "plan_active, expected",
        [
            pytest.param(PlanType.NEEDS_MEMORIZE, True, id="needs_memorize"),
            pytest.param(None, False, id="no_plan"),
            pytest.param(PlanType.TRAVEL, False, id="travel_plan"),
            pytest.param("needs_memorize", True, id="string_needs_memorize"),
        ],
    )
    def test_memorize_scenarios(self, plan_active: str | None, expected: bool) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.plan.active = plan_active

        result = _should_memorize_spells(state, ctx)
        assert result is expected


# ---------------------------------------------------------------------------
# _should_summon_pet
# ---------------------------------------------------------------------------


class TestShouldSummonPet:
    """SUMMON_PET condition: spell available, pet dead, not engaged, enough mana."""

    @pytest.mark.parametrize(
        "has_spell, pet_alive, engaged, mana_current, expected",
        [
            pytest.param(True, False, False, 500, True, id="all_good_summon"),
            pytest.param(False, False, False, 500, False, id="no_spell_available"),
            pytest.param(True, True, False, 500, False, id="pet_alive_skip"),
            pytest.param(True, False, True, 500, False, id="engaged_skip"),
            pytest.param(True, False, False, 50, False, id="mana_too_low"),
        ],
    )
    def test_summon_scenarios(
        self,
        has_spell: bool,
        pet_alive: bool,
        engaged: bool,
        mana_current: int,
        expected: bool,
    ) -> None:
        spell = FakeSpell(name="Summon Dead", mana_cost=200) if has_spell else None
        provider = _fake_provider(spell)

        state = make_game_state(mana_current=mana_current, mana_max=500)
        ctx = AgentContext()
        ctx.pet.alive = pet_alive
        ctx.combat.engaged = engaged

        result = _should_summon_pet(state, ctx, provider)
        assert result is expected

    def test_no_spell_configured_returns_false(self) -> None:
        """When no pet spell is loaded (default), summon always returns False."""
        # Don't inject -- use the real get_spell_by_role which returns
        # None when no loadout is configured.
        state = make_game_state(mana_current=500, mana_max=500)
        ctx = AgentContext()
        ctx.pet.alive = False

        result = _should_summon_pet(state, ctx)
        assert result is False


# ---------------------------------------------------------------------------
# _should_buff
# ---------------------------------------------------------------------------


class TestShouldBuff:
    """BUFF condition: flag enabled, spell available, not engaged/sitting, cooldown elapsed."""

    @pytest.fixture(autouse=True)
    def _enable_buff_flag(self) -> None:
        flags.shielding_buff = True

    def test_buff_flag_disabled(self) -> None:
        flags.shielding_buff = False
        state = make_game_state()
        ctx = AgentContext()

        rs = _MaintenanceRuleState()
        assert _should_buff(state, ctx, rs, _fake_provider(FakeSpell())) is False

    def test_no_spell_returns_false(self) -> None:
        state = make_game_state()
        ctx = AgentContext()

        rs = _MaintenanceRuleState()
        assert _should_buff(state, ctx, rs, _fake_provider(None)) is False

    def test_engaged_blocks_buff(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.engaged = True

        rs = _MaintenanceRuleState()
        assert _should_buff(state, ctx, rs, _fake_provider(FakeSpell())) is False

    def test_sitting_blocks_buff(self) -> None:
        state = make_game_state(stand_state=1)  # sitting
        ctx = AgentContext()

        rs = _MaintenanceRuleState()
        assert _should_buff(state, ctx, rs, _fake_provider(FakeSpell())) is False

    def test_low_mana_blocks_buff(self) -> None:
        state = make_game_state(mana_current=50, mana_max=500)
        ctx = AgentContext()

        rs = _MaintenanceRuleState()
        assert _should_buff(state, ctx, rs, _fake_provider(FakeSpell(mana_cost=200))) is False

    def test_recently_cast_blocks_buff(self) -> None:
        """Buff cast within 30s should not trigger recast."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.player.last_buff_time = 0.0

        # last_buff_cast was very recent
        rs = _MaintenanceRuleState(last_buff_cast=time.time() - 10.0)
        assert _should_buff(state, ctx, rs, _fake_provider(FakeSpell())) is False

    def test_buff_triggers_when_all_conditions_met(self) -> None:
        """Buff should trigger when all gates pass and cooldown elapsed."""
        state = make_game_state(mana_current=500, mana_max=500)
        ctx = AgentContext()
        ctx.player.last_buff_time = 0.0

        rs = _MaintenanceRuleState(last_buff_cast=0.0)  # long ago
        # spell_id=0 skips buff_array check
        assert _should_buff(state, ctx, rs, _fake_provider(FakeSpell(spell_id=0))) is True

    def test_pull_in_progress_blocks_buff(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.pull_target_id = 123

        rs = _MaintenanceRuleState()
        assert _should_buff(state, ctx, rs, _fake_provider(FakeSpell())) is False

    def test_npc_targeted_blocks_buff(self) -> None:
        """Buff blocked when an NPC is targeted (need to Escape first)."""
        from tests.factories import make_spawn

        npc = make_spawn(spawn_id=300, name="a_skeleton")
        state = make_game_state(target=npc)
        ctx = AgentContext()

        rs = _MaintenanceRuleState()
        assert _should_buff(state, ctx, rs, _fake_provider(FakeSpell())) is False

    def test_memorize_pending_blocks_buff(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.plan.active = PlanType.NEEDS_MEMORIZE

        rs = _MaintenanceRuleState()
        assert _should_buff(state, ctx, rs, _fake_provider(FakeSpell())) is False

    def test_buff_active_in_buff_array_skips(self) -> None:
        """When buff spell_id is present in the buff array, skip recast."""
        state = make_game_state(
            mana_current=500,
            mana_max=500,
            buffs=((1001, 50),),  # buff active with 50 ticks
        )
        ctx = AgentContext()
        ctx.player.last_buff_time = 0.0

        rs = _MaintenanceRuleState(last_buff_cast=0.0)
        assert _should_buff(state, ctx, rs, _fake_provider(FakeSpell(spell_id=1001))) is False

    def test_buff_not_in_array_logs_once(self) -> None:
        """When spell_id is set but buff not in array, logs info once then returns True."""
        state = make_game_state(mana_current=500, mana_max=500)
        ctx = AgentContext()
        ctx.player.last_buff_time = 0.0

        rs = _MaintenanceRuleState(last_buff_cast=0.0, buff_logged=False)
        result = _should_buff(state, ctx, rs, _fake_provider(FakeSpell(spell_id=999)))
        assert result is True
        assert rs.buff_logged is True  # logged the first time

    def test_buff_not_in_array_already_logged(self) -> None:
        """When buff_logged is True, still returns True but doesn't re-log."""
        state = make_game_state(mana_current=500, mana_max=500)
        ctx = AgentContext()
        ctx.player.last_buff_time = 0.0

        rs = _MaintenanceRuleState(last_buff_cast=0.0, buff_logged=True)
        result = _should_buff(state, ctx, rs, _fake_provider(FakeSpell(spell_id=999)))
        assert result is True
        assert rs.buff_logged is True  # unchanged

    def test_ctx_last_buff_time_recent_blocks(self) -> None:
        """ctx.player.last_buff_time within 30s blocks buff even if rs time is old."""
        state = make_game_state(mana_current=500, mana_max=500)
        ctx = AgentContext()
        ctx.player.last_buff_time = time.time() - 10.0  # 10s ago

        rs = _MaintenanceRuleState(last_buff_cast=0.0)  # long ago
        assert _should_buff(state, ctx, rs, _fake_provider(FakeSpell())) is False


# ---------------------------------------------------------------------------
# _score_memorize
# ---------------------------------------------------------------------------


class TestScoreMemorize:
    """Score function for MEMORIZE_SPELLS rule."""

    def test_needs_memorize_returns_1(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.plan.active = PlanType.NEEDS_MEMORIZE

        assert _score_memorize(state, ctx) == 1.0

    def test_no_plan_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.plan.active = None

        assert _score_memorize(state, ctx) == 0.0

    def test_travel_plan_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL

        assert _score_memorize(state, ctx) == 0.0


# ---------------------------------------------------------------------------
# _score_summon_pet
# ---------------------------------------------------------------------------


class TestScoreSummonPet:
    """Score function for SUMMON_PET rule."""

    def test_all_good_returns_1(self) -> None:
        provider = _fake_provider(FakeSpell(name="Summon Dead", mana_cost=200))
        state = make_game_state(mana_current=500, mana_max=500)
        ctx = AgentContext()
        ctx.pet.alive = False

        assert _score_summon_pet(state, ctx, provider) == 1.0

    def test_no_spell_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()

        assert _score_summon_pet(state, ctx, _fake_provider(None)) == 0.0

    def test_pet_alive_returns_0(self) -> None:
        provider = _fake_provider(FakeSpell(name="Summon Dead", mana_cost=200))
        state = make_game_state(mana_current=500, mana_max=500)
        ctx = AgentContext()
        ctx.pet.alive = True

        assert _score_summon_pet(state, ctx, provider) == 0.0

    def test_engaged_returns_0(self) -> None:
        provider = _fake_provider(FakeSpell(name="Summon Dead", mana_cost=200))
        state = make_game_state(mana_current=500, mana_max=500)
        ctx = AgentContext()
        ctx.pet.alive = False
        ctx.combat.engaged = True

        assert _score_summon_pet(state, ctx, provider) == 0.0

    def test_low_mana_returns_0(self) -> None:
        provider = _fake_provider(FakeSpell(name="Summon Dead", mana_cost=200))
        state = make_game_state(mana_current=50, mana_max=500)
        ctx = AgentContext()
        ctx.pet.alive = False

        assert _score_summon_pet(state, ctx, provider) == 0.0


# ---------------------------------------------------------------------------
# _score_buff
# ---------------------------------------------------------------------------


class TestScoreBuff:
    """Score function for BUFF rule."""

    @pytest.fixture(autouse=True)
    def _enable_buff_flag(self) -> None:
        flags.shielding_buff = True

    def test_flag_disabled_returns_0(self) -> None:
        flags.shielding_buff = False
        state = make_game_state()
        ctx = AgentContext()
        rs = _MaintenanceRuleState()

        assert _score_buff(state, ctx, rs, _fake_provider(FakeSpell())) == 0.0

    def test_no_spell_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        rs = _MaintenanceRuleState()

        assert _score_buff(state, ctx, rs, _fake_provider(None)) == 0.0

    def test_engaged_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.engaged = True
        rs = _MaintenanceRuleState()

        assert _score_buff(state, ctx, rs, _fake_provider(FakeSpell())) == 0.0

    def test_recently_cast_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.player.last_buff_time = 0.0

        rs = _MaintenanceRuleState(last_buff_cast=time.time() - 10.0)
        assert _score_buff(state, ctx, rs, _fake_provider(FakeSpell())) == 0.0

    def test_moderately_old_cast_returns_positive(self) -> None:
        """When both timers are 30-600s ago, score should be > 0 (inverse_linear range)."""
        state = make_game_state()
        ctx = AgentContext()
        now = time.time()
        ctx.player.last_buff_time = now - 60.0  # 60s ago

        rs = _MaintenanceRuleState(last_buff_cast=now - 60.0)
        score = _score_buff(state, ctx, rs, _fake_provider(FakeSpell()))
        assert score > 0.0

    def test_very_old_cast_returns_0(self) -> None:
        """When both timers are > 600s ago, inverse_linear returns 0."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.player.last_buff_time = 0.0  # epoch -> huge elapsed

        rs = _MaintenanceRuleState(last_buff_cast=0.0)
        score = _score_buff(state, ctx, rs, _fake_provider(FakeSpell()))
        assert score == 0.0


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


class TestRegister:
    """Integration test: register() wires rules to the brain."""

    def test_register_adds_three_rules(self) -> None:
        """register() should add MEMORIZE_SPELLS, SUMMON_PET, and BUFF rules."""
        brain = MagicMock()
        ctx = AgentContext()

        def read_state_fn() -> GameState:
            return make_game_state()

        register(brain, ctx, read_state_fn)

        assert brain.add_rule.call_count == 3
        rule_names = [call.args[0] for call in brain.add_rule.call_args_list]
        assert "MEMORIZE_SPELLS" in rule_names
        assert "SUMMON_PET" in rule_names
        assert "BUFF" in rule_names

    def test_buff_tick_wrapper_records_success(self) -> None:
        """Buff tick wrapper updates last_buff_cast and ctx on SUCCESS."""
        from routines.buff import BuffRoutine

        class _SuccessBuff(BuffRoutine):
            def tick(self, state: GameState) -> RoutineStatus:
                return RoutineStatus.SUCCESS

        brain = MagicMock()
        ctx = AgentContext()
        ctx.player.last_buff_time = 0.0

        def read_state_fn() -> GameState:
            return make_game_state()

        register(brain, ctx, read_state_fn, buff_routine=_SuccessBuff(ctx=ctx, read_state_fn=read_state_fn))

        # Find the BUFF rule's routine (3rd add_rule call)
        buff_call = brain.add_rule.call_args_list[2]
        buff_routine = buff_call.args[2]

        before = time.time()
        state = make_game_state()
        result = buff_routine.tick(state)
        after = time.time()

        assert result == RoutineStatus.SUCCESS
        assert ctx.player.last_buff_time >= before
        assert ctx.player.last_buff_time <= after

    def test_buff_tick_wrapper_no_update_on_running(self) -> None:
        """Buff tick wrapper does NOT update timestamps on RUNNING."""
        from routines.buff import BuffRoutine

        class _RunningBuff(BuffRoutine):
            def tick(self, state: GameState) -> RoutineStatus:
                return RoutineStatus.RUNNING

        brain = MagicMock()
        ctx = AgentContext()
        ctx.player.last_buff_time = 0.0

        def read_state_fn() -> GameState:
            return make_game_state()

        register(brain, ctx, read_state_fn, buff_routine=_RunningBuff(ctx=ctx, read_state_fn=read_state_fn))

        buff_call = brain.add_rule.call_args_list[2]
        buff_routine = buff_call.args[2]

        state = make_game_state()
        result = buff_routine.tick(state)

        assert result == RoutineStatus.RUNNING
        assert ctx.player.last_buff_time == 0.0  # unchanged
