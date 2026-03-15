"""Tests for brain.rules.combat -- combat, acquire, pull, engage_add conditions.

Condition functions are called directly with GameState + AgentContext.
Feature flags are set per-fixture to isolate tests from global state.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from brain.context import AgentContext
from brain.rules.combat import (
    _acquire_not_ready,
    _acquire_suppressed,
    _CombatRuleState,
    _has_close_pull_target,
    _has_pending_loot,
    _score_acquire,
    _score_engage_add,
    _score_in_combat,
    _score_pull,
    _should_acquire,
    _should_combat,
    _should_engage_add,
    _should_pull,
)
from core.features import flags
from core.types import LootMode, PlanType
from tests.factories import make_game_state, make_spawn


@pytest.fixture(autouse=True)
def _enable_flags() -> None:
    """Enable combat-relevant flags for all tests in this module."""
    flags.pull = True
    flags.flee = True


# ---------------------------------------------------------------------------
# _should_combat
# ---------------------------------------------------------------------------


class TestShouldCombat:
    """IN_COMBAT condition: engaged with live target, or auto-engage candidate."""

    @pytest.mark.parametrize(
        "engaged, target_hp, has_target, auto_engage, expected",
        [
            pytest.param(True, 100, True, None, True, id="engaged_live_target"),
            pytest.param(True, 0, True, None, False, id="engaged_target_dead"),
            pytest.param(True, 0, False, None, False, id="engaged_no_target_no_pet"),
            pytest.param(False, 0, False, None, False, id="not_engaged_no_candidate"),
            pytest.param(
                False,
                0,
                False,
                (200, "a_skeleton", 10.0, 10.0, 8),
                True,
                id="auto_engage_candidate",
            ),
        ],
    )
    def test_basic_scenarios(
        self,
        engaged: bool,
        target_hp: int,
        has_target: bool,
        auto_engage: tuple | None,
        expected: bool,
    ) -> None:
        target = make_spawn(hp_current=target_hp, hp_max=100) if has_target else None
        state = make_game_state(target=target)
        ctx = AgentContext()
        ctx.combat.engaged = engaged
        ctx.combat.auto_engage_candidate = auto_engage

        result = _should_combat(state, ctx)
        assert result is expected

    def test_engaged_target_gone_pet_fighting_nearby(self) -> None:
        """Engaged, target gone, but pet fighting a damaged NPC nearby."""
        damaged_npc = make_spawn(
            spawn_id=300,
            name="a_bat",
            x=10.0,
            y=10.0,
            hp_current=50,
            hp_max=100,
        )
        state = make_game_state(target=None, spawns=(damaged_npc,))
        ctx = AgentContext()
        ctx.combat.engaged = True
        ctx.pet.alive = True

        assert _should_combat(state, ctx) is True

    def test_engaged_target_gone_no_pet(self) -> None:
        """Engaged, target gone, no pet -- no combat."""
        state = make_game_state(target=None)
        ctx = AgentContext()
        ctx.combat.engaged = True
        ctx.pet.alive = False

        assert _should_combat(state, ctx) is False

    def test_engaged_damaged_npc_too_far(self) -> None:
        """Engaged, target gone, pet alive, but nearby NPC is >100u away."""
        far_npc = make_spawn(
            spawn_id=300,
            name="a_bat",
            x=200.0,
            y=200.0,
            hp_current=50,
            hp_max=100,
        )
        state = make_game_state(target=None, spawns=(far_npc,))
        ctx = AgentContext()
        ctx.combat.engaged = True
        ctx.pet.alive = True

        assert _should_combat(state, ctx) is False


# ---------------------------------------------------------------------------
# _should_acquire
# ---------------------------------------------------------------------------


class TestShouldAcquire:
    """ACQUIRE condition: all resource/state gates must pass."""

    @pytest.mark.parametrize(
        "pet_alive, engaged, hp_pct, mana_pct, pull_target_id, expected",
        [
            pytest.param(True, False, 0.90, 0.80, None, True, id="all_good_acquire"),
            pytest.param(False, False, 0.90, 0.80, None, False, id="no_pet_blocked"),
            pytest.param(True, True, 0.90, 0.80, None, False, id="engaged_blocked"),
            pytest.param(True, False, 0.50, 0.80, None, False, id="hp_low_blocked"),
            pytest.param(True, False, 0.90, 0.80, 123, False, id="pull_in_progress_blocked"),
        ],
    )
    def test_gate_scenarios(
        self,
        pet_alive: bool,
        engaged: bool,
        hp_pct: float,
        mana_pct: float,
        pull_target_id: int | None,
        expected: bool,
    ) -> None:
        hp = int(hp_pct * 1000)
        mana = int(mana_pct * 500)
        state = make_game_state(hp_current=hp, hp_max=1000, mana_current=mana, mana_max=500)
        ctx = AgentContext()
        ctx.pet.alive = pet_alive
        ctx.pet.spawn_id = 50 if pet_alive else None
        ctx.combat.engaged = engaged
        ctx.combat.pull_target_id = pull_target_id

        result = _should_acquire(state, ctx)
        assert result is expected

    def test_pull_flag_disabled_blocks(self) -> None:
        flags.pull = False
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.spawn_id = 50

        assert _should_acquire(state, ctx) is False

    def test_nearby_player_blocks_acquire(self) -> None:
        """A nearby player within 250u triggers AFK safety suppression."""
        nearby_player = make_spawn(
            spawn_id=500,
            name="OtherPlayer",
            spawn_type=0,  # Player
            x=10.0,
            y=10.0,
        )
        state = make_game_state(
            hp_current=900,
            hp_max=1000,
            mana_current=400,
            mana_max=500,
            spawns=(nearby_player,),
        )
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.spawn_id = 50
        # ctx.world is None, so it falls through to ctx.nearby_player_count()

        assert _should_acquire(state, ctx) is False

    def test_mana_low_but_pet_solo_ok(self) -> None:
        """Low mana (> 20%) with healthy pet and HP >= 85% is allowed (pet solo)."""
        # Pet spawn in the spawn list so _should_acquire can check pet HP
        pet_spawn = make_spawn(
            spawn_id=50,
            name="Kabaler`s_pet",
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
        )
        state = make_game_state(
            hp_current=900,
            hp_max=1000,
            mana_current=150,  # 30% mana
            mana_max=500,
            spawns=(pet_spawn,),
        )
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.spawn_id = 50

        # mana_pct=0.30 >= 0.20, hp_pct=0.90 >= 0.85, pet_hp=1.0 >= 0.70 -> mana_ok=True
        assert _should_acquire(state, ctx) is True

    def test_mana_very_low_blocks(self) -> None:
        """Mana below 20% blocks acquire even with healthy pet."""
        pet_spawn = make_spawn(
            spawn_id=50,
            name="Kabaler`s_pet",
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
        )
        state = make_game_state(
            hp_current=900,
            hp_max=1000,
            mana_current=50,  # 10% mana
            mana_max=500,
            spawns=(pet_spawn,),
        )
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.spawn_id = 50

        assert _should_acquire(state, ctx) is False

    def test_active_plan_blocks_acquire(self) -> None:
        """Non-travel active plan blocks acquire."""
        from core.types import PlanType

        state = make_game_state(hp_current=900, hp_max=1000, mana_current=400, mana_max=500)
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.spawn_id = 50
        ctx.plan.active = PlanType.NEEDS_MEMORIZE

        assert _should_acquire(state, ctx) is False

    def test_pet_hp_low_blocks_acquire(self) -> None:
        """Pet HP below 50% blocks acquire (let pet rest)."""
        pet_spawn = make_spawn(
            spawn_id=50,
            name="Kabaler`s_pet",
            x=5.0,
            y=5.0,
            hp_current=30,  # 30%
            hp_max=100,
        )
        state = make_game_state(
            hp_current=900,
            hp_max=1000,
            mana_current=400,
            mana_max=500,
            spawns=(pet_spawn,),
        )
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.spawn_id = 50

        assert _should_acquire(state, ctx) is False


# ---------------------------------------------------------------------------
# _should_pull
# ---------------------------------------------------------------------------


class TestShouldPull:
    """PULL condition: pull_target_id set and not engaged."""

    @pytest.mark.parametrize(
        "pull_target_id, engaged, expected",
        [
            pytest.param(123, False, True, id="target_set_not_engaged"),
            pytest.param(None, False, False, id="no_target"),
            pytest.param(123, True, False, id="engaged_blocks_pull"),
        ],
    )
    def test_basic_scenarios(
        self,
        pull_target_id: int | None,
        engaged: bool,
        expected: bool,
    ) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.pull_target_id = pull_target_id
        ctx.combat.engaged = engaged

        result = _should_pull(state, ctx)
        assert result is expected


# ---------------------------------------------------------------------------
# _should_engage_add
# ---------------------------------------------------------------------------


class TestShouldEngageAdd:
    """ENGAGE_ADD condition: pet has add, or NPC targeting player/pet."""

    def test_pet_has_add(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.has_add = True
        ctx.combat.engaged = False

        rs = _CombatRuleState()
        assert _should_engage_add(state, ctx, rs) is True

    def test_pet_not_alive_returns_false(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = False
        ctx.combat.engaged = False

        rs = _CombatRuleState()
        assert _should_engage_add(state, ctx, rs) is False

    def test_engaged_resets_and_returns_false(self) -> None:
        """When engaged, engage_add is handled by IN_COMBAT; returns False."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.has_add = True
        ctx.combat.engaged = True

        rs = _CombatRuleState(add_first_seen=5.0)
        assert _should_engage_add(state, ctx, rs) is False
        assert rs.add_first_seen == 0.0

    def test_npc_targeting_player_triggers_add(self) -> None:
        """An NPC targeting the player within 100u triggers engage_add."""
        attacker = make_spawn(
            spawn_id=300,
            name="a_skeleton",
            x=10.0,
            y=10.0,
            hp_current=100,
            hp_max=100,
            target_name="TestPlayer",
        )
        state = make_game_state(spawns=(attacker,))
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.name = "Kabaler`s_pet"
        ctx.combat.engaged = False

        rs = _CombatRuleState()
        result = _should_engage_add(state, ctx, rs)
        assert result is True
        assert ctx.pet.has_add is True

    def test_npc_targeting_pet_triggers_add(self) -> None:
        """An NPC targeting the pet within 100u triggers engage_add."""
        attacker = make_spawn(
            spawn_id=300,
            name="a_bat",
            x=10.0,
            y=10.0,
            hp_current=100,
            hp_max=100,
            target_name="Kabaler`s_pet",
        )
        state = make_game_state(spawns=(attacker,))
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.name = "Kabaler`s_pet"
        ctx.combat.engaged = False

        rs = _CombatRuleState()
        result = _should_engage_add(state, ctx, rs)
        assert result is True

    def test_npc_too_far_no_add(self) -> None:
        """NPC targeting player but >100u away does not trigger add."""
        far_attacker = make_spawn(
            spawn_id=300,
            name="a_skeleton",
            x=200.0,
            y=200.0,
            hp_current=100,
            hp_max=100,
            target_name="TestPlayer",
        )
        state = make_game_state(spawns=(far_attacker,))
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.name = "Kabaler`s_pet"
        ctx.combat.engaged = False

        rs = _CombatRuleState()
        result = _should_engage_add(state, ctx, rs)
        assert result is False

    def test_dead_npc_targeting_player_ignored(self) -> None:
        """A dead NPC targeting the player should not trigger add."""
        dead_attacker = make_spawn(
            spawn_id=300,
            name="a_skeleton",
            x=10.0,
            y=10.0,
            hp_current=0,
            hp_max=100,
            target_name="TestPlayer",
        )
        state = make_game_state(spawns=(dead_attacker,))
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.name = "Kabaler`s_pet"
        ctx.combat.engaged = False

        rs = _CombatRuleState()
        result = _should_engage_add(state, ctx, rs)
        assert result is False

    def test_pull_target_excluded_from_add(self) -> None:
        """The current pull target should not be considered an add."""
        pull_npc = make_spawn(
            spawn_id=300,
            name="a_skeleton",
            x=10.0,
            y=10.0,
            hp_current=100,
            hp_max=100,
            target_name="TestPlayer",
        )
        state = make_game_state(spawns=(pull_npc,))
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.name = "Kabaler`s_pet"
        ctx.combat.engaged = False
        ctx.combat.pull_target_id = 300  # same as the NPC

        rs = _CombatRuleState()
        result = _should_engage_add(state, ctx, rs)
        assert result is False


# ---------------------------------------------------------------------------
# _score_in_combat
# ---------------------------------------------------------------------------


class TestScoreInCombat:
    """Score function for IN_COMBAT rule."""

    def test_engaged_live_target_returns_1(self) -> None:
        target = make_spawn(hp_current=80, hp_max=100)
        state = make_game_state(target=target)
        ctx = AgentContext()
        ctx.combat.engaged = True

        assert _score_in_combat(state, ctx) == 1.0

    def test_engaged_dead_target_pet_fighting_nearby_returns_08(self) -> None:
        damaged_npc = make_spawn(spawn_id=300, name="a_bat", x=10.0, y=10.0, hp_current=50, hp_max=100)
        state = make_game_state(target=None, spawns=(damaged_npc,))
        ctx = AgentContext()
        ctx.combat.engaged = True
        ctx.pet.alive = True

        assert _score_in_combat(state, ctx) == 0.8

    def test_engaged_no_target_no_pet_returns_0(self) -> None:
        state = make_game_state(target=None)
        ctx = AgentContext()
        ctx.combat.engaged = True
        ctx.pet.alive = False

        assert _score_in_combat(state, ctx) == 0.0

    def test_engaged_pet_alive_but_npc_too_far_returns_0(self) -> None:
        far_npc = make_spawn(spawn_id=300, name="a_bat", x=200.0, y=200.0, hp_current=50, hp_max=100)
        state = make_game_state(target=None, spawns=(far_npc,))
        ctx = AgentContext()
        ctx.combat.engaged = True
        ctx.pet.alive = True

        assert _score_in_combat(state, ctx) == 0.0

    def test_engaged_pet_alive_npc_full_hp_returns_0(self) -> None:
        """NPC at full HP (not damaged) should not count as pet fighting."""
        full_npc = make_spawn(spawn_id=300, name="a_bat", x=10.0, y=10.0, hp_current=100, hp_max=100)
        state = make_game_state(target=None, spawns=(full_npc,))
        ctx = AgentContext()
        ctx.combat.engaged = True
        ctx.pet.alive = True

        assert _score_in_combat(state, ctx) == 0.0

    def test_not_engaged_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.engaged = False

        assert _score_in_combat(state, ctx) == 0.0


# ---------------------------------------------------------------------------
# _score_engage_add
# ---------------------------------------------------------------------------


class TestScoreEngageAdd:
    """Score function for ENGAGE_ADD rule."""

    def test_engaged_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.engaged = True
        ctx.pet.alive = True
        ctx.pet.has_add = True

        assert _score_engage_add(state, ctx) == 0.0

    def test_pet_not_alive_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.engaged = False
        ctx.pet.alive = False

        assert _score_engage_add(state, ctx) == 0.0

    def test_pet_has_add_returns_1(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.engaged = False
        ctx.pet.alive = True
        ctx.pet.has_add = True

        assert _score_engage_add(state, ctx) == 1.0

    def test_no_add_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.engaged = False
        ctx.pet.alive = True
        ctx.pet.has_add = False

        assert _score_engage_add(state, ctx) == 0.0


# ---------------------------------------------------------------------------
# _score_pull
# ---------------------------------------------------------------------------


class TestScorePull:
    """Score function for PULL rule."""

    def test_pull_target_set_not_engaged_returns_1(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.pull_target_id = 123
        ctx.combat.engaged = False

        assert _score_pull(state, ctx) == 1.0

    def test_no_pull_target_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.pull_target_id = None

        assert _score_pull(state, ctx) == 0.0

    def test_engaged_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.pull_target_id = 123
        ctx.combat.engaged = True

        assert _score_pull(state, ctx) == 0.0


# ---------------------------------------------------------------------------
# _score_acquire
# ---------------------------------------------------------------------------


class TestScoreAcquire:
    """Score function for ACQUIRE rule."""

    def test_pull_disabled_returns_0(self) -> None:
        flags.pull = False
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True

        assert _score_acquire(state, ctx) == 0.0

    def test_in_active_combat_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.engaged = True
        ctx.pet.alive = True

        assert _score_acquire(state, ctx) == 0.0

    def test_no_pet_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = False

        assert _score_acquire(state, ctx) == 0.0

    def test_healthy_state_returns_positive(self) -> None:
        """Full HP, mana, pet alive -> positive score."""
        state = make_game_state(hp_current=900, hp_max=1000, mana_current=400, mana_max=500)
        ctx = AgentContext()
        ctx.pet.alive = True

        score = _score_acquire(state, ctx, _get_spell=lambda role: None)
        assert score > 0.0

    def test_low_hp_reduces_score(self) -> None:
        state = make_game_state(hp_current=550, hp_max=1000, mana_current=400, mana_max=500)
        ctx = AgentContext()
        ctx.pet.alive = True

        score = _score_acquire(state, ctx, _get_spell=lambda role: None)
        # hp_pct=0.55 -> linear(0.55, 0.50, 0.90) is small but > 0
        assert 0.0 < score < 1.0

    def test_with_dot_spell_mana_factor(self) -> None:
        """When a DOT spell exists, mana_factor is computed from mana_current."""
        from eq.loadout import SpellRole

        fake_dot = SimpleNamespace(mana_cost=100)
        state = make_game_state(hp_current=900, hp_max=1000, mana_current=200, mana_max=500)
        ctx = AgentContext()
        ctx.pet.alive = True

        score = _score_acquire(
            state,
            ctx,
            _get_spell=lambda role: fake_dot if role == SpellRole.DOT else None,
        )
        assert score > 0.0


# ---------------------------------------------------------------------------
# _has_close_pull_target
# ---------------------------------------------------------------------------


class TestHasClosePullTarget:
    """Checks for nearby valid pull targets during TRAVEL plan."""

    def test_close_valid_npc_returns_true(self) -> None:
        npc = make_spawn(
            spawn_id=300,
            name="a_moss_snake",
            level=8,
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
            owner_id=0,
        )
        state = make_game_state(level=10, spawns=(npc,))
        ctx = AgentContext()

        assert _has_close_pull_target(state, ctx) is True

    def test_npc_too_far_returns_false(self) -> None:
        npc = make_spawn(
            spawn_id=300,
            name="a_moss_snake",
            level=8,
            x=100.0,
            y=100.0,
            hp_current=100,
            hp_max=100,
            owner_id=0,
        )
        state = make_game_state(level=10, spawns=(npc,))
        ctx = AgentContext()

        assert _has_close_pull_target(state, ctx) is False

    def test_dead_npc_ignored(self) -> None:
        npc = make_spawn(
            spawn_id=300,
            name="a_moss_snake",
            level=8,
            x=5.0,
            y=5.0,
            hp_current=0,
            hp_max=100,
            owner_id=0,
        )
        state = make_game_state(level=10, spawns=(npc,))
        ctx = AgentContext()

        assert _has_close_pull_target(state, ctx) is False

    def test_pet_owned_npc_ignored(self) -> None:
        """NPC with owner_id != 0 should be skipped."""
        npc = make_spawn(
            spawn_id=300,
            name="pet_summon",
            level=8,
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
            owner_id=1,
        )
        state = make_game_state(level=10, spawns=(npc,))
        ctx = AgentContext()

        assert _has_close_pull_target(state, ctx) is False

    def test_wrong_con_color_returns_false(self) -> None:
        """NPC that cons RED should not trigger pull target."""
        npc = make_spawn(
            spawn_id=300,
            name="a_dragon",
            level=60,
            x=5.0,
            y=5.0,
            hp_current=5000,
            hp_max=5000,
            owner_id=0,
        )
        state = make_game_state(level=10, spawns=(npc,))
        ctx = AgentContext()

        assert _has_close_pull_target(state, ctx) is False

    def test_custom_target_cons_respected(self) -> None:
        """Zone target_cons overrides default set."""
        from perception.combat_eval import Con

        npc = make_spawn(
            spawn_id=300,
            name="a_moss_snake",
            level=8,
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
            owner_id=0,
        )
        state = make_game_state(level=10, spawns=(npc,))
        ctx = AgentContext()
        # Set target_cons to only allow RED -- blue/white NPC should be rejected
        ctx.zone.target_cons = {Con.RED}

        assert _has_close_pull_target(state, ctx) is False


# ---------------------------------------------------------------------------
# _has_pending_loot
# ---------------------------------------------------------------------------


class TestHasPendingLoot:
    """Checks for recent unlooted defeats that should delay acquire."""

    def test_no_defeats_returns_false(self) -> None:
        ctx = AgentContext()
        assert _has_pending_loot(ctx) is False

    def test_recent_unlooted_defeat_returns_true(self) -> None:
        from brain.state.kill_tracker import DefeatInfo

        ctx = AgentContext()
        ctx.defeat_tracker.defeat_history.append(
            DefeatInfo(name="a_skeleton", x=10.0, y=10.0, time=time.time(), looted=False)
        )
        assert _has_pending_loot(ctx) is True

    def test_old_defeat_returns_false(self) -> None:
        from brain.state.kill_tracker import DefeatInfo

        ctx = AgentContext()
        ctx.defeat_tracker.defeat_history.append(
            DefeatInfo(name="a_skeleton", x=10.0, y=10.0, time=time.time() - 10, looted=False)
        )
        assert _has_pending_loot(ctx) is False

    def test_looted_defeat_returns_false(self) -> None:
        from brain.state.kill_tracker import DefeatInfo

        ctx = AgentContext()
        ctx.defeat_tracker.defeat_history.append(
            DefeatInfo(name="a_skeleton", x=10.0, y=10.0, time=time.time(), looted=True)
        )
        assert _has_pending_loot(ctx) is False

    def test_smart_loot_mode_filters_by_resource_targets(self) -> None:
        """SMART loot mode only waits for resource target defeats."""
        from brain.state.kill_tracker import DefeatInfo

        flags.looting = True
        old_loot_mode = flags.loot_mode
        flags.loot_mode = LootMode.SMART

        ctx = AgentContext()
        ctx.loot.resource_targets = ["a_skeleton"]
        ctx.defeat_tracker.defeat_history.append(
            DefeatInfo(name="a_bat", x=10.0, y=10.0, time=time.time(), looted=False)
        )
        # "a_bat" is not in resource_targets, so should not count
        result = _has_pending_loot(ctx)
        flags.loot_mode = old_loot_mode
        assert result is False

    def test_smart_loot_mode_matches_resource_target(self) -> None:
        """SMART loot mode returns True when defeat matches resource target."""
        from brain.state.kill_tracker import DefeatInfo

        flags.looting = True
        old_loot_mode = flags.loot_mode
        flags.loot_mode = LootMode.SMART

        ctx = AgentContext()
        ctx.loot.resource_targets = ["a_skeleton"]
        ctx.defeat_tracker.defeat_history.append(
            DefeatInfo(name="a_skeleton", x=10.0, y=10.0, time=time.time(), looted=False)
        )
        result = _has_pending_loot(ctx)
        flags.loot_mode = old_loot_mode
        assert result is True


# ---------------------------------------------------------------------------
# _acquire_suppressed  (additional branches)
# ---------------------------------------------------------------------------


class TestAcquireSuppressed:
    """Tests for _acquire_suppressed branches not covered by TestShouldAcquire."""

    def test_looting_with_pending_loot_blocks(self) -> None:
        """Looting flag + pending loot suppresses acquire."""
        from brain.state.kill_tracker import DefeatInfo

        flags.looting = True
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.defeat_tracker.defeat_history.append(
            DefeatInfo(name="a_skeleton", x=10.0, y=10.0, time=time.time(), looted=False)
        )
        assert _acquire_suppressed(ctx, state) is True

    def test_travel_plan_with_close_target_allows(self) -> None:
        """TRAVEL plan with a close valid pull target should NOT suppress."""
        npc = make_spawn(
            spawn_id=300,
            name="a_moss_snake",
            level=8,
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
            owner_id=0,
        )
        state = make_game_state(level=10, spawns=(npc,))
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.plan.active = PlanType.TRAVEL
        assert _acquire_suppressed(ctx, state) is False

    def test_travel_plan_no_close_target_blocks(self) -> None:
        """TRAVEL plan with no close target suppresses acquire."""
        state = make_game_state(level=10)
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.plan.active = PlanType.TRAVEL
        assert _acquire_suppressed(ctx, state) is True

    def test_world_model_nearby_player_blocks(self) -> None:
        """When ctx.world is set, use world.nearby_player_count."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.world = MagicMock()
        ctx.world.nearby_player_count.return_value = 1

        assert _acquire_suppressed(ctx, state) is True

    def test_world_model_no_nearby_player_allows(self) -> None:
        """When ctx.world is set and no player nearby, not suppressed."""
        state = make_game_state(hp_current=900, hp_max=1000, mana_current=400, mana_max=500)
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.world = MagicMock()
        ctx.world.nearby_player_count.return_value = 0

        assert _acquire_suppressed(ctx, state) is False


# ---------------------------------------------------------------------------
# _acquire_not_ready  (additional branches)
# ---------------------------------------------------------------------------


class TestAcquireNotReady:
    """Tests for _acquire_not_ready branches."""

    def test_world_damaged_npcs_near_blocks(self) -> None:
        """Pet fighting nearby (damaged NPCs via world model)."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.world = MagicMock()
        ctx.world.damaged_npcs_near.return_value = [MagicMock()]

        assert _acquire_not_ready(ctx, state) is True

    def test_pet_too_far_blocks(self) -> None:
        """Pet spawn found but > 200u away blocks acquire."""
        pet_spawn = make_spawn(spawn_id=50, name="pet", x=300.0, y=300.0)
        state = make_game_state(spawns=(pet_spawn,))
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.spawn_id = 50

        assert _acquire_not_ready(ctx, state) is True

    def test_pet_not_alive_second_check(self) -> None:
        """If pet dies between first and second check, blocks."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = False

        assert _acquire_not_ready(ctx, state) is True

    def test_player_hp_low_blocks(self) -> None:
        """Player HP < 70% blocks acquire."""
        state = make_game_state(hp_current=600, hp_max=1000)
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.spawn_id = 50

        assert _acquire_not_ready(ctx, state) is True

    def test_pull_target_already_set_blocks(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.spawn_id = 50
        ctx.combat.pull_target_id = 123

        assert _acquire_not_ready(ctx, state) is True

    def test_world_threats_within_blocks(self) -> None:
        """Threats within 80u via world model blocks acquire."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.spawn_id = 50
        ctx.world = MagicMock()
        ctx.world.damaged_npcs_near.return_value = []
        ctx.world.threats_within.return_value = [MagicMock()]

        assert _acquire_not_ready(ctx, state) is True

    def test_all_clear_returns_false(self) -> None:
        """All checks pass -> not blocked."""
        pet_spawn = make_spawn(spawn_id=50, name="pet", x=5.0, y=5.0, hp_current=100, hp_max=100)
        state = make_game_state(hp_current=900, hp_max=1000, spawns=(pet_spawn,))
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.spawn_id = 50

        assert _acquire_not_ready(ctx, state) is False


# ---------------------------------------------------------------------------
# _should_acquire  (danger-aware gating)
# ---------------------------------------------------------------------------


class TestAcquireDangerGating:
    """Danger-aware gating in _should_acquire via fight_history."""

    def _make_ready_ctx(self) -> AgentContext:
        """Create a ctx that passes all suppression/readiness checks."""
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.spawn_id = 50
        return ctx

    def _make_ready_state(self, hp_pct: float = 0.90, mana_pct: float = 0.80) -> object:
        hp = int(hp_pct * 1000)
        mana = int(mana_pct * 500)
        pet_spawn = make_spawn(spawn_id=50, name="pet", x=5.0, y=5.0, hp_current=100, hp_max=100)
        return make_game_state(
            hp_current=hp, hp_max=1000, mana_current=mana, mana_max=500, spawns=(pet_spawn,)
        )

    def test_high_danger_low_hp_blocks(self) -> None:
        """danger > 0.5 with hp < 80% blocks acquire."""
        npc = make_spawn(spawn_id=300, name="a_skeleton", x=10.0, y=10.0, hp_current=100, hp_max=100)
        pet_spawn = make_spawn(spawn_id=50, name="pet", x=5.0, y=5.0, hp_current=100, hp_max=100)
        state = make_game_state(
            hp_current=750,
            hp_max=1000,
            mana_current=400,
            mana_max=500,
            spawns=(npc, pet_spawn),
        )
        ctx = self._make_ready_ctx()

        fh = MagicMock()
        fh.has_learned.return_value = True
        fh.learned_danger.return_value = 0.6
        ctx.fight_history = fh

        assert _should_acquire(state, ctx) is False

    def test_very_high_danger_low_mana_blocks(self) -> None:
        """danger > 0.7 with mana < 40% blocks acquire."""
        npc = make_spawn(spawn_id=300, name="a_skeleton", x=10.0, y=10.0, hp_current=100, hp_max=100)
        pet_spawn = make_spawn(spawn_id=50, name="pet", x=5.0, y=5.0, hp_current=100, hp_max=100)
        state = make_game_state(
            hp_current=900,
            hp_max=1000,
            mana_current=150,
            mana_max=500,
            spawns=(npc, pet_spawn),
        )
        ctx = self._make_ready_ctx()

        fh = MagicMock()
        fh.has_learned.return_value = True
        fh.learned_danger.return_value = 0.8
        ctx.fight_history = fh

        # mana_pct=0.30 >= 0.20, hp_pct=0.90 >= 0.85, pet_hp=1.0 >= 0.70 -> mana_ok via pet solo
        # But danger > 0.7 and mana_pct < 0.40 blocks
        assert _should_acquire(state, ctx) is False

    def test_danger_below_threshold_allows(self) -> None:
        """danger <= 0.5 does not block even with lower HP."""
        npc = make_spawn(spawn_id=300, name="a_skeleton", x=10.0, y=10.0, hp_current=100, hp_max=100)
        pet_spawn = make_spawn(spawn_id=50, name="pet", x=5.0, y=5.0, hp_current=100, hp_max=100)
        state = make_game_state(
            hp_current=900,
            hp_max=1000,
            mana_current=400,
            mana_max=500,
            spawns=(npc, pet_spawn),
        )
        ctx = self._make_ready_ctx()

        fh = MagicMock()
        fh.has_learned.return_value = True
        fh.learned_danger.return_value = 0.3
        ctx.fight_history = fh

        assert _should_acquire(state, ctx) is True

    def test_not_learned_mob_no_danger_gating(self) -> None:
        """Mobs without learned data don't trigger danger gating."""
        npc = make_spawn(spawn_id=300, name="a_skeleton", x=10.0, y=10.0, hp_current=100, hp_max=100)
        pet_spawn = make_spawn(spawn_id=50, name="pet", x=5.0, y=5.0, hp_current=100, hp_max=100)
        state = make_game_state(
            hp_current=900,
            hp_max=1000,
            mana_current=400,
            mana_max=500,
            spawns=(npc, pet_spawn),
        )
        ctx = self._make_ready_ctx()

        fh = MagicMock()
        fh.has_learned.return_value = False
        ctx.fight_history = fh

        assert _should_acquire(state, ctx) is True

    def test_learned_danger_none_no_block(self) -> None:
        """learned_danger returns None -> no block."""
        npc = make_spawn(spawn_id=300, name="a_skeleton", x=10.0, y=10.0, hp_current=100, hp_max=100)
        pet_spawn = make_spawn(spawn_id=50, name="pet", x=5.0, y=5.0, hp_current=100, hp_max=100)
        state = make_game_state(
            hp_current=900,
            hp_max=1000,
            mana_current=400,
            mana_max=500,
            spawns=(npc, pet_spawn),
        )
        ctx = self._make_ready_ctx()

        fh = MagicMock()
        fh.has_learned.return_value = True
        fh.learned_danger.return_value = None
        ctx.fight_history = fh

        assert _should_acquire(state, ctx) is True

    def test_far_npc_not_danger_checked(self) -> None:
        """NPCs > 200u away are skipped in danger check."""
        npc = make_spawn(spawn_id=300, name="a_skeleton", x=500.0, y=500.0, hp_current=100, hp_max=100)
        pet_spawn = make_spawn(spawn_id=50, name="pet", x=5.0, y=5.0, hp_current=100, hp_max=100)
        state = make_game_state(
            hp_current=900,
            hp_max=1000,
            mana_current=400,
            mana_max=500,
            spawns=(npc, pet_spawn),
        )
        ctx = self._make_ready_ctx()

        fh = MagicMock()
        fh.has_learned.return_value = True
        fh.learned_danger.return_value = 0.9
        ctx.fight_history = fh

        # NPC is too far to be considered
        assert _should_acquire(state, ctx) is True


# ---------------------------------------------------------------------------
# _should_engage_add  (debounce logic)
# ---------------------------------------------------------------------------


class TestEngageAddDebounce:
    """Debounce logic for damaged NPC detection in _should_engage_add."""

    def test_damaged_npc_near_pet_starts_debounce(self) -> None:
        """First detection of damaged NPC starts debounce timer, returns False."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.has_add = False
        ctx.combat.engaged = False
        ctx.combat.pull_target_id = None

        world = MagicMock()
        world.pet_spawn = None
        world.damaged_npcs_near.return_value = [MagicMock()]
        ctx.world = world

        rs = _CombatRuleState()
        result = _should_engage_add(state, ctx, rs)
        assert result is False
        assert rs.add_first_seen > 0.0  # timer started

    def test_damaged_npc_debounce_expired_fires(self) -> None:
        """After 1.5s debounce, damaged NPC detection fires engage_add."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.has_add = False
        ctx.combat.engaged = False
        ctx.combat.pull_target_id = None

        world = MagicMock()
        world.pet_spawn = None
        world.damaged_npcs_near.return_value = [MagicMock()]
        ctx.world = world

        rs = _CombatRuleState(add_first_seen=time.time() - 2.0)  # 2s ago > 1.5s
        result = _should_engage_add(state, ctx, rs)
        assert result is True

    def test_damaged_npc_debounce_not_expired_yet(self) -> None:
        """Within 1.5s debounce, returns False."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.has_add = False
        ctx.combat.engaged = False
        ctx.combat.pull_target_id = None

        world = MagicMock()
        world.pet_spawn = None
        world.damaged_npcs_near.return_value = [MagicMock()]
        ctx.world = world

        rs = _CombatRuleState(add_first_seen=time.time() - 0.5)  # 0.5s ago < 1.5s
        result = _should_engage_add(state, ctx, rs)
        assert result is False

    def test_no_damaged_npc_resets_debounce(self) -> None:
        """When no damaged NPCs found, debounce timer resets."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.has_add = False
        ctx.combat.engaged = False
        ctx.combat.pull_target_id = None

        world = MagicMock()
        world.pet_spawn = None
        world.damaged_npcs_near.return_value = []
        ctx.world = world

        rs = _CombatRuleState(add_first_seen=time.time() - 2.0)
        result = _should_engage_add(state, ctx, rs)
        assert result is False
        assert rs.add_first_seen == 0.0

    def test_damaged_npc_uses_pet_spawn_position(self) -> None:
        """When world.pet_spawn exists, use its position for damaged_npcs_near."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.has_add = False
        ctx.combat.engaged = False
        ctx.combat.pull_target_id = None

        pet_spawn = SimpleNamespace(x=100.0, y=200.0)
        world = MagicMock()
        world.pet_spawn = pet_spawn
        world.damaged_npcs_near.return_value = [MagicMock()]
        ctx.world = world

        rs = _CombatRuleState()
        _should_engage_add(state, ctx, rs)

        # Should have been called with pet's position
        world.damaged_npcs_near.assert_called_once()
