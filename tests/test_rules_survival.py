"""Tests for brain.rules.survival -- flee, rest, death recovery, evade conditions.

Condition functions are called directly with GameState + AgentContext.
Feature flags are set per-fixture to isolate tests from global state.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from brain.context import AgentContext
from brain.rules.survival import (
    _check_core_safety_floors,
    _fight_winnable,
    _mob_attacking_player,
    _next_pull_mana_estimate,
    _rest_exit_check,
    _rest_suppressed,
    _score_death_recovery,
    _score_evade,
    _score_feign_death,
    _score_flee,
    _score_rest,
    _should_death_recover,
    _should_evade,
    _should_feign_death,
    _should_flee,
    _should_rest,
    _SurvivalRuleState,
    compute_flee_urgency,
    feign_death_condition,
    flee_condition,
    register,
    reset_flee_hysteresis,
    rest_needs_check,
)
from core.features import flags
from core.types import DeathRecoveryMode, Point
from eq.loadout import Spell
from perception.state import GameState
from tests.factories import make_game_state, make_spawn


@pytest.fixture(autouse=True)
def _enable_flags() -> None:
    """Enable survival-relevant flags for all tests in this module."""
    flags.flee = True
    flags.rest = True


# ---------------------------------------------------------------------------
# _should_flee
# ---------------------------------------------------------------------------


class TestShouldFlee:
    """Flee condition: urgency-based with hysteresis + safety floors."""

    @pytest.mark.parametrize(
        "hp_pct, imminent_threat, imminent_con, pet_alive, expected",
        [
            pytest.param(0.20, False, "", True, True, id="safety_floor_hp_below_40pct"),
            pytest.param(0.90, False, "", True, False, id="healthy_no_threat"),
            pytest.param(0.50, True, "red", True, True, id="red_imminent_threat"),
            pytest.param(0.80, True, "yellow", False, True, id="yellow_threat_no_pet"),
            pytest.param(0.80, False, "", False, False, id="no_pet_no_threat_healthy"),
            pytest.param(0.35, False, "", False, True, id="low_hp_no_pet"),
        ],
    )
    def test_basic_scenarios(
        self,
        hp_pct: float,
        imminent_threat: bool,
        imminent_con: str,
        pet_alive: bool,
        expected: bool,
    ) -> None:
        hp = int(hp_pct * 1000)
        state = make_game_state(hp_current=hp, hp_max=1000)
        ctx = AgentContext()
        ctx.pet.alive = pet_alive
        ctx.threat.imminent_threat = imminent_threat
        ctx.threat.imminent_threat_con = imminent_con

        result = _should_flee(state, ctx)
        assert result is expected

    def test_flee_disabled_returns_false(self) -> None:
        flags.flee = False
        state = make_game_state(hp_current=100, hp_max=1000)
        ctx = AgentContext()
        assert _should_flee(state, ctx) is False

    def test_npc_attacking_player_no_pet(self) -> None:
        """NPC targeting player with no pet alive triggers flee safety floor."""
        attacker = make_spawn(
            spawn_id=200,
            name="a_skeleton",
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
            target_name="TestPlayer",
        )
        state = make_game_state(
            hp_current=800,
            hp_max=1000,
            spawns=(attacker,),
        )
        ctx = AgentContext()
        ctx.pet.alive = False
        ctx.combat.engaged = False

        assert _should_flee(state, ctx) is True

    def test_npc_attacking_player_with_pet_no_flee(self) -> None:
        """NPC nearby but pet alive -- no-pet safety floor doesn't fire."""
        attacker = make_spawn(
            spawn_id=200,
            name="a_skeleton",
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
            target_name="TestPlayer",
        )
        state = make_game_state(
            hp_current=900,
            hp_max=1000,
            spawns=(attacker,),
        )
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.combat.engaged = False

        assert _should_flee(state, ctx) is False

    def test_hysteresis_stays_active_above_exit(self) -> None:
        """Once flee is active, it stays active until urgency drops below exit threshold."""
        state = make_game_state(hp_current=200, hp_max=1000)
        ctx = AgentContext()
        ctx.pet.alive = True

        # First call triggers flee
        result1 = _should_flee(state, ctx)
        assert result1 is True
        assert ctx.combat.flee_urgency_active is True

        # Increase HP somewhat but still above exit threshold
        state2 = make_game_state(hp_current=500, hp_max=1000)
        result2 = _should_flee(state2, ctx)
        # With hp=50%, urgency is still significant -- stays active
        assert result2 is True

    def test_hysteresis_disengages_when_safe(self) -> None:
        """Flee disengages when urgency drops below exit threshold."""
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.combat.flee_urgency_active = True

        # Full HP -> urgency ~0 -> below exit threshold
        state = make_game_state(hp_current=1000, hp_max=1000)
        result = _should_flee(state, ctx)
        assert result is False
        assert ctx.combat.flee_urgency_active is False


# ---------------------------------------------------------------------------
# compute_flee_urgency
# ---------------------------------------------------------------------------


class TestComputeFleeUrgency:
    """Flee urgency score: 0.0=safe, 1.0=critical."""

    @pytest.mark.parametrize(
        "hp_pct, expected_min, expected_max",
        [
            pytest.param(1.0, 0.0, 0.05, id="full_hp_near_zero"),
            pytest.param(0.35, 0.40, 0.80, id="low_hp_moderate_urgency"),
            pytest.param(0.10, 0.80, 1.0, id="critical_hp_high_urgency"),
        ],
    )
    def test_hp_axis(self, hp_pct: float, expected_min: float, expected_max: float) -> None:
        hp = int(hp_pct * 1000)
        state = make_game_state(hp_current=hp, hp_max=1000)
        ctx = AgentContext()
        ctx.pet.alive = True

        urgency = compute_flee_urgency(ctx, state)
        assert expected_min <= urgency <= expected_max

    def test_red_threat_adds_urgency(self) -> None:
        state = make_game_state(hp_current=700, hp_max=1000)
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.threat.imminent_threat = True
        ctx.threat.imminent_threat_con = "red"

        urgency = compute_flee_urgency(ctx, state)
        # RED threat adds 0.5 to base HP urgency
        assert urgency >= 0.5

    def test_urgency_clamped_to_unit(self) -> None:
        """Urgency is always in [0, 1]."""
        state = make_game_state(hp_current=10, hp_max=1000)
        ctx = AgentContext()
        ctx.pet.alive = False
        ctx.threat.imminent_threat = True
        ctx.threat.imminent_threat_con = "red"

        urgency = compute_flee_urgency(ctx, state)
        assert 0.0 <= urgency <= 1.0


# ---------------------------------------------------------------------------
# _should_rest
# ---------------------------------------------------------------------------


class TestShouldRest:
    """Rest condition: HP/mana deficit triggers rest, combat blocks it."""

    @pytest.mark.parametrize(
        "hp_pct, mana_pct, engaged, resting_already, expected",
        [
            pytest.param(0.50, 0.80, False, False, True, id="hp_low_enter_rest"),
            pytest.param(0.90, 0.15, False, False, True, id="mana_very_low_enter_rest"),
            pytest.param(0.95, 0.80, False, False, False, id="healthy_no_rest"),
            pytest.param(0.50, 0.50, True, False, False, id="engaged_no_rest"),
        ],
    )
    def test_basic_scenarios(
        self,
        hp_pct: float,
        mana_pct: float,
        engaged: bool,
        resting_already: bool,
        expected: bool,
    ) -> None:
        hp = int(hp_pct * 1000)
        mana = int(mana_pct * 500)
        state = make_game_state(hp_current=hp, hp_max=1000, mana_current=mana, mana_max=500)
        ctx = AgentContext()
        ctx.combat.engaged = engaged
        # Set rest thresholds to their defaults
        ctx.rest_hp_entry = 0.85
        ctx.rest_mana_entry = 0.40
        # Set buff/flee times far in the past so suppression doesn't fire
        ctx.player.last_buff_time = 0.0
        ctx.player.last_flee_time = 0.0

        rs = _SurvivalRuleState(resting=resting_already)

        result = _should_rest(state, ctx, rs)
        assert result is expected

    def test_rest_disabled_returns_false(self) -> None:
        flags.rest = False
        state = make_game_state(hp_current=100, hp_max=1000)
        ctx = AgentContext()
        rs = _SurvivalRuleState()
        assert _should_rest(state, ctx, rs) is False
        # resting flag should be reset when rest is disabled
        rs = _SurvivalRuleState(resting=True)
        assert _should_rest(state, ctx, rs) is False
        assert rs.resting is False

    def test_resting_exits_when_thresholds_met(self) -> None:
        """When already resting, exit once HP and mana thresholds are met."""
        state = make_game_state(hp_current=950, hp_max=1000, mana_current=400, mana_max=500)
        ctx = AgentContext()
        ctx.rest_hp_threshold = 0.92
        ctx.rest_mana_threshold = 0.60
        ctx.player.last_buff_time = 0.0
        ctx.player.last_flee_time = 0.0
        rs = _SurvivalRuleState(resting=True)

        result = _should_rest(state, ctx, rs)
        assert result is False
        assert rs.resting is False

    def test_imminent_threat_cancels_rest(self) -> None:
        state = make_game_state(hp_current=500, hp_max=1000)
        ctx = AgentContext()
        ctx.threat.imminent_threat = True
        ctx.player.last_buff_time = 0.0
        ctx.player.last_flee_time = 0.0
        rs = _SurvivalRuleState(resting=True)

        result = _should_rest(state, ctx, rs)
        assert result is False
        assert rs.resting is False

    def test_evasion_point_cancels_rest(self) -> None:
        state = make_game_state(hp_current=500, hp_max=1000)
        ctx = AgentContext()
        ctx.threat.evasion_point = Point(100.0, 200.0, 0.0)
        ctx.player.last_buff_time = 0.0
        ctx.player.last_flee_time = 0.0
        rs = _SurvivalRuleState(resting=True)

        result = _should_rest(state, ctx, rs)
        assert result is False

    def test_recently_buffed_suppresses_rest_entry(self) -> None:
        """Rest suppressed for 10s after buff cast (HP% dip from max increase)."""
        state = make_game_state(hp_current=800, hp_max=1000)
        ctx = AgentContext()
        ctx.player.last_buff_time = time.time()  # just now
        ctx.player.last_flee_time = 0.0
        rs = _SurvivalRuleState()

        result = _should_rest(state, ctx, rs)
        assert result is False

    def test_mana_low_but_pet_healthy_and_hp_ok_skips_rest(self) -> None:
        """With pet HP >= 70% and player HP >= 85%, mana > 20% doesn't trigger rest."""
        state = make_game_state(hp_current=900, hp_max=1000, mana_current=150, mana_max=500)
        ctx = AgentContext()
        ctx.rest_mana_entry = 0.40  # mana_pct=0.30 < 0.40 entry
        ctx.player.last_buff_time = 0.0
        ctx.player.last_flee_time = 0.0
        # No world model, so pet HP check is skipped (pet_hp defaults to 1.0)
        rs = _SurvivalRuleState()

        result = _should_rest(state, ctx, rs)
        assert result is False

    def test_mana_critical_always_triggers_rest(self) -> None:
        """Mana below 20% always triggers rest regardless of pet/HP."""
        state = make_game_state(hp_current=900, hp_max=1000, mana_current=50, mana_max=500)
        ctx = AgentContext()
        ctx.rest_mana_entry = 0.40
        ctx.player.last_buff_time = 0.0
        ctx.player.last_flee_time = 0.0
        rs = _SurvivalRuleState()

        result = _should_rest(state, ctx, rs)
        assert result is True

    def test_fresh_state_per_call(self) -> None:
        """Separate _SurvivalRuleState instances should not cross-contaminate."""
        state = make_game_state(hp_current=500, hp_max=1000)
        ctx = AgentContext()
        ctx.player.last_buff_time = 0.0
        ctx.player.last_flee_time = 0.0

        rs_a = _SurvivalRuleState()
        rs_b = _SurvivalRuleState()

        _should_rest(state, ctx, rs_a)
        assert rs_a.resting is True

        # rs_b is independent
        assert rs_b.resting is False


# ---------------------------------------------------------------------------
# _should_death_recover
# ---------------------------------------------------------------------------


class TestShouldDeathRecover:
    """Death recovery condition: player dead + recovery mode enabled."""

    @pytest.mark.parametrize(
        "dead, recovery_mode, deaths, expected",
        [
            pytest.param(True, DeathRecoveryMode.ON, 0, True, id="dead_recovery_on"),
            pytest.param(False, DeathRecoveryMode.ON, 0, False, id="alive_no_recovery"),
            pytest.param(True, DeathRecoveryMode.OFF, 0, False, id="dead_recovery_off"),
            pytest.param(True, DeathRecoveryMode.SMART, 0, True, id="dead_smart_first_death"),
            pytest.param(True, DeathRecoveryMode.SMART, 1, True, id="dead_smart_second_death"),
            pytest.param(True, DeathRecoveryMode.SMART, 2, False, id="dead_smart_third_death_blocked"),
        ],
    )
    def test_death_recovery_scenarios(
        self,
        dead: bool,
        recovery_mode: DeathRecoveryMode,
        deaths: int,
        expected: bool,
    ) -> None:
        flags.death_recovery = recovery_mode
        state = make_game_state()
        ctx = AgentContext()
        ctx.player.dead = dead
        ctx.player.deaths = deaths

        result = _should_death_recover(state, ctx)
        assert result is expected


# ---------------------------------------------------------------------------
# _should_evade
# ---------------------------------------------------------------------------


class TestShouldEvade:
    """Evade condition: evasion point set + not engaged (or patrol evade)."""

    @pytest.mark.parametrize(
        "evasion_point, engaged, patrol_evade, expected",
        [
            pytest.param((100.0, 200.0), False, False, True, id="evasion_point_not_engaged"),
            pytest.param(None, False, False, False, id="no_evasion_point"),
            pytest.param((100.0, 200.0), True, False, False, id="engaged_no_patrol_evade"),
        ],
    )
    def test_basic_scenarios(
        self,
        evasion_point: tuple | None,
        engaged: bool,
        patrol_evade: bool,
        expected: bool,
    ) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.threat.evasion_point = evasion_point
        ctx.threat.patrol_evade = patrol_evade
        ctx.combat.engaged = engaged

        rs = _SurvivalRuleState()

        result = _should_evade(state, ctx, rs)
        assert result is expected

    def test_patrol_evade_during_combat(self) -> None:
        """RED patrol collision during combat triggers evade (with debounce)."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.threat.evasion_point = Point(100.0, 200.0, 0.0)
        ctx.threat.patrol_evade = True
        ctx.combat.engaged = True

        # last_patrol_evade long enough ago to pass the 8s debounce
        rs = _SurvivalRuleState()

        result = _should_evade(state, ctx, rs)
        assert result is True

    def test_patrol_evade_debounce(self) -> None:
        """Patrol evade during combat is debounced (8s cooldown)."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.threat.evasion_point = Point(100.0, 200.0, 0.0)
        ctx.threat.patrol_evade = True
        ctx.combat.engaged = True

        # last_patrol_evade was very recent
        rs = _SurvivalRuleState(last_patrol_evade=time.time() - 2.0)

        result = _should_evade(state, ctx, rs)
        assert result is False

    def test_evade_logged_flag_set_on_first_call(self) -> None:
        """evade_logged is set to True on first evade detection."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.threat.evasion_point = Point(100.0, 200.0, 0.0)

        rs = _SurvivalRuleState()

        _should_evade(state, ctx, rs)
        assert rs.evade_logged is True

    def test_evade_logged_resets_when_clear(self) -> None:
        """evade_logged resets to False when no evasion point."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.threat.evasion_point = None

        rs = _SurvivalRuleState(evade_logged=True)

        _should_evade(state, ctx, rs)
        assert rs.evade_logged is False


# ---------------------------------------------------------------------------
# Property-based tests: flee urgency invariants
# ---------------------------------------------------------------------------


class TestFleeUrgencyProperties:
    @given(hp_pct=st.floats(0.0, 1.0, allow_nan=False))
    def test_urgency_always_in_unit_interval(self, hp_pct: float) -> None:
        """Flee urgency is always in [0.0, 1.0] for any HP."""
        hp = max(1, int(hp_pct * 1000))
        state = make_game_state(hp_current=hp, hp_max=1000)
        ctx = AgentContext()
        ctx.pet.alive = True
        urgency = compute_flee_urgency(ctx, state)
        assert 0.0 <= urgency <= 1.0

    @given(
        hp1=st.floats(0.0, 1.0, allow_nan=False),
        hp2=st.floats(0.0, 1.0, allow_nan=False),
    )
    def test_urgency_monotonic_in_hp(self, hp1: float, hp2: float) -> None:
        """Lower HP should produce >= urgency than higher HP, all else equal."""
        assume(abs(hp1 - hp2) > 0.05)  # avoid degenerate near-equal
        ctx = AgentContext()
        ctx.pet.alive = True
        s1 = make_game_state(hp_current=max(1, int(hp1 * 1000)), hp_max=1000)
        s2 = make_game_state(hp_current=max(1, int(hp2 * 1000)), hp_max=1000)
        u1 = compute_flee_urgency(ctx, s1)
        u2 = compute_flee_urgency(ctx, s2)
        if hp1 < hp2:
            assert u1 >= u2
        else:
            assert u2 >= u1


# ---------------------------------------------------------------------------
# _check_core_safety_floors
# ---------------------------------------------------------------------------


class TestCheckCoreSafetyFloors:
    """Direct tests for _check_core_safety_floors helper."""

    def test_hp_below_40_triggers_floor(self) -> None:
        state = make_game_state(hp_current=350, hp_max=1000)
        ctx = AgentContext()
        assert _check_core_safety_floors(ctx, state, "test") is True

    def test_hp_above_40_no_floor(self) -> None:
        state = make_game_state(hp_current=500, hp_max=1000)
        ctx = AgentContext()
        assert _check_core_safety_floors(ctx, state, "test") is None

    def test_pet_died_unwinnable_fight(self) -> None:
        """Pet just died in active combat with unwinnable fight triggers floor."""
        target = make_spawn(spawn_id=300, hp_current=80, hp_max=100)
        state = make_game_state(hp_current=500, hp_max=1000, target=target)
        ctx = AgentContext()
        ctx.pet.prev_alive = True
        ctx.pet.alive = False
        ctx.combat.engaged = True  # in_active_combat
        assert _check_core_safety_floors(ctx, state, "test") is True

    def test_pet_died_winnable_fight_no_floor(self) -> None:
        """Pet died but fight is winnable (player HP > 60%, mob HP < 50%) -- no floor."""
        target = make_spawn(spawn_id=300, hp_current=40, hp_max=100)
        state = make_game_state(hp_current=700, hp_max=1000, target=target)
        ctx = AgentContext()
        ctx.pet.prev_alive = True
        ctx.pet.alive = False
        ctx.combat.engaged = True
        assert _check_core_safety_floors(ctx, state, "test") is None

    def test_red_imminent_threat_triggers_floor(self) -> None:
        state = make_game_state(hp_current=800, hp_max=1000)
        ctx = AgentContext()
        ctx.threat.imminent_threat = True
        ctx.threat.imminent_threat_con = "red"
        assert _check_core_safety_floors(ctx, state, "test") is True

    def test_yellow_threat_no_floor(self) -> None:
        state = make_game_state(hp_current=800, hp_max=1000)
        ctx = AgentContext()
        ctx.threat.imminent_threat = True
        ctx.threat.imminent_threat_con = "yellow"
        assert _check_core_safety_floors(ctx, state, "test") is None


# ---------------------------------------------------------------------------
# _fight_winnable
# ---------------------------------------------------------------------------


class TestFightWinnable:
    def test_no_target(self) -> None:
        state = make_game_state(hp_current=800, hp_max=1000)
        assert _fight_winnable(state) is False

    def test_target_hp_max_zero(self) -> None:
        target = make_spawn(spawn_id=300, hp_current=50, hp_max=0)
        state = make_game_state(hp_current=800, hp_max=1000, target=target)
        assert _fight_winnable(state) is False

    def test_winnable(self) -> None:
        """Player HP > 60% and mob HP < 50% -> winnable."""
        target = make_spawn(spawn_id=300, hp_current=40, hp_max=100)
        state = make_game_state(hp_current=700, hp_max=1000, target=target)
        assert _fight_winnable(state) is True

    def test_not_winnable_player_too_low(self) -> None:
        """Player HP <= 60% -> not winnable even if mob is low."""
        target = make_spawn(spawn_id=300, hp_current=40, hp_max=100)
        state = make_game_state(hp_current=500, hp_max=1000, target=target)
        assert _fight_winnable(state) is False

    def test_not_winnable_mob_too_healthy(self) -> None:
        """Mob HP >= 50% -> not winnable."""
        target = make_spawn(spawn_id=300, hp_current=60, hp_max=100)
        state = make_game_state(hp_current=800, hp_max=1000, target=target)
        assert _fight_winnable(state) is False


# ---------------------------------------------------------------------------
# _mob_attacking_player
# ---------------------------------------------------------------------------


class TestMobAttackingPlayer:
    def test_npc_targeting_player_within_range(self) -> None:
        attacker = make_spawn(
            spawn_id=200,
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
            target_name="TestPlayer",
        )
        state = make_game_state(spawns=(attacker,))
        assert _mob_attacking_player(state) is True

    def test_npc_targeting_player_out_of_range(self) -> None:
        attacker = make_spawn(
            spawn_id=200,
            x=500.0,
            y=500.0,
            hp_current=100,
            hp_max=100,
            target_name="TestPlayer",
        )
        state = make_game_state(spawns=(attacker,))
        assert _mob_attacking_player(state) is False

    def test_npc_not_targeting_player(self) -> None:
        attacker = make_spawn(
            spawn_id=200,
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
            target_name="SomeoneElse",
        )
        state = make_game_state(spawns=(attacker,))
        assert _mob_attacking_player(state) is False

    def test_dead_npc_ignored(self) -> None:
        attacker = make_spawn(
            spawn_id=200,
            x=5.0,
            y=5.0,
            hp_current=0,
            hp_max=100,
            target_name="TestPlayer",
        )
        state = make_game_state(spawns=(attacker,))
        assert _mob_attacking_player(state) is False

    def test_player_spawn_ignored(self) -> None:
        """Players (spawn_type=0) are not NPCs."""
        player = make_spawn(
            spawn_id=200,
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
            target_name="TestPlayer",
            spawn_type=0,
        )
        state = make_game_state(spawns=(player,))
        assert _mob_attacking_player(state) is False

    def test_no_spawns(self) -> None:
        state = make_game_state(spawns=())
        assert _mob_attacking_player(state) is False


# ---------------------------------------------------------------------------
# compute_flee_urgency -- additional edge cases
# ---------------------------------------------------------------------------


class TestFleeUrgencyEdgeCases:
    """Cover branches in compute_flee_urgency not hit by TestComputeFleeUrgency."""

    def test_pet_died_unwinnable_adds_urgency(self) -> None:
        """Pet died mid-combat + unwinnable fight -> +0.4 urgency."""
        target = make_spawn(spawn_id=300, hp_current=80, hp_max=100)
        state = make_game_state(hp_current=300, hp_max=1000, target=target)
        ctx = AgentContext()
        ctx.pet.prev_alive = True
        ctx.pet.alive = False
        ctx.combat.engaged = True
        urgency = compute_flee_urgency(ctx, state)
        # HP axis alone at 30% ~ 0.76, plus 0.4 = clamped to 1.0
        assert urgency >= 0.65

    def test_extra_npcs_add_urgency(self) -> None:
        """Multiple damaged NPCs within 40u add urgency."""
        npc1 = make_spawn(spawn_id=200, x=5.0, y=5.0, hp_current=80, hp_max=100)
        npc2 = make_spawn(spawn_id=201, x=10.0, y=10.0, hp_current=60, hp_max=100)
        npc3 = make_spawn(spawn_id=202, x=15.0, y=15.0, hp_current=40, hp_max=100)
        state = make_game_state(hp_current=500, hp_max=1000, spawns=(npc1, npc2, npc3))
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.combat.engaged = True
        # With 3 damaged NPCs, extra_npc_count-1 = 2 -> +0.30
        urgency_with = compute_flee_urgency(ctx, state)

        state_no_adds = make_game_state(hp_current=500, hp_max=1000)
        ctx2 = AgentContext()
        ctx2.pet.alive = True
        ctx2.combat.engaged = True
        urgency_without = compute_flee_urgency(ctx2, state_no_adds)
        assert urgency_with > urgency_without

    def test_target_nearly_dead_reduces_urgency(self) -> None:
        """Target NPC with < 15% HP reduces urgency by 0.25."""
        target = make_spawn(spawn_id=300, hp_current=10, hp_max=100)
        state_low = make_game_state(hp_current=500, hp_max=1000, target=target)
        ctx = AgentContext()
        ctx.pet.alive = True

        target_healthy = make_spawn(spawn_id=300, hp_current=80, hp_max=100)
        state_healthy = make_game_state(hp_current=500, hp_max=1000, target=target_healthy)
        ctx2 = AgentContext()
        ctx2.pet.alive = True

        urgency_low_target = compute_flee_urgency(ctx, state_low)
        urgency_healthy_target = compute_flee_urgency(ctx2, state_healthy)
        assert urgency_low_target < urgency_healthy_target

    def test_fight_history_danger_adds_urgency(self) -> None:
        """Learned danger > 0.7 from FightHistory adds urgency."""
        state = make_game_state(hp_current=500, hp_max=1000)
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.combat.engaged = True
        ctx.defeat_tracker.last_fight_name = "a_skeleton"
        ctx.fight_history = SimpleNamespace(
            learned_danger=lambda name: 0.9,
            get_all_stats=lambda: {},
            has_learned=lambda k: False,
        )
        urgency_danger = compute_flee_urgency(ctx, state)

        ctx2 = AgentContext()
        ctx2.pet.alive = True
        ctx2.combat.engaged = True
        urgency_no_danger = compute_flee_urgency(ctx2, state)
        assert urgency_danger > urgency_no_danger

    def test_pet_hp_low_adds_urgency(self) -> None:
        """Pet HP < 30% adds urgency."""
        state = make_game_state(hp_current=700, hp_max=1000)
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.world = SimpleNamespace(pet_hp_pct=0.20)
        urgency_low_pet = compute_flee_urgency(ctx, state)

        ctx2 = AgentContext()
        ctx2.pet.alive = True
        ctx2.world = SimpleNamespace(pet_hp_pct=0.80)
        urgency_healthy_pet = compute_flee_urgency(ctx2, state)
        assert urgency_low_pet > urgency_healthy_pet

    def test_no_pet_mob_on_player_adds_urgency(self) -> None:
        """No pet + mob attacking player adds 0.5 urgency."""
        attacker = make_spawn(
            spawn_id=200,
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
            target_name="TestPlayer",
        )
        state = make_game_state(hp_current=700, hp_max=1000, spawns=(attacker,))
        ctx = AgentContext()
        ctx.pet.alive = False
        ctx.combat.engaged = False
        urgency = compute_flee_urgency(ctx, state)
        assert urgency >= 0.5


# ---------------------------------------------------------------------------
# feign_death_condition (standalone)
# ---------------------------------------------------------------------------


class TestFeignDeathCondition:
    def test_no_fd_spell_returns_false(self) -> None:
        state = make_game_state(hp_current=100, hp_max=1000)
        ctx = AgentContext()
        assert feign_death_condition(ctx, state, fd_spell=None) is False

    def test_fd_spell_no_gem_returns_false(self) -> None:
        fd = Spell("Feign Death", gem=0, cast_time=1.0, mana_cost=10)
        state = make_game_state(hp_current=100, hp_max=1000, mana_current=500, mana_max=500)
        ctx = AgentContext()
        assert feign_death_condition(ctx, state, fd_spell=fd) is False

    def test_fd_spell_wrong_name_returns_false(self) -> None:
        fd = Spell("Lifetap", gem=1, cast_time=1.0, mana_cost=10)
        state = make_game_state(hp_current=100, hp_max=1000, mana_current=500, mana_max=500)
        ctx = AgentContext()
        assert feign_death_condition(ctx, state, fd_spell=fd) is False

    def test_insufficient_mana_returns_false(self) -> None:
        fd = Spell("Feign Death", gem=1, cast_time=1.0, mana_cost=100)
        state = make_game_state(hp_current=100, hp_max=1000, mana_current=50, mana_max=500)
        ctx = AgentContext()
        assert feign_death_condition(ctx, state, fd_spell=fd) is False

    def test_hp_below_40_triggers(self) -> None:
        fd = Spell("Feign Death", gem=1, cast_time=1.0, mana_cost=10)
        state = make_game_state(hp_current=350, hp_max=1000, mana_current=500, mana_max=500)
        ctx = AgentContext()
        assert feign_death_condition(ctx, state, fd_spell=fd) is True

    def test_red_threat_triggers(self) -> None:
        fd = Spell("Feign Death", gem=1, cast_time=1.0, mana_cost=10)
        state = make_game_state(hp_current=800, hp_max=1000, mana_current=500, mana_max=500)
        ctx = AgentContext()
        ctx.threat.imminent_threat = True
        ctx.threat.imminent_threat_con = "red"
        assert feign_death_condition(ctx, state, fd_spell=fd) is True

    def test_no_safety_floor_returns_false(self) -> None:
        fd = Spell("Feign Death", gem=1, cast_time=1.0, mana_cost=10)
        state = make_game_state(hp_current=800, hp_max=1000, mana_current=500, mana_max=500)
        ctx = AgentContext()
        assert feign_death_condition(ctx, state, fd_spell=fd) is False


# ---------------------------------------------------------------------------
# rest_needs_check
# ---------------------------------------------------------------------------


class TestRestNeedsCheck:
    def test_hp_low(self) -> None:
        state = make_game_state(hp_current=700, hp_max=1000, mana_current=500, mana_max=500)
        ctx = AgentContext()
        ctx.rest_hp_entry = 0.85
        hp_low, mana_low, pet_low = rest_needs_check(ctx, state)
        assert hp_low is True
        assert mana_low is False

    def test_mana_low_critical(self) -> None:
        state = make_game_state(hp_current=900, hp_max=1000, mana_current=50, mana_max=500)
        ctx = AgentContext()
        ctx.rest_mana_entry = 0.40
        hp_low, mana_low, pet_low = rest_needs_check(ctx, state)
        assert mana_low is True

    def test_mana_low_suppressed_when_pet_healthy_hp_ok(self) -> None:
        """Mana 30% but pet HP >= 70% and player HP >= 85% -> mana_low suppressed."""
        state = make_game_state(hp_current=900, hp_max=1000, mana_current=150, mana_max=500)
        ctx = AgentContext()
        ctx.rest_mana_entry = 0.40
        hp_low, mana_low, pet_low = rest_needs_check(ctx, state)
        assert mana_low is False

    def test_pet_low(self) -> None:
        state = make_game_state(hp_current=950, hp_max=1000, mana_current=400, mana_max=500)
        ctx = AgentContext()
        ctx.rest_hp_entry = 0.85
        ctx.rest_mana_entry = 0.40
        ctx.pet.alive = True
        world = SimpleNamespace(pet_hp_pct=0.40)
        hp_low, mana_low, pet_low = rest_needs_check(ctx, state, world=world)
        assert pet_low is True

    def test_pet_hp_negative_treated_as_full(self) -> None:
        """pet_hp_pct < 0 means unknown -> treated as 1.0."""
        state = make_game_state(hp_current=950, hp_max=1000, mana_current=400, mana_max=500)
        ctx = AgentContext()
        ctx.pet.alive = True
        world = SimpleNamespace(pet_hp_pct=-1.0)
        hp_low, mana_low, pet_low = rest_needs_check(ctx, state, world=world)
        assert pet_low is False

    def test_all_healthy(self) -> None:
        state = make_game_state(hp_current=950, hp_max=1000, mana_current=400, mana_max=500)
        ctx = AgentContext()
        ctx.rest_hp_entry = 0.85
        ctx.rest_mana_entry = 0.40
        hp_low, mana_low, pet_low = rest_needs_check(ctx, state)
        assert not hp_low and not mana_low and not pet_low

    def test_zero_mana_max(self) -> None:
        """Classes with 0 mana_max never report mana_low."""
        state = make_game_state(hp_current=950, hp_max=1000, mana_current=0, mana_max=0)
        ctx = AgentContext()
        ctx.rest_mana_entry = 0.40
        hp_low, mana_low, pet_low = rest_needs_check(ctx, state)
        assert mana_low is False

    def test_hp_deficit_too_small_no_rest(self) -> None:
        """HP % is below entry but deficit < 5 -> no hp_low."""
        state = make_game_state(hp_current=99, hp_max=100, mana_current=400, mana_max=500)
        ctx = AgentContext()
        ctx.rest_hp_entry = 0.999  # very high threshold
        hp_low, mana_low, pet_low = rest_needs_check(ctx, state)
        assert hp_low is False


# ---------------------------------------------------------------------------
# _score_death_recovery
# ---------------------------------------------------------------------------


class TestScoreDeathRecovery:
    def test_alive_returns_zero(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.player.dead = False
        assert _score_death_recovery(state, ctx) == 0.0

    def test_dead_recovery_on(self) -> None:
        from core.types import DeathRecoveryMode

        flags.death_recovery = DeathRecoveryMode.ON
        state = make_game_state()
        ctx = AgentContext()
        ctx.player.dead = True
        assert _score_death_recovery(state, ctx) == 1.0

    def test_dead_recovery_off(self) -> None:
        from core.types import DeathRecoveryMode

        flags.death_recovery = DeathRecoveryMode.OFF
        state = make_game_state()
        ctx = AgentContext()
        ctx.player.dead = True
        assert _score_death_recovery(state, ctx) == 0.0


# ---------------------------------------------------------------------------
# _should_feign_death (via brain rule interface)
# ---------------------------------------------------------------------------


class TestShouldFeignDeath:
    def test_flee_disabled(self) -> None:
        flags.flee = False
        state = make_game_state()
        ctx = AgentContext()
        assert _should_feign_death(state, ctx) is False

    @patch("brain.rules.survival.get_spell_by_role", return_value=None)
    def test_no_fd_spell(self, _mock: object) -> None:
        state = make_game_state()
        ctx = AgentContext()
        assert _should_feign_death(state, ctx) is False

    @patch(
        "brain.rules.survival.get_spell_by_role",
        return_value=Spell("Feign Death", gem=1, cast_time=1.0, mana_cost=10),
    )
    def test_mana_too_low(self, _mock: object) -> None:
        state = make_game_state(mana_current=5, mana_max=500)
        ctx = AgentContext()
        assert _should_feign_death(state, ctx) is False

    @patch(
        "brain.rules.survival.get_spell_by_role",
        return_value=Spell("Feign Death", gem=1, cast_time=1.0, mana_cost=10),
    )
    def test_hp_floor_triggers(self, _mock: object) -> None:
        state = make_game_state(hp_current=300, hp_max=1000, mana_current=500, mana_max=500)
        ctx = AgentContext()
        assert _should_feign_death(state, ctx) is True

    @patch(
        "brain.rules.survival.get_spell_by_role",
        return_value=Spell("Feign Death", gem=1, cast_time=1.0, mana_cost=10),
    )
    def test_no_trigger_returns_false(self, _mock: object) -> None:
        state = make_game_state(hp_current=800, hp_max=1000, mana_current=500, mana_max=500)
        ctx = AgentContext()
        assert _should_feign_death(state, ctx) is False


# ---------------------------------------------------------------------------
# _score_feign_death
# ---------------------------------------------------------------------------


class TestScoreFeignDeath:
    def test_flee_disabled_returns_zero(self) -> None:
        flags.flee = False
        state = make_game_state()
        ctx = AgentContext()
        assert _score_feign_death(state, ctx) == 0.0

    @patch("brain.rules.survival.get_spell_by_role", return_value=None)
    def test_no_spell_returns_zero(self, _mock: object) -> None:
        state = make_game_state()
        ctx = AgentContext()
        assert _score_feign_death(state, ctx) == 0.0

    @patch(
        "brain.rules.survival.get_spell_by_role",
        return_value=Spell("Feign Death", gem=1, cast_time=1.0, mana_cost=10),
    )
    def test_low_mana_returns_zero(self, _mock: object) -> None:
        state = make_game_state(mana_current=5, mana_max=500)
        ctx = AgentContext()
        assert _score_feign_death(state, ctx) == 0.0

    @patch(
        "brain.rules.survival.get_spell_by_role",
        return_value=Spell("Feign Death", gem=1, cast_time=1.0, mana_cost=10),
    )
    def test_hp_below_40_returns_urgency(self, _mock: object) -> None:
        state = make_game_state(hp_current=300, hp_max=1000, mana_current=500, mana_max=500)
        ctx = AgentContext()
        score = _score_feign_death(state, ctx)
        assert 0.0 < score <= 1.0

    @patch(
        "brain.rules.survival.get_spell_by_role",
        return_value=Spell("Feign Death", gem=1, cast_time=1.0, mana_cost=10),
    )
    def test_pet_died_unwinnable(self, _mock: object) -> None:
        target = make_spawn(spawn_id=300, hp_current=80, hp_max=100)
        state = make_game_state(
            hp_current=500,
            hp_max=1000,
            mana_current=500,
            mana_max=500,
            target=target,
        )
        ctx = AgentContext()
        ctx.pet.prev_alive = True
        ctx.pet.alive = False
        ctx.combat.engaged = True
        assert _score_feign_death(state, ctx) == 1.0

    @patch(
        "brain.rules.survival.get_spell_by_role",
        return_value=Spell("Feign Death", gem=1, cast_time=1.0, mana_cost=10),
    )
    def test_red_threat(self, _mock: object) -> None:
        state = make_game_state(hp_current=800, hp_max=1000, mana_current=500, mana_max=500)
        ctx = AgentContext()
        ctx.threat.imminent_threat = True
        ctx.threat.imminent_threat_con = "red"
        assert _score_feign_death(state, ctx) == 1.0

    @patch(
        "brain.rules.survival.get_spell_by_role",
        return_value=Spell("Feign Death", gem=1, cast_time=1.0, mana_cost=10),
    )
    def test_no_trigger_returns_zero(self, _mock: object) -> None:
        state = make_game_state(hp_current=800, hp_max=1000, mana_current=500, mana_max=500)
        ctx = AgentContext()
        assert _score_feign_death(state, ctx) == 0.0


# ---------------------------------------------------------------------------
# _score_flee
# ---------------------------------------------------------------------------


class TestScoreFlee:
    def test_flee_disabled_returns_zero(self) -> None:
        flags.flee = False
        state = make_game_state()
        ctx = AgentContext()
        assert _score_flee(state, ctx) == 0.0

    def test_returns_urgency(self) -> None:
        state = make_game_state(hp_current=200, hp_max=1000)
        ctx = AgentContext()
        ctx.pet.alive = True
        score = _score_flee(state, ctx)
        assert score > 0.0

    def test_subthreshold_urgency_returns_zero(self) -> None:
        state = make_game_state(hp_current=990, hp_max=1000)
        ctx = AgentContext()
        ctx.pet.alive = True

        assert _score_flee(state, ctx) == 0.0

    def test_safety_floor_returns_full_score(self) -> None:
        attacker = make_spawn(target_name="TestPlayer", x=10.0, y=0.0)
        state = make_game_state(spawns=(attacker,))
        ctx = AgentContext()
        ctx.pet.alive = False

        assert _score_flee(state, ctx) == 1.0


# ---------------------------------------------------------------------------
# _score_rest
# ---------------------------------------------------------------------------


class TestScoreRest:
    def test_rest_disabled_returns_zero(self) -> None:
        flags.rest = False
        state = make_game_state()
        ctx = AgentContext()
        assert _score_rest(state, ctx) == 0.0

    def test_in_active_combat_returns_zero(self) -> None:
        state = make_game_state(hp_current=300, hp_max=1000)
        ctx = AgentContext()
        ctx.combat.engaged = True
        assert _score_rest(state, ctx) == 0.0

    def test_hp_low_returns_positive(self) -> None:
        state = make_game_state(hp_current=500, hp_max=1000, mana_current=500, mana_max=500)
        ctx = AgentContext()
        ctx.rest_hp_entry = 0.85
        ctx.rest_hp_threshold = 0.92
        ctx.rest_mana_entry = 0.40
        ctx.rest_mana_threshold = 0.60
        score = _score_rest(state, ctx)
        assert score > 0.0

    def test_mana_low_returns_positive(self) -> None:
        state = make_game_state(hp_current=950, hp_max=1000, mana_current=100, mana_max=500)
        ctx = AgentContext()
        ctx.rest_hp_entry = 0.85
        ctx.rest_hp_threshold = 0.92
        ctx.rest_mana_entry = 0.40
        ctx.rest_mana_threshold = 0.60
        score = _score_rest(state, ctx)
        assert score > 0.0

    def test_zero_mana_max(self) -> None:
        state = make_game_state(hp_current=950, hp_max=1000, mana_current=0, mana_max=0)
        ctx = AgentContext()
        ctx.rest_hp_entry = 0.85
        ctx.rest_hp_threshold = 0.92
        ctx.rest_mana_entry = 0.40
        ctx.rest_mana_threshold = 0.60
        score = _score_rest(state, ctx)
        # mana_score should be 0, hp is healthy so overall should be ~0
        assert score >= 0.0


# ---------------------------------------------------------------------------
# _score_evade
# ---------------------------------------------------------------------------


class TestScoreEvade:
    def test_engaged_returns_zero(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.engaged = True
        ctx.threat.evasion_point = Point(100.0, 200.0, 0.0)
        assert _score_evade(state, ctx) == 0.0

    def test_evasion_point_set(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.threat.evasion_point = Point(100.0, 200.0, 0.0)
        assert _score_evade(state, ctx) == 1.0

    def test_no_evasion_point(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        assert _score_evade(state, ctx) == 0.0


# ---------------------------------------------------------------------------
# _next_pull_mana_estimate
# ---------------------------------------------------------------------------


class TestNextPullManaEstimate:
    def test_no_fight_history(self) -> None:
        ctx = AgentContext()
        ctx.fight_history = None
        assert _next_pull_mana_estimate(ctx) is None

    def test_no_learned_data(self) -> None:
        ctx = AgentContext()
        ctx.fight_history = SimpleNamespace(
            get_all_stats=lambda: {"a_skeleton": object()},
            has_learned=lambda k: False,
            learned_mana=lambda k: None,
        )
        assert _next_pull_mana_estimate(ctx) is None

    def test_with_learned_data(self) -> None:
        ctx = AgentContext()
        ctx.fight_history = SimpleNamespace(
            get_all_stats=lambda: {"a_skeleton": object(), "a_bat": object()},
            has_learned=lambda k: True,
            learned_mana=lambda k: 100 if k == "a_skeleton" else 200,
        )
        result = _next_pull_mana_estimate(ctx)
        assert result == 150  # average of 100 and 200

    def test_partial_learned_data(self) -> None:
        """Some mobs learned, some return None for mana."""
        ctx = AgentContext()
        ctx.fight_history = SimpleNamespace(
            get_all_stats=lambda: {"a_skeleton": object(), "a_bat": object()},
            has_learned=lambda k: True,
            learned_mana=lambda k: 100 if k == "a_skeleton" else None,
        )
        result = _next_pull_mana_estimate(ctx)
        assert result == 100


# ---------------------------------------------------------------------------
# _rest_suppressed -- additional edge cases
# ---------------------------------------------------------------------------


class TestRestSuppressed:
    def _make_ctx(self) -> AgentContext:
        ctx = AgentContext()
        ctx.player.last_buff_time = 0.0
        ctx.player.last_flee_time = 0.0
        return ctx

    def test_recently_fled_suppresses(self) -> None:
        state = make_game_state(hp_current=500, hp_max=1000)
        ctx = self._make_ctx()
        ctx.player.last_flee_time = time.time()
        rs = _SurvivalRuleState()
        assert _rest_suppressed(ctx, state, rs) is True

    def test_in_combat_dot_pet_critical_allows_rest(self) -> None:
        """in_combat from DoT but pet critical -> rest allowed (not suppressed here)."""
        state = make_game_state(hp_current=800, hp_max=1000, in_combat=True)
        ctx = self._make_ctx()
        ctx.combat.engaged = False
        ctx.pet.alive = True
        ctx.world = SimpleNamespace(
            pet_hp_pct=0.30,
            any_hostile_npc_within=lambda r: False,
            threats_within=lambda r: [],
        )
        rs = _SurvivalRuleState()
        assert _rest_suppressed(ctx, state, rs) is False

    def test_in_combat_not_engaged_pet_ok_suppresses(self) -> None:
        """in_combat from DoT but pet is fine -> rest suppressed."""
        state = make_game_state(hp_current=800, hp_max=1000, in_combat=True)
        ctx = self._make_ctx()
        ctx.combat.engaged = False
        ctx.pet.alive = True
        ctx.world = SimpleNamespace(pet_hp_pct=0.80)
        rs = _SurvivalRuleState()
        assert _rest_suppressed(ctx, state, rs) is True

    def test_in_combat_not_engaged_no_pet_suppresses(self) -> None:
        """in_combat but no pet -> rest suppressed."""
        state = make_game_state(hp_current=800, hp_max=1000, in_combat=True)
        ctx = self._make_ctx()
        ctx.combat.engaged = False
        ctx.pet.alive = False
        rs = _SurvivalRuleState()
        assert _rest_suppressed(ctx, state, rs) is True

    def test_hp_dropping_while_resting_cancels(self) -> None:
        """HP dropping significantly while resting -> cancel rest."""
        state = make_game_state(hp_current=600, hp_max=1000)
        ctx = self._make_ctx()
        ctx.player.rest_start_time = time.time() - 10.0  # resting for 10s
        ctx.player.last_rest_hp = 0.80  # started resting at 80% HP
        rs = _SurvivalRuleState(resting=True)
        # HP is now 60%, dropped > 10% from 80% -> should cancel
        assert _rest_suppressed(ctx, state, rs) is True
        assert rs.resting is False

    def test_hostile_npc_within_20_suppresses(self) -> None:
        state = make_game_state(hp_current=500, hp_max=1000)
        ctx = self._make_ctx()
        ctx.world = SimpleNamespace(
            any_hostile_npc_within=lambda r: True,
            threats_within=lambda r: [],
        )
        rs = _SurvivalRuleState()
        assert _rest_suppressed(ctx, state, rs) is True

    def test_threats_within_50_suppresses(self) -> None:
        state = make_game_state(hp_current=500, hp_max=1000)
        ctx = self._make_ctx()
        ctx.world = SimpleNamespace(
            any_hostile_npc_within=lambda r: False,
            threats_within=lambda r: [object()] if r == 50 else [],
        )
        rs = _SurvivalRuleState()
        assert _rest_suppressed(ctx, state, rs) is True

    def test_engaged_suppresses(self) -> None:
        state = make_game_state(hp_current=500, hp_max=1000)
        ctx = self._make_ctx()
        ctx.combat.engaged = True
        rs = _SurvivalRuleState()
        assert _rest_suppressed(ctx, state, rs) is True

    def test_target_is_damaged_npc_suppresses(self) -> None:
        """Targeting a damaged NPC (not our pet) suppresses rest."""
        target = make_spawn(spawn_id=300, hp_current=80, hp_max=100)
        state = make_game_state(hp_current=500, hp_max=1000, target=target)
        ctx = self._make_ctx()
        ctx.pet.spawn_id = 999  # different from target
        rs = _SurvivalRuleState()
        assert _rest_suppressed(ctx, state, rs) is True


# ---------------------------------------------------------------------------
# _rest_exit_check -- learned mana threshold
# ---------------------------------------------------------------------------


class TestRestExitCheck:
    def test_thresholds_met_exits(self) -> None:
        state = make_game_state(hp_current=950, hp_max=1000, mana_current=400, mana_max=500)
        ctx = AgentContext()
        ctx.rest_hp_threshold = 0.92
        ctx.rest_mana_threshold = 0.60
        rs = _SurvivalRuleState(resting=True)
        assert _rest_exit_check(ctx, state, rs) is True
        assert rs.resting is False

    def test_thresholds_not_met(self) -> None:
        state = make_game_state(hp_current=500, hp_max=1000, mana_current=100, mana_max=500)
        ctx = AgentContext()
        ctx.rest_hp_threshold = 0.92
        ctx.rest_mana_threshold = 0.60
        rs = _SurvivalRuleState(resting=True)
        assert _rest_exit_check(ctx, state, rs) is False

    def test_learned_mana_lowers_threshold(self) -> None:
        """Learned mana cost -> exit rest earlier."""
        # Mana at 50%, default threshold 60%, but learned cost is low
        state = make_game_state(hp_current=950, hp_max=1000, mana_current=250, mana_max=500)
        ctx = AgentContext()
        ctx.rest_hp_threshold = 0.92
        ctx.rest_mana_threshold = 0.60
        # Learned cost: 100 mana. needed_pct = (100 * 1.2) / 500 = 0.24
        # Clamped to max(0.40, min(0.24, 0.60)) = 0.40
        # mana_pct = 0.50 >= 0.40 -> exits rest
        ctx.fight_history = SimpleNamespace(
            get_all_stats=lambda: {"a_skeleton": object()},
            has_learned=lambda k: True,
            learned_mana=lambda k: 100,
        )
        rs = _SurvivalRuleState(resting=True)
        assert _rest_exit_check(ctx, state, rs) is True

    def test_pet_hp_low_prevents_exit(self) -> None:
        """Pet HP below 90% prevents rest exit even if HP/mana are ok."""
        state = make_game_state(hp_current=950, hp_max=1000, mana_current=400, mana_max=500)
        ctx = AgentContext()
        ctx.rest_hp_threshold = 0.92
        ctx.rest_mana_threshold = 0.60
        ctx.pet.alive = True
        ctx.world = SimpleNamespace(pet_hp_pct=0.50)
        rs = _SurvivalRuleState(resting=True)
        assert _rest_exit_check(ctx, state, rs) is False

    def test_pet_hp_unknown_allows_exit(self) -> None:
        """pet_hp_pct < 0 (unknown) -> treated as ok."""
        state = make_game_state(hp_current=950, hp_max=1000, mana_current=400, mana_max=500)
        ctx = AgentContext()
        ctx.rest_hp_threshold = 0.92
        ctx.rest_mana_threshold = 0.60
        ctx.pet.alive = True
        ctx.world = SimpleNamespace(pet_hp_pct=-1.0)
        rs = _SurvivalRuleState(resting=True)
        assert _rest_exit_check(ctx, state, rs) is True

    def test_zero_mana_max_treats_mana_ok(self) -> None:
        state = make_game_state(hp_current=950, hp_max=1000, mana_current=0, mana_max=0)
        ctx = AgentContext()
        ctx.rest_hp_threshold = 0.92
        ctx.rest_mana_threshold = 0.60
        rs = _SurvivalRuleState(resting=True)
        assert _rest_exit_check(ctx, state, rs) is True


# ---------------------------------------------------------------------------
# _should_rest -- pet low entry + pet_low in _should_rest
# ---------------------------------------------------------------------------


class TestShouldRestPetLow:
    def test_pet_low_triggers_rest(self) -> None:
        state = make_game_state(hp_current=950, hp_max=1000, mana_current=400, mana_max=500)
        ctx = AgentContext()
        ctx.rest_hp_entry = 0.85
        ctx.rest_mana_entry = 0.40
        ctx.rest_hp_threshold = 0.99  # never exit
        ctx.rest_mana_threshold = 0.99
        ctx.player.last_buff_time = 0.0
        ctx.player.last_flee_time = 0.0
        ctx.pet.alive = True
        ctx.world = SimpleNamespace(
            pet_hp_pct=0.40,
            any_hostile_npc_within=lambda r: False,
            threats_within=lambda r: [],
        )
        rs = _SurvivalRuleState()
        result = _should_rest(state, ctx, rs)
        assert result is True
        assert rs.resting is True


# ---------------------------------------------------------------------------
# flee_condition -- additional safety floor branches
# ---------------------------------------------------------------------------


class TestFleeConditionSafetyFloors:
    def test_3_damaged_npcs_train_floor(self) -> None:
        """3+ damaged NPCs (train) triggers flee safety floor."""
        npc1 = make_spawn(spawn_id=200, x=5.0, y=5.0, hp_current=80, hp_max=100)
        npc2 = make_spawn(spawn_id=201, x=10.0, y=10.0, hp_current=60, hp_max=100)
        npc3 = make_spawn(spawn_id=202, x=15.0, y=15.0, hp_current=40, hp_max=100)
        state = make_game_state(hp_current=900, hp_max=1000, spawns=(npc1, npc2, npc3))
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.combat.engaged = True
        result = flee_condition(ctx, state)
        assert result is True
        assert ctx.combat.flee_urgency_active is True

    def test_2_damaged_npcs_pet_dead_floor(self) -> None:
        """2 damaged NPCs + pet dead triggers flee."""
        npc1 = make_spawn(spawn_id=200, x=5.0, y=5.0, hp_current=80, hp_max=100)
        npc2 = make_spawn(spawn_id=201, x=10.0, y=10.0, hp_current=60, hp_max=100)
        state = make_game_state(hp_current=900, hp_max=1000, spawns=(npc1, npc2))
        ctx = AgentContext()
        ctx.pet.alive = False
        ctx.combat.engaged = True
        result = flee_condition(ctx, state)
        assert result is True

    def test_2_damaged_npcs_pet_hp_low_floor(self) -> None:
        """2 damaged NPCs + pet HP < 70% triggers flee."""
        npc1 = make_spawn(spawn_id=200, x=5.0, y=5.0, hp_current=80, hp_max=100)
        npc2 = make_spawn(spawn_id=201, x=10.0, y=10.0, hp_current=60, hp_max=100)
        state = make_game_state(hp_current=900, hp_max=1000, spawns=(npc1, npc2))
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.combat.engaged = True
        ctx.world = SimpleNamespace(
            pet_hp_pct=0.50,
            damaged_npcs_near=lambda pos, r: [npc1, npc2],
        )
        result = flee_condition(ctx, state)
        assert result is True

    def test_2_damaged_npcs_pet_healthy_no_floor(self) -> None:
        """2 damaged NPCs + pet HP >= 70% -> no extra floor."""
        npc1 = make_spawn(spawn_id=200, x=5.0, y=5.0, hp_current=80, hp_max=100)
        npc2 = make_spawn(spawn_id=201, x=10.0, y=10.0, hp_current=60, hp_max=100)
        state = make_game_state(hp_current=900, hp_max=1000, spawns=(npc1, npc2))
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.combat.engaged = True
        ctx.world = SimpleNamespace(
            pet_hp_pct=0.90,
            damaged_npcs_near=lambda pos, r: [npc1, npc2],
        )
        # With healthy HP and pet, urgency should be low
        result = flee_condition(ctx, state)
        assert result is False

    def test_npc_attacking_during_pull(self) -> None:
        """NPC attacking player during pull (pull_target_id set) triggers flee."""
        attacker = make_spawn(
            spawn_id=200,
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
            target_name="TestPlayer",
        )
        state = make_game_state(hp_current=800, hp_max=1000, spawns=(attacker,))
        ctx = AgentContext()
        ctx.pet.alive = False
        ctx.combat.engaged = False
        ctx.combat.pull_target_id = 300  # mid-pull
        result = flee_condition(ctx, state)
        assert result is True

    def test_reset_flee_hysteresis(self) -> None:
        ctx = AgentContext()
        ctx.combat.flee_urgency_active = True
        reset_flee_hysteresis(ctx)
        assert ctx.combat.flee_urgency_active is False


# ---------------------------------------------------------------------------
# register() -- smoke test
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_adds_rules(self) -> None:
        """Verify register() adds all 5 survival rules to the brain."""

        rules_added: list[str] = []

        class FakeBrain:
            def add_rule(self, name: str, *args: object, **kwargs: object) -> None:
                rules_added.append(name)

        brain = FakeBrain()
        ctx = AgentContext()

        def read_state_fn() -> GameState:
            return make_game_state()

        register(brain, ctx, read_state_fn)

        assert "DEATH_RECOVERY" in rules_added
        assert "FEIGN_DEATH" in rules_added
        assert "FLEE" in rules_added
        assert "REST" in rules_added
        assert "EVADE" in rules_added
        assert len(rules_added) == 5
