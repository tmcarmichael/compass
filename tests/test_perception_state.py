"""Tests for perception.state and perception.queries.

Covers all properties and computed fields of SpawnData, GameState, and
DoorData, plus the query functions: _looks_like_pet_name, is_pet,
is_our_pet, live_npcs, nearby_live_npcs, count_nearby_npcs,
count_nearby_social.
"""

from __future__ import annotations

import pytest

from core.constants import XP_SCALE_MAX
from core.types import Point
from perception.queries import (
    _looks_like_pet_name,
    count_nearby_npcs,
    count_nearby_social,
    is_our_pet,
    is_pet,
    live_npcs,
    nearby_live_npcs,
)
from perception.state import DoorData, GameState, SpawnData
from tests.factories import make_game_state, make_spawn

# ===================================================================
# SpawnData properties
# ===================================================================


class TestSpawnDataTypeProperties:
    """spawn_type -> is_npc, is_player, is_corpse, is_player_corpse."""

    @pytest.mark.parametrize(
        "spawn_type, expected",
        [(0, False), (1, True), (2, False), (3, False)],
    )
    def test_is_npc(self, spawn_type: int, expected: bool) -> None:
        assert make_spawn(spawn_type=spawn_type).is_npc is expected

    @pytest.mark.parametrize(
        "spawn_type, expected",
        [(0, True), (1, False), (2, False), (3, False)],
    )
    def test_is_player(self, spawn_type: int, expected: bool) -> None:
        assert make_spawn(spawn_type=spawn_type).is_player is expected

    @pytest.mark.parametrize(
        "spawn_type, expected",
        [(0, False), (1, False), (2, True), (3, True)],
    )
    def test_is_corpse(self, spawn_type: int, expected: bool) -> None:
        assert make_spawn(spawn_type=spawn_type).is_corpse is expected

    @pytest.mark.parametrize(
        "spawn_type, expected",
        [(0, False), (1, False), (2, False), (3, True)],
    )
    def test_is_player_corpse(self, spawn_type: int, expected: bool) -> None:
        assert make_spawn(spawn_type=spawn_type).is_player_corpse is expected


class TestSpawnDataOwnership:
    def test_is_owned_when_owner_set(self) -> None:
        assert make_spawn(owner_id=42).is_owned is True

    def test_not_owned_default(self) -> None:
        assert make_spawn(owner_id=0).is_owned is False


class TestSpawnDataBodyState:
    @pytest.mark.parametrize(
        "body_state, prop, expected",
        [
            ("f", "is_feigning", True),
            ("n", "is_feigning", False),
            ("d", "is_dead_body", True),
            ("n", "is_dead_body", False),
            ("i", "is_invisible", True),
            ("n", "is_invisible", False),
            # Cross-checks: each flag is exclusive for normal state
            ("d", "is_feigning", False),
            ("d", "is_invisible", False),
            ("f", "is_dead_body", False),
            ("f", "is_invisible", False),
            ("i", "is_feigning", False),
            ("i", "is_dead_body", False),
        ],
    )
    def test_body_state_flags(self, body_state: str, prop: str, expected: bool) -> None:
        s = make_spawn(body_state=body_state)
        assert getattr(s, prop) is expected


# ===================================================================
# GameState computed properties
# ===================================================================


class TestGameStateHpPct:
    def test_full_hp(self) -> None:
        assert make_game_state(hp_current=1000, hp_max=1000).hp_pct == pytest.approx(1.0)

    def test_half_hp(self) -> None:
        assert make_game_state(hp_current=500, hp_max=1000).hp_pct == pytest.approx(0.5)

    def test_zero_current(self) -> None:
        assert make_game_state(hp_current=0, hp_max=1000).hp_pct == pytest.approx(0.0)

    def test_hp_max_zero_returns_one(self) -> None:
        """Edge case: hp_max=0 should return 1.0, not crash."""
        assert make_game_state(hp_current=0, hp_max=0).hp_pct == pytest.approx(1.0)


class TestGameStateManaPct:
    def test_full_mana(self) -> None:
        assert make_game_state(mana_current=500, mana_max=500).mana_pct == pytest.approx(1.0)

    def test_half_mana(self) -> None:
        assert make_game_state(mana_current=250, mana_max=500).mana_pct == pytest.approx(0.5)

    def test_mana_max_zero_returns_one(self) -> None:
        assert make_game_state(mana_current=0, mana_max=0).mana_pct == pytest.approx(1.0)


class TestGameStateXpPct:
    def test_zero_raw(self) -> None:
        assert make_game_state(xp_pct_raw=0).xp_pct == pytest.approx(0.0)

    def test_negative_raw(self) -> None:
        assert make_game_state(xp_pct_raw=-10).xp_pct == pytest.approx(0.0)

    def test_midpoint(self) -> None:
        assert make_game_state(xp_pct_raw=165).xp_pct == pytest.approx(0.5)

    def test_max_raw(self) -> None:
        assert make_game_state(xp_pct_raw=XP_SCALE_MAX).xp_pct == pytest.approx(1.0)

    def test_above_max_clamps(self) -> None:
        assert make_game_state(xp_pct_raw=XP_SCALE_MAX + 100).xp_pct == pytest.approx(1.0)


class TestGameStateTarget:
    def test_has_target_with_spawn(self) -> None:
        assert make_game_state(target=make_spawn()).has_target is True

    def test_has_target_none(self) -> None:
        assert make_game_state(target=None).has_target is False


class TestGameStateSitStand:
    @pytest.mark.parametrize(
        "stand_state, sitting, medding, standing",
        [
            (0, False, False, True),  # standing
            (1, True, False, False),  # sitting
            (2, True, True, False),  # medding (also counts as sitting)
            (3, False, False, True),  # other
        ],
    )
    def test_sit_stand_states(self, stand_state: int, sitting: bool, medding: bool, standing: bool) -> None:
        s = make_game_state(stand_state=stand_state)
        assert s.is_sitting is sitting
        assert s.is_medding is medding
        assert s.is_standing is standing


class TestGameStateCasting:
    def test_casting_mode_1(self) -> None:
        assert make_game_state(casting_mode=1).is_casting is True

    @pytest.mark.parametrize("mode", [0, 2, 3])
    def test_not_casting(self, mode: int) -> None:
        assert make_game_state(casting_mode=mode).is_casting is False


class TestGameStateGameMode:
    @pytest.mark.parametrize(
        "game_mode, in_game, zoning, char_select",
        [
            (0, False, False, True),  # char select
            (3, False, True, False),  # zoning
            (4, False, True, False),  # loading
            (5, True, False, False),  # in game
            (253, False, True, False),  # pre-ingame
        ],
    )
    def test_game_modes(self, game_mode: int, in_game: bool, zoning: bool, char_select: bool) -> None:
        s = make_game_state(game_mode=game_mode)
        assert s.is_in_game is in_game
        assert s.is_zoning is zoning
        assert s.is_at_char_select is char_select


class TestGameStateMoney:
    def test_all_zeroes(self) -> None:
        assert make_game_state(money_pp=0, money_gp=0, money_sp=0, money_cp=0).money_total_cp == 0

    def test_mixed(self) -> None:
        s = make_game_state(money_pp=2, money_gp=3, money_sp=4, money_cp=5)
        assert s.money_total_cp == 2 * 1000 + 3 * 100 + 4 * 10 + 5

    def test_only_copper(self) -> None:
        assert make_game_state(money_cp=99).money_total_cp == 99


class TestGameStateDead:
    def test_dead(self) -> None:
        assert make_game_state(body_state="d").is_dead is True

    def test_not_dead(self) -> None:
        assert make_game_state(body_state="n").is_dead is False


class TestGameStateBuffs:
    def test_has_buff_present(self) -> None:
        s = make_game_state(buffs=((100, 5), (200, 10)))
        assert s.has_buff(100) is True
        assert s.has_buff(200) is True

    def test_has_buff_absent(self) -> None:
        s = make_game_state(buffs=((100, 5),))
        assert s.has_buff(999) is False

    def test_has_buff_empty(self) -> None:
        assert make_game_state(buffs=()).has_buff(1) is False

    def test_buff_ticks_present(self) -> None:
        s = make_game_state(buffs=((100, 7),))
        assert s.buff_ticks(100) == 7

    def test_buff_ticks_absent(self) -> None:
        s = make_game_state(buffs=((100, 7),))
        assert s.buff_ticks(999) == 0

    def test_buff_ticks_empty(self) -> None:
        assert make_game_state(buffs=()).buff_ticks(1) == 0


class TestGameStateNearbyFilters:
    def test_nearby_npcs(self) -> None:
        npc = make_spawn(spawn_id=1, spawn_type=1)
        player = make_spawn(spawn_id=2, spawn_type=0)
        corpse = make_spawn(spawn_id=3, spawn_type=2)
        s = make_game_state(spawns=(npc, player, corpse))
        assert len(s.nearby_npcs) == 1
        assert s.nearby_npcs[0].spawn_id == 1

    def test_nearby_players(self) -> None:
        npc = make_spawn(spawn_id=1, spawn_type=1)
        p1 = make_spawn(spawn_id=2, spawn_type=0)
        p2 = make_spawn(spawn_id=3, spawn_type=0)
        s = make_game_state(spawns=(npc, p1, p2))
        assert len(s.nearby_players) == 2

    def test_empty_spawns(self) -> None:
        s = make_game_state(spawns=())
        assert s.nearby_npcs == ()
        assert s.nearby_players == ()


# ===================================================================
# DoorData creation
# ===================================================================


class TestDoorData:
    def test_create_with_defaults(self) -> None:
        d = DoorData(door_id=1, name="GATE01", x=10.0, y=20.0, z=0.0, heading=128.0, is_open=True)
        assert d.door_id == 1
        assert d.name == "GATE01"
        assert d.is_open is True
        assert d.width == 10.0  # default

    def test_custom_width(self) -> None:
        d = DoorData(door_id=2, name="BIGDOOR", x=0.0, y=0.0, z=0.0, heading=0.0, is_open=False, width=25.0)
        assert d.width == 25.0
        assert d.is_open is False

    def test_frozen(self) -> None:
        d = DoorData(door_id=1, name="D", x=0.0, y=0.0, z=0.0, heading=0.0, is_open=True)
        with pytest.raises(AttributeError):
            d.is_open = False  # type: ignore[misc]


# ===================================================================
# perception.queries  -- _looks_like_pet_name
# ===================================================================


class TestLooksLikePetName:
    @pytest.mark.parametrize(
        "name",
        [
            "Garann000",  # classic pet name pattern
            "Xabcde123",  # exactly 8 chars, ends 3 digits
            "Jabober456",
            "Magefire789",
            "GARANN000",  # all caps still matches (isalpha + first upper)
        ],
    )
    def test_pet_names_match(self, name: str) -> None:
        assert _looks_like_pet_name(name) is True

    @pytest.mark.parametrize(
        "name, reason",
        [
            ("a_skeleton", "underscore NPC name"),
            ("a_fire_beetle", "underscore NPC name"),
            ("Ab123", "too short (< 7 chars)"),
            ("Ab1234", "too short (exactly 6 chars)"),
            ("garann000", "starts lowercase"),
            ("Gara_n000", "contains underscore"),
            ("Garannxyz", "last 3 not digits"),
        ],
    )
    def test_non_pet_names(self, name: str, reason: str) -> None:
        assert _looks_like_pet_name(name) is False, reason

    def test_minimum_length_boundary(self) -> None:
        # 7 chars: prefix=4 alpha + 3 digits -> passes if prefix[0].isupper()
        assert _looks_like_pet_name("Abcd123") is True
        # 6 chars -> too short
        assert _looks_like_pet_name("Abc123") is False


# ===================================================================
# perception.queries  -- is_pet / is_our_pet
# ===================================================================


class TestIsPet:
    def test_owner_id_nonzero_is_pet(self) -> None:
        """owner_id != 0 is definitive, regardless of name."""
        s = make_spawn(owner_id=42, name="a_skeleton", spawn_type=1)
        assert is_pet(s) is True

    def test_pet_name_heuristic(self) -> None:
        """No owner_id but pet-like name -> pet."""
        s = make_spawn(owner_id=0, name="Garann000", spawn_type=1)
        assert is_pet(s) is True

    def test_regular_npc_not_pet(self) -> None:
        s = make_spawn(owner_id=0, name="a_skeleton", spawn_type=1)
        assert is_pet(s) is False

    def test_player_never_pet(self) -> None:
        """Players (spawn_type=0) are never pets, even with owner_id."""
        s = make_spawn(owner_id=42, spawn_type=0)
        assert is_pet(s) is False

    def test_corpse_never_pet(self) -> None:
        """Corpses are not NPCs so is_pet returns False."""
        s = make_spawn(owner_id=42, spawn_type=2)
        assert is_pet(s) is False


class TestIsOurPet:
    def test_our_pet_by_owner_id(self) -> None:
        s = make_spawn(owner_id=1, spawn_type=1)
        assert is_our_pet(s, player_spawn_id=1) is True

    def test_not_our_pet_different_owner(self) -> None:
        s = make_spawn(owner_id=99, spawn_type=1)
        assert is_our_pet(s, player_spawn_id=1) is False

    def test_fallback_to_heuristic_when_owner_zero(self) -> None:
        """When owner_id=0, falls back to name heuristic (can't distinguish)."""
        pet = make_spawn(owner_id=0, name="Garann000", spawn_type=1)
        assert is_our_pet(pet, player_spawn_id=1) is True

    def test_non_pet_fallback(self) -> None:
        npc = make_spawn(owner_id=0, name="a_skeleton", spawn_type=1)
        assert is_our_pet(npc, player_spawn_id=1) is False


# ===================================================================
# perception.queries  -- live_npcs
# ===================================================================


class TestLiveNpcs:
    def _state_with(self, *spawns: SpawnData) -> GameState:
        return make_game_state(spawns=spawns)

    def test_basic_npc(self) -> None:
        npc = make_spawn(spawn_id=1, spawn_type=1, hp_current=100)
        result = list(live_npcs(self._state_with(npc)))
        assert len(result) == 1

    def test_excludes_corpses(self) -> None:
        corpse = make_spawn(spawn_id=1, spawn_type=2, hp_current=0)
        assert list(live_npcs(self._state_with(corpse))) == []

    def test_excludes_players(self) -> None:
        player = make_spawn(spawn_id=1, spawn_type=0, hp_current=100)
        assert list(live_npcs(self._state_with(player))) == []

    def test_excludes_dead_npc(self) -> None:
        dead = make_spawn(spawn_id=1, spawn_type=1, hp_current=0)
        assert list(live_npcs(self._state_with(dead))) == []

    def test_excludes_feigning(self) -> None:
        feign = make_spawn(spawn_id=1, spawn_type=1, hp_current=100, body_state="f")
        assert list(live_npcs(self._state_with(feign))) == []

    def test_excludes_dead_body_state(self) -> None:
        dead = make_spawn(spawn_id=1, spawn_type=1, hp_current=100, body_state="d")
        assert list(live_npcs(self._state_with(dead))) == []

    def test_excludes_pets_by_default(self) -> None:
        pet = make_spawn(spawn_id=1, spawn_type=1, hp_current=100, owner_id=42)
        assert list(live_npcs(self._state_with(pet))) == []

    def test_includes_pets_when_not_excluded(self) -> None:
        pet = make_spawn(spawn_id=1, spawn_type=1, hp_current=100, owner_id=42)
        result = list(live_npcs(self._state_with(pet), exclude_pets=False))
        assert len(result) == 1

    def test_mixed_spawns(self) -> None:
        npc = make_spawn(spawn_id=1, spawn_type=1, hp_current=100)
        corpse = make_spawn(spawn_id=2, spawn_type=2, hp_current=0)
        player = make_spawn(spawn_id=3, spawn_type=0, hp_current=100)
        dead_npc = make_spawn(spawn_id=4, spawn_type=1, hp_current=0)
        pet = make_spawn(spawn_id=5, spawn_type=1, hp_current=100, owner_id=10)
        result = list(live_npcs(self._state_with(npc, corpse, player, dead_npc, pet)))
        assert len(result) == 1
        assert result[0].spawn_id == 1


# ===================================================================
# perception.queries  -- nearby_live_npcs
# ===================================================================


class TestNearbyLiveNpcs:
    def test_within_radius(self) -> None:
        npc = make_spawn(spawn_id=1, x=10.0, y=0.0, spawn_type=1, hp_current=100)
        state = make_game_state(spawns=(npc,))
        result = nearby_live_npcs(state, Point(0.0, 0.0, 0.0), radius=20.0)
        assert len(result) == 1
        assert result[0][0].spawn_id == 1
        assert result[0][1] == pytest.approx(10.0)

    def test_outside_radius(self) -> None:
        npc = make_spawn(spawn_id=1, x=100.0, y=0.0, spawn_type=1, hp_current=100)
        state = make_game_state(spawns=(npc,))
        result = nearby_live_npcs(state, Point(0.0, 0.0, 0.0), radius=20.0)
        assert result == []

    def test_sorted_by_distance(self) -> None:
        far = make_spawn(spawn_id=1, x=30.0, y=0.0, spawn_type=1, hp_current=100)
        close = make_spawn(spawn_id=2, x=5.0, y=0.0, spawn_type=1, hp_current=100)
        mid = make_spawn(spawn_id=3, x=15.0, y=0.0, spawn_type=1, hp_current=100)
        state = make_game_state(spawns=(far, close, mid))
        result = nearby_live_npcs(state, Point(0.0, 0.0, 0.0), radius=50.0)
        ids = [r[0].spawn_id for r in result]
        assert ids == [2, 3, 1]

    def test_excludes_pets_by_default(self) -> None:
        pet = make_spawn(spawn_id=1, x=5.0, y=0.0, spawn_type=1, hp_current=100, owner_id=42)
        state = make_game_state(spawns=(pet,))
        assert nearby_live_npcs(state, Point(0.0, 0.0, 0.0), radius=50.0) == []

    def test_includes_pets_when_requested(self) -> None:
        pet = make_spawn(spawn_id=1, x=5.0, y=0.0, spawn_type=1, hp_current=100, owner_id=42)
        state = make_game_state(spawns=(pet,))
        result = nearby_live_npcs(state, Point(0.0, 0.0, 0.0), radius=50.0, exclude_pets=False)
        assert len(result) == 1


# ===================================================================
# perception.queries  -- count_nearby_npcs
# ===================================================================


class TestCountNearbyNpcs:
    def test_counts_in_range(self) -> None:
        n1 = make_spawn(spawn_id=1, x=5.0, y=0.0, spawn_type=1, hp_current=100)
        n2 = make_spawn(spawn_id=2, x=10.0, y=0.0, spawn_type=1, hp_current=100)
        state = make_game_state(spawns=(n1, n2))
        assert count_nearby_npcs(state, Point(0.0, 0.0, 0.0), radius=20.0) == 2

    def test_excludes_out_of_range(self) -> None:
        near = make_spawn(spawn_id=1, x=5.0, y=0.0, spawn_type=1, hp_current=100)
        far = make_spawn(spawn_id=2, x=100.0, y=0.0, spawn_type=1, hp_current=100)
        state = make_game_state(spawns=(near, far))
        assert count_nearby_npcs(state, Point(0.0, 0.0, 0.0), radius=20.0) == 1

    def test_exclude_id(self) -> None:
        n1 = make_spawn(spawn_id=1, x=5.0, y=0.0, spawn_type=1, hp_current=100)
        n2 = make_spawn(spawn_id=2, x=10.0, y=0.0, spawn_type=1, hp_current=100)
        state = make_game_state(spawns=(n1, n2))
        assert count_nearby_npcs(state, Point(0.0, 0.0, 0.0), radius=20.0, exclude_id=1) == 1

    def test_excludes_pets_by_default(self) -> None:
        pet = make_spawn(spawn_id=1, x=5.0, y=0.0, spawn_type=1, hp_current=100, owner_id=42)
        state = make_game_state(spawns=(pet,))
        assert count_nearby_npcs(state, Point(0.0, 0.0, 0.0), radius=20.0) == 0

    def test_empty(self) -> None:
        state = make_game_state(spawns=())
        assert count_nearby_npcs(state, Point(0.0, 0.0, 0.0), radius=100.0) == 0

    def test_boundary_uses_strict_less_than(self) -> None:
        """NPC at exactly the radius boundary: d < radius is strict."""
        npc = make_spawn(spawn_id=1, x=20.0, y=0.0, spawn_type=1, hp_current=100)
        state = make_game_state(spawns=(npc,))
        # Exactly at radius=20 -> d=20.0, condition is d < radius -> False
        assert count_nearby_npcs(state, Point(0.0, 0.0, 0.0), radius=20.0) == 0
        # Just outside -> still 0
        assert count_nearby_npcs(state, Point(0.0, 0.0, 0.0), radius=20.01) == 1


# ===================================================================
# perception.queries  -- count_nearby_social
# ===================================================================


class TestCountNearbySocial:
    def test_empty_social_groups(self) -> None:
        target = make_spawn(spawn_id=1, name="a_skeleton01", x=0.0, y=0.0)
        state = make_game_state(spawns=(target,))
        assert count_nearby_social(target, state, social_groups={}) == 0

    def test_target_not_in_any_group(self) -> None:
        target = make_spawn(spawn_id=1, name="a_bat", x=0.0, y=0.0, spawn_type=1, hp_current=100)
        ally = make_spawn(spawn_id=2, name="a_skeleton", x=5.0, y=0.0, spawn_type=1, hp_current=100)
        state = make_game_state(spawns=(target, ally))
        groups: dict[str, frozenset[str]] = {"a_skeleton": frozenset({"a_skeleton", "a_zombie"})}
        assert count_nearby_social(target, state, groups, radius=50.0) == 0

    def test_counts_group_members_near_target(self) -> None:
        target = make_spawn(spawn_id=1, name="a_skeleton01", x=0.0, y=0.0, spawn_type=1, hp_current=100)
        ally = make_spawn(spawn_id=2, name="a_skeleton", x=10.0, y=0.0, spawn_type=1, hp_current=100)
        state = make_game_state(spawns=(target, ally))
        groups: dict[str, frozenset[str]] = {"a_skeleton": frozenset({"a_skeleton", "a_zombie"})}
        assert count_nearby_social(target, state, groups, radius=50.0) == 1

    def test_excludes_target_itself(self) -> None:
        """The target NPC should not count itself."""
        target = make_spawn(spawn_id=1, name="a_skeleton", x=0.0, y=0.0, spawn_type=1, hp_current=100)
        state = make_game_state(spawns=(target,))
        groups: dict[str, frozenset[str]] = {"a_skeleton": frozenset({"a_skeleton"})}
        assert count_nearby_social(target, state, groups, radius=50.0) == 0

    def test_excludes_out_of_radius(self) -> None:
        target = make_spawn(spawn_id=1, name="a_skeleton", x=0.0, y=0.0, spawn_type=1, hp_current=100)
        far_ally = make_spawn(spawn_id=2, name="a_skeleton", x=200.0, y=0.0, spawn_type=1, hp_current=100)
        state = make_game_state(spawns=(target, far_ally))
        groups: dict[str, frozenset[str]] = {"a_skeleton": frozenset({"a_skeleton"})}
        assert count_nearby_social(target, state, groups, radius=50.0) == 0

    def test_multiple_group_members(self) -> None:
        target = make_spawn(spawn_id=1, name="a_skeleton", x=0.0, y=0.0, spawn_type=1, hp_current=100)
        a1 = make_spawn(spawn_id=2, name="a_zombie", x=5.0, y=0.0, spawn_type=1, hp_current=100)
        a2 = make_spawn(spawn_id=3, name="a_skeleton", x=10.0, y=0.0, spawn_type=1, hp_current=100)
        state = make_game_state(spawns=(target, a1, a2))
        groups: dict[str, frozenset[str]] = {"a_skeleton": frozenset({"a_skeleton", "a_zombie"})}
        assert count_nearby_social(target, state, groups, radius=50.0) == 2

    def test_name_digit_stripping(self) -> None:
        """Target name a_skeleton01 -> base a_skeleton, looked up in groups."""
        target = make_spawn(spawn_id=1, name="a_skeleton01", x=0.0, y=0.0, spawn_type=1, hp_current=100)
        ally = make_spawn(spawn_id=2, name="a_skeleton02", x=5.0, y=0.0, spawn_type=1, hp_current=100)
        state = make_game_state(spawns=(target, ally))
        groups: dict[str, frozenset[str]] = {"a_skeleton": frozenset({"a_skeleton"})}
        assert count_nearby_social(target, state, groups, radius=50.0) == 1

    def test_does_not_exclude_pets(self) -> None:
        """count_nearby_social calls live_npcs(exclude_pets=False), so pets count."""
        target = make_spawn(spawn_id=1, name="a_skeleton", x=0.0, y=0.0, spawn_type=1, hp_current=100)
        pet_ally = make_spawn(
            spawn_id=2,
            name="a_skeleton",
            x=5.0,
            y=0.0,
            spawn_type=1,
            hp_current=100,
            owner_id=99,
        )
        state = make_game_state(spawns=(target, pet_ally))
        groups: dict[str, frozenset[str]] = {"a_skeleton": frozenset({"a_skeleton"})}
        assert count_nearby_social(target, state, groups, radius=50.0) == 1
