"""Tests for perception.combat_eval -- difficulty evaluation and NPC classification.

Covers con_color level-difference formula across all three level brackets,
FIGHTABLE_CONS / THREAT_CONS / AGGRESSIVE_DISPOSITIONS frozensets,
pet detection re-export, is_valid_target, is_threat, avoid-name management,
get_disposition lookups, find_targets sorting, and mob-blocked LOS.
"""

from __future__ import annotations

import pytest

from core.types import Disposition
from perception.combat_eval import (
    AGGRESSIVE_DISPOSITIONS,
    FIGHTABLE_CONS,
    PASSIVE_DISPOSITIONS,
    THREAT_CONS,
    Con,
    check_mob_blocked_los,
    con_color,
    configure_avoid_names,
    find_targets,
    get_avoid_names,
    get_disposition,
    get_zone_avoid_mobs,
    is_threat,
    is_valid_target,
    set_avoid_names,
    set_zone_avoid_mobs,
)
from tests.factories import make_spawn

# ---------------------------------------------------------------------------
# con_color: low-level bracket (player_level < 8)
# ---------------------------------------------------------------------------


class TestConColorLowLevel:
    """Con ranges are tighter at low levels (1-7)."""

    @pytest.mark.parametrize(
        ("player_level", "mob_level", "expected"),
        [
            (5, 8, Con.RED),  # diff +3
            (5, 9, Con.RED),  # diff +4
            (5, 6, Con.YELLOW),  # diff +1
            (5, 7, Con.YELLOW),  # diff +2
            (5, 5, Con.WHITE),  # diff 0
            (5, 3, Con.BLUE),  # diff -2
            (5, 4, Con.BLUE),  # diff -1
            (5, 2, Con.LIGHT_BLUE),  # diff -3
            (5, 1, Con.GREEN),  # diff -4
            (7, 3, Con.GREEN),  # diff -4
        ],
    )
    def test_low_level_cons(self, player_level: int, mob_level: int, expected: Con) -> None:
        assert con_color(player_level, mob_level) == expected


# ---------------------------------------------------------------------------
# con_color: mid-level bracket (8 <= player_level < 20)
# ---------------------------------------------------------------------------


class TestConColorMidLevel:
    @pytest.mark.parametrize(
        ("player_level", "mob_level", "expected"),
        [
            (12, 15, Con.RED),  # diff +3
            (12, 13, Con.YELLOW),  # diff +1
            (12, 14, Con.YELLOW),  # diff +2
            (12, 12, Con.WHITE),  # diff 0
            (12, 9, Con.BLUE),  # diff -3
            (12, 10, Con.BLUE),  # diff -2
            (12, 7, Con.LIGHT_BLUE),  # diff -5
            (12, 6, Con.GREEN),  # diff -6
            (19, 22, Con.RED),  # top of mid bracket
        ],
    )
    def test_mid_level_cons(self, player_level: int, mob_level: int, expected: Con) -> None:
        assert con_color(player_level, mob_level) == expected


# ---------------------------------------------------------------------------
# con_color: high-level bracket (player_level >= 20)
# ---------------------------------------------------------------------------


class TestConColorHighLevel:
    @pytest.mark.parametrize(
        ("player_level", "mob_level", "expected"),
        [
            (30, 33, Con.RED),  # diff +3
            (30, 35, Con.RED),  # diff +5
            (30, 31, Con.YELLOW),  # diff +1
            (30, 32, Con.YELLOW),  # diff +2
            (30, 30, Con.WHITE),  # diff 0
            (30, 25, Con.BLUE),  # diff -5
            (30, 27, Con.BLUE),  # diff -3
            (30, 21, Con.LIGHT_BLUE),  # diff -9
            (30, 20, Con.GREEN),  # diff -10
            (50, 50, Con.WHITE),  # same level
        ],
    )
    def test_high_level_cons(self, player_level: int, mob_level: int, expected: Con) -> None:
        assert con_color(player_level, mob_level) == expected


# ---------------------------------------------------------------------------
# Frozenset constants
# ---------------------------------------------------------------------------


class TestConSets:
    def test_fightable_cons_contents(self) -> None:
        assert Con.WHITE in FIGHTABLE_CONS
        assert Con.BLUE in FIGHTABLE_CONS
        assert Con.LIGHT_BLUE in FIGHTABLE_CONS
        assert Con.YELLOW in FIGHTABLE_CONS
        assert Con.RED not in FIGHTABLE_CONS
        assert Con.GREEN not in FIGHTABLE_CONS

    def test_threat_cons_contents(self) -> None:
        assert Con.YELLOW in THREAT_CONS
        assert Con.RED in THREAT_CONS
        assert Con.WHITE not in THREAT_CONS
        assert Con.BLUE not in THREAT_CONS

    def test_aggressive_dispositions_hostile(self) -> None:
        assert Disposition.SCOWLING in AGGRESSIVE_DISPOSITIONS
        assert Disposition.READY_TO_ATTACK in AGGRESSIVE_DISPOSITIONS
        assert Disposition.THREATENING in AGGRESSIVE_DISPOSITIONS

    def test_aggressive_excludes_passive(self) -> None:
        assert Disposition.INDIFFERENT not in AGGRESSIVE_DISPOSITIONS
        assert Disposition.ALLY not in AGGRESSIVE_DISPOSITIONS

    def test_passive_dispositions_complete(self) -> None:
        assert Disposition.ALLY in PASSIVE_DISPOSITIONS
        assert Disposition.INDIFFERENT in PASSIVE_DISPOSITIONS
        assert Disposition.DUBIOUS in PASSIVE_DISPOSITIONS

    def test_passive_excludes_hostile(self) -> None:
        assert Disposition.SCOWLING not in PASSIVE_DISPOSITIONS
        assert Disposition.READY_TO_ATTACK not in PASSIVE_DISPOSITIONS

    def test_no_overlap_aggressive_passive(self) -> None:
        assert AGGRESSIVE_DISPOSITIONS & PASSIVE_DISPOSITIONS == frozenset()


# ---------------------------------------------------------------------------
# Avoid-name management
# ---------------------------------------------------------------------------


class TestAvoidNames:
    def test_default_avoid_names_include_guard(self) -> None:
        # Reset to defaults
        configure_avoid_names({})
        names = get_avoid_names()
        assert "Guard" in names
        assert "Merchant" in names

    def test_set_avoid_names_replaces(self) -> None:
        original = get_avoid_names()
        try:
            set_avoid_names(frozenset({"TestBoss"}))
            assert get_avoid_names() == frozenset({"TestBoss"})
        finally:
            set_avoid_names(original)

    def test_configure_avoid_names_from_config(self) -> None:
        original = get_avoid_names()
        try:
            configure_avoid_names(
                {"avoid_npcs": {"global_prefixes": ["Knight"], "zone_specific": ["a_big_dragon"]}}
            )
            names = get_avoid_names()
            assert "Knight" in names
            assert "a_big_dragon" in names
        finally:
            set_avoid_names(original)

    def test_zone_avoid_mobs_roundtrip(self) -> None:
        original = get_zone_avoid_mobs()
        try:
            set_zone_avoid_mobs(frozenset({"a_fire_beetle"}))
            assert "a_fire_beetle" in get_zone_avoid_mobs()
        finally:
            set_zone_avoid_mobs(original)


# ---------------------------------------------------------------------------
# get_disposition
# ---------------------------------------------------------------------------


class TestGetDisposition:
    def test_no_zone_dispositions_returns_unknown(self) -> None:
        assert get_disposition("a_skeleton", None) == Disposition.UNKNOWN

    def test_empty_zone_dispositions_returns_unknown(self) -> None:
        assert get_disposition("a_skeleton", {}) == Disposition.UNKNOWN

    def test_matching_prefix(self) -> None:
        zone_disps = {"scowling": ["a_skeleton", "a_bat"]}
        assert get_disposition("a_skeleton001", zone_disps) == Disposition.SCOWLING

    def test_no_matching_prefix(self) -> None:
        zone_disps = {"scowling": ["a_bat"]}
        assert get_disposition("a_skeleton001", zone_disps) == Disposition.UNKNOWN

    def test_invalid_disposition_name_skipped(self) -> None:
        zone_disps = {"bogus_standing": ["a_skeleton"]}
        assert get_disposition("a_skeleton001", zone_disps) == Disposition.UNKNOWN


# ---------------------------------------------------------------------------
# is_valid_target
# ---------------------------------------------------------------------------


class TestIsValidTarget:
    def test_valid_npc_at_fightable_con(self) -> None:
        # Reset avoid names to defaults so "a_skeleton" is not avoided
        original = get_avoid_names()
        try:
            configure_avoid_names({})
            spawn = make_spawn(name="a_skeleton", level=10, hp_current=100, hp_max=100)
            assert is_valid_target(spawn, player_level=10) is True
        finally:
            set_avoid_names(original)

    def test_dead_npc_rejected(self) -> None:
        spawn = make_spawn(name="a_skeleton", level=10, hp_current=0, hp_max=100)
        assert is_valid_target(spawn, player_level=10) is False

    def test_player_rejected(self) -> None:
        spawn = make_spawn(name="SomePlayer", spawn_type=0, level=10, hp_current=100, hp_max=100)
        assert is_valid_target(spawn, player_level=10) is False

    def test_guard_rejected(self) -> None:
        original = get_avoid_names()
        try:
            configure_avoid_names({})
            spawn = make_spawn(name="Guard_Talion", level=10, hp_current=100, hp_max=100)
            assert is_valid_target(spawn, player_level=10) is False
        finally:
            set_avoid_names(original)

    def test_level_zero_rejected(self) -> None:
        spawn = make_spawn(name="a_skeleton", level=0, hp_current=100, hp_max=100)
        assert is_valid_target(spawn, player_level=10) is False


# ---------------------------------------------------------------------------
# is_threat
# ---------------------------------------------------------------------------


class TestIsThreat:
    def test_red_unknown_disposition_is_threat(self) -> None:
        spawn = make_spawn(name="a_dragon", level=40, hp_current=500, hp_max=500)
        assert is_threat(spawn, player_level=10) is True

    def test_blue_unknown_disposition_not_threat(self) -> None:
        spawn = make_spawn(name="a_rat", level=8, hp_current=50, hp_max=50)
        assert is_threat(spawn, player_level=10) is False

    def test_passive_high_level_not_threat(self) -> None:
        zone_disps = {"indifferent": ["a_guard"]}
        spawn = make_spawn(name="a_guard_captain", level=50, hp_current=500, hp_max=500)
        assert is_threat(spawn, player_level=10, zone_dispositions=zone_disps) is False

    def test_aggressive_yellow_is_threat(self) -> None:
        zone_disps = {"scowling": ["a_orc"]}
        spawn = make_spawn(name="a_orc_centurion", level=12, hp_current=150, hp_max=150)
        assert is_threat(spawn, player_level=10, zone_dispositions=zone_disps) is True

    def test_player_not_threat(self) -> None:
        spawn = make_spawn(name="SomePlayer", spawn_type=0, level=50, hp_current=5000, hp_max=5000)
        assert is_threat(spawn, player_level=10) is False


# ---------------------------------------------------------------------------
# find_targets
# ---------------------------------------------------------------------------


class TestFindTargets:
    def test_empty_spawns(self) -> None:
        result = find_targets((), 0.0, 0.0, player_level=10)
        assert result == []

    def test_sorts_white_before_blue(self) -> None:
        original = get_avoid_names()
        try:
            configure_avoid_names({})
            white = make_spawn(
                spawn_id=1, name="a_skeleton", level=10, x=50.0, y=50.0, hp_current=100, hp_max=100
            )
            blue = make_spawn(spawn_id=2, name="a_rat", level=7, x=50.0, y=50.0, hp_current=100, hp_max=100)
            result = find_targets((white, blue), 0.0, 0.0, player_level=10)
            assert len(result) == 2
            assert result[0][0].spawn_id == 1  # white first
        finally:
            set_avoid_names(original)

    def test_beyond_max_distance_excluded(self) -> None:
        original = get_avoid_names()
        try:
            configure_avoid_names({})
            far = make_spawn(name="a_skeleton", level=10, x=9000.0, y=9000.0, hp_current=100, hp_max=100)
            result = find_targets((far,), 0.0, 0.0, player_level=10, max_distance=200.0)
            assert result == []
        finally:
            set_avoid_names(original)


# ---------------------------------------------------------------------------
# check_mob_blocked_los
# ---------------------------------------------------------------------------


class TestCheckMobBlockedLos:
    def test_clear_los_returns_none(self) -> None:
        blocker = make_spawn(spawn_id=99, name="a_bat", x=500.0, y=500.0, z=0.0, hp_current=50, hp_max=50)
        result = check_mob_blocked_los(0.0, 0.0, 0.0, 100.0, 0.0, 0.0, (blocker,), target_id=0)
        assert result is None

    def test_mob_on_line_blocks(self) -> None:
        blocker = make_spawn(spawn_id=99, name="a_giant", x=50.0, y=0.0, z=0.0, hp_current=100, hp_max=100)
        result = check_mob_blocked_los(0.0, 0.0, 0.0, 100.0, 0.0, 0.0, (blocker,), target_id=0)
        assert result is not None
        assert result.spawn_id == 99

    def test_target_excluded_from_block_check(self) -> None:
        target = make_spawn(spawn_id=50, name="a_skeleton", x=50.0, y=0.0, z=0.0, hp_current=100, hp_max=100)
        result = check_mob_blocked_los(0.0, 0.0, 0.0, 100.0, 0.0, 0.0, (target,), target_id=50)
        assert result is None

    def test_overlapping_caster_target_returns_none(self) -> None:
        blocker = make_spawn(spawn_id=99, name="a_bat", x=0.5, y=0.0, z=0.0, hp_current=50, hp_max=50)
        result = check_mob_blocked_los(0.0, 0.0, 0.0, 0.1, 0.0, 0.0, (blocker,), target_id=0)
        assert result is None
