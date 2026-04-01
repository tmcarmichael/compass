"""Tests for eq/loadout.py: spell loadout management, gem assignment, role lookup.

Covers Spell properties, gems_for_level, role-based lookup.
_CLASS_SPELL_PRIORITIES is empty in the public release, so configure_loadout
returns empty for all classes. Tests that depend on priority tables are removed.
"""

from __future__ import annotations

import pytest

from eq.spells import SpellDB

# ---------------------------------------------------------------------------
# Helpers -- reuse the _spell factory
# ---------------------------------------------------------------------------


def _spell_data(
    id: int = 1,
    name: str = "Test Spell",
    *,
    mana_cost: int = 50,
    cast_time_ms: int = 2500,
    duration_ticks: int = 0,
    beneficial: bool = False,
    resist_type: int = 1,
    target_type: int = 5,
    effect_ids: tuple = (254,) * 12,
    base_values: tuple = (0,) * 12,
    class_levels: tuple | None = None,
    range: int = 200,
    recast_ms: int = 0,
):
    from eq.spells import SpellData

    if class_levels is None:
        class_levels = (255,) * 16
    return SpellData(
        id=id,
        name=name,
        range=range,
        cast_time_ms=cast_time_ms,
        recovery_ms=0,
        recast_ms=recast_ms,
        duration_ticks=duration_ticks,
        mana_cost=mana_cost,
        cast_message="",
        cast_on_other="",
        fade_message="",
        class_levels=class_levels,
        beneficial=beneficial,
        resist_type=resist_type,
        target_type=target_type,
        effect_ids=effect_ids,
        base_values=base_values,
        max_values=(0,) * 12,
        duration_formula=0,
        aoe_range=0,
        pushback=0.0,
    )


def _make_necro_db() -> SpellDB:
    """SpellDB with a few necro spells at known levels for class_id=11."""

    # class_levels index 10 = necromancer
    def necro_levels(lvl: int) -> tuple:
        return (255,) * 10 + (lvl,) + (255,) * 5

    db = SpellDB()
    spells = [
        _spell_data(
            id=338, name="Cavorting Bones", cast_time_ms=5000, mana_cost=15, class_levels=necro_levels(1)
        ),
        _spell_data(
            id=340,
            name="Disease Cloud",
            cast_time_ms=1500,
            mana_cost=10,
            duration_ticks=6,
            class_levels=necro_levels(1),
        ),
        _spell_data(id=341, name="Lifetap", cast_time_ms=1500, mana_cost=9, class_levels=necro_levels(1)),
        _spell_data(
            id=348, name="Poison Bolt", cast_time_ms=1750, mana_cost=30, class_levels=necro_levels(4)
        ),
        _spell_data(
            id=288,
            name="Minor Shielding",
            cast_time_ms=2500,
            mana_cost=10,
            beneficial=True,
            class_levels=necro_levels(1),
        ),
        _spell_data(id=351, name="Bone Walk", cast_time_ms=7000, mana_cost=80, class_levels=necro_levels(8)),
        _spell_data(id=641, name="Dark Pact", cast_time_ms=3000, mana_cost=5, class_levels=necro_levels(8)),
        _spell_data(
            id=353,
            name="Mend Bones",
            cast_time_ms=3500,
            mana_cost=25,
            recast_ms=7000,
            class_levels=necro_levels(8),
        ),
        _spell_data(
            id=522, name="Gather Shadows", cast_time_ms=5000, mana_cost=35, class_levels=necro_levels(8)
        ),
        _spell_data(
            id=246,
            name="Lesser Shielding",
            cast_time_ms=2500,
            mana_cost=25,
            beneficial=True,
            class_levels=necro_levels(8),
        ),
        _spell_data(
            id=900,
            name="Fear",
            cast_time_ms=1500,
            mana_cost=20,
            class_levels=necro_levels(4),
        ),
        _spell_data(
            id=901,
            name="Gate",
            cast_time_ms=5000,
            mana_cost=70,
            class_levels=necro_levels(4),
        ),
        _spell_data(
            id=902,
            name="Clinging Darkness",
            cast_time_ms=2000,
            mana_cost=25,
            class_levels=necro_levels(4),
        ),
    ]
    for s in spells:
        db._by_id[s.id] = s
        db._by_name[s.name.lower()] = s
    return db


# ---------------------------------------------------------------------------
# Spell dataclass tests
# ---------------------------------------------------------------------------


class TestSpellObject:
    """Tests for eq.loadout.Spell properties."""

    def test_truthy_when_memorized(self) -> None:
        from eq.loadout import Spell

        s = Spell("Test", gem=3, cast_time=2.5, mana_cost=50)
        assert bool(s) is True

    def test_falsy_when_not_memorized(self) -> None:
        from eq.loadout import Spell

        s = Spell("Test", gem=0, cast_time=2.5, mana_cost=50)
        assert bool(s) is False

    def test_mana_efficiency(self) -> None:
        from eq.loadout import Spell

        s = Spell("Test", gem=1, cast_time=2.5, mana_cost=100, est_damage=300.0)
        assert s.mana_efficiency == pytest.approx(3.0)

    def test_mana_efficiency_zero_cost(self) -> None:
        from eq.loadout import Spell

        s = Spell("Test", gem=1, cast_time=2.5, mana_cost=0, est_damage=100.0)
        assert s.mana_efficiency == 0.0

    def test_dps_per_mana(self) -> None:
        from eq.loadout import Spell

        # 100 damage over 2s cast = 50 dps, / 50 mana = 1.0
        s = Spell("Test", gem=1, cast_time=2.0, mana_cost=50, est_damage=100.0)
        assert s.dps_per_mana == pytest.approx(1.0)

    def test_dps_per_mana_zero_fields(self) -> None:
        from eq.loadout import Spell

        s = Spell("Test", gem=1, cast_time=0, mana_cost=50, est_damage=100.0)
        assert s.dps_per_mana == 0.0


# ---------------------------------------------------------------------------
# gems_for_level
# ---------------------------------------------------------------------------


class TestGemsForLevel:
    @pytest.mark.parametrize("level", [1, 10, 30, 60])
    def test_always_8(self, level: int) -> None:
        from eq.loadout import gems_for_level

        assert gems_for_level(level) == 8


# ---------------------------------------------------------------------------
# configure_loadout -- empty priorities returns empty
# ---------------------------------------------------------------------------


class TestConfigureLoadout:
    def test_unknown_class_returns_empty(self) -> None:
        from eq.loadout import configure_loadout

        db = _make_necro_db()
        assigned = configure_loadout(class_id=99, level=60, db=db)
        assert assigned == {}

    def test_any_class_returns_empty_with_no_priorities(self) -> None:
        """With empty _CLASS_SPELL_PRIORITIES, all classes return empty."""
        from eq.loadout import configure_loadout

        db = _make_necro_db()
        assigned = configure_loadout(class_id=11, level=10, db=db)
        assert assigned == {}


# ---------------------------------------------------------------------------
# compute_desired_loadout -- empty priorities returns empty
# ---------------------------------------------------------------------------


class TestComputeDesiredLoadout:
    def test_unknown_class_empty(self) -> None:
        from eq.loadout import compute_desired_loadout

        db = _make_necro_db()
        assert compute_desired_loadout(class_id=99, level=60, db=db) == {}

    def test_any_class_empty_with_no_priorities(self) -> None:
        """With empty _CLASS_SPELL_PRIORITIES, all classes return empty."""
        from eq.loadout import compute_desired_loadout

        db = _make_necro_db()
        assert compute_desired_loadout(class_id=11, level=10, db=db) == {}


# ---------------------------------------------------------------------------
# configure_from_memory
# ---------------------------------------------------------------------------


class TestConfigureFromMemory:
    def test_sets_gems_from_dict(self) -> None:
        from eq.loadout import configure_from_memory

        db = _make_necro_db()
        memorized = {1: 340, 2: 341, 3: 338}  # Disease Cloud, Lifetap, Cavorting Bones
        assigned = configure_from_memory(memorized, class_id=11, db=db)
        assert assigned[1] == "Disease Cloud"
        assert assigned[2] == "Lifetap"
        assert assigned[3] == "Cavorting Bones"

    def test_unknown_spell_id_skipped(self) -> None:
        from eq.loadout import configure_from_memory

        db = _make_necro_db()
        memorized = {1: 99999}
        assigned = configure_from_memory(memorized, class_id=11, db=db)
        assert len(assigned) == 0


# ---------------------------------------------------------------------------
# check_spell_loadout -- with empty priorities, desired is empty
# ---------------------------------------------------------------------------


class TestCheckSpellLoadout:
    def test_empty_desired_returns_empty_changes(self) -> None:
        from eq.loadout import check_spell_loadout

        db = _make_necro_db()
        memorized = {1: 340, 2: 341}
        # With empty priorities, desired is empty, so memorized items are "extra"
        changes = check_spell_loadout(memorized, class_id=11, level=10, db=db)
        # No desired spells means nothing to memorize -- only "clear" actions
        # But since desired is empty, missing is also empty, so changes depend on
        # the implementation. With empty desired, actual - desired = actual,
        # but missing (desired - actual) = empty, so no "memorize" actions.
        # Clarification: if desired is empty and memorized has items, actual_ids - desired_ids != empty
        # but missing = desired_ids - actual_ids = empty set, so the check returns [].
        assert changes == []
