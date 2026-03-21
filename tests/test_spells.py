"""Tests for eq/spells.py: spell data model and database queries.

SpellData is a frozen dataclass parsed from the game's spell database.
SpellDB provides typed queries by class, level, effect, and resist type.
These tests verify the data model contracts and query correctness using
hand-constructed SpellData fixtures (no game files needed).

_CLASS_ID_TO_FIELD_OFFSET is empty in the public release, so class-based
queries (available_for, min_level_for_class) return empty/255. SpellDB.load()
is stubbed to return 0.
"""

from __future__ import annotations

import pytest

from eq.spells import SPA, ResistType, SpellData, SpellDB, SpellRole, TargetType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spell(
    id: int = 1,
    name: str = "Test Spell",
    *,
    mana_cost: int = 50,
    cast_time_ms: int = 2500,
    duration_ticks: int = 0,
    beneficial: bool = False,
    resist_type: int = ResistType.MAGIC,
    target_type: int = TargetType.SINGLE,
    effect_ids: tuple = (254,) * 12,
    base_values: tuple = (0,) * 12,
    class_levels: tuple | None = None,
) -> SpellData:
    """Construct a SpellData with sensible defaults."""
    if class_levels is None:
        class_levels = (255,) * 16  # no class can use by default
    return SpellData(
        id=id,
        name=name,
        range=200,
        cast_time_ms=cast_time_ms,
        recovery_ms=0,
        recast_ms=0,
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


# ---------------------------------------------------------------------------
# SpellData properties
# ---------------------------------------------------------------------------


class TestSpellDataProperties:
    def test_cast_time_conversion(self) -> None:
        s = _spell(cast_time_ms=2500)
        assert s.cast_time == pytest.approx(2.5)

    def test_duration_seconds(self) -> None:
        s = _spell(duration_ticks=5)
        assert s.duration_seconds == pytest.approx(30.0)

    def test_is_dot(self) -> None:
        s = _spell(
            duration_ticks=5,
            beneficial=False,
            effect_ids=(SPA.CURRENT_HP,) + (254,) * 11,
            base_values=(-100,) + (0,) * 11,
        )
        assert s.is_dot is True

    def test_is_dd(self) -> None:
        s = _spell(
            duration_ticks=0,
            beneficial=False,
            effect_ids=(SPA.CURRENT_HP,) + (254,) * 11,
            base_values=(-200,) + (0,) * 11,
        )
        assert s.is_dd is True

    def test_is_heal(self) -> None:
        s = _spell(
            beneficial=True,
            effect_ids=(SPA.CURRENT_HP,) + (254,) * 11,
            base_values=(500,) + (0,) * 11,
        )
        assert s.is_heal is True

    def test_not_dot_when_beneficial(self) -> None:
        s = _spell(duration_ticks=5, beneficial=True)
        assert s.is_dot is False

    def test_resist_name(self) -> None:
        s = _spell(resist_type=ResistType.FIRE)
        assert s.resist_name == "fire"

    def test_target_name(self) -> None:
        s = _spell(target_type=TargetType.SELF)
        assert s.target_name == "self"

    def test_min_level_for_class_unknown(self) -> None:
        """With empty _CLASS_ID_TO_FIELD_OFFSET, all classes return 255."""
        s = _spell()
        assert s.min_level_for_class(1) == 255
        assert s.min_level_for_class(11) == 255

    def test_active_effects(self) -> None:
        s = _spell(
            effect_ids=(SPA.CURRENT_HP, SPA.ARMOR_CLASS, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254),
            base_values=(-100, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )
        effects = s.active_effects
        assert len(effects) == 2
        assert effects[0][0] == SPA.CURRENT_HP
        assert effects[1][0] == SPA.ARMOR_CLASS


# ---------------------------------------------------------------------------
# SpellDB queries
# ---------------------------------------------------------------------------


class TestSpellDB:
    @pytest.fixture
    def db(self) -> SpellDB:
        """A SpellDB populated with hand-constructed spells."""
        db = SpellDB()
        # Manually insert spells (bypassing file load)
        spells = [
            _spell(
                id=340,
                name="Disease Cloud",
                mana_cost=10,
                duration_ticks=6,
                beneficial=False,
                resist_type=ResistType.DISEASE,
                effect_ids=(SPA.CURRENT_HP,) + (254,) * 11,
                base_values=(-8,) + (0,) * 11,
                class_levels=(255,) * 10 + (1,) + (255,) * 5,
            ),
            _spell(
                id=502,
                name="Lifespike",
                mana_cost=18,
                cast_time_ms=1750,
                beneficial=False,
                resist_type=ResistType.MAGIC,
                effect_ids=(SPA.CURRENT_HP,) + (254,) * 11,
                base_values=(-45,) + (0,) * 11,
                class_levels=(255,) * 10 + (4,) + (255,) * 5,
            ),
            _spell(
                id=246,
                name="Lesser Shielding",
                mana_cost=25,
                beneficial=True,
                resist_type=ResistType.MAGIC,
                effect_ids=(SPA.ARMOR_CLASS, SPA.MAX_HP) + (254,) * 10,
                base_values=(12, 15) + (0,) * 10,
                class_levels=(255,) * 10 + (8,) + (255,) * 5,
            ),
        ]
        for s in spells:
            db._by_id[s.id] = s
            db._by_name[s.name.lower()] = s
        return db

    def test_get_by_id(self, db: SpellDB) -> None:
        assert db.get(340) is not None
        assert db.get(340).name == "Disease Cloud"

    def test_get_missing(self, db: SpellDB) -> None:
        assert db.get(99999) is None

    def test_find_by_name(self, db: SpellDB) -> None:
        assert db.find("disease cloud") is not None
        assert db.find("DISEASE CLOUD") is not None

    def test_find_missing(self, db: SpellDB) -> None:
        assert db.find("nonexistent") is None

    def test_available_for_empty_with_stubbed_offsets(self, db: SpellDB) -> None:
        """With empty _CLASS_ID_TO_FIELD_OFFSET, available_for returns []."""
        available = db.available_for(class_id=11, level=10)
        assert available == []

    def test_by_resist_type(self, db: SpellDB) -> None:
        disease_spells = db.by_resist_type(ResistType.DISEASE)
        assert len(disease_spells) == 1
        assert disease_spells[0].name == "Disease Cloud"

    def test_dots(self, db: SpellDB) -> None:
        dots = db.dots()
        assert len(dots) == 1
        assert dots[0].name == "Disease Cloud"

    def test_contains(self, db: SpellDB) -> None:
        assert 340 in db
        assert 99999 not in db

    def test_len(self, db: SpellDB) -> None:
        assert len(db) == 3


# ---------------------------------------------------------------------------
# SpellDB by target type
# ---------------------------------------------------------------------------


class TestSpellDBByTargetType:
    @pytest.fixture
    def db(self) -> SpellDB:
        db = SpellDB()
        for s in [
            _spell(
                id=340,
                name="Disease Cloud",
                duration_ticks=5,
                target_type=TargetType.SINGLE,
                base_values=(-20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                effect_ids=(SPA.CURRENT_HP, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254),
            ),
            _spell(
                id=580,
                name="Lifespike",
                target_type=TargetType.SINGLE,
                base_values=(-100, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                effect_ids=(SPA.CURRENT_HP, SPA.CURRENT_HP, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254),
            ),
            _spell(
                id=209,
                name="Spirit of Wolf",
                beneficial=True,
                target_type=TargetType.SINGLE,
                effect_ids=(SPA.MOVEMENT_SPEED, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254),
            ),
        ]:
            db._by_id[s.id] = s
            db._by_name[s.name.lower()] = s
        return db

    def test_by_target_type(self, db: SpellDB) -> None:
        single_spells = db.by_target_type(TargetType.SINGLE)
        assert len(single_spells) == 3

    def test_by_target_type_no_match(self, db: SpellDB) -> None:
        corpse_spells = db.by_target_type(TargetType.CORPSE)
        assert corpse_spells == []


class TestSpellDBByEffect:
    @pytest.fixture
    def db(self) -> SpellDB:
        db = SpellDB()
        for s in [
            _spell(
                id=340,
                name="Disease Cloud",
                duration_ticks=5,
                base_values=(-20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                effect_ids=(SPA.CURRENT_HP, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254),
            ),
            _spell(
                id=580,
                name="Lifespike",
                base_values=(-100, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                effect_ids=(SPA.CURRENT_HP, SPA.CURRENT_HP, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254),
            ),
        ]:
            db._by_id[s.id] = s
            db._by_name[s.name.lower()] = s
        return db

    def test_by_effect(self, db: SpellDB) -> None:
        hp_spells = db.by_effect(SPA.CURRENT_HP)
        names = {s.name for s in hp_spells}
        assert "Disease Cloud" in names
        assert "Lifespike" in names

    def test_by_effect_no_match(self, db: SpellDB) -> None:
        assert db.by_effect(SPA.SEE_INVIS) == []


class TestSpellDBHeals:
    def test_heals(self) -> None:
        db = SpellDB()
        heal = _spell(
            id=10,
            name="Minor Healing",
            beneficial=True,
            effect_ids=(SPA.CURRENT_HP,) + (254,) * 11,
            base_values=(200,) + (0,) * 11,
        )
        non_heal = _spell(
            id=11,
            name="Lifespike",
            beneficial=False,
            effect_ids=(SPA.CURRENT_HP,) + (254,) * 11,
            base_values=(-45,) + (0,) * 11,
        )
        db._by_id[heal.id] = heal
        db._by_id[non_heal.id] = non_heal
        heals = db.heals()
        assert len(heals) == 1
        assert heals[0].name == "Minor Healing"

    def test_heals_empty(self) -> None:
        db = SpellDB()
        assert db.heals() == []


class TestSpellDBAvailableForEdgeCases:
    def test_available_for_unknown_class(self) -> None:
        db = SpellDB()
        db._by_id[1] = _spell(id=1, name="Test")
        # class_id 99 not in _CLASS_ID_TO_FIELD_OFFSET
        assert db.available_for(class_id=99, level=60) == []


class TestSpellDataRecastTime:
    def test_recast_time(self) -> None:
        s = _spell()
        # Default recast_ms=0
        assert s.recast_time == 0.0

    def test_recast_time_nonzero(self) -> None:
        s = SpellData(
            id=1,
            name="Test",
            range=0,
            cast_time_ms=0,
            recovery_ms=0,
            recast_ms=5000,
            duration_ticks=0,
            mana_cost=0,
            cast_message="",
            cast_on_other="",
            fade_message="",
            class_levels=(255,) * 16,
            beneficial=False,
            resist_type=0,
            target_type=5,
            effect_ids=(254,) * 12,
            base_values=(0,) * 12,
            max_values=(0,) * 12,
            duration_formula=0,
            aoe_range=0,
            pushback=0.0,
        )
        assert s.recast_time == 5.0


class TestSpellDataMinLevelEdge:
    def test_min_level_for_unknown_class(self) -> None:
        s = _spell()
        assert s.min_level_for_class(99) == 255


class TestSpellDataIsDetrimental:
    def test_is_detrimental(self) -> None:
        s = _spell(beneficial=False)
        assert s.is_detrimental is True

    def test_not_detrimental(self) -> None:
        s = _spell(beneficial=True)
        assert s.is_detrimental is False


class TestResistTypeNames:
    def test_known_names(self) -> None:
        assert ResistType.label_for(0) == "unresistable"
        assert ResistType.label_for(2) == "fire"
        assert ResistType.label_for(5) == "disease"

    def test_unknown_resist(self) -> None:
        assert ResistType.label_for(99) == "unknown(99)"


class TestTargetTypeNames:
    def test_known_names(self) -> None:
        assert TargetType.label_for(5) == "single"
        assert TargetType.label_for(6) == "self"

    def test_unknown_target(self) -> None:
        assert TargetType.label_for(99) == "unknown(99)"


class TestSpellDBLoad:
    """Test SpellDB.load() stub behavior."""

    def test_load_returns_zero(self, tmp_path) -> None:
        """Stubbed load() always returns 0."""
        db = SpellDB()
        assert db.load(tmp_path / "spells_us.txt") == 0

    def test_load_leaves_db_empty(self, tmp_path) -> None:
        """Stubbed load() does not populate the database."""
        db = SpellDB()
        db.load(tmp_path / "spells_us.txt")
        assert len(db) == 0


# ---------------------------------------------------------------------------
# SpellRole enum
# ---------------------------------------------------------------------------


class TestSpellRole:
    def test_is_str(self) -> None:
        assert isinstance(SpellRole.DOT, str)
        assert SpellRole.DOT == "dot"

    def test_all_roles_unique(self) -> None:
        values = [r.value for r in SpellRole]
        assert len(values) == len(set(values))
