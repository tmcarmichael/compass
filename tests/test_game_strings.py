"""Tests for eq/game_strings.py: EQ game string databases.

Covers GameStrings, DatabaseStrings, load_game_strings, and load_database_strings.
MsgID is empty in the public release; load functions return empty instances.
Data class methods are tested with hand-constructed fixtures.
"""

from __future__ import annotations

from eq.game_strings import (
    DatabaseStrings,
    DBStrEntry,
    DBStrType,
    GameStrings,
    MsgID,
    load_database_strings,
    load_game_strings,
)

# ---------------------------------------------------------------------------
# MsgID class (stubbed -- no constants)
# ---------------------------------------------------------------------------


class TestMsgID:
    def test_class_exists(self) -> None:
        assert MsgID is not None

    def test_is_class(self) -> None:
        assert isinstance(MsgID, type)


# ---------------------------------------------------------------------------
# GameStrings data class
# ---------------------------------------------------------------------------


class TestGameStrings:
    def test_empty_game_strings(self) -> None:
        gs = GameStrings()
        assert len(gs) == 0
        assert gs.get(100) is None
        assert 100 not in gs

    def test_get_existing(self) -> None:
        gs = GameStrings(_by_id={100: "Out of range"})
        assert gs.get(100) == "Out of range"
        assert 100 in gs
        assert len(gs) == 1

    def test_get_missing(self) -> None:
        gs = GameStrings(_by_id={100: "Out of range"})
        assert gs.get(999) is None
        assert 999 not in gs

    def test_search_case_insensitive(self) -> None:
        gs = GameStrings(
            _by_id={
                100: "Your target is out of range!",
                101: "Target not found",
                200: "You gain experience!",
            }
        )
        results = gs.search("target")
        assert len(results) == 2
        ids = {r[0] for r in results}
        assert ids == {100, 101}

    def test_search_no_results(self) -> None:
        gs = GameStrings(_by_id={100: "Out of range"})
        results = gs.search("fizzle")
        assert results == []

    def test_search_empty_db(self) -> None:
        gs = GameStrings()
        assert gs.search("anything") == []

    def test_contains(self) -> None:
        gs = GameStrings(_by_id={100: "msg"})
        assert 100 in gs
        assert 200 not in gs


# ---------------------------------------------------------------------------
# load_game_strings (stubbed -- returns empty)
# ---------------------------------------------------------------------------


class TestLoadGameStrings:
    def test_returns_game_strings(self) -> None:
        gs = load_game_strings("/nonexistent/path")
        assert isinstance(gs, GameStrings)

    def test_returns_empty(self) -> None:
        gs = load_game_strings("/nonexistent/path")
        assert len(gs) == 0


# ---------------------------------------------------------------------------
# DBStrType constants
# ---------------------------------------------------------------------------


class TestDBStrType:
    def test_known_types(self) -> None:
        assert DBStrType.SPELL_DESCRIPTION == 6
        assert DBStrType.RACE_SINGULAR == 11
        assert DBStrType.CLASS_NAME == 13
        assert DBStrType.AA_NAME == 1


# ---------------------------------------------------------------------------
# DBStrEntry
# ---------------------------------------------------------------------------


class TestDBStrEntry:
    def test_frozen(self) -> None:
        e = DBStrEntry(id=1, type=6, value="Test")
        assert e.id == 1
        assert e.type == 6
        assert e.value == "Test"


# ---------------------------------------------------------------------------
# DatabaseStrings data class
# ---------------------------------------------------------------------------


class TestDatabaseStrings:
    def _make_db(self) -> DatabaseStrings:
        db = DatabaseStrings()
        db._entries[(100, 6)] = "Spell desc A"
        db._entries[(200, 6)] = "Spell desc B"
        db._entries[(1, 11)] = "Human"
        db._entries[(2, 11)] = "Barbarian"
        db._entries[(1, 13)] = "Warrior"
        db._entries[(500, 1)] = "AA Innate Run Speed"
        db._by_type[6] = {100: "Spell desc A", 200: "Spell desc B"}
        db._by_type[11] = {1: "Human", 2: "Barbarian"}
        db._by_type[13] = {1: "Warrior"}
        db._by_type[1] = {500: "AA Innate Run Speed"}
        return db

    def test_get(self) -> None:
        db = self._make_db()
        assert db.get(100, 6) == "Spell desc A"
        assert db.get(999, 6) is None

    def test_by_type(self) -> None:
        db = self._make_db()
        spells = db.by_type(6)
        assert len(spells) == 2
        assert spells[100] == "Spell desc A"

    def test_by_type_missing(self) -> None:
        db = self._make_db()
        assert db.by_type(99) == {}

    def test_spell_description(self) -> None:
        db = self._make_db()
        assert db.spell_description(100) == "Spell desc A"
        assert db.spell_description(999) is None

    def test_race_name(self) -> None:
        db = self._make_db()
        assert db.race_name(1) == "Human"
        assert db.race_name(999) is None

    def test_class_name(self) -> None:
        db = self._make_db()
        assert db.class_name(1) == "Warrior"
        assert db.class_name(999) is None

    def test_aa_name(self) -> None:
        db = self._make_db()
        assert db.aa_name(500) == "AA Innate Run Speed"
        assert db.aa_name(999) is None

    def test_search_unfiltered(self) -> None:
        db = self._make_db()
        results = db.search("spell")
        assert len(results) == 2
        assert all(isinstance(r, DBStrEntry) for r in results)

    def test_search_filtered_by_type(self) -> None:
        db = self._make_db()
        results = db.search("human", entry_type=11)
        assert len(results) == 1
        assert results[0].value == "Human"

    def test_search_case_insensitive(self) -> None:
        db = self._make_db()
        results = db.search("SPELL")
        assert len(results) == 2

    def test_search_no_match(self) -> None:
        db = self._make_db()
        assert db.search("zzzzz") == []

    def test_search_filtered_no_match(self) -> None:
        db = self._make_db()
        results = db.search("Human", entry_type=6)  # Human is type 11, not 6
        assert results == []

    def test_len(self) -> None:
        db = self._make_db()
        assert len(db) == 6


# ---------------------------------------------------------------------------
# load_database_strings (stubbed -- returns empty)
# ---------------------------------------------------------------------------


class TestLoadDatabaseStrings:
    def test_returns_database_strings(self) -> None:
        db = load_database_strings("/nonexistent/path")
        assert isinstance(db, DatabaseStrings)

    def test_returns_empty(self) -> None:
        db = load_database_strings("/nonexistent/path")
        assert len(db) == 0
