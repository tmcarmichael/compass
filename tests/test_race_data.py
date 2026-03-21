"""Tests for eq/race_data.py: race/model/animation data.

Covers RACE_NAMES lookups, RaceModelData properties, RaceDB queries, and
load_race_data. RACE_NAMES is empty in the public release; tests verify
the data classes and stub behavior.
"""

from __future__ import annotations

from eq.race_data import RACE_NAMES, RaceDB, RaceModelData, load_race_data


# ---------------------------------------------------------------------------
# RACE_NAMES static lookup (stubbed to empty)
# ---------------------------------------------------------------------------


class TestRaceNames:
    def test_is_dict(self) -> None:
        assert isinstance(RACE_NAMES, dict)

    def test_unknown_race_not_in_dict(self) -> None:
        assert 999 not in RACE_NAMES


# ---------------------------------------------------------------------------
# RaceModelData
# ---------------------------------------------------------------------------


class TestRaceModelData:
    def test_name_unknown_race(self) -> None:
        """With empty RACE_NAMES, all races fall through to Race_N."""
        r = RaceModelData(race_id=9999, male_model_ids=(0,), female_model_ids=(0,))
        assert r.name == "Race_9999"

    def test_name_race_1_falls_back(self) -> None:
        """With empty RACE_NAMES, even race_id=1 returns Race_1."""
        r = RaceModelData(race_id=1, male_model_ids=(100,), female_model_ids=(200,))
        assert r.name == "Race_1"

    def test_male_base_model(self) -> None:
        r = RaceModelData(race_id=1, male_model_ids=(42, 10, 20), female_model_ids=(50,))
        assert r.male_base_model == 42

    def test_female_base_model(self) -> None:
        r = RaceModelData(race_id=1, male_model_ids=(42,), female_model_ids=(50, 60))
        assert r.female_base_model == 50

    def test_male_base_model_empty(self) -> None:
        r = RaceModelData(race_id=1, male_model_ids=(), female_model_ids=(50,))
        assert r.male_base_model == 0

    def test_female_base_model_empty(self) -> None:
        r = RaceModelData(race_id=1, male_model_ids=(42,), female_model_ids=())
        assert r.female_base_model == 0

    def test_frozen(self) -> None:
        r = RaceModelData(race_id=1, male_model_ids=(1,), female_model_ids=(2,))
        assert r.race_id == 1  # frozen means we can read but not write


# ---------------------------------------------------------------------------
# RaceDB
# ---------------------------------------------------------------------------


class TestRaceDB:
    def _make_db(self) -> RaceDB:
        db = RaceDB()
        db._by_id[1] = RaceModelData(
            race_id=1,
            male_model_ids=tuple(range(14)),
            female_model_ids=tuple(range(100, 114)),
        )
        db._by_id[2] = RaceModelData(
            race_id=2,
            male_model_ids=tuple(range(200, 214)),
            female_model_ids=tuple(range(300, 314)),
        )
        return db

    def test_get_existing(self) -> None:
        db = self._make_db()
        r = db.get(1)
        assert r is not None
        assert r.race_id == 1

    def test_get_missing(self) -> None:
        db = self._make_db()
        assert db.get(999) is None

    def test_name_unknown_returns_fallback(self) -> None:
        """Unknown race ID not in db falls back to RACE_NAMES or Race_N."""
        db = self._make_db()
        assert db.name(999) == "Race_999"

    def test_all_races_sorted(self) -> None:
        db = self._make_db()
        races = db.all_races()
        assert len(races) == 2
        assert races[0].race_id == 1
        assert races[1].race_id == 2

    def test_len(self) -> None:
        db = self._make_db()
        assert len(db) == 2

    def test_contains(self) -> None:
        db = self._make_db()
        assert 1 in db
        assert 999 not in db

    def test_empty_db(self) -> None:
        db = RaceDB()
        assert len(db) == 0
        assert db.get(1) is None
        assert db.all_races() == []


# ---------------------------------------------------------------------------
# load_race_data (stubbed -- returns empty RaceDB)
# ---------------------------------------------------------------------------


class TestLoadRaceData:
    def test_returns_race_db(self) -> None:
        db = load_race_data("/nonexistent/path")
        assert isinstance(db, RaceDB)

    def test_returns_empty(self) -> None:
        db = load_race_data("/nonexistent/path")
        assert len(db) == 0
