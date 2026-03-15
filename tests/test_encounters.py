"""Tests for brain.learning.encounters  -- fight outcome learning.

FightHistory records per-entity-type encounter data and produces learned
estimates after a minimum sample threshold. These tests verify the
learning convergence contract and persistence roundtrip.
"""

from __future__ import annotations

import pytest

from brain.learning.encounters import MIN_FIGHTS_FOR_LEARNED, FightHistory, FightRecord


def _record_fights(
    fh: FightHistory,
    mob: str,
    n: int,
    *,
    duration: float = 30.0,
    mana_spent: int = 200,
    hp_delta: float = -0.05,
) -> None:
    """Helper: record n identical fights against a mob."""
    for _ in range(n):
        fh.record(
            mob_name=mob,
            duration=duration,
            mana_spent=mana_spent,
            hp_delta=hp_delta,
            casts=10,
            pet_heals=2,
            pet_died=False,
            defeated=True,
        )


# ---------------------------------------------------------------------------
# FightRecord dataclass
# ---------------------------------------------------------------------------


class TestFightRecord:
    def test_defaults(self) -> None:
        r = FightRecord(
            duration=10.0,
            mana_spent=100,
            hp_delta=-0.05,
            casts=5,
            pet_heals=1,
            pet_died=False,
            defeated=True,
        )
        assert r.adds == 0
        assert r.mob_level == 0
        assert r.strategy == ""
        assert r.fitness == 0.0

    def test_required_fields(self) -> None:
        cls: type = FightRecord
        with pytest.raises(TypeError, match="missing.*required"):
            cls()


# ---------------------------------------------------------------------------
# FightHistory  -- empty state
# ---------------------------------------------------------------------------


class TestEmptyHistory:
    def test_no_stats(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        assert fh.get_all_stats() == {}

    def test_has_learned_false(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        assert fh.has_learned("a_skeleton") is False

    def test_learned_duration_none(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        assert fh.learned_duration("a_skeleton") is None


# ---------------------------------------------------------------------------
# Learning threshold
# ---------------------------------------------------------------------------


class TestLearningThreshold:
    def test_below_threshold_returns_none(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        _record_fights(fh, "a_skeleton", MIN_FIGHTS_FOR_LEARNED - 1)
        assert fh.learned_duration("a_skeleton") is None
        assert fh.has_learned("a_skeleton") is False

    def test_at_threshold_returns_value(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        _record_fights(fh, "a_skeleton", MIN_FIGHTS_FOR_LEARNED, duration=25.0)
        dur = fh.learned_duration("a_skeleton")
        assert dur is not None
        assert dur == pytest.approx(25.0)

    def test_learned_mana_converges(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        _record_fights(fh, "a_skeleton", MIN_FIGHTS_FOR_LEARNED, mana_spent=300)
        mana = fh.learned_mana("a_skeleton")
        assert mana is not None
        assert mana == pytest.approx(300, abs=1)

    def test_has_learned_true_at_threshold(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        _record_fights(fh, "a_skeleton", MIN_FIGHTS_FOR_LEARNED)
        assert fh.has_learned("a_skeleton") is True


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------


class TestNameNormalization:
    def test_strips_trailing_digits(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        _record_fights(fh, "a_skeleton007", 1)
        _record_fights(fh, "a_skeleton003", 1)
        stats = fh.get_stats("a_skeleton")
        assert stats is not None
        assert stats.fights == 2


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    def test_caps_at_max_samples(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        _record_fights(fh, "a_skeleton", 50, duration=10.0)
        stats = fh.get_stats("a_skeleton")
        assert stats is not None
        # MAX_SAMPLES is 30
        assert stats.fights <= 30

    def test_recent_data_dominates(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        # Record 30 slow fights, then 30 fast fights
        _record_fights(fh, "a_skeleton", 30, duration=100.0)
        _record_fights(fh, "a_skeleton", 30, duration=10.0)
        dur = fh.learned_duration("a_skeleton")
        assert dur is not None
        # Window should have dropped the slow fights
        assert dur < 50.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_roundtrip(self, fight_history_factory, tmp_path) -> None:
        fh = fight_history_factory("roundtrip_zone")
        _record_fights(fh, "a_skeleton", 10, duration=20.0, mana_spent=150)
        fh.save()

        # Load into fresh instance
        fh2 = FightHistory(zone="roundtrip_zone", data_dir=str(tmp_path))
        assert fh2.has_learned("a_skeleton") is True
        assert fh2.learned_duration("a_skeleton") == pytest.approx(20.0)

    def test_save_not_dirty_noop(self, fight_history_factory) -> None:
        """Save is a no-op when _dirty is False (no new records since last save)."""
        fh = fight_history_factory()
        fh.save()  # nothing dirty, should not create file
        # No exception is sufficient

    def test_load_legacy_format(self, tmp_path) -> None:
        """Load pre-v1 format (no 'v' or 'npcs' key, just {name: [records]})."""
        import json

        legacy = {
            "a_bat": [
                {
                    "dur": 10.0,
                    "mana": 100,
                    "hp": -0.05,
                    "casts": 3,
                    "pet_h": 1,
                    "pet_d": False,
                    "defeat": True,
                }
            ]
        }
        path = tmp_path / "legacy_fights.json"
        with open(path, "w") as f:
            json.dump(legacy, f)

        fh = FightHistory(zone="legacy", data_dir=str(tmp_path))
        stats = fh.get_stats("a_bat")
        assert stats is not None
        assert stats.fights == 1


# ---------------------------------------------------------------------------
# Strategy fitness tracking
# ---------------------------------------------------------------------------


class TestStrategyFitness:
    def test_record_with_strategy_preserved(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        fh.record(
            mob_name="a_skeleton",
            duration=20.0,
            mana_spent=150,
            hp_delta=-0.05,
            casts=5,
            pet_heals=1,
            pet_died=False,
            defeated=True,
            strategy="pet_tank",
        )
        records = fh._records.get("a_skeleton", [])
        assert len(records) == 1
        assert records[0].strategy == "pet_tank"

    def test_fitness_appended_to_recent(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        fh.record(
            mob_name="a_skeleton",
            duration=20.0,
            mana_spent=150,
            hp_delta=-0.05,
            casts=5,
            pet_heals=1,
            pet_died=False,
            defeated=True,
            fitness=0.85,
        )
        assert len(fh._recent_fitness) == 1
        assert fh._recent_fitness[0] == (0.85, "a_skeleton")

    def test_zero_fitness_not_appended(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        fh.record(
            mob_name="a_skeleton",
            duration=20.0,
            mana_spent=150,
            hp_delta=-0.05,
            casts=5,
            pet_heals=1,
            pet_died=False,
            defeated=True,
            fitness=0.0,
        )
        assert len(fh._recent_fitness) == 0


# ---------------------------------------------------------------------------
# drain_recent_fitness
# ---------------------------------------------------------------------------


class TestDrainRecentFitness:
    def test_returns_and_clears(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        fh.record(
            mob_name="a_bat",
            duration=15.0,
            mana_spent=100,
            hp_delta=-0.03,
            casts=4,
            pet_heals=0,
            pet_died=False,
            defeated=True,
            fitness=0.7,
        )
        fh.record(
            mob_name="a_skeleton",
            duration=25.0,
            mana_spent=200,
            hp_delta=-0.08,
            casts=6,
            pet_heals=1,
            pet_died=False,
            defeated=True,
            fitness=0.5,
        )

        result = fh.drain_recent_fitness()
        assert len(result) == 2
        assert result[0] == (0.7, "a_bat")
        assert result[1] == (0.5, "a_skeleton")

        # Second drain returns empty
        result2 = fh.drain_recent_fitness()
        assert result2 == []


# ---------------------------------------------------------------------------
# Learned add probability and types
# ---------------------------------------------------------------------------


class TestLearnedAdds:
    def test_add_probability_insufficient_data(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        _record_fights(fh, "a_skeleton", MIN_FIGHTS_FOR_LEARNED - 1)
        assert fh.learned_add_probability("a_skeleton") is None

    def test_add_probability_computed(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        for i in range(MIN_FIGHTS_FOR_LEARNED):
            fh.record(
                mob_name="a_skeleton",
                duration=20.0,
                mana_spent=150,
                hp_delta=-0.05,
                casts=5,
                pet_heals=1,
                pet_died=False,
                defeated=True,
                adds=1 if i < 2 else 0,
            )
        prob = fh.learned_add_probability("a_skeleton")
        assert prob is not None
        assert prob == pytest.approx(2 / MIN_FIGHTS_FOR_LEARNED)

    def test_add_types_insufficient_data(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        _record_fights(fh, "a_skeleton", MIN_FIGHTS_FOR_LEARNED - 1)
        assert fh.learned_add_types("a_skeleton") == {}

    def test_add_types_computed(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        for i in range(MIN_FIGHTS_FOR_LEARNED):
            extra = ("a_bat",) if i < 3 else ()
            fh.record(
                mob_name="a_skeleton",
                duration=20.0,
                mana_spent=150,
                hp_delta=-0.05,
                casts=5,
                pet_heals=1,
                pet_died=False,
                defeated=True,
                extra_npc_types=extra,
            )
        types = fh.learned_add_types("a_skeleton")
        assert "a_bat" in types
        assert types["a_bat"] == pytest.approx(3 / MIN_FIGHTS_FOR_LEARNED)


# ---------------------------------------------------------------------------
# Additional learned getters
# ---------------------------------------------------------------------------


class TestLearnedGetters:
    def test_learned_danger(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        _record_fights(fh, "a_skeleton", MIN_FIGHTS_FOR_LEARNED, hp_delta=-0.3)
        danger = fh.learned_danger("a_skeleton")
        assert danger is not None
        assert 0.0 <= danger <= 1.0

    def test_learned_danger_insufficient_data(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        _record_fights(fh, "a_skeleton", MIN_FIGHTS_FOR_LEARNED - 1)
        assert fh.learned_danger("a_skeleton") is None

    def test_learned_adds_avg(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        for _ in range(MIN_FIGHTS_FOR_LEARNED):
            fh.record(
                mob_name="a_skeleton",
                duration=20.0,
                mana_spent=150,
                hp_delta=-0.05,
                casts=5,
                pet_heals=1,
                pet_died=False,
                defeated=True,
                adds=2,
            )
        avg = fh.learned_adds("a_skeleton")
        assert avg is not None
        assert avg == pytest.approx(2.0)

    def test_learned_adds_insufficient(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        _record_fights(fh, "a_skeleton", MIN_FIGHTS_FOR_LEARNED - 1)
        assert fh.learned_adds("a_skeleton") is None

    def test_learned_pet_death_rate(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        for i in range(MIN_FIGHTS_FOR_LEARNED):
            fh.record(
                mob_name="a_skeleton",
                duration=20.0,
                mana_spent=150,
                hp_delta=-0.05,
                casts=5,
                pet_heals=1,
                pet_died=(i == 0),  # pet dies in 1 out of N fights
                defeated=True,
            )
        rate = fh.learned_pet_death_rate("a_skeleton")
        assert rate is not None
        assert rate == pytest.approx(1 / MIN_FIGHTS_FOR_LEARNED)

    def test_learned_pet_death_rate_insufficient(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        _record_fights(fh, "a_skeleton", MIN_FIGHTS_FOR_LEARNED - 1)
        assert fh.learned_pet_death_rate("a_skeleton") is None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_empty_summary(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        assert "no data" in fh.summary()

    def test_summary_with_data(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        _record_fights(fh, "a_skeleton", MIN_FIGHTS_FOR_LEARNED, duration=30.0)
        _record_fights(fh, "a_bat", 2, duration=10.0)
        summary = fh.summary()
        assert "a_skeleton" in summary
        assert "a_bat" in summary
        assert "*" in summary  # learned indicator for skeleton

    def test_summary_shows_adds(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        for _ in range(MIN_FIGHTS_FOR_LEARNED):
            fh.record(
                mob_name="a_skeleton",
                duration=20.0,
                mana_spent=150,
                hp_delta=-0.05,
                casts=5,
                pet_heals=1,
                pet_died=False,
                defeated=True,
                adds=2,
            )
        summary = fh.summary()
        assert "adds=" in summary

    def test_summary_shows_pet_deaths_warning(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        for _ in range(MIN_FIGHTS_FOR_LEARNED):
            fh.record(
                mob_name="a_skeleton",
                duration=20.0,
                mana_spent=150,
                hp_delta=-0.1,
                casts=5,
                pet_heals=1,
                pet_died=True,  # 100% pet death rate
                defeated=True,
            )
        summary = fh.summary()
        assert "PET_DEATHS!" in summary


# ---------------------------------------------------------------------------
# Recompute with empty records
# ---------------------------------------------------------------------------


class TestRecompute:
    def test_recompute_empty_removes_stats(self, fight_history_factory) -> None:
        fh = fight_history_factory()
        _record_fights(fh, "a_skeleton", 3)
        assert fh.get_stats("a_skeleton") is not None
        fh._records["a_skeleton"] = []
        fh._recompute("a_skeleton")
        assert fh.get_stats("a_skeleton") is None
