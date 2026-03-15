"""Tests for brain.learning.danger_memory -- persistent danger avoidance learning.

Covers DangerRecord, DangerMemory.record_death, record_flee, danger_penalty
with time-decay, should_avoid threshold, plan failure tracking, and JSON
persistence via tmp_path.
"""

from __future__ import annotations

from brain.learning.danger_memory import AVOID_DEATHS, DECAY_HOURS, DangerMemory, DangerRecord

# ---------------------------------------------------------------------------
# DangerRecord dataclass
# ---------------------------------------------------------------------------


class TestDangerRecord:
    def test_defaults(self) -> None:
        r = DangerRecord(entity_type="a_skeleton")
        assert r.deaths == 0
        assert r.flees == 0
        assert r.last_incident == 0.0
        assert r.incidents == []


# ---------------------------------------------------------------------------
# DangerMemory: recording events
# ---------------------------------------------------------------------------


class TestDangerMemoryRecording:
    def test_record_death_increments(self, tmp_path: object) -> None:
        dm = DangerMemory(zone="test", data_dir=str(tmp_path))
        dm.record_death("a_dragon")
        dm.record_death("a_dragon")
        assert dm._records["a_dragon"].deaths == 2

    def test_record_flee_increments(self, tmp_path: object) -> None:
        dm = DangerMemory(zone="test", data_dir=str(tmp_path))
        dm.record_flee("a_bat")
        assert dm._records["a_bat"].flees == 1

    def test_recent_deaths_session_only(self, tmp_path: object) -> None:
        dm = DangerMemory(zone="test", data_dir=str(tmp_path))
        assert dm.recent_deaths == 0
        dm.record_death("a_dragon")
        dm.record_death("a_bat")
        assert dm.recent_deaths == 2

    def test_record_death_with_context(self, tmp_path: object) -> None:
        dm = DangerMemory(zone="test", data_dir=str(tmp_path))
        dm.record_death("a_dragon", context={"level": 50})
        assert dm._records["a_dragon"].incidents[0]["level"] == 50


# ---------------------------------------------------------------------------
# DangerMemory: danger_penalty
# ---------------------------------------------------------------------------


class TestDangerPenalty:
    def test_unknown_entity_zero(self, tmp_path: object) -> None:
        dm = DangerMemory(zone="test", data_dir=str(tmp_path))
        assert dm.danger_penalty("never_seen") == 0.0

    def test_deaths_increase_penalty(self, tmp_path: object) -> None:
        dm = DangerMemory(zone="test", data_dir=str(tmp_path))
        dm.record_death("a_dragon")
        p1 = dm.danger_penalty("a_dragon")
        dm.record_death("a_dragon")
        p2 = dm.danger_penalty("a_dragon")
        assert p2 > p1 > 0.0

    def test_penalty_capped_at_one(self, tmp_path: object) -> None:
        dm = DangerMemory(zone="test", data_dir=str(tmp_path))
        for _ in range(10):
            dm.record_death("a_dragon")
        assert dm.danger_penalty("a_dragon") <= 1.0

    def test_decay_reduces_penalty(self, tmp_path: object) -> None:

        t = [1_000_000.0]
        dm = DangerMemory(zone="test", data_dir=str(tmp_path), clock=lambda: t[0])
        dm.record_death("a_dragon")
        dm.record_death("a_dragon")
        dm.record_death("a_dragon")
        fresh_penalty = dm.danger_penalty("a_dragon")

        # Advance time by one decay half-life
        t[0] += DECAY_HOURS * 3600
        decayed_penalty = dm.danger_penalty("a_dragon")
        assert decayed_penalty < fresh_penalty
        # Should be roughly half (within tolerance for raw*decay math)
        assert abs(decayed_penalty - fresh_penalty * 0.5) < 0.05


# ---------------------------------------------------------------------------
# DangerMemory: should_avoid
# ---------------------------------------------------------------------------


class TestShouldAvoid:
    def test_below_threshold_not_avoided(self, tmp_path: object) -> None:
        dm = DangerMemory(zone="test", data_dir=str(tmp_path))
        dm.record_death("a_dragon")
        assert dm.should_avoid("a_dragon") is False

    def test_at_threshold_avoided(self, tmp_path: object) -> None:
        dm = DangerMemory(zone="test", data_dir=str(tmp_path))
        for _ in range(AVOID_DEATHS):
            dm.record_death("a_dragon")
        assert dm.should_avoid("a_dragon") is True


# ---------------------------------------------------------------------------
# DangerMemory: plan failure tracking
# ---------------------------------------------------------------------------


class TestPlanFailure:
    def test_plan_step_penalty_below_threshold(self, tmp_path: object) -> None:
        dm = DangerMemory(zone="test", data_dir=str(tmp_path))
        dm.record_plan_failure("pull", "target_moved")
        assert dm.plan_step_penalty("pull") == 0.0

    def test_plan_step_penalty_above_threshold(self, tmp_path: object) -> None:
        dm = DangerMemory(zone="test", data_dir=str(tmp_path))
        for _ in range(5):
            dm.record_plan_failure("pull", "target_moved")
        penalty = dm.plan_step_penalty("pull")
        assert penalty > 0.0


# ---------------------------------------------------------------------------
# DangerMemory: persistence
# ---------------------------------------------------------------------------


class TestDangerMemoryPersistence:
    def test_save_and_reload(self, tmp_path: object) -> None:
        dm1 = DangerMemory(zone="persist_test", data_dir=str(tmp_path))
        dm1.record_death("a_dragon")
        dm1.record_death("a_dragon")
        dm1.record_flee("a_bat")
        dm1.save()

        dm2 = DangerMemory(zone="persist_test", data_dir=str(tmp_path))
        assert dm2._records["a_dragon"].deaths == 2
        assert dm2._records["a_bat"].flees == 1

    def test_load_missing_file_no_crash(self, tmp_path: object) -> None:
        dm = DangerMemory(zone="nonexistent_zone", data_dir=str(tmp_path))
        assert len(dm._records) == 0
