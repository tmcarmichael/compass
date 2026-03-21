"""Tests for brain.state.kill_tracker -- DefeatTracker history and matching.

Covers record_kill dedup, defeat cycle timing, cleanup/pruning,
find_unlootable_kill matching logic, clean_kill_history, defeats_in_window,
and last_kill_age.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from brain.state.kill_tracker import DefeatInfo, DefeatTracker
from core.types import Point

# ---------------------------------------------------------------------------
# record_kill
# ---------------------------------------------------------------------------


class TestRecordKill:
    def test_basic_record(self) -> None:
        tracker = DefeatTracker()
        tracker.record_kill(100, name="a_skeleton", pos=Point(10.0, 20.0, 0.0))
        assert tracker.defeats == 1
        assert len(tracker.defeat_history) == 1
        assert tracker.defeat_history[0].name == "a_skeleton"

    def test_dedup_same_spawn_within_30s(self) -> None:
        tracker = DefeatTracker()
        tracker.record_kill(100, name="a_skeleton")
        tracker.record_kill(100, name="a_skeleton")
        assert tracker.defeats == 1

    def test_no_dedup_different_spawn(self) -> None:
        tracker = DefeatTracker()
        tracker.record_kill(100, name="a_skeleton")
        tracker.record_kill(101, name="a_skeleton")
        assert tracker.defeats == 2

    def test_no_name_skips_history_entry(self) -> None:
        tracker = DefeatTracker()
        tracker.record_kill(100)
        assert tracker.defeats == 1
        assert len(tracker.defeat_history) == 0

    def test_defeat_cycle_timing(self) -> None:
        tracker = DefeatTracker()
        with patch("brain.state.kill_tracker.time") as mock_time:
            mock_time.time.return_value = 100.0
            tracker.record_kill(100, name="a")
            mock_time.time.return_value = 130.0
            tracker.record_kill(101, name="b")
        assert len(tracker.defeat_cycle_times) == 1
        assert tracker.defeat_cycle_times[0] == 30.0

    def test_defeat_cycle_pruning(self) -> None:
        """Cycle times pruned when >500 entries."""
        tracker = DefeatTracker()
        tracker.defeat_cycle_times = list(range(501))
        tracker._last_kill_time = 1.0
        with patch("brain.state.kill_tracker.time") as mock_time:
            mock_time.time.return_value = 2.0
            tracker.record_kill(999, name="x")
        # Should be pruned to last 250
        assert len(tracker.defeat_cycle_times) == 250

    def test_defeat_times_pruning(self) -> None:
        """defeat_times pruned to 1-hour window when >500 entries."""
        tracker = DefeatTracker()
        now = time.time()
        # 501 times, all within last hour
        tracker.defeat_times = [now - i for i in range(501)]
        with patch("brain.state.kill_tracker.time") as mock_time:
            mock_time.time.return_value = now
            tracker.record_kill(999, name="x")
        assert len(tracker.defeat_times) <= 502  # original + 1 new, then pruned

    def test_defeat_history_pruning(self) -> None:
        """defeat_history pruned to last 50 when >100 entries."""
        tracker = DefeatTracker()
        now = time.time()
        for i in range(101):
            tracker.defeat_history.append(DefeatInfo(spawn_id=i, name=f"npc_{i}", x=0, y=0, time=now))
        with patch("brain.state.kill_tracker.time") as mock_time:
            mock_time.time.return_value = now + 31  # past dedup window
            tracker.record_kill(999, name="final")
        assert len(tracker.defeat_history) == 50

    def test_recent_kills_pruning(self) -> None:
        """recent_kills pruned to last 50 when >100 entries."""
        tracker = DefeatTracker()
        tracker.recent_kills = [(i, 0.0) for i in range(101)]
        with patch("brain.state.kill_tracker.time") as mock_time:
            mock_time.time.return_value = 1.0
            tracker.record_kill(999, name="x")
        assert len(tracker.recent_kills) == 50


# ---------------------------------------------------------------------------
# find_unlootable_kill
# ---------------------------------------------------------------------------


class TestFindUnlootableKill:
    def test_basic_name_and_proximity_match(self) -> None:
        tracker = DefeatTracker()
        info = DefeatInfo(spawn_id=100, name="a_skeleton", x=10.0, y=10.0, time=time.time())
        tracker.defeat_history.append(info)
        result = tracker.find_unlootable_kill("a_skeleton's_corpse", Point(10.0, 10.0, 0.0))
        assert result is info

    def test_already_looted_skipped(self) -> None:
        tracker = DefeatTracker()
        info = DefeatInfo(spawn_id=100, name="a_skeleton", x=10.0, y=10.0, time=time.time(), looted=True)
        tracker.defeat_history.append(info)
        result = tracker.find_unlootable_kill("a_skeleton's_corpse", Point(10.0, 10.0, 0.0))
        assert result is None

    def test_corpse_spawn_id_match_looted_returns_none(self) -> None:
        """If corpse_spawn_id matches a LOOTED defeat, skip entirely."""
        tracker = DefeatTracker()
        info = DefeatInfo(spawn_id=100, name="a_skeleton", x=10.0, y=10.0, time=time.time(), looted=True)
        tracker.defeat_history.append(info)
        result = tracker.find_unlootable_kill(
            "a_skeleton's_corpse", Point(10.0, 10.0, 0.0), corpse_spawn_id=100
        )
        assert result is None

    def test_spawn_id_exact_match(self) -> None:
        tracker = DefeatTracker()
        info = DefeatInfo(spawn_id=100, name="a_skeleton", x=10.0, y=10.0, time=time.time())
        tracker.defeat_history.append(info)
        result = tracker.find_unlootable_kill(
            "a_skeleton's_corpse", Point(100.0, 100.0, 0.0), corpse_spawn_id=100
        )
        assert result is info

    def test_too_far_no_match(self) -> None:
        tracker = DefeatTracker()
        info = DefeatInfo(spawn_id=100, name="a_skeleton", x=10.0, y=10.0, time=time.time())
        tracker.defeat_history.append(info)
        result = tracker.find_unlootable_kill("a_skeleton's_corpse", Point(500.0, 500.0, 0.0))
        assert result is None

    def test_name_mismatch_no_match(self) -> None:
        tracker = DefeatTracker()
        info = DefeatInfo(spawn_id=100, name="a_bat", x=10.0, y=10.0, time=time.time())
        tracker.defeat_history.append(info)
        result = tracker.find_unlootable_kill("a_skeleton's_corpse", Point(10.0, 10.0, 0.0))
        assert result is None

    def test_numeric_suffix_stripped(self) -> None:
        """Names with trailing digits (e.g. a_skeleton01) match stripped base."""
        tracker = DefeatTracker()
        info = DefeatInfo(spawn_id=100, name="a_skeleton01", x=10.0, y=10.0, time=time.time())
        tracker.defeat_history.append(info)
        result = tracker.find_unlootable_kill("a_skeleton's_corpse", Point(10.0, 10.0, 0.0))
        assert result is info

    def test_no_corpse_suffix(self) -> None:
        """Corpse name without 's_corpse is matched as-is."""
        tracker = DefeatTracker()
        info = DefeatInfo(spawn_id=100, name="a_skeleton", x=10.0, y=10.0, time=time.time())
        tracker.defeat_history.append(info)
        result = tracker.find_unlootable_kill("a_skeleton", Point(10.0, 10.0, 0.0))
        assert result is info


# ---------------------------------------------------------------------------
# clean_kill_history
# ---------------------------------------------------------------------------


class TestCleanKillHistory:
    def test_removes_old_entries(self) -> None:
        tracker = DefeatTracker()
        now = time.time()
        old = DefeatInfo(spawn_id=1, name="old", x=0, y=0, time=now - 400)
        recent = DefeatInfo(spawn_id=2, name="recent", x=0, y=0, time=now - 10)
        tracker.defeat_history = [old, recent]
        tracker.clean_kill_history()
        assert len(tracker.defeat_history) == 1
        assert tracker.defeat_history[0].name == "recent"


# ---------------------------------------------------------------------------
# defeats_in_window
# ---------------------------------------------------------------------------


class TestDefeatsInWindow:
    def test_counts_recent_defeats(self) -> None:
        tracker = DefeatTracker()
        now = time.time()
        tracker.defeat_times = [now - 10, now - 100, now - 400]
        count = tracker.defeats_in_window(300.0)
        assert count == 2

    def test_empty_returns_zero(self) -> None:
        tracker = DefeatTracker()
        assert tracker.defeats_in_window() == 0


# ---------------------------------------------------------------------------
# last_kill_age
# ---------------------------------------------------------------------------


class TestLastKillAge:
    def test_no_kills_returns_large(self) -> None:
        tracker = DefeatTracker()
        assert tracker.last_kill_age() == 9999.0

    def test_recent_kill(self) -> None:
        tracker = DefeatTracker()
        tracker.defeat_times = [time.time() - 5.0]
        age = tracker.last_kill_age()
        assert 4.0 < age < 7.0
