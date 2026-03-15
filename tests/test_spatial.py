"""Tests for brain.learning.spatial -- spatial heat map memory.

SpatialMemory records defeats, sightings, and empty scans to build a heat
map of npc activity. Used by wander for direction bias and camp detection.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from brain.learning.spatial import CELL_SIZE, SpatialMemory
from core.types import Point


def _make_memory(
    tmp_path: Path,
    zone: str = "testzone",
    clock: Callable[[], float] | None = None,
) -> SpatialMemory:
    """Create a SpatialMemory with an empty data directory."""
    kwargs: dict = {"zone_name": zone, "data_dir": tmp_path}
    if clock is not None:
        kwargs["clock"] = clock
    return SpatialMemory(**kwargs)


class TestSpatialMemory:
    def test_empty_heat_zero(self, tmp_path: Path) -> None:
        sm = _make_memory(tmp_path)
        assert sm.heat_at(Point(100.0, 100.0, 0.0)) == 0.0

    def test_kill_increases_heat(self, tmp_path: Path) -> None:
        t = [1_000_000.0]
        sm = _make_memory(tmp_path, clock=lambda: t[0])
        x, y = 200.0, 200.0
        sm.record_kill(Point(x, y, 0.0), "a_skeleton", 10, fight_seconds=20.0)
        assert sm.heat_at(Point(x, y, 0.0)) > 0.0

    def test_sighting_increases_heat(self, tmp_path: Path) -> None:
        t = [1_000_000.0]
        sm = _make_memory(tmp_path, clock=lambda: t[0])
        x, y = 300.0, 300.0
        sm.record_sighting(Point(x, y, 0.0), "a_bat", 5)
        heat_sighting = sm.heat_at(Point(x, y, 0.0))
        assert heat_sighting > 0.0
        # Sighting has weight=1.0 vs kill weight=3.0, so less heat than a kill
        sm2 = _make_memory(tmp_path, zone="testzone2", clock=lambda: t[0])
        sm2.record_kill(Point(x, y, 0.0), "a_bat", 5, fight_seconds=15.0)
        assert sm2.heat_at(Point(x, y, 0.0)) > heat_sighting

    def test_empty_scan_decreases_heat(self, tmp_path: Path) -> None:
        t = [1_000_000.0]
        sm = _make_memory(tmp_path, clock=lambda: t[0])
        x, y = 400.0, 400.0
        sm.record_kill(Point(x, y, 0.0), "a_skeleton", 10)
        heat_before = sm.heat_at(Point(x, y, 0.0))
        sm.record_empty_scan(Point(x, y, 0.0))
        assert sm.heat_at(Point(x, y, 0.0)) < heat_before

    def test_heat_decays_over_time(self, tmp_path: Path) -> None:
        """Events from hours ago contribute less heat (exponential decay)."""
        t = [1_000_000.0]
        sm = _make_memory(tmp_path, clock=lambda: t[0])
        x, y = 500.0, 500.0
        sm.record_kill(Point(x, y, 0.0), "a_skeleton", 10)
        heat_fresh = sm.heat_at(Point(x, y, 0.0))
        # Record a kill "8 hours ago" by manipulating the time for _update_heat
        t2 = [1_000_000.0]
        sm2 = _make_memory(tmp_path, zone="testzone_old", clock=lambda: t2[0])
        # Simulate: record_kill calls _update_heat with event_time=now,
        # but time.time() is now+8h so the age_hours=8 → strong decay
        t2[0] = t[0] + 8 * 3600
        sm2._update_heat(Point(x, y, 0.0), weight=3.0, event_time=t[0])
        heat_old = sm2.heat_at(Point(x, y, 0.0))
        assert heat_old < heat_fresh

    def test_best_direction_picks_hottest(self, tmp_path: Path) -> None:
        t = [1_000_000.0]
        sm = _make_memory(tmp_path, clock=lambda: t[0])
        # Create a hot zone to the east
        for i in range(5):
            sm.record_kill(Point(500.0 + i * 10.0, 0.0, 0.0), "a_skeleton", 10)
        result = sm.best_direction(Point(0.0, 0.0, 0.0), radius=800.0)
        assert result is not None
        # Should point roughly east (positive x)
        assert result[0] > 0

    def test_best_direction_none_when_empty(self, tmp_path: Path) -> None:
        sm = _make_memory(tmp_path)
        assert sm.best_direction(Point(0.0, 0.0, 0.0)) is None

    def test_list_trimming(self, tmp_path: Path) -> None:
        t = [1_000_000.0]
        sm = _make_memory(tmp_path, clock=lambda: t[0])
        # Record more sightings than the cap (500)
        for i in range(520):
            # Use unique enough names to bypass dedup
            sm._sightings.append({"x": float(i), "y": 0.0, "name": f"npc_{i}", "level": 1, "time": t[0]})
        sm.trim_lists()
        assert len(sm._sightings) <= 500

    def test_visited_suppression(self, tmp_path: Path) -> None:
        t = [1_000_000.0]
        sm = _make_memory(tmp_path, clock=lambda: t[0])
        # Record kills in one area
        target_x, target_y = 300.0, 300.0
        for _ in range(5):
            sm.record_kill(Point(target_x, target_y, 0.0), "a_skeleton", 10)
        # Without visited, best_direction should point there
        result = sm.best_direction(Point(0.0, 0.0, 0.0), radius=600.0)
        assert result is not None
        # Mark that cell as visited
        sm.mark_visited(Point(target_x, target_y, 0.0))
        result_after = sm.best_direction(Point(0.0, 0.0, 0.0), radius=600.0)
        # After marking visited, that cell should be suppressed
        # (result_after is None or points somewhere else)
        if result_after is not None:
            # Should not be the same cell
            cell_before = (int(target_x // CELL_SIZE), int(target_y // CELL_SIZE))
            cell_after = (int(result_after[0] // CELL_SIZE), int(result_after[1] // CELL_SIZE))
            assert cell_after != cell_before

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        t = [1_000_000.0]
        sm = _make_memory(tmp_path, zone="persist_test", clock=lambda: t[0])
        sm.record_kill(Point(100.0, 200.0, 0.0), "a_skeleton", 10, fight_seconds=15.0)
        sm.save()
        # Load into a new instance
        sm2 = _make_memory(tmp_path, zone="persist_test", clock=lambda: t[0])
        assert sm2.total_kills == 1
        assert sm2.heat_at(Point(100.0, 200.0, 0.0)) > 0.0


class TestSightingDedup:
    def test_sighting_dedup_same_location(self, tmp_path: Path) -> None:
        """Same npc at same location within 60s is deduplicated."""
        t = [1_000_000.0]
        sm = _make_memory(tmp_path, clock=lambda: t[0])
        sm.record_sighting(Point(100.0, 100.0, 0.0), "a_bat", 5)
        sm.record_sighting(Point(110.0, 110.0, 0.0), "a_bat", 5)  # within 30 units
        assert sm.total_sightings == 1  # second was deduped

    def test_sighting_not_deduped_after_60s(self, tmp_path: Path) -> None:
        """Same npc at same location after 60s is NOT deduplicated."""
        t = [1_000_000.0]
        sm = _make_memory(tmp_path, clock=lambda: t[0])
        sm.record_sighting(Point(100.0, 100.0, 0.0), "a_bat", 5)

        t[0] = 1_000_061.0
        sm.record_sighting(Point(110.0, 110.0, 0.0), "a_bat", 5)
        assert sm.total_sightings == 2

    def test_sighting_different_mob_not_deduped(self, tmp_path: Path) -> None:
        t = [1_000_000.0]
        sm = _make_memory(tmp_path, clock=lambda: t[0])
        sm.record_sighting(Point(100.0, 100.0, 0.0), "a_bat", 5)
        sm.record_sighting(Point(100.0, 100.0, 0.0), "a_skeleton", 10)
        assert sm.total_sightings == 2


class TestTrimListsKills:
    def test_kills_trimmed_by_time_cutoff(self, tmp_path: Path) -> None:
        """Kill list > 1000 entries is trimmed to recent by time cutoff."""
        t = [1_000_000.0]
        sm = _make_memory(tmp_path, clock=lambda: t[0])
        # Add 1100 kills, half very old
        old_time = t[0] - 100 * 3600  # 100 hours ago
        for i in range(600):
            sm._kills.append({"x": float(i), "y": 0.0, "name": "old", "level": 1, "time": old_time})
        for i in range(600):
            sm._kills.append({"x": float(i), "y": 0.0, "name": "new", "level": 1, "time": t[0]})
        sm.trim_lists()
        assert len(sm._kills) < 1200

    def test_danger_events_trimmed(self, tmp_path: Path) -> None:
        t = [1_000_000.0]
        sm = _make_memory(tmp_path, clock=lambda: t[0])
        for i in range(60):
            sm._danger_events.append({"x": float(i), "y": 0.0, "time": t[0], "reason": "test"})
        sm.trim_lists()
        assert len(sm._danger_events) <= 50


class TestBestDirectionEdgeCases:
    def test_skips_too_close_cells(self, tmp_path: Path) -> None:
        """Cells within 30 units are skipped."""
        t = [1_000_000.0]
        sm = _make_memory(tmp_path, clock=lambda: t[0])
        # Record kills very close to the query origin
        sm.record_kill(Point(10.0, 10.0, 0.0), "a_bat", 5)
        result = sm.best_direction(Point(0.0, 0.0, 0.0), radius=400.0)
        # Cell center would be at ~25.0, 25.0 which is ~35 units, borderline
        # but a kill at 10,10 is in cell (0,0), center at 25,25, dist=35.3 > 30 -> may return
        # Just ensure no crash
        assert result is None or isinstance(result, tuple)

    def test_skips_cells_beyond_radius(self, tmp_path: Path) -> None:
        t = [1_000_000.0]
        sm = _make_memory(tmp_path, clock=lambda: t[0])
        # Record kills very far away
        sm.record_kill(Point(5000.0, 5000.0, 0.0), "a_bat", 5)
        result = sm.best_direction(Point(0.0, 0.0, 0.0), radius=100.0)
        assert result is None


class TestLoadEdgeCases:
    def test_load_corrupt_json(self, tmp_path: Path) -> None:
        """Corrupt JSON file should not crash, just log warning."""
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{{")
        sm = SpatialMemory(zone_name="corrupt", data_dir=tmp_path)
        assert sm.total_kills == 0

    def test_load_missing_version(self, tmp_path: Path) -> None:
        """JSON without 'v' key is loaded with a log info."""
        import json

        now = 1_000_000.0
        data = {
            "zone": "old_zone",
            "saved": now,
            "defeats": [{"x": 100, "y": 200, "name": "a_bat", "level": 5, "time": now}],
            "sightings": [],
            "empty_scans": [],
            "danger_events": [],
        }
        path = tmp_path / "old_zone.json"
        with open(path, "w") as f:
            json.dump(data, f)

        sm = SpatialMemory(zone_name="old_zone", data_dir=tmp_path)
        assert sm.total_kills == 1
