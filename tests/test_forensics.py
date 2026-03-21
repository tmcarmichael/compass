"""Tests for util.forensics  -- ring buffer for incident telemetry.

The ForensicsBuffer holds the last N ticks of compact state in memory.
On death or invariant breach, it flushes to disk as JSONL. These tests
verify capacity, flush mechanics, and entry structure.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.factories import make_game_state
from util.forensics import ForensicsBuffer


def _fill_buffer(buf: ForensicsBuffer, n: int) -> None:
    """Record n ticks with incrementing tick IDs."""
    state = make_game_state(hp_current=900, hp_max=1000, mana_current=400)
    for i in range(n):
        buf.record_tick(tick_id=i, state=state, active_routine="combat")


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------


class TestBasicOps:
    def test_empty_snapshot(self, tmp_path: Path) -> None:
        buf = ForensicsBuffer("sess-1", tmp_path)
        assert buf.snapshot() == []

    def test_record_adds_entry(self, tmp_path: Path) -> None:
        buf = ForensicsBuffer("sess-1", tmp_path)
        state = make_game_state()
        buf.record_tick(0, state, active_routine="rest")
        snap = buf.snapshot()
        assert len(snap) == 1

    def test_snapshot_returns_copy(self, tmp_path: Path) -> None:
        buf = ForensicsBuffer("sess-1", tmp_path)
        _fill_buffer(buf, 3)
        snap = buf.snapshot()
        snap.clear()
        assert len(buf.snapshot()) == 3


# ---------------------------------------------------------------------------
# Ring buffer capacity
# ---------------------------------------------------------------------------


class TestCapacity:
    def test_max_entries_enforced(self, tmp_path: Path) -> None:
        cap = 10
        buf = ForensicsBuffer("sess-1", tmp_path, max_entries=cap)
        _fill_buffer(buf, cap + 5)
        assert len(buf.snapshot()) == cap

    def test_oldest_entries_dropped(self, tmp_path: Path) -> None:
        cap = 5
        buf = ForensicsBuffer("sess-1", tmp_path, max_entries=cap)
        _fill_buffer(buf, 10)
        snap = buf.snapshot()
        tick_ids = [e["tick"] for e in snap]
        assert tick_ids[0] >= 5  # oldest entries (0-4) should be gone


# ---------------------------------------------------------------------------
# Flush to disk
# ---------------------------------------------------------------------------


class TestFlush:
    def test_writes_valid_jsonl(self, tmp_path: Path) -> None:
        buf = ForensicsBuffer("sess-1", tmp_path)
        _fill_buffer(buf, 5)
        buf.flush("death")

        jsonl_files = list(tmp_path.glob("*forensics*"))
        assert len(jsonl_files) == 1

        lines = jsonl_files[0].read_text().strip().split("\n")
        for line in lines:
            json.loads(line)  # should not raise

    def test_flush_clears_buffer(self, tmp_path: Path) -> None:
        buf = ForensicsBuffer("sess-1", tmp_path)
        _fill_buffer(buf, 5)
        buf.flush("death")
        assert len(buf.snapshot()) == 0

    def test_flush_noop_when_empty(self, tmp_path: Path) -> None:
        buf = ForensicsBuffer("sess-1", tmp_path)
        buf.flush("death")
        # Should not create a file for empty buffer
        jsonl_files = list(tmp_path.glob("*forensics*"))
        assert len(jsonl_files) == 0


# ---------------------------------------------------------------------------
# Entry structure
# ---------------------------------------------------------------------------


class TestEntryStructure:
    def test_expected_fields(self, tmp_path: Path) -> None:
        buf = ForensicsBuffer("sess-1", tmp_path)
        state = make_game_state(
            hp_current=800, hp_max=1000, mana_current=300, x=100.5, y=200.3, heading=128.0
        )
        buf.record_tick(42, state, active_routine="pull", engaged=True)
        entry = buf.snapshot()[0]

        assert entry["tick"] == 42
        assert "ts" in entry
        assert entry["hp"] == 800
        assert entry["routine"] == "pull"
        assert entry["engaged"] is True

    def test_target_fields_when_present(self, tmp_path: Path) -> None:
        from tests.factories import make_spawn

        buf = ForensicsBuffer("sess-1", tmp_path)
        target = make_spawn(name="a_bat", hp_current=50)
        state = make_game_state(target=target)
        buf.record_tick(10, state)
        entry = buf.snapshot()[0]

        assert entry["tgt"] == "a_bat"
        assert entry["tgt_hp"] == 50

    def test_no_target_fields(self, tmp_path: Path) -> None:
        buf = ForensicsBuffer("sess-1", tmp_path)
        state = make_game_state()
        buf.record_tick(10, state)
        entry = buf.snapshot()[0]

        assert entry["tgt"] == ""
        assert entry["tgt_hp"] == 0


# ---------------------------------------------------------------------------
# compact_world
# ---------------------------------------------------------------------------


class TestCompactWorld:
    def test_empty_spawns(self) -> None:
        from util.forensics import compact_world

        state = make_game_state()
        result = compact_world(state)
        assert result["npcs"] == []
        assert result["players"] == 0

    def test_npcs_sorted_by_distance(self) -> None:
        from tests.factories import make_spawn
        from util.forensics import compact_world

        far_npc = make_spawn(spawn_id=1, name="a_skeleton", x=300.0, y=0.0, hp_current=100)
        near_npc = make_spawn(spawn_id=2, name="a_bat", x=50.0, y=0.0, hp_current=80)
        state = make_game_state(x=0.0, y=0.0, spawns=[far_npc, near_npc])
        result = compact_world(state)

        assert len(result["npcs"]) == 2
        assert result["npcs"][0]["name"] == "a_bat"  # closer
        assert result["npcs"][1]["name"] == "a_skeleton"  # farther

    def test_dead_npcs_excluded(self) -> None:
        from tests.factories import make_spawn
        from util.forensics import compact_world

        dead_npc = make_spawn(spawn_id=1, name="a_skeleton", x=50.0, y=0.0, hp_current=0)
        state = make_game_state(spawns=[dead_npc])
        result = compact_world(state)

        assert len(result["npcs"]) == 0

    def test_players_counted(self) -> None:
        from tests.factories import make_spawn
        from util.forensics import compact_world

        player = make_spawn(spawn_id=1, name="SomePlayer", x=50.0, y=0.0, spawn_type=0)
        npc = make_spawn(spawn_id=2, name="a_bat", x=100.0, y=0.0, hp_current=50)
        state = make_game_state(spawns=[player, npc])
        result = compact_world(state)

        assert result["players"] == 1
        assert len(result["npcs"]) == 1

    def test_max_npcs_respected(self) -> None:
        from tests.factories import make_spawn
        from util.forensics import compact_world

        npcs = [
            make_spawn(spawn_id=i, name=f"npc_{i}", x=float(i * 10), y=0.0, hp_current=100)
            for i in range(1, 12)
        ]
        state = make_game_state(spawns=npcs)
        result = compact_world(state, max_npcs=3)

        assert len(result["npcs"]) == 3


# ---------------------------------------------------------------------------
# Flush mechanics
# ---------------------------------------------------------------------------


class TestFlushHeaderAndMultiple:
    def test_header_contains_trigger_and_session(self, tmp_path: Path) -> None:
        buf = ForensicsBuffer("sess-42", tmp_path)
        _fill_buffer(buf, 3)
        buf.flush("invariant:mana")

        jsonl_file = tmp_path / "sess-42_forensics.jsonl"
        lines = jsonl_file.read_text().strip().split("\n")
        header = json.loads(lines[0])

        assert header["event"] == "forensics_dump"
        assert header["trigger"] == "invariant:mana"
        assert header["session_id"] == "sess-42"
        assert header["entries"] == 3
        assert header["flush_number"] == 1

    def test_multiple_flushes_append(self, tmp_path: Path) -> None:
        buf = ForensicsBuffer("sess-1", tmp_path)
        _fill_buffer(buf, 2)
        buf.flush("death")
        _fill_buffer(buf, 3)
        buf.flush("invariant:x")

        jsonl_file = tmp_path / "sess-1_forensics.jsonl"
        lines = jsonl_file.read_text().strip().split("\n")
        # First flush: 1 header + 2 entries = 3 lines
        # Second flush: 1 header + 3 entries = 4 lines
        assert len(lines) == 7

    def test_flush_count_increments(self, tmp_path: Path) -> None:
        buf = ForensicsBuffer("sess-1", tmp_path)
        _fill_buffer(buf, 1)
        buf.flush("a")
        assert buf._flush_count == 1
        _fill_buffer(buf, 1)
        buf.flush("b")
        assert buf._flush_count == 2


# ---------------------------------------------------------------------------
# Close (shutdown flush)
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_flushes_remaining(self, tmp_path: Path) -> None:
        buf = ForensicsBuffer("sess-1", tmp_path)
        _fill_buffer(buf, 5)
        buf.close()

        jsonl_file = tmp_path / "sess-1_forensics.jsonl"
        assert jsonl_file.exists()
        lines = jsonl_file.read_text().strip().split("\n")
        header = json.loads(lines[0])
        assert header["trigger"] == "shutdown"

    def test_close_empty_buffer_no_file(self, tmp_path: Path) -> None:
        buf = ForensicsBuffer("sess-1", tmp_path)
        buf.close()

        jsonl_files = list(tmp_path.glob("*forensics*"))
        assert len(jsonl_files) == 0
