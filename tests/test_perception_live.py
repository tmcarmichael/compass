"""Tests for the perception layer requiring a live EQ client process.

All tests are marked with @pytest.mark.live and will be skipped
when eqgame.exe is not running. Designed for local Windows testing
when the EQ client is active.
"""

from __future__ import annotations

import subprocess

import pytest

# ---------------------------------------------------------------------------
# Live-process check
# ---------------------------------------------------------------------------

EQ_PID = 4532


def _eq_process_running() -> bool:
    """Return True if eqgame.exe is in the process list."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq eqgame.exe", "/NH"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "eqgame.exe" in result.stdout.lower()
    except Exception:
        return False


def _offsets_configured() -> bool:
    """Return True if perception offsets have non-zero root pointers."""
    try:
        from perception import offsets

        return offsets.PLAYER_SPAWN_PTR != 0
    except Exception:
        return False


live = pytest.mark.skipif(not _eq_process_running(), reason="Requires live EQ client")

# Tests that require both a running process AND configured offsets
live_configured = pytest.mark.skipif(
    not (_eq_process_running() and _offsets_configured()),
    reason="Requires live EQ client with configured offsets",
)


# ---------------------------------------------------------------------------
# Pure unit tests (no live process needed)
# ---------------------------------------------------------------------------


class TestReadStats:
    """Test the ReadStats helper class (pure logic, no process access)."""

    def test_initial_state(self):
        from perception.reader import ReadStats

        stats = ReadStats()
        assert stats.success == 0
        assert stats.fail == 0
        assert stats.consecutive_fail == 0
        assert stats.total == 0

    def test_initial_success_rate_is_one(self):
        from perception.reader import ReadStats

        stats = ReadStats()
        assert stats.success_rate == 1.0

    def test_record_ok(self):
        from perception.reader import ReadStats

        stats = ReadStats()
        stats.record_ok()
        assert stats.success == 1
        assert stats.total == 1
        assert stats.consecutive_fail == 0
        assert stats.success_rate == 1.0

    def test_record_fail(self):
        from perception.reader import ReadStats

        stats = ReadStats()
        stats.record_fail()
        assert stats.fail == 1
        assert stats.total == 1
        assert stats.consecutive_fail == 1
        assert stats.success_rate == 0.0

    def test_mixed_records(self):
        from perception.reader import ReadStats

        stats = ReadStats()
        stats.record_ok()
        stats.record_ok()
        stats.record_fail()
        assert stats.success == 2
        assert stats.fail == 1
        assert stats.total == 3
        assert stats.consecutive_fail == 1
        assert abs(stats.success_rate - 2 / 3) < 1e-9

    def test_consecutive_fail_resets_on_ok(self):
        from perception.reader import ReadStats

        stats = ReadStats()
        stats.record_fail()
        stats.record_fail()
        stats.record_fail()
        assert stats.consecutive_fail == 3
        stats.record_ok()
        assert stats.consecutive_fail == 0


class TestCalcBaseMana:
    """Test CharReader.calc_base_mana() static method with known values."""

    def test_zero_level(self):
        from perception.char_reader import CharReader

        assert CharReader.calc_base_mana(200, 0) == 0

    def test_zero_stat(self):
        from perception.char_reader import CharReader

        assert CharReader.calc_base_mana(0, 50) == 0

    def test_negative_inputs(self):
        from perception.char_reader import CharReader

        assert CharReader.calc_base_mana(-10, 50) == 0
        assert CharReader.calc_base_mana(200, -5) == 0

    def test_low_stat(self):
        """With wis_or_int <= 100, uses the 'low stat' formula branch."""
        from perception.char_reader import CharReader

        result = CharReader.calc_base_mana(80, 30)
        assert result > 0
        # Verify formula: mind_lesser = max(0, (80-199)//2) = 0
        # mind_factor = 80 - 0 = 80
        # stat <= 100: ((5 * (80 + 200)) // 2) * 3 * 30 // 100
        # = (5*280//2) * 3 * 30 // 100 = 700 * 90 // 100 = 630
        assert result == 630

    def test_high_stat(self):
        """With wis_or_int > 100, uses the 'high stat' formula branch."""
        from perception.char_reader import CharReader

        result = CharReader.calc_base_mana(200, 60)
        assert result > 0
        # mind_lesser = max(0, (200-199)//2) = 0
        # mind_factor = 200 - 0 = 200
        # stat > 100: ((5 * (200 + 20)) // 2) * 3 * 60 // 40
        # = (5*220//2) * 3 * 60 // 40 = 550 * 180 // 40 = 2475
        assert result == 2475

    def test_diminishing_returns_above_199(self):
        """Stats above 199 have diminishing returns via mind_lesser."""
        from perception.char_reader import CharReader

        result_200 = CharReader.calc_base_mana(200, 60)
        result_250 = CharReader.calc_base_mana(250, 60)
        # 250 should give more than 200, but less than linear
        assert result_250 > result_200
        # mind_lesser = (250-199)//2 = 25
        # mind_factor = 250 - 25 = 225
        # ((5*245)//2) * 3 * 60 // 40 = 612 * 180 // 40 = 2754
        assert result_250 == 2754

    def test_level_1_mana(self):
        """Level 1 caster with 100 int should have modest mana."""
        from perception.char_reader import CharReader

        result = CharReader.calc_base_mana(100, 1)
        assert result > 0
        assert result < 100  # reasonable for level 1


# ---------------------------------------------------------------------------
# Live tests (require eqgame.exe running)
# ---------------------------------------------------------------------------


@live
class TestMemoryReaderConstruction:
    """Test MemoryReader construction and teardown with live process."""

    def test_open_and_handle(self):
        from perception.reader import MemoryReader

        reader = MemoryReader(EQ_PID)
        try:
            assert reader._handle is not None
            assert reader._handle != 0
        finally:
            reader.close()

    def test_close(self):
        from perception.reader import MemoryReader

        reader = MemoryReader(EQ_PID)
        reader.close()
        assert reader._handle is None

    def test_double_close_safe(self):
        from perception.reader import MemoryReader

        reader = MemoryReader(EQ_PID)
        reader.close()
        reader.close()  # should not raise


@live_configured
class TestReadState:
    """Test read_state() with live process data."""

    @pytest.fixture(autouse=True)
    def _reader(self):
        from perception.reader import MemoryReader

        self.reader = MemoryReader(EQ_PID)
        yield
        self.reader.close()

    def test_basic_state(self):
        state = self.reader.read_state()
        assert state.hp_max > 0
        assert 1 <= state.level <= 65
        assert isinstance(state.name, str)
        assert len(state.name) > 0

    def test_position_non_zero(self):
        state = self.reader.read_state()
        # At least one coordinate should be non-zero if character is in world
        assert state.x != 0 or state.y != 0 or state.z != 0

    def test_state_types(self):
        state = self.reader.read_state()
        assert isinstance(state.hp_current, int)
        assert isinstance(state.hp_max, int)
        assert isinstance(state.level, int)
        assert isinstance(state.x, float)
        assert isinstance(state.y, float)
        assert isinstance(state.z, float)
        assert isinstance(state.heading, float)

    def test_spawns(self):
        state = self.reader.read_state(include_spawns=True)
        assert isinstance(state.spawns, tuple)
        # Player should always be in the spawn list
        assert len(state.spawns) >= 1

    def test_spawn_data_fields(self):
        state = self.reader.read_state(include_spawns=True)
        assert len(state.spawns) >= 1
        for spawn in state.spawns[:5]:  # check first 5
            assert isinstance(spawn.name, str)
            assert len(spawn.name) > 0
            assert isinstance(spawn.hp_current, int)
            assert isinstance(spawn.hp_max, int)
            assert isinstance(spawn.x, float)
            assert isinstance(spawn.y, float)
            assert isinstance(spawn.z, float)
            assert isinstance(spawn.spawn_id, int)
            assert spawn.spawn_id > 0


@live_configured
class TestManaReading:
    """Test mana reads with live process."""

    @pytest.fixture(autouse=True)
    def _reader(self):
        from perception.reader import MemoryReader

        self.reader = MemoryReader(EQ_PID)
        yield
        self.reader.close()

    def test_mana_values(self):
        state = self.reader.read_state()
        assert state.mana_current >= 0
        # Either both > 0 (caster) or mana_current == 0 (non-caster)
        if state.mana_max > 0:
            assert state.mana_current <= state.mana_max + 100  # allow small buffer


@live_configured
class TestBuffReading:
    """Test buff reading from live process."""

    @pytest.fixture(autouse=True)
    def _reader(self):
        from perception.reader import MemoryReader

        self.reader = MemoryReader(EQ_PID)
        yield
        self.reader.close()

    def test_read_buffs_returns_tuple(self):
        buffs = self.reader.read_buffs()
        assert isinstance(buffs, tuple)

    def test_buff_entries_are_pairs(self):
        buffs = self.reader.read_buffs()
        for entry in buffs:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            spell_id, ticks = entry
            assert isinstance(spell_id, int)
            assert isinstance(ticks, int)
            assert spell_id > 0


@live_configured
class TestCharName:
    """Test character name reading."""

    @pytest.fixture(autouse=True)
    def _reader(self):
        from perception.reader import MemoryReader

        self.reader = MemoryReader(EQ_PID)
        yield
        self.reader.close()

    def test_read_char_name(self):
        name = self.reader.read_char_name()
        assert isinstance(name, str)
        assert len(name) > 0
        assert len(name) <= 64


@live_configured
class TestHealthCheck:
    """Test health_check() with live process data."""

    @pytest.fixture(autouse=True)
    def _reader(self):
        from perception.reader import MemoryReader

        self.reader = MemoryReader(EQ_PID)
        yield
        self.reader.close()

    def test_health_check_keys(self):
        state = self.reader.read_state()
        result = self.reader.health_check(state)
        assert isinstance(result, dict)
        expected_keys = {"hp", "level", "position", "mana", "weight", "money", "xp", "buffs", "casting_mode"}
        assert expected_keys == set(result.keys())

    def test_health_check_values_are_tuples(self):
        state = self.reader.read_state()
        result = self.reader.health_check(state)
        for key, value in result.items():
            assert isinstance(value, tuple), f"health_check[{key!r}] should be a tuple"
            assert len(value) == 2, f"health_check[{key!r}] should be (value, status)"
            _val, status = value
            assert isinstance(status, str)


@live_configured
class TestSpawnReaderParse:
    """Test SpawnReader._parse_spawn_from_buffer() with live data."""

    @pytest.fixture(autouse=True)
    def _reader(self):
        from perception.reader import MemoryReader

        self.reader = MemoryReader(EQ_PID)
        yield
        self.reader.close()

    def test_parse_spawn_from_buffer(self):
        from perception.reader import SPAWN_BUF_SIZE

        # Read the player spawn base and get raw bytes
        base = self.reader._get_spawn_base()
        buf = self.reader._read_bytes(base, SPAWN_BUF_SIZE)
        assert len(buf) == SPAWN_BUF_SIZE

        fields = self.reader._parse_spawn_from_buffer(buf)
        assert isinstance(fields, dict)
        assert "name" in fields
        assert "x" in fields
        assert "y" in fields
        assert "z" in fields
        assert "hp_current" in fields
        assert "hp_max" in fields
        assert "level" in fields
        assert "spawn_id" in fields

        # Verify the parsed fields match what read_state would give us
        assert isinstance(fields["name"], str)
        assert len(fields["name"]) > 0
        assert fields["hp_max"] > 0
        assert 1 <= fields["level"] <= 65
        assert fields["spawn_id"] > 0
