"""Tests for perception reader modules: CharReader, SpawnReader, InventoryReader, ReadStats.

Uses mock-based testing to exercise parsing and memory-layout logic without
a live target process.
"""

from __future__ import annotations

import struct

import pytest

from perception import offsets
from perception.char_reader import CharReader
from perception.inventory_reader import InventoryReader
from perception.reader import ReadStats
from perception.spawn_reader import SPAWN_BUF_SIZE, SpawnReader
from perception.state import SpawnData

# ---------------------------------------------------------------------------
# MockReader: fake MemoryReader that serves pre-loaded memory bytes
# ---------------------------------------------------------------------------


class MockReader:
    """Minimal mock of MemoryReader for testing sub-readers.

    Pre-load ``_memory`` with ``{address: bytes_value}`` mappings; reads at
    those addresses will return the stored bytes. Reads at unmapped addresses
    raise ``OSError`` like a real process-memory read failure.
    """

    def __init__(self) -> None:
        self._profile_base_cache: int | None = 0x1000
        self._profile_chain_failed: bool = False
        self._profile_retry_after: float = 0.0
        self._observed_mana_max: int = 0
        self._weight_garbage_warned: bool = False
        self._memory: dict[int, bytes] = {}

    # -- Profile/charinfo pointers -------------------------------------------

    def _resolve_profile_base(self) -> int | None:
        return self._profile_base_cache

    def _get_charinfo_base(self) -> int | None:
        return 0x2000

    # -- Low-level read primitives -------------------------------------------

    def _read_bytes(self, address: int, size: int) -> bytes:
        if address in self._memory:
            data = self._memory[address]
            if len(data) < size:
                raise OSError(f"Mock data at 0x{address:08X} too short ({len(data)} < {size})")
            return data[:size]
        raise OSError(f"No mock data at 0x{address:08X}")

    def _read_int32(self, address: int) -> int:
        return struct.unpack("<i", self._read_bytes(address, 4))[0]

    def _read_uint32(self, address: int) -> int:
        return struct.unpack("<I", self._read_bytes(address, 4))[0]

    def _read_string(self, address: int, max_len: int = 64) -> str:
        data = self._read_bytes(address, max_len)
        end = data.find(b"\x00")
        return data[:end].decode("ascii") if end != -1 else data.decode("ascii")

    def _read_pointer(self, address: int) -> int:
        return self._read_uint32(address)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROFILE_BASE = 0x1000
CHARINFO_BASE = 0x2000


def _store_int32(mock: MockReader, address: int, value: int) -> None:
    """Write a single int32 into mock memory."""
    mock._memory[address] = struct.pack("<i", value)


def _store_uint32(mock: MockReader, address: int, value: int) -> None:
    """Write a single uint32 into mock memory."""
    mock._memory[address] = struct.pack("<I", value)


def _store_bytes(mock: MockReader, address: int, data: bytes) -> None:
    """Write a raw byte block into mock memory."""
    mock._memory[address] = data


def build_spawn_buffer(
    name: bytes = b"a_skeleton",
    x: float = 100.0,
    y: float = 200.0,
    z: float = 0.0,
    level: int = 10,
    spawn_type: int = 1,
    spawn_id: int = 42,
    hp_current: int = 100,
    hp_max: int = 100,
    **kwargs: object,
) -> bytes:
    """Build a entity struct byte buffer (752 bytes) with the given fields."""
    buf = bytearray(SPAWN_BUF_SIZE)

    # Position block: Y, X, Z, velY, velX, velZ, speed, heading
    speed = float(kwargs.get("speed", 0.0))
    heading = float(kwargs.get("heading", 128.0))
    struct.pack_into(
        "<8f",
        buf,
        offsets.Y,
        y,
        x,
        z,
        0.0,
        0.0,
        0.0,
        speed,
        heading,
    )

    # Name (null-terminated inside 64-byte field)
    buf[offsets.NAME : offsets.NAME + len(name)] = name

    # Byte fields
    buf[offsets.TYPE] = spawn_type
    buf[offsets.LEVEL] = level
    buf[offsets.HIDE] = int(kwargs.get("hide", 0))
    buf[offsets.BODY_STATE] = int(kwargs.get("body_state", ord("n")))
    buf[offsets.CLASS] = int(kwargs.get("mob_class", 0))

    # Multi-byte fields
    struct.pack_into("<I", buf, offsets.SPAWN_ID, spawn_id)
    struct.pack_into("<I", buf, offsets.OWNER_SPAWN_ID, int(kwargs.get("owner_id", 0)))
    struct.pack_into("<I", buf, offsets.RACE, int(kwargs.get("race", 0)))
    struct.pack_into("<i", buf, offsets.HP_CURRENT, hp_current)
    struct.pack_into("<i", buf, offsets.HP_MAX, hp_max)

    # Pointers
    struct.pack_into("<I", buf, offsets.NEXT, int(kwargs.get("next_ptr", 0)))
    struct.pack_into("<I", buf, offsets.PREV, int(kwargs.get("prev_ptr", 0)))
    struct.pack_into("<I", buf, offsets.ACTORCLIENT_PTR, int(kwargs.get("ac_ptr", 0)))
    struct.pack_into("<I", buf, offsets.CHARINFO_PTR, int(kwargs.get("ci_ptr", 0)))
    struct.pack_into("<i", buf, offsets.ZONE_ID, int(kwargs.get("zone_id", 0)))

    if offsets.NO_REGEN_FLAG and offsets.NO_REGEN_FLAG < SPAWN_BUF_SIZE:
        buf[offsets.NO_REGEN_FLAG] = int(kwargs.get("no_regen", 0))

    return bytes(buf)


# ═══════════════════════════════════════════════════════════════════════════
# ReadStats
# ═══════════════════════════════════════════════════════════════════════════


class TestReadStats:
    """ReadStats is a simple counter -- verify the invariants."""

    def test_initial_state(self) -> None:
        s = ReadStats()
        assert s.success == 0
        assert s.fail == 0
        assert s.total == 0
        assert s.consecutive_fail == 0
        assert s.success_rate == 1.0  # defined as 1.0 when total==0

    def test_record_ok(self) -> None:
        s = ReadStats()
        s.record_ok()
        assert s.success == 1
        assert s.fail == 0
        assert s.total == 1
        assert s.consecutive_fail == 0
        assert s.success_rate == 1.0

    def test_record_fail(self) -> None:
        s = ReadStats()
        s.record_fail()
        assert s.success == 0
        assert s.fail == 1
        assert s.total == 1
        assert s.consecutive_fail == 1
        assert s.success_rate == 0.0

    def test_consecutive_fail_resets_on_ok(self) -> None:
        s = ReadStats()
        s.record_fail()
        s.record_fail()
        s.record_fail()
        assert s.consecutive_fail == 3
        s.record_ok()
        assert s.consecutive_fail == 0
        assert s.success == 1
        assert s.fail == 3
        assert s.total == 4

    def test_mixed_sequence(self) -> None:
        s = ReadStats()
        for _ in range(7):
            s.record_ok()
        for _ in range(3):
            s.record_fail()
        assert s.total == 10
        assert s.success == 7
        assert s.fail == 3
        assert s.consecutive_fail == 3
        assert s.success_rate == pytest.approx(0.7)

    def test_success_rate_precision(self) -> None:
        s = ReadStats()
        s.record_ok()
        s.record_fail()
        s.record_ok()
        assert s.success_rate == pytest.approx(2 / 3)


# ═══════════════════════════════════════════════════════════════════════════
# CharReader -- calc_base_mana (pure math, no mock needed)
# ═══════════════════════════════════════════════════════════════════════════


class TestCalcBaseMana:
    """CharReader.calc_base_mana: pre-SoF mana formula."""

    def test_zero_level(self) -> None:
        assert CharReader.calc_base_mana(100, 0) == 0

    def test_negative_level(self) -> None:
        assert CharReader.calc_base_mana(100, -5) == 0

    def test_zero_stat(self) -> None:
        assert CharReader.calc_base_mana(0, 50) == 0

    def test_negative_stat(self) -> None:
        assert CharReader.calc_base_mana(-10, 50) == 0

    def test_both_zero(self) -> None:
        assert CharReader.calc_base_mana(0, 0) == 0

    def test_stat_100_level_50(self) -> None:
        # stat <= 100 path: mind_lesser = max(0, (100-199)//2) = 0
        # mind_factor = 100 - 0 = 100
        # ((5 * (100 + 200)) // 2) * 3 * 50 // 100
        # = (1500 // 2) * 3 * 50 // 100
        # = 750 * 150 // 100
        # = 112500 // 100 = 1125
        assert CharReader.calc_base_mana(100, 50) == 1125

    def test_stat_200_level_60(self) -> None:
        # stat > 100 path: mind_lesser = max(0, (200-199)//2) = 0
        # mind_factor = 200 - 0 = 200
        # ((5 * (200 + 20)) // 2) * 3 * 60 // 40
        # = (1100 // 2) * 3 * 60 // 40
        # = 550 * 180 // 40
        # = 99000 // 40 = 2475
        assert CharReader.calc_base_mana(200, 60) == 2475

    def test_stat_50_level_1(self) -> None:
        # stat <= 100: mind_lesser = 0, mind_factor = 50
        # ((5 * (50 + 200)) // 2) * 3 * 1 // 100
        # = (1250 // 2) * 3 // 100
        # = 625 * 3 // 100
        # = 1875 // 100 = 18
        assert CharReader.calc_base_mana(50, 1) == 18

    def test_stat_1_level_1(self) -> None:
        # Minimum positive inputs
        # mind_lesser = 0, mind_factor = 1
        # ((5 * (1 + 200)) // 2) * 3 * 1 // 100
        # = (1005 // 2) * 3 // 100
        # = 502 * 3 // 100
        # = 1506 // 100 = 15
        assert CharReader.calc_base_mana(1, 1) == 15

    def test_high_stat_diminishing_returns(self) -> None:
        # stat=300: mind_lesser = max(0, (300-199)//2) = 50
        # mind_factor = 300 - 50 = 250
        # ((5 * (250 + 20)) // 2) * 3 * 60 // 40
        # = (1350 // 2) * 3 * 60 // 40
        # = 675 * 180 // 40
        # = 121500 // 40 = 3037
        assert CharReader.calc_base_mana(300, 60) == 3037

    def test_boundary_stat_101(self) -> None:
        # Exactly at 101 -- should use the stat>100 branch
        # mind_lesser = max(0, (101-199)//2) = max(0, -49) = 0
        # mind_factor = 101
        # ((5 * (101 + 20)) // 2) * 3 * 50 // 40
        # = (605 // 2) * 3 * 50 // 40
        # = 302 * 150 // 40
        # = 45300 // 40 = 1132
        assert CharReader.calc_base_mana(101, 50) == 1132


# ═══════════════════════════════════════════════════════════════════════════
# CharReader -- mock-based methods
# ═══════════════════════════════════════════════════════════════════════════


class TestCharReaderProfileMana:
    """CharReader._read_profile_mana: mana read via profile chain."""

    def test_valid_mana(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_MANA
        _store_int32(mock, addr, 1500)
        assert cr._read_profile_mana() == 1500

    def test_mana_zero(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        _store_int32(mock, PROFILE_BASE + offsets.PROFILE_MANA, 0)
        assert cr._read_profile_mana() == 0

    def test_mana_upper_bound(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        _store_int32(mock, PROFILE_BASE + offsets.PROFILE_MANA, 50000)
        assert cr._read_profile_mana() == 50000

    def test_mana_out_of_range_returns_none(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        _store_int32(mock, PROFILE_BASE + offsets.PROFILE_MANA, 50001)
        assert cr._read_profile_mana() is None

    def test_mana_negative_returns_none(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        _store_int32(mock, PROFILE_BASE + offsets.PROFILE_MANA, -1)
        assert cr._read_profile_mana() is None

    def test_profile_unavailable_returns_none(self) -> None:
        mock = MockReader()
        mock._profile_base_cache = None
        cr = CharReader(mock)
        assert cr._read_profile_mana() is None


class TestCharReaderCharinfoMana:
    """CharReader._read_charinfo_mana: wrapper returning 0 on failure."""

    def test_returns_profile_mana_on_success(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        _store_int32(mock, PROFILE_BASE + offsets.PROFILE_MANA, 999)
        assert cr._read_charinfo_mana() == 999

    def test_returns_zero_on_failure(self) -> None:
        mock = MockReader()
        mock._profile_base_cache = None
        cr = CharReader(mock)
        assert cr._read_charinfo_mana() == 0


class TestCharReaderWeight:
    """CharReader._read_charinfo_weight: CHARINFO+0x0048."""

    def test_valid_weight(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        _store_int32(mock, CHARINFO_BASE + offsets.CHARINFO_WEIGHT, 150)
        assert cr._read_charinfo_weight() == 150

    def test_zero_weight(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        _store_int32(mock, CHARINFO_BASE + offsets.CHARINFO_WEIGHT, 0)
        assert cr._read_charinfo_weight() == 0

    def test_garbage_weight_returns_zero(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        _store_int32(mock, CHARINFO_BASE + offsets.CHARINFO_WEIGHT, 99999)
        assert cr._read_charinfo_weight() == 0

    def test_garbage_weight_sets_warning_flag(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        _store_int32(mock, CHARINFO_BASE + offsets.CHARINFO_WEIGHT, 99999)
        cr._read_charinfo_weight()
        assert mock._weight_garbage_warned is True

    def test_negative_weight_returns_zero(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        _store_int32(mock, CHARINFO_BASE + offsets.CHARINFO_WEIGHT, -5)
        assert cr._read_charinfo_weight() == 0

    def test_charinfo_unavailable_returns_zero(self) -> None:
        mock = MockReader()
        mock._get_charinfo_base = lambda: None  # type: ignore[assignment]
        cr = CharReader(mock)
        assert cr._read_charinfo_weight() == 0

    def test_read_error_returns_zero(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        # No data stored at the weight address -- OSError
        assert cr._read_charinfo_weight() == 0


class TestCharReaderMoney:
    """CharReader._read_profile_money and read_money."""

    def _store_money(self, mock: MockReader, pp: int, gp: int, sp: int, cp: int) -> None:
        addr = PROFILE_BASE + offsets.PROFILE_PP
        _store_bytes(mock, addr, struct.pack("<4i", pp, gp, sp, cp))

    def test_valid_money(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        self._store_money(mock, 100, 50, 25, 10)
        assert cr._read_profile_money() == (100, 50, 25, 10)

    def test_zero_money(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        self._store_money(mock, 0, 0, 0, 0)
        assert cr._read_profile_money() == (0, 0, 0, 0)

    def test_out_of_range_returns_none(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        self._store_money(mock, 20_000_000, 0, 0, 0)
        assert cr._read_profile_money() is None

    def test_negative_money_returns_none(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        self._store_money(mock, -1, 0, 0, 0)
        assert cr._read_profile_money() is None

    def test_profile_unavailable_returns_none(self) -> None:
        mock = MockReader()
        mock._profile_base_cache = None
        cr = CharReader(mock)
        assert cr._read_profile_money() is None

    def test_read_money_wrapper_success(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        self._store_money(mock, 10, 20, 30, 40)
        assert cr.read_money() == (10, 20, 30, 40)

    def test_read_money_wrapper_failure(self) -> None:
        mock = MockReader()
        mock._profile_base_cache = None
        cr = CharReader(mock)
        assert cr.read_money() == (0, 0, 0, 0)


class TestCharReaderBuffs:
    """CharReader buff array reads."""

    def _build_buff_array(self, buffs: list[tuple[int, int]]) -> bytes:
        """Build a 500-byte buff array from (spell_id, ticks) pairs.

        Up to 25 slots; remaining slots are zeroed.
        """
        data = bytearray(offsets.PROFILE_BUFF_COUNT * offsets.PROFILE_BUFF_SLOT_SIZE)
        for i, (spell_id, ticks) in enumerate(buffs):
            base = i * offsets.PROFILE_BUFF_SLOT_SIZE
            struct.pack_into("<i", data, base + offsets.PROFILE_BUFF_SPELL_ID_OFF, spell_id)
            struct.pack_into("<i", data, base + offsets.PROFILE_BUFF_TICKS_OFF, ticks)
        return bytes(data)

    def test_active_buffs(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_BUFF_BASE
        _store_bytes(
            mock,
            addr,
            self._build_buff_array(
                [
                    (100, 30),
                    (200, 60),
                ]
            ),
        )
        result = cr._read_profile_buffs()
        assert result == ((100, 30), (200, 60))

    def test_filters_zero_spell_id(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_BUFF_BASE
        _store_bytes(
            mock,
            addr,
            self._build_buff_array(
                [
                    (0, 10),  # empty slot
                    (150, 20),  # active
                ]
            ),
        )
        result = cr._read_profile_buffs()
        assert result == ((150, 20),)

    def test_filters_negative_spell_id(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_BUFF_BASE
        _store_bytes(
            mock,
            addr,
            self._build_buff_array(
                [
                    (-1, 10),
                    (300, 5),
                ]
            ),
        )
        result = cr._read_profile_buffs()
        assert result == ((300, 5),)

    def test_filters_spell_id_above_10000(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_BUFF_BASE
        _store_bytes(
            mock,
            addr,
            self._build_buff_array(
                [
                    (10001, 10),
                    (500, 15),
                ]
            ),
        )
        result = cr._read_profile_buffs()
        assert result == ((500, 15),)

    def test_all_empty_slots(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_BUFF_BASE
        _store_bytes(mock, addr, self._build_buff_array([]))
        result = cr._read_profile_buffs()
        assert result == ()

    def test_profile_unavailable(self) -> None:
        mock = MockReader()
        mock._profile_base_cache = None
        cr = CharReader(mock)
        assert cr._read_profile_buffs() is None

    def test_read_buffs_wrapper(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_BUFF_BASE
        _store_bytes(mock, addr, self._build_buff_array([(42, 10)]))
        assert cr.read_buffs() == ((42, 10),)

    def test_read_buffs_failure_returns_empty(self) -> None:
        mock = MockReader()
        mock._profile_base_cache = None
        cr = CharReader(mock)
        assert cr.read_buffs() == ()

    def test_get_buff_ticks_found(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_BUFF_BASE
        _store_bytes(mock, addr, self._build_buff_array([(100, 30), (200, 60)]))
        assert cr.get_buff_ticks(200) == 60

    def test_get_buff_ticks_not_found(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_BUFF_BASE
        _store_bytes(mock, addr, self._build_buff_array([(100, 30)]))
        assert cr.get_buff_ticks(999) == -1

    def test_get_buff_ticks_profile_unavailable(self) -> None:
        mock = MockReader()
        mock._profile_base_cache = None
        cr = CharReader(mock)
        assert cr.get_buff_ticks(100) == -1

    def test_is_buff_active_true(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_BUFF_BASE
        _store_bytes(mock, addr, self._build_buff_array([(100, 5)]))
        assert cr.is_buff_active(100) is True

    def test_is_buff_active_zero_ticks(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_BUFF_BASE
        _store_bytes(mock, addr, self._build_buff_array([(100, 0)]))
        assert cr.is_buff_active(100) is False

    def test_is_buff_active_missing(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_BUFF_BASE
        _store_bytes(mock, addr, self._build_buff_array([]))
        assert cr.is_buff_active(999) is False


class TestCharReaderGems:
    """CharReader._read_profile_gems and read_memorized_spells."""

    def _build_gems(self, spells: list[int]) -> bytes:
        """Build a gem array from spell IDs. Pad with -1 to 10 slots."""
        padded = spells + [-1] * (offsets.PROFILE_SPELL_GEM_COUNT - len(spells))
        return struct.pack(f"<{offsets.PROFILE_SPELL_GEM_COUNT}i", *padded)

    def test_valid_gems(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_SPELL_GEMS
        _store_bytes(mock, addr, self._build_gems([100, 200, 300]))
        result = cr._read_profile_gems()
        # 1-indexed slots
        assert result == {1: 100, 2: 200, 3: 300}

    def test_empty_gems(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_SPELL_GEMS
        _store_bytes(mock, addr, self._build_gems([]))
        result = cr._read_profile_gems()
        assert result == {}

    def test_gems_filters_invalid_ids(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_SPELL_GEMS
        _store_bytes(mock, addr, self._build_gems([-1, 0, 150, 10000]))
        result = cr._read_profile_gems()
        # -1 filtered, 0 filtered, 150 valid, 10000 filtered (must be < 10000)
        assert result == {3: 150}

    def test_profile_unavailable(self) -> None:
        mock = MockReader()
        mock._profile_base_cache = None
        cr = CharReader(mock)
        assert cr._read_profile_gems() is None

    def test_read_memorized_spells_success(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_SPELL_GEMS
        _store_bytes(mock, addr, self._build_gems([500, -1, 600]))
        result = cr.read_memorized_spells()
        assert result == {1: 500, 3: 600}

    def test_read_memorized_spells_failure(self) -> None:
        mock = MockReader()
        mock._profile_base_cache = None
        cr = CharReader(mock)
        assert cr.read_memorized_spells() == {}


class TestCharReaderSpellbook:
    """CharReader spellbook reads."""

    def _build_spellbook(self, spells_by_slot: dict[int, int]) -> bytes:
        """Build a 400-slot spellbook. Unset slots are 0xFFFFFFFF (-1 as uint32)."""
        data = bytearray(offsets.PROFILE_SPELLBOOK_SIZE * 4)
        # Fill with -1 (empty)
        for i in range(offsets.PROFILE_SPELLBOOK_SIZE):
            struct.pack_into("<i", data, i * 4, -1)
        # Set specific slots
        for slot, spell_id in spells_by_slot.items():
            struct.pack_into("<i", data, slot * 4, spell_id)
        return bytes(data)

    def test_read_spellbook(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_SPELLBOOK
        _store_bytes(mock, addr, self._build_spellbook({0: 100, 5: 200, 10: 300}))
        result = cr.read_spellbook()
        assert result == {100, 200, 300}

    def test_read_spellbook_empty(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_SPELLBOOK
        _store_bytes(mock, addr, self._build_spellbook({}))
        result = cr.read_spellbook()
        assert result == set()

    def test_read_spellbook_profile_unavailable(self) -> None:
        mock = MockReader()
        mock._profile_base_cache = None
        cr = CharReader(mock)
        assert cr.read_spellbook() == set()

    def test_spellbook_filters_invalid(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_SPELLBOOK
        _store_bytes(mock, addr, self._build_spellbook({0: 0, 1: -1, 2: 10000, 3: 500}))
        result = cr.read_spellbook()
        # 0 filtered, -1 filtered, 10000 filtered (must be < 10000), 500 valid
        assert result == {500}

    def test_spellbook_slot_for_found(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_SPELLBOOK
        _store_bytes(mock, addr, self._build_spellbook({7: 456}))
        assert cr.spellbook_slot_for(456) == 7

    def test_spellbook_slot_for_not_found(self) -> None:
        mock = MockReader()
        cr = CharReader(mock)
        addr = PROFILE_BASE + offsets.PROFILE_SPELLBOOK
        _store_bytes(mock, addr, self._build_spellbook({}))
        assert cr.spellbook_slot_for(999) is None

    def test_spellbook_slot_for_profile_unavailable(self) -> None:
        mock = MockReader()
        mock._profile_base_cache = None
        cr = CharReader(mock)
        assert cr.spellbook_slot_for(100) is None


# ═══════════════════════════════════════════════════════════════════════════
# SpawnReader
# ═══════════════════════════════════════════════════════════════════════════


class TestParseSpawnFromBuffer:
    """SpawnReader._parse_spawn_from_buffer: pure buffer parsing."""

    def _make_reader(self) -> SpawnReader:
        mock = MockReader()
        return SpawnReader(mock)

    def test_basic_npc(self) -> None:
        sr = self._make_reader()
        buf = build_spawn_buffer(
            name=b"a_skeleton",
            x=100.0,
            y=200.0,
            z=10.0,
            level=15,
            spawn_type=1,
            spawn_id=42,
            hp_current=500,
            hp_max=500,
        )
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["name"] == "a_skeleton"
        assert fields["x"] == pytest.approx(100.0)
        assert fields["y"] == pytest.approx(200.0)
        assert fields["z"] == pytest.approx(10.0)
        assert fields["level"] == 15
        assert fields["spawn_type"] == 1
        assert fields["spawn_id"] == 42
        assert fields["hp_current"] == 500
        assert fields["hp_max"] == 500
        assert fields["body_state"] == "n"
        assert fields["heading"] == pytest.approx(128.0)
        assert fields["speed"] == pytest.approx(0.0)

    def test_player_type(self) -> None:
        sr = self._make_reader()
        buf = build_spawn_buffer(spawn_type=0, name=b"Soandso", spawn_id=1)
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["spawn_type"] == 0
        assert fields["name"] == "Soandso"

    def test_dead_body_state(self) -> None:
        sr = self._make_reader()
        buf = build_spawn_buffer(body_state=ord("d"))
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["body_state"] == "d"

    def test_feign_body_state(self) -> None:
        sr = self._make_reader()
        buf = build_spawn_buffer(body_state=ord("f"))
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["body_state"] == "f"

    def test_invalid_body_state_defaults_to_n(self) -> None:
        sr = self._make_reader()
        buf = build_spawn_buffer(body_state=0xFF)
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["body_state"] == "n"

    def test_owner_id(self) -> None:
        sr = self._make_reader()
        buf = build_spawn_buffer(owner_id=999)
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["owner_id"] == 999

    def test_race(self) -> None:
        sr = self._make_reader()
        buf = build_spawn_buffer(race=130)
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["race"] == 130

    def test_mob_class(self) -> None:
        sr = self._make_reader()
        buf = build_spawn_buffer(mob_class=5)
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["mob_class"] == 5

    def test_hide_flag(self) -> None:
        sr = self._make_reader()
        buf = build_spawn_buffer(hide=1)
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["hide"] == 1

    def test_pointers_in_parsed_fields(self) -> None:
        sr = self._make_reader()
        buf = build_spawn_buffer(
            next_ptr=0xAAAA,
            prev_ptr=0xBBBB,
            ac_ptr=0xCCCC,
            ci_ptr=0xDDDD,
            zone_id=50,
        )
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["_next_ptr"] == 0xAAAA
        assert fields["_prev_ptr"] == 0xBBBB
        assert fields["_actorclient_ptr"] == 0xCCCC
        assert fields["_charinfo_ptr"] == 0xDDDD
        assert fields["_zone_id"] == 50

    def test_no_regen_flag(self) -> None:
        sr = self._make_reader()
        buf = build_spawn_buffer(no_regen=1)
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["_no_regen_flag"] == 1

    def test_velocity_fields(self) -> None:
        sr = self._make_reader()
        # Velocity fields are at the position block; default build_spawn_buffer
        # sets them to 0.0
        buf = build_spawn_buffer()
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["velocity_y"] == pytest.approx(0.0)
        assert fields["velocity_x"] == pytest.approx(0.0)
        assert fields["velocity_z"] == pytest.approx(0.0)

    def test_negative_hp(self) -> None:
        sr = self._make_reader()
        buf = build_spawn_buffer(hp_current=-10, hp_max=100)
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["hp_current"] == -10
        assert fields["hp_max"] == 100

    def test_zero_spawn_id(self) -> None:
        sr = self._make_reader()
        buf = build_spawn_buffer(spawn_id=0)
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["spawn_id"] == 0

    def test_long_name_truncated(self) -> None:
        sr = self._make_reader()
        # Name field is 64 bytes; use a name exactly at the limit
        long_name = b"a" * 63 + b"\x00"
        buf = build_spawn_buffer(name=long_name)
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["name"] == "a" * 63


class TestSpawnFromParsed:
    """SpawnReader._spawn_from_parsed strips internal keys."""

    def test_strips_internal_keys(self) -> None:
        mock = MockReader()
        sr = SpawnReader(mock)
        buf = build_spawn_buffer(
            name=b"test_mob",
            x=10.0,
            y=20.0,
            z=0.0,
            level=5,
            spawn_type=1,
            spawn_id=99,
            hp_current=50,
            hp_max=100,
            next_ptr=0x1234,
            prev_ptr=0x5678,
            ac_ptr=0x9ABC,
            ci_ptr=0xDEF0,
            zone_id=10,
            no_regen=1,
        )
        fields = sr._parse_spawn_from_buffer(buf)
        spawn = SpawnReader._spawn_from_parsed(fields)
        assert isinstance(spawn, SpawnData)
        assert spawn.spawn_id == 99
        assert spawn.name == "test_mob"
        assert spawn.x == pytest.approx(10.0)
        assert spawn.y == pytest.approx(20.0)
        assert spawn.level == 5
        assert spawn.spawn_type == 1
        assert spawn.hp_current == 50
        assert spawn.hp_max == 100
        # Internal keys must not be present on SpawnData
        assert not hasattr(spawn, "_next_ptr")
        assert not hasattr(spawn, "_prev_ptr")
        assert not hasattr(spawn, "_actorclient_ptr")
        assert not hasattr(spawn, "_charinfo_ptr")
        assert not hasattr(spawn, "_zone_id")

    def test_spawn_data_properties(self) -> None:
        mock = MockReader()
        sr = SpawnReader(mock)
        # NPC
        buf = build_spawn_buffer(spawn_type=1)
        fields = sr._parse_spawn_from_buffer(buf)
        spawn = SpawnReader._spawn_from_parsed(fields)
        assert spawn.is_npc is True
        assert spawn.is_player is False

        # Player
        buf = build_spawn_buffer(spawn_type=0)
        fields = sr._parse_spawn_from_buffer(buf)
        spawn = SpawnReader._spawn_from_parsed(fields)
        assert spawn.is_npc is False
        assert spawn.is_player is True


class TestSpawnReaderMultipleBuffers:
    """Ensure parsing works with diverse inputs."""

    def test_corpse_type(self) -> None:
        mock = MockReader()
        sr = SpawnReader(mock)
        buf = build_spawn_buffer(
            name=b"a_skeleton_corpse",
            spawn_type=2,
            body_state=ord("d"),
            hp_current=0,
            hp_max=100,
        )
        fields = sr._parse_spawn_from_buffer(buf)
        spawn = SpawnReader._spawn_from_parsed(fields)
        assert spawn.spawn_type == 2
        assert spawn.body_state == "d"
        assert spawn.hp_current == 0

    def test_mounted_body_state(self) -> None:
        mock = MockReader()
        sr = SpawnReader(mock)
        buf = build_spawn_buffer(body_state=ord("o"))
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["body_state"] == "o"

    def test_invis_body_state(self) -> None:
        mock = MockReader()
        sr = SpawnReader(mock)
        buf = build_spawn_buffer(body_state=ord("i"))
        fields = sr._parse_spawn_from_buffer(buf)
        assert fields["body_state"] == "i"


# ═══════════════════════════════════════════════════════════════════════════
# InventoryReader
# ═══════════════════════════════════════════════════════════════════════════


class TestInventoryReaderLoot:
    """InventoryReader.read_loot_items: metadata read from loot window."""

    def test_loot_window_closed(self) -> None:
        mock = MockReader()
        ir = InventoryReader(mock)
        # LOOT_WND_PTR resolves to 0 (window closed)
        _store_uint32(mock, offsets.LOOT_WND_PTR, 0)
        assert ir.read_loot_items() == ()

    def test_loot_window_read_error(self) -> None:
        mock = MockReader()
        ir = InventoryReader(mock)
        # No data at LOOT_WND_PTR -- OSError
        assert ir.read_loot_items() == ()

    def test_loot_items_with_valid_slots(self) -> None:
        mock = MockReader()
        ir = InventoryReader(mock)
        loot_wnd = 0x50000
        _store_uint32(mock, offsets.LOOT_WND_PTR, loot_wnd)
        # Set up 3 slots with valid metadata values, rest with -1
        for slot in range(offsets.LOOT_WND_ITEM_SLOTS):
            off = offsets.LOOT_WND_METADATA_OFFSET + slot * 4
            if slot < 3:
                _store_int32(mock, loot_wnd + off, slot * 10)
            else:
                _store_int32(mock, loot_wnd + off, -1)
        result = ir.read_loot_items()
        assert result == (0, 10, 20)

    def test_loot_items_all_empty(self) -> None:
        mock = MockReader()
        ir = InventoryReader(mock)
        loot_wnd = 0x50000
        _store_uint32(mock, offsets.LOOT_WND_PTR, loot_wnd)
        for slot in range(offsets.LOOT_WND_ITEM_SLOTS):
            off = offsets.LOOT_WND_METADATA_OFFSET + slot * 4
            _store_int32(mock, loot_wnd + off, -1)
        result = ir.read_loot_items()
        assert result == ()


class TestInventoryReaderLootDeep:
    """InventoryReader.read_loot_items_deep: CONTENTS->ITEMINFO chain."""

    def test_loot_window_closed(self) -> None:
        mock = MockReader()
        ir = InventoryReader(mock)
        _store_uint32(mock, offsets.LOOT_WND_PTR, 0)
        assert ir.read_loot_items_deep() == []

    def test_loot_deep_with_items(self) -> None:
        mock = MockReader()
        ir = InventoryReader(mock)
        loot_wnd = 0x50000
        _store_uint32(mock, offsets.LOOT_WND_PTR, loot_wnd)

        # Set up slot 0 with a valid chain (pointers must be > 0x10000)
        contents_ptr = 0x60000
        iteminfo_ptr = 0x70000
        contents_addr = loot_wnd + offsets.LOOT_WND_CONTENTS_OFFSET + 0 * 4
        _store_uint32(mock, contents_addr, contents_ptr)
        _store_uint32(mock, contents_ptr + offsets.CONTENTS_ITEMINFO_PTR, iteminfo_ptr)

        # Item name and ID
        name_bytes = b"Rusty Sword\x00" + b"\x00" * 52  # 64 bytes total
        _store_bytes(mock, iteminfo_ptr + offsets.ITEMINFO_NAME, name_bytes)
        _store_uint32(mock, iteminfo_ptr + offsets.ITEMINFO_ITEM_NUMBER, 12345)

        # Other slots: null pointers
        for slot in range(1, offsets.LOOT_WND_ITEM_SLOTS):
            slot_addr = loot_wnd + offsets.LOOT_WND_CONTENTS_OFFSET + slot * 4
            _store_uint32(mock, slot_addr, 0)

        result = ir.read_loot_items_deep()
        assert len(result) == 1
        assert result[0] == (0, "Rusty Sword", 12345)

    def test_loot_deep_skips_null_contents(self) -> None:
        mock = MockReader()
        ir = InventoryReader(mock)
        loot_wnd = 0x50000
        _store_uint32(mock, offsets.LOOT_WND_PTR, loot_wnd)
        # All slots are null pointers
        for slot in range(offsets.LOOT_WND_ITEM_SLOTS):
            slot_addr = loot_wnd + offsets.LOOT_WND_CONTENTS_OFFSET + slot * 4
            _store_uint32(mock, slot_addr, 0)
        result = ir.read_loot_items_deep()
        assert result == []

    def test_loot_deep_skips_low_pointer(self) -> None:
        mock = MockReader()
        ir = InventoryReader(mock)
        loot_wnd = 0x50000
        _store_uint32(mock, offsets.LOOT_WND_PTR, loot_wnd)
        # Slot 0 has a low pointer (< 0x10000), should be skipped
        slot_addr = loot_wnd + offsets.LOOT_WND_CONTENTS_OFFSET
        _store_uint32(mock, slot_addr, 0x0001)
        # Fill remaining
        for slot in range(1, offsets.LOOT_WND_ITEM_SLOTS):
            sa = loot_wnd + offsets.LOOT_WND_CONTENTS_OFFSET + slot * 4
            _store_uint32(mock, sa, 0)
        result = ir.read_loot_items_deep()
        assert result == []


class TestInventoryReaderInventory:
    """InventoryReader.read_inventory: bag reading via profile chain."""

    def test_profile_unavailable(self) -> None:
        mock = MockReader()
        mock._profile_base_cache = None
        ir = InventoryReader(mock)
        assert ir.read_inventory() == []

    def test_empty_bags(self) -> None:
        mock = MockReader()
        ir = InventoryReader(mock)
        # All bag slot pointers are 0
        for bag in range(offsets.PROFILE_BAG_COUNT):
            bag_off = offsets.PROFILE_BAG_START + bag * offsets.PROFILE_BAG_STRIDE
            _store_uint32(mock, PROFILE_BASE + bag_off, 0)
        result = ir.read_inventory()
        assert result == []


class TestInventoryReaderCountItem:
    """InventoryReader.count_item: total quantity of a named item."""

    def test_count_item_empty_inventory(self) -> None:
        mock = MockReader()
        mock._profile_base_cache = None
        ir = InventoryReader(mock)
        assert ir.count_item("Bone Chips") == 0

    def test_count_item_case_insensitive(self) -> None:
        mock = MockReader()
        ir = InventoryReader(mock)
        # Patch read_inventory to return test data
        ir.read_inventory = lambda: [  # type: ignore[assignment]
            ("Bone Chips", 1001, 5),
            ("bone chips", 1001, 3),
            ("Bat Wing", 1002, 2),
        ]
        assert ir.count_item("Bone Chips") == 8
        assert ir.count_item("bone chips") == 8
        assert ir.count_item("BONE CHIPS") == 8
        assert ir.count_item("Bat Wing") == 2
        assert ir.count_item("Missing Item") == 0
