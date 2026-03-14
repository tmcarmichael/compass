"""Tests for MemoryReader orchestration layer (perception/reader.py).

Bypasses Win32 process attachment by constructing MemoryReader instances
via object.__new__ and manually setting required attributes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from perception.reader import MemoryReader, ReadStats
from tests.factories import make_game_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_reader() -> MemoryReader:
    """Create a MemoryReader without Win32 process attachment."""
    reader = object.__new__(MemoryReader)
    reader._pid = 0
    reader._handle = None
    reader._observed_mana_max = 0
    reader._profile_base_cache = None
    reader._profile_chain_failed = False
    reader._profile_retry_count = 0
    reader._profile_retry_after = 0.0
    reader._weight_garbage_warned = False
    reader._vbase_cache = None
    reader._read_stats = {
        "state": ReadStats(),
        "profile_chain": ReadStats(),
        "mana": ReadStats(),
        "spawns": ReadStats(),
    }
    reader._stats_warned = set()
    return reader


# ===========================================================================
# ReadStats
# ===========================================================================


class TestReadStats:
    def test_initial_state(self):
        s = ReadStats()
        assert s.success == 0
        assert s.fail == 0
        assert s.consecutive_fail == 0
        assert s.total == 0

    def test_record_ok_increments(self):
        s = ReadStats()
        s.record_ok()
        assert s.success == 1
        assert s.total == 1
        assert s.fail == 0
        assert s.consecutive_fail == 0

    def test_record_ok_resets_consecutive_fail(self):
        s = ReadStats()
        s.record_fail()
        s.record_fail()
        assert s.consecutive_fail == 2
        s.record_ok()
        assert s.consecutive_fail == 0
        assert s.success == 1
        assert s.fail == 2
        assert s.total == 3

    def test_record_fail_increments(self):
        s = ReadStats()
        s.record_fail()
        assert s.fail == 1
        assert s.total == 1
        assert s.consecutive_fail == 1
        assert s.success == 0

    def test_record_fail_consecutive(self):
        s = ReadStats()
        for _ in range(5):
            s.record_fail()
        assert s.consecutive_fail == 5
        assert s.fail == 5
        assert s.total == 5

    def test_success_rate_empty(self):
        s = ReadStats()
        assert s.success_rate == 1.0

    def test_success_rate_all_ok(self):
        s = ReadStats()
        for _ in range(10):
            s.record_ok()
        assert s.success_rate == 1.0

    def test_success_rate_all_fail(self):
        s = ReadStats()
        for _ in range(10):
            s.record_fail()
        assert s.success_rate == 0.0

    def test_success_rate_mixed(self):
        s = ReadStats()
        for _ in range(7):
            s.record_ok()
        for _ in range(3):
            s.record_fail()
        assert s.success_rate == pytest.approx(0.7)

    def test_interleaved_ok_and_fail(self):
        s = ReadStats()
        s.record_fail()
        s.record_fail()
        s.record_ok()
        s.record_fail()
        assert s.success == 1
        assert s.fail == 3
        assert s.total == 4
        assert s.consecutive_fail == 1  # reset by record_ok, then 1 more fail


# ===========================================================================
# MemoryReader health_check
# ===========================================================================


class TestHealthCheck:
    def test_health_check_all_ok(self):
        reader = make_test_reader()
        reader._profile_base_cache = 0x1000
        state = make_game_state(
            hp_current=500,
            hp_max=1000,
            mana_current=200,
            mana_max=500,
            level=20,
            weight=50,
            money_pp=10,
        )
        result = reader.health_check(state)
        assert result["hp"][1] == "ok"
        assert result["level"][1] == "ok"
        assert result["mana"][1] == "ok"

    def test_health_check_zero_hp_max(self):
        reader = make_test_reader()
        state = make_game_state(hp_current=0, hp_max=0, mana_current=0, level=0)
        result = reader.health_check(state)
        assert result["hp"][1] == "zero"
        assert result["level"][1] == "garbage"

    def test_health_check_position_zero(self):
        reader = make_test_reader()
        state = make_game_state(x=0.0, y=0.0, z=0.0)
        result = reader.health_check(state)
        assert result["position"][1] == "zero"

    def test_health_check_position_nonzero(self):
        reader = make_test_reader()
        state = make_game_state(x=100.0, y=-200.0, z=5.0)
        result = reader.health_check(state)
        assert result["position"][1] == "ok"

    def test_health_check_mana_source_profile(self):
        reader = make_test_reader()
        reader._profile_base_cache = 0x2000
        state = make_game_state(mana_current=100, mana_max=500)
        result = reader.health_check(state)
        assert "profile" in result["mana"][0]

    def test_health_check_mana_source_charinfo(self):
        reader = make_test_reader()
        reader._profile_base_cache = None
        state = make_game_state(mana_current=100, mana_max=500)
        result = reader.health_check(state)
        assert "charinfo" in result["mana"][0]

    def test_health_check_mana_zero(self):
        reader = make_test_reader()
        state = make_game_state(mana_current=0, mana_max=500)
        result = reader.health_check(state)
        assert result["mana"][1] == "zero"

    def test_health_check_weight_zero(self):
        reader = make_test_reader()
        state = make_game_state(weight=0)
        result = reader.health_check(state)
        assert result["weight"][1] == "zero"

    def test_health_check_weight_positive(self):
        reader = make_test_reader()
        state = make_game_state(weight=75)
        result = reader.health_check(state)
        assert result["weight"][1] == "ok"

    def test_health_check_money_all_zero(self):
        reader = make_test_reader()
        state = make_game_state(money_pp=0, money_gp=0, money_sp=0, money_cp=0)
        result = reader.health_check(state)
        assert result["money"][1] == "unknown"

    def test_health_check_money_garbage(self):
        reader = make_test_reader()
        state = make_game_state(money_pp=999999)
        result = reader.health_check(state)
        assert result["money"][1] == "garbage"

    def test_health_check_money_normal(self):
        reader = make_test_reader()
        state = make_game_state(money_pp=50, money_gp=20)
        result = reader.health_check(state)
        assert result["money"][1] == "ok"

    def test_health_check_xp_garbage(self):
        reader = make_test_reader()
        # xp_pct_raw > XP_SCALE_MAX (330) is garbage
        state = make_game_state(xp_pct_raw=500)
        result = reader.health_check(state)
        assert result["xp"][1] == "garbage"

    def test_health_check_xp_ok(self):
        reader = make_test_reader()
        state = make_game_state(xp_pct_raw=165)
        result = reader.health_check(state)
        assert result["xp"][1] == "ok"

    def test_health_check_buffs_present(self):
        reader = make_test_reader()
        state = make_game_state(buffs=((100, 5), (200, 10)))
        result = reader.health_check(state)
        assert result["buffs"][1] == "ok"
        assert "2 active" in result["buffs"][0]

    def test_health_check_buffs_empty(self):
        reader = make_test_reader()
        state = make_game_state(buffs=())
        result = reader.health_check(state)
        assert result["buffs"][1] == "empty"

    def test_health_check_casting_mode(self):
        reader = make_test_reader()
        state = make_game_state(casting_mode=1)
        result = reader.health_check(state)
        assert result["casting_mode"][0] == 1
        assert result["casting_mode"][1] == "ok"

    def test_health_check_level_boundaries(self):
        reader = make_test_reader()
        # Level 1 is ok
        state = make_game_state(level=1)
        assert reader.health_check(state)["level"][1] == "ok"
        # Level 65 is ok
        state = make_game_state(level=65)
        assert reader.health_check(state)["level"][1] == "ok"
        # Level 66 is garbage
        state = make_game_state(level=66)
        assert reader.health_check(state)["level"][1] == "garbage"
        # Level 0 is garbage
        state = make_game_state(level=0)
        assert reader.health_check(state)["level"][1] == "garbage"


# ===========================================================================
# health_stats
# ===========================================================================


class TestHealthStats:
    def test_health_stats_initial(self):
        reader = make_test_reader()
        stats = reader.health_stats()
        for source in ("state", "profile_chain", "mana", "spawns"):
            assert stats[source]["success"] == 0
            assert stats[source]["fail"] == 0
            assert stats[source]["total"] == 0
            assert stats[source]["success_rate"] == 1.0
            assert stats[source]["consecutive_fail"] == 0

    def test_health_stats_after_events(self):
        reader = make_test_reader()
        reader._read_stats["state"].record_ok()
        reader._read_stats["state"].record_ok()
        reader._read_stats["state"].record_fail()
        reader._read_stats["mana"].record_fail()

        stats = reader.health_stats()
        assert stats["state"]["success"] == 2
        assert stats["state"]["fail"] == 1
        assert stats["state"]["total"] == 3
        assert stats["state"]["success_rate"] == pytest.approx(0.667, abs=0.001)
        assert stats["mana"]["fail"] == 1
        assert stats["mana"]["consecutive_fail"] == 1


# ===========================================================================
# log_health_check
# ===========================================================================


class TestLogHealthCheck:
    def test_log_health_check_does_not_raise(self):
        reader = make_test_reader()
        state = make_game_state(hp_current=500, hp_max=1000, mana_current=100, level=10)
        reader.log_health_check(state)

    def test_log_health_check_with_problems(self):
        reader = make_test_reader()
        state = make_game_state(hp_current=0, hp_max=0, level=0, mana_current=0)
        reader.log_health_check(state)


# ===========================================================================
# _check_read_health (perception watchdog)
# ===========================================================================


class TestCheckReadHealth:
    def test_no_warning_under_100_samples(self):
        reader = make_test_reader()
        # 99 total with low success rate -- should not warn
        for _ in range(9):
            reader._read_stats["state"].record_ok()
        for _ in range(90):
            reader._read_stats["state"].record_fail()
        assert reader._read_stats["state"].total == 99
        reader._check_read_health()
        assert "state" not in reader._stats_warned

    def test_warning_over_100_low_success_rate(self):
        reader = make_test_reader()
        for _ in range(80):
            reader._read_stats["state"].record_ok()
        for _ in range(21):
            reader._read_stats["state"].record_fail()
        assert reader._read_stats["state"].total == 101
        assert reader._read_stats["state"].success_rate < 0.90
        reader._check_read_health()
        assert "state" in reader._stats_warned

    def test_no_warning_above_90_percent(self):
        reader = make_test_reader()
        for _ in range(95):
            reader._read_stats["state"].record_ok()
        for _ in range(5):
            reader._read_stats["state"].record_fail()
        assert reader._read_stats["state"].total == 100
        assert reader._read_stats["state"].success_rate >= 0.90
        reader._check_read_health()
        assert "state" not in reader._stats_warned

    def test_warning_only_once(self):
        reader = make_test_reader()
        for _ in range(80):
            reader._read_stats["state"].record_ok()
        for _ in range(21):
            reader._read_stats["state"].record_fail()
        reader._check_read_health()
        assert "state" in reader._stats_warned
        # Record more failures and check again -- should not warn again
        for _ in range(10):
            reader._read_stats["state"].record_fail()
        # No assertion needed; just verifying it doesn't duplicate the warning
        reader._check_read_health()

    def test_profile_chain_3_consecutive_clears_cache(self):
        reader = make_test_reader()
        reader._profile_base_cache = 0x5000
        pc = reader._read_stats["profile_chain"]
        # Need 100+ total for check to apply
        for _ in range(100):
            pc.record_ok()
        # 3 consecutive failures
        pc.record_fail()
        pc.record_fail()
        pc.record_fail()
        assert pc.consecutive_fail == 3
        reader._check_read_health()
        # Cache should be cleared
        assert reader._profile_base_cache is None
        assert pc.consecutive_fail == 0
        assert reader._profile_retry_count == 1

    def test_profile_chain_no_clear_without_cache(self):
        reader = make_test_reader()
        reader._profile_base_cache = None
        pc = reader._read_stats["profile_chain"]
        for _ in range(100):
            pc.record_ok()
        pc.record_fail()
        pc.record_fail()
        pc.record_fail()
        reader._check_read_health()
        # No cache to clear, so retry count stays 0
        assert reader._profile_retry_count == 0

    def test_profile_chain_gives_up_after_5_retries(self):
        reader = make_test_reader()
        reader._profile_base_cache = 0x5000
        reader._profile_retry_count = 5
        pc = reader._read_stats["profile_chain"]
        for _ in range(100):
            pc.record_ok()
        pc.record_fail()
        pc.record_fail()
        pc.record_fail()
        reader._check_read_health()
        # Should give up: retry_count incremented to 6, cache NOT cleared
        assert reader._profile_retry_count == 6
        assert reader._profile_base_cache == 0x5000
        assert pc.consecutive_fail == 0


# ===========================================================================
# _read_casting_state
# ===========================================================================


class TestReadCastingState:
    def test_read_casting_state_idle(self):
        reader = make_test_reader()
        reader._read_int32 = MagicMock(return_value=0)
        casting, spell = reader._read_casting_state()
        assert casting == 0
        assert spell == -1

    def test_read_casting_state_active(self):
        reader = make_test_reader()
        # First call returns casting mode, second returns spell ID
        reader._read_int32 = MagicMock(side_effect=[1, 42])
        with patch("perception.offsets.CASTING_SPELL_ID_PTR", 0x1234):
            casting, spell = reader._read_casting_state()
        assert casting == 1
        assert spell == 42

    def test_read_casting_state_clamps_negative(self):
        reader = make_test_reader()
        reader._read_int32 = MagicMock(return_value=-1)
        casting, spell = reader._read_casting_state()
        assert casting == 0
        assert spell == -1

    def test_read_casting_state_clamps_high(self):
        reader = make_test_reader()
        reader._read_int32 = MagicMock(return_value=99)
        casting, spell = reader._read_casting_state()
        assert casting == 0
        assert spell == -1

    def test_read_casting_state_oserror(self):
        reader = make_test_reader()
        reader._read_int32 = MagicMock(side_effect=OSError("fail"))
        casting, spell = reader._read_casting_state()
        assert casting == 0
        assert spell == -1

    def test_read_casting_state_spell_id_oserror(self):
        """Spell ID read fails but casting mode succeeded."""
        reader = make_test_reader()
        reader._read_int32 = MagicMock(side_effect=[1, OSError("fail")])
        with patch("perception.offsets.CASTING_SPELL_ID_PTR", 0x1234):
            casting, spell = reader._read_casting_state()
        assert casting == 1
        assert spell == -1

    def test_read_casting_state_spell_id_out_of_range(self):
        reader = make_test_reader()
        reader._read_int32 = MagicMock(side_effect=[2, 99999])
        with patch("perception.offsets.CASTING_SPELL_ID_PTR", 0x1234):
            casting, spell = reader._read_casting_state()
        assert casting == 2
        assert spell == -1


# ===========================================================================
# _read_engine_state
# ===========================================================================


class TestReadEngineState:
    def test_read_engine_state_defaults_on_null_pointers(self):
        reader = make_test_reader()
        reader._read_pointer = MagicMock(return_value=0)
        game_mode, xp_raw, defeat_count, engine_zone_id = reader._read_engine_state()
        assert game_mode == 5  # default
        assert xp_raw == 0
        assert defeat_count == 0
        assert engine_zone_id == 0

    def test_read_engine_state_reads_game_mode(self):
        reader = make_test_reader()
        # engine_ptr returns nonzero, game_mode reads 3 (zoning)
        reader._read_pointer = MagicMock(side_effect=[0x1000, 0])
        reader._read_int32 = MagicMock(return_value=3)
        game_mode, xp_raw, defeat_count, engine_zone_id = reader._read_engine_state()
        assert game_mode == 3

    def test_read_engine_state_oserror_engine(self):
        reader = make_test_reader()
        reader._read_pointer = MagicMock(side_effect=OSError("fail"))
        game_mode, xp_raw, defeat_count, engine_zone_id = reader._read_engine_state()
        assert game_mode == 5
        assert xp_raw == 0


# ===========================================================================
# Delegation methods
# ===========================================================================


class TestDelegation:
    def test_read_money_delegates(self):
        reader = make_test_reader()
        mock_char = MagicMock()
        mock_char.read_money.return_value = (100, 50, 25, 10)
        reader._char = mock_char
        assert reader.read_money() == (100, 50, 25, 10)
        mock_char.read_money.assert_called_once()

    def test_read_buffs_delegates(self):
        reader = make_test_reader()
        mock_char = MagicMock()
        mock_char.read_buffs.return_value = ((10, 5),)
        reader._char = mock_char
        assert reader.read_buffs() == ((10, 5),)
        mock_char.read_buffs.assert_called_once()

    def test_is_buff_active_delegates(self):
        reader = make_test_reader()
        mock_char = MagicMock()
        mock_char.is_buff_active.return_value = True
        reader._char = mock_char
        assert reader.is_buff_active(42) is True
        mock_char.is_buff_active.assert_called_once_with(42)

    def test_read_loot_items_delegates(self):
        reader = make_test_reader()
        mock_inv = MagicMock()
        mock_inv.read_loot_items.return_value = (1, 2, 3)
        reader._inv = mock_inv
        assert reader.read_loot_items() == (1, 2, 3)
        mock_inv.read_loot_items.assert_called_once()

    def test_read_spawns_delegates(self):
        reader = make_test_reader()
        mock_spawn = MagicMock()
        mock_spawn.read_spawns.return_value = []
        reader._spawn = mock_spawn
        assert reader.read_spawns() == []
        mock_spawn.read_spawns.assert_called_once()

    def test_read_memorized_spells_delegates(self):
        reader = make_test_reader()
        mock_char = MagicMock()
        mock_char.read_memorized_spells.return_value = {0: 100, 1: 200}
        reader._char = mock_char
        assert reader.read_memorized_spells() == {0: 100, 1: 200}
        mock_char.read_memorized_spells.assert_called_once()

    def test_read_spellbook_delegates(self):
        reader = make_test_reader()
        mock_char = MagicMock()
        mock_char.read_spellbook.return_value = {100, 200}
        reader._char = mock_char
        assert reader.read_spellbook() == {100, 200}
        mock_char.read_spellbook.assert_called_once()

    def test_count_item_delegates(self):
        reader = make_test_reader()
        mock_inv = MagicMock()
        mock_inv.count_item.return_value = 5
        reader._inv = mock_inv
        assert reader.count_item("Bone Chips") == 5
        mock_inv.count_item.assert_called_once_with("Bone Chips")

    def test_read_target_delegates(self):
        reader = make_test_reader()
        mock_spawn = MagicMock()
        mock_spawn._read_target.return_value = None
        reader._spawn = mock_spawn
        assert reader._read_target() is None
        mock_spawn._read_target.assert_called_once()


# ===========================================================================
# __getattr__ lazy initialization
# ===========================================================================


class TestGetAttrLazyInit:
    def test_char_auto_creates(self):
        reader = object.__new__(MemoryReader)
        reader._pid = 0
        reader._handle = None
        char = reader._char
        from perception.char_reader import CharReader

        assert isinstance(char, CharReader)
        # Second access returns the same instance
        assert reader._char is char

    def test_inv_auto_creates(self):
        reader = object.__new__(MemoryReader)
        reader._pid = 0
        reader._handle = None
        inv = reader._inv
        from perception.inventory_reader import InventoryReader

        assert isinstance(inv, InventoryReader)
        assert reader._inv is inv

    def test_spawn_auto_creates(self):
        reader = object.__new__(MemoryReader)
        reader._pid = 0
        reader._handle = None
        spawn = reader._spawn
        from perception.spawn_reader import SpawnReader

        assert isinstance(spawn, SpawnReader)
        assert reader._spawn is spawn

    def test_read_stats_auto_creates(self):
        reader = object.__new__(MemoryReader)
        stats = reader._read_stats
        assert isinstance(stats, dict)
        assert set(stats.keys()) == {"state", "profile_chain", "mana", "spawns"}
        for v in stats.values():
            assert isinstance(v, ReadStats)
        # Second access returns the same dict
        assert reader._read_stats is stats

    def test_stats_warned_auto_creates(self):
        reader = object.__new__(MemoryReader)
        warned = reader._stats_warned
        assert isinstance(warned, set)
        assert len(warned) == 0
        assert reader._stats_warned is warned

    def test_profile_defaults_auto_create(self):
        reader = object.__new__(MemoryReader)
        assert reader._observed_mana_max == 0
        assert reader._profile_base_cache is None
        assert reader._profile_chain_failed is False
        assert reader._profile_retry_count == 0
        assert reader._profile_retry_after == 0.0

    def test_unknown_attribute_raises(self):
        reader = object.__new__(MemoryReader)
        with pytest.raises(AttributeError, match="no_such_attribute"):
            _ = reader.no_such_attribute
