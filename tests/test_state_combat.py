"""Tests for brain.state.combat -- CombatState tracking.

Covers time_since_spell, record_spell_cast, clear_spell_cast,
and should_recast_dot with mocked spell database.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from brain.state.combat import CombatState

# ---------------------------------------------------------------------------
# spell cast tracking
# ---------------------------------------------------------------------------


class TestSpellCastTracking:
    def test_time_since_spell_never_cast(self) -> None:
        cs = CombatState()
        elapsed = cs.time_since_spell(42)
        # Should be very large (time.time() - 0)
        assert elapsed > 1_000_000

    def test_record_and_time_since(self) -> None:
        cs = CombatState()
        cs.record_spell_cast(42)
        elapsed = cs.time_since_spell(42)
        assert elapsed < 1.0

    def test_clear_spell_cast(self) -> None:
        cs = CombatState()
        cs.record_spell_cast(42)
        cs.clear_spell_cast(42)
        elapsed = cs.time_since_spell(42)
        assert elapsed > 1_000_000

    def test_clear_nonexistent_no_error(self) -> None:
        cs = CombatState()
        cs.clear_spell_cast(999)  # should not raise


# ---------------------------------------------------------------------------
# should_recast_dot
# ---------------------------------------------------------------------------


class TestShouldRecastDot:
    def test_no_spell_returns_false(self) -> None:
        cs = CombatState()
        assert cs.should_recast_dot(None) is False

    def test_no_spell_id_returns_false(self) -> None:
        spell = MagicMock()
        spell.spell_id = 0
        cs = CombatState()
        assert cs.should_recast_dot(spell) is False

    def test_recast_when_duration_exceeded(self) -> None:
        """Spell with 5-tick duration (30s). Should recast after 24s (30-6)."""
        cs = CombatState()
        spell = MagicMock()
        spell.spell_id = 100

        # Record cast 25 seconds ago
        cs.spell_cast_times[100] = time.time() - 25.0

        # Mock spell DB: 5 ticks = 30s duration
        mock_sd = MagicMock()
        mock_sd.duration_ticks = 5
        mock_sd.duration_seconds = 30.0

        mock_db = MagicMock()
        mock_db.get.return_value = mock_sd

        with patch("eq.loadout.get_spell_db", return_value=mock_db):
            assert cs.should_recast_dot(spell) is True

    def test_no_recast_when_freshly_cast(self) -> None:
        """Spell just cast 2 seconds ago should NOT need recast."""
        cs = CombatState()
        spell = MagicMock()
        spell.spell_id = 100

        cs.spell_cast_times[100] = time.time() - 2.0

        mock_sd = MagicMock()
        mock_sd.duration_ticks = 5
        mock_sd.duration_seconds = 30.0

        mock_db = MagicMock()
        mock_db.get.return_value = mock_sd

        with patch("eq.loadout.get_spell_db", return_value=mock_db):
            assert cs.should_recast_dot(spell) is False

    def test_fallback_duration_no_spell_data(self) -> None:
        """When spell not in DB, fallback to 30s recast interval."""
        cs = CombatState()
        spell = MagicMock()
        spell.spell_id = 100

        # Cast 35 seconds ago -> exceeds 30s fallback
        cs.spell_cast_times[100] = time.time() - 35.0

        mock_db = MagicMock()
        mock_db.get.return_value = None

        with patch("eq.loadout.get_spell_db", return_value=mock_db):
            assert cs.should_recast_dot(spell) is True

    def test_fallback_duration_zero_ticks(self) -> None:
        """When spell has 0 duration_ticks, fallback to 30s."""
        cs = CombatState()
        spell = MagicMock()
        spell.spell_id = 100

        cs.spell_cast_times[100] = time.time() - 35.0

        mock_sd = MagicMock()
        mock_sd.duration_ticks = 0
        mock_sd.duration_seconds = 0.0

        mock_db = MagicMock()
        mock_db.get.return_value = mock_sd

        with patch("eq.loadout.get_spell_db", return_value=mock_db):
            assert cs.should_recast_dot(spell) is True
