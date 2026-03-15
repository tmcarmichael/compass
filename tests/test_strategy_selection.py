"""Tests for routines.strategies.selection -- strategy selection logic.

Covers level-bracket defaults, con-color downgrades, danger upgrades,
pet death rate upgrades, no-fear fallback, safe blue downgrade,
forced exploration mechanism, and is_exploration_active state tracking.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from perception.combat_eval import Con
from routines.strategies import selection
from routines.strategies.selection import (
    EXPLORE_INTERVAL,
    CombatStrategy,
    is_exploration_active,
    select_strategy,
    select_strategy_with_exploration,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_exploration_state() -> None:
    """Clear module-level exploration counters and flag between tests."""
    selection._explore_counts.clear()
    selection._exploration_active = False


@pytest.fixture(autouse=True)
def _clean_exploration():
    """Reset exploration state before every test."""
    _reset_exploration_state()
    yield
    _reset_exploration_state()


def _fear_available(role):
    """Simulate a loaded fear spell for get_spell_by_role."""
    if role == "fear":
        return object()  # truthy sentinel
    return None


def _no_fear(role):
    """Simulate no fear spell loaded."""
    return None


# ---------------------------------------------------------------------------
# Level bracket defaults (no overrides)
# ---------------------------------------------------------------------------


class TestLevelBracketDefaults:
    """Baseline strategy for each level range with no con/danger context."""

    @pytest.mark.parametrize("level", [1, 4, 7])
    def test_pet_tank_low_levels(self, level):
        assert select_strategy(level) == CombatStrategy.PET_TANK

    @pytest.mark.parametrize("level", [8, 12, 15])
    def test_pet_and_dot_mid_levels(self, level):
        assert select_strategy(level) == CombatStrategy.PET_AND_DOT

    @pytest.mark.parametrize("level", [16, 30, 48])
    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    def test_fear_kite_with_fear(self, _mock, level):
        assert select_strategy(level) == CombatStrategy.FEAR_KITE

    @pytest.mark.parametrize("level", [49, 55, 60])
    def test_endgame_high_levels(self, level):
        assert select_strategy(level) == CombatStrategy.ENDGAME


class TestLevelBracketBoundaries:
    """Exact boundary values between brackets."""

    def test_boundary_7_pet_tank(self):
        assert select_strategy(7) == CombatStrategy.PET_TANK

    def test_boundary_8_pet_and_dot(self):
        assert select_strategy(8) == CombatStrategy.PET_AND_DOT

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    def test_boundary_15_pet_and_dot(self, _mock):
        assert select_strategy(15) == CombatStrategy.PET_AND_DOT

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    def test_boundary_16_fear_kite(self, _mock):
        assert select_strategy(16) == CombatStrategy.FEAR_KITE

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    def test_boundary_48_fear_kite(self, _mock):
        assert select_strategy(48) == CombatStrategy.FEAR_KITE

    def test_boundary_49_endgame(self):
        assert select_strategy(49) == CombatStrategy.ENDGAME


# ---------------------------------------------------------------------------
# Con color downgrades (L16+)
# ---------------------------------------------------------------------------


class TestConColorDowngrades:
    """Trivial npcs (LIGHT_BLUE/GREEN) at L16+ downgrade to PET_AND_DOT."""

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    @pytest.mark.parametrize("con", [Con.LIGHT_BLUE, Con.GREEN])
    def test_trivial_npc_downgrade(self, _mock, con):
        result = select_strategy(level=25, con=con)
        assert result == CombatStrategy.PET_AND_DOT

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    @pytest.mark.parametrize("con", [Con.WHITE, Con.YELLOW, Con.BLUE])
    def test_non_trivial_keeps_fear_kite(self, _mock, con):
        result = select_strategy(level=25, con=con)
        assert result == CombatStrategy.FEAR_KITE


# ---------------------------------------------------------------------------
# Safe blue downgrade (L16+)
# ---------------------------------------------------------------------------


class TestSafeBlueDowngrade:
    """Blue con with low danger at L16+ downgrades to PET_AND_DOT."""

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    def test_safe_blue_downgrades(self, _mock):
        result = select_strategy(level=30, con=Con.BLUE, danger=0.1)
        assert result == CombatStrategy.PET_AND_DOT

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    def test_blue_at_threshold_does_not_downgrade(self, _mock):
        """danger=0.2 is NOT < 0.2, so no downgrade."""
        result = select_strategy(level=30, con=Con.BLUE, danger=0.2)
        assert result == CombatStrategy.FEAR_KITE

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    def test_dangerous_blue_keeps_fear_kite(self, _mock):
        result = select_strategy(level=30, con=Con.BLUE, danger=0.5)
        assert result == CombatStrategy.FEAR_KITE

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    def test_safe_white_no_downgrade(self, _mock):
        """Low danger + WHITE con does NOT trigger the blue downgrade."""
        result = select_strategy(level=30, con=Con.WHITE, danger=0.1)
        assert result == CombatStrategy.FEAR_KITE


# ---------------------------------------------------------------------------
# Danger upgrades (L8-15)
# ---------------------------------------------------------------------------


class TestDangerUpgrades:
    """Known-dangerous npcs at L8-15 upgrade to FEAR_KITE if fear available."""

    def test_high_danger_white_with_fear_upgrades(self):
        result = select_strategy(level=12, con=Con.WHITE, danger=0.7, has_fear=True)
        assert result == CombatStrategy.FEAR_KITE

    def test_high_danger_yellow_with_fear_upgrades(self):
        result = select_strategy(level=12, con=Con.YELLOW, danger=0.7, has_fear=True)
        assert result == CombatStrategy.FEAR_KITE

    def test_high_danger_without_fear_stays_pet_and_dot(self):
        result = select_strategy(level=12, con=Con.WHITE, danger=0.7, has_fear=False)
        assert result == CombatStrategy.PET_AND_DOT

    def test_moderate_danger_stays_pet_and_dot(self):
        result = select_strategy(level=12, con=Con.WHITE, danger=0.5, has_fear=True)
        assert result == CombatStrategy.PET_AND_DOT

    def test_danger_threshold_boundary(self):
        """danger=0.6 is NOT > 0.6, so no upgrade."""
        result = select_strategy(level=12, con=Con.WHITE, danger=0.6, has_fear=True)
        assert result == CombatStrategy.PET_AND_DOT

    def test_high_danger_blue_con_no_upgrade(self):
        """Only WHITE/YELLOW qualify for danger upgrade."""
        result = select_strategy(level=12, con=Con.BLUE, danger=0.7, has_fear=True)
        assert result == CombatStrategy.PET_AND_DOT


# ---------------------------------------------------------------------------
# Pet death rate upgrades
# ---------------------------------------------------------------------------


class TestPetDeathRateUpgrades:
    """High pet death rate forces FEAR_KITE to protect pet."""

    def test_pet_death_rate_upgrade_mid_level(self):
        result = select_strategy(level=12, pet_death_rate=0.35, has_fear=True)
        assert result == CombatStrategy.FEAR_KITE

    def test_pet_death_rate_threshold_boundary(self):
        """pet_death_rate=0.30 is NOT > 0.30, so no upgrade."""
        result = select_strategy(level=12, pet_death_rate=0.30, has_fear=True)
        assert result == CombatStrategy.PET_AND_DOT

    def test_pet_death_rate_without_fear_stays(self):
        result = select_strategy(level=12, pet_death_rate=0.50, has_fear=False)
        assert result == CombatStrategy.PET_AND_DOT

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    def test_pet_death_rate_at_l16_already_fear_kite(self, _mock):
        """L16+ with fear already defaults to FEAR_KITE, so pet death rate is moot."""
        result = select_strategy(level=20, con=Con.WHITE, pet_death_rate=0.50)
        assert result == CombatStrategy.FEAR_KITE

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_no_fear)
    def test_pet_death_rate_at_l16_no_fear_stays_pet_and_dot(self, _mock):
        """L16+ without fear: pet death rate can't upgrade without fear spell."""
        result = select_strategy(level=20, con=Con.WHITE, pet_death_rate=0.50)
        assert result == CombatStrategy.PET_AND_DOT


# ---------------------------------------------------------------------------
# No fear spell fallback (L16+)
# ---------------------------------------------------------------------------


class TestNoFearFallback:
    """L16+ without a fear spell falls back to PET_AND_DOT."""

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_no_fear)
    def test_no_fear_falls_back(self, _mock):
        result = select_strategy(level=25, con=Con.WHITE)
        assert result == CombatStrategy.PET_AND_DOT

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_no_fear)
    def test_no_fear_with_danger_falls_back(self, _mock):
        result = select_strategy(level=25, con=Con.WHITE, danger=0.8)
        assert result == CombatStrategy.PET_AND_DOT


# ---------------------------------------------------------------------------
# Endgame always wins
# ---------------------------------------------------------------------------


class TestEndgameOverrides:
    """Endgame bracket ignores all other parameters."""

    def test_endgame_ignores_con(self):
        assert select_strategy(50, con=Con.GREEN) == CombatStrategy.ENDGAME

    def test_endgame_ignores_danger(self):
        assert select_strategy(55, danger=0.9) == CombatStrategy.ENDGAME

    def test_endgame_ignores_fear(self):
        assert select_strategy(60, has_fear=True) == CombatStrategy.ENDGAME

    def test_endgame_ignores_pet_death_rate(self):
        assert select_strategy(49, pet_death_rate=0.99) == CombatStrategy.ENDGAME


# ---------------------------------------------------------------------------
# Exploration mechanism
# ---------------------------------------------------------------------------


class TestExplorationMechanism:
    """Forced exploration after EXPLORE_INTERVAL encounters."""

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    def test_no_exploration_before_interval(self, _mock):
        """Under EXPLORE_INTERVAL encounters, no exploration triggered."""
        for _ in range(EXPLORE_INTERVAL - 1):
            strat, is_exp = select_strategy_with_exploration(
                "a_fire_beetle",
                level=25,
                con=Con.WHITE,
                has_fear=True,
            )
            assert strat == CombatStrategy.FEAR_KITE
            assert is_exp is False

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    def test_exploration_at_interval(self, _mock):
        """Exactly at EXPLORE_INTERVAL, an alternative strategy is forced."""
        for _ in range(EXPLORE_INTERVAL - 1):
            select_strategy_with_exploration(
                "a_fire_beetle",
                level=25,
                con=Con.WHITE,
                has_fear=True,
            )

        strat, is_exp = select_strategy_with_exploration(
            "a_fire_beetle",
            level=25,
            con=Con.WHITE,
            has_fear=True,
        )
        assert is_exp is True
        assert strat != CombatStrategy.FEAR_KITE

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    def test_exploration_resets_counter(self, _mock):
        """After exploration, counter resets and next N-1 calls are normal."""
        for _ in range(EXPLORE_INTERVAL):
            select_strategy_with_exploration(
                "a_fire_beetle",
                level=25,
                con=Con.WHITE,
                has_fear=True,
            )

        # The 20th call triggered exploration and reset. Next calls are normal.
        strat, is_exp = select_strategy_with_exploration(
            "a_fire_beetle",
            level=25,
            con=Con.WHITE,
            has_fear=True,
        )
        assert is_exp is False
        assert strat == CombatStrategy.FEAR_KITE

    def test_exploration_separate_per_entity(self):
        """Different entity names have independent counters."""
        for _ in range(EXPLORE_INTERVAL - 2):
            select_strategy_with_exploration("a_fire_beetle", level=12)

        # fire_beetle at 18, one below interval -- should NOT trigger
        _, is_exp = select_strategy_with_exploration("a_fire_beetle", level=12)
        assert is_exp is False

        # skeleton starts fresh, nowhere near interval
        _, is_exp = select_strategy_with_exploration("a_skeleton", level=12)
        assert is_exp is False

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    def test_exploration_picks_viable_alternative(self, _mock):
        """Alternative strategy is viable for the level (e.g. no ENDGAME at L25)."""
        for _ in range(EXPLORE_INTERVAL - 1):
            select_strategy_with_exploration(
                "a_fire_beetle",
                level=25,
                con=Con.WHITE,
                has_fear=True,
            )

        alt, is_exp = select_strategy_with_exploration(
            "a_fire_beetle",
            level=25,
            con=Con.WHITE,
            has_fear=True,
        )
        assert is_exp is True
        # At L25 with fear, alternative should NOT be ENDGAME
        assert alt != CombatStrategy.ENDGAME
        # PET_TANK is excluded at L16+ by viability
        assert alt != CombatStrategy.PET_TANK

    def test_exploration_low_level_no_fear_limited_alternatives(self):
        """At L8-15 without fear, PET_AND_DOT is best; alt is PET_TANK."""
        for _ in range(EXPLORE_INTERVAL - 1):
            select_strategy_with_exploration(
                "a_bat",
                level=10,
                has_fear=False,
            )

        alt, is_exp = select_strategy_with_exploration(
            "a_bat",
            level=10,
            has_fear=False,
        )
        assert is_exp is True
        # PET_TANK is viable below L16, FEAR_KITE needs fear, ENDGAME needs L49
        assert alt == CombatStrategy.PET_TANK


# ---------------------------------------------------------------------------
# is_exploration_active state
# ---------------------------------------------------------------------------


class TestIsExplorationActive:
    """Module-level exploration flag tracking."""

    @pytest.fixture(autouse=True)
    def _reset_exploration(self) -> None:
        from routines.strategies.selection import _exploration_state

        _exploration_state["active"] = False

    def test_initially_false(self):
        assert is_exploration_active() is False

    def test_false_during_normal_selection(self):
        select_strategy_with_exploration("a_beetle", level=5)
        assert is_exploration_active() is False

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    def test_true_after_exploration_triggered(self, _mock):
        for _ in range(EXPLORE_INTERVAL):
            select_strategy_with_exploration(
                "a_beetle",
                level=25,
                con=Con.WHITE,
                has_fear=True,
            )
        assert is_exploration_active() is True

    @patch("routines.strategies.selection.get_spell_by_role", side_effect=_fear_available)
    def test_cleared_after_normal_call(self, _mock):
        """Flag is cleared on the next non-exploratory call."""
        for _ in range(EXPLORE_INTERVAL):
            select_strategy_with_exploration(
                "a_beetle",
                level=25,
                con=Con.WHITE,
                has_fear=True,
            )
        assert is_exploration_active() is True

        # Next call is normal -> flag cleared
        select_strategy_with_exploration(
            "a_beetle",
            level=25,
            con=Con.WHITE,
            has_fear=True,
        )
        assert is_exploration_active() is False


# ---------------------------------------------------------------------------
# CombatStrategy enum
# ---------------------------------------------------------------------------


class TestCombatStrategyEnum:
    """Verify the enum values match expected string labels."""

    def test_values(self):
        assert CombatStrategy.PET_TANK == "pet_tank"
        assert CombatStrategy.PET_AND_DOT == "pet_and_dot"
        assert CombatStrategy.FEAR_KITE == "fear_kite"
        assert CombatStrategy.ENDGAME == "endgame"

    def test_is_str_enum(self):
        assert isinstance(CombatStrategy.PET_TANK, str)


# ---------------------------------------------------------------------------
# select_strategy_with_exploration return types
# ---------------------------------------------------------------------------


class TestSelectStrategyWithExplorationBasic:
    """Non-exploration basic behavior of select_strategy_with_exploration."""

    def test_returns_tuple(self):
        result = select_strategy_with_exploration("a_beetle", level=5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_strategy_matches_base(self):
        strat, _ = select_strategy_with_exploration("a_beetle", level=5)
        assert strat == CombatStrategy.PET_TANK

    def test_not_exploratory_first_call(self):
        _, is_exp = select_strategy_with_exploration("a_beetle", level=5)
        assert is_exp is False
