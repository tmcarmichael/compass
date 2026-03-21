"""Tests for runtime.zone_progression -- level-based zone sequencing.

check_zone_progression() determines whether a player should move to the
next zone based on level vs. the configured progression table.
Tests pass the table via the `progression` parameter (no monkeypatching).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from runtime.zone_progression import check_zone_progression

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TABLE = [
    ("gfaydark", 1, 10),
    ("crushbone", 8, 15),
    ("lfaydark", 13, 20),
    ("unrest", 18, 30),
]


def _mock_ctx() -> MagicMock:
    """Minimal AgentContext mock (unused fields are fine as MagicMock)."""
    return MagicMock()


def _check(zone: str, level: int, table: list | None = None) -> str | None:
    return check_zone_progression(_mock_ctx(), zone, player_level=level, progression=table or _TABLE)


# ---------------------------------------------------------------------------
# check_zone_progression
# ---------------------------------------------------------------------------


class TestCheckZoneProgression:
    def test_returns_none_when_zone_not_in_table(self) -> None:
        assert _check("freportn", 10) is None

    def test_stays_in_zone_when_level_within_range(self) -> None:
        assert _check("gfaydark", 8) is None

    def test_stays_at_max_level_boundary(self) -> None:
        assert _check("gfaydark", 10) is None

    def test_advances_to_next_zone_when_outleveled(self) -> None:
        assert _check("gfaydark", 11) == "crushbone"

    def test_skips_zones_that_dont_fit(self) -> None:
        assert _check("crushbone", 21) == "unrest"

    def test_returns_none_when_no_viable_zone_left(self) -> None:
        assert _check("unrest", 31) is None

    def test_returns_none_at_last_zone_within_range(self) -> None:
        assert _check("unrest", 25) is None


class TestZoneProgressionTableEmpty:
    def test_empty_table_returns_none(self) -> None:
        assert _check("gfaydark", 10, table=[]) is None
