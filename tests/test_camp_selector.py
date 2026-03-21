"""Tests for runtime.camp_selector -- camp scoring and selection logic.

score_camp() ranks camps by level-fit + proximity. select_camp() picks the
best camp from a list. Tests cover level-range matching, fallback penalties,
linear camp midpoint computation, and edge cases.
"""

from __future__ import annotations

import pytest

from core.constants import LEVEL_RANGE_PENALTY
from core.types import Point
from runtime.camp_selector import FALLBACK_PENALTY, score_camp, select_camp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _camp(
    name: str = "camp_a",
    cx: float = 100.0,
    cy: float = 100.0,
    level_range: list[int] | None = None,
    fallback: bool = False,
    camp_type: str = "circular",
    patrol_waypoints: list[dict] | None = None,
) -> dict:
    """Construct a minimal camp dict."""
    c: dict = {
        "name": name,
        "center": {"x": cx, "y": cy},
    }
    if level_range is not None:
        c["level_range"] = level_range
    if fallback:
        c["fallback"] = True
    if camp_type != "circular":
        c["camp_type"] = camp_type
    if patrol_waypoints is not None:
        c["patrol_waypoints"] = patrol_waypoints
    return c


# ---------------------------------------------------------------------------
# score_camp
# ---------------------------------------------------------------------------


class TestScoreCamp:
    """score_camp returns a float: lower is better."""

    def test_no_level_range_scores_by_distance(self) -> None:
        camp = _camp(cx=100, cy=0)
        score = score_camp(camp, Point(0.0, 0.0, 0.0), player_level=10)
        assert score == pytest.approx(100.0, abs=0.1)

    def test_within_level_range_scores_distance_only(self) -> None:
        camp = _camp(cx=100, cy=0, level_range=[5, 15])
        score = score_camp(camp, Point(0.0, 0.0, 0.0), player_level=10)
        assert score == pytest.approx(100.0, abs=0.1)

    def test_outside_level_range_adds_gap_penalty(self) -> None:
        camp = _camp(cx=100, cy=0, level_range=[5, 8])
        # Player level 10, range max 8 -> gap = 2
        score = score_camp(camp, Point(0.0, 0.0, 0.0), player_level=10)
        expected = 100.0 + 2 * LEVEL_RANGE_PENALTY
        assert score == pytest.approx(expected, abs=0.1)

    def test_fallback_camp_gets_penalty(self) -> None:
        normal = _camp(cx=100, cy=0, level_range=[5, 15])
        fb = _camp(cx=100, cy=0, level_range=[5, 15], fallback=True)
        score_normal = score_camp(normal, Point(0.0, 0.0, 0.0), player_level=10)
        score_fb = score_camp(fb, Point(0.0, 0.0, 0.0), player_level=10)
        assert score_fb - score_normal == pytest.approx(FALLBACK_PENALTY)

    def test_player_level_zero_ignores_level_range(self) -> None:
        camp = _camp(cx=100, cy=0, level_range=[5, 15])
        score = score_camp(camp, Point(0.0, 0.0, 0.0), player_level=0)
        # level_range present but player_level=0 -> treated as no level check
        assert score == pytest.approx(100.0, abs=0.1)

    def test_linear_camp_midpoint_odd_waypoints(self) -> None:
        """LINEAR camp with no explicit center uses patrol midpoint."""
        wps = [{"x": 0.0, "y": 0.0}, {"x": 100.0, "y": 0.0}, {"x": 200.0, "y": 0.0}]
        camp = _camp(cx=0, cy=0, camp_type="linear", patrol_waypoints=wps)
        # Midpoint = wps[1] = (100, 0). Player at origin -> dist = 100
        score = score_camp(camp, Point(0.0, 0.0, 0.0), player_level=10)
        assert score == pytest.approx(100.0, abs=0.1)

    def test_linear_camp_midpoint_even_waypoints(self) -> None:
        """LINEAR camp with even waypoints averages two middle points."""
        wps = [
            {"x": 0.0, "y": 0.0},
            {"x": 100.0, "y": 0.0},
            {"x": 200.0, "y": 0.0},
            {"x": 300.0, "y": 0.0},
        ]
        camp = _camp(cx=0, cy=0, camp_type="linear", patrol_waypoints=wps)
        # Even: average of wps[1] and wps[2] = (150, 0)
        score = score_camp(camp, Point(0.0, 0.0, 0.0), player_level=10)
        assert score == pytest.approx(150.0, abs=0.1)

    def test_linear_camp_with_explicit_center_uses_center(self) -> None:
        """When center is non-zero, even a linear camp uses the explicit center."""
        wps = [{"x": 0.0, "y": 0.0}, {"x": 1000.0, "y": 0.0}]
        camp = _camp(cx=50, cy=0, camp_type="linear", patrol_waypoints=wps)
        # cx=50 is non-zero, so center is used directly
        score = score_camp(camp, Point(0.0, 0.0, 0.0), player_level=10)
        assert score == pytest.approx(50.0, abs=0.1)


# ---------------------------------------------------------------------------
# select_camp
# ---------------------------------------------------------------------------


class TestSelectCamp:
    """select_camp picks the best camp by score."""

    def test_empty_list_returns_empty_dict(self) -> None:
        assert select_camp([], Point(100.0, 100.0, 0.0), 10) == {}

    def test_single_camp_returned(self) -> None:
        camps = [_camp("only")]
        result = select_camp(camps, Point(100.0, 100.0, 0.0), 10)
        assert result["name"] == "only"

    def test_selects_closest_camp_same_level_range(self) -> None:
        near = _camp("near", cx=110, cy=100, level_range=[5, 15])
        far = _camp("far", cx=500, cy=500, level_range=[5, 15])
        result = select_camp([far, near], Point(100.0, 100.0, 0.0), 10)
        assert result["name"] == "near"

    def test_prefers_level_fit_over_proximity(self) -> None:
        close_bad = _camp("close_bad", cx=110, cy=100, level_range=[1, 3])
        far_good = _camp("far_good", cx=200, cy=200, level_range=[8, 12])
        result = select_camp([close_bad, far_good], Point(100.0, 100.0, 0.0), 10)
        assert result["name"] == "far_good"

    def test_origin_player_uses_fallback_name(self) -> None:
        a = _camp("alpha")
        b = _camp("beta")
        result = select_camp([a, b], Point(0.0, 0.0, 0.0), 10, fallback_name="beta")
        assert result["name"] == "beta"

    def test_origin_player_missing_fallback_uses_first(self) -> None:
        a = _camp("alpha")
        b = _camp("beta")
        result = select_camp([a, b], Point(0.0, 0.0, 0.0), 10, fallback_name="nonexistent")
        assert result["name"] == "alpha"

    def test_origin_player_no_fallback_uses_first(self) -> None:
        a = _camp("alpha")
        b = _camp("beta")
        result = select_camp([a, b], Point(0.0, 0.0, 0.0), 10)
        assert result["name"] == "alpha"

    def test_skips_camps_with_zero_center(self) -> None:
        """Camps with center (0,0) are skipped during scored selection."""
        no_center = _camp("no_center", cx=0, cy=0)
        real = _camp("real", cx=200, cy=200, level_range=[5, 15])
        result = select_camp([no_center, real], Point(100.0, 100.0, 0.0), 10)
        assert result["name"] == "real"

    def test_fallback_camp_loses_to_normal_camp(self) -> None:
        normal = _camp("normal", cx=100, cy=100, level_range=[5, 15])
        fb = _camp("fallback", cx=100, cy=100, level_range=[5, 15], fallback=True)
        result = select_camp([fb, normal], Point(100.0, 100.0, 0.0), 10)
        assert result["name"] == "normal"
