"""Tests for nav/map_data.py: zone map parsing, spatial indexing, and path queries.

Covers segment parsing, POI extraction, grid-based spatial indexing,
path_blocked intersection checks, nearest_segment_dist, and find_detour.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.types import Point
from nav.map_data import (
    POI,
    Segment,
    ZoneMap,
    _cross,
    _point_segment_dist,
    _segments_intersect,
    load_zone_map,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zone(segments: list[Segment] | None = None, pois: list[POI] | None = None) -> ZoneMap:
    return ZoneMap(segments or [], pois or [])


# ---------------------------------------------------------------------------
# Segment / POI dataclasses
# ---------------------------------------------------------------------------


class TestSegmentAndPOI:
    def test_segment_frozen(self) -> None:
        seg: Any = Segment(0, 0, 10, 10)
        with pytest.raises(AttributeError):
            seg.x1 = 5

    def test_poi_frozen(self) -> None:
        poi: Any = POI(1.0, 2.0, 3.0, "Camp")
        assert poi.label == "Camp"
        with pytest.raises(AttributeError):
            poi.label = "Other"


# ---------------------------------------------------------------------------
# Cross product
# ---------------------------------------------------------------------------


class TestCross:
    def test_zero(self) -> None:
        assert _cross(1, 0, 1, 0) == 0.0

    def test_perpendicular(self) -> None:
        assert _cross(1, 0, 0, 1) == 1.0
        assert _cross(0, 1, 1, 0) == -1.0


# ---------------------------------------------------------------------------
# Segment intersection (no buffer)
# ---------------------------------------------------------------------------


class TestSegmentsIntersect:
    def test_crossing(self) -> None:
        assert _segments_intersect(0, 0, 10, 10, 0, 10, 10, 0) is True

    def test_parallel_no_intersect(self) -> None:
        assert _segments_intersect(0, 0, 10, 0, 0, 5, 10, 5) is False

    def test_disjoint(self) -> None:
        assert _segments_intersect(0, 0, 1, 0, 5, 5, 6, 5) is False

    def test_t_shape_touch(self) -> None:
        # Segment A: (0,5)->(10,5), B: (5,0)->(5,10) -- cross at (5,5)
        assert _segments_intersect(0, 5, 10, 5, 5, 0, 5, 10) is True

    def test_buffer_makes_near_miss_intersect(self) -> None:
        # Two horizontal segments 2 units apart -- no intersect at buffer=0
        assert _segments_intersect(0, 0, 10, 0, 0, 2, 10, 2, buffer=0) is False
        # With buffer=3, they are within distance
        assert _segments_intersect(0, 0, 10, 0, 0, 2, 10, 2, buffer=3) is True


# ---------------------------------------------------------------------------
# Point-to-segment distance
# ---------------------------------------------------------------------------


class TestPointSegmentDist:
    def test_point_on_segment(self) -> None:
        seg = Segment(0, 0, 10, 0)
        assert _point_segment_dist(5, 0, seg) == pytest.approx(0.0)

    def test_point_above_midpoint(self) -> None:
        seg = Segment(0, 0, 10, 0)
        assert _point_segment_dist(5, 3, seg) == pytest.approx(3.0)

    def test_point_past_endpoint(self) -> None:
        seg = Segment(0, 0, 10, 0)
        assert _point_segment_dist(15, 0, seg) == pytest.approx(5.0)

    def test_degenerate_zero_length_segment(self) -> None:
        seg = Segment(5, 5, 5, 5)
        assert _point_segment_dist(8, 9, seg) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# ZoneMap grid indexing
# ---------------------------------------------------------------------------


class TestZoneMapGrid:
    def test_empty_map(self) -> None:
        zm = _zone()
        assert zm.segments == []
        assert zm.pois == []
        assert zm._grid == {}

    def test_segment_indexed_in_grid(self) -> None:
        seg = Segment(10, 10, 20, 20)
        zm = _zone([seg])
        assert len(zm._grid) > 0
        # At least one cell should reference segment index 0
        assert any(0 in idxs for idxs in zm._grid.values())

    def test_pos_to_cell(self) -> None:
        zm = _zone()
        assert zm._pos_to_cell(0, 0) == (0, 0)
        assert zm._pos_to_cell(49, 49) == (0, 0)
        assert zm._pos_to_cell(50, 50) == (1, 1)
        assert zm._pos_to_cell(-1, -1) == (-1, -1)


# ---------------------------------------------------------------------------
# path_blocked
# ---------------------------------------------------------------------------


class TestPathBlocked:
    def test_clear_path(self) -> None:
        # Wall at y=100, path below it
        seg = Segment(0, 100, 200, 100)
        zm = _zone([seg])
        assert zm.path_blocked(50, 10, 150, 10) is None

    def test_blocked_path(self) -> None:
        # Horizontal wall at y=50, path crosses it
        seg = Segment(0, 50, 200, 50)
        zm = _zone([seg])
        result = zm.path_blocked(100, 0, 100, 100)
        assert result is not None
        assert result == seg

    def test_blocked_with_buffer(self) -> None:
        # Path passes 2 units from the wall -- clear with buffer=1, blocked with buffer=5
        seg = Segment(0, 50, 200, 50)
        zm = _zone([seg])
        assert zm.path_blocked(100, 47, 100, 47, buffer=1) is None
        assert zm.path_blocked(100, 0, 100, 100, buffer=5) is not None


# ---------------------------------------------------------------------------
# nearest_segment_dist
# ---------------------------------------------------------------------------


class TestNearestSegmentDist:
    def test_nearby(self) -> None:
        seg = Segment(0, 0, 100, 0)
        zm = _zone([seg])
        dist = zm.nearest_segment_dist(50, 10)
        assert dist == pytest.approx(10.0)

    def test_no_segments_returns_inf(self) -> None:
        zm = _zone()
        assert zm.nearest_segment_dist(0, 0) == float("inf")


# ---------------------------------------------------------------------------
# find_detour
# ---------------------------------------------------------------------------


class TestFindDetour:
    def test_no_block_returns_none(self) -> None:
        seg = Segment(0, 100, 200, 100)
        zm = _zone([seg])
        assert zm.find_detour(50, 10, 150, 10) is None

    def test_returns_point_around_wall(self) -> None:
        # Vertical wall from (50,0) to (50,100); path (0,50) -> (100,50)
        seg = Segment(50, 0, 50, 100)
        zm = _zone([seg])
        wp = zm.find_detour(0, 50, 100, 50)
        assert wp is not None
        assert isinstance(wp, Point)
        # Waypoint should be offset from the wall in y direction
        assert wp.x != pytest.approx(50.0, abs=1.0)

    def test_degenerate_segment_returns_none(self) -> None:
        seg = Segment(50, 50, 50, 50)
        zm = _zone([seg])
        # Even if path goes through the point, degenerate segment has length ~0
        assert zm.find_detour(50, 0, 50, 100) is None


# ---------------------------------------------------------------------------
# load_zone_map file parser
# ---------------------------------------------------------------------------


class TestLoadZoneMap:
    def test_missing_file_returns_empty(self, tmp_path) -> None:
        zm = load_zone_map(tmp_path / "nonexistent.txt")
        assert zm.segments == []
        assert zm.pois == []

    def test_parses_L_lines(self, tmp_path) -> None:
        mapfile = tmp_path / "zone.txt"
        mapfile.write_text("L 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 255, 255, 255\n")
        zm = load_zone_map(mapfile)
        assert len(zm.segments) == 1
        assert zm.segments[0] == Segment(0.0, 0.0, 100.0, 0.0)

    def test_parses_P_lines(self, tmp_path) -> None:
        mapfile = tmp_path / "zone.txt"
        mapfile.write_text("P 10.0, 20.0, 30.0, 255, 0, 0, 3, Innothule\n")
        zm = load_zone_map(mapfile)
        assert len(zm.pois) == 1
        assert zm.pois[0].label == "Innothule"
        assert zm.pois[0].x == pytest.approx(10.0)

    def test_ignores_blank_and_unknown_lines(self, tmp_path) -> None:
        mapfile = tmp_path / "zone.txt"
        mapfile.write_text("\n# comment\nGARBAGE\nL 1,2,3,4,5,6,0,0,0\n")
        zm = load_zone_map(mapfile)
        assert len(zm.segments) == 1

    def test_skips_malformed_L_lines(self, tmp_path) -> None:
        mapfile = tmp_path / "zone.txt"
        mapfile.write_text("L 1, 2\nL a,b,c,d,e,f\nL 1,2,3,4,5,6,0,0,0\n")
        zm = load_zone_map(mapfile)
        # First too few parts, second non-numeric, third valid
        assert len(zm.segments) == 1

    def test_skips_malformed_P_lines(self, tmp_path) -> None:
        mapfile = tmp_path / "zone.txt"
        mapfile.write_text("P 1,2\nP a,b,c,0,0,0,3,Label\nP 1,2,3,0,0,0,3,OK\n")
        zm = load_zone_map(mapfile)
        # First too few parts, second non-numeric coords, third valid
        assert len(zm.pois) == 1


# ---------------------------------------------------------------------------
# Hypothesis: segment intersection symmetry
# ---------------------------------------------------------------------------


@given(
    x1=st.floats(-500, 500),
    y1=st.floats(-500, 500),
    x2=st.floats(-500, 500),
    y2=st.floats(-500, 500),
)
@settings(max_examples=100)
def test_intersection_symmetric(x1: float, y1: float, x2: float, y2: float) -> None:
    """Intersection should be symmetric: A crosses B iff B crosses A."""
    a = _segments_intersect(x1, y1, x2, y2, 0, -100, 0, 100)
    b = _segments_intersect(0, -100, 0, 100, x1, y1, x2, y2)
    assert a == b
