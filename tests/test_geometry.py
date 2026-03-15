"""Tests for nav.geometry  -- pure math over EQ's coordinate system.

EQ quirks: heading 0-512 (not 360°), Y-axis inverted, heading 0 = south.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from core.types import Point
from nav.geometry import (
    angle_diff,
    distance_2d,
    heading_to,
    normalize_heading,
    point_to_polyline,
    point_to_segment,
)

# Hypothesis strategies for EQ-specific domains
eq_heading = st.floats(min_value=-1024, max_value=1024, allow_nan=False, allow_infinity=False)
eq_coord = st.floats(min_value=-10_000, max_value=10_000, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# distance_2d
# ---------------------------------------------------------------------------


class TestDistance2D:
    @given(x=eq_coord, y=eq_coord)
    def test_zero_to_self(self, x: float, y: float) -> None:
        assert distance_2d(x, y, x, y) == 0.0

    @given(x1=eq_coord, y1=eq_coord, x2=eq_coord, y2=eq_coord)
    def test_non_negative(self, x1: float, y1: float, x2: float, y2: float) -> None:
        assert distance_2d(x1, y1, x2, y2) >= 0.0

    @given(x1=eq_coord, y1=eq_coord, x2=eq_coord, y2=eq_coord)
    def test_symmetric(self, x1: float, y1: float, x2: float, y2: float) -> None:
        assert distance_2d(x1, y1, x2, y2) == pytest.approx(distance_2d(x2, y2, x1, y1))

    def test_345_triangle(self) -> None:
        assert distance_2d(0, 0, 3, 4) == pytest.approx(5.0)

    def test_known_distance(self) -> None:
        assert distance_2d(10, 20, 13, 24) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# normalize_heading
# ---------------------------------------------------------------------------


class TestNormalizeHeading:
    @given(h=eq_heading)
    def test_output_range(self, h: float) -> None:
        result = normalize_heading(h)
        assert 0.0 <= result <= 512.0  # float precision: -1e-15 → 512.0

    @given(h=eq_heading)
    def test_idempotent(self, h: float) -> None:
        once = normalize_heading(h)
        twice = normalize_heading(once)
        # 0.0 and 512.0 are equivalent headings (full circle)
        if {once, twice} == {0.0, 512.0}:
            return
        assert once == pytest.approx(twice, abs=1e-9)

    @pytest.mark.parametrize(
        "input_h, expected",
        [
            (0.0, 0.0),
            (512.0, 0.0),
            (-1.0, 511.0),
            (640.0, 128.0),
            (1024.0, 0.0),
        ],
    )
    def test_wrapping(self, input_h: float, expected: float) -> None:
        assert normalize_heading(input_h) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# angle_diff
# ---------------------------------------------------------------------------


class TestAngleDiff:
    @given(a=eq_heading, b=eq_heading)
    def test_output_range(self, a: float, b: float) -> None:
        diff = angle_diff(a, b)
        assert -256.0 < diff <= 256.0

    @given(h=eq_heading)
    def test_zero_diff_to_self(self, h: float) -> None:
        assert angle_diff(h, h) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "current, target, expected",
        [
            (0, 128, 128),
            (0, 384, -128),
            (500, 12, 24),  # wrapping forward
            (10, 500, -22),  # wrapping backward
        ],
    )
    def test_known_values(self, current: float, target: float, expected: float) -> None:
        assert angle_diff(current, target) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# heading_to
# ---------------------------------------------------------------------------


class TestHeadingTo:
    @given(x1=eq_coord, y1=eq_coord, x2=eq_coord, y2=eq_coord)
    def test_output_range(self, x1: float, y1: float, x2: float, y2: float) -> None:
        assume(distance_2d(x1, y1, x2, y2) > 0.01)
        result = heading_to(x1, y1, x2, y2)
        assert 0.0 <= result <= 512.0  # float precision: boundary can hit 512.0

    def test_cardinal_south(self) -> None:
        # Y+ is south in EQ, heading 0 = south
        # Due south: target has higher y
        h = heading_to(0, 0, 0, 100)
        assert h == pytest.approx(0.0, abs=1.0) or h == pytest.approx(512.0, abs=1.0)


# ---------------------------------------------------------------------------
# point_to_segment
# ---------------------------------------------------------------------------


class TestPointToSegment:
    @given(px=eq_coord, py=eq_coord, ax=eq_coord, ay=eq_coord, bx=eq_coord, by=eq_coord)
    def test_distance_non_negative(
        self, px: float, py: float, ax: float, ay: float, bx: float, by: float
    ) -> None:
        dist, _, _, _ = point_to_segment(px, py, ax, ay, bx, by)
        assert dist >= 0.0

    @given(px=eq_coord, py=eq_coord, ax=eq_coord, ay=eq_coord, bx=eq_coord, by=eq_coord)
    def test_t_in_range(self, px: float, py: float, ax: float, ay: float, bx: float, by: float) -> None:
        _, _, _, t = point_to_segment(px, py, ax, ay, bx, by)
        assert 0.0 <= t <= 1.0

    def test_point_at_endpoint_a(self) -> None:
        dist, nx, ny, t = point_to_segment(0, 0, 0, 0, 10, 0)
        assert dist == pytest.approx(0.0)
        assert t == pytest.approx(0.0)

    def test_point_at_endpoint_b(self) -> None:
        dist, nx, ny, t = point_to_segment(10, 0, 0, 0, 10, 0)
        assert dist == pytest.approx(0.0)
        assert t == pytest.approx(1.0)

    def test_perpendicular_to_midpoint(self) -> None:
        dist, nx, ny, t = point_to_segment(5, 3, 0, 0, 10, 0)
        assert dist == pytest.approx(3.0)
        assert t == pytest.approx(0.5)

    def test_degenerate_segment(self) -> None:
        # A == B: distance to the single point
        dist, _, _, _ = point_to_segment(3, 4, 0, 0, 0, 0)
        assert dist == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# point_to_polyline
# ---------------------------------------------------------------------------


class TestPointToPolyline:
    def test_single_segment(self) -> None:
        wps = [Point(0.0, 0.0, 0.0), Point(10.0, 0.0, 0.0)]
        dist, nx, ny, seg_idx, path_t = point_to_polyline(5, 3, wps)
        assert dist == pytest.approx(3.0)
        assert seg_idx == 0
        assert 0.0 <= path_t <= 1.0

    def test_path_t_near_start(self) -> None:
        wps = [Point(0.0, 0.0, 0.0), Point(100.0, 0.0, 0.0), Point(200.0, 0.0, 0.0)]
        _, _, _, _, path_t = point_to_polyline(1, 0, wps)
        assert path_t < 0.1

    def test_path_t_near_end(self) -> None:
        wps = [Point(0.0, 0.0, 0.0), Point(100.0, 0.0, 0.0), Point(200.0, 0.0, 0.0)]
        _, _, _, _, path_t = point_to_polyline(199, 0, wps)
        assert path_t > 0.9


# ---------------------------------------------------------------------------
# 3D distance: Point.dist_to
# ---------------------------------------------------------------------------


class TestPoint3DDistance:
    """Verify 3D distance via Point.dist_to."""

    def test_basic_3d(self) -> None:
        a = Point(0.0, 0.0, 0.0)
        b = Point(3.0, 4.0, 0.0)
        assert a.dist_to(b) == pytest.approx(5.0)

    def test_vertical_only(self) -> None:
        a = Point(0.0, 0.0, 0.0)
        b = Point(0.0, 0.0, 10.0)
        assert a.dist_to(b) == pytest.approx(10.0)

    def test_full_3d(self) -> None:
        a = Point(1.0, 2.0, 3.0)
        b = Point(4.0, 6.0, 3.0)
        assert a.dist_to(b) == pytest.approx(5.0)

    def test_self_distance_is_zero(self) -> None:
        p = Point(42.0, -17.0, 100.0)
        assert p.dist_to(p) == 0.0

    @given(
        x1=eq_coord,
        y1=eq_coord,
        z1=eq_coord,
        x2=eq_coord,
        y2=eq_coord,
        z2=eq_coord,
    )
    def test_3d_symmetry(self, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> None:
        a = Point(x1, y1, z1)
        b = Point(x2, y2, z2)
        assert a.dist_to(b) == pytest.approx(b.dist_to(a))

    @given(
        x1=eq_coord,
        y1=eq_coord,
        z1=eq_coord,
        x2=eq_coord,
        y2=eq_coord,
        z2=eq_coord,
    )
    def test_3d_geq_2d(self, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> None:
        """3D distance is always >= 2D distance (Pythagorean extension)."""
        d3 = Point(x1, y1, z1).dist_to(Point(x2, y2, z2))
        d2 = distance_2d(x1, y1, x2, y2)
        assert d3 >= d2 - 1e-9  # float tolerance

    @given(
        x1=eq_coord,
        y1=eq_coord,
        z1=eq_coord,
        x2=eq_coord,
        y2=eq_coord,
        z2=eq_coord,
        x3=eq_coord,
        y3=eq_coord,
        z3=eq_coord,
    )
    def test_triangle_inequality(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
        x3: float,
        y3: float,
        z3: float,
    ) -> None:
        """d(a,c) <= d(a,b) + d(b,c) for any three points."""
        a = Point(x1, y1, z1)
        b = Point(x2, y2, z2)
        c = Point(x3, y3, z3)
        assert a.dist_to(c) <= a.dist_to(b) + b.dist_to(c) + 1e-9
