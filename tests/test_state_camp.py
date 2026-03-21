"""Tests for brain.state.camp -- CampConfig spatial anchor logic.

Covers distance_to_camp for CIRCULAR and LINEAR modes, effective_camp_distance,
patrol_position, nearest_point_on_path, point_along_path, path_total_length,
and _inside_bounds.
"""

from __future__ import annotations

import math

import pytest

from brain.state.camp import CampConfig
from core.types import CampType, Point
from tests.factories import make_game_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_camp(**overrides) -> CampConfig:
    """A LINEAR camp with two waypoints along the X axis."""
    defaults = dict(
        camp_type=CampType.LINEAR,
        patrol_waypoints=[Point(0.0, 0.0, 0.0), Point(1000.0, 0.0, 0.0)],
        corridor_width=100.0,
    )
    defaults.update(overrides)
    return CampConfig(**defaults)


def _circular_camp(**overrides) -> CampConfig:
    defaults = dict(
        camp_x=500.0,
        camp_y=500.0,
        roam_radius=250.0,
    )
    defaults.update(overrides)
    return CampConfig(**defaults)


# ---------------------------------------------------------------------------
# distance_to_camp
# ---------------------------------------------------------------------------


class TestDistanceToCampCircular:
    def test_outside_no_bounds(self) -> None:
        camp = _circular_camp()
        state = make_game_state(x=0.0, y=0.0)
        d = camp.distance_to_camp(state)
        expected = math.hypot(500.0, 500.0)
        assert abs(d - expected) < 0.1

    def test_inside_bounds_returns_zero(self) -> None:
        camp = _circular_camp(
            bounds_x_min=400.0,
            bounds_x_max=600.0,
            bounds_y_min=400.0,
            bounds_y_max=600.0,
        )
        state = make_game_state(x=500.0, y=500.0)
        assert camp.distance_to_camp(state) == 0.0

    def test_outside_bounds_returns_distance(self) -> None:
        camp = _circular_camp(
            bounds_x_min=400.0,
            bounds_x_max=600.0,
            bounds_y_min=400.0,
            bounds_y_max=600.0,
        )
        state = make_game_state(x=0.0, y=0.0)
        d = camp.distance_to_camp(state)
        # Outside bounds -> falls through to distance_2d
        assert d > 0.0


class TestDistanceToCampLinear:
    def test_on_path_within_corridor(self) -> None:
        camp = _linear_camp()
        state = make_game_state(x=500.0, y=50.0)  # 50 from path, corridor=100
        d = camp.distance_to_camp(state)
        assert d == 0.0  # within corridor

    def test_outside_corridor(self) -> None:
        camp = _linear_camp()
        state = make_game_state(x=500.0, y=200.0)  # 200 from path, corridor=100
        d = camp.distance_to_camp(state)
        assert d == pytest.approx(100.0, abs=1.0)  # 200 - 100 corridor

    def test_single_waypoint_falls_back_to_circular(self) -> None:
        camp = CampConfig(
            camp_type=CampType.LINEAR,
            patrol_waypoints=[Point(0.0, 0.0, 0.0)],
            camp_x=100.0,
            camp_y=100.0,
        )
        state = make_game_state(x=0.0, y=0.0)
        d = camp.distance_to_camp(state)
        assert d == pytest.approx(math.hypot(100.0, 100.0), abs=0.1)


# ---------------------------------------------------------------------------
# effective_camp_distance
# ---------------------------------------------------------------------------


class TestEffectiveCampDistance:
    def test_linear_within_corridor(self) -> None:
        camp = _linear_camp()
        d = camp.effective_camp_distance(Point(500.0, 50.0, 0.0))
        assert d == 0.0

    def test_linear_outside_corridor(self) -> None:
        camp = _linear_camp()
        d = camp.effective_camp_distance(Point(500.0, 300.0, 0.0))
        assert d == pytest.approx(200.0, abs=1.0)

    def test_circular_inside_bounds(self) -> None:
        camp = _circular_camp(
            bounds_x_min=400.0,
            bounds_x_max=600.0,
            bounds_y_min=400.0,
            bounds_y_max=600.0,
        )
        assert camp.effective_camp_distance(Point(500.0, 500.0, 0.0)) == 0.0

    def test_circular_outside_bounds(self) -> None:
        camp = _circular_camp(
            bounds_x_min=400.0,
            bounds_x_max=600.0,
            bounds_y_min=400.0,
            bounds_y_max=600.0,
        )
        d = camp.effective_camp_distance(Point(0.0, 0.0, 0.0))
        assert d > 0.0

    def test_circular_no_bounds(self) -> None:
        camp = _circular_camp()
        d = camp.effective_camp_distance(Point(0.0, 0.0, 0.0))
        expected = math.hypot(500.0, 500.0)
        assert abs(d - expected) < 0.1

    def test_linear_single_waypoint_fallback(self) -> None:
        camp = CampConfig(
            camp_type=CampType.LINEAR,
            patrol_waypoints=[Point(0.0, 0.0, 0.0)],
            camp_x=100.0,
            camp_y=0.0,
        )
        d = camp.effective_camp_distance(Point(0.0, 0.0, 0.0))
        # Falls back to circular: distance_2d(0,0,100,0) = 100
        assert d == pytest.approx(100.0, abs=0.1)


# ---------------------------------------------------------------------------
# patrol_position
# ---------------------------------------------------------------------------


class TestPatrolPosition:
    def test_circular_returns_zero(self) -> None:
        camp = _circular_camp()
        assert camp.patrol_position(Point(500.0, 500.0, 0.0)) == 0.0

    def test_linear_at_start(self) -> None:
        camp = _linear_camp()
        frac = camp.patrol_position(Point(0.0, 0.0, 0.0))
        assert frac == pytest.approx(0.0, abs=0.01)

    def test_linear_at_end(self) -> None:
        camp = _linear_camp()
        frac = camp.patrol_position(Point(1000.0, 0.0, 0.0))
        assert frac == pytest.approx(1.0, abs=0.01)

    def test_linear_midpoint(self) -> None:
        camp = _linear_camp()
        frac = camp.patrol_position(Point(500.0, 0.0, 0.0))
        assert frac == pytest.approx(0.5, abs=0.01)

    def test_single_waypoint_returns_zero(self) -> None:
        camp = CampConfig(
            camp_type=CampType.LINEAR,
            patrol_waypoints=[Point(0.0, 0.0, 0.0)],
        )
        assert camp.patrol_position(Point(100.0, 100.0, 0.0)) == 0.0


# ---------------------------------------------------------------------------
# nearest_point_on_path
# ---------------------------------------------------------------------------


class TestNearestPointOnPath:
    def test_circular_returns_camp_center(self) -> None:
        camp = _circular_camp()
        pt = camp.nearest_point_on_path(Point(0.0, 0.0, 0.0))
        assert pt == Point(500.0, 500.0, 0.0)

    def test_linear_projects_onto_path(self) -> None:
        camp = _linear_camp()
        pt = camp.nearest_point_on_path(Point(500.0, 200.0, 0.0))
        # Should be projected to (500, 0) on the X-axis path
        assert pt.x == pytest.approx(500.0, abs=1.0)
        assert pt.y == pytest.approx(0.0, abs=1.0)

    def test_single_waypoint_returns_camp_center(self) -> None:
        camp = CampConfig(
            camp_type=CampType.LINEAR,
            patrol_waypoints=[Point(0.0, 0.0, 0.0)],
            camp_x=42.0,
            camp_y=99.0,
        )
        assert camp.nearest_point_on_path(Point(0.0, 0.0, 0.0)) == Point(42.0, 99.0, 0.0)


# ---------------------------------------------------------------------------
# point_along_path
# ---------------------------------------------------------------------------


class TestPointAlongPath:
    def test_t_zero_returns_start(self) -> None:
        camp = _linear_camp()
        pt = camp.point_along_path(0.0)
        assert pt.x == pytest.approx(0.0, abs=0.1)
        assert pt.y == pytest.approx(0.0, abs=0.1)

    def test_t_one_returns_end(self) -> None:
        camp = _linear_camp()
        pt = camp.point_along_path(1.0)
        assert pt.x == pytest.approx(1000.0, abs=0.1)
        assert pt.y == pytest.approx(0.0, abs=0.1)

    def test_t_half_returns_midpoint(self) -> None:
        camp = _linear_camp()
        pt = camp.point_along_path(0.5)
        assert pt.x == pytest.approx(500.0, abs=0.1)
        assert pt.y == pytest.approx(0.0, abs=0.1)

    def test_clamped_below_zero(self) -> None:
        camp = _linear_camp()
        pt = camp.point_along_path(-1.0)
        assert pt.x == pytest.approx(0.0, abs=0.1)

    def test_clamped_above_one(self) -> None:
        camp = _linear_camp()
        pt = camp.point_along_path(2.0)
        assert pt.x == pytest.approx(1000.0, abs=0.1)

    def test_fewer_than_two_waypoints(self) -> None:
        camp = CampConfig(
            patrol_waypoints=[Point(42.0, 99.0, 0.0)],
            camp_x=10.0,
            camp_y=20.0,
        )
        pt = camp.point_along_path(0.5)
        assert pt == Point(10.0, 20.0, 0.0)

    def test_multi_segment_interpolation(self) -> None:
        """Three waypoints: (0,0) -> (100,0) -> (100,100). Total length 200."""
        camp = CampConfig(
            camp_type=CampType.LINEAR,
            patrol_waypoints=[Point(0.0, 0.0, 0.0), Point(100.0, 0.0, 0.0), Point(100.0, 100.0, 0.0)],
        )
        # t=0.25 -> 50 units along first segment
        pt = camp.point_along_path(0.25)
        assert pt.x == pytest.approx(50.0, abs=1.0)
        assert pt.y == pytest.approx(0.0, abs=1.0)

        # t=0.75 -> 50 units into second segment
        pt2 = camp.point_along_path(0.75)
        assert pt2.x == pytest.approx(100.0, abs=1.0)
        assert pt2.y == pytest.approx(50.0, abs=1.0)

    def test_zero_length_path(self) -> None:
        """Degenerate case: two identical waypoints."""
        camp = CampConfig(
            camp_type=CampType.LINEAR,
            patrol_waypoints=[Point(50.0, 50.0, 0.0), Point(50.0, 50.0, 0.0)],
        )
        pt = camp.point_along_path(0.5)
        assert pt == Point(50.0, 50.0, 0.0)


# ---------------------------------------------------------------------------
# path_total_length
# ---------------------------------------------------------------------------


class TestPathTotalLength:
    def test_single_segment(self) -> None:
        camp = _linear_camp()
        assert camp.path_total_length() == pytest.approx(1000.0)

    def test_multi_segment(self) -> None:
        camp = CampConfig(
            patrol_waypoints=[Point(0.0, 0.0, 0.0), Point(100.0, 0.0, 0.0), Point(100.0, 100.0, 0.0)],
        )
        assert camp.path_total_length() == pytest.approx(200.0)

    def test_no_waypoints(self) -> None:
        camp = CampConfig()
        assert camp.path_total_length() == 0.0


# ---------------------------------------------------------------------------
# _inside_bounds (edge cases)
# ---------------------------------------------------------------------------


class TestInsideBounds:
    def test_no_bounds_returns_false(self) -> None:
        camp = CampConfig()
        assert camp._inside_bounds(Point(0.0, 0.0, 0.0)) is False

    def test_partial_bounds_x_min_only(self) -> None:
        camp = CampConfig(bounds_x_min=10.0)
        assert camp._inside_bounds(Point(5.0, 0.0, 0.0)) is False
        assert camp._inside_bounds(Point(15.0, 0.0, 0.0)) is True

    def test_partial_bounds_x_max_only(self) -> None:
        camp = CampConfig(bounds_x_max=100.0)
        assert camp._inside_bounds(Point(50.0, 0.0, 0.0)) is True
        assert camp._inside_bounds(Point(150.0, 0.0, 0.0)) is False

    def test_partial_bounds_y_min_only(self) -> None:
        camp = CampConfig(bounds_y_min=-50.0)
        assert camp._inside_bounds(Point(0.0, -100.0, 0.0)) is False
        assert camp._inside_bounds(Point(0.0, 0.0, 0.0)) is True

    def test_partial_bounds_y_max_only(self) -> None:
        camp = CampConfig(bounds_y_max=200.0)
        assert camp._inside_bounds(Point(0.0, 100.0, 0.0)) is True
        assert camp._inside_bounds(Point(0.0, 300.0, 0.0)) is False

    def test_full_bounds_inside(self) -> None:
        camp = CampConfig(
            bounds_x_min=0.0,
            bounds_x_max=100.0,
            bounds_y_min=0.0,
            bounds_y_max=100.0,
        )
        assert camp._inside_bounds(Point(50.0, 50.0, 0.0)) is True

    def test_full_bounds_outside_x_min(self) -> None:
        camp = CampConfig(
            bounds_x_min=0.0,
            bounds_x_max=100.0,
            bounds_y_min=0.0,
            bounds_y_max=100.0,
        )
        assert camp._inside_bounds(Point(-1.0, 50.0, 0.0)) is False

    def test_full_bounds_outside_y_max(self) -> None:
        camp = CampConfig(
            bounds_x_min=0.0,
            bounds_x_max=100.0,
            bounds_y_min=0.0,
            bounds_y_max=100.0,
        )
        assert camp._inside_bounds(Point(50.0, 200.0, 0.0)) is False
