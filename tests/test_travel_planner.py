"""Tests for nav.travel_planner -- multi-leg route planning with tunnels.

Covers parse_tunnel_routes config parsing, plan_travel_legs routing logic
with and without tunnel routes and waypoint graphs, TravelLeg dataclass,
and subdivide_waypoints segment splitting.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.types import Point, TravelMode, TravelWaypoint
from nav.travel_planner import (
    TravelLeg,
    TunnelRoute,
    parse_tunnel_routes,
    plan_travel_legs,
    subdivide_waypoints,
)
from nav.waypoint_graph import WaypointGraph

# ---------------------------------------------------------------------------
# parse_tunnel_routes
# ---------------------------------------------------------------------------


class TestParseTunnelRoutes:
    def test_empty_config(self) -> None:
        assert parse_tunnel_routes({}) == []

    def test_no_tunnel_routes_key(self) -> None:
        assert parse_tunnel_routes({"waypoints": []}) == []

    def test_missing_waypoint_skipped(self) -> None:
        config = {
            "waypoints": [{"name": "a", "x": 0.0, "y": 0.0}],
            "tunnel_routes": [
                {"name": "t1", "from_waypoint": "a", "to_waypoint": "b", "points": [{"x": 1, "y": 2}]}
            ],
        }
        result = parse_tunnel_routes(config)
        assert result == []

    def test_valid_tunnel_route_parsed(self) -> None:
        config = {
            "waypoints": [
                {"name": "north", "x": 100.0, "y": 200.0},
                {"name": "south", "x": 300.0, "y": 400.0},
            ],
            "tunnel_routes": [
                {
                    "name": "ns_tunnel",
                    "from_waypoint": "north",
                    "to_waypoint": "south",
                    "points": [
                        {"x": 150.0, "y": 250.0, "action": "left_click"},
                        {"x": 200.0, "y": 300.0},
                    ],
                }
            ],
        }
        result = parse_tunnel_routes(config)
        assert len(result) == 1
        route = result[0]
        assert route.name == "ns_tunnel"
        assert route.from_waypoint == "north"
        assert route.to_waypoint == "south"
        assert route.from_pos == Point(100.0, 200.0, 0.0)
        assert route.to_pos == Point(300.0, 400.0, 0.0)
        assert len(route.points) == 2
        assert route.points[0].action == "left_click"
        assert route.points[1].action == ""

    def test_empty_points_skipped(self) -> None:
        config = {
            "waypoints": [
                {"name": "a", "x": 0.0, "y": 0.0},
                {"name": "b", "x": 10.0, "y": 10.0},
            ],
            "tunnel_routes": [{"name": "empty", "from_waypoint": "a", "to_waypoint": "b", "points": []}],
        }
        result = parse_tunnel_routes(config)
        assert result == []

    def test_multiple_routes(self) -> None:
        config = {
            "waypoints": [
                {"name": "a", "x": 0.0, "y": 0.0},
                {"name": "b", "x": 100.0, "y": 100.0},
                {"name": "c", "x": 200.0, "y": 200.0},
            ],
            "tunnel_routes": [
                {"name": "ab", "from_waypoint": "a", "to_waypoint": "b", "points": [{"x": 50, "y": 50}]},
                {"name": "bc", "from_waypoint": "b", "to_waypoint": "c", "points": [{"x": 150, "y": 150}]},
            ],
        }
        result = parse_tunnel_routes(config)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# TravelLeg dataclass
# ---------------------------------------------------------------------------


class TestTravelLeg:
    def test_fields(self) -> None:
        leg = TravelLeg(target_x=1.0, target_y=2.0, mode=TravelMode.PATHFIND)
        assert leg.target_x == 1.0
        assert leg.target_y == 2.0
        assert leg.mode == TravelMode.PATHFIND
        assert leg.waypoints is None

    def test_manual_leg_with_waypoints(self) -> None:
        wps = (TravelWaypoint(10.0, 20.0, 0.0, "click"),)
        leg = TravelLeg(target_x=10.0, target_y=20.0, mode=TravelMode.MANUAL, waypoints=wps)
        assert leg.waypoints is not None
        assert len(leg.waypoints) == 1

    def test_frozen(self) -> None:
        leg: Any = TravelLeg(target_x=1.0, target_y=2.0, mode=TravelMode.PATHFIND)
        with pytest.raises(AttributeError):
            leg.target_x = 99.0


# ---------------------------------------------------------------------------
# plan_travel_legs: fallback (no tunnels)
# ---------------------------------------------------------------------------


class TestPlanTravelLegsFallback:
    def test_no_tunnels_single_pathfind_leg(self) -> None:
        legs = plan_travel_legs([], 0.0, 0.0, 500.0, 500.0)
        assert len(legs) == 1
        assert legs[0].mode == TravelMode.PATHFIND
        assert legs[0].target_x == 500.0
        assert legs[0].target_y == 500.0

    def test_tunnel_not_matching_falls_back(self) -> None:
        route = TunnelRoute(
            name="far_tunnel",
            from_waypoint="x",
            to_waypoint="y",
            from_pos=Point(9000.0, 9000.0, 0.0),
            to_pos=Point(9500.0, 9500.0, 0.0),
            points=(TravelWaypoint(9100.0, 9100.0, 0.0),),
        )
        legs = plan_travel_legs([route], 0.0, 0.0, 100.0, 100.0)
        assert len(legs) == 1
        assert legs[0].mode == TravelMode.PATHFIND


# ---------------------------------------------------------------------------
# plan_travel_legs: tunnel matching
# ---------------------------------------------------------------------------


class TestPlanTravelLegsTunnel:
    def _make_route(self) -> TunnelRoute:
        return TunnelRoute(
            name="test_tunnel",
            from_waypoint="entrance",
            to_waypoint="exit",
            from_pos=Point(0.0, 0.0, 0.0),
            to_pos=Point(1000.0, 1000.0, 0.0),
            points=(TravelWaypoint(100.0, 100.0, 0.0), TravelWaypoint(500.0, 500.0, 0.0)),
        )

    def test_forward_direct_match(self) -> None:
        route = self._make_route()
        legs = plan_travel_legs([route], 0.0, 0.0, 1000.0, 1000.0)
        assert len(legs) == 1
        assert legs[0].mode == TravelMode.MANUAL
        assert legs[0].waypoints is not None

    def test_reverse_direct_match(self) -> None:
        route = self._make_route()
        legs = plan_travel_legs([route], 1000.0, 1000.0, 0.0, 0.0)
        assert len(legs) == 1
        assert legs[0].mode == TravelMode.MANUAL

    def test_target_near_exit_adds_pathfind_leg(self) -> None:
        route = self._make_route()
        # Start far from entrance, target near exit
        legs = plan_travel_legs([route], 5000.0, 5000.0, 1000.0, 1000.0)
        assert len(legs) == 2
        assert legs[0].mode == TravelMode.PATHFIND
        assert legs[1].mode == TravelMode.MANUAL


# ---------------------------------------------------------------------------
# plan_travel_legs: waypoint graph
# ---------------------------------------------------------------------------


class TestPlanTravelLegsGraph:
    def _make_graph(self) -> WaypointGraph:
        g = WaypointGraph()
        g.add_node("start", 0.0, 0.0)
        g.add_node("mid", 500.0, 500.0)
        g.add_node("end", 1000.0, 1000.0)
        g.add_edge("start", "mid")
        g.add_edge("mid", "end")
        return g

    def test_graph_route_generates_legs(self) -> None:
        g = self._make_graph()
        legs = plan_travel_legs([], 0.0, 0.0, 1000.0, 1000.0, waypoint_graph=g)
        # Should be 2 legs: start->mid, mid->end
        assert len(legs) == 2
        assert all(leg.mode == TravelMode.PATHFIND for leg in legs)

    def test_graph_same_node_falls_back(self) -> None:
        g = self._make_graph()
        # Start and target both near "start" node
        legs = plan_travel_legs([], 0.0, 0.0, 5.0, 5.0, waypoint_graph=g)
        assert len(legs) == 1
        assert legs[0].mode == TravelMode.PATHFIND


# ---------------------------------------------------------------------------
# subdivide_waypoints
# ---------------------------------------------------------------------------


class TestSubdivideWaypoints:
    def test_short_segment_unchanged(self) -> None:
        wps = [Point(10.0, 0.0, 0.0)]
        result = subdivide_waypoints(0.0, 0.0, wps, max_segment=200.0)
        assert len(result) == 1
        assert result[0] == Point(10.0, 0.0, 0.0)

    def test_long_segment_subdivided(self) -> None:
        wps = [Point(500.0, 0.0, 0.0)]
        result = subdivide_waypoints(0.0, 0.0, wps, max_segment=200.0, step=150.0)
        assert len(result) > 1
        # Final point should be near the original waypoint
        assert abs(result[-1].x - 500.0) < 1.0


# ---------------------------------------------------------------------------
# _find_tunnel_between -- coordinate-based fallback
# ---------------------------------------------------------------------------


class TestFindTunnelBetween:
    def _make_route(self) -> TunnelRoute:
        return TunnelRoute(
            name="coord_tunnel",
            from_waypoint="entrance",
            to_waypoint="exit",
            from_pos=Point(100.0, 100.0, 0.0),
            to_pos=Point(900.0, 900.0, 0.0),
            points=(TravelWaypoint(500.0, 500.0, 0.0),),
        )

    def test_name_match_forward(self) -> None:
        from nav.travel_planner import _find_tunnel_between

        route = self._make_route()
        result = _find_tunnel_between([route], "entrance", "exit", 0, 0, 0, 0)
        assert result is not None
        assert len(result) == 1

    def test_name_match_reversed(self) -> None:
        from nav.travel_planner import _find_tunnel_between

        route = self._make_route()
        result = _find_tunnel_between([route], "exit", "entrance", 0, 0, 0, 0)
        assert result is not None
        assert len(result) == 1

    def test_coord_fallback_forward(self) -> None:
        from nav.travel_planner import _find_tunnel_between

        route = self._make_route()
        # Names don't match, but coords are close
        result = _find_tunnel_between(
            [route], "unknown_a", "unknown_b", 110.0, 110.0, 910.0, 910.0, threshold=500.0
        )
        assert result is not None

    def test_coord_fallback_reversed(self) -> None:
        from nav.travel_planner import _find_tunnel_between

        route = self._make_route()
        result = _find_tunnel_between(
            [route], "unknown_a", "unknown_b", 910.0, 910.0, 110.0, 110.0, threshold=500.0
        )
        assert result is not None

    def test_no_match(self) -> None:
        from nav.travel_planner import _find_tunnel_between

        route = self._make_route()
        result = _find_tunnel_between(
            [route], "far_a", "far_b", 5000.0, 5000.0, 6000.0, 6000.0, threshold=100.0
        )
        assert result is None


# ---------------------------------------------------------------------------
# plan_travel_legs: tunnel case 4 (target near from, reversed)
# ---------------------------------------------------------------------------


class TestPlanTravelLegsTunnelCase4:
    def _make_route(self) -> TunnelRoute:
        return TunnelRoute(
            name="test_tunnel",
            from_waypoint="entrance",
            to_waypoint="exit",
            from_pos=Point(0.0, 0.0, 0.0),
            to_pos=Point(1000.0, 1000.0, 0.0),
            points=(TravelWaypoint(100.0, 100.0, 0.0), TravelWaypoint(500.0, 500.0, 0.0)),
        )

    def test_target_near_from_adds_reverse_path(self) -> None:
        """Case 4: target near 'from' endpoint, start far from 'to'."""
        route = self._make_route()
        # Start is far, target is near from_pos (0,0)
        legs = plan_travel_legs([route], 5000.0, 5000.0, 0.0, 0.0)
        assert len(legs) == 2
        assert legs[0].mode == TravelMode.PATHFIND
        assert legs[1].mode == TravelMode.MANUAL


# ---------------------------------------------------------------------------
# plan_travel_legs: waypoint graph with tunnel edge
# ---------------------------------------------------------------------------


class TestPlanTravelLegsGraphTunnel:
    def test_graph_with_tunnel_edge(self) -> None:
        """Waypoint graph route uses tunnel when available on edge."""
        g = WaypointGraph()
        g.add_node("a", 0.0, 0.0)
        g.add_node("b", 500.0, 500.0)
        g.add_node("c", 1000.0, 1000.0)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        # Tunnel between b and c
        tunnel = TunnelRoute(
            name="bc_tunnel",
            from_waypoint="b",
            to_waypoint="c",
            from_pos=Point(500.0, 500.0, 0.0),
            to_pos=Point(1000.0, 1000.0, 0.0),
            points=(TravelWaypoint(700.0, 700.0, 0.0),),
        )
        legs = plan_travel_legs([tunnel], 0.0, 0.0, 1000.0, 1000.0, waypoint_graph=g)
        # Should be a->b (pathfind), b->c (tunnel/manual)
        assert len(legs) == 2
        modes = [leg.mode for leg in legs]
        assert TravelMode.MANUAL in modes

    def test_graph_start_far_from_first_node_prepends_leg(self) -> None:
        """When start is far from the first graph node, a pathfind leg is prepended."""
        g = WaypointGraph()
        g.add_node("mid", 500.0, 500.0)
        g.add_node("end", 1000.0, 1000.0)
        g.add_edge("mid", "end")
        # Start at 400,400 -- near "mid" (within threshold), but >50 units away
        legs = plan_travel_legs([], 400.0, 400.0, 1000.0, 1000.0, waypoint_graph=g)
        # Should have a prepend leg (to mid) + mid->end
        assert len(legs) >= 2
        assert legs[0].mode == TravelMode.PATHFIND

    def test_graph_path_trimming(self) -> None:
        """When agent is closer to a mid-path node, earlier nodes are trimmed."""
        g = WaypointGraph()
        g.add_node("a", 0.0, 0.0)
        g.add_node("b", 100.0, 100.0)
        g.add_node("c", 200.0, 200.0)
        g.add_node("d", 300.0, 300.0)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "d")
        # Start at (95, 95) -- closer to "b" than "a"
        legs = plan_travel_legs([], 95.0, 95.0, 300.0, 300.0, waypoint_graph=g)
        # Path should be trimmed to start from "b", so b->c->d = 2 legs
        assert len(legs) == 2

    def test_graph_no_match_both_nodes_missing(self) -> None:
        """Nodes not near any waypoint -> falls back to single A* leg."""
        g = WaypointGraph()
        g.add_node("a", 0.0, 0.0)
        g.add_node("b", 100.0, 100.0)
        g.add_edge("a", "b")
        legs = plan_travel_legs([], 5000.0, 5000.0, 6000.0, 6000.0, waypoint_graph=g)
        assert len(legs) == 1
        assert legs[0].mode == TravelMode.PATHFIND


# ---------------------------------------------------------------------------
# find_tunnel_route (deprecated wrapper)
# ---------------------------------------------------------------------------


class TestFindTunnelRoute:
    def test_single_tunnel_returns_waypoints(self) -> None:
        import warnings

        from nav.travel_planner import find_tunnel_route

        route = TunnelRoute(
            name="test",
            from_waypoint="a",
            to_waypoint="b",
            from_pos=Point(0.0, 0.0, 0.0),
            to_pos=Point(100.0, 100.0, 0.0),
            points=(TravelWaypoint(50.0, 50.0, 0.0),),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = find_tunnel_route([route], 0.0, 0.0, 100.0, 100.0)
        assert result is not None
        assert len(result) == 1

    def test_no_tunnel_returns_none(self) -> None:
        import warnings

        from nav.travel_planner import find_tunnel_route

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = find_tunnel_route([], 0.0, 0.0, 100.0, 100.0)
        assert result is None

    def test_emits_deprecation_warning(self) -> None:
        import warnings

        from nav.travel_planner import find_tunnel_route

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            find_tunnel_route([], 0.0, 0.0, 100.0, 100.0)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
