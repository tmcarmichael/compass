"""Tests for waypoint graph intra-zone navigation (src/nav/waypoint_graph.py)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nav.waypoint_graph import WaypointGraph, parse_waypoint_graph

# ---------------------------------------------------------------------------
# Adding waypoints and edges
# ---------------------------------------------------------------------------


class TestAddNodesAndEdges:
    def test_add_node(self) -> None:
        g = WaypointGraph()
        g.add_node("camp", 100.0, 200.0)
        assert "camp" in g.coords
        assert g.coords["camp"].x == 100.0
        assert g.coords["camp"].y == 200.0

    def test_add_node_creates_edge_set(self) -> None:
        g = WaypointGraph()
        g.add_node("camp", 0.0, 0.0)
        assert "camp" in g.edges
        assert isinstance(g.edges["camp"], set)
        assert len(g.edges["camp"]) == 0

    def test_add_edge_bidirectional(self) -> None:
        g = WaypointGraph()
        g.add_node("a", 0.0, 0.0)
        g.add_node("b", 100.0, 0.0)
        g.add_edge("a", "b")
        assert "b" in g.edges["a"]
        assert "a" in g.edges["b"]

    def test_add_edge_creates_missing_edge_sets(self) -> None:
        """add_edge auto-creates edge sets for nodes not yet added via add_node."""
        g = WaypointGraph()
        g.add_edge("x", "y")
        assert "y" in g.edges["x"]
        assert "x" in g.edges["y"]

    def test_multiple_edges(self) -> None:
        g = WaypointGraph()
        g.add_node("center", 0.0, 0.0)
        g.add_node("north", 0.0, 100.0)
        g.add_node("south", 0.0, -100.0)
        g.add_edge("center", "north")
        g.add_edge("center", "south")
        assert g.edges["center"] == {"north", "south"}

    def test_repr(self) -> None:
        g = WaypointGraph()
        g.add_node("a", 0.0, 0.0)
        g.add_node("b", 1.0, 1.0)
        g.add_edge("a", "b")
        r = repr(g)
        assert "WaypointGraph" in r
        assert "2 nodes" in r
        assert "1 edges" in r


# ---------------------------------------------------------------------------
# Finding nearest waypoint
# ---------------------------------------------------------------------------


class TestNearestNode:
    def test_nearest_node_basic(self) -> None:
        g = WaypointGraph()
        g.add_node("close", 10.0, 10.0)
        g.add_node("far", 500.0, 500.0)
        result = g.nearest_node(0.0, 0.0)
        assert result == "close"

    def test_nearest_node_with_threshold(self) -> None:
        g = WaypointGraph()
        g.add_node("far", 1000.0, 1000.0)
        result = g.nearest_node(0.0, 0.0, threshold=100.0)
        assert result is None

    def test_nearest_node_returns_none_when_empty(self) -> None:
        g = WaypointGraph()
        result = g.nearest_node(0.0, 0.0)
        assert result is None

    def test_nearest_node_exact_match(self) -> None:
        g = WaypointGraph()
        g.add_node("here", 50.0, 50.0)
        result = g.nearest_node(50.0, 50.0)
        assert result == "here"

    @pytest.mark.parametrize(
        "px,py,expected",
        [
            (0.0, 0.0, "sw"),
            (100.0, 100.0, "ne"),
            (100.0, 0.0, "se"),
            (0.0, 100.0, "nw"),
        ],
    )
    def test_nearest_node_four_corners(self, px: float, py: float, expected: str) -> None:
        g = WaypointGraph()
        g.add_node("sw", 0.0, 0.0)
        g.add_node("se", 100.0, 0.0)
        g.add_node("nw", 0.0, 100.0)
        g.add_node("ne", 100.0, 100.0)
        result = g.nearest_node(px, py)
        assert result == expected

    @given(
        x=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        y=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_nearest_node_always_within_threshold(self, x: float, y: float) -> None:
        """Property: if nearest_node returns a name, it is within threshold."""
        g = WaypointGraph()
        g.add_node("origin", 0.0, 0.0)
        threshold = 2000.0
        result = g.nearest_node(x, y, threshold=threshold)
        if result is not None:
            p = g.coords[result]
            dist = math.hypot(x - p.x, y - p.y)
            assert dist < threshold


# ---------------------------------------------------------------------------
# Path finding between connected waypoints
# ---------------------------------------------------------------------------


class TestFindPath:
    def test_direct_path(self) -> None:
        g = WaypointGraph()
        g.add_node("a", 0.0, 0.0)
        g.add_node("b", 100.0, 0.0)
        g.add_edge("a", "b")
        path = g.find_path("a", "b")
        assert path == ["a", "b"]

    def test_multi_hop_path(self) -> None:
        g = WaypointGraph()
        g.add_node("a", 0.0, 0.0)
        g.add_node("b", 50.0, 0.0)
        g.add_node("c", 100.0, 0.0)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        path = g.find_path("a", "c")
        assert path == ["a", "b", "c"]

    def test_shortest_path_preferred(self) -> None:
        """BFS finds shortest path when multiple exist."""
        g = WaypointGraph()
        g.add_node("a", 0.0, 0.0)
        g.add_node("b", 50.0, 0.0)
        g.add_node("c", 100.0, 0.0)
        g.add_node("d", 100.0, 50.0)
        # Direct: a -> d
        g.add_edge("a", "d")
        # Long: a -> b -> c -> d
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "d")
        path = g.find_path("a", "d")
        assert path == ["a", "d"]

    def test_path_is_bidirectional(self) -> None:
        g = WaypointGraph()
        g.add_node("a", 0.0, 0.0)
        g.add_node("b", 100.0, 0.0)
        g.add_edge("a", "b")
        assert g.find_path("a", "b") is not None
        assert g.find_path("b", "a") is not None


# ---------------------------------------------------------------------------
# No path between disconnected waypoints
# ---------------------------------------------------------------------------


class TestDisconnectedPath:
    def test_no_path_disconnected(self) -> None:
        g = WaypointGraph()
        g.add_node("a", 0.0, 0.0)
        g.add_node("b", 100.0, 0.0)
        g.add_node("c", 200.0, 0.0)
        g.add_edge("a", "b")
        # c is disconnected
        path = g.find_path("a", "c")
        assert path is None

    def test_no_path_unknown_start(self) -> None:
        g = WaypointGraph()
        g.add_node("a", 0.0, 0.0)
        path = g.find_path("unknown", "a")
        assert path is None

    def test_no_path_unknown_end(self) -> None:
        g = WaypointGraph()
        g.add_node("a", 0.0, 0.0)
        path = g.find_path("a", "unknown")
        assert path is None

    def test_two_disconnected_components(self) -> None:
        g = WaypointGraph()
        g.add_node("a1", 0.0, 0.0)
        g.add_node("a2", 10.0, 0.0)
        g.add_edge("a1", "a2")
        g.add_node("b1", 1000.0, 0.0)
        g.add_node("b2", 1010.0, 0.0)
        g.add_edge("b1", "b2")
        assert g.find_path("a1", "b1") is None
        assert g.find_path("b2", "a1") is None


# ---------------------------------------------------------------------------
# Self-path (same waypoint)
# ---------------------------------------------------------------------------


class TestSelfPath:
    def test_same_node_returns_single_element(self) -> None:
        g = WaypointGraph()
        g.add_node("camp", 0.0, 0.0)
        path = g.find_path("camp", "camp")
        assert path == ["camp"]


# ---------------------------------------------------------------------------
# parse_waypoint_graph from zone config
# ---------------------------------------------------------------------------


class TestParseWaypointGraph:
    def test_empty_config(self) -> None:
        g = parse_waypoint_graph({})
        assert len(g.coords) == 0

    def test_waypoints_only(self) -> None:
        config = {
            "waypoints": [
                {"name": "camp", "x": 100.0, "y": 200.0},
                {"name": "bridge", "x": 300.0, "y": 400.0},
            ]
        }
        g = parse_waypoint_graph(config)
        assert len(g.coords) == 2
        assert "camp" in g.coords
        assert "bridge" in g.coords

    def test_waypoints_with_edges(self) -> None:
        config = {
            "waypoints": [
                {"name": "a", "x": 0.0, "y": 0.0},
                {"name": "b", "x": 100.0, "y": 0.0},
                {"name": "c", "x": 200.0, "y": 0.0},
            ],
            "waypoint_edges": [{"points": ["a", "b", "c"]}],
        }
        g = parse_waypoint_graph(config)
        assert "b" in g.edges["a"]
        assert "a" in g.edges["b"]
        assert "c" in g.edges["b"]
        assert "b" in g.edges["c"]

    def test_tunnel_routes_add_edges(self) -> None:
        config = {
            "waypoints": [
                {"name": "entrance", "x": 0.0, "y": 0.0},
                {"name": "exit", "x": 500.0, "y": 0.0},
            ],
            "tunnel_routes": [
                {"from_waypoint": "entrance", "to_waypoint": "exit"},
            ],
        }
        g = parse_waypoint_graph(config)
        assert "exit" in g.edges["entrance"]
        assert "entrance" in g.edges["exit"]
