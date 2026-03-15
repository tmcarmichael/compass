"""Tests for zone graph routing (src/nav/zone_graph.py)."""

from __future__ import annotations

from typing import Any

import pytest

from nav.zone_graph import ZoneConnection, ZoneGraph, normalize_zone_name

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conn(from_zone: str, to_zone: str) -> ZoneConnection:
    """Create a minimal ZoneConnection for testing."""
    return ZoneConnection(
        from_zone=from_zone,
        to_zone=to_zone,
        zoneline_x=0.0,
        zoneline_y=0.0,
        arrival_x=0.0,
        arrival_y=0.0,
    )


# ---------------------------------------------------------------------------
# Adding connections
# ---------------------------------------------------------------------------


class TestAddConnection:
    def test_add_single_connection(self) -> None:
        g = ZoneGraph()
        g.add_connection(_conn("a", "b"))
        conns = g.get_connections("a")
        assert len(conns) == 1
        assert conns[0].to_zone == "b"

    def test_add_multiple_connections_from_same_zone(self) -> None:
        g = ZoneGraph()
        g.add_connection(_conn("a", "b"))
        g.add_connection(_conn("a", "c"))
        conns = g.get_connections("a")
        assert len(conns) == 2
        destinations = {c.to_zone for c in conns}
        assert destinations == {"b", "c"}

    def test_get_connections_empty_for_unknown_zone(self) -> None:
        g = ZoneGraph()
        assert g.get_connections("nonexistent") == []

    def test_get_connection_specific(self) -> None:
        g = ZoneGraph()
        g.add_connection(_conn("a", "b"))
        g.add_connection(_conn("a", "c"))
        conn = g.get_connection("a", "b")
        assert conn is not None
        assert conn.to_zone == "b"

    def test_get_connection_returns_none_when_missing(self) -> None:
        g = ZoneGraph()
        g.add_connection(_conn("a", "b"))
        assert g.get_connection("a", "c") is None

    def test_zones_property(self) -> None:
        g = ZoneGraph()
        g.add_connection(_conn("a", "b"))
        g.add_connection(_conn("b", "c"))
        zones = g.zones
        assert set(zones) == {"a", "b", "c"}

    def test_repr(self) -> None:
        g = ZoneGraph()
        g.add_connection(_conn("a", "b"))
        r = repr(g)
        assert "ZoneGraph" in r


# ---------------------------------------------------------------------------
# BFS shortest path
# ---------------------------------------------------------------------------


class TestFindRoute:
    def test_direct_connection(self) -> None:
        g = ZoneGraph()
        g.add_connection(_conn("a", "b"))
        route = g.find_route("a", "b")
        assert route is not None
        assert len(route) == 1
        assert route[0].from_zone == "a"
        assert route[0].to_zone == "b"

    def test_multi_hop_path(self) -> None:
        g = ZoneGraph()
        g.add_connection(_conn("a", "b"))
        g.add_connection(_conn("b", "c"))
        g.add_connection(_conn("c", "d"))
        route = g.find_route("a", "d")
        assert route is not None
        assert len(route) == 3
        assert [c.from_zone for c in route] == ["a", "b", "c"]
        assert [c.to_zone for c in route] == ["b", "c", "d"]

    def test_shortest_path_preferred(self) -> None:
        """BFS finds shortest path when multiple paths exist."""
        g = ZoneGraph()
        # Direct path: a -> d (1 hop)
        g.add_connection(_conn("a", "d"))
        # Longer path: a -> b -> c -> d (3 hops)
        g.add_connection(_conn("a", "b"))
        g.add_connection(_conn("b", "c"))
        g.add_connection(_conn("c", "d"))

        route = g.find_route("a", "d")
        assert route is not None
        assert len(route) == 1

    def test_bidirectional_connections(self) -> None:
        g = ZoneGraph()
        g.add_connection(_conn("a", "b"))
        g.add_connection(_conn("b", "a"))
        # Forward
        route_fwd = g.find_route("a", "b")
        assert route_fwd is not None
        assert len(route_fwd) == 1
        # Backward
        route_bwd = g.find_route("b", "a")
        assert route_bwd is not None
        assert len(route_bwd) == 1


# ---------------------------------------------------------------------------
# Disconnected zones
# ---------------------------------------------------------------------------


class TestDisconnected:
    def test_no_path_when_disconnected(self) -> None:
        g = ZoneGraph()
        g.add_connection(_conn("a", "b"))
        g.add_connection(_conn("c", "d"))
        route = g.find_route("a", "d")
        assert route is None

    def test_no_path_from_unknown_zone(self) -> None:
        g = ZoneGraph()
        g.add_connection(_conn("a", "b"))
        route = g.find_route("x", "b")
        assert route is None

    def test_one_way_connection_blocks_reverse(self) -> None:
        g = ZoneGraph()
        g.add_connection(_conn("a", "b"))  # a->b only
        route = g.find_route("b", "a")
        assert route is None


# ---------------------------------------------------------------------------
# Self-path (same zone)
# ---------------------------------------------------------------------------


class TestSelfPath:
    def test_same_zone_returns_empty_list(self) -> None:
        g = ZoneGraph()
        g.add_connection(_conn("a", "b"))
        route = g.find_route("a", "a")
        assert route is not None
        assert route == []

    def test_same_zone_no_connections(self) -> None:
        """Self-path works even without any connections."""
        g = ZoneGraph()
        route = g.find_route("anywhere", "anywhere")
        assert route == []


# ---------------------------------------------------------------------------
# ZoneConnection dataclass
# ---------------------------------------------------------------------------


class TestZoneConnection:
    def test_frozen(self) -> None:
        c: Any = _conn("a", "b")
        with pytest.raises(AttributeError):
            c.from_zone = "x"

    def test_fields_accessible(self) -> None:
        c = ZoneConnection(
            from_zone="nektulos",
            to_zone="ecommons",
            zoneline_x=100.0,
            zoneline_y=200.0,
            arrival_x=300.0,
            arrival_y=400.0,
        )
        assert c.from_zone == "nektulos"
        assert c.zoneline_x == 100.0
        assert c.arrival_y == 400.0


# ---------------------------------------------------------------------------
# Zone name normalization
# ---------------------------------------------------------------------------


class TestNormalizeZoneName:
    @pytest.mark.parametrize(
        "display,expected",
        [
            ("Nektulos Forest", "nektulos"),
            ("East Commonlands", "ecommons"),
            ("Greater Faydark", "gfaydark"),
        ],
    )
    def test_known_names(self, display: str, expected: str) -> None:
        assert normalize_zone_name(display) == expected

    def test_unknown_name_lowercased(self) -> None:
        result = normalize_zone_name("Some Unknown Zone")
        assert result == "someunknownzone"

    def test_unknown_name_with_hyphens(self) -> None:
        result = normalize_zone_name("Neriak - Third Gate")
        # Not in lookup, so fallback: lowercase, strip spaces and hyphens
        assert result == "neriakthirdgate"


# ---------------------------------------------------------------------------
# _poi_label_to_zone (private but exercised for coverage)
# ---------------------------------------------------------------------------


class TestPoiLabelToZone:
    def test_known_label(self) -> None:
        from nav.zone_graph import _poi_label_to_zone

        assert _poi_label_to_zone("to_east_commonlands") == "ecommons"

    def test_known_label_with_whitespace(self) -> None:
        from nav.zone_graph import _poi_label_to_zone

        assert _poi_label_to_zone("  to_neriak  ") == "neriaka"

    def test_unknown_label_returns_none(self) -> None:
        from nav.zone_graph import _poi_label_to_zone

        assert _poi_label_to_zone("some_random_poi") is None

    def test_case_insensitive(self) -> None:
        from nav.zone_graph import _poi_label_to_zone

        assert _poi_label_to_zone("To_East_Commonlands") == "ecommons"


# ---------------------------------------------------------------------------
# build_zone_graph
# ---------------------------------------------------------------------------


class TestBuildZoneGraph:
    def test_missing_directory_returns_empty_graph(self, tmp_path: object) -> None:
        from pathlib import Path

        from nav.zone_graph import build_zone_graph

        nonexistent = Path(str(tmp_path)) / "does_not_exist"
        graph = build_zone_graph(nonexistent)
        assert graph.zones == []

    def test_empty_directory_returns_graph_with_manual_connections(self, tmp_path: object) -> None:
        """An empty maps dir still adds manual connections."""
        from nav.zone_graph import build_zone_graph

        graph = build_zone_graph(str(tmp_path))
        # Manual connections include ecommons<->nro, nro<->oasis, oasis<->sro
        assert len(graph.zones) > 0
        assert graph.get_connection("ecommons", "nro") is not None
        assert graph.get_connection("nro", "oasis") is not None
        assert graph.get_connection("oasis", "sro") is not None

    def test_map_file_with_poi_creates_connections(self, tmp_path: object) -> None:
        """Map file with a To_ POI creates a zone connection."""
        from pathlib import Path

        from nav.zone_graph import build_zone_graph

        maps_dir = Path(str(tmp_path))
        # Create a nektulos.txt with a POI pointing to ecommons
        nektulos_map = maps_dir / "nektulos.txt"
        nektulos_map.write_text(
            "L 100.0, 200.0, 0.0, 300.0, 400.0, 0.0, 255, 0, 0\n"
            "P -615.7, -1483.7, 0.0, 0, 0, 0, 3, to_east_commonlands\n"
        )
        graph = build_zone_graph(maps_dir)
        conn = graph.get_connection("nektulos", "ecommons")
        assert conn is not None
        assert conn.from_zone == "nektulos"
        assert conn.to_zone == "ecommons"

    def test_overlay_files_skipped(self, tmp_path: object) -> None:
        """Files like zone_1.txt (overlay) are skipped."""
        from pathlib import Path

        from nav.zone_graph import build_zone_graph

        maps_dir = Path(str(tmp_path))
        # Create overlay file nektulos_1.txt -- should be skipped
        overlay = maps_dir / "nektulos_1.txt"
        overlay.write_text("P -615.7, -1483.7, 0.0, 0, 0, 0, 3, to_east_commonlands\n")
        graph = build_zone_graph(maps_dir)
        # No nektulos connection from overlay
        conn = graph.get_connection("nektulos_1", "ecommons")
        assert conn is None

    def test_manual_connections_not_duplicated(self, tmp_path: object) -> None:
        """If a map file already adds ecommons->nro, the manual one is skipped."""
        from pathlib import Path

        from nav.zone_graph import build_zone_graph

        maps_dir = Path(str(tmp_path))
        # Create ecommons.txt with a POI to NRO
        ec_map = maps_dir / "ecommons.txt"
        ec_map.write_text("P 160.0, -222.0, 0.0, 0, 0, 0, 3, to_north_ro\n")
        graph = build_zone_graph(maps_dir)
        # ecommons->nro should exist (from map POI)
        conns = graph.get_connections("ecommons")
        nro_conns = [c for c in conns if c.to_zone == "nro"]
        # Should only be one (either from map or manual, not both)
        assert len(nro_conns) == 1

    def test_arrival_coords_populated(self, tmp_path: object) -> None:
        """Connections with known arrival coords get them populated."""
        from pathlib import Path

        from nav.zone_graph import build_zone_graph

        maps_dir = Path(str(tmp_path))
        nektulos_map = maps_dir / "nektulos.txt"
        nektulos_map.write_text("P -615.7, -1483.7, 0.0, 0, 0, 0, 3, to_east_commonlands\n")
        graph = build_zone_graph(maps_dir)
        conn = graph.get_connection("nektulos", "ecommons")
        assert conn is not None
        # nektulos->ecommons has arrival coords in _ARRIVAL_COORDS
        assert conn.arrival_x != 0.0 or conn.arrival_y != 0.0
