"""Zone connection graph for multi-zone navigation.

Defines which zones connect to which, with zoneline coordinates in both
the source and destination zones. Supports pathfinding across zone
boundaries (BFS shortest path through the graph).

Zone connections are populated from map POI labels ("To_X")
and supplemented with hardcoded arrival coordinates.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from core.types import Point
from nav.map_data import load_zone_map

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ZoneConnection:
    """A one-way connection from one zone to another."""

    from_zone: str  # source zone short name
    to_zone: str  # destination zone short name
    zoneline_x: float  # game coordinates of the zoneline IN source zone
    zoneline_y: float
    arrival_x: float  # where you appear IN destination zone
    arrival_y: float


# -- Zone name normalization ------------------------------------------

# "You have entered X" display name -> zone short name
ZONE_DISPLAY_TO_SHORT: dict[str, str] = {
    "Nektulos Forest": "nektulos",
    "East Commonlands": "ecommons",
    "West Commonlands": "commons",
    "Neriak - Commons": "neriaka",
    "Neriak - Foreign Quarter": "neriakb",
    "Neriak Third Gate": "neriakc",
    "Lavastorm Mountains": "lavastorm",
    "The Nektulos Forest": "nektulos",
    "Greater Faydark": "gfaydark",
    "Toxxulia Forest": "tox",
    "Northern Desert of Ro": "nro",
    "Southern Desert of Ro": "sro",
    "Oasis of Marr": "oasis",
}

# Map POI label -> destination zone short name
_POI_LABEL_TO_ZONE: dict[str, str] = {
    "to_east_commonlands": "ecommons",
    "to_neriak": "neriaka",
    "to_lavastorm": "lavastorm",
    "to_corathus": "corathus",
    "to_nektulos": "nektulos",
    "to_nektulos_forest": "nektulos",
    "nektulos_forrest": "nektulos",  # typo in lavastorm map
    "nektulos_forest": "nektulos",
    "to_neriak_commons": "neriaka",
    "to_neriak_foreign_quarter": "neriakb",
    "to_foreign_quarter": "neriakb",
    "to_third_gate": "neriakc",
    "to_west_commonlands": "commons",
    "to_lavastorm_mountains": "lavastorm",
    "to_north_ro": "nro",
    "to_northern_desert_of_ro": "nro",
    "to_oasis": "oasis",
    "to_oasis_of_marr": "oasis",
    "to_south_ro": "sro",
    "to_southern_desert_of_ro": "sro",
}


def normalize_zone_name(display_name: str) -> str:
    """Convert display zone name to short name. Returns lowercase if unknown."""
    short = ZONE_DISPLAY_TO_SHORT.get(display_name)
    if short:
        return short
    # Fallback: lowercase, replace spaces with nothing
    return display_name.lower().replace(" ", "").replace("-", "")


def _poi_label_to_zone(label: str) -> str | None:
    """Convert a map POI label like 'To_East_Commonlands' to zone short name."""
    key = label.lower().strip()
    return _POI_LABEL_TO_ZONE.get(key)


# -- Hardcoded arrival coordinates ------------------------------------
# Where you appear after crossing a zoneline. Extracted from map POIs
# in the DESTINATION zone (the "to_SourceZone" POIs).

_ARRIVAL_COORDS: dict[tuple[str, str], Point] = {
    # (from_zone, to_zone) -> Point(arrival_x, arrival_y, arrival_z) in destination zone
    ("nektulos", "ecommons"): Point(-615.7, -1483.7, 0.0),  # EC's "to_Nektulos" POI
    ("ecommons", "nektulos"): Point(108.3, 1699.4, 0.0),  # Nektulos's "To_East_Commonlands"
    ("nektulos", "neriaka"): Point(-155.2, -20.7, 0.0),  # Neriaka's "to_Nektulos_Forest"
    ("neriaka", "nektulos"): Point(653.4, -752.6, 0.0),  # Nektulos's "To_Neriak"
    ("nektulos", "lavastorm"): Point(7.1, -177.7, 0.0),  # Lavastorm's "Nektulos_Forrest"
    ("lavastorm", "nektulos"): Point(-317.1, -1619.8, 0.0),  # Nektulos's "To_Lavastorm"
    ("neriaka", "neriakb"): Point(
        0.0, 0.0, 0.0
    ),  # Arrival coordinates not recorded; using zone safe point (0, 0)
    ("neriakb", "neriaka"): Point(
        0.0, 0.0, 0.0
    ),  # Arrival coordinates not recorded; using zone safe point (0, 0)
    # NRO <-> EC (through tunnel)
    ("ecommons", "nro"): Point(2903.0, 2604.0, 0.0),  # NRO's "EC Zoneline" waypoint
    ("nro", "ecommons"): Point(160.0, -222.0, 0.0),  # EC's "NRO Zoneline" waypoint
    # Oasis <-> NRO
    ("nro", "oasis"): Point(121.0, 2461.0, 0.0),  # Oasis's "NRO Zoneline" waypoint
    ("oasis", "nro"): Point(0.0, 0.0, 0.0),  # Arrival coordinates not recorded; using zone safe point (0, 0)
    # Oasis <-> SRO
    ("oasis", "sro"): Point(168.0, 1473.0, 0.0),  # SRO's "Oasis Zoneline" waypoint
    ("sro", "oasis"): Point(173.0, -1845.0, 0.0),  # Oasis's "SRO Zoneline" waypoint
}


# ======================================================================
# Zone Graph
# ======================================================================


class ZoneGraph:
    """Graph of zone connections for multi-zone pathfinding."""

    def __init__(self) -> None:
        self._connections: dict[str, list[ZoneConnection]] = {}

    def add_connection(self, conn: ZoneConnection) -> None:
        if conn.from_zone not in self._connections:
            self._connections[conn.from_zone] = []
        self._connections[conn.from_zone].append(conn)

    def get_connections(self, zone: str) -> list[ZoneConnection]:
        """Get all outbound connections from a zone."""
        return self._connections.get(zone, [])

    def get_connection(self, from_zone: str, to_zone: str) -> ZoneConnection | None:
        """Get the connection between two specific zones."""
        for conn in self.get_connections(from_zone):
            if conn.to_zone == to_zone:
                return conn
        return None

    def find_route(self, from_zone: str, to_zone: str) -> list[ZoneConnection] | None:
        """BFS shortest path through the zone graph.

        Returns ordered list of ZoneConnections to follow, or None.
        """
        if from_zone == to_zone:
            return []

        from collections import deque

        # parent[zone] = the ZoneConnection that first reached it
        parent: dict[str, ZoneConnection | None] = {from_zone: None}
        queue: deque[str] = deque([from_zone])

        while queue:
            current_zone = queue.popleft()
            for conn in self.get_connections(current_zone):
                if conn.to_zone in parent:
                    continue
                parent[conn.to_zone] = conn
                if conn.to_zone == to_zone:
                    # Reconstruct connection list from parent pointers
                    route: list[ZoneConnection] = []
                    zone = to_zone
                    while parent[zone] is not None:
                        conn_back = parent[zone]
                        assert conn_back is not None
                        route.append(conn_back)
                        zone = conn_back.from_zone
                    route.reverse()
                    return route
                queue.append(conn.to_zone)

        return None

    @property
    def zones(self) -> list[str]:
        """All zones that have outbound connections."""
        zones = set(self._connections.keys())
        for conns in self._connections.values():
            for c in conns:
                zones.add(c.to_zone)
        return sorted(zones)

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._connections.values())
        return f"ZoneGraph({len(self.zones)} zones, {total} connections)"


# ======================================================================
# Build zone graph from map files
# ======================================================================


def build_zone_graph(maps_dir: str | Path) -> ZoneGraph:
    """Build zone graph by scanning all map files for "To_X" POIs.

    Args:
        maps_dir: Path to maps directory (e.g., .../maps/)

    Returns:
        ZoneGraph with connections extracted from POI labels.
    """
    maps_dir = Path(maps_dir)
    graph = ZoneGraph()

    if not maps_dir.exists():
        log.warning("[NAV] Maps directory not found: %s", maps_dir)
        return graph

    # Suppress per-file logging during bulk scan (90+ map files)
    import logging as _logging

    map_logger = _logging.getLogger("compass.nav.map_data")
    prev_level = map_logger.level
    map_logger.setLevel(_logging.WARNING)

    for map_file in sorted(maps_dir.glob("*.txt")):
        # Zone short name from filename (e.g., "nektulos.txt" -> "nektulos")
        # Skip overlay files like "nektulos_1.txt"
        stem = map_file.stem
        if "_" in stem and stem.split("_")[-1].isdigit():
            continue

        zone_map = load_zone_map(map_file)

        for poi in zone_map.pois:
            dest_zone = _poi_label_to_zone(poi.label)
            if dest_zone is None:
                continue

            # Look up arrival coordinates in destination zone
            arrival = _ARRIVAL_COORDS.get((stem, dest_zone), (0.0, 0.0))

            conn = ZoneConnection(
                from_zone=stem,
                to_zone=dest_zone,
                zoneline_x=poi.x,
                zoneline_y=poi.y,
                arrival_x=arrival[0],
                arrival_y=arrival[1],
            )
            graph.add_connection(conn)
            log.debug("[NAV] Zone connection: %s -> %s at (%.0f, %.0f)", stem, dest_zone, poi.x, poi.y)

    map_logger.setLevel(prev_level)

    # -- Manual connections for zones without "To_X" POIs --
    _manual = [
        # EC <-> NRO (through tunnel)
        ("ecommons", "nro", 160.0, -222.0),  # EC zoneline coords in EC
        ("nro", "ecommons", 2903.0, 2604.0),  # NRO EC Zoneline coords in NRO
        # NRO <-> Oasis
        ("nro", "oasis", 0.0, 0.0),  # Zoneline coordinates not recorded; using (0, 0)
        ("oasis", "nro", 121.0, 2461.0),  # Oasis NRO Zoneline coords
        # Oasis <-> SRO
        ("oasis", "sro", 173.0, -1845.0),  # Oasis SRO Zoneline coords
        ("sro", "oasis", 168.0, 1473.0),  # SRO Oasis Zoneline coords
    ]
    for from_z, to_z, zl_x, zl_y in _manual:
        # Skip if already added from map files
        if graph.get_connection(from_z, to_z):
            continue
        arrival = _ARRIVAL_COORDS.get((from_z, to_z), (0.0, 0.0))
        conn = ZoneConnection(
            from_zone=from_z,
            to_zone=to_z,
            zoneline_x=zl_x,
            zoneline_y=zl_y,
            arrival_x=arrival[0],
            arrival_y=arrival[1],
        )
        graph.add_connection(conn)
        log.debug("[NAV] Zone connection (manual): %s -> %s at (%.0f, %.0f)", from_z, to_z, zl_x, zl_y)

    log.debug("[NAV] Zone graph: %s", graph)
    return graph
