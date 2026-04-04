"""Waypoint graph  -  intra-zone routing through safe intermediate points.

Named waypoints connected by bidirectional edges. Travel between non-adjacent
waypoints routes through intermediate nodes via BFS. Each edge is traversed
via A* pathfinding unless a tunnel_route provides a manual path.

Used by travel planning for multi-leg routes and corpse recovery.
"""

import logging
from collections import deque

from core.types import Point

log = logging.getLogger(__name__)


class WaypointGraph:
    """Graph of named waypoints within a zone.

    Waypoints are connected by edges (bidirectional). Travel between
    non-adjacent waypoints routes through intermediate nodes via BFS.
    Each edge is traversed via A* pathfinding unless a tunnel_route
    provides a manual path for that specific edge.
    """

    def __init__(self) -> None:
        self.coords: dict[str, Point] = {}
        self.edges: dict[str, set[str]] = {}

    def add_node(self, name: str, pos: Point) -> None:
        self.coords[name] = pos
        if name not in self.edges:
            self.edges[name] = set()

    def add_edge(self, a: str, b: str) -> None:
        """Add bidirectional edge between two waypoints."""
        if a not in self.edges:
            self.edges[a] = set()
        if b not in self.edges:
            self.edges[b] = set()
        self.edges[a].add(b)
        self.edges[b].add(a)

    def nearest_node(self, pos: Point, threshold: float = 500.0) -> str | None:
        """Find the nearest waypoint within threshold distance (2D)."""
        best_name = None
        best_dist = threshold
        for name, pt in self.coords.items():
            d = pos.dist_2d(pt)
            if d < best_dist:
                best_dist = d
                best_name = name
        return best_name

    def find_path(self, start: str, end: str) -> list[str] | None:
        """BFS shortest path through waypoint names. Returns node list."""
        if start == end:
            return [start]
        if start not in self.edges or end not in self.edges:
            return None

        parent: dict[str, str] = {start: start}
        queue: deque[str] = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor in self.edges.get(current, set()):
                if neighbor in parent:
                    continue
                parent[neighbor] = current
                if neighbor == end:
                    # Reconstruct path from parent pointers
                    path = [end]
                    node = end
                    while parent[node] != node:
                        node = parent[node]
                        path.append(node)
                    path.reverse()
                    return path
                queue.append(neighbor)
        return None

    def __repr__(self) -> str:
        edge_count = sum(len(v) for v in self.edges.values()) // 2
        return f"WaypointGraph({len(self.coords)} nodes, {edge_count} edges)"


def parse_waypoint_graph(zone_config: dict) -> WaypointGraph:
    """Build waypoint graph from zone config.

    Reads [[waypoints]] for node coords and [[waypoint_edges]] for
    connectivity. Also extra_npcs tunnel_route endpoints as edges.
    """
    graph = WaypointGraph()

    # Add all waypoints as nodes
    for wp in zone_config.get("waypoints", []):
        graph.add_node(wp["name"], Point(wp["x"], wp["y"], wp.get("z", 0.0)))

    # Add edges from [[waypoint_edges]] chains
    for edge_def in zone_config.get("waypoint_edges", []):
        points = edge_def.get("points", [])
        for i in range(len(points) - 1):
            a, b = points[i], points[i + 1]
            if a in graph.coords and b in graph.coords:
                graph.add_edge(a, b)
            else:
                missing = [n for n in (a, b) if n not in graph.coords]
                log.warning("[NAV] Waypoint edge: unknown node(s) %s", missing)

    # Add tunnel route endpoints as edges (they define connectivity too)
    for tr in zone_config.get("tunnel_routes", []):
        a = tr.get("from_waypoint", "")
        b = tr.get("to_waypoint", "")
        if a in graph.coords and b in graph.coords:
            graph.add_edge(a, b)

    if graph.coords:
        edge_count = sum(len(v) for v in graph.edges.values()) // 2
        log.info("[NAV] Waypoint graph: %d nodes, %d edges", len(graph.coords), edge_count)

    return graph
