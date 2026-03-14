"""Travel planning: multi-leg route computation with waypoint graphs + tunnel routes.

Pure planning logic -- no routine state machines. Computes TravelLeg sequences
that callers (TravelRoutine, MultiLegTravelRoutine) execute sequentially.

Planning priority:
  1. Waypoint graph BFS: if both endpoints match named waypoints, walk the graph
     and generate a leg per edge (A* or tunnel).
  2. Tunnel routes: if target matches a tunnel endpoint, chain A* to entrance +
     manual through tunnel.
  3. Fallback: single A* leg.
"""

import logging
import math
import warnings
from dataclasses import dataclass

from core.types import Point, TravelMode, TravelWaypoint
from nav.waypoint_graph import WaypointGraph

log = logging.getLogger(__name__)


def subdivide_waypoints(
    start_x: float, start_y: float, waypoints: list[Point], max_segment: float = 200.0, step: float = 150.0
) -> list[Point]:
    """Break long waypoint segments into sub-segments for reliable move_to_point."""
    result: list[Point] = []
    prev: Point = Point(start_x, start_y, 0.0)
    for wp in waypoints:
        seg_dist = math.hypot(wp[0] - prev[0], wp[1] - prev[1])
        if seg_dist > max_segment:
            steps = int(seg_dist / step) + 1
            for s in range(1, steps + 1):
                frac = s / steps
                ix = prev[0] + (wp[0] - prev[0]) * frac
                iy = prev[1] + (wp[1] - prev[1]) * frac
                result.append(Point(ix, iy, 0.0))
        else:
            result.append(wp)
        prev = wp
    return result


@dataclass(frozen=True, slots=True)
class TunnelRoute:
    """A parsed tunnel route between two named waypoints."""

    name: str
    from_waypoint: str
    to_waypoint: str
    from_pos: Point
    to_pos: Point
    points: tuple[TravelWaypoint, ...]


def parse_tunnel_routes(zone_config: dict) -> list[TunnelRoute]:
    """Parse tunnel routes from zone config, resolving waypoint names to coords."""
    waypoints: dict[str, Point] = {}
    for wp in zone_config.get("waypoints", []):
        waypoints[wp["name"]] = Point(wp["x"], wp["y"], wp.get("z", 0.0))

    routes: list[TunnelRoute] = []
    for tr in zone_config.get("tunnel_routes", []):
        from_name = tr.get("from_waypoint", "")
        to_name = tr.get("to_waypoint", "")
        if from_name not in waypoints or to_name not in waypoints:
            log.warning("[TRAVEL] Tunnel route '%s': unknown waypoint(s)", tr.get("name", "?"))
            continue
        pts = tuple(
            TravelWaypoint(p["x"], p["y"], p.get("z", 0.0), p.get("action", "")) for p in tr.get("points", [])
        )
        if not pts:
            continue
        routes.append(
            TunnelRoute(
                name=tr.get("name", "?"),
                from_waypoint=from_name,
                to_waypoint=to_name,
                from_pos=waypoints[from_name],
                to_pos=waypoints[to_name],
                points=pts,
            )
        )
        log.info(
            "[TRAVEL] Tunnel route '%s': %d waypoints, %s -> %s",
            tr.get("name", "?"),
            len(pts),
            from_name,
            to_name,
        )
    return routes


@dataclass(frozen=True, slots=True)
class TravelLeg:
    """One leg of a multi-leg travel plan."""

    target_x: float
    target_y: float
    mode: TravelMode
    waypoints: tuple[TravelWaypoint, ...] | None = None


def _find_tunnel_between(
    tunnel_routes: list[TunnelRoute],
    from_name: str,
    to_name: str,
    from_x: float,
    from_y: float,
    to_x: float,
    to_y: float,
    threshold: float = 500.0,
) -> tuple[TravelWaypoint, ...] | None:
    """Check if a tunnel route exists between two waypoints.

    Matches by waypoint name first, then falls back to coordinate proximity.
    Returns manual waypoints (possibly reversed) or None.
    """
    for route in tunnel_routes:
        # Name-based match (exact)
        if route.from_waypoint == from_name and route.to_waypoint == to_name:
            return route.points
        if route.from_waypoint == to_name and route.to_waypoint == from_name:
            return tuple(reversed(route.points))

        # Coordinate-based fallback
        fx, fy = route.from_pos.x, route.from_pos.y
        tx, ty = route.to_pos.x, route.to_pos.y

        if math.hypot(from_x - fx, from_y - fy) < threshold and math.hypot(to_x - tx, to_y - ty) < threshold:
            return route.points
        if math.hypot(from_x - tx, from_y - ty) < threshold and math.hypot(to_x - fx, to_y - fy) < threshold:
            return tuple(reversed(route.points))

    return None


def _plan_via_waypoint_graph(
    waypoint_graph: WaypointGraph,
    tunnel_routes: list[TunnelRoute],
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
    threshold: float,
) -> list[TravelLeg] | None:
    """Build travel legs using waypoint graph BFS. Returns None if not applicable."""
    start_node = waypoint_graph.nearest_node(start_x, start_y, threshold)
    target_node = waypoint_graph.nearest_node(target_x, target_y, threshold)

    if not start_node or not target_node:
        log.debug(
            "[TRAVEL] Waypoint graph skip: start_node=%s target_node=%s "
            "(threshold=%.0f, start=(%.0f,%.0f), target=(%.0f,%.0f))",
            start_node,
            target_node,
            threshold,
            start_x,
            start_y,
            target_x,
            target_y,
        )
        return None
    if start_node == target_node:
        log.debug("[TRAVEL] Waypoint graph skip: same node '%s'", start_node)
        return None

    path = waypoint_graph.find_path(start_node, target_node)
    if not path or len(path) < 2:
        return None

    # Trim path: skip nodes we've already passed
    path = _trim_passed_nodes(waypoint_graph, path, start_x, start_y)

    legs: list[TravelLeg] = []

    # If start is far from first node, prepend A* leg to it
    _first = waypoint_graph.coords[path[0]]
    first_x, first_y = _first.x, _first.y
    if math.hypot(start_x - first_x, start_y - first_y) > 50:
        legs.append(TravelLeg(first_x, first_y, TravelMode.PATHFIND))

    # Generate a leg for each edge in the path
    for i in range(len(path) - 1):
        from_name = path[i]
        to_name = path[i + 1]
        _fp = waypoint_graph.coords[from_name]
        fx, fy = _fp.x, _fp.y
        _tp = waypoint_graph.coords[to_name]
        tx, ty = _tp.x, _tp.y

        manual_pts = _find_tunnel_between(tunnel_routes, from_name, to_name, fx, fy, tx, ty, threshold)
        if manual_pts:
            legs.append(TravelLeg(tx, ty, TravelMode.MANUAL, manual_pts))
            log.info("[TRAVEL] Travel plan: %s -> %s (tunnel, %d wps)", from_name, to_name, len(manual_pts))
        else:
            legs.append(TravelLeg(tx, ty, TravelMode.PATHFIND))
            log.info("[TRAVEL] Travel plan: %s -> %s (A*)", from_name, to_name)

    log.info("[TRAVEL] Travel plan: %d legs via waypoint graph (%s)", len(legs), " -> ".join(path))
    return legs


def _trim_passed_nodes(
    waypoint_graph: WaypointGraph,
    path: list[str],
    start_x: float,
    start_y: float,
) -> list[str]:
    """Trim leading nodes that the agent has already passed."""
    best_idx = 0
    _first = waypoint_graph.coords[path[0]]
    best_dist = math.hypot(start_x - _first.x, start_y - _first.y)
    for idx in range(1, len(path) - 1):  # never skip target
        _curr = waypoint_graph.coords[path[idx]]
        d = math.hypot(start_x - _curr.x, start_y - _curr.y)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    if best_idx > 0:
        skipped = path[:best_idx]
        path = path[best_idx:]
        log.info("[TRAVEL] Trimmed %d node(s) already passed: %s", len(skipped), " -> ".join(skipped))
    return path


def _plan_via_tunnel_routes(
    tunnel_routes: list[TunnelRoute],
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
    threshold: float,
) -> list[TravelLeg] | None:
    """Match target to a tunnel route endpoint. Returns legs or None."""
    for route in tunnel_routes:
        fx, fy = route.from_pos.x, route.from_pos.y
        tx, ty = route.to_pos.x, route.to_pos.y
        pts_rev = tuple(reversed(route.points))

        d_start_from = math.hypot(start_x - fx, start_y - fy)
        d_start_to = math.hypot(start_x - tx, start_y - ty)
        d_target_to = math.hypot(target_x - tx, target_y - ty)
        d_target_from = math.hypot(target_x - fx, target_y - fy)

        # Case 1: Forward direct
        if d_start_from < threshold and d_target_to < threshold:
            log.info(
                "[TRAVEL] Travel plan: tunnel '%s' FORWARD (%d waypoints)", route.name, len(route.points)
            )
            return [TravelLeg(tx, ty, TravelMode.MANUAL, route.points)]

        # Case 2: Reverse direct
        if d_start_to < threshold and d_target_from < threshold:
            log.info("[TRAVEL] Travel plan: tunnel '%s' REVERSED (%d waypoints)", route.name, len(pts_rev))
            return [TravelLeg(fx, fy, TravelMode.MANUAL, pts_rev)]

        # Case 3: Target near "to", start far from "from"
        if d_target_to < threshold:
            log.info("[TRAVEL] Travel plan: A* + tunnel '%s' FORWARD", route.name)
            return [
                TravelLeg(fx, fy, TravelMode.PATHFIND),
                TravelLeg(tx, ty, TravelMode.MANUAL, route.points),
            ]

        # Case 4: Target near "from", start far from "to"
        if d_target_from < threshold:
            log.info("[TRAVEL] Travel plan: A* + tunnel '%s' REVERSED", route.name)
            return [
                TravelLeg(tx, ty, TravelMode.PATHFIND),
                TravelLeg(fx, fy, TravelMode.MANUAL, pts_rev),
            ]

    return None


def plan_travel_legs(
    tunnel_routes: list[TunnelRoute],
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
    threshold: float = 500.0,
    waypoint_graph: WaypointGraph | None = None,
) -> list[TravelLeg]:
    """Plan travel legs using waypoint graph + tunnel routes.

    Planning priority:
      1. Waypoint graph: if both start and target match waypoints,
         BFS the graph and generate legs for each edge (A* or tunnel).
      2. Tunnel routes: if target matches a tunnel endpoint, chain
         A* to entrance + manual through tunnel.
      3. Fallback: single A* leg.

    Returns list of TravelLeg. Callers execute legs sequentially.
    """

    # -- Try waypoint graph first --
    if waypoint_graph and waypoint_graph.coords:
        graph_legs = _plan_via_waypoint_graph(
            waypoint_graph,
            tunnel_routes,
            start_x,
            start_y,
            target_x,
            target_y,
            threshold,
        )
        if graph_legs is not None:
            return graph_legs

    # -- Tunnel route matching (no waypoint graph or no graph match) --
    tunnel_legs = _plan_via_tunnel_routes(
        tunnel_routes,
        start_x,
        start_y,
        target_x,
        target_y,
        threshold,
    )
    if tunnel_legs is not None:
        return tunnel_legs

    # -- Fallback: single A* leg --
    return [TravelLeg(target_x, target_y, TravelMode.PATHFIND)]


def find_tunnel_route(
    tunnel_routes: list[TunnelRoute],
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
    threshold: float = 500.0,
) -> list[TravelWaypoint] | None:
    """Simple wrapper: returns manual waypoints if travel is a single tunnel leg.

    .. deprecated::
        Use plan_travel_legs() directly.
    """
    warnings.warn(
        "find_tunnel_route() is deprecated, use plan_travel_legs() directly",
        DeprecationWarning,
        stacklevel=2,
    )
    legs = plan_travel_legs(tunnel_routes, start_x, start_y, target_x, target_y, threshold)
    if len(legs) == 1 and legs[0].mode == TravelMode.MANUAL and legs[0].waypoints:
        return list(legs[0].waypoints)
    return None
