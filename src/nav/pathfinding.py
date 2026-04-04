"""A* pathfinding on the terrain heightmap grid.

Finds optimal paths through walkable terrain, avoiding water, lava,
cliffs, and steep slopes. Returns waypoints in game coordinates.

Paths are varied after computation: intermediate waypoints receive small
perpendicular offsets so no two traversals follow an identical line.
"""

from __future__ import annotations

import heapq
import logging
import math
import random
import time
from typing import TYPE_CHECKING

from core.types import Point

if TYPE_CHECKING:
    from nav.terrain.heightmap import ZoneTerrain

log = logging.getLogger(__name__)

# Cost multipliers for different terrain
_COST_NORMAL = 1.0
_COST_STEEP = 3.0  # traversable but penalized
_COST_DIAGONAL = 1.414  # sqrt(2) for diagonal movement
_COST_IMPASSABLE = 1e6  # blocks path (Z-delta too large)

# Hazard flags that block pathfinding
_BLOCKED = 0x04 | 0x08 | 0x20 | 0x40  # WATER | LAVA | CLIFF | OBSTACLE

# Maximum Z change between adjacent cells for traversal (EQ step height)
_WALKABLE_CLIMB = 15.0


def _validate_endpoints(
    terrain: ZoneTerrain,
    sc: int,
    sr: int,
    gc: int,
    gr: int,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    near_z: float,
) -> tuple[int, int, int, int] | None:
    """Validate start/goal are on-grid and snap blocked cells to nearest walkable.

    Returns (start_col, start_row, goal_col, goal_row) or None if unreachable.
    """
    if terrain._grid_idx(sc, sr) < 0:
        log.warning("[NAV] Pathfind: start (%.0f,%.0f) is off grid", start_x, start_y)
        return None
    if terrain._grid_idx(gc, gr) < 0:
        log.warning("[NAV] Pathfind: goal (%.0f,%.0f) is off grid", goal_x, goal_y)
        return None

    snap_radius = max(10, int(100.0 / terrain.cell_size))
    if not _cell_walkable(terrain, sc, sr, near_z):
        snapped = _snap_to_walkable(terrain, sc, sr, radius=snap_radius, near_z=near_z)
        if snapped:
            sc, sr = snapped
        else:
            log.warning("[NAV] Pathfind: start is blocked and no walkable cell nearby")
            return None

    if not _cell_walkable(terrain, gc, gr, near_z):
        snapped = _snap_to_walkable(terrain, gc, gr, radius=snap_radius, near_z=near_z)
        if snapped:
            gc, gr = snapped
        else:
            log.warning("[NAV] Pathfind: goal is blocked and no walkable cell nearby")
            return None

    return sc, sr, gc, gr


def _log_path_found(
    path: list[Point],
    terrain: ZoneTerrain,
    t0: float,
    explored: int,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
) -> None:
    """Log JPS path result with detour analysis for avoidance zones."""
    elapsed = time.perf_counter() - t0
    avoidance_active = bool(terrain._avoidance_zones or terrain._dynamic_avoidance)
    direct_dist = math.hypot(start_x - goal_x, start_y - goal_y)
    path_dist = (
        sum(
            math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)
        )
        if len(path) > 1
        else direct_dist
    )
    detour_pct = ((path_dist / max(direct_dist, 1.0)) - 1.0) * 100

    if avoidance_active and detour_pct > 50:
        log.warning(
            "[NAV] JPS: avoidance detour +%.0f%% (direct=%.0f, path=%.0f, %d zones)",
            detour_pct,
            direct_dist,
            path_dist,
            len(terrain._dynamic_avoidance),
        )
    elif avoidance_active and detour_pct > 10:
        log.info(
            "[NAV] JPS: path %d waypoints, %d nodes in %.2fs (detour +%.0f%%, %d avoidance zones)",
            len(path),
            explored,
            elapsed,
            detour_pct,
            len(terrain._dynamic_avoidance),
        )
    else:
        log.info("[NAV] JPS: path %d waypoints, %d nodes explored in %.2fs", len(path), explored, elapsed)


def find_path(
    terrain: ZoneTerrain,
    start: Point,
    goal: Point,
    max_nodes: int = 50000,
    jitter: float = 3.0,
) -> list[Point] | None:
    """Find a path using Jump Point Search (JPS) on the terrain grid.

    JPS prunes symmetric A* paths by jumping along straight/diagonal
    lines until hitting obstacles or forced neighbors. Same optimal
    grid path, 5-10x fewer node expansions than standard A*.

    Hybrid cost model: JPS determines reachability (walkable cells),
    then actual terrain costs (steep, Z-gradient, avoidance) are
    summed along each jump for correct edge weights.

    Args:
        terrain: ZoneTerrain instance with loaded heightmap.
        start: Start position as Point (x, y, z in game coordinates).
        goal: Goal position as Point (x, y, z in game coordinates).
        max_nodes: Maximum jump points to explore before giving up.
        jitter: Path humanization amount (0=straight, 12=natural).

    Returns:
        List of Point(x, y, z) waypoints from start to goal,
        or None if no path exists.
    """
    t0 = time.perf_counter()
    near_z = start.z

    # Scale jitter to cell_size so deviation is consistent (~1-2 cells)
    # regardless of grid resolution. 12u at 10u cells = 1.2 cells.
    jitter = jitter * min(terrain.cell_size, 10.0) / 10.0

    # Convert game coords to grid cells
    sc, sr = terrain._game_to_grid(start.x, start.y)
    gc, gr = terrain._game_to_grid(goal.x, goal.y)

    # Validate and snap endpoints to walkable cells
    endpoints = _validate_endpoints(
        terrain,
        sc,
        sr,
        gc,
        gr,
        start.x,
        start.y,
        goal.x,
        goal.y,
        near_z,
    )
    if endpoints is None:
        return None
    sc, sr, gc, gr = endpoints

    # A* search
    # Node: (col, row)
    # Priority queue: (f_cost, counter, col, row)
    open_set: list[tuple[float, int, int, int]] = []
    counter = 0
    start_node = (sc, sr)
    goal_node = (gc, gr)

    g_cost: dict[tuple[int, int], float] = {start_node: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    h = _heuristic(sc, sr, gc, gr)
    heapq.heappush(open_set, (h, counter, sc, sr))
    counter += 1

    cols = terrain._cols
    rows = terrain._rows
    explored = 0

    # Build a Z-filtered walkability bitfield so the search never routes
    # through cells on a different vertical level (bridges, multi-floor).
    if not math.isnan(near_z) and terrain._z_ceiling:
        wb, wbc = terrain.build_walk_bits_z(near_z)
    else:
        if not terrain._walk_bits:
            terrain._build_walk_bits()
        wb = terrain._walk_bits
        wbc = terrain._walk_byte_cols

    # All 8 directions for the start node (no parent to prune from)
    _ALL_DIRS = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    while open_set:
        if explored >= max_nodes:
            elapsed = time.perf_counter() - t0
            log.warning(
                "[NAV] JPS: hit node limit %d in %.2fs (start=(%d,%d) goal=(%d,%d))",
                max_nodes,
                elapsed,
                sc,
                sr,
                gc,
                gr,
            )
            return None

        _f, _, cc, cr = heapq.heappop(open_set)
        current = (cc, cr)

        if current in closed:
            continue
        closed.add(current)
        explored += 1

        if current == goal_node:
            path = _jps_reconstruct(came_from, current, terrain, near_z)
            path = _simplify_path(path, terrain, wb_override=wb, wbc_override=wbc)
            path = vary_path(
                path, terrain, jitter_range=jitter, near_z=near_z, wb_override=wb, wbc_override=wbc
            )
            _log_path_found(path, terrain, t0, explored, start.x, start.y, goal.x, goal.y)
            return path

        current_g = g_cost.get(current, float("inf"))

        # JPS: pruned directions based on how we arrived at this node
        parent = came_from.get(current)
        if parent is not None:
            directions = _jps_pruned_dirs(wb, wbc, cc, cr, parent[0], parent[1], cols, rows)
        else:
            directions = _ALL_DIRS

        for dc, dr in directions:
            if dc != 0 and dr != 0:
                jp = _jps_jump_diagonal(wb, wbc, cc, cr, dc, dr, gc, gr, cols, rows)
            else:
                jp = _jps_jump_straight(wb, wbc, cc, cr, dc, dr, gc, gr, cols, rows)

            if jp is None:
                continue
            nc, nr = jp
            if (nc, nr) in closed:
                continue

            step_cost = _jps_path_cost(terrain, cc, cr, nc, nr)
            tentative_g = current_g + step_cost

            if tentative_g < g_cost.get((nc, nr), float("inf")):
                g_cost[(nc, nr)] = tentative_g
                came_from[(nc, nr)] = current
                f_cost = tentative_g + _heuristic(nc, nr, gc, gr)
                heapq.heappush(open_set, (f_cost, counter, nc, nr))
                counter += 1

    # JPS failed -- fall back to standard A* with bitfield.
    # JPS can miss paths on open terrain with sparse obstacles
    # (cardinal jumps scan to walls without finding forced neighbors).
    log.info(
        "[NAV] JPS exhausted (%d nodes in %.2fs), falling back to A*", explored, time.perf_counter() - t0
    )
    return _find_path_astar(terrain, start.x, start.y, goal.x, goal.y, max_nodes, jitter, t0, wb, wbc, near_z)


def _heuristic(c1: int, r1: int, c2: int, r2: int) -> float:
    """Octile distance heuristic (consistent for 8-directional grid)."""
    dc = abs(c2 - c1)
    dr = abs(r2 - r1)
    return max(dc, dr) + (_COST_DIAGONAL - 1.0) * min(dc, dr)


def _cell_walkable(terrain: ZoneTerrain, col: int, row: int, near_z: float = float("nan")) -> bool:
    """Check if a grid cell is walkable with Z-level filtering.

    When near_z is provided (not NaN), cells on a different vertical
    level (bridges, multi-floor buildings) are rejected. Bridge cells
    with water/lava underneath are only walkable for agents at bridge
    height, not ground level.
    """
    idx = terrain._grid_idx(col, row)
    if idx < 0:
        return False
    wx = terrain._min_x + col * terrain.cell_size
    wy = terrain._min_y + row * terrain.cell_size
    game_x = wy  # WLD Y -> game X
    game_y = wx  # WLD X -> game Y
    if not terrain.is_walkable(game_x, game_y):
        return False
    if not math.isnan(near_z):
        # Multi-level check: reject bridge cells when agent is at ground
        # level and ground surface is hazardous (water/lava).
        from nav.terrain.heightmap import SURFACE_BRIDGE, SURFACE_LAVA, SURFACE_WATER

        f = terrain._flags[idx]
        z_ceil = terrain.get_z_ceiling(game_x, game_y)
        if not math.isnan(z_ceil):
            z_ground = terrain.get_z(game_x, game_y)
            mid = (z_ground + z_ceil) / 2.0
            on_upper = near_z > mid
            if not on_upper and (f & SURFACE_BRIDGE) and (f & (SURFACE_WATER | SURFACE_LAVA)):
                return False
        level_z = terrain.get_level_z(game_x, game_y, near_z)
        if abs(level_z - near_z) > _WALKABLE_CLIMB:
            return False
    return True


def _cell_cost(terrain: ZoneTerrain, col: int, row: int, from_col: int = -1, from_row: int = -1) -> float:
    """Return movement cost for a cell, or -1 if impassable.
    Uses terrain.is_walkable for full override + Z-threshold support.
    Penalizes steep Z changes between cells to prefer flat paths."""
    idx = terrain._grid_idx(col, row)
    if idx < 0:
        return -1

    # Convert grid -> game coords once, reuse for walkability + avoidance
    wx = terrain._min_x + col * terrain.cell_size
    wy = terrain._min_y + row * terrain.cell_size
    game_x, game_y = wy, wx  # WLD -> game coords (swapped)

    if not terrain.is_walkable(game_x, game_y):
        return -1

    f = terrain._flags[idx]
    cost = _COST_STEEP if (f & 0x02) else _COST_NORMAL

    # Z-gradient penalty: prefer flat terrain; reject impassable climbs
    if from_col >= 0:
        from_idx = terrain._grid_idx(from_col, from_row)
        if from_idx >= 0:
            dz = abs(terrain._z[idx] - terrain._z[from_idx])
            if dz > _WALKABLE_CLIMB:
                return -1  # impassable cliff
            if dz > 3.0:
                cost += dz * 0.3  # penalty scales with Z change

    # Avoidance zone penalty: cost near danger nodes
    if terrain._avoidance_zones or terrain._dynamic_avoidance:
        cost += terrain.avoidance_cost(game_x, game_y)

    return cost


def _snap_to_walkable(
    terrain: ZoneTerrain, col: int, row: int, radius: int = 5, near_z: float = float("nan")
) -> tuple[int, int] | None:
    """Find nearest walkable cell within radius, respecting Z-level."""
    best = None
    best_dist = float("inf")
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nc, nr = col + dc, row + dr
            if _cell_walkable(terrain, nc, nr, near_z):
                d = dc * dc + dr * dr
                if d < best_dist:
                    best_dist = d
                    best = (nc, nr)
    return best


# ======================================================================
# Standard A* fallback (bitfield-accelerated)
# ======================================================================


def _find_path_astar(
    terrain: ZoneTerrain,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    max_nodes: int,
    jitter: float,
    t0: float,
    wb: bytearray,
    wbc: int,
    near_z: float = float("nan"),
) -> list[Point] | None:
    """Standard A* with bitfield walkability. Fallback when JPS fails.

    JPS can miss paths on open terrain with sparse obstacles because
    cardinal jumps scan to walls without finding forced neighbors.
    Standard A* expands all 8 neighbors and always finds a path if one
    exists within the node budget.
    """
    sc, sr = terrain._game_to_grid(start_x, start_y)
    gc, gr = terrain._game_to_grid(goal_x, goal_y)
    cols = terrain._cols
    rows = terrain._rows

    snap_radius = max(10, int(100.0 / terrain.cell_size))
    if not _bit_walkable(wb, wbc, sc, sr, cols, rows):
        snapped = _snap_to_walkable(terrain, sc, sr, radius=snap_radius, near_z=near_z)
        if snapped:
            sc, sr = snapped
        else:
            return None
    if not _bit_walkable(wb, wbc, gc, gr, cols, rows):
        snapped = _snap_to_walkable(terrain, gc, gr, radius=snap_radius, near_z=near_z)
        if snapped:
            gc, gr = snapped
        else:
            return None

    open_set: list[tuple[float, int, int, int]] = []
    counter = 0
    g_cost: dict[tuple[int, int], float] = {(sc, sr): 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    h = _heuristic(sc, sr, gc, gr)
    heapq.heappush(open_set, (h, counter, sc, sr))
    counter += 1
    explored = 0

    neighbors = [
        (-1, 0, False),
        (1, 0, False),
        (0, -1, False),
        (0, 1, False),
        (-1, -1, True),
        (-1, 1, True),
        (1, -1, True),
        (1, 1, True),
    ]
    bw = _bit_walkable

    while open_set:
        if explored >= max_nodes:
            elapsed = time.perf_counter() - t0
            log.warning("[NAV] A*: hit node limit %d in %.2fs", max_nodes, elapsed)
            return None

        _f, _, cc, cr = heapq.heappop(open_set)
        current = (cc, cr)
        if current in closed:
            continue
        closed.add(current)
        explored += 1

        if cc == gc and cr == gr:
            path = _jps_reconstruct(came_from, current, terrain, near_z)
            path = _simplify_path(path, terrain, wb_override=wb, wbc_override=wbc)
            path = vary_path(
                path, terrain, jitter_range=jitter, near_z=near_z, wb_override=wb, wbc_override=wbc
            )
            elapsed = time.perf_counter() - t0
            log.info("[NAV] A* fallback: path %d waypoints, %d nodes in %.2fs", len(path), explored, elapsed)
            return path

        current_g = g_cost[current]

        for dc, dr, diag in neighbors:
            nc, nr = cc + dc, cr + dr
            if not bw(wb, wbc, nc, nr, cols, rows):
                continue
            if diag and (
                not bw(wb, wbc, cc + dc, cr, cols, rows) or not bw(wb, wbc, cc, cr + dr, cols, rows)
            ):
                continue

            cost = _fast_cell_cost(terrain, nc, nr, cc, cr)
            if cost < 0:
                continue  # impassable (cliff)
            step = cost * (_COST_DIAGONAL if diag else _COST_NORMAL)
            tentative_g = current_g + step

            if tentative_g < g_cost.get((nc, nr), float("inf")):
                g_cost[(nc, nr)] = tentative_g
                came_from[(nc, nr)] = current
                heapq.heappush(open_set, (tentative_g + _heuristic(nc, nr, gc, gr), counter, nc, nr))
                counter += 1

    elapsed = time.perf_counter() - t0
    log.warning("[NAV] A* fallback: no path in %.2fs (%d nodes)", elapsed, explored)
    return None


# ======================================================================
# Jump Point Search (JPS) -- bitfield-accelerated
# ======================================================================


def _sign(x: int) -> int:
    """Return -1, 0, or 1."""
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _bit_walkable(wb: bytearray, wbc: int, col: int, row: int, cols: int, rows: int) -> bool:
    """O(1) walkability check via precomputed bitfield.

    Single array lookup + bit mask. No coordinate conversion, no override
    iteration, no feature flag import. ~10x faster than _cell_walkable().
    """
    if col < 0 or col >= cols or row < 0 or row >= rows:
        return False
    return bool(wb[row * wbc + (col >> 3)] & (1 << (col & 7)))


def _fast_cell_cost(terrain: ZoneTerrain, col: int, row: int, from_col: int, from_row: int) -> float:
    """Cell cost for bitfield-validated cells (skip walkability check).

    The JPS jump already confirmed walkability via bitfield, so we skip
    the full is_walkable() chain and read flags directly.
    Returns -1 for impassable cliff transitions (dz > WALKABLE_CLIMB).
    """
    cols = terrain._cols
    idx = row * cols + col
    f = terrain._flags[idx]
    cost = _COST_STEEP if (f & 0x02) else _COST_NORMAL

    from_idx = from_row * cols + from_col
    dz = abs(terrain._z[idx] - terrain._z[from_idx])
    if dz > _WALKABLE_CLIMB:
        return -1  # impassable cliff
    if dz > 3.0:
        cost += dz * 0.3

    if terrain._avoidance_zones or terrain._dynamic_avoidance:
        wx = terrain._min_x + col * terrain.cell_size
        wy = terrain._min_y + row * terrain.cell_size
        cost += terrain.avoidance_cost(wy, wx)

    return cost


def _jps_jump_straight(
    wb: bytearray,
    wbc: int,
    col: int,
    row: int,
    dc: int,
    dr: int,
    gc: int,
    gr: int,
    cols: int,
    rows: int,
) -> tuple[int, int] | None:
    """Jump in a cardinal direction until obstacle, forced neighbor, or goal.

    Uses precomputed bitfield for O(1) walkability checks.
    Returns the jump point (col, row), or None if blocked / out of bounds.
    """
    bw = _bit_walkable  # local ref for speed
    max_dist = cols if dc != 0 else rows
    for _ in range(max_dist):
        col += dc
        row += dr

        if not bw(wb, wbc, col, row, cols, rows):
            return None

        if col == gc and row == gr:
            return (col, row)

        # Forced neighbor detection
        if dc != 0:
            if not bw(wb, wbc, col, row - 1, cols, rows) and bw(wb, wbc, col + dc, row - 1, cols, rows):
                return (col, row)
            if not bw(wb, wbc, col, row + 1, cols, rows) and bw(wb, wbc, col + dc, row + 1, cols, rows):
                return (col, row)
        else:
            if not bw(wb, wbc, col - 1, row, cols, rows) and bw(wb, wbc, col - 1, row + dr, cols, rows):
                return (col, row)
            if not bw(wb, wbc, col + 1, row, cols, rows) and bw(wb, wbc, col + 1, row + dr, cols, rows):
                return (col, row)

    return None


def _jps_jump_diagonal(
    wb: bytearray,
    wbc: int,
    col: int,
    row: int,
    dc: int,
    dr: int,
    gc: int,
    gr: int,
    cols: int,
    rows: int,
) -> tuple[int, int] | None:
    """Jump diagonally until obstacle or a cardinal sub-jump finds a point.

    Uses precomputed bitfield. Corner-cutting prevention: both cardinal
    neighbors of the diagonal step must be walkable.
    """
    bw = _bit_walkable
    max_dist = min(cols, rows)
    for _ in range(max_dist):
        col += dc
        row += dr

        if not bw(wb, wbc, col, row, cols, rows):
            return None

        # Corner-cutting prevention
        if not bw(wb, wbc, col - dc, row, cols, rows) or not bw(wb, wbc, col, row - dr, cols, rows):
            return None

        if col == gc and row == gr:
            return (col, row)

        # Cardinal sub-jumps
        if _jps_jump_straight(wb, wbc, col, row, dc, 0, gc, gr, cols, rows) is not None:
            return (col, row)
        if _jps_jump_straight(wb, wbc, col, row, 0, dr, gc, gr, cols, rows) is not None:
            return (col, row)

    return None


def _jps_pruned_dirs(
    wb: bytearray,
    wbc: int,
    col: int,
    row: int,
    pcol: int,
    prow: int,
    cols: int,
    rows: int,
) -> list[tuple[int, int]]:
    """Return pruned search directions from a JPS node given its parent.

    Uses precomputed bitfield for forced neighbor detection.
    Cardinal nodes: continue straight + diagonals toward forced neighbors.
    Diagonal nodes: continue diagonal + both cardinal components.
    """
    dc = _sign(col - pcol)
    dr = _sign(row - prow)
    bw = _bit_walkable

    dirs: list[tuple[int, int]] = []

    if dc != 0 and dr != 0:
        # Diagonal arrival: natural = diagonal + both cardinal components
        dirs.append((dc, dr))
        dirs.append((dc, 0))
        dirs.append((0, dr))
    elif dc != 0:
        # Horizontal arrival
        dirs.append((dc, 0))
        if not bw(wb, wbc, col, row - 1, cols, rows):
            dirs.append((dc, -1))
        if not bw(wb, wbc, col, row + 1, cols, rows):
            dirs.append((dc, 1))
    elif dr != 0:
        # Vertical arrival
        dirs.append((0, dr))
        if not bw(wb, wbc, col - 1, row, cols, rows):
            dirs.append((-1, dr))
        if not bw(wb, wbc, col + 1, row, cols, rows):
            dirs.append((1, dr))

    return dirs


def _jps_path_cost(
    terrain: ZoneTerrain,
    c1: int,
    r1: int,
    c2: int,
    r2: int,
) -> float:
    """Sum actual terrain costs along a straight/diagonal line between cells.

    Uses _fast_cell_cost (skips walkability check -- already validated by
    bitfield during the jump). Only computes steep/Z-gradient/avoidance.
    """
    dc = _sign(c2 - c1)
    dr = _sign(r2 - r1)
    diag = dc != 0 and dr != 0
    base = _COST_DIAGONAL if diag else _COST_NORMAL

    total = 0.0
    c, r = c1, r1
    while c != c2 or r != r2:
        nc, nr = c + dc, r + dr
        cell_cost = _fast_cell_cost(terrain, nc, nr, c, r)
        if cell_cost < 0:
            return float("inf")  # cliff along jump -- reject this path
        total += cell_cost * base
        c, r = nc, nr
    return total


def _jps_reconstruct(
    came_from: dict[tuple[int, int], tuple[int, int]],
    current: tuple[int, int],
    terrain: ZoneTerrain,
    near_z: float = float("nan"),
) -> list[Point]:
    """Reconstruct path from JPS came_from, filling cells between jump points."""
    jump_path: list[tuple[int, int]] = [current]
    while current in came_from:
        current = came_from[current]
        jump_path.append(current)
    jump_path.reverse()

    grid_path: list[tuple[int, int]] = []
    for i in range(len(jump_path) - 1):
        c1, r1 = jump_path[i]
        c2, r2 = jump_path[i + 1]
        dc = _sign(c2 - c1)
        dr = _sign(r2 - r1)
        c, r = c1, r1
        while c != c2 or r != r2:
            grid_path.append((c, r))
            c += dc
            r += dr
    grid_path.append(jump_path[-1])

    # Convert grid cells to game coordinates with level-aware Z
    waypoints: list[Point] = []
    for col, row in grid_path:
        wx = terrain._min_x + (col + 0.5) * terrain.cell_size
        wy = terrain._min_y + (row + 0.5) * terrain.cell_size
        gx, gy = wy, wx  # WLD -> game (swap)
        if not math.isnan(near_z):
            _z = terrain.get_level_z(gx, gy, near_z)
        else:
            _z = terrain.get_z(gx, gy)
        waypoints.append(Point(gx, gy, _z if _z == _z else 0.0))  # NaN guard
    return waypoints


def _reconstruct(
    came_from: dict,
    current: tuple[int, int],
    terrain: ZoneTerrain,
    near_z: float = float("nan"),
) -> list[Point]:
    """Reconstruct path from A* came_from map. Returns game coordinates with Z."""
    grid_path = [current]
    while current in came_from:
        current = came_from[current]
        grid_path.append(current)
    grid_path.reverse()

    waypoints: list[Point] = []
    for col, row in grid_path:
        wx = terrain._min_x + (col + 0.5) * terrain.cell_size
        wy = terrain._min_y + (row + 0.5) * terrain.cell_size
        gx, gy = wy, wx  # WLD -> game (swap back)
        if not math.isnan(near_z):
            _z = terrain.get_level_z(gx, gy, near_z)
        else:
            _z = terrain.get_z(gx, gy)
        waypoints.append(Point(gx, gy, _z if _z == _z else 0.0))
    return waypoints


def _simplify_path(
    path: list[Point],
    terrain: ZoneTerrain,
    wb_override: bytearray | None = None,
    wbc_override: int | None = None,
) -> list[Point]:
    """Two-phase path smoothing: greedy LOS + clearance-aware funnel.

    Phase 1 (greedy forward): scan forward from each anchor, skip
    collinear/visible waypoints. Fast -- handles bulk grid artifacts.

    Phase 2 (reverse funnel): for each anchor in the Phase 1 result,
    try the FARTHEST remaining point first (reverse scan). Prefer paths
    with clearance margin; fall back to center-only for narrow passages.
    Finds shortcuts that Phase 1 misses due to its early-break heuristic.

    When *wb_override*/*wbc_override* are provided, LOS checks use the
    Z-filtered bitfield so smoothing respects multi-level terrain.
    """
    if len(path) <= 2:
        return path

    # Phase 1: greedy forward LOS skip
    phase1: list[Point] = [path[0]]
    i = 0
    while i < len(path) - 1:
        best_j = i + 1
        for j in range(i + 2, len(path)):
            if _clear_line(
                terrain, path[i][0], path[i][1], path[j][0], path[j][1], wb_override, wbc_override
            ):
                best_j = j
            else:
                break
        phase1.append(path[best_j])
        i = best_j

    # Phase 2: reverse-scan funnel with clearance
    return _funnel_smooth(phase1, terrain, wb_override=wb_override, wbc_override=wbc_override)


def _funnel_smooth(
    path: list[Point],
    terrain: ZoneTerrain,
    clearance: float = 2.0,
    wb_override: bytearray | None = None,
    wbc_override: int | None = None,
) -> list[Point]:
    """Clearance-aware reverse-scan funnel smoothing.

    For each anchor, try the farthest remaining waypoint first. This
    finds the maximum shortcut in one check (vs Phase 1's forward scan
    that stops at the first LOS failure).

    Prefers paths with clearance margin (2u on each side) so the agent
    doesn't wall-hug. Falls back to center-only LOS for narrow
    passages (doorways, corridors < 5u wide).

    Operates on the already-reduced Phase 1 output (~10-30 points),
    so the O(n^2) reverse scan is fast.
    """
    if len(path) <= 2:
        return path

    result: list[Point] = [path[0]]
    i = 0

    while i < len(path) - 1:
        best_j = i + 1

        # Try farthest point first -- with clearance
        for j in range(len(path) - 1, i + 1, -1):
            if _clear_line_wide(
                terrain, path[i][0], path[i][1], path[j][0], path[j][1], clearance, wb_override, wbc_override
            ):
                best_j = j
                break

        # Fallback: center-only for narrow passages
        if best_j == i + 1 and i + 2 < len(path):
            for j in range(len(path) - 1, i + 1, -1):
                if _clear_line(
                    terrain, path[i][0], path[i][1], path[j][0], path[j][1], wb_override, wbc_override
                ):
                    best_j = j
                    break

        result.append(path[best_j])
        i = best_j

    return result


def vary_path(
    path: list[Point],
    terrain: ZoneTerrain,
    jitter_range: float = 3.0,
    near_z: float = float("nan"),
    wb_override: bytearray | None = None,
    wbc_override: int | None = None,
) -> list[Point]:
    """Add small perpendicular offsets to intermediate waypoints.

    Applies a uniform lateral drift (1-3u) to each interior point.
    Start/end waypoints are never modified. All offset positions are
    validated against terrain before use.

    When *near_z* and a Z-filtered bitfield are provided, jitter
    validation uses the bitfield (respecting multi-level terrain) and
    Z lookup uses ``get_level_z`` to stay on the correct floor.
    """
    if len(path) <= 2:
        return path

    # Pre-fetch grid constants for bitfield walkability checks
    _use_zbit = wb_override is not None and wbc_override is not None
    if _use_zbit:
        _cs = terrain.cell_size
        _mx = terrain._min_x
        _my = terrain._min_y
        _cols = terrain._cols
        _rows = terrain._rows
        _bw = _bit_walkable

    _has_near_z = not math.isnan(near_z)

    result: list[Point] = [path[0]]

    for i in range(1, len(path) - 1):
        px, py = path[i].x, path[i].y

        if jitter_range > 0 and i < len(path) - 1:
            dx = path[i + 1].x - path[i - 1].x
            dy = path[i + 1].y - path[i - 1].y
            seg_len = math.hypot(dx, dy)
            if seg_len > 1.0:
                perp_x = -dy / seg_len
                perp_y = dx / seg_len
                offset = random.uniform(-jitter_range, jitter_range)
                jx = px + perp_x * offset
                jy = py + perp_y * offset
                # Z-aware walkability: use the filtered bitfield when available
                if _use_zbit and wb_override is not None and wbc_override is not None:
                    col = int((jy - _mx) / _cs)
                    row = int((jx - _my) / _cs)
                    walkable = _bw(wb_override, wbc_override, col, row, _cols, _rows)
                else:
                    walkable = terrain.is_walkable(jx, jy)
                if walkable:
                    px, py = jx, jy

        # Z-aware height: use get_level_z to stay on the correct floor
        if _has_near_z:
            _z = terrain.get_level_z(px, py, near_z)
        else:
            _z = terrain.get_z(px, py)
        result.append(Point(px, py, _z if _z == _z else 0.0))

    result.append(path[-1])
    return result


def _clear_line(
    terrain: ZoneTerrain,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    wb_override: bytearray | None = None,
    wbc_override: int | None = None,
) -> bool:
    """Check if a straight line between two game-coord points is walkable.

    Uses precomputed bitfield when available (~10x faster than
    _cell_walkable per sample). Falls back to _cell_walkable otherwise.

    When *wb_override*/*wbc_override* are provided (e.g. the Z-filtered
    bitfield from ``build_walk_bits_z``), they take precedence over the
    cached Z-agnostic ``terrain._walk_bits``.
    """
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 0.01:
        return True

    cs = terrain.cell_size
    steps = max(1, int(dist / (cs * 0.5)))

    # Fast path: bitfield-accelerated (no coord conversion overhead)
    wb = wb_override if wb_override is not None else terrain._walk_bits
    if wb:
        wbc = wbc_override if wbc_override is not None else terrain._walk_byte_cols
        cols = terrain._cols
        rows = terrain._rows
        mx = terrain._min_x
        my = terrain._min_y
        bw = _bit_walkable
        for i in range(steps + 1):
            t = i / steps
            px, py = x1 + dx * t, y1 + dy * t
            # Inline game -> grid: game_y -> WLD_x -> col, game_x -> WLD_y -> row
            col = int((py - mx) / cs)
            row = int((px - my) / cs)
            if not bw(wb, wbc, col, row, cols, rows):
                return False
        return True

    # Slow fallback
    for i in range(steps + 1):
        t = i / steps
        px, py = x1 + dx * t, y1 + dy * t
        col, row = terrain._game_to_grid(px, py)
        if not _cell_walkable(terrain, col, row):
            return False
    return True


def _clear_line_wide(
    terrain: ZoneTerrain,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    clearance: float = 2.0,
    wb_override: bytearray | None = None,
    wbc_override: int | None = None,
) -> bool:
    """Check if a line with clearance margin is fully walkable.

    Tests center line plus two parallel lines offset by clearance.
    Ensures the agent has breathing room and won't wall-hug.
    """
    if not _clear_line(terrain, x1, y1, x2, y2, wb_override, wbc_override):
        return False

    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        return True

    # Perpendicular offset
    px = -dy / dist * clearance
    py = dx / dist * clearance

    if not _clear_line(terrain, x1 + px, y1 + py, x2 + px, y2 + py, wb_override, wbc_override):
        return False
    if not _clear_line(terrain, x1 - px, y1 - py, x2 - px, y2 - py, wb_override, wbc_override):
        return False
    return True
