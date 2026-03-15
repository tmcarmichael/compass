"""Comprehensive tests for nav.pathfinding -- A*/JPS pathfinding on terrain grids.

Covers all public and internal functions: heuristic, bitfield walkability,
coordinate snapping, cell cost, JPS internals (jump straight, jump diagonal,
pruned directions), path reconstruction, simplification, variation, LOS
checks, and the full find_path / _find_path_astar entry points.

All test data is synthetic -- no game assets needed. Uses _flat_grid from
test_heightmap to build ZoneTerrain objects, then blocks specific cells.
"""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from core.types import Point
from nav.pathfinding import (
    _bit_walkable,
    _cell_cost,
    _cell_walkable,
    _clear_line,
    _clear_line_wide,
    _fast_cell_cost,
    _find_path_astar,
    _heuristic,
    _jps_jump_diagonal,
    _jps_jump_straight,
    _jps_path_cost,
    _jps_pruned_dirs,
    _jps_reconstruct,
    _sign,
    _simplify_path,
    _snap_to_walkable,
    find_path,
    vary_path,
)
from nav.terrain.heightmap import (
    SURFACE_OBSTACLE,
    SURFACE_WALKABLE,
    SURFACE_WATER,
    ZoneTerrain,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _flat_grid(cols: int, rows: int, z: float = 0.0, cell_size: float = 1.0) -> ZoneTerrain:
    """Create a ZoneTerrain with a flat walkable grid (no meshes needed).

    Copied from tests/test_heightmap.py for self-containment.
    """
    from nav.terrain.heightmap import MAT_UNKNOWN

    t = ZoneTerrain(cell_size=cell_size)
    t._min_x = 0.0
    t._min_y = 0.0
    t._cols = cols
    t._rows = rows
    total = cols * rows
    t._z = [z] * total
    t._z_ceiling = [float("nan")] * total
    t._flags = [SURFACE_WALKABLE] * total
    t._normal_z = [0.9] * total
    t._material_id = [MAT_UNKNOWN] * total
    t._region_id = [0] * total
    return t


def _path_grid(cols: int, rows: int, blocked_cells: list[tuple[int, int]] | None = None) -> ZoneTerrain:
    """Create a walkable grid with specific cells blocked via OBSTACLE flag.

    Coordinates are (col, row) in grid space.
    """
    t = _flat_grid(cols, rows, z=0.0)
    t._build_walk_bits()
    if blocked_cells:
        for c, r in blocked_cells:
            idx = r * cols + c
            t._flags[idx] |= SURFACE_OBSTACLE
        t._build_walk_bits()  # rebuild after flagging
    return t


def _grid_to_game(terrain: ZoneTerrain, col: int, row: int) -> tuple[float, float]:
    """Convert grid (col, row) to game (game_x, game_y).

    Reverse of _game_to_grid:
      _game_to_wld: (game_x, game_y) -> (game_y, game_x) = (wld_x, wld_y)
      grid: col = int((wld_x - min_x) / cell_size)
            row = int((wld_y - min_y) / cell_size)

    So: wld_x = min_x + col * cell_size  (+ 0.5 for center)
        wld_y = min_y + row * cell_size  (+ 0.5 for center)
        game_x = wld_y, game_y = wld_x
    """
    cs = terrain.cell_size
    wld_x = terrain._min_x + (col + 0.5) * cs
    wld_y = terrain._min_y + (row + 0.5) * cs
    return (wld_y, wld_x)  # game_x = wld_y, game_y = wld_x


# ------------------------------------------------------------------
# _sign
# ------------------------------------------------------------------


class TestSign:
    def test_positive(self):
        assert _sign(5) == 1
        assert _sign(1) == 1
        assert _sign(1000) == 1

    def test_negative(self):
        assert _sign(-3) == -1
        assert _sign(-1) == -1
        assert _sign(-999) == -1

    def test_zero(self):
        assert _sign(0) == 0


# ------------------------------------------------------------------
# _heuristic -- octile distance
# ------------------------------------------------------------------


class TestHeuristic:
    def test_same_point(self):
        assert _heuristic(5, 5, 5, 5) == pytest.approx(0.0)

    def test_cardinal_horizontal(self):
        # Pure horizontal: dc=10, dr=0 -> max(10,0) + (1.414-1)*0 = 10
        assert _heuristic(0, 0, 10, 0) == pytest.approx(10.0)

    def test_cardinal_vertical(self):
        # Pure vertical: dc=0, dr=7 -> max(0,7) + (1.414-1)*0 = 7
        assert _heuristic(3, 0, 3, 7) == pytest.approx(7.0)

    def test_diagonal(self):
        # dc=5, dr=5 -> max(5,5) + (1.414-1)*5 = 5 + 2.07 = 7.07
        assert _heuristic(0, 0, 5, 5) == pytest.approx(5 + 0.414 * 5)

    def test_asymmetric(self):
        # dc=3, dr=7 -> max(3,7) + 0.414*min(3,7) = 7 + 1.242 = 8.242
        assert _heuristic(0, 0, 3, 7) == pytest.approx(7 + 0.414 * 3)

    def test_symmetric(self):
        # Heuristic should be symmetric
        assert _heuristic(2, 3, 8, 5) == pytest.approx(_heuristic(8, 5, 2, 3))


# ------------------------------------------------------------------
# _bit_walkable -- bitfield lookup
# ------------------------------------------------------------------


class TestBitWalkable:
    def test_walkable_cell(self):
        t = _path_grid(10, 10)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        assert _bit_walkable(wb, wbc, 5, 5, 10, 10) is True

    def test_blocked_cell(self):
        t = _path_grid(10, 10, blocked_cells=[(3, 4)])
        wb, wbc = t._walk_bits, t._walk_byte_cols
        assert _bit_walkable(wb, wbc, 3, 4, 10, 10) is False

    def test_out_of_bounds_negative_col(self):
        t = _path_grid(10, 10)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        assert _bit_walkable(wb, wbc, -1, 0, 10, 10) is False

    def test_out_of_bounds_negative_row(self):
        t = _path_grid(10, 10)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        assert _bit_walkable(wb, wbc, 0, -1, 10, 10) is False

    def test_out_of_bounds_col_too_large(self):
        t = _path_grid(10, 10)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        assert _bit_walkable(wb, wbc, 10, 0, 10, 10) is False

    def test_out_of_bounds_row_too_large(self):
        t = _path_grid(10, 10)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        assert _bit_walkable(wb, wbc, 0, 10, 10, 10) is False

    def test_corner_cells_walkable(self):
        t = _path_grid(10, 10)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        assert _bit_walkable(wb, wbc, 0, 0, 10, 10) is True
        assert _bit_walkable(wb, wbc, 9, 0, 10, 10) is True
        assert _bit_walkable(wb, wbc, 0, 9, 10, 10) is True
        assert _bit_walkable(wb, wbc, 9, 9, 10, 10) is True

    def test_water_cell_blocked(self):
        t = _flat_grid(10, 10)
        idx = 3 * 10 + 5  # row=3, col=5
        t._flags[idx] |= SURFACE_WATER
        t._build_walk_bits()
        wb, wbc = t._walk_bits, t._walk_byte_cols
        assert _bit_walkable(wb, wbc, 5, 3, 10, 10) is False

    def test_all_cells_blocked(self):
        """Every cell blocked -> no cell is walkable."""
        blocked = [(c, r) for r in range(5) for c in range(5)]
        t = _path_grid(5, 5, blocked_cells=blocked)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        for r in range(5):
            for c in range(5):
                assert _bit_walkable(wb, wbc, c, r, 5, 5) is False


# ------------------------------------------------------------------
# _cell_walkable and _cell_cost
# ------------------------------------------------------------------


class TestCellWalkable:
    def test_walkable_cell(self):
        t = _flat_grid(10, 10)
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            assert _cell_walkable(t, 5, 5) is True

    def test_blocked_cell(self):
        t = _flat_grid(10, 10)
        t._flags[5 * 10 + 3] |= SURFACE_OBSTACLE
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            assert _cell_walkable(t, 3, 5) is False

    def test_out_of_bounds(self):
        t = _flat_grid(10, 10)
        assert _cell_walkable(t, -1, 0) is False
        assert _cell_walkable(t, 10, 0) is False


class TestCellCost:
    def test_normal_cost(self):
        t = _flat_grid(10, 10)
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            cost = _cell_cost(t, 5, 5)
            assert cost == pytest.approx(1.0)

    def test_steep_cost(self):
        t = _flat_grid(10, 10)
        from nav.terrain.heightmap import SURFACE_STEEP

        t._flags[5 * 10 + 5] |= SURFACE_STEEP
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            cost = _cell_cost(t, 5, 5)
            assert cost == pytest.approx(3.0)

    def test_blocked_returns_negative(self):
        t = _flat_grid(10, 10)
        t._flags[5 * 10 + 3] |= SURFACE_OBSTACLE
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            cost = _cell_cost(t, 3, 5)
            assert cost == -1

    def test_out_of_bounds_returns_negative(self):
        t = _flat_grid(10, 10)
        cost = _cell_cost(t, -1, 0)
        assert cost == -1

    def test_z_gradient_penalty(self):
        """Large Z difference adds extra cost."""
        t = _flat_grid(10, 10, z=0.0)
        t._z[5 * 10 + 5] = 10.0  # cell (5,5) at Z=10
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            cost_with_gradient = _cell_cost(t, 5, 5, from_col=4, from_row=5)
            cost_without = _cell_cost(t, 4, 5)
            assert cost_with_gradient > cost_without


# ------------------------------------------------------------------
# _fast_cell_cost
# ------------------------------------------------------------------


class TestFastCellCost:
    def test_normal_cell(self):
        t = _path_grid(10, 10)
        cost = _fast_cell_cost(t, 5, 5, 4, 5)
        assert cost == pytest.approx(1.0)

    def test_z_gradient_penalty(self):
        t = _path_grid(10, 10)
        t._z[5 * 10 + 5] = 10.0  # large Z change from neighbor
        cost = _fast_cell_cost(t, 5, 5, 4, 5)
        # dz=10 > 3.0 -> cost += 10 * 0.3 = 3.0
        assert cost == pytest.approx(1.0 + 10.0 * 0.3)

    def test_steep_cell(self):
        t = _path_grid(10, 10)
        from nav.terrain.heightmap import SURFACE_STEEP

        t._flags[5 * 10 + 5] |= SURFACE_STEEP
        cost = _fast_cell_cost(t, 5, 5, 4, 5)
        assert cost == pytest.approx(3.0)


# ------------------------------------------------------------------
# _snap_to_walkable
# ------------------------------------------------------------------


class TestSnapToWalkable:
    def test_snap_from_blocked_to_adjacent(self):
        """Blocked cell at (5,5) should snap to one of the walkable neighbors."""
        t = _path_grid(10, 10, blocked_cells=[(5, 5)])
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            result = _snap_to_walkable(t, 5, 5, radius=2)
            assert result is not None
            sc, sr = result
            # Should be adjacent (distance <= 2)
            assert abs(sc - 5) <= 2
            assert abs(sr - 5) <= 2
            # The snapped cell itself must be walkable
            assert _cell_walkable(t, sc, sr) is True

    def test_snap_picks_closest(self):
        """When multiple walkable cells exist, snap picks the closest."""
        # Block everything except (7, 5)
        blocked = [(c, r) for r in range(10) for c in range(10) if not (c == 7 and r == 5)]
        t = _path_grid(10, 10, blocked_cells=blocked)
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            result = _snap_to_walkable(t, 5, 5, radius=5)
            assert result == (7, 5)

    def test_snap_returns_none_when_all_blocked(self):
        """If no walkable cells in radius, return None."""
        blocked = [(c, r) for r in range(10) for c in range(10)]
        t = _path_grid(10, 10, blocked_cells=blocked)
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            result = _snap_to_walkable(t, 5, 5, radius=3)
            assert result is None

    def test_snap_already_walkable(self):
        """If the cell itself is walkable, snap returns it (distance=0)."""
        t = _path_grid(10, 10)
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            result = _snap_to_walkable(t, 5, 5, radius=3)
            assert result == (5, 5)


# ------------------------------------------------------------------
# _jps_jump_straight -- JPS cardinal jump
# ------------------------------------------------------------------


class TestJpsJumpStraight:
    def test_jump_to_goal(self):
        """Straight jump east should find the goal."""
        t = _path_grid(20, 10)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        # Start at (2, 5), goal at (10, 5), direction east (dc=1, dr=0)
        jp = _jps_jump_straight(wb, wbc, 2, 5, 1, 0, 10, 5, 20, 10)
        assert jp == (10, 5)

    def test_jump_blocked(self):
        """Jump into a wall returns None."""
        # Wall at col=5 across all rows
        blocked = [(5, r) for r in range(10)]
        t = _path_grid(20, 10, blocked_cells=blocked)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        # Start at (2, 5), jump east -- hits wall at col=5
        jp = _jps_jump_straight(wb, wbc, 2, 5, 1, 0, 15, 5, 20, 10)
        assert jp is None

    def test_jump_forced_neighbor(self):
        """Obstacle adjacent to scan line creates a forced neighbor."""
        # Block cell at (6, 4) -- one above the scan row 5
        blocked = [(6, 4)]
        t = _path_grid(20, 10, blocked_cells=blocked)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        # Jump east from (2, 5), goal far away at (18, 5)
        # At col=6, row-1=4 is blocked, and col+dc=7, row-1=4 is walkable -> forced neighbor
        jp = _jps_jump_straight(wb, wbc, 2, 5, 1, 0, 18, 5, 20, 10)
        assert jp is not None
        assert jp[0] == 6  # jump point at col=6

    def test_jump_south_to_goal(self):
        """Jump south (dr=1) to goal."""
        t = _path_grid(10, 20)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        jp = _jps_jump_straight(wb, wbc, 3, 2, 0, 1, 3, 15, 10, 20)
        assert jp == (3, 15)

    def test_jump_to_edge(self):
        """Jump west from near left edge hits boundary -> None."""
        t = _path_grid(10, 10)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        # Start at (2, 5), jump west (dc=-1), goal far right
        jp = _jps_jump_straight(wb, wbc, 2, 5, -1, 0, 9, 5, 10, 10)
        # Reaches col=0 eventually, then col=-1 -> out of bounds -> None (unless forced neighbor)
        # With an open grid no forced neighbors exist, so it scans to the edge
        assert jp is None


# ------------------------------------------------------------------
# _jps_jump_diagonal -- JPS diagonal jump
# ------------------------------------------------------------------


class TestJpsJumpDiagonal:
    def test_jump_diagonal_to_goal(self):
        """Diagonal jump should find goal on the diagonal."""
        t = _path_grid(20, 20)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        # Start at (2, 2), goal at (8, 8), direction NE (dc=1, dr=1)
        jp = _jps_jump_diagonal(wb, wbc, 2, 2, 1, 1, 8, 8, 20, 20)
        assert jp == (8, 8)

    def test_jump_diagonal_blocked(self):
        """Diagonal jump that hits a blocked cell on the diagonal returns None."""
        # Block cell (3,3) directly on the diagonal path from (2,2) direction (1,1).
        # Also block (4,2) and (2,4) to prevent corner cutting detection,
        # and block neighbors to avoid forced-neighbor sub-jump hits.
        blocked = [(3, 3), (4, 2), (2, 4)]
        t = _path_grid(20, 20, blocked_cells=blocked)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        jp = _jps_jump_diagonal(wb, wbc, 2, 2, 1, 1, 15, 15, 20, 20)
        # Step 1: (3,3) is blocked -> returns None immediately
        assert jp is None

    def test_jump_diagonal_corner_cutting_blocked(self):
        """Diagonal blocked by corner-cutting prevention."""
        # Block (3, 2) -- the cardinal neighbor col-dc=3, row=2 when at (4, 3)
        # Actually: corner cutting checks col-dc and row-dr neighbors
        # If we jump (1,1) from (2,2), next cell (3,3), check (2,3) and (3,2)
        blocked = [(2, 3)]  # blocks corner cutting at step (3,3)
        t = _path_grid(20, 20, blocked_cells=blocked)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        jp = _jps_jump_diagonal(wb, wbc, 2, 2, 1, 1, 15, 15, 20, 20)
        assert jp is None

    def test_jump_diagonal_finds_cardinal_sub_jump(self):
        """Diagonal jump stops when a cardinal sub-jump finds a jump point."""
        # Create a forced neighbor for the horizontal sub-jump:
        # Block (7, 5) so that at diagonal position (6, 6), horizontal sub-jump
        # from (6,6) east finds a forced neighbor at col=7 (blocked row-1=5)
        # Actually: _jps_jump_straight checks forced neighbors, so we need
        # a pattern where the cardinal scan from a diagonal position finds one.
        # Block (8, 5) so that scanning east from (6,6), at col=8 row-1=5 is blocked
        # and col+1=9 row-1=5 is walkable -> forced neighbor at (8,6)
        blocked = [(8, 5)]
        t = _path_grid(20, 20, blocked_cells=blocked)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        # Diagonal NE from (2, 2), goal at (18, 18)
        jp = _jps_jump_diagonal(wb, wbc, 2, 2, 1, 1, 18, 18, 20, 20)
        assert jp is not None
        # Should stop at the diagonal position where the cardinal sub-jump found something
        # At diagonal step col=6+, the horizontal sub-jump from col=6 should find (8,6)
        # Actually at diagonal step 4: (6,6), then horizontal sub-jump east finds forced at 8
        assert jp[0] <= 8


# ------------------------------------------------------------------
# _jps_pruned_dirs -- JPS direction pruning
# ------------------------------------------------------------------


class TestJpsPrunedDirs:
    def test_horizontal_arrival_straight(self):
        """Horizontal arrival: continue east, no forced neighbors."""
        t = _path_grid(10, 10)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        # Arrived from (3,5) to (4,5) -> dc=1, dr=0 -> horizontal
        dirs = _jps_pruned_dirs(wb, wbc, 4, 5, 3, 5, 10, 10)
        assert (1, 0) in dirs
        # No forced neighbors on open grid
        assert len(dirs) == 1

    def test_horizontal_arrival_forced_neighbor(self):
        """Horizontal arrival with obstacle creates forced diagonal."""
        # Block (4, 4) -- above current cell (4, 5)
        blocked = [(4, 4)]
        t = _path_grid(10, 10, blocked_cells=blocked)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        dirs = _jps_pruned_dirs(wb, wbc, 4, 5, 3, 5, 10, 10)
        assert (1, 0) in dirs  # natural direction
        assert (1, -1) in dirs  # forced diagonal toward blocked row-1

    def test_vertical_arrival_straight(self):
        """Vertical arrival: continue south, no forced neighbors."""
        t = _path_grid(10, 10)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        dirs = _jps_pruned_dirs(wb, wbc, 5, 4, 5, 3, 10, 10)
        assert (0, 1) in dirs
        assert len(dirs) == 1

    def test_vertical_arrival_forced_neighbor(self):
        """Vertical arrival with obstacle creates forced diagonal."""
        # Block (4, 4) -- left of current (5, 4)
        blocked = [(4, 4)]
        t = _path_grid(10, 10, blocked_cells=blocked)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        dirs = _jps_pruned_dirs(wb, wbc, 5, 4, 5, 3, 10, 10)
        assert (0, 1) in dirs
        assert (-1, 1) in dirs  # forced: left blocked, so diagonal left-down

    def test_diagonal_arrival(self):
        """Diagonal arrival: continue diagonal + both cardinal components."""
        t = _path_grid(10, 10)
        wb, wbc = t._walk_bits, t._walk_byte_cols
        # Arrived from (3,3) to (4,4) -> dc=1, dr=1 -> diagonal
        dirs = _jps_pruned_dirs(wb, wbc, 4, 4, 3, 3, 10, 10)
        assert (1, 1) in dirs  # continue diagonal
        assert (1, 0) in dirs  # horizontal component
        assert (0, 1) in dirs  # vertical component
        assert len(dirs) == 3


# ------------------------------------------------------------------
# _jps_path_cost
# ------------------------------------------------------------------


class TestJpsPathCost:
    def test_horizontal_cost(self):
        """Cost along a horizontal line of normal cells."""
        t = _path_grid(20, 10)
        cost = _jps_path_cost(t, 2, 5, 7, 5)
        # 5 steps east, each costs 1.0 * 1.0 (normal, cardinal base)
        assert cost == pytest.approx(5.0)

    def test_diagonal_cost(self):
        """Cost along a diagonal line of normal cells."""
        t = _path_grid(20, 20)
        cost = _jps_path_cost(t, 2, 2, 5, 5)
        # 3 diagonal steps, each costs 1.0 * sqrt(2)
        assert cost == pytest.approx(3 * 1.414, abs=0.01)


# ------------------------------------------------------------------
# _jps_reconstruct
# ------------------------------------------------------------------


class TestJpsReconstruct:
    def test_single_node(self):
        """Single node -> single waypoint."""
        t = _path_grid(10, 10)
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        path = _jps_reconstruct(came_from, (5, 5), t)
        assert len(path) == 1

    def test_straight_path(self):
        """Reconstruct a straight horizontal path between two jump points."""
        t = _path_grid(20, 10)
        came_from = {(10, 5): (2, 5)}
        path = _jps_reconstruct(came_from, (10, 5), t)
        # From (2,5) to (10,5) = 8 steps + endpoint = 9 cells
        assert len(path) == 9

    def test_diagonal_path(self):
        """Reconstruct a diagonal path."""
        t = _path_grid(20, 20)
        came_from = {(5, 5): (2, 2)}
        path = _jps_reconstruct(came_from, (5, 5), t)
        # From (2,2) to (5,5) = 3 steps + endpoint = 4 cells
        assert len(path) == 4


# ------------------------------------------------------------------
# _simplify_path -- greedy LOS + funnel smoothing
# ------------------------------------------------------------------


class TestSimplifyPath:
    def test_short_path_unchanged(self):
        """Path with <= 2 points returns unchanged."""
        t = _path_grid(10, 10)
        p1 = [(1.0, 1.0)]
        assert _simplify_path(p1, t) == p1
        p2 = [(1.0, 1.0), (5.0, 5.0)]
        assert _simplify_path(p2, t) == p2

    def test_collinear_points_simplified(self):
        """Collinear points on an open grid should simplify to start+end."""
        t = _path_grid(20, 20)
        # Straight line in game coords -- all on walkable terrain
        # Game coords where game_x maps to row, game_y to col
        path = [(0.5 + i, 0.5) for i in range(15)]
        result = _simplify_path(path, t)
        # Should collapse to just start and end (clear LOS)
        assert len(result) <= 3
        assert result[0] == path[0]
        assert result[-1] == path[-1]

    def test_path_around_wall_preserved(self):
        """Path that detours around a wall keeps the turning waypoints."""
        # Block a column at col=5 for rows 0-8, leaving row=9 open
        blocked = [(5, r) for r in range(9)]
        t = _path_grid(20, 20, blocked_cells=blocked)
        # Path goes: (row=5 col=2) -> (row=9 col=2) -> (row=9 col=8) -> (row=5 col=8)
        gx0, gy0 = _grid_to_game(t, 2, 5)
        gx1, gy1 = _grid_to_game(t, 2, 9)
        gx2, gy2 = _grid_to_game(t, 8, 9)
        gx3, gy3 = _grid_to_game(t, 8, 5)
        path = [
            (gx0, gy0),
            (gx1, gy1),
            (gx2, gy2),
            (gx3, gy3),
        ]
        result = _simplify_path(path, t)
        # Endpoints preserved
        assert result[0] == path[0]
        assert result[-1] == path[-1]
        # At least some waypoints kept (can't shortcut through wall)
        assert len(result) >= 2


# ------------------------------------------------------------------
# vary_path -- jitter
# ------------------------------------------------------------------


class TestVaryPath:
    def test_short_path_unchanged(self):
        """Paths with 2 or fewer points are not varied."""
        t = _path_grid(10, 10)
        p = [Point(1.0, 1.0, 0.0), Point(5.0, 5.0, 0.0)]
        assert vary_path(p, t, jitter_range=3.0) == p

    def test_start_end_unchanged(self):
        """Start and end points are never modified."""
        t = _path_grid(20, 20)
        path = [Point(0.5, 0.5, 0.0), Point(5.5, 5.5, 0.0), Point(10.5, 10.5, 0.0), Point(15.5, 15.5, 0.0)]
        random.seed(42)
        result = vary_path(path, t, jitter_range=3.0)
        assert result[0] == path[0]
        assert result[-1] == path[-1]

    def test_interior_points_may_shift(self):
        """Interior waypoints may be offset (with sufficient jitter)."""
        t = _path_grid(30, 30)
        path = [Point(0.5, 0.5, 0.0), Point(10.5, 10.5, 0.0), Point(20.5, 20.5, 0.0), Point(29.5, 29.5, 0.0)]
        random.seed(123)
        result = vary_path(path, t, jitter_range=5.0)
        # At least one interior point should differ
        assert any(result[i] != path[i] for i in range(1, len(path) - 1))
        assert len(result) == len(path)

    def test_zero_jitter_preserves_path(self):
        """Zero jitter -> no modification."""
        t = _path_grid(20, 20)
        path = [Point(0.5, 0.5, 0.0), Point(5.5, 5.5, 0.0), Point(10.5, 10.5, 0.0)]
        result = vary_path(path, t, jitter_range=0.0)
        assert result == path


# ------------------------------------------------------------------
# _clear_line -- LOS check
# ------------------------------------------------------------------


class TestClearLine:
    def test_clear_on_open_grid(self):
        """Straight line across open terrain is clear."""
        t = _path_grid(20, 20)
        gx0, gy0 = _grid_to_game(t, 2, 2)
        gx1, gy1 = _grid_to_game(t, 15, 15)
        assert _clear_line(t, gx0, gy0, gx1, gy1) is True

    def test_blocked_by_obstacle(self):
        """Line through a blocked cell is not clear."""
        blocked = [(8, r) for r in range(20)]  # wall at col=8
        t = _path_grid(20, 20, blocked_cells=blocked)
        gx0, gy0 = _grid_to_game(t, 2, 5)
        gx1, gy1 = _grid_to_game(t, 15, 5)
        assert _clear_line(t, gx0, gy0, gx1, gy1) is False

    def test_same_point(self):
        """Same start and end is always clear."""
        t = _path_grid(10, 10)
        assert _clear_line(t, 5.0, 5.0, 5.0, 5.0) is True

    def test_uses_bitfield_fast_path(self):
        """With walk_bits present, the fast path is used."""
        t = _path_grid(20, 20)
        assert t._walk_bits  # bitfield built
        gx0, gy0 = _grid_to_game(t, 1, 1)
        gx1, gy1 = _grid_to_game(t, 10, 10)
        assert _clear_line(t, gx0, gy0, gx1, gy1) is True

    def test_fallback_without_bitfield(self):
        """Without walk_bits, falls back to _cell_walkable."""
        t = _path_grid(20, 20)
        t._walk_bits = bytearray()  # clear bitfield
        gx0, gy0 = _grid_to_game(t, 1, 1)
        gx1, gy1 = _grid_to_game(t, 10, 10)
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            assert _clear_line(t, gx0, gy0, gx1, gy1) is True


# ------------------------------------------------------------------
# _clear_line_wide -- LOS with clearance margin
# ------------------------------------------------------------------


class TestClearLineWide:
    def test_clear_on_open_grid(self):
        """Wide LOS check on open grid passes."""
        t = _path_grid(30, 30)
        gx0, gy0 = _grid_to_game(t, 5, 5)
        gx1, gy1 = _grid_to_game(t, 25, 25)
        assert _clear_line_wide(t, gx0, gy0, gx1, gy1, clearance=2.0) is True

    def test_blocked_center(self):
        """If center line is blocked, wide check fails."""
        blocked = [(15, r) for r in range(30)]
        t = _path_grid(30, 30, blocked_cells=blocked)
        gx0, gy0 = _grid_to_game(t, 5, 5)
        gx1, gy1 = _grid_to_game(t, 25, 5)
        assert _clear_line_wide(t, gx0, gy0, gx1, gy1, clearance=2.0) is False

    def test_blocked_parallel_only(self):
        """If center is clear but a parallel offset hits a wall, wide check fails."""
        # _grid_to_game(t, col, row) -> game_x = row+0.5, game_y = col+0.5
        # So (col=2,row=10) -> game(10.5, 2.5), (col=25,row=10) -> game(10.5, 25.5)
        # This is a vertical line in game space at game_x=10.5.
        # Perpendicular is along game_x. clearance=3 offsets game_x by +/-3.
        # game_x=10.5+3=13.5 -> row=13, game_x=10.5-3=7.5 -> row=7.
        # Block those rows to make the parallel offsets fail.
        blocked_above = [(c, 7) for c in range(30)]
        blocked_below = [(c, 13) for c in range(30)]
        t = _path_grid(30, 30, blocked_cells=blocked_above + blocked_below)
        gx0, gy0 = _grid_to_game(t, 2, 10)
        gx1, gy1 = _grid_to_game(t, 25, 10)
        # Center line at row=10 is clear
        assert _clear_line(t, gx0, gy0, gx1, gy1) is True
        # Parallel offset by 3 hits blocked rows 7 and 13
        assert _clear_line_wide(t, gx0, gy0, gx1, gy1, clearance=3.0) is False

    def test_very_short_line(self):
        """Very short line (< 1.0) returns True if center is clear."""
        t = _path_grid(10, 10)
        gx0, gy0 = _grid_to_game(t, 5, 5)
        gx1 = gx0 + 0.1
        gy1 = gy0 + 0.1
        assert _clear_line_wide(t, gx0, gy0, gx1, gy1, clearance=2.0) is True


# ------------------------------------------------------------------
# find_path -- full JPS entry point
# ------------------------------------------------------------------


class TestFindPath:
    """Integration tests for find_path using synthetic grids."""

    def test_straight_line_open_terrain(self):
        """Straight path on open terrain should return a path."""
        t = _path_grid(30, 30)
        gx0, gy0 = _grid_to_game(t, 2, 2)
        gx1, gy1 = _grid_to_game(t, 25, 25)
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            path = find_path(t, gx0, gy0, gx1, gy1, jitter=0.0)
        assert path is not None
        assert len(path) >= 2
        # Start and end should be near the requested positions
        assert abs(path[0][0] - gx0) < 2.0
        assert abs(path[0][1] - gy0) < 2.0
        assert abs(path[-1][0] - gx1) < 2.0
        assert abs(path[-1][1] - gy1) < 2.0

    def test_path_around_wall(self):
        """Path should route around a wall of blocked cells."""
        # Wall at col=15 from row=5 to row=25
        blocked = [(15, r) for r in range(5, 26)]
        t = _path_grid(30, 30, blocked_cells=blocked)
        gx0, gy0 = _grid_to_game(t, 10, 15)
        gx1, gy1 = _grid_to_game(t, 20, 15)
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            path = find_path(t, gx0, gy0, gx1, gy1, jitter=0.0)
        assert path is not None
        assert len(path) >= 2

    def test_path_through_corridor(self):
        """Path through a narrow gap in a wall."""
        # Wall at col=15 rows 0-12 and 14-29, gap at row=13
        blocked = [(15, r) for r in range(13)] + [(15, r) for r in range(14, 30)]
        t = _path_grid(30, 30, blocked_cells=blocked)
        gx0, gy0 = _grid_to_game(t, 10, 13)
        gx1, gy1 = _grid_to_game(t, 20, 13)
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            path = find_path(t, gx0, gy0, gx1, gy1, jitter=0.0)
        assert path is not None

    def test_start_off_grid_returns_none(self):
        """Start position off the grid returns None."""
        t = _path_grid(10, 10)
        # Game coords that map to a cell way off grid
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            path = find_path(t, -100.0, -100.0, 5.0, 5.0)
        assert path is None

    def test_goal_off_grid_returns_none(self):
        """Goal position off the grid returns None."""
        t = _path_grid(10, 10)
        gx0, gy0 = _grid_to_game(t, 5, 5)
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            path = find_path(t, gx0, gy0, -100.0, -100.0)
        assert path is None

    def test_start_blocked_snaps_to_walkable(self):
        """Blocked start snaps to nearby walkable cell and finds a path."""
        blocked = [(5, 5)]
        t = _path_grid(30, 30, blocked_cells=blocked)
        gx0, gy0 = _grid_to_game(t, 5, 5)
        gx1, gy1 = _grid_to_game(t, 20, 20)
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            path = find_path(t, gx0, gy0, gx1, gy1, jitter=0.0)
        assert path is not None
        assert len(path) >= 2

    def test_no_path_walled_off(self):
        """Completely walled-off goal returns None when unreachable."""
        # Create an impassable wall across the entire grid at col=15 (all rows).
        # Start on the left, goal on the right. The goal can snap to a walkable
        # cell on the right side, but no path exists through the wall.
        blocked_wall = [(15, r) for r in range(30)]
        t = _path_grid(30, 30, blocked_cells=blocked_wall)
        gx0, gy0 = _grid_to_game(t, 5, 15)
        gx1, gy1 = _grid_to_game(t, 25, 15)
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            path = find_path(t, gx0, gy0, gx1, gy1, max_nodes=10000, jitter=0.0)
        assert path is None

    def test_same_start_and_goal(self):
        """Same start and goal should return a short path."""
        t = _path_grid(20, 20)
        gx, gy = _grid_to_game(t, 10, 10)
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            path = find_path(t, gx, gy, gx, gy, jitter=0.0)
        assert path is not None
        assert len(path) >= 1
        # Path should be very short -- just the point itself
        assert len(path) <= 3

    def test_max_nodes_limit(self):
        """Hitting the node limit returns None."""
        t = _path_grid(50, 50)
        gx0, gy0 = _grid_to_game(t, 2, 2)
        gx1, gy1 = _grid_to_game(t, 48, 48)
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            # Very low node limit -- should hit it on a 50x50 grid
            result = find_path(t, gx0, gy0, gx1, gy1, max_nodes=1, jitter=0.0)
        # With max_nodes=1 JPS gets one expansion, then hits the limit.
        # The A* fallback also gets max_nodes=1. Very likely returns None.
        assert result is None


# ------------------------------------------------------------------
# _find_path_astar -- A* fallback
# ------------------------------------------------------------------


class TestFindPathAstar:
    def test_straight_line(self):
        """A* finds a straight-line path on open terrain."""
        t = _path_grid(20, 20)
        gx0, gy0 = _grid_to_game(t, 2, 2)
        gx1, gy1 = _grid_to_game(t, 15, 15)
        import time

        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            path = _find_path_astar(
                t,
                gx0,
                gy0,
                gx1,
                gy1,
                max_nodes=50000,
                jitter=0.0,
                t0=time.perf_counter(),
                wb=t._walk_bits,
                wbc=t._walk_byte_cols,
            )
        assert path is not None
        assert len(path) >= 2

    def test_around_wall(self):
        """A* routes around a wall."""
        blocked = [(10, r) for r in range(5, 16)]
        t = _path_grid(20, 20, blocked_cells=blocked)
        gx0, gy0 = _grid_to_game(t, 5, 10)
        gx1, gy1 = _grid_to_game(t, 15, 10)
        import time

        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            path = _find_path_astar(
                t,
                gx0,
                gy0,
                gx1,
                gy1,
                max_nodes=50000,
                jitter=0.0,
                t0=time.perf_counter(),
                wb=t._walk_bits,
                wbc=t._walk_byte_cols,
            )
        assert path is not None

    def test_no_path(self):
        """A* returns None when goal is unreachable (impassable wall)."""
        # Block entire column 10 across all rows -> left/right disconnect
        blocked = [(10, r) for r in range(20)]
        t = _path_grid(20, 20, blocked_cells=blocked)
        gx0, gy0 = _grid_to_game(t, 2, 10)
        gx1, gy1 = _grid_to_game(t, 15, 10)
        import time

        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            path = _find_path_astar(
                t,
                gx0,
                gy0,
                gx1,
                gy1,
                max_nodes=50000,
                jitter=0.0,
                t0=time.perf_counter(),
                wb=t._walk_bits,
                wbc=t._walk_byte_cols,
            )
        assert path is None

    def test_start_blocked_snaps(self):
        """A* snaps a blocked start to a nearby walkable cell."""
        blocked = [(5, 5)]
        t = _path_grid(20, 20, blocked_cells=blocked)
        gx0, gy0 = _grid_to_game(t, 5, 5)
        gx1, gy1 = _grid_to_game(t, 15, 15)
        import time

        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            path = _find_path_astar(
                t,
                gx0,
                gy0,
                gx1,
                gy1,
                max_nodes=50000,
                jitter=0.0,
                t0=time.perf_counter(),
                wb=t._walk_bits,
                wbc=t._walk_byte_cols,
            )
        assert path is not None

    def test_same_start_and_goal(self):
        """A* with same start and goal returns a short path."""
        t = _path_grid(20, 20)
        gx, gy = _grid_to_game(t, 10, 10)
        import time

        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            path = _find_path_astar(
                t,
                gx,
                gy,
                gx,
                gy,
                max_nodes=50000,
                jitter=0.0,
                t0=time.perf_counter(),
                wb=t._walk_bits,
                wbc=t._walk_byte_cols,
            )
        assert path is not None
        assert len(path) <= 3
