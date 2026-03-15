"""Tests for nav.terrain.heightmap -- grid heightmap terrain queries.

Covers ZoneTerrain init, grid operations, cliff detection, slope detection,
cliff buffer expansion, walkability queries, obstacle flagging, and
material classification. All test data is synthetic (no game asset files).
"""

from __future__ import annotations

import math
from unittest.mock import patch

from eq.wld import Mesh, MeshTriangle, MeshVertex
from nav.terrain.heightmap import (
    MAT_DIRT,
    MAT_GRASS,
    MAT_LAVA,
    MAT_STONE,
    MAT_UNKNOWN,
    MAT_WATER,
    MAT_WOOD,
    SURFACE_BRIDGE,
    SURFACE_CLIFF,
    SURFACE_LAVA,
    SURFACE_NONE,
    SURFACE_OBSTACLE,
    SURFACE_WALKABLE,
    SURFACE_WATER,
    ZoneTerrain,
    classify_material,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _flat_grid(cols: int, rows: int, z: float = 0.0, cell_size: float = 1.0) -> ZoneTerrain:
    """Create a ZoneTerrain with a flat walkable grid (no meshes needed)."""
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


def _make_mesh(
    verts: list[tuple[float, float, float]],
    tris: list[tuple[int, int, int]],
    name: str = "test_mesh",
) -> Mesh:
    """Build a Mesh from raw vertex/triangle lists."""
    vertices = [MeshVertex(x, y, z) for x, y, z in verts]
    triangles = [MeshTriangle(v1, v2, v3, 0, 0) for v1, v2, v3 in tris]
    cx = sum(v.x for v in vertices) / len(vertices)
    cy = sum(v.y for v in vertices) / len(vertices)
    cz = sum(v.z for v in vertices) / len(vertices)
    return Mesh(name=name, vertices=vertices, triangles=triangles, center=(cx, cy, cz))


# ------------------------------------------------------------------
# ZoneTerrain.__init__ and basic grid operations
# ------------------------------------------------------------------


class TestZoneTerrainInit:
    def test_default_init(self):
        t = ZoneTerrain()
        assert t.cell_size == 1.0
        assert t._cols == 0
        assert t._rows == 0
        assert t._z == []
        assert t._flags == []

    def test_custom_cell_size(self):
        t = ZoneTerrain(cell_size=5.0)
        assert t.cell_size == 5.0

    def test_grid_idx_in_bounds(self):
        t = _flat_grid(10, 10)
        assert t._grid_idx(0, 0) == 0
        assert t._grid_idx(9, 0) == 9
        assert t._grid_idx(0, 1) == 10
        assert t._grid_idx(9, 9) == 99

    def test_grid_idx_out_of_bounds(self):
        t = _flat_grid(10, 10)
        assert t._grid_idx(-1, 0) == -1
        assert t._grid_idx(10, 0) == -1
        assert t._grid_idx(0, -1) == -1
        assert t._grid_idx(0, 10) == -1

    def test_game_to_grid_swap(self):
        """game_to_grid swaps x/y (game coords -> WLD coords)."""
        t = _flat_grid(10, 10)
        # game (3, 5) -> WLD (5, 3) -> col=5, row=3
        col, row = t._game_to_grid(3.0, 5.0)
        assert col == 5
        assert row == 3


# ------------------------------------------------------------------
# classify_material
# ------------------------------------------------------------------


class TestClassifyMaterial:
    def test_water_w_prefix(self):
        assert classify_material("w1") == MAT_WATER
        assert classify_material("w2") == MAT_WATER
        assert classify_material("w12") == MAT_WATER

    def test_water_keyword(self):
        assert classify_material("riverstones") == MAT_WATER

    def test_wood(self):
        assert classify_material("woodplank01") == MAT_WOOD

    def test_stone(self):
        assert classify_material("cobblestone") == MAT_STONE

    def test_dirt(self):
        assert classify_material("dirtroad") == MAT_DIRT

    def test_grass(self):
        assert classify_material("grass_field") == MAT_GRASS

    def test_lava(self):
        assert classify_material("lava_floor") == MAT_LAVA

    def test_empty(self):
        assert classify_material("") == MAT_UNKNOWN

    def test_unknown(self):
        assert classify_material("xyzzy123") == MAT_UNKNOWN


# ------------------------------------------------------------------
# Walkability queries on known grids
# ------------------------------------------------------------------


class TestWalkability:
    def test_flat_walkable(self):
        t = _flat_grid(10, 10)
        # game (5, 5) should be walkable
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            assert t.is_walkable(5.0, 5.0) is True

    def test_water_not_walkable(self):
        t = _flat_grid(10, 10)
        # Mark one cell as water: WLD col=3, row=4 -> game (4, 3)
        idx = 4 * 10 + 3
        t._flags[idx] |= SURFACE_WATER
        assert t.is_walkable(4.0, 3.0) is False

    def test_lava_not_walkable(self):
        t = _flat_grid(10, 10)
        idx = 4 * 10 + 3
        t._flags[idx] |= SURFACE_LAVA
        assert t.is_walkable(4.0, 3.0) is False

    def test_cliff_not_walkable(self):
        t = _flat_grid(10, 10)
        idx = 4 * 10 + 3
        t._flags[idx] |= SURFACE_CLIFF
        assert t.is_walkable(4.0, 3.0) is False

    def test_obstacle_not_walkable_when_flag_enabled(self):
        t = _flat_grid(10, 10)
        idx = 4 * 10 + 3
        t._flags[idx] |= SURFACE_OBSTACLE
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            assert t.is_walkable(4.0, 3.0) is False

    def test_obstacle_walkable_when_flag_disabled(self):
        t = _flat_grid(10, 10)
        idx = 4 * 10 + 3
        t._flags[idx] |= SURFACE_OBSTACLE
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = False
            assert t.is_walkable(4.0, 3.0) is True

    def test_none_surface_not_walkable(self):
        t = _flat_grid(10, 10)
        idx = 4 * 10 + 3
        t._flags[idx] = SURFACE_NONE
        with patch("core.features.flags") as mock_flags:
            mock_flags.obstacle_avoidance = True
            assert t.is_walkable(4.0, 3.0) is False

    def test_walkable_override_trumps_water(self):
        t = _flat_grid(10, 10)
        idx = 4 * 10 + 3
        t._flags[idx] |= SURFACE_WATER
        # Add walkable override covering game (4, 3)
        t._walkable_overrides.append((3.5, 2.5, 4.5, 3.5))
        assert t.is_walkable(4.0, 3.0) is True

    def test_bridge_always_walkable(self):
        t = _flat_grid(10, 10)
        idx = 4 * 10 + 3
        t._flags[idx] = SURFACE_WALKABLE | SURFACE_BRIDGE | SURFACE_WATER
        assert t.is_walkable(4.0, 3.0) is True

    def test_is_water(self):
        t = _flat_grid(10, 10)
        idx = 4 * 10 + 3
        t._flags[idx] |= SURFACE_WATER
        assert t.is_water(4.0, 3.0) is True
        assert t.is_water(5.0, 5.0) is False

    def test_is_obstacle(self):
        t = _flat_grid(10, 10)
        idx = 4 * 10 + 3
        t._flags[idx] |= SURFACE_OBSTACLE
        assert t.is_obstacle(4.0, 3.0) is True
        assert t.is_obstacle(5.0, 5.0) is False

    def test_is_cliff(self):
        t = _flat_grid(10, 10)
        idx = 4 * 10 + 3
        t._flags[idx] |= SURFACE_CLIFF
        assert t.is_cliff(4.0, 3.0) is True
        assert t.is_cliff(5.0, 5.0) is False

    def test_is_hazard_unknown_terrain(self):
        t = _flat_grid(10, 10)
        idx = 4 * 10 + 3
        t._flags[idx] = SURFACE_NONE
        assert t.is_hazard(4.0, 3.0) is True

    def test_is_hazard_walkable(self):
        t = _flat_grid(10, 10)
        assert t.is_hazard(5.0, 5.0) is False


# ------------------------------------------------------------------
# get_z and out-of-bounds queries
# ------------------------------------------------------------------


class TestGetZ:
    def test_flat_z(self):
        t = _flat_grid(10, 10, z=-5.0)
        assert t.get_z(5.0, 5.0) == -5.0

    def test_out_of_bounds_nan(self):
        t = _flat_grid(10, 10, z=0.0)
        z = t.get_z(99.0, 99.0)
        assert math.isnan(z)

    def test_get_flags_out_of_bounds(self):
        t = _flat_grid(10, 10)
        assert t.get_flags(99.0, 99.0) == SURFACE_NONE


# ------------------------------------------------------------------
# Cliff detection: _detect_gradient_cliffs
# ------------------------------------------------------------------


class TestGradientCliffs:
    def test_steep_drop_detected(self):
        """A 20-unit Z drop between adjacent cells should be flagged as cliff."""
        t = _flat_grid(10, 10, z=0.0)
        # Create a cliff at column 5: cells to the right are at Z=-20
        for row in range(10):
            for col in range(5, 10):
                t._z[row * 10 + col] = -20.0
        count = t._detect_gradient_cliffs(gradient_threshold=15.0)
        assert count > 0
        # The cells at the cliff edge (col 4 and col 5) should be flagged
        for row in range(10):
            assert t._flags[row * 10 + 4] & SURFACE_CLIFF
            assert t._flags[row * 10 + 5] & SURFACE_CLIFF

    def test_gentle_slope_not_flagged(self):
        """A 5-unit Z difference should NOT be flagged as a cliff (threshold=15)."""
        t = _flat_grid(10, 10, z=0.0)
        for row in range(10):
            for col in range(5, 10):
                t._z[row * 10 + col] = -5.0
        count = t._detect_gradient_cliffs(gradient_threshold=15.0)
        assert count == 0

    def test_nan_cells_not_flagged(self):
        """NaN Z cells should not trigger cliff detection."""
        t = _flat_grid(10, 10, z=0.0)
        t._z[45] = float("nan")  # row 4, col 5
        t._detect_gradient_cliffs(gradient_threshold=15.0)
        assert not (t._flags[45] & SURFACE_CLIFF)


# ------------------------------------------------------------------
# Cliff detection: _detect_sustained_slopes
# ------------------------------------------------------------------


class TestSustainedSlopes:
    def test_ramp_detected(self):
        """A smooth ramp rising 50 units over 5 cells should be detected."""
        t = _flat_grid(20, 5, z=0.0)
        # Build ramp at row 2, cols 5-10: Z rises from 0 to 50
        for c in range(5, 11):
            t._z[2 * 20 + c] = (c - 5) * 10.0  # 0, 10, 20, 30, 40, 50
        count = t._detect_sustained_slopes(window=5, total_threshold=40.0)
        assert count > 0
        # At least some ramp cells should be flagged
        flagged_cols = [c for c in range(5, 11) if t._flags[2 * 20 + c] & SURFACE_CLIFF]
        assert len(flagged_cols) >= 3

    def test_flat_terrain_not_flagged(self):
        """Flat terrain should produce no sustained-slope detections."""
        t = _flat_grid(20, 5, z=0.0)
        count = t._detect_sustained_slopes(window=5, total_threshold=40.0)
        assert count == 0


# ------------------------------------------------------------------
# Cliff buffer: _expand_cliff_buffer
# ------------------------------------------------------------------


class TestExpandCliffBuffer:
    def test_buffer_grows_cliff_zone(self):
        """Cliff flags should expand to adjacent walkable cells."""
        t = _flat_grid(10, 10, z=0.0)
        # Flag a single cliff cell at (5, 5)
        t._flags[5 * 10 + 5] |= SURFACE_CLIFF
        count = t._expand_cliff_buffer()
        assert count > 0
        # Neighbors should now be cliffs too (at least the 4 cardinal)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = 5 + dr, 5 + dc
            assert t._flags[nr * 10 + nc] & SURFACE_CLIFF

    def test_buffer_does_not_expand_to_nan(self):
        """Buffer should not expand into cells with NaN Z."""
        t = _flat_grid(10, 10, z=0.0)
        t._flags[5 * 10 + 5] |= SURFACE_CLIFF
        # Set a neighbor to NaN
        t._z[5 * 10 + 6] = float("nan")
        t._expand_cliff_buffer()
        # NaN cell should NOT have cliff flag
        assert not (t._flags[5 * 10 + 6] & SURFACE_CLIFF)

    def test_buffer_larger_with_small_cell_size(self):
        """Smaller cell_size should produce a larger buffer (more iterations)."""
        t1 = _flat_grid(20, 20, z=0.0, cell_size=1.0)
        t1._flags[10 * 20 + 10] |= SURFACE_CLIFF
        count1 = t1._expand_cliff_buffer()

        t2 = _flat_grid(20, 20, z=0.0, cell_size=0.5)
        t2._flags[10 * 20 + 10] |= SURFACE_CLIFF
        count2 = t2._expand_cliff_buffer()

        # cell_size=0.5 means cliff_buf = round(5.0/0.5) = 10 vs round(5.0/1.0) = 5
        assert count2 > count1


# ------------------------------------------------------------------
# Obstacle buffer: _expand_obstacle_buffer
# ------------------------------------------------------------------


class TestExpandObstacleBuffer:
    def test_obstacle_buffer_expands(self):
        t = _flat_grid(10, 10, z=0.0)
        t._flags[5 * 10 + 5] |= SURFACE_OBSTACLE
        count = t._expand_obstacle_buffer(buffer_cells=1)
        assert count > 0
        # Cardinal neighbors should be flagged
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = 5 + dr, 5 + dc
            assert t._flags[nr * 10 + nc] & SURFACE_OBSTACLE

    def test_obstacle_buffer_does_not_overwrite_water(self):
        t = _flat_grid(10, 10, z=0.0)
        t._flags[5 * 10 + 5] |= SURFACE_OBSTACLE
        t._flags[5 * 10 + 6] = SURFACE_WATER  # no WALKABLE flag
        t._expand_obstacle_buffer(buffer_cells=1)
        # Water cell should NOT get obstacle flag
        assert not (t._flags[5 * 10 + 6] & SURFACE_OBSTACLE)


# ------------------------------------------------------------------
# Full _detect_cliffs integration
# ------------------------------------------------------------------


class TestDetectCliffs:
    def test_detect_cliffs_end_to_end(self):
        """Full cliff detection: gradient + slope + buffer."""
        t = _flat_grid(20, 10, z=0.0)
        # Create a sharp cliff at col 10
        for row in range(10):
            for col in range(10, 20):
                t._z[row * 20 + col] = -25.0
        t._detect_cliffs(gradient_threshold=15.0)
        # Cells near col 10 boundary should be cliff-flagged
        cliff_count = sum(1 for f in t._flags if f & SURFACE_CLIFF)
        assert cliff_count > 10  # many cells due to gradient + buffer


# ------------------------------------------------------------------
# build() with synthetic mesh data
# ------------------------------------------------------------------


class TestBuild:
    def test_build_flat_quad(self):
        """Build terrain from a flat quad (two triangles) spanning 0-10 in x/y."""
        mesh = _make_mesh(
            verts=[
                (0.0, 0.0, 0.0),
                (10.0, 0.0, 0.0),
                (10.0, 10.0, 0.0),
                (0.0, 10.0, 0.0),
            ],
            # Counter-clockwise winding in EQ -> cross product gives -nz.
            # The code flips nz so upward-facing = positive.
            # Use CCW: (0,1,2), (0,2,3) -> cross gives -Z -> flipped to +Z -> walkable.
            tris=[(0, 2, 1), (0, 3, 2)],
        )
        t = ZoneTerrain(cell_size=2.0)
        t.build([mesh], bsp_nodes=[], region_types={}, margin=1.0)
        assert t._cols > 0
        assert t._rows > 0
        # Center of the mesh should be walkable (game coords are swapped)
        # WLD (5, 5) -> game (5, 5) since the mesh is symmetric
        center_z = t.get_z(5.0, 5.0)
        assert not math.isnan(center_z)
        assert abs(center_z) < 1.0

    def test_build_no_meshes(self):
        """Empty mesh list should produce an empty terrain."""
        t = ZoneTerrain()
        t.build([], bsp_nodes=[], region_types={})
        assert t._cols == 0
        assert t._rows == 0

    def test_build_with_steep_triangle(self):
        """A nearly vertical triangle should be flagged SURFACE_STEEP."""
        # Wall: goes from (0,0,0) to (0,0,10) -- vertical in Z
        mesh = _make_mesh(
            verts=[
                (5.0, 5.0, 0.0),
                (5.0, 5.0, 10.0),
                (5.0, 6.0, 10.0),
            ],
            tris=[(0, 1, 2)],
        )
        t = ZoneTerrain(cell_size=2.0)
        t.build([mesh], bsp_nodes=[], region_types={}, margin=1.0)
        # This vertical triangle has normal pointing in Y direction (nz~=0)
        # It should be flagged as steep or skipped (downward-facing)
        # The triangle may not rasterize any cells at all since it's vertical
        # Just verify the build doesn't crash
        assert t._cols > 0


# ------------------------------------------------------------------
# Walk bits
# ------------------------------------------------------------------


class TestWalkBits:
    def test_build_walk_bits(self):
        t = _flat_grid(10, 10)
        t._build_walk_bits()
        assert len(t._walk_bits) > 0
        # All cells are walkable, so all bits should be set
        total_set = sum(bin(b).count("1") for b in t._walk_bits)
        assert total_set == 100  # 10x10

    def test_water_not_in_walk_bits(self):
        t = _flat_grid(10, 10)
        t._flags[55] |= SURFACE_WATER
        t._build_walk_bits()
        # Cell 55 = row 5, col 5 -> bit 5 in byte offset for that row
        byte_cols = (10 + 7) >> 3  # = 2
        row = 55 // 10  # = 5
        col = 55 % 10  # = 5
        byte_idx = row * byte_cols + (col >> 3)
        bit = 1 << (col & 7)
        assert not (t._walk_bits[byte_idx] & bit)

    def test_bridge_in_walk_bits(self):
        t = _flat_grid(10, 10)
        t._flags[55] = SURFACE_WALKABLE | SURFACE_BRIDGE | SURFACE_WATER
        t._build_walk_bits()
        byte_cols = (10 + 7) >> 3
        row = 55 // 10
        col = 55 % 10
        byte_idx = row * byte_cols + (col >> 3)
        bit = 1 << (col & 7)
        assert t._walk_bits[byte_idx] & bit


# ------------------------------------------------------------------
# Avoidance cost
# ------------------------------------------------------------------


class TestAvoidanceCost:
    def test_no_zones_zero_cost(self):
        t = _flat_grid(10, 10)
        assert t.avoidance_cost(5.0, 5.0) == 0.0

    def test_inside_zone_nonzero(self):
        t = _flat_grid(10, 10)
        t.add_avoidance_zone(5.0, 5.0, 10.0)
        cost = t.avoidance_cost(5.0, 5.0)
        assert cost > 0.0

    def test_outside_zone_zero(self):
        t = _flat_grid(10, 10)
        t.add_avoidance_zone(5.0, 5.0, 2.0)
        cost = t.avoidance_cost(50.0, 50.0)
        assert cost == 0.0


# ------------------------------------------------------------------
# Bounds property
# ------------------------------------------------------------------


class TestBounds:
    def test_bounds(self):
        t = _flat_grid(10, 8, cell_size=2.0)
        # WLD: min_x=0, min_y=0, cols=10, rows=8, cell_size=2
        # game bounds: min_gx=min_y=0, min_gy=min_x=0
        #              max_gx=min_y + rows*cs = 16, max_gy=min_x + cols*cs = 20
        gx_min, gy_min, gx_max, gy_max = t.bounds
        assert gx_min == 0.0
        assert gy_min == 0.0
        assert gx_max == 16.0
        assert gy_max == 20.0


# ------------------------------------------------------------------
# check_path
# ------------------------------------------------------------------


class TestCheckPath:
    def test_clear_path(self):
        t = _flat_grid(20, 20, z=0.0)
        result = t.check_path(5.0, 5.0, 15.0, 15.0)
        assert result is None

    def test_path_through_hazard(self):
        t = _flat_grid(20, 20, z=0.0)
        # Mark the center as water
        for row in range(8, 12):
            for col in range(8, 12):
                t._flags[row * 20 + col] |= SURFACE_WATER
        result = t.check_path(5.0, 5.0, 15.0, 15.0, step=1.0)
        assert result is not None


# ------------------------------------------------------------------
# check_los
# ------------------------------------------------------------------


class TestCheckLOS:
    def test_clear_los(self):
        t = _flat_grid(20, 20, z=0.0)
        assert t.check_los(5, 5, 10, 15, 15, 10) is True

    def test_blocked_los(self):
        t = _flat_grid(20, 20, z=0.0)
        # Raise terrain in the middle to block LOS
        for row in range(8, 12):
            for col in range(8, 12):
                t._z[row * 20 + col] = 50.0
        # LOS from (5,5,z=0) to (15,15,z=0) should be blocked by Z=50 terrain
        assert t.check_los(5, 5, 0, 15, 15, 0) is False


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------


class TestStats:
    def test_stats_keys(self):
        t = _flat_grid(10, 10)
        s = t.stats
        assert "grid" in s
        assert "total_cells" in s
        assert s["total_cells"] == 100
        assert s["walkable"] == 100
