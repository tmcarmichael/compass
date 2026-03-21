"""Tests for eq.placeables -- obstacle footprint computation.

Covers get_model_radius (prefix matching), compute_obstacle_cells
(circle rasterization), _GridBounds construction, _rasterize_triangle
(barycentric rasterization), and compute_mesh_footprint_cells.
All test data is synthetic (no game asset files).
"""

from __future__ import annotations

import math

from eq.placeables import (
    _DEFAULT_RADIUS,
    _GridBounds,
    _rasterize_triangle,
    compute_mesh_footprint_cells,
    compute_obstacle_cells,
    get_model_radius,
)
from eq.wld import Mesh, MeshTriangle, MeshVertex, ObjectPlacement

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _placement(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    model_name: str = "rock",
    heading: float = 0.0,
    scale: float = 1.0,
) -> ObjectPlacement:
    return ObjectPlacement(
        name="test_placement",
        model_name=model_name,
        x=x,
        y=y,
        z=z,
        heading=heading,
        scale=scale,
    )


def _make_mesh(
    verts: list[tuple[float, float, float]],
    tris: list[tuple[int, int, int]],
    name: str = "test_mesh",
) -> Mesh:
    vertices = [MeshVertex(x, y, z) for x, y, z in verts]
    triangles = [MeshTriangle(v1, v2, v3, 0, 0) for v1, v2, v3 in tris]
    cx = sum(v.x for v in vertices) / len(vertices)
    cy = sum(v.y for v in vertices) / len(vertices)
    cz = sum(v.z for v in vertices) / len(vertices)
    return Mesh(name=name, vertices=vertices, triangles=triangles, center=(cx, cy, cz))


# ------------------------------------------------------------------
# get_model_radius
# ------------------------------------------------------------------


class TestGetModelRadius:
    def test_unknown_returns_default(self):
        assert get_model_radius("zzz_unknown_model_xyz") == _DEFAULT_RADIUS

    def test_any_name_returns_default_when_table_empty(self):
        """With empty _MODEL_RADII, all names fall through to _DEFAULT_RADIUS."""
        assert get_model_radius("rock") == _DEFAULT_RADIUS
        assert get_model_radius("pine") == _DEFAULT_RADIUS
        assert get_model_radius("boulder") == _DEFAULT_RADIUS


# ------------------------------------------------------------------
# _GridBounds NamedTuple
# ------------------------------------------------------------------


class TestGridBounds:
    def test_construction(self):
        b = _GridBounds(
            min_x=0.0,
            min_y=0.0,
            cell_size=1.0,
            min_col=0,
            max_col=9,
            min_row=0,
            max_row=9,
        )
        assert b.min_x == 0.0
        assert b.cell_size == 1.0
        assert b.max_col == 9

    def test_is_named_tuple(self):
        b = _GridBounds(0.0, 0.0, 1.0, 0, 9, 0, 9)
        assert b._fields == ("min_x", "min_y", "cell_size", "min_col", "max_col", "min_row", "max_row")


# ------------------------------------------------------------------
# compute_obstacle_cells
# ------------------------------------------------------------------


class TestComputeObstacleCells:
    def test_basic_circle(self):
        """Place an object at (50, 50) with mesh_radius=12 in a 100x100 grid."""
        p = _placement(x=50.0, y=50.0, model_name="obj")
        cells = compute_obstacle_cells(
            p,
            grid_min_x=0.0,
            grid_min_y=0.0,
            cols=100,
            rows=100,
            cell_size=1.0,
            mesh_radius=12.0,
        )
        assert len(cells) > 0
        # All cells should be within radius of placement
        for col, row in cells:
            cx = 0.0 + (col + 0.5) * 1.0
            cy = 0.0 + (row + 0.5) * 1.0
            dist = math.hypot(cx - 50.0, cy - 50.0)
            assert dist <= 12.0 + 0.01

    def test_zero_radius_no_cells(self):
        """mesh_radius=0 should produce no obstacle cells."""
        p = _placement(x=50.0, y=50.0, model_name="obj")
        cells = compute_obstacle_cells(p, 0.0, 0.0, 100, 100, 1.0, mesh_radius=0.0)
        assert cells == []

    def test_override_radius(self):
        """Passing mesh_radius overrides name-based lookup."""
        p = _placement(x=50.0, y=50.0, model_name="whatever")
        cells = compute_obstacle_cells(p, 0.0, 0.0, 100, 100, 1.0, mesh_radius=5.0)
        assert len(cells) > 0

    def test_scale_affects_radius(self):
        """Larger scale -> more cells covered."""
        p1 = _placement(x=50.0, y=50.0, model_name="obj", scale=1.0)
        p2 = _placement(x=50.0, y=50.0, model_name="obj", scale=2.0)
        cells1 = compute_obstacle_cells(p1, 0.0, 0.0, 100, 100, 1.0, mesh_radius=12.0)
        cells2 = compute_obstacle_cells(p2, 0.0, 0.0, 100, 100, 1.0, mesh_radius=12.0)
        assert len(cells2) > len(cells1)

    def test_edge_of_grid(self):
        """Placement at grid edge should be clipped, not crash."""
        p = _placement(x=0.0, y=0.0, model_name="obj")
        cells = compute_obstacle_cells(p, 0.0, 0.0, 100, 100, 1.0, mesh_radius=12.0)
        # Should still produce some cells (the in-bounds portion)
        assert len(cells) > 0
        for col, row in cells:
            assert 0 <= col < 100
            assert 0 <= row < 100

    def test_approximate_cell_count(self):
        """Circle area ~= pi*r^2, cell count should be roughly that."""
        p = _placement(x=50.0, y=50.0, model_name="obj")
        cells = compute_obstacle_cells(p, 0.0, 0.0, 100, 100, 1.0, mesh_radius=12.0)
        expected_area = math.pi * 12.0 * 12.0
        # Allow 10% tolerance
        assert abs(len(cells) - expected_area) / expected_area < 0.10

    def test_zero_radius_override(self):
        """mesh_radius=0 should produce no cells."""
        p = _placement(x=50.0, y=50.0, model_name="obj")
        cells = compute_obstacle_cells(p, 0.0, 0.0, 100, 100, 1.0, mesh_radius=0.0)
        assert cells == []


# ------------------------------------------------------------------
# _rasterize_triangle
# ------------------------------------------------------------------


class TestRasterizeTriangle:
    def test_simple_triangle(self):
        """A triangle covering a known area should mark those cells."""
        tri = MeshTriangle(v1=0, v2=1, v3=2, flags=0, material_idx=0)
        # Triangle: (1,1) -> (9,1) -> (5,9) -- roughly centered in 10x10
        world_xs = [1.0, 9.0, 5.0]
        world_ys = [1.0, 1.0, 9.0]
        bounds = _GridBounds(0.0, 0.0, 1.0, 0, 9, 0, 9)
        cells: set[tuple[int, int]] = set()
        _rasterize_triangle(tri, world_xs, world_ys, bounds, cells)
        assert len(cells) > 0
        # The center cell (5, 5) should be inside the triangle
        assert (5, 5) in cells or (4, 4) in cells  # depends on rounding

    def test_degenerate_triangle_no_cells(self):
        """A degenerate (collinear) triangle should produce no cells."""
        tri = MeshTriangle(v1=0, v2=1, v3=2, flags=0, material_idx=0)
        world_xs = [0.0, 5.0, 10.0]
        world_ys = [0.0, 0.0, 0.0]  # all on same line
        bounds = _GridBounds(0.0, 0.0, 1.0, 0, 10, 0, 10)
        cells: set[tuple[int, int]] = set()
        _rasterize_triangle(tri, world_xs, world_ys, bounds, cells)
        assert len(cells) == 0

    def test_out_of_range_vertex_index(self):
        """Triangle referencing out-of-range vertex should be skipped."""
        tri = MeshTriangle(v1=0, v2=1, v3=99, flags=0, material_idx=0)
        world_xs = [1.0, 5.0]
        world_ys = [1.0, 5.0]
        bounds = _GridBounds(0.0, 0.0, 1.0, 0, 9, 0, 9)
        cells: set[tuple[int, int]] = set()
        _rasterize_triangle(tri, world_xs, world_ys, bounds, cells)
        assert len(cells) == 0

    def test_already_marked_cells_skipped(self):
        """Pre-existing cells in the set should not be re-processed."""
        tri = MeshTriangle(v1=0, v2=1, v3=2, flags=0, material_idx=0)
        world_xs = [1.0, 9.0, 5.0]
        world_ys = [1.0, 1.0, 9.0]
        bounds = _GridBounds(0.0, 0.0, 1.0, 0, 9, 0, 9)
        cells: set[tuple[int, int]] = set()
        _rasterize_triangle(tri, world_xs, world_ys, bounds, cells)
        first_count = len(cells)
        # Rasterize same triangle again -- count should not change
        _rasterize_triangle(tri, world_xs, world_ys, bounds, cells)
        assert len(cells) == first_count


# ------------------------------------------------------------------
# compute_mesh_footprint_cells
# ------------------------------------------------------------------


class TestComputeMeshFootprintCells:
    def test_simple_quad_footprint(self):
        """A square mesh placed at the grid center should produce cells."""
        mesh = _make_mesh(
            verts=[
                (-5.0, -5.0, 0.0),
                (5.0, -5.0, 0.0),
                (5.0, 5.0, 0.0),
                (-5.0, 5.0, 0.0),
            ],
            tris=[(0, 1, 2), (0, 2, 3)],
        )
        # model_name gets _DEFAULT_RADIUS (non-zero), so footprint proceeds
        p = _placement(x=50.0, y=50.0, model_name="obj", heading=0.0, scale=1.0)
        cells = compute_mesh_footprint_cells(
            mesh, p, grid_min_x=0.0, grid_min_y=0.0, cols=100, rows=100, cell_size=1.0
        )
        assert len(cells) > 0

    def test_empty_mesh_no_cells(self):
        """Mesh with no vertices/triangles should return empty."""
        mesh = Mesh(name="empty", vertices=[], triangles=[], center=(0, 0, 0))
        p = _placement(x=50.0, y=50.0, model_name="obj")
        cells = compute_mesh_footprint_cells(mesh, p, 0.0, 0.0, 100, 100, 1.0)
        assert cells == []

    def test_scale_increases_footprint(self):
        """Scale > 1 should produce more cells than scale 1."""
        mesh = _make_mesh(
            verts=[(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)],
            tris=[(0, 1, 2), (0, 2, 3)],
        )
        p1 = _placement(x=50.0, y=50.0, model_name="obj", scale=1.0)
        p2 = _placement(x=50.0, y=50.0, model_name="obj", scale=2.0)
        cells1 = compute_mesh_footprint_cells(mesh, p1, 0.0, 0.0, 100, 100, 1.0)
        cells2 = compute_mesh_footprint_cells(mesh, p2, 0.0, 0.0, 100, 100, 1.0)
        assert len(cells2) > len(cells1)

    def test_heading_rotates_footprint(self):
        """Non-zero heading should rotate the footprint."""
        # Rectangular mesh: wide in X, narrow in Y
        mesh = _make_mesh(
            verts=[(-10, -2, 0), (10, -2, 0), (10, 2, 0), (-10, 2, 0)],
            tris=[(0, 1, 2), (0, 2, 3)],
        )
        p0 = _placement(x=50.0, y=50.0, model_name="obj", heading=0.0)
        p90 = _placement(x=50.0, y=50.0, model_name="obj", heading=128.0)  # 90 degrees
        cells0 = set(compute_mesh_footprint_cells(mesh, p0, 0.0, 0.0, 100, 100, 1.0))
        cells90 = set(compute_mesh_footprint_cells(mesh, p90, 0.0, 0.0, 100, 100, 1.0))
        # Should have similar count but different cell positions
        assert len(cells0) > 0
        assert len(cells90) > 0
        assert cells0 != cells90
