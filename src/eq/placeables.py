"""Object placement footprint computation for obstacle avoidance.

Model radius data removed from public release.

Provides model radius lookup and AABB computation for placed zone
objects (trees, rocks, buildings). Used during terrain cache build
to bake SURFACE_OBSTACLE flags into the heightmap grid.

Tier 1: Name-based radius lookup (covers 80%+ of collisions).
Tier 2: Mesh AABB computation (handles irregular shapes).
Tier 3: Mesh-accurate triangle rasterization (exact footprints).
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

from eq.wld import Mesh, MeshTriangle, ObjectPlacement

log = logging.getLogger(__name__)


# ======================================================================
# Model radius lookup table
# ======================================================================

_MODEL_RADII: dict[str, float] = {}

_DEFAULT_RADIUS = 5.0

# Maximum footprint size in game units. Objects larger than this are
# likely terrain features (already in the heightmap mesh) or buildings
# whose interiors should remain walkable.
_MAX_FOOTPRINT = 200.0


def get_model_radius(model_name: str) -> float:
    """Return collision radius for a model name (case-insensitive prefix match).

    Returns 0.0 for known walkable/passthrough objects.
    Returns _DEFAULT_RADIUS for unknown models.
    """
    lower = model_name.lower()

    # Exact match first (fastest path)
    if lower in _MODEL_RADII:
        return _MODEL_RADII[lower]

    # Prefix match: check if model name starts with any known prefix
    # Sort by length descending so longer prefixes match first
    for prefix, radius in sorted(_MODEL_RADII.items(), key=lambda kv: len(kv[0]), reverse=True):
        if lower.startswith(prefix):
            return radius

    return _DEFAULT_RADIUS


# ======================================================================
# Mesh AABB computation (Tier 2)
# ======================================================================


def compute_mesh_aabb(
    mesh: Mesh,
) -> tuple[float, float, float, float, float, float]:
    """Compute axis-aligned bounding box from mesh vertices.

    Returns (min_x, min_y, min_z, max_x, max_y, max_z) relative to
    mesh center. All coordinates in WLD space.
    """
    if not mesh.vertices:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    xs = [v.x for v in mesh.vertices]
    ys = [v.y for v in mesh.vertices]
    zs = [v.z for v in mesh.vertices]

    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def compute_mesh_footprint_radius(
    mesh: Mesh,
) -> float:
    """Compute 2D footprint radius from mesh AABB.

    Returns the maximum horizontal distance from mesh center to any
    vertex, giving a circular approximation of the footprint.
    """
    if not mesh.vertices:
        return 0.0

    cx, cy, _cz = mesh.center
    max_r_sq = 0.0
    for v in mesh.vertices:
        dx = v.x - cx
        dy = v.y - cy
        r_sq = dx * dx + dy * dy
        if r_sq > max_r_sq:
            max_r_sq = r_sq

    return math.sqrt(max_r_sq)


# ======================================================================
# Footprint cell computation
# ======================================================================


def compute_obstacle_cells(
    placement: ObjectPlacement,
    grid_min_x: float,
    grid_min_y: float,
    cols: int,
    rows: int,
    cell_size: float,
    mesh_radius: float | None = None,
) -> list[tuple[int, int]]:
    """Compute grid cells covered by a placed object's footprint.

    Args:
        placement: The object placement (WLD coordinates).
        grid_min_x: Grid origin X (WLD).
        grid_min_y: Grid origin Y (WLD).
        cols: Grid column count.
        rows: Grid row count.
        cell_size: Grid cell size.
        mesh_radius: Override radius from mesh AABB (Tier 2).
                     If None, uses name-based lookup (Tier 1).

    Returns:
        List of (col, row) grid cells within the object's footprint.
    """
    if mesh_radius is not None:
        radius = mesh_radius
    else:
        radius = get_model_radius(placement.model_name)

    if radius <= 0.0:
        return []  # walkable/passthrough object

    # Apply scale
    radius *= placement.scale

    # Cap at maximum footprint size
    if radius > _MAX_FOOTPRINT / 2:
        log.debug(
            "Object %s footprint capped: %.0f -> %.0f", placement.model_name, radius, _MAX_FOOTPRINT / 2
        )
        radius = _MAX_FOOTPRINT / 2

    # Compute grid cell range covered by the circle
    min_col = max(0, int((placement.x - radius - grid_min_x) / cell_size))
    max_col = min(cols - 1, int((placement.x + radius - grid_min_x) / cell_size))
    min_row = max(0, int((placement.y - radius - grid_min_y) / cell_size))
    max_row = min(rows - 1, int((placement.y + radius - grid_min_y) / cell_size))

    # Circle rasterization: include cells whose center is within radius
    cells: list[tuple[int, int]] = []
    r_sq = radius * radius

    for row in range(min_row, max_row + 1):
        cy = grid_min_y + (row + 0.5) * cell_size
        dy = cy - placement.y
        dy_sq = dy * dy
        if dy_sq > r_sq:
            continue
        for col in range(min_col, max_col + 1):
            cx = grid_min_x + (col + 0.5) * cell_size
            dx = cx - placement.x
            if dx * dx + dy_sq <= r_sq:
                cells.append((col, row))

    return cells


# ======================================================================
# Mesh-accurate footprint (Tier 3)
# ======================================================================


class _GridBounds(NamedTuple):
    """Grid-space bounding box passed to the triangle rasterizer."""

    min_x: float
    min_y: float
    cell_size: float
    min_col: int
    max_col: int
    min_row: int
    max_row: int


def _rasterize_triangle(
    tri: MeshTriangle,
    world_xs: list[float],
    world_ys: list[float],
    bounds: _GridBounds,
    cells: set[tuple[int, int]],
) -> None:
    """Rasterize a single mesh triangle onto the grid via barycentric coords.

    Adds covered (col, row) pairs to *cells* in place.
    """
    if tri.v1 >= len(world_xs) or tri.v2 >= len(world_xs) or tri.v3 >= len(world_xs):
        return

    v1x, v1y = world_xs[tri.v1], world_ys[tri.v1]
    v2x, v2y = world_xs[tri.v2], world_ys[tri.v2]
    v3x, v3y = world_xs[tri.v3], world_ys[tri.v3]

    # Per-triangle bounding box (clipped to mesh bbox)
    t_min_col = max(bounds.min_col, int((min(v1x, v2x, v3x) - bounds.min_x) / bounds.cell_size))
    t_max_col = min(bounds.max_col, int((max(v1x, v2x, v3x) - bounds.min_x) / bounds.cell_size))
    t_min_row = max(bounds.min_row, int((min(v1y, v2y, v3y) - bounds.min_y) / bounds.cell_size))
    t_max_row = min(bounds.max_row, int((max(v1y, v2y, v3y) - bounds.min_y) / bounds.cell_size))

    # Barycentric denominator
    denom = (v2y - v3y) * (v1x - v3x) + (v3x - v2x) * (v1y - v3y)
    if abs(denom) < 1e-10:
        return
    inv_denom = 1.0 / denom

    for row in range(t_min_row, t_max_row + 1):
        py = bounds.min_y + (row + 0.5) * bounds.cell_size
        for col in range(t_min_col, t_max_col + 1):
            if (col, row) in cells:
                continue  # already marked
            px = bounds.min_x + (col + 0.5) * bounds.cell_size

            w1 = ((v2y - v3y) * (px - v3x) + (v3x - v2x) * (py - v3y)) * inv_denom
            w2 = ((v3y - v1y) * (px - v3x) + (v1x - v3x) * (py - v3y)) * inv_denom
            w3 = 1.0 - w1 - w2

            if w1 >= -0.001 and w2 >= -0.001 and w3 >= -0.001:
                cells.add((col, row))


def compute_mesh_footprint_cells(
    mesh: Mesh,
    placement: ObjectPlacement,
    grid_min_x: float,
    grid_min_y: float,
    cols: int,
    rows: int,
    cell_size: float,
) -> list[tuple[int, int]]:
    """Compute grid cells by rasterizing actual mesh triangles.

    Projects mesh triangles to 2D, applies placement transform
    (scale, rotate by heading, translate), and rasterizes onto the
    grid using barycentric coordinates. Returns exact footprint
    cells rather than a circular approximation.

    Falls back to empty list if mesh has no usable triangles.
    Caller should fall back to circular radius in that case.

    Args:
        mesh: Object mesh from _obj.s3d (vertices in model space).
        placement: Object placement (WLD coordinates + heading + scale).
        grid_min_x: Grid origin X (WLD).
        grid_min_y: Grid origin Y (WLD).
        cols: Grid column count.
        rows: Grid row count.
        cell_size: Grid cell size.

    Returns:
        List of (col, row) grid cells within the mesh footprint.
    """
    if not mesh.vertices or not mesh.triangles:
        return []

    # Check if this is a walkable/passthrough object by name
    if get_model_radius(placement.model_name) <= 0.0:
        return []

    # Transform mesh vertices from model space to world WLD space:
    #   1. Subtract mesh center to get local coords
    #   2. Scale by placement.scale
    #   3. Rotate by heading (EQ 0-512 system)
    #   4. Translate to placement position
    angle = placement.heading * math.pi * 2.0 / 512.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    scale = placement.scale
    cx, cy, _cz = mesh.center

    world_xs: list[float] = []
    world_ys: list[float] = []
    for v in mesh.vertices:
        lx = (v.x - cx) * scale
        ly = (v.y - cy) * scale
        wx = placement.x + cos_a * lx - sin_a * ly
        wy = placement.y + sin_a * lx + cos_a * ly
        world_xs.append(wx)
        world_ys.append(wy)

    # Bounding box in grid space
    bb_min_col = max(0, int((min(world_xs) - grid_min_x) / cell_size))
    bb_max_col = min(cols - 1, int((max(world_xs) - grid_min_x) / cell_size))
    bb_min_row = max(0, int((min(world_ys) - grid_min_y) / cell_size))
    bb_max_row = min(rows - 1, int((max(world_ys) - grid_min_y) / cell_size))

    # Cap at maximum footprint size (large meshes are terrain, not obstacles)
    extent_x = (bb_max_col - bb_min_col + 1) * cell_size
    extent_y = (bb_max_row - bb_min_row + 1) * cell_size
    if extent_x > _MAX_FOOTPRINT or extent_y > _MAX_FOOTPRINT:
        log.debug("Mesh footprint %s capped: %.0fx%.0f exceeds max", placement.model_name, extent_x, extent_y)
        return []

    # Rasterize each triangle via barycentric coordinates
    cells: set[tuple[int, int]] = set()
    bounds = _GridBounds(grid_min_x, grid_min_y, cell_size, bb_min_col, bb_max_col, bb_min_row, bb_max_row)

    for tri in mesh.triangles:
        _rasterize_triangle(tri, world_xs, world_ys, bounds, cells)

    return list(cells)
