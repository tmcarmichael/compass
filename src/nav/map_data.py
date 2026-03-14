"""Parse zone map files and provide segment-based obstacle queries.

Map format (.txt files):
  L x1, y1, z1, x2, y2, z2, r, g, b    -  wall/terrain line segment
  P x, y, z, r, g, b, size, label       -  POI label

Coordinate convention: map (x, y) corresponds to (state.x, state.y)
in our memory reader  -  i.e., map_x = offset 0x34, map_y = offset 0x30.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from core.types import Point

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Segment:
    """A 2D line segment from the zone map (Z dropped for XY pathfinding)."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class POI:
    """A point of interest from the zone map."""

    x: float
    y: float
    z: float
    label: str


class ZoneMap:
    """Parsed zone map with segment intersection queries."""

    def __init__(self, segments: list[Segment], pois: list[POI]) -> None:
        self.segments = segments
        self.pois = pois
        # Spatial index: grid cells mapping to segment indices
        self._cell_size = 50.0
        self._grid: dict[tuple[int, int], list[int]] = {}
        self._build_grid()

    def _build_grid(self) -> None:
        """Build spatial hash grid for fast segment lookup."""
        for i, seg in enumerate(self.segments):
            cells = self._segment_cells(seg)
            for cell in cells:
                if cell not in self._grid:
                    self._grid[cell] = []
                self._grid[cell].append(i)
        log.info(
            "[NAV] ZoneMap: %d segments indexed into %d grid cells (cell_size=%.0f)",
            len(self.segments),
            len(self._grid),
            self._cell_size,
        )

    def _pos_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor(x / self._cell_size)), int(math.floor(y / self._cell_size)))

    def _segment_cells(self, seg: Segment) -> set[tuple[int, int]]:
        """Return all grid cells a segment passes through."""
        cells = set()
        cx1, cy1 = self._pos_to_cell(seg.x1, seg.y1)
        cx2, cy2 = self._pos_to_cell(seg.x2, seg.y2)
        # Walk from (cx1,cy1) to (cx2,cy2), adding all cells
        min_cx, max_cx = min(cx1, cx2), max(cx1, cx2)
        min_cy, max_cy = min(cy1, cy2), max(cy1, cy2)
        for gx in range(min_cx, max_cx + 1):
            for gy in range(min_cy, max_cy + 1):
                cells.add((gx, gy))
        return cells

    def _cells_along_ray(self, x1: float, y1: float, x2: float, y2: float) -> set[tuple[int, int]]:
        """Return all grid cells a ray from (x1,y1) to (x2,y2) passes through."""
        cells = set()
        cx1, cy1 = self._pos_to_cell(x1, y1)
        cx2, cy2 = self._pos_to_cell(x2, y2)
        min_cx, max_cx = min(cx1, cx2), max(cx1, cx2)
        min_cy, max_cy = min(cy1, cy2), max(cy1, cy2)
        for gx in range(min_cx, max_cx + 1):
            for gy in range(min_cy, max_cy + 1):
                cells.add((gx, gy))
        return cells

    def path_blocked(self, x1: float, y1: float, x2: float, y2: float, buffer: float = 3.0) -> Segment | None:
        """Check if a straight-line path intersects any map segment.

        Args:
            x1, y1: Start position.
            x2, y2: End position.
            buffer: Minimum clearance from segments (character width).

        Returns:
            The first blocking Segment, or None if path is clear.
        """
        # Collect candidate segments from grid cells along the ray
        cells = self._cells_along_ray(x1, y1, x2, y2)
        checked = set()

        for cell in cells:
            for seg_idx in self._grid.get(cell, ()):
                if seg_idx in checked:
                    continue
                checked.add(seg_idx)
                seg = self.segments[seg_idx]
                if _segments_intersect(x1, y1, x2, y2, seg.x1, seg.y1, seg.x2, seg.y2, buffer):
                    return seg
        return None

    def nearest_segment_dist(self, x: float, y: float) -> float:
        """Distance from (x, y) to the nearest map segment. For diagnostics."""
        best = float("inf")
        cx, cy = self._pos_to_cell(x, y)
        # Check local cells + neighbors
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for seg_idx in self._grid.get((cx + dx, cy + dy), ()):
                    d = _point_segment_dist(x, y, self.segments[seg_idx])
                    if d < best:
                        best = d
        return best

    def find_detour(self, x1: float, y1: float, x2: float, y2: float, buffer: float = 3.0) -> Point | None:
        """Find a waypoint that avoids the blocking segment.

        Returns an intermediate (x, y) point offset perpendicular to the
        blocking segment, or None if no block.
        """
        seg = self.path_blocked(x1, y1, x2, y2, buffer)
        if seg is None:
            return None

        # Compute perpendicular offset to go around the segment
        seg_dx = seg.x2 - seg.x1
        seg_dy = seg.y2 - seg.y1
        seg_len = math.hypot(seg_dx, seg_dy)
        if seg_len < 0.01:
            return None

        # Perpendicular direction (normalized)
        perp_x = -seg_dy / seg_len
        perp_y = seg_dx / seg_len

        # Midpoint of the segment
        mid_x = (seg.x1 + seg.x2) / 2
        mid_y = (seg.y1 + seg.y2) / 2

        # Try both sides of the segment, pick the one closer to destination
        offset = buffer + 15.0  # clearance beyond buffer
        wp_a = Point(mid_x + perp_x * offset, mid_y + perp_y * offset, 0.0)
        wp_b = Point(mid_x - perp_x * offset, mid_y - perp_y * offset, 0.0)

        dist_a = math.hypot(wp_a.x - x2, wp_a.y - y2)
        dist_b = math.hypot(wp_b.x - x2, wp_b.y - y2)

        return wp_a if dist_a < dist_b else wp_b


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    """2D cross product."""
    return ax * by - ay * bx


def _segments_intersect(
    ax1: float,
    ay1: float,
    ax2: float,
    ay2: float,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
    buffer: float = 0.0,
) -> bool:
    """Check if two 2D line segments intersect or come within buffer distance."""
    # If buffer > 0, first check minimum distance
    if buffer > 0:
        if _segment_segment_dist(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) < buffer:
            return True
        return False

    # Standard cross-product segment intersection test
    dx = ax2 - ax1
    dy = ay2 - ay1
    ex = bx2 - bx1
    ey = by2 - by1

    denom = _cross(dx, dy, ex, ey)
    if abs(denom) < 1e-10:
        return False  # Parallel

    fx = bx1 - ax1
    fy = by1 - ay1

    t = _cross(fx, fy, ex, ey) / denom
    u = _cross(fx, fy, dx, dy) / denom

    return 0 <= t <= 1 and 0 <= u <= 1


def _point_segment_dist(px: float, py: float, seg: Segment) -> float:
    """Minimum distance from point (px, py) to a line segment."""
    dx = seg.x2 - seg.x1
    dy = seg.y2 - seg.y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-10:
        return math.hypot(px - seg.x1, py - seg.y1)

    t = max(0, min(1, ((px - seg.x1) * dx + (py - seg.y1) * dy) / len_sq))
    proj_x = seg.x1 + t * dx
    proj_y = seg.y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _segment_segment_dist(
    ax1: float, ay1: float, ax2: float, ay2: float, bx1: float, by1: float, bx2: float, by2: float
) -> float:
    """Minimum distance between two line segments."""
    # Check if they intersect (distance = 0)
    if _segments_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, buffer=0):
        return 0.0

    # Otherwise, min of point-to-segment distances for all 4 endpoints
    sa = Segment(bx1, by1, bx2, by2)
    sb = Segment(ax1, ay1, ax2, ay2)
    return min(
        _point_segment_dist(ax1, ay1, sa),
        _point_segment_dist(ax2, ay2, sa),
        _point_segment_dist(bx1, by1, sb),
        _point_segment_dist(bx2, by2, sb),
    )


def load_zone_map(map_path: str | Path) -> ZoneMap:
    """Parse an EQ map file into a ZoneMap."""
    segments = []
    pois = []
    path = Path(map_path)

    if not path.exists():
        log.warning("[NAV] Map file not found: %s", path)
        return ZoneMap([], [])

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("L "):
                parts = line[2:].split(",")
                if len(parts) >= 6:
                    try:
                        x1, y1, _z1 = float(parts[0]), float(parts[1]), float(parts[2])
                        x2, y2, _z2 = float(parts[3]), float(parts[4]), float(parts[5])
                        segments.append(Segment(x1, y1, x2, y2))
                    except (
                        ValueError,
                        IndexError,
                    ):
                        continue

            elif line.startswith("P "):
                parts = line[2:].split(",")
                if len(parts) >= 8:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        label = parts[7].strip()
                        pois.append(POI(x, y, z, label))
                    except (
                        ValueError,
                        IndexError,
                    ):
                        continue

    log.info("[NAV] Loaded zone map %s: %d segments, %d POIs", path.name, len(segments), len(pois))
    return ZoneMap(segments, pois)
