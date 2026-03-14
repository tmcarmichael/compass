"""Pure math utilities for EQ's coordinate and heading system.

EQ Specifics:
- Heading: 0-512 float (512 = full circle, NOT 360 degrees)
- Y axis: Inverted vs typical convention (Y increases going south)
- Heading 0: South (derived from heading_to's atan2(-dx, dy))
- Coordinate system: /loc reports Y, X, Z (not X, Y, Z)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.types import Point

HEADING_MAX = 512.0
HEADING_HALF = 256.0


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance in the XY plane."""
    return math.hypot(x2 - x1, y2 - y1)


def normalize_heading(heading: float) -> float:
    """Normalize a heading to [0, 512)."""
    return heading % HEADING_MAX


def angle_diff(current: float, target: float) -> float:
    """Signed shortest angular difference from current to target heading.

    Returns a value in (-256, 256].
    Positive = need to increase heading = turn LEFT (CCW).
    Negative = need to decrease heading = turn RIGHT (CW).
    Verified empirically: Shift+A (left) increases heading, Shift+D (right) decreases.
    """
    diff = normalize_heading(target - current)
    if diff > HEADING_HALF:
        diff -= HEADING_MAX
    return diff


def point_to_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> tuple[float, float, float, float]:
    """Closest point on segment AB to point P.

    Returns (distance, nearest_x, nearest_y, t) where t in [0,1] is the
    fractional position along the segment (0=A, 1=B).
    """
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-10:
        # Degenerate segment (A == B)
        return (math.hypot(px - ax, py - ay), ax, ay, 0.0)
    # Project P onto line AB, clamped to [0, 1]
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    nx = ax + t * dx
    ny = ay + t * dy
    return (math.hypot(px - nx, py - ny), nx, ny, t)


def point_to_polyline(
    px: float,
    py: float,
    waypoints: Sequence[Point],
) -> tuple[float, float, float, int, float]:
    """Closest point on a polyline to point P (XY projection).

    Returns (distance, nearest_x, nearest_y, segment_idx, path_t) where
    path_t in [0.0, 1.0] is the fractional position along the total path
    length (0.0 = first waypoint, 1.0 = last waypoint).
    """
    if len(waypoints) < 2:
        if waypoints:
            w = waypoints[0]
            return (math.hypot(px - w.x, py - w.y), w.x, w.y, 0, 0.0)
        return (0.0, px, py, 0, 0.0)

    # Pre-compute segment lengths for path_t calculation
    seg_lengths: list[float] = []
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        seg_lengths.append(math.hypot(b.x - a.x, b.y - a.y))
    total_length = sum(seg_lengths)

    best_dist = float("inf")
    best_nx, best_ny = px, py
    best_seg = 0
    best_path_t = 0.0

    length_before = 0.0
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        dist, nx, ny, t = point_to_segment(px, py, a.x, a.y, b.x, b.y)
        if dist < best_dist:
            best_dist = dist
            best_nx, best_ny = nx, ny
            best_seg = i
            if total_length > 1e-10:
                best_path_t = (length_before + t * seg_lengths[i]) / total_length
            else:
                best_path_t = 0.0
        length_before += seg_lengths[i]

    return (best_dist, best_nx, best_ny, best_seg, best_path_t)


def heading_to(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    """Calculate the EQ heading from one point to another.

    Uses atan2 and converts to EQ's 0-512 heading system.
    Heading 0 = North (+Y direction). 128=West, 256=South, 384=East.
    Heading increases CCW (left turn).
    atan2(dx, dy) gives angle from +Y axis  -  matches heading 0 = +Y.
    """
    dx = to_x - from_x
    dy = to_y - from_y

    angle_rad = math.atan2(dx, dy)
    heading = (angle_rad / (2 * math.pi)) * HEADING_MAX
    return normalize_heading(heading)
