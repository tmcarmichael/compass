"""Camp/zone spatial configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.types import CampType, DangerZone, Point
from nav.geometry import point_to_polyline
from perception.state import GameState


@dataclass(slots=True, kw_only=True)
class CampConfig:
    """Spatial anchors: where to hunt, where to flee, what's dangerous."""

    camp_pos: Point = Point(0.0, 0.0, 0.0)
    roam_radius: float = 250.0
    guard_pos: Point = Point(0.0, 0.0, 0.0)
    flee_pos: Point = Point(0.0, 0.0, 0.0)
    flee_waypoints: list[Point] = field(default_factory=list)
    hunt_min_dist: float = 50.0
    hunt_max_dist: float = 300.0
    danger_points: list[DangerZone] = field(default_factory=list)
    base_roam_radius: float = 0.0  # original roam_radius before tuning adjustments
    # Zone boundaries -- agent will not wander past these
    bounds_x_min: float | None = None
    bounds_x_max: float | None = None
    bounds_y_min: float | None = None
    bounds_y_max: float | None = None

    # -- LINEAR camp fields --
    camp_type: str = CampType.CIRCULAR
    patrol_waypoints: list[Point] = field(default_factory=list)
    corridor_width: float = 200.0

    def distance_to_camp(self, state: GameState) -> float:
        """Distance to camp area. Returns 0 if inside the camp zone."""
        if self.camp_type == CampType.LINEAR and len(self.patrol_waypoints) >= 2:
            dist: float = point_to_polyline(state.pos, self.patrol_waypoints)[0]
            return max(0.0, dist - self.corridor_width)
        # CIRCULAR: bounds-aware
        if self._inside_bounds(state.pos):
            return 0.0
        return state.pos.dist_2d(self.camp_pos)

    def effective_camp_distance(self, pos: Point) -> float:
        """Distance from camp area for scoring. Returns 0 if inside."""
        if self.camp_type == CampType.LINEAR and len(self.patrol_waypoints) >= 2:
            dist: float = point_to_polyline(pos, self.patrol_waypoints)[0]
            return max(0.0, dist - self.corridor_width)
        # CIRCULAR: bounds-aware
        if self._inside_bounds(pos):
            return 0.0
        return pos.dist_2d(self.camp_pos)

    def patrol_position(self, pos: Point) -> float:
        """Fractional position along the patrol path (0.0 to 1.0).

        Only meaningful for LINEAR camps. Returns 0.0 for CIRCULAR.
        """
        if self.camp_type != CampType.LINEAR or len(self.patrol_waypoints) < 2:
            return 0.0
        frac: float = point_to_polyline(pos, self.patrol_waypoints)[4]
        return frac

    def nearest_point_on_path(self, pos: Point) -> Point:
        """Closest point on the patrol polyline. For camp-return navigation.

        For CIRCULAR camps, returns camp center.
        """
        if self.camp_type != CampType.LINEAR or len(self.patrol_waypoints) < 2:
            return self.camp_pos
        _, nx, ny, _, _ = point_to_polyline(pos, self.patrol_waypoints)
        return Point(nx, ny, 0.0)

    def point_along_path(self, path_t: float) -> Point:
        """Get the (x, y) position at fractional path_t along the polyline.

        path_t is clamped to [0, 1]. Used by wander to pick patrol targets.
        """
        wps = self.patrol_waypoints
        if len(wps) < 2:
            return self.camp_pos
        path_t = max(0.0, min(1.0, path_t))

        # Compute segment lengths
        seg_lengths: list[float] = []
        for i in range(len(wps) - 1):
            seg_lengths.append(math.hypot(wps[i + 1].x - wps[i].x, wps[i + 1].y - wps[i].y))
        total = sum(seg_lengths)
        if total < 1e-10:
            return wps[0]

        target_dist = path_t * total
        accumulated = 0.0
        for i, seg_len in enumerate(seg_lengths):
            if accumulated + seg_len >= target_dist or i == len(seg_lengths) - 1:
                # Interpolate within this segment
                remain = target_dist - accumulated
                t = remain / seg_len if seg_len > 1e-10 else 0.0
                t = max(0.0, min(1.0, t))
                ax, ay, az = wps[i]
                bx, by, bz = wps[i + 1]
                return Point(ax + t * (bx - ax), ay + t * (by - ay), az + t * (bz - az))
            accumulated += seg_len
        return wps[-1]

    def path_total_length(self) -> float:
        """Total length of the patrol polyline."""
        wps = self.patrol_waypoints
        total = 0.0
        for i in range(len(wps) - 1):
            total += math.hypot(wps[i + 1].x - wps[i].x, wps[i + 1].y - wps[i].y)
        return total

    def _inside_bounds(self, pos: Point) -> bool:
        """True if position is inside the configured bounds rectangle."""
        has_bounds = (
            self.bounds_x_min is not None
            or self.bounds_x_max is not None
            or self.bounds_y_min is not None
            or self.bounds_y_max is not None
        )
        if not has_bounds:
            return False
        if self.bounds_x_min is not None and pos.x < self.bounds_x_min:
            return False
        if self.bounds_x_max is not None and pos.x > self.bounds_x_max:
            return False
        if self.bounds_y_min is not None and pos.y < self.bounds_y_min:
            return False
        if self.bounds_y_max is not None and pos.y > self.bounds_y_max:
            return False
        return True
