"""Patrol detection: geometric cycle detection for NPC movement patterns.

Detects patrolling npcs by tracking position traces over a 60-second window
and looking for return-to-origin cycles. Extracted from world_model.py to
isolate the geometric analysis from core spawn tracking.

Usage:
    # PatrolMixin is mixed into _MobHistory in world_model.py
    # patrol_safe_window() is called from WorldModel
"""

import math
from collections import deque

from core.types import Point

# -- Constants ---------------------------------------------------------------

PATROL_RETURN_DIST = 50.0  # npc returned within this distance = patrol
PATROL_TRACE_WINDOW = 60.0  # seconds of position trace to keep for patrol


# -- Patrol Mixin for _MobHistory -------------------------------------------


class PatrolMixin:
    """Mixin that extra_npcs patrol detection to _MobHistory.

    Expects the host class to have:
        patrol_trace: list | None
        patrol_period: float
        _patrol_checked: float
    """

    patrol_trace: deque[tuple[float, float, float]] | None
    patrol_period: float
    _patrol_checked: float

    def _update_patrol_trace(self, t: float, pos: Point) -> None:
        """Add a position sample to the patrol trace and run detection.

        Called from _MobHistory.add() each tick. Sparse-samples (1 per 2s)
        over a 60s sliding window. Runs cycle detection every 10s.
        """
        if self.patrol_trace is None:
            self.patrol_trace = deque()
        if not self.patrol_trace or (t - self.patrol_trace[-1][0]) >= 2.0:
            self.patrol_trace.append((t, pos.x, pos.y))
        # Trim to 60s window
        trace_cutoff = t - PATROL_TRACE_WINDOW
        while self.patrol_trace and self.patrol_trace[0][0] < trace_cutoff:
            self.patrol_trace.popleft()

        # Run patrol detection every 10s
        if t - self._patrol_checked >= 10.0:
            self._detect_patrol()
            self._patrol_checked = t

    def _detect_patrol(self) -> None:
        """Detect patrol pattern: npc moved away then returned to a prior position.

        Requires the npc reached at least 60u from the old position at some
        intermediate point -- prevents stationary npcs from false-triggering.
        """
        trace = self.patrol_trace
        if not trace or len(trace) < 5:
            return
        recent_t, recent_x, recent_y = trace[-1]
        for i in range(len(trace) - 3):
            old_t, old_x, old_y = trace[i]
            dt = recent_t - old_t
            if dt < 10.0:
                continue  # too recent, not a full cycle
            dx = recent_x - old_x
            dy = recent_y - old_y
            return_dist = math.sqrt(dx * dx + dy * dy)
            if return_dist < PATROL_RETURN_DIST:
                # Verify the npc actually moved away (>= 60u from start) at some point
                max_displacement = 0.0
                for j in range(i + 1, len(trace) - 1):
                    _, mx, my = trace[j]
                    md = math.sqrt((mx - old_x) ** 2 + (my - old_y) ** 2)
                    if md > max_displacement:
                        max_displacement = md
                if max_displacement >= 60.0:
                    self.patrol_period = dt
                    return
        self.patrol_period = 0.0


# -- Patrol-aware queries (called on WorldModel) ----------------------------


def patrol_safe_window(
    profiles: list,
    trackers: dict,
    target_pos: Point,
    fight_duration: float,
    threat_radius: float = 80.0,
) -> float:
    """Seconds until a patrolling threat enters threat range of target.

    Returns float('inf') if no patrol will arrive during fight_duration.
    Returns 0.0 if a patrol is already within threat range.

    Used by acquire/pull to avoid pulling when a patrol is about to return.

    Args:
        profiles: list of MobProfile from WorldModel._profiles
        trackers: dict of spawn_id -> _MobHistory from WorldModel._trackers
        target_pos: position of the fight location
        fight_duration: expected fight duration in seconds
        threat_radius: radius within which a patrol is considered a threat
    """
    min_arrival = float("inf")
    for p in profiles:
        if not p.is_patrolling or not p.is_threat:
            continue
        # Current distance to target
        d = p.spawn.pos.dist_to(target_pos)
        if d < threat_radius:
            return 0.0  # already in range

        # Predict position along patrol using current velocity
        tracker = trackers.get(p.spawn.spawn_id)
        if not tracker:
            continue
        vx, vy, _vz = tracker.velocity(p.spawn)
        spd = math.sqrt(vx * vx + vy * vy)
        if spd < 0.5:
            continue  # stationary right now

        # Simple linear prediction: time to reach threat_radius
        # Distance to target minus threat radius, divided by speed
        approach_dist = max(0, d - threat_radius)
        eta = approach_dist / spd
        if eta < fight_duration:
            min_arrival = min(min_arrival, eta)

    return min_arrival


def patrolling_threats(profiles: list) -> list:
    """All patrolling threat npcs currently tracked.

    Args:
        profiles: list of MobProfile from WorldModel._profiles
    """
    return [p for p in profiles if p.is_patrolling and p.is_threat]
