"""Travel routine: navigate to a destination, optionally crossing zone boundaries.

Supports both intra-zone travel (A* pathfind to a point) and inter-zone
travel (pathfind to zoneline, walk into it, let brain_runner handle the
zone transition).

Phases:
  PATHFIND     -  compute A* path to target (or zoneline if crossing zones)
  WALK         -  follow waypoints via move_to_point
  ZONE_CROSS   -  walk into zoneline (non-blocking, brain_runner detects zone)
  DONE         -  arrived at destination

Zone crossing is non-blocking. Brain_runner detects the transition via
engine_zone_id / game_mode / log message and calls enter_zoning() which
deactivates this routine. PlanTravelRoutine detects the zone change on
re-entry and advances to the next hop.

Travel planning (TravelLeg, plan_travel_legs, etc.) lives in
compass.nav.travel_planner and is re-exported here for backward compatibility.
"""

from __future__ import annotations

import enum
import logging
import time
from typing import TYPE_CHECKING, override

from core.types import FailureCategory, Point, ReadStateFn, TravelMode, TravelWaypoint
from nav.movement import get_terrain, move_to_point
from nav.travel_planner import (
    TravelLeg,
    find_tunnel_route,
    parse_tunnel_routes,
    plan_travel_legs,
    subdivide_waypoints,
)
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus

# Re-export planning symbols for backward compatibility
__all__ = [
    "TravelLeg",
    "find_tunnel_route",
    "parse_tunnel_routes",
    "plan_travel_legs",
    "subdivide_waypoints",
    "TravelRoutine",
    "MultiLegTravelRoutine",
    "PlanTravelRoutine",
]

if TYPE_CHECKING:
    from brain.context import AgentContext

log = logging.getLogger(__name__)


class _Phase(enum.IntEnum):
    PATHFIND = 0
    WALK = 1
    ZONE_CROSS = 2
    DONE = 3


class TravelRoutine(RoutineBase):
    """Navigate to a game-coordinate destination with optional zone crossing.

    For intra-zone travel: pathfinds and walks there.
    For inter-zone travel: walks to zoneline, crosses, and signals completion
    so the caller can reload zone data and create a new TravelRoutine for
    the next zone in the route.
    """

    def __init__(
        self,
        target_x: float,
        target_y: float,
        read_state_fn: ReadStateFn,
        zoneline: bool = False,
        manual_waypoints: list[TravelWaypoint] | None = None,
    ) -> None:
        """
        Args:
            target_x, target_y: Destination in game coordinates.
            read_state_fn: Callable returning GameState.
            zoneline: If True, target is a zoneline  -  walk into it and
                      wait for zone change.
            manual_waypoints: If provided, use these instead of A*.
                              For tunnel/cave routes where A* fails.
        """
        self._target_x = target_x
        self._target_y = target_y
        self._read_state_fn = read_state_fn
        self._zoneline = zoneline
        self._manual_waypoints = manual_waypoints

        self._phase = _Phase.PATHFIND
        self._waypoints: list[Point] = []
        self._wp_idx = 0
        self._zone_cross_start = 0.0
        self._zoned = False  # True once zone change detected
        self._pre_zone_id = 0  # zone_id before crossing

    @property
    def zoned(self) -> bool:
        """True if we successfully crossed into a new zone."""
        return self._zoned

    @override
    def enter(self, state: GameState) -> None:
        self._phase = _Phase.PATHFIND
        self._pre_zone_id = state.zone_id
        dist = state.pos.dist_to(Point(self._target_x, self._target_y, 0.0))
        log.info(
            "[TRAVEL] Travel: target (%.0f,%.0f) dist=%.0f zoneline=%s zone_id=%d",
            self._target_x,
            self._target_y,
            dist,
            self._zoneline,
            state.zone_id,
        )

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        if self._phase == _Phase.PATHFIND:
            return self._tick_pathfind(state)
        elif self._phase == _Phase.WALK:
            return self._tick_walk(state)
        elif self._phase == _Phase.ZONE_CROSS:
            return self._tick_zone_cross(state)
        return RoutineStatus.SUCCESS

    @override
    def exit(self, state: GameState) -> None:
        from motor.actions import move_forward_stop

        move_forward_stop()  # safety: release forward key if held during zone cross
        log.info("[TRAVEL] Travel: exit phase=%s zoned=%s", self._phase.name, self._zoned)

    # -- Phase implementations ----------------------------------------

    def _tick_pathfind(self, state: GameState) -> RoutineStatus:
        """Compute path using manual waypoints, A* terrain, or direct walk."""
        if self._manual_waypoints:
            # Manual route (tunnel/cave)  -  extract (x, y) from TravelWaypoints
            self._waypoints = [Point(wp[0], wp[1], 0.0) for wp in self._manual_waypoints]
            log.info("[TRAVEL] Travel: using %d manual waypoints (tunnel route)", len(self._waypoints))
        else:
            terrain = get_terrain()
            raw_wps = []

            if terrain:
                path = terrain.find_path(state.x, state.y, self._target_x, self._target_y)
                if path and len(path) > 1:
                    raw_wps = path[1:]
                    log.info("[TRAVEL] Travel: A* path with %d waypoints", len(raw_wps))
                else:
                    raw_wps = [Point(self._target_x, self._target_y, 0.0)]
                    log.info("[TRAVEL] Travel: no A* path, walking direct")
            else:
                raw_wps = [Point(self._target_x, self._target_y, 0.0)]

            # Subdivide long segments for reliable move_to_point
            self._waypoints = subdivide_waypoints(state.x, state.y, raw_wps)
            log.info(
                "[TRAVEL] Travel: %d raw -> %d waypoints after subdivision",
                len(raw_wps),
                len(self._waypoints),
            )

        self._wp_idx = 0
        self._phase = _Phase.WALK
        return RoutineStatus.RUNNING

    def _tick_walk(self, state: GameState) -> RoutineStatus:
        """Follow waypoints sequentially."""
        if self._wp_idx >= len(self._waypoints):
            if self._zoneline:
                self._phase = _Phase.ZONE_CROSS
                self._zone_cross_start = time.time()
                log.info("[TRAVEL] Travel: reached zoneline, crossing...")
                return RoutineStatus.RUNNING
            else:
                self._phase = _Phase.DONE
                log.info("[TRAVEL] Travel: arrived at destination")
                return RoutineStatus.SUCCESS

        wp = self._waypoints[self._wp_idx]
        is_final = self._wp_idx == len(self._waypoints) - 1
        tol = 12.0 if is_final else 18.0

        arrived = move_to_point(
            wp.x,
            wp.y,
            self._read_state_fn,
            arrival_tolerance=tol,
            timeout=30.0,
        )

        if arrived:
            self._wp_idx += 1
        else:
            # Movement failed  -  skip to next waypoint or abort
            self._wp_idx += 1
            if self._wp_idx >= len(self._waypoints):
                log.warning("[TRAVEL] Travel: failed to reach final waypoint")
                self.failure_reason = "waypoint_unreachable"
                self.failure_category = FailureCategory.EXECUTION
                return RoutineStatus.FAILURE

        return RoutineStatus.RUNNING

    def _tick_zone_cross(self, state: GameState) -> RoutineStatus:
        """Walk into the zoneline. Non-blocking.

        Brain_runner detects the zone transition (via engine_zone_id,
        game_mode, or log message) and calls enter_zoning() which
        deactivates this routine. PlanTravelRoutine detects the zone
        change on re-entry and advances the hop.
        """
        elapsed = time.time() - self._zone_cross_start

        if elapsed > 20.0:
            from motor.actions import move_forward_stop

            move_forward_stop()
            log.warning("[TRAVEL] Travel: zone cross timeout after %.0fs", elapsed)
            return RoutineStatus.FAILURE

        # Walk toward the zoneline point
        dist = state.pos.dist_to(Point(self._target_x, self._target_y, 0.0))
        if dist > 5.0:
            move_to_point(
                self._target_x,
                self._target_y,
                self._read_state_fn,
                arrival_tolerance=3.0,
                timeout=5.0,
            )
        else:
            # At the zoneline -- walk forward to trigger crossing
            from motor.actions import move_forward_start

            move_forward_start()

        return RoutineStatus.RUNNING


class MultiLegTravelRoutine(RoutineBase):
    """Execute a multi-leg travel plan (A* + tunnel route chains).

    Takes a list of TravelLeg and executes them sequentially.
    Each leg creates a TravelRoutine (pathfind or manual waypoints).
    """

    def __init__(self, legs: list[TravelLeg], read_state_fn: ReadStateFn) -> None:
        self._legs = legs
        self._read_state_fn = read_state_fn
        self._leg_idx = 0
        self._inner: TravelRoutine | None = None

    def _start_leg(self, state: GameState) -> None:
        if self._leg_idx >= len(self._legs):
            return
        leg = self._legs[self._leg_idx]
        manual = list(leg.waypoints) if leg.mode == TravelMode.MANUAL and leg.waypoints else None
        self._inner = TravelRoutine(leg.target_x, leg.target_y, self._read_state_fn, manual_waypoints=manual)
        self._inner.enter(state)
        log.info(
            "[TRAVEL] MultiLeg: leg %d/%d mode=%s target=(%.0f,%.0f)",
            self._leg_idx + 1,
            len(self._legs),
            leg.mode,
            leg.target_x,
            leg.target_y,
        )

    @override
    def enter(self, state: GameState) -> None:
        self._leg_idx = 0
        self._start_leg(state)

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        if not self._inner:
            return RoutineStatus.SUCCESS

        result = self._inner.tick(state)

        if result == RoutineStatus.SUCCESS:
            self._leg_idx += 1
            if self._leg_idx >= len(self._legs):
                log.info("[TRAVEL] MultiLeg: all %d legs complete", len(self._legs))
                return RoutineStatus.SUCCESS
            self._start_leg(state)
            return RoutineStatus.RUNNING

        if result == RoutineStatus.FAILURE:
            log.warning("[TRAVEL] MultiLeg: leg %d/%d failed", self._leg_idx + 1, len(self._legs))
            return RoutineStatus.FAILURE

        return RoutineStatus.RUNNING

    @override
    @property
    def locked(self) -> bool:
        return self._inner is not None and self._inner.locked

    @override
    def exit(self, state: GameState) -> None:
        if self._inner:
            self._inner.exit(state)
        self._inner = None


class PlanTravelRoutine(RoutineBase):
    """Travel routine driven by ctx.plan.travel route.

    Reads the current hop from ctx.plan.travel.route[hop_index],
    creates a TravelRoutine for it, and delegates. When the hop
    completes (or brain_runner detects a zone transition and re-enters
    us), advances hop_index. When all hops done, clears the plan.

    Zone crossing is non-blocking: the inner TravelRoutine walks into
    the zoneline, brain_runner detects the transition and deactivates
    us, then on re-entry we detect the zone_id change and advance.
    """

    def __init__(self, ctx: AgentContext | None = None, read_state_fn: ReadStateFn | None = None) -> None:
        self._ctx = ctx
        self._read_state_fn = read_state_fn
        self._inner: TravelRoutine | None = None

    @override
    def enter(self, state: GameState) -> None:
        assert self._ctx is not None
        route = self._ctx.plan.travel.route
        hop = self._ctx.plan.travel.hop_index

        # Detect zone change from brain_runner's zone transition handling:
        # if zone_id differs from pre-hop, the previous hop succeeded.
        pre_hop_zone = self._ctx.plan.travel.pre_hop_zone_id
        if pre_hop_zone > 0 and state.zone_id != pre_hop_zone:
            hop += 1
            self._ctx.plan.travel.hop_index = hop
            log.info(
                "[TRAVEL] PlanTravel: zone changed %d -> %d -- hop advanced to %d/%d",
                pre_hop_zone,
                state.zone_id,
                hop,
                len(route),
            )

        if hop >= len(route):
            dest = self._ctx.plan.travel.destination or "?"
            log.info("[TRAVEL] PlanTravel: all hops complete -- arrived at %s!", dest)
            self._ctx.plan.active = None
            self._inner = None
            return

        conn = route[hop]
        # Store zone_id before this hop so we can detect zone change on re-entry
        self._ctx.plan.travel.pre_hop_zone_id = state.zone_id
        log.info(
            "[TRAVEL] PlanTravel: hop %d/%d  -  %s to %s via (%.0f,%.0f)",
            hop + 1,
            len(route),
            conn.from_zone,
            conn.to_zone,
            conn.zoneline_x,
            conn.zoneline_y,
        )
        assert self._read_state_fn is not None
        self._inner = TravelRoutine(conn.zoneline_x, conn.zoneline_y, self._read_state_fn, zoneline=True)
        self._inner.enter(state)

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        if not self._inner:
            # All hops done (plan cleared in enter)
            return RoutineStatus.SUCCESS

        result = self._inner.tick(state)
        assert self._ctx is not None

        if result == RoutineStatus.SUCCESS:
            # Hop completed via normal route (intra-zone)
            route = self._ctx.plan.travel.route
            hop = self._ctx.plan.travel.hop_index
            self._ctx.plan.travel.hop_index = hop + 1

            if hop + 1 >= len(route):
                dest = self._ctx.plan.travel.destination or "?"
                log.info("[TRAVEL] PlanTravel: arrived at %s!", dest)
                self._ctx.plan.active = None
                return RoutineStatus.SUCCESS

            # More hops -- will re-enter on next brain tick
            self._inner = None
            return RoutineStatus.SUCCESS

        if result == RoutineStatus.FAILURE:
            log.warning("[TRAVEL] PlanTravel: hop failed, aborting travel")
            self._ctx.plan.active = None
            return RoutineStatus.FAILURE

        return RoutineStatus.RUNNING

    @override
    def exit(self, state: GameState) -> None:
        if self._inner:
            self._inner.exit(state)
        self._inner = None
