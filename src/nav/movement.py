"""Closed-loop point-to-point movement with fast stuck recovery and obstacle avoidance.

Decomposed into three classes:
- StuckRecovery: owns escalation logic (strafe, backup, turn decisions)
- HeadingController: owns face_heading with tolerance
- MovementController: composes the above, owns the main movement loop

Module-level functions (move_to_point, etc.) are thin wrappers
that delegate to a module-level _controller singleton.
"""

from __future__ import annotations

import logging
import math
import random
import threading as _threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.types import ReadStateFn
    from nav.map_data import ZoneMap
    from nav.terrain.heightmap import ZoneTerrain
    from perception.state import GameState

from core.features import flags
from core.types import Point
from motor.actions import (
    _action_down,
    _action_up,
    face_heading,
    is_sitting,
    jittered_sleep,
    move_backward_start,
    move_backward_stop,
    move_forward_start,
    move_forward_stop,
    stand,
)
from nav.geometry import angle_diff, distance_2d, heading_to, normalize_heading
from nav.stuck import StuckDetector

log = logging.getLogger(__name__)

STUCK_AVOIDANCE_RADIUS = 20.0

# Thread-safe cancel signal  -  set by FLEE or orchestrator to interrupt move_to_point.
# Must remain module-level because it is set from outside (FLEE, orchestrator stop).
_cancel_event = _threading.Event()


# ---------------------------------------------------------------------------
# RecoveryAction  -  describes what unstick maneuver to perform
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecoveryAction:
    """Describes a stuck-recovery maneuver."""

    action_type: str  # "strafe_left", "strafe_right", "backup", "turn"
    duration: float  # seconds for the primary action
    # For compound actions (backup+strafe+turn), extra fields:
    strafe_dir: int = 0  # -1=left, 1=right, 0=none
    strafe_duration: float = 0.0
    turn_amount: float = 0.0  # heading delta (signed)


# ---------------------------------------------------------------------------
# StuckRecovery  -  owns escalation logic for getting unstuck
# ---------------------------------------------------------------------------


class StuckRecovery:
    """Escalating recovery when movement is stuck.

    Tracks attempt count and first strafe direction. Returns RecoveryAction
    describing what to do, or None if not stuck. The caller is responsible
    for executing the action via motor commands.

    Escalation:
      1: Strafe first_dir
      2: Strafe opposite of first_dir
      3: Backup + strafe + turn
      4+: Bigger backup + random strafe + random turn
    """

    def __init__(self) -> None:
        self._attempt: int = 0
        self._first_dir: int = random.choice([-1, 1])
        self.logger = logging.getLogger(__name__ + ".StuckRecovery")

    @property
    def attempt(self) -> int:
        return self._attempt

    def reset(self) -> None:
        """Reset attempt counter and pick a new random first direction."""
        self._attempt = 0
        self._first_dir = random.choice([-1, 1])

    def next_recovery(self) -> RecoveryAction:
        """Return the next escalating recovery action.

        Increments the internal attempt counter.
        """
        self._attempt += 1
        attempt = self._attempt
        first_dir = self._first_dir

        if attempt <= 1:
            # Strafe one direction + small turn to route around obstacle
            strafe_time = random.uniform(0.8, 1.3)
            turn_amount = random.uniform(20, 40) * first_dir
            direction = "left" if first_dir < 0 else "right"
            action_type = "strafe_left" if first_dir < 0 else "strafe_right"
            self.logger.info(
                "[POSITION] Recovery %d: strafe %s %.1fs + turn %.0f",
                attempt,
                direction,
                strafe_time,
                turn_amount,
            )
            return RecoveryAction(
                action_type=action_type,
                duration=strafe_time,
                turn_amount=turn_amount,
            )

        elif attempt == 2:
            # Try the other direction + bigger turn
            opp = -first_dir
            strafe_time = random.uniform(0.8, 1.3)
            turn_amount = random.uniform(30, 50) * opp
            direction = "left" if opp < 0 else "right"
            action_type = "strafe_left" if opp < 0 else "strafe_right"
            self.logger.info(
                "[POSITION] Recovery %d: strafe %s %.1fs + turn %.0f",
                attempt,
                direction,
                strafe_time,
                turn_amount,
            )
            return RecoveryAction(
                action_type=action_type,
                duration=strafe_time,
                turn_amount=turn_amount,
            )

        elif attempt == 3:
            # Back up + strafe to get around obstacle
            backup_time = random.uniform(0.7, 1.2)
            strafe_time = random.uniform(0.8, 1.3)
            turn_amount = random.uniform(30, 60) * first_dir
            self.logger.info(
                "[POSITION] Recovery %d: backup %.1fs + strafe + turn %.0f", attempt, backup_time, turn_amount
            )
            return RecoveryAction(
                action_type="backup",
                duration=backup_time,
                strafe_dir=first_dir,
                strafe_duration=strafe_time,
                turn_amount=turn_amount,
            )

        else:
            # Bigger backup + strafe + turn to really get around
            backup_time = random.uniform(0.8, 1.5)
            strafe_time = random.uniform(0.8, 1.5)
            rand_dir = random.choice([-1, 1])
            turn_amount = random.uniform(60, 120) * random.choice([-1, 1])
            self.logger.info(
                "[POSITION] Recovery %d: backup %.1fs + strafe + turn %.0f", attempt, backup_time, turn_amount
            )
            return RecoveryAction(
                action_type="backup",
                duration=backup_time,
                strafe_dir=rand_dir,
                strafe_duration=strafe_time,
                turn_amount=turn_amount,
            )


# ---------------------------------------------------------------------------
# HeadingController  -  owns face_heading with tolerance
# ---------------------------------------------------------------------------


class HeadingController:
    """Computes desired heading and delegates to motor face_heading."""

    def __init__(self) -> None:
        self._last_desired: float = 0.0
        self.logger = logging.getLogger(__name__ + ".HeadingController")

    @property
    def last_desired(self) -> float:
        """The last heading passed to face_toward."""
        return self._last_desired

    def face_toward(
        self,
        target_x: float,
        target_y: float,
        current_x: float,
        current_y: float,
        current_heading: float,
        read_heading_fn: Callable[[], float] | None = None,
        tolerance: float = 5.0,
    ) -> bool:
        """Compute heading toward target and turn to face it.

        Args:
            target_x, target_y: destination coordinates.
            current_x, current_y: character's current position.
            current_heading: character's current heading (0-512).
            read_heading_fn: callable returning live heading (for closed-loop).
                If None, only computes desired heading without turning.
            tolerance: heading tolerance in EQ heading units.

        Returns:
            True if already facing (within tolerance), False if turn was needed.
        """
        desired = heading_to(Point(current_x, current_y, 0.0), Point(target_x, target_y, 0.0))
        self._last_desired = desired
        heading_error = abs(angle_diff(current_heading, desired))

        if heading_error <= tolerance:
            return True

        if read_heading_fn is not None:
            face_heading(desired, read_heading_fn, tolerance=tolerance)
        return False

    def is_facing(self, current_heading: float, tolerance: float = 20.0) -> bool:
        """Check if current heading is within tolerance of last desired heading."""
        facing: bool = abs(angle_diff(current_heading, self._last_desired)) <= tolerance
        return facing


# ---------------------------------------------------------------------------
# Helper: execute a RecoveryAction via motor commands
# ---------------------------------------------------------------------------


def _strafe(direction: int, duration: float) -> None:
    """Strafe left (direction=-1) or right (direction=1) for duration seconds.
    Checks cancel event to avoid ghost keys on stop."""
    action = "strafe_left" if direction < 0 else "strafe_right"
    _action_down(action)
    try:
        # Poll cancel in short bursts instead of one long sleep
        remaining = duration
        while remaining > 0:
            if _cancel_event.is_set():
                break
            step = min(remaining, 0.1)
            jittered_sleep(step)
            remaining -= step
    finally:
        _action_up(action)


def _execute_recovery(action: RecoveryAction, read_state_fn: ReadStateFn) -> None:
    """Execute a RecoveryAction using motor commands.

    For simple strafes, just strafe in the indicated direction.
    For backup actions, backup then strafe then turn.
    """
    move_forward_stop()
    jittered_sleep(0.05)

    if action.action_type in ("strafe_left", "strafe_right"):
        direction = -1 if action.action_type == "strafe_left" else 1
        _strafe(direction, action.duration)
        # Turn after strafe to avoid re-facing the same wall
        if action.turn_amount != 0 and not _cancel_event.is_set():
            state = read_state_fn()
            new_heading = normalize_heading(state.heading + action.turn_amount)
            face_heading(new_heading, lambda: read_state_fn().heading, tolerance=5.0)

    elif action.action_type == "backup":
        # Backup phase (cancel-aware)
        move_backward_start()
        remaining = action.duration
        while remaining > 0 and not _cancel_event.is_set():
            step = min(remaining, 0.1)
            jittered_sleep(step)
            remaining -= step
        move_backward_stop()
        if _cancel_event.is_set():
            return
        jittered_sleep(0.1)

        # Strafe phase
        if action.strafe_dir != 0 and action.strafe_duration > 0:
            _strafe(action.strafe_dir, action.strafe_duration)

        # Turn phase
        if action.turn_amount != 0:
            state = read_state_fn()
            new_heading = normalize_heading(state.heading + action.turn_amount)
            face_heading(new_heading, lambda: read_state_fn().heading, tolerance=5.0)


# ---------------------------------------------------------------------------
# MovementController  -  composes StuckRecovery + HeadingController
# ---------------------------------------------------------------------------


class MovementController:
    """Encapsulates movement state: zone map, terrain, stuck points, and cancel flag.

    Composes StuckRecovery for unstuck escalation and HeadingController for
    heading management. All previously module-level globals (stuck points,
    zone_map, terrain) are instance state.
    """

    def __init__(self, terrain: ZoneTerrain | None = None, zone_map: ZoneMap | None = None) -> None:
        self.zone_map = zone_map
        self.terrain = terrain
        self.cancel_requested = False

        # Stuck point memory  -  locations where the agent got stuck
        self._stuck_points: list[Point] = []
        self._stuck_lock = _threading.Lock()
        self._stuck_event_count: int = 0

        # Composed helpers
        self.heading = HeadingController()

        self.logger = logging.getLogger(__name__ + ".MovementController")

    # -- Stuck point management --

    def record_stuck_point(self, pos: Point) -> None:
        """Record a stuck location for future avoidance."""
        with self._stuck_lock:
            for sp in self._stuck_points:
                if abs(pos.x - sp.x) < STUCK_AVOIDANCE_RADIUS and abs(pos.y - sp.y) < STUCK_AVOIDANCE_RADIUS:
                    return
            self._stuck_points.append(pos)
            count = len(self._stuck_points)
        log.info(
            "[POSITION] Stuck point recorded: (%.0f, %.0f) -- total %d known stuck points",
            pos.x,
            pos.y,
            count,
        )

    def is_near_stuck_point(self, pos: Point) -> bool:
        """Check if a position is near a known stuck point (XY)."""
        for sp in self._stuck_points:
            if pos.dist_2d(sp) < STUCK_AVOIDANCE_RADIUS:
                return True
        return False

    def get_stuck_points(self) -> list[Point]:
        """Return all known stuck points (for persistence)."""
        with self._stuck_lock:
            return list(self._stuck_points)

    def load_stuck_points(self, points: list[Point]) -> None:
        """Load stuck points from persistence."""
        with self._stuck_lock:
            self._stuck_points.clear()
            self._stuck_points.extend(points)
        if points:
            log.info("[NAV] Loaded %d stuck points from memory", len(points))

    @property
    def stuck_event_count(self) -> int:
        return self._stuck_event_count

    # -- Main movement API --

    def move_to_point(
        self,
        target: Point,
        read_state_fn: ReadStateFn,
        arrival_tolerance: float = 15.0,
        heading_tolerance: float = 5.0,
        timeout: float = 30.0,
        max_stuck_recoveries: int = 6,
        check_fn: Callable[[], bool] | None = None,
    ) -> bool:
        """Move the character to a target point using closed-loop control.

        Features:
        - A* pathfinding when terrain data is available (multi-waypoint)
        - Fast stuck detection (~0.5s via position tracking)
        - Lightweight recovery (turn first, backup only if needed)
        - Map-based raycast obstacle avoidance (fallback if no terrain)
        - Cliff/water/lava detection via terrain heightmap
        - Cancel support for clean Stop button behavior
        - Optional check_fn: called each tick, returns True to stop early

        Returns:
            True if arrived, False if timed out, stuck, or cancelled.
        """
        # If terrain is available and distance is significant, try A* pathfinding
        if self.terrain:
            state = read_state_fn()
            dist = state.pos.dist_2d(target)
            if dist > 50:
                path = self.terrain.find_path(state.pos, target)
                if path and len(path) > 2:
                    # Follow A* waypoints sequentially (skip first = current pos)
                    for i, wp in enumerate(path[1:], 1):
                        is_final = i == len(path) - 1
                        tol = arrival_tolerance if is_final else arrival_tolerance * 1.5
                        ok = self._move_to_point_inner(
                            wp,
                            read_state_fn,
                            arrival_tolerance=tol,
                            heading_tolerance=heading_tolerance,
                            timeout=timeout,
                            max_stuck_recoveries=max_stuck_recoveries,
                            check_fn=check_fn,
                        )
                        if not ok:
                            return False
                    return True

        return self._move_to_point_inner(
            target,
            read_state_fn,
            arrival_tolerance=arrival_tolerance,
            heading_tolerance=heading_tolerance,
            timeout=timeout,
            max_stuck_recoveries=max_stuck_recoveries,
            check_fn=check_fn,
        )

    def _move_to_point_inner(
        self,
        target: Point,
        read_state_fn: ReadStateFn,
        arrival_tolerance: float = 15.0,
        heading_tolerance: float = 5.0,
        timeout: float = 30.0,
        max_stuck_recoveries: int = 6,
        check_fn: Callable[[], bool] | None = None,
    ) -> bool:
        """Inner movement loop  -  walk to a single target point."""
        stuck_detector = StuckDetector(check_seconds=1.0, min_distance=3.0)
        recovery = StuckRecovery()
        start_time = time.perf_counter()
        current_target_x = target.x
        current_target_y = target.y
        detour_active = False
        moving = False
        last_pos_log = 0.0
        pos_log_interval = 2.0  # log position every 2s during movement
        last_check_fn = 0.0  # throttle check_fn to ~1s intervals
        last_face_fail = 0.0  # cooldown after face_heading failure
        face_fail_cooldown = 5.0  # seconds to skip face_heading after it fails
        # Predictive detour state
        last_detour_x: float | None = None
        last_detour_y: float | None = None
        repeat_detour_count = 0

        try:
            while True:
                if self.cancel_requested:
                    state = read_state_fn()
                    log.info(
                        "[POSITION] move_to_point cancelled at (%.0f, %.0f) dist=%.0f to target",
                        state.x,
                        state.y,
                        state.pos.dist_2d(target),
                    )
                    return False

                # External check  -  caller can stop walk early (e.g. npc nearby)
                now_check = time.perf_counter()
                if check_fn and now_check - last_check_fn > 1.0:
                    last_check_fn = now_check
                    if check_fn():
                        return False

                elapsed = now_check - start_time
                if elapsed > timeout:
                    state = read_state_fn()
                    log.warning(
                        "[POSITION] move_to_point TIMEOUT %.1fs at (%.0f, %.0f) dist=%.0f",
                        timeout,
                        state.x,
                        state.y,
                        state.pos.dist_2d(target),
                    )
                    return False

                state = read_state_fn()
                dist_to_final = state.pos.dist_2d(target)

                # Periodic position logging during movement
                now = time.perf_counter()
                if moving and now - last_pos_log > pos_log_interval:
                    log.debug(
                        "[POSITION] Moving: pos=(%.0f,%.0f) dist=%.0f heading=%.0f t=%.1fs",
                        state.x,
                        state.y,
                        dist_to_final,
                        state.heading,
                        elapsed,
                    )
                    last_pos_log = now

                if dist_to_final <= arrival_tolerance:
                    log.info(
                        "[POSITION] Arrived at (%.0f, %.0f) dist=%.0f in %.1fs",
                        target.x,
                        target.y,
                        dist_to_final,
                        elapsed,
                    )
                    return True

                # If on a detour, check if we reached the detour waypoint
                if detour_active and self._detour_reached(
                    state, current_target_x, current_target_y, arrival_tolerance
                ):
                    current_target_x = target.x
                    current_target_y = target.y
                    detour_active = False

                # Sitting guard: if the character is sitting, movement keys
                # have no effect and stuck detection will never converge.
                if moving and state.speed_run == 0 and (state.is_sitting or is_sitting()):
                    self._force_stand_during_move(state, stuck_detector)
                    moving = False
                    continue

                # Stuck detection  -  fast response (speed-based + displacement)
                if moving and stuck_detector.check(state.pos, speed=state.speed_run):
                    stuck_result = self._handle_stuck(
                        state,
                        recovery,
                        read_state_fn,
                        stuck_detector,
                        target,
                        dist_to_final,
                        elapsed,
                        max_stuck_recoveries,
                    )
                    if stuck_result is False:
                        return False
                    moving = False
                    if detour_active:
                        current_target_x = target.x
                        current_target_y = target.y
                        detour_active = False
                    continue

                # Predictive obstacle scan: check terrain ahead and detour
                # before hitting the obstacle (validate-then-move pattern).
                if (
                    self.terrain
                    and not detour_active
                    and repeat_detour_count < 3
                    and flags.obstacle_avoidance
                ):
                    detour = self._compute_predictive_detour(
                        state,
                        current_target_x,
                        current_target_y,
                        target.x,
                        target.y,
                        last_detour_x,
                        last_detour_y,
                        repeat_detour_count,
                    )
                    if detour is not None:
                        current_target_x, current_target_y = detour[0], detour[1]
                        last_detour_x, last_detour_y = detour[0], detour[1]
                        repeat_detour_count = detour[2]
                        detour_active = True

                # Face toward target and start/continue moving
                desired = heading_to(state.pos, Point(current_target_x, current_target_y, 0.0))
                moving, last_face_fail = self._face_and_move(
                    desired,
                    read_state_fn,
                    heading_tolerance,
                    moving,
                    last_face_fail,
                    face_fail_cooldown,
                )

                jittered_sleep(0.15)
        finally:
            move_forward_stop()
            move_backward_stop()
            # Release strafe keys in case we were interrupted mid-recovery
            _action_up("strafe_left")
            _action_up("strafe_right")

    def _compute_predictive_detour(
        self,
        state: GameState,
        current_x: float,
        current_y: float,
        target_x: float,
        target_y: float,
        last_detour_x: float | None,
        last_detour_y: float | None,
        repeat_count: int,
    ) -> tuple[float, float, int] | None:
        """Scan terrain ahead and compute a detour waypoint to avoid obstacles.

        Returns (detour_x, detour_y, updated_repeat_count) or None.
        """
        # Predictive terrain scanning disabled (stuck recovery handles obstacles).
        # The call site handles None gracefully (no detour applied).
        return None

    @staticmethod
    def _face_and_move(
        desired: float,
        read_state_fn: ReadStateFn,
        heading_tolerance: float,
        moving: bool,
        last_face_fail: float,
        face_fail_cooldown: float,
    ) -> tuple[bool, float]:
        """Face desired heading and ensure forward movement. Returns (moving, last_face_fail)."""
        heading_error = abs(angle_diff(read_state_fn().heading, desired))
        face_on_cooldown = (time.perf_counter() - last_face_fail) < face_fail_cooldown

        if heading_error > 20.0:
            if face_on_cooldown:
                if not moving:
                    move_forward_start()
                    moving = True
            else:
                move_forward_stop()
                moving = False
                ok = face_heading(desired, lambda: read_state_fn().heading, tolerance=heading_tolerance)
                if not ok:
                    last_face_fail = time.perf_counter()
                    log.debug("[POSITION] face_heading failed, cooldown %.0fs", face_fail_cooldown)
                move_forward_start()
                moving = True
        elif not moving:
            if not face_on_cooldown:
                ok = face_heading(desired, lambda: read_state_fn().heading, tolerance=heading_tolerance)
                if not ok:
                    last_face_fail = time.perf_counter()
            move_forward_start()
            moving = True
        return moving, last_face_fail

    @staticmethod
    @staticmethod
    def _detour_reached(
        state: GameState,
        detour_x: float,
        detour_y: float,
        arrival_tolerance: float,
    ) -> bool:
        """Check if we reached the detour waypoint and should resume direct path."""
        dist = distance_2d(state.x, state.y, detour_x, detour_y)
        if dist <= arrival_tolerance * 1.5:
            log.debug("[POSITION] Reached detour waypoint, resuming direct path")
            return True
        return False

    @staticmethod
    def _force_stand_during_move(state: GameState, stuck_detector: StuckDetector) -> None:
        """Force-stand when character is sitting during movement."""
        log.warning(
            "[POSITION] move_to_point: character is sitting "
            "(stand_state=%d internal=%s) -- force-standing before move",
            state.stand_state,
            is_sitting(),
        )
        move_forward_stop()
        stand()
        jittered_sleep(0.4)
        stuck_detector.reset()

    def _handle_stuck(
        self,
        state: GameState,
        recovery: StuckRecovery,
        read_state_fn: ReadStateFn,
        stuck_detector: StuckDetector,
        target: Point,
        dist_to_final: float,
        elapsed: float,
        max_stuck_recoveries: int,
    ) -> bool | None:
        """Check stuck budget and execute recovery. Returns False to abort, None to continue."""
        if recovery.attempt >= max_stuck_recoveries:
            log.warning(
                "[POSITION] Stuck %d times, giving up on (%.1f, %.1f)",
                recovery.attempt + 1,
                target.x,
                target.y,
            )
            return False
        self._execute_stuck_recovery(
            state,
            recovery,
            read_state_fn,
            target,
            dist_to_final,
            elapsed,
            max_stuck_recoveries,
        )
        stuck_detector.reset()
        return None

    def _execute_stuck_recovery(
        self,
        state: GameState,
        recovery: StuckRecovery,
        read_state_fn: ReadStateFn,
        target: Point,
        dist_to_final: float,
        elapsed: float,
        max_stuck_recoveries: int,
    ) -> None:
        """Log stuck context, execute escalating recovery, log result."""
        desired = heading_to(state.pos, target)
        hdg_err = angle_diff(state.heading, desired)
        terrain_tag = ""
        if self.terrain:
            if self.terrain.is_obstacle(state.x, state.y):
                terrain_tag = " [OBSTACLE]"
            elif not self.terrain.is_walkable(state.x, state.y):
                terrain_tag = " [unwalkable]"
        log.info(
            "[POSITION] Stuck at (%.0f,%.0f)%s hdg=%.0f desired=%.0f "
            "err=%.0f speed=%.1f dist=%.0f "
            "goal=(%.0f,%.0f) elapsed=%.1fs (recovery %d/%d)",
            state.x,
            state.y,
            terrain_tag,
            state.heading,
            desired,
            hdg_err,
            state.speed_run,
            dist_to_final,
            target.x,
            target.y,
            elapsed,
            recovery.attempt + 1,
            max_stuck_recoveries,
        )
        pre_x, pre_y = state.x, state.y
        self._stuck_event_count += 1

        action = recovery.next_recovery()
        _execute_recovery(action, read_state_fn)

        post = read_state_fn()
        delta = distance_2d(pre_x, pre_y, post.x, post.y)
        log.info(
            "[POSITION] Post-recovery pos=(%.0f,%.0f) hdg=%.0f moved=%.0f dist_to_goal=%.0f",
            post.x,
            post.y,
            post.heading,
            delta,
            post.pos.dist_2d(target),
        )


# ---------------------------------------------------------------------------
# MovementPhase  -  non-blocking, tick-resumable movement
# ---------------------------------------------------------------------------


class MovementPhase:
    """Non-blocking point-to-point movement that yields every tick.

    Unlike the blocking ``move_to_point()`` which loops internally with
    ``jittered_sleep(0.15)``, this class is called once per brain tick
    via ``tick()`` and returns immediately. The caller (wander, travel)
    calls ``tick()`` each brain cycle and checks the result.

    Usage::

        phase = MovementPhase(target_x, target_y, read_state_fn)
        # Each brain tick:
        result = phase.tick()
        if result is not None:
            arrived = result  # True=arrived, False=failed/timeout
    """

    def __init__(
        self,
        target_x: float,
        target_y: float,
        read_state_fn: ReadStateFn,
        arrival_tolerance: float = 15.0,
        heading_tolerance: float = 5.0,
        timeout: float = 30.0,
        max_stuck_recoveries: int = 6,
    ) -> None:
        self._target_x = target_x
        self._target_y = target_y
        self._read_state_fn = read_state_fn
        self._arrival_tolerance = arrival_tolerance
        self._heading_tolerance = heading_tolerance
        self._timeout = timeout
        self._max_stuck_recoveries = max_stuck_recoveries

        self._start_time = time.perf_counter()
        self._stuck_detector = StuckDetector(check_seconds=1.0, min_distance=3.0)
        self._recovery = StuckRecovery()
        self._moving = False
        self._done = False
        self._result: bool | None = None
        # Recovery sub-phase: when stuck recovery is executing, we run it
        # in a blocking micro-step (single recovery action) then resume.
        # This keeps recovery atomic (0.5-2s) but yields between walks.
        self._recovery_pending = False
        # Face heading tracking
        self._last_face_fail = 0.0
        self._face_fail_cooldown = 5.0

    @property
    def done(self) -> bool:
        """True when movement is complete (arrived or failed)."""
        return self._done

    @property
    def arrived(self) -> bool:
        """True if movement completed successfully (reached target)."""
        return self._result is True

    def cancel(self) -> None:
        """Stop movement immediately."""
        if self._moving:
            move_forward_stop()
            self._moving = False
        self._done = True
        self._result = False

    def tick(self) -> bool | None:
        """Advance movement by one tick.

        Returns:
            None -- still moving, call again next tick
            True -- arrived at target
            False -- failed (timeout, max stuck recoveries, cancelled)
        """
        if self._done:
            return self._result

        # Cancel check
        if _cancel_event.is_set():
            self._finish(False, "cancelled")
            return False

        # Timeout check
        elapsed = time.perf_counter() - self._start_time
        if elapsed > self._timeout:
            state = self._read_state_fn()
            log.warning(
                "[POSITION] MovementPhase TIMEOUT %.1fs at (%.0f, %.0f) dist=%.0f",
                self._timeout,
                state.x,
                state.y,
                distance_2d(state.x, state.y, self._target_x, self._target_y),
            )
            self._finish(False, "timeout")
            return False

        state = self._read_state_fn()
        dist = distance_2d(state.x, state.y, self._target_x, self._target_y)

        # Arrival check
        if dist <= self._arrival_tolerance:
            log.info(
                "[POSITION] MovementPhase arrived at (%.0f, %.0f) dist=%.0f in %.1fs",
                self._target_x,
                self._target_y,
                dist,
                elapsed,
            )
            self._finish(True, "arrived")
            return True

        # Sitting guard
        if self._moving and state.speed_run == 0 and (state.is_sitting or is_sitting()):
            log.warning("[POSITION] MovementPhase: sitting -- force-standing")
            move_forward_stop()
            self._moving = False
            stand()
            self._stuck_detector.reset()
            return None  # yield, re-check next tick

        # Stuck detection (only while actively moving)
        if self._moving and self._stuck_detector.check(state.pos, speed=state.speed_run):
            if self._recovery.attempt >= self._max_stuck_recoveries:
                log.warning("[POSITION] MovementPhase stuck %d times, giving up", self._recovery.attempt + 1)
                self._finish(False, "stuck")
                return False

            desired = heading_to(state.pos, Point(self._target_x, self._target_y, 0.0))
            hdg_err = angle_diff(state.heading, desired)
            log.info(
                "[POSITION] MovementPhase stuck at (%.0f,%.0f) hdg=%.0f err=%.0f dist=%.0f (recovery %d/%d)",
                state.x,
                state.y,
                state.heading,
                hdg_err,
                dist,
                self._recovery.attempt + 1,
                self._max_stuck_recoveries,
            )

            # Execute recovery atomically (blocking 0.5-2s) then yield
            action = self._recovery.next_recovery()
            _execute_recovery(action, self._read_state_fn)
            self._stuck_detector.reset()
            self._moving = False
            return None  # yield after recovery, re-face next tick

        # Face toward target and start walking
        desired = heading_to(state.pos, Point(self._target_x, self._target_y, 0.0))
        self._face_and_walk(state, desired)

        # No sleep -- return immediately, brain will call us again next tick
        return None

    def _face_and_walk(self, state: GameState, desired: float) -> None:
        """Face desired heading and ensure forward movement is started."""
        heading_error = abs(angle_diff(state.heading, desired))
        face_on_cooldown = (time.perf_counter() - self._last_face_fail) < self._face_fail_cooldown

        if heading_error > 20.0:
            if face_on_cooldown:
                if not self._moving:
                    move_forward_start()
                    self._moving = True
            else:
                if self._moving:
                    move_forward_stop()
                    self._moving = False
                ok = face_heading(
                    desired, lambda: self._read_state_fn().heading, tolerance=self._heading_tolerance
                )
                if not ok:
                    self._last_face_fail = time.perf_counter()
                move_forward_start()
                self._moving = True
        elif not self._moving:
            if not face_on_cooldown:
                ok = face_heading(
                    desired, lambda: self._read_state_fn().heading, tolerance=self._heading_tolerance
                )
                if not ok:
                    self._last_face_fail = time.perf_counter()
            move_forward_start()
            self._moving = True

    def _finish(self, result: bool, reason: str) -> None:
        """Stop movement and record result."""
        if self._moving:
            move_forward_stop()
            self._moving = False
        move_backward_stop()
        _action_up("strafe_left")
        _action_up("strafe_right")
        self._done = True
        self._result = result
        log.debug("[POSITION] MovementPhase finished: %s (%s)", result, reason)


# ---------------------------------------------------------------------------
# Module-level singleton  -  backward-compatible API
# ---------------------------------------------------------------------------

_controller = MovementController()


def get_stuck_event_count() -> int:
    return _controller.stuck_event_count


def is_near_stuck_point(pos: Point) -> bool:
    """Check if a position is near a known stuck point."""
    return _controller.is_near_stuck_point(pos)


def get_stuck_points() -> list[Point]:
    """Return all known stuck points (for persistence)."""
    return _controller.get_stuck_points()


def load_stuck_points(points: list[Point]) -> None:
    """Load stuck points from persistence."""
    _controller.load_stuck_points(points)


def get_terrain() -> ZoneTerrain | None:
    """Return the current terrain heightmap (or None)."""
    return _controller.terrain


def set_terrain(terrain: ZoneTerrain | None) -> None:
    """Set terrain heightmap for A* pathfinding and hazard detection.

    Called by brain_lifecycle after each zone load/transition. Passing None
    clears terrain (disables A* pathfinding, falls back to stuck recovery).
    """
    _controller.terrain = terrain
    if terrain:
        log.info("[NAV] Movement: terrain loaded (%s)", terrain.stats.get("grid", "?"))
    else:
        log.info("[NAV] Movement: terrain cleared (no cache)")


def check_spell_los(
    player_x: float, player_y: float, player_z: float, target_x: float, target_y: float, target_z: float
) -> bool:
    """Check if terrain blocks line-of-sight for spellcasting.

    Returns True if LOS is clear, False if blocked by terrain.
    Returns True (assume clear) if no terrain data loaded.

    Eye-height offsets prevent rays from hugging the ground on downhill
    casts. Margin 5u tolerates rolling terrain undulations (3-5u amplitude).
    """
    terrain = _controller.terrain
    if not terrain:
        return True
    layer_z = terrain.get_z(player_x, player_y)
    if math.isnan(layer_z):
        return True
    CASTER_EYE_HEIGHT = 5.0
    MOB_TORSO_HEIGHT = 3.0
    clear: bool = terrain.check_los(
        player_x,
        player_y,
        player_z + CASTER_EYE_HEIGHT,
        target_x,
        target_y,
        target_z + MOB_TORSO_HEIGHT,
        margin=5.0,
    )
    return clear


def request_movement_cancel() -> None:
    """Request all in-progress movement to stop immediately (thread-safe)."""
    _cancel_event.set()
    _controller.cancel_requested = True


def clear_movement_cancel() -> None:
    """Clear the cancel flag (call at start of each brain tick)."""
    _cancel_event.clear()
    _controller.cancel_requested = False


def move_to_point(
    target: Point,
    read_state_fn: ReadStateFn,
    arrival_tolerance: float = 15.0,
    heading_tolerance: float = 5.0,
    timeout: float = 30.0,
    max_stuck_recoveries: int = 6,
    check_fn: Callable[[], bool] | None = None,
) -> bool:
    """Move the character to a target point using closed-loop control.

    Thin wrapper that delegates to the module-level MovementController singleton.
    """
    return _controller.move_to_point(
        target,
        read_state_fn,
        arrival_tolerance=arrival_tolerance,
        heading_tolerance=heading_tolerance,
        timeout=timeout,
        max_stuck_recoveries=max_stuck_recoveries,
        check_fn=check_fn,
    )
