"""Wander routine: roam the hunting zone looking for npcs.

Stays within a hunting zone defined by distance from zone center
(ctx.camp.hunt_min_dist to ctx.camp.hunt_max_dist). Biases toward npc density.
Avoids nearby players and danger points. Walks 150-240u (300u when returning).

Wander is purely about movement  -  it does NOT check for targets, tab,
or try to hand off to acquire. The brain handles routine transitions.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, override

from core.constants import PULL_ABORT_DISTANCE
from core.timing import interruptible_sleep
from core.types import Point, ReadStateFn
from nav.movement import MovementPhase, get_terrain
from perception.combat_eval import get_avoid_names, is_valid_target
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus, make_flee_predicate
from util.log_tiers import VERBOSE

if TYPE_CHECKING:
    from brain.context import AgentContext

log = logging.getLogger(__name__)

# -- Domain-local constants --
THREAT_DETECT_RANGE = 80.0  # stop wander / trigger evade within this range

# Default hunting zone (overridden by ctx.camp.hunt_min_dist/hunt_max_dist)
DEFAULT_HUNT_MIN = 50.0
DEFAULT_HUNT_MAX = 300.0


class WanderRoutine(RoutineBase):
    """Roam randomly near camp, biased toward npc density."""

    def __init__(
        self,
        camp_x: float,
        camp_y: float,
        read_state_fn: ReadStateFn | None = None,
        ctx: AgentContext | None = None,
    ) -> None:
        self._camp_x = camp_x
        self._camp_y = camp_y
        self._read_state_fn = read_state_fn
        self._ctx = ctx
        self._walked = False
        self._pause_until = 0.0
        self._last_wander_angle: float | None = None  # anti-oscillation
        self._flee_check: Callable[[], bool] | None = None
        # Breadcrumb trail: (x, y, timestamp) of recent positions.
        # Wander biases away from recent breadcrumbs to cover more area.
        self._breadcrumbs: list[tuple[float, float, float]] = []
        self._breadcrumb_ttl = 90.0  # seconds before breadcrumb expires
        # LINEAR patrol state
        self._patrol_forward: bool = True  # patrol direction along polyline
        # Non-blocking movement (replaces blocking move_to_point)
        self._movement_phase: MovementPhase | None = None
        self._walk_origin_x: float = 0.0
        self._walk_origin_y: float = 0.0

    def _unexplored_angle(self, state: GameState) -> float | None:
        """Return angle toward least-explored area based on breadcrumbs.

        Divides the circle into 8 sectors. Counts breadcrumbs in each.
        Returns the angle of the sector with the fewest recent breadcrumbs.
        Returns None if no breadcrumbs yet (no data to bias from).
        """
        if len(self._breadcrumbs) < 3:
            return None  # not enough data
        now_t = time.time()
        # 8 sectors of 45 degrees each
        sector_counts = [0] * 8
        for bx, by, bt in self._breadcrumbs:
            if now_t - bt > self._breadcrumb_ttl:
                continue
            angle = math.atan2(by - state.y, bx - state.x)
            sector = int((angle + math.pi) / (math.pi / 4)) % 8
            sector_counts[sector] += 1
        # Find the sector with fewest visits
        min_count = min(sector_counts)
        # Pick randomly among tied minimums
        candidates = [i for i, c in enumerate(sector_counts) if c == min_count]
        best_sector = random.choice(candidates)
        # Convert sector back to angle (center of sector)
        return best_sector * (math.pi / 4) - math.pi + (math.pi / 8)

    def _mob_density_angle(self, state: GameState) -> float | None:
        """Compute angle (radians) toward highest valid NPC density.

        Only considers npcs that pass is_valid_target  -  ignores green cons,
        guards, pets, etc. Weights closer npcs more heavily.
        Returns None if no valid npcs found.
        """
        valid_npcs = []
        for spawn in state.spawns:
            if not is_valid_target(spawn, state.level):
                continue
            if spawn.hp_current <= 0:
                continue
            dist = state.pos.dist_to(spawn.pos)
            if 40 < dist < 600:
                valid_npcs.append((spawn, dist))

        if not valid_npcs:
            return None

        # Weighted center of mass  -  closer NPCs pull harder
        wx, wy, total_w = 0.0, 0.0, 0.0
        for spawn, dist in valid_npcs:
            w = 1.0 / max(dist, 10.0)
            wx += spawn.x * w
            wy += spawn.y * w
            total_w += w

        cx = wx / total_w
        cy = wy / total_w
        return math.atan2(cy - state.y, cx - state.x)

    def _player_avoidance_angle(self, state: GameState) -> float | None:
        """If a player is within 25-250 units, return angle AWAY from them."""
        for spawn in state.spawns:
            if not spawn.is_player or spawn.name == state.name:
                continue
            dist = state.pos.dist_to(spawn.pos)
            if dist < 250:
                # Move away from the player
                angle_to_player = math.atan2(spawn.y - state.y, spawn.x - state.x)
                return angle_to_player + math.pi  # opposite direction
        return None

    def _guard_avoidance_angle(self, state: GameState) -> float | None:
        """If a roaming avoid-NPC is within 250 units, return angle AWAY."""
        for spawn in state.spawns:
            if not spawn.is_npc:
                continue
            # Check if this NPC is a guard/merchant/etc from avoid names
            is_guard = any(avoid in spawn.name for avoid in get_avoid_names())
            if not is_guard:
                continue
            dist = state.pos.dist_to(spawn.pos)
            if dist < PULL_ABORT_DISTANCE:
                angle_to_guard = math.atan2(spawn.y - state.y, spawn.x - state.x)
                log.info(
                    "[POSITION] Wander: avoiding guard '%s' at dist=%.0f  -  moving away", spawn.name, dist
                )
                return angle_to_guard + math.pi
        return None

    def _linear_walk_target(self, state: GameState) -> tuple[float, float, float]:
        """Pick a patrol target along the LINEAR polyline path.

        Returns (target_x, target_y, walk_dist). The agent walks along the
        polyline in its current patrol direction, reversing at endpoints.
        Adds perpendicular jitter for natural movement.
        """
        if self._ctx is None:
            log.warning("[POSITION] Wander: LINEAR patrol skipped -- no ctx")
            return (state.x, state.y, 0.0)
        camp = self._ctx.camp

        path_t = camp.patrol_position(state.pos)
        path_len = camp.path_total_length()

        # Reverse at endpoints (with hysteresis to prevent oscillation)
        if path_t > 0.90 and self._patrol_forward:
            self._patrol_forward = False
            log.info("[POSITION] Wander: LINEAR patrol reversing at path end (t=%.2f)", path_t)
        elif path_t < 0.10 and not self._patrol_forward:
            self._patrol_forward = True
            log.info("[POSITION] Wander: LINEAR patrol reversing at path start (t=%.2f)", path_t)

        # Advance 120-200u along the polyline in current direction
        advance_dist = random.uniform(120, 200)
        advance_t = advance_dist / max(path_len, 1.0)

        if self._patrol_forward:
            new_t = min(1.0, path_t + advance_t)
        else:
            new_t = max(0.0, path_t - advance_t)

        # Get the base point on the polyline
        _base = camp.point_along_path(new_t)
        base_x, base_y = _base.x, _base.y

        # Add perpendicular jitter for natural movement (up to 30% of corridor)
        jitter = random.gauss(0, camp.corridor_width * 0.15)
        # Need the perpendicular direction at this path segment
        # Approximate: direction along path at new_t
        dt = 0.02
        t_ahead = min(1.0, new_t + dt)
        t_behind = max(0.0, new_t - dt)
        _pa = camp.point_along_path(t_behind)
        ax, ay = _pa.x, _pa.y
        _pb = camp.point_along_path(t_ahead)
        bx, by = _pb.x, _pb.y
        seg_dx = bx - ax
        seg_dy = by - ay
        seg_len = math.hypot(seg_dx, seg_dy)
        if seg_len > 0.1:
            # Perpendicular direction (rotate 90 degrees)
            perp_x = -seg_dy / seg_len
            perp_y = seg_dx / seg_len
            base_x += perp_x * jitter
            base_y += perp_y * jitter

        # Bias toward npc density if available
        mob_angle = self._mob_density_angle(state)
        if mob_angle is not None and random.random() < 0.3:
            # Blend: shift target 20-40u toward npc density
            blend = random.uniform(20, 40)
            base_x += math.cos(mob_angle) * blend
            base_y += math.sin(mob_angle) * blend

        walk_dist = state.pos.dist_to(Point(base_x, base_y, 0.0))
        return (base_x, base_y, walk_dist)

    # -------------------------------------------------------------------------
    # tick() sub-methods
    # -------------------------------------------------------------------------

    def _tick_movement_phase(self, state: GameState) -> RoutineStatus | None:
        """Continue an in-progress non-blocking walk.

        Returns RUNNING if still walking or just completed (with bookkeeping).
        Returns None if self._movement_phase is None (no walk in progress).
        """
        if self._movement_phase is None:
            return None

        result = self._movement_phase.tick()
        if result is None:
            return RoutineStatus.RUNNING  # still walking
        # Movement complete -- post-walk bookkeeping
        if self._read_state_fn:
            ns = self._read_state_fn()
        else:
            ns = state
        actual = Point(self._walk_origin_x, self._walk_origin_y, 0.0).dist_to(ns.pos)
        new_guard_dist = (
            ns.pos.dist_to(Point(self._ctx.camp.guard_x, self._ctx.camp.guard_y, 0.0)) if self._ctx else 0
        )
        log.info(
            "[POSITION] Wander: walk %s  -  now at (%.0f, %.0f) walked=%.0f guard_dist=%.0f",
            "arrived" if result else "stopped_early",
            ns.x,
            ns.y,
            actual,
            new_guard_dist,
        )
        if self._ctx:
            self._ctx.metrics.wander_total_distance += actual
            self._ctx.metrics.wander_count += 1
        # Breadcrumb + visited
        now_t = time.time()
        self._breadcrumbs.append((ns.x, ns.y, now_t))
        if self._ctx and self._ctx.spatial_memory:
            self._ctx.spatial_memory.mark_visited(ns.pos)
        self._breadcrumbs = [(x, y, t) for x, y, t in self._breadcrumbs if now_t - t < self._breadcrumb_ttl]
        self._movement_phase = None
        self._walked = True
        if random.random() < 0.30:
            pause = random.uniform(0.3, 0.8)
            self._pause_until = time.time() + pause
            log.debug("[POSITION] Wander: pausing %.1fs", pause)
        else:
            self._pause_until = 0.0
        return RoutineStatus.RUNNING

    def _pick_linear_target(
        self,
        state: GameState,
        hunt_min: float,
        hunt_max: float,
    ) -> tuple[float, float, float, bool] | RoutineStatus:
        """Pick a walk target for LINEAR camp type.

        Returns (target_x, target_y, dist, chose_resource) on normal paths.
        Returns RoutineStatus.SUCCESS when a TRAVEL plan is triggered.
        """
        assert self._ctx is not None

        avoid_angle = self._player_avoidance_angle(state)
        guard_avoid = self._guard_avoidance_angle(state)

        target_x: float
        target_y: float
        dist: float

        if avoid_angle is not None:
            # Player nearby: move away instead of patrolling
            dist = random.uniform(80, 120)
            angle = avoid_angle + random.gauss(0, 0.5)
            target_x = state.x + dist * math.cos(angle)
            target_y = state.y + dist * math.sin(angle)
            log.info("[POSITION] Wander: LINEAR moving AWAY from nearby player")
        elif guard_avoid is not None:
            dist = random.uniform(80, 120)
            angle = guard_avoid + random.gauss(0, 0.5)
            target_x = state.x + dist * math.cos(angle)
            target_y = state.y + dist * math.sin(angle)
        else:
            # Check if outside corridor -- walk back to path first
            perp_dist = self._ctx.camp.distance_to_camp(state)
            # Use actual walk distance (not just perpendicular) to decide
            # whether wander's short terrain walks can handle it
            _np = self._ctx.camp.nearest_point_on_path(state.pos)
            nx, ny = _np.x, _np.y
            actual_walk = state.pos.dist_to(_np)
            if perp_dist > 150 or actual_walk > 300:
                # Too far for wander's terrain-safe walks -- use TRAVEL
                # (A* pathfinding handles obstacles properly)
                from core.types import PlanType

                self._ctx.plan.active = PlanType.TRAVEL
                self._ctx.plan.travel.target_x = nx
                self._ctx.plan.travel.target_y = ny
                log.info(
                    "[POSITION] Wander: LINEAR too far off corridor "
                    "(perp=%.0fu walk=%.0fu) -- TRAVEL to (%.0f,%.0f)",
                    perp_dist,
                    actual_walk,
                    nx,
                    ny,
                )
                self._walked = True
                return RoutineStatus.SUCCESS
            if perp_dist > 50:
                _np2 = self._ctx.camp.nearest_point_on_path(state.pos)
                nx, ny = _np2.x, _np2.y
                target_x, target_y = nx, ny
                log.log(
                    VERBOSE, "[POSITION] Wander: LINEAR returning to path (%.0fu off corridor)", perp_dist
                )
            else:
                # Normal patrol along the polyline
                acq_fails = self._ctx.metrics.consecutive_acquire_fails
                if acq_fails >= 5:
                    # Drought: head to the far end of the path
                    path_t = self._ctx.camp.patrol_position(state.pos)
                    if path_t > 0.5:
                        self._patrol_forward = False
                    else:
                        self._patrol_forward = True
                    log.info(
                        "[POSITION] Wander: LINEAR DROUGHT  -  %d fails, heading to %s end",
                        acq_fails,
                        "start" if not self._patrol_forward else "far",
                    )
                target_x, target_y, _ = self._linear_walk_target(state)

            log.log(
                VERBOSE,
                "[POSITION] Wander: LINEAR patrol %s to (%.0f, %.0f) dist=%.0f path_t=%.2f",
                "forward" if self._patrol_forward else "backward",
                target_x,
                target_y,
                state.pos.dist_to(Point(target_x, target_y, 0.0)),
                self._ctx.camp.patrol_position(state.pos),
            )
            dist = state.pos.dist_to(Point(target_x, target_y, 0.0))

        # Set angle/dist so shared post-processing (threat avoidance,
        # bounds clamping, terrain) works for LINEAR too
        dist = state.pos.dist_to(Point(target_x, target_y, 0.0))
        return (target_x, target_y, dist, False)

    def _apply_forward_bias_and_dist(
        self,
        angle: float,
        outside_zone: bool,
    ) -> tuple[float, float]:
        """Apply forward-momentum bias to angle, then compute walk distance.

        Returns (angle, dist).
        """
        # Forward bias: prefer continuing roughly forward
        if self._last_wander_angle is not None and not math.isnan(angle):
            if random.random() < 0.70:
                forward = self._last_wander_angle
                angle = forward + random.gauss(0, math.radians(45))
            reverse = (self._last_wander_angle + math.pi) % (2 * math.pi)
            diff = abs((angle - reverse + math.pi) % (2 * math.pi) - math.pi)
            if diff < math.radians(60):
                offset = random.choice([-1, 1]) * math.radians(90)
                angle = (self._last_wander_angle + offset) % (2 * math.pi)

        dist = random.uniform(100, 150) if outside_zone else random.uniform(120, 200)
        return (angle, dist)

    def _choose_direction(
        self,
        state: GameState,
        guard_dist: float,
        hunt_min: float,
        hunt_max: float,
        avoid_angle: float | None,
        guard_avoid: float | None,
        mob_angle: float | None,
    ) -> float:
        """Select a wander angle via the priority chain.

        Checks escalation, zone bounds, avoidance, density, prediction,
        spatial memory, and unexplored sectors in priority order.
        """
        acq_fails = self._ctx.metrics.consecutive_acquire_fails if self._ctx else 0

        if acq_fails >= 5 and guard_dist <= hunt_max and self._ctx and self._ctx.spatial_memory:
            return self._direction_drought(state, acq_fails)
        if acq_fails >= 3 and guard_dist <= hunt_max:
            return self._direction_escalation(state, acq_fails, guard_dist, hunt_max)
        if guard_dist < hunt_min:
            assert self._ctx is not None
            angle = math.atan2(state.y - self._ctx.camp.guard_y, state.x - self._ctx.camp.guard_x)
            angle += random.gauss(0, 0.5)
            log.info(
                "[POSITION] Wander: too close to center (%.0f < %.0f)  -  walking into hunting zone",
                guard_dist,
                hunt_min,
            )
            return angle
        if guard_dist > hunt_max:
            assert self._ctx is not None
            angle = math.atan2(self._ctx.camp.guard_y - state.y, self._ctx.camp.guard_x - state.x)
            angle += random.gauss(0, 0.5)
            log.info(
                "[POSITION] Wander: beyond hunting zone (%.0f > %.0f)  -  walking back", guard_dist, hunt_max
            )
            return angle
        if avoid_angle is not None:
            log.info("[POSITION] Wander: moving AWAY from nearby player")
            return avoid_angle + random.gauss(0, 0.5)
        if guard_avoid is not None:
            return guard_avoid + random.gauss(0, 0.5)
        if mob_angle is not None and random.random() < 0.7:
            log.log(VERBOSE, "[POSITION] Wander: heading toward npc density")
            return mob_angle + random.gauss(0, 0.6)
        if self._ctx and self._ctx.spawn_predictor:
            return self._direction_from_predictor(state, hunt_min, hunt_max)
        if self._ctx and self._ctx.spatial_memory:
            return self._direction_from_spatial(state)
        return self._direction_from_unexplored(state)

    def _direction_drought(self, state: GameState, acq_fails: int) -> float:
        """Direction when in drought mode (5+ acquire failures)."""
        assert self._ctx is not None
        if not self._ctx.spatial_memory:
            return random.uniform(0, 2 * math.pi)
        best = self._ctx.spatial_memory.best_direction(state.pos)
        if best:
            angle = math.atan2(best[1] - state.y, best[0] - state.x)
            angle += random.gauss(0, 0.3)
            log.info(
                "[POSITION] Wander: DROUGHT MODE  -  %d acquire failures, "
                "heading toward hot zone (%.0f,%.0f)",
                acq_fails,
                best[0],
                best[1],
            )
            return angle
        log.info(
            "[POSITION] Wander: DROUGHT MODE  -  %d acquire failures, no spatial data  -  random direction",
            acq_fails,
        )
        return random.uniform(0, 2 * math.pi)

    def _direction_escalation(
        self, state: GameState, acq_fails: int, guard_dist: float, hunt_max: float
    ) -> float:
        """Direction when in escalation mode (3-4 acquire failures)."""
        unexplored = self._unexplored_angle(state)
        if unexplored is not None:
            log.info(
                "[POSITION] Wander: ESCALATION  -  %d acquire failures, heading toward unexplored sector",
                acq_fails,
            )
            return unexplored + random.gauss(0, 0.4)
        if guard_dist > hunt_max * 0.6:
            assert self._ctx is not None
            angle = math.atan2(self._ctx.camp.guard_y - state.y, self._ctx.camp.guard_x - state.x)
            angle += random.gauss(0, 0.8)
            log.info(
                "[POSITION] Wander: ESCALATION  -  %d acquire failures, drifting toward camp (dist=%.0f)",
                acq_fails,
                guard_dist,
            )
            return angle
        log.info("[POSITION] Wander: ESCALATION  -  %d acquire failures, random direction", acq_fails)
        return random.uniform(0, 2 * math.pi)

    def _direction_from_predictor(self, state: GameState, hunt_min: float, hunt_max: float) -> float:
        """Direction from spawn predictor, falling back to spatial memory."""
        assert self._ctx is not None
        if not self._ctx.spawn_predictor:
            return self._direction_from_spatial(state)
        best_cells = self._ctx.spawn_predictor.best_cells(3, time.time())
        chosen = None
        for pt, secs in best_cells:
            if secs > 120:
                break
            d = state.pos.dist_to(pt)
            if hunt_min <= d <= hunt_max:
                chosen = pt
                break
        if chosen is not None:
            angle = math.atan2(chosen.y - state.y, chosen.x - state.x)
            angle += random.gauss(0, 0.3)
            log.log(
                VERBOSE, "[POSITION] Wander: heading toward predicted respawn (%.0f,%.0f)", chosen.x, chosen.y
            )
            return angle
        if self._ctx.spatial_memory:
            best = self._ctx.spatial_memory.best_direction(state.pos)
            if best:
                angle = math.atan2(best[1] - state.y, best[0] - state.x)
                angle += random.gauss(0, 0.5)
                log.log(
                    VERBOSE,
                    "[POSITION] Wander: heading toward learned hot zone (%.0f,%.0f)",
                    best[0],
                    best[1],
                )
                return angle
        return random.uniform(0, 2 * math.pi)

    def _direction_from_spatial(self, state: GameState) -> float:
        """Direction from spatial memory, falling back to unexplored."""
        assert self._ctx is not None
        if not self._ctx.spatial_memory:
            return self._direction_from_unexplored(state)
        best = self._ctx.spatial_memory.best_direction(state.pos)
        if best:
            angle = math.atan2(best[1] - state.y, best[0] - state.x)
            angle += random.gauss(0, 0.5)
            log.log(
                VERBOSE,
                "[POSITION] Wander: heading toward learned hot zone (%.0f,%.0f)",
                best[0],
                best[1],
            )
            return angle
        return self._direction_from_unexplored(state)

    def _direction_from_unexplored(self, state: GameState) -> float:
        """Direction from unexplored sector, falling back to random."""
        unexplored = self._unexplored_angle(state)
        if unexplored is not None:
            log.info("[POSITION] Wander: heading toward unexplored sector")
            return unexplored + random.gauss(0, 0.4)
        return random.uniform(0, 2 * math.pi)

    def _pick_circular_target(
        self,
        state: GameState,
        guard_dist: float,
        hunt_min: float,
        hunt_max: float,
        outside_zone: bool,
    ) -> tuple[float, float, float, bool]:
        """Pick a walk target for CIRCULAR camp type.

        Returns (target_x, target_y, dist, False).
        """
        avoid_angle = self._player_avoidance_angle(state)
        guard_avoid = self._guard_avoidance_angle(state)
        mob_angle = self._mob_density_angle(state)

        angle = self._choose_direction(
            state, guard_dist, hunt_min, hunt_max, avoid_angle, guard_avoid, mob_angle
        )

        angle, dist = self._apply_forward_bias_and_dist(angle, outside_zone)

        target_x = state.x + dist * math.cos(angle)
        target_y = state.y + dist * math.sin(angle)
        return (target_x, target_y, dist, False)

    def _shorten_walk_for_npcs(
        self,
        state: GameState,
        target_x: float,
        target_y: float,
        angle: float,
        dist: float,
    ) -> tuple[float, float]:
        """Shorten the walk if the destination is too close to a guard NPC or
        aggressive threat npc. Returns (target_x, target_y).
        """
        # Check if destination is too close to any known guard NPC.
        # Guards are in avoid names. If destination lands within 250u
        # of a guard, shorten the walk to stop 250u from the guard.
        for spawn in state.spawns:
            if not spawn.is_npc:
                continue
            is_guard = any(avoid in spawn.name for avoid in get_avoid_names())
            if not is_guard:
                continue
            dest_to_guard = Point(target_x, target_y, 0.0).dist_to(spawn.pos)
            if dest_to_guard < PULL_ABORT_DISTANCE:
                # Shorten walk: stop before reaching guard
                safe_dist = dist * 0.5  # halve the walk distance
                target_x = state.x + safe_dist * math.cos(angle)
                target_y = state.y + safe_dist * math.sin(angle)
                log.info(
                    "[POSITION] Wander: destination too close to guard '%s' "
                    "(%.0fu) -- shortened walk to %.0fu",
                    spawn.name,
                    dest_to_guard,
                    safe_dist,
                )
                break

        # Aggressive npc avoidance: check if path passes near
        # scowling npcs (wolves, bears, skeletons). Shorten walk
        # to avoid walking into threat range.
        from perception.combat_eval import is_threat

        zone_disp = self._ctx.zone.zone_dispositions if self._ctx else None
        for spawn in state.spawns:
            if not spawn.is_npc or spawn.hp_current <= 0:
                continue
            if not is_threat(spawn, state.level, zone_disp):
                continue
            # Check if destination is within 40u of this threat
            threat_to_dest = Point(target_x, target_y, 0.0).dist_to(spawn.pos)
            if threat_to_dest < 40:
                # Also check if the path midpoint passes near it
                mid_x = (state.x + target_x) / 2
                mid_y = (state.y + target_y) / 2
                threat_to_mid = Point(mid_x, mid_y, 0.0).dist_to(spawn.pos)
                if threat_to_mid < 50:
                    safe_dist = dist * 0.4
                    target_x = state.x + safe_dist * math.cos(angle)
                    target_y = state.y + safe_dist * math.sin(angle)
                    log.info(
                        "[POSITION] Wander: path near aggressive '%s' lv%d "
                        "(%.0fu from dest) -- shortened to %.0fu",
                        spawn.name,
                        spawn.level,
                        threat_to_dest,
                        safe_dist,
                    )
                    break

        return (target_x, target_y)

    def _apply_safety_constraints(
        self,
        state: GameState,
        target_x: float,
        target_y: float,
        angle: float,
        dist: float,
    ) -> tuple[float, float, float]:
        """Apply guard proximity, NPC avoidance, zone bounds, danger points.

        Returns (target_x, target_y, angle) after all constraint adjustments.
        No-ops and returns inputs unchanged when self._ctx is None.
        """
        if not self._ctx:
            return (target_x, target_y, angle)

        hunt_min = self._ctx.camp.hunt_min_dist
        hunt_max = self._ctx.camp.hunt_max_dist

        # Clamp destination within hunting zone (guard distance band)
        # Also check guard proximity at destination -- prevents walking
        # past guards when chasing resource npcs in their direction.
        dest_guard_dist = Point(target_x, target_y, 0.0).dist_to(
            Point(self._ctx.camp.guard_x, self._ctx.camp.guard_y, 0.0)
        )
        if dest_guard_dist > hunt_max:
            angle_to_guards = math.atan2(self._ctx.camp.guard_y - state.y, self._ctx.camp.guard_x - state.x)
            angle = angle_to_guards + random.uniform(-0.5, 0.5)
            target_x = state.x + dist * math.cos(angle)
            target_y = state.y + dist * math.sin(angle)
        elif dest_guard_dist < hunt_min:
            angle_from_guards = math.atan2(state.y - self._ctx.camp.guard_y, state.x - self._ctx.camp.guard_x)
            angle = angle_from_guards + random.uniform(-0.5, 0.5)
            target_x = state.x + dist * math.cos(angle)
            target_y = state.y + dist * math.sin(angle)

        target_x, target_y = self._shorten_walk_for_npcs(state, target_x, target_y, angle, dist)

        # Zone boundary clamping: keep within configured bounds
        camp_cfg = self._ctx.camp
        if hasattr(camp_cfg, "bounds_x_min") and camp_cfg.bounds_x_min is not None:
            if target_x < camp_cfg.bounds_x_min + 50:
                target_x = camp_cfg.bounds_x_min + random.uniform(50, 80)
                log.info(
                    "[POSITION] Wander: clamped x to %.0f (zone wall at %.0f)",
                    target_x,
                    camp_cfg.bounds_x_min,
                )
        if hasattr(camp_cfg, "bounds_x_max") and camp_cfg.bounds_x_max is not None:
            if target_x > camp_cfg.bounds_x_max - 50:
                target_x = camp_cfg.bounds_x_max - random.uniform(50, 80)
        if hasattr(camp_cfg, "bounds_y_min") and camp_cfg.bounds_y_min is not None:
            if target_y < camp_cfg.bounds_y_min + 50:
                target_y = camp_cfg.bounds_y_min + random.uniform(50, 80)
        if hasattr(camp_cfg, "bounds_y_max") and camp_cfg.bounds_y_max is not None:
            if target_y > camp_cfg.bounds_y_max - 50:
                target_y = camp_cfg.bounds_y_max - random.uniform(50, 80)

        # Danger point avoidance: reject destinations too close
        for dp_x, dp_y, _, dp_min, dp_name in self._ctx.camp.danger_points:
            dp_dist = Point(target_x, target_y, 0.0).dist_to(Point(dp_x, dp_y, 0.0))
            if dp_dist < dp_min:
                # Redirect away from danger point
                away = math.atan2(state.y - dp_y, state.x - dp_x)
                angle = away + random.uniform(-0.3, 0.3)
                target_x = state.x + dist * math.cos(angle)
                target_y = state.y + dist * math.sin(angle)
                log.info("[POSITION] Wander: avoiding danger '%s'  -  redirected", dp_name)

        return (target_x, target_y, angle)

    def _terrain_check(
        self,
        state: GameState,
        target_x: float,
        target_y: float,
        angle: float,
        dist: float,
    ) -> tuple[float, float] | None:
        """Validate destination against terrain and pathfinding.

        Uses find_path() to confirm a navigable route exists (not just
        endpoint walkability). If the target is unreachable, retries
        with random directions. Returns (target_x, target_y) on a path
        that A* can solve, or None if all directions are blocked.
        """
        terrain = get_terrain()
        if not terrain:
            return (target_x, target_y)

        for _retry in range(5):
            if not terrain.is_walkable(target_x, target_y):
                # Target itself is in water/obstacle/cliff -- try elsewhere
                angle = random.uniform(0, 2 * math.pi)
                target_x = state.x + dist * math.cos(angle)
                target_y = state.y + dist * math.sin(angle)
                continue

            # Endpoint walkable -- verify A* can actually reach it
            path = terrain.find_path(state.x, state.y, target_x, target_y, jitter=0)
            if path is not None:
                return (target_x, target_y)

            # A* failed (obstacles block the route) -- try new direction
            log.log(VERBOSE, "[POSITION] Wander: no path to (%.0f,%.0f), retrying", target_x, target_y)
            angle = random.uniform(0, 2 * math.pi)
            target_x = state.x + dist * math.cos(angle)
            target_y = state.y + dist * math.sin(angle)

        # All retries exhausted -- shorten walk as last resort
        dist = min(dist, 60.0)
        target_x = state.x + dist * math.cos(angle)
        target_y = state.y + dist * math.sin(angle)
        if terrain.is_walkable(target_x, target_y):
            path = terrain.find_path(state.x, state.y, target_x, target_y, jitter=0)
            if path is not None:
                log.log(VERBOSE, "[POSITION] Wander: short walk fallback to (%.0f,%.0f)", target_x, target_y)
                return (target_x, target_y)

        log.log(VERBOSE, "[POSITION] Wander: terrain blocked all directions -- staying put")
        self._walked = True
        self._pause_until = time.time() + 2.0
        return None

    @override
    def enter(self, state: GameState) -> None:
        self._walked = False
        self._pause_until = 0.0
        self._movement_phase = None
        if self._read_state_fn and self._ctx:
            self._flee_check = make_flee_predicate(self._read_state_fn, self._ctx)
        else:
            self._flee_check = None
        log.info(
            "[POSITION] Wander: starting from (%.0f, %.0f) camp=(%.0f, %.0f) camp_dist=%.0f",
            state.x,
            state.y,
            self._camp_x,
            self._camp_y,
            state.pos.dist_to(Point(self._camp_x, self._camp_y, 0.0)),
        )

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        # Determine guard distance for hunting zone awareness
        hunt_min = self._ctx.camp.hunt_min_dist if self._ctx else DEFAULT_HUNT_MIN
        hunt_max = self._ctx.camp.hunt_max_dist if self._ctx else DEFAULT_HUNT_MAX
        guard_dist = 9999.0
        if self._ctx:
            guard_dist = state.pos.dist_to(Point(self._ctx.camp.guard_x, self._ctx.camp.guard_y, 0.0))
        outside_zone = guard_dist < hunt_min or guard_dist > hunt_max

        # Phase 1b: Movement in progress (non-blocking continuation)
        movement_result = self._tick_movement_phase(state)
        if movement_result is not None:
            return movement_result

        # Phase 1a: Direction selection + start walk
        if not self._walked:
            from core.types import CampType

            target_x = state.x  # safe default (no movement)
            target_y = state.y
            dist: float
            angle: float

            if (
                self._ctx
                and self._ctx.camp.camp_type == CampType.LINEAR
                and len(self._ctx.camp.patrol_waypoints) >= 2
            ):
                # -- LINEAR camp: patrol along polyline path --
                linear_result = self._pick_linear_target(state, hunt_min, hunt_max)
                if isinstance(linear_result, RoutineStatus):
                    return linear_result
                target_x, target_y, dist, _ = linear_result
                angle = math.atan2(target_y - state.y, target_x - state.x)
            else:
                # -- CIRCULAR camp: existing angle-based target selection --
                target_x, target_y, dist, _ = self._pick_circular_target(
                    state, guard_dist, hunt_min, hunt_max, outside_zone
                )
                angle = math.atan2(target_y - state.y, target_x - state.x)

            # -- Shared post-processing (both LINEAR and CIRCULAR) --
            target_x, target_y, angle = self._apply_safety_constraints(state, target_x, target_y, angle, dist)

            # Terrain safety: reject destinations in water, lava, or off cliffs
            terrain_result = self._terrain_check(state, target_x, target_y, angle, dist)
            if terrain_result is None:
                return RoutineStatus.RUNNING
            target_x, target_y = terrain_result

            self._last_wander_angle = angle  # remember for anti-oscillation

            walk_dist = state.pos.dist_to(Point(target_x, target_y, 0.0))
            log.info(
                "[POSITION] Wander: walking to (%.0f, %.0f) dist=%.0f from (%.0f, %.0f) guard_dist=%.0f",
                target_x,
                target_y,
                walk_dist,
                state.x,
                state.y,
                guard_dist,
            )

            # Start non-blocking movement -- brain re-evaluates ACQUIRE/FLEE
            # every tick, so the old _wander_check closure is unnecessary.
            self._walk_origin_x = state.x
            self._walk_origin_y = state.y
            if self._read_state_fn:
                self._movement_phase = MovementPhase(
                    target_x,
                    target_y,
                    self._read_state_fn,
                    arrival_tolerance=10.0,
                    timeout=20.0,
                )
            else:
                self._walked = True
            return RoutineStatus.RUNNING

        # Phase 2: Brief pause (only if pause was set)
        if self._pause_until > 0 and time.time() < self._pause_until:
            remaining = self._pause_until - time.time()
            interruptible_sleep(min(remaining, 0.5), self._flee_check)
            return RoutineStatus.RUNNING

        return RoutineStatus.SUCCESS

    @override
    def exit(self, state: GameState) -> None:
        if self._movement_phase is not None:
            self._movement_phase.cancel()
            self._movement_phase = None
