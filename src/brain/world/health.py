"""Runtime health monitor  -  detects silent failures and approaching threats.

Runs every brain tick with near-zero cost. Three tiers:
  Tier 1 (every tick): data integrity checks  -  mana/HP/position sanity
  Tier 2 (every 30s):  deep consistency checks  -  defeat tracking, pet state
  Tier 3 (every tick):  threat proximity  -  YELLOW/RED npcs approaching

Tier 1-2 only log warnings. Tier 3 sets flags on AgentContext that the
brain's FLEE rule reads to trigger evasive action.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any

from core.types import Con, Point
from nav.geometry import heading_to
from perception.combat_eval import _DEFAULT_AVOID_PREFIXES, con_color, is_pet

if TYPE_CHECKING:
    from brain.context import AgentContext
    from perception.state import GameState

_AVOID_PREFIXES = _DEFAULT_AVOID_PREFIXES
_PATROL_MIN_SPEED = 2.0  # units/tick below which we ignore patrol collision
_PATROL_EVADE_RADIUS = 70.0  # closest-pass distance to trigger combat evade

log = logging.getLogger("compass.health")


class HealthMonitor:
    """Lightweight observer  -  validates state each tick, flags threats."""

    def __init__(self) -> None:
        # Tier 1: previous-tick values for transition detection
        self._prev_x: float = 0.0
        self._prev_y: float = 0.0
        self._prev_mana: int = -1
        self._prev_weight: int = -1
        self._prev_level: int = 0
        self._zero_mana_ticks: int = 0
        self._mana_warned: bool = False

        # Tier 2: periodic deep check state
        self._last_deep_check: float = 0.0
        self._prev_kills: int = 0
        self._prev_combat_count: int = 0
        self._last_mana_change: float = 0.0

        # Tier 3: threat tracking
        self._threat_positions: dict[int, Point] = {}  # spawn_id -> position
        self._last_threat_log: dict[str, float] = {}  # throttle key -> last log time

    def tick(self, state: GameState, ctx: AgentContext) -> None:
        """Called every brain tick. Tier 1 + Tier 3. Must be <1ms."""
        self._check_vitals(state)
        self._check_position(state)
        self._check_threats(state, ctx)

    def deep_check(self, state: GameState, ctx: AgentContext) -> None:
        """Called every 30s alongside snapshot. Tier 2."""
        now = time.time()
        if now - self._last_deep_check < 25.0:
            return
        self._last_deep_check = now

        self._check_kill_integrity(ctx)
        self._check_pet_integrity(state, ctx)
        self._check_engaged_integrity(state, ctx)
        self._check_buff_integrity(state, ctx)
        self._check_mana_frozen(state, ctx)

    # -- Tier 1: Every-tick vitals ------------------------------------

    def _check_vitals(self, state: GameState) -> None:
        # Mana reads 0 for multiple ticks -> character info may be stale
        if state.mana_current == 0 and state.mana_max == 0:
            self._zero_mana_ticks += 1
            if self._zero_mana_ticks == 30 and not self._mana_warned:
                log.warning(
                    "[HEALTH] mana reads 0 for 30 ticks  -  character info may be stale or not yet loaded"
                )
                self._mana_warned = True
        else:
            self._zero_mana_ticks = 0
            self._mana_warned = False
            if self._last_mana_change == 0.0 or state.mana_current != self._prev_mana:
                self._last_mana_change = time.time()

        # HP sanity
        if state.hp_max > 0:
            if state.hp_current > state.hp_max * 1.1:
                log.warning("[HEALTH] HP %d exceeds max %d", state.hp_current, state.hp_max)
            if state.hp_current < 0:
                log.warning("[HEALTH] HP is negative: %d", state.hp_current)

        # Weight dropped to 0 unexpectedly
        if self._prev_weight > 10 and state.weight == 0:
            log.warning(
                "[HEALTH] weight dropped from %d to 0  -  character state read may have failed",
                self._prev_weight,
            )

        # Level changed
        if self._prev_level > 0 and state.level != self._prev_level:
            log.info("[PERCEPTION] level changed: %d -> %d", self._prev_level, state.level)

        self._prev_mana = state.mana_current
        self._prev_weight = state.weight
        self._prev_level = state.level

    def _check_position(self, state: GameState) -> None:
        if self._prev_x == 0.0 and self._prev_y == 0.0:
            self._prev_x, self._prev_y = state.x, state.y
            return

        dist = math.hypot(state.x - self._prev_x, state.y - self._prev_y)
        if dist > 500:
            log.warning(
                "[HEALTH] position jumped %.0fu in one tick "
                "(%.0f,%.0f) -> (%.0f,%.0f)  -  zone load or corruption",
                dist,
                self._prev_x,
                self._prev_y,
                state.x,
                state.y,
            )

        self._prev_x, self._prev_y = state.x, state.y

    # -- Tier 2: Periodic deep checks --------------------------------

    def _check_kill_integrity(self, ctx: AgentContext) -> None:
        # Defeats with empty names
        empty_kills = sum(1 for k in ctx.defeat_tracker.defeat_history if not k.name)
        if empty_kills > 0:
            log.warning(
                "[HEALTH] %d defeats in history have empty names  -  target was lost before recording",
                empty_kills,
            )

        # Combat happened but no defeats recorded
        combat_count = ctx.metrics.routine_counts.get("IN_COMBAT", 0)
        if combat_count > self._prev_combat_count and ctx.defeat_tracker.defeats == self._prev_kills:
            combats_without_kill = combat_count - self._prev_combat_count
            if combats_without_kill >= 2:
                log.warning(
                    "[HEALTH] %d combats completed but defeats still %d  -  defeat recording may be broken",
                    combats_without_kill,
                    ctx.defeat_tracker.defeats,
                )
        self._prev_kills = ctx.defeat_tracker.defeats
        self._prev_combat_count = combat_count

    def _check_pet_integrity(self, state: GameState, ctx: AgentContext) -> None:
        if not ctx.pet.alive or not ctx.pet.spawn_id:
            return
        # Pet alive but not in spawn list
        found = False
        for sp in state.spawns:
            if sp.spawn_id == ctx.pet.spawn_id:
                found = True
                break
        if not found:
            log.warning(
                "[HEALTH] pet alive=True id=%d but not in spawn list (%d spawns)",
                ctx.pet.spawn_id,
                len(state.spawns),
            )

    def _check_engaged_integrity(self, state: GameState, ctx: AgentContext) -> None:
        if ctx.combat.engaged and not state.target and not ctx.combat.pull_target_id:
            log.warning("[HEALTH] engaged=True but no target and no pull_target_id")

    def _check_buff_integrity(self, state: GameState, ctx: AgentContext) -> None:
        """Verify buff data makes sense."""
        for spell_id, ticks in state.buffs:
            if spell_id < 0:
                log.warning("[HEALTH] buff with negative spell_id: %d", spell_id)
            if ticks < 0:
                log.warning("[HEALTH] buff spell_id=%d has negative ticks: %d", spell_id, ticks)

    def _check_mana_frozen(self, state: GameState, ctx: AgentContext) -> None:
        if state.mana_max == 0:
            return
        # Don't warn if mana is at max  -  that's normal (agent not casting)
        if state.mana_max > 0 and state.mana_current >= state.mana_max * 0.98:
            return
        elapsed = time.time() - self._last_mana_change if self._last_mana_change > 0 else 0
        if elapsed > 120 and state.is_standing:
            log.warning(
                "[HEALTH] mana unchanged for %.0fs while not sitting  -  mana read may be stuck at %d",
                elapsed,
                state.mana_current,
            )

    # -- Tier 3: Threat proximity + trajectory prediction ------------

    def _evaluate_single_spawn_threat(self, sp: Any, state: GameState, ctx: AgentContext) -> dict | None:
        """Evaluate a single spawn for threat potential.

        Returns a dict with threat info to apply to ctx, or None if not a threat.
        The dict may contain keys: imminent, approaching, evasion_point, patrol_evade,
        con, position, skip_traj.
        """
        # Skip NPCs in the avoid set (known non-hostile high-level)
        base_name = sp.name.split("_")[0] if sp.name else ""
        if base_name in _AVOID_PREFIXES:
            return None

        con = con_color(state.level, sp.level)
        if con not in (Con.YELLOW, Con.RED):
            return None

        # Skip npcs with known non-aggressive disposition
        if ctx.zone.zone_dispositions:
            from perception.combat_eval import PASSIVE_DISPOSITIONS, get_disposition

            disp = get_disposition(sp.name, ctx.zone.zone_dispositions)
            if disp in PASSIVE_DISPOSITIONS:
                return None

        dist = state.pos.dist_to(sp.pos)
        result: dict = {"position": sp.pos, "con": con}

        # Imminent threat: in threat range
        if dist < 40:
            result["imminent"] = True
            _tkey = f"aggro_{sp.spawn_id}"
            _now = time.time()
            if _now - self._last_threat_log.get(_tkey, 0) > 5.0:
                log.warning(
                    "[HEALTH] THREAT '%s' lv%d (%s) at %.0fu  -  IN THREAT RANGE",
                    sp.name,
                    sp.level,
                    con,
                    dist,
                )
                self._last_threat_log[_tkey] = _now
            result["evasion_point"] = self._compute_evasion(state.x, state.y, sp.x, sp.y, sp.x, sp.y, state)
            return result

        # Skip if too far to matter
        if dist > 150:
            return result

        # Heading-based early warning
        heading_result = self._check_heading_threat(sp, state, ctx, con, dist, result)
        if heading_result is not None:
            return heading_result

        # Trajectory prediction (fallback)
        return self._check_trajectory_threat(sp, state, ctx, con, dist, result)

    def _check_heading_threat(
        self,
        sp: Any,
        state: GameState,
        ctx: AgentContext,
        con: Con,
        dist: float,
        result: dict,
    ) -> dict | None:
        """Check if spawn is charging toward the player based on heading.

        Returns the updated result dict if a heading threat was detected,
        or None if heading detection did not trigger.
        """
        if sp.speed <= 0.5:
            return None
        if ctx.combat.engaged and (con != Con.RED or sp.target_name != ""):
            return None

        mob_facing = sp.heading  # 0-512 scale
        angle_to_player = heading_to(sp.x, sp.y, state.x, state.y)
        heading_error = abs(mob_facing - angle_to_player) % 512
        if heading_error > 256:
            heading_error = 512 - heading_error

        if heading_error >= 60:
            return None

        time_to_aggro = (dist - 40) / sp.speed if sp.speed > 0 else 999.0
        if time_to_aggro >= 5.0:
            return None

        result["approaching"] = sp
        evasion = self._compute_evasion(state.x, state.y, sp.x, sp.y, state.x, state.y, state)
        result["evasion_point"] = evasion
        if ctx.combat.engaged:
            result["patrol_evade"] = True
        _tkey = f"heading_{sp.spawn_id}"
        _now = time.time()
        if _now - self._last_threat_log.get(_tkey, 0) > 5.0:
            log.warning(
                "[HEALTH] THREAT '%s' lv%d (%s) CHARGING "
                "(heading): dist=%.0f, heading_err=%.0f, "
                "speed=%.1f, eta=%.1fs  -  "
                "evasion=(%.0f,%.0f)%s",
                sp.name,
                sp.level,
                con,
                dist,
                heading_error,
                sp.speed,
                time_to_aggro,
                evasion[0],
                evasion[1],
                " [COMBAT EVADE]" if ctx.combat.engaged else "",
            )
            self._last_threat_log[_tkey] = _now
        result["skip_traj"] = True
        return result

    def _check_trajectory_threat(
        self,
        sp: Any,
        state: GameState,
        ctx: AgentContext,
        con: Con,
        dist: float,
        result: dict,
    ) -> dict:
        """Check if spawn's trajectory will pass near the player.

        Returns the (possibly updated) result dict.
        """
        if sp.velocity_x != 0.0 or sp.velocity_y != 0.0:
            vx = sp.velocity_x
            vy = sp.velocity_y
        else:
            prev_pos = self._threat_positions.get(sp.spawn_id)
            if prev_pos is None:
                return result
            vx = sp.x - prev_pos.x
            vy = sp.y - prev_pos.y

        speed = math.hypot(vx, vy)

        if speed < 0.5:
            return result

        if ctx.combat.engaged:
            if con != Con.RED or sp.target_name != "" or speed < _PATROL_MIN_SPEED:
                return result

        predict_x = sp.x + vx * 50
        predict_y = sp.y + vy * 50

        evade_radius = _PATROL_EVADE_RADIUS if ctx.combat.engaged else 60
        closest = self._closest_point_on_line(state.x, state.y, sp.x, sp.y, predict_x, predict_y)
        closest_dist = state.pos.dist_to(closest)

        if closest_dist < evade_radius and dist < 120:
            result["approaching"] = sp
            evasion = self._compute_evasion(state.x, state.y, sp.x, sp.y, predict_x, predict_y, state)
            result["evasion_point"] = evasion
            if ctx.combat.engaged:
                result["patrol_evade"] = True
            log.warning(
                "[HEALTH] THREAT '%s' lv%d (%s) pathing toward us: "
                "dist=%.0f, closest_pass=%.0f, speed=%.1f  -  "
                "evasion=(%.0f,%.0f)%s",
                sp.name,
                sp.level,
                con,
                dist,
                closest_dist,
                speed * 10,
                evasion[0],
                evasion[1],
                " [COMBAT EVADE]" if ctx.combat.engaged else "",
            )

        return result

    def _check_threats(self, state: GameState, ctx: AgentContext) -> None:
        """Scan for YELLOW/RED cons, predict trajectory, compute evasion."""
        # Reset flags each tick
        ctx.threat.approaching_threat = None
        ctx.threat.imminent_threat = False
        ctx.threat.imminent_threat_con = ""
        ctx.threat.evasion_point = None
        ctx.threat.patrol_evade = False

        current_positions: dict[int, Point] = {}

        for sp in state.spawns:
            if not sp.is_npc or sp.hp_current <= 0:
                continue
            if is_pet(sp):
                continue

            result = self._evaluate_single_spawn_threat(sp, state, ctx)
            if result is None:
                continue

            if "position" in result:
                current_positions[sp.spawn_id] = result["position"]

            if result.get("imminent"):
                ctx.threat.imminent_threat = True
                ctx.threat.imminent_threat_con = result["con"]
                ctx.threat.evasion_point = result.get("evasion_point")
                continue

            if result.get("approaching"):
                ctx.threat.approaching_threat = result["approaching"]
                ctx.threat.evasion_point = result.get("evasion_point")
                if result.get("patrol_evade"):
                    ctx.threat.patrol_evade = True

        self._threat_positions = current_positions
        # Prune stale throttle entries to prevent unbounded growth
        _now = time.time()
        if len(self._last_threat_log) > 50:
            self._last_threat_log = {k: v for k, v in self._last_threat_log.items() if _now - v < 60.0}

    def _closest_point_on_line(
        self,
        px: float,
        py: float,
        lx1: float,
        ly1: float,
        lx2: float,
        ly2: float,
        z: float = 0.0,
    ) -> Point:
        """Find the point on line segment (lx1,ly1)->(lx2,ly2) closest to (px,py)."""
        dx, dy = lx2 - lx1, ly2 - ly1
        len_sq = dx * dx + dy * dy
        if len_sq < 0.01:
            return Point(lx1, ly1, z)
        t = max(0, min(1, ((px - lx1) * dx + (py - ly1) * dy) / len_sq))
        return Point(lx1 + t * dx, ly1 + t * dy, z)

    def _compute_evasion(
        self,
        player_x: float,
        player_y: float,
        mob_x: float,
        mob_y: float,
        mob_dest_x: float,
        mob_dest_y: float,
        state: GameState | None = None,
    ) -> Point:
        """Compute a sidestep point perpendicular to the npc's path.

        Moves 60u perpendicular to the npc's travel direction,
        picking the side that's farther from the npc.
        """
        pz = state.z if state else 0.0
        # Npc's direction of travel
        dx = mob_dest_x - mob_x
        dy = mob_dest_y - mob_y
        path_len = math.hypot(dx, dy)
        if path_len < 0.1:
            # Npc not moving  -  just move directly away
            away_x = player_x - mob_x
            away_y = player_y - mob_y
            away_len = math.hypot(away_x, away_y)
            if away_len < 0.1:
                return Point(player_x + 60, player_y, pz)
            return Point(player_x + away_x / away_len * 60, player_y + away_y / away_len * 60, pz)

        # Perpendicular directions
        perp_x = -dy / path_len
        perp_y = dx / path_len

        # Two sidestep options
        opt_a = Point(player_x + perp_x * 60, player_y + perp_y * 60, pz)
        opt_b = Point(player_x - perp_x * 60, player_y - perp_y * 60, pz)

        # Pick the one farther from the npc
        dist_a = opt_a.dist_to(Point(mob_x, mob_y, opt_a.z))
        dist_b = opt_b.dist_to(Point(mob_x, mob_y, opt_b.z))

        return opt_a if dist_a > dist_b else opt_b
