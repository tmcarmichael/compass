"""Camp scoring, selection, and progression.

Extracted from agent.py (A-3) to keep build_context under 80 LOC.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from core.constants import LEVEL_RANGE_PENALTY
from core.types import DangerZone, Point
from util.log_tiers import EVENT

if TYPE_CHECKING:
    from brain.context import AgentContext

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Camp scoring
# ------------------------------------------------------------------

FALLBACK_PENALTY = 10_000  # camp scoring: penalty for fallback camps


def score_camp(camp: dict[str, Any], player_pos: Point, player_level: int) -> float:
    """Score a camp for selection.  Lower is better.

    Camps whose level_range contains the player level are strongly preferred.
    Among those, the closest camp wins.  Camps outside the level range get a
    large penalty (10_000 units) so they only win when nothing else fits.
    Camps marked ``fallback = true`` get an additional penalty so they are
    only chosen when no primary camp fits the level range.
    """
    cc = camp.get("center", {})
    cx, cy = float(cc.get("x", 0.0)), float(cc.get("y", 0.0))
    # LINEAR camps without explicit center: compute from patrol waypoints
    if cx == 0.0 and cy == 0.0 and camp.get("camp_type") == "linear":
        wps = camp.get("patrol_waypoints", [])
        if len(wps) >= 2:
            mid_idx = len(wps) // 2
            if len(wps) % 2 == 0:
                # Even: average the two middle waypoints
                a, b = wps[mid_idx - 1], wps[mid_idx]
                cx = (a["x"] + b["x"]) / 2
                cy = (a["y"] + b["y"]) / 2
            else:
                cx = wps[mid_idx]["x"]
                cy = wps[mid_idx]["y"]
    dist = ((player_pos.x - cx) ** 2 + (player_pos.y - cy) ** 2) ** 0.5

    # Fallback camps only win when nothing else fits
    penalty = FALLBACK_PENALTY if camp.get("fallback", False) else 0.0

    lr = camp.get("level_range", [])
    if lr and player_level > 0:
        lo, hi = lr[0], lr[1]
        if lo <= player_level <= hi:
            return float(dist + penalty)  # level fit -- proximity + fallback penalty
        # Outside range: penalty proportional to how far off
        gap = min(abs(player_level - lo), abs(player_level - hi))
        return float(dist + gap * LEVEL_RANGE_PENALTY + penalty)
    return float(dist + penalty)  # no level_range defined -- proximity + fallback


def select_camp(
    camps: list[dict[str, Any]], player_pos: Point, player_level: int, fallback_name: str = ""
) -> dict[str, Any]:
    """Pick the best camp from *camps* by level fit + proximity.

    If player position is unknown (0, 0), falls back to *fallback_name*
    or the first camp.  Returns an empty dict when *camps* is empty.
    """
    if not camps:
        return {}

    camp = camps[0]

    if player_pos.x != 0.0 or player_pos.y != 0.0:
        best_score = float("inf")
        for c in camps:
            cc = c.get("center", {})
            if cc.get("x", 0.0) == 0.0 and cc.get("y", 0.0) == 0.0:
                continue
            s = score_camp(c, player_pos, player_level)
            lr = c.get("level_range", [])
            log.debug(
                "[TRAVEL]   Camp '%s': score=%.0f (dist=%.0f, level_range=%s)",
                c.get("name", "?"),
                s,
                ((player_pos.x - cc.get("x", 0)) ** 2 + (player_pos.y - cc.get("y", 0)) ** 2) ** 0.5,
                lr,
            )
            if s < best_score:
                best_score = s
                camp = c
        log.log(
            EVENT,
            "[TRAVEL] Selected camp: '%s' (score=%.0f, player_level=%d)",
            camp.get("name", "unknown"),
            best_score,
            player_level,
        )
    else:
        # Fallback: match by name
        if fallback_name:
            for c in camps:
                if c.get("name") == fallback_name:
                    camp = c
                    break
            else:
                log.warning("[TRAVEL] Camp '%s' not found, using first camp", fallback_name)
        log.log(EVENT, "[TRAVEL] Active camp: '%s' (from config)", camp.get("name", "unknown"))

    return camp


# ------------------------------------------------------------------
# Camp application
# ------------------------------------------------------------------


def apply_camp(ctx: AgentContext, camp: dict[str, Any]) -> None:
    """Apply a camp dict to ctx spatial fields. Reusable for camp switches."""
    camp_center = camp.get("center", {})
    safe_spot = camp.get("safe_spot", camp_center)
    flee_spot = camp.get("flee_spot", safe_spot)

    # -- LINEAR camp support --
    from core.types import CampType

    camp_type_str = camp.get("camp_type", "circular")
    ctx.camp.camp_type = camp_type_str

    # Parse patrol waypoints for LINEAR camps
    ctx.camp.patrol_waypoints = []
    for wp in camp.get("patrol_waypoints", []):
        ctx.camp.patrol_waypoints.append(Point(wp["x"], wp["y"], wp.get("z", 0.0)))
    ctx.camp.corridor_width = camp.get("corridor_width", 200.0)

    is_linear = camp_type_str == CampType.LINEAR and len(ctx.camp.patrol_waypoints) >= 2

    if is_linear and not camp_center:
        # Auto-compute center from path midpoint
        mid = ctx.camp.point_along_path(0.5)
        camp_center = {"x": mid[0], "y": mid[1]}

    ctx.camp.camp_x = camp_center.get("x", 0.0)
    ctx.camp.camp_y = camp_center.get("y", 0.0)
    ctx.camp.guard_x = safe_spot.get("x", 0.0)
    ctx.camp.guard_y = safe_spot.get("y", 0.0)
    ctx.camp.flee_x = flee_spot.get("x", 0.0)
    ctx.camp.flee_y = flee_spot.get("y", 0.0)
    ctx.camp.hunt_min_dist = camp.get("hunt_min_dist", 50.0)

    if is_linear:
        # For LINEAR: roam_radius = corridor_width so all existing
        # roam_radius comparisons work without per-consumer changes.
        ctx.camp.roam_radius = ctx.camp.corridor_width
        path_len = ctx.camp.path_total_length()
        ctx.camp.hunt_max_dist = camp.get("hunt_max_dist", path_len + ctx.camp.corridor_width)
        # Set guard to path midpoint so guard_dist checks stay sane
        mid = ctx.camp.point_along_path(0.5)
        ctx.camp.guard_x = mid[0]
        ctx.camp.guard_y = mid[1]
        log.info(
            "[TRAVEL] LINEAR camp: %d waypoints, corridor=%.0f, path=%.0fu",
            len(ctx.camp.patrol_waypoints),
            ctx.camp.corridor_width,
            path_len,
        )
    else:
        ctx.camp.roam_radius = camp.get("roam_radius", 250.0)
        ctx.camp.hunt_max_dist = camp.get("hunt_max_dist", 300.0)

    # Zone boundaries (wander will not go past these)
    ctx.camp.bounds_x_min = camp.get("bounds_x_min", None)
    ctx.camp.bounds_x_max = camp.get("bounds_x_max", None)
    ctx.camp.bounds_y_min = camp.get("bounds_y_min", None)
    ctx.camp.bounds_y_max = camp.get("bounds_y_max", None)
    bounds = [
        b
        for b in [ctx.camp.bounds_x_min, ctx.camp.bounds_x_max, ctx.camp.bounds_y_min, ctx.camp.bounds_y_max]
        if b is not None
    ]
    if bounds:
        log.info(
            "[TRAVEL] Zone bounds: x=[%s..%s] y=[%s..%s]",
            ctx.camp.bounds_x_min,
            ctx.camp.bounds_x_max,
            ctx.camp.bounds_y_min,
            ctx.camp.bounds_y_max,
        )

    # Danger points
    ctx.camp.danger_points = []
    for dp in camp.get("danger_points", []):
        ctx.camp.danger_points.append(
            DangerZone(dp["x"], dp["y"], dp.get("z", 0.0), dp["min_distance"], dp.get("name", "?"))
        )
        log.info(
            "[TRAVEL] Danger point: '%s' at (%.0f, %.0f) min_dist=%.0f",
            dp.get("name", "?"),
            dp["x"],
            dp["y"],
            dp["min_distance"],
        )

    # Flee waypoints
    ctx.camp.flee_waypoints = []
    for wp in camp.get("flee_waypoints", []):
        ctx.camp.flee_waypoints.append(Point(wp["x"], wp["y"], wp.get("z", 0.0)))
    if ctx.camp.flee_waypoints:
        log.info("[TRAVEL] Flee waypoints: %d points to zoneline", len(ctx.camp.flee_waypoints))

    # Avoid npcs
    from perception.combat_eval import set_zone_avoid_mobs

    avoid = set()
    for mob_name in camp.get("avoid_mobs", []):
        avoid.add(mob_name.lower())
        log.info("[TRAVEL] Avoid npc: %s", mob_name)
    set_zone_avoid_mobs(frozenset(avoid))

    ctx.zone.active_camp_name = camp.get("name", "unknown")
    log.info("[TRAVEL] Camp applied: '%s'", ctx.zone.active_camp_name)


# ------------------------------------------------------------------
# Camp progression (on level-up)
# ------------------------------------------------------------------


def check_camp_progression(ctx: AgentContext, player_level: int, player_pos: Point) -> str | None:
    """Re-evaluate camps after a level-up.  Returns new camp name if switched,
    None if current camp is still best.

    Called from the brain loop on level-up events.
    """
    camps = ctx.zone.zone_camps
    if not camps or len(camps) < 2:
        return None

    current = ctx.zone.active_camp_name
    best_score = float("inf")
    best_camp = None
    for c in camps:
        cc = c.get("center", {})
        if cc.get("x", 0.0) == 0.0 and cc.get("y", 0.0) == 0.0:
            continue
        s = score_camp(c, player_pos, player_level)
        log.info(
            "[TRAVEL]   Camp re-eval '%s': score=%.0f (level_range=%s)",
            c.get("name", "?"),
            s,
            c.get("level_range", []),
        )
        if s < best_score:
            best_score = s
            best_camp = c

    if not best_camp or best_camp.get("name") == current:
        log.log(EVENT, "[TRAVEL] Camp progression: staying at '%s'", current)
        return None

    new_name = str(best_camp.get("name", "unknown"))
    log.log(EVENT, "[TRAVEL] Camp progression: '%s' -> '%s' (level %d)", current, new_name, player_level)
    apply_camp(ctx, best_camp)
    return new_name
