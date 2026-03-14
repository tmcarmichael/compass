"""Target filtering: decide whether a specific npc is acceptable to engage.

Stateless functions that evaluate a candidate target against game state,
agent context, and zone policy. Used by AcquireRoutine to validate Tab results.

Distinct from combat_eval.py which handles con colors and raw target lists --
these functions answer "should we fight THIS npc right NOW?"
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from perception.combat_eval import con_color, get_avoid_names, is_valid_target

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.world.model import MobProfile

from core.constants import (
    DAMAGED_TARGET_HP_THRESHOLD,
    GUARD_CHECK_RADIUS,
    NEARBY_AGGRO_RADIUS,
    PLAYER_MOB_PROXIMITY,
    SCAN_RADIUS,
    SITTING_MANA_REGEN_RATE,
    SOCIAL_NPC_RADIUS,
    THREAT_RADIUS_BASE,
    THREAT_RADIUS_PER_LEVEL,
)
from perception.queries import count_nearby_npcs, count_nearby_social
from perception.state import GameState, SpawnData

log = logging.getLogger(__name__)


def guard_nearby(target: SpawnData, state: GameState) -> bool:
    """Check if a guard/avoid NPC is within guard-check radius of target."""
    for spawn in state.spawns:
        if not spawn.is_npc:
            continue
        if any(avoid in spawn.name for avoid in get_avoid_names()):
            d = target.pos.dist_to(spawn.pos)
            if d < GUARD_CHECK_RADIUS:
                return True
    return False


def social_npc_count(target: SpawnData, state: GameState, ctx: AgentContext | None) -> int:
    """Count how many social-group members are near the target npc."""
    if not ctx or not ctx.zone.social_mob_group:
        return 0
    result: int = count_nearby_social(target, state, ctx.zone.social_mob_group, SOCIAL_NPC_RADIUS)
    return result


def nearby_npc_count(target: SpawnData, state: GameState, radius: float = NEARBY_AGGRO_RADIUS) -> int:
    """Count living NPCs near the target (regardless of social group).

    Npcs clustered within threat radius will likely add when we pull.
    """
    result: int = count_nearby_npcs(state, target.pos, radius, exclude_id=target.spawn_id)
    return result


def estimate_exposure(state: GameState, profile: MobProfile, ctx: AgentContext | None) -> float:
    """Estimate total exposure time: fight + mandatory rest if mana low.

    If projected post-fight mana drops below the rest entry threshold,
    the agent will sit to regen. This rest period is vulnerable to patrols.
    Returns total seconds of exposure (fight + rest).
    """
    fight_time: float = profile.fight_duration_est
    if not ctx or state.mana_max <= 0:
        return fight_time

    # Project mana after fight
    projected_mana = state.mana_current - profile.mana_cost_est
    projected_pct = projected_mana / state.mana_max

    # Will rest be needed?
    rest_entry = ctx.rest_mana_entry
    rest_exit = ctx.rest_mana_threshold
    if projected_pct >= rest_entry:
        return fight_time  # no rest needed

    # Estimate rest duration: mana regen ~2/tick sitting (6s tick = ~0.33/s)
    mana_needed: float = (rest_exit * state.mana_max) - max(projected_mana, 0)
    rest_time: float = mana_needed / SITTING_MANA_REGEN_RATE

    total: float = fight_time + rest_time
    return total


def _threat_blocks_path(target: SpawnData, state: GameState, ctx: AgentContext, dist: float) -> bool:
    """Return True if a threat's avoidance zone blocks the path to target."""
    if not ctx.world:
        return False
    for tp in ctx.world.threats_within(200):
        # Skip the target itself -- a npc can't block the path to itself
        if tp.spawn.spawn_id == target.spawn_id:
            continue
        # Check if threat is between us and the target
        threat_to_target = tp.spawn.pos.dist_to(target.pos)
        threat_to_player = tp.spawn.pos.dist_to(state.pos)
        threat_radius = tp.spawn.level * THREAT_RADIUS_PER_LEVEL + THREAT_RADIUS_BASE
        # Threat is "in the way" if it's closer to both player and
        # target than the avoidance radius, and closer to target
        # than we are (we'd walk toward it)
        if threat_to_target < threat_radius and threat_to_player < dist + threat_radius:
            log.info(
                "[TARGET] Acquire: REJECT '%s' - threat '%s'(%s lv%d) blocks path at %.0fu (radius=%.0f)",
                target.name,
                tp.spawn.name,
                tp.con if hasattr(tp, "con") else "?",
                tp.spawn.level,
                threat_to_target,
                threat_radius,
            )
            return True
    return False


def is_acceptable_target(target: SpawnData, state: GameState, ctx: AgentContext | None) -> bool:
    """Real-time validation of a tab target.

    Checks con color, HP, distance, pets, guards, recent defeats,
    and threat path blocking.
    """
    if not is_valid_target(target, state.level):
        log.info("[TARGET] Acquire: REJECT '%s' - is_valid_target=False", target.name)
        return False

    if ctx and ctx.zone.target_cons:
        con = con_color(state.level, target.level)
        if con not in ctx.zone.target_cons:
            log.info("[TARGET] Acquire: REJECT '%s' - con=%s not in target_cons", target.name, con)
            return False

    if target.hp_current < target.hp_max:
        if target.hp_current < target.hp_max * DAMAGED_TARGET_HP_THRESHOLD:
            log.info(
                "[TARGET] Acquire: REJECT '%s' - damaged HP=%d/%d",
                target.name,
                target.hp_current,
                target.hp_max,
            )
            return False

    # Skip npcs near other players (claimed, about to engage, or being fought)
    for spawn in state.spawns:
        if (
            spawn.is_player
            and spawn.name != state.name
            and target.pos.dist_to(spawn.pos) < PLAYER_MOB_PROXIMITY
        ):
            log.info(
                "[TARGET] Acquire: REJECT '%s' - player '%s' within %.0fu of npc",
                target.name,
                spawn.name,
                target.pos.dist_to(spawn.pos),
            )
            return False

    dist = state.pos.dist_to(target.pos)
    if dist > SCAN_RADIUS:
        log.info("[TARGET] Acquire: REJECT '%s' - dist=%.0f > SCAN_RADIUS=%d", target.name, dist, SCAN_RADIUS)
        return False

    if ctx:
        ctx.defeat_tracker.clear_recent_kills()
        recent_ids = {sid for sid, _ in ctx.defeat_tracker.recent_kills}
        if target.spawn_id in recent_ids:
            log.info("[TARGET] Acquire: REJECT '%s' - recently defeated", target.name)
            return False

    if guard_nearby(target, state):
        log.info("[TARGET] Acquire: REJECT '%s' - guard nearby", target.name)
        return False

    # Reject targets where path crosses a threat's avoidance zone
    if ctx and _threat_blocks_path(target, state, ctx, dist):
        return False

    return True
