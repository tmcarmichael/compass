"""Flee urgency computation.

Extracted from brain.rules.survival so that routines.base can import
flee thresholds and the urgency function without creating a back-edge
in the import DAG (routines -> brain.rules is forbidden).

Both brain.rules.survival and routines.base import from this module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.context_views import SurvivalView
    from perception.state import GameState

log = logging.getLogger(__name__)

# -- Flee urgency thresholds (hysteresis) --
FLEE_URGENCY_ENTER = 0.65  # start fleeing at this urgency
FLEE_URGENCY_EXIT = 0.35  # stop fleeing below this


# -- Private urgency axes ----------------------------------------------------


def _fight_winnable(state: GameState) -> bool:
    """True if player can finish the current fight without a pet.

    Conditions: player HP > 60% AND target npc HP < 50%.
    At these thresholds a few Lifespikes will finish the npc.
    """
    t = state.target
    if not t or t.hp_max <= 0:
        log.debug("[DECISION] _fight_winnable: no valid target")
        return False
    mob_hp: float = t.hp_current / max(t.hp_max, 1)
    winnable: bool = state.hp_pct > 0.60 and mob_hp < 0.50
    return winnable


def _count_damaged_npcs(ctx: SurvivalView, state: GameState) -> int:
    """Count damaged NPCs within 40u (for add detection)."""
    from brain.rules.skip_log import damaged_npcs_near

    return len(damaged_npcs_near(ctx, state, state.pos, 40))


def _mob_attacking_player(state: GameState) -> bool:
    """Return True if any NPC within 30u is targeting the player."""
    for sp in state.spawns:
        if (
            sp.is_npc
            and sp.hp_current > 0
            and sp.target_name == state.name
            and state.pos.dist_to(sp.pos) < 30
        ):
            return True
    return False


def _urgency_hp(hp_pct: float) -> float:
    """HP axis: power curve scaled so ~35% HP yields ~0.65."""
    result: float = ((1.0 - hp_pct) ** 1.8) * 1.45
    return result


def _urgency_pet_died(ctx: SurvivalView, state: GameState, in_combat: bool) -> float:
    """Pet died mid-combat with unwinnable fight: +0.4."""
    if in_combat and ctx.pet.just_died() and not _fight_winnable(state):
        return 0.4
    return 0.0


def _urgency_adds(ctx: SurvivalView, state: GameState, in_combat: bool) -> float:
    """Extra damaged NPCs nearby: +0.15 per add."""
    if not in_combat:
        return 0.0
    count = _count_damaged_npcs(ctx, state)
    return 0.15 * max(0, count - 1)


def _urgency_target_dying(state: GameState) -> float:
    """Target nearly dead (<15% HP): -0.25 to finish the defeat."""
    t = state.target
    if t and t.is_npc and t.hp_max > 0 and t.hp_current > 0:
        if t.hp_current / t.hp_max < 0.15:
            return -0.25
    return 0.0


def _urgency_learned_danger(ctx: SurvivalView, in_combat: bool) -> float:
    """Learned danger from FightHistory (>0.7): +0.2."""
    fh = ctx.fight_history
    if not fh or not in_combat:
        return 0.0
    name = ctx.defeat_tracker.last_fight_name
    if not name:
        return 0.0
    danger = fh.learned_danger(name)
    return 0.2 if danger is not None and danger > 0.7 else 0.0


def _urgency_pet_hp_low(ctx: SurvivalView) -> float:
    """Pet HP below 30%: +0.1."""
    world = ctx.world
    if ctx.pet.alive and world:
        pet_hp = world.pet_hp_pct
        if 0 <= pet_hp < 0.30:
            return 0.1
    return 0.0


def _urgency_red_threat(ctx: SurvivalView) -> float:
    """RED imminent threat: +0.5."""
    if ctx.threat.imminent_threat and ctx.threat.imminent_threat_con == "red":
        return 0.5
    return 0.0


def _urgency_mob_on_player(ctx: SurvivalView, state: GameState) -> float:
    """No pet + NPC attacking player: +0.5."""
    if not ctx.pet.alive and not ctx.combat.engaged and _mob_attacking_player(state):
        return 0.5
    return 0.0


# -- Public API ---------------------------------------------------------------


def compute_flee_urgency(ctx: SurvivalView, state: GameState) -> float:
    """Composite flee urgency score (0.0 = safe, 1.0 = critical).

    Eight axes contribute additively, then clamp to [0, 1].
    """
    in_combat = ctx.in_active_combat

    hp = _urgency_hp(state.hp_pct)
    pet_died = _urgency_pet_died(ctx, state, in_combat)
    adds = _urgency_adds(ctx, state, in_combat)
    target_dying = _urgency_target_dying(state)
    danger = _urgency_learned_danger(ctx, in_combat)
    pet_low = _urgency_pet_hp_low(ctx)
    red = _urgency_red_threat(ctx)
    mob_on = _urgency_mob_on_player(ctx, state)

    result: float = max(0.0, min(1.0, hp + pet_died + adds + target_dying + danger + pet_low + red + mob_on))

    if result >= FLEE_URGENCY_ENTER:
        factors = []
        if hp > 0.01:
            factors.append(f"hp={hp:.2f}")
        if pet_died > 0:
            factors.append("pet_died")
        if adds > 0:
            factors.append(f"extra_npcs={adds / 0.15:.0f}")
        if red > 0:
            factors.append("RED_threat")
        if mob_on > 0:
            factors.append("mob_on_player")
        log.info(
            "[DECISION] Flee urgency: %.3f [%s] combat=%s",
            result,
            " ".join(factors) if factors else "hp_only",
            in_combat,
        )

    return result
