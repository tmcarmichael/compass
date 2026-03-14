"""Combat strategy selection based on level, con color, and learned data.

Four strategies that define how the combat routine prioritizes spells:
- PET_TANK (L1-7): Pet does damage, conserve mana
- PET_AND_DOT (L8-15): Pet + DoT + conditional lifetap/DD (current behavior)
- FEAR_KITE (L16-48): Fear -> DoT stack -> lifetap -> re-Fear cycle
- ENDGAME (L49-60): Full DoT stack, powerful lifetaps, Lich sustain

Context-aware overrides within level brackets use con color and learned
danger scores to downgrade for trivial npcs (save mana) or upgrade for
known-dangerous npcs (fear kite early).

Forced exploration: every EXPLORE_INTERVAL encounters with the same
strategy per entity type, one encounter uses an alternative strategy
to gather comparative fitness data for learning.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from eq.loadout import SpellRole, get_spell_by_role
from perception.combat_eval import Con
from util.log_tiers import VERBOSE

log = logging.getLogger(__name__)

# -- Forced Exploration -------------------------------------------------------
# Every EXPLORE_INTERVAL encounters with the same best strategy, force one
# encounter with an alternative to gather comparative data.
EXPLORE_INTERVAL = 20

# entity_type -> {strategy -> count since last exploration}
_explore_counts: dict[str, dict[str, int]] = {}

# Flag set when the current encounter is exploratory (read by combat routine
# to mark the fight record, so gradient tuner can skip it).
# Mutable container avoids the need for a `global` statement in writers.
_exploration_state: dict[str, bool] = {"active": False}


class CombatStrategy(StrEnum):
    PET_TANK = "pet_tank"
    PET_AND_DOT = "pet_and_dot"
    FEAR_KITE = "fear_kite"
    ENDGAME = "endgame"


def select_strategy(
    level: int,
    con: Con | None = None,
    danger: float | None = None,
    has_fear: bool = False,
    pet_death_rate: float | None = None,
) -> CombatStrategy:
    """Select combat strategy based on level, con color, and learned danger.

    Hard gates enforce spell availability (no fear kite without fear spell).
    Within each level bracket, context-aware overrides adjust strategy:
    - Trivial npcs (LIGHT_BLUE/GREEN at L16+) downgrade to PET_AND_DOT
    - Safe blue npcs (danger < 0.2) downgrade to PET_AND_DOT
    - Known-dangerous npcs (danger > 0.6, WHITE/YELLOW) upgrade to FEAR_KITE
      if fear spell is available (even before L16)
    """
    # Endgame always takes priority
    if level >= 49:
        return CombatStrategy.ENDGAME

    # Hard gate: below L8 is pet-tank only (no DoTs yet)
    if level < 8:
        return CombatStrategy.PET_TANK

    # L8-15: PET_AND_DOT default, with early fear kite for dangerous npcs
    if level < 16:
        if danger is not None and danger > 0.6 and con in (Con.WHITE, Con.YELLOW) and has_fear:
            log.info(
                "[COMBAT] Strategy override: PET_AND_DOT -> FEAR_KITE (danger=%.2f, con=%s, L%d)",
                danger,
                con if con else "?",
                level,
            )
            return CombatStrategy.FEAR_KITE
        # Learned pet death rate > 30% -> fear kite to protect pet
        if pet_death_rate is not None and pet_death_rate > 0.30 and has_fear:
            log.info(
                "[COMBAT] Strategy override: PET_AND_DOT -> FEAR_KITE (pet_death_rate=%.0f%%, L%d)",
                pet_death_rate * 100,
                level,
            )
            return CombatStrategy.FEAR_KITE
        return CombatStrategy.PET_AND_DOT

    # L16+: FEAR_KITE default (if fear available), with downgrades
    fear = get_spell_by_role(SpellRole.FEAR)
    level_default = CombatStrategy.FEAR_KITE if fear else CombatStrategy.PET_AND_DOT

    # Downgrade: trivial npcs (LIGHT_BLUE/GREEN) -> PET_AND_DOT (save mana)
    if con in (Con.LIGHT_BLUE, Con.GREEN):
        if level_default != CombatStrategy.PET_AND_DOT:
            log.info(
                "[COMBAT] Strategy override: %s -> PET_AND_DOT (trivial npc, con=%s, L%d)",
                level_default,
                con if con else "?",
                level,
            )
        return CombatStrategy.PET_AND_DOT

    # Downgrade: safe blue npcs (learned danger < 0.2) -> PET_AND_DOT
    if danger is not None and danger < 0.2 and con == Con.BLUE:
        if level_default != CombatStrategy.PET_AND_DOT:
            log.info(
                "[COMBAT] Strategy override: %s -> PET_AND_DOT (safe blue, danger=%.2f, L%d)",
                level_default,
                danger,
                level,
            )
        return CombatStrategy.PET_AND_DOT

    # Upgrade: learned pet death rate > 30% -> FEAR_KITE to protect pet
    if (
        pet_death_rate is not None
        and pet_death_rate > 0.30
        and fear
        and level_default != CombatStrategy.FEAR_KITE
    ):
        log.info(
            "[COMBAT] Strategy override: %s -> FEAR_KITE (pet_death_rate=%.0f%%, L%d)",
            level_default,
            pet_death_rate * 100,
            level,
        )
        return CombatStrategy.FEAR_KITE

    return level_default


def select_strategy_with_exploration(
    entity_name: str,
    level: int,
    con: Con | None = None,
    danger: float | None = None,
    has_fear: bool = False,
    pet_death_rate: float | None = None,
) -> tuple[CombatStrategy, bool]:
    """Select strategy with forced exploration for learning.

    Returns (strategy, is_exploratory). When is_exploratory is True,
    the encounter outcome should not penalize scoring weights.
    """
    best = select_strategy(level, con, danger, has_fear, pet_death_rate)

    # Track encounters per strategy per entity type
    if entity_name not in _explore_counts:
        _explore_counts[entity_name] = {}
    counts = _explore_counts[entity_name]
    counts[best.value] = counts.get(best.value, 0) + 1

    # Check if exploration is due
    if counts.get(best.value, 0) >= EXPLORE_INTERVAL:
        alt = _pick_alternative(best, level, has_fear)
        if alt is not None:
            counts[best.value] = 0
            _exploration_state["active"] = True
            log.log(VERBOSE, "[COMBAT] Strategy sample: %s -> %s for %s", best.value, alt.value, entity_name)
            return alt, True

    _exploration_state["active"] = False
    return best, False


def is_exploration_active() -> bool:
    """True if the current encounter is exploratory (for fight record flagging)."""
    return _exploration_state["active"]


def evaluate_mid_fight_switch(
    current: CombatStrategy,
    time_in_combat: float,
    pet_hp: float,
    pet_alive: bool,
    fear_available: bool,
    fear_mana_cost: int,
    player_mana: int,
    player_level: int,
    learned_duration: float | None,
    fear_phase: int | None,
    pet_death_rate: float | None,
    fight_count: int,
) -> tuple[CombatStrategy, str] | None:
    """Pure function: evaluate whether combat strategy should switch mid-fight.

    Returns (new_strategy, reason) if a switch is warranted, None otherwise.
    Extracted from CombatRoutine._evaluate_strategy_switch for testability.
    """
    if time_in_combat < 10.0:
        return None

    # Pet dying fast -> switch to fear_kite if fear available
    if (
        current in (CombatStrategy.PET_TANK, CombatStrategy.PET_AND_DOT)
        and pet_hp >= 0
        and pet_hp < 0.35
        and pet_alive
        and fear_available
        and player_mana >= fear_mana_cost
    ):
        return CombatStrategy.FEAR_KITE, f"pet HP critical ({pet_hp * 100:.0f}%)"

    # Fight dragging (>2x learned average) -> escalate to pet_and_dot
    if current == CombatStrategy.PET_TANK and learned_duration is not None:
        if time_in_combat > learned_duration * 2.0 and time_in_combat > 20:
            return (
                CombatStrategy.PET_AND_DOT,
                f"fight dragging ({time_in_combat:.0f}s > 2x learned {learned_duration:.0f}s)",
            )

    # Fear resisted -> fall back
    if current == CombatStrategy.FEAR_KITE and fear_phase == 3:
        fallback = CombatStrategy.PET_AND_DOT if player_level >= 8 else CombatStrategy.PET_TANK
        return fallback, "fear resisted/failed"

    # High pet death rate from history -> switch to fear kite
    if (
        pet_death_rate is not None
        and fight_count >= 5
        and pet_death_rate > 0.3
        and fear_available
        and current != CombatStrategy.FEAR_KITE
        and player_mana >= fear_mana_cost
        and time_in_combat > 15.0
    ):
        return (
            CombatStrategy.FEAR_KITE,
            f"high pet death rate ({pet_death_rate:.0%}) from history ({fight_count} fights)",
        )

    return None


def _pick_alternative(best: CombatStrategy, level: int, has_fear: bool) -> CombatStrategy | None:
    """Pick a viable alternative strategy different from best.

    Returns None if no alternative is viable at this level.
    """
    # Candidates ordered by preference (most different from best first)
    candidates = [
        CombatStrategy.PET_AND_DOT,
        CombatStrategy.FEAR_KITE,
        CombatStrategy.PET_TANK,
        CombatStrategy.ENDGAME,
    ]

    for c in candidates:
        if c == best:
            continue
        # Viability checks
        if c == CombatStrategy.PET_TANK and level >= 16:
            continue  # too weak for high levels
        if c == CombatStrategy.FEAR_KITE and not has_fear:
            continue  # can't fear without the spell
        if c == CombatStrategy.ENDGAME and level < 49:
            continue  # endgame spells not available
        return c
    return None
