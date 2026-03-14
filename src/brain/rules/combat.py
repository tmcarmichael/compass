"""Combat rules: IN_COMBAT, ENGAGE_ADD, ACQUIRE, PULL."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from brain.rule_def import Consideration
from brain.rules.skip_log import SkipLog
from brain.scoring.curves import linear
from core.features import flags
from core.types import Con, LootMode, PlanType, Point
from core.types import normalize_entity_name as normalize_mob_name
from eq.loadout import Spell, SpellRole, get_spell_by_role
from perception.combat_eval import con_color
from perception.state import GameState
from routines.acquire import AcquireRoutine
from routines.combat import CombatRoutine
from routines.engage_add import EngageAddRoutine
from routines.pull import PullRoutine

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.decision import Brain
    from core.types import ReadStateFn


@dataclass
class _CombatRuleState:
    """Mutable state shared across combat rule closures."""

    add_first_seen: float = 0.0


log = logging.getLogger(__name__)
_skip = SkipLog(log)


# -- Module-level extracted condition/score functions --


def _should_combat(
    state: GameState,
    ctx: AgentContext,
) -> bool:
    """Pure predicate: should the IN_COMBAT routine activate?

    No ctx mutations. Threat detection and auto-engage candidate
    identification are handled by scan_auto_engage() in the pre-rule
    pipeline. Add detection is handled by detect_adds().
    """
    # Standard engaged combat
    if ctx.combat.engaged:
        # Target visible and alive
        if state.target is not None and state.target.is_npc and state.target.hp_current > 0:
            return True
        # Engaged but target gone/dead -- pet may still be fighting
        # Check for any damaged NPC nearby (pet's current fight)
        if ctx.pet.alive:
            for sp in state.spawns:
                if sp.is_npc and sp.hp_current > 0 and sp.hp_current < sp.hp_max:
                    d = state.pos.dist_to(sp.pos)
                    if d < 100:
                        return True
        _skip("Combat", "engaged but no live target or pet NPC nearby")
        return False

    # Auto-engage: candidate identified by pre-rule scan_auto_engage()
    if ctx.combat.auto_engage_candidate is not None:
        return True

    _skip("Combat", "not engaged, no auto-engage candidate")
    return False


def _score_in_combat(state: GameState, ctx: AgentContext) -> float:
    # Actively engaged with a live target
    if ctx.combat.engaged:
        if state.target is not None and state.target.is_npc and state.target.hp_current > 0:
            return 1.0
        # Pet fighting a nearby damaged NPC
        if ctx.pet.alive:
            for sp in state.spawns:
                if sp.is_npc and sp.hp_current > 0 and sp.hp_current < sp.hp_max:
                    d = state.pos.dist_to(sp.pos)
                    if d < 100:
                        return 0.8
    return 0.0


def _engage_add_suppressed(ctx: AgentContext, state: GameState, rs: _CombatRuleState) -> bool:
    """Return True if EngageAdd should be suppressed (skipped)."""
    if ctx.combat.engaged:
        rs.add_first_seen = 0.0
        _skip("EngageAdd", "engaged (handled by IN_COMBAT)")
        return True
    if not ctx.pet.alive:
        rs.add_first_seen = 0.0
        _skip("EngageAdd", "pet not alive")
        return True
    return False


def _should_engage_add(
    state: GameState,
    ctx: AgentContext,
    rs: _CombatRuleState,
) -> bool:
    if _engage_add_suppressed(ctx, state, rs):
        return False
    if ctx.pet.has_add:
        rs.add_first_seen = 0.0
        return True
    pull_id = ctx.combat.pull_target_id or 0

    # Check 1: target_name scan -- catches full-HP npcs attacking
    # player or pet (damaged_npcs_near misses these)
    pet_name = ctx.pet.name or ""
    for sp in state.spawns:
        if not sp.is_npc or sp.hp_current <= 0:
            continue
        if sp.spawn_id == pull_id:
            continue
        if not sp.target_name:
            continue
        targeting_us = sp.target_name == state.name or (pet_name and sp.target_name == pet_name)
        if targeting_us:
            d = state.pos.dist_to(sp.pos)
            if d < 100:
                who = "player" if sp.target_name == state.name else "pet"
                log.warning(
                    "[COMBAT] EngageAdd: '%s' lv%d targeting %s at %.0fu (HP=%d/%d)",
                    sp.name,
                    sp.level,
                    who,
                    d,
                    sp.hp_current,
                    sp.hp_max,
                )
                ctx.pet.has_add = True
                rs.add_first_seen = 0.0
                return True

    # Check 2: damaged NPCs near pet (already in combat, HP reduced)
    world = ctx.world
    pet_x, pet_y = state.x, state.y
    if world and world.pet_spawn:
        pet_x, pet_y = world.pet_spawn.x, world.pet_spawn.y
    has_damaged = False
    if world:
        has_damaged = bool(world.damaged_npcs_near(Point(pet_x, pet_y, 0.0), 60, exclude_id=pull_id))
    if has_damaged:
        now = time.time()
        if rs.add_first_seen == 0.0:
            rs.add_first_seen = now
            log.debug("[DECISION] EngageAdd: damaged NPC detected, starting 1.5s debounce")
            return False
        if now - rs.add_first_seen >= 1.5:
            rs.add_first_seen = 0.0
            return True
        log.debug("[DECISION] EngageAdd: debounce wait (%.1fs/1.5s)", now - rs.add_first_seen)
        return False
    rs.add_first_seen = 0.0
    return False


def _score_engage_add(state: GameState, ctx: AgentContext) -> float:
    if ctx.combat.engaged:
        return 0.0
    if not ctx.pet.alive:
        return 0.0
    if ctx.pet.has_add:
        return 1.0
    return 0.0


def _has_close_pull_target(state: GameState, ctx: AgentContext) -> bool:
    """Return True if a valid pull target is within 50u during TRAVEL plan."""
    target_cons = ctx.zone.target_cons if ctx.zone.target_cons else {Con.BLUE, Con.LIGHT_BLUE, Con.WHITE}
    for sp in state.spawns:
        if sp.is_npc and sp.hp_current > 0 and sp.owner_id == 0:
            d = state.pos.dist_to(sp.pos)
            if d < 50:
                tc = con_color(state.level, sp.level)
                if tc in target_cons:
                    log.info(
                        "[TARGET] Acquire: bypassing TRAVEL plan -- '%s' (%s) at %.0fu (pull before threat)",
                        sp.name,
                        tc.name,
                        d,
                    )
                    return True
    return False


def _has_pending_loot(ctx: AgentContext) -> bool:
    """True if a recent unlooted defeat should delay acquire."""
    smart = flags.loot_mode == LootMode.SMART
    resource_bases = set()
    if smart and ctx.loot.resource_targets:
        resource_bases = {n.lower() for n in ctx.loot.resource_targets}
    for k in ctx.defeat_tracker.defeat_history:
        if k.looted or time.time() - k.time >= 3:
            continue
        if smart and resource_bases:
            defeat_base = normalize_mob_name(k.name)
            if defeat_base not in resource_bases:
                continue
        return True
    return False


def _acquire_suppressed(ctx: AgentContext, state: GameState) -> bool:
    """Return True if Acquire should be suppressed (skipped).

    Checks feature flags, pet availability, looting delays, active plans,
    nearby players, and combat engagement.
    """
    if not flags.pull:
        _skip("Acquire", "pull disabled")
        return True
    if not ctx.pet.alive:
        _skip("Acquire", "no pet")
        return True
    if flags.looting and _has_pending_loot(ctx):
        _skip("Acquire", "recent unlooted defeat (waiting for loot)")
        return True
    if ctx.plan.active is not None:
        # Allow ACQUIRE to fire through a drift-back TRAVEL plan when
        # a valid target is within 50u (about to walk into threat --
        # mirrors _should_travel yield logic at 30u but with margin
        # so acquire can Tab + approach before contact).
        # All other plans (NEEDS_MEMORIZE) hard-block ACQUIRE.
        if ctx.plan.active == PlanType.TRAVEL and flags.pull and ctx.pet.alive:
            if not _has_close_pull_target(state, ctx):
                _skip("Acquire", f"active plan '{ctx.plan.active}'")
                return True
        else:
            _skip("Acquire", f"active plan '{ctx.plan.active}'")
            return True
    # Only suppress acquire if player is VERY close (camping same spot)
    world = ctx.world
    if world:
        if world.nearby_player_count(250) > 0:
            _skip("Acquire", "player nearby (AFK safety)")
            return True
    else:
        if ctx.nearby_player_count(state, radius=250) > 0:
            _skip("Acquire", "player nearby (AFK safety)")
            return True
    if ctx.combat.engaged:
        _skip("Acquire", "engaged")
        return True
    return False


def _pet_too_far(ctx: AgentContext, state: GameState) -> bool:
    """Return True (with skip log) if the pet is beyond 200u."""
    if ctx.pet.alive and ctx.pet.spawn_id:
        for sp in state.spawns:
            if sp.spawn_id == ctx.pet.spawn_id:
                pet_dist = state.pos.dist_to(sp.pos)
                if pet_dist > 200:
                    _skip("Acquire", f"pet too far ({pet_dist:.0f}u)")
                    return True
                break
    return False


def _pet_hp_low(ctx: AgentContext, state: GameState) -> bool:
    """Return True (with skip log) if pet HP is below 50%."""
    if ctx.pet.alive and ctx.pet.spawn_id:
        for sp in state.spawns:
            if sp.spawn_id == ctx.pet.spawn_id:
                php = sp.hp_current / max(sp.hp_max, 1)
                if php < 0.50:
                    _skip("Acquire", f"pet HP low ({php * 100:.0f}%%)")
                    return True
                break
    return False


def _acquire_not_ready(ctx: AgentContext, state: GameState) -> bool:
    """Return True if Acquire should be skipped due to readiness state.

    Checks pet fighting, pet distance, pet HP, player HP, pull_target_id,
    and nearby damaged NPCs / threats.
    """
    world = ctx.world
    # Don't acquire while pet is actively fighting something
    if world and ctx.pet.alive:
        if world.damaged_npcs_near(state.pos, 150):
            _skip("Acquire", "pet fighting")
            return True
    # Don't acquire if pet is very far away
    if _pet_too_far(ctx, state):
        return True
    if not ctx.pet.alive:
        _skip("Acquire", "no pet (second check)")
        return True
    # Don't start new fights when pet is struggling -- REST instead
    if _pet_hp_low(ctx, state):
        return True
    if state.hp_pct < 0.70:
        _skip("Acquire", "low HP")
        return True
    if ctx.combat.pull_target_id is not None:
        _skip("Acquire", "pull_target_id already set")
        return True
    if world:
        if world.damaged_npcs_near(state.pos, 100):
            _skip("Acquire", "damaged NPC nearby")
            return True
        if world.threats_within(80):
            _skip("Acquire", "threat nearby")
            return True
    return False


def _danger_blocks_acquire(state: GameState, ctx: AgentContext) -> bool:
    """Return True if learned danger for a nearby NPC blocks acquire."""
    fh = ctx.fight_history
    if not fh:
        return False
    for sp in state.spawns:
        if not sp.is_npc or sp.hp_current <= 0:
            continue
        if state.pos.dist_to(sp.pos) > 200:
            continue
        mob_base = normalize_mob_name(sp.name)
        if not fh.has_learned(mob_base):
            continue
        danger = fh.learned_danger(mob_base)
        if danger is None:
            continue
        if danger > 0.5 and state.hp_pct < 0.80:
            log.info(
                "[TARGET] Acquire: SKIP danger=%.2f HP=%.0f%% for '%s'",
                danger,
                state.hp_pct * 100,
                mob_base,
            )
            return True
        if danger > 0.7 and state.mana_pct < 0.40:
            log.info(
                "[TARGET] Acquire: SKIP danger=%.2f mana=%.0f%% for '%s'",
                danger,
                state.mana_pct * 100,
                mob_base,
            )
            return True
    return False


def _should_acquire(state: GameState, ctx: AgentContext) -> bool:
    if _acquire_suppressed(ctx, state):
        return False
    if _acquire_not_ready(ctx, state):
        return False
    # Note: don't suppress acquire based on camp distance. The agent should
    # grind its way back to camp, not refuse to fight npcs that are right here.
    # Marathon mana gate: require >= 50% mana before pulling.
    # Exception: if pet is healthy and HP is fine, pet can solo-pull at low mana
    # (matches rest hysteresis logic that suppresses rest in same conditions).
    _pet_hp = 1.0
    if ctx.pet.alive and ctx.pet.spawn_id:
        for _sp in state.spawns:
            if _sp.spawn_id == ctx.pet.spawn_id:
                _pet_hp = _sp.hp_current / max(_sp.hp_max, 1)
                break
    if state.mana_pct >= 0.20 and state.hp_pct >= 0.85 and ctx.pet.alive and _pet_hp >= 0.70:
        mana_ok = True  # pet can pull without mana
    else:
        mana_ok = state.mana_pct >= 0.50 or state.mana_max == 0
    if not mana_ok:
        _skip("Acquire", "low mana")
        return False
    # Danger-aware gating: if fight_history has learned data for nearby
    # npcs, require higher HP/mana before engaging dangerous targets.
    # Falls back to existing flat gates for unknown npcs.
    if _danger_blocks_acquire(state, ctx):
        return False
    return True


def _score_acquire(
    state: GameState,
    ctx: AgentContext,
    _get_spell: Callable[[str], Spell | None] = get_spell_by_role,
) -> float:
    if not flags.pull:
        return 0.0
    if ctx.in_active_combat:
        return 0.0
    if not ctx.pet.alive:
        return 0.0
    # Product of readiness factors
    hp_factor: float = linear(state.hp_pct, 0.50, 0.90)
    mana_factor = 1.0
    dot = _get_spell(SpellRole.DOT)
    if dot and state.mana_max > 0:
        mana_factor = linear(state.mana_current, 0, dot.mana_cost * 2)
    pet_factor = 1.0 if ctx.pet.alive else 0.0
    return hp_factor * mana_factor * pet_factor


def _should_pull(state: GameState, ctx: AgentContext) -> bool:
    if ctx.combat.pull_target_id is None:
        _skip("Pull", "no pull_target_id")
        return False
    if ctx.combat.engaged:
        _skip("Pull", "engaged")
        return False
    return True


def _score_pull(state: GameState, ctx: AgentContext) -> float:
    if ctx.combat.pull_target_id is None:
        return 0.0
    if ctx.combat.engaged:
        return 0.0
    return 1.0


def register(
    brain: Brain,
    ctx: AgentContext,
    read_state_fn: ReadStateFn,
    spell_provider: Callable[[str], Spell | None] | None = None,
) -> None:
    """Register combat rules."""

    _get_spell = spell_provider or get_spell_by_role

    combat = CombatRoutine(ctx=ctx, read_state_fn=read_state_fn)

    brain.add_rule(
        "IN_COMBAT",
        lambda s: _should_combat(s, ctx),
        combat,
        score_fn=lambda s: _score_in_combat(s, ctx),
        tier=1,
        weight=40,
    )

    # ENGAGE_ADD  -  pet fighting social add we haven't targeted
    engage_extra_npc = EngageAddRoutine(ctx=ctx, read_state_fn=read_state_fn)
    _rs = _CombatRuleState()

    brain.add_rule(
        "ENGAGE_ADD",
        lambda s: _should_engage_add(s, ctx, _rs),
        engage_extra_npc,
        failure_cooldown=5.0,
        score_fn=lambda s: _score_engage_add(s, ctx),
        tier=1,
        weight=30,
    )

    # ACQUIRE  -  find a target to pull
    acquire = AcquireRoutine(ctx=ctx, read_state_fn=read_state_fn)

    acquire_considerations = [
        Consideration(
            name="hp_readiness",
            input_fn=lambda s, _ctx: s.hp_pct,
            curve=lambda v: linear(v, 0.50, 0.90),
        ),
        Consideration(
            name="mana_readiness",
            input_fn=lambda s, _ctx: s.mana_pct if s.mana_max > 0 else 1.0,
            curve=lambda v: linear(v, 0.20, 0.60),
        ),
        Consideration(
            name="pet_alive",
            input_fn=lambda s, _ctx: 1.0 if ctx.pet.alive else 0.0,
            curve=lambda v: v,  # hard gate: 0.0 if no pet
        ),
    ]

    brain.add_rule(
        "ACQUIRE",
        lambda s: _should_acquire(s, ctx),
        acquire,
        failure_cooldown=3.0,
        score_fn=lambda s: _score_acquire(s, ctx, _get_spell),
        considerations=acquire_considerations,
        tier=1,
        weight=30,
    )

    # PULL  -  have a target, not yet engaged
    pull = PullRoutine(ctx=ctx, read_state_fn=read_state_fn)

    brain.add_rule(
        "PULL",
        lambda s: _should_pull(s, ctx),
        pull,
        max_lock_seconds=90.0,
        score_fn=lambda s: _score_pull(s, ctx),
        tier=1,
        weight=35,
    )
