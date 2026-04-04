"""Maintenance rules: MEMORIZE_SPELLS, SUMMON_PET, BUFF."""

from __future__ import annotations

import logging
import time as _time_mod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from brain.rules.skip_log import SkipLog
from brain.scoring.curves import inverse_linear
from core.features import flags
from core.types import PlanType
from eq.loadout import Spell, SpellRole, get_spell_by_role
from perception.state import GameState
from routines.base import RoutineStatus
from routines.buff import BuffRoutine
from routines.memorize_spells import MemorizeSpellsRoutine
from routines.summon_pet import SummonPetRoutine

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.context_views import MaintenanceView
    from brain.decision import Brain
    from core.types import ReadStateFn

# Helper functions are typed against MaintenanceView.  See context_views.py.


@dataclass
class _MaintenanceRuleState:
    """Mutable state shared across maintenance rule closures."""

    last_buff_cast: float = 0.0
    buff_logged: bool = False


log = logging.getLogger(__name__)
_skip = SkipLog(log)


# -- Module-level extracted condition/score functions --


def _should_memorize_spells(state: GameState, ctx: MaintenanceView) -> bool:
    if ctx.plan.active == PlanType.NEEDS_MEMORIZE:
        return True
    _skip("Memorize", "no NEEDS_MEMORIZE plan")
    return False


def _score_memorize(state: GameState, ctx: MaintenanceView) -> float:
    return 1.0 if ctx.plan.active == PlanType.NEEDS_MEMORIZE else 0.0


def _should_summon_pet(
    state: GameState,
    ctx: MaintenanceView,
    get_spell: Callable[[SpellRole], Spell | None] = get_spell_by_role,
) -> bool:
    pet_spell = get_spell(SpellRole.PET_SUMMON)
    if not pet_spell:
        _skip("SummonPet", "no pet spell at this level")
        return False
    if ctx.pet.alive:
        _skip("SummonPet", "pet already alive")
        return False
    if ctx.combat.engaged:
        _skip("SummonPet", "engaged")
        return False
    if state.mana_current < pet_spell.mana_cost:
        _skip("SummonPet", "low mana")
        return False
    return True


def _score_summon_pet(
    state: GameState,
    ctx: MaintenanceView,
    get_spell: Callable[[SpellRole], Spell | None] = get_spell_by_role,
) -> float:
    pet_spell = get_spell(SpellRole.PET_SUMMON)
    if not pet_spell:
        return 0.0
    if ctx.pet.alive:
        return 0.0
    if ctx.combat.engaged:
        return 0.0
    if state.mana_current < pet_spell.mana_cost:
        return 0.0
    return 1.0


def _should_buff(
    state: GameState,
    ctx: MaintenanceView,
    rs: _MaintenanceRuleState,
    get_spell: Callable[[SpellRole], Spell | None] = get_spell_by_role,
) -> bool:
    """Reapply self-buff if not active.

    Three checks (any = buff active, skip):
    1. Memory: has_buff() -- spell ID present in buff array
    2. Memory: buff_ticks() > 10 -- remaining duration confirms active
    3. Time-based fallback: cast < 30s ago (recent cast deduplication)
    """
    if not flags.shielding_buff:
        _skip("Buff", "shielding disabled")
        return False
    if ctx.plan.active == PlanType.NEEDS_MEMORIZE:
        _skip("Buff", "memorize pending -- gem loadout in flux")
        return False
    buff_spell = get_spell(SpellRole.SELF_BUFF)
    if not buff_spell:
        _skip("Buff", "no spell configured")
        return False
    if ctx.combat.engaged:
        _skip("Buff", "engaged")
        return False
    if state.is_sitting:
        _skip("Buff", "sitting")
        return False
    if ctx.combat.pull_target_id is not None:
        _skip("Buff", "pull in progress")
        return False
    if state.mana_current < buff_spell.mana_cost:
        _skip("Buff", "low mana")
        return False
    # NPC targeted: press Escape to clear before buffing rather than silently
    # blocking. Without clearing, the agent stays frozen forever when wander
    # leaves a npc targeted and acquire is suppressed by player proximity.
    if state.target and state.target.is_npc:
        _skip("Buff", "NPC targeted")
        return False
    # If we JUST cast the buff, don't recast (rapid recast prevention)
    # Check both the wrapper time AND the ctx time (ctx time set on cast START,
    # wrapper time set on SUCCESS -- ctx covers mid-cast deactivation)
    now = _time_mod.time()
    elapsed = now - rs.last_buff_cast
    ctx_elapsed = now - ctx.player.last_buff_time
    if elapsed < 30.0 or ctx_elapsed < 30.0:
        _skip("Buff", f"recently cast ({min(elapsed, ctx_elapsed):.0f}s ago)")
        return False
    # Memory check: is the buff active? (has_buff checks ID presence,
    # buff_ticks checks duration -- either confirming = skip)
    if buff_spell.spell_id:
        has = state.has_buff(buff_spell.spell_id)
        ticks = state.buff_ticks(buff_spell.spell_id)
        if has or ticks > 10:
            _skip("Buff", "active")
            return False  # buff confirmed active
        if not rs.buff_logged:
            log.info(
                "[CAST] Buff check: spell %d (%s) not in buff array (has=%s ticks=%d last_cast=%.0fs ago)",
                buff_spell.spell_id,
                buff_spell.name,
                has,
                ticks,
                elapsed,
            )
            rs.buff_logged = True
    return True


def _score_buff(
    state: GameState,
    ctx: MaintenanceView,
    rs: _MaintenanceRuleState,
    get_spell: Callable[[SpellRole], Spell | None] = get_spell_by_role,
) -> float:
    if not flags.shielding_buff:
        return 0.0
    buff_spell = get_spell(SpellRole.SELF_BUFF)
    if not buff_spell:
        return 0.0
    if ctx.combat.engaged:
        return 0.0
    now = _time_mod.time()
    elapsed = now - rs.last_buff_cast
    ctx_elapsed = now - ctx.player.last_buff_time
    if elapsed < 30.0 or ctx_elapsed < 30.0:
        return 0.0
    score: float = inverse_linear(min(elapsed, ctx_elapsed), 30, 600)
    return score


def register(
    brain: Brain,
    ctx: AgentContext,
    read_state_fn: ReadStateFn,
    spell_provider: Callable[[SpellRole], Spell | None] | None = None,
    buff_routine: BuffRoutine | None = None,
) -> None:
    """Register maintenance rules."""

    _get_spell = spell_provider or get_spell_by_role

    # MEMORIZE_SPELLS  -  fires on "needs_memorize" signal (set at startup, always)
    memorize = MemorizeSpellsRoutine(ctx=ctx, read_state_fn=read_state_fn)

    brain.add_rule(
        "MEMORIZE_SPELLS",
        lambda s: _should_memorize_spells(s, ctx),
        memorize,
        failure_cooldown=300.0,
        score_fn=lambda s: _score_memorize(s, ctx),
        tier=2,
        weight=15,
    )

    summon_pet = SummonPetRoutine(
        read_state_fn=read_state_fn,
        ctx=ctx,
    )

    brain.add_rule(
        "SUMMON_PET",
        lambda s: _should_summon_pet(s, ctx, _get_spell),
        summon_pet,
        score_fn=lambda s: _score_summon_pet(s, ctx, _get_spell),
        tier=2,
        weight=20,
    )

    buff = buff_routine or BuffRoutine(ctx=ctx, read_state_fn=read_state_fn)
    _rs = _MaintenanceRuleState(last_buff_cast=_time_mod.time())  # init to NOW so startup doesn't recast

    # Wrap tick to record successful cast time
    _orig_tick = buff.tick

    def _buff_tick_wrapper(state: GameState) -> RoutineStatus:
        result = _orig_tick(state)
        if result == RoutineStatus.SUCCESS:
            now = _time_mod.time()
            _rs.last_buff_cast = now
            ctx.player.last_buff_time = now  # suppress REST after buff
            _rs.buff_logged = False
        return result

    object.__setattr__(buff, "tick", _buff_tick_wrapper)  # monkey-patch tick

    brain.add_rule(
        "BUFF",
        lambda s: _should_buff(s, ctx, _rs, _get_spell),
        buff,
        failure_cooldown=5.0,
        score_fn=lambda s: _score_buff(s, ctx, _rs, _get_spell),
        tier=2,
        weight=15,
    )
