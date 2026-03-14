"""PET_TANK strategy (L1-7): Pet does the work, player assists with spells.

Always apply DoT when available -- roughly halves defeat time.
Lifetap when HP < 70% (urgent self-heal), pet HP < 70% (DPS assist),
fight > 15s (dragging), or at full mana (don't waste regen).

When pet HP drops below 50%, enter PET_SAVE mode: prioritize lifetap
casts at a short cooldown to accelerate the fight and relieve pressure
on the pet.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, override

from core.features import flags
from core.types import ManaMode
from eq.loadout import SpellRole, get_spell_by_role, rank_damage_spells
from perception.combat_eval import Con
from routines.base import RoutineStatus
from routines.strategies.base import CastContext, CastStrategy

if TYPE_CHECKING:
    from brain.context import AgentContext
    from eq.loadout import Spell
    from routines.strategies.base import CombatCaster

log = logging.getLogger(__name__)


class PetTankStrategy(CastStrategy):
    def __init__(self, combat_routine: CombatCaster, ctx: AgentContext) -> None:
        super().__init__(combat_routine, ctx)
        self._efficiency_logged = False

    @override
    def reset(self) -> None:
        self._efficiency_logged = False

    @override
    def execute(self, cc: CastContext) -> RoutineStatus | None:
        if not self._efficiency_logged:
            ranked = rank_damage_spells()
            if ranked:
                parts = [f"{s.name}={s.mana_efficiency:.1f}dmg/mana" for s in ranked]
                log.info("[CAST] SPELL EFFICIENCY: %s", " > ".join(parts))
            self._efficiency_logged = True

        lifetap = get_spell_by_role(SpellRole.LIFETAP)
        dot = get_spell_by_role(SpellRole.DOT)

        # Mana mode: LOW = pet solos, MEDIUM = existing, HIGH = cast everything
        no_cast = False
        if flags.mana_mode == ManaMode.LOW:
            no_cast = not cc.has_adds
        elif flags.mana_mode == ManaMode.HIGH:
            no_cast = False

        # Detect PET_SAVE condition: pet struggling, player healthy
        pet_save = cc.pet_hp >= 0 and cc.pet_hp < 0.50 and cc.state.hp_pct > 0.60

        # In LOW mode, only emergency lifetap (HP < 50%) -- skip normal priority
        if not no_cast:
            result = self._try_lifetap(cc, lifetap)
            if result is not None:
                return result
        elif cc.state.hp_pct < 0.50:
            # Emergency override: lifetap even in LOW mode
            result = self._try_lifetap(cc, lifetap)
            if result is not None:
                return result

        result = self._try_pet_save_lifetap(cc, lifetap, pet_save)
        if result is not None:
            return result

        if not no_cast:
            result = self._try_dot(cc, dot, pet_save)
            if result is not None:
                return result

            result = self._try_lifetap_sustain(cc, lifetap)
            if result is not None:
                return result

        return None

    def _try_lifetap(
        self,
        cc: CastContext,
        lifetap: Spell | None,
    ) -> RoutineStatus | None:
        """Priority 1: Urgent self-heal."""
        if lifetap and cc.state.mana_current >= lifetap.mana_cost and not cc.out_of_range:
            if cc.state.hp_pct < 0.70 and (not cc.recently_sat or cc.state.hp_pct < 0.50):
                time_since_lt = cc.now - (self._ctx.combat.last_lifetap_time if self._ctx else 0)
                cooldown = 3.0 if cc.state.hp_pct < 0.50 else 8.0
                if time_since_lt > cooldown:
                    log.info(
                        "[CAST] Combat: CAST %s (pet_tank urgent) HP=%.0f%% Mana=%d Pet=%.0f%% dist=%.0f",
                        lifetap.name,
                        cc.state.hp_pct * 100,
                        cc.state.mana_current,
                        cc.pet_hp * 100 if cc.pet_hp >= 0 else -1,
                        cc.dist,
                    )
                    self._combat._cast_spell(lifetap.gem, lifetap.cast_time, cc.now, cc.state, cc.target)
                    if self._ctx:
                        self._ctx.combat.last_lifetap_time = cc.now
                        self._ctx.combat.record_spell_cast(lifetap.spell_id)
                    return RoutineStatus.RUNNING
        return None

    def _try_pet_save_lifetap(
        self,
        cc: CastContext,
        lifetap: Spell | None,
        pet_save: bool,
    ) -> RoutineStatus | None:
        """Priority 1.5: PET_SAVE -- accelerated lifetap casts."""
        if pet_save and lifetap and not cc.out_of_range and not cc.recently_sat:
            if cc.state.mana_current >= lifetap.mana_cost:
                time_since_lt = cc.now - (self._ctx.combat.last_lifetap_time if self._ctx else 0)
                if time_since_lt > 2.0:
                    log.info(
                        "[CAST] Combat: CAST %s (PET_SAVE pet=%.0f%% npc=%.0f%%) HP=%.0f%% Mana=%d dist=%.0f",
                        lifetap.name,
                        cc.pet_hp * 100,
                        cc.target_hp * 100,
                        cc.state.hp_pct * 100,
                        cc.state.mana_current,
                        cc.dist,
                    )
                    self._combat._cast_spell(lifetap.gem, lifetap.cast_time, cc.now, cc.state, cc.target)
                    if self._ctx:
                        self._ctx.combat.last_lifetap_time = cc.now
                        self._ctx.combat.record_spell_cast(lifetap.spell_id)
                    return RoutineStatus.RUNNING
        return None

    def _try_dot(
        self,
        cc: CastContext,
        dot: Spell | None,
        pet_save: bool,
    ) -> RoutineStatus | None:
        """Priority 2: DoT on BLUE+ cons (skip light_blue, skip PET_SAVE).

        HIGH mana mode overrides con check -- DoT on everything.
        """
        if flags.mana_mode == ManaMode.HIGH:
            skip_dot = pet_save  # only skip during pet_save
        else:
            skip_dot = cc.tc in (Con.LIGHT_BLUE, Con.GREEN) or pet_save
        if dot and not cc.out_of_range and not cc.is_undead and not skip_dot:
            if cc.state.mana_current >= dot.mana_cost:
                should_recast = self._ctx.combat.should_recast_dot(dot) if self._ctx else True
                if should_recast and not cc.recently_sat:
                    time_since_dot = cc.now - (self._ctx.combat.last_dot_time if self._ctx else 0)
                    log.info(
                        "[CAST] Combat: CAST %s (pet_tank, %s con, %.0fs since last) "
                        "HP=%.0f%% Mana=%d Pet=%.0f%% dist=%.0f",
                        dot.name,
                        cc.tc,
                        time_since_dot,
                        cc.state.hp_pct * 100,
                        cc.state.mana_current,
                        cc.pet_hp * 100 if cc.pet_hp >= 0 else -1,
                        cc.dist,
                    )
                    self._combat._cast_spell(dot.gem, dot.cast_time, cc.now, cc.state, cc.target, is_dot=True)
                    if self._ctx:
                        self._ctx.combat.record_spell_cast(dot.spell_id)
                    return RoutineStatus.RUNNING
        return None

    def _try_lifetap_sustain(
        self,
        cc: CastContext,
        lifetap: Spell | None,
    ) -> RoutineStatus | None:
        """Priority 3: Help pet (DPS assist -- accelerate defeats)."""
        if lifetap and not cc.out_of_range and not cc.recently_sat:
            if cc.state.mana_current >= lifetap.mana_cost:
                time_since_lt = cc.now - (self._ctx.combat.last_lifetap_time if self._ctx else 0)
                should_help = False
                reason = ""

                if cc.state.mana_current >= cc.state.mana_max:
                    should_help = True
                    reason = "full mana"
                elif cc.time_in_combat < 5.0 and cc.target_hp > 0.50 and cc.tc in (Con.WHITE, Con.YELLOW):
                    # Early burst on equal/tough npcs only -- save mana on easy npcs
                    should_help = True
                    reason = "early burst"
                elif cc.pet_hp >= 0 and cc.pet_hp < 0.50:
                    # Pet in real danger -- help regardless of npc HP
                    should_help = True
                    reason = f"pet HP {cc.pet_hp * 100:.0f}%%"
                elif (
                    cc.tc in (Con.WHITE, Con.YELLOW, Con.BLUE)
                    and cc.target_hp > 0.30
                    and cc.pet_hp >= 0
                    and cc.pet_hp < 0.70
                ):
                    # DPS assist on BLUE+ when pet is taking real damage
                    should_help = True
                    reason = f"pet assist ({cc.tc} con, pet {cc.pet_hp * 100:.0f}%%, npc {cc.target_hp * 100:.0f}%%)"
                elif (
                    cc.time_in_combat > 25.0
                    and cc.target_hp > 0.50
                    and cc.tc not in (Con.LIGHT_BLUE, Con.GREEN)
                ):
                    # Only help on long fights vs tough npcs with high HP remaining
                    should_help = True
                    reason = f"fight dragging {cc.time_in_combat:.0f}s"

                if should_help and time_since_lt > 5.0:
                    log.info(
                        "[CAST] Combat: CAST %s (pet_tank, %s) HP=%.0f%% Mana=%d Pet=%.0f%% dist=%.0f",
                        lifetap.name,
                        reason,
                        cc.state.hp_pct * 100,
                        cc.state.mana_current,
                        cc.pet_hp * 100 if cc.pet_hp >= 0 else -1,
                        cc.dist,
                    )
                    self._combat._cast_spell(lifetap.gem, lifetap.cast_time, cc.now, cc.state, cc.target)
                    if self._ctx:
                        self._ctx.combat.last_lifetap_time = cc.now
                        self._ctx.combat.record_spell_cast(lifetap.spell_id)
                    return RoutineStatus.RUNNING
        return None
