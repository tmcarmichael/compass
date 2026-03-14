"""PET_AND_DOT strategy (L8-15): Pet tanks, player assists with DoTs and DDs.

Priority 1: Lifetap when HP < 80% (urgent at 60%)
Priority 2: DD on extra_npcs/long fights/high mana
Priority 3: DoT reapply every ~30s
Priority 4: Opportunistic cast at full mana
Mana conservation: LIGHT_BLUE/BLUE = pet-only unless fight drags
"""

from __future__ import annotations

import logging
import random
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

DOT_RECAST_INTERVAL = 30.0


class PetAndDotStrategy(CastStrategy):
    def __init__(self, combat_routine: CombatCaster, ctx: AgentContext) -> None:
        super().__init__(combat_routine, ctx)
        self._last_pb_time: float = 0.0
        self._efficiency_logged: bool = False

    @override
    def reset(self) -> None:
        self._last_pb_time = 0.0
        self._efficiency_logged = False
        self._escalated = False

    @override
    def execute(self, cc: CastContext) -> RoutineStatus | None:
        # Log spell efficiency ranking once per fight
        if not self._efficiency_logged:
            ranked = rank_damage_spells()
            if ranked:
                parts = [f"{s.name}={s.mana_efficiency:.1f}dmg/mana" for s in ranked]
                log.info("[CAST] SPELL EFFICIENCY: %s", " > ".join(parts))
            self._efficiency_logged = True

        time_since_dot = cc.now - (self._ctx.combat.last_dot_time if self._ctx else 0)
        time_since_lt = cc.now - (self._ctx.combat.last_lifetap_time if self._ctx else 0)
        time_since_pb = cc.now - self._last_pb_time

        # Mana conservation driven by flags.mana_mode:
        #   LOW:    no casts (emergency lifetap only at HP < 50%)
        #   MEDIUM: skip GREEN/LIGHT_BLUE (pet solos), PB on WHITE only
        #   HIGH:   cast on everything
        # Exception: adds, pet struggling, or fight dragging always escalate
        if flags.mana_mode == ManaMode.LOW:
            no_cast = not cc.has_adds
            pb_ok = False
        elif flags.mana_mode == ManaMode.HIGH:
            no_cast = False
            pb_ok = True
        else:  # MEDIUM (default -- existing behavior)
            no_cast = cc.tc in (Con.GREEN, Con.LIGHT_BLUE) and not cc.has_adds
            pb_ok = cc.tc == Con.WHITE

        # Escalation: if fight drags, remove restrictions
        if no_cast and cc.time_in_combat > 20.0:
            no_cast = False
            if not self._escalated:
                log.info(
                    "[COMBAT] Combat: ESCALATE  -  fight %.0fs, npc HP=%.0f%%",
                    cc.time_in_combat,
                    cc.target_hp * 100,
                )
                self._escalated = True

        lifetap = get_spell_by_role(SpellRole.LIFETAP)
        dot = get_spell_by_role(SpellRole.DOT)
        dot2 = get_spell_by_role(SpellRole.DOT_2)
        dd = dot2 if dot2 and dot2.gem else get_spell_by_role(SpellRole.DD)

        # Priority 0: Lifetap when player HP < 95%
        result = self._try_lifetap(cc, lifetap, no_cast, time_since_lt)
        if result is not None:
            return result

        # Priority 1: DD on WHITE only (skip BLUE and below)
        result = self._try_dd(cc, dd, not pb_ok, False, time_since_pb)
        if result is not None:
            return result

        # Priority 2: DoT on BLUE+ (skip GREEN/LIGHT_BLUE)
        result = self._try_dot(cc, dot, time_since_dot, no_cast)
        if result is not None:
            return result

        result = self._try_efficiency_spell(
            cc,
            lifetap,
            dot,
            no_cast,
            False,
        )
        if result is not None:
            return result

        return None

    def _try_lifetap(
        self,
        cc: CastContext,
        lifetap: Spell | None,
        no_cast: bool,
        time_since_lt: float,
    ) -> RoutineStatus | None:
        """Priority 0: Lifetap when player HP < 95%.

        Emergency override: in LOW mana mode, cast lifetap when HP < 50%
        even though no_cast is True (survival takes priority over mana).
        """
        emergency_override = no_cast and cc.state.hp_pct < 0.50
        if (
            lifetap
            and cc.state.mana_current >= lifetap.mana_cost
            and (not no_cast or emergency_override)
            and not cc.out_of_range
        ):
            lt_threshold = 0.95
            urgent = cc.state.hp_pct < 0.60
            cooldown = 3.0 if urgent else 8.0
            if (
                cc.state.hp_pct < lt_threshold
                and time_since_lt > cooldown
                and (urgent or not cc.recently_sat)
            ):
                log.info(
                    "[CAST] Combat: CAST %s  -  HP=%.0f%% (%s) mana=%d dist=%.0f",
                    lifetap.name,
                    cc.state.hp_pct * 100,
                    "URGENT" if urgent else "opportunistic",
                    cc.state.mana_current,
                    cc.dist,
                )
                self._combat._cast_spell(lifetap.gem, lifetap.cast_time, cc.now, cc.state, cc.target)
                if self._ctx:
                    self._ctx.combat.last_lifetap_time = cc.now
                    self._ctx.combat.record_spell_cast(lifetap.spell_id)
                return RoutineStatus.RUNNING
        return None

    def _try_dd(
        self,
        cc: CastContext,
        dd: Spell | None,
        easy_mob: bool,
        pet_handling_it: bool,
        time_since_pb: float,
    ) -> RoutineStatus | None:
        """Priority 2: DD when situation demands."""
        if (
            dd
            and cc.state.mana_current >= dd.mana_cost
            and time_since_pb > 10.0
            and not easy_mob
            and not pet_handling_it
            and not cc.recently_sat
            and not cc.out_of_range
            and not cc.is_undead
        ):
            use_pb = False
            reason = ""
            if cc.has_adds:
                use_pb = True
                reason = "adds"
            elif cc.time_in_combat > 45.0 and cc.tc in (Con.YELLOW, Con.WHITE):
                use_pb = True
                reason = f"fight dragging ({cc.tc} con)"
            elif cc.state.mana_pct > 0.90 and cc.target_hp > 0.50 and cc.tc == Con.YELLOW:
                use_pb = True
                reason = f"high mana + {cc.tc} con"
            if use_pb:
                log.info(
                    "[CAST] Combat: CAST %s  -  %s mana=%d dist=%.0f",
                    dd.name,
                    reason,
                    cc.state.mana_current,
                    cc.dist,
                )
                self._combat._cast_spell(dd.gem, dd.cast_time, cc.now, cc.state, cc.target)
                self._last_pb_time = cc.now
                if self._ctx:
                    self._ctx.combat.record_spell_cast(dd.spell_id)
                return RoutineStatus.RUNNING
        return None

    def _try_dot(
        self,
        cc: CastContext,
        dot: Spell | None,
        time_since_dot: float,
        easy_mob: bool = False,
    ) -> RoutineStatus | None:
        """Priority 3: DoT reapply. Skip on easy npcs (pet solos)."""
        if easy_mob:
            return None
        if (
            dot
            and cc.state.mana_current >= dot.mana_cost
            and not cc.recently_sat
            and not cc.out_of_range
            and not cc.is_undead
        ):
            should_recast = (
                self._ctx.combat.should_recast_dot(dot) if self._ctx else time_since_dot > DOT_RECAST_INTERVAL
            )
            if should_recast:
                log.info(
                    "[CAST] Combat: CAST %s (reapply, %.0fs since last) mana=%d dist=%.0f",
                    dot.name,
                    time_since_dot,
                    cc.state.mana_current,
                    cc.dist,
                )
                self._combat._cast_spell(dot.gem, dot.cast_time, cc.now, cc.state, cc.target, is_dot=True)
                if self._ctx:
                    self._ctx.combat.record_spell_cast(dot.spell_id)
                return RoutineStatus.RUNNING
        return None

    def _try_efficiency_spell(
        self,
        cc: CastContext,
        lifetap: Spell | None,
        dot: Spell | None,
        easy_mob: bool,
        pet_handling_it: bool,
    ) -> RoutineStatus | None:
        """Priority 4: Opportunistic cast at full mana."""
        if cc.state.mana_pct >= 1.0 and not easy_mob and not pet_handling_it and not cc.out_of_range:
            choices = []
            if lifetap:
                choices.append(lifetap)
            if dot:
                choices.append(dot)
            if choices:
                spell = random.choice(choices)
                is_dot_cast = spell is dot
                log.info(
                    "[CAST] Combat: CAST %s (full mana, opportunistic) mana=%d dist=%.0f",
                    spell.name,
                    cc.state.mana_current,
                    cc.dist,
                )
                self._combat._cast_spell(
                    spell.gem, spell.cast_time, cc.now, cc.state, cc.target, is_dot=is_dot_cast
                )
                if self._ctx:
                    if spell is lifetap:
                        self._ctx.combat.last_lifetap_time = cc.now
                    self._ctx.combat.record_spell_cast(spell.spell_id)
                return RoutineStatus.RUNNING
        return None
