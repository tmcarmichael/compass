"""ENDGAME strategy (L49-60): Full DoT stack, aggressive lifetaps, DDs.

- Full DoT stack (DOT + DOT_2) on all npcs
- Lifetap usage (HP < 90%)
- Powerful DDs when mana allows
- Snare fleeing npcs
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, override

from eq.loadout import SpellRole, get_spell_by_role
from routines.base import RoutineStatus
from routines.strategies.base import CastContext, CastStrategy

if TYPE_CHECKING:
    from brain.context import AgentContext
    from eq.loadout import Spell
    from routines.strategies.base import CombatCaster

log = logging.getLogger(__name__)


class EndgameStrategy(CastStrategy):
    def __init__(self, combat_routine: CombatCaster, ctx: AgentContext) -> None:
        super().__init__(combat_routine, ctx)
        self._last_pb_time: float = 0.0

    @override
    def reset(self) -> None:
        self._last_pb_time = 0.0

    @override
    def execute(self, cc: CastContext) -> RoutineStatus | None:
        lifetap = get_spell_by_role(SpellRole.LIFETAP)
        dot = get_spell_by_role(SpellRole.DOT)
        dot2 = get_spell_by_role(SpellRole.DOT_2)
        dd = get_spell_by_role(SpellRole.DD)
        snare = get_spell_by_role(SpellRole.SNARE)

        result = self._try_lifetap(cc, lifetap)
        if result is not None:
            return result

        result = self._try_dot(cc, dot)
        if result is not None:
            return result

        result = self._try_dot2(cc, dot2)
        if result is not None:
            return result

        result = self._try_dd(cc, dd)
        if result is not None:
            return result

        result = self._try_snare(cc, snare)
        if result is not None:
            return result

        result = self._try_any_spell(cc, lifetap, dot)
        if result is not None:
            return result

        return None

    def _try_lifetap(self, cc: CastContext, lifetap: Spell | None) -> RoutineStatus | None:
        """Priority 1: Lifetap."""
        if lifetap and cc.state.mana_current >= lifetap.mana_cost and not cc.out_of_range:
            urgent = cc.state.hp_pct < 0.60
            lt_threshold = 0.95 if cc.has_adds else 0.90
            time_since_lt = cc.now - (self._ctx.combat.last_lifetap_time if self._ctx else 0)
            cooldown = 3.0 if urgent else 8.0
            if (
                cc.state.hp_pct < lt_threshold
                and time_since_lt > cooldown
                and (urgent or not cc.recently_sat)
            ):
                log.info(
                    "[CAST] Combat: CAST %s  -  HP=%.0f%% (endgame %s) mana=%d",
                    lifetap.name,
                    cc.state.hp_pct * 100,
                    "URGENT" if urgent else "sustain",
                    cc.state.mana_current,
                )
                self._combat._cast_spell(lifetap.gem, lifetap.cast_time, cc.now, cc.state, cc.target)
                if self._ctx:
                    self._ctx.combat.last_lifetap_time = cc.now
                    self._ctx.combat.record_spell_cast(lifetap.spell_id)
                return RoutineStatus.RUNNING
        return None

    def _try_dot(self, cc: CastContext, dot: Spell | None) -> RoutineStatus | None:
        """Priority 2: Primary DoT."""
        if (
            dot
            and not cc.out_of_range
            and not cc.recently_sat
            and not cc.is_undead
            and cc.state.mana_current >= dot.mana_cost
        ):
            should = self._ctx.combat.should_recast_dot(dot) if self._ctx else True
            if should:
                log.info("[CAST] Combat: CAST %s (endgame DoT) mana=%d", dot.name, cc.state.mana_current)
                self._combat._cast_spell(dot.gem, dot.cast_time, cc.now, cc.state, cc.target, is_dot=True)
                if self._ctx:
                    self._ctx.combat.record_spell_cast(dot.spell_id)
                return RoutineStatus.RUNNING
        return None

    def _try_dot2(self, cc: CastContext, dot2: Spell | None) -> RoutineStatus | None:
        """Priority 3: Secondary DoT."""
        if (
            dot2
            and not cc.out_of_range
            and not cc.recently_sat
            and not cc.is_undead
            and cc.state.mana_current >= dot2.mana_cost
        ):
            should = self._ctx.combat.should_recast_dot(dot2) if self._ctx else True
            if should:
                log.info(
                    "[CAST] Combat: CAST %s (endgame secondary DoT) mana=%d", dot2.name, cc.state.mana_current
                )
                self._combat._cast_spell(dot2.gem, dot2.cast_time, cc.now, cc.state, cc.target)
                if self._ctx:
                    self._ctx.combat.record_spell_cast(dot2.spell_id)
                return RoutineStatus.RUNNING
        return None

    def _try_dd(self, cc: CastContext, dd: Spell | None) -> RoutineStatus | None:
        """Priority 4: DD burst."""
        if dd and not cc.out_of_range and not cc.recently_sat and not cc.is_undead:
            time_since_pb = cc.now - self._last_pb_time
            if cc.state.mana_current >= dd.mana_cost and time_since_pb > 8.0:
                use_dd = False
                reason = ""
                if cc.has_adds:
                    use_dd, reason = True, "adds"
                elif cc.state.mana_pct > 0.60 and cc.target_hp > 0.30:
                    use_dd, reason = True, "mana available"
                elif cc.time_in_combat > 30.0:
                    use_dd, reason = True, "fight dragging"
                if use_dd:
                    log.info(
                        "[CAST] Combat: CAST %s  -  %s (endgame) mana=%d",
                        dd.name,
                        reason,
                        cc.state.mana_current,
                    )
                    self._combat._cast_spell(dd.gem, dd.cast_time, cc.now, cc.state, cc.target)
                    self._last_pb_time = cc.now
                    if self._ctx:
                        self._ctx.combat.record_spell_cast(dd.spell_id)
                    return RoutineStatus.RUNNING
        return None

    def _try_snare(self, cc: CastContext, snare: Spell | None) -> RoutineStatus | None:
        """Priority 5: Snare fleeing npc."""
        if (
            snare
            and not cc.out_of_range
            and not cc.recently_sat
            and cc.target_hp < 0.20
            and cc.target.speed > 0.2
            and cc.state.mana_current >= snare.mana_cost
        ):
            ts = self._ctx.combat.time_since_spell(snare.spell_id) if self._ctx else 999
            if ts > 15.0:
                log.info(
                    "[CAST] Combat: CAST %s (endgame, npc fleeing) mana=%d", snare.name, cc.state.mana_current
                )
                self._combat._cast_spell(snare.gem, snare.cast_time, cc.now, cc.state, cc.target)
                if self._ctx:
                    self._ctx.combat.record_spell_cast(snare.spell_id)
                return RoutineStatus.RUNNING
        return None

    def _try_any_spell(
        self,
        cc: CastContext,
        lifetap: Spell | None,
        dot: Spell | None,
    ) -> RoutineStatus | None:
        """Priority 6: Opportunistic at high mana."""
        if cc.state.mana_pct >= 0.95 and not cc.out_of_range:
            choices = []
            if lifetap:
                choices.append(lifetap)
            if dot and not cc.is_undead:
                choices.append(dot)
            if choices:
                spell = random.choice(choices)
                is_dot_cast = spell is dot
                log.info(
                    "[CAST] Combat: CAST %s (endgame opportunistic) mana=%d",
                    spell.name,
                    cc.state.mana_current,
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
