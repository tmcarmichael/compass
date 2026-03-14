"""FEAR_KITE strategy (L16-48): Fear -> DoT stack -> lifetap -> re-Fear.

Phase machine:
  INITIAL    -> Cast Fear, then stack DoTs
  FEARED     -> DoTs ticking, npc running. Med/Lifetap while waiting.
  RE_FEAR    -> Fear broke (npc approaching), re-cast Fear.
  PET_TANKING -> Fear resisted or unavailable, fall back to pet_and_dot.

Skips fear when multiple same-type NPCs are nearby to avoid pulling the pack.
Tracks fear outcome rate and warns when add rate is high.
"""

from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from brain.context import AgentContext
    from eq.loadout import Spell
    from routines.strategies.base import CombatCaster
    from routines.strategies.pet_and_dot import PetAndDotStrategy

from eq.loadout import SpellRole, get_spell_by_role
from perception.combat_eval import Con
from routines.base import RoutineStatus
from routines.strategies.base import CastContext, CastStrategy

log = logging.getLogger(__name__)

# Zone safety: skip fear if this many same-type NPCs are within range
SOCIAL_FEAR_THRESHOLD = 3
SOCIAL_FEAR_RADIUS = 80.0  # scan radius for same-type NPCs

# Success tracking: warn if extra_npcs rate exceeds this fraction
FEAR_ADDS_WARN_THRESHOLD = 0.30


class _FearPhase(enum.IntEnum):
    INITIAL = 0
    FEARED = 1
    RE_FEAR = 2
    PET_TANKING = 3


class FearPullTracker:
    """Simple counter for fear-pull success/extra_npcs rate."""

    def __init__(self) -> None:
        self.total: int = 0
        self.with_adds: int = 0

    @property
    def adds_rate(self) -> float:
        return self.with_adds / self.total if self.total > 0 else 0.0

    def record(self, *, had_adds: bool) -> None:
        self.total += 1
        if had_adds:
            self.with_adds += 1
        if self.total >= 10 and self.adds_rate > FEAR_ADDS_WARN_THRESHOLD:
            log.warning(
                "[COMBAT] FearPull: adds rate %.0f%% (%d/%d) -- high add rate detected",
                self.adds_rate * 100,
                self.with_adds,
                self.total,
            )


class FearKiteStrategy(CastStrategy):
    def __init__(self, combat_routine: CombatCaster, ctx: AgentContext) -> None:
        super().__init__(combat_routine, ctx)
        self._fear_phase = _FearPhase.INITIAL
        self._last_fear_time: float = 0.0
        self._last_mob_dist: float = 0.0
        self._pet_and_dot: PetAndDotStrategy | None = None  # lazy import to avoid circular
        self._fear_pull_recorded = False  # track once per fight

    @override
    def reset(self) -> None:
        self._fear_phase = _FearPhase.INITIAL
        self._last_fear_time = 0.0
        self._last_mob_dist = 0.0
        self._fear_pull_recorded = False

    def _get_pet_and_dot(self) -> PetAndDotStrategy:
        if self._pet_and_dot is None:
            from routines.strategies.pet_and_dot import PetAndDotStrategy

            self._pet_and_dot = PetAndDotStrategy(self._combat, self._ctx)
        return self._pet_and_dot

    def _get_fear_tracker(self) -> FearPullTracker:
        """Get or lazily initialize the fear-pull tracker on ctx.combat."""
        if self._ctx and self._ctx.combat.fear_tracker is None:
            self._ctx.combat.fear_tracker = FearPullTracker()
        if self._ctx:
            tracker: FearPullTracker | None = self._ctx.combat.fear_tracker
            if tracker is not None:
                return tracker
        # Fallback: no ctx available (shouldn't happen in practice)
        return FearPullTracker()

    @override
    def execute(self, cc: CastContext) -> RoutineStatus | None:
        # Easy npcs: skip fear-kiting
        if cc.tc in (Con.LIGHT_BLUE, Con.BLUE) and not cc.has_adds:
            return self._get_pet_and_dot().execute(cc)

        fear = get_spell_by_role(SpellRole.FEAR)
        dot = get_spell_by_role(SpellRole.DOT)
        dot2 = get_spell_by_role(SpellRole.DOT_2)
        lifetap = get_spell_by_role(SpellRole.LIFETAP)
        snare = get_spell_by_role(SpellRole.SNARE)

        if not fear or cc.is_undead:
            if cc.is_undead and self._fear_phase != _FearPhase.PET_TANKING:
                log.info("[COMBAT] Combat: undead/no fear, skipping fear-kite")
                self._fear_phase = _FearPhase.PET_TANKING
            return self._get_pet_and_dot().execute(cc)

        # Zone safety: skip fear if too many same-type NPCs nearby
        # Feared npc runs wildly and can pull the entire pack
        if self._fear_phase == _FearPhase.INITIAL:
            nearby_same = self._count_same_type_nearby(cc)
            if nearby_same >= SOCIAL_FEAR_THRESHOLD:
                log.warning(
                    "[COMBAT] Combat: SKIP FEAR -- %d same-type NPCs within %.0fu "
                    "(threshold=%d), falling back to pet_and_dot",
                    nearby_same,
                    SOCIAL_FEAR_RADIUS,
                    SOCIAL_FEAR_THRESHOLD,
                )
                self._fear_phase = _FearPhase.PET_TANKING
                return self._get_pet_and_dot().execute(cc)

        # Fear duration from spell DB (replaces hardcoded 24s)
        fear_duration = self._get_fear_duration(fear)

        time_since_fear = cc.now - self._last_fear_time if self._last_fear_time > 0 else 999
        mob_approaching = cc.dist < self._last_mob_dist - 5.0
        fear_may_have_broken = self._fear_phase == _FearPhase.FEARED and time_since_fear > fear_duration * 0.7
        fear_definitely_broke = (
            self._fear_phase == _FearPhase.FEARED and mob_approaching and time_since_fear > 3.0
        )
        self._last_mob_dist = cc.dist

        # Urgent self-heal
        result = self._try_urgent_lifetap(cc, lifetap)
        if result is not None:
            return result

        # Phase transitions
        if fear_definitely_broke and self._fear_phase == _FearPhase.FEARED:
            log.info("[COMBAT] Combat: FEAR BROKE (dist=%.0f, %.0fs since fear)", cc.dist, time_since_fear)
            self._fear_phase = _FearPhase.RE_FEAR
        if fear_may_have_broken and self._fear_phase == _FearPhase.FEARED:
            log.info("[COMBAT] Combat: fear expiring (%.0fs/%.0fs)", time_since_fear, fear_duration)
            self._fear_phase = _FearPhase.RE_FEAR

        # Phase dispatch
        if self._fear_phase == _FearPhase.INITIAL:
            return self._execute_initial(cc, fear)
        if self._fear_phase == _FearPhase.FEARED:
            return self._execute_feared(cc, dot, dot2, lifetap)
        if self._fear_phase == _FearPhase.RE_FEAR:
            return self._execute_re_fear(cc, fear, snare, time_since_fear)
        if self._fear_phase == _FearPhase.PET_TANKING:
            return self._execute_pet_tanking(cc)

        return None

    def _get_fear_duration(self, fear: Spell) -> float:
        """Look up fear duration from spell DB, fallback to 24s."""
        from eq.loadout import get_spell_db

        fear_sd = get_spell_db().get(fear.spell_id) if fear.spell_id else None
        fear_duration: float
        if fear_sd and fear_sd.duration_ticks > 0:
            fear_duration = float(fear_sd.duration_seconds)
            log.debug(
                "[CAST] Combat: fear duration %.0fs (from spell DB, %d ticks)",
                fear_duration,
                fear_sd.duration_ticks,
            )
        else:
            fear_duration = 24.0
            log.debug("[CAST] Combat: fear duration %.0fs (default -- no spell DB data)", fear_duration)
        return fear_duration

    def _try_urgent_lifetap(self, cc: CastContext, lifetap: Spell | None) -> RoutineStatus | None:
        """Cast lifetap urgently when HP < 60%."""
        if (
            lifetap
            and cc.state.hp_pct < 0.60
            and cc.state.mana_current >= lifetap.mana_cost
            and not cc.out_of_range
        ):
            time_since_lt = cc.now - (self._ctx.combat.last_lifetap_time if self._ctx else 0)
            if time_since_lt > 3.0 and (not cc.recently_sat or cc.state.hp_pct < 0.50):
                log.info(
                    "[CAST] Combat: CAST %s  -  HP=%.0f%% (fear_kite URGENT) mana=%d",
                    lifetap.name,
                    cc.state.hp_pct * 100,
                    cc.state.mana_current,
                )
                self._combat._cast_spell(lifetap.gem, lifetap.cast_time, cc.now, cc.state, cc.target)
                if self._ctx:
                    self._ctx.combat.last_lifetap_time = cc.now
                    self._ctx.combat.record_spell_cast(lifetap.spell_id)
                return RoutineStatus.RUNNING
        return None

    def _execute_initial(self, cc: CastContext, fear: Spell) -> RoutineStatus | None:
        """INITIAL phase: cast Fear to start kiting."""
        if cc.state.mana_current >= fear.mana_cost and not cc.out_of_range and not cc.recently_sat:
            log.info(
                "[CAST] Combat: CAST %s (initial fear) mana=%d dist=%.0f",
                fear.name,
                cc.state.mana_current,
                cc.dist,
            )
            self._combat._cast_spell(fear.gem, fear.cast_time, cc.now, cc.state, cc.target)
            self._last_fear_time = cc.now
            if self._ctx:
                self._ctx.combat.record_spell_cast(fear.spell_id)
            self._fear_phase = _FearPhase.FEARED
            return RoutineStatus.RUNNING
        if cc.out_of_range or cc.state.mana_current < fear.mana_cost:
            self._fear_phase = _FearPhase.PET_TANKING
        return None

    def _execute_feared(
        self,
        cc: CastContext,
        dot: Spell | None,
        dot2: Spell | None,
        lifetap: Spell | None,
    ) -> RoutineStatus | None:
        """FEARED phase: stack DoTs, sustain lifetap while fear is active."""
        if not self._fear_pull_recorded:
            self._fear_pull_recorded = True
            self._get_fear_tracker().record(had_adds=cc.has_adds)
        for spell, is_dot in [(dot, True), (dot2, False)]:
            if (
                spell
                and not cc.out_of_range
                and not cc.recently_sat
                and cc.state.mana_current >= spell.mana_cost
            ):
                should = self._ctx.combat.should_recast_dot(spell) if self._ctx else True
                if should:
                    log.info(
                        "[CAST] Combat: CAST %s (feared, stacking) mana=%d", spell.name, cc.state.mana_current
                    )
                    self._combat._cast_spell(
                        spell.gem, spell.cast_time, cc.now, cc.state, cc.target, is_dot=is_dot
                    )
                    if self._ctx:
                        self._ctx.combat.record_spell_cast(spell.spell_id)
                    return RoutineStatus.RUNNING

        if (
            lifetap
            and cc.state.hp_pct < 0.80
            and not cc.out_of_range
            and not cc.recently_sat
            and cc.state.mana_current >= lifetap.mana_cost
        ):
            time_since_lt = cc.now - (self._ctx.combat.last_lifetap_time if self._ctx else 0)
            if time_since_lt > 8.0:
                log.info(
                    "[CAST] Combat: CAST %s (feared, sustain) mana=%d", lifetap.name, cc.state.mana_current
                )
                self._combat._cast_spell(lifetap.gem, lifetap.cast_time, cc.now, cc.state, cc.target)
                if self._ctx:
                    self._ctx.combat.last_lifetap_time = cc.now
                    self._ctx.combat.record_spell_cast(lifetap.spell_id)
                return RoutineStatus.RUNNING
        return None

    def _execute_re_fear(
        self,
        cc: CastContext,
        fear: Spell,
        snare: Spell | None,
        time_since_fear: float,
    ) -> RoutineStatus | None:
        """RE_FEAR phase: re-cast fear, or snare as fallback."""
        if cc.state.mana_current >= fear.mana_cost and not cc.out_of_range and not cc.recently_sat:
            log.info(
                "[CAST] Combat: CAST %s (RE-FEAR, %.0fs since last) mana=%d",
                fear.name,
                time_since_fear,
                cc.state.mana_current,
            )
            self._combat._cast_spell(fear.gem, fear.cast_time, cc.now, cc.state, cc.target)
            self._last_fear_time = cc.now
            if self._ctx:
                self._ctx.combat.record_spell_cast(fear.spell_id)
            self._fear_phase = _FearPhase.FEARED
            return RoutineStatus.RUNNING
        if snare and cc.state.mana_current >= snare.mana_cost and not cc.out_of_range and not cc.recently_sat:
            ts = self._ctx.combat.time_since_spell(snare.spell_id) if self._ctx else 999
            if ts > 10.0:
                log.info("[CAST] Combat: CAST %s (snare fallback) mana=%d", snare.name, cc.state.mana_current)
                self._combat._cast_spell(snare.gem, snare.cast_time, cc.now, cc.state, cc.target)
                if self._ctx:
                    self._ctx.combat.record_spell_cast(snare.spell_id)
                return RoutineStatus.RUNNING
        self._fear_phase = _FearPhase.PET_TANKING
        return None

    def _execute_pet_tanking(self, cc: CastContext) -> RoutineStatus | None:
        """PET_TANKING phase: delegate to pet_and_dot fallback."""
        # Record fear-pull outcome if fear was cast before falling back
        if not self._fear_pull_recorded and self._last_fear_time > 0:
            self._fear_pull_recorded = True
            self._get_fear_tracker().record(had_adds=cc.has_adds)
        return self._get_pet_and_dot().execute(cc)

    def _count_same_type_nearby(self, cc: CastContext) -> int:
        """Count living NPCs with the same base name near the target.

        Uses the spawns from cc.state to avoid importing perception.queries
        (which would create a circular dependency from strategies/).
        """
        target = cc.target
        base = target.name.rstrip("0123456789")
        count = 0
        for spawn in cc.state.spawns:
            if not spawn.is_npc or spawn.hp_current <= 0:
                continue
            if spawn.spawn_id == target.spawn_id:
                continue
            if spawn.name.rstrip("0123456789") == base:
                d = target.pos.dist_to(spawn.pos)
                if d < SOCIAL_FEAR_RADIUS:
                    count += 1
        return count
