"""Pet combat management: healing, status tracking, recall decisions.

Extracted from CombatRoutine to isolate pet-specific logic.
Used by CombatRoutine.tick() when pet needs healing during a fight.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from brain.context import AgentContext
    from routines.casting import CastingPhase

from core.constants import PET_HEAL_THRESHOLD
from core.timing import interruptible_sleep
from core.types import ReadStateFn
from eq.loadout import SpellRole, get_spell_by_role
from motor.actions import (
    clear_target,
    pet_attack,
    pet_back_off,
    tab_target,
)
from perception.state import GameState, SpawnData
from util.log_tiers import VERBOSE

log = logging.getLogger(__name__)


def should_heal_pet(
    pet_hp_pct: float,
    pet_dist: float,
    mana: int,
    heal_cost: int,
    heal_recast: float,
    time_since_heal: float,
    recently_sat: bool,
    emergency_ttd: bool,
    target_hp: float,
    target_speed: float,
) -> str:
    """Pure function: decide whether to heal pet and how.

    Returns: "HEAL", "RECALL_THEN_HEAL", "SKIP_FLEEING", "SKIP_FAR",
             "SKIP_NO_NEED", or "SKIP_NO_MANA".
    """
    # Threshold check
    PET_HEAL_THRESHOLD = 0.70
    needs = (
        (pet_hp_pct <= PET_HEAL_THRESHOLD or emergency_ttd)
        and mana >= heal_cost
        and time_since_heal > heal_recast + 0.5
        and (not recently_sat or pet_hp_pct < 0.50 or emergency_ttd)
    )
    if not needs:
        if mana < heal_cost:
            return "SKIP_NO_MANA"
        return "SKIP_NO_NEED"

    # Mob fleeing check
    mob_fleeing = target_hp < 0.25 and target_speed > 0.2
    if mob_fleeing:
        return "SKIP_FLEEING"

    if pet_dist > 100:
        return "RECALL_THEN_HEAL"

    if pet_dist >= 200:
        return "SKIP_FAR"

    return "HEAL"


class PetCombatManager:
    """Manages pet during combat: healing, status monitoring.

    Holds its own heal timer and count. The parent CombatRoutine
    calls try_heal() each tick to check if pet needs Mend Bones.
    """

    def __init__(self, ctx: AgentContext | None = None, read_state_fn: ReadStateFn | None = None) -> None:
        self._ctx = ctx
        self._read_state_fn = read_state_fn
        self._last_heal_time: float = 0.0
        self._heals: int = 0
        self._flee_chase_logged: bool = False
        self._cast_phase: CastingPhase | None = None  # CastingPhase when healing
        self._heal_mob_target_id: int | None = None  # npc to retarget after heal
        self._flee_check = None  # set by parent CombatRoutine

    def reset(self) -> None:
        """Reset state for a new fight."""
        self._heals = 0
        self._flee_chase_logged = False
        self._cast_phase = None
        self._heal_mob_target_id = None

    @property
    def heals(self) -> int:
        return self._heals

    def get_pet_status(self, state: GameState) -> tuple[float, float]:
        """Returns (pet_hp_pct, pet_dist) from spawn list."""
        pet_hp_pct = 1.0
        pet_dist = 9999.0
        if self._ctx and self._ctx.pet.alive and self._ctx.pet.spawn_id:
            for sp in state.spawns:
                if sp.spawn_id == self._ctx.pet.spawn_id:
                    pet_hp_pct = sp.hp_current / max(sp.hp_max, 1)
                    pet_dist = state.pos.dist_to(sp.pos)
                    break
        return pet_hp_pct, pet_dist

    def try_heal(
        self,
        state: GameState,
        target: SpawnData | None,
        *,
        medding: bool = False,
        recently_sat: bool,
        stand_from_med_fn: Callable[[], None],
    ) -> bool | str | None:
        """Attempt to heal pet if needed. Returns:
        - True if pet was healed (caller should return RUNNING)
        - False if npc died during heal (caller should return SUCCESS)
        - None if no heal was needed (caller continues normal tick)
        """
        pet_heal = get_spell_by_role(SpellRole.PET_HEAL)
        if not pet_heal or not self._ctx or not self._ctx.pet.alive:
            return None

        now = time.time()
        pet_hp_pct, pet_dist = self.get_pet_status(state)
        target_hp = (target.hp_current / max(target.hp_max, 1)) if target else 1.0

        time_since_heal = now - max(
            self._last_heal_time,
            self._ctx.pet.last_heal_time if self._ctx else 0,
        )

        # Emergency heal: pet will die in < 5 seconds based on damage rate
        emergency_ttd = False
        if self._ctx and hasattr(self._ctx, "world") and self._ctx.world:
            pet_ttd = self._ctx.world.pet_time_to_death()
            if pet_ttd is not None and pet_ttd < 5.0:
                emergency_ttd = True

        needs_heal = (
            (pet_hp_pct <= PET_HEAL_THRESHOLD or emergency_ttd)
            and state.mana_current >= pet_heal.mana_cost
            and time_since_heal > pet_heal.recast + 0.5
            and (not recently_sat or pet_hp_pct < 0.50 or emergency_ttd)
        )

        if not needs_heal:
            return None

        if emergency_ttd:
            log.info(
                "[PET] Combat: EMERGENCY pet heal  -  TTD=%.1fs, pet HP=%.0f%%", pet_ttd, pet_hp_pct * 100
            )

        # Don't recall if npc is fleeing (low HP + moving)
        mob_fleeing = target_hp < 0.25 and target and target.speed > 0.2

        if pet_dist > 100 and not mob_fleeing:
            log.info("[PET] Combat: pet needs heal but too far (%.0fu)  -  recalling", pet_dist)
            stand_from_med_fn()
            pet_back_off()
            interruptible_sleep(random.uniform(2.0, 3.0), self._flee_check)
            if self._read_state_fn:
                ns = self._read_state_fn()
                for sp in ns.spawns:
                    if sp.spawn_id == self._ctx.pet.spawn_id:
                        pet_dist = ns.pos.dist_to(sp.pos)
                        break
        elif mob_fleeing:
            if not self._flee_chase_logged:
                log.info(
                    "[PET] Combat: pet needs heal but npc fleeing (HP=%.0f%%)  -  letting pet chase",
                    target_hp * 100,
                )
                self._flee_chase_logged = True
            return None  # skip heal, let pet finish

        if pet_dist >= 200:
            return None  # pet too far even after recall attempt

        stand_from_med_fn()
        log.log(
            VERBOSE,
            "[PET] Combat: HEAL PET  -  Mend Bones (pet HP=%.0f%%) mana=%d",
            pet_hp_pct * 100,
            state.mana_current,
        )

        mob_target_id = target.spawn_id if target else None

        # Target pet + cast heal gem (no macro needed)
        from motor.actions import pet_heal as _pet_heal_action

        _pet_heal_action()
        self._last_heal_time = now
        if self._ctx:
            self._ctx.pet.last_heal_time = now
        self._heals += 1
        if self._ctx:
            self._ctx.metrics.total_casts += 1

        # Non-blocking cast: start phase and return "healing" sentinel
        from routines.casting import CastingPhase

        self._cast_phase = CastingPhase(pet_heal.cast_time, pet_heal.name, self._read_state_fn)
        self._heal_mob_target_id = mob_target_id
        return "healing"  # sentinel: combat routine should return RUNNING

    def tick_heal(self) -> bool | None:
        """Poll pet heal cast progress. Call each tick while healing.

        Returns:
            None  -- still casting, caller should return RUNNING
            True  -- heal complete, retargeted npc
            False -- heal complete, retarget failed
        """
        if self._cast_phase is None:
            return True  # no active cast

        result = self._cast_phase.tick()
        from routines.casting import CastResult

        if result == CastResult.CASTING:
            return None  # still casting

        # Cast done
        self._cast_phase = None
        log.log(VERBOSE, "[PET] Combat: pet heal cast complete, retargeting npc")

        # Retarget the npc
        mob_target_id = self._heal_mob_target_id
        found_target = False
        tab_target()
        interruptible_sleep(0.3, self._flee_check)
        if mob_target_id and self._read_state_fn:
            for _tab in range(6):
                check = self._read_state_fn()
                if check.target and check.target.spawn_id == mob_target_id:
                    found_target = True
                    break
                tab_target()
                interruptible_sleep(0.25, self._flee_check)

        if found_target:
            # Log what we're sending the pet after (forensic: catches wrong-target threat)
            if self._read_state_fn:
                _rs = self._read_state_fn()
                log.info(
                    "[PET] Combat: pet_attack -> '%s' id=%d dist=%.0f",
                    _rs.target.name if _rs.target else "?",
                    _rs.target.spawn_id if _rs.target else 0,
                    _rs.pos.dist_to(_rs.target.pos) if _rs.target else 0,
                )
            pet_attack()
            interruptible_sleep(0.3, self._flee_check)
            return True

        # Log what the last tab landed on before we clear
        if self._read_state_fn:
            _rs = self._read_state_fn()
            if _rs.target:
                log.warning(
                    "[PET] Combat: retarget failed -- last tab was '%s' id=%d (clearing, NOT sending pet)",
                    _rs.target.name,
                    _rs.target.spawn_id,
                )
        log.warning("[PET] Combat: retarget failed after 6 tabs  -  npc may have reset (out of tab range)")
        # Don't blindly pet_attack -- the last tab may have landed on
        # a zone-avoided npc (e.g. will-o-wisp). Clear target to be safe.
        clear_target()
        interruptible_sleep(0.2, self._flee_check)
        # Return True (heal complete) not False (which falsely records a defeat).
        # If npc truly reset, stale fight detector (npc 100% HP >80u >15s)
        # will disengage the combat routine.
        return True
