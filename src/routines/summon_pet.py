"""Summon pet routine: cast pet spell and verify the pet appeared.

Non-blocking phase machine: cast, wait for cast completion, verify pet
spawned nearby. Retries up to MAX_SUMMON_ATTEMPTS on failure.
When no pet spell is available at the current level, skip.
Recasting the pet spell automatically replaces the current pet.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, override

from core.timing import interruptible_sleep
from core.types import FailureCategory, ReadStateFn
from eq.loadout import SpellRole, get_spell_by_role
from motor.actions import press_gem, stand
from perception.combat_eval import is_pet
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus, make_flee_predicate
from util.log_tiers import EVENT
from util.structured_log import log_event

if TYPE_CHECKING:
    from collections.abc import Callable

    from brain.context import AgentContext
    from routines.casting import CastingPhase
log = logging.getLogger(__name__)

CAST_WAIT = 11.0  # 7s cast + buffer

MAX_SUMMON_ATTEMPTS = 8  # Total attempts before giving up


class SummonPetRoutine(RoutineBase):
    """Cast pet spell, verify spawned, retry on failure."""

    def __init__(self, read_state_fn: ReadStateFn | None = None, ctx: AgentContext | None = None) -> None:
        self._retries = 0
        self._read_state_fn = read_state_fn
        self._ctx = ctx
        self._summoned = False
        self._pre_summon_ids: set[int] = set()
        self._cast_phase: CastingPhase | None = None
        self._flee_check: Callable[[], bool] | None = None
        self._casting_started = False  # lock flag: True once cast begins

    @property
    def locked(self) -> bool:
        """Lock during cast + acceptance to prevent brain from deactivating
        before we can verify the pet appeared."""
        return self._casting_started and not self._summoned

    def _snapshot_pet_ids(self, state: GameState) -> set[int]:
        ids = set()
        for spawn in state.spawns:
            if is_pet(spawn) and spawn.is_npc:
                ids.add(spawn.spawn_id)
        return ids

    def _find_new_pet(self, state: GameState) -> tuple[int, str, int]:
        """Find the newly summoned pet. Returns (spawn_id, name, level)."""
        post_ids = self._snapshot_pet_ids(state)
        new_ids = post_ids - self._pre_summon_ids

        if new_ids:
            new_id = new_ids.pop()
            for spawn in state.spawns:
                if spawn.spawn_id == new_id:
                    return new_id, spawn.name, spawn.level
            return new_id, "", 0

        # Fallback: closest pet-named NPC
        best_dist = 9999.0
        best = (0, "", 0)
        for spawn in state.spawns:
            if is_pet(spawn) and spawn.is_npc:
                d = state.pos.dist_to(spawn.pos)
                if d < best_dist:
                    best_dist = d
                    best = (spawn.spawn_id, spawn.name, spawn.level)
        return best

    @override
    def enter(self, state: GameState) -> None:
        self._retries = 0
        self._summoned = False
        self._casting_started = False
        self._cast_phase = None
        if self._read_state_fn and self._ctx:
            self._flee_check = make_flee_predicate(self._read_state_fn, self._ctx)
        else:
            self._flee_check = None
        self._pre_summon_ids = self._snapshot_pet_ids(state)

        pet_spell = get_spell_by_role(SpellRole.PET_SUMMON)
        if pet_spell:
            log.info("[PET] SummonPet: summoning via %s (id=%d)", pet_spell.name, pet_spell.spell_id)

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        # Non-blocking cast polling
        if self._cast_phase is not None:
            result = self._cast_phase.tick()
            from routines.casting import CastResult

            if result == CastResult.CASTING:
                return RoutineStatus.RUNNING
            self._cast_phase = None
            return self._check_summon_result(state)

        if self._summoned:
            return RoutineStatus.SUCCESS

        if self._retries >= MAX_SUMMON_ATTEMPTS:
            log.warning("[PET] SummonPet: failed after %d attempts", self._retries)
            self.failure_reason = "max_retries"
            self.failure_category = FailureCategory.EXECUTION
            return RoutineStatus.FAILURE

        # Force stand via hotbar macro -- never use the toggle key here.
        # The _sitting flag drifts when memorize is interrupted mid-sit.
        if state.is_sitting:
            from motor.actions import force_standing, press_hotbar

            press_hotbar(8)
            interruptible_sleep(0.5, self._flee_check)
            force_standing()
            log.info("[PET] SummonPet: forced /stand (was sitting)")
        else:
            stand()
            interruptible_sleep(0.3, self._flee_check)

        # Cast pet spell  -  use whatever pet summon is currently memorized
        pet_spell = get_spell_by_role(SpellRole.PET_SUMMON)
        pet_gem = pet_spell.gem if pet_spell else 0
        if not pet_gem:
            log.warning("[PET] SummonPet: no pet summon spell memorized!")
            self.failure_reason = "no_spell"
            self.failure_category = FailureCategory.PRECONDITION
            return RoutineStatus.FAILURE

        # Sit to regen mana if needed before casting
        if pet_spell and state.mana_current < pet_spell.mana_cost:
            if not state.is_sitting:
                from motor.actions import sit

                sit()
                log.info(
                    "[PET] SummonPet: low mana (%d/%d, need %d) -- sitting to med",
                    state.mana_current,
                    state.mana_max,
                    pet_spell.mana_cost,
                )
            if state.mana_current < pet_spell.mana_cost:
                return RoutineStatus.RUNNING  # keep waiting for mana
            stand()
            interruptible_sleep(0.5, self._flee_check)

        log.info(
            "[PET] SummonPet: casting %s (gem %d), attempt %d",
            pet_spell.name if pet_spell else "?",
            pet_gem,
            self._retries + 1,
        )
        self._casting_started = True  # lock routine until acceptance check
        press_gem(pet_gem)

        # Non-blocking cast: create phase and return RUNNING
        cast_time = pet_spell.cast_time if pet_spell else 7.0
        from routines.casting import CastingPhase

        self._cast_phase = CastingPhase(
            cast_time, pet_spell.name if pet_spell else "pet summon", self._read_state_fn, timeout_buffer=3.0
        )
        return RoutineStatus.RUNNING

    def _check_summon_result(self, state: GameState) -> RoutineStatus:
        """Verify pet spawned after cast completed. Called on next tick."""
        check_state = self._read_state_fn() if self._read_state_fn else state
        has_pet = any(
            is_pet(s) and s.is_npc and s.hp_current > 0 and check_state.pos.dist_to(s.pos) < 100
            for s in check_state.spawns
        )
        if has_pet:
            pet_id, pet_name, pet_level = self._find_new_pet(check_state)

            log.log(EVENT, "[PET] SummonPet: pet '%s' level %d (id=%d)", pet_name, pet_level, pet_id)
            log_event(
                log,
                "summon_pet_result",
                f"[PET] Pet '{pet_name}' level {pet_level} summoned",
                pet_name=pet_name,
                pet_level=pet_level,
                pet_id=pet_id,
            )

            if self._ctx:
                self._ctx.pet.alive = True
                self._ctx.pet.spawn_id = pet_id
                self._ctx.pet.name = pet_name

            self._summoned = True
            return RoutineStatus.SUCCESS

        self._retries += 1
        log.info("[PET] SummonPet: pet not detected, retry %d/%d", self._retries, MAX_SUMMON_ATTEMPTS)
        interruptible_sleep(1.0, self._flee_check)
        return RoutineStatus.RUNNING

    @override
    def exit(self, state: GameState) -> None:
        pass
