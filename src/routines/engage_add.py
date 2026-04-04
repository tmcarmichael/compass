"""Engage add routine: detect and target a npc already fighting our pet.

When a social npc threatens the pet during a pull (e.g., spiderlings assist
each other), the agent may finish the primary target without noticing the add.
This routine quickly tabs to the add, sends pet attack, and sets engaged=True
so combat routine takes over.

Pet back off is ONLY used when redirecting pet to a NEW target (true add).
If the pet is already fighting this npc, just tab and engage -- no redirect.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from brain.context import AgentContext

from core.timing import interruptible_sleep
from core.types import FailureCategory, ReadStateFn
from motor.actions import face_heading, pet_attack, tab_target
from nav.geometry import heading_to
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus, make_flee_predicate

log = logging.getLogger(__name__)


class EngageAddRoutine(RoutineBase):
    """Quickly target and engage a npc that's already fighting our pet."""

    def __init__(self, ctx: AgentContext | None = None, read_state_fn: ReadStateFn | None = None) -> None:
        self._ctx = ctx
        self._read_state_fn = read_state_fn
        self._flee_check: Callable[[], bool] | None = None

    @override
    def enter(self, state: GameState) -> None:
        if self._read_state_fn and self._ctx:
            self._flee_check = make_flee_predicate(self._read_state_fn, self._ctx)
        else:
            self._flee_check = None
        log.info("[COMBAT] EngageAdd: detected npc fighting pet  -  engaging")

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        if not self._read_state_fn or not self._ctx:
            self.failure_reason = "no_context"
            self.failure_category = FailureCategory.PRECONDITION
            return RoutineStatus.FAILURE

        # Find the damaged NPC (the add fighting our pet)
        from perception.queries import is_pet

        add = None
        extra_npc_dist = 999.0
        for spawn in state.spawns:
            if (
                spawn.is_npc
                and not is_pet(spawn)
                and spawn.hp_max > 0
                and spawn.hp_current > 0
                and spawn.hp_current < spawn.hp_max
            ):
                d = state.pos.dist_to(spawn.pos)
                if d < 300 and d < extra_npc_dist:
                    add = spawn
                    extra_npc_dist = d

        if not add:
            log.info("[COMBAT] EngageAdd: no damaged NPC found, aborting")
            self.failure_reason = "no_add_found"
            self.failure_category = FailureCategory.PRECONDITION
            return RoutineStatus.FAILURE

        # Check if this is a genuinely NEW npc vs the one pet is already on
        is_new_add = self._ctx.combat.pull_target_id and add.spawn_id != self._ctx.combat.pull_target_id

        # Face the add
        exact = heading_to(state.pos, add.pos)
        jittered = (exact + random.gauss(0, 5.0)) % 512.0
        rsf = self._read_state_fn
        assert rsf is not None
        face_heading(jittered, lambda: rsf().heading, tolerance=10.0)

        # Tab to target it
        log.info(
            "[COMBAT] EngageAdd: targeting '%s' id=%d dist=%.0f HP=%d/%d new_add=%s",
            add.name,
            add.spawn_id,
            extra_npc_dist,
            add.hp_current,
            add.hp_max,
            is_new_add,
        )
        tab_target()
        interruptible_sleep(0.3, self._flee_check)

        # Verify we got a valid target (not our own pet)
        for _tab_attempt in range(2):
            check = self._read_state_fn()
            if check.target and check.target.is_npc and check.target.hp_current > 0:
                if is_pet(check.target):
                    log.info("[COMBAT] EngageAdd: tab landed on own pet, retrying")
                    tab_target()
                    interruptible_sleep(0.3, self._flee_check)
                    continue
                if is_new_add:
                    log.info("[COMBAT] EngageAdd: TRUE ADD  -  redirecting pet to '%s'", check.target.name)
                    from motor.actions import redirect_pet

                    redirect_pet()
                else:
                    log.info("[COMBAT] EngageAdd: same npc pet is fighting  -  just engaging")
                    pet_attack()

                self._ctx.combat.engaged = True
                self._ctx.player.engagement_start = time.time()
                self._ctx.combat.pull_target_id = check.target.spawn_id
                self._ctx.pet.has_add = False
                log.info(
                    "[COMBAT] EngageAdd: ENGAGED '%s' id=%d  -  transitioning to combat",
                    check.target.name,
                    check.target.spawn_id,
                )
                return RoutineStatus.SUCCESS

            # Tab didn't get it, try again
            tab_target()
            interruptible_sleep(0.3, self._flee_check)

        log.warning("[COMBAT] EngageAdd: failed to target add after 2 Tabs")
        self._ctx.pet.has_add = False  # clear anyway to prevent spam
        self.failure_reason = "target_failed"
        self.failure_category = FailureCategory.EXECUTION
        return RoutineStatus.FAILURE

    @override
    def exit(self, state: GameState) -> None:
        pass
