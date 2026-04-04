"""Death recovery routine: respawn at bind point, return to camp, resume.

When the player dies in EQ:
1. They appear at their bind point (a fixed zone location)
2. They need to run back to their corpse OR continue grinding without it
3. They need to resummon pet, rebuff, and re-engage

For now, we skip corpse recovery (requires inventory reading) and just
return to the current camp to resume grinding.

Phases:
  WAIT_RESPAWN  -  wait for HP > 0 (player respawns)
  STABILIZE     -  wait for zone to load, stand up
  RETURN_CAMP   -  navigate back to grinding camp
  READY         -  done, brain rules handle pet/buff/grind
"""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, override

from core.timing import interruptible_sleep

if TYPE_CHECKING:
    from brain.context import AgentContext
from core.types import FailureCategory, Point, ReadStateFn
from nav.movement import move_to_point
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus, make_flee_predicate

log = logging.getLogger(__name__)

# -- Domain-local constants --
CORPSE_RECOVERY_TIMEOUT = 120.0  # max seconds to search for corpse


class _Phase(enum.IntEnum):
    WAIT_RESPAWN = 0
    STABILIZE = 1
    RETURN_CAMP = 2
    READY = 3


class DeathRecoveryRoutine(RoutineBase):
    """Handle death recovery: wait for respawn, return to camp."""

    def __init__(self, ctx: AgentContext | None = None, read_state_fn: ReadStateFn | None = None) -> None:
        self._ctx = ctx
        self._read_state_fn = read_state_fn
        self._phase = _Phase.WAIT_RESPAWN
        self._respawn_start = 0.0
        self._stabilize_start = 0.0
        self._flee_check: Callable[[], bool] | None = None

    @override
    @property
    def locked(self) -> bool:
        """Lock during respawn wait and stabilization."""
        return self._phase in (_Phase.WAIT_RESPAWN, _Phase.STABILIZE)

    @override
    def enter(self, state: GameState) -> None:
        self._phase = _Phase.WAIT_RESPAWN
        self._respawn_start = time.time()
        if self._read_state_fn and self._ctx:
            self._flee_check = make_flee_predicate(self._read_state_fn, self._ctx)
        else:
            self._flee_check = None
        log.info(
            "[LIFECYCLE] DeathRecovery: starting recovery. Last known pos=(%.0f, %.0f)",
            self._ctx.player.last_known_x if self._ctx else 0,
            self._ctx.player.last_known_y if self._ctx else 0,
        )

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        if self._phase == _Phase.WAIT_RESPAWN:
            return self._tick_wait_respawn(state)
        elif self._phase == _Phase.STABILIZE:
            return self._tick_stabilize(state)
        elif self._phase == _Phase.RETURN_CAMP:
            return self._tick_return_camp(state)
        return RoutineStatus.SUCCESS

    @override
    def exit(self, state: GameState) -> None:
        log.info("[LIFECYCLE] DeathRecovery: exit phase=%s", self._phase.name)

    def _tick_wait_respawn(self, state: GameState) -> RoutineStatus:
        """Wait for player HP to become > 0 (respawned)."""
        elapsed = time.time() - self._respawn_start

        if state.hp_current > 0 and state.hp_max > 0:
            log.info(
                "[LIFECYCLE] DeathRecovery: respawned! HP=%d/%d pos=(%.0f, %.0f) after %.1fs",
                state.hp_current,
                state.hp_max,
                state.x,
                state.y,
                elapsed,
            )
            self._phase = _Phase.STABILIZE
            self._stabilize_start = time.time()
            return RoutineStatus.RUNNING

        # Timeout after 120s (loading screen + respawn)
        if elapsed > CORPSE_RECOVERY_TIMEOUT:
            log.warning("[LIFECYCLE] DeathRecovery: respawn timeout after %.0fs", elapsed)
            self.failure_reason = "timeout"
            self.failure_category = FailureCategory.TIMEOUT
            return RoutineStatus.FAILURE

        # Wait patiently
        if int(elapsed) % 10 == 0 and elapsed > 1:
            log.info("[LIFECYCLE] DeathRecovery: waiting for respawn... %.0fs", elapsed)
        interruptible_sleep(1.0, self._flee_check)
        return RoutineStatus.RUNNING

    def _tick_stabilize(self, state: GameState) -> RoutineStatus:
        """Wait a few seconds for zone to stabilize after respawn."""
        elapsed = time.time() - self._stabilize_start
        if elapsed < 5.0:
            interruptible_sleep(1.0, self._flee_check)
            return RoutineStatus.RUNNING

        # Stand up (respawn puts you sitting)
        from motor.actions import stand

        if state.stand_state != 0:  # 0 = standing
            stand()
            interruptible_sleep(1.0, self._flee_check)

        log.info(
            "[LIFECYCLE] DeathRecovery: stabilized. Pos=(%.0f, %.0f) HP=%.0f%%",
            state.x,
            state.y,
            state.hp_pct * 100,
        )

        # Reset death state so brain rules can resume
        if self._ctx:
            self._ctx.player.dead = False
            self._ctx.pet.alive = False
            self._ctx.pet.spawn_id = None
            self._ctx.pet.name = ""
            self._ctx.combat.engaged = False
            self._ctx.combat.pull_target_id = None

        # Check if we're already near camp
        if self._ctx:
            camp_dist = state.pos.dist_to(self._ctx.camp.camp_pos)
            if camp_dist < 100:
                log.info("[LIFECYCLE] DeathRecovery: already near camp (dist=%.0f), done", camp_dist)
                self._phase = _Phase.READY
                return RoutineStatus.SUCCESS

        self._phase = _Phase.RETURN_CAMP
        return RoutineStatus.RUNNING

    def _tick_return_camp(self, state: GameState) -> RoutineStatus:
        """Walk back to camp center."""
        if not self._ctx or not self._read_state_fn:
            return RoutineStatus.SUCCESS

        camp_x = self._ctx.camp.camp_pos.x
        camp_y = self._ctx.camp.camp_pos.y
        dist = state.pos.dist_to(self._ctx.camp.camp_pos)

        if dist < 50:
            log.info(
                "[LIFECYCLE] DeathRecovery: arrived at camp (dist=%.0f). Resuming normal operation.", dist
            )
            self._phase = _Phase.READY
            return RoutineStatus.SUCCESS

        log.info("[LIFECYCLE] DeathRecovery: walking to camp (%.0f, %.0f) dist=%.0f", camp_x, camp_y, dist)
        arrived = move_to_point(
            Point(camp_x, camp_y, 0.0), self._read_state_fn, arrival_tolerance=40.0, timeout=60.0
        )

        if arrived:
            log.info("[LIFECYCLE] DeathRecovery: arrived at camp. Resuming.")
            self._phase = _Phase.READY
            return RoutineStatus.SUCCESS

        log.warning("[LIFECYCLE] DeathRecovery: failed to reach camp, retrying")
        return RoutineStatus.RUNNING
