"""Feign Death routine: cast FD to drop all threat.

Feign Death (level 16) is a 1.5s cast that makes the player appear dead.
All npcs reset threat. After 5-10s, stand up and verify npcs have left.
If FD is resisted or npcs don't reset, return FAILURE (brain falls through to FLEE).
"""

from __future__ import annotations

import logging
import random
import time
from enum import StrEnum
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from brain.context import AgentContext
    from routines.casting import CastingPhase

from core.timing import interruptible_sleep
from core.types import FailureCategory, ReadStateFn
from eq.loadout import SpellRole, get_spell_by_role
from motor.actions import press_gem, stand
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus

log = logging.getLogger(__name__)


class _FDPhase(StrEnum):
    """Feign Death routine phases."""

    CAST = "CAST"
    FEIGNING = "FEIGNING"
    STAND = "STAND"


# Stand state values observed in target client
_STAND_STATE_STANDING = 0
_STAND_STATE_SITTING = 1
_STAND_STATE_FEIGNING = 110  # client uses 110 for feign death

# How long to stay feigning before standing (jittered)
_FEIGN_WAIT_MIN = 5.0
_FEIGN_WAIT_MAX = 10.0
_FEIGN_MAX_DURATION = 30.0  # absolute cap on feign duration (prevents infinite extension)

# Distance within which an NPC is considered "still on us"
_HOSTILE_RADIUS = 30.0

# Max time to wait for cast to land + stand_state change
_CAST_TIMEOUT = 4.0


class FeignDeathRoutine(RoutineBase):
    """Cast Feign Death to drop all threat, wait, then stand and verify."""

    def __init__(self, ctx: AgentContext | None = None, read_state_fn: ReadStateFn | None = None) -> None:
        self._ctx = ctx
        self._read_state_fn = read_state_fn
        self._phase = _FDPhase.CAST  # CAST -> CASTING -> FEIGNING -> STAND
        self._phase_start = 0.0
        self._feign_duration = 0.0  # randomized each attempt
        self._locked = False
        self._cast_phase: CastingPhase | None = None

    @override
    @property
    def locked(self) -> bool:
        return self._locked

    @override
    def enter(self, state: GameState) -> None:
        self._phase = _FDPhase.CAST
        self._phase_start = time.time()
        self._feign_duration = 0.0
        self._locked = True
        self._cast_phase = None

        # Clear combat state -- we're trying to escape, not fight
        if self._ctx:
            self._ctx.combat.engaged = False
            self._ctx.combat.pull_target_id = None

        log.warning(
            "[CAST] FeignDeath: casting FD (HP=%.0f%%, mana=%d, pos=(%.0f,%.0f))",
            state.hp_pct * 100,
            state.mana_current,
            state.x,
            state.y,
        )

        # Stand up if sitting -- can't cast while sitting
        if state.is_sitting:
            stand()
            interruptible_sleep(0.4)

    def _nearby_hostile_count(self, state: GameState) -> int:
        """Count living NPCs within hostile radius of the player."""
        count = 0
        for s in state.spawns:
            if s.is_npc and s.hp_current > 0:
                d = state.pos.dist_to(s.pos)
                if d < _HOSTILE_RADIUS:
                    count += 1
        return count

    def _any_npc_approaching(self, old_state: GameState, new_state: GameState) -> bool:
        """Check if any NPC got closer between two state snapshots."""
        old_spawns = {s.spawn_id: s for s in old_state.spawns if s.is_npc}
        for s in new_state.spawns:
            if not s.is_npc or s.hp_current <= 0:
                continue
            d_new = new_state.pos.dist_to(s.pos)
            if d_new > _HOSTILE_RADIUS:
                continue
            old = old_spawns.get(s.spawn_id)
            if old is not None:
                d_old = old_state.pos.dist_to(old.pos)
                if d_new < d_old - 2.0:  # approaching (with noise margin)
                    return True
        return False

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        if not self._read_state_fn:
            log.error("[LIFECYCLE] FeignDeath: no read_state_fn")
            self.failure_reason = "no_context"
            self.failure_category = FailureCategory.PRECONDITION
            return RoutineStatus.FAILURE

        # -- Phase 1: CAST --
        if self._phase == _FDPhase.CAST:
            return self._tick_cast(state)

        # -- Phase 2: FEIGNING --
        if self._phase == _FDPhase.FEIGNING:
            return self._tick_feigning(state)

        # -- Phase 3: STAND --
        if self._phase == _FDPhase.STAND:
            return self._tick_stand(state)

        log.error("[LIFECYCLE] FeignDeath: unknown phase %s", self._phase)
        self.failure_reason = "unknown_phase"
        self.failure_category = FailureCategory.UNKNOWN
        return RoutineStatus.FAILURE

    def _tick_cast(self, state: GameState) -> RoutineStatus:
        """Press FD gem and wait for cast (non-blocking)."""
        spell = get_spell_by_role(SpellRole.UTILITY)
        if not spell or not spell.gem:
            log.warning("[CAST] FeignDeath: FD spell not memorized -- aborting")
            self.failure_reason = "no_spell"
            self.failure_category = FailureCategory.PRECONDITION
            return RoutineStatus.FAILURE

        # First call: start the cast
        if self._cast_phase is None:
            press_gem(spell.gem)
            from routines.casting import CastingPhase

            self._cast_phase = CastingPhase(spell.cast_time, "Feign Death", self._read_state_fn)
            return RoutineStatus.RUNNING

        # Subsequent calls: poll cast progress
        result = self._cast_phase.tick()
        from routines.casting import CastResult

        if result == CastResult.CASTING:
            return RoutineStatus.RUNNING

        # Cast complete -- clear phase and proceed to verification
        self._cast_phase = None

        # Brief settle time for stand_state to update
        interruptible_sleep(0.3)

        # Verify: check if stand_state changed to feigning
        assert self._read_state_fn is not None
        ns = self._read_state_fn()
        if ns.stand_state == _STAND_STATE_FEIGNING:
            log.info("[CAST] FeignDeath: FD landed -- feigning (stand_state=%d)", ns.stand_state)
        elif ns.stand_state in (_STAND_STATE_STANDING, _STAND_STATE_SITTING):
            # FD was likely resisted -- stand_state didn't change
            log.warning(
                "[CAST] FeignDeath: FD may have been RESISTED (stand_state=%d, expected %d)",
                ns.stand_state,
                _STAND_STATE_FEIGNING,
            )
            self.failure_reason = "resisted"
            self.failure_category = FailureCategory.EXECUTION
            return RoutineStatus.FAILURE
        else:
            # Unknown stand_state -- assume FD landed (might be a different
            # feign code than expected). Log it so we can calibrate.
            log.info(
                "[CAST] FeignDeath: unexpected stand_state=%d after cast "
                "-- assuming FD landed (may need calibration)",
                ns.stand_state,
            )

        # Transition to FEIGNING phase
        self._feign_duration = random.uniform(_FEIGN_WAIT_MIN, _FEIGN_WAIT_MAX)
        self._phase = _FDPhase.FEIGNING
        self._phase_start = time.time()
        log.info("[LIFECYCLE] FeignDeath: will feign for %.1fs", self._feign_duration)
        return RoutineStatus.RUNNING

    def _tick_feigning(self, state: GameState) -> RoutineStatus:
        """Wait on the ground while npcs reset threat."""
        elapsed = time.time() - self._phase_start

        # Poll periodically -- check if NPCs are still approaching
        poll_interval = 1.0
        remaining = self._feign_duration - elapsed
        wait_time = min(poll_interval, max(remaining, 0.1))
        interruptible_sleep(wait_time)

        assert self._read_state_fn is not None
        ns = self._read_state_fn()

        # Check if we're still feigning (EQ might have broken FD)
        if ns.stand_state not in (_STAND_STATE_FEIGNING,) and ns.stand_state > 2:
            # Still in some non-standing state -- probably fine
            pass
        elif ns.stand_state == _STAND_STATE_STANDING:
            # Something knocked us out of FD (damage, spell, etc.)
            log.warning(
                "[LIFECYCLE] FeignDeath: knocked out of FD at %.1fs (stand_state=%d)", elapsed, ns.stand_state
            )
            self.failure_reason = "knocked_out"
            self.failure_category = FailureCategory.ENVIRONMENT
            return RoutineStatus.FAILURE

        # Check if any NPC is still approaching (FD didn't stick)
        if self._any_npc_approaching(state, ns):
            log.info("[LIFECYCLE] FeignDeath: NPC still approaching at %.1fs -- extending feign", elapsed)
            # Extend the duration slightly (capped to prevent infinite feign)
            if remaining <= 0:
                if self._feign_duration < _FEIGN_MAX_DURATION:
                    self._feign_duration += 3.0
                else:
                    log.warning(
                        "[LIFECYCLE] FeignDeath: max feign duration (%.0fs) reached -- standing",
                        _FEIGN_MAX_DURATION,
                    )

        if elapsed >= self._feign_duration:
            # Time to stand up
            self._phase = _FDPhase.STAND
            self._phase_start = time.time()
            log.info("[LIFECYCLE] FeignDeath: feign complete (%.1fs) -- standing up", elapsed)

        return RoutineStatus.RUNNING

    def _tick_stand(self, state: GameState) -> RoutineStatus:
        """Stand up and verify npcs have left."""
        stand()
        interruptible_sleep(1.0)

        assert self._read_state_fn is not None
        ns = self._read_state_fn()

        hostile_count = self._nearby_hostile_count(ns)
        if hostile_count > 0:
            log.warning(
                "[LIFECYCLE] FeignDeath: FAILED -- %d hostile NPC(s) still "
                "within %.0fu after standing (HP=%.0f%%)",
                hostile_count,
                _HOSTILE_RADIUS,
                ns.hp_pct * 100,
            )
            self.failure_reason = "hostiles_remain"
            self.failure_category = FailureCategory.ENVIRONMENT
            return RoutineStatus.FAILURE

        log.info(
            "[LIFECYCLE] FeignDeath: SUCCESS -- no hostiles within %.0fu, HP=%.0f%%, mana=%d",
            _HOSTILE_RADIUS,
            ns.hp_pct * 100,
            ns.mana_current,
        )
        return RoutineStatus.SUCCESS

    @override
    def exit(self, state: GameState) -> None:
        self._locked = False
        self._phase = _FDPhase.CAST

        # Ensure player is standing (not still feigning)
        if state.stand_state != _STAND_STATE_STANDING:
            stand()

        log.info(
            "[LIFECYCLE] FeignDeath: exited -- HP=%.0f%% pos=(%.0f, %.0f)",
            state.hp_pct * 100,
            state.x,
            state.y,
        )
