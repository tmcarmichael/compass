"""Combat phase state machine: non-blocking multi-step sequences.

Extracted from combat.py to keep the main CombatRoutine focused on
spell rotation and fight management. Phases replace blocking sleep()
sequences so the brain loop stays responsive (FLEE can fire between steps).
"""

from __future__ import annotations

import logging
import random
from enum import StrEnum
from typing import TYPE_CHECKING

from motor.actions import (
    move_forward_start,
    move_forward_stop,
    pet_attack,
)
from perception.state import GameState
from routines.base import RoutineStatus

if TYPE_CHECKING:
    from routines.combat import CombatRoutine

log = logging.getLogger(__name__)


class CombatPhase(StrEnum):
    """Non-blocking combat sub-phases that replace blocking sleep() sequences."""

    NONE = ""
    LOS_RECALL = "LOS_RECALL"
    LOS_WALK = "LOS_WALK"
    PET_RECALL = "PET_RECALL"
    BACKSTEP = "BACKSTEP"


class CombatPhaseManager:
    """Manages non-blocking phase state machine for combat sequences.

    Phases: LOS recall/walk (line-of-sight correction), pet recall
    (pull pet back from distance), and backstep (emergency distance).

    Args:
        combat: The parent CombatRoutine instance (for _stand_from_med,
                _face_target, _record_kill, _target_killed, _combat_recalled).
        read_state_fn: Callable to read fresh GameState.
    """

    def __init__(self, combat: CombatRoutine) -> None:
        self._combat = combat
        self.phase: str = CombatPhase.NONE
        self.deadline: float = 0.0
        self.data: dict = {}

    def reset(self) -> None:
        """Reset phase state. Call on combat start."""
        self.phase = CombatPhase.NONE
        self.deadline = 0.0
        self.data = {}

    @property
    def active(self) -> bool:
        """True if a non-blocking phase is in progress."""
        return bool(self.phase)

    def tick(self, state: GameState, now: float) -> RoutineStatus:
        """Handle non-blocking multi-step sequences. Returns RoutineStatus."""
        phase = self.phase

        # Check if target died during any phase -- early exit
        target = state.target
        if target and target.hp_current <= 0:
            log.info("[COMBAT] Combat: target died during %s phase -- ending phase", phase)
            self._cleanup(phase)
            self.phase = CombatPhase.NONE
            self.data.clear()
            self._combat._target_killed = True
            fight_time = now - self._combat._combat_start
            self._combat._record_kill(target, fight_time)
            return RoutineStatus.SUCCESS

        match phase:
            case CombatPhase.LOS_RECALL:
                return self._tick_los_recall(state, now)
            case CombatPhase.LOS_WALK:
                return self._tick_los_walk(state, now)
            case CombatPhase.PET_RECALL:
                return self._tick_pet_recall(state, now)
            case CombatPhase.BACKSTEP:
                return self._tick_backstep(state, now)
            case _:
                # Unknown phase -- clear and log
                log.warning("[COMBAT] Combat: unknown phase '%s', clearing", phase)
                self.phase = CombatPhase.NONE
                self.data.clear()
                return RoutineStatus.RUNNING

    def cleanup(self, phase: str | None = None) -> None:
        """Stop any in-progress movement when a phase ends early.

        If phase is None, uses the current phase.
        """
        self._cleanup(phase if phase is not None else self.phase)

    def _cleanup(self, phase: str) -> None:
        """Stop any in-progress movement when a phase ends early."""
        match phase:
            case CombatPhase.LOS_WALK | CombatPhase.BACKSTEP:
                move_forward_stop()

    def _tick_los_recall(self, state: GameState, now: float) -> RoutineStatus:
        """LOS_RECALL: pet back off, wait for disengage, then transition to LOS_WALK."""
        if not self.data.get("recalled"):
            # First tick: recall pet
            from motor.actions import _pet_command

            self._combat._stand_from_med()
            _pet_command("back off")
            self.data["recalled"] = True
            self.deadline = now + 1.5  # wait for pet to disengage
            log.info("[COMBAT] Combat: LOS phase -- pet recalled, waiting 1.5s")
            return RoutineStatus.RUNNING
        if now < self.deadline:
            return RoutineStatus.RUNNING
        # Transition: resend pet + walk toward npc
        pet_attack()
        self.phase = CombatPhase.LOS_WALK
        target = state.target
        if target:
            self._combat._face_target(state, target)
        move_forward_start()
        walk_time = random.uniform(0.8, 1.5)
        self.deadline = now + walk_time
        log.info("[COMBAT] Combat: LOS phase -- pet resent, walking %.1fs toward npc", walk_time)
        return RoutineStatus.RUNNING

    def _tick_los_walk(self, state: GameState, now: float) -> RoutineStatus:
        """LOS_WALK: walk toward npc to get LOS, then stop."""
        if now < self.deadline:
            return RoutineStatus.RUNNING
        move_forward_stop()
        self._combat._los_blocked_until = now + 3.0
        self.phase = CombatPhase.NONE
        self.data.clear()
        log.info("[COMBAT] Combat: LOS phase complete -- suppressing casts for 3s")
        return RoutineStatus.RUNNING

    def _tick_pet_recall(self, state: GameState, now: float) -> RoutineStatus:
        """PET_RECALL: wait after pet back off, then resend pet attack."""
        if now < self.deadline:
            return RoutineStatus.RUNNING
        # Re-check target (TOCTOU: npc may have died during wait)
        if self._combat._read_state_fn:
            ns = self._combat._read_state_fn()
            if not ns.target or ns.target.hp_current <= 0:
                log.info("[COMBAT] Combat: target died during pet recall")
                self._combat._combat_recalled = True
                self.phase = CombatPhase.NONE
                self.data.clear()
                return RoutineStatus.RUNNING  # next tick catches death
        pet_attack()
        self._combat._combat_recalled = True
        self.phase = CombatPhase.NONE
        self.data.clear()
        log.info("[COMBAT] Combat: pet recall complete -- pet resent")
        return RoutineStatus.RUNNING

    def _tick_backstep(self, state: GameState, now: float) -> RoutineStatus:
        """BACKSTEP: walk forward briefly to create distance from npc."""
        if now < self.deadline:
            return RoutineStatus.RUNNING
        move_forward_stop()
        self.phase = CombatPhase.NONE
        self.data.clear()
        log.info("[COMBAT] Combat: emergency backstep complete")
        return RoutineStatus.RUNNING
