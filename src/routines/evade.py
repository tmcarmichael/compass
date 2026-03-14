"""Evade routine: sidestep away from approaching YELLOW/RED threats.

Non-blocking: uses move_to_point with a flee predicate so FLEE can
interrupt within ~100ms. Clears evasion_point on exit (handles
both success and interruption).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, override

from core.types import ReadStateFn
from nav.movement import move_to_point
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus, make_flee_predicate

if TYPE_CHECKING:
    from brain.context import AgentContext

log = logging.getLogger(__name__)


class EvadeRoutine(RoutineBase):
    """Sidestep to a pre-calculated evasion point to avoid a threat.

    Uses check_fn on move_to_point so FLEE can interrupt mid-movement.
    """

    def __init__(self, ctx: AgentContext, read_state_fn: ReadStateFn) -> None:
        self._ctx = ctx
        self._read_state_fn = read_state_fn
        self._done = False
        self._flee_check = make_flee_predicate(read_state_fn, ctx)

    @override
    def enter(self, state: GameState) -> None:
        self._done = False

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        if self._done or not self._ctx.threat.evasion_point:
            return RoutineStatus.SUCCESS
        ep = self._ctx.threat.evasion_point
        log.info("[POSITION] Evade: sidestepping to (%.0f, %.0f)", ep.x, ep.y)
        arrived = move_to_point(
            ep.x, ep.y, self._read_state_fn, arrival_tolerance=15.0, timeout=8.0, check_fn=self._flee_check
        )
        self._done = True
        if not arrived:
            log.info("[POSITION] Evade: movement interrupted (flee or timeout)")
        return RoutineStatus.SUCCESS

    @override
    def exit(self, state: GameState) -> None:
        # Clear evasion point on any exit (success or interruption)
        # so EVADE doesn't re-fire for a stale point next tick.
        self._ctx.threat.evasion_point = None
