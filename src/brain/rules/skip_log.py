"""SkipLog and damaged_npcs_near -- shared helpers for brain rule modules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.types import Point
from util.log_tiers import VERBOSE

if TYPE_CHECKING:
    from typing import Protocol

    from brain.world.model import WorldModel
    from perception.state import GameState

    class _HasWorld(Protocol):
        world: WorldModel | None


def damaged_npcs_near(ctx: _HasWorld, state: GameState, pos: Point, radius: float) -> list[object]:
    """Get damaged NPCs within radius, with fallback if WorldModel is None."""
    world = ctx.world
    if world:
        return list(world.damaged_npcs_near(pos, radius))

    from perception.combat_eval import is_pet

    return [
        s
        for s in state.spawns
        if s.is_npc
        and not is_pet(s)
        and s.hp_max > 0
        and 0 < s.hp_current < s.hp_max
        and pos.dist_to(s.pos) < radius
    ]


class SkipLog:
    """Log rule skip reasons only on state transitions, not every 10 Hz tick.

    The brain evaluates all rules every tick. Lower-priority rules log "skip"
    reasons that repeat identically for seconds or minutes (e.g. "engaged in
    combat" fires 10x/sec for a 30s fight = 300 identical lines). This class
    deduplicates by tracking the last reason per rule name.
    """

    __slots__ = ("_last", "_log")

    def __init__(self, logger: logging.Logger) -> None:
        self._last: dict[str, str] = {}
        self._log = logger

    def __call__(self, rule: str, reason: str) -> None:
        """Log skip reason only when it differs from last call for this rule."""
        if self._last.get(rule) != reason:
            self._log.log(VERBOSE, "[DECISION] %s skip: %s", rule, reason)
            self._last[rule] = reason
