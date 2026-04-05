"""Base class, status enum, and sub-routine composition for routines."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING

from core.types import FailureCategory

if TYPE_CHECKING:
    from brain.context import AgentContext
    from core.types import ReadStateFn
    from perception.state import GameState

log = logging.getLogger(__name__)


def make_flee_predicate(read_state_fn: ReadStateFn, ctx: AgentContext | None) -> Callable[[], bool]:
    """Create a lightweight FLEE check for interruptible_sleep.

    Returns a callable that returns True when FLEE urgency exceeds the
    entry threshold. Used by combat/pull routines to break out of
    blocking sleeps when danger is detected.

    Catches all exceptions -- a failing predicate should never prevent
    the sleep from completing normally.
    """

    def check() -> bool:
        try:
            from brain.flee import (
                FLEE_URGENCY_ENTER,
                compute_flee_urgency,
            )

            if ctx is None:
                return False
            state = read_state_fn()
            urgency = compute_flee_urgency(ctx, state)
            should_flee: bool = urgency >= FLEE_URGENCY_ENTER
            return should_flee
        except Exception:
            return False

    return check


class RoutineStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class RoutineBase(ABC):
    """Abstract base for all routines (rest, travel, combat, etc.)."""

    failure_reason: str = ""
    failure_category: str | FailureCategory = FailureCategory.UNKNOWN
    # Cooperative tick budget: set by Brain before each tick() call.
    # Routines can check time.perf_counter() > _tick_deadline to yield early.
    _tick_deadline: float = 0.0

    @abstractmethod
    def enter(self, state: GameState) -> None:
        """Called when the routine is first activated."""

    @abstractmethod
    def tick(self, state: GameState) -> RoutineStatus:
        """Called each tick while the routine is active.

        Returns RoutineStatus indicating whether to keep running,
        or whether the routine succeeded/failed.
        """

    @abstractmethod
    def exit(self, state: GameState) -> None:
        """Called when the routine is deactivated."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def locked(self) -> bool:
        """If True, brain won't deactivate this routine for non-emergency rules.

        Override in subclasses that have critical phases (e.g., pull mid-flight).
        """
        return False


# ---------------------------------------------------------------------------
#  Sub-routine composition (roadmap 3.4)
# ---------------------------------------------------------------------------


class SubRoutine(ABC):
    """Lightweight routine fragment without brain integration.

    SubRoutines are composed inside a parent RoutineBase via RoutineComposer.
    They share the same enter/tick/exit lifecycle but are never registered
    as brain rules directly.

    Unlike RoutineBase, SubRoutines can declare themselves as not-ready
    (should_run returns False) so the composer skips them.
    """

    def enter(self, state: GameState) -> None:
        """Optional setup when parent routine activates."""
        return

    @abstractmethod
    def tick(self, state: GameState) -> RoutineStatus:
        """Execute one tick. Return RUNNING to keep going, SUCCESS/FAILURE to yield."""

    def exit(self, state: GameState) -> None:
        """Optional cleanup when parent routine deactivates."""
        return

    def should_run(self, state: GameState) -> bool:
        """Return True if this sub-routine has work to do right now.

        The composer calls this before tick(). Return False to let
        lower-priority sub-routines run instead. Default: always ready.
        """
        return True

    @property
    def name(self) -> str:
        return self.__class__.__name__


class RoutineComposer:
    """Runs multiple SubRoutines with micro-priority ordering.

    Each tick, evaluates sub-routines from highest priority (lowest number)
    to lowest. The first sub-routine whose should_run() returns True gets
    its tick() called. This mirrors the brain's rule pattern at routine scale.

    Usage inside a RoutineBase:

        class CombatRoutine(RoutineBase):
            def __init__(self):
                self.composer = RoutineComposer([
                    (10, PetHealSub()),    # highest priority
                    (20, SelfHealSub()),
                    (30, DotSub()),
                    (40, BurstSub()),
                    (50, MedSub()),        # lowest priority
                ])

            def enter(self, state):
                self.composer.enter(state)

            def tick(self, state):
                result = self.composer.tick(state)
                if result is not None:
                    return result
                return RoutineStatus.RUNNING  # nothing to do this tick

            def exit(self, state):
                self.composer.exit(state)
    """

    def __init__(self, sub_routines: list[tuple[int, SubRoutine]]) -> None:
        # Sort by priority (lowest number = highest priority)
        self._subs: list[tuple[int, SubRoutine]] = sorted(sub_routines, key=lambda x: x[0])
        self._active_name: str = ""

    def enter(self, state: GameState) -> None:
        """Forward enter to all sub-routines."""
        for _prio, sub in self._subs:
            sub.enter(state)
        self._active_name = ""

    def tick(self, state: GameState) -> RoutineStatus | None:
        """Evaluate sub-routines by priority. Returns result of winning sub, or None.

        Only one sub-routine runs per tick (highest priority that's ready).
        Returns None if no sub-routine is ready.
        """
        for _prio, sub in self._subs:
            if sub.should_run(state):
                result = sub.tick(state)
                if self._active_name != sub.name:
                    if self._active_name:
                        log.debug("SubRoutine: %s -> %s", self._active_name, sub.name)
                    self._active_name = sub.name
                return result
        self._active_name = ""
        return None

    def exit(self, state: GameState) -> None:
        """Forward exit to all sub-routines."""
        for _prio, sub in self._subs:
            sub.exit(state)
        self._active_name = ""

    @property
    def active_name(self) -> str:
        """Name of the sub-routine that ran last tick, or empty string."""
        return self._active_name

    def add(self, priority: int, sub: SubRoutine) -> None:
        """Add a sub-routine dynamically (e.g., when new spells unlock)."""
        self._subs.append((priority, sub))
        self._subs.sort(key=lambda x: x[0])

    def remove(self, name: str) -> bool:
        """Remove a sub-routine by class name. Returns True if found."""
        before = len(self._subs)
        self._subs = [(p, s) for p, s in self._subs if s.name != name]
        return len(self._subs) < before
