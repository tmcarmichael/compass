"""Cast strategy interface for level-adaptive spell rotation.

Each strategy decides what to cast given the current fight state.
The CombatRoutine delegates spell selection to the active strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from perception.combat_eval import Con
from routines.base import RoutineStatus

if TYPE_CHECKING:
    from brain.context import AgentContext
    from perception.state import GameState, SpawnData


@runtime_checkable
class CombatCaster(Protocol):
    """Protocol for the combat routine's spell casting interface.

    Strategies use this to cast spells without importing CombatRoutine
    (which lives in routines/ -- below brain/ in the data flow).
    """

    def _cast_spell(
        self,
        gem: int,
        cast_time: float,
        now: float,
        state: GameState,
        target: SpawnData | None,
        *,
        is_dot: bool = False,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CastContext:
    """Immutable snapshot of fight state passed to spell strategies."""

    state: GameState
    target: SpawnData
    now: float
    dist: float
    target_hp: float  # 0.0-1.0
    tc: Con
    time_in_combat: float
    out_of_range: bool
    recently_sat: bool
    is_undead: bool
    has_adds: bool
    mob_on_player: bool
    pet_hp: float  # 0.0-1.0 or -1 if unknown
    pet_dist: float


class CastStrategy(ABC):
    """Base class for level-adapted spell rotation logic."""

    def __init__(self, combat_routine: CombatCaster, ctx: AgentContext) -> None:
        self._combat = combat_routine
        self._ctx = ctx

    @abstractmethod
    def execute(self, cc: CastContext) -> RoutineStatus | None:
        """Return RUNNING if cast started, None if nothing to cast."""
        ...

    def reset(self) -> None:
        """Reset per-fight state. Called on CombatRoutine.enter()."""
        return
