"""Threat detection state: approaching npcs, evasion points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.types import Point

if TYPE_CHECKING:
    from perception.state import SpawnData


@dataclass(slots=True, kw_only=True)
class ThreatState:
    """Threat detection results, set by HealthMonitor each tick."""

    approaching_threat: SpawnData | None = None
    imminent_threat: bool = False
    imminent_threat_con: str = ""
    evasion_point: Point | None = None
    patrol_evade: bool = False  # True = evasion_point is for RED patrol sidestep during combat

    def clear(self) -> None:
        self.approaching_threat = None
        self.imminent_threat = False
        self.imminent_threat_con = ""
        self.evasion_point = None
        self.patrol_evade = False
