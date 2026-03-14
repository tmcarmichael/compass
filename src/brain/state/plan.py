"""Strategic plan state  -  typed plan data instead of raw dicts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nav.zone_graph import ZoneConnection


@dataclass(slots=True, kw_only=True)
class TravelPlan:
    """Plan data for travel (both cross-zone and intra-zone waypoint)."""

    destination: str = ""
    route: list[ZoneConnection] = field(default_factory=list)
    hop_index: int = 0
    # Intra-zone waypoint travel
    waypoint: bool = False
    target_x: float = 0.0
    target_y: float = 0.0
    # Cross-zone hop tracking
    pre_hop_zone_id: int = 0


@dataclass(slots=True, kw_only=True)
class PlanState:
    """Current strategic plan (or None for normal grinding)."""

    active: str | None = None  # "travel", "needs_memorize", etc.
    travel: TravelPlan = field(default_factory=TravelPlan)
    data: dict[str, Any] = field(default_factory=dict)  # legacy -- migrate to typed fields

    def set_data(self, v: dict[str, Any]) -> None:
        """Set plan data dict and sync typed travel fields."""
        self.data = dict(v)
        if "route" in v:
            self.travel.route = v.get("route", [])
            self.travel.hop_index = v.get("hop_index", 0)
            self.travel.destination = v.get("destination", "")
        if v.get("waypoint"):
            self.travel.waypoint = True
            self.travel.target_x = v.get("target_x", 0.0)
            self.travel.target_y = v.get("target_y", 0.0)
            self.travel.destination = v.get("destination", "")

    def clear(self) -> None:
        """Reset to no active plan."""
        self.active = None
        self.travel = TravelPlan()
        self.data = {}
