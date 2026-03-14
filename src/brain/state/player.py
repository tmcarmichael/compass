"""Player state: death tracking, last known position."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, kw_only=True)
class PlayerState:
    """Player death and position tracking."""

    dead: bool = False
    deaths: int = 0
    last_known_x: float = 0.0
    last_known_y: float = 0.0
    last_known_z: float = 0.0
    engagement_start: float = 0.0
    last_buff_time: float = 0.0
    last_flee_time: float = 0.0
    rest_start_time: float = 0.0
    last_rest_hp: float = 1.0
