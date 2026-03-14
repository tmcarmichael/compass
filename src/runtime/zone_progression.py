"""Zone progression: level-based zone mapping and travel initiation.

Extracted from agent.py (A-3) to keep build_context under 80 LOC.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.types import PlanType

if TYPE_CHECKING:
    from brain.context import AgentContext

log = logging.getLogger(__name__)

# Zone progression order for agent leveling path.
# (zone_short_name, min_level, max_level)
# Agent moves to next zone when player level exceeds current zone's max.
# Configure this for your environment -- list zones in leveling order.
ZONE_PROGRESSION: list[tuple[str, int, int]] = [
    # ("zone_name", min_level, max_level),
]


def check_zone_progression(
    ctx: AgentContext,
    current_zone: str,
    player_level: int,
    client_path: str = "",
    progression: list[tuple[str, int, int]] | None = None,
) -> str | None:
    """Check if the player should move to a new zone.

    Returns target zone short name if progression needed, None otherwise.
    Called after check_camp_progression finds no viable in-zone camp.
    """
    _table = progression if progression is not None else ZONE_PROGRESSION
    # Find current zone in progression order
    current_idx = None
    for i, (zone, lo, hi) in enumerate(_table):
        if zone == current_zone:
            current_idx = i
            break

    if current_idx is None:
        log.info("[TRAVEL] Zone progression: '%s' not in progression order", current_zone)
        return None

    # Check if current zone still has viable level range
    _, _, current_max = _table[current_idx]
    if player_level <= current_max:
        log.info(
            "[TRAVEL] Zone progression: level %d still fits '%s' (max=%d)",
            player_level,
            current_zone,
            current_max,
        )
        return None

    # Find next viable zone
    for zone, lo, hi in _table[current_idx + 1 :]:
        if lo <= player_level <= hi:
            log.info(
                "[TRAVEL] Zone progression: level %d outgrew '%s' -> '%s' (range %d-%d)",
                player_level,
                current_zone,
                zone,
                lo,
                hi,
            )
            return zone

    log.warning(
        "[TRAVEL] Zone progression: no viable zone for level %d after '%s'", player_level, current_zone
    )
    return None


def initiate_zone_travel(
    ctx: AgentContext, current_zone: str, target_zone: str, client_path: str = ""
) -> bool:
    """Set up a travel plan to move to target_zone.

    Returns True if travel plan was set up, False if no route found.
    """
    from pathlib import Path

    from nav.zone_graph import build_zone_graph

    maps_dir = Path(client_path) / "maps" if client_path else Path()
    graph = build_zone_graph(maps_dir)
    route = graph.find_route(current_zone, target_zone)

    if not route:
        log.warning("[TRAVEL] Zone travel: no route from '%s' to '%s'", current_zone, target_zone)
        return False

    log.info(
        "[TRAVEL] Zone travel: route %s -> %s (%d hops: %s)",
        current_zone,
        target_zone,
        len(route),
        " -> ".join(c.to_zone for c in route),
    )

    ctx.plan.active = PlanType.TRAVEL
    ctx.plan.set_data(
        {
            "route": route,
            "destination": target_zone,
            "hop_index": 0,
            "zone_progression": True,  # flag for post-travel zone reload
        }
    )
    return True
