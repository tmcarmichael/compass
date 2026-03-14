"""AgentSession: explicit wiring layer for agent lifecycle resources.

Replaces implicit module-level singletons with an owned container.
Created once per agent start in orchestrator.py. Routines access via
ctx.session for explicit, testable resource paths.

Backward compatible: module-level flags and get_terrain() still work
(they delegate to the same instances). AgentSession is the opt-in
explicit path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.features import FeatureFlags
    from nav.movement import MovementController
    from nav.terrain.heightmap import ZoneTerrain
    from perception.reader import MemoryReader


@dataclass(slots=True)
class AgentSession:
    """Owns agent lifecycle resources for one session.

    Single source of truth for session wiring. Every resource that
    must survive across ticks but not across sessions lives here.

    Created by orchestrator.start_agent(), attached to ctx.session.
    """

    flags: FeatureFlags
    movement: MovementController
    terrain: ZoneTerrain | None = None
    reader: MemoryReader | None = None
    config: dict[str, Any] = field(default_factory=dict)
    zone_config: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""

    @classmethod
    def from_globals(cls, config: dict | None = None) -> AgentSession:
        """Create session wrapping current module-level singletons.

        Backward-compatible factory: uses the existing flags and
        _controller singletons so all code that imports them directly
        continues to work.
        """
        from core.features import flags
        from nav.movement import _controller

        return cls(
            flags=flags,
            movement=_controller,
            config=config or {},
        )

    def health_summary(self) -> dict[str, str]:
        """Return one-line health status for each subsystem.

        Used by SESSION READY log and diagnostic endpoints.
        """
        terrain_status = "None (no cache)"
        if self.terrain:
            s = self.terrain.stats
            terrain_status = f"{s.get('grid', '?')}, {s.get('obstacle', 0)} obstacles"

        return {
            "terrain": terrain_status,
            "movement": "terrain wired" if self.movement.terrain else "no terrain",
            "session_id": self.session_id or "?",
        }
