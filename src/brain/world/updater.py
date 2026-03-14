"""World state updater: health check, world model, threat avoidance, player status.

Extracted from brain_runner.py to separate world observation from
brain loop orchestration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.world.health import HealthMonitor
    from brain.world.tracker import StateChangeTracker
    from perception.state import GameState

brain_log = logging.getLogger("compass.brain_loop")


class DeathHandler(Protocol):
    """Interface for the death-handling callback used by WorldStateUpdater."""

    def _handle_death(self, ctx: AgentContext, reason: str) -> bool: ...


class WorldStateUpdater:
    """Updates world model, tracks player status, detects death.

    Composed into BrainRunner -- not a subclass. The runner passes
    itself so the updater can access shared state (_handle_death).

    Args:
        runner: Any object implementing the DeathHandler protocol.
    """

    def __init__(self, runner: DeathHandler) -> None:
        self._runner = runner

    def update_world_state(
        self,
        state: GameState,
        ctx: AgentContext,
        health_monitor: HealthMonitor,
        state_tracker: StateChangeTracker,
    ) -> None:
        """Health check, world model update, threat avoidance, pet status."""
        health_monitor.tick(state, ctx)

        if hasattr(ctx, "world") and ctx.world:
            ctx.world.update(state)

            # Feed active threats into A* avoidance zones
            from nav.movement import get_terrain

            _terrain = get_terrain()
            if _terrain and hasattr(_terrain, "update_dynamic_avoidance"):
                threat_zones = []
                for tp in ctx.world.threats_within(300):
                    radius = tp.spawn.level * 3.0 + 40.0
                    threat_zones.append((tp.spawn.x, tp.spawn.y, radius))
                _terrain.update_dynamic_avoidance(threat_zones)

        ctx.update_pet_status(state)
        state_tracker.update(state, ctx)

    def check_player_status(self, state: GameState, ctx: AgentContext) -> bool:
        """Track position and detect death.

        Returns True if death detected (caller should break the loop).
        """
        # Track last known position (for corpse location on death)
        if state.hp_current > 0:
            ctx.player.last_known_x = state.x
            ctx.player.last_known_y = state.y
            ctx.player.last_known_z = state.z

        # Death detection from body_state + HP confirmation
        if state.is_dead and not ctx.player.dead and state.hp_current <= 0 and state.hp_max > 0:
            if self._runner._handle_death(ctx, "body_state"):
                return True

        # Death detection from memory (HP=0)
        if state.hp_current <= 0 and state.hp_max > 0 and not ctx.player.dead:
            if self._runner._handle_death(ctx, "hp_zero"):
                return True

        return False
