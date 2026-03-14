"""Simplified world state for GOAP planning.

A frozen, hashable snapshot of the properties that matter for action
preconditions and effects. Built from GameState + AgentContext each time
a plan is generated. Small enough to copy cheaply for A* state-space search.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from brain.context import AgentContext
    from perception.state import GameState


@dataclass(frozen=True, slots=True)
class PlanWorldState:
    """Hashable world state for the GOAP planner's state-space search."""

    hp_pct: float = 1.0  # 0.0-1.0
    mana_pct: float = 1.0  # 0.0-1.0
    pet_alive: bool = True
    engaged: bool = False
    has_target: bool = False
    corpse_nearby: bool = False
    buffs_active: bool = True
    spells_ready: bool = True
    inventory_pct: float = 0.0  # 0.0-1.0 (weight/capacity)
    at_camp: bool = True  # within roam radius
    targets_available: int = 0  # scored targets with score > 0
    nearby_threats: int = 0  # hostile NPCs approaching

    def with_changes(self, **kwargs: Any) -> PlanWorldState:
        """Return a copy with specified fields changed (delegates to dataclasses.replace)."""
        return replace(self, **kwargs)


def build_world_state(state: GameState, ctx: AgentContext) -> PlanWorldState:
    """Build PlanWorldState from current GameState and AgentContext."""
    # Target availability from world model
    targets = 0
    if hasattr(ctx, "world") and ctx.world is not None:
        targets = len([t for t in getattr(ctx.world, "_targets", []) if t.score > 0])

    # Nearby threats
    threats = 0
    if hasattr(ctx, "world") and ctx.world is not None:
        threats = len(getattr(ctx.world, "_threats", []))

    # Corpse nearby (unlootable corpse within range)
    corpse = ctx.has_unlootable_corpse(state, max_dist=100.0)

    # Buffs active (check if any tracked buff is missing)
    buffs = True
    if hasattr(state, "buffs") and state.buffs is not None:
        # If we have buff data and no active buffs, they're not active
        # This is a simplification -- the buff routine checks specific buffs
        buffs = len(state.buffs) > 0

    # Inventory percentage (weight gained vs threshold)
    inv_pct = 0.0
    if ctx.inventory.weight_threshold > 0:
        gained = (
            state.weight - ctx.inventory.weight_baseline
            if state.weight > ctx.inventory.weight_baseline
            else 0
        )
        inv_pct = gained / ctx.inventory.weight_threshold

    # At camp
    at_camp = True
    if ctx.camp.roam_radius > 0:
        from core.types import Point

        d = state.pos.dist_to(Point(ctx.camp.camp_x, ctx.camp.camp_y, 0.0))
        at_camp = d <= ctx.camp.roam_radius * 1.2  # slight buffer

    return PlanWorldState(
        hp_pct=state.hp_pct,
        mana_pct=state.mana_pct,
        pet_alive=ctx.pet.alive,
        engaged=ctx.combat.engaged,
        has_target=state.has_target,
        corpse_nearby=corpse,
        buffs_active=buffs,
        spells_ready=True,  # assume ready unless memorize rule fires
        inventory_pct=min(1.0, inv_pct),
        at_camp=at_camp,
        targets_available=targets,
        nearby_threats=threats,
    )
