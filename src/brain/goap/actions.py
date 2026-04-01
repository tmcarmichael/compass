"""GOAP action definitions mapping to existing routines.

Each PlanAction has preconditions (what must be true), effects (how the
world state changes), and cost estimation (from learned data or heuristics).
Actions compose into plans; the planner searches for the cheapest sequence
that satisfies a goal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from brain.goap.world_state import PlanWorldState

if TYPE_CHECKING:
    from brain.context import AgentContext

log = logging.getLogger(__name__)

# Default heuristic costs (seconds) when learned data unavailable
_DEFAULT_COSTS: dict[str, float] = {
    "rest": 30.0,
    "acquire": 5.0,
    "pull": 8.0,
    "defeat": 25.0,
    "buff": 12.0,
    "summon_pet": 15.0,
    "memorize": 20.0,
    "wander": 30.0,
}


@dataclass(frozen=True, slots=True)
class PlanAction:
    """A single action the planner can include in a plan."""

    name: str
    routine_name: str  # maps to brain rule name (e.g., "REST")

    def preconditions_met(self, ws: PlanWorldState) -> bool:
        """Check if this action can execute in the given world state."""
        return False  # overridden by subclasses

    def apply_effects(self, ws: PlanWorldState) -> PlanWorldState:
        """Return new world state after this action executes."""
        return ws  # overridden by subclasses

    def estimate_cost(self, ctx: AgentContext | None) -> float:
        """Estimated duration in seconds from learned data or heuristic."""
        return _DEFAULT_COSTS.get(self.name, 20.0)


# -- Concrete Actions ---------------------------------------------------------


class RestAction(PlanAction):
    def preconditions_met(self, ws: PlanWorldState) -> bool:
        return not ws.engaged and ws.nearby_threats == 0

    def apply_effects(self, ws: PlanWorldState) -> PlanWorldState:
        return ws.with_changes(hp_pct=0.95, mana_pct=0.80)

    def estimate_cost(self, ctx: AgentContext | None) -> float:
        if not ctx:
            return _DEFAULT_COSTS["rest"]
        # Estimate from current mana deficit
        deficit = 0.80 - (ctx.combat.last_mana_pct if hasattr(ctx.combat, "last_mana_pct") else 0.5)
        return max(10.0, deficit * 60.0)  # rough: 1% mana per second


class AcquireAction(PlanAction):
    def preconditions_met(self, ws: PlanWorldState) -> bool:
        return ws.targets_available > 0 and ws.mana_pct > 0.25 and not ws.engaged and ws.pet_alive

    def apply_effects(self, ws: PlanWorldState) -> PlanWorldState:
        return ws.with_changes(has_target=True)

    def estimate_cost(self, ctx: AgentContext | None) -> float:
        return _DEFAULT_COSTS["acquire"]


class PullAction(PlanAction):
    def preconditions_met(self, ws: PlanWorldState) -> bool:
        return ws.has_target and not ws.engaged and ws.pet_alive

    def apply_effects(self, ws: PlanWorldState) -> PlanWorldState:
        return ws.with_changes(engaged=True)

    def estimate_cost(self, ctx: AgentContext | None) -> float:
        return _DEFAULT_COSTS["pull"]


class DefeatAction(PlanAction):
    """Combat until target is defeated."""

    def preconditions_met(self, ws: PlanWorldState) -> bool:
        engaged: bool = ws.engaged
        return engaged

    def apply_effects(self, ws: PlanWorldState) -> PlanWorldState:
        # Use learned average resource costs when available
        mana_delta = self._learned_mana_delta
        hp_delta = self._learned_hp_delta
        return ws.with_changes(
            engaged=False,
            has_target=False,
            corpse_nearby=True,
            mana_pct=max(0.0, ws.mana_pct - mana_delta),
            hp_pct=max(0.0, ws.hp_pct - hp_delta),
        )

    def estimate_cost(self, ctx: AgentContext | None) -> float:
        if not ctx or not ctx.fight_history:
            return _DEFAULT_COSTS["defeat"]
        # Update cached resource deltas from learned data
        self._update_learned_deltas(ctx)
        # Use average learned duration across recent encounters
        all_stats = ctx.fight_history.get_all_stats()
        if all_stats:
            durations: list[float] = [s.avg_duration for s in all_stats.values() if s.fights >= 3]
            if durations:
                avg: float = sum(durations) / len(durations)
                return avg
        return _DEFAULT_COSTS["defeat"]

    # Cached learned resource deltas (updated when estimate_cost is called)
    _learned_mana_delta: float = 0.30
    _learned_hp_delta: float = 0.10

    def _update_learned_deltas(self, ctx: AgentContext) -> None:
        """Update cached resource deltas from fight history averages."""
        fh = ctx.fight_history
        if not fh:
            return
        all_stats = fh.get_all_stats()
        if not all_stats:
            return
        # Average across all learned entity types
        mana_costs: list[float] = []
        hp_costs: list[float] = []
        for stats in all_stats.values():
            if stats.fights < 3:
                continue
            # avg_mana is absolute mana; convert to fraction of max
            max_mana = getattr(ctx, "_last_max_mana", 500)
            if max_mana > 0:
                mana_costs.append(stats.avg_mana / max_mana)
            hp_costs.append(stats.avg_hp_lost)
        if mana_costs:
            object.__setattr__(self, "_learned_mana_delta", sum(mana_costs) / len(mana_costs))
        if hp_costs:
            object.__setattr__(self, "_learned_hp_delta", sum(hp_costs) / len(hp_costs))


class BuffAction(PlanAction):
    def preconditions_met(self, ws: PlanWorldState) -> bool:
        return not ws.buffs_active and not ws.engaged and ws.nearby_threats == 0

    def apply_effects(self, ws: PlanWorldState) -> PlanWorldState:
        return ws.with_changes(buffs_active=True)

    def estimate_cost(self, ctx: AgentContext | None) -> float:
        return _DEFAULT_COSTS["buff"]


class SummonPetAction(PlanAction):
    def preconditions_met(self, ws: PlanWorldState) -> bool:
        return not ws.pet_alive and not ws.engaged and ws.mana_pct > 0.20

    def apply_effects(self, ws: PlanWorldState) -> PlanWorldState:
        return ws.with_changes(
            pet_alive=True,
            mana_pct=max(0.0, ws.mana_pct - 0.15),  # pet summon mana cost
        )

    def estimate_cost(self, ctx: AgentContext | None) -> float:
        return _DEFAULT_COSTS["summon_pet"]


class MemorizeAction(PlanAction):
    def preconditions_met(self, ws: PlanWorldState) -> bool:
        return not ws.spells_ready and not ws.engaged

    def apply_effects(self, ws: PlanWorldState) -> PlanWorldState:
        return ws.with_changes(spells_ready=True)

    def estimate_cost(self, ctx: AgentContext | None) -> float:
        return _DEFAULT_COSTS["memorize"]


class WanderAction(PlanAction):
    def preconditions_met(self, ws: PlanWorldState) -> bool:
        return ws.targets_available == 0 and not ws.engaged

    def apply_effects(self, ws: PlanWorldState) -> PlanWorldState:
        # Probabilistic: wandering may reveal targets
        return ws.with_changes(targets_available=1)

    def estimate_cost(self, ctx: AgentContext | None) -> float:
        """Use spawn prediction to reduce wander cost when respawns are imminent.

        If the spawn predictor has enough data, the expected time-to-next-respawn
        in nearby cells replaces the default heuristic.  This makes the planner
        prefer wander-then-fight plans when targets are predicted to appear soon,
        converting random wandering into directed positioning.
        """
        if not ctx or not ctx.spawn_predictor:
            return _DEFAULT_COSTS["wander"]
        import time as _time

        best = ctx.spawn_predictor.best_cells(3, _time.time())
        if best:
            # Use the shortest predicted wait among nearby cells
            min_wait = min(secs for _, secs in best)
            # Blend: at least 5s (travel time), at most the default
            return max(5.0, min(min_wait, _DEFAULT_COSTS["wander"]))
        return _DEFAULT_COSTS["wander"]


# -- Action Set ---------------------------------------------------------------


def build_action_set() -> list[PlanAction]:
    """Build the standard set of 11 actions for GOAP planning."""
    return [
        RestAction(name="rest", routine_name="REST"),
        AcquireAction(name="acquire", routine_name="ACQUIRE"),
        PullAction(name="pull", routine_name="PULL"),
        DefeatAction(name="defeat", routine_name="IN_COMBAT"),
        BuffAction(name="buff", routine_name="BUFF"),
        SummonPetAction(name="summon_pet", routine_name="SUMMON_PET"),
        MemorizeAction(name="memorize", routine_name="MEMORIZE_SPELLS"),
        WanderAction(name="wander", routine_name="WANDER"),
    ]
