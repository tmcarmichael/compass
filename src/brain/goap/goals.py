"""GOAP goal definitions with satisfaction and insistence functions.

Five persistent goals ordered by priority. The planner serves the most
insistent goal -- the one that most urgently needs attention. Goals are
evaluated on plan generation, not every tick.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from brain.goap.world_state import PlanWorldState

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Goal:
    """A persistent desire with measurable satisfaction."""

    name: str
    priority: int  # lower = higher priority (for tie-breaking)

    def satisfaction(self, ws: PlanWorldState) -> float:
        """0.0 = completely unsatisfied, 1.0 = fully satisfied."""
        return 0.0

    def insistence(self, ws: PlanWorldState) -> float:
        """How urgently this goal needs attention. Higher = more urgent.

        Combines dissatisfaction with priority weighting.
        """
        return (1.0 - self.satisfaction(ws)) * (1.0 / (1 + self.priority))


class SurviveGoal(Goal):
    """Stay alive. HP above safe threshold, no imminent threats."""

    def satisfaction(self, ws: PlanWorldState) -> float:
        if ws.hp_pct <= 0:
            return 0.0
        s: float = ws.hp_pct
        if ws.nearby_threats > 0:
            s *= 0.5
        return min(1.0, s)

    def insistence(self, ws: PlanWorldState) -> float:
        if ws.hp_pct < 0.30:
            return 1.0  # critical: override everything
        if ws.nearby_threats > 0 and ws.hp_pct < 0.50:
            return 0.9
        return (1.0 - self.satisfaction(ws)) * 1.0  # highest weight


class MaintainReadinessGoal(Goal):
    """Keep pet alive, buffs active, spells memorized."""

    def satisfaction(self, ws: PlanWorldState) -> float:
        s = 1.0
        if not ws.pet_alive:
            s -= 0.4
        if not ws.buffs_active:
            s -= 0.3
        if not ws.spells_ready:
            s -= 0.3
        return max(0.0, s)


class ManageResourcesGoal(Goal):
    """Maintain mana and HP above comfortable thresholds."""

    def satisfaction(self, ws: PlanWorldState) -> float:
        # Mana is the primary resource constraint
        mana_sat: float = min(1.0, ws.mana_pct / 0.70)  # 70% = fully satisfied
        hp_sat: float = min(1.0, ws.hp_pct / 0.90)  # 90% = fully satisfied
        result: float = mana_sat * 0.7 + hp_sat * 0.3  # mana-weighted
        return result


class GainXPGoal(Goal):
    """Defeat NPCs to accumulate experience."""

    def satisfaction(self, ws: PlanWorldState) -> float:
        if ws.corpse_nearby:
            return 0.85  # recent defeat completed the grind cycle
        # Always partially unsatisfied (always want more XP)
        # Satisfaction increases when targets are available and resources permit
        if ws.targets_available == 0:
            return 0.3  # no targets: partially satisfied (can't grind)
        if ws.mana_pct < 0.25:
            return 0.4  # low mana: can't fight effectively
        if ws.engaged:
            return 0.6  # in combat: making progress
        return 0.5  # targets available, ready to engage

    def insistence(self, ws: PlanWorldState) -> float:
        # Only insistent when we have resources AND targets
        if ws.mana_pct < 0.25 or ws.hp_pct < 0.40:
            return 0.1  # too depleted to fight
        if ws.targets_available == 0:
            return 0.2  # nothing to fight
        return (1.0 - self.satisfaction(ws)) * 0.7  # steady pull


class ManageInventoryGoal(Goal):
    """Keep inventory weight under control."""

    def satisfaction(self, ws: PlanWorldState) -> float:
        # Fully satisfied when inventory < 70%, drops as it fills
        if ws.inventory_pct < 0.70:
            return 1.0
        # Linear dropoff from 0.7 to 1.0
        dropoff: float = max(0.0, 1.0 - (ws.inventory_pct - 0.70) / 0.30)
        return dropoff

    def insistence(self, ws: PlanWorldState) -> float:
        if ws.inventory_pct > 0.90:
            return 0.8  # urgent: almost full
        if ws.inventory_pct > 0.80:
            return 0.5  # should offload soon
        return (1.0 - self.satisfaction(ws)) * 0.3  # low priority


# -- Goal Set Construction ----------------------------------------------------


def build_goal_set() -> list[Goal]:
    """Build the standard set of 5 goals ordered by priority."""
    return [
        SurviveGoal(name="SURVIVE", priority=0),
        MaintainReadinessGoal(name="MAINTAIN_READINESS", priority=1),
        ManageResourcesGoal(name="MANAGE_RESOURCES", priority=2),
        GainXPGoal(name="GAIN_XP", priority=3),
        ManageInventoryGoal(name="MANAGE_INVENTORY", priority=4),
    ]
