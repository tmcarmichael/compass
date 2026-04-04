"""Narrowed view protocols for AgentContext consumers.

Each rule module depends on a subset of AgentContext's surface area.
These Protocol classes make that dependency explicit at the type level:
mypy enforces that closures only access attributes defined on their view.

AgentContext satisfies all views via structural subtyping -- no changes
to AgentContext are needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from brain.learning.encounters import FightHistory
    from brain.state.camp import CampConfig
    from brain.state.combat import CombatState
    from brain.state.kill_tracker import DefeatTracker
    from brain.state.loot_config import LootConfig
    from brain.state.pet import PetState
    from brain.state.plan import PlanState
    from brain.state.player import PlayerState
    from brain.state.threat import ThreatState
    from brain.state.zone import ZoneState
    from brain.world.model import WorldModel
    from nav.travel_planner import TunnelRoute
    from nav.waypoint_graph import WaypointGraph
    from perception.state import GameState


@runtime_checkable
class SurvivalView(Protocol):
    """Context surface used by survival rules (FLEE, REST, FEIGN_DEATH, etc.)."""

    combat: CombatState
    pet: PetState
    player: PlayerState
    threat: ThreatState
    defeat_tracker: DefeatTracker
    fight_history: FightHistory | None
    world: WorldModel | None

    # Rest thresholds
    rest_hp_entry: float
    rest_mana_entry: float
    rest_hp_threshold: float
    rest_mana_threshold: float

    @property
    def in_active_combat(self) -> bool: ...


@runtime_checkable
class CombatView(Protocol):
    """Context surface used by combat rules (IN_COMBAT, ACQUIRE, PULL, etc.)."""

    combat: CombatState
    pet: PetState
    zone: ZoneState
    plan: PlanState
    defeat_tracker: DefeatTracker
    fight_history: FightHistory | None
    loot: LootConfig
    world: WorldModel | None

    @property
    def in_active_combat(self) -> bool: ...

    def nearby_player_count(self, state: GameState, radius: float = ...) -> int: ...


@runtime_checkable
class MaintenanceView(Protocol):
    """Context surface used by maintenance rules (BUFF, SUMMON_PET, etc.)."""

    combat: CombatState
    pet: PetState
    plan: PlanState
    player: PlayerState


@runtime_checkable
class NavigationView(Protocol):
    """Context surface used by navigation rules (TRAVEL, WANDER)."""

    combat: CombatState
    pet: PetState
    plan: PlanState
    camp: CampConfig
    world: WorldModel | None
    tunnel_routes: list[TunnelRoute]
    waypoint_graph: WaypointGraph | None
