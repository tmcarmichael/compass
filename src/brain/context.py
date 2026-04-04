"""AgentContext: mutable state shared across brain rules and routines.

Contains 12 focused sub-state objects for clean grouping:
  ctx.combat     -  CombatState (engaged, pull_target_id, dot/lifetap timers)
  ctx.pet        -  PetState (alive, spawn_id, name, has_add)
  ctx.camp       -  CampConfig (camp_pos, guard_pos, hunt zone, danger points)
  ctx.inventory  -  InventoryState (weight tracking, loot count)
  ctx.plan       -  PlanState (typed travel plans)
  ctx.player     -  PlayerState (death, position, engagement)
  ctx.defeat_tracker - DefeatTracker (defeats, corpse matching)
  ctx.metrics    -  SessionMetrics (counters, timing, XP)
  ctx.threat     -  ThreatState (approaching npcs, evasion)
  ctx.loot       -  LootConfig (resource targets, npc knowledge)
  ctx.zone       -  ZoneState (zone config, dispositions)
  ctx.diag       -  DiagnosticState (rule eval, events)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from brain.state.camp import CampConfig
from brain.state.combat import CombatState
from brain.state.diagnostic import DiagnosticState
from brain.state.inventory import InventoryState
from brain.state.kill_tracker import DefeatTracker
from brain.state.loot_config import LootConfig
from brain.state.metrics import SessionMetrics
from brain.state.pet import PetState
from brain.state.plan import PlanState
from brain.state.player import PlayerState
from brain.state.threat import ThreatState
from brain.state.zone import ZoneState
from core.types import Point
from perception.state import GameState
from util.thread_guard import assert_brain_thread

if TYPE_CHECKING:
    from brain.goap.spawn_predictor import SpawnPredictor
    from brain.learning.danger_memory import DangerMemory
    from brain.learning.encounters import FightHistory
    from brain.learning.spatial import SpatialMemory
    from brain.world.model import WorldModel
    from nav.travel_planner import TunnelRoute
    from nav.waypoint_graph import WaypointGraph
    from perception.reader import MemoryReader
    from runtime.agent_session import AgentSession

log = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Mutable agent state that persists across ticks.

    Thread ownership (concurrency-audit.md):
      BRAIN-ONLY: Written by brain thread, read by brain thread only.
        No cross-thread reads. Safe without lock.
      RULE-SAFE: Written BEFORE rule eval (brain_runner tick start).
        Rules see current-tick values. Routines see same-tick values.
      ROUTINE-WRITTEN: Written by routines DURING tick (after rule eval).
        Rules see PREVIOUS tick's value (1 tick behind, ~100ms).
      CROSS-THREAD: Read by display/reporting thread. Simple types (int/float/bool/str)
        are GIL-atomic. Dicts/lists need lock if iterated by secondary thread.
    """

    # -- Session (explicit wiring layer) --
    session: AgentSession | None = None

    # -- Thread safety --
    lock: threading.Lock = field(default_factory=threading.Lock)

    # -- Focused sub-state objects --
    combat: CombatState = field(default_factory=CombatState)
    pet: PetState = field(default_factory=PetState)
    camp: CampConfig = field(default_factory=CampConfig)
    inventory: InventoryState = field(default_factory=InventoryState)
    plan: PlanState = field(default_factory=PlanState)

    # -- Sub-state objects (Phase 2 decomposition) --
    player: PlayerState = field(default_factory=PlayerState)
    defeat_tracker: DefeatTracker = field(default_factory=DefeatTracker)
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    threat: ThreatState = field(default_factory=ThreatState)
    loot: LootConfig = field(default_factory=LootConfig)
    zone: ZoneState = field(default_factory=ZoneState)
    diag: DiagnosticState = field(default_factory=DiagnosticState)

    # -- Infrastructure (shared refs, not sub-state) --
    stop_event: threading.Event | None = None  # set by brain_runner for session_end
    reader: MemoryReader | None = None
    world: WorldModel | None = None
    spatial_memory: SpatialMemory | None = None
    fight_history: FightHistory | None = None
    danger_memory: DangerMemory | None = None
    spawn_predictor: SpawnPredictor | None = None
    tunnel_routes: list[TunnelRoute] = field(default_factory=list)
    waypoint_graph: WaypointGraph | None = None

    # -- Rest thresholds (shared between rule and routine) --
    rest_hp_entry: float = 0.85
    rest_mana_entry: float = 0.40
    rest_hp_threshold: float = 0.92
    rest_mana_threshold: float = 0.60

    # ==================================================================
    # Convenience properties
    # ==================================================================

    @property
    def in_active_combat(self) -> bool:
        """True if engaged or mid-pull."""
        return self.combat.engaged or self.combat.pull_target_id is not None

    # ==================================================================
    # Methods (delegating where appropriate)
    # ==================================================================

    def begin_engagement(
        self, target_id: int, name: str = "", x: float = 0.0, y: float = 0.0, level: int = 0
    ) -> None:
        """Atomically set all engagement-related fields."""
        assert_brain_thread("begin_engagement")
        with self.lock:
            self.combat.engaged = True
            self.combat.pull_target_id = target_id
            self.defeat_tracker.last_fight_name = name
            self.defeat_tracker.last_fight_id = target_id
            self.defeat_tracker.last_fight_x = x
            self.defeat_tracker.last_fight_y = y
            self.defeat_tracker.last_fight_level = level
            self.player.engagement_start = time.time()

    def clear_engagement(self) -> None:
        """Atomically clear all engagement-related fields."""
        assert_brain_thread("clear_engagement")
        with self.lock:
            self.combat.engaged = False
            self.combat.pull_target_id = None
            self.defeat_tracker.last_fight_name = ""
            self.defeat_tracker.last_fight_id = 0
            self.defeat_tracker.last_fight_x = 0.0
            self.defeat_tracker.last_fight_y = 0.0
            self.player.engagement_start = 0.0

    def update_pet_status(self, state: GameState) -> None:
        assert_brain_thread("update_pet_status")
        self.pet.update(state)

    def has_unlootable_corpse(self, state: GameState, max_dist: float = 120.0) -> bool:
        self.defeat_tracker.clean_kill_history()

        for spawn in state.spawns:
            if not spawn.is_corpse:
                continue
            dist = state.pos.dist_to(spawn.pos)
            if dist > max_dist:
                continue
            defeat = self.defeat_tracker.find_unlootable_kill(
                spawn.name, spawn.pos, corpse_spawn_id=spawn.spawn_id
            )
            if defeat:
                return True
            # Log WHY matching failed
            for k in self.defeat_tracker.defeat_history:
                if not k.looted and time.time() - k.time < 300:
                    kdist = spawn.pos.dist_to(k.pos)
                    name_match = k.name in spawn.name or spawn.name.startswith(k.name)
                    log.debug(
                        "Loot match: corpse='%s'@(%.0f,%.0f) vs defeat='%s'@(%.0f,%.0f) "
                        "name_match=%s dist=%.0f (max=60)",
                        spawn.name,
                        spawn.x,
                        spawn.y,
                        k.name,
                        k.x,
                        k.y,
                        name_match,
                        kdist,
                    )
        return False

    _ORIGIN = Point(0.0, 0.0, 0.0)

    def record_kill(self, spawn_id: int, name: str = "", pos: Point = _ORIGIN) -> None:
        """Record a defeat with thread-safety assertion."""
        assert_brain_thread("record_kill")
        self.defeat_tracker.record_kill(spawn_id, name=name, pos=pos)

    def nearest_player_dist(self, state: GameState) -> float:
        best = 9999.0
        for spawn in state.spawns:
            if spawn.spawn_type == 0 and spawn.name != state.name:
                dist = state.pos.dist_to(spawn.pos)
                if dist < best:
                    best = dist
        return best

    def nearby_player_count(self, state: GameState, radius: float = 200.0) -> int:
        count = 0
        for spawn in state.spawns:
            if spawn.spawn_type == 0 and spawn.name != state.name:
                dist = state.pos.dist_to(spawn.pos)
                if dist <= radius:
                    count += 1
        return count

    def session_summary(self) -> str:
        from brain.session_summary import format_session_summary

        return format_session_summary(self)
