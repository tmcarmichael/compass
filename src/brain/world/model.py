"""World Model: interprets raw GameState into rich, relational, predictive data.

Transforms the flat spawn list into MobProfiles with velocity, isolation scores,
fight duration estimates, threat levels, and camp distance. Updated every brain
tick. Brain rules and routines read from the world model instead of doing their
own inline spawn scanning.

Usage:
    world = WorldModel(ctx)
    world.update(state)  # every tick
    best = world.best_target  # scored and ranked
    threats = world.threats   # sorted by urgency
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from brain.scoring.pareto import (
    AxisPriorities,
    compute_axes,
    compute_priorities,
    log_pareto_selection,
    pareto_frontier,
    select_from_frontier,
)
from brain.scoring.target import (
    MobProfile,
    ScoringWeights,
    estimate_fight_duration,
    estimate_mana_cost,
    load_scoring_weights,
    log_top_targets,
    score_target,
)
from brain.world.patrol import (
    PatrolMixin,
)
from brain.world.patrol import (
    patrol_safe_window as _patrol_safe_window,
)
from brain.world.patrol import (
    patrolling_threats as _patrolling_threats,
)
from core.features import flags
from core.types import Con, Disposition, Point
from core.types import normalize_entity_name as normalize_mob_name
from nav.geometry import heading_to
from perception.combat_eval import (
    FIGHTABLE_CONS,
    con_color,
    get_disposition,
    is_pet,
)
from perception.state import GameState, SpawnData

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.learning.spatial import SpatialMemory

log = logging.getLogger(__name__)

# Re-export for backward compatibility (callers import from world_model)
__all__ = [
    "MobProfile",
    "ScoringWeights",
    "load_scoring_weights",
    "WorldModel",
]


# -- Npc Tracker: per-spawn temporal history ----------------------------------


@dataclass(slots=True)
class _MobHistory(PatrolMixin):
    """Sliding window of recent positions and HP for a single spawn."""

    spawn_id: int
    name: str
    positions: deque  # deque[(time, x, y), ...]  max ~30 entries (3s at 10Hz)
    hp_samples: deque  # deque[(time, hp_pct), ...]  max ~30 entries (3s at 10Hz)
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_kill_time: float = 0.0  # when we last defeated this base name here

    # Patrol detection: sparse 60s trace (1 sample/2s max)
    patrol_trace: deque[tuple[float, float, float]] | None = None
    patrol_period: float = 0.0  # detected period in seconds (0 = not patrolling)
    _patrol_checked: float = 0.0  # last time we ran patrol detection

    def add(self, t: float, pos: Point, hp_pct: float = -1.0) -> None:
        self.positions.append((t, pos.x, pos.y, pos.z))
        if hp_pct >= 0:
            self.hp_samples.append((t, hp_pct))
        self.last_seen = t
        if not self.first_seen:
            self.first_seen = t
        # Keep last 3 seconds of history (for velocity)
        cutoff = t - 3.0
        while self.positions and self.positions[0][0] < cutoff:
            self.positions.popleft()
        while self.hp_samples and self.hp_samples[0][0] < cutoff:
            self.hp_samples.popleft()

        # Delegate patrol trace update to PatrolMixin
        self._update_patrol_trace(t, pos)

    def velocity(self, spawn: SpawnData | None = None) -> tuple[float, float, float]:
        """Get velocity (units/sec). Prefers direct memory read when available.

        Args:
            spawn: SpawnData with velocity_x/velocity_y/velocity_z from entity struct.
                   Falls back to position-delta computation if None or zero.
        """
        # Prefer instant memory-read velocity (no lag)
        if spawn is not None:
            vx = getattr(spawn, "velocity_x", 0.0)
            vy = getattr(spawn, "velocity_y", 0.0)
            vz = getattr(spawn, "velocity_z", 0.0)
            if vx != 0.0 or vy != 0.0 or vz != 0.0:
                return (vx, vy, vz)
        # Fallback: compute from position history (~0.3s lag)
        if len(self.positions) < 2:
            return (0.0, 0.0, 0.0)
        t0, x0, y0, z0 = self.positions[0]
        t1, x1, y1, z1 = self.positions[-1]
        dt = t1 - t0
        if dt < 0.3:
            return (0.0, 0.0, 0.0)
        return ((x1 - x0) / dt, (y1 - y0) / dt, (z1 - z0) / dt)

    def speed(self) -> float:
        vx, vy, _vz = self.velocity()
        return math.sqrt(vx * vx + vy * vy)

    def predicted_pos(self, seconds: float) -> Point:
        """Predict position N seconds from now."""
        if not self.positions:
            return Point(0.0, 0.0, 0.0)
        _, x, y, z = self.positions[-1]
        vx, vy, vz = self.velocity()
        return Point(x + vx * seconds, y + vy * seconds, z + vz * seconds)

    def hp_rate(self) -> float:
        """HP% lost per second. Positive = taking damage (HP decreasing).

        Uses oldest and newest HP samples in the 3s window.
        Returns 0.0 if insufficient data (<0.5s span or <2 samples).
        """
        if len(self.hp_samples) < 2:
            return 0.0
        t0, hp0 = self.hp_samples[0]
        t1, hp1 = self.hp_samples[-1]
        dt = t1 - t0
        if dt < 0.5:
            return 0.0
        # hp0 > hp1 means npc is losing HP, return positive rate
        return float((hp0 - hp1) / dt * 100.0)  # convert fraction to %/sec


# -- World Model --------------------------------------------------------------


class WorldModel:
    """Interprets GameState into actionable intelligence every tick.

    Call update(state) each brain tick. Then read properties for
    rich npc data, threats, and target recommendations.
    """

    def __init__(self, ctx: AgentContext | None = None, weights: ScoringWeights | None = None) -> None:
        self._ctx = ctx
        self._weights = weights or ScoringWeights()
        self._trackers: dict[int, _MobHistory] = {}  # spawn_id -> history
        self._profiles: list[MobProfile] = []
        self._threats: list[MobProfile] = []
        self._targets: list[MobProfile] = []
        self._last_update = 0.0

        # Fight duration model: base_name -> list of recent durations
        self._fight_durations: dict[str, list[float]] = {}

        # Player / pet tracking (rebuilt every tick)
        self._players: list[tuple[SpawnData, float]] = []  # (spawn, dist)
        self._our_pet: SpawnData | None = None
        self._nearby_pets: list[tuple[SpawnData, float]] = []
        self._last_target_log: float = 0.0

        # Pet HP history for damage rate / TTD estimation
        self._pet_hp_history: deque[tuple[float, float]] = deque()  # (timestamp, hp_pct)

        # Player position (updated each tick for heading calculations)
        self._last_player_pos: Point = Point(0.0, 0.0, 0.0)

        # Profiling: last update() wall time in ms
        self.update_ms: float = 0.0

        # Score breakdown for best target (gradient weight learning)
        self._last_target_breakdown: dict[str, float] = {}

    def _update_spawn_tracking(self, state: GameState, now: float) -> None:
        """Classify spawns into players, pets, and identify our pet."""
        ctx = self._ctx
        _pet_id = ctx.pet.spawn_id if ctx else 0

        self._players = []
        self._our_pet = None
        self._nearby_pets = []

        for spawn in state.spawns:
            if spawn.spawn_type == 0 and spawn.name != state.name:
                d = state.pos.dist_to(spawn.pos)
                self._players.append((spawn, d))
            if spawn.is_npc and spawn.hp_current > 0 and is_pet(spawn):
                d = state.pos.dist_to(spawn.pos)
                self._nearby_pets.append((spawn, d))
                if _pet_id and spawn.spawn_id == _pet_id:
                    self._our_pet = spawn

    def _update_pet_state(self, now: float) -> None:
        """Record pet HP for damage rate tracking."""
        if self._our_pet and self._our_pet.hp_max > 0:
            pet_pct = self._our_pet.hp_current / self._our_pet.hp_max
            self._pet_hp_history.append((now, pet_pct))
            # Keep last 5 seconds of history
            cutoff = now - 5.0
            while self._pet_hp_history and self._pet_hp_history[0][0] <= cutoff:
                self._pet_hp_history.popleft()
        elif not self._our_pet:
            # Pet not found (dead or despawned) -- clear history
            self._pet_hp_history.clear()

    @staticmethod
    def _count_nearby_npcs(
        spawn: SpawnData,
        mob_base: str,
        npc_grid: dict[tuple[int, int], list[tuple]],
        grid_cell: float,
        social_groups: dict,
    ) -> tuple[int, int]:
        """Count nearby and social NPCs via the spatial grid.

        Returns (nearby_count, social_count).
        """
        nearby = 0
        social_npcs = 0
        cx = int(spawn.x // grid_cell)
        cy = int(spawn.y // grid_cell)
        social_group = social_groups.get(mob_base)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other, other_base in npc_grid.get((cx + dx, cy + dy), ()):
                    if other.spawn_id == spawn.spawn_id:
                        continue
                    if spawn.pos.dist_to(other.pos) < 40:
                        nearby += 1
                        if social_group and other_base in social_group:
                            social_npcs += 1
        return nearby, social_npcs

    @staticmethod
    def _assess_threat(con: Con, disp: Disposition, dist: float) -> tuple[bool, float]:
        """Compute threat flag and threat level for a single NPC.

        Returns (is_threat, threat_level).
        """
        is_threat = False
        if disp in (Disposition.SCOWLING, Disposition.READY_TO_ATTACK, Disposition.THREATENING):
            is_threat = con in (Con.YELLOW, Con.RED, Con.WHITE)
        elif disp == Disposition.UNKNOWN:
            is_threat = con in (Con.YELLOW, Con.RED)

        threat = 0.0
        if con == Con.RED:
            threat = 1.0
        elif con == Con.YELLOW:
            threat = 0.6
        elif is_threat:
            threat = 0.3
        if threat > 0 and dist < 300:
            threat *= max(0.1, (300 - dist) / 300)

        return is_threat, threat

    def _build_npc_profile(
        self,
        spawn: SpawnData,
        now: float,
        player_pos: Point,
        player_level: int,
        camp_x: float,
        camp_y: float,
        zone_disps: Any,
        social_groups: dict,
        npc_grid: dict[tuple[int, int], list[tuple]],
        grid_cell: float,
        ctx: AgentContext | None,
    ) -> MobProfile:
        """Build a MobProfile for a single living NPC spawn."""
        sid = spawn.spawn_id

        # Update tracker
        if sid not in self._trackers:
            self._trackers[sid] = _MobHistory(
                spawn_id=sid, name=spawn.name, positions=deque(), hp_samples=deque(), first_seen=now
            )
        tracker = self._trackers[sid]
        hp_pct = spawn.hp_current / spawn.hp_max if spawn.hp_max > 0 else 1.0
        tracker.add(now, spawn.pos, hp_pct=hp_pct)

        # Basic calculations
        dist = player_pos.dist_to(spawn.pos)
        camp_dist = (
            ctx.camp.effective_camp_distance(spawn.pos)
            if ctx
            else Point(camp_x, camp_y, 0.0).dist_to(spawn.pos)
        )
        con = con_color(player_level, spawn.level) if player_level > 0 else Con.WHITE
        disp = get_disposition(spawn.name, zone_disps)
        mob_base = normalize_mob_name(spawn.name)

        # Velocity + prediction (prefer memory-read velocity)
        vel = tracker.velocity(spawn)
        spd = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
        pred_5s = tracker.predicted_pos(5.0)

        # Isolation: count nearby NPCs via spatial grid (O(1) per npc)
        nearby, social_npcs = self._count_nearby_npcs(spawn, mob_base, npc_grid, grid_cell, social_groups)
        isolation = 1.0 / (1.0 + nearby)  # 1.0 = alone, 0.5 = 1 neighbor, etc.

        # Fight duration estimate (Thompson-sampled for exploration)
        fight_est = estimate_fight_duration(
            mob_base,
            con,
            spawn.level,
            ctx=ctx,
            fight_durations=self._fight_durations,
            sample=True,
        )
        mana_est = estimate_mana_cost(
            con,
            fight_est,
            mob_base=mob_base,
            ctx=ctx,
            sample=True,
        )

        # Threat assessment
        is_threat, threat = self._assess_threat(con, disp, dist)

        # Add probability (Thompson-sampled from Beta posterior)
        add_prob = 0.0
        fh = ctx.fight_history if ctx else None
        if fh:
            add_prob = fh.sample_add_probability(mob_base)

        return MobProfile(
            spawn=spawn,
            con=con,
            disposition=disp,
            distance=dist,
            camp_distance=camp_dist,
            isolation_score=isolation,
            nearby_npc_count=nearby,
            social_npc_count=social_npcs,
            is_moving=spd > 0.5,
            speed=spd,
            velocity=vel,
            predicted_pos_5s=pred_5s,
            fight_duration_est=fight_est,
            mana_cost_est=mana_est,
            threat_level=threat,
            is_threat=is_threat,
            is_patrolling=tracker.patrol_period > 0,
            patrol_period=tracker.patrol_period,
            extra_npc_probability=add_prob,
        )

    def update(self, state: GameState) -> None:
        """Rebuild all profiles from current state. Called every tick."""
        _t0 = time.perf_counter()
        now = time.time()
        self._last_update = now

        ctx = self._ctx
        player_pos = state.pos
        self._last_player_pos = player_pos
        player_level = state.level
        camp_x = ctx.camp.camp_pos.x if ctx else 0.0
        camp_y = ctx.camp.camp_pos.y if ctx else 0.0
        zone_disps = ctx.zone.zone_dispositions if ctx else None
        social_groups = ctx.zone.social_mob_group if ctx else {}

        # Track players and pets (lightweight, rebuilt every tick)
        self._update_spawn_tracking(state, now)

        # Record pet HP for damage rate tracking
        self._update_pet_state(now)

        # Track all living NPCs
        seen_ids = set()

        # Pre-filter living NPCs for spatial grid (O(n) instead of O(n^2))
        _GRID_CELL = 40.0
        living_npcs = []
        for spawn in state.spawns:
            if not spawn.is_npc or spawn.hp_current <= 0:
                continue
            if is_pet(spawn):
                continue
            living_npcs.append(spawn)

        # Build spatial grid: cell -> list of (spawn, base_name)
        npc_grid: dict[tuple[int, int], list[tuple]] = {}
        for spawn in living_npcs:
            cx = int(spawn.x // _GRID_CELL)
            cy = int(spawn.y // _GRID_CELL)
            base = normalize_mob_name(spawn.name)
            cell = (cx, cy)
            if cell not in npc_grid:
                npc_grid[cell] = []
            npc_grid[cell].append((spawn, base))

        profiles = []

        for spawn in living_npcs:
            seen_ids.add(spawn.spawn_id)
            profile = self._build_npc_profile(
                spawn,
                now,
                player_pos,
                player_level,
                camp_x,
                camp_y,
                zone_disps,
                social_groups,
                npc_grid,
                _GRID_CELL,
                ctx,
            )
            profiles.append(profile)

        # Prune stale trackers (npc despawned or died)
        stale = [sid for sid in self._trackers if sid not in seen_ids]
        for sid in stale:
            del self._trackers[sid]

        # Sort and categorize
        self._profiles = profiles
        self._threats = sorted(
            [p for p in profiles if p.is_threat and p.distance < 300], key=lambda p: -p.threat_level
        )

        # Score and rank targets by utility (delegated to mob_scoring)
        target_candidates = [
            p
            for p in profiles
            if p.con in FIGHTABLE_CONS and p.distance < 250 and p.spawn.hp_current >= p.spawn.hp_max
        ]
        for p in target_candidates:
            p.score = score_target(
                p,
                self._weights,
                self._profiles,
                self._players,
                ctx=ctx,
                player_x=self._last_player_pos.x,
                player_y=self._last_player_pos.y,
                fight_durations=self._fight_durations,
            )

        # Pareto multi-objective selection (when enabled)
        if flags.pareto_scoring and target_candidates and state:
            self._targets = self._pareto_rank(target_candidates, state, ctx)
        else:
            self._targets = sorted(target_candidates, key=lambda p: -p.score)

        # Capture breakdown for best target (gradient weight learning)
        if self._targets and self._targets[0].score > 0:
            bd: dict[str, float] = {}
            score_target(
                self._targets[0],
                self._weights,
                self._profiles,
                self._players,
                ctx=ctx,
                player_x=self._last_player_pos.x,
                player_y=self._last_player_pos.y,
                fight_durations=self._fight_durations,
                breakdown=bd,
            )
            self._last_target_breakdown = bd
        else:
            self._last_target_breakdown = {}

        # Periodic target diagnostics
        self._last_target_log = log_top_targets(self._targets, self._last_target_log)

        self.update_ms = (time.perf_counter() - _t0) * 1000

    def _pareto_rank(
        self, candidates: list[MobProfile], state: GameState, ctx: AgentContext | None
    ) -> list[MobProfile]:
        """Rank targets using multi-objective Pareto filtering.

        Computes 4-axis scores per target, identifies the Pareto frontier,
        and selects the best using state-aware priority weights. Returns
        all candidates sorted with the Pareto-selected best first.
        """
        # Compute axis scores for each candidate
        axis_list = []
        for p in candidates:
            if p.score <= 0:
                continue  # hard-rejected by score_target
            mob_base = normalize_mob_name(p.spawn.name)
            axes = compute_axes(
                p, self._weights, state, ctx, fight_durations=self._fight_durations, mob_base=mob_base
            )
            axis_list.append(axes)

        if not axis_list:
            return sorted(candidates, key=lambda p: -p.score)

        # Find Pareto frontier
        frontier = pareto_frontier(axis_list)
        if not frontier:
            return sorted(candidates, key=lambda p: -p.score)

        # Compute state-aware priorities
        phase = "grinding"
        if ctx and hasattr(ctx, "diag") and ctx.diag:
            pd = getattr(ctx.diag, "phase_detector", None)
            if pd is not None:
                phase = pd.current_phase
        base_priorities = AxisPriorities(
            efficiency=self._weights.pareto_eff_weight,
            safety=self._weights.pareto_saf_weight,
            resource=self._weights.pareto_res_weight,
            accessibility=self._weights.pareto_acc_weight,
        )
        priorities = compute_priorities(state, ctx, phase, base_priorities)

        # Select best from frontier
        best = select_from_frontier(frontier, priorities)
        if best is not None:
            log_pareto_selection(frontier, best, priorities, len(candidates))
            # Assign Pareto-weighted score for consistent ranking
            total = priorities.efficiency + priorities.safety + priorities.resource + priorities.accessibility
            if total > 0:
                best.profile.score = (
                    (
                        best.efficiency * priorities.efficiency
                        + best.safety * priorities.safety
                        + best.resource * priorities.resource
                        + best.accessibility * priorities.accessibility
                    )
                    / total
                    * 200
                )  # scale to comparable range with weighted sum

        # Return all candidates, Pareto-selected first, then by original score
        selected_id = best.profile.spawn.spawn_id if best else -1
        result = sorted(candidates, key=lambda p: (0 if p.spawn.spawn_id == selected_id else 1, -p.score))
        return result

    def record_fight(self, mob_base_name: str, duration: float) -> None:
        """Record a fight duration for the model to learn from."""
        key = normalize_mob_name(mob_base_name)
        if key not in self._fight_durations:
            self._fight_durations[key] = []
        self._fight_durations[key].append(duration)
        # Keep last 20 samples per npc type
        if len(self._fight_durations[key]) > 20:
            self._fight_durations[key] = self._fight_durations[key][-20:]

    def load_from_spatial(self, spatial_memory: SpatialMemory) -> None:
        """Seed fight durations from SpatialMemory's persisted defeat history.

        SpatialMemory already persists fight_s in each defeat entry.
        This reconstructs _fight_durations on startup so the agent
        starts each session already knowing how long fights take.
        """
        count = 0
        for defeat in getattr(spatial_memory, "_kills", []):
            fight_s = defeat.get("fight_s", 0)
            if fight_s <= 0:
                continue
            name = defeat.get("name", "")
            if not name:
                continue
            key = normalize_mob_name(name)
            if key not in self._fight_durations:
                self._fight_durations[key] = []
            self._fight_durations[key].append(fight_s)
            count += 1
        # Cap each npc type at 20 samples
        for key in self._fight_durations:
            if len(self._fight_durations[key]) > 20:
                self._fight_durations[key] = self._fight_durations[key][-20:]
        if count > 0:
            log.info(
                "[TARGET] WorldModel: loaded %d fight durations for %d npc types from spatial memory",
                count,
                len(self._fight_durations),
            )

    # -- Properties ----------------------------------------------------

    @property
    def profiles(self) -> list[MobProfile]:
        """All living NPC profiles, updated this tick."""
        return self._profiles

    @property
    def threats(self) -> list[MobProfile]:
        """Threatening npcs sorted by urgency (highest threat first)."""
        return self._threats

    @property
    def targets(self) -> list[MobProfile]:
        """Valid combat targets sorted by score (best first)."""
        return self._targets

    @property
    def best_target(self) -> MobProfile | None:
        """Highest-scored target, or None."""
        return self._targets[0] if self._targets else None

    def mob_targeting_player(self, player_name: str) -> bool:
        """True if the current combat target is attacking the player (not pet).

        Uses the NPC's target_name field from the spawn record.
        Returns False if target is attacking pet or no target engaged.
        """
        ctx = self._ctx
        if not ctx or not ctx.combat.engaged:
            return False
        target_id = ctx.combat.pull_target_id
        if not target_id:
            return False
        for p in self._profiles:
            if p.spawn.spawn_id == target_id:
                targeting_player: bool = p.spawn.target_name == player_name
                return targeting_player
        return False

    @property
    def mob_density(self) -> float:
        """NPCs within 200u of camp center."""
        return sum(1 for p in self._profiles if p.camp_distance < 200)

    @property
    def threat_count(self) -> int:
        return len(self._threats)

    def get_profile(self, spawn_id: int) -> MobProfile | None:
        """Get profile for a specific spawn by ID."""
        for p in self._profiles:
            if p.spawn.spawn_id == spawn_id:
                return p
        return None

    # -- Player queries ---------------------------------------------

    def nearest_player_dist(self) -> float:
        """Distance to nearest other player, or 9999.0 if none."""
        if not self._players:
            return 9999.0
        return min(d for _, d in self._players)

    def nearby_player_count(self, radius: float = 200.0) -> int:
        """Count of other players within radius."""
        return sum(1 for _, d in self._players if d <= radius)

    # -- Pet queries ------------------------------------------------

    @property
    def pet_spawn(self) -> SpawnData | None:
        """Our tracked pet's SpawnData this tick, or None."""
        return self._our_pet

    @property
    def pet_hp_pct(self) -> float:
        """Our pet's HP as 0.0-1.0, or -1.0 if not found in spawns."""
        p = self._our_pet
        if p and p.hp_max > 0:
            pct: float = p.hp_current / p.hp_max
            return pct
        return -1.0

    def has_pet_nearby(self, radius: float = 100.0) -> bool:
        """True if any pet NPC is within radius of player."""
        return any(d < radius for _, d in self._nearby_pets)

    def pet_damage_rate(self) -> float:
        """Pet HP% lost per second. Positive = taking damage. 0 = stable.

        Uses linear regression (OLS slope) over a 5-second sliding window.
        More robust than oldest/newest -- a single heal tick won't invert
        the estimate when the overall trend is still damage.
        Requires at least 3 samples spanning 0.5+ seconds.
        """
        n = len(self._pet_hp_history)
        if n < 3:
            return 0.0
        t0 = self._pet_hp_history[0][0]
        t_last = self._pet_hp_history[-1][0]
        if t_last - t0 < 0.5:
            return 0.0
        # OLS slope: sum((t-t_mean)(hp-hp_mean)) / sum((t-t_mean)^2)
        sum_t = sum_hp = 0.0
        for t, hp in self._pet_hp_history:
            sum_t += t - t0
            sum_hp += hp
        t_mean = sum_t / n
        hp_mean = sum_hp / n
        num = denom = 0.0
        for t, hp in self._pet_hp_history:
            dt = (t - t0) - t_mean
            num += dt * (hp - hp_mean)
            denom += dt * dt
        if denom < 1e-9:
            return 0.0
        slope = num / denom  # HP fraction per second (negative = losing HP)
        return max(0.0, -slope * 100)  # convert to %/sec, positive = damage

    def pet_time_to_death(self) -> float | None:
        """Seconds until pet HP reaches 0 at current damage rate.

        Returns None if pet is not taking damage or insufficient data.
        Returns 0.0 if pet is already at 0 HP.
        """
        rate = self.pet_damage_rate()
        if rate <= 0 or not self._pet_hp_history:
            return None
        current_pct = self._pet_hp_history[-1][1] * 100
        if current_pct <= 0:
            return 0.0
        return current_pct / rate

    # -- NPC proximity queries --------------------------------------

    def any_npc_within(self, radius: float) -> bool:
        """True if any living non-pet NPC is within radius of player."""
        return any(p.distance < radius for p in self._profiles)

    def any_hostile_npc_within(self, radius: float) -> bool:
        """True if any hostile NPC (KOS or unknown YELLOW/RED) is within radius.

        Filters out INDIFFERENT/passive npcs that will never threat, so patrol
        npcs passing through camp don't block rest.
        """
        from perception.combat_eval import AGGRESSIVE_DISPOSITIONS

        for p in self._profiles:
            if p.distance >= radius:
                continue
            if p.disposition in AGGRESSIVE_DISPOSITIONS:
                return True
            if p.disposition == Disposition.UNKNOWN and p.is_threat:
                return True
        return False

    def threats_within(self, radius: float) -> list[MobProfile]:
        """Threat profiles (YELLOW/RED/aggressive) within radius of player."""
        return [p for p in self._profiles if p.is_threat and p.distance < radius]

    def patrol_safe_window(
        self, target_pos: Point, fight_duration: float, threat_radius: float = 80.0
    ) -> float:
        """Seconds until a patrolling threat enters threat range of target.

        Returns float('inf') if no patrol will arrive during fight_duration.
        Returns 0.0 if a patrol is already within threat range.

        Used by acquire/pull to avoid pulling when a patrol is about to return.
        """
        result: float = _patrol_safe_window(
            self._profiles, self._trackers, target_pos, fight_duration, threat_radius
        )
        return result

    def patrolling_threats(self) -> list[MobProfile]:
        """All patrolling threat npcs currently tracked."""
        result: list[MobProfile] = _patrolling_threats(self._profiles)
        return result

    def damaged_npcs_near(self, pos: Point, radius: float, exclude_id: int = 0) -> list[MobProfile]:
        """Living NPCs with HP < max within radius of position."""
        result = []
        for p in self._profiles:
            if p.spawn.spawn_id == exclude_id:
                continue
            if p.spawn.hp_current >= p.spawn.hp_max:
                continue
            d = pos.dist_to(p.spawn.pos)
            if d < radius:
                result.append(p)
        return result

    # -- Heading analysis --------------------------------------------

    @staticmethod
    def is_approaching(spawn: SpawnData, player_pos: Point) -> bool:
        """True if *spawn* is facing toward the player AND moving.

        Uses the spawn's heading (0-512) and speed fields.
        A heading error < 60 units (~42 degrees) with speed > 0.5
        counts as "approaching".  Useful for early threat detection,
        target scoring, and acquire safety checks.
        """
        if spawn.speed <= 0.5:
            return False
        angle_to_player = heading_to(spawn.pos, player_pos)
        heading_error: float = abs(spawn.heading - angle_to_player) % 512
        if heading_error > 256:
            heading_error = 512 - heading_error
        approaching: bool = heading_error < 60
        return approaching

    @staticmethod
    def heading_error_to(spawn: SpawnData, target_pos: Point) -> float:
        """Heading error (0-256) between spawn's facing and the direction to a point.

        0 = facing directly at the point, 256 = facing directly away.
        """
        angle_to_target = heading_to(spawn.pos, target_pos)
        err: float = abs(spawn.heading - angle_to_target) % 512
        if err > 256:
            err = 512 - err
        return err

    # -- Target damage rate ---------------------------------------

    def target_damage_rate(self, spawn_id: int = 0) -> float:
        """HP% lost per second on a target npc. Positive = taking damage.

        If spawn_id is 0, uses the current target from context.
        Returns 0.0 if no target, target not tracked, or insufficient data.
        """
        if spawn_id == 0 and self._ctx:
            sid = self._ctx.combat.pull_target_id
            if not sid:
                return 0.0
            spawn_id = sid
        if spawn_id == 0:
            return 0.0
        tracker = self._trackers.get(spawn_id)
        if not tracker:
            return 0.0
        return tracker.hp_rate()
