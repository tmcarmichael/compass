"""Target scoring: consideration-based utility with 15-factor scoring function.

MobProfile (rich NPC interpretation), ScoringWeights (curve-parameterized
scoring constants), and the utility scoring logic for target selection.

Usage:
    from brain.scoring import MobProfile, ScoringWeights, score_target
    weights = load_scoring_weights("path/to/weights.json")
    score = score_target(profile, weights, profiles, players, ctx=ctx)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from brain.scoring.curves import bell, inverse_logistic, polynomial
from core.types import Con, Disposition, Point
from core.types import normalize_entity_name as normalize_mob_name
from nav.geometry import heading_to
from perception.state import SpawnData
from util.log_tiers import VERBOSE

if TYPE_CHECKING:
    from brain.context import AgentContext

log = logging.getLogger(__name__)

# Throttle per-npc drift logs to avoid spam (scored every tick per npc)
_drift_logged: dict[str, float] = {}  # mob_name -> last_log_time
_DRIFT_LOG_INTERVAL = 30.0

# Throttle player-proximity reject logs (same npc+player fires every tick)
_player_reject_logged: dict[str, float] = {}  # "npc:player" -> last_log_time
_PLAYER_REJECT_LOG_INTERVAL = 30.0

# Cap throttle dicts to prevent unbounded growth in long sessions
_MAX_THROTTLE_ENTRIES = 100


def _prune_throttle(d: dict[str, float], now: float, interval: float) -> None:
    """Remove stale entries from a throttle dict when it exceeds cap."""
    if len(d) <= _MAX_THROTTLE_ENTRIES:
        return
    stale = [k for k, t in d.items() if now - t > interval * 2]
    for k in stale:
        del d[k]
    # If still over cap after pruning stale, drop oldest half
    if len(d) > _MAX_THROTTLE_ENTRIES:
        by_age = sorted(d, key=lambda k: d.get(k, 0.0))
        for k in by_age[: len(d) // 2]:
            del d[k]


# -- Npc Profile: rich interpretation of a single NPC -------------------------


@dataclass(slots=True)
class MobProfile:
    """Rich, scored interpretation of a single NPC."""

    spawn: SpawnData
    con: Con
    disposition: Disposition
    distance: float
    camp_distance: float  # distance from camp center to npc

    # Spatial
    isolation_score: float  # 0.0 (clustered) to 1.0 (isolated)
    nearby_npc_count: int  # NPCs within 40u of this npc
    social_npc_count: int  # social group members nearby

    # Temporal
    is_moving: bool
    speed: float
    velocity: tuple[float, float, float]
    predicted_pos_5s: Point

    # Combat estimation
    fight_duration_est: float  # predicted seconds to defeat
    mana_cost_est: int  # predicted mana expenditure

    # Threat
    threat_level: float  # 0.0 (harmless) to 1.0 (lethal)
    is_threat: bool  # YELLOW/RED or aggressive disposition

    # Patrol detection
    is_patrolling: bool = False
    patrol_period: float = 0.0  # cycle time in seconds (0 = not patrolling)

    # Learned add probability (from encounter history)
    extra_npc_probability: float = 0.0  # 0.0-1.0, probability of getting extra_npcs

    # Scoring (filled by scorer, default 0)
    score: float = 0.0


# -- Scoring Weights ----------------------------------------------------------


@dataclass(slots=True)
class ScoringWeights:
    """Configurable weights for consideration-based target scoring.

    Each factor uses a response curve (bell, logistic, polynomial) instead of
    step functions. Curve parameters are tunable by the gradient learner,
    allowing the agent to learn optimal scoring shapes per zone.
    """

    # -- Difficulty preference (discrete, no curve needed) --
    con_white: int = 100
    con_blue: int = 75
    con_light_blue: int = 35
    con_yellow: int = 20

    # -- Resource target bonus --
    resource_bonus: int = 40

    # -- Distance (bell curve: sweet spot with smooth falloff) --
    dist_ideal: float = 60.0  # center of sweet spot (game units)
    dist_width: float = 100.0  # distance from center to near-zero score
    dist_peak: float = 30.0  # maximum score at ideal distance

    # -- Isolation (polynomial: diminishing returns) --
    isolation_peak: float = 50.0  # max score at full isolation (1.0)
    isolation_exp: float = 0.7  # <1 = diminishing returns, >1 = accelerating
    social_npc_penalty: int = 30  # per social add (linear, stacks)

    # -- Camp proximity (logistic falloff beyond roam radius) --
    camp_peak: float = 30.0  # max bonus at camp center
    camp_falloff_k: float = 0.04  # steepness of dropoff beyond roam radius

    # -- Movement --
    moving_penalty: int = 15

    # -- Caster NPC --
    caster_penalty: int = 25

    # -- Loot value --
    loot_value_scale: float = 5.0  # points per 100cp expected value
    loot_value_cap: int = 30

    # -- Heading safety (multipliers, not additive) --
    heading_facing_mult: float = 0.7  # NPC facing player
    heading_away_mult: float = 1.1  # NPC facing away

    # -- Spatial memory heat --
    heat_multiplier: float = 0.1
    heat_cap: float = 5.0

    # -- Learned encounter efficiency --
    fast_defeat_bonus: int = 10  # avg duration < 20s
    slow_defeat_penalty: int = 10  # avg duration > 40s

    # -- Pareto axis priority weights (gradient-learnable) --
    pareto_eff_weight: float = 0.35
    pareto_saf_weight: float = 0.25
    pareto_res_weight: float = 0.20
    pareto_acc_weight: float = 0.20

    # -- Hard reject thresholds --
    avoid_npc_proximity: float = 100.0  # min distance from named NPCs in avoid set
    player_proximity: float = 30.0
    social_npc_hard_limit: int = 3


def load_scoring_weights(path: str | None = None) -> ScoringWeights:
    """Load scoring weights from a JSON file, or return defaults."""
    if not path:
        return ScoringWeights()
    try:
        with open(path) as f:
            data = json.load(f)
        return ScoringWeights(**{k: v for k, v in data.items() if hasattr(ScoringWeights, k)})
    except (OSError, json.JSONDecodeError, TypeError) as e:
        log.warning("[TARGET] Failed to load scoring weights from %s: %s", path, e)
        return ScoringWeights()


# -- Fight estimation --------------------------------------------------------


def estimate_fight_duration(
    mob_base: str,
    con: Con,
    mob_level: int,
    ctx: AgentContext | None = None,
    fight_durations: dict | None = None,
    sample: bool = False,
) -> float:
    """Estimate fight duration from learned data, session data, or heuristic.

    Args:
        mob_base: base npc name (stripped of trailing digits)
        con: con color for the npc
        mob_level: npc's level
        ctx: AgentContext (optional, for FightHistory access)
        fight_durations: dict of mob_base -> list[float] durations (session data)
        sample: if True, Thompson-sample from posterior instead of point estimate
    """
    key = mob_base.lower()

    # Priority 1: FightHistory learned data (persistent across sessions)
    fh = ctx.fight_history if ctx else None
    if fh:
        if sample and fh.get_stats(key) is not None:
            return fh.sample_duration(key)
        learned = fh.learned_duration(key)
        if learned is not None:
            return float(learned)

    # Priority 2: Session-local fight durations
    if fight_durations and key in fight_durations and fight_durations[key]:
        durations = fight_durations[key]
        return float(sum(durations) / len(durations))

    # Priority 3: Heuristic based on con and level
    match con:
        case Con.LIGHT_BLUE:
            return 15.0 + mob_level * 1.5
        case Con.BLUE:
            return 20.0 + mob_level * 2.0
        case Con.WHITE:
            return 25.0 + mob_level * 2.0
        case Con.YELLOW:
            return 35.0 + mob_level * 2.5
        case _:
            return 20.0


def estimate_mana_cost(
    con: Con,
    fight_duration: float,
    mob_base: str = "",
    ctx: AgentContext | None = None,
    sample: bool = False,
) -> int:
    """Estimate mana expenditure from learned data or heuristic.

    Args:
        con: con color for the npc
        fight_duration: estimated fight duration in seconds
        mob_base: base npc name (for FightHistory lookup)
        ctx: AgentContext (optional, for FightHistory access)
        sample: if True, Thompson-sample from posterior instead of point estimate
    """
    # Priority 1: FightHistory learned data
    if mob_base:
        fh = ctx.fight_history if ctx else None
        if fh:
            if sample and fh.get_stats(mob_base) is not None:
                return fh.sample_mana(mob_base)
            learned = fh.learned_mana(mob_base)
            if learned is not None:
                return int(learned)

    # Priority 2: Heuristic based on con
    match con:
        case Con.LIGHT_BLUE:
            return 0  # pet only
        case Con.BLUE:
            return 10  # DC only
        case Con.WHITE:
            return 20 if fight_duration > 25 else 10
        case Con.YELLOW:
            return 40
        case _:
            return 10


# -- Target Scorer -----------------------------------------------------------


def _hard_reject(
    p: MobProfile,
    w: ScoringWeights,
    profiles: list[MobProfile],
    players: list[tuple[SpawnData, float]],
    ctx: AgentContext | None,
    avoid_names: frozenset[str] | set[str],
) -> bool:
    """Return True if target should be rejected (score 0).

    Checks guard proximity, player proximity, social extra_npcs, faction.
    """
    # Named NPC proximity exclusion: reject if too close to an avoided NPC
    for other in profiles:
        if other is p:
            continue
        if any(avoid in other.spawn.name for avoid in avoid_names):
            gd = p.spawn.pos.dist_to(other.spawn.pos)
            if gd < w.avoid_npc_proximity:
                return True

    # Player proximity
    for player_spawn, _ in players:
        pd = p.spawn.pos.dist_to(player_spawn.pos)
        if pd < w.player_proximity:
            _key = f"{p.spawn.name}:{player_spawn.name}"
            _now = time.monotonic()
            if _now - _player_reject_logged.get(_key, 0.0) >= _PLAYER_REJECT_LOG_INTERVAL:
                log.log(
                    VERBOSE, "[TARGET] REJECT '%s': player '%s' at %.0fu", p.spawn.name, player_spawn.name, pd
                )
                _player_reject_logged[_key] = _now
                _prune_throttle(_player_reject_logged, _now, _PLAYER_REJECT_LOG_INTERVAL)
            return True

    # Social extra_npcs hard limit
    if p.social_npc_count >= w.social_npc_hard_limit:
        log.debug(
            "[TARGET] REJECT '%s': %d social extra_npcs >= %d limit",
            p.spawn.name,
            p.social_npc_count,
            w.social_npc_hard_limit,
        )
        return True

    # Con color not in target_cons (avoids walk-toward-then-reject loops)
    if ctx and ctx.zone.target_cons:
        if p.con not in ctx.zone.target_cons:
            return True

    # Zone-avoided npcs
    mob_base = normalize_mob_name(p.spawn.name)
    from perception.combat_eval import get_zone_avoid_mobs

    if mob_base in get_zone_avoid_mobs():
        return True

    return False


def _score_con(p: MobProfile, w: ScoringWeights) -> float:
    """Score based on con color (XP value + fight safety)."""
    con_scores = {
        Con.WHITE: w.con_white,
        Con.BLUE: w.con_blue,
        Con.LIGHT_BLUE: w.con_light_blue,
        Con.YELLOW: w.con_yellow,
    }
    return con_scores.get(p.con, 0)


def _score_camp(
    p: MobProfile, w: ScoringWeights, ctx: AgentContext, player_x: float, player_y: float
) -> float:
    """Score based on camp proximity using logistic falloff."""
    s = 0.0
    roam = ctx.camp.roam_radius
    # Use bounds-aware distance so corridor-end npcs get full bonus
    effective_dist = ctx.camp.effective_camp_distance(p.spawn.pos)
    # Smooth S-curve: full bonus inside roam, drops off beyond
    proximity: float = inverse_logistic(effective_dist, roam, w.camp_falloff_k)
    s += w.camp_peak * proximity

    # Camp drift penalty: penalize npcs that would pull player further
    player_camp_dist: float = ctx.camp.effective_camp_distance(Point(player_x, player_y, 0.0))
    if player_camp_dist > roam * 0.5 and p.camp_distance > player_camp_dist:
        drift_penalty = (p.camp_distance - player_camp_dist) * 0.1
        s -= drift_penalty
        if drift_penalty > 20:
            _now = time.time()
            _last = _drift_logged.get(p.spawn.name, 0.0)
            if _now - _last >= _DRIFT_LOG_INTERVAL:
                _drift_logged[p.spawn.name] = _now
                _prune_throttle(_drift_logged, _now, _DRIFT_LOG_INTERVAL)
                log.log(
                    VERBOSE,
                    "[TARGET] Drift penalty: '%s' -%.0f (npc %.0fu from camp, player %.0fu)",
                    p.spawn.name,
                    drift_penalty,
                    p.camp_distance,
                    player_camp_dist,
                )
    return s


def _learned_add_penalty(mob_base: str, w: ScoringWeights, ctx: AgentContext | None) -> float:
    """Penalize npcs with known social add history from FightHistory."""
    if not ctx or not ctx.fight_history:
        return 0.0
    learned_adds = ctx.fight_history.learned_adds(mob_base)
    if learned_adds is not None and learned_adds > 0.5:
        penalty: float = learned_adds * w.social_npc_penalty * 0.5
        return penalty
    return 0.0


def _score_distance(p: MobProfile, w: ScoringWeights) -> float:
    """Score based on distance using bell curve (smooth sweet spot)."""
    result: float = w.dist_peak * bell(p.distance, w.dist_ideal, w.dist_width)
    return result


def _score_factors(
    p: MobProfile,
    w: ScoringWeights,
    ctx: AgentContext | None,
    player_x: float,
    player_y: float,
    fight_durations: dict | None,
    bd: dict[str, float] | None,
) -> float:
    """Compute soft score from all weighted factors.

    When bd (breakdown) is provided, each factor's contribution is recorded
    for gradient weight learning. Zero cost when bd is None.
    """
    s: float = 0.0

    # Con preference
    con_pts = _score_con(p, w)
    s += con_pts

    # Resource target bonus
    mob_base = normalize_mob_name(p.spawn.name)
    resource_pts = 0.0
    if ctx and ctx.loot.resource_targets:
        resource_names = {n.lower() for n in ctx.loot.resource_targets}
        if mob_base in resource_names:
            resource_pts = float(w.resource_bonus)
    s += resource_pts

    # Distance
    dist_pts = _score_distance(p, w)
    s += dist_pts

    # Isolation: diminishing returns curve (first 50% matters most)
    iso_pts = w.isolation_peak * polynomial(p.isolation_score, 0.0, 1.0, w.isolation_exp)
    s += iso_pts

    # Social add penalty (stacks -- each add is pull risk)
    social_pts: float = -(p.social_npc_count * w.social_npc_penalty)
    social_pts -= _learned_add_penalty(mob_base, w, ctx)
    # Learned add probability penalty (from encounter history)
    if p.extra_npc_probability > 0.1:
        social_pts -= p.extra_npc_probability * w.social_npc_penalty * 1.5
    # Danger memory penalty (deaths/flees from this entity type)
    if ctx and ctx.danger_memory:
        danger_pen = ctx.danger_memory.danger_penalty(mob_base)
        if danger_pen > 0:
            social_pts -= danger_pen * w.social_npc_penalty * 2.0
    s += social_pts

    # Camp proximity + drift penalty
    camp_pts = _score_camp(p, w, ctx, player_x, player_y) if ctx else 0.0
    s += camp_pts

    # Moving targets are harder to pull
    move_pts = -float(w.moving_penalty) if p.is_moving else 0.0
    s += move_pts

    # Caster npcs are dangerous (nuke, heal, gate)
    caster_pts = 0.0
    if ctx and ctx.loot.caster_mob_names:
        if mob_base in ctx.loot.caster_mob_names or p.spawn.name in ctx.loot.caster_mob_names:
            caster_pts = -float(w.caster_penalty)
    s += caster_pts

    # Loot value bonus
    loot_pts = _score_loot(mob_base, w, ctx)
    s += loot_pts

    # Heading safety (multiplier)
    heading_mult = _heading_multiplier(p, w, player_x, player_y)
    s *= heading_mult

    # Spatial memory bonus (multiplier)
    heat_mult = _heat_multiplier(p, w, ctx)
    s *= heat_mult

    # Learned efficiency: prefer npc types with known fast defeats
    efficiency_pts = _score_efficiency(mob_base, w, fight_durations)
    s += efficiency_pts

    # Populate breakdown for gradient tuner if requested
    if bd is not None:
        bd["con_pref"] = con_pts
        bd["resource"] = resource_pts
        bd["distance"] = dist_pts
        bd["isolation"] = iso_pts
        bd["social_add"] = social_pts
        bd["camp_proximity"] = camp_pts
        bd["movement"] = move_pts
        bd["caster"] = caster_pts
        bd["loot_value"] = loot_pts
        bd["heading"] = heading_mult - 1.0
        bd["spatial_heat"] = heat_mult - 1.0
        bd["learned_efficiency"] = efficiency_pts

    return s


def _score_loot(mob_base: str, w: ScoringWeights, ctx: AgentContext | None) -> float:
    """Loot value bonus from npc knowledge."""
    if not ctx or not ctx.loot.mob_loot_values:
        return 0.0
    loot_ev: float = ctx.loot.mob_loot_values.get(mob_base, 0)
    if loot_ev > 0:
        result: float = min(float(w.loot_value_cap), loot_ev / 100.0 * w.loot_value_scale)
        return result
    return 0.0


def _heading_multiplier(p: MobProfile, w: ScoringWeights, player_x: float, player_y: float) -> float:
    """Heading safety multiplier: facing us = risky, facing away = safer."""
    if p.distance >= 80 or p.distance <= 0:
        return 1.0
    angle_to_player = heading_to(p.spawn.pos, Point(player_x, player_y, 0.0))
    heading_error = abs(p.spawn.heading - angle_to_player) % 512
    if heading_error > 256:
        heading_error = 512 - heading_error
    if heading_error < 60:
        return w.heading_facing_mult
    if heading_error > 180:
        return w.heading_away_mult
    return 1.0


def _heat_multiplier(p: MobProfile, w: ScoringWeights, ctx: AgentContext | None) -> float:
    """Spatial memory heat multiplier."""
    if not ctx or not ctx.spatial_memory:
        return 1.0
    heat: float = ctx.spatial_memory.heat_at(p.spawn.pos)
    result: float = 1.0 + w.heat_multiplier * min(heat, w.heat_cap)
    return result


def _score_efficiency(mob_base: str, w: ScoringWeights, fight_durations: dict | None) -> float:
    """Learned encounter efficiency bonus/penalty."""
    if not fight_durations or mob_base not in fight_durations:
        return 0.0
    durations = fight_durations[mob_base]
    avg_dur = sum(durations) / len(durations)
    if avg_dur < 20:
        return float(w.fast_defeat_bonus)
    if avg_dur > 40:
        return -float(w.slow_defeat_penalty)
    return 0.0


def score_target(
    p: MobProfile,
    weights: ScoringWeights,
    profiles: list[MobProfile],
    players: list[tuple[SpawnData, float]],
    ctx: AgentContext | None = None,
    player_x: float = 0.0,
    player_y: float = 0.0,
    fight_durations: dict | None = None,
    breakdown: dict[str, float] | None = None,
) -> float:
    """Utility score for target selection. Higher = better target.

    Hard constraints reject (return 0.0) before soft scoring.
    All numeric weights come from the weights parameter (ScoringWeights).

    Args:
        p: MobProfile to score
        weights: ScoringWeights with all tunable constants
        profiles: all MobProfile instances this tick (for guard proximity)
        players: list of (SpawnData, distance) for other players
        ctx: AgentContext (optional, for zone/loot config access)
        player_x: player X position (for heading calculations)
        player_y: player Y position (for heading calculations)
        fight_durations: dict of mob_base -> list[float] (session fight data)
        breakdown: if provided, populated with per-factor score contributions
                   for gradient weight learning (zero cost when None)
    """
    from perception.combat_eval import get_avoid_names

    if _hard_reject(p, weights, profiles, players, ctx, get_avoid_names()):
        return 0.0

    return _score_factors(p, weights, ctx, player_x, player_y, fight_durations, breakdown)


def log_top_targets(targets: list[MobProfile], last_log_time: float) -> float:
    """Log top 3 scored targets every 30s for diagnostics.

    Args:
        targets: sorted list of MobProfile (best first)
        last_log_time: timestamp of the last log emission

    Returns:
        Updated last_log_time (caller should store this).
    """
    now = time.time()
    if now - last_log_time < 30.0:
        return last_log_time
    if not targets:
        return now
    parts = []
    for p in targets[:3]:
        parts.append(
            f"'{p.spawn.name}' {p.con} dist={p.distance:.0f} iso={p.isolation_score:.2f} score={p.score:.0f}"
        )
    log.log(VERBOSE, "[TARGET] WorldModel targets: %s", " | ".join(parts))
    return now
