"""Multi-objective target selection with Pareto-optimal filtering.

Scores each target on 4 independent axes (efficiency, safety, resource cost,
accessibility). The Pareto frontier identifies targets not dominated on any
axis. From the frontier, the agent selects based on priority weights that
shift with session phase, resource levels, and learned preferences.

Usage:
    axes = compute_axes(profile, weights, state, ctx, fight_durations)
    frontier = pareto_frontier(all_axes)
    priorities = compute_priorities(state, ctx, phase)
    best = select_from_frontier(frontier, priorities)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from brain.scoring.curves import bell, inverse_logistic, polynomial
from brain.scoring.target import (
    MobProfile,
    ScoringWeights,
    _score_con,
    _score_efficiency,
    _score_loot,
)
from util.log_tiers import VERBOSE

if TYPE_CHECKING:
    from brain.context import AgentContext
    from perception.state import GameState

log = logging.getLogger(__name__)


@dataclass(slots=True)
class AxisScores:
    """Per-target scores on 4 independent axes, each 0.0-1.0."""

    profile: MobProfile
    efficiency: float = 0.0
    safety: float = 0.0
    resource: float = 0.0
    accessibility: float = 0.0

    @property
    def name(self) -> str:
        n: str = self.profile.spawn.name
        return n


@dataclass(frozen=True, slots=True)
class AxisPriorities:
    """Priority weights for selecting from the Pareto frontier.

    Values are positive floats. Higher = more important axis.
    Normalized to sum=1.0 before use.
    """

    efficiency: float = 0.35
    safety: float = 0.25
    resource: float = 0.20
    accessibility: float = 0.20


# -- Axis Computation ---------------------------------------------------------


def compute_axes(
    p: MobProfile,
    w: ScoringWeights,
    state: GameState,
    ctx: AgentContext | None,
    fight_durations: dict[str, list[float]] | None = None,
    mob_base: str = "",
) -> AxisScores:
    """Score one target on all 4 axes. Each axis is 0.0-1.0."""
    return AxisScores(
        profile=p,
        efficiency=_axis_efficiency(p, w, ctx, fight_durations, mob_base),
        safety=_axis_safety(p, w),
        resource=_axis_resource(p, state),
        accessibility=_axis_accessibility(p, w, ctx),
    )


def _axis_efficiency(
    p: MobProfile,
    w: ScoringWeights,
    ctx: AgentContext | None,
    fight_durations: dict[str, list[float]] | None,
    mob_base: str,
) -> float:
    """How profitable is this target? XP value, loot, encounter speed."""
    s: float = _score_con(p, w)  # 0-100
    s += _score_loot(mob_base, w, ctx)  # 0-30
    s += _score_efficiency(mob_base, w, fight_durations)  # -10 to +10
    # Normalize to 0.0-1.0 (theoretical max ~140)
    return max(0.0, min(1.0, s / 130.0))


def _axis_safety(p: MobProfile, w: ScoringWeights) -> float:
    """How safe is this engagement? Isolation, add risk, threat level."""
    s: float = polynomial(p.isolation_score, 0.0, 1.0, w.isolation_exp)
    s -= p.social_npc_count * 0.15
    s -= p.extra_npc_probability * 0.3
    s -= p.threat_level * 0.2
    # Heading: penalize if NPC is facing player (harder to approach safely)
    # Use a simple threshold on heading data if available
    if p.distance > 0 and p.distance < 80:
        # Approximate facing check from MobProfile data
        # Full heading math is in the scoring function; here use threat_level
        # as a proxy (threat already factors in facing for aggressive NPCs)
        pass
    return max(0.0, min(1.0, s))


def _axis_resource(p: MobProfile, state: GameState) -> float:
    """How expensive is this encounter in mana? Low cost = high score."""
    max_mana = state.mana_max if state.mana_max > 0 else 1
    cost_pct: float = p.mana_cost_est / max_mana
    return max(0.0, min(1.0, 1.0 - cost_pct))


def _axis_accessibility(p: MobProfile, w: ScoringWeights, ctx: AgentContext | None) -> float:
    """How quickly can the agent reach this target?"""
    dist_score: float = bell(p.distance, w.dist_ideal, w.dist_width)
    camp_score = 0.5  # default if no camp context
    if ctx and ctx.camp.roam_radius > 0:
        effective_dist = ctx.camp.effective_camp_distance(p.spawn.pos)
        camp_score = inverse_logistic(effective_dist, ctx.camp.roam_radius, w.camp_falloff_k)
    move_penalty = 0.15 if p.is_moving else 0.0
    return max(0.0, min(1.0, dist_score * 0.5 + camp_score * 0.5 - move_penalty))


# -- Pareto Frontier ----------------------------------------------------------


def pareto_frontier(candidates: list[AxisScores]) -> list[AxisScores]:
    """Return non-dominated targets. O(n^2) but n < 50 in practice."""
    if not candidates:
        return []
    frontier: list[AxisScores] = []
    for t in candidates:
        dominated = False
        for other in candidates:
            if other is t:
                continue
            if _dominates(other, t):
                dominated = True
                break
        if not dominated:
            frontier.append(t)
    return frontier


def _dominates(a: AxisScores, b: AxisScores) -> bool:
    """True if a dominates b (>= on all axes, > on at least one)."""
    ge = (
        a.efficiency >= b.efficiency
        and a.safety >= b.safety
        and a.resource >= b.resource
        and a.accessibility >= b.accessibility
    )
    gt = (
        a.efficiency > b.efficiency
        or a.safety > b.safety
        or a.resource > b.resource
        or a.accessibility > b.accessibility
    )
    return ge and gt


# -- Priority-Weighted Selection ----------------------------------------------


def select_from_frontier(frontier: list[AxisScores], priorities: AxisPriorities) -> AxisScores | None:
    """Select the best target from the Pareto frontier using priority weights."""
    if not frontier:
        return None
    # Normalize priorities
    total = priorities.efficiency + priorities.safety + priorities.resource + priorities.accessibility
    if total <= 0:
        total = 1.0
    pe = priorities.efficiency / total
    ps = priorities.safety / total
    pr = priorities.resource / total
    pa = priorities.accessibility / total

    return max(
        frontier, key=lambda t: t.efficiency * pe + t.safety * ps + t.resource * pr + t.accessibility * pa
    )


def compute_priorities(
    state: GameState, ctx: AgentContext | None, phase: str, base: AxisPriorities | None = None
) -> AxisPriorities:
    """Compute state-aware priority weights for Pareto selection.

    Shifts default weights based on:
    - Current mana/HP levels
    - Session phase (incident -> safety, idle -> accessibility)
    - Pareto weight fields from ScoringWeights (gradient-learned)
    """
    b = base or AxisPriorities()
    eff = b.efficiency
    saf = b.safety
    res = b.resource
    acc = b.accessibility

    # Low mana -> weight resource cost
    if state.mana_pct < 0.40:
        res += 0.15
        eff -= 0.10
        acc -= 0.05

    # Low HP -> weight safety
    if state.hp_pct < 0.60:
        saf += 0.20
        eff -= 0.15
        acc -= 0.05

    # Session phase adjustments
    if phase == "incident":
        saf += 0.20
        eff -= 0.15
        acc -= 0.05
    elif phase == "idle":
        acc += 0.15
        eff -= 0.10
        res -= 0.05
    elif phase == "startup":
        saf += 0.10
        acc += 0.05
        eff -= 0.10
        res -= 0.05

    # Clamp negatives
    eff = max(0.05, eff)
    saf = max(0.05, saf)
    res = max(0.05, res)
    acc = max(0.05, acc)

    # Normalize
    total = eff + saf + res + acc
    return AxisPriorities(
        efficiency=eff / total,
        safety=saf / total,
        resource=res / total,
        accessibility=acc / total,
    )


def log_pareto_selection(
    frontier: list[AxisScores], selected: AxisScores, priorities: AxisPriorities, total_candidates: int
) -> None:
    """Log Pareto selection at T2 (INFO) with axis scores."""
    log.info(
        "[TARGET] Pareto: %d of %d candidates on frontier, selected '%s' "
        "(eff=%.2f saf=%.2f res=%.2f acc=%.2f) "
        "priorities=(%.2f/%.2f/%.2f/%.2f)",
        len(frontier),
        total_candidates,
        selected.name,
        selected.efficiency,
        selected.safety,
        selected.resource,
        selected.accessibility,
        priorities.efficiency,
        priorities.safety,
        priorities.resource,
        priorities.accessibility,
    )
    # Log frontier details at VERBOSE
    if len(frontier) > 1:
        for t in frontier[:5]:
            if t is not selected:
                log.log(
                    VERBOSE,
                    "[TARGET] Pareto frontier: '%s' (eff=%.2f saf=%.2f res=%.2f acc=%.2f)",
                    t.name,
                    t.efficiency,
                    t.safety,
                    t.resource,
                    t.accessibility,
                )
