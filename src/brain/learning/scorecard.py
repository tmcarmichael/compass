"""Session evaluation: measures agent performance across key metrics.

Produces a 0-100 score per category. Rate-based deductions make
short and long sessions comparable.

The feedback loop (evaluate_and_tune) reads category scores and adjusts
agent parameters within bounded ranges (+/-50% of defaults). Adjustments
persist to data/memory/<zone>_tuning.json and are reloaded on startup.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypedDict

if TYPE_CHECKING:
    from brain.context import AgentContext

_tune_log = logging.getLogger("compass.tuning")


class ScorecardResult(TypedDict):
    """Typed shape of the dict returned by compute_scorecard()."""

    pathing: int
    defeat_rate: int
    pull_success: int
    targeting: int
    survival: int
    mana_efficiency: int
    uptime: int
    overall: int
    grade: str
    _hours: float
    _kills: int
    _deaths: int
    _flees: int
    _stuck: int


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _score_inverse(rate_per_hour: float, perfect: float, failing: float) -> int:
    """Score inversely proportional to a rate. Lower rate = higher score.

    perfect: rate at which score is 100
    failing: rate at which score is 0
    Linear interpolation between them.
    """
    if rate_per_hour <= perfect:
        return 100
    if rate_per_hour >= failing:
        return 0
    return int(_clamp(100 * (1 - (rate_per_hour - perfect) / (failing - perfect))))


def _score_direct(rate_per_hour: float, failing: float, perfect: float) -> int:
    """Score directly proportional to a rate. Higher rate = higher score.

    failing: rate at which score is 0
    perfect: rate at which score is 100
    """
    if rate_per_hour >= perfect:
        return 100
    if rate_per_hour <= failing:
        return 0
    return int(_clamp(100 * (rate_per_hour - failing) / (perfect - failing)))


def _score_pct(success_rate: float) -> int:
    """Score from a 0.0-1.0 success rate. 1.0 = 100, 0.0 = 0."""
    return int(_clamp(success_rate * 100))


def compute_scorecard(ctx: AgentContext) -> dict[str, int | float | str]:
    """Compute performance scores from session context.

    Returns dict with category scores (0-100) and an overall grade.
    """
    from nav.movement import get_stuck_event_count

    elapsed = time.time() - ctx.metrics.session_start
    hours = max(elapsed / 3600, 0.01)

    scores: dict[str, int | float | str] = {}

    # -- PATHING (stuck events per hour) --
    # 0 stuck/hr = 100, 10+/hr = 0
    stuck = get_stuck_event_count()
    stuck_per_hr = stuck / hours
    scores["pathing"] = _score_inverse(stuck_per_hr, 0.0, 10.0)

    # -- COMBAT EFFICIENCY (defeats per hour, scaled by level expectations) --
    # At low levels (1-5): 20 dph = perfect, 5 dph = failing
    # At higher levels: expectations decrease
    dph = ctx.defeat_tracker.defeats / hours
    scores["defeat_rate"] = _score_direct(dph, 5.0, 20.0)

    # -- PULL SUCCESS (pulls that reach combat vs aborts/timeouts) --
    total_pulls = ctx.metrics.routine_counts.get("PULL", 0)
    pull_fails = ctx.metrics.routine_failures.get("PULL", 0)
    if total_pulls > 0:
        pull_success = (total_pulls - pull_fails) / total_pulls
        scores["pull_success"] = _score_pct(pull_success)
    else:
        scores["pull_success"] = 50  # no data

    # -- ACQUIRE EFFICIENCY (tabs per successful acquire) --
    # 1.0 avg tabs = perfect, 5+ = poor
    if ctx.metrics.acquire_tab_totals:
        avg_tabs = sum(ctx.metrics.acquire_tab_totals) / len(ctx.metrics.acquire_tab_totals)
        scores["targeting"] = _score_inverse(avg_tabs, 1.0, 5.0)
    else:
        scores["targeting"] = 50

    # -- SURVIVAL (deaths + flees per hour) --
    # 0 deaths + 0 flees = 100; 1 death/hr = 30; 3+ deaths/hr = 0
    death_rate = ctx.player.deaths / hours
    flee_rate = ctx.metrics.flee_count / hours
    # Deaths heavily penalized, flees are minor (fleeing is smart)
    survival_penalty = death_rate * 70 + flee_rate * 5
    scores["survival"] = int(_clamp(100 - survival_penalty))

    # -- MANA EFFICIENCY (casts per defeat -- lower is more efficient) --
    # 1-2 casts/defeat = perfect, 6+ = wasteful
    if ctx.defeat_tracker.defeats > 0 and ctx.metrics.total_casts > 0:
        casts_per_kill = ctx.metrics.total_casts / ctx.defeat_tracker.defeats
        scores["mana_efficiency"] = _score_inverse(casts_per_kill, 1.5, 6.0)
    else:
        scores["mana_efficiency"] = 50

    # -- UPTIME (% of session spent in combat + pulling vs resting + idle) --
    # Lock-protected: brain thread mutates routine_time in-place
    combat_time = ctx.metrics.total_combat_time
    with ctx.lock:
        pull_time = ctx.metrics.routine_time.get("PULL", 0)
        active_time = combat_time + pull_time
        total_tracked = sum(ctx.metrics.routine_time.values()) if ctx.metrics.routine_time else elapsed
    if total_tracked > 0:
        active_pct = active_time / total_tracked
        # 60%+ active = perfect, 20% = failing
        scores["uptime"] = _score_direct(active_pct, 0.20, 0.60)
    else:
        scores["uptime"] = 50

    # -- OVERALL: weighted average --
    weights = {
        "defeat_rate": 25,
        "survival": 20,
        "pull_success": 15,
        "uptime": 15,
        "pathing": 10,
        "targeting": 5,
        "mana_efficiency": 10,
    }
    total_weight = sum(weights.values())
    weighted_sum = sum(int(scores[k]) * weights[k] for k in weights)
    scores["overall"] = int(_clamp(weighted_sum / total_weight))

    # Letter grade
    overall = int(scores["overall"])
    if overall >= 90:
        scores["grade"] = "A"
    elif overall >= 80:
        scores["grade"] = "B"
    elif overall >= 70:
        scores["grade"] = "C"
    elif overall >= 60:
        scores["grade"] = "D"
    else:
        scores["grade"] = "F"

    # Store raw stats for context
    scores["_hours"] = round(hours, 2)
    scores["_kills"] = ctx.defeat_tracker.defeats
    scores["_deaths"] = ctx.player.deaths
    scores["_flees"] = ctx.metrics.flee_count
    scores["_stuck"] = stuck

    return scores


def format_scorecard(scores: dict[str, int | float | str]) -> str:
    """Format scorecard as a human-readable log block."""
    lines = [
        "",
        "SESSION SCORECARD",
        "-" * 40,
    ]

    categories = [
        ("defeat_rate", "Throughput", "npcs per hour"),
        ("survival", "Survival", "deaths and flees"),
        ("pull_success", "Pull Success", "pulls reaching combat"),
        ("uptime", "Uptime", "active vs idle time"),
        ("pathing", "Pathing", "stuck events"),
        ("targeting", "Targeting", "tabs per acquire"),
        ("mana_efficiency", "Mana Efficiency", "casts per npc"),
    ]

    for key, label, desc in categories:
        val = int(scores.get(key, 0))
        bar = _bar(val)
        lines.append(f"  {label:<18} {bar} {val:>3}/100  ({desc})")

    overall = int(scores.get("overall", 0))
    grade = scores.get("grade", "?")
    lines.append("-" * 40)
    lines.append(f"  OVERALL            {_bar(overall)} {overall:>3}/100  Grade: {grade}")
    lines.append("")

    # Raw stats footnote
    hrs = scores.get("_hours", 0)
    lines.append(
        f"  [{hrs:.1f}hr | {scores.get('_kills', 0)} npcs | "
        f"{scores.get('_deaths', 0)} deaths | "
        f"{scores.get('_flees', 0)} flees | "
        f"{scores.get('_stuck', 0)} stuck]"
    )

    return "\n".join(lines)


def _bar(value: int, width: int = 20) -> str:
    """ASCII progress bar."""
    filled = int(value / 100 * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


# -- Encounter Fitness: training signal for gradient weight learning ----------


def encounter_fitness(
    duration: float, mana_spent: int, max_mana: int, hp_delta: float, defeated: bool, expected_duration: float
) -> float:
    """Compute fitness score for a single encounter outcome.

    Returns 0.0 (fled/died) to 1.0 (perfect encounter).
    This is the universal training signal for scoring weight learning.

    Components (0.5 baseline + 0.2 duration + 0.2 resource + 0.1 safety):
        - Baseline 0.5 for a successful defeat
        - Duration efficiency: faster relative to expectation
        - Resource efficiency: less mana spent relative to pool
        - Safety: less HP lost during encounter
    """
    if not defeated:
        return 0.0
    score = 0.5
    # Duration efficiency (0.0 to 0.2): faster = better
    if expected_duration > 0 and duration > 0:
        ratio = min(expected_duration / duration, 1.0)
        score += 0.2 * ratio
    else:
        score += 0.1  # no expectation data, assume average
    # Resource efficiency (0.0 to 0.2): less mana = better
    if max_mana > 0:
        spent_pct = mana_spent / max_mana
        score += 0.2 * (1.0 - min(spent_pct, 1.0))
    else:
        score += 0.1
    # Safety (0.0 to 0.1): less HP lost = better
    # hp_delta is negative (e.g. -0.15 = lost 15% HP)
    score += 0.1 * max(0.0, 1.0 + hp_delta)
    return min(score, 1.0)


# -- Feedback Loop: auto-tune parameters from scorecard -----------------------


@dataclass(slots=True)
class TuningParams:
    """Adjustable agent parameters derived from scorecard feedback.

    Each parameter has a default and bounded range (+/-50% of default).
    """

    roam_radius_mult: float = 1.0  # multiplier on camp roam_radius
    social_npc_limit: int = 3  # hard reject threshold for social extra_npcs
    mana_conserve_level: int = 0  # 0=normal, 1=tighter, 2=pet-only-for-blue

    # Bounds (class-level constants, excluded from asdict/serialization)
    _ROAM_MIN: ClassVar[float] = 0.5
    _ROAM_MAX: ClassVar[float] = 1.5
    _SOCIAL_MIN: ClassVar[int] = 2
    _SOCIAL_MAX: ClassVar[int] = 5


def _clamp_tuning(p: TuningParams) -> TuningParams:
    """Enforce bounds on all tuning parameters."""
    p.roam_radius_mult = max(p._ROAM_MIN, min(p._ROAM_MAX, p.roam_radius_mult))
    p.social_npc_limit = max(p._SOCIAL_MIN, min(p._SOCIAL_MAX, p.social_npc_limit))
    p.mana_conserve_level = max(0, min(2, p.mana_conserve_level))
    return p


def _tune_roam(p: TuningParams, defeat_rate: int) -> list[str]:
    changes: list[str] = []
    if defeat_rate < 40:
        old = p.roam_radius_mult
        p.roam_radius_mult = min(p._ROAM_MAX, p.roam_radius_mult + 0.15)
        if p.roam_radius_mult != old:
            changes.append(
                f"roam_radius_mult {old:.2f} -> {p.roam_radius_mult:.2f} (defeat_rate={defeat_rate} < 40)"
            )
    elif defeat_rate > 80:
        old = p.roam_radius_mult
        p.roam_radius_mult = max(p._ROAM_MIN, p.roam_radius_mult - 0.05)
        if p.roam_radius_mult != old:
            changes.append(
                f"roam_radius_mult {old:.2f} -> {p.roam_radius_mult:.2f} (defeat_rate={defeat_rate} > 80, tightening)"
            )
    return changes


def _tune_social(p: TuningParams, pull_success: int, survival: int) -> list[str]:
    changes: list[str] = []
    if pull_success < 50:
        old = p.social_npc_limit
        p.social_npc_limit = max(p._SOCIAL_MIN, p.social_npc_limit - 1)
        if p.social_npc_limit != old:
            changes.append(
                f"social_npc_limit {old} -> {p.social_npc_limit} (pull_success={pull_success} < 50)"
            )
    elif pull_success > 85 and survival > 80:
        old = p.social_npc_limit
        p.social_npc_limit = min(p._SOCIAL_MAX, p.social_npc_limit + 1)
        if p.social_npc_limit != old:
            changes.append(
                f"social_npc_limit {old} -> {p.social_npc_limit} (pull_success={pull_success} > 85, relaxing)"
            )
    return changes


def _tune_mana(p: TuningParams, mana_eff: int) -> list[str]:
    changes: list[str] = []
    if mana_eff < 40:
        old = p.mana_conserve_level
        p.mana_conserve_level = min(2, p.mana_conserve_level + 1)
        if p.mana_conserve_level != old:
            changes.append(
                f"mana_conserve_level {old} -> {p.mana_conserve_level} (mana_efficiency={mana_eff} < 40)"
            )
    elif mana_eff > 80:
        old = p.mana_conserve_level
        p.mana_conserve_level = max(0, p.mana_conserve_level - 1)
        if p.mana_conserve_level != old:
            changes.append(
                f"mana_conserve_level {old} -> {p.mana_conserve_level} (mana_efficiency={mana_eff} > 80, loosening)"
            )
    return changes


def evaluate_and_tune(scores: dict, current: TuningParams | None = None) -> TuningParams:
    """Adjust tuning parameters based on scorecard grades.

    Returns updated TuningParams. Every adjustment is logged with
    before/after values and the scorecard trigger.
    """
    p = current or TuningParams()

    defeat_rate = scores.get("defeat_rate", 50)
    survival = scores.get("survival", 50)
    pull_success = scores.get("pull_success", 50)
    mana_eff = scores.get("mana_efficiency", 50)

    changes = _tune_roam(p, defeat_rate)
    changes.extend(_tune_social(p, pull_success, survival))
    changes.extend(_tune_mana(p, mana_eff))

    # Survival critically low -> tighten everything
    if survival < 30:
        old_roam = p.roam_radius_mult
        p.roam_radius_mult = max(p._ROAM_MIN, p.roam_radius_mult - 0.1)
        old_social = p.social_npc_limit
        p.social_npc_limit = max(p._SOCIAL_MIN, p.social_npc_limit - 1)
        if p.roam_radius_mult != old_roam or p.social_npc_limit != old_social:
            changes.append(
                f"SURVIVAL EMERGENCY: roam {old_roam:.2f}->{p.roam_radius_mult:.2f}, "
                f"social {old_social}->{p.social_npc_limit} "
                f"(survival={survival} < 30)"
            )

    p = _clamp_tuning(p)

    if changes:
        for c in changes:
            _tune_log.info("[TUNING] %s", c)
    else:
        _tune_log.info(
            "[TUNING] no adjustments (scores: defeat=%d surv=%d pull=%d mana=%d)",
            defeat_rate,
            survival,
            pull_success,
            mana_eff,
        )

    return p


def save_tuning(params: TuningParams, zone: str, data_dir: str = "data/memory") -> None:
    """Persist tuning params to data/memory/<zone>_tuning.json."""
    path = str(Path(data_dir) / f"{zone}_tuning.json")
    data = {
        "v": 1,
        "roam_radius_mult": params.roam_radius_mult,
        "social_npc_limit": params.social_npc_limit,
        "mana_conserve_level": params.mana_conserve_level,
    }
    try:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        _tune_log.info("[TUNING] saved to %s", path)
    except OSError as e:
        _tune_log.warning("[TUNING] failed to save to %s: %s", path, e)


def load_tuning(zone: str, data_dir: str = "data/memory") -> TuningParams:
    """Load tuning params from data/memory/<zone>_tuning.json, or defaults."""
    path = str(Path(data_dir) / f"{zone}_tuning.json")
    try:
        with open(path) as f:
            data = json.load(f)
        p = TuningParams(
            roam_radius_mult=data.get("roam_radius_mult", 1.0),
            social_npc_limit=data.get("social_npc_limit", 3),
            mana_conserve_level=data.get("mana_conserve_level", 0),
        )
        p = _clamp_tuning(p)
        _tune_log.info(
            "[TUNING] loaded from %s (roam=%.2f social=%d mana_conserve=%d)",
            path,
            p.roam_radius_mult,
            p.social_npc_limit,
            p.mana_conserve_level,
        )
        return p
    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
    ):
        return TuningParams()


def apply_tuning(params: TuningParams, ctx: AgentContext) -> None:
    """Apply tuning parameters to live AgentContext.

    Modifies ctx.camp.roam_radius and world model scoring weights.
    Logged so the effect is visible in session logs.
    """
    # Roam radius
    if ctx.camp.base_roam_radius == 0.0:
        ctx.camp.base_roam_radius = ctx.camp.roam_radius
    new_roam = ctx.camp.base_roam_radius * params.roam_radius_mult
    if abs(new_roam - ctx.camp.roam_radius) > 1.0:
        _tune_log.info(
            "[TUNING] roam_radius %.0f -> %.0f (mult=%.2f)",
            ctx.camp.roam_radius,
            new_roam,
            params.roam_radius_mult,
        )
        ctx.camp.roam_radius = new_roam

    # Social add limit -> scoring weights
    if hasattr(ctx, "world") and ctx.world is not None and hasattr(ctx.world, "_weights"):
        w = ctx.world._weights
        if w.social_npc_hard_limit != params.social_npc_limit:
            _tune_log.info(
                "[TUNING] social_npc_hard_limit %d -> %d", w.social_npc_hard_limit, params.social_npc_limit
            )
            w.social_npc_hard_limit = params.social_npc_limit
