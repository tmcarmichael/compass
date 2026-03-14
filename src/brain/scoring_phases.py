"""Utility scoring selection methods (Phases 1-4).

Extracted from Brain to keep the decision engine focused on coordination.
Each function takes the Brain instance and delegates to its internal state.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from brain.phase_modifiers import get_phase_modifier
from brain.rule_def import RuleDef, score_from_considerations
from routines.base import RoutineBase

if TYPE_CHECKING:
    from brain.decision import Brain
    from perception.state import GameState

log = logging.getLogger(__name__)


def compute_divergence(brain: Brain, state: GameState, now: float, binary_winner: str) -> None:
    """Phase 1: compute scores for all rules, log when score-based
    selection would differ from binary selection."""
    scores: dict[str, float] = {}
    best_score = 0.0
    score_winner = ""

    for r in brain._rules:
        if r.name in brain._cooldowns and now < brain._cooldowns[r.name]:
            scores[r.name] = -1.0  # on cooldown
            continue
        try:
            s = r.score_fn(state)
        except Exception:
            s = 0.0
        scores[r.name] = s
        if s > best_score:
            best_score = s
            score_winner = r.name

    brain.rule_scores = scores
    brain._score_winner = score_winner

    if score_winner and score_winner != binary_winner:
        log.debug(
            "[DECISION] UTIL_DIVERGE: binary=%s score=%s (%.2f)",
            binary_winner or "NONE",
            score_winner,
            best_score,
        )

    # Expose scores to dashboard
    if brain._ctx:
        brain._ctx.diag.rule_scores = scores


def select_by_tier(
    brain: Brain, state: GameState, now: float, rule_eval: dict, diag_results: list, rule_times: dict
) -> tuple[RoutineBase | None, str, bool]:
    """Phase 2: within each tier, select by highest score.
    Between tiers, higher tier (lower number) wins if any rule scores > 0."""
    tier_groups: dict[int, list[RuleDef]] = defaultdict(list)

    for r in brain._rules:
        if r.name in brain._cooldowns and now < brain._cooldowns[r.name]:
            remaining = brain._cooldowns[r.name] - now
            rule_eval[r.name] = f"cooldown({remaining:.0f}s)"
            diag_results.append(f"{r.name}=CD")
            rule_times[r.name] = 0.0
            continue
        tier_groups[r.tier].append(r)

    # Get session phase for contextual score modifiers
    phase = "grinding"
    goap_hint = ""
    if brain._ctx and hasattr(brain._ctx, "diag") and brain._ctx.diag:
        pd = getattr(brain._ctx.diag, "phase_detector", None)
        if pd is not None:
            phase = pd.current_phase
        goap_hint = getattr(brain._ctx.diag, "goap_suggestion", "")

    for tier in sorted(tier_groups):
        scored: list[tuple[float, RuleDef]] = []
        for r in tier_groups[tier]:
            t0 = time.perf_counter()
            s = r.score_fn(state)
            # Apply session phase modifier (startup, incident, idle, etc.)
            s *= get_phase_modifier(phase, r.name)
            # GOAP planner boost: prefer the planned action
            if goap_hint and r.name == goap_hint and s > 0:
                s *= 1.5  # 50% score boost for GOAP-suggested action
            rule_times[r.name] = (time.perf_counter() - t0) * 1000
            rule_eval[r.name] = f"{s:.2f}" if s > 0 else "0"
            diag_results.append(f"{r.name}={s:.2f}")
            if s > 0.0:
                scored.append((s, r))
        if scored:
            best = max(scored, key=lambda x: x[0])[1]
            return best.routine, best.name, best.emergency

    return None, "", False


def select_weighted(
    brain: Brain, state: GameState, now: float, rule_eval: dict, diag_results: list, rule_times: dict
) -> tuple[RoutineBase | None, str, bool]:
    """Phase 3+: weighted cross-tier scoring.
    Emergency rules retain hard priority. Non-emergency rules compete
    by weight * score."""
    emergency: list[tuple[float, RuleDef]] = []
    normal: list[tuple[float, RuleDef]] = []

    # Session phase for contextual modifiers
    phase = "grinding"
    goap_hint = ""
    if brain._ctx and hasattr(brain._ctx, "diag") and brain._ctx.diag:
        pd = getattr(brain._ctx.diag, "phase_detector", None)
        if pd is not None:
            phase = pd.current_phase
        goap_hint = getattr(brain._ctx.diag, "goap_suggestion", "")

    for r in brain._rules:
        if r.name in brain._cooldowns and now < brain._cooldowns[r.name]:
            remaining = brain._cooldowns[r.name] - now
            rule_eval[r.name] = f"cooldown({remaining:.0f}s)"
            diag_results.append(f"{r.name}=CD")
            rule_times[r.name] = 0.0
            continue
        t0 = time.perf_counter()
        s = r.score_fn(state)
        # Apply session phase modifier
        s *= get_phase_modifier(phase, r.name)
        # GOAP planner boost
        if goap_hint and r.name == goap_hint and s > 0:
            s *= 1.5
        rule_times[r.name] = (time.perf_counter() - t0) * 1000
        weighted = r.weight * s
        rule_eval[r.name] = f"{weighted:.1f}" if s > 0 else "0"
        diag_results.append(f"{r.name}={weighted:.1f}")
        if s <= 0.0:
            continue
        if r.emergency:
            emergency.append((s, r))
        else:
            normal.append((weighted, r))

    if emergency:
        best = max(emergency, key=lambda x: x[0])[1]
        return best.routine, best.name, best.emergency
    if normal:
        best = max(normal, key=lambda x: x[0])[1]
        return best.routine, best.name, best.emergency

    return None, "", False


def select_with_considerations(
    brain: Brain, state: GameState, now: float, rule_eval: dict, diag_results: list, rule_times: dict
) -> tuple[RoutineBase | None, str, bool]:
    """Phase 4: consideration-based scoring with weighted geometric mean.

    Rules with considerations use score_from_considerations() instead of
    score_fn(). Rules without considerations fall back to score_fn().
    Otherwise identical to Phase 3 (emergency hard priority, weighted
    cross-tier selection for non-emergency).
    """
    emergency: list[tuple[float, RuleDef]] = []
    normal: list[tuple[float, RuleDef]] = []

    phase = "grinding"
    goap_hint = ""
    if brain._ctx and hasattr(brain._ctx, "diag") and brain._ctx.diag:
        pd = getattr(brain._ctx.diag, "phase_detector", None)
        if pd is not None:
            phase = pd.current_phase
        goap_hint = getattr(brain._ctx.diag, "goap_suggestion", "")

    for r in brain._rules:
        if r.name in brain._cooldowns and now < brain._cooldowns[r.name]:
            remaining = brain._cooldowns[r.name] - now
            rule_eval[r.name] = f"cooldown({remaining:.0f}s)"
            diag_results.append(f"{r.name}=CD")
            rule_times[r.name] = 0.0
            continue
        t0 = time.perf_counter()
        # Phase 4: prefer considerations over score_fn when defined
        if r.considerations and brain._ctx:
            s = score_from_considerations(r.considerations, state, brain._ctx)
        else:
            s = r.score_fn(state)
        s *= get_phase_modifier(phase, r.name)
        if goap_hint and r.name == goap_hint and s > 0:
            s *= 1.5
        rule_times[r.name] = (time.perf_counter() - t0) * 1000
        weighted = r.weight * s
        rule_eval[r.name] = f"{weighted:.1f}" if s > 0 else "0"
        diag_results.append(f"{r.name}={weighted:.1f}")
        if s <= 0.0:
            continue
        if r.emergency:
            emergency.append((s, r))
        else:
            normal.append((weighted, r))

    if emergency:
        best = max(emergency, key=lambda x: x[0])[1]
        return best.routine, best.name, best.emergency
    if normal:
        best = max(normal, key=lambda x: x[0])[1]
        return best.routine, best.name, best.emergency

    return None, "", False
