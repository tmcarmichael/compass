"""Utility scoring selection methods (Phases 0-4).

Extracted from Brain to keep the decision engine focused on coordination.
Each phase implements the PhaseSelector protocol and is chosen at init time
via build_phase_selector(), replacing the if/elif dispatch chain.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Protocol

from brain.phase_modifiers import get_phase_modifier
from brain.rule_def import RuleDef, score_from_considerations
from routines.base import RoutineBase

if TYPE_CHECKING:
    from brain.decision import Brain
    from perception.state import GameState

log = logging.getLogger(__name__)

# -- Type alias for the common return shape of all phase selectors --
_SelectResult = tuple[RoutineBase | None, str, bool]


class PhaseSelector(Protocol):
    """Strategy interface for rule selection phases.

    Each phase implements select() with the same signature so the Brain
    can dispatch without knowing which phase is active.
    """

    def select(
        self,
        brain: Brain,
        state: GameState,
        now: float,
        rule_eval: dict[str, str],
        diag_results: list[str],
        rule_times: dict[str, float],
    ) -> _SelectResult: ...


# Score multiplier applied when GOAP planner suggests a specific action.
# Retained for scoring diagnostics even though GOAP is now authoritative.
GOAP_BOOST = 1.5


def _resolve_phase_context(brain: Brain) -> tuple[str, str]:
    """Return (session_phase, goap_hint) from brain diagnostics."""
    phase = "grinding"
    goap_hint = ""
    if brain._ctx and hasattr(brain._ctx, "diag") and brain._ctx.diag:
        pd = getattr(brain._ctx.diag, "phase_detector", None)
        if pd is not None:
            phase = pd.current_phase
        goap_hint = getattr(brain._ctx.diag, "goap_suggestion", "")
    return phase, goap_hint


def _apply_modifiers(score: float, phase: str, rule_name: str, goap_hint: str) -> float:
    """Apply session-phase modifier and GOAP boost to a raw score."""
    score *= get_phase_modifier(phase, rule_name)
    if goap_hint and rule_name == goap_hint and score > 0:
        score *= GOAP_BOOST
    return score


def _goap_authoritative_check(
    brain: Brain,
    state: GameState,
    now: float,
    goap_hint: str,
    rule_eval: dict[str, str],
    diag_results: list[str],
    rule_times: dict[str, float],
) -> _SelectResult | None:
    """If GOAP suggests a routine and it passes eligibility, return it directly.

    Emergency rules override GOAP: if any emergency rule fires, returns None
    so the caller falls through to normal selection. Otherwise, if the GOAP-
    suggested rule passes condition/cooldown/breaker checks, selects it
    without competing on score.

    Returns None when GOAP should not override (no hint, hint blocked, or
    emergency rule active).
    """
    if not goap_hint:
        return None

    goap_rule: RuleDef | None = None
    emergency_fired = False

    for r in brain._rules:
        # Quick emergency scan: if any emergency rule fires, GOAP defers
        if r.emergency:
            if r.name not in brain._cooldowns or now >= brain._cooldowns[r.name]:
                breaker = brain._breakers.get(r.name)
                if not breaker or breaker.allow():
                    try:
                        if r.condition(state):
                            emergency_fired = True
                            break
                    except Exception:
                        pass

        if r.name == goap_hint:
            goap_rule = r

    if emergency_fired or goap_rule is None:
        return None

    # Check GOAP rule eligibility (cooldown, breaker, condition)
    if _is_rule_blocked(brain, goap_rule, state, now, rule_eval, diag_results, rule_times):
        return None

    return goap_rule.routine, goap_rule.name, goap_rule.emergency


def _is_rule_blocked(
    brain: Brain,
    r: RuleDef,
    state: GameState,
    now: float,
    rule_eval: dict,
    diag_results: list,
    rule_times: dict,
) -> bool:
    """Check condition, cooldown, and circuit breaker.  Returns True if
    rule should be skipped.

    Shared by Phases 2/3/4 to avoid duplicating eligibility logic.
    The condition check is critical: without it, a rule whose predicate
    returns False (e.g. ACQUIRE when suppressed by hard gates) could
    still win selection based on its utility score alone.
    """
    if r.name in brain._cooldowns and now < brain._cooldowns[r.name]:
        remaining = brain._cooldowns[r.name] - now
        rule_eval[r.name] = f"cooldown({remaining:.0f}s)"
        diag_results.append(f"{r.name}=CD")
        rule_times[r.name] = 0.0
        return True
    breaker = brain._breakers.get(r.name)
    if breaker and not breaker.allow():
        rule_eval[r.name] = "OPEN"
        diag_results.append(f"{r.name}=OPEN")
        rule_times[r.name] = 0.0
        return True
    try:
        cond = r.condition(state)
    except Exception as exc:
        log.warning("[DECISION] Rule %s condition raised %s -- skipping", r.name, exc)
        rule_eval[r.name] = "ERROR"
        diag_results.append(f"{r.name}=ERROR")
        rule_times[r.name] = 0.0
        return True
    if not cond:
        rule_eval[r.name] = "cond=F"
        diag_results.append(f"{r.name}=cond_F")
        rule_times[r.name] = 0.0
        return True
    return False


def _safe_score(fn: object, *args: object) -> float:
    """Call a scoring function, returning 0.0 on any exception.

    Logs at WARNING so failures are visible but don't crash the tick.
    """
    try:
        result: float = fn(*args)  # type: ignore[operator]
        return result
    except Exception:
        log.warning(
            "[DECISION] Score function %s raised, defaulting to 0.0",
            getattr(fn, "__name__", fn),
            exc_info=True,
        )
        return 0.0


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
        breaker = brain._breakers.get(r.name)
        if breaker and not breaker.allow():
            scores[r.name] = -2.0  # circuit-broken
            continue
        # Check condition so telemetry doesn't report ineligible "winners"
        try:
            if not r.condition(state):
                scores[r.name] = -3.0  # condition false
                continue
        except Exception:
            scores[r.name] = -4.0  # condition error
            continue
        s = _safe_score(r.score_fn, state)
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
    Between tiers, higher tier (lower number) wins if any rule scores > 0.
    GOAP suggestion is authoritative (selected directly) unless an emergency
    rule fires."""
    phase, goap_hint = _resolve_phase_context(brain)

    # GOAP authoritative: select directly if eligible and no emergency
    goap_result = _goap_authoritative_check(brain, state, now, goap_hint, rule_eval, diag_results, rule_times)
    if goap_result is not None:
        return goap_result

    tier_groups: dict[int, list[RuleDef]] = defaultdict(list)

    for r in brain._rules:
        if _is_rule_blocked(brain, r, state, now, rule_eval, diag_results, rule_times):
            continue
        tier_groups[r.tier].append(r)

    for tier in sorted(tier_groups):
        scored: list[tuple[float, RuleDef]] = []
        for r in tier_groups[tier]:
            t0 = time.perf_counter()
            s = _apply_modifiers(_safe_score(r.score_fn, state), phase, r.name, goap_hint)
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
    Emergency rules retain hard priority. GOAP suggestion is authoritative
    (selected directly) unless an emergency rule fires. Non-emergency rules
    compete by weight * score as fallback."""
    phase, goap_hint = _resolve_phase_context(brain)

    # GOAP authoritative: select directly if eligible and no emergency
    goap_result = _goap_authoritative_check(brain, state, now, goap_hint, rule_eval, diag_results, rule_times)
    if goap_result is not None:
        return goap_result

    emergency: list[tuple[float, RuleDef]] = []
    normal: list[tuple[float, RuleDef]] = []

    for r in brain._rules:
        if _is_rule_blocked(brain, r, state, now, rule_eval, diag_results, rule_times):
            continue
        t0 = time.perf_counter()
        s = _apply_modifiers(_safe_score(r.score_fn, state), phase, r.name, goap_hint)
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
    GOAP suggestion is authoritative (selected directly) unless an emergency
    rule fires. Otherwise identical to Phase 3 (emergency hard priority,
    weighted cross-tier selection for non-emergency).
    """
    phase, goap_hint = _resolve_phase_context(brain)

    # GOAP authoritative: select directly if eligible and no emergency
    goap_result = _goap_authoritative_check(brain, state, now, goap_hint, rule_eval, diag_results, rule_times)
    if goap_result is not None:
        return goap_result

    emergency: list[tuple[float, RuleDef]] = []
    normal: list[tuple[float, RuleDef]] = []

    for r in brain._rules:
        if _is_rule_blocked(brain, r, state, now, rule_eval, diag_results, rule_times):
            continue
        t0 = time.perf_counter()
        # Phase 4: prefer considerations over score_fn when defined
        if r.considerations and brain._ctx:
            raw = _safe_score(score_from_considerations, r.considerations, state, brain._ctx)
        else:
            raw = _safe_score(r.score_fn, state)
        s = _apply_modifiers(raw, phase, r.name, goap_hint)
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


# =====================================================================
# Phase selector implementations
# =====================================================================


def select_binary(
    brain: Brain,
    state: GameState,
    now: float,
    rule_eval: dict[str, str],
    diag_results: list[str],
    rule_times: dict[str, float],
) -> _SelectResult:
    """Phase 0/1: binary conditions, insertion-order priority.

    Short-circuits after winner found: skips condition evaluation for
    lower-priority rules. This is safe because detection logic (threat
    scan, add detection) runs pre-rule in the tick pipeline, not inside
    condition functions.
    """
    selected: RoutineBase | None = None
    selected_name = ""
    selected_emergency = False

    for r in brain._rules:
        if r.name in brain._cooldowns and now < brain._cooldowns[r.name]:
            remaining = brain._cooldowns[r.name] - now
            rule_eval[r.name] = f"cooldown({remaining:.0f}s)"
            diag_results.append(f"{r.name}=CD")
            rule_times[r.name] = 0.0
            continue
        breaker = brain._breakers.get(r.name)
        if breaker and not breaker.allow():
            rule_eval[r.name] = "OPEN"
            diag_results.append(f"{r.name}=OPEN")
            rule_times[r.name] = 0.0
            continue
        if selected is not None:
            rule_eval[r.name] = "skip"
            diag_results.append(f"{r.name}=skip")
            rule_times[r.name] = 0.0
            continue
        t0 = brain.perf_clock()
        try:
            matched = r.condition(state)
        except Exception as exc:
            rule_times[r.name] = (brain.perf_clock() - t0) * 1000
            rule_eval[r.name] = "ERROR"
            diag_results.append(f"{r.name}=ERROR")
            log.warning("[DECISION] Rule %s condition raised %s -- skipping", r.name, exc)
            continue
        rule_times[r.name] = (brain.perf_clock() - t0) * 1000
        rule_eval[r.name] = "YES" if matched else "no"
        diag_results.append(f"{r.name}={'YES' if matched else 'no'}")
        if matched:
            selected = r.routine
            selected_name = r.name
            selected_emergency = r.emergency

    return selected, selected_name, selected_emergency


class BinarySelector:
    """Phase 0/1: binary condition evaluation, insertion-order priority."""

    def select(
        self,
        brain: Brain,
        state: GameState,
        now: float,
        rule_eval: dict[str, str],
        diag_results: list[str],
        rule_times: dict[str, float],
    ) -> _SelectResult:
        return select_binary(brain, state, now, rule_eval, diag_results, rule_times)


class TierSelector:
    """Phase 2: score-based selection within priority tiers."""

    def select(
        self,
        brain: Brain,
        state: GameState,
        now: float,
        rule_eval: dict[str, str],
        diag_results: list[str],
        rule_times: dict[str, float],
    ) -> _SelectResult:
        return select_by_tier(brain, state, now, rule_eval, diag_results, rule_times)


class WeightedSelector:
    """Phase 3: weighted cross-tier scoring."""

    def select(
        self,
        brain: Brain,
        state: GameState,
        now: float,
        rule_eval: dict[str, str],
        diag_results: list[str],
        rule_times: dict[str, float],
    ) -> _SelectResult:
        return select_weighted(brain, state, now, rule_eval, diag_results, rule_times)


class ConsiderationSelector:
    """Phase 4: consideration-based scoring with weighted geometric mean."""

    def select(
        self,
        brain: Brain,
        state: GameState,
        now: float,
        rule_eval: dict[str, str],
        diag_results: list[str],
        rule_times: dict[str, float],
    ) -> _SelectResult:
        return select_with_considerations(brain, state, now, rule_eval, diag_results, rule_times)


def build_phase_selector(phase: int) -> PhaseSelector:
    """Factory: return the appropriate selector for a utility phase level."""
    if phase >= 4:
        return ConsiderationSelector()
    if phase >= 3:
        return WeightedSelector()
    if phase >= 2:
        return TierSelector()
    return BinarySelector()
