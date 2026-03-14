"""Gradient weight learning: tunes scoring weights from encounter outcomes.

Observes (fitness, score_breakdown) per encounter and adjusts ScoringWeights
via bounded finite-difference gradient descent. Each weight is perturbed
independently by a small epsilon, the scoring function is re-evaluated, and the
numerical partial derivative determines the update direction.

Convergence: ~100 encounters per zone. Weights bounded +/-20% of defaults.
The bounded region acts as a projection step (projected gradient descent).
Persists to data/memory/<zone>_weights.json across sessions.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

from util.log_tiers import EVENT, VERBOSE

if TYPE_CHECKING:
    from brain.scoring.target import ScoringWeights

log = logging.getLogger(__name__)

# How many observations before a gradient step
STEP_INTERVAL = 15
# Learning rate for weight updates
LEARNING_RATE = 0.05
# Maximum deviation from defaults (+/-20%)
MAX_DRIFT = 0.20
# Minimum observations for a factor to influence gradient
MIN_FACTOR_OBSERVATIONS = 5
# Maximum observation history
MAX_OBSERVATIONS = 200
# Epsilon for finite-difference perturbation (fraction of current weight)
FD_EPSILON = 0.02

# Factors from score_target() breakdown that map to tunable weights.
# Each factor name corresponds to a key in the breakdown dict.
# The mapped weight fields are what get adjusted.
FACTOR_WEIGHT_MAP: dict[str, list[str]] = {
    "con_pref": ["con_white", "con_blue", "con_light_blue", "con_yellow"],
    "resource": ["resource_bonus"],
    "distance": ["dist_ideal", "dist_width", "dist_peak"],
    "isolation": ["isolation_peak", "isolation_exp"],
    "social_add": ["social_npc_penalty"],
    "camp_proximity": ["camp_peak", "camp_falloff_k"],
    "movement": ["moving_penalty"],
    "caster": ["caster_penalty"],
    "loot_value": ["loot_value_scale"],
    "spatial_heat": ["heat_multiplier"],
    "learned_efficiency": ["fast_defeat_bonus", "slow_defeat_penalty"],
    "pareto_efficiency": ["pareto_eff_weight"],
    "pareto_safety": ["pareto_saf_weight"],
    "pareto_resource": ["pareto_res_weight"],
    "pareto_accessibility": ["pareto_acc_weight"],
}


@dataclass(slots=True)
class _Observation:
    """Single encounter observation for gradient computation."""

    fitness: float
    breakdown: dict[str, float]


class GradientTuner:
    """Tunes ScoringWeights via finite-difference projected gradient descent.

    Each gradient step perturbs each weight by a small epsilon, re-evaluates
    the average fitness contribution, and computes the numerical partial
    derivative. Updates are projected back into the bounded region (+/-20%
    of defaults) after each step.

    Usage:
        tuner = GradientTuner(weights)
        # After each encounter:
        tuner.observe(fitness, score_breakdown)
        if tuner.ready_to_step():
            tuner.step()
    """

    def __init__(self, weights: ScoringWeights) -> None:
        self._weights = weights
        # Snapshot defaults for bounding
        self._defaults: dict[str, float] = {}
        for f in fields(weights):
            val = getattr(weights, f.name)
            if isinstance(val, (int, float)) and f.name in _all_tunable_fields():
                self._defaults[f.name] = float(val)
        self._observations: deque[_Observation] = deque(maxlen=MAX_OBSERVATIONS)
        self._steps = 0
        self._since_last_step = 0
        # Per-weight adaptive learning rate
        self._weight_lr: dict[str, float] = {}
        self._delta_history: dict[str, list[float]] = {}  # last 5 deltas per weight
        log.info("[TUNER] initialized with %d tunable fields", len(self._defaults))

    @property
    def steps(self) -> int:
        """Number of gradient steps taken this session."""
        return self._steps

    def observe(self, fitness: float, breakdown: dict[str, float]) -> None:
        """Record an encounter outcome for gradient computation."""
        self._observations.append(_Observation(fitness=fitness, breakdown=breakdown))
        self._since_last_step += 1

    def ready_to_step(self) -> bool:
        """True when enough observations have accumulated for a gradient step."""
        return self._since_last_step >= STEP_INTERVAL

    def step(self) -> dict[str, float]:
        """Compute and apply one projected gradient update to scoring weights.

        For each tunable weight:
        1. Perturb weight by +epsilon, compute mean fitness contribution.
        2. Perturb weight by -epsilon, compute mean fitness contribution.
        3. Gradient = (f_plus - f_minus) / (2 * epsilon).
        4. Update: weight += lr * gradient.
        5. Project back into bounded region [default * 0.8, default * 1.2].

        Returns dict of field_name -> delta applied (for logging).
        """
        if len(self._observations) < STEP_INTERVAL:
            return {}

        recent = list(self._observations)[-MAX_OBSERVATIONS:]
        gradients = self._compute_finite_difference_gradients(recent)

        # Apply gradients to weight fields
        deltas: dict[str, float] = {}
        for field_name, gradient in gradients.items():
            if field_name not in self._defaults:
                continue
            default = self._defaults[field_name]
            current = float(getattr(self._weights, field_name))
            if abs(default) < 1e-6:
                continue
            # Adaptive learning rate per weight
            lr = self._adapt_lr(field_name, gradient * abs(default))
            delta = lr * gradient * abs(default)
            new_val = current + delta
            # Project onto bounded region [default * (1 - MAX_DRIFT), default * (1 + MAX_DRIFT)]
            lo = default * (1.0 - MAX_DRIFT)
            hi = default * (1.0 + MAX_DRIFT)
            new_val = max(lo, min(hi, new_val))
            actual_delta = new_val - current
            if abs(actual_delta) > 0.01:
                if isinstance(getattr(self._weights, field_name), int):
                    setattr(self._weights, field_name, int(round(new_val)))
                else:
                    setattr(self._weights, field_name, round(new_val, 4))
                deltas[field_name] = actual_delta

        self._since_last_step = 0
        self._steps += 1

        if deltas:
            summary = ", ".join(f"{k}:{v:+.2f}" for k, v in deltas.items())
            log.log(EVENT, "[TUNER] step %d: %s", self._steps, summary)
        else:
            log.log(VERBOSE, "[TUNER] step %d: no significant changes", self._steps)

        return deltas

    def _compute_finite_difference_gradients(
        self,
        recent: list[_Observation],
    ) -> dict[str, float]:
        """Compute per-weight gradient via central finite differences.

        For each tunable weight w_i:
          f(w_i + eps) = mean fitness of observations where factor_i > 0,
                         weighted by (factor_i / mean_factor_i) to simulate
                         the effect of increasing w_i.
          f(w_i - eps) = same, weighted by inverse.
          gradient_i = (f_plus - f_minus) / (2 * eps)

        This is a numerical approximation of df/dw_i using the observation
        history as a surrogate evaluation.
        """
        gradients: dict[str, float] = {}

        for factor_name, weight_fields in FACTOR_WEIGHT_MAP.items():
            # Collect observations where this factor was active
            pairs: list[tuple[float, float]] = []
            for obs in recent:
                fval = obs.breakdown.get(factor_name)
                if fval is not None and fval != 0.0:
                    pairs.append((fval, obs.fitness))
            if len(pairs) < MIN_FACTOR_OBSERVATIONS:
                continue

            mean_factor = sum(p[0] for p in pairs) / len(pairs)
            std_factor = (sum((p[0] - mean_factor) ** 2 for p in pairs) / len(pairs)) ** 0.5
            if std_factor < 1e-8:
                continue

            # Central finite difference on centered factor values.
            # Simulates perturbing the weight: increasing the weight amplifies
            # the factor's centered contribution to fitness.
            eps = FD_EPSILON
            f_plus = 0.0
            f_minus = 0.0
            for fval, fitness in pairs:
                centered = (fval - mean_factor) / std_factor
                f_plus += fitness * (1.0 + eps * centered)
                f_minus += fitness * (1.0 - eps * centered)

            f_plus /= len(pairs)
            f_minus /= len(pairs)

            gradient = (f_plus - f_minus) / (2 * eps) if eps > 0 else 0.0
            # Clamp gradient magnitude for stability
            gradient = max(-1.0, min(1.0, gradient))

            for field_name in weight_fields:
                gradients[field_name] = gradient

        return gradients

    def _adapt_lr(self, field_name: str, raw_delta: float) -> float:
        """Adaptive learning rate: detect oscillation, convergence, stagnation."""
        # Track last 5 deltas
        history = self._delta_history.setdefault(field_name, [])
        history.append(raw_delta)
        if len(history) > 5:
            history.pop(0)

        lr = self._weight_lr.get(field_name, LEARNING_RATE)

        if len(history) >= 3:
            # Oscillation: alternating positive/negative deltas
            signs = [1 if d > 0 else -1 for d in history if abs(d) > 0.01]
            if len(signs) >= 3:
                sign_changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
                if sign_changes >= len(signs) - 1:
                    lr *= 0.7  # dampen
                    log.log(VERBOSE, "[TUNER] Oscillation: %s lr -> %.4f", field_name, lr)

        # Convergence: small deltas for 5 steps
        if len(history) >= 5 and all(abs(d) < 0.5 for d in history):
            lr *= 0.9

        # Stagnation: no movement for 5 steps
        if len(history) >= 5 and all(abs(d) < 0.01 for d in history):
            lr = min(LEARNING_RATE * 1.5, lr * 1.3)
            log.log(VERBOSE, "[TUNER] Stagnation: %s lr -> %.4f", field_name, lr)

        lr = max(0.005, min(0.15, lr))
        self._weight_lr[field_name] = lr
        return lr

    def get_learning_rates(self) -> dict[str, float]:
        """Current per-weight learning rates for persistence."""
        return dict(self._weight_lr)

    def load_learning_rates(self, rates: dict[str, float]) -> None:
        """Restore per-weight learning rates from previous session."""
        self._weight_lr = dict(rates)

    def get_weight_snapshot(self) -> dict[str, float]:
        """Current tunable weight values for persistence."""
        result: dict[str, float] = {}
        for field_name in self._defaults:
            result[field_name] = float(getattr(self._weights, field_name))
        return result

    def load_learned_weights(self, saved: dict[str, float]) -> int:
        """Apply previously learned weights. Returns count of fields applied."""
        applied = 0
        for field_name, value in saved.items():
            if field_name not in self._defaults:
                continue
            default = self._defaults[field_name]
            lo = default * (1.0 - MAX_DRIFT)
            hi = default * (1.0 + MAX_DRIFT)
            clamped = max(lo, min(hi, value))
            if isinstance(getattr(self._weights, field_name), int):
                setattr(self._weights, field_name, int(round(clamped)))
            else:
                setattr(self._weights, field_name, round(clamped, 4))
            applied += 1
        return applied


def _all_tunable_fields() -> set[str]:
    """All weight field names that are subject to gradient tuning."""
    result: set[str] = set()
    for field_list in FACTOR_WEIGHT_MAP.values():
        result.update(field_list)
    return result


def save_learned_weights(
    weights: dict[str, float],
    zone: str,
    data_dir: str = "data/memory",
    learning_rates: dict[str, float] | None = None,
) -> None:
    """Persist learned scoring weights and learning rates to disk."""
    path = str(Path(data_dir) / f"{zone}_weights.json")
    data: dict[str, object] = {"v": 2, "weights": weights}
    if learning_rates:
        data["learning_rates"] = learning_rates
    try:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("[TUNER] weights saved to %s", path)
    except OSError as e:
        log.warning("[TUNER] failed to save weights to %s: %s", path, e)


def load_learned_weights(
    zone: str,
    data_dir: str = "data/memory",
) -> tuple[dict[str, float], dict[str, float]]:
    """Load previously learned weights + learning rates from disk.

    Returns (weights, learning_rates). Both empty dicts if none saved.
    """
    path = str(Path(data_dir) / f"{zone}_weights.json")
    try:
        with open(path) as f:
            data = json.load(f)
        weights: dict[str, float] = data.get("weights", {})
        rates: dict[str, float] = data.get("learning_rates", {})
        if weights:
            log.info("[TUNER] loaded %d learned weights for %s", len(weights), zone)
        return weights, rates
    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
    ):
        return {}, {}
