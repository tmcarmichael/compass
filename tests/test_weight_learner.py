"""Tests for brain.scoring.weight_learner -- gradient weight tuning.

GradientTuner observes (fitness, score_breakdown) per encounter and adjusts
ScoringWeights via bounded gradient descent. Pearson correlation drives the
gradient direction.
"""

from __future__ import annotations

from brain.scoring.target import ScoringWeights
from brain.scoring.weight_learner import (
    FACTOR_WEIGHT_MAP,
    MAX_DRIFT,
    STEP_INTERVAL,
    GradientTuner,
)


def _make_tuner() -> tuple[GradientTuner, ScoringWeights]:
    w = ScoringWeights()
    return GradientTuner(w), w


def _breakdown(factor: str, value: float) -> dict[str, float]:
    """Build a minimal breakdown dict with one factor set."""
    return {factor: value}


class TestGradientTuner:
    def test_initial_no_observations(self) -> None:
        tuner, _ = _make_tuner()
        assert tuner.steps == 0

    def test_observe_accumulates(self) -> None:
        tuner, _ = _make_tuner()
        tuner.observe(0.8, {"con_pref": 50.0})
        tuner.observe(0.3, {"con_pref": 20.0})
        assert not tuner.ready_to_step()

    def test_not_ready_below_threshold(self) -> None:
        tuner, _ = _make_tuner()
        for _ in range(STEP_INTERVAL - 1):
            tuner.observe(0.5, {"con_pref": 30.0})
        assert not tuner.ready_to_step()

    def test_ready_at_threshold(self) -> None:
        tuner, _ = _make_tuner()
        for _ in range(STEP_INTERVAL):
            tuner.observe(0.5, {"con_pref": 30.0})
        assert tuner.ready_to_step()

    def test_step_returns_deltas(self) -> None:
        tuner, _ = _make_tuner()
        # Create enough observations with varying con_pref and fitness
        for i in range(STEP_INTERVAL):
            fitness = 0.1 + (i / STEP_INTERVAL) * 0.8
            tuner.observe(fitness, {"con_pref": float(10 + i * 5)})
        deltas = tuner.step()
        # step() returns a dict (may be empty if changes are too small)
        assert isinstance(deltas, dict)
        assert tuner.steps == 1

    def test_weights_bounded(self) -> None:
        """After step, no tunable weight exceeds +/-20% of its default."""
        tuner, w = _make_tuner()
        defaults = ScoringWeights()
        # Push strongly biased observations
        for i in range(STEP_INTERVAL * 3):
            tuner.observe(1.0 if i % 2 == 0 else 0.0, {"con_pref": float(100 * (i % 2))})
            if tuner.ready_to_step():
                tuner.step()
        # Verify bounds on all tunable fields
        for factor_fields in FACTOR_WEIGHT_MAP.values():
            for field_name in factor_fields:
                default = float(getattr(defaults, field_name))
                if abs(default) < 1e-6:
                    continue  # skip zero-default fields
                current = float(getattr(w, field_name))
                lo = default * (1.0 - MAX_DRIFT)
                hi = default * (1.0 + MAX_DRIFT)
                assert lo - 0.01 <= current <= hi + 0.01, f"{field_name}: {current} not in [{lo}, {hi}]"

    def test_positive_correlation_increases_weight(self) -> None:
        """When high factor contribution correlates with high fitness, weight goes up."""
        tuner, w = _make_tuner()
        # Observations: high con_pref -> high fitness
        for i in range(STEP_INTERVAL):
            con_val = 50.0 + i * 3.0
            fitness = 0.5 + i * 0.03
            tuner.observe(fitness, {"con_pref": con_val})
        deltas = tuner.step()
        # con_pref maps to con_white (among others) -- should increase or stay
        if "con_white" in deltas:
            assert deltas["con_white"] > 0

    def test_negative_correlation_decreases_weight(self) -> None:
        """When high factor contribution correlates with low fitness, weight goes down."""
        tuner, w = _make_tuner()
        # Observations: high distance factor → low fitness (anti-correlated)
        for i in range(STEP_INTERVAL):
            dist_val = 10.0 + i * 2.0
            fitness = 0.8 - i * 0.04
            tuner.observe(fitness, {"distance": dist_val})
        deltas = tuner.step()
        # distance maps to dist_ideal, dist_width, dist_peak
        for field in ("dist_ideal", "dist_width", "dist_peak"):
            if field in deltas:
                assert deltas[field] < 0

    def test_weight_snapshot_roundtrip(self) -> None:
        tuner, w = _make_tuner()
        snap = tuner.get_weight_snapshot()
        # Create a fresh weights object and tuner, load the snapshot
        w2 = ScoringWeights()
        tuner2 = GradientTuner(w2)
        applied = tuner2.load_learned_weights(snap)
        assert applied == len(snap)
        # Verify all values match
        for field_name, value in snap.items():
            assert float(getattr(w2, field_name)) == value

    def test_learning_rate_persistence(self) -> None:
        tuner, _ = _make_tuner()
        # Perform a step to populate learning rates
        for i in range(STEP_INTERVAL):
            tuner.observe(float(i) / STEP_INTERVAL, {"con_pref": float(20 + i)})
        tuner.step()
        rates = tuner.get_learning_rates()
        # Load into a new tuner
        tuner2, _ = _make_tuner()
        tuner2.load_learning_rates(rates)
        assert tuner2.get_learning_rates() == rates

    def test_step_resets_counter(self) -> None:
        tuner, _ = _make_tuner()
        for _ in range(STEP_INTERVAL):
            tuner.observe(0.5, {"con_pref": 30.0})
        assert tuner.ready_to_step()
        tuner.step()
        assert not tuner.ready_to_step()
