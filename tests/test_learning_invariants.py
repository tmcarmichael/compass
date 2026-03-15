"""Property-based invariant tests for the learning subsystems.

Verifies structural guarantees that must hold regardless of input:
- Gradient weight drift stays within ±20% of defaults
- Scorecard tuning parameters stay within declared bounds
- Encounter history respects MIN_FIGHTS_FOR_LEARNED and MAX_SAMPLES
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from brain.learning.encounters import (
    MAX_SAMPLES,
    MIN_FIGHTS_FOR_LEARNED,
    FightHistory,
)
from brain.learning.scorecard import TuningParams, evaluate_and_tune
from brain.scoring.target import ScoringWeights
from brain.scoring.weight_learner import MAX_DRIFT, GradientTuner

# ---------------------------------------------------------------------------
# Weight drift bounds
# ---------------------------------------------------------------------------


class TestWeightDriftBounds:
    """Scoring weights must stay within ±MAX_DRIFT of defaults."""

    @given(
        fitness_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=5,
            max_size=80,
        ),
    )
    @settings(max_examples=50)
    def test_weights_never_exceed_drift_bounds(self, fitness_values: list[float]) -> None:
        weights = ScoringWeights()
        tuner = GradientTuner(weights)

        # Build a plausible breakdown (all factors contribute equally)
        breakdown = {
            "con_pref": 50.0,
            "distance": 20.0,
            "isolation": 30.0,
            "social_add": -10.0,
            "camp_proximity": 15.0,
            "movement": 0.0,
            "caster": 0.0,
            "loot_value": 5.0,
            "heading": 0.0,
            "spatial_heat": 0.0,
            "learned_efficiency": 0.0,
            "resource": 0.0,
        }

        for fitness in fitness_values:
            tuner.observe(fitness, breakdown)
            if tuner.ready_to_step():
                tuner.step()

        # Verify every tunable field stays within bounds
        defaults = ScoringWeights()
        for field_name in tuner._defaults:
            default_val = float(getattr(defaults, field_name))
            if abs(default_val) < 1e-6:
                continue
            current_val = float(getattr(weights, field_name))
            lo = default_val * (1.0 - MAX_DRIFT)
            hi = default_val * (1.0 + MAX_DRIFT)
            assert lo - 0.01 <= current_val <= hi + 0.01, (
                f"Weight '{field_name}' drifted to {current_val:.4f}, "
                f"outside [{lo:.4f}, {hi:.4f}] (default={default_val})"
            )


# ---------------------------------------------------------------------------
# Scorecard tuning bounds
# ---------------------------------------------------------------------------


class TestScorecardTuningBounds:
    """Tuning parameters must stay within declared min/max regardless of input."""

    @given(
        defeat_rate=st.integers(min_value=0, max_value=100),
        survival=st.integers(min_value=0, max_value=100),
        pull_success=st.integers(min_value=0, max_value=100),
        mana_eff=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=200)
    def test_tuning_stays_within_bounds(
        self, defeat_rate: int, survival: int, pull_success: int, mana_eff: int
    ) -> None:
        scores = {
            "defeat_rate": defeat_rate,
            "survival": survival,
            "pull_success": pull_success,
            "mana_efficiency": mana_eff,
            "uptime": 50,
            "pathing": 50,
            "targeting": 50,
        }
        params = evaluate_and_tune(scores)

        assert TuningParams._ROAM_MIN <= params.roam_radius_mult <= TuningParams._ROAM_MAX
        assert TuningParams._SOCIAL_MIN <= params.social_npc_limit <= TuningParams._SOCIAL_MAX
        assert 0 <= params.mana_conserve_level <= 2

    @given(
        rounds=st.integers(min_value=1, max_value=20),
        defeat_rate=st.integers(min_value=0, max_value=100),
        survival=st.integers(min_value=0, max_value=100),
        pull_success=st.integers(min_value=0, max_value=100),
        mana_eff=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50)
    def test_repeated_tuning_stays_bounded(
        self, rounds: int, defeat_rate: int, survival: int, pull_success: int, mana_eff: int
    ) -> None:
        """Multiple consecutive tuning rounds can't escape bounds."""
        scores = {
            "defeat_rate": defeat_rate,
            "survival": survival,
            "pull_success": pull_success,
            "mana_efficiency": mana_eff,
            "uptime": 50,
            "pathing": 50,
            "targeting": 50,
        }
        params = TuningParams()
        for _ in range(rounds):
            params = evaluate_and_tune(scores, current=params)

        assert TuningParams._ROAM_MIN <= params.roam_radius_mult <= TuningParams._ROAM_MAX
        assert TuningParams._SOCIAL_MIN <= params.social_npc_limit <= TuningParams._SOCIAL_MAX
        assert 0 <= params.mana_conserve_level <= 2


# ---------------------------------------------------------------------------
# Encounter history invariants
# ---------------------------------------------------------------------------


class TestEncounterHistoryInvariants:
    """FightHistory respects MIN_FIGHTS and MAX_SAMPLES contracts."""

    def test_learned_duration_requires_min_fights(self, tmp_path: Path) -> None:
        """learned_duration returns None until MIN_FIGHTS_FOR_LEARNED records exist."""
        fh = FightHistory(zone="test", data_dir=str(tmp_path))
        for i in range(MIN_FIGHTS_FOR_LEARNED - 1):
            fh.record(
                mob_name="a_skeleton",
                duration=20.0 + i,
                mana_spent=50,
                hp_delta=-0.05,
                casts=2,
                pet_heals=0,
                pet_died=False,
                defeated=True,
            )
            assert fh.learned_duration("a_skeleton") is None, (
                f"learned_duration should be None with only {i + 1} fights"
            )

        # One more should cross the threshold
        fh.record(
            mob_name="a_skeleton",
            duration=25.0,
            mana_spent=50,
            hp_delta=-0.05,
            casts=2,
            pet_heals=0,
            pet_died=False,
            defeated=True,
        )
        assert fh.learned_duration("a_skeleton") is not None

    def test_rolling_window_never_exceeds_max_samples(self, tmp_path: Path) -> None:
        """Recording more than MAX_SAMPLES fights trims the oldest."""
        fh = FightHistory(zone="test", data_dir=str(tmp_path))
        for i in range(MAX_SAMPLES + 20):
            fh.record(
                mob_name="a_bat",
                duration=10.0 + (i * 0.1),
                mana_spent=20,
                hp_delta=-0.02,
                casts=1,
                pet_heals=0,
                pet_died=False,
                defeated=True,
            )
        records = fh._records.get("a_bat", [])
        assert len(records) <= MAX_SAMPLES, f"Expected <= {MAX_SAMPLES} records, got {len(records)}"

    def test_learned_data_overrides_after_threshold(self, tmp_path: Path) -> None:
        """After MIN_FIGHTS, learned data reflects actual outcomes."""
        fh = FightHistory(zone="test", data_dir=str(tmp_path))
        target_duration = 15.0
        for _ in range(MIN_FIGHTS_FOR_LEARNED + 5):
            fh.record(
                mob_name="a_moss_snake",
                duration=target_duration,
                mana_spent=30,
                hp_delta=-0.03,
                casts=1,
                pet_heals=0,
                pet_died=False,
                defeated=True,
            )
        learned = fh.learned_duration("a_moss_snake")
        assert learned is not None
        assert abs(learned - target_duration) < 1.0, (
            f"Learned duration {learned} should be close to {target_duration}"
        )
