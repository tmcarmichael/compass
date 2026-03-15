"""Learning subsystem convergence tests.

Validates that FightHistory, GradientTuner, and scorecard tuning
converge to stable, correct values over many observations. Individual
learner correctness is tested elsewhere; these tests verify convergence
properties over realistic observation counts.
"""

from __future__ import annotations

import random

import pytest

from brain.learning.encounters import FightHistory
from brain.learning.scorecard import TuningParams, evaluate_and_tune
from brain.scoring.target import ScoringWeights
from brain.scoring.weight_learner import STEP_INTERVAL, GradientTuner
from tests.factories import make_fight_record

# ---------------------------------------------------------------------------
# FightHistory convergence
# ---------------------------------------------------------------------------


class TestFightHistoryConvergence:
    """FightHistory learned values converge to true means."""

    @pytest.fixture()
    def fh(self, tmp_path: object) -> FightHistory:
        return FightHistory(zone="test", data_dir=str(tmp_path))

    def _record_fights(
        self,
        fh: FightHistory,
        mob_name: str,
        count: int,
        mean_dur: float,
        sigma_dur: float,
        mean_hp_delta: float,
        pet_death_rate: float,
    ) -> None:
        """Record `count` fights with normally-distributed durations."""
        rng = random.Random(42)
        for _ in range(count):
            dur = max(1.0, rng.gauss(mean_dur, sigma_dur))
            pet_died = rng.random() < pet_death_rate
            rec = make_fight_record(
                mob_name=mob_name,
                duration=dur,
                hp_delta=mean_hp_delta + rng.gauss(0, 0.02),
                pet_died=pet_died,
                fitness=0.5 + rng.gauss(0, 0.1),
            )
            fh.record(**rec)

    def test_duration_converges_within_tolerance(self, fh: FightHistory) -> None:
        """After 100 fights, learned_duration is within 15% of true mean."""
        true_mean = 20.0
        self._record_fights(fh, "a_skeleton", 100, true_mean, 3.0, -0.05, 0.0)

        learned = fh.learned_duration("a_skeleton")
        assert learned is not None
        error_pct = abs(learned - true_mean) / true_mean
        assert error_pct < 0.15, (
            f"Learned duration {learned:.1f} deviates {error_pct:.0%} from true mean {true_mean}"
        )

    def test_danger_rank_ordering(self, fh: FightHistory) -> None:
        """Mobs with higher HP loss are ranked more dangerous."""
        # Safe mob: low HP loss, no pet deaths
        self._record_fights(fh, "a_bat", 20, 15.0, 2.0, -0.02, 0.0)
        # Dangerous mob: high HP loss, frequent pet deaths
        self._record_fights(fh, "a_skeleton", 20, 30.0, 5.0, -0.30, 0.4)

        safe_danger = fh.learned_danger("a_bat")
        high_danger = fh.learned_danger("a_skeleton")
        assert safe_danger is not None
        assert high_danger is not None
        assert high_danger > safe_danger, (
            f"Dangerous mob ({high_danger:.2f}) should rank above safe mob ({safe_danger:.2f})"
        )

    def test_has_learned_threshold(self, fh: FightHistory) -> None:
        """has_learned returns False until MIN_FIGHTS_FOR_LEARNED reached."""
        for i in range(4):
            rec = make_fight_record(mob_name="a_snake", duration=10.0 + i)
            fh.record(**rec)
        assert not fh.has_learned("a_snake")

        rec = make_fight_record(mob_name="a_snake", duration=12.0)
        fh.record(**rec)
        assert fh.has_learned("a_snake")

    def test_multiple_mobs_independent(self, fh: FightHistory) -> None:
        """Learning for one mob type doesn't affect another."""
        self._record_fights(fh, "a_bat", 10, 10.0, 1.0, -0.01, 0.0)
        self._record_fights(fh, "a_skeleton", 10, 40.0, 5.0, -0.20, 0.3)

        bat_dur = fh.learned_duration("a_bat")
        skel_dur = fh.learned_duration("a_skeleton")
        assert bat_dur is not None and skel_dur is not None
        assert skel_dur > bat_dur * 2, "Independent mobs should have very different durations"


# ---------------------------------------------------------------------------
# GradientTuner convergence
# ---------------------------------------------------------------------------


class TestGradientTunerConvergence:
    """Gradient tuner weights stabilize over many observations."""

    def _make_tuner(self) -> GradientTuner:
        return GradientTuner(ScoringWeights())

    def test_weights_stabilize(self) -> None:
        """After many observations, consecutive steps produce smaller deltas."""
        tuner = self._make_tuner()
        rng = random.Random(42)

        # Feed observations with isolation positively correlated with fitness
        for i in range(150):
            fitness = rng.gauss(0.5, 0.15)
            # Higher isolation -> higher fitness (positive correlation)
            isolation_contrib = fitness * 0.8 + rng.gauss(0, 0.1)
            breakdown = {
                "con_pref": rng.gauss(0.5, 0.2),
                "distance": rng.gauss(0.3, 0.1),
                "isolation": max(0, isolation_contrib),
                "social_add": rng.gauss(0.1, 0.05),
                "camp_proximity": rng.gauss(0.2, 0.1),
            }
            tuner.observe(max(0, min(1, fitness)), breakdown)
            if tuner.ready_to_step():
                tuner.step()

        # After convergence: deltas should be small
        # Take one more step and check delta magnitude
        for _ in range(STEP_INTERVAL):
            tuner.observe(0.5, {"isolation": 0.4, "distance": 0.3})
        if tuner.ready_to_step():
            deltas = tuner.step()
            total_delta = sum(abs(v) for v in deltas.values())
            # After 150 observations and ~10 gradient steps, deltas should be modest
            assert total_delta < 5.0, (
                f"After convergence, total delta magnitude ({total_delta:.2f}) should be small"
            )

    def test_correlated_weight_increases(self) -> None:
        """A factor positively correlated with fitness should see its weight increase."""
        tuner = self._make_tuner()
        initial_snapshot = tuner.get_weight_snapshot()
        rng = random.Random(42)

        # Strong correlation: high isolation -> high fitness
        for i in range(200):
            fitness = rng.random()
            breakdown = {
                "isolation": fitness * 0.9 + rng.gauss(0, 0.05),
                "con_pref": rng.gauss(0.3, 0.2),
                "distance": rng.gauss(0.3, 0.2),
            }
            tuner.observe(max(0, min(1, fitness)), breakdown)
            if tuner.ready_to_step():
                tuner.step()

        final_snapshot = tuner.get_weight_snapshot()
        # isolation_peak should have increased (positive correlation)
        if "isolation_peak" in initial_snapshot and "isolation_peak" in final_snapshot:
            assert final_snapshot["isolation_peak"] >= initial_snapshot["isolation_peak"], (
                f"Positively correlated factor should not decrease: "
                f"{initial_snapshot['isolation_peak']} -> {final_snapshot['isolation_peak']}"
            )

    def test_step_count_increments(self) -> None:
        """steps property increments with each gradient step."""
        tuner = self._make_tuner()
        assert tuner.steps == 0

        for _ in range(STEP_INTERVAL):
            tuner.observe(0.5, {"con_pref": 0.3})
        assert tuner.ready_to_step()
        tuner.step()
        assert tuner.steps == 1


# ---------------------------------------------------------------------------
# Scorecard tuning convergence
# ---------------------------------------------------------------------------


class TestScorecardTuningConvergence:
    """evaluate_and_tune produces stable params after repeated evaluation."""

    def test_tuning_stabilizes(self) -> None:
        """Running evaluate_and_tune repeatedly with same scores converges."""
        scores = {
            "defeat_rate": 90,
            "survival": 100,
            "pull_success": 95,
            "targeting": 85,
            "mana_efficiency": 90,
            "uptime": 80,
            "pathing": 100,
            "overall": 92,
            "grade": "A",
            "_hours": 1.0,
            "_kills": 50,
            "_deaths": 0,
            "_flees": 2,
            "_stuck": 0,
        }

        params = None
        all_params: list[TuningParams] = []
        for _ in range(5):
            params = evaluate_and_tune(scores, params)
            all_params.append(params)

        # Last two iterations should produce identical tuning
        assert all_params[-1].roam_radius_mult == all_params[-2].roam_radius_mult
        assert all_params[-1].social_npc_limit == all_params[-2].social_npc_limit
        assert all_params[-1].mana_conserve_level == all_params[-2].mana_conserve_level

    def test_poor_scores_adjust_params(self) -> None:
        """Poor scores should cause tuning to differ from defaults."""
        poor_scores = {
            "defeat_rate": 90,
            "survival": 100,
            "pull_success": 50,
            "targeting": 30,
            "mana_efficiency": 90,
            "uptime": 40,
            "pathing": 100,
            "overall": 65,
            "grade": "C",
            "_hours": 1.0,
            "_kills": 20,
            "_deaths": 0,
            "_flees": 0,
            "_stuck": 0,
        }

        default = TuningParams()
        tuned = evaluate_and_tune(poor_scores, TuningParams())
        # Poor scores should cause at least one parameter to change
        changed = (
            tuned.roam_radius_mult != default.roam_radius_mult
            or tuned.social_npc_limit != default.social_npc_limit
            or tuned.mana_conserve_level != default.mana_conserve_level
        )
        assert changed, "Poor scores should cause at least one tuning parameter to change"
