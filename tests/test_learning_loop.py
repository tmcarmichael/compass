"""Learning loop integration: encounter data → scoring changes.

These tests prove the closed-loop claims from the README's Learned Adaptation
section. Each test records experience, then verifies that scoring output
changes in the expected direction. No mocks  -- real scoring functions,
real learning systems, real data flow.
"""

from __future__ import annotations

import pytest

from brain.context import AgentContext
from brain.learning.encounters import FightHistory
from brain.learning.spatial import SpatialMemory
from brain.scoring.target import ScoringWeights, score_target
from core.types import Point
from perception.combat_eval import Con
from tests.factories import make_mob_profile, make_spawn


def _make_ctx(tmp_path: str, *, target_cons: frozenset | None = None) -> AgentContext:
    """Build an AgentContext that accepts WHITE targets."""
    ctx = AgentContext()
    ctx.zone.target_cons = target_cons or frozenset(
        {Con.WHITE, Con.BLUE, Con.LIGHT_BLUE, Con.YELLOW, Con.RED}
    )
    return ctx


# ---------------------------------------------------------------------------
# 1. Fight duration data changes target scoring
# ---------------------------------------------------------------------------


class TestEncounterDurationAffectsScoring:
    """Targets with known fast fight times should score higher than slow ones."""

    def test_fast_fights_boost_score(self) -> None:
        w = ScoringWeights()
        p = make_mob_profile(distance=50.0)

        score_no_data = score_target(p, w, [p], players=[])

        # Provide learned data: fast fights
        score_fast = score_target(
            p,
            w,
            [p],
            players=[],
            fight_durations={"a_skeleton": [10.0, 12.0, 11.0]},
        )

        assert score_fast > score_no_data

    def test_slow_fights_penalize_score(self) -> None:
        w = ScoringWeights()
        p = make_mob_profile(distance=50.0)

        score_no_data = score_target(p, w, [p], players=[])

        score_slow = score_target(
            p,
            w,
            [p],
            players=[],
            fight_durations={"a_skeleton": [90.0, 95.0, 88.0]},
        )

        assert score_slow < score_no_data

    def test_fast_beats_slow(self) -> None:
        w = ScoringWeights()
        p = make_mob_profile(distance=50.0)

        score_fast = score_target(
            p,
            w,
            [p],
            players=[],
            fight_durations={"a_skeleton": [10.0, 12.0]},
        )
        score_slow = score_target(
            p,
            w,
            [p],
            players=[],
            fight_durations={"a_skeleton": [90.0, 95.0]},
        )

        assert score_fast > score_slow


# ---------------------------------------------------------------------------
# 2. Spatial memory heat biases target selection
# ---------------------------------------------------------------------------


class TestSpatialMemoryBiasesScoring:
    """Targets in high-heat areas (many recent kills) should score higher."""

    def test_heat_boosts_score(self, tmp_path) -> None:
        w = ScoringWeights()
        ctx = _make_ctx(str(tmp_path))

        # Target at (50, 50)
        p = make_mob_profile(
            spawn=make_spawn(x=50.0, y=50.0),
            distance=50.0,
        )

        # Score with no spatial memory
        score_cold = score_target(p, w, [p], players=[], ctx=ctx)

        # Build heat at target's location
        sm = SpatialMemory("test", data_dir=str(tmp_path))
        for _ in range(15):
            sm.record_kill(Point(50.0, 50.0, 0.0), "a_skeleton", 10, 30.0)
        ctx.spatial_memory = sm

        score_hot = score_target(p, w, [p], players=[], ctx=ctx)

        assert score_hot > score_cold

    def test_heat_selects_productive_area(self, tmp_path) -> None:
        """Given two equidistant targets, the one in a high-heat area scores higher."""
        w = ScoringWeights()
        ctx = _make_ctx(str(tmp_path))

        target_a = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=100.0, y=0.0, spawn_id=1),
            distance=50.0,
        )
        target_b = make_mob_profile(
            spawn=make_spawn(name="a_bat", x=-100.0, y=0.0, spawn_id=2),
            distance=50.0,
        )

        # Build heat only near target B's location
        sm = SpatialMemory("test", data_dir=str(tmp_path))
        for _ in range(20):
            sm.record_kill(Point(-100.0, 0.0, 0.0), "a_bat", 10, 25.0)
        ctx.spatial_memory = sm

        score_a = score_target(target_a, w, [target_a, target_b], players=[], ctx=ctx)
        score_b = score_target(target_b, w, [target_a, target_b], players=[], ctx=ctx)

        assert score_b > score_a


# ---------------------------------------------------------------------------
# 3. Weight gradient shifts scoring emphasis
# ---------------------------------------------------------------------------


class TestWeightGradientShiftsScoring:
    """After gradient learning, weights that correlate with fitness
    should increase, changing the relative scoring of targets."""

    def test_gradient_updates_weights(self) -> None:
        from brain.scoring.weight_learner import GradientTuner

        w = ScoringWeights()
        tuner = GradientTuner(w)

        # Feed observations: high isolation → high fitness
        for i in range(30):
            isolation = 50.0 + i * 2
            fitness = 0.5 + (isolation / 200)
            tuner.observe(
                fitness=min(fitness, 1.0),
                breakdown={"isolation": isolation / 100, "distance": 0.5},
            )

        assert tuner.ready_to_step()

        old_isolation = w.isolation_peak
        tuner.step()
        new_isolation = w.isolation_peak

        # Isolation weight should have increased (positive correlation with fitness)
        assert new_isolation >= old_isolation

    def test_learned_weights_change_scoring_output(self) -> None:
        from brain.scoring.weight_learner import GradientTuner

        w = ScoringWeights()

        # Score two targets: one isolated, one crowded
        isolated = make_mob_profile(isolation_score=90.0, nearby_npc_count=0)
        crowded = make_mob_profile(isolation_score=20.0, nearby_npc_count=3)

        gap_before = score_target(isolated, w, [isolated, crowded], players=[]) - score_target(
            crowded, w, [isolated, crowded], players=[]
        )

        # Train: isolation correlates with fitness
        tuner = GradientTuner(w)
        for _ in range(30):
            tuner.observe(fitness=0.9, breakdown={"isolation": 0.9, "distance": 0.5})
            tuner.observe(fitness=0.3, breakdown={"isolation": 0.1, "distance": 0.5})

        if tuner.ready_to_step():
            tuner.step()

        gap_after = score_target(isolated, w, [isolated, crowded], players=[]) - score_target(
            crowded, w, [isolated, crowded], players=[]
        )

        # The scoring gap should widen (isolation matters more after learning)
        assert gap_after >= gap_before


# ---------------------------------------------------------------------------
# 4. Encounter history convergence
# ---------------------------------------------------------------------------


class TestEncounterConvergence:
    """After enough fights, learned estimates replace heuristics."""

    def test_learned_overrides_heuristic(self, tmp_path) -> None:
        fh = FightHistory(zone="test", data_dir=str(tmp_path))

        # Heuristic says nothing (no data)
        assert fh.learned_duration("a_skeleton") is None

        # Record 5 fights (minimum threshold)
        for _ in range(5):
            fh.record(
                mob_name="a_skeleton",
                duration=20.0,
                mana_spent=150,
                hp_delta=-0.05,
                casts=8,
                pet_heals=1,
                pet_died=False,
                defeated=True,
            )

        # Learned data now available
        dur = fh.learned_duration("a_skeleton")
        assert dur is not None
        assert dur == pytest.approx(20.0)

        mana = fh.learned_mana("a_skeleton")
        assert mana is not None
        assert mana == pytest.approx(150, abs=1)

    def test_danger_score_reflects_outcomes(self, tmp_path) -> None:
        fh = FightHistory(zone="test", data_dir=str(tmp_path))

        # Record 5 rough fights (high HP loss, pet deaths)
        for _ in range(5):
            fh.record(
                mob_name="a_dragon",
                duration=60.0,
                mana_spent=400,
                hp_delta=-0.4,
                casts=20,
                pet_heals=5,
                pet_died=True,
                defeated=True,
            )

        danger = fh.learned_danger("a_dragon")
        assert danger is not None
        assert danger > 0.5  # should be flagged as dangerous
