"""Tests for brain.learning.scorecard -- encounter fitness and session tuning.

encounter_fitness scores individual encounters (training signal for weight
learning). evaluate_and_tune adjusts agent parameters from scorecard grades.
TuningParams persist to disk for cross-session learning.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from brain.learning.scorecard import (
    TuningParams,
    _bar,
    _clamp,
    _clamp_tuning,
    _score_direct,
    _score_inverse,
    _score_pct,
    apply_tuning,
    compute_scorecard,
    encounter_fitness,
    evaluate_and_tune,
    format_scorecard,
    load_tuning,
    save_tuning,
)


class TestEncounterFitness:
    def test_fitness_defeated_positive(self) -> None:
        f = encounter_fitness(
            duration=20.0,
            mana_spent=100,
            max_mana=500,
            hp_delta=-0.1,
            defeated=True,
            expected_duration=25.0,
        )
        assert f > 0.0

    def test_fitness_fled_zero(self) -> None:
        f = encounter_fitness(
            duration=10.0,
            mana_spent=50,
            max_mana=500,
            hp_delta=-0.2,
            defeated=False,
            expected_duration=25.0,
        )
        assert f == 0.0

    def test_fitness_fast_bonus(self) -> None:
        """Faster-than-expected encounter scores higher."""
        fast = encounter_fitness(
            duration=10.0,
            mana_spent=100,
            max_mana=500,
            hp_delta=-0.1,
            defeated=True,
            expected_duration=30.0,
        )
        slow = encounter_fitness(
            duration=50.0,
            mana_spent=100,
            max_mana=500,
            hp_delta=-0.1,
            defeated=True,
            expected_duration=30.0,
        )
        assert fast > slow

    def test_fitness_slow_penalty(self) -> None:
        """Slower encounter gets a lower duration component."""
        slow = encounter_fitness(
            duration=60.0,
            mana_spent=100,
            max_mana=500,
            hp_delta=-0.1,
            defeated=True,
            expected_duration=20.0,
        )
        # The duration ratio min(20/60, 1.0) = 0.333 contributes 0.2 * 0.333
        assert slow < 0.9

    def test_fitness_bounded(self) -> None:
        """Fitness is always 0.0-1.0."""
        # Best case
        best = encounter_fitness(
            duration=5.0,
            mana_spent=0,
            max_mana=500,
            hp_delta=0.0,
            defeated=True,
            expected_duration=30.0,
        )
        assert 0.0 <= best <= 1.0
        # Worst defeated case
        worst = encounter_fitness(
            duration=100.0,
            mana_spent=500,
            max_mana=500,
            hp_delta=-1.0,
            defeated=True,
            expected_duration=10.0,
        )
        assert 0.0 <= worst <= 1.0


class TestEvaluateAndTune:
    def test_tuning_low_kills_expands_roam(self) -> None:
        scores = {"defeat_rate": 20, "survival": 80, "pull_success": 70, "mana_efficiency": 50}
        p = evaluate_and_tune(scores)
        assert p.roam_radius_mult > 1.0

    def test_tuning_low_survival_tightens(self) -> None:
        scores = {"defeat_rate": 50, "survival": 20, "pull_success": 70, "mana_efficiency": 50}
        p = evaluate_and_tune(scores, TuningParams())
        assert p.roam_radius_mult < 1.0 or p.social_npc_limit < 3

    def test_tuning_bounds_enforced(self) -> None:
        """_clamp_tuning keeps parameters within their defined ranges."""
        p = TuningParams(roam_radius_mult=10.0, social_npc_limit=100, mana_conserve_level=99)
        p = _clamp_tuning(p)
        assert p.roam_radius_mult <= TuningParams._ROAM_MAX
        assert p.social_npc_limit <= TuningParams._SOCIAL_MAX
        assert p.mana_conserve_level <= 2

    def test_tuning_defaults(self) -> None:
        p = TuningParams()
        assert p.roam_radius_mult == 1.0
        assert p.social_npc_limit == 3
        assert p.mana_conserve_level == 0

    def test_tuning_persistence_roundtrip(self, tmp_path: Path) -> None:
        p = TuningParams(roam_radius_mult=1.3, social_npc_limit=4, mana_conserve_level=1)
        save_tuning(p, "testzone", data_dir=str(tmp_path))
        loaded = load_tuning("testzone", data_dir=str(tmp_path))
        assert loaded.roam_radius_mult == pytest.approx(1.3)
        assert loaded.social_npc_limit == 4
        assert loaded.mana_conserve_level == 1

    def test_tuning_low_pull_success_tightens_social(self) -> None:
        scores = {"defeat_rate": 50, "survival": 80, "pull_success": 30, "mana_efficiency": 50}
        p = evaluate_and_tune(scores, TuningParams(social_npc_limit=3))
        assert p.social_npc_limit < 3

    def test_tuning_high_performance_relaxes(self) -> None:
        scores = {"defeat_rate": 50, "survival": 90, "pull_success": 90, "mana_efficiency": 50}
        p = evaluate_and_tune(scores, TuningParams(social_npc_limit=3))
        assert p.social_npc_limit >= 3

    def test_tuning_poor_mana_conserves(self) -> None:
        scores = {"defeat_rate": 50, "survival": 80, "pull_success": 70, "mana_efficiency": 20}
        p = evaluate_and_tune(scores, TuningParams(mana_conserve_level=0))
        assert p.mana_conserve_level > 0

    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        loaded = load_tuning("nonexistent", data_dir=str(tmp_path))
        assert loaded.roam_radius_mult == 1.0
        assert loaded.social_npc_limit == 3

    def test_clamp_lower_bounds(self) -> None:
        p = TuningParams(roam_radius_mult=0.0, social_npc_limit=0, mana_conserve_level=-5)
        p = _clamp_tuning(p)
        assert p.roam_radius_mult >= TuningParams._ROAM_MIN
        assert p.social_npc_limit >= TuningParams._SOCIAL_MIN
        assert p.mana_conserve_level >= 0

    def test_tuning_high_defeat_rate_tightens_roam(self) -> None:
        """High defeat_rate (>80) tightens roam radius."""
        scores = {"defeat_rate": 90, "survival": 80, "pull_success": 70, "mana_efficiency": 50}
        p = evaluate_and_tune(scores, TuningParams(roam_radius_mult=1.0))
        assert p.roam_radius_mult < 1.0

    def test_tuning_good_mana_loosens(self) -> None:
        """High mana_efficiency (>80) loosens conservation level."""
        scores = {"defeat_rate": 50, "survival": 80, "pull_success": 70, "mana_efficiency": 90}
        p = evaluate_and_tune(scores, TuningParams(mana_conserve_level=1))
        assert p.mana_conserve_level < 1

    def test_tuning_no_changes_mid_scores(self) -> None:
        """Mid-range scores produce no changes."""
        scores = {"defeat_rate": 50, "survival": 50, "pull_success": 60, "mana_efficiency": 50}
        p = evaluate_and_tune(scores, TuningParams())
        assert p.roam_radius_mult == 1.0
        assert p.social_npc_limit == 3
        assert p.mana_conserve_level == 0

    def test_survival_emergency(self) -> None:
        """Survival < 30 triggers emergency tightening."""
        scores = {"defeat_rate": 50, "survival": 10, "pull_success": 70, "mana_efficiency": 50}
        p = evaluate_and_tune(scores, TuningParams(roam_radius_mult=1.0, social_npc_limit=3))
        # Should tighten both roam and social
        assert p.roam_radius_mult < 1.0 or p.social_npc_limit < 3


# ---------------------------------------------------------------------------
# Scoring helper functions
# ---------------------------------------------------------------------------


class TestScoringHelpers:
    def test_clamp_within_range(self) -> None:
        assert _clamp(50.0) == 50.0

    def test_clamp_above_max(self) -> None:
        assert _clamp(150.0) == 100.0

    def test_clamp_below_min(self) -> None:
        assert _clamp(-10.0) == 0.0

    def test_score_inverse_perfect(self) -> None:
        assert _score_inverse(0.0, 0.0, 10.0) == 100

    def test_score_inverse_failing(self) -> None:
        assert _score_inverse(10.0, 0.0, 10.0) == 0

    def test_score_inverse_midpoint(self) -> None:
        result = _score_inverse(5.0, 0.0, 10.0)
        assert result == 50

    def test_score_direct_perfect(self) -> None:
        assert _score_direct(20.0, 5.0, 20.0) == 100

    def test_score_direct_failing(self) -> None:
        assert _score_direct(5.0, 5.0, 20.0) == 0

    def test_score_direct_midpoint(self) -> None:
        result = _score_direct(12.5, 5.0, 20.0)
        assert result == 50

    def test_score_pct_full(self) -> None:
        assert _score_pct(1.0) == 100

    def test_score_pct_zero(self) -> None:
        assert _score_pct(0.0) == 0

    def test_score_pct_half(self) -> None:
        assert _score_pct(0.5) == 50


class TestBar:
    def test_bar_100(self) -> None:
        result = _bar(100)
        assert "#" * 20 in result

    def test_bar_0(self) -> None:
        result = _bar(0)
        assert "." * 20 in result

    def test_bar_50(self) -> None:
        result = _bar(50)
        assert "#" * 10 in result


# ---------------------------------------------------------------------------
# compute_scorecard
# ---------------------------------------------------------------------------


def _make_scorecard_ctx(
    *,
    defeats: int = 10,
    deaths: int = 0,
    flee_count: int = 0,
    total_casts: int = 30,
    session_start: float = 0.0,
    pulls: int = 20,
    pull_fails: int = 2,
    acquire_tab_totals: list | None = None,
    routine_time: dict | None = None,
    total_combat_time: float = 1000.0,
) -> SimpleNamespace:
    """Minimal ctx for compute_scorecard."""
    import time

    if session_start == 0.0:
        session_start = time.time() - 3600.0  # 1 hour ago

    return SimpleNamespace(
        metrics=SimpleNamespace(
            session_start=session_start,
            routine_counts=defaultdict(int, {"PULL": pulls}),
            routine_failures=defaultdict(int, {"PULL": pull_fails}),
            acquire_tab_totals=acquire_tab_totals or [],
            flee_count=flee_count,
            total_casts=total_casts,
            total_combat_time=total_combat_time,
            routine_time=routine_time or {"PULL": 500.0, "IN_COMBAT": 1500.0},
        ),
        defeat_tracker=SimpleNamespace(defeats=defeats),
        player=SimpleNamespace(deaths=deaths),
        lock=threading.Lock(),
    )


class TestComputeScorecard:
    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_basic_scorecard(self, _mock_stuck: MagicMock) -> None:
        ctx = _make_scorecard_ctx()
        scores = compute_scorecard(ctx)
        assert "overall" in scores
        assert "grade" in scores
        assert 0 <= scores["overall"] <= 100

    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_all_grades(self, _mock_stuck: MagicMock) -> None:
        """Verify grade assignment thresholds."""

        # Grade A: high kills, no deaths, good uptime
        ctx = _make_scorecard_ctx(
            defeats=20,
            deaths=0,
            flee_count=0,
            total_casts=30,
            pulls=20,
            pull_fails=0,
            acquire_tab_totals=[1.0] * 20,
            total_combat_time=2160.0,
            routine_time={"PULL": 400.0, "IN_COMBAT": 2160.0},
        )
        scores = compute_scorecard(ctx)
        assert scores["grade"] in ("A", "B")

    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_no_pull_data_default(self, _mock_stuck: MagicMock) -> None:
        ctx = _make_scorecard_ctx(pulls=0, pull_fails=0)
        scores = compute_scorecard(ctx)
        assert scores["pull_success"] == 50

    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_no_cast_data_default(self, _mock_stuck: MagicMock) -> None:
        ctx = _make_scorecard_ctx(defeats=0, total_casts=0)
        scores = compute_scorecard(ctx)
        assert scores["mana_efficiency"] == 50

    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_no_acquire_data_default(self, _mock_stuck: MagicMock) -> None:
        ctx = _make_scorecard_ctx(acquire_tab_totals=[])
        scores = compute_scorecard(ctx)
        assert scores["targeting"] == 50

    @patch("nav.movement.get_stuck_event_count", return_value=15)
    def test_high_stuck_lowers_pathing(self, _mock_stuck: MagicMock) -> None:
        ctx = _make_scorecard_ctx()
        scores = compute_scorecard(ctx)
        assert scores["pathing"] == 0

    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_raw_stats_present(self, _mock_stuck: MagicMock) -> None:
        ctx = _make_scorecard_ctx(defeats=5, deaths=2, flee_count=3)
        scores = compute_scorecard(ctx)
        assert scores["_kills"] == 5
        assert scores["_deaths"] == 2
        assert scores["_flees"] == 3
        assert scores["_hours"] > 0

    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_empty_routine_time_uses_elapsed(self, _mock_stuck: MagicMock) -> None:
        ctx = _make_scorecard_ctx(routine_time={}, total_combat_time=0.0)
        scores = compute_scorecard(ctx)
        # Should not crash; uptime falls back to elapsed
        assert "uptime" in scores


# ---------------------------------------------------------------------------
# format_scorecard
# ---------------------------------------------------------------------------


class TestFormatScorecard:
    def test_format_output(self) -> None:
        scores = {
            "defeat_rate": 80,
            "survival": 90,
            "pull_success": 70,
            "uptime": 60,
            "pathing": 100,
            "targeting": 50,
            "mana_efficiency": 75,
            "overall": 78,
            "grade": "C",
            "_hours": 1.5,
            "_kills": 20,
            "_deaths": 1,
            "_flees": 2,
            "_stuck": 0,
        }
        result = format_scorecard(scores)
        assert "SESSION SCORECARD" in result
        assert "Grade: C" in result
        assert "Throughput" in result
        assert "Survival" in result
        assert "1.5hr" in result


# ---------------------------------------------------------------------------
# encounter_fitness edge cases
# ---------------------------------------------------------------------------


class TestEncounterFitnessEdgeCases:
    def test_no_expected_duration(self) -> None:
        """expected_duration=0 -> average bonus (0.1) for duration component."""
        f = encounter_fitness(
            duration=10.0,
            mana_spent=100,
            max_mana=500,
            hp_delta=-0.1,
            defeated=True,
            expected_duration=0.0,
        )
        assert 0.5 < f <= 1.0

    def test_zero_max_mana(self) -> None:
        """max_mana=0 -> average bonus (0.1) for resource component."""
        f = encounter_fitness(
            duration=10.0,
            mana_spent=100,
            max_mana=0,
            hp_delta=-0.1,
            defeated=True,
            expected_duration=20.0,
        )
        assert 0.5 < f <= 1.0

    def test_heavy_hp_loss(self) -> None:
        """hp_delta = -1.0 means 100% HP lost -> safety component is 0."""
        f = encounter_fitness(
            duration=10.0,
            mana_spent=100,
            max_mana=500,
            hp_delta=-1.0,
            defeated=True,
            expected_duration=20.0,
        )
        # Safety: 0.1 * max(0, 1 + (-1.0)) = 0
        assert f < 0.95


# ---------------------------------------------------------------------------
# apply_tuning
# ---------------------------------------------------------------------------


class TestApplyTuning:
    def test_sets_base_roam_radius(self) -> None:
        params = TuningParams(roam_radius_mult=1.2)
        ctx = SimpleNamespace(
            camp=SimpleNamespace(
                base_roam_radius=0.0,
                roam_radius=100.0,
            ),
            world=None,
        )
        apply_tuning(params, ctx)
        assert ctx.camp.base_roam_radius == 100.0
        assert ctx.camp.roam_radius == pytest.approx(120.0)

    def test_updates_social_limit(self) -> None:
        params = TuningParams(social_npc_limit=4)
        ctx = SimpleNamespace(
            camp=SimpleNamespace(
                base_roam_radius=100.0,
                roam_radius=100.0,
            ),
            world=SimpleNamespace(
                _weights=SimpleNamespace(social_npc_hard_limit=3),
            ),
        )
        apply_tuning(params, ctx)
        assert ctx.world._weights.social_npc_hard_limit == 4

    def test_no_world_no_crash(self) -> None:
        params = TuningParams()
        ctx = SimpleNamespace(
            camp=SimpleNamespace(
                base_roam_radius=100.0,
                roam_radius=100.0,
            ),
            world=None,
        )
        apply_tuning(params, ctx)  # should not crash

    def test_small_roam_change_not_applied(self) -> None:
        """Roam radius change < 1.0 unit is not applied."""
        params = TuningParams(roam_radius_mult=1.005)
        ctx = SimpleNamespace(
            camp=SimpleNamespace(
                base_roam_radius=100.0,
                roam_radius=100.0,
            ),
            world=None,
        )
        apply_tuning(params, ctx)
        assert ctx.camp.roam_radius == 100.0  # unchanged
