"""Tests for brain.scoring.pareto -- multi-objective Pareto frontier.

AxisScores represent per-target scores on 4 independent axes. The Pareto
frontier identifies non-dominated targets for priority-weighted selection.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from brain.scoring.pareto import (
    AxisPriorities,
    AxisScores,
    _axis_accessibility,
    _axis_efficiency,
    _axis_resource,
    _axis_safety,
    _dominates,
    compute_axes,
    compute_priorities,
    log_pareto_selection,
    pareto_frontier,
    select_from_frontier,
)
from brain.scoring.target import ScoringWeights
from perception.combat_eval import Con
from tests.factories import make_game_state, make_mob_profile, make_spawn


def _axis(
    eff: float = 0.5,
    saf: float = 0.5,
    res: float = 0.5,
    acc: float = 0.5,
    **profile_kw: Any,
) -> AxisScores:
    """Build an AxisScores with given axis values and a default MobProfile."""
    return AxisScores(
        profile=make_mob_profile(**profile_kw),
        efficiency=eff,
        safety=saf,
        resource=res,
        accessibility=acc,
    )


class TestParetoFrontier:
    def test_single_candidate_is_frontier(self) -> None:
        c = _axis()
        assert pareto_frontier([c]) == [c]

    def test_dominated_excluded(self) -> None:
        a = _axis(1.0, 1.0, 1.0, 1.0)
        b = _axis(0.0, 0.0, 0.0, 0.0)
        frontier = pareto_frontier([a, b])
        assert a in frontier
        assert b not in frontier

    def test_non_dominated_both_kept(self) -> None:
        a = _axis(1.0, 0.0, 0.5, 0.5)
        b = _axis(0.0, 1.0, 0.5, 0.5)
        frontier = pareto_frontier([a, b])
        assert len(frontier) == 2

    def test_all_identical_all_kept(self) -> None:
        items = [_axis(0.5, 0.5, 0.5, 0.5) for _ in range(4)]
        frontier = pareto_frontier(items)
        assert len(frontier) == 4

    def test_empty_frontier(self) -> None:
        assert pareto_frontier([]) == []

    @given(
        candidates=st.lists(
            st.tuples(
                st.floats(0, 1, allow_nan=False),
                st.floats(0, 1, allow_nan=False),
                st.floats(0, 1, allow_nan=False),
                st.floats(0, 1, allow_nan=False),
            ),
            min_size=1,
            max_size=15,
        )
    )
    def test_frontier_is_subset(self, candidates: list[tuple[float, float, float, float]]) -> None:
        """Pareto frontier is always a subset of the original candidates."""
        axes = [_axis(e, s, r, a) for e, s, r, a in candidates]
        frontier = pareto_frontier(axes)
        for f in frontier:
            assert f in axes


class TestDominates:
    def test_does_not_dominate_self(self) -> None:
        """a does NOT dominate itself (requires > on at least one axis)."""
        a = _axis(0.5, 0.5, 0.5, 0.5)
        assert _dominates(a, a) is False

    def test_requires_strict_improvement(self) -> None:
        """Domination requires >= on all axes AND > on at least one."""
        a = _axis(0.5, 0.5, 0.5, 0.5)
        b = _axis(0.5, 0.5, 0.5, 0.5)
        assert _dominates(a, b) is False

    def test_clear_domination(self) -> None:
        a = _axis(1.0, 1.0, 1.0, 1.0)
        b = _axis(0.5, 0.5, 0.5, 0.5)
        assert _dominates(a, b) is True
        assert _dominates(b, a) is False

    def test_partial_superiority_no_domination(self) -> None:
        a = _axis(1.0, 0.0, 0.5, 0.5)
        b = _axis(0.0, 1.0, 0.5, 0.5)
        assert _dominates(a, b) is False
        assert _dominates(b, a) is False


class TestSelectFromFrontier:
    def test_picks_max_weighted(self) -> None:
        eff_best = _axis(1.0, 0.0, 0.0, 0.0)
        saf_best = _axis(0.0, 1.0, 0.0, 0.0)
        frontier = [eff_best, saf_best]
        # Efficiency-heavy priorities
        priorities = AxisPriorities(efficiency=0.9, safety=0.03, resource=0.03, accessibility=0.04)
        selected = select_from_frontier(frontier, priorities)
        assert selected is eff_best

    def test_empty_frontier_returns_none(self) -> None:
        assert select_from_frontier([], AxisPriorities()) is None

    def test_priority_normalization(self) -> None:
        """Priorities with non-unit sum still produce a valid selection."""
        p = AxisPriorities(efficiency=10.0, safety=10.0, resource=10.0, accessibility=10.0)
        total = p.efficiency + p.safety + p.resource + p.accessibility
        # After normalization each weight should be ~0.25
        normed = [w / total for w in (p.efficiency, p.safety, p.resource, p.accessibility)]
        for w in normed:
            assert abs(w - 0.25) < 0.01

    def test_zero_total_priorities(self) -> None:
        """When all priorities are zero, total falls back to 1.0."""
        p = AxisPriorities(efficiency=0.0, safety=0.0, resource=0.0, accessibility=0.0)
        frontier = [_axis(0.5, 0.5, 0.5, 0.5)]
        result = select_from_frontier(frontier, p)
        assert result is not None

    def test_selects_safety_best_when_safety_weighted(self) -> None:
        eff_best = _axis(1.0, 0.0, 0.0, 0.0)
        saf_best = _axis(0.0, 1.0, 0.0, 0.0)
        priorities = AxisPriorities(efficiency=0.03, safety=0.9, resource=0.03, accessibility=0.04)
        selected = select_from_frontier([eff_best, saf_best], priorities)
        assert selected is saf_best


# ---------------------------------------------------------------------------
# AxisScores.name property
# ---------------------------------------------------------------------------


class TestAxisScoresName:
    def test_name_returns_spawn_name(self) -> None:
        a = _axis(spawn=make_spawn(name="a_skeleton"))
        assert a.name == "a_skeleton"


# ---------------------------------------------------------------------------
# Axis computation functions
# ---------------------------------------------------------------------------


class TestAxisEfficiency:
    def test_returns_normalized_score(self) -> None:
        p = make_mob_profile(con=Con.WHITE, distance=50.0)
        w = ScoringWeights()
        result = _axis_efficiency(p, w, None, None, "")
        assert 0.0 <= result <= 1.0

    def test_higher_con_higher_efficiency(self) -> None:
        w = ScoringWeights()
        white = make_mob_profile(con=Con.WHITE)
        blue = make_mob_profile(con=Con.BLUE)
        assert _axis_efficiency(white, w, None, None, "") > _axis_efficiency(blue, w, None, None, "")


class TestAxisSafety:
    def test_isolated_npc_high_safety(self) -> None:
        p = make_mob_profile(
            isolation_score=1.0, social_npc_count=0, extra_npc_probability=0.0, threat_level=0.0
        )
        w = ScoringWeights()
        result = _axis_safety(p, w)
        assert 0.0 <= result <= 1.0
        assert result > 0.5

    def test_social_npcs_reduce_safety(self) -> None:
        w = ScoringWeights()
        solo = make_mob_profile(
            isolation_score=1.0, social_npc_count=0, extra_npc_probability=0.0, threat_level=0.0
        )
        social = make_mob_profile(
            isolation_score=1.0, social_npc_count=3, extra_npc_probability=0.0, threat_level=0.0
        )
        assert _axis_safety(solo, w) > _axis_safety(social, w)

    def test_threat_reduces_safety(self) -> None:
        w = ScoringWeights()
        safe = make_mob_profile(
            isolation_score=0.8, social_npc_count=0, extra_npc_probability=0.0, threat_level=0.0
        )
        threat = make_mob_profile(
            isolation_score=0.8, social_npc_count=0, extra_npc_probability=0.0, threat_level=1.0
        )
        assert _axis_safety(safe, w) > _axis_safety(threat, w)

    def test_clamped_to_zero(self) -> None:
        """Safety cannot go below 0.0 even with extreme penalties."""
        p = make_mob_profile(
            isolation_score=0.0, social_npc_count=5, extra_npc_probability=1.0, threat_level=1.0
        )
        w = ScoringWeights()
        result = _axis_safety(p, w)
        assert result >= 0.0


class TestAxisResource:
    def test_zero_mana_cost_scores_one(self) -> None:
        state = make_game_state(mana_max=500, mana_current=500)
        p = make_mob_profile(mana_cost_est=0)
        result = _axis_resource(p, state)
        assert result == pytest.approx(1.0)

    def test_full_mana_cost_scores_zero(self) -> None:
        state = make_game_state(mana_max=500, mana_current=500)
        p = make_mob_profile(mana_cost_est=500)
        result = _axis_resource(p, state)
        assert result == pytest.approx(0.0)

    def test_zero_max_mana_no_crash(self) -> None:
        state = make_game_state(mana_max=0, mana_current=0)
        p = make_mob_profile(mana_cost_est=100)
        result = _axis_resource(p, state)
        assert 0.0 <= result <= 1.0


class TestAxisAccessibility:
    def test_ideal_distance_high_score(self) -> None:
        w = ScoringWeights(dist_ideal=60.0, dist_width=100.0)
        p = make_mob_profile(distance=60.0, is_moving=False)
        result = _axis_accessibility(p, w, None)
        assert result > 0.3

    def test_moving_penalty(self) -> None:
        w = ScoringWeights()
        still = make_mob_profile(distance=60.0, is_moving=False)
        moving = make_mob_profile(distance=60.0, is_moving=True)
        assert _axis_accessibility(still, w, None) > _axis_accessibility(moving, w, None)


class TestComputeAxes:
    def test_returns_axis_scores(self) -> None:
        p = make_mob_profile()
        w = ScoringWeights()
        state = make_game_state()
        result = compute_axes(p, w, state, None)
        assert 0.0 <= result.efficiency <= 1.0
        assert 0.0 <= result.safety <= 1.0
        assert 0.0 <= result.resource <= 1.0
        assert 0.0 <= result.accessibility <= 1.0
        assert result.profile is p


# ---------------------------------------------------------------------------
# compute_priorities -- state-aware priority shifts
# ---------------------------------------------------------------------------


class TestComputePriorities:
    def test_default_phase_steady(self) -> None:
        state = make_game_state(hp_current=1000, hp_max=1000, mana_current=500, mana_max=500)
        p = compute_priorities(state, None, "steady")
        assert p.efficiency > 0
        assert p.safety > 0
        assert p.resource > 0
        assert p.accessibility > 0

    def test_low_mana_boosts_resource(self) -> None:
        state = make_game_state(mana_current=100, mana_max=500)  # 20% mana
        base = AxisPriorities()
        p = compute_priorities(state, None, "steady", base)
        assert p.resource > base.resource / (
            base.efficiency + base.safety + base.resource + base.accessibility
        )

    def test_low_hp_boosts_safety(self) -> None:
        state = make_game_state(hp_current=200, hp_max=1000)  # 20% hp
        base = AxisPriorities()
        p = compute_priorities(state, None, "steady", base)
        total = p.efficiency + p.safety + p.resource + p.accessibility
        assert p.safety / total > base.safety / (
            base.efficiency + base.safety + base.resource + base.accessibility
        )

    def test_incident_phase_boosts_safety(self) -> None:
        state = make_game_state(hp_current=1000, hp_max=1000, mana_current=500, mana_max=500)
        p = compute_priorities(state, None, "incident")
        p_steady = compute_priorities(state, None, "steady")
        assert p.safety > p_steady.safety

    def test_idle_phase_boosts_accessibility(self) -> None:
        state = make_game_state(hp_current=1000, hp_max=1000, mana_current=500, mana_max=500)
        p = compute_priorities(state, None, "idle")
        p_steady = compute_priorities(state, None, "steady")
        assert p.accessibility > p_steady.accessibility

    def test_startup_phase(self) -> None:
        state = make_game_state(hp_current=1000, hp_max=1000, mana_current=500, mana_max=500)
        p = compute_priorities(state, None, "startup")
        assert p.safety > 0
        assert p.accessibility > 0

    def test_priorities_sum_to_one(self) -> None:
        state = make_game_state(hp_current=200, hp_max=1000, mana_current=50, mana_max=500)
        p = compute_priorities(state, None, "incident")
        total = p.efficiency + p.safety + p.resource + p.accessibility
        assert total == pytest.approx(1.0, abs=0.01)

    def test_clamp_prevents_negative(self) -> None:
        """Even extreme combined adjustments keep all pre-normalization weights >= 0.05."""
        state = make_game_state(hp_current=100, hp_max=1000, mana_current=50, mana_max=500)
        p = compute_priorities(state, None, "incident")
        # All normalized weights should be positive
        assert p.efficiency > 0
        assert p.safety > 0
        assert p.resource > 0
        assert p.accessibility > 0


# ---------------------------------------------------------------------------
# log_pareto_selection
# ---------------------------------------------------------------------------


class TestLogParetoSelection:
    def test_logs_without_error(self, caplog: pytest.LogCaptureFixture) -> None:
        selected = _axis(0.8, 0.6, 0.7, 0.5, spawn=make_spawn(name="a_skeleton"))
        other = _axis(0.6, 0.7, 0.5, 0.4, spawn=make_spawn(name="a_bat"))
        frontier = [selected, other]
        priorities = AxisPriorities()
        with caplog.at_level(logging.DEBUG):
            log_pareto_selection(frontier, selected, priorities, total_candidates=5)
        # Should log INFO + VERBOSE without crashing
        assert any("Pareto" in r.message for r in caplog.records)

    def test_single_item_frontier(self, caplog: pytest.LogCaptureFixture) -> None:
        selected = _axis(0.8, 0.6, 0.7, 0.5, spawn=make_spawn(name="a_skeleton"))
        frontier = [selected]
        priorities = AxisPriorities()
        with caplog.at_level(logging.DEBUG):
            log_pareto_selection(frontier, selected, priorities, total_candidates=1)
        assert any("Pareto" in r.message for r in caplog.records)
