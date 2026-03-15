"""Tests for brain.scoring.target -- consideration-based utility scoring.

Uses make_mob_profile() factory for constructing MobProfile instances.
Tests cover score_target, hard rejects, distance/isolation/con/social factors.
"""

from __future__ import annotations

import json
import time as _time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from brain.scoring.target import (
    _MAX_THROTTLE_ENTRIES,
    MobProfile,
    ScoringWeights,
    _hard_reject,
    _heading_multiplier,
    _heat_multiplier,
    _learned_add_penalty,
    _prune_throttle,
    _score_camp,
    _score_efficiency,
    _score_loot,
    estimate_fight_duration,
    estimate_mana_cost,
    load_scoring_weights,
    log_top_targets,
    score_target,
)
from core.types import Point
from perception.combat_eval import (
    Con,
    get_avoid_names,
    get_zone_avoid_mobs,
    set_avoid_names,
    set_zone_avoid_mobs,
)
from tests.factories import make_mob_profile, make_spawn


def _score(profile: MobProfile, **kw: Any) -> float:
    """Score a single profile with no context, no players, default weights."""
    weights = kw.pop("weights", ScoringWeights()) if "weights" in kw else ScoringWeights()
    return score_target(
        p=profile,
        weights=weights,
        profiles=[profile],
        players=[],
        ctx=None,
        player_x=0.0,
        player_y=0.0,
    )


class TestScoreTarget:
    def test_score_is_non_negative(self) -> None:
        p = make_mob_profile()
        s = _score(p)
        assert s >= 0.0

    def test_closer_scores_higher(self) -> None:
        near = make_mob_profile(distance=30.0)
        far = make_mob_profile(distance=200.0)
        assert _score(near) > _score(far)

    def test_isolated_scores_higher(self) -> None:
        iso = make_mob_profile(isolation_score=1.0)
        clustered = make_mob_profile(isolation_score=0.1)
        assert _score(iso) > _score(clustered)

    def test_social_adds_penalty(self) -> None:
        solo = make_mob_profile(social_npc_count=0)
        social = make_mob_profile(social_npc_count=3)
        # social_npc_count=3 hits the hard limit (default social_npc_hard_limit=3)
        # so it will be rejected by _hard_reject. Use count=2 instead.
        social = make_mob_profile(social_npc_count=2)
        assert _score(solo) > _score(social)

    def test_threat_scores_zero(self) -> None:
        """is_threat=True means YELLOW/RED con -- may be rejected or score low."""
        threat = make_mob_profile(is_threat=True, con=Con.YELLOW)
        safe = make_mob_profile(is_threat=False, con=Con.WHITE)
        assert _score(safe) > _score(threat)

    @pytest.mark.parametrize(
        "con_a, con_b",
        [
            (Con.WHITE, Con.BLUE),
            (Con.BLUE, Con.LIGHT_BLUE),
        ],
    )
    def test_con_ordering(self, con_a: Con, con_b: Con) -> None:
        """Higher-XP con colors score higher, all else equal."""
        a = make_mob_profile(con=con_a, distance=60.0)
        b = make_mob_profile(con=con_b, distance=60.0)
        assert _score(a) > _score(b)

    def test_moving_penalty(self) -> None:
        still = make_mob_profile(is_moving=False)
        moving = make_mob_profile(is_moving=True)
        assert _score(still) > _score(moving)

    def test_score_breakdown_populated(self) -> None:
        p = make_mob_profile()
        bd: dict[str, float] = {}
        score_target(
            p=p,
            weights=ScoringWeights(),
            profiles=[p],
            players=[],
            ctx=None,
            player_x=0.0,
            player_y=0.0,
            breakdown=bd,
        )
        expected_keys = {
            "con_pref",
            "resource",
            "distance",
            "isolation",
            "social_add",
            "camp_proximity",
            "movement",
            "caster",
            "loot_value",
            "heading",
            "spatial_heat",
            "learned_efficiency",
        }
        assert expected_keys.issubset(set(bd.keys()))


# ---------------------------------------------------------------------------
# Hard-reject conditions
# ---------------------------------------------------------------------------


class TestHardReject:
    """Tests for _hard_reject covering all rejection paths."""

    def _weights(self, **kw: Any) -> ScoringWeights:
        return ScoringWeights(**kw)

    def test_guard_proximity_rejects(self) -> None:
        """NPC near a guard (name containing 'Guard') is rejected."""
        original = get_avoid_names()
        try:
            set_avoid_names(frozenset({"Guard"}))
            target = make_mob_profile(spawn=make_spawn(name="a_skeleton", x=100.0, y=100.0))
            guard = make_mob_profile(spawn=make_spawn(name="Guard_Arias", x=110.0, y=100.0))
            w = self._weights(avoid_npc_proximity=100.0)
            assert _hard_reject(target, w, [target, guard], [], None, frozenset({"Guard"}))
        finally:
            set_avoid_names(original)

    def test_guard_far_away_not_rejected(self) -> None:
        """NPC far from a guard is NOT rejected."""
        target = make_mob_profile(spawn=make_spawn(name="a_skeleton", x=100.0, y=100.0))
        guard = make_mob_profile(spawn=make_spawn(name="Guard_Arias", x=500.0, y=500.0))
        w = self._weights(avoid_npc_proximity=100.0)
        rejected = _hard_reject(target, w, [target, guard], [], None, frozenset({"Guard"}))
        assert not rejected

    def test_player_proximity_rejects(self) -> None:
        """NPC near another player (within player_proximity) is rejected."""
        target = make_mob_profile(spawn=make_spawn(name="a_skeleton", x=100.0, y=100.0))
        player_spawn = make_spawn(name="SomePlayer", x=110.0, y=100.0)
        w = self._weights(player_proximity=30.0)
        rejected = _hard_reject(target, w, [target], [(player_spawn, 10.0)], None, frozenset())
        assert rejected

    def test_player_far_away_not_rejected(self) -> None:
        """NPC far from players is NOT rejected."""
        target = make_mob_profile(spawn=make_spawn(name="a_skeleton", x=100.0, y=100.0))
        player_spawn = make_spawn(name="SomePlayer", x=500.0, y=500.0)
        w = self._weights(player_proximity=30.0)
        rejected = _hard_reject(target, w, [target], [(player_spawn, 500.0)], None, frozenset())
        assert not rejected

    def test_social_hard_limit_rejects(self) -> None:
        """NPC with social_npc_count >= social_npc_hard_limit is rejected."""
        target = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=100.0, y=100.0),
            social_npc_count=4,
        )
        w = self._weights(social_npc_hard_limit=3)
        rejected = _hard_reject(target, w, [target], [], None, frozenset())
        assert rejected

    def test_social_below_limit_not_rejected(self) -> None:
        """NPC with social_npc_count < social_npc_hard_limit passes."""
        target = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=100.0, y=100.0),
            social_npc_count=2,
        )
        w = self._weights(social_npc_hard_limit=3)
        rejected = _hard_reject(target, w, [target], [], None, frozenset())
        assert not rejected

    def test_con_not_in_target_cons_rejects(self) -> None:
        """NPC with con not in ctx.zone.target_cons is rejected."""
        target = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=100.0, y=100.0),
            con=Con.YELLOW,
        )
        ctx = SimpleNamespace(
            zone=SimpleNamespace(target_cons=frozenset({Con.BLUE, Con.WHITE})),
        )
        w = self._weights()
        rejected = _hard_reject(target, w, [target], [], ctx, frozenset())
        assert rejected

    def test_con_in_target_cons_not_rejected(self) -> None:
        """NPC with con in ctx.zone.target_cons passes."""
        target = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=100.0, y=100.0),
            con=Con.WHITE,
        )
        ctx = SimpleNamespace(
            zone=SimpleNamespace(target_cons=frozenset({Con.BLUE, Con.WHITE})),
        )
        w = self._weights()
        rejected = _hard_reject(target, w, [target], [], ctx, frozenset())
        assert not rejected

    def test_empty_target_cons_allows_all(self) -> None:
        """When target_cons is empty (falsy), no con-based rejection."""
        target = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=100.0, y=100.0),
            con=Con.YELLOW,
        )
        ctx = SimpleNamespace(
            zone=SimpleNamespace(target_cons=frozenset()),
        )
        w = self._weights()
        rejected = _hard_reject(target, w, [target], [], ctx, frozenset())
        assert not rejected

    def test_zone_avoid_mob_rejects(self) -> None:
        """NPC whose base name is in zone_avoid_mobs is rejected."""
        original = get_zone_avoid_mobs()
        try:
            set_zone_avoid_mobs(frozenset({"a_skeleton"}))
            target = make_mob_profile(
                spawn=make_spawn(name="a_skeleton007", x=100.0, y=100.0),
            )
            w = self._weights()
            rejected = _hard_reject(target, w, [target], [], None, frozenset())
            assert rejected
        finally:
            set_zone_avoid_mobs(original)


# ---------------------------------------------------------------------------
# Camp scoring with drift penalty
# ---------------------------------------------------------------------------


class TestScoreCamp:
    """Tests for _score_camp: proximity bonus and drift penalty."""

    def _make_ctx(
        self, roam_radius: float = 250.0, camp_x: float = 0.0, camp_y: float = 0.0
    ) -> SimpleNamespace:
        """Build a minimal ctx with a camp that delegates effective_camp_distance."""

        camp = SimpleNamespace(
            camp_x=camp_x,
            camp_y=camp_y,
            roam_radius=roam_radius,
            effective_camp_distance=lambda pos: pos.dist_to(Point(camp_x, camp_y, 0.0)),
        )
        return SimpleNamespace(camp=camp)

    def test_mob_close_to_camp_high_score(self) -> None:
        """NPC near camp center gets a high camp proximity score."""
        ctx = self._make_ctx(roam_radius=250.0)
        mob = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=10.0, y=10.0),
            camp_distance=14.0,
        )
        w = ScoringWeights(camp_peak=30.0, camp_falloff_k=0.04)
        score = _score_camp(mob, w, ctx, player_x=0.0, player_y=0.0)
        # Close to camp => high score (near camp_peak)
        assert score > 25.0

    def test_mob_far_from_camp_low_score(self) -> None:
        """NPC far beyond roam radius gets a low camp proximity score."""
        ctx = self._make_ctx(roam_radius=100.0)
        mob = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=500.0, y=500.0),
            camp_distance=707.0,
        )
        w = ScoringWeights(camp_peak=30.0, camp_falloff_k=0.04)
        score = _score_camp(mob, w, ctx, player_x=0.0, player_y=0.0)
        # Far beyond roam => near-zero score
        assert score < 5.0

    def test_camp_drift_penalty_applied(self) -> None:
        """When player is far from camp and NPC is even further, drift penalty kicks in."""
        ctx = self._make_ctx(roam_radius=100.0)
        # Player at 80u from camp (> 50% of roam=100)
        player_x, player_y = 80.0, 0.0
        # NPC at 200u from camp (further than player)
        mob = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=200.0, y=0.0),
            camp_distance=200.0,
        )
        w = ScoringWeights(camp_peak=30.0, camp_falloff_k=0.04)
        score_with_drift = _score_camp(mob, w, ctx, player_x=player_x, player_y=player_y)

        # Compare to same NPC when player is at camp center (no drift penalty)
        score_no_drift = _score_camp(mob, w, ctx, player_x=0.0, player_y=0.0)
        assert score_with_drift < score_no_drift

    def test_no_drift_penalty_when_player_near_camp(self) -> None:
        """No drift penalty when player is within 50% of roam radius."""
        ctx = self._make_ctx(roam_radius=200.0)
        # Player at 50u from camp (< 50% of roam=200 which is 100)
        player_x, player_y = 50.0, 0.0
        mob = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=150.0, y=0.0),
            camp_distance=150.0,
        )
        w = ScoringWeights(camp_peak=30.0, camp_falloff_k=0.04)
        score_near = _score_camp(mob, w, ctx, player_x=player_x, player_y=player_y)
        score_center = _score_camp(mob, w, ctx, player_x=0.0, player_y=0.0)
        # Should be equal because player_camp_dist(50) <= roam*0.5(100)
        assert score_near == score_center


# ---------------------------------------------------------------------------
# Heading multiplier edge cases
# ---------------------------------------------------------------------------


class TestHeadingMultiplier:
    """Tests for _heading_multiplier: facing vs. away vs. far."""

    def test_npc_facing_player_penalty(self) -> None:
        """NPC facing toward the player gets heading_facing_mult (< 1.0)."""
        # NPC at (100, 100), player at (100, 200). heading_to -> heading ~0 (north).
        # NPC heading = 0 means facing north toward player.
        from nav.geometry import heading_to

        npc_x, npc_y = 100.0, 100.0
        player_x, player_y = 100.0, 200.0
        h = heading_to(npc_x, npc_y, player_x, player_y)
        # Set NPC heading to match the angle toward player (facing player)
        mob = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=npc_x, y=npc_y, heading=h),
            distance=50.0,
        )
        w = ScoringWeights(heading_facing_mult=0.7, heading_away_mult=1.1)
        mult = _heading_multiplier(mob, w, player_x, player_y)
        assert mult == pytest.approx(0.7)

    def test_npc_facing_away_bonus(self) -> None:
        """NPC facing away from the player gets heading_away_mult (> 1.0)."""
        from nav.geometry import heading_to

        npc_x, npc_y = 100.0, 100.0
        player_x, player_y = 100.0, 200.0
        h = heading_to(npc_x, npc_y, player_x, player_y)
        # Opposite direction: heading 180 degrees away in 512-unit system = h + 256
        away_heading = (h + 256) % 512
        mob = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=npc_x, y=npc_y, heading=away_heading),
            distance=50.0,
        )
        w = ScoringWeights(heading_facing_mult=0.7, heading_away_mult=1.1)
        mult = _heading_multiplier(mob, w, player_x, player_y)
        assert mult == pytest.approx(1.1)

    def test_distance_beyond_80_no_effect(self) -> None:
        """When distance >= 80, heading multiplier is always 1.0."""
        mob = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=100.0, y=100.0, heading=0.0),
            distance=80.0,
        )
        w = ScoringWeights(heading_facing_mult=0.7, heading_away_mult=1.1)
        mult = _heading_multiplier(mob, w, 100.0, 200.0)
        assert mult == 1.0

    def test_distance_zero_no_effect(self) -> None:
        """When distance <= 0, heading multiplier is always 1.0."""
        mob = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=100.0, y=100.0, heading=0.0),
            distance=0.0,
        )
        w = ScoringWeights(heading_facing_mult=0.7, heading_away_mult=1.1)
        mult = _heading_multiplier(mob, w, 100.0, 200.0)
        assert mult == 1.0

    def test_perpendicular_heading_neutral(self) -> None:
        """NPC facing perpendicular (heading error ~128) returns 1.0."""
        from nav.geometry import heading_to

        npc_x, npc_y = 100.0, 100.0
        player_x, player_y = 100.0, 200.0
        h = heading_to(npc_x, npc_y, player_x, player_y)
        # 90 degrees off in 512 system = 128 units. That's between 60 and 180.
        perp_heading = (h + 128) % 512
        mob = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=npc_x, y=npc_y, heading=perp_heading),
            distance=50.0,
        )
        w = ScoringWeights(heading_facing_mult=0.7, heading_away_mult=1.1)
        mult = _heading_multiplier(mob, w, player_x, player_y)
        assert mult == 1.0


# ---------------------------------------------------------------------------
# Heat multiplier with spatial memory mock
# ---------------------------------------------------------------------------


class TestHeatMultiplier:
    """Tests for _heat_multiplier with mocked spatial memory."""

    def test_no_ctx_returns_one(self) -> None:
        """Without ctx, heat multiplier is 1.0."""
        mob = make_mob_profile()
        w = ScoringWeights()
        assert _heat_multiplier(mob, w, None) == 1.0

    def test_no_spatial_memory_returns_one(self) -> None:
        """With ctx but no spatial_memory, heat multiplier is 1.0."""
        ctx = SimpleNamespace(spatial_memory=None)
        mob = make_mob_profile()
        w = ScoringWeights()
        assert _heat_multiplier(mob, w, ctx) == 1.0

    def test_zero_heat_returns_one(self) -> None:
        """At a cold location (heat=0), multiplier is 1.0."""
        sm = MagicMock()
        sm.heat_at.return_value = 0.0
        ctx = SimpleNamespace(spatial_memory=sm)
        mob = make_mob_profile(spawn=make_spawn(x=100.0, y=100.0))
        w = ScoringWeights(heat_multiplier=0.1, heat_cap=5.0)
        assert _heat_multiplier(mob, w, ctx) == pytest.approx(1.0)
        sm.heat_at.assert_called_once()

    def test_high_heat_boosts_multiplier(self) -> None:
        """At a hot location, multiplier is > 1.0."""
        sm = MagicMock()
        sm.heat_at.return_value = 3.0
        ctx = SimpleNamespace(spatial_memory=sm)
        mob = make_mob_profile(spawn=make_spawn(x=50.0, y=50.0))
        w = ScoringWeights(heat_multiplier=0.1, heat_cap=5.0)
        result = _heat_multiplier(mob, w, ctx)
        expected = 1.0 + 0.1 * 3.0  # 1.3
        assert result == pytest.approx(expected)

    def test_heat_capped_at_heat_cap(self) -> None:
        """Heat values beyond heat_cap are clamped."""
        sm = MagicMock()
        sm.heat_at.return_value = 100.0  # way above cap
        ctx = SimpleNamespace(spatial_memory=sm)
        mob = make_mob_profile()
        w = ScoringWeights(heat_multiplier=0.1, heat_cap=5.0)
        result = _heat_multiplier(mob, w, ctx)
        expected = 1.0 + 0.1 * 5.0  # capped at 5.0
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Efficiency scoring with fight_history mock
# ---------------------------------------------------------------------------


class TestScoreEfficiency:
    """Tests for _score_efficiency with fight duration data."""

    def test_no_durations_returns_zero(self) -> None:
        """Without fight_durations data, efficiency score is 0."""
        w = ScoringWeights()
        assert _score_efficiency("a_skeleton", w, None) == 0.0
        assert _score_efficiency("a_skeleton", w, {}) == 0.0

    def test_mob_not_in_durations_returns_zero(self) -> None:
        """If mob base name not in fight_durations, returns 0."""
        w = ScoringWeights()
        assert _score_efficiency("a_skeleton", w, {"a_bat": [15.0, 18.0]}) == 0.0

    def test_fast_defeats_bonus(self) -> None:
        """Average duration < 20s earns fast_defeat_bonus."""
        w = ScoringWeights(fast_defeat_bonus=10, slow_defeat_penalty=10)
        durations = {"a_skeleton": [12.0, 15.0, 18.0]}  # avg = 15.0
        result = _score_efficiency("a_skeleton", w, durations)
        assert result == pytest.approx(10.0)

    def test_slow_defeats_penalty(self) -> None:
        """Average duration > 40s incurs slow_defeat_penalty."""
        w = ScoringWeights(fast_defeat_bonus=10, slow_defeat_penalty=10)
        durations = {"a_skeleton": [42.0, 45.0, 50.0]}  # avg = 45.67
        result = _score_efficiency("a_skeleton", w, durations)
        assert result == pytest.approx(-10.0)

    def test_medium_duration_returns_zero(self) -> None:
        """Average duration between 20s and 40s returns 0."""
        w = ScoringWeights(fast_defeat_bonus=10, slow_defeat_penalty=10)
        durations = {"a_skeleton": [25.0, 30.0, 35.0]}  # avg = 30.0
        result = _score_efficiency("a_skeleton", w, durations)
        assert result == 0.0

    def test_boundary_at_20_returns_zero(self) -> None:
        """avg_dur == 20 is NOT < 20, so returns 0."""
        w = ScoringWeights(fast_defeat_bonus=10, slow_defeat_penalty=10)
        durations = {"a_skeleton": [20.0]}
        result = _score_efficiency("a_skeleton", w, durations)
        assert result == 0.0

    def test_boundary_at_40_returns_zero(self) -> None:
        """avg_dur == 40 is NOT > 40, so returns 0."""
        w = ScoringWeights(fast_defeat_bonus=10, slow_defeat_penalty=10)
        durations = {"a_skeleton": [40.0]}
        result = _score_efficiency("a_skeleton", w, durations)
        assert result == 0.0


# ---------------------------------------------------------------------------
# _prune_throttle
# ---------------------------------------------------------------------------


class TestPruneThrottle:
    """Tests for _prune_throttle: stale entry cleanup in throttle dicts."""

    def test_no_prune_under_cap(self) -> None:
        """When dict is under cap, prune is a no-op."""
        d = {"a": 1.0, "b": 2.0}
        _prune_throttle(d, 100.0, 30.0)
        assert len(d) == 2

    def test_prune_stale_entries(self) -> None:
        """Stale entries (older than 2x interval) are removed."""
        now = 1000.0
        interval = 30.0
        d = {}
        # Create entries well beyond cap
        for i in range(_MAX_THROTTLE_ENTRIES + 10):
            d[f"key_{i}"] = now - 100.0  # all stale (100s > 2 * 30s)
        _prune_throttle(d, now, interval)
        assert len(d) == 0

    def test_prune_drops_oldest_half_when_still_over_cap(self) -> None:
        """When all entries are fresh, oldest half is dropped if over cap."""
        now = 1000.0
        interval = 30.0
        d = {}
        for i in range(_MAX_THROTTLE_ENTRIES + 20):
            d[f"key_{i}"] = now - 10.0 + i * 0.001  # all fresh
        _prune_throttle(d, now, interval)
        assert len(d) <= _MAX_THROTTLE_ENTRIES


# ---------------------------------------------------------------------------
# load_scoring_weights from file
# ---------------------------------------------------------------------------


class TestLoadScoringWeights:
    def test_default_when_no_path(self) -> None:
        w = load_scoring_weights(None)
        assert isinstance(w, ScoringWeights)
        assert w.con_white == 100

    def test_load_from_json(self, tmp_path) -> None:
        data = {"con_white": 50, "con_blue": 30, "dist_ideal": 80.0}
        f = tmp_path / "weights.json"
        f.write_text(json.dumps(data))
        w = load_scoring_weights(str(f))
        assert w.con_white == 50
        assert w.con_blue == 30
        assert w.dist_ideal == 80.0

    def test_unknown_keys_ignored(self, tmp_path) -> None:
        data = {"con_white": 50, "unknown_field": 999}
        f = tmp_path / "weights.json"
        f.write_text(json.dumps(data))
        w = load_scoring_weights(str(f))
        assert w.con_white == 50

    def test_invalid_json_returns_defaults(self, tmp_path) -> None:
        f = tmp_path / "weights.json"
        f.write_text("NOT JSON")
        w = load_scoring_weights(str(f))
        assert w.con_white == 100  # default

    def test_missing_file_returns_defaults(self) -> None:
        w = load_scoring_weights("/nonexistent/path/weights.json")
        assert w.con_white == 100


# ---------------------------------------------------------------------------
# estimate_fight_duration
# ---------------------------------------------------------------------------


class TestEstimateFightDuration:
    def test_session_local_durations(self) -> None:
        """Session-local fight_durations take priority over heuristic."""
        result = estimate_fight_duration(
            "a_skeleton", Con.WHITE, 10, fight_durations={"a_skeleton": [12.0, 14.0]}
        )
        assert result == pytest.approx(13.0)

    def test_heuristic_light_blue(self) -> None:
        result = estimate_fight_duration("a_skeleton", Con.LIGHT_BLUE, 10)
        assert result == 15.0 + 10 * 1.5

    def test_heuristic_blue(self) -> None:
        result = estimate_fight_duration("a_skeleton", Con.BLUE, 10)
        assert result == 20.0 + 10 * 2.0

    def test_heuristic_white(self) -> None:
        result = estimate_fight_duration("a_skeleton", Con.WHITE, 10)
        assert result == 25.0 + 10 * 2.0

    def test_heuristic_yellow(self) -> None:
        result = estimate_fight_duration("a_skeleton", Con.YELLOW, 10)
        assert result == 35.0 + 10 * 2.5

    def test_heuristic_default(self) -> None:
        """Unknown con color falls through to default."""
        result = estimate_fight_duration("a_skeleton", Con.GREEN, 10)
        assert result == 20.0

    def test_empty_session_durations_falls_through(self) -> None:
        """Empty list in fight_durations falls through to heuristic."""
        result = estimate_fight_duration("a_skeleton", Con.WHITE, 10, fight_durations={"a_skeleton": []})
        assert result == 25.0 + 10 * 2.0

    def test_fight_history_priority(self) -> None:
        """FightHistory learned data takes priority over session data."""
        fh = MagicMock()
        fh.learned_duration.return_value = 42.0
        ctx = SimpleNamespace(fight_history=fh)
        result = estimate_fight_duration(
            "a_skeleton", Con.WHITE, 10, ctx=ctx, fight_durations={"a_skeleton": [10.0]}
        )
        assert result == 42.0

    def test_fight_history_none_falls_through(self) -> None:
        """FightHistory returning None falls through to session data."""
        fh = MagicMock()
        fh.learned_duration.return_value = None
        ctx = SimpleNamespace(fight_history=fh)
        result = estimate_fight_duration(
            "a_skeleton", Con.WHITE, 10, ctx=ctx, fight_durations={"a_skeleton": [20.0]}
        )
        assert result == 20.0


# ---------------------------------------------------------------------------
# estimate_mana_cost
# ---------------------------------------------------------------------------


class TestEstimateMana:
    def test_light_blue_zero(self) -> None:
        assert estimate_mana_cost(Con.LIGHT_BLUE, 30.0) == 0

    def test_blue_cost(self) -> None:
        assert estimate_mana_cost(Con.BLUE, 30.0) == 10

    def test_white_short_fight(self) -> None:
        assert estimate_mana_cost(Con.WHITE, 20.0) == 10

    def test_white_long_fight(self) -> None:
        assert estimate_mana_cost(Con.WHITE, 30.0) == 20

    def test_yellow(self) -> None:
        assert estimate_mana_cost(Con.YELLOW, 30.0) == 40

    def test_default_con(self) -> None:
        assert estimate_mana_cost(Con.GREEN, 30.0) == 10

    def test_fight_history_priority(self) -> None:
        fh = MagicMock()
        fh.learned_mana.return_value = 99
        ctx = SimpleNamespace(fight_history=fh)
        result = estimate_mana_cost(Con.WHITE, 30.0, mob_base="a_skeleton", ctx=ctx)
        assert result == 99

    def test_fight_history_none_falls_through(self) -> None:
        fh = MagicMock()
        fh.learned_mana.return_value = None
        ctx = SimpleNamespace(fight_history=fh)
        result = estimate_mana_cost(Con.WHITE, 30.0, mob_base="a_skeleton", ctx=ctx)
        assert result == 20  # heuristic for WHITE with dur > 25


# ---------------------------------------------------------------------------
# _score_loot
# ---------------------------------------------------------------------------


class TestScoreLoot:
    def test_no_ctx_returns_zero(self) -> None:
        w = ScoringWeights()
        assert _score_loot("a_skeleton", w, None) == 0.0

    def test_no_loot_values_returns_zero(self) -> None:
        ctx = SimpleNamespace(loot=SimpleNamespace(mob_loot_values=None))
        w = ScoringWeights()
        assert _score_loot("a_skeleton", w, ctx) == 0.0

    def test_mob_not_in_loot_returns_zero(self) -> None:
        ctx = SimpleNamespace(loot=SimpleNamespace(mob_loot_values={"a_bat": 500}))
        w = ScoringWeights()
        assert _score_loot("a_skeleton", w, ctx) == 0.0

    def test_loot_value_scored(self) -> None:
        ctx = SimpleNamespace(loot=SimpleNamespace(mob_loot_values={"a_skeleton": 1000}))
        w = ScoringWeights(loot_value_scale=5.0, loot_value_cap=30)
        result = _score_loot("a_skeleton", w, ctx)
        # 1000 / 100.0 * 5.0 = 50.0, capped at 30
        assert result == 30.0

    def test_loot_value_under_cap(self) -> None:
        ctx = SimpleNamespace(loot=SimpleNamespace(mob_loot_values={"a_skeleton": 200}))
        w = ScoringWeights(loot_value_scale=5.0, loot_value_cap=30)
        result = _score_loot("a_skeleton", w, ctx)
        # 200 / 100.0 * 5.0 = 10.0
        assert result == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _learned_add_penalty
# ---------------------------------------------------------------------------


class TestLearnedAddPenalty:
    def test_no_ctx(self) -> None:
        w = ScoringWeights()
        assert _learned_add_penalty("a_skeleton", w, None) == 0.0

    def test_no_fight_history(self) -> None:
        ctx = SimpleNamespace(fight_history=None)
        w = ScoringWeights()
        assert _learned_add_penalty("a_skeleton", w, ctx) == 0.0

    def test_low_adds_no_penalty(self) -> None:
        fh = MagicMock()
        fh.learned_adds.return_value = 0.3  # below 0.5 threshold
        ctx = SimpleNamespace(fight_history=fh)
        w = ScoringWeights()
        assert _learned_add_penalty("a_skeleton", w, ctx) == 0.0

    def test_high_adds_penalty(self) -> None:
        fh = MagicMock()
        fh.learned_adds.return_value = 1.5
        ctx = SimpleNamespace(fight_history=fh)
        w = ScoringWeights(social_npc_penalty=30)
        result = _learned_add_penalty("a_skeleton", w, ctx)
        assert result == pytest.approx(1.5 * 30 * 0.5)

    def test_none_adds_no_penalty(self) -> None:
        fh = MagicMock()
        fh.learned_adds.return_value = None
        ctx = SimpleNamespace(fight_history=fh)
        w = ScoringWeights()
        assert _learned_add_penalty("a_skeleton", w, ctx) == 0.0


# ---------------------------------------------------------------------------
# score_target with context: resource, caster, danger, extra_npc
# ---------------------------------------------------------------------------


class TestScoreTargetWithContext:
    """Tests for _score_factors paths needing a ctx mock."""

    def _make_ctx(self, **kw) -> SimpleNamespace:
        """Build a minimal ctx with loot, danger_memory, fight_history, spatial_memory, zone, camp."""

        camp = SimpleNamespace(
            camp_x=0.0,
            camp_y=0.0,
            roam_radius=250.0,
            effective_camp_distance=lambda pos: pos.dist_to(Point(0.0, 0.0, 0.0)),
        )
        defaults = dict(
            loot=SimpleNamespace(
                resource_targets=[],
                mob_loot_values={},
                caster_mob_names=set(),
            ),
            danger_memory=None,
            fight_history=None,
            spatial_memory=None,
            zone=SimpleNamespace(target_cons=frozenset()),
            camp=camp,
        )
        defaults.update(kw)
        return SimpleNamespace(**defaults)

    def test_resource_target_bonus(self) -> None:
        ctx = self._make_ctx(
            loot=SimpleNamespace(
                resource_targets=["a_skeleton"],
                mob_loot_values={},
                caster_mob_names=set(),
            ),
        )
        target = make_mob_profile(spawn=make_spawn(name="a_skeleton", x=50.0, y=50.0))
        w = ScoringWeights(resource_bonus=40)
        from brain.scoring.target import _score_factors

        score_with = _score_factors(target, w, ctx, 0.0, 0.0, None, None)
        ctx_no_resource = self._make_ctx()
        score_without = _score_factors(target, w, ctx_no_resource, 0.0, 0.0, None, None)
        assert score_with > score_without

    def test_caster_penalty(self) -> None:
        ctx = self._make_ctx(
            loot=SimpleNamespace(
                resource_targets=[],
                mob_loot_values={},
                caster_mob_names={"a_skeleton"},
            ),
        )
        target = make_mob_profile(spawn=make_spawn(name="a_skeleton", x=50.0, y=50.0))
        w = ScoringWeights(caster_penalty=25)
        from brain.scoring.target import _score_factors

        score_caster = _score_factors(target, w, ctx, 0.0, 0.0, None, None)
        ctx_no_caster = self._make_ctx()
        score_normal = _score_factors(target, w, ctx_no_caster, 0.0, 0.0, None, None)
        assert score_caster < score_normal

    def test_extra_npc_probability_penalty(self) -> None:
        target_high = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=50.0, y=50.0),
            extra_npc_probability=0.8,
        )
        target_low = make_mob_profile(
            spawn=make_spawn(name="a_skeleton", x=50.0, y=50.0),
            extra_npc_probability=0.0,
        )
        ctx = self._make_ctx()
        w = ScoringWeights()
        from brain.scoring.target import _score_factors

        s_high = _score_factors(target_high, w, ctx, 0.0, 0.0, None, None)
        s_low = _score_factors(target_low, w, ctx, 0.0, 0.0, None, None)
        assert s_low > s_high

    def test_danger_memory_penalty(self) -> None:
        dm = MagicMock()
        dm.danger_penalty.return_value = 2.0
        ctx = self._make_ctx(danger_memory=dm)
        target = make_mob_profile(spawn=make_spawn(name="a_skeleton", x=50.0, y=50.0))
        w = ScoringWeights()
        from brain.scoring.target import _score_factors

        score_danger = _score_factors(target, w, ctx, 0.0, 0.0, None, None)
        ctx_no_danger = self._make_ctx()
        score_safe = _score_factors(target, w, ctx_no_danger, 0.0, 0.0, None, None)
        assert score_safe > score_danger


# ---------------------------------------------------------------------------
# log_top_targets
# ---------------------------------------------------------------------------


class TestLogTopTargets:
    def test_logs_when_interval_passed(self) -> None:
        targets = [
            make_mob_profile(spawn=make_spawn(name="a_skeleton"), con=Con.WHITE, distance=50.0),
            make_mob_profile(spawn=make_spawn(name="a_bat"), con=Con.BLUE, distance=80.0),
        ]
        targets[0].score = 100.0
        targets[1].score = 80.0
        last = 0.0  # ancient time
        new_time = log_top_targets(targets, last)
        assert new_time > last

    def test_does_not_log_when_recent(self) -> None:
        targets = [make_mob_profile()]
        last = _time.time()  # just now
        new_time = log_top_targets(targets, last)
        assert new_time == last  # unchanged

    def test_empty_targets(self) -> None:
        new_time = log_top_targets([], 0.0)
        assert new_time > 0  # returns now
