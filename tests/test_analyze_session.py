"""Tests for util.analyze_session -- pure parsing/analysis functions.

Exercises the sub-analysis helpers that parse structured event dicts
without running the CLI main() or printing output.
"""

from __future__ import annotations

from util.analyze_session import (
    _failure_chains,
    _kill_cycles,
    _kill_droughts,
    _mana_utilization,
    _moving_windows,
    _recommendations,
    _routine_distribution,
    _spatial_analysis,
    _wander_effectiveness,
    analyze,
    load_events,
)

# ---------------------------------------------------------------------------
# Helpers -- build minimal event dicts
# ---------------------------------------------------------------------------


def _routine_start(t: float, routine: str, elapsed: float = 0.0) -> dict:
    return {"event": "routine_start", "t": t, "routine": routine, "elapsed": elapsed or t}


def _routine_end(t: float, routine: str, result: str, elapsed: float = 0.0) -> dict:
    return {"event": "routine_end", "t": t, "routine": routine, "result": result, "elapsed": elapsed or t}


def _fight_end(
    t: float,
    npc: str = "a_skeleton",
    duration: float = 20.0,
    mana_spent: int = 100,
    casts: int = 3,
    hp_start: float = 1.0,
    hp_end: float = 0.95,
    defeats: int = 1,
    elapsed: float = 0.0,
    **extra: object,
) -> dict:
    return {
        "event": "fight_end",
        "t": t,
        "npc": npc,
        "duration": duration,
        "mana_spent": mana_spent,
        "casts": casts,
        "hp_start": hp_start,
        "hp_end": hp_end,
        "defeats": defeats,
        "elapsed": elapsed or t,
        **extra,
    }


def _snapshot(
    t: float,
    x: float = 100.0,
    y: float = 200.0,
    camp_dist: float = 50.0,
    mana: int = 300,
    mana_max: int = 500,
    elapsed: float = 0.0,
    **extra: object,
) -> dict:
    return {
        "event": "snapshot",
        "t": t,
        "x": x,
        "y": y,
        "camp_dist": camp_dist,
        "mana": mana,
        "mana_max": mana_max,
        "elapsed": elapsed or t,
        **extra,
    }


# ===========================================================================
# _routine_distribution
# ===========================================================================


class TestRoutineDistribution:
    def test_basic_distribution(self) -> None:
        events = [
            _routine_start(0, "ACQUIRE"),
            _routine_end(5, "ACQUIRE", "SUCCESS"),
            _routine_start(5, "PULL"),
            _routine_end(8, "PULL", "SUCCESS"),
        ]
        times, counts, results = _routine_distribution(events)
        assert times["ACQUIRE"] == 5.0
        assert times["PULL"] == 3.0
        assert counts["ACQUIRE"] == 1
        assert counts["PULL"] == 1
        assert results["ACQUIRE"]["SUCCESS"] == 1

    def test_empty_events(self) -> None:
        times, counts, results = _routine_distribution([])
        assert len(times) == 0
        assert len(counts) == 0

    def test_multiple_invocations(self) -> None:
        events = [
            _routine_start(0, "ACQUIRE"),
            _routine_end(3, "ACQUIRE", "SUCCESS"),
            _routine_start(5, "ACQUIRE"),
            _routine_end(9, "ACQUIRE", "FAILURE"),
        ]
        times, counts, results = _routine_distribution(events)
        assert times["ACQUIRE"] == 7.0  # 3 + 4
        assert counts["ACQUIRE"] == 2
        assert results["ACQUIRE"]["SUCCESS"] == 1
        assert results["ACQUIRE"]["FAILURE"] == 1

    def test_end_without_start_ignored(self) -> None:
        events = [
            _routine_end(5, "ACQUIRE", "SUCCESS"),
        ]
        times, counts, results = _routine_distribution(events)
        assert times.get("ACQUIRE", 0) == 0
        assert results["ACQUIRE"]["SUCCESS"] == 1


# ===========================================================================
# _kill_cycles
# ===========================================================================


class TestKillCycles:
    def test_single_cycle(self) -> None:
        events = [
            _routine_start(0, "ACQUIRE"),
            _routine_end(3, "ACQUIRE", "SUCCESS"),
            _routine_start(3, "PULL"),
            _routine_end(5, "PULL", "SUCCESS"),
            _routine_start(5, "IN_COMBAT"),
            _routine_end(25, "IN_COMBAT", "SUCCESS"),
            _fight_end(25),
        ]
        cycles = _kill_cycles(events)
        assert len(cycles) == 1
        assert cycles[0]["ACQUIRE"] == 3.0
        assert cycles[0]["PULL"] == 2.0
        assert cycles[0]["IN_COMBAT"] == 20.0

    def test_idle_between_cycles(self) -> None:
        events = [
            _routine_start(0, "ACQUIRE"),
            _routine_end(2, "ACQUIRE", "SUCCESS"),
            _fight_end(5),
            # 10s gap
            _routine_start(15, "ACQUIRE"),
            _routine_end(18, "ACQUIRE", "SUCCESS"),
            _fight_end(20),
        ]
        cycles = _kill_cycles(events)
        assert len(cycles) == 2
        assert "_idle" in cycles[1]
        assert cycles[1]["_idle"] == 10.0  # 15 - 5

    def test_empty_events(self) -> None:
        assert _kill_cycles([]) == []


# ===========================================================================
# _failure_chains
# ===========================================================================


class TestFailureChains:
    def test_detects_chain_of_three(self) -> None:
        events = [
            _routine_end(10, "ACQUIRE", "FAILURE", elapsed=10),
            _routine_end(20, "ACQUIRE", "FAILURE", elapsed=20),
            _routine_end(30, "ACQUIRE", "FAILURE", elapsed=30),
        ]
        chains = _failure_chains(events)
        assert len(chains) == 1
        assert chains[0]["routine"] == "ACQUIRE"
        assert chains[0]["count"] == 3

    def test_chain_broken_by_success(self) -> None:
        events = [
            _routine_end(10, "ACQUIRE", "FAILURE", elapsed=10),
            _routine_end(20, "ACQUIRE", "FAILURE", elapsed=20),
            _routine_end(30, "ACQUIRE", "SUCCESS", elapsed=30),
            _routine_end(40, "ACQUIRE", "FAILURE", elapsed=40),
        ]
        chains = _failure_chains(events)
        # Only 2 consecutive failures before success -- not >= 3
        assert len(chains) == 0

    def test_no_failures(self) -> None:
        events = [
            _routine_end(10, "ACQUIRE", "SUCCESS", elapsed=10),
        ]
        chains = _failure_chains(events)
        assert len(chains) == 0

    def test_empty_events(self) -> None:
        assert _failure_chains([]) == []

    def test_trailing_chain_flushed(self) -> None:
        events = [
            _routine_end(10, "WANDER", "FAILURE", elapsed=10),
            _routine_end(20, "WANDER", "FAILURE", elapsed=20),
            _routine_end(30, "WANDER", "FAILURE", elapsed=30),
            _routine_end(40, "WANDER", "FAILURE", elapsed=40),
        ]
        chains = _failure_chains(events)
        assert len(chains) == 1
        assert chains[0]["count"] == 4

    def test_different_routine_breaks_chain(self) -> None:
        events = [
            _routine_end(10, "ACQUIRE", "FAILURE", elapsed=10),
            _routine_end(20, "ACQUIRE", "FAILURE", elapsed=20),
            _routine_end(30, "ACQUIRE", "FAILURE", elapsed=30),
            _routine_end(40, "PULL", "FAILURE", elapsed=40),
        ]
        chains = _failure_chains(events)
        assert len(chains) == 1
        assert chains[0]["routine"] == "ACQUIRE"
        assert chains[0]["count"] == 3


# ===========================================================================
# _moving_windows
# ===========================================================================


class TestMovingWindows:
    def test_single_window_short_session(self) -> None:
        fights = [_fight_end(30, elapsed=30), _fight_end(50, elapsed=50)]
        windows = _moving_windows(fights, 120, window_s=300)
        assert len(windows) >= 1
        assert windows[0]["defeats"] == 2

    def test_no_fights_returns_empty(self) -> None:
        assert _moving_windows([], 600) == []

    def test_short_duration_returns_empty(self) -> None:
        fights = [_fight_end(10, elapsed=10)]
        assert _moving_windows(fights, 30) == []

    def test_multiple_windows(self) -> None:
        fights = [
            _fight_end(60, elapsed=60),
            _fight_end(120, elapsed=120),
            _fight_end(400, elapsed=400),
        ]
        windows = _moving_windows(fights, 600, window_s=300)
        assert len(windows) >= 2
        # First window should contain the first two fights
        assert windows[0]["defeats"] >= 2


# ===========================================================================
# _kill_droughts
# ===========================================================================


class TestKillDroughts:
    def test_detects_drought(self) -> None:
        events = [_routine_start(30, "WANDER", elapsed=30)]
        fights = [_fight_end(10, elapsed=10), _fight_end(100, elapsed=100)]
        droughts = _kill_droughts(events, fights, 120, threshold_s=45)
        # Gap between fight at 10s and 100s is 90s > 45s
        assert len(droughts) >= 1
        assert droughts[0]["gap"] == 90.0

    def test_no_droughts_when_close(self) -> None:
        fights = [_fight_end(10, elapsed=10), _fight_end(30, elapsed=30)]
        droughts = _kill_droughts([], fights, 50, threshold_s=45)
        assert len(droughts) == 0

    def test_drought_context_includes_routines(self) -> None:
        events = [
            _routine_start(15, "WANDER", elapsed=15),
            _routine_start(20, "WANDER", elapsed=20),
        ]
        fights = [_fight_end(10, elapsed=10), _fight_end(80, elapsed=80)]
        droughts = _kill_droughts(events, fights, 100, threshold_s=45)
        assert len(droughts) >= 1
        assert "WANDER" in droughts[0]["context"]


# ===========================================================================
# _spatial_analysis
# ===========================================================================


class TestSpatialAnalysis:
    def test_basic_spatial(self) -> None:
        snapshots = [
            _snapshot(10, x=100, y=200, camp_dist=50),
            _snapshot(20, x=120, y=220, camp_dist=60),
            _snapshot(30, x=110, y=210, camp_dist=55),
        ]
        result = _spatial_analysis(snapshots)
        assert result["min_x"] == 100
        assert result["max_x"] == 120
        assert result["min_y"] == 200
        assert result["max_y"] == 220
        assert result["avg_camp"] == 55.0
        assert result["area"] == 20 * 20  # 400

    def test_empty_snapshots(self) -> None:
        assert _spatial_analysis([]) == {}

    def test_single_snapshot(self) -> None:
        result = _spatial_analysis([_snapshot(10, x=50, y=60, camp_dist=10)])
        assert result["area"] == 0
        assert result["avg_camp"] == 10


# ===========================================================================
# _mana_utilization
# ===========================================================================


class TestManaUtilization:
    def test_basic_mana_stats(self) -> None:
        snapshots = [
            _snapshot(10, mana=500, mana_max=500),  # full
            _snapshot(20, mana=250, mana_max=500),
            _snapshot(30, mana=500, mana_max=500),  # full
        ]
        fights = [_fight_end(15, casts=5, mana_spent=200)]
        result = _mana_utilization(snapshots, fights)
        # 2 out of 3 snapshots are at 100% mana (>=98%)
        assert abs(result["pct_full"] - 66.67) < 1.0
        assert result["total_casts"] == 5
        assert result["total_mana_spent"] == 200

    def test_empty_snapshots(self) -> None:
        assert _mana_utilization([], []) == {}

    def test_no_fights_zero_casts(self) -> None:
        snapshots = [_snapshot(10, mana=100, mana_max=500)]
        result = _mana_utilization(snapshots, [])
        assert result["total_casts"] == 0


# ===========================================================================
# _wander_effectiveness
# ===========================================================================


class TestWanderEffectiveness:
    def test_basic_wander(self) -> None:
        events = [
            _routine_start(0, "WANDER"),
            _routine_end(10, "WANDER", "SUCCESS"),
            # ACQUIRE success within 15s of wander end
            _routine_end(15, "ACQUIRE", "SUCCESS"),
        ]
        events[-1]["elapsed"] = 100  # session end marker
        result = _wander_effectiveness(events)
        assert result["total"] == 1
        assert result["led_to_kill"] == 1
        assert result["total_time"] == 10.0

    def test_no_wanders(self) -> None:
        events = [_routine_start(0, "ACQUIRE"), _routine_end(5, "ACQUIRE", "SUCCESS")]
        assert _wander_effectiveness(events) == {}

    def test_thrash_detection(self) -> None:
        events = [
            _routine_start(0, "WANDER"),
            _routine_end(0.5, "WANDER", "SUCCESS"),  # < 1.5s = thrash
            _routine_start(5, "WANDER"),
            _routine_end(5.3, "WANDER", "SUCCESS"),  # < 1.5s = thrash
        ]
        events[-1]["elapsed"] = 100
        result = _wander_effectiveness(events)
        assert result["thrash_count"] == 2


# ===========================================================================
# _recommendations
# ===========================================================================


class TestRecommendations:
    def test_zero_casts_recommendation(self) -> None:
        mana = {"total_casts": 0, "pct_full": 90}
        fights = [_fight_end(10)]
        recs = _recommendations([], fights, [], [], [], mana, {}, {}, [], 600)
        issues = [r["issue"] for r in recs]
        assert any("Zero spells" in i for i in issues)

    def test_high_mana_recommendation(self) -> None:
        mana = {"total_casts": 5, "pct_full": 95}
        fights = [_fight_end(10)]
        recs = _recommendations([], fights, [], [], [], mana, {}, {}, [], 600)
        issues = [r["issue"] for r in recs]
        assert any("Mana full" in i for i in issues)

    def test_no_recommendations_for_clean_session(self) -> None:
        mana = {"total_casts": 50, "pct_full": 30}
        recs = _recommendations([], [_fight_end(10)], [], [], [], mana, {}, {}, [], 600)
        # Should have no high-priority issues
        high = [r for r in recs if r["priority"] == "HIGH"]
        assert len(high) == 0

    def test_camp_drift_recommendation(self) -> None:
        spatial = {"max_camp": 400, "avg_camp": 200}
        mana = {"total_casts": 10, "pct_full": 30}
        recs = _recommendations([], [], [], [], [], mana, {}, spatial, [], 600)
        issues = [r["issue"] for r in recs]
        assert any("Drifted" in i for i in issues)

    def test_slow_fights_recommendation(self) -> None:
        fights = [_fight_end(10, duration=50, npc="a_skeleton")]
        mana = {"total_casts": 10, "pct_full": 30}
        recs = _recommendations([], fights, [], [], [], mana, {}, {}, [], 600)
        issues = [r["issue"] for r in recs]
        assert any("over 40s" in i for i in issues)

    def test_acquire_fail_rate_recommendation(self) -> None:
        events = [
            _routine_start(0, "ACQUIRE"),
            _routine_start(5, "ACQUIRE"),
            _routine_start(10, "ACQUIRE"),
        ]
        failures = [
            _routine_end(3, "ACQUIRE", "FAILURE", elapsed=3),
            _routine_end(8, "ACQUIRE", "FAILURE", elapsed=8),
        ]
        mana = {"total_casts": 10, "pct_full": 30}
        recs = _recommendations(events, [], failures, [], [], mana, {}, {}, [], 600)
        issues = [r["issue"] for r in recs]
        assert any("ACQUIRE fails" in i for i in issues)

    def test_recommendations_sorted_by_priority(self) -> None:
        mana = {"total_casts": 0, "pct_full": 95}
        spatial = {"max_camp": 400, "avg_camp": 200}
        fights = [_fight_end(10)]
        recs = _recommendations([], fights, [], [], [], mana, {}, spatial, [], 600)
        if len(recs) >= 2:
            priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            for i in range(len(recs) - 1):
                assert priority_order[recs[i]["priority"]] <= priority_order[recs[i + 1]["priority"]]


# ===========================================================================
# load_events
# ===========================================================================


class TestLoadEvents:
    def test_load_from_file(self, tmp_path) -> None:
        p = tmp_path / "test.jsonl"
        import json

        lines = [
            json.dumps({"event": "snapshot", "t": 1.0}),
            json.dumps({"event": "fight_end", "t": 2.0}),
        ]
        p.write_text("\n".join(lines) + "\n")
        events = load_events(str(p))
        assert len(events) == 2
        assert events[0]["event"] == "snapshot"

    def test_load_empty_file(self, tmp_path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        events = load_events(str(p))
        assert events == []


# ===========================================================================
# analyze -- integration smoke test
# ===========================================================================


class TestAnalyze:
    def test_empty_events(self) -> None:
        result = analyze([])
        assert result == {}

    def test_minimal_session(self) -> None:
        events = [
            _snapshot(10, elapsed=10),
            _routine_start(15, "ACQUIRE", elapsed=15),
            _routine_end(18, "ACQUIRE", "SUCCESS", elapsed=18),
            _fight_end(30, elapsed=30, defeats=1),
            _snapshot(40, elapsed=40),
        ]
        result = analyze(events)
        assert result["defeats"] == 1
        assert result["fights"] == 1
        assert result["duration_min"] > 0

    def test_session_with_failures(self) -> None:
        events = [
            _routine_start(0, "ACQUIRE", elapsed=0),
            _routine_end(3, "ACQUIRE", "FAILURE", elapsed=3),
            _routine_start(5, "ACQUIRE", elapsed=5),
            _routine_end(8, "ACQUIRE", "SUCCESS", elapsed=8),
            _fight_end(20, elapsed=20, defeats=1),
            _snapshot(30, elapsed=30),
        ]
        result = analyze(events)
        assert result["acquire_fails"] == 1
        assert result["defeats"] == 1
