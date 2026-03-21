"""Tests for util.session_reporter -- XP tracking and narrative builder.

Only the deterministic, isolatable paths are tested here.
write_session_report and periodic_snapshot have deep runtime dependencies
and are excluded.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.constants import XP_SCALE_MAX
from tests.factories import make_game_state
from util.session_reporter import SessionReporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner() -> SimpleNamespace:
    """Minimal BrainRunner stub so SessionReporter.__init__ succeeds."""
    return SimpleNamespace(
        _session_id="test-session-001",
        _current_zone="testzone",
        _brain=SimpleNamespace(_active_name="none"),
    )


def _make_ctx(
    *,
    xp_last_raw: int = 0,
    xp_gained_pct: float = 0.0,
    defeats: int = 0,
    deaths: int = 0,
    session_start: float = 0.0,
    last_fight_name: str = "",
    last_fight_id: int = 0,
    last_fight_x: float = 0.0,
    last_fight_y: float = 0.0,
    recent_kills: list | None = None,
    defeat_cycle_times: list | None = None,
) -> SimpleNamespace:
    """Minimal AgentContext mock for session reporter tests."""
    return SimpleNamespace(
        metrics=SimpleNamespace(
            xp_last_raw=xp_last_raw,
            xp_gained_pct=xp_gained_pct,
            session_start=session_start,
            flee_count=0,
            rest_count=0,
            total_casts=0,
        ),
        defeat_tracker=SimpleNamespace(
            defeats=defeats,
            xp_gains=0,
            last_fight_name=last_fight_name,
            last_fight_id=last_fight_id,
            last_fight_x=last_fight_x,
            last_fight_y=last_fight_y,
            recent_kills=recent_kills if recent_kills is not None else [],
            defeat_cycle_times=defeat_cycle_times if defeat_cycle_times is not None else [],
        ),
        player=SimpleNamespace(deaths=deaths),
        pet=SimpleNamespace(alive=True),
        record_kill=MagicMock(),
        diag=SimpleNamespace(phase_detector=None),
    )


def _make_reporter() -> SessionReporter:
    """Import and construct SessionReporter with a stub runner."""
    from util.session_reporter import SessionReporter

    return SessionReporter(_make_runner())


# ===========================================================================
# track_xp -- XP increase detection and defeat recording
# ===========================================================================


class TestTrackXpDetectsDefeat:
    """XP increase above xp_last_raw -> ctx.record_kill called."""

    def test_xp_increase_records_defeat_with_last_fight_info(self) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(
            xp_last_raw=100,
            last_fight_name="a_skeleton",
            last_fight_id=42,
            last_fight_x=10.0,
            last_fight_y=20.0,
        )
        state = make_game_state(xp_pct_raw=120)
        now = 1000.0

        reporter.track_xp(state, ctx, now)

        ctx.record_kill.assert_called_once()
        assert ctx.defeat_tracker.xp_gains == 1

    def test_xp_increase_falls_back_to_target_name(self) -> None:
        """When last_fight_name is empty, fall back to state.target."""
        reporter = _make_reporter()
        ctx = _make_ctx(xp_last_raw=100)
        from tests.factories import make_spawn

        target = make_spawn(spawn_id=77, name="a_bat", x=5.0, y=6.0)
        state = make_game_state(xp_pct_raw=130, target=target)
        now = 1000.0

        reporter.track_xp(state, ctx, now)

        ctx.record_kill.assert_called_once()

    def test_xp_increase_falls_back_to_unknown_no_target(self) -> None:
        """No last_fight and no target -> defeat_name='unknown', id=0."""
        reporter = _make_reporter()
        ctx = _make_ctx(xp_last_raw=100)
        state = make_game_state(xp_pct_raw=110, x=30.0, y=40.0)
        now = 1000.0

        reporter.track_xp(state, ctx, now)

        ctx.record_kill.assert_called_once()


class TestTrackXpNoDefeat:
    """XP unchanged or zero -> no defeat recorded."""

    def test_xp_same_no_defeat(self) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(xp_last_raw=100)
        state = make_game_state(xp_pct_raw=100)

        reporter.track_xp(state, ctx, 1000.0)

        ctx.record_kill.assert_not_called()
        assert ctx.defeat_tracker.xp_gains == 0

    def test_xp_zero_no_defeat(self) -> None:
        """If xp_pct_raw is 0, nothing should fire (even if last_raw > 0)."""
        reporter = _make_reporter()
        ctx = _make_ctx(xp_last_raw=100)
        state = make_game_state(xp_pct_raw=0)

        reporter.track_xp(state, ctx, 1000.0)

        ctx.record_kill.assert_not_called()

    def test_xp_last_raw_zero_no_defeat(self) -> None:
        """First tick: xp_last_raw == 0 -> skip detection even if xp_pct_raw > 0."""
        reporter = _make_reporter()
        ctx = _make_ctx(xp_last_raw=0)
        state = make_game_state(xp_pct_raw=150)

        reporter.track_xp(state, ctx, 1000.0)

        ctx.record_kill.assert_not_called()


class TestTrackXpDedup:
    """Recent kills within 5s are deduped -- no second record_kill."""

    def test_recent_kill_suppresses_duplicate(self) -> None:
        reporter = _make_reporter()
        now = 1000.0
        ctx = _make_ctx(
            xp_last_raw=100,
            last_fight_name="a_skeleton",
            last_fight_id=42,
            recent_kills=[(42, now - 2.0)],  # killed 2s ago
        )
        state = make_game_state(xp_pct_raw=120)

        reporter.track_xp(state, ctx, now)

        ctx.record_kill.assert_not_called()
        # xp_gains still increments (the gain is real, just deduped for recording)
        assert ctx.defeat_tracker.xp_gains == 1

    def test_old_kill_does_not_suppress(self) -> None:
        reporter = _make_reporter()
        now = 1000.0
        ctx = _make_ctx(
            xp_last_raw=100,
            last_fight_name="a_skeleton",
            last_fight_id=42,
            recent_kills=[(42, now - 10.0)],  # killed 10s ago -- outside 5s window
        )
        state = make_game_state(xp_pct_raw=120)

        reporter.track_xp(state, ctx, now)

        ctx.record_kill.assert_called_once()


class TestTrackXpLevelWrap:
    """XP wraps from high to low when leveling up (xp_pct_raw goes 300 -> 50)."""

    def test_level_wrap_accumulates_xp_correctly(self) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(xp_last_raw=300, xp_gained_pct=0.0)
        # XP dropped from 300 to 50 -- level wrap
        state = make_game_state(xp_pct_raw=50)

        reporter.track_xp(state, ctx, 1000.0)

        # No defeat recorded (xp went DOWN, not UP)
        ctx.record_kill.assert_not_called()

        # But xp_gained_pct should account for the wrap:
        #   (XP_SCALE_MAX - 300) / XP_SCALE_MAX * 100  +  50 / XP_SCALE_MAX * 100
        expected_remainder = (XP_SCALE_MAX - 300) / float(XP_SCALE_MAX) * 100
        expected_new = 50 / float(XP_SCALE_MAX) * 100
        expected = expected_remainder + expected_new
        assert abs(ctx.metrics.xp_gained_pct - expected) < 0.01

    def test_normal_xp_gain_accumulates(self) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(xp_last_raw=100, xp_gained_pct=0.0)
        state = make_game_state(xp_pct_raw=150)

        reporter.track_xp(state, ctx, 1000.0)

        expected = (150 - 100) / float(XP_SCALE_MAX) * 100
        assert abs(ctx.metrics.xp_gained_pct - expected) < 0.01


class TestTrackXpUpdatesLastRaw:
    """xp_last_raw is updated to current value after tracking."""

    def test_last_raw_updated(self) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(xp_last_raw=100)
        state = make_game_state(xp_pct_raw=200)

        reporter.track_xp(state, ctx, 1000.0)

        assert ctx.metrics.xp_last_raw == 200

    def test_last_raw_not_updated_for_zero(self) -> None:
        """xp_pct_raw == 0 -> xp_last_raw stays unchanged."""
        reporter = _make_reporter()
        ctx = _make_ctx(xp_last_raw=100)
        state = make_game_state(xp_pct_raw=0)

        reporter.track_xp(state, ctx, 1000.0)

        assert ctx.metrics.xp_last_raw == 100


class TestTrackXpLargeDeltaIgnored:
    """XP delta >= 100 is ignored (guards against misreads)."""

    def test_delta_100_no_defeat(self) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(xp_last_raw=50)
        state = make_game_state(xp_pct_raw=200)  # delta=150 >= 100

        reporter.track_xp(state, ctx, 1000.0)

        # xp_gains increments (it checks delta < 100 *after* incrementing)
        # Actually, looking at the code: the 0 < xp_delta < 100 check
        # gates xp_gains increment as well.
        assert ctx.defeat_tracker.xp_gains == 0
        ctx.record_kill.assert_not_called()


# ===========================================================================
# _build_narrative -- phase history formatting
# ===========================================================================


class TestBuildNarrativeNoPhases:
    """No phase detector -> single-phase fallback."""

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "B+"})
    def test_no_phases_with_defeats(self, _mock_scorecard: MagicMock) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(defeats=10, deaths=1, session_start=0.0)
        elapsed = 3600.0  # 1 hour

        lines = reporter._build_narrative(ctx, elapsed)

        assert len(lines) >= 2
        assert "GRINDING" in lines[0]
        assert "10 defeats" in lines[0]
        assert "SUMMARY" in lines[-1]
        assert "10 defeats" in lines[-1]
        assert "1 death," in lines[-1]
        assert "grade B+" in lines[-1]

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "?"})
    def test_no_phases_no_defeats(self, _mock_scorecard: MagicMock) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(defeats=0, deaths=0, session_start=0.0)
        elapsed = 600.0  # 10 min

        lines = reporter._build_narrative(ctx, elapsed)

        assert "SESSION" in lines[0]
        assert "no defeats" in lines[0]
        assert "SUMMARY" in lines[-1]
        assert "0 defeats" in lines[-1]


class TestBuildNarrativeWithPhases:
    """Phase detector present -> multi-line narrative."""

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "A"})
    def test_grinding_phase_with_defeats(self, _mock_scorecard: MagicMock) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(
            defeats=20,
            deaths=0,
            session_start=1000.0,
            defeat_cycle_times=[30.0, 35.0, 28.0, 32.0],
        )
        # Phase: grinding, started at session_start, 1800s duration, 20 defeats, 40 dph
        phase_history = [("grinding", 1000.0, 1800.0, 20, 40.0)]
        ctx.diag = SimpleNamespace(
            phase_detector=SimpleNamespace(
                finalize=MagicMock(),
                history=phase_history,
            ),
        )
        elapsed = 1800.0

        lines = reporter._build_narrative(ctx, elapsed)

        assert "GRINDING" in lines[0]
        assert "20 defeats" in lines[0]
        assert "40 dph" in lines[0]
        assert "SUMMARY" in lines[-1]

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "C"})
    def test_resting_phase(self, _mock_scorecard: MagicMock) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(defeats=5, deaths=0, session_start=0.0)
        phase_history = [("resting", 0.0, 120.0, 0, 0.0)]
        ctx.diag = SimpleNamespace(
            phase_detector=SimpleNamespace(
                finalize=MagicMock(),
                history=phase_history,
            ),
        )

        lines = reporter._build_narrative(ctx, 120.0)

        assert "RESTING" in lines[0]
        assert "mana recovery" in lines[0]

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "?"})
    def test_incident_phase(self, _mock_scorecard: MagicMock) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(defeats=3, deaths=1, session_start=0.0)
        phase_history = [("incident", 0.0, 60.0, 0, 0.0)]
        ctx.diag = SimpleNamespace(
            phase_detector=SimpleNamespace(
                finalize=MagicMock(),
                history=phase_history,
            ),
        )

        lines = reporter._build_narrative(ctx, 60.0)

        assert "INCIDENT" in lines[0]
        assert "flee/death" in lines[0]

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "?"})
    def test_startup_phase(self, _mock_scorecard: MagicMock) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(defeats=0, deaths=0, session_start=0.0)
        phase_history = [("startup", 0.0, 30.0, 0, 0.0)]
        ctx.diag = SimpleNamespace(
            phase_detector=SimpleNamespace(
                finalize=MagicMock(),
                history=phase_history,
            ),
        )

        lines = reporter._build_narrative(ctx, 30.0)

        assert "STARTUP" in lines[0]
        assert "warmup" in lines[0]
        assert "pet alive" in lines[0]

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "?"})
    def test_idle_phase(self, _mock_scorecard: MagicMock) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(defeats=0, deaths=0, session_start=0.0)
        phase_history = [("idle", 0.0, 90.0, 0, 0.0)]
        ctx.diag = SimpleNamespace(
            phase_detector=SimpleNamespace(
                finalize=MagicMock(),
                history=phase_history,
            ),
        )

        lines = reporter._build_narrative(ctx, 90.0)

        assert "IDLE" in lines[0]
        assert "no activity" in lines[0]

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "?"})
    def test_unknown_phase_type(self, _mock_scorecard: MagicMock) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(defeats=0, deaths=0, session_start=0.0)
        phase_history = [("custom_phase", 0.0, 45.0, 0, 0.0)]
        ctx.diag = SimpleNamespace(
            phase_detector=SimpleNamespace(
                finalize=MagicMock(),
                history=phase_history,
            ),
        )

        lines = reporter._build_narrative(ctx, 45.0)

        assert "CUSTOM_PHASE" in lines[0]
        assert "45s" in lines[0]

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "A+"})
    def test_grinding_with_notable_subevent(self, _mock_scorecard: MagicMock) -> None:
        """Grinding with >3 defeats produces a 'Notable' sub-line."""
        reporter = _make_reporter()
        ctx = _make_ctx(
            defeats=5,
            deaths=0,
            session_start=0.0,
            defeat_cycle_times=[25.0, 30.0, 28.0, 27.0, 32.0],
        )
        phase_history = [("grinding", 0.0, 600.0, 5, 30.0)]
        ctx.diag = SimpleNamespace(
            phase_detector=SimpleNamespace(
                finalize=MagicMock(),
                history=phase_history,
            ),
        )

        lines = reporter._build_narrative(ctx, 600.0)

        notable_lines = [l for l in lines if "Notable" in l]
        assert len(notable_lines) == 1
        assert "avg" in notable_lines[0]
        assert "s/cycle" in notable_lines[0]

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "B"})
    def test_grinding_no_defeats_format(self, _mock_scorecard: MagicMock) -> None:
        """Grinding phase with 0 defeats shows different format."""
        reporter = _make_reporter()
        ctx = _make_ctx(defeats=0, deaths=0, session_start=0.0)
        phase_history = [("grinding", 0.0, 300.0, 0, 0.0)]
        ctx.diag = SimpleNamespace(
            phase_detector=SimpleNamespace(
                finalize=MagicMock(),
                history=phase_history,
            ),
        )

        lines = reporter._build_narrative(ctx, 300.0)

        assert "GRINDING" in lines[0]
        assert "no defeats" in lines[0]


class TestBuildNarrativeSummary:
    """Summary line always present, pluralizes deaths correctly."""

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "A"})
    def test_summary_singular_death(self, _mock_scorecard: MagicMock) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(defeats=5, deaths=1, session_start=0.0)

        lines = reporter._build_narrative(ctx, 3600.0)

        summary = lines[-1]
        assert "1 death," in summary  # singular
        assert "deaths" not in summary

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "A"})
    def test_summary_plural_deaths(self, _mock_scorecard: MagicMock) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(defeats=5, deaths=3, session_start=0.0)

        lines = reporter._build_narrative(ctx, 3600.0)

        summary = lines[-1]
        assert "3 deaths" in summary

    @patch("util.session_reporter.compute_scorecard", side_effect=TypeError("broken"))
    def test_summary_scorecard_failure_uses_question_mark(self, _mock_scorecard: MagicMock) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(defeats=0, deaths=0, session_start=0.0)

        lines = reporter._build_narrative(ctx, 600.0)

        assert "grade ?" in lines[-1]


class TestBuildNarrativeMultiPhase:
    """Multi-phase session produces multiple narrative lines."""

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "B"})
    def test_three_phases(self, _mock_scorecard: MagicMock) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(
            defeats=15,
            deaths=1,
            session_start=0.0,
            defeat_cycle_times=[30.0] * 15,
        )
        phase_history = [
            ("startup", 0.0, 30.0, 0, 0.0),
            ("grinding", 30.0, 1770.0, 15, 30.5),
            ("resting", 1800.0, 120.0, 0, 0.0),
        ]
        ctx.diag = SimpleNamespace(
            phase_detector=SimpleNamespace(
                finalize=MagicMock(),
                history=phase_history,
            ),
        )

        lines = reporter._build_narrative(ctx, 1920.0)

        # 3 phase lines + notable sub-event (15 > 3) + summary
        assert "STARTUP" in lines[0]
        assert "GRINDING" in lines[1]
        assert "RESTING" in lines[3] or "RESTING" in lines[2]
        assert "SUMMARY" in lines[-1]
        assert "15 defeats" in lines[-1]


# ===========================================================================
# SessionReporter.__init__ -- derivative tracking defaults
# ===========================================================================


class TestBuildNarrativeStartupNoPet:
    """Startup phase with pet not alive shows 'no pet'."""

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "?"})
    def test_startup_no_pet(self, _mock_scorecard: MagicMock) -> None:
        reporter = _make_reporter()
        ctx = _make_ctx(defeats=0, deaths=0, session_start=0.0)
        ctx.pet = SimpleNamespace(alive=False)
        phase_history = [("startup", 0.0, 30.0, 0, 0.0)]
        ctx.diag = SimpleNamespace(
            phase_detector=SimpleNamespace(
                finalize=MagicMock(),
                history=phase_history,
            ),
        )

        lines = reporter._build_narrative(ctx, 30.0)

        assert "STARTUP" in lines[0]
        assert "no pet" in lines[0]


class TestBuildNarrativeGrindingCycleAvg:
    """Grinding with defeats includes avg cycle time in line when defeat_cycle_times available."""

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "B"})
    def test_grinding_with_empty_cycle_times(self, _mock_scorecard: MagicMock) -> None:
        """Grinding with defeats but empty cycle times omits avg cycle string."""
        reporter = _make_reporter()
        ctx = _make_ctx(
            defeats=3,
            deaths=0,
            session_start=0.0,
            defeat_cycle_times=[],
        )
        phase_history = [("grinding", 0.0, 600.0, 3, 18.0)]
        ctx.diag = SimpleNamespace(
            phase_detector=SimpleNamespace(
                finalize=MagicMock(),
                history=phase_history,
            ),
        )

        lines = reporter._build_narrative(ctx, 600.0)

        assert "GRINDING" in lines[0]
        assert "3 defeats" in lines[0]
        # No "avg" in line since defeat_cycle_times is empty
        assert "avg" not in lines[0]

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "B"})
    def test_grinding_notable_subevent_not_triggered_at_3(self, _mock_scorecard: MagicMock) -> None:
        """Grinding with exactly 3 defeats does NOT produce 'Notable' sub-line (threshold >3)."""
        reporter = _make_reporter()
        ctx = _make_ctx(
            defeats=3,
            deaths=0,
            session_start=0.0,
            defeat_cycle_times=[30.0, 28.0, 32.0],
        )
        phase_history = [("grinding", 0.0, 600.0, 3, 18.0)]
        ctx.diag = SimpleNamespace(
            phase_detector=SimpleNamespace(
                finalize=MagicMock(),
                history=phase_history,
            ),
        )

        lines = reporter._build_narrative(ctx, 600.0)

        notable = [l for l in lines if "Notable" in l]
        assert len(notable) == 0


class TestTrackXpEdgeCases:
    """Additional edge cases for track_xp."""

    def test_xp_delta_exactly_100_not_recorded(self) -> None:
        """XP delta of exactly 100 should NOT be recorded (guard: 0 < delta < 100)."""
        reporter = _make_reporter()
        ctx = _make_ctx(xp_last_raw=100, last_fight_name="a_skeleton", last_fight_id=1)
        state = make_game_state(xp_pct_raw=200)  # delta=100, not < 100

        reporter.track_xp(state, ctx, 1000.0)

        assert ctx.defeat_tracker.xp_gains == 0
        ctx.record_kill.assert_not_called()

    def test_xp_last_raw_updated_even_without_defeat(self) -> None:
        """xp_last_raw updates to current even when xp went down (level wrap)."""
        reporter = _make_reporter()
        ctx = _make_ctx(xp_last_raw=300, xp_gained_pct=0.0)
        state = make_game_state(xp_pct_raw=50)

        reporter.track_xp(state, ctx, 1000.0)

        # xp_last_raw updates since xp_pct_raw > 0
        assert ctx.metrics.xp_last_raw == 50

    def test_xp_zero_current_and_previous_no_crash(self) -> None:
        """Both xp_pct_raw=0 and xp_last_raw=0 should not crash."""
        reporter = _make_reporter()
        ctx = _make_ctx(xp_last_raw=0, xp_gained_pct=0.0)
        state = make_game_state(xp_pct_raw=0)

        reporter.track_xp(state, ctx, 1000.0)

        assert ctx.metrics.xp_gained_pct == 0.0
        ctx.record_kill.assert_not_called()


class TestInit:
    def test_initial_derivative_state(self) -> None:
        reporter = _make_reporter()
        assert reporter._prev_mana == 0
        assert reporter._prev_mana_time == 0.0
        assert reporter._prev_camp_dist == 0.0
        assert reporter._prev_camp_dist_time == 0.0


# ===========================================================================
# write_session_report -- JSON file output
# ===========================================================================


def _make_report_ctx(
    *,
    defeats: int = 10,
    deaths: int = 0,
    flees: int = 1,
    rests: int = 2,
    total_casts: int = 50,
    xp_gained_pct: float = 2.5,
    session_start: float = 0.0,
) -> SimpleNamespace:
    """Build a comprehensive ctx mock for write_session_report."""
    import threading

    return SimpleNamespace(
        metrics=SimpleNamespace(
            session_start=session_start or (time.time() - 3600),
            xp_gained_pct=xp_gained_pct,
            xp_last_raw=100,
            flee_count=flees,
            rest_count=rests,
            total_casts=total_casts,
            acquire_tab_totals=[3, 4, 5],
            acquire_invalid_tabs=1,
            consecutive_acquire_fails=0,
            snapshot_collections=lambda: {
                "routine_time": {"IN_COMBAT": 600.0, "ACQUIRE": 200.0},
                "routine_counts": {"IN_COMBAT": 10, "ACQUIRE": 15},
                "routine_failures": {"ACQUIRE": 3},
                "total_cycle_times": [30.0, 35.0, 28.0],
            },
            trim_lists=MagicMock(),
            record_xp_sample=MagicMock(),
            xp_per_hour=MagicMock(return_value=1.5),
            time_to_level=MagicMock(return_value=None),
        ),
        defeat_tracker=SimpleNamespace(
            defeats=defeats,
            xp_gains=0,
            last_fight_name="",
            last_fight_id=0,
            last_fight_x=0.0,
            last_fight_y=0.0,
            recent_kills=[],
            defeat_cycle_times=[30.0, 35.0, 28.0],
            defeat_rate_window=MagicMock(return_value=10.0),
            last_kill_age=MagicMock(return_value=30.0),
        ),
        player=SimpleNamespace(deaths=deaths),
        pet=SimpleNamespace(alive=True),
        record_kill=MagicMock(),
        lock=threading.Lock(),
        inventory=SimpleNamespace(loot_count=5),
        fight_history=None,
        diag=SimpleNamespace(
            phase_detector=None,
            metrics=None,
            invariants=None,
            tick_overbudget_count=0,
        ),
    )


class TestWriteSessionReport:
    """Tests for the JSON session report writer."""

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "A", "overall": 85})
    @patch("util.session_reporter.get_stuck_event_count", return_value=3)
    def test_writes_valid_json(self, _mock_stuck, _mock_score, tmp_path) -> None:
        import json

        reporter = _make_reporter()
        ctx = _make_report_ctx()
        reporter.write_session_report(ctx, str(tmp_path))

        report_path = tmp_path / "test-session-001_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["v"] == 1
        assert data["defeats"] == 10
        assert data["deaths"] == 0
        assert data["flees"] == 1
        assert data["rests"] == 2
        assert data["casts"] == 50
        assert "scorecard" in data
        assert data["scorecard"]["grade"] == "A"
        assert data["stuck_events"] == 3

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "B"})
    @patch("util.session_reporter.get_stuck_event_count", return_value=0)
    def test_includes_routine_time(self, _mock_stuck, _mock_score, tmp_path) -> None:
        import json

        reporter = _make_reporter()
        ctx = _make_report_ctx()
        reporter.write_session_report(ctx, str(tmp_path))

        data = json.loads((tmp_path / "test-session-001_report.json").read_text())
        assert "routine_time" in data
        assert "IN_COMBAT" in data["routine_time"]
        assert "routine_counts" in data
        assert "routine_failures" in data

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "C"})
    @patch("util.session_reporter.get_stuck_event_count", return_value=0)
    def test_combat_efficiency_fields(self, _mock_stuck, _mock_score, tmp_path) -> None:
        import json

        reporter = _make_reporter()
        ctx = _make_report_ctx(defeats=10, total_casts=50)
        reporter.write_session_report(ctx, str(tmp_path))

        data = json.loads((tmp_path / "test-session-001_report.json").read_text())
        assert "casts_per_kill" in data
        assert data["casts_per_kill"] == 5.0
        assert "avg_cycle_s" in data

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "D"})
    @patch("util.session_reporter.get_stuck_event_count", return_value=0)
    def test_acquire_stats(self, _mock_stuck, _mock_score, tmp_path) -> None:
        import json

        reporter = _make_reporter()
        ctx = _make_report_ctx()
        reporter.write_session_report(ctx, str(tmp_path))

        data = json.loads((tmp_path / "test-session-001_report.json").read_text())
        assert "avg_tabs_per_acquire" in data
        assert data["acquire_invalid_tabs"] == 1

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "A"})
    @patch("util.session_reporter.get_stuck_event_count", return_value=0)
    def test_tick_budget_included_when_overbudget(self, _mock_stuck, _mock_score, tmp_path) -> None:
        import json

        reporter = _make_reporter()
        ctx = _make_report_ctx()
        ctx.diag.tick_overbudget_count = 5
        ctx.diag.tick_overbudget_max_ms = 120.5
        ctx.diag.tick_overbudget_last_routine = "IN_COMBAT"
        reporter.write_session_report(ctx, str(tmp_path))

        data = json.loads((tmp_path / "test-session-001_report.json").read_text())
        assert "tick_budget" in data
        assert data["tick_budget"]["overbudget_count"] == 5
        assert data["tick_budget"]["worst_routine"] == "IN_COMBAT"

    @patch("util.session_reporter.compute_scorecard", side_effect=TypeError("broken"))
    @patch("util.session_reporter.get_stuck_event_count", return_value=0)
    def test_scorecard_failure_still_writes(self, _mock_stuck, _mock_score, tmp_path) -> None:
        import json

        reporter = _make_reporter()
        ctx = _make_report_ctx()
        reporter.write_session_report(ctx, str(tmp_path))

        data = json.loads((tmp_path / "test-session-001_report.json").read_text())
        # Report still written, just no scorecard key
        assert data["defeats"] == 10
        assert "scorecard" not in data

    @patch("util.session_reporter.compute_scorecard", return_value={"grade": "A"})
    @patch("util.session_reporter.get_stuck_event_count", return_value=0)
    def test_xp_fields(self, _mock_stuck, _mock_score, tmp_path) -> None:
        import json

        reporter = _make_reporter()
        ctx = _make_report_ctx(xp_gained_pct=3.5)
        reporter.write_session_report(ctx, str(tmp_path))

        data = json.loads((tmp_path / "test-session-001_report.json").read_text())
        assert data["xp_gained_pct"] == 3.5
        assert "xp_per_hr" in data
