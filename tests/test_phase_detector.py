"""Tests for util.phase_detector -- operational phase classification."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

from tests.factories import make_game_state
from util.phase_detector import PhaseDetector


def _make_ctx(
    dead: bool = False,
    engaged: bool = False,
    defeats: int = 10,
    last_kill_age: float = 5.0,
    flee_active: bool = False,
    rest_active: bool = False,
) -> SimpleNamespace:
    """Minimal AgentContext mock for phase detector tests."""
    rule_eval: dict[str, str] = {}
    if flee_active:
        rule_eval["FLEE"] = "YES"
    else:
        rule_eval["FLEE"] = "no"
    if rest_active:
        rule_eval["REST"] = "YES"
    else:
        rule_eval["REST"] = "no"

    return SimpleNamespace(
        player=SimpleNamespace(dead=dead),
        combat=SimpleNamespace(engaged=engaged),
        defeat_tracker=SimpleNamespace(
            defeats=defeats,
            last_kill_age=lambda: last_kill_age,
        ),
        diag=SimpleNamespace(last_rule_evaluation=rule_eval),
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_starts_in_startup(self) -> None:
        pd = PhaseDetector()
        assert pd.current_phase == "startup"

    def test_empty_history(self) -> None:
        pd = PhaseDetector()
        assert pd.history == []


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------


class TestClassify:
    def test_startup_within_60s(self) -> None:
        pd = PhaseDetector()
        state = make_game_state()
        ctx = _make_ctx()
        now = pd._session_start + 30  # 30s into session
        result = pd._classify(state, ctx, now)
        assert result == "startup"

    def test_grinding_after_startup(self) -> None:
        pd = PhaseDetector()
        pd._current_phase = "grinding"  # past startup
        state = make_game_state()
        ctx = _make_ctx(defeats=5, last_kill_age=10.0)
        now = pd._session_start + 120
        result = pd._classify(state, ctx, now)
        assert result == "grinding"

    def test_incident_on_death(self) -> None:
        pd = PhaseDetector()
        pd._current_phase = "grinding"
        state = make_game_state()
        ctx = _make_ctx(dead=True)
        now = pd._session_start + 120
        result = pd._classify(state, ctx, now)
        assert result == "incident"

    def test_incident_on_flee(self) -> None:
        pd = PhaseDetector()
        pd._current_phase = "grinding"
        state = make_game_state()
        ctx = _make_ctx(flee_active=True)
        now = pd._session_start + 120
        result = pd._classify(state, ctx, now)
        assert result == "incident"

    def test_resting_on_low_mana(self) -> None:
        pd = PhaseDetector()
        pd._current_phase = "grinding"
        state = make_game_state(mana_current=100, mana_max=500)  # 20% mana
        ctx = _make_ctx(rest_active=True)
        now = pd._session_start + 120
        result = pd._classify(state, ctx, now)
        assert result == "resting"

    def test_rest_ignored_at_high_mana(self) -> None:
        pd = PhaseDetector()
        pd._current_phase = "grinding"
        state = make_game_state(mana_current=400, mana_max=500)  # 80% mana
        ctx = _make_ctx(rest_active=True)
        now = pd._session_start + 120
        result = pd._classify(state, ctx, now)
        assert result == "grinding"  # brief rest between defeats, not a rest phase

    def test_idle_no_defeats_for_120s(self) -> None:
        pd = PhaseDetector()
        pd._current_phase = "grinding"
        state = make_game_state()
        ctx = _make_ctx(defeats=5, last_kill_age=150.0, engaged=False)
        now = pd._session_start + 300
        result = pd._classify(state, ctx, now)
        assert result == "idle"

    def test_not_idle_when_engaged(self) -> None:
        pd = PhaseDetector()
        pd._current_phase = "grinding"
        state = make_game_state()
        ctx = _make_ctx(defeats=5, last_kill_age=150.0, engaged=True)
        now = pd._session_start + 300
        result = pd._classify(state, ctx, now)
        assert result == "grinding"

    def test_not_idle_with_zero_defeats(self) -> None:
        pd = PhaseDetector()
        pd._current_phase = "grinding"
        state = make_game_state()
        ctx = _make_ctx(defeats=0, last_kill_age=999.0)
        now = pd._session_start + 300
        result = pd._classify(state, ctx, now)
        assert result == "grinding"  # no defeats yet, still ramping up

    def test_startup_holds_even_on_death(self) -> None:
        """Startup phase holds for 60s regardless of other signals."""
        pd = PhaseDetector()
        state = make_game_state()
        ctx = _make_ctx(dead=True)
        now = pd._session_start + 10  # still in startup window
        result = pd._classify(state, ctx, now)
        assert result == "startup"

    def test_resting_stays_resting(self) -> None:
        """Once in resting phase, REST active keeps it there."""
        pd = PhaseDetector()
        pd._current_phase = "resting"
        state = make_game_state(mana_current=400, mana_max=500)  # high mana now
        ctx = _make_ctx(rest_active=True)
        now = pd._session_start + 200
        result = pd._classify(state, ctx, now)
        assert result == "resting"


# ---------------------------------------------------------------------------
# Phase transitions (check method)
# ---------------------------------------------------------------------------


class TestCheck:
    @patch("util.phase_detector.log_event")
    def test_transition_emits_event(self, mock_log) -> None:
        pd = PhaseDetector()
        state = make_game_state()
        ctx = _make_ctx(dead=True)
        now = pd._session_start + 120
        pd._current_phase = "grinding"
        pd._phase_start = now - 60
        pd.check(state, ctx, now)
        assert pd.current_phase == "incident"
        assert mock_log.called
        _, kwargs = mock_log.call_args
        assert kwargs["old_phase"] == "grinding"
        assert kwargs["new_phase"] == "incident"

    @patch("util.phase_detector.log_event")
    def test_no_transition_no_event(self, mock_log) -> None:
        pd = PhaseDetector()
        pd._current_phase = "grinding"
        state = make_game_state()
        ctx = _make_ctx()
        now = pd._session_start + 120
        pd.check(state, ctx, now)
        assert not mock_log.called

    @patch("util.phase_detector.log_event")
    def test_history_appended_on_transition(self, mock_log) -> None:
        pd = PhaseDetector()
        pd._current_phase = "grinding"
        pd._phase_start = time.time() - 60
        state = make_game_state()
        ctx = _make_ctx(dead=True)
        now = time.time()
        pd.check(state, ctx, now)
        assert len(pd.history) == 1
        phase, start, duration, defeats, dph = pd.history[0]
        assert phase == "grinding"


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------


class TestFinalize:
    def test_finalize_appends_current_phase(self) -> None:
        pd = PhaseDetector()
        pd._current_phase = "grinding"
        pd._phase_start = time.time() - 120
        pd._phase_kills = 3
        ctx = _make_ctx(defeats=10)
        pd.finalize(ctx)
        assert len(pd.history) == 1
        phase, _, _, defeats_in, _ = pd.history[0]
        assert phase == "grinding"
        assert defeats_in == 7  # 10 - 3
