"""Tests for util.cycle_tracker -- composite narrative events for defeat cycles."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from tests.factories import make_game_state
from util.cycle_tracker import CycleTracker


def _make_ctx(cycle_id: int = 1, defeats: int = 5, session_start: float = 0.0) -> SimpleNamespace:
    """Minimal AgentContext mock for cycle tracker tests."""
    return SimpleNamespace(
        defeat_tracker=SimpleNamespace(cycle_id=cycle_id, defeats=defeats),
        metrics=SimpleNamespace(session_start=session_start),
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_starts_inactive(self) -> None:
        t = CycleTracker()
        assert t._active is False
        assert t.cycle_id == 0


# ---------------------------------------------------------------------------
# Acquire starts/abandons cycle
# ---------------------------------------------------------------------------


class TestAcquire:
    def test_acquire_success_starts_cycle(self) -> None:
        t = CycleTracker()
        state = make_game_state(mana_current=400, x=10.0, y=20.0)
        ctx = _make_ctx(cycle_id=7)
        t.on_routine_end("ACQUIRE", "success", state, ctx, {"tabs": 3, "target": "a_skeleton"})
        assert t._active is True
        assert t.cycle_id == 7
        assert t._acquire_tabs == 3
        assert t._target_name == "a_skeleton"
        assert t._mana_start == 400

    def test_acquire_failure_stays_inactive(self) -> None:
        t = CycleTracker()
        state = make_game_state()
        ctx = _make_ctx()
        t.on_routine_end("ACQUIRE", "failure", state, ctx)
        assert t._active is False

    def test_acquire_failure_deactivates_existing_cycle(self) -> None:
        t = CycleTracker()
        state = make_game_state()
        ctx = _make_ctx(cycle_id=1)
        t.on_routine_end("ACQUIRE", "success", state, ctx)
        assert t._active is True
        t.on_routine_end("ACQUIRE", "failure", state, ctx)
        assert t._active is False


# ---------------------------------------------------------------------------
# Pull phase
# ---------------------------------------------------------------------------


class TestPull:
    def test_pull_success_records_data(self) -> None:
        t = CycleTracker()
        state = make_game_state()
        ctx = _make_ctx()
        t.on_routine_end("ACQUIRE", "success", state, ctx)
        t.on_routine_end(
            "PULL", "success", state, ctx, {"strategy": "dot", "duration": 3.5, "dot_retries": 1}
        )
        assert t._active is True
        assert t._pull_strategy == "dot"
        assert t._pull_duration == 3.5
        assert t._pull_dot_retries == 1

    def test_pull_failure_abandons_cycle(self) -> None:
        t = CycleTracker()
        state = make_game_state()
        ctx = _make_ctx()
        t.on_routine_end("ACQUIRE", "success", state, ctx)
        t.on_routine_end("PULL", "failure", state, ctx)
        assert t._active is False

    def test_pull_without_acquire_is_noop(self) -> None:
        t = CycleTracker()
        state = make_game_state()
        ctx = _make_ctx()
        t.on_routine_end("PULL", "success", state, ctx)
        assert t._active is False

    def test_pull_fills_target_name_if_missing(self) -> None:
        t = CycleTracker()
        state = make_game_state()
        ctx = _make_ctx()
        t.on_routine_end("ACQUIRE", "success", state, ctx, {"tabs": 1})
        assert t._target_name == ""
        t.on_routine_end("PULL", "success", state, ctx, {"target": "a_bat"})
        assert t._target_name == "a_bat"


# ---------------------------------------------------------------------------
# Combat completes cycle
# ---------------------------------------------------------------------------


class TestCombat:
    @patch("util.cycle_tracker.log_event")
    def test_combat_success_emits_event(self, mock_log_event) -> None:
        t = CycleTracker()
        state = make_game_state(mana_current=300, x=100.0, y=200.0)
        ctx = _make_ctx(cycle_id=5, defeats=10, session_start=0.0)
        t.on_routine_end("ACQUIRE", "success", state, ctx, {"target": "a_skeleton"})
        t.on_routine_end("PULL", "success", state, ctx)
        t.on_routine_end(
            "IN_COMBAT",
            "success",
            state,
            ctx,
            {
                "duration": 15.0,
                "casts": 5,
                "mana_spent": 200,
                "hp_delta": -0.1,
                "adds": 0,
                "strategy": "pet_tank",
            },
        )
        assert t._active is False
        assert mock_log_event.called
        call_kwargs = mock_log_event.call_args
        assert call_kwargs[1]["npc"] == "a_skeleton"
        assert call_kwargs[1]["fight_duration"] == 15.0
        assert call_kwargs[1]["fight_casts"] == 5

    def test_combat_failure_ends_cycle_without_emit(self) -> None:
        t = CycleTracker()
        state = make_game_state()
        ctx = _make_ctx()
        t.on_routine_end("ACQUIRE", "success", state, ctx)
        t.on_routine_end("IN_COMBAT", "failure", state, ctx)
        assert t._active is False

    def test_combat_without_acquire_is_noop(self) -> None:
        t = CycleTracker()
        state = make_game_state()
        ctx = _make_ctx()
        t.on_routine_end("IN_COMBAT", "success", state, ctx)
        assert t._active is False


# ---------------------------------------------------------------------------
# Full cycle
# ---------------------------------------------------------------------------


class TestFullCycle:
    @patch("util.cycle_tracker.log_event")
    def test_acquire_pull_combat_emits_complete(self, mock_log_event) -> None:
        t = CycleTracker()
        state = make_game_state(mana_current=500)
        ctx = _make_ctx(cycle_id=3, defeats=8)
        t.on_routine_end("ACQUIRE", "success", state, ctx, {"tabs": 2, "target": "a_bat"})
        t.on_routine_end("PULL", "success", state, ctx, {"strategy": "pet", "duration": 2.0})
        t.on_routine_end(
            "IN_COMBAT", "success", state, ctx, {"duration": 10.0, "casts": 3, "mana_spent": 100}
        )
        assert mock_log_event.call_count == 1
        assert mock_log_event.call_args[1]["cycle_id"] == 3

    @patch("util.cycle_tracker.log_event")
    def test_sequential_cycles_increment(self, mock_log_event) -> None:
        t = CycleTracker()
        state = make_game_state()
        for cid in (1, 2, 3):
            ctx = _make_ctx(cycle_id=cid)
            t.on_routine_end("ACQUIRE", "success", state, ctx, {"target": f"mob_{cid}"})
            t.on_routine_end("IN_COMBAT", "success", state, ctx, {"duration": 5.0})
        assert mock_log_event.call_count == 3


# ---------------------------------------------------------------------------
# Unrelated routines
# ---------------------------------------------------------------------------


class TestUnrelatedRoutines:
    def test_wander_ignored(self) -> None:
        t = CycleTracker()
        state = make_game_state()
        ctx = _make_ctx()
        t.on_routine_end("WANDER", "success", state, ctx)
        assert t._active is False

    def test_rest_ignored(self) -> None:
        t = CycleTracker()
        state = make_game_state()
        ctx = _make_ctx()
        t.on_routine_end("ACQUIRE", "success", state, ctx)
        t.on_routine_end("REST", "success", state, ctx)
        assert t._active is True  # REST doesn't break cycle
