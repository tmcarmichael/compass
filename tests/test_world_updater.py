"""Tests for brain.world.updater -- WorldStateUpdater.

Tests update_world_state delegation and check_player_status death detection
using lightweight stubs for BrainRunner, HealthMonitor, and StateChangeTracker.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from brain.world.updater import WorldStateUpdater
from tests.factories import make_game_state

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _make_player(
    dead: bool = False,
    last_known_x: float = 0.0,
    last_known_y: float = 0.0,
    last_known_z: float = 0.0,
) -> Any:
    return SimpleNamespace(
        dead=dead,
        last_known_x=last_known_x,
        last_known_y=last_known_y,
        last_known_z=last_known_z,
    )


def _make_ctx(
    player_dead: bool = False,
    has_world: bool = False,
) -> Any:
    ctx = SimpleNamespace(
        player=_make_player(dead=player_dead),
        pet=SimpleNamespace(alive=False, spawn_id=0, name=""),
    )
    ctx.update_pet_status = lambda state: None
    if has_world:
        ctx.world = SimpleNamespace(
            update=lambda state: None,
            threats_within=lambda r: [],
        )
    else:
        ctx.world = None
    return ctx


class _StubHealthMonitor:
    def __init__(self) -> None:
        self.tick_count = 0

    def tick(self, state: Any, ctx: Any) -> None:
        self.tick_count += 1


class _StubStateTracker:
    def __init__(self) -> None:
        self.update_count = 0

    def update(self, state: Any, ctx: Any) -> None:
        self.update_count += 1


class _StubRunner:
    def __init__(self, handle_death_returns: bool = True) -> None:
        self._handle_death_returns = handle_death_returns
        self.death_calls: list[str] = []

    def _handle_death(self, ctx: Any, reason: str) -> bool:
        self.death_calls.append(reason)
        return self._handle_death_returns


# ---------------------------------------------------------------------------
# update_world_state
# ---------------------------------------------------------------------------


class TestUpdateWorldState:
    def test_calls_health_monitor_tick(self) -> None:
        runner = _StubRunner()
        updater = WorldStateUpdater(runner)
        hm = _StubHealthMonitor()
        st = _StubStateTracker()
        ctx = _make_ctx()
        state = make_game_state()
        updater.update_world_state(state, ctx, hm, st)
        assert hm.tick_count == 1

    def test_calls_state_tracker_update(self) -> None:
        runner = _StubRunner()
        updater = WorldStateUpdater(runner)
        hm = _StubHealthMonitor()
        st = _StubStateTracker()
        ctx = _make_ctx()
        state = make_game_state()
        updater.update_world_state(state, ctx, hm, st)
        assert st.update_count == 1

    def test_no_world_model_no_crash(self) -> None:
        runner = _StubRunner()
        updater = WorldStateUpdater(runner)
        hm = _StubHealthMonitor()
        st = _StubStateTracker()
        ctx = _make_ctx(has_world=False)
        state = make_game_state()
        updater.update_world_state(state, ctx, hm, st)
        # Should not crash even without world model


# ---------------------------------------------------------------------------
# check_player_status -- position tracking
# ---------------------------------------------------------------------------


class TestCheckPlayerStatusPosition:
    def test_tracks_position_when_alive(self) -> None:
        runner = _StubRunner()
        updater = WorldStateUpdater(runner)
        ctx = _make_ctx()
        state = make_game_state(x=100.0, y=200.0, z=5.0, hp_current=500)
        updater.check_player_status(state, ctx)
        assert ctx.player.last_known_x == 100.0
        assert ctx.player.last_known_y == 200.0
        assert ctx.player.last_known_z == 5.0

    def test_no_position_update_at_zero_hp(self) -> None:
        runner = _StubRunner()
        updater = WorldStateUpdater(runner)
        ctx = _make_ctx()
        ctx.player.last_known_x = 50.0
        state = make_game_state(x=999.0, y=999.0, hp_current=0)
        updater.check_player_status(state, ctx)
        # Position should not update to 999
        assert ctx.player.last_known_x == 50.0


# ---------------------------------------------------------------------------
# check_player_status -- death detection
# ---------------------------------------------------------------------------


class TestCheckPlayerStatusDeath:
    def test_body_state_death(self) -> None:
        runner = _StubRunner()
        updater = WorldStateUpdater(runner)
        ctx = _make_ctx(player_dead=False)
        state = make_game_state(body_state="d", hp_current=0, hp_max=1000)
        result = updater.check_player_status(state, ctx)
        assert result is True
        assert "body_state" in runner.death_calls

    def test_hp_zero_death(self) -> None:
        runner = _StubRunner()
        updater = WorldStateUpdater(runner)
        ctx = _make_ctx(player_dead=False)
        state = make_game_state(body_state="n", hp_current=0, hp_max=1000)
        result = updater.check_player_status(state, ctx)
        assert result is True
        assert "hp_zero" in runner.death_calls

    def test_no_death_when_alive(self) -> None:
        runner = _StubRunner()
        updater = WorldStateUpdater(runner)
        ctx = _make_ctx(player_dead=False)
        state = make_game_state(hp_current=500, hp_max=1000)
        result = updater.check_player_status(state, ctx)
        assert result is False
        assert runner.death_calls == []

    def test_no_death_when_already_dead(self) -> None:
        runner = _StubRunner()
        updater = WorldStateUpdater(runner)
        ctx = _make_ctx(player_dead=True)
        state = make_game_state(body_state="d", hp_current=0, hp_max=1000)
        result = updater.check_player_status(state, ctx)
        assert result is False

    def test_no_death_when_hp_max_zero(self) -> None:
        """hp_max=0 means character info not loaded yet."""
        runner = _StubRunner()
        updater = WorldStateUpdater(runner)
        ctx = _make_ctx(player_dead=False)
        state = make_game_state(hp_current=0, hp_max=0)
        result = updater.check_player_status(state, ctx)
        assert result is False

    def test_handle_death_returns_false(self) -> None:
        """When _handle_death returns False, check_player_status returns False."""
        runner = _StubRunner(handle_death_returns=False)
        updater = WorldStateUpdater(runner)
        ctx = _make_ctx(player_dead=False)
        state = make_game_state(body_state="d", hp_current=0, hp_max=1000)
        result = updater.check_player_status(state, ctx)
        # First death check (body_state) returns False, so continues to hp_zero
        assert result is False

    def test_successive_alive_updates(self) -> None:
        """Multiple alive ticks should track position without death."""
        runner = _StubRunner()
        updater = WorldStateUpdater(runner)
        ctx = _make_ctx()
        for i in range(5):
            state = make_game_state(x=float(i * 10), y=0.0, hp_current=1000, hp_max=1000)
            result = updater.check_player_status(state, ctx)
            assert result is False
        assert ctx.player.last_known_x == 40.0
