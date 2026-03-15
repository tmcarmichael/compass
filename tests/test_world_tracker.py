"""Tests for brain.world.tracker -- StateChangeTracker edge-triggered logging.

Covers pet alive/dead transitions, engaged/disengaged, target changes,
imminent threat, pull target, sitting/standing, plan activation, in-combat flag,
and pet-has-add transitions. Uses make_game_state and make_spawn for realistic inputs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from brain.world.tracker import StateChangeTracker
from tests.factories import make_game_state, make_spawn

# ---------------------------------------------------------------------------
# Stub context
# ---------------------------------------------------------------------------


def _make_ctx(
    pet_alive: bool = False,
    pet_spawn_id: int | None = None,
    pet_name: str = "",
    engaged: bool = False,
    pull_target_name: str = "",
    pull_target_id: int | None = None,
    imminent_threat: bool = False,
    imminent_threat_con: str = "",
    plan_active: str | None = None,
    pet_has_add: bool = False,
) -> Any:
    ctx = SimpleNamespace(
        pet=SimpleNamespace(alive=pet_alive, spawn_id=pet_spawn_id, name=pet_name, has_add=pet_has_add),
        combat=SimpleNamespace(
            engaged=engaged,
            pull_target_name=pull_target_name,
            pull_target_id=pull_target_id,
        ),
        threat=SimpleNamespace(
            imminent_threat=imminent_threat,
            imminent_threat_con=imminent_threat_con,
        ),
        plan=SimpleNamespace(active=plan_active, travel=""),
    )
    ctx.nearby_player_count = lambda state, radius=250: 0
    ctx.nearest_player_dist = lambda state: 999.0
    return ctx


# ---------------------------------------------------------------------------
# Pet state transitions
# ---------------------------------------------------------------------------


class TestPetTransitions:
    def test_pet_death_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        ctx_alive = _make_ctx(pet_alive=True)
        state = make_game_state()
        tracker.update(state, ctx_alive)

        ctx_dead = _make_ctx(pet_alive=False)
        with caplog.at_level(10):
            tracker.update(state, ctx_dead)
        assert any("pet DIED" in r.message for r in caplog.records)

    def test_pet_alive_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        ctx_dead = _make_ctx(pet_alive=False)
        state = make_game_state()
        tracker.update(state, ctx_dead)

        ctx_alive = _make_ctx(pet_alive=True, pet_spawn_id=100, pet_name="Kabal")
        with caplog.at_level(10):
            tracker.update(state, ctx_alive)
        assert any("pet ALIVE" in r.message for r in caplog.records)

    def test_no_log_on_first_tick(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        ctx = _make_ctx(pet_alive=True)
        state = make_game_state()
        with caplog.at_level(10):
            tracker.update(state, ctx)
        assert not any("pet" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Engaged transitions
# ---------------------------------------------------------------------------


class TestEngagedTransitions:
    def test_engage_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        state = make_game_state()
        # First tick: not engaged
        tracker.update(state, _make_ctx(engaged=False))
        # Second tick: engaged
        with caplog.at_level(10):
            tracker.update(state, _make_ctx(engaged=True, pull_target_name="a_skeleton"))
        assert any("ENGAGED" in r.message for r in caplog.records)

    def test_disengage_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        state = make_game_state()
        tracker.update(state, _make_ctx(engaged=True))
        with caplog.at_level(10):
            tracker.update(state, _make_ctx(engaged=False))
        assert any("disengaged" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Target changes
# ---------------------------------------------------------------------------


class TestTargetChanges:
    def test_new_target_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        target = make_spawn(spawn_id=200, name="a_bat", x=30.0, y=40.0, level=8, hp_current=80, hp_max=100)
        state = make_game_state(target=target)
        with caplog.at_level(10):
            tracker.update(state, _make_ctx())
        assert any("a_bat" in r.message for r in caplog.records)

    def test_target_cleared_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        target = make_spawn(spawn_id=200, name="a_bat")
        state_with = make_game_state(target=target)
        tracker.update(state_with, _make_ctx())

        state_without = make_game_state(target=None)
        with caplog.at_level(10):
            tracker.update(state_without, _make_ctx())
        assert any("target cleared" in r.message for r in caplog.records)

    def test_same_target_no_log(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        target = make_spawn(spawn_id=200, name="a_bat")
        state = make_game_state(target=target)
        tracker.update(state, _make_ctx())
        caplog.clear()
        with caplog.at_level(10):
            tracker.update(state, _make_ctx())
        assert not any("a_bat" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Imminent threat transitions
# ---------------------------------------------------------------------------


class TestThreatTransitions:
    def test_threat_detected_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        state = make_game_state()
        tracker.update(state, _make_ctx(imminent_threat=False))
        with caplog.at_level(10):
            tracker.update(state, _make_ctx(imminent_threat=True, imminent_threat_con="RED"))
        assert any("imminent threat" in r.message for r in caplog.records)

    def test_threat_cleared_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        state = make_game_state()
        tracker.update(state, _make_ctx(imminent_threat=True))
        with caplog.at_level(10):
            tracker.update(state, _make_ctx(imminent_threat=False))
        assert any("threat cleared" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Pull target transitions
# ---------------------------------------------------------------------------


class TestPullTarget:
    def test_pull_target_set_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        state = make_game_state()
        tracker.update(state, _make_ctx(pull_target_id=None))
        with caplog.at_level(10):
            tracker.update(state, _make_ctx(pull_target_id=300, pull_target_name="a_skeleton"))
        assert any("pull target set" in r.message for r in caplog.records)

    def test_pull_target_cleared_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        state = make_game_state()
        tracker.update(state, _make_ctx(pull_target_id=300))
        with caplog.at_level(10):
            tracker.update(state, _make_ctx(pull_target_id=None))
        assert any("pull target cleared" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Plan transitions
# ---------------------------------------------------------------------------


class TestPlanTransitions:
    def test_plan_activated_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        state = make_game_state()
        tracker.update(state, _make_ctx(plan_active=None))
        with caplog.at_level(10):
            tracker.update(state, _make_ctx(plan_active="travel"))
        assert any("plan activated" in r.message for r in caplog.records)

    def test_plan_completed_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        state = make_game_state()
        tracker.update(state, _make_ctx(plan_active="travel"))
        with caplog.at_level(10):
            tracker.update(state, _make_ctx(plan_active=None))
        assert any("plan completed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# In-combat flag
# ---------------------------------------------------------------------------


class TestInCombatFlag:
    def test_in_combat_set_and_cleared(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        ctx = _make_ctx()
        state_idle = make_game_state(in_combat=False)
        tracker.update(state_idle, ctx)

        state_combat = make_game_state(in_combat=True)
        with caplog.at_level(10):
            tracker.update(state_combat, ctx)
        assert any("in_combat flag SET" in r.message for r in caplog.records)
        caplog.clear()

        with caplog.at_level(10):
            tracker.update(state_idle, ctx)
        assert any("in_combat flag CLEARED" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Pet has add
# ---------------------------------------------------------------------------


class TestPetHasAdd:
    def test_pet_add_detected(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        state = make_game_state()
        tracker.update(state, _make_ctx(pet_has_add=False))
        with caplog.at_level(10):
            tracker.update(state, _make_ctx(pet_has_add=True))
        assert any("pet has ADD" in r.message for r in caplog.records)

    def test_pet_add_cleared(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = StateChangeTracker()
        state = make_game_state()
        tracker.update(state, _make_ctx(pet_has_add=True))
        with caplog.at_level(10):
            tracker.update(state, _make_ctx(pet_has_add=False))
        assert any("pet add cleared" in r.message for r in caplog.records)
