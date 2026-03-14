"""Pytest fixtures and hypothesis configuration.

Factory functions live in tests/factories.py  -- this file is for
pytest-discovered fixtures and hypothesis profile registration only.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

from brain.goap.world_state import PlanWorldState
from brain.learning.encounters import FightHistory
from perception.state import GameState
from tests.factories import make_game_state

# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------

settings.register_profile("ci", max_examples=200)
settings.register_profile("dev", max_examples=50, suppress_health_check=[HealthCheck.too_slow])


# ---------------------------------------------------------------------------
# Motor recording backend (auto-used by all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _recording_motor():
    """Install a RecordingMotor backend for all tests.

    Prevents motor actions from sleeping or sending input during tests.
    Restores the default backend after each test.
    """
    from motor.actions import set_backend
    from motor.recording import RecordingMotor

    recorder = RecordingMotor()
    set_backend(recorder)
    yield recorder
    # Restore default backend
    from motor.actions import _DefaultBackend

    set_backend(_DefaultBackend())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def game_state() -> GameState:
    return make_game_state()


@pytest.fixture
def plan_world_state() -> PlanWorldState:
    return PlanWorldState()


@pytest.fixture
def fight_history_factory(tmp_path: Path) -> Callable[[str], FightHistory]:
    """Returns a callable that creates a FightHistory writing to tmp_path."""

    def _factory(zone: str = "testzone") -> FightHistory:
        return FightHistory(zone=zone, data_dir=str(tmp_path))

    return _factory
