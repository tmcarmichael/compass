"""Tests for runtime.agent_session -- AgentSession dataclass.

AgentSession is a container for session lifecycle resources. Tests verify
construction, default values, and health_summary output.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from runtime.agent_session import AgentSession

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestAgentSessionConstruction:
    def test_minimal_construction(self) -> None:
        flags = MagicMock()
        movement = MagicMock()
        session = AgentSession(flags=flags, movement=movement)
        assert session.flags is flags
        assert session.movement is movement
        assert session.terrain is None
        assert session.reader is None
        assert session.config == {}
        assert session.zone_config == {}
        assert session.session_id == ""

    def test_full_construction(self) -> None:
        session = AgentSession(
            flags=MagicMock(),
            movement=MagicMock(),
            terrain=MagicMock(),
            reader=MagicMock(),
            config={"key": "val"},
            zone_config={"zone": "data"},
            session_id="test-session-001",
        )
        assert session.config["key"] == "val"
        assert session.zone_config["zone"] == "data"
        assert session.session_id == "test-session-001"


# ---------------------------------------------------------------------------
# health_summary
# ---------------------------------------------------------------------------


class TestHealthSummary:
    def test_no_terrain_reports_none(self) -> None:
        movement = MagicMock()
        movement.terrain = None
        session = AgentSession(flags=MagicMock(), movement=movement, terrain=None)
        summary = session.health_summary()
        assert summary["terrain"] == "None (no cache)"
        assert "no terrain" in summary["movement"]
        assert summary["session_id"] == "?"

    def test_with_terrain(self) -> None:
        terrain = MagicMock()
        terrain.stats = {"grid": "256x256", "obstacle": 42}
        movement = MagicMock()
        movement.terrain = terrain
        session = AgentSession(
            flags=MagicMock(),
            movement=movement,
            terrain=terrain,
            session_id="abc",
        )
        summary = session.health_summary()
        assert "256x256" in summary["terrain"]
        assert "42 obstacles" in summary["terrain"]
        assert "terrain wired" in summary["movement"]
        assert summary["session_id"] == "abc"
