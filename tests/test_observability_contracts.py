"""Tests for observability system contracts.

Verifies that the logging, forensics, and decision receipt systems
produce the output documented in the architecture.
"""

from __future__ import annotations

import logging

from brain.decision import Brain
from routines.base import RoutineBase, RoutineStatus
from tests.factories import make_game_state
from util.forensics import ForensicsBuffer
from util.structured_log import log_event

# ---------------------------------------------------------------------------
# Stub routine for brain tests
# ---------------------------------------------------------------------------


class _StubRoutine(RoutineBase):
    def enter(self, state):
        pass

    def tick(self, state):
        return RoutineStatus.RUNNING

    def exit(self, state):
        pass


# ---------------------------------------------------------------------------
# Decision receipt completeness
# ---------------------------------------------------------------------------


class TestDecisionReceipts:
    def test_receipt_covers_all_registered_rules(self) -> None:
        """After brain.tick(), last_rule_eval must have an entry for every rule."""
        brain = Brain(ctx=None, utility_phase=0)
        brain.add_rule("FLEE", lambda s: False, _StubRoutine(), emergency=True)
        brain.add_rule("REST", lambda s: False, _StubRoutine())
        brain.add_rule("ACQUIRE", lambda s: False, _StubRoutine())
        brain.add_rule("WANDER", lambda s: True, _StubRoutine())

        state = make_game_state()
        brain.tick(state)

        receipt = brain.last_rule_eval
        expected_rules = {"FLEE", "REST", "ACQUIRE", "WANDER"}
        assert set(receipt.keys()) == expected_rules, (
            f"Receipt keys {set(receipt.keys())} != registered rules {expected_rules}"
        )

    def test_receipt_values_are_valid(self) -> None:
        """Receipt values must be one of: YES, no, skip, cooldown(...), OPEN."""
        brain = Brain(ctx=None, utility_phase=0)
        brain.add_rule("FLEE", lambda s: False, _StubRoutine(), emergency=True)
        brain.add_rule("WANDER", lambda s: True, _StubRoutine())

        brain.tick(make_game_state())

        valid_prefixes = ("YES", "no", "skip", "cooldown", "OPEN")
        for rule_name, value in brain.last_rule_eval.items():
            assert any(value.startswith(p) for p in valid_prefixes), (
                f"Rule '{rule_name}' has invalid receipt value: {value!r}"
            )

    def test_exactly_one_yes_per_tick(self) -> None:
        """Exactly one rule should match (YES) per tick in phase 0."""
        brain = Brain(ctx=None, utility_phase=0)
        brain.add_rule("REST", lambda s: s.hp_pct < 0.5, _StubRoutine())
        brain.add_rule("WANDER", lambda s: True, _StubRoutine())

        brain.tick(make_game_state(hp_current=1000, hp_max=1000))

        yes_count = sum(1 for v in brain.last_rule_eval.values() if v == "YES")
        assert yes_count == 1


# ---------------------------------------------------------------------------
# Forensics ring buffer
# ---------------------------------------------------------------------------


class TestForensicsBuffer:
    def test_capacity_is_300(self) -> None:
        """Buffer evicts oldest entries beyond 300."""
        buf = ForensicsBuffer(session_id="test", session_dir="/tmp", max_entries=300)
        state = make_game_state()
        for i in range(350):
            buf.record_tick(i, state, active_routine="WANDER")
        assert len(buf._buffer) == 300

    def test_oldest_evicted_first(self) -> None:
        """After overflow, the oldest tick should be gone."""
        buf = ForensicsBuffer(session_id="test", session_dir="/tmp", max_entries=300)
        state = make_game_state()
        for i in range(350):
            buf.record_tick(i, state)
        oldest = buf._buffer[0]
        assert oldest["tick"] == 50  # ticks 0-49 evicted

    def test_flush_produces_data(self, tmp_path) -> None:
        """Flushing a non-empty buffer writes content."""
        buf = ForensicsBuffer(session_id="test", session_dir=str(tmp_path), max_entries=300)
        state = make_game_state()
        for i in range(10):
            buf.record_tick(i, state)
        buf.flush("test_trigger")

        path = tmp_path / "test_forensics.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 11  # 1 header + 10 entries


# ---------------------------------------------------------------------------
# Structured event schema
# ---------------------------------------------------------------------------


class TestStructuredEventSchema:
    def test_log_event_produces_valid_fields(self, caplog) -> None:
        """log_event() attaches event name and data via LogRecord extra."""
        test_logger = logging.getLogger("test.events")
        with caplog.at_level(logging.INFO, logger="test.events"):
            log_event(
                test_logger,
                "fight_end",
                "combat completed",
                duration=20.5,
                mana_spent=50,
            )

        assert len(caplog.records) >= 1
        record = caplog.records[-1]
        # log_event attaches extra={"event": ..., "event_data": ...}
        assert getattr(record, "event", None) == "fight_end"
        event_data = getattr(record, "event_data", {})
        assert event_data.get("duration") == 20.5
        assert event_data.get("mana_spent") == 50
