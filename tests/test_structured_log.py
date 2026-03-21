"""Tests for util/structured_log.py: structured event logging, throttling, filters.

Covers iso_ts formatting, ElapsedFilter tag injection, StructuredHandler JSONL
writing, log_event pipeline, emit_throttled rate limiting, and DecisionThrottle.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from util.structured_log import (
    DecisionThrottle,
    ElapsedFilter,
    StructuredHandler,
    emit_throttled,
    iso_ts,
    log_event,
    reset_throttle_state,
)

# ---------------------------------------------------------------------------
# iso_ts
# ---------------------------------------------------------------------------


class TestIsoTs:
    def test_epoch_zero(self) -> None:
        result = iso_ts(0.0)
        assert result == "1970-01-01T00:00:00.000+00:00"

    def test_known_epoch(self) -> None:
        # 2024-01-01 00:00:00 UTC = 1704067200
        result = iso_ts(1704067200.0)
        assert result.startswith("2024-01-01T00:00:00")

    def test_millisecond_precision(self) -> None:
        result = iso_ts(1704067200.123)
        assert ".123" in result

    @pytest.mark.parametrize(
        "epoch",
        [0.0, 1.0, 1_000_000_000.0, 1_704_067_200.5],
    )
    def test_always_contains_timezone(self, epoch: float) -> None:
        result = iso_ts(epoch)
        assert "+00:00" in result


# ---------------------------------------------------------------------------
# ElapsedFilter
# ---------------------------------------------------------------------------


class TestElapsedFilter:
    def _make_record(self, msg: str, created: float) -> logging.LogRecord:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=None,
            exc_info=None,
        )
        record.created = created
        return record

    def test_injects_elapsed_into_tagged_message(self) -> None:
        f = ElapsedFilter(start_time=100.0)
        record = self._make_record("[COMBAT] Fight started", 126.5)
        f.filter(record)
        assert "+26s" in record.msg or "+27s" in record.msg

    def test_skips_untagged_messages(self) -> None:
        f = ElapsedFilter(start_time=100.0)
        record = self._make_record("Regular message", 110.0)
        f.filter(record)
        assert record.msg == "Regular message"

    def test_does_not_double_inject(self) -> None:
        f = ElapsedFilter(start_time=100.0)
        record = self._make_record("[TAG] +5s Already has elapsed", 110.0)
        f.filter(record)
        # Should NOT inject again since after_tag starts with "+"
        assert record.msg.count("+") == 1

    def test_always_returns_true(self) -> None:
        f = ElapsedFilter(start_time=0.0)
        record = self._make_record("anything", 1.0)
        assert f.filter(record) is True


# ---------------------------------------------------------------------------
# StructuredHandler
# ---------------------------------------------------------------------------


class TestStructuredHandler:
    def test_writes_jsonl(self, tmp_path) -> None:
        filepath = tmp_path / "events.jsonl"
        handler = StructuredHandler(filepath, session_id="test-session")
        try:
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0, msg="hello", args=None, exc_info=None
            )
            record.__dict__["event"] = "test_event"
            record.__dict__["event_data"] = {"key": "value"}
            handler.emit(record)

            lines = filepath.read_text().strip().split("\n")
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["event"] == "test_event"
            assert data["key"] == "value"
            assert data["session_id"] == "test-session"
        finally:
            handler.close()

    def test_skips_non_structured_records(self, tmp_path) -> None:
        filepath = tmp_path / "events.jsonl"
        handler = StructuredHandler(filepath)
        try:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="plain msg",
                args=None,
                exc_info=None,
            )
            handler.emit(record)
            assert filepath.read_text() == ""
        finally:
            handler.close()

    def test_tick_id_updated(self, tmp_path) -> None:
        filepath = tmp_path / "events.jsonl"
        handler = StructuredHandler(filepath)
        try:
            handler.set_tick_id(42)
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0, msg="", args=None, exc_info=None
            )
            record.__dict__["event"] = "tick_test"
            record.__dict__["event_data"] = {}
            handler.emit(record)

            data = json.loads(filepath.read_text().strip())
            assert data["tick_id"] == 42
        finally:
            handler.close()

    def test_circuit_breaker(self, tmp_path) -> None:
        filepath = tmp_path / "events.jsonl"
        handler = StructuredHandler(filepath)
        try:
            handler._errors = handler._max_errors  # trip the breaker
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0, msg="", args=None, exc_info=None
            )
            record.__dict__["event"] = "should_skip"
            record.__dict__["event_data"] = {}
            handler.emit(record)
            assert filepath.read_text() == ""
        finally:
            handler.close()

    def test_flush(self, tmp_path) -> None:
        filepath = tmp_path / "events.jsonl"
        handler = StructuredHandler(filepath)
        try:
            handler.flush()  # should not raise
        finally:
            handler.close()


# ---------------------------------------------------------------------------
# log_event
# ---------------------------------------------------------------------------


class TestLogEvent:
    def test_emits_through_logger(self, tmp_path) -> None:
        filepath = tmp_path / "events.jsonl"
        logger = logging.getLogger("test_log_event")
        logger.setLevel(logging.DEBUG)
        handler = StructuredHandler(filepath)
        logger.addHandler(handler)
        try:
            log_event(logger, "my_event", "Human message", npc="skeleton", hp=100)
            lines = filepath.read_text().strip().split("\n")
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["event"] == "my_event"
            assert data["npc"] == "skeleton"
            assert data["hp"] == 100
        finally:
            logger.removeHandler(handler)
            handler.close()

    def test_default_msg_is_event_name(self, tmp_path) -> None:
        """When msg is empty, event name is used as the log message."""
        filepath = tmp_path / "events.jsonl"
        logger = logging.getLogger("test_log_event_default")
        logger.setLevel(logging.DEBUG)
        handler = StructuredHandler(filepath)
        logger.addHandler(handler)
        try:
            log_event(logger, "my_event")
            data = json.loads(filepath.read_text().strip())
            assert data["event"] == "my_event"
        finally:
            logger.removeHandler(handler)
            handler.close()


# ---------------------------------------------------------------------------
# emit_throttled
# ---------------------------------------------------------------------------


class TestEmitThrottled:
    def setup_method(self) -> None:
        reset_throttle_state()

    def test_first_call_emits(self, tmp_path) -> None:
        filepath = tmp_path / "events.jsonl"
        logger = logging.getLogger("test_throttle_emit")
        logger.setLevel(logging.DEBUG)
        handler = StructuredHandler(filepath)
        logger.addHandler(handler)
        try:
            result = emit_throttled(logger, "slow_tick", "Tick was slow", min_interval=5.0)
            assert result is True
            assert filepath.read_text().strip() != ""
        finally:
            logger.removeHandler(handler)
            handler.close()

    def test_second_call_throttled(self) -> None:
        logger = logging.getLogger("test_throttle_block")
        logger.setLevel(logging.DEBUG)
        result1 = emit_throttled(logger, "dup_event", min_interval=60.0)
        assert result1 is True
        result2 = emit_throttled(logger, "dup_event", min_interval=60.0)
        assert result2 is False

    def test_different_events_not_throttled(self) -> None:
        logger = logging.getLogger("test_throttle_diff")
        logger.setLevel(logging.DEBUG)
        assert emit_throttled(logger, "event_a", min_interval=60.0) is True
        assert emit_throttled(logger, "event_b", min_interval=60.0) is True

    def test_reset_clears_state(self) -> None:
        logger = logging.getLogger("test_throttle_reset")
        logger.setLevel(logging.DEBUG)
        emit_throttled(logger, "cleared_event", min_interval=60.0)
        reset_throttle_state()
        result = emit_throttled(logger, "cleared_event", min_interval=60.0)
        assert result is True


# ---------------------------------------------------------------------------
# DecisionThrottle
# ---------------------------------------------------------------------------


class TestDecisionThrottle:
    def test_emits_at_interval(self, tmp_path) -> None:
        filepath = tmp_path / "decisions.jsonl"
        logger = logging.getLogger("test_decision_throttle")
        logger.setLevel(logging.DEBUG)
        handler = StructuredHandler(filepath)
        logger.addHandler(handler)
        try:
            throttle = DecisionThrottle(logger, interval=3)
            state = MagicMock()
            state.hp_pct = 0.8
            state.mana_pct = 0.5
            state.x = 100.0
            state.y = 200.0
            state.target = None

            # Calls 1 and 2 should not emit (interval=3, emit on 3rd)
            for i in range(2):
                throttle.record(
                    tick_id=i,
                    state=state,
                    rule_eval={},
                    rule_scores={},
                    selected="idle",
                    active="idle",
                    locked=False,
                    tick_ms=5.0,
                    routine_ms=1.0,
                    engaged=False,
                    pet_alive=True,
                )
            assert filepath.read_text() == ""

            # Call 3 should emit
            throttle.record(
                tick_id=3,
                state=state,
                rule_eval={},
                rule_scores={},
                selected="idle",
                active="idle",
                locked=False,
                tick_ms=5.0,
                routine_ms=1.0,
                engaged=False,
                pet_alive=True,
            )
            lines = filepath.read_text().strip().split("\n")
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["event"] == "decision"
        finally:
            logger.removeHandler(handler)
            handler.close()
