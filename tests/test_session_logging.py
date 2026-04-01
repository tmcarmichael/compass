"""Tests for runtime.session_logging helper module."""

from __future__ import annotations

import logging

from brain.context import AgentContext
from runtime.session_logging import LogCaptureHandler, setup_session_logging
from util.structured_log import log_event


def test_log_capture_handler_routes_formatted_messages() -> None:
    captured: list[tuple[str, str]] = []
    handler = LogCaptureHandler(lambda msg, level: captured.append((msg, level)))
    logger = logging.getLogger("test.session_logging.capture")
    logger.handlers[:] = []
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    try:
        logger.info("hello")
    finally:
        logger.removeHandler(handler)
        handler.close()

    assert captured == [("hello", "INFO")]


def test_setup_session_logging_installs_diag_resources_and_closes_cleanly(tmp_path) -> None:
    ctx = AgentContext()
    handles = setup_session_logging(ctx, tmp_path, "sess-1")

    assert ctx.diag.structured_handler is handles.structured_handler
    assert ctx.diag.decision_throttle is handles.decision_throttle
    assert ctx.diag.forensics is handles.forensics
    assert ctx.diag.metrics is handles.metrics_engine
    assert ctx.diag.invariants is handles.invariant_checker

    log_event(logging.getLogger("compass"), "test_event", "session logging test", value=1)
    handles.close()

    events_file = tmp_path / "sess-1_events.jsonl"
    assert events_file.exists()
    assert logging.getLogger("compass.decisions").propagate is True
