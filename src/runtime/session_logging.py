"""Session logging bootstrap and teardown helpers.

Keeps the runtime orchestrator focused on lifecycle flow rather than the
details of handler wiring, forensics setup, and diagnostic module bootstrapping.
"""

from __future__ import annotations

import logging
import queue
import time
from collections.abc import Callable
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

from util.log_tiers import EVENT, VERBOSE
from util.structured_log import DecisionThrottle, ElapsedFilter, StructuredHandler, reset_throttle_state

if TYPE_CHECKING:
    from brain.context import AgentContext
    from util.forensics import ForensicsBuffer
    from util.invariants import InvariantChecker
    from util.metrics import MetricsEngine


log = logging.getLogger(__name__)


def prune_session_logs(session_dir: Path, max_age_days: int = 7, keep_min: int = 5) -> None:
    """Delete old session JSONL files while keeping recent artifacts."""
    now = time.time()
    max_age = max_age_days * 86400
    pruned = 0
    for suffix in ("_events.jsonl", "_decisions.jsonl"):
        files = sorted(session_dir.glob(f"*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files[keep_min:]:
            try:
                if now - f.stat().st_mtime > max_age:
                    f.unlink()
                    pruned += 1
            except OSError:
                pass
    if pruned:
        log.info("[LIFECYCLE] Pruned %d old session JSONL files", pruned)


class LogCaptureHandler(logging.Handler):
    """Routes log records into the orchestrator ring buffer."""

    def __init__(self, sink: Callable[[str, str], None]) -> None:
        super().__init__(level=logging.INFO)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink(self.format(record), record.levelname)
        except (ValueError, AttributeError, OSError) as exc:
            import sys

            print(f"LogCapture.emit failed: {exc}", file=sys.stderr)


@dataclass(slots=True)
class SessionLoggingHandles:
    """Owned session logging resources that must be torn down together."""

    queue_handler: logging.Handler
    queue_listener: QueueListener
    structured_handler: StructuredHandler
    decision_handler: StructuredHandler
    decision_throttle: DecisionThrottle
    forensics: ForensicsBuffer
    metrics_engine: MetricsEngine
    invariant_checker: InvariantChecker
    text_handlers: tuple[logging.Handler, ...]

    def close(self) -> None:
        """Detach and close all handlers/resources created for a session."""
        self.forensics.close()

        compass_logger = logging.getLogger("compass")
        decision_logger = logging.getLogger("compass.decisions")

        decision_logger.removeHandler(self.decision_handler)
        decision_logger.propagate = True
        self.decision_handler.close()

        compass_logger.removeHandler(self.structured_handler)
        self.structured_handler.close()

        compass_logger.removeHandler(self.queue_handler)
        self.queue_listener.stop()
        self.queue_handler.close()

        for handler in self.text_handlers:
            handler.close()


def setup_session_logging(ctx: AgentContext, session_dir: Path, session_id: str) -> SessionLoggingHandles:
    """Create session-scoped logging, forensics, and observability resources."""
    from util.cycle_tracker import CycleTracker
    from util.forensics import ForensicsBuffer
    from util.incident_reporter import IncidentReporter
    from util.invariants import InvariantChecker, register_builtin_invariants
    from util.metrics import MetricsEngine
    from util.phase_detector import PhaseDetector

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    elapsed_filter = ElapsedFilter(start_time=time.time())
    reset_throttle_state()

    events_text_handler = RotatingFileHandler(str(session_dir / f"{session_id}_events.log"), maxBytes=10_000_000, backupCount=2)
    events_text_handler.setLevel(EVENT)
    events_text_handler.setFormatter(fmt)
    events_text_handler.addFilter(elapsed_filter)

    session_file_handler = RotatingFileHandler(str(session_dir / f"{session_id}.log"), maxBytes=50_000_000, backupCount=3)
    session_file_handler.setLevel(logging.INFO)
    session_file_handler.setFormatter(fmt)
    session_file_handler.addFilter(elapsed_filter)

    verbose_handler = RotatingFileHandler(
        str(session_dir / f"{session_id}_verbose.log"), maxBytes=50_000_000, backupCount=3
    )
    verbose_handler.setLevel(VERBOSE)
    verbose_handler.setFormatter(fmt)
    verbose_handler.addFilter(elapsed_filter)

    debug_handler = RotatingFileHandler(str(session_dir / f"{session_id}_debug.log"), maxBytes=50_000_000, backupCount=3)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(fmt)
    debug_handler.addFilter(elapsed_filter)

    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)
    queue_handler = QueueHandler(log_queue)
    queue_handler.setLevel(logging.DEBUG)
    queue_listener = QueueListener(
        log_queue,
        events_text_handler,
        session_file_handler,
        verbose_handler,
        debug_handler,
        respect_handler_level=True,
    )
    queue_listener.start()

    compass_logger = logging.getLogger("compass")
    compass_logger.addHandler(queue_handler)

    structured_handler = StructuredHandler(session_dir / f"{session_id}_events.jsonl", session_id)
    compass_logger.addHandler(structured_handler)
    ctx.diag.structured_handler = structured_handler

    decision_handler = StructuredHandler(session_dir / f"{session_id}_decisions.jsonl", session_id)
    decision_logger = logging.getLogger("compass.decisions")
    decision_logger.addHandler(decision_handler)
    decision_logger.setLevel(logging.DEBUG)
    decision_logger.propagate = False
    decision_throttle = DecisionThrottle(decision_logger, interval=5)
    ctx.diag.decision_throttle = decision_throttle

    forensics = ForensicsBuffer(session_id, str(session_dir))
    ctx.diag.forensics = forensics

    metrics_engine = MetricsEngine()
    ctx.diag.metrics = metrics_engine

    invariant_checker = InvariantChecker()
    register_builtin_invariants(invariant_checker)
    ctx.diag.invariants = invariant_checker

    ctx.diag.cycle_tracker = CycleTracker()
    ctx.diag.incident_reporter = IncidentReporter()
    ctx.diag.phase_detector = PhaseDetector()

    return SessionLoggingHandles(
        queue_handler=queue_handler,
        queue_listener=queue_listener,
        structured_handler=structured_handler,
        decision_handler=decision_handler,
        decision_throttle=decision_throttle,
        forensics=forensics,
        metrics_engine=metrics_engine,
        invariant_checker=invariant_checker,
        text_handlers=(events_text_handler, session_file_handler, verbose_handler, debug_handler),
    )
