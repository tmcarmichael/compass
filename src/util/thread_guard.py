"""Thread ownership assertions for AgentContext fields.

Warning-only: violations are logged, not raised. A crash at 3 AM is worse
than a logged violation. Session logs surface violations for offline fixes.

Performance: ~20ns per check (C-level thread ident lookup). At 10 Hz on
the handful of instrumented mutation sites, total cost is <1us/tick.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

__all__ = ["set_brain_thread", "assert_brain_thread", "violation_count"]

# Set once by brain_runner.run() at startup
_brain_thread_id: int = 0

# Violation tracking (read by invariant checker for session reports)
_violation_count: int = 0
_last_violation_time: float = 0.0
_COOLDOWN: float = 10.0  # min seconds between warnings for same thread


def set_brain_thread() -> None:
    """Called once from brain_runner.run() on the brain thread."""
    global _brain_thread_id
    _brain_thread_id = threading.current_thread().ident or 0
    log.debug("[THREAD] Brain thread registered: id=%d", _brain_thread_id)


def assert_brain_thread(field_name: str) -> None:
    """Warn if called from a non-brain thread. Non-fatal in production."""
    global _violation_count, _last_violation_time
    if _brain_thread_id == 0:
        return  # not yet initialized -- skip check
    current = threading.current_thread().ident or 0
    if current != _brain_thread_id:
        _violation_count += 1
        now = time.monotonic()
        if now - _last_violation_time > _COOLDOWN:
            _last_violation_time = now
            log.warning(
                "[THREAD] OWNERSHIP VIOLATION #%d: '%s' accessed from thread %d (brain=%d)",
                _violation_count,
                field_name,
                current,
                _brain_thread_id,
            )


def violation_count() -> int:
    """Current violation count (for invariant checker / session report)."""
    return _violation_count
