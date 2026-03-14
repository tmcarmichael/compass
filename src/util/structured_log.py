"""Structured event logging through the stdlib logging pipeline.

Adds a StructuredHandler that intercepts LogRecords carrying an 'event'
attribute (set via extra={}) and writes them as JSON lines to a .jsonl file.
All events share timestamps with the text log -- same pipeline, same ordering.

Usage:
    from util.structured_log import log_event
    log_event(log, "fight_end", "Fight ended: skeleton in 7.2s",
              npc="skeleton", duration=7.2, casts=3)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from perception.state import GameState

log = logging.getLogger(__name__)


def iso_ts(epoch: float) -> str:
    """Convert Unix epoch to ISO 8601 UTC string with ms precision."""
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat(timespec="milliseconds")


class ElapsedFilter(logging.Filter):
    """Injects elapsed time into tagged log messages for cross-file correlation.

    Tagged messages like ``[COMBAT] Fight started`` become
    ``[COMBAT] +26.5s Fight started``. The ``+26.5s`` matches the
    ``"elapsed": 26.5`` field in _events.jsonl, enabling agents to jump
    between text log and JSONL by searching for the elapsed value.

    Only modifies messages that start with ``[`` (tagged lines).
    Attached to the session file handler, not the structured handler.
    """

    def __init__(self, start_time: float = 0.0) -> None:
        super().__init__()
        self._start_time = start_time or time.time()

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if msg.startswith("["):
            elapsed = record.created - self._start_time
            # Find end of tag: [TAG] -> insert after "] "
            bracket_end = msg.find("] ")
            if bracket_end > 0:
                after_tag = msg[bracket_end + 2 :]
                # Skip if elapsed already injected (avoid +Xs +Xs doubling)
                if not after_tag.startswith("+"):
                    record.msg = f"{msg[: bracket_end + 2]}+{elapsed:.0f}s {after_tag}"
                    record.args = None  # msg is already formatted
        return True


class StructuredHandler(logging.Handler):
    """Logging handler that writes structured events to a JSONL file.

    Intercepts LogRecords that have an 'event' attribute (set via extra={}).
    Writes one JSON line per structured event. Ignores non-structured records.

    Attached to the same compass logger as the text and dashboard handlers --
    same pipeline, same timestamps, same ordering.
    """

    def __init__(self, filepath: Path | str, session_id: str = "") -> None:
        super().__init__(level=logging.DEBUG)
        self._file = open(filepath, "a", encoding="utf-8")  # closed in close()
        self._session_id = session_id
        self._tick_id: int = 0
        self._start_time: float = time.time()
        self._errors: int = 0
        self._max_errors: int = 20  # circuit breaker

    def emit(self, record: logging.LogRecord) -> None:
        """Write JSON line if record has structured event data."""
        event = getattr(record, "event", None)
        if event is None:
            return  # not a structured event, skip

        if self._errors >= self._max_errors:
            return  # circuit breaker tripped

        event_data = getattr(record, "event_data", {})
        line = {
            "event": event,
            "ts": iso_ts(record.created),
            "elapsed": round(record.created - self._start_time, 2),
            "session_id": self._session_id,
            "tick_id": self._tick_id,
            "level": record.levelname,
            "logger": record.name,
            **event_data,
        }
        try:
            self._file.write(json.dumps(line, separators=(",", ":")) + "\n")
            self._file.flush()
            self._errors = 0  # reset on success
        except (OSError, ValueError, TypeError) as exc:
            self._errors += 1
            if self._errors <= 3:
                # Use stderr to avoid recursion through the logging system
                import sys

                print(f"StructuredHandler write error ({self._errors}): {exc}", file=sys.stderr)

    def set_tick_id(self, tick_id: int) -> None:
        """Update the current tick_id. Called by brain_runner each tick."""
        self._tick_id = tick_id

    def flush(self) -> None:
        """Flush the JSONL file buffer."""
        try:
            self._file.flush()
        except OSError as exc:
            log.debug("StructuredHandler.flush failed: %s", exc)

    def close(self) -> None:
        """Flush and close the JSONL file."""
        try:
            self._file.flush()
            self._file.close()
        except OSError as exc:
            log.debug("StructuredHandler.close failed: %s", exc)
        super().close()


class DecisionThrottle:
    """Rate-limited decision receipt logger. Emits every Nth call (~2Hz at 10Hz tick).

    Decision receipts capture the brain's full decision state: which rule won,
    what alternatives existed, scores, and key world state. This is the
    "30 seconds before death" forensic tool.

    Uses a dedicated logger so decisions can go to a separate JSONL file
    while still flowing through the stdlib logging pipeline.
    """

    def __init__(self, logger: logging.Logger, interval: int = 5) -> None:
        self._logger = logger
        self._interval = interval
        self._count: int = 0

    def record(
        self,
        tick_id: int,
        state: GameState,
        rule_eval: dict[str, str],
        rule_scores: dict[str, float],
        selected: str,
        active: str,
        locked: bool,
        tick_ms: float,
        routine_ms: float,
        engaged: bool,
        pet_alive: bool,
    ) -> None:
        """Record one decision receipt. Rate-limited to ~2Hz internally."""
        self._count += 1
        if self._count % self._interval != 0:
            return
        # Build compact state -- avoid importing GameState (use duck typing)
        hp = getattr(state, "hp_pct", 0.0)
        mana = getattr(state, "mana_pct", 0.0)
        x = getattr(state, "x", 0.0)
        y = getattr(state, "y", 0.0)
        target = getattr(state, "target", None)
        target_name = target.name if target else ""
        target_hp = target.hp_current if target else 0
        target_dist = 0.0
        if target:
            dx = x - target.x
            dy = y - target.y
            target_dist = (dx * dx + dy * dy) ** 0.5
        log_event(
            self._logger,
            "decision",
            level=logging.DEBUG,
            tick_id=tick_id,
            rule_eval=rule_eval,
            rule_scores=rule_scores if rule_scores else {},
            selected=selected,
            active=active,
            locked=locked,
            hp=round(hp, 3),
            mana=round(mana, 3),
            pos_x=round(x),
            pos_y=round(y),
            target=target_name,
            target_hp=target_hp,
            target_dist=round(target_dist),
            engaged=engaged,
            pet=pet_alive,
            tick_ms=round(tick_ms, 1),
            routine_ms=round(routine_ms, 1),
        )


def log_event(
    logger: logging.Logger, event: str, msg: str = "", level: int = logging.INFO, **fields: object
) -> None:
    """Emit a structured event through the logging pipeline.

    The event name and fields are attached to the LogRecord via extra={}.
    StructuredHandler extracts them to JSONL. Text handler logs the message.
    All other handlers (dashboard, file) see the record normally.

    Args:
        logger: The logger to emit through (use the module's log).
        event: Event type name (e.g. "fight_end", "pull_result").
        msg: Human-readable message for the text log. Defaults to event name.
        level: Log level (default INFO).
        **fields: Structured fields for the JSONL record.
    """
    logger.log(level, msg or event, extra={"event": event, "event_data": fields})


# Throttle state for emit_throttled (module-level, shared across callers)
_throttle_times: dict[str, float] = {}


def reset_throttle_state() -> None:
    """Clear throttle timers. Call at session start to prevent cross-session bleed."""
    _throttle_times.clear()


def emit_throttled(
    logger: logging.Logger,
    event: str,
    msg: str = "",
    min_interval: float = 5.0,
    level: int = logging.WARNING,
    **fields: object,
) -> bool:
    """Emit a structured event at most once per min_interval seconds.

    Use for hot-path warnings (slow ticks, repeated anomalies) to prevent
    log flooding while still surfacing the issue.

    Returns True if the event was emitted, False if throttled.
    """
    now = time.time()
    last = _throttle_times.get(event, 0.0)
    if now - last < min_interval:
        return False
    _throttle_times[event] = now
    log_event(logger, event, msg, level=level, **fields)
    return True
