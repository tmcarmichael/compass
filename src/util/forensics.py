"""Forensics ring buffer and world state snapshot helpers.

The ForensicsBuffer holds the last N ticks of compact state in memory.
Zero disk I/O during normal operation. On death, crash, or invariant
breach, the buffer is flushed through the logging pipeline to a
_forensics.jsonl file.

compact_world() produces a small dict of nearby NPCs for embedding
in key events (fight_end, flee_trigger, player_death).
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from util.structured_log import iso_ts

if TYPE_CHECKING:
    from perception.state import GameState

log = logging.getLogger(__name__)


def compact_world(state: GameState, max_npcs: int = 6) -> dict[str, object]:
    """Build a compact world snapshot for embedding in events.

    Returns nearby live NPCs sorted by distance, plus player count.

    Args:
        state: Current game state with position and spawn data.
        max_npcs: Max NPCs to include (closest first).

    Returns:
        {"npcs": [{"name", "id", "hp", "dist"}], "players": int}
    """
    px = state.x
    py = state.y

    npcs = []
    players = 0
    for s in state.spawns:
        if s.is_player:
            players += 1
            continue
        if not s.is_npc:
            continue
        if s.hp_current <= 0:
            continue
        dx = px - s.x
        dy = py - s.y
        dist = (dx * dx + dy * dy) ** 0.5
        npcs.append(
            {
                "name": s.name,
                "id": s.spawn_id,
                "hp": s.hp_current,
                "dist": round(dist),
            }
        )

    npcs.sort(key=lambda n: n["dist"])
    return {"npcs": npcs[:max_npcs], "players": players}


class ForensicsBuffer:
    """Bounded ring buffer of compact per-tick state.

    Normal operation: records ~150 bytes/entry into a deque.
    300 entries = ~45KB max memory. Zero disk I/O.

    On critical events (death, crash, invariant breach), flushes
    the entire buffer to a JSONL file through the logging pipeline.

    Thread ownership: brain thread writes via record_tick() and flush().
    """

    def __init__(self, session_id: str, session_dir: str | Path, max_entries: int = 300) -> None:
        self._buffer: deque[dict[str, object]] = deque(maxlen=max_entries)
        self._file_path = Path(session_dir) / f"{session_id}_forensics.jsonl"
        self._session_id = session_id
        self._flush_count: int = 0

    def snapshot(self) -> list[dict[str, object]]:
        """Return a copy of the ring buffer for incident reporting."""
        return list(self._buffer)

    def record_tick(
        self, tick_id: int, state: GameState, active_routine: str = "", engaged: bool = False
    ) -> None:
        """Record compact tick state into ring buffer. Called every tick (10Hz)."""
        target = state.target
        self._buffer.append(
            {
                "tick": tick_id,
                "ts": iso_ts(time.time()),
                "hp": state.hp_current,
                "hp_max": state.hp_max,
                "mana": state.mana_current,
                "x": round(state.x, 1),
                "y": round(state.y, 1),
                "hdg": round(state.heading),
                "tgt": target.name if target else "",
                "tgt_hp": target.hp_current if target else 0,
                "engaged": engaged,
                "routine": active_routine,
            }
        )

    def flush(self, trigger: str) -> None:
        """Dump ring buffer to disk. Called on death, crash, invariant breach.

        Writes a header line then all buffered entries as JSON lines.
        """
        if not self._buffer:
            return

        self._flush_count += 1
        try:
            with open(self._file_path, "a", encoding="utf-8") as f:
                header = {
                    "event": "forensics_dump",
                    "trigger": trigger,
                    "session_id": self._session_id,
                    "ts": iso_ts(time.time()),
                    "entries": len(self._buffer),
                    "flush_number": self._flush_count,
                }
                f.write(json.dumps(header, separators=(",", ":")) + "\n")
                for entry in self._buffer:
                    f.write(json.dumps(entry, separators=(",", ":")) + "\n")
            log.info(
                "[LIFECYCLE] Forensics: flushed %d ticks (trigger=%s) to %s",
                len(self._buffer),
                trigger,
                self._file_path.name,
            )
        except OSError as exc:
            log.warning("[LIFECYCLE] Forensics: flush failed: %s", exc)

        # Clear buffer after every flush to prevent re-dumping stale data.
        # Previous design kept buffer for cascading invariant violations,
        # but this caused 2.5MB+ of redundant forensics per session.
        self._buffer.clear()

    def close(self) -> None:
        """Flush any remaining entries on shutdown."""
        if self._buffer:
            self.flush("shutdown")
