"""Spatial memory  -  the agent learns where npcs spawn and builds camps at runtime.

Records defeats, npc sightings, and empty areas. Over time builds a heat map
of where good targets are. Persists across sessions per zone.

Used by:
  - Wander: bias direction toward known npc-dense areas
  - Acquire: log sightings and empty scans
  - Auto-camp: detect defeat clusters and form camps dynamically
"""

import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.types import Point

log = logging.getLogger(__name__)

# Grid cell size for the heat map (in game units)
CELL_SIZE = 50.0

# How fast old data decays (defeats older than this contribute less)
DECAY_HOURS = 4.0


class SpatialMemory:
    """Persistent spatial knowledge about a zone."""

    def __init__(
        self, zone_name: str, data_dir: str | Path = "data/memory", clock: Callable[[], float] = time.time
    ) -> None:
        self.zone_name = zone_name
        self._data_dir = Path(data_dir)
        self._clock = clock
        self._kills: list[dict[str, Any]] = []  # {x, y, name, level, time, fight_s}
        self._sightings: list[dict[str, Any]] = []  # {x, y, name, level, time}
        self._empty_scans: list[dict[str, Any]] = []  # {x, y, time}
        self._danger_events: list[dict[str, Any]] = []  # {x, y, time, reason}

        # Heat map: cell -> score (higher = more npcs defeated/seen here)
        self._heat: dict[tuple[int, int], float] = {}
        # Recently visited cells (not persisted -- session only)
        self._visited: dict[tuple[int, int], float] = {}

        self._load()

    # -- Public API ---------------------------------------------------

    def record_kill(self, pos: Point, name: str, level: int, fight_seconds: float = 0.0) -> None:
        """Record a defeat at this location."""
        now = self._clock()
        self._kills.append(
            {
                "x": round(pos.x),
                "y": round(pos.y),
                "name": name,
                "level": level,
                "time": now,
                "fight_s": round(fight_seconds, 1),
            }
        )
        self._update_heat(pos, weight=3.0, event_time=now)
        log.debug("SpatialMemory: defeat '%s' lv%d at (%.0f,%.0f)", name, level, pos.x, pos.y)

    def record_sighting(self, pos: Point, name: str, level: int) -> None:
        """Record seeing a valid npc at this location."""
        now = self._clock()
        for s in self._sightings[-20:]:
            if s["name"] == name and abs(s["x"] - pos.x) < 30 and abs(s["y"] - pos.y) < 30:
                if now - s["time"] < 60:
                    return
        self._sightings.append(
            {
                "x": round(pos.x),
                "y": round(pos.y),
                "name": name,
                "level": level,
                "time": now,
            }
        )
        self._update_heat(pos, weight=1.0, event_time=now)

    def record_empty_scan(self, pos: Point) -> None:
        """Record that no valid npcs were found near this location."""
        now = self._clock()
        self._empty_scans.append(
            {
                "x": round(pos.x),
                "y": round(pos.y),
                "time": now,
            }
        )
        self._update_heat(pos, weight=-0.5, event_time=now)

    def trim_lists(self) -> None:
        """Cap rolling lists to prevent unbounded growth in long sessions.

        Uses the same limits as save() so runtime memory matches what
        would be persisted to disk.
        """
        cutoff = self._clock() - DECAY_HOURS * 3600 * 3
        if len(self._kills) > 1000:
            self._kills = [k for k in self._kills if k["time"] > cutoff]
        if len(self._sightings) > 500:
            self._sightings = self._sightings[-500:]
        if len(self._empty_scans) > 200:
            self._empty_scans = self._empty_scans[-200:]
        if len(self._danger_events) > 50:
            self._danger_events = self._danger_events[-50:]

    def mark_visited(self, pos: Point) -> None:
        """Mark a cell as recently visited. Suppresses best_direction for 3 min."""
        cell = (int(pos.x // CELL_SIZE), int(pos.y // CELL_SIZE))
        self._visited[cell] = self._clock()

    def best_direction(self, pos: Point, radius: float = 400.0) -> Point | None:
        """Return the position of the highest-scoring area within radius.

        Used by wander to bias toward known npc-dense areas.
        Skips cells visited in the last 3 minutes to prevent retracing.
        Returns None if no data.
        """
        if not self._heat:
            return None

        now = self._clock()
        visit_cooldown = 180.0  # 3 minutes

        best_score = -999.0
        best_pos = None
        fx, fy = int(pos.x // CELL_SIZE), int(pos.y // CELL_SIZE)
        cell_radius = int(radius // CELL_SIZE) + 1

        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell = (fx + dx, fy + dy)
                score = self._heat.get(cell, 0.0)
                if score <= 0:
                    continue
                last_visit = self._visited.get(cell, 0.0)
                if now - last_visit < visit_cooldown:
                    continue
                wx = (cell[0] + 0.5) * CELL_SIZE
                wy = (cell[1] + 0.5) * CELL_SIZE
                dist = math.hypot(wx - pos.x, wy - pos.y)
                if dist < 30 or dist > radius:
                    continue
                adjusted = score / max(dist / 100, 1.0)
                if adjusted > best_score:
                    best_score = adjusted
                    best_pos = Point(wx, wy, pos.z)

        return best_pos

    @property
    def total_kills(self) -> int:
        return len(self._kills)

    @property
    def total_sightings(self) -> int:
        return len(self._sightings)

    def heat_at(self, pos: Point) -> float:
        """Get heat score at a position."""
        cell = (int(pos.x // CELL_SIZE), int(pos.y // CELL_SIZE))
        return self._heat.get(cell, 0.0)

    # -- Heat map -----------------------------------------------------

    def _update_heat(self, pos: Point, weight: float, event_time: float = 0.0) -> None:
        cell = (int(pos.x // CELL_SIZE), int(pos.y // CELL_SIZE))
        # Apply exponential time decay: recent events count more
        if event_time > 0:
            age_hours = (self._clock() - event_time) / 3600.0
            decay = math.exp(-age_hours / DECAY_HOURS)
            weight *= decay
        self._heat[cell] = self._heat.get(cell, 0.0) + weight

    # -- Persistence --------------------------------------------------

    def save(self) -> None:
        """Save memory to disk."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        path = self._data_dir / f"{self.zone_name}.json"

        # Prune old data before saving
        cutoff = self._clock() - DECAY_HOURS * 3600 * 3  # keep 3x decay window
        self._kills = [k for k in self._kills if k["time"] > cutoff]
        self._sightings = self._sightings[-500:]  # cap sightings
        self._empty_scans = self._empty_scans[-200:]
        self._danger_events = self._danger_events[-50:]

        data = {
            "v": 1,
            "zone": self.zone_name,
            "saved": self._clock(),
            "defeats": self._kills,
            "sightings": self._sightings,
            "empty_scans": self._empty_scans,
            "danger_events": self._danger_events,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info(
            "[SPATIAL] saved %d defeats, %d sightings to %s",
            len(self._kills),
            len(self._sightings),
            path.name,
        )

    def _load(self) -> None:
        """Load memory from disk."""
        path = self._data_dir / f"{self.zone_name}.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            if "v" not in data:
                log.info("[SPATIAL] no schema version in %s (pre-v1)", path.name)
            self._kills = data.get("defeats", [])
            self._sightings = data.get("sightings", [])
            self._empty_scans = data.get("empty_scans", [])
            self._danger_events = data.get("danger_events", [])

            # Rebuild heat map from loaded data (with time decay)
            for k in self._kills:
                self._update_heat(Point(k["x"], k["y"], 0.0), weight=3.0, event_time=k.get("time", 0))
            for s in self._sightings:
                self._update_heat(Point(s["x"], s["y"], 0.0), weight=1.0, event_time=s.get("time", 0))
            for e in self._empty_scans:
                self._update_heat(Point(e["x"], e["y"], 0.0), weight=-0.5, event_time=e.get("time", 0))
            for d in self._danger_events:
                self._update_heat(Point(d["x"], d["y"], 0.0), weight=-5.0, event_time=d.get("time", 0))

            log.info(
                "[SPATIAL] loaded %d defeats, %d sightings from %s",
                len(self._kills),
                len(self._sightings),
                path.name,
            )
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("[SPATIAL] failed to load %s: %s", path, e)
