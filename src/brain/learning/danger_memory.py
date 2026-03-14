"""Danger memory: persistent learning from deaths and flees.

Records which entity types caused deaths/flees and produces a penalty
score that feeds into target selection. Prevents the agent from
repeatedly engaging the same dangerous entity type.

Danger decays over 48 hours so the agent retries when stronger.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from util.log_tiers import EVENT, VERBOSE

log = logging.getLogger(__name__)

# Decay: 48 hours half-life for danger records
DECAY_HOURS = 48.0
# Threshold for temporary avoidance
AVOID_DEATHS = 3
# Max records per entity type
MAX_INCIDENTS = 20


@dataclass(slots=True)
class DangerRecord:
    """Aggregated danger data for one entity type."""

    entity_type: str
    deaths: int = 0
    flees: int = 0
    last_incident: float = 0.0
    incidents: list[dict[str, object]] = field(default_factory=list)


class DangerMemory:
    """Persistent memory of dangerous entity types from deaths and flees."""

    def __init__(self, zone: str = "", data_dir: str = "", clock: Callable[[], float] = time.time) -> None:
        self._zone = zone
        self._data_dir = data_dir or str(Path("data") / "memory")
        self._clock = clock
        self._records: dict[str, DangerRecord] = {}
        self._recent_deaths = 0  # deaths this session (not persisted)
        self._dirty = False  # write throttle: save on next flush, not every event
        self._load()

    @property
    def recent_deaths(self) -> int:
        """Deaths recorded this session."""
        return self._recent_deaths

    def record_death(self, entity_type: str, context: dict[str, object] | None = None) -> None:
        """Record a death caused by this entity type."""
        r = self._get_or_create(entity_type)
        r.deaths += 1
        r.last_incident = self._clock()
        r.incidents.append({"type": "death", "time": self._clock(), **(context or {})})
        if len(r.incidents) > MAX_INCIDENTS:
            r.incidents = r.incidents[-MAX_INCIDENTS:]
        self._recent_deaths += 1
        self._dirty = True  # flushed by save() at session end or periodic
        log.log(
            EVENT,
            "[LEARNING] Danger memory: death from '%s' (total deaths=%d, flees=%d)",
            entity_type,
            r.deaths,
            r.flees,
        )

    def record_flee(self, entity_type: str, context: dict[str, object] | None = None) -> None:
        """Record a flee triggered by this entity type."""
        r = self._get_or_create(entity_type)
        r.flees += 1
        r.last_incident = self._clock()
        r.incidents.append({"type": "flee", "time": self._clock(), **(context or {})})
        if len(r.incidents) > MAX_INCIDENTS:
            r.incidents = r.incidents[-MAX_INCIDENTS:]
        self._dirty = True  # flushed by save() at session end or periodic
        log.log(
            VERBOSE,
            "[LEARNING] Danger memory: flee from '%s' (total deaths=%d, flees=%d)",
            entity_type,
            r.deaths,
            r.flees,
        )

    def danger_penalty(self, entity_type: str) -> float:
        """0.0 (safe) to 1.0 (deadly). Feeds into target scoring.

        3+ deaths = 1.0. Decays over 48 hours from last incident.
        """
        r = self._records.get(entity_type)
        if r is None:
            return 0.0
        # Decay factor based on time since last incident
        # Guard: if last_incident is unset/corrupted (0.0), treat as fresh
        if r.last_incident <= 0:
            hours_ago = 0.0
        else:
            hours_ago = (self._clock() - r.last_incident) / 3600
        decay = 0.5 ** (hours_ago / DECAY_HOURS)
        # Raw danger from incident counts
        raw = min(1.0, r.deaths * 0.35 + r.flees * 0.10)
        return float(raw * decay)

    def should_avoid(self, entity_type: str) -> bool:
        """True if 3+ deaths from this type with significant recency."""
        r = self._records.get(entity_type)
        if r is None:
            return False
        if r.deaths < AVOID_DEATHS:
            return False
        # Only avoid if recent (within 2x decay period)
        hours_ago = (self._clock() - r.last_incident) / 3600
        return hours_ago < DECAY_HOURS * 2

    def record_plan_failure(self, step_name: str, reason: str) -> None:
        """Track plan step failures for cost penalty learning."""
        key = f"__plan__{step_name}"
        r = self._get_or_create(key)
        r.flees += 1  # reuse flees counter for plan failures
        r.last_incident = self._clock()
        r.incidents.append({"type": "plan_fail", "time": self._clock(), "reason": reason})
        if len(r.incidents) > MAX_INCIDENTS:
            r.incidents = r.incidents[-MAX_INCIDENTS:]

    def plan_step_penalty(self, step_name: str) -> float:
        """Extra cost (seconds) to add for plan steps that frequently fail."""
        key = f"__plan__{step_name}"
        r = self._records.get(key)
        if r is None or r.flees < 3:
            return 0.0
        # 5 seconds per failure beyond the threshold
        return (r.flees - 2) * 5.0

    def _get_or_create(self, entity_type: str) -> DangerRecord:
        if entity_type not in self._records:
            self._records[entity_type] = DangerRecord(entity_type=entity_type)
        return self._records[entity_type]

    # -- Persistence -----------------------------------------------------------

    def _path(self) -> str:
        return str(Path(self._data_dir) / f"{self._zone}_danger.json")

    def _load(self) -> None:
        path = self._path()
        if not Path(path).exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for name, rd in data.get("records", {}).items():
                self._records[name] = DangerRecord(
                    entity_type=name,
                    deaths=rd.get("deaths", 0),
                    flees=rd.get("flees", 0),
                    last_incident=rd.get("last", 0.0),
                    incidents=rd.get("incidents", []),
                )
            if self._records:
                log.info("[LIFECYCLE] DangerMemory: loaded %d records for %s", len(self._records), self._zone)
        except (OSError, json.JSONDecodeError, TypeError) as e:
            log.warning("[LIFECYCLE] DangerMemory: load failed: %s", e)

    def _save(self) -> None:
        path = self._path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        records: dict[str, dict[str, object]] = {}
        for name, r in self._records.items():
            records[name] = {
                "deaths": r.deaths,
                "flees": r.flees,
                "last": r.last_incident,
                "incidents": r.incidents[-MAX_INCIDENTS:],
            }
        try:
            with open(path, "w") as f:
                json.dump({"v": 1, "records": records}, f, separators=(",", ":"))
        except OSError as e:
            log.warning("[LIFECYCLE] DangerMemory: save failed: %s", e)

    def save(self) -> None:
        """Public save for session cleanup. Also called periodically."""
        self._save()
        self._dirty = False
