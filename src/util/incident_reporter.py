"""Incident reporter -- composite narrative events for deaths and flees.

When a terminal event occurs (player death, flee, pet death during combat),
assembles a causal chain from the forensics ring buffer and recent cycle
data. Emits a single incident event that tells the complete story.

Thread ownership: brain thread only (called from brain_runner on death/flee).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from util.structured_log import log_event

if TYPE_CHECKING:
    from brain.context import AgentContext
    from perception.state import GameState

log = logging.getLogger(__name__)


class IncidentReporter:
    """Builds composite incident reports from forensics and cycle data.

    The forensics ring buffer provides the last ~300 ticks (30s) of
    compact state. This reporter reads it to trace HP drain sequences,
    npc HP regeneration during flee, distance traveled, and duration.
    """

    def __init__(self) -> None:
        self._last_death_time: float = 0.0
        self._last_flee_time: float = 0.0

    def report_death(
        self, state: GameState, ctx: AgentContext, source: str = "", buffer: list[dict] | None = None
    ) -> None:
        """Build and emit an incident report for player death.

        Args:
            state: Current GameState at time of death.
            ctx: AgentContext with forensics buffer and cycle data.
            source: Death source string (e.g. "hp_zero").
            buffer: Pre-captured forensics snapshot. If None, reads from
                ctx.diag.forensics (which may be empty if already flushed).
        """
        if time.time() - self._last_death_time < 5.0:
            return  # debounce duplicate death detections
        self._last_death_time = time.time()

        if buffer is None:
            forensics = getattr(getattr(ctx, "diag", None), "forensics", None)
            buffer = forensics.snapshot() if forensics else []
        cycle_tracker = getattr(getattr(ctx, "diag", None), "cycle_tracker", None)

        report = self._build_report("player_death", buffer, state, ctx)
        report["source"] = source
        report["cycle_id"] = cycle_tracker.cycle_id if cycle_tracker else 0
        report["defeats_before_incident"] = getattr(getattr(ctx, "defeat_tracker", None), "defeats", 0)

        summary = self._build_summary("death", report)
        report["summary"] = summary

        log_event(log, "incident", summary, level=logging.CRITICAL, **report)

    def report_flee(self, state: GameState, ctx: AgentContext, trigger_reason: str = "") -> None:
        """Build and emit an incident report for flee activation.

        Only emits once per flee (debounced). The flee routine may
        emit multiple flee_trigger events as it runs.
        """
        if time.time() - self._last_flee_time < 30.0:
            return  # one flee incident per 30s max
        self._last_flee_time = time.time()

        forensics = getattr(getattr(ctx, "diag", None), "forensics", None)
        buffer = forensics.snapshot() if forensics else []
        cycle_tracker = getattr(getattr(ctx, "diag", None), "cycle_tracker", None)

        report = self._build_report("flee", buffer, state, ctx)
        report["trigger_reason"] = trigger_reason
        report["cycle_id"] = cycle_tracker.cycle_id if cycle_tracker else 0

        summary = self._build_summary("flee", report)
        report["summary"] = summary

        log_event(log, "incident", summary, level=logging.WARNING, **report)

    def _build_report(
        self, incident_type: str, buffer: Sequence[dict], state: GameState, ctx: AgentContext
    ) -> dict:
        """Extract narrative data from the forensics ring buffer."""
        report: dict = {"type": incident_type}

        if not buffer:
            report["hp_sequence"] = []
            report["mob_hp_sequence"] = []
            report["flee_distance"] = 0
            report["flee_duration_s"] = 0
            report["trigger_mob"] = ""
            report["mana_at_trigger"] = 0
            report["guards_nearby"] = False
            return report

        # HP drain sequence (deduplicated)
        hp_seq: list[float] = []
        prev_hp = -1.0
        for entry in buffer:
            hp = entry.get("hp", 0)
            hp_max = entry.get("hp_max", 1)
            hp_pct = round(hp / max(hp_max, 1), 3) if hp_max > 0 else 0.0
            if hp_pct != prev_hp:
                hp_seq.append(hp_pct)
                prev_hp = hp_pct
        report["hp_sequence"] = hp_seq[-20:]  # last 20 distinct values

        # Npc HP sequence (if chasing npc visible)
        mob_hp_seq: list[int] = []
        prev_mob_hp = -1
        trigger_mob = ""
        for entry in buffer:
            tgt = entry.get("tgt", "")
            tgt_hp = entry.get("tgt_hp", 0)
            if tgt and tgt_hp > 0:
                if not trigger_mob:
                    trigger_mob = tgt
                if tgt == trigger_mob and tgt_hp != prev_mob_hp:
                    mob_hp_seq.append(tgt_hp)
                    prev_mob_hp = tgt_hp
        report["mob_hp_sequence"] = mob_hp_seq[-10:]
        report["trigger_mob"] = trigger_mob

        # Flee distance (start to end position)
        first = buffer[0]
        last = buffer[-1]
        dx = last.get("x", 0) - first.get("x", 0)
        dy = last.get("y", 0) - first.get("y", 0)
        report["flee_distance"] = round((dx * dx + dy * dy) ** 0.5)

        # Flee duration
        first_ts = first.get("tick", 0)
        last_ts = last.get("tick", 0)
        # Approximate: 10 ticks/sec
        tick_delta = last_ts - first_ts
        report["flee_duration_s"] = round(tick_delta / 10.0, 1)

        # Mana at start of incident
        report["mana_at_trigger"] = first.get("mana", 0)

        # Position
        report["pos_start_x"] = round(first.get("x", 0))
        report["pos_start_y"] = round(first.get("y", 0))
        report["pos_end_x"] = round(last.get("x", 0))
        report["pos_end_y"] = round(last.get("y", 0))

        # Guards nearby (check current spawns)
        guards_nearby = False
        spawns = getattr(state, "spawns", ())
        px = getattr(state, "x", 0.0)
        py = getattr(state, "y", 0.0)
        for s in spawns:
            name = getattr(s, "name", "")
            if "Guard" in name or "Captain" in name or "guard" in name:
                sx = getattr(s, "x", 0.0)
                sy = getattr(s, "y", 0.0)
                d = ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
                if d < 300:
                    guards_nearby = True
                    break
        report["guards_nearby"] = guards_nearby

        return report

    def _build_summary(self, incident_type: str, report: dict) -> str:
        """Build a human-readable summary string from the report."""
        npc = report.get("trigger_mob", "unknown")
        reason = report.get("trigger_reason", "")

        if incident_type == "death":
            flee_dist = report.get("flee_distance", 0)
            flee_dur = report.get("flee_duration_s", 0)
            parts = [f"Died fighting {npc}"]
            if reason:
                parts.append(f"trigger={reason}")
            if flee_dist > 50:
                parts.append(f"fled {flee_dist}u over {flee_dur}s")
            guards = report.get("guards_nearby", False)
            if guards:
                parts.append("guards nearby but did not save")
            return " | ".join(parts)

        else:  # flee
            parts = [f"Fled from {npc}"]
            if reason:
                parts.append(f"trigger={reason}")
            hp_seq = report.get("hp_sequence", [])
            if hp_seq:
                parts.append(f"HP {hp_seq[0]:.0%}->{hp_seq[-1]:.0%}")
            return " | ".join(parts)
