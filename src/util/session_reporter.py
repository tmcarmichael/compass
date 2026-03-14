"""Session reporting: JSON session report, periodic snapshots, XP tracking.

Extracted from brain_runner.py. These are pure observation/logging
methods with no side effects on brain state.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from brain.learning.scorecard import compute_scorecard
from core import __version__
from core.constants import XP_SCALE_MAX
from nav.movement import get_stuck_event_count
from perception.combat_eval import is_valid_target
from util.event_schemas import SessionEndEvent, SnapshotEvent
from util.log_tiers import EVENT
from util.structured_log import log_event

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.runner.loop import BrainRunner
    from brain.world.health import HealthMonitor
    from perception.state import GameState

brain_log = logging.getLogger("compass.brain_loop")


class SessionReporter:
    """Handles session reporting, periodic snapshots, and XP tracking.

    Composed into BrainRunner -- not a subclass. The runner passes
    references to its own state (ctx, brain, reader, etc.).

    Args:
        runner: The parent BrainRunner instance.
    """

    def __init__(self, runner: BrainRunner) -> None:
        self._runner = runner
        # Derivative tracking for snapshot rate-of-change fields
        self._prev_mana: int = 0
        self._prev_mana_time: float = 0.0
        self._prev_camp_dist: float = 0.0
        self._prev_camp_dist_time: float = 0.0

    def write_session_report(self, ctx: AgentContext, session_dir: str) -> None:
        """Write machine-readable session report JSON at shutdown."""
        try:
            elapsed = time.time() - ctx.metrics.session_start
            hours = max(elapsed / 3600, 0.01)
            dph = ctx.defeat_tracker.defeats / hours

            from datetime import datetime

            start_epoch = ctx.metrics.session_start
            report = {
                "v": 1,
                "version": __version__,
                "session_id": self._runner._session_id,
                "zone": self._runner._current_zone,
                "session_start_epoch": round(start_epoch, 1),
                "session_start_local": datetime.fromtimestamp(start_epoch).strftime("%Y-%m-%d %H:%M:%S"),
                "duration_s": round(elapsed, 1),
                "defeats": ctx.defeat_tracker.defeats,
                "defeats_per_hr": round(dph, 1),
                "deaths": ctx.player.deaths,
                "flees": ctx.metrics.flee_count,
                "rests": ctx.metrics.rest_count,
                "casts": ctx.metrics.total_casts,
            }

            # XP
            report["xp_gained_pct"] = round(ctx.metrics.xp_gained_pct, 2)
            xp_hr = ctx.metrics.xp_per_hour()
            report["xp_per_hr"] = round(xp_hr, 1) if xp_hr > 0 else 0

            # Snapshot mutable collections under lock (free-threaded safety)
            with ctx.lock:
                mc = ctx.metrics.snapshot_collections()

            # Routine time breakdown
            report["routine_time"] = (
                {k: round(v, 1) for k, v in mc["routine_time"].items()} if mc["routine_time"] else {}
            )

            # Routine counts and failures
            report["routine_counts"] = dict(mc["routine_counts"])
            report["routine_failures"] = dict(mc["routine_failures"])

            # Combat efficiency
            if ctx.defeat_tracker.defeats > 0 and ctx.metrics.total_casts > 0:
                report["casts_per_kill"] = round(ctx.metrics.total_casts / ctx.defeat_tracker.defeats, 2)
            if mc["total_cycle_times"]:
                report["avg_cycle_s"] = round(sum(mc["total_cycle_times"]) / len(mc["total_cycle_times"]), 1)

            # Acquire stats
            if ctx.metrics.acquire_tab_totals:
                report["avg_tabs_per_acquire"] = round(
                    sum(ctx.metrics.acquire_tab_totals) / len(ctx.metrics.acquire_tab_totals), 2
                )
            report["acquire_invalid_tabs"] = ctx.metrics.acquire_invalid_tabs

            # Stuck events
            report["stuck_events"] = get_stuck_event_count()

            # Loot stats
            report["corpses_looted"] = ctx.inventory.loot_count

            # Scorecard
            try:
                scores = compute_scorecard(ctx)
                report["scorecard"] = {k: v for k, v in scores.items() if not k.startswith("_")}
            except (AttributeError, TypeError, ZeroDivisionError, KeyError) as e:
                brain_log.debug("[SNAPSHOT] Failed to compute scorecard: %s", e)

            # Fight history summary (top npcs by danger)
            if ctx.fight_history:
                try:
                    mob_stats = {}
                    for name, stats in ctx.fight_history.get_all_stats().items():
                        mob_stats[name] = {
                            "fights": stats.fights,
                            "defeats": stats.defeats,
                            "avg_duration": round(stats.avg_duration, 1),
                            "avg_mana": round(stats.avg_mana),
                            "danger": round(stats.danger_score, 2),
                            "pet_death_rate": round(stats.pet_death_rate, 2),
                        }
                    report["mob_stats"] = mob_stats
                except (AttributeError, TypeError, ZeroDivisionError) as e:
                    brain_log.debug("[SNAPSHOT] Failed to build fight history summary: %s", e)

            # Metrics engine summary (tick percentiles, success rates)
            if ctx.diag.metrics:
                report["metrics"] = ctx.diag.metrics.summary()
            if ctx.diag.invariants:
                report["invariants"] = ctx.diag.invariants.summary()

            # Tick budget stats (Phase 2a hardening)
            if ctx.diag.tick_overbudget_count > 0:
                report["tick_budget"] = {
                    "overbudget_count": ctx.diag.tick_overbudget_count,
                    "max_ms": round(ctx.diag.tick_overbudget_max_ms, 1),
                    "worst_routine": ctx.diag.tick_overbudget_last_routine,
                }

            # Session narrative -- phase-by-phase index
            try:
                report["narrative"] = self._build_narrative(ctx, elapsed)
            except (AttributeError, TypeError, ZeroDivisionError, IndexError) as e:
                brain_log.debug("[NARRATIVE] Failed to build narrative: %s", e)

            # Write report
            report_path = str(Path(session_dir) / f"{self._runner._session_id}_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            brain_log.info("[LIFECYCLE] Session report: %s", report_path)

            scorecard = report.get("scorecard", {})
            grade = scorecard.get("grade", "?") if isinstance(scorecard, dict) else "?"
            log_event(
                brain_log,
                "session_end",
                f"[LIFECYCLE] Session: {elapsed / 60:.0f}min, {ctx.defeat_tracker.defeats} defeats, {ctx.player.deaths} deaths, grade={grade}",
                **SessionEndEvent(
                    duration=round(elapsed, 1),
                    defeats=ctx.defeat_tracker.defeats,
                    deaths=ctx.player.deaths,
                    flees=ctx.metrics.flee_count,
                    dph=round(dph, 1),
                    xp_pct=round(ctx.metrics.xp_gained_pct, 2),
                    grade=grade,
                    loot_count=ctx.inventory.loot_count,
                ),
            )

        except Exception as e:
            brain_log.warning("[LIFECYCLE] Session report: failed to write: %s", e)

    def _format_phase_line(
        self,
        ctx: AgentContext,
        phase_name: str,
        start_min: float,
        end_min: float,
        duration_s: float,
        defeats: int,
        dph: float,
    ) -> str:
        """Format a single phase entry for the narrative."""
        label = phase_name.upper()
        prefix = f"{start_min:.0f}-{end_min:.0f}min: {label}"

        if phase_name == "grinding" and defeats > 0:
            avg_cycle = ""
            if ctx.defeat_tracker.defeat_cycle_times:
                recent = ctx.defeat_tracker.defeat_cycle_times[-defeats:] if defeats > 0 else []
                if recent:
                    avg_s = sum(recent) / len(recent)
                    avg_cycle = f", avg {avg_s:.0f}s/cycle"
            return f"{prefix} -- {defeats} defeats at {dph:.0f} dph{avg_cycle}"
        if phase_name == "grinding":
            return f"{prefix} -- no defeats, {duration_s:.0f}s"
        if phase_name == "resting":
            return f"{prefix} -- mana recovery, {duration_s:.0f}s"
        if phase_name == "incident":
            return f"{prefix} -- flee/death, {duration_s:.0f}s"
        if phase_name == "startup":
            pet_str = "pet alive" if ctx.pet.alive else "no pet"
            return f"{prefix} -- warmup, {pet_str}"
        if phase_name == "idle":
            return f"{prefix} -- no activity, {duration_s:.0f}s"
        return f"{prefix} -- {duration_s:.0f}s"

    def _build_narrative(self, ctx: AgentContext, elapsed: float) -> list[str]:
        """Build a phase-by-phase session narrative index.

        Uses phase detector history, defeat tracker, and session metrics
        to produce human-readable narrative lines for the report JSON.

        Args:
            ctx: AgentContext with session data.
            elapsed: Total session duration in seconds.

        Returns:
            List of narrative strings describing session phases and summary.
        """
        lines: list[str] = []
        session_start = ctx.metrics.session_start

        # Finalize phase detector so the current (still-active) phase is included
        phase_detector = getattr(ctx.diag, "phase_detector", None)
        if phase_detector:
            phase_detector.finalize(ctx)

        # Build phase lines from phase detector history
        phases = phase_detector.history if phase_detector else []
        if phases:
            for phase_name, start_epoch, duration_s, defeats, dph in phases:
                offset_s = start_epoch - session_start
                start_min = max(offset_s / 60, 0)
                end_min = start_min + duration_s / 60
                lines.append(
                    self._format_phase_line(
                        ctx,
                        phase_name,
                        start_min,
                        end_min,
                        duration_s,
                        defeats,
                        dph,
                    )
                )
                # Notable sub-events within grinding phases
                if phase_name == "grinding" and defeats > 3:
                    recent = ctx.defeat_tracker.defeat_cycle_times[-defeats:]
                    if recent:
                        avg_s_cycle = sum(recent) / len(recent)
                        lines.append(f"  Notable: avg {avg_s_cycle:.0f}s/cycle")
        else:
            # No phase detector -- build a simple single-phase narrative
            defeats = ctx.defeat_tracker.defeats
            hours = max(elapsed / 3600, 0.01)
            dph = defeats / hours
            if defeats > 0:
                lines.append(f"0-{elapsed / 60:.0f}min: GRINDING -- {defeats} defeats at {dph:.0f} dph")
            else:
                lines.append(f"0-{elapsed / 60:.0f}min: SESSION -- {elapsed:.0f}s, no defeats")

        # Summary line
        defeats = ctx.defeat_tracker.defeats
        deaths = ctx.player.deaths
        hours = max(elapsed / 3600, 0.01)
        dph = defeats / hours
        grade = "?"
        try:
            scores = compute_scorecard(ctx)
            grade = str(scores.get("grade", "?"))
        except Exception:
            pass
        lines.append(
            f"SUMMARY: {defeats} defeats, {deaths} death{'s' if deaths != 1 else ''}, "
            f"grade {grade}, {dph:.1f} dph"
        )

        brain_log.debug("[NARRATIVE] Built %d narrative lines for session report", len(lines))
        return lines

    def track_xp(self, state: GameState, ctx: AgentContext, now: float) -> None:
        """Detect XP-based defeats from memory and track XP rate."""
        if state.xp_pct_raw > ctx.metrics.xp_last_raw and ctx.metrics.xp_last_raw > 0:
            xp_delta = state.xp_pct_raw - ctx.metrics.xp_last_raw
            if 0 < xp_delta < 100:
                ctx.defeat_tracker.xp_gains += 1
                defeat_name = ctx.defeat_tracker.last_fight_name or (
                    state.target.name if state.target else "unknown"
                )
                defeat_id = ctx.defeat_tracker.last_fight_id or (state.target.spawn_id if state.target else 0)
                defeat_x = ctx.defeat_tracker.last_fight_x or (state.target.x if state.target else state.x)
                defeat_y = ctx.defeat_tracker.last_fight_y or (state.target.y if state.target else state.y)
                recent = (
                    any(now - t < 5.0 for _, t in ctx.defeat_tracker.recent_kills[-1:])
                    if ctx.defeat_tracker.recent_kills
                    else False
                )
                if not recent:
                    brain_log.info(
                        "[LIFECYCLE] XP gain detected (memory): delta=%d, npc='%s'", xp_delta, defeat_name
                    )
                    from core.types import Point

                    ctx.record_kill(defeat_id, name=defeat_name, pos=Point(defeat_x, defeat_y, state.z))

        if state.xp_pct_raw > 0 and ctx.metrics.xp_last_raw > 0:
            if state.xp_pct_raw < ctx.metrics.xp_last_raw:
                ctx.metrics.xp_gained_pct += (
                    (XP_SCALE_MAX - ctx.metrics.xp_last_raw) / float(XP_SCALE_MAX) * 100
                )
                ctx.metrics.xp_gained_pct += state.xp_pct_raw / float(XP_SCALE_MAX) * 100
            elif state.xp_pct_raw > ctx.metrics.xp_last_raw:
                ctx.metrics.xp_gained_pct += (
                    (state.xp_pct_raw - ctx.metrics.xp_last_raw) / float(XP_SCALE_MAX) * 100
                )
        if state.xp_pct_raw > 0:
            ctx.metrics.xp_last_raw = state.xp_pct_raw

    def _build_snapshot_event(
        self,
        state: GameState,
        ctx: AgentContext,
        now: float,
        dph: float,
        kph_5m: float,
        defeat_age: float,
        live_npcs: list,
        player_count: int,
        camp_dist: float,
    ) -> SnapshotEvent:
        """Compute derivative fields and build the structured snapshot event."""
        mana_current = getattr(state, "mana_current", 0)
        snap_kwargs: SnapshotEvent = SnapshotEvent(
            hp_pct=round(state.hp_pct, 3),
            mana_pct=round(state.mana_pct, 3),
            pos_x=round(state.x),
            pos_y=round(state.y),
            defeats=ctx.defeat_tracker.defeats,
            dph=round(dph, 1),
            engaged=ctx.combat.engaged,
            routine=self._runner._brain._active_name or "none",
            pet_alive=ctx.pet.alive,
            npcs_nearby=len(live_npcs),
            camp_dist=round(camp_dist),
            players=player_count,
            acq_fails=ctx.metrics.consecutive_acquire_fails,
            last_kill_age=round(defeat_age),
        )

        # Mana rate (mana/min, negative = draining)
        if self._prev_mana_time > 0:
            dt_min = (now - self._prev_mana_time) / 60.0
            if dt_min > 0.1:
                mana_delta = mana_current - self._prev_mana
                snap_kwargs["mana_rate"] = round(mana_delta / dt_min, 1)
        self._prev_mana = mana_current
        self._prev_mana_time = now

        snap_kwargs["kph_5min"] = round(kph_5m, 1)
        snap_kwargs["kph_delta"] = round(kph_5m - dph, 1)

        # Camp drift rate (units/min)
        if self._prev_camp_dist_time > 0:
            dt_min = (now - self._prev_camp_dist_time) / 60.0
            if dt_min > 0.1:
                drift = camp_dist - self._prev_camp_dist
                snap_kwargs["camp_drift_rate"] = round(drift / dt_min, 1)
        self._prev_camp_dist = camp_dist
        self._prev_camp_dist_time = now

        if ctx.defeat_tracker.defeat_cycle_times:
            recent = ctx.defeat_tracker.defeat_cycle_times[-10:]
            snap_kwargs["avg_cycle_s"] = round(sum(recent) / len(recent), 1)

        phase_detector = getattr(ctx.diag, "phase_detector", None)
        if phase_detector:
            snap_kwargs["phase"] = phase_detector.current_phase

        return snap_kwargs

    def periodic_snapshot(
        self, state: GameState, ctx: AgentContext, now: float, health_monitor: HealthMonitor
    ) -> None:
        """Log 30-second snapshot: player stats, nearby NPCs, defeat rate, XP, drought."""
        live_npcs = []
        for s in state.spawns:
            if s.is_npc and s.hp_current > 0:
                d = state.pos.dist_to(s.pos)
                live_npcs.append((s, d))
        live_npcs.sort(key=lambda x: x[1])
        pose = {0: "standing", 1: "sitting", 2: "medding"}.get(state.stand_state, f"?{state.stand_state}")
        player_dist = ctx.nearest_player_dist(state)
        player_count = ctx.nearby_player_count(state, radius=200)
        elapsed = now - ctx.metrics.session_start
        dph = ctx.defeat_tracker.defeats / max(elapsed / 3600, 0.01)
        target_info = ""
        if state.target:
            td = state.pos.dist_to(state.target.pos)
            target_info = (
                f" Target='{state.target.name}' "
                f"id={state.target.spawn_id} dist={td:.0f} "
                f"HP={state.target.hp_current}/{state.target.hp_max}"
            )
        kph_5m = ctx.defeat_tracker.defeat_rate_window(300)
        defeats_5m = ctx.defeat_tracker.defeats_in_window(300)
        defeat_age = ctx.defeat_tracker.last_kill_age()
        drought_str = f" DROUGHT={defeat_age:.0f}s" if defeat_age > 60 else ""
        active_routine = self._runner._brain._active
        locked_str = ""
        if active_routine and active_routine.locked:
            locked_str = f" LOCKED={self._runner._brain._active_name}"

        ctx.metrics.record_xp_sample(now, state.xp_pct_raw)
        xp_hr = ctx.metrics.xp_per_hour()
        ttl = ctx.metrics.time_to_level()
        ttl_str = f" TTL={ttl:.0f}min" if ttl else ""
        xp_rate_str = f" XP_rate={xp_hr:.1f}%/hr{ttl_str}" if xp_hr > 0 else ""

        brain_log.log(
            EVENT,
            "[SNAPSHOT] SNAPSHOT | HP=%.0f%% Mana=%.0f%% Pos=(%.0f,%.0f) Hdg=%.0f "
            "Pet=%s(id=%s,'%s') Engaged=%s "
            "Npcs=%d(%.1f/hr) 5min=%d(%.1f/hr) XP=%d "
            "NPCs=%d Camp_dist=%.0f Players=%d(nearest=%.0f) "
            "Routine=%s Pose=%s AcqFails=%d "
            "LastKill=%.0fs%s%s%s%s",
            state.hp_pct * 100,
            state.mana_pct * 100,
            state.x,
            state.y,
            state.heading,
            "yes" if ctx.pet.alive else "no",
            ctx.pet.spawn_id or "?",
            ctx.pet.name or "?",
            "yes" if ctx.combat.engaged else "no",
            ctx.defeat_tracker.defeats,
            dph,
            defeats_5m,
            kph_5m,
            ctx.defeat_tracker.xp_gains,
            len(live_npcs),
            ctx.camp.distance_to_camp(state),
            player_count,
            player_dist,
            self._runner._brain._active_name or "none",
            pose,
            ctx.metrics.consecutive_acquire_fails,
            defeat_age,
            drought_str,
            locked_str,
            target_info,
            xp_rate_str,
        )
        # Build structured snapshot and log it
        camp_dist = ctx.camp.distance_to_camp(state)
        snap_kwargs = self._build_snapshot_event(
            state,
            ctx,
            now,
            dph,
            kph_5m,
            defeat_age,
            live_npcs,
            player_count,
            camp_dist,
        )
        log_event(brain_log, "snapshot", "[SNAPSHOT] snapshot periodic", **snap_kwargs)
        for s, d in live_npcs[:6]:
            brain_log.debug(
                "[SNAPSHOT]   NPC: '%s' id=%d dist=%.0f pos=(%.0f,%.0f) HP=%d/%d lv=%d",
                s.name,
                s.spawn_id,
                d,
                s.x,
                s.y,
                s.hp_current,
                s.hp_max,
                s.level,
            )

        # Tick profiling
        brain = self._runner._brain
        if hasattr(brain, "tick_total_ms"):
            world_ms = getattr(ctx.world, "update_ms", 0)
            brain_log.debug(
                "[SNAPSHOT]   TICK: total=%.1fms rules=%.1fms routine=%.1fms world=%.1fms",
                brain.tick_total_ms,
                sum(brain.rule_times.values()),
                brain.routine_tick_ms,
                world_ms,
            )

        # Grind drought warning
        if defeat_age > 60 and ctx.defeat_tracker.defeats > 0 and not ctx.combat.engaged:
            valid_nearby = sum(1 for s, d in live_npcs if d < 200 and is_valid_target(s, state.level))
            brain_log.warning(
                "[SNAPSHOT] DROUGHT: %.0fs since last npc | "
                "valid_mobs_nearby=%d camp_dist=%.0f acq_fails=%d routine=%s",
                defeat_age,
                valid_nearby,
                ctx.camp.distance_to_camp(state),
                ctx.metrics.consecutive_acquire_fails,
                self._runner._brain._active_name or "none",
            )

        # Periodic inventory scan (every 10s -- memory read is microseconds)
        if now - ctx.inventory.last_scan_time > 10:
            try:
                items = self._runner._reader.read_inventory()
                ctx.inventory.update_items(items, now)
                brain_log.debug("[SNAPSHOT] Inventory scan: %d items", ctx.inventory.slots_used())
            except (OSError, RuntimeError, ValueError) as e:
                brain_log.warning("[SNAPSHOT] Inventory scan failed: %s", e)

        health_monitor.deep_check(state, ctx)
