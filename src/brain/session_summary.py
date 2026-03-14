"""Session summary formatting: end-of-session report generation.

Extracted from brain.context to separate presentation logic from
mutable agent state. All functions are pure formatters that take
data snapshots, not live mutable state.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.state.inventory import InventoryState
    from brain.state.kill_tracker import DefeatTracker
    from brain.state.metrics import SessionMetrics

log = logging.getLogger(__name__)


def format_routine_stats(mc: dict) -> list[str]:
    """Format routine counts, failures, and acquire modes from metrics snapshot."""
    lines: list[str] = []
    if mc["routine_counts"]:
        parts = [f"{k}={v}" for k, v in sorted(mc["routine_counts"].items())]
        lines.append(f"Routine activations: {', '.join(parts)}")
    if mc["routine_failures"]:
        parts = [f"{k}={v}" for k, v in sorted(mc["routine_failures"].items())]
        lines.append(f"Routine failures: {', '.join(parts)}")
    if mc["acquire_modes"]:
        parts = [f"{k}={v}" for k, v in sorted(mc["acquire_modes"].items())]
        lines.append(f"Acquire modes: {', '.join(parts)}")
    return lines


def format_pull_stats(mc: dict, metrics: SessionMetrics, defeat_tracker: DefeatTracker) -> list[str]:
    """Format pull distances, engage times, and mana efficiency."""
    lines: list[str] = []
    if metrics.total_casts > 0 and defeat_tracker.defeats > 0:
        lines.append(f"Mana efficiency: {metrics.total_casts / defeat_tracker.defeats:.1f} casts/npc")
    if mc["pull_distances"]:
        avg_d = sum(mc["pull_distances"]) / len(mc["pull_distances"])
        lines.append(
            f"Pull distance: avg={avg_d:.0f} min={min(mc['pull_distances']):.0f} "
            f"max={max(mc['pull_distances']):.0f}"
        )
    if mc["pull_engage_times"]:
        avg_e = sum(mc["pull_engage_times"]) / len(mc["pull_engage_times"])
        lines.append(
            f"Pet engage time: avg={avg_e:.1f}s min={min(mc['pull_engage_times']):.1f}s "
            f"max={max(mc['pull_engage_times']):.1f}s"
        )
    if metrics.pull_dc_fizzles > 0:
        lines.append(f"DC fizzles: {metrics.pull_dc_fizzles}")
    if metrics.pull_pet_only_count > 0:
        lines.append(f"Pet-only pulls: {metrics.pull_pet_only_count}/{len(mc['pull_distances'])}")
    return lines


def format_cycle_stats(
    mc: dict,
    metrics: SessionMetrics,
    defeat_tracker: DefeatTracker,
    inventory: InventoryState,
) -> list[str]:
    """Format cycle times, pull distances, engage times, and combat stats."""
    lines: list[str] = []
    if mc["total_cycle_times"]:
        avg = sum(mc["total_cycle_times"]) / len(mc["total_cycle_times"])
        fastest = min(mc["total_cycle_times"])
        slowest = max(mc["total_cycle_times"])
        lines.append(
            f"Grind cycle: avg={avg:.1f}s fastest={fastest:.1f}s "
            f"slowest={slowest:.1f}s ({len(mc['total_cycle_times'])} cycles)"
        )
    lines.extend(format_pull_stats(mc, metrics, defeat_tracker))
    if mc["acquire_tab_totals"]:
        avg_t = sum(mc["acquire_tab_totals"]) / len(mc["acquire_tab_totals"])
        lines.append(
            f"Acquire tabs/success: avg={avg_t:.1f} "
            f"invalid_tabs={metrics.acquire_invalid_tabs} "
            f"approach_forced={metrics.acquire_approach_forced}"
        )
    if metrics.wander_count > 0:
        avg_wd = metrics.wander_total_distance / metrics.wander_count
        lines.append(f"Wander: {metrics.wander_count}x avg_dist={avg_wd:.0f}")
    if metrics.total_combat_time > 0:
        cast_pct = metrics.total_cast_time / metrics.total_combat_time * 100
        idle_pct = 100 - cast_pct
        lines.append(
            f"Combat time: {metrics.total_combat_time:.0f}s (casting {cast_pct:.0f}%, idle {idle_pct:.0f}%)"
        )
    if defeat_tracker.xp_gains > 0:
        lines.append(
            f"XP npcs: {defeat_tracker.xp_gains}/{defeat_tracker.defeats} "
            f"({defeat_tracker.xp_gains / defeat_tracker.defeats * 100:.0f}% gave XP)"
        )
    if inventory.loot_count > 0:
        lines.append(f"Corpses looted: {inventory.loot_count}")

    # Time spent in each routine
    if mc["routine_time"]:
        sorted_rt = sorted(mc["routine_time"].items(), key=lambda x: -x[1])
        total_rt = sum(v for _, v in sorted_rt)
        parts = []
        for name, secs in sorted_rt:
            pct = secs / total_rt * 100 if total_rt > 0 else 0
            parts.append(f"{name}={secs:.0f}s({pct:.0f}%)")
        lines.append(f"Time by routine: {', '.join(parts)}")

    # Defeat cycle timing (time between consecutive defeats)
    if defeat_tracker.defeat_cycle_times:
        avg_cycle = sum(defeat_tracker.defeat_cycle_times) / len(defeat_tracker.defeat_cycle_times)
        lines.append(
            f"Time between npcs: avg={avg_cycle:.0f}s "
            f"min={min(defeat_tracker.defeat_cycle_times):.0f}s "
            f"max={max(defeat_tracker.defeat_cycle_times):.0f}s"
        )

    # Stuck events
    from nav.movement import get_stuck_event_count

    stuck = get_stuck_event_count()
    if stuck > 0:
        lines.append(f"Stuck events: {stuck}")

    return lines


def format_xp_stats(ctx: AgentContext) -> list[str]:
    """Format XP rate, time to level, and perception stats."""
    lines: list[str] = []
    xp_hr = ctx.metrics.xp_per_hour()
    if xp_hr > 0:
        ttl = ctx.metrics.time_to_level()
        ttl_str = f" TTL={ttl:.0f}min" if ttl else ""
        lines.append(f"XP rate: {xp_hr:.1f}%/hr{ttl_str} (gained {ctx.metrics.xp_gained_pct:.1f}% total)")

    # Perception health
    if ctx.reader and hasattr(ctx.reader, "_read_stats"):
        parts = []
        for source, stats in sorted(ctx.reader._read_stats.items()):
            if stats.total > 0:
                parts.append(f"{source}={stats.success}/{stats.total}({stats.success_rate * 100:.0f}%)")
        if parts:
            lines.append(f"Perception: {', '.join(parts)}")

    return lines


def format_session_summary(ctx: AgentContext) -> str:
    """Generate the full end-of-session summary string."""
    elapsed = time.time() - ctx.metrics.session_start
    minutes = elapsed / 60
    dph = ctx.defeat_tracker.defeats / max(elapsed / 3600, 0.01)

    # Snapshot all mutable collections under lock (free-threaded safety)
    with ctx.lock:
        mc = ctx.metrics.snapshot_collections()

    from core import __version__

    lines = [
        "=" * 60,
        f"SESSION SUMMARY (v{__version__})",
        "=" * 60,
        f"Duration: {minutes:.1f} min | Npcs: {ctx.defeat_tracker.defeats} | DPH: {dph:.1f}",
        f"Rests: {ctx.metrics.rest_count} | Flees: {ctx.metrics.flee_count} | Casts: {ctx.metrics.total_casts}",
    ]

    lines.extend(format_routine_stats(mc))
    lines.extend(format_cycle_stats(mc, ctx.metrics, ctx.defeat_tracker, ctx.inventory))
    lines.extend(format_xp_stats(ctx))

    lines.append("=" * 60)

    # Fight outcome learning summary
    if ctx.fight_history:
        try:
            lines.append(ctx.fight_history.summary())
        except (KeyError, ValueError, ZeroDivisionError) as e:
            log.debug("Fight history summary failed: %s", e)

    # Performance scorecard
    try:
        from brain.learning.scorecard import compute_scorecard, format_scorecard

        scores = compute_scorecard(ctx)
        lines.append(format_scorecard(scores))
    except (ImportError, KeyError, ValueError, ZeroDivisionError) as e:
        log.debug("Scorecard computation failed: %s", e)

    return "\n".join(lines)
