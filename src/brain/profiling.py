"""Tick profiling and budget tracking.

Extracted from Brain to keep the decision engine focused on coordination.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from util.structured_log import log_event

if TYPE_CHECKING:
    from brain.decision import Brain

log = logging.getLogger(__name__)


def tick_profiling(brain: Brain, tick_start: float) -> None:
    """Record tick timing, warn on slow ticks, publish breaker states."""
    brain.tick_total_ms = (brain.perf_clock() - tick_start) * 1000
    if brain.tick_total_ms > 1000:
        slowest_rule = max(brain.rule_times, key=lambda k: brain.rule_times[k], default="?")
        slowest_ms = brain.rule_times.get(slowest_rule, 0)
        # Report the actual bottleneck: routine if it dominated, else slowest rule
        if brain.routine_tick_ms > sum(brain.rule_times.values()):
            bottleneck = brain._ticked_routine_name or "?"
            bottleneck_ms = brain.routine_tick_ms
        else:
            bottleneck = slowest_rule
            bottleneck_ms = slowest_ms
        log.warning(
            "[DECISION] SLOW TICK: %.0fms (rules=%.0fms routine=%.0fms slowest=%s:%.0fms)",
            brain.tick_total_ms,
            sum(brain.rule_times.values()),
            brain.routine_tick_ms,
            bottleneck,
            bottleneck_ms,
        )

    # Tick budget tracking (200ms soft budget)
    if brain.tick_total_ms > 200 and brain._ctx:
        brain._ctx.diag.tick_overbudget_count += 1
        if brain.tick_total_ms > brain._ctx.diag.tick_overbudget_max_ms:
            brain._ctx.diag.tick_overbudget_max_ms = brain.tick_total_ms
            brain._ctx.diag.tick_overbudget_last_routine = brain._ticked_routine_name or ""
        log_event(
            log,
            "tick_overbudget",
            f"[DECISION] Tick {brain.tick_total_ms:.0f}ms (budget=200ms, "
            f"routine={brain._ticked_routine_name or 'NONE'}:{brain.routine_tick_ms:.0f}ms)",
            tick_ms=round(brain.tick_total_ms, 1),
            routine_ms=round(brain.routine_tick_ms, 1),
            routine=brain._ticked_routine_name or "NONE",
        )

    # Publish breaker states for SSE snapshot (atomic ref swap)
    if brain._ctx and brain._breakers:
        brain._ctx.diag.breaker_states = {
            n: b.state for n, b in brain._breakers.items() if b.state != "CLOSED"
        }
