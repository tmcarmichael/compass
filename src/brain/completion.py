"""Routine tick execution and completion handling.

Extracted from Brain to keep the decision engine focused on coordination.
Handles ticking the active routine, processing SUCCESS/FAILURE outcomes,
notifying cycle trackers, and hard-killing stuck routines.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from routines.base import RoutineStatus
from util.event_schemas import RoutineEndEvent
from util.structured_log import log_event

if TYPE_CHECKING:
    from brain.decision import Brain
    from perception.state import GameState

log = logging.getLogger(__name__)


def tick_active_routine(brain: Brain, state: GameState, now: float) -> None:
    """Tick the active routine and handle SUCCESS/FAILURE/hard-kill outcomes."""
    if brain._active is None:
        brain.routine_tick_ms = 0.0
        return

    rt0 = brain.perf_clock()
    brain._active._tick_deadline = rt0 + 0.200  # 200ms cooperative budget
    status = brain._active.tick(state)
    brain.routine_tick_ms = (brain.perf_clock() - rt0) * 1000

    # Capture name before completion/hard-kill clears it (for profiling)
    brain._ticked_routine_name = brain._active_name

    if status == RoutineStatus.RUNNING:
        if brain.routine_tick_ms > 5000:
            hard_kill_routine(brain, state, now)
        return

    # Routine completed (SUCCESS or FAILURE)
    handle_routine_completion(brain, state, status, now)


def handle_routine_completion(brain: Brain, state: GameState, status: RoutineStatus, now: float) -> None:
    """Log, record metrics, notify trackers, and apply cooldowns after completion."""
    reason = getattr(brain._active, "failure_reason", "") or ""
    fail_cat = getattr(brain._active, "failure_category", "unknown")
    entity_id = 0
    if brain._ctx and brain._ctx.combat:
        entity_id = brain._ctx.combat.pull_target_id or 0
    _re_kwargs: RoutineEndEvent = RoutineEndEvent(
        routine=brain._active_name,
        result=str(status),
        hp_pct=round(state.hp_pct, 3),
        mana_pct=round(state.mana_pct, 3),
        reason=reason,
        failure_category=fail_cat if status == RoutineStatus.FAILURE else "",
        entity_id=entity_id,
    )
    # Attach defeat cycle correlation ID to combat-cycle routines
    _CYCLE_ROUTINES = {"ACQUIRE", "PULL", "IN_COMBAT", "LOOT", "ENGAGE_ADD"}
    if brain._active_name in _CYCLE_ROUTINES and brain._ctx and brain._ctx.defeat_tracker.cycle_id > 0:
        _re_kwargs["cycle_id"] = brain._ctx.defeat_tracker.cycle_id
    log_event(
        log,
        "routine_end",
        f"[DECISION] Routine {brain._active_name} {status}"
        f" (HP={state.hp_pct * 100:.0f}% mana={state.mana_pct * 100:.0f}%)",
        **_re_kwargs,
    )
    # Record routine outcome in metrics engine
    if brain._ctx and brain._ctx.diag.metrics:
        brain._ctx.diag.metrics.record_routine_outcome(brain._active_name, status == RoutineStatus.SUCCESS)
    assert brain._active is not None
    brain._active.exit(state)

    notify_cycle_tracker(brain, state, status)

    # Clear engagement state after combat completes (not pull:
    # pull SUCCESS means fight is starting, engagement should persist)
    if brain._ctx and brain._active_name == "IN_COMBAT":
        brain._ctx.clear_engagement()

    # Apply failure cooldown, track failures, update circuit breaker
    if status == RoutineStatus.FAILURE:
        if brain._ctx:
            brain._ctx.metrics.routine_failures[brain._active_name] += 1
        breaker = brain._breakers.get(brain._active_name)
        if breaker:
            # Only trip breaker on execution failures, not precondition
            # (e.g., "no npcs nearby" is not a stuck state worth tripping for)
            from core.types import FailureCategory as _FC

            if fail_cat != _FC.PRECONDITION:
                breaker.record_failure()
        for r in brain._rules:
            if r.routine is brain._active and r.failure_cooldown > 0:
                brain._cooldowns[r.name] = now + r.failure_cooldown
                log.debug("[DECISION] Brain: %s on cooldown for %.0fs", r.name, r.failure_cooldown)
                break
    elif status == RoutineStatus.SUCCESS:
        breaker = brain._breakers.get(brain._active_name)
        if breaker:
            breaker.record_success()

    brain._active = None
    brain._active_name = ""
    brain._active_start_time = 0.0


def notify_cycle_tracker(brain: Brain, state: GameState, status: RoutineStatus) -> None:
    """Notify the cycle tracker of routine completion (narrative defeat cycle events)."""
    if not brain._ctx or not brain._ctx.diag.cycle_tracker:
        return
    _SUMMARY_ATTRS = {
        "IN_COMBAT": "last_fight_summary",
        "ACQUIRE": "last_acquire_summary",
        "PULL": "last_pull_summary",
    }
    attr = _SUMMARY_ATTRS.get(brain._active_name)
    _cycle_data = getattr(brain._active, attr, None) if attr else None
    brain._ctx.diag.cycle_tracker.on_routine_end(
        brain._active_name, str(status), state, brain._ctx, event_data=_cycle_data
    )


def hard_kill_routine(brain: Brain, state: GameState, now: float) -> None:
    """Force-exit a routine that returned RUNNING but took >5 s."""
    # Threshold must exceed combat casting pipeline (2-3s legit)
    # to avoid killing active fights mid-cast.
    log.error("[DECISION] HARD KILL: %s took %.0fms, forcing exit", brain._active_name, brain.routine_tick_ms)
    assert brain._active is not None
    brain._active.failure_reason = "hard_kill"
    from core.types import FailureCategory as _FC

    brain._active.failure_category = _FC.TIMEOUT
    log_event(
        log,
        "routine_hard_kill",
        f"[DECISION] HARD KILL: {brain._active_name} took {brain.routine_tick_ms:.0f}ms",
        routine=brain._active_name,
        tick_ms=round(brain.routine_tick_ms, 1),
        failure_category="TIMEOUT",
    )
    brain._active.exit(state)
    if brain._ctx:
        brain._ctx.metrics.routine_failures[brain._active_name] += 1
    breaker = brain._breakers.get(brain._active_name)
    if breaker:
        breaker.record_failure()
    for r in brain._rules:
        if r.routine is brain._active and r.failure_cooldown > 0:
            brain._cooldowns[r.name] = now + r.failure_cooldown
            break
    brain._active = None
    brain._active_name = ""
    brain._active_start_time = 0.0
