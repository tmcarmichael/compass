"""Routine transition handling, stale engagement cleanup, and decision receipts.

Extracted from Brain to keep the decision engine focused on coordination.
Handles activation/deactivation of routines, lock-in enforcement, zombie
engagement detection, and structured decision receipt logging.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from brain.phase_modifiers import get_phase_modifier
from routines.base import RoutineBase

if TYPE_CHECKING:
    from brain.decision import Brain
    from perception.state import GameState

log = logging.getLogger(__name__)


def handle_transition(
    brain: Brain,
    state: GameState,
    selected: RoutineBase | None,
    selected_name: str,
    selected_emergency: bool,
    now: float,
) -> None:
    """Clear stale engagement state and handle routine activation/deactivation."""
    # Safety: clear stale engaged state, prevents permanent stuck.
    # Grace period: don't clear in the first 10s after engagement starts
    # (pull just succeeded, target may not be tab-targeted yet).
    if brain._ctx and brain._ctx.combat.engaged:
        maybe_clear_stale_engagement(brain, state, selected, now)

    if selected is brain._active:
        return

    # Check lock timeout before honoring the lock
    if brain._active is not None and brain._active.locked and brain._active_start_time > 0:
        maybe_force_lock_exit(brain, state, now)

    # Locked routines can only be interrupted by emergency rules
    if brain._active is not None and brain._active.locked and not selected_emergency:
        lock_key = f"{brain._active_name}:{selected_name}"
        if lock_key != brain._last_lock_blocked:
            log.info(
                "[DECISION] Brain: %s is LOCKED  -  %s cannot interrupt",
                brain._active_name,
                selected_name or "NONE",
            )
            brain._last_lock_blocked = lock_key
        return

    brain._last_lock_blocked = ""
    if brain._active is not None:
        if brain._active.locked:
            log.info("[DECISION] Brain: EMERGENCY %s overriding locked %s", selected_name, brain._active_name)
            # Clear stale GOAP suggestion on emergency interrupt
            if brain._ctx and hasattr(brain._ctx, "diag") and brain._ctx.diag:
                brain._ctx.diag.goap_suggestion = ""
        log.info("[DECISION] Brain: deactivating %s", brain._active.name)
        # Track time spent in outgoing routine.
        # Lock-protected: another thread may iterate routine_time
        # via sum(values()) in scorecard. Dict mutation during
        # iteration raises RuntimeError.
        if brain._ctx and brain._active_name and brain._active_start_time > 0:
            elapsed = now - brain._active_start_time
            with brain._ctx.lock:
                brain._ctx.metrics.routine_time[brain._active_name] = (
                    brain._ctx.metrics.routine_time.get(brain._active_name, 0) + elapsed
                )
        brain._active.exit(state)

    brain._active = selected
    brain._active_name = selected_name
    brain._active_start_time = now
    if brain._ctx:
        brain._ctx.metrics.routine_start_time = now
    if brain._active is not None:
        target_str = ""
        if state.target:
            target_str = f" Target='{state.target.name}' id={state.target.spawn_id}"
        log.info(
            "[DECISION] Brain: -> %s | HP=%.0f%% Mana=%.0f%% Pos=(%.0f,%.0f) Pet=%s%s",
            selected_name,
            state.hp_pct * 100,
            state.mana_pct * 100,
            state.x,
            state.y,
            "yes" if (brain._ctx and brain._ctx.pet.alive) else "no",
            target_str,
        )
        # Decision receipt: trace the full path through all layers
        log_decision_receipt(brain, selected_name, selected_emergency, state)
        if brain._ctx:
            brain._ctx.metrics.routine_counts[selected_name] += 1
        brain._active.failure_reason = ""
        brain._active.enter(state)


def log_decision_receipt(brain: Brain, rule_name: str, emergency: bool, state: GameState) -> None:
    """Log a unified decision receipt tracing the full path through all layers.

    One log line that answers: which layer drove this decision, what was
    the GOAP plan state, what was the utility score, what phase modifier
    was applied, and what was the alternative?
    """
    if not brain._ctx or not brain._ctx.diag:
        return

    parts: list[str] = [f"[DECISION] RECEIPT: {rule_name}"]

    # Source: GOAP plan, utility scoring, or priority fallback?
    goap_hint = getattr(brain._ctx.diag, "goap_suggestion", "")
    if emergency:
        parts.append("source=EMERGENCY")
    elif goap_hint and goap_hint == rule_name:
        parts.append("source=GOAP_PLAN")
    elif brain.utility_phase >= 2:
        parts.append(f"source=UTILITY_PHASE_{brain.utility_phase}")
    else:
        parts.append("source=PRIORITY_RULES")

    # GOAP plan context
    if goap_hint:
        parts.append(f"goap_step='{goap_hint}'")
    else:
        parts.append("goap=none")

    # Phase modifier
    phase = "grinding"
    pd = getattr(brain._ctx.diag, "phase_detector", None)
    if pd is not None:
        phase = pd.current_phase
    modifier = get_phase_modifier(phase, rule_name)
    if modifier != 1.0:
        parts.append(f"phase={phase}({modifier:.2f}x)")
    else:
        parts.append(f"phase={phase}")

    # Score (from last evaluation if available)
    score_str = brain._ctx.diag.last_rule_evaluation.get(rule_name, "")
    if score_str and score_str != "0":
        parts.append(f"score={score_str}")

    # Top alternative (highest-scoring rule that was NOT selected)
    evals = brain._ctx.diag.last_rule_evaluation
    alt_name = ""
    alt_score = ""
    for rname, rval in evals.items():
        if rname == rule_name or rval in ("0", ""):
            continue
        if not alt_name or rval > alt_score:
            alt_name = rname
            alt_score = rval
    if alt_name:
        parts.append(f"alt={alt_name}({alt_score})")

    log.info(" | ".join(parts))


def maybe_clear_stale_engagement(
    brain: Brain, state: GameState, selected: RoutineBase | None, now: float
) -> None:
    """Detect and clear zombie engaged state; record inferred defeats."""
    if not brain._ctx:
        return
    _stale = False
    _reason = ""
    engage_age = now - brain._ctx.player.engagement_start if brain._ctx.player.engagement_start > 0 else 999
    if selected is None and engage_age > 10.0:
        if not state.target or not state.target.is_npc or state.target.hp_current <= 0:
            _stale = True
            _reason = f"no rule matched + target gone/dead ({engage_age:.0f}s)"
    # Zombie engagement: engaged for 60s+ but combat/pull not active
    if (
        not _stale
        and brain._ctx.player.engagement_start > 0
        and now - brain._ctx.player.engagement_start > 60.0
        and brain._active_name not in ("IN_COMBAT", "PULL", "FLEE")
    ):
        _stale = True
        _reason = "zombie ({:.0f}s, active={})".format(
            now - brain._ctx.player.engagement_start, brain._active_name or "NONE"
        )
    if not _stale:
        return

    # Only record defeat if we can confirm the target actually died (HP=0),
    # not just that the target pointer is null (could be evade/despawn).
    defeat_name = ""
    if (
        state.target
        and state.target.hp_current <= 0
        and (
            state.target.spawn_id == brain._ctx.defeat_tracker.last_fight_id
            or state.target.spawn_id == brain._ctx.combat.pull_target_id
        )
    ):
        defeat_name = state.target.name
        brain._ctx.record_kill(
            state.target.spawn_id,
            name=state.target.name,
            pos=state.target.pos,
        )
    elif (
        not state.target
        and brain._ctx.defeat_tracker.last_fight_name
        and brain._ctx.defeat_tracker.last_fight_id
    ):
        last_hp = brain._ctx.combat.last_mob_hp_pct
        if last_hp < 0.20:
            # NPC was nearly dead when it vanished: infer defeat
            log.info(
                "[STATE] Brain: target gone at %.0f%% HP  -  recording defeat for '%s' id=%d (inferred)",
                last_hp * 100,
                brain._ctx.defeat_tracker.last_fight_name,
                brain._ctx.defeat_tracker.last_fight_id,
            )
            from core.types import Point

            brain._ctx.record_kill(
                brain._ctx.defeat_tracker.last_fight_id,
                name=brain._ctx.defeat_tracker.last_fight_name,
                pos=Point(
                    brain._ctx.defeat_tracker.last_fight_x, brain._ctx.defeat_tracker.last_fight_y, 0.0
                ),
            )
            defeat_name = brain._ctx.defeat_tracker.last_fight_name
        else:
            log.warning(
                "[STATE] Brain: target gone at %.0f%% HP (possible evade)  -  "
                "NOT recording defeat for '%s' id=%d",
                last_hp * 100,
                brain._ctx.defeat_tracker.last_fight_name,
                brain._ctx.defeat_tracker.last_fight_id,
            )
    if defeat_name:
        log.warning("[STATE] Brain: cleared stale engaged + recorded defeat '%s' (%s)", defeat_name, _reason)
    else:
        log.warning("[STATE] Brain: cleared stale engaged (%s)", _reason)
    brain._ctx.clear_engagement()


def maybe_force_lock_exit(brain: Brain, state: GameState, now: float) -> None:
    """Force-exit a locked routine that has exceeded its max_lock_seconds budget."""
    _max_lock = 0.0
    for r in brain._rules:
        if r.routine is brain._active:
            _max_lock = r.max_lock_seconds
            break
    if _max_lock <= 0:
        return
    lock_dur = now - brain._active_start_time
    if lock_dur <= _max_lock:
        return

    _timed_out_name = brain._active_name
    log.warning(
        "[DECISION] Brain: %s LOCK TIMEOUT after %.0fs (limit %.0fs), forcing exit",
        _timed_out_name,
        lock_dur,
        _max_lock,
    )
    assert brain._active is not None
    brain._active.exit(state)
    # Apply failure cooldown
    for r in brain._rules:
        if r.routine is brain._active and r.failure_cooldown > 0:
            brain._cooldowns[r.name] = now + r.failure_cooldown
            break
    brain._active = None
    brain._active_name = ""
    brain._active_start_time = 0.0
    # Clear stale combat state to prevent re-fire loop
    # and check for active threats so FLEE fires next tick
    if brain._ctx and _timed_out_name in ("PULL", "IN_COMBAT", "ENGAGE_ADD"):
        brain._ctx.clear_engagement()
        for sp in state.spawns:
            if sp.is_npc and sp.hp_current > 0 and sp.target_name == state.name:
                brain._ctx.threat.imminent_threat = True
                log.warning(
                    "[STATE] Brain: npc '%s' still targeting player after %s timeout, flagging threat",
                    sp.name,
                    _timed_out_name,
                )
                break
