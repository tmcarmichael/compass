"""Phase detector -- identifies operational mode transitions in long sessions.

Tracks rolling defeat rate and routine distribution to detect when the agent
shifts between grinding, resting, incident (flee/death), and idle phases.
Emits phase_change events at transitions, giving sessions a chapter structure.

Thread ownership: brain thread only (called from periodic snapshot handler).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from util.structured_log import log_event

if TYPE_CHECKING:
    from brain.context import AgentContext
    from perception.state import GameState

log = logging.getLogger(__name__)


class PhaseDetector:
    """Detects operational phase transitions and emits phase_change events.

    Phases:
        startup   -- first 60s of session, before steady state
        grinding  -- actively defeating (DPH > 0, not resting/fleeing)
        resting   -- extended rest (>30s in REST routine)
        incident  -- flee or death in progress
        idle      -- no defeats for >120s, not in combat or rest

    Called every 30s from periodic_snapshot. Uses routine name and
    defeat timing to classify the current phase.
    """

    def __init__(self) -> None:
        self._current_phase: str = "startup"
        self._phase_start: float = time.time()
        self._phase_kills: int = 0
        self._session_start: float = time.time()
        self._last_kills: int = 0
        # In-memory history for session narrative (consumed at shutdown)
        # Each entry: (phase, start_epoch, duration_s, defeats_in_phase, dph)
        self.history: list[tuple[str, float, float, int, float]] = []

    @property
    def current_phase(self) -> str:
        return self._current_phase

    def check(self, state: GameState, ctx: AgentContext, now: float) -> None:
        """Evaluate current phase and emit phase_change if transitioned.

        Args:
            state: Current GameState.
            ctx: AgentContext.
            now: Current time.time().
        """
        new_phase = self._classify(state, ctx, now)

        if new_phase != self._current_phase:
            old_phase = self._current_phase
            phase_duration = now - self._phase_start

            defeats_tracker = getattr(ctx, "defeat_tracker", None)
            total_kills = getattr(defeats_tracker, "defeats", 0) if defeats_tracker else 0
            defeats_in_phase = total_kills - self._phase_kills

            # Compute DPH for the completed phase
            phase_hours = max(phase_duration / 3600, 0.01)
            kph_in_phase = round(defeats_in_phase / phase_hours, 1)

            log_event(
                log,
                "phase_change",
                f"Phase: {old_phase} -> {new_phase} "
                f"({phase_duration:.0f}s, {defeats_in_phase} defeats, "
                f"{kph_in_phase} dph)",
                old_phase=old_phase,
                new_phase=new_phase,
                phase_duration_s=round(phase_duration, 1),
                defeats_in_phase=defeats_in_phase,
                kph_in_phase=kph_in_phase,
                elapsed=round(now - self._session_start, 1),
                total_kills=total_kills,
            )

            # Record completed phase for session narrative
            self.history.append(
                (
                    old_phase,
                    self._phase_start,
                    round(phase_duration, 1),
                    defeats_in_phase,
                    kph_in_phase,
                )
            )

            self._current_phase = new_phase
            self._phase_start = now
            self._phase_kills = total_kills

    def finalize(self, ctx: AgentContext) -> None:
        """Close the current phase and append it to history.

        Called at session shutdown so the final (still-active) phase
        is included in the narrative.
        """
        now = time.time()
        duration = now - self._phase_start
        defeats_tracker = getattr(ctx, "defeat_tracker", None)
        total_kills = getattr(defeats_tracker, "defeats", 0) if defeats_tracker else 0
        defeats_in_phase = total_kills - self._phase_kills
        phase_hours = max(duration / 3600, 0.01)
        kph_in_phase = round(defeats_in_phase / phase_hours, 1)
        self.history.append(
            (
                self._current_phase,
                self._phase_start,
                round(duration, 1),
                defeats_in_phase,
                kph_in_phase,
            )
        )

    def _classify(self, state: GameState, ctx: AgentContext, now: float) -> str:
        """Classify current operational phase."""
        elapsed = now - self._session_start

        # First 60 seconds is always startup
        if elapsed < 60 and self._current_phase == "startup":
            return "startup"

        # Check for incident (flee/death)
        player = getattr(ctx, "player", None)
        if player and getattr(player, "dead", False):
            return "incident"

        # Check active routine name via diag
        diag = getattr(ctx, "diag", None)
        last_eval = getattr(diag, "last_rule_evaluation", {}) if diag else {}

        # Infer active routine from brain state
        # FLEE active -> incident
        flee_val = last_eval.get("FLEE", "")
        if flee_val and flee_val != "no" and not flee_val.startswith("cooldown"):
            return "incident"

        # REST active for extended period -> resting
        rest_val = last_eval.get("REST", "")
        if rest_val and rest_val != "no" and not rest_val.startswith("cooldown"):
            # Only if we've been resting for >30s
            if self._current_phase == "resting":
                return "resting"
            # Transitioning to rest -- check if it's brief (between defeats)
            # or extended. Use mana as a signal.
            mana_pct = getattr(state, "mana_pct", 1.0)
            if mana_pct < 0.5:
                return "resting"  # low mana = genuine rest phase

        # Check for idle (no defeats for >120s, not engaged)
        combat = getattr(ctx, "combat", None)
        engaged = getattr(combat, "engaged", False) if combat else False
        defeats_tracker = getattr(ctx, "defeat_tracker", None)
        if defeats_tracker and not engaged:
            defeat_age = defeats_tracker.last_kill_age()
            total_kills = getattr(defeats_tracker, "defeats", 0)
            if defeat_age > 120 and total_kills > 0:
                return "idle"

        # Default: grinding (actively defeating or between defeats)
        return "grinding"
