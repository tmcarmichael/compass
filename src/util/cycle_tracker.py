"""Defeat cycle tracker -- composite narrative events for acquire->pull->combat cycles.

Accumulates per-routine data as each routine in a defeat cycle completes,
then emits a single cycle_complete event when the cycle ends (combat SUCCESS
or cycle abandoned). This transforms 5-8 independent events into one
self-contained narrative record per npc engagement.

Thread ownership: brain thread only (called from decision.py routine completion).
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


class CycleTracker:
    """Accumulates routine data across one acquire->pull->combat cycle.

    Usage:
        tracker.on_routine_end(routine_name, status, state, ctx)

    When combat exits SUCCESS, emits a composite cycle_complete event
    with all accumulated data. When a cycle is abandoned (FAILURE at any
    stage, or a new ACQUIRE starts before combat), the partial cycle is
    discarded.
    """

    def __init__(self) -> None:
        self._current_cycle_id: int = 0
        self._active: bool = False
        # Accumulated per-cycle data
        self._acquire_tabs: int = 0
        self._acquire_time: float = 0.0
        self._target_name: str = ""
        self._entity_id: int = 0
        self._pull_strategy: str = ""
        self._pull_duration: float = 0.0
        self._pull_dot_retries: int = 0
        self._fight_duration: float = 0.0
        self._fight_casts: int = 0
        self._fight_mana_spent: int = 0
        self._fight_hp_delta: float = 0.0
        self._fight_adds: int = 0
        self._fight_strategy: str = ""
        self._fight_med_time: float = 0.0
        self._mana_start: int = 0
        self._mana_end: int = 0
        self._cycle_start: float = 0.0
        self._pos_x: int = 0
        self._pos_y: int = 0

    @property
    def cycle_id(self) -> int:
        """Read cycle_id from the canonical source (defeat_tracker).

        defeat_tracker.cycle_id is incremented by acquire.py on enter().
        This avoids dual counters drifting out of sync.
        """
        return self._current_cycle_id

    def on_routine_end(
        self, routine: str, status: str, state: GameState, ctx: AgentContext, event_data: dict | None = None
    ) -> None:
        """Called by decision.py when any routine completes.

        Args:
            routine: Routine name (ACQUIRE, PULL, IN_COMBAT, etc.)
            status: "success" or "failure"
            state: Current GameState
            ctx: AgentContext
            event_data: Extra event data from the routine's own log_event call.
                        Contains fields like tabs, duration, strategy, etc.
        """
        data = event_data or {}

        if routine == "ACQUIRE" and status == "success":
            # New cycle starts. cycle_id comes from defeat_tracker
            # (incremented by acquire.py enter(), before this notification)
            self._current_cycle_id = getattr(getattr(ctx, "defeat_tracker", None), "cycle_id", 0)
            self._active = True
            self._cycle_start = time.time()
            self._acquire_tabs = data.get("tabs", 0)
            self._acquire_time = 0.0
            self._target_name = data.get("target", "")
            self._entity_id = data.get("entity_id", 0)
            self._pull_strategy = ""
            self._pull_duration = 0.0
            self._pull_dot_retries = 0
            self._fight_duration = 0.0
            self._fight_casts = 0
            self._fight_mana_spent = 0
            self._fight_hp_delta = 0.0
            self._fight_adds = 0
            self._fight_strategy = ""
            self._fight_med_time = 0.0
            self._mana_start = getattr(state, "mana_current", 0)
            self._mana_end = 0
            self._pos_x = round(getattr(state, "x", 0.0))
            self._pos_y = round(getattr(state, "y", 0.0))

        elif routine == "ACQUIRE" and status == "failure":
            # Failed acquire -- not a cycle
            self._active = False

        elif routine == "PULL" and self._active:
            if status == "failure":
                log.debug("CycleTracker: PULL failed -- abandoning cycle %d", self.cycle_id)
                self._active = False
                return
            self._pull_strategy = data.get("strategy", "")
            self._pull_duration = data.get("duration", 0.0)
            self._pull_dot_retries = data.get("dot_retries", 0)
            if not self._target_name:
                self._target_name = data.get("target", "")

        elif routine == "IN_COMBAT" and self._active:
            self._mana_end = getattr(state, "mana_current", 0)
            self._pos_x = round(getattr(state, "x", 0.0))
            self._pos_y = round(getattr(state, "y", 0.0))
            if status == "success":
                self._emit_cycle(state, ctx, data)
            # Cycle ends regardless of combat outcome
            self._active = False

    def _emit_cycle(self, state: GameState, ctx: AgentContext, fight_data: dict) -> None:
        """Emit a composite cycle_complete event."""
        self._fight_duration = fight_data.get("duration", 0.0)
        self._fight_casts = fight_data.get("casts", 0)
        self._fight_mana_spent = fight_data.get("mana_spent", 0)
        self._fight_hp_delta = fight_data.get("hp_delta", 0.0)
        self._fight_adds = fight_data.get("adds", 0)
        self._fight_strategy = fight_data.get("strategy", "")
        self._fight_med_time = fight_data.get("med_time", 0.0)

        cycle_total = time.time() - self._cycle_start if self._cycle_start > 0 else 0.0
        defeats_so_far = getattr(getattr(ctx, "defeat_tracker", None), "defeats", 0)

        # Rolling DPH from context
        elapsed = time.time() - getattr(getattr(ctx, "metrics", None), "session_start", time.time())
        hours = max(elapsed / 3600, 0.01)
        kph_rolling = round(defeats_so_far / hours, 1)

        log_event(
            log,
            "cycle_complete",
            f"Cycle #{self.cycle_id}: {self._target_name} | "
            f"{round(self._fight_duration)}s fight, {self._fight_casts} casts, "
            f"{self._fight_mana_spent} mana | {cycle_total:.0f}s total",
            cycle_id=self.cycle_id,
            npc=self._target_name,
            entity_id=self._entity_id,
            acquire_tabs=self._acquire_tabs,
            pull_strategy=self._pull_strategy,
            pull_duration=round(self._pull_duration, 1),
            pull_dot_retries=self._pull_dot_retries,
            fight_duration=round(self._fight_duration, 1),
            fight_casts=self._fight_casts,
            fight_mana_spent=self._fight_mana_spent,
            fight_hp_delta=round(self._fight_hp_delta, 3),
            fight_adds=self._fight_adds,
            fight_strategy=self._fight_strategy,
            fight_med_time=round(self._fight_med_time, 1),
            mana_start=self._mana_start,
            mana_end=self._mana_end,
            cycle_total_s=round(cycle_total, 1),
            defeats_so_far=defeats_so_far,
            kph_rolling=kph_rolling,
            pos_x=self._pos_x,
            pos_y=self._pos_y,
        )
