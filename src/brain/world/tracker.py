"""State change tracker: edge-triggered logging for key state transitions.

Only logs when a value CHANGES, not on every tick. Eliminates spam while
ensuring every meaningful transition is captured in the log.

Usage:
    tracker = StateChangeTracker()
    # In brain loop, after reading state:
    tracker.update(state, ctx)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from perception.state import GameState
from util.log_tiers import VERBOSE

if TYPE_CHECKING:
    from brain.context import AgentContext

log = logging.getLogger("compass.brain_loop")


class StateChangeTracker:
    """Tracks key state values and logs only on transitions."""

    def __init__(self) -> None:
        # Previous values (None = not yet seen)
        self._pet_alive: bool | None = None
        self._engaged: bool | None = None
        self._target_id: int = 0
        self._target_name: str = ""
        self._player_count: int = 0
        self._imminent_threat: bool = False
        self._pull_target_id: int | None = None
        self._is_sitting: bool | None = None
        self._plan_active: str | None = None
        self._in_combat_flag: bool | None = None
        self._pet_has_add: bool = False

    def update(self, state: GameState, ctx: AgentContext | None) -> None:
        """Check all tracked values and log transitions."""
        self._track_pet(ctx)
        self._track_engagement(ctx)
        self._track_target(state)
        self._track_players(state, ctx)
        self._track_threat(ctx)
        self._track_pull_target(ctx)
        self._track_sitting(state, ctx)
        self._track_plan(ctx)
        self._track_combat_flag(state)
        self._track_pet_add(ctx)

    def _track_pet(self, ctx: AgentContext | None) -> None:
        pet_alive = ctx.pet.alive if ctx else False
        if self._pet_alive is not None and pet_alive != self._pet_alive:
            if pet_alive:
                assert ctx is not None
                log.info(
                    "[STATE] STATE: pet ALIVE (id=%s name='%s')", ctx.pet.spawn_id or "?", ctx.pet.name or "?"
                )
            else:
                log.warning("[STATE] STATE: pet DIED")
        self._pet_alive = pet_alive

    def _track_engagement(self, ctx: AgentContext | None) -> None:
        engaged = ctx.combat.engaged if ctx else False
        if self._engaged is not None and engaged != self._engaged:
            if engaged:
                assert ctx is not None
                name = ctx.combat.pull_target_name or "?"
                log.info("[STATE] STATE: ENGAGED -> '%s'", name)
            else:
                log.info("[STATE] STATE: disengaged")
        self._engaged = engaged

    def _track_target(self, state: GameState) -> None:
        tid = state.target.spawn_id if state.target else 0
        tname = state.target.name if state.target else ""
        if tid != self._target_id and tid > 0:
            dist = state.pos.dist_to(state.target.pos) if state.target else 0
            log.log(
                VERBOSE,
                "[STATE] STATE: target -> '%s' id=%d dist=%.0f lv=%d HP=%d/%d",
                tname,
                tid,
                dist,
                state.target.level if state.target else 0,
                state.target.hp_current if state.target else 0,
                state.target.hp_max if state.target else 0,
            )
        elif tid == 0 and self._target_id > 0:
            log.log(VERBOSE, "[STATE] STATE: target cleared (was '%s')", self._target_name)
        self._target_id = tid
        self._target_name = tname

    def _track_players(self, state: GameState, ctx: AgentContext | None) -> None:
        if ctx:
            pc = ctx.nearby_player_count(state, radius=250)
            if pc > 0 and self._player_count == 0:
                pd = ctx.nearest_player_dist(state)
                log.info("[STATE] STATE: player detected within 250u (dist=%.0f, count=%d)", pd, pc)
            elif pc == 0 and self._player_count > 0:
                log.info("[STATE] STATE: no players nearby (cleared)")
            self._player_count = pc

    def _track_threat(self, ctx: AgentContext | None) -> None:
        threat = ctx.threat.imminent_threat if ctx else False
        if threat and not self._imminent_threat:
            con = ctx.threat.imminent_threat_con if ctx else "?"
            log.warning("[STATE] STATE: imminent threat (%s)", con)
        elif not threat and self._imminent_threat:
            log.info("[STATE] STATE: threat cleared")
        self._imminent_threat = threat

    def _track_pull_target(self, ctx: AgentContext | None) -> None:
        ptid = ctx.combat.pull_target_id if ctx else None
        if ptid != self._pull_target_id:
            if ptid is not None and self._pull_target_id is None:
                assert ctx is not None
                pname = ctx.combat.pull_target_name or "?"
                log.info("[STATE] STATE: pull target set -> '%s' id=%d", pname, ptid)
            elif ptid is None and self._pull_target_id is not None:
                log.info("[STATE] STATE: pull target cleared")
        self._pull_target_id = ptid

    def _track_sitting(self, state: GameState, ctx: AgentContext | None) -> None:
        sitting = state.is_sitting
        if self._is_sitting is not None and sitting != self._is_sitting:
            suppress = (ctx and ctx.combat.engaged) or state.casting_mode == 6
            if not suppress:
                log.debug("[STATE] STATE: %s", "sitting" if sitting else "standing")
        self._is_sitting = sitting

    def _track_plan(self, ctx: AgentContext | None) -> None:
        plan = ctx.plan.active if ctx else None
        if plan != self._plan_active:
            if plan is not None:
                log.info(
                    "[STATE] STATE: plan activated -> '%s' travel=%s", plan, ctx.plan.travel if ctx else ""
                )
            elif self._plan_active is not None:
                log.info("[STATE] STATE: plan completed ('%s')", self._plan_active)
        self._plan_active = plan

    def _track_combat_flag(self, state: GameState) -> None:
        icf = state.in_combat
        if self._in_combat_flag is not None and icf != self._in_combat_flag:
            if icf:
                log.log(VERBOSE, "[STATE] STATE: in_combat flag SET (game detects combat)")
            else:
                log.log(VERBOSE, "[STATE] STATE: in_combat flag CLEARED")
        self._in_combat_flag = icf

    def _track_pet_add(self, ctx: AgentContext | None) -> None:
        pet_add = ctx.pet.has_add if ctx else False
        if pet_add and not self._pet_has_add:
            log.warning("[STATE] STATE: pet has ADD (npc fighting pet besides target)")
        elif not pet_add and self._pet_has_add:
            log.info("[STATE] STATE: pet add cleared")
        self._pet_has_add = pet_add
