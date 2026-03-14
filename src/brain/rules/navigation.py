"""Navigation rules: TRAVEL, WANDER.

EVADE was moved to survival.py for higher priority (after REST, before combat).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from brain.rules.skip_log import SkipLog
from core.features import flags
from core.types import Point, TravelMode
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus
from routines.travel import (
    MultiLegTravelRoutine,
    PlanTravelRoutine,
    TravelRoutine,
    plan_travel_legs,
)
from routines.wander import WanderRoutine
from util.log_tiers import VERBOSE

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.decision import Brain
    from core.types import ReadStateFn


@dataclass
class _NavigationRuleState:
    """Mutable state shared across navigation rule closures."""

    waypoint_travel: RoutineBase | None = None


log = logging.getLogger(__name__)
_skip = SkipLog(log)


# -- Module-level extracted condition/score functions --


def _should_travel(
    state: GameState,
    ctx: AgentContext,
    rs: _NavigationRuleState,
) -> bool:
    if ctx.plan.active != "travel":
        rs.waypoint_travel = None
        _skip("Travel", "no travel plan")
        return False
    if ctx.combat.engaged:
        _skip("Travel", "engaged")
        return False
    # Yield to ACQUIRE if a targetable npc is very close on the path.
    # Better to pull properly than walk face-first into threat.
    if flags.pull and ctx.pet.alive:
        from core.types import Con
        from perception.combat_eval import con_color

        for sp in state.spawns:
            if sp.is_npc and sp.hp_current > 0 and sp.owner_id == 0:
                d = state.pos.dist_to(sp.pos)
                if d < 30:
                    tc = con_color(state.level, sp.level)
                    if tc in (Con.BLUE, Con.LIGHT_BLUE, Con.WHITE):
                        _skip("Travel", f"yielding to ACQUIRE ('{sp.name}' {tc.name})")
                        return False
    if ctx.plan.travel.waypoint:
        return True  # intra-zone waypoint
    # Direct x/y target: convert to waypoint format for _get_travel_routine.
    if ctx.plan.travel.target_x != 0.0 and not ctx.plan.travel.waypoint:
        tx = ctx.plan.travel.target_x
        ty = ctx.plan.travel.target_y
        dist = state.pos.dist_to(Point(tx, ty, 0.0))
        if dist < 100:
            log.info("[TRAVEL] Travel: arrived near destination (dist=%.0f)", dist)
            ctx.plan.active = None
            return False
        ctx.plan.travel.waypoint = True
        ctx.plan.travel.destination = "camp"
        log.log(VERBOSE, "[TRAVEL] Travel: heading to (%.0f,%.0f) dist=%.0f", tx, ty, dist)
        return True
    route = ctx.plan.travel.route
    hop = ctx.plan.travel.hop_index
    if not route or hop >= len(route):
        log.log(
            VERBOSE,
            "[DECISION] Travel skip: route exhausted (hop=%d/%d) -- clearing plan",
            hop,
            len(route) if route else 0,
        )
        ctx.plan.active = None
        return False
    return True


def _score_travel(state: GameState, ctx: AgentContext) -> float:
    if ctx.plan.active != "travel":
        return 0.0
    if ctx.combat.engaged:
        return 0.0
    return 1.0


def _should_wander(state: GameState, ctx: AgentContext) -> bool:
    if not flags.wander:
        return False
    if ctx.plan.active is not None:
        _skip("Wander", f"active plan '{ctx.plan.active}'")
        return False
    if ctx.combat.engaged:
        _skip("Wander", "engaged")
        return False
    # Don't wander further from pet if it's very far
    if ctx.pet.alive and ctx.pet.spawn_id:
        for sp in state.spawns:
            if sp.spawn_id == ctx.pet.spawn_id:
                pet_dist = state.pos.dist_to(sp.pos)
                if pet_dist > 200:
                    _skip("Wander", f"pet too far ({pet_dist:.0f}u)")
                    return False
                break
    # Don't wander if pet is actively fighting something nearby
    if ctx.pet.alive:
        world = ctx.world
        if world:
            damaged = world.damaged_npcs_near(state.pos, 150)
            if damaged:
                _skip("Wander", "pet fighting")
                return False
    # If far from camp, set TRAVEL to return rather than wander further
    if ctx.camp.camp_x or ctx.camp.camp_y:
        camp_dist = ctx.camp.distance_to_camp(state)
        if camp_dist > 400 and not ctx.combat.engaged:
            from core.types import CampType, PlanType

            ctx.plan.active = PlanType.TRAVEL
            if ctx.camp.camp_type == CampType.LINEAR:
                nearest = ctx.camp.nearest_point_on_path(state.pos)
                ctx.plan.travel.target_x = nearest.x
                ctx.plan.travel.target_y = nearest.y
            else:
                ctx.plan.travel.target_x = ctx.camp.camp_x
                ctx.plan.travel.target_y = ctx.camp.camp_y
            log.info("[POSITION] Wander: outside camp range %.0fu -- returning", camp_dist)
            return False
    return True


def _score_wander(state: GameState, ctx: AgentContext) -> float:
    if not flags.wander:
        return 0.0
    if ctx.combat.engaged:
        return 0.0
    # Always available as fallback
    return 1.0


def register(brain: Brain, ctx: AgentContext, read_state_fn: ReadStateFn) -> None:
    """Register navigation rules."""

    # TRAVEL  -  navigate to destination when travel plan is active
    # Supports both cross-zone (PlanTravelRoutine) and intra-zone waypoints
    plan_travel = PlanTravelRoutine(ctx=ctx, read_state_fn=read_state_fn)
    _rs = _NavigationRuleState()

    def _get_travel_routine() -> RoutineBase:
        """Return the appropriate routine for the current travel plan."""
        if ctx.plan.travel.waypoint:
            # Intra-zone waypoint  -  plan via waypoint graph + tunnel routes
            tx = ctx.plan.travel.target_x
            ty = ctx.plan.travel.target_y
            if _rs.waypoint_travel is None:
                state = read_state_fn()
                legs = plan_travel_legs(
                    ctx.tunnel_routes, state.x, state.y, tx, ty, waypoint_graph=ctx.waypoint_graph
                )
                if len(legs) > 1 or (legs and legs[0].mode == TravelMode.MANUAL):
                    rt = MultiLegTravelRoutine(legs, read_state_fn)
                    _rs.waypoint_travel = rt
                    return rt
                # Simple A* travel
                _rs.waypoint_travel = TravelRoutine(tx, ty, read_state_fn, zoneline=False)
            assert _rs.waypoint_travel is not None
            return _rs.waypoint_travel
        return plan_travel

    class _TravelDispatcher(RoutineBase):
        """Dispatches to waypoint or plan travel based on plan_data."""

        def enter(self, state: GameState) -> None:
            r = _get_travel_routine()
            r.enter(state)

        def tick(self, state: GameState) -> RoutineStatus:
            r = _get_travel_routine()
            result = r.tick(state)
            if result == RoutineStatus.SUCCESS:
                dest = ctx.plan.travel.destination or "?"
                log.info("[TRAVEL] Travel: arrived at %s", dest)
                ctx.plan.active = None
                _rs.waypoint_travel = None
            elif result == RoutineStatus.FAILURE:
                log.warning("[TRAVEL] Travel: failed to reach destination")
                ctx.plan.active = None
                _rs.waypoint_travel = None
            return result

        @property
        def locked(self) -> bool:
            r = _get_travel_routine()
            return getattr(r, "locked", False)

        def exit(self, state: GameState) -> None:
            r = _get_travel_routine()
            r.exit(state)

    travel = _TravelDispatcher()

    brain.add_rule(
        "TRAVEL",
        lambda s: _should_travel(s, ctx, _rs),
        travel,
        max_lock_seconds=300.0,
        score_fn=lambda s: _score_travel(s, ctx),
        tier=4,
        weight=5,
    )

    # WANDER  -  fallback, roam randomly near camp
    wander = WanderRoutine(
        camp_x=ctx.camp.camp_x,
        camp_y=ctx.camp.camp_y,
        read_state_fn=read_state_fn,
        ctx=ctx,
    )

    brain.add_rule(
        "WANDER",
        lambda s: _should_wander(s, ctx),
        wander,
        score_fn=lambda s: _score_wander(s, ctx),
        tier=4,
        weight=5,
    )
