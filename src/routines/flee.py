"""Flee routine: disengage and run to zoneline via waypoints.

When triggered, follows the full waypoint chain to zone out.
Locked behavior  -  only FLEE emergency can re-trigger, nothing else
can interrupt the run. Agent must reach safety before resuming normal ops.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, override

from core.timing import interruptible_sleep
from core.types import FailureCategory, Point, ReadStateFn
from motor.actions import move_forward_stop, pet_back_off
from nav.movement import move_to_point
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus
from util.event_schemas import FleeTriggerEvent
from util.forensics import compact_world
from util.structured_log import log_event

if TYPE_CHECKING:
    from brain.context import AgentContext

log = logging.getLogger(__name__)

SAFE_DISTANCE = 30.0
MAX_WAYPOINT_RETRIES = 3
TOTAL_FLEE_TIMEOUT = 120.0  # seconds before giving up on entire flee


def should_attempt_gate(
    pet_alive: bool,
    has_gate_spell: bool,
    gate_gem_set: bool,
    mana_current: int,
    gate_mana_cost: int,
) -> bool:
    """Pure function: decide whether to attempt Gate escape during flee.

    Gate is preferred over running when the pet is dead and the player
    has enough mana for the spell. Returns True if Gate should be attempted.
    """
    if pet_alive:
        return False
    if not has_gate_spell or not gate_gem_set:
        return False
    return mana_current >= gate_mana_cost


class FleeRoutine(RoutineBase):
    """Run to zoneline via waypoint chain when in danger."""

    def __init__(self, ctx: AgentContext | None = None, read_state_fn: ReadStateFn | None = None) -> None:
        self._ctx = ctx
        self._read_state_fn = read_state_fn
        self._locked = False
        self._waypoint_idx = 0
        self._waypoint_retries = 0
        self._flee_start = 0.0
        self._defenseless_wait_until = 0.0  # idle at guards when no pet
        self._wait_hp = 1.0  # HP when defenseless wait started
        self._gated = False  # True after successful gate teleport

    @override
    @property
    def locked(self) -> bool:
        return self._locked

    @override
    def enter(self, state: GameState) -> None:
        self._gated = False
        target = state.target
        pet_str = "dead"
        if self._ctx and self._ctx.pet.alive:
            php = self._ctx.world.pet_hp_pct if self._ctx.world else -1
            pet_str = f"{php * 100:.0f}%" if php >= 0 else "alive"
        target_name = target.name if target else "none"
        target_dist = state.pos.dist_to(target.pos) if target else 0
        log_event(
            log,
            "flee_trigger",
            f"[LIFECYCLE] FLEE: HP={state.hp_pct * 100:.0f}% mana={state.mana_current} target='{target_name}'",
            level=logging.WARNING,
            **FleeTriggerEvent(
                hp_pct=round(state.hp_pct, 3),
                mana=state.mana_current,
                pet=pet_str,
                pos_x=round(state.x),
                pos_y=round(state.y),
                target=target_name,
                target_dist=round(target_dist),
                entity_id=target.spawn_id if target else 0,
                world=compact_world(state),
            ),
        )
        if self._ctx:
            self._ctx.metrics.flee_count += 1
            self._ctx.combat.engaged = False
            self._ctx.combat.pull_target_id = None
            # Emit composite incident report for flee
            if self._ctx.diag.incident_reporter:
                _flee_reason = f"hp={state.hp_pct:.0%} pet={pet_str} npc={target_name}"
                self._ctx.diag.incident_reporter.report_flee(state, self._ctx, trigger_reason=_flee_reason)
        # Log nearby add count for debugging
        if self._ctx and self._ctx.world:
            damaged = self._ctx.world.damaged_npcs_near(state.pos, 40)
            if damaged:
                log.warning("[LIFECYCLE] Flee: %d damaged npcs nearby at flee time", len(damaged))
        # Gate escape: no pet + Gate memorized + enough mana.
        # Gate is always preferred over running when defenseless -- instant
        # escape regardless of HP. After gate, brain re-initializes (memorize
        # spells, summon pet) then travels back to camp.
        from eq.loadout import SpellRole, get_spell_by_role

        gate = get_spell_by_role(SpellRole.GATE)
        pet_dead = not (self._ctx and self._ctx.pet.alive)
        can_gate = gate and gate.gem and pet_dead and state.mana_current >= gate.mana_cost
        if can_gate:
            assert gate is not None
            log.warning(
                "[CAST] Flee: GATE ESCAPE -- pet dead + HP=%.0f%% + mana=%d >= %d",
                state.hp_pct * 100,
                state.mana_current,
                gate.mana_cost,
            )
            move_forward_stop()
            from motor.actions import press_gem, stand

            stand()
            interruptible_sleep(0.5)
            # Try Gate up to 3 times (npc hits interrupt the 5s cast)
            for attempt in range(3):
                press_gem(gate.gem)
                interruptible_sleep(gate.cast_time + 1.0)
                # Check if we gated (position changes dramatically on gate)
                if self._read_state_fn:
                    post = self._read_state_fn()
                    if state.pos.dist_to(post.pos) > 500:
                        log.info("[CAST] Flee: GATE SUCCESS -- teleported to bind point")
                        self._locked = True
                        self._gated = True
                        self._flee_start = time.time()
                        return
                    if post.mana_current < gate.mana_cost:
                        log.warning(
                            "[CAST] Flee: Gate cast %d failed + not enough mana to retry "
                            "-- falling back to run",
                            attempt + 1,
                        )
                        break
                    log.warning(
                        "[CAST] Flee: Gate cast %d interrupted -- retrying (%d/3)", attempt + 1, attempt + 2
                    )
            # Gate failed all attempts -- fall through to normal flee
            log.warning("[CAST] Flee: Gate failed -- falling back to run")

        # Tell pet to follow (skip if pet is already dead)
        if self._ctx and self._ctx.pet.alive:
            pet_back_off()
        else:
            log.info("[LIFECYCLE] Flee: skip pet_back_off -- pet is dead")
        move_forward_stop()
        interruptible_sleep(0.3)
        self._waypoint_idx = 0
        self._waypoint_retries = 0
        self._flee_start = time.time()
        self._locked = True
        # Recovery wait: always sit+med 60s at guards after flee.
        # Guards handle pursuing npc, player recovers HP/mana.
        self._defenseless_wait_until = -1.0  # sentinel: set real deadline on arrival

    def _get_waypoints(self) -> list[Point]:
        """Build waypoint list: flee_waypoints if available, else flee_spot."""
        if self._ctx and self._ctx.camp.flee_waypoints:
            return list(self._ctx.camp.flee_waypoints)

        # Fallback to single flee point
        if self._ctx:
            flee_x: float = (
                self._ctx.camp.flee_pos.x if self._ctx.camp.flee_pos.x != 0 else self._ctx.camp.guard_pos.x
            )
            flee_y: float = (
                self._ctx.camp.flee_pos.y if self._ctx.camp.flee_pos.y != 0 else self._ctx.camp.guard_pos.y
            )
            return [Point(flee_x, flee_y, 0.0)]
        return []

    def _tick_recovery_wait(self, ns: GameState) -> RoutineStatus | None:
        """Handle recovery wait at guards after reaching final waypoint.

        Returns a RoutineStatus to short-circuit tick, or None to fall through.
        """
        if self._defenseless_wait_until == 0.0:
            return None

        if self._defenseless_wait_until < 0:
            # First arrival -- sit down and start 60s recovery
            self._defenseless_wait_until = time.time() + 60.0
            self._wait_hp = ns.hp_pct
            from motor.actions import sit

            sit()
            log.info(
                "[ACTION] Flee: SAFE at guards -- sitting to med "
                "(until combat clears, max 60s) HP=%.0f%% Mana=%.0f%%",
                ns.hp_pct * 100,
                ns.mana_pct * 100,
            )
            return RoutineStatus.RUNNING

        # ABORT wait if npc is still hitting us (HP dropping)
        if ns.hp_pct < self._wait_hp - 0.05:
            return self._abort_recovery(ns)

        self._wait_hp = ns.hp_pct
        # Early exit: combat cleared (guards defeated npc or npc deaggro'd)
        if not ns.in_combat:
            log.info("[LIFECYCLE] Flee: combat cleared -- guards handled npc")
            from motor.actions import stand

            stand()
            self._defenseless_wait_until = 0.0
            # Fall through to TRAVEL plan + SUCCESS
        else:
            remaining = self._defenseless_wait_until - time.time()
            if remaining > 0:
                if int(remaining) % 10 == 0:
                    log.debug(
                        "[LIFECYCLE] Flee: recovery wait -- %.0fs remaining HP=%.0f%% Mana=%.0f%%",
                        remaining,
                        ns.hp_pct * 100,
                        ns.mana_pct * 100,
                    )
                return RoutineStatus.RUNNING

        # Recovery complete -- stand up
        from motor.actions import stand

        stand()
        log.info("[LIFECYCLE] Flee: recovery complete -- standing up")
        self._defenseless_wait_until = 0.0
        return None

    def _abort_recovery(self, ns: GameState) -> RoutineStatus:
        """Abort recovery wait and attempt Gate or restart flee path."""
        log.warning(
            "[LIFECYCLE] Flee: recovery ABORTED -- HP dropped %.0f%% -> %.0f%% (npc still attacking)",
            self._wait_hp * 100,
            ns.hp_pct * 100,
        )
        self._defenseless_wait_until = 0.0
        from motor.actions import stand

        stand()
        # Try Gate escape if mana allows
        from eq.loadout import SpellRole, get_spell_by_role

        gate = get_spell_by_role(SpellRole.GATE)
        if gate and gate.gem and ns.mana_current >= gate.mana_cost:
            log.warning("[CAST] Flee: attempting GATE after recovery abort (mana=%d)", ns.mana_current)
            from motor.actions import press_gem

            press_gem(gate.gem)
            interruptible_sleep(gate.cast_time + 1.0)
            post = self._read_state_fn() if self._read_state_fn else ns
            if ns.pos.dist_to(post.pos) > 500:
                log.info("[CAST] Flee: GATE SUCCESS after recovery abort")
                self._gated = True
                return RoutineStatus.SUCCESS
            log.warning("[CAST] Flee: Gate interrupted -- keep running")
        self._waypoint_idx = 0
        self._waypoint_retries = 0
        log.warning("[LIFECYCLE] Flee: restarting flee path (keep running)")
        return RoutineStatus.RUNNING

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        # Gate teleport succeeded -- skip waypoint walking entirely
        if self._gated:
            log.info("[CAST] Flee: gate teleport complete -- SUCCESS")
            return RoutineStatus.SUCCESS

        if not self._ctx or not self._read_state_fn:
            log.error("[LIFECYCLE] Flee: no context or read_state_fn")
            self.failure_reason = "no_context"
            self.failure_category = FailureCategory.PRECONDITION
            return RoutineStatus.FAILURE

        waypoints = self._get_waypoints()
        if not waypoints:
            log.error("[LIFECYCLE] Flee: no waypoints configured")
            self.failure_reason = "no_waypoints"
            self.failure_category = FailureCategory.PRECONDITION
            return RoutineStatus.FAILURE

        # Total flee timeout  -  if we've been fleeing too long, give up
        if time.time() - self._flee_start > TOTAL_FLEE_TIMEOUT:
            log.error(
                "[LIFECYCLE] Flee: TOTAL TIMEOUT after %.0fs  -  giving up", time.time() - self._flee_start
            )
            self.failure_reason = "timeout"
            self.failure_category = FailureCategory.TIMEOUT
            return RoutineStatus.FAILURE

        if self._waypoint_idx >= len(waypoints):
            # Verify we actually reached the final waypoint (not just skipped all)
            ns = self._read_state_fn()
            final_wp = waypoints[-1]
            dist_to_final = ns.pos.dist_to(final_wp)
            if dist_to_final > SAFE_DISTANCE * 2:
                # We skipped waypoints but never reached safety
                log.error(
                    "[LIFECYCLE] Flee: exhausted waypoints but %.0fu from final "
                    "waypoint  -  NOT safe (threshold %.0f)",
                    dist_to_final,
                    SAFE_DISTANCE * 2,
                )
                self.failure_reason = "not_safe"
                self.failure_category = FailureCategory.EXECUTION
                return RoutineStatus.FAILURE
            # Recovery wait: sit + med for 60s at guards.
            # Guards handle the pursuing npc. Player recovers HP/mana.
            # Always wait regardless of pet status -- safe recovery after flee.
            recovery = self._tick_recovery_wait(ns)
            if recovery is not None:
                return recovery
            # Set TRAVEL plan back to camp before exiting
            if self._ctx and self._ctx.camp.camp_pos.x:
                from core.types import PlanType

                self._ctx.plan.active = PlanType.TRAVEL
                self._ctx.plan.travel.target_x = self._ctx.camp.camp_pos.x
                self._ctx.plan.travel.target_y = self._ctx.camp.camp_pos.y
                log.info(
                    "[LIFECYCLE] Flee: set TRAVEL back to camp (%.0f, %.0f)",
                    self._ctx.camp.camp_pos.x,
                    self._ctx.camp.camp_pos.y,
                )
            log.info(
                "[LIFECYCLE] Flee: SAFE  -  reached safety at (%.0f, %.0f) after %d waypoints",
                ns.x,
                ns.y,
                len(waypoints),
            )
            return RoutineStatus.SUCCESS

        wp = waypoints[self._waypoint_idx]
        # Jitter each waypoint slightly
        jitter_x = wp.x + random.uniform(-10, 10)
        jitter_y = wp.y + random.uniform(-10, 10)
        flee_dist = state.pos.dist_to(Point(jitter_x, jitter_y, 0.0))

        log.info(
            "[POSITION] Flee: running to waypoint %d/%d (%.0f, %.0f) dist=%.0f retry=%d/%d",
            self._waypoint_idx + 1,
            len(waypoints),
            jitter_x,
            jitter_y,
            flee_dist,
            self._waypoint_retries,
            MAX_WAYPOINT_RETRIES,
        )

        arrived = move_to_point(
            Point(jitter_x, jitter_y, 0.0),
            self._read_state_fn,
            arrival_tolerance=SAFE_DISTANCE,
            timeout=45.0,
        )

        if arrived:
            self._waypoint_idx += 1
            self._waypoint_retries = 0
            ns = self._read_state_fn()
            if self._waypoint_idx >= len(waypoints):
                log.info("[POSITION] Flee: reached final waypoint at (%.0f, %.0f)", ns.x, ns.y)
            else:
                log.info(
                    "[POSITION] Flee: reached waypoint %d/%d at (%.0f, %.0f)",
                    self._waypoint_idx,
                    len(waypoints),
                    ns.x,
                    ns.y,
                )
        else:
            self._waypoint_retries += 1
            ns = self._read_state_fn()
            if self._waypoint_retries >= MAX_WAYPOINT_RETRIES:
                log.warning(
                    "[POSITION] Flee: waypoint %d/%d UNREACHABLE after %d retries "
                    "at (%.0f, %.0f)  -  skipping to next",
                    self._waypoint_idx + 1,
                    len(waypoints),
                    self._waypoint_retries,
                    ns.x,
                    ns.y,
                )
                self._waypoint_idx += 1
                self._waypoint_retries = 0
            else:
                log.warning(
                    "[POSITION] Flee: FAILED to reach waypoint %d/%d  -  "
                    "at (%.0f, %.0f) dist=%.0f, retry %d/%d",
                    self._waypoint_idx + 1,
                    len(waypoints),
                    ns.x,
                    ns.y,
                    ns.pos.dist_to(Point(jitter_x, jitter_y, 0.0)),
                    self._waypoint_retries,
                    MAX_WAYPOINT_RETRIES,
                )

        return RoutineStatus.RUNNING

    @override
    def exit(self, state: GameState) -> None:
        self._locked = False
        if self._ctx:
            self._ctx.combat.engaged = False
            self._ctx.combat.pull_target_id = None
            self._ctx.player.last_flee_time = time.time()
        log.info(
            "[LIFECYCLE] Flee: ended  -  HP=%.0f%% pos=(%.0f, %.0f)", state.hp_pct * 100, state.x, state.y
        )
