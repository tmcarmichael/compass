"""Acquire routine: find and target a nearby npc via Tab targeting.

Tab targets nearest NPC, validates the result, and falls back to an
approach walk when valid targets are beyond Tab range.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, override

from core.constants import SCAN_RADIUS
from core.timing import interruptible_sleep
from core.types import FailureCategory, GrindStyle, Point, ReadStateFn
from motor.actions import (
    clear_target,
    cycle_target,
    face_heading,
    move_forward_start,
    move_forward_stop,
    stand,
    tab_target,
)
from nav.geometry import heading_to
from perception.combat_eval import (
    Con,
    con_color,
    find_targets,
    is_valid_target,
)
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus, make_flee_predicate
from routines.target_filter import (
    estimate_exposure,
    guard_nearby,
    is_acceptable_target,
    nearby_npc_count,
    social_npc_count,
)
from util.event_schemas import AcquireResultEvent
from util.log_tiers import EVENT, VERBOSE
from util.structured_log import log_event

if TYPE_CHECKING:
    from collections.abc import Callable

    from brain.context import AgentContext
    from brain.world.model import MobProfile, WorldModel
    from perception.state import SpawnData

log = logging.getLogger(__name__)

# WorldModel scoring is the single authority for target ranking.
_WM_ACCEPT_THRESHOLD = 50.0  # minimum WorldModel score to accept a tab target
MAX_TABS = 8
CAMP_SIT_ENGAGE_RADIUS = 250.0  # max distance for camp-sit proximity acquire


class AcquireRoutine(RoutineBase):
    """Find and target a npc using Tab + real-time validation."""

    def __init__(self, ctx: AgentContext | None = None, read_state_fn: ReadStateFn | None = None) -> None:
        self._ctx = ctx
        self._read_state_fn = read_state_fn
        self._tab_count = 0
        self._locked = False
        self._seen_ids: set[int] = set()  # track all tab results to detect cycling
        self._empty_tabs = 0  # consecutive tabs with no NPC result
        self._approach_done = False  # True after walk-closer fallback
        self._has_targets = False  # set in enter() if valid npcs exist
        self._best_target: SpawnData | None = None  # highest-scored npc for approach fallback
        self._camp_sit_target: SpawnData | None = None  # proximity-acquired target for CAMP_SIT
        self._approach_active = False
        self._approach_start_x = 0.0
        # Stashed acquire summary for cycle tracker (populated in exit())
        self.last_acquire_summary: dict = {}
        self._approach_start_y = 0.0
        self._approach_walk_dist = 0.0
        self._approach_deadline = 0.0
        self._approach_best_id = 0
        self._approach_is_best = False
        self._flee_check: Callable[[], bool] | None = None

    @override
    @property
    def locked(self) -> bool:
        return self._locked

    def _get_camp_mob_names(self) -> set[str]:
        """Return the set of allowed mob names for the active camp, or empty if unconstrained."""
        camp_mob_names: set[str] = set()
        if self._ctx and self._ctx.zone.zone_config:
            for camp in self._ctx.zone.zone_config.get("camps", []):
                if camp.get("name") == self._ctx.zone.active_camp_name:
                    for mn in camp.get("mob_names", []):
                        camp_mob_names.add(mn.lower())
                    break
        return camp_mob_names

    def _find_nearest_camp_sit_npc(
        self, state: GameState, camp_mob_names: set[str]
    ) -> tuple[SpawnData | None, float]:
        """Find the nearest valid NPC within CAMP_SIT engage radius.

        Returns (best_mob, best_dist).
        """
        best_mob = None
        best_dist = CAMP_SIT_ENGAGE_RADIUS + 1.0

        for npc in state.spawns:
            if not npc.is_npc:
                continue
            if npc.hp_current <= 0:
                log.debug("CampSit: skip '%s' -- dead (hp=%d)", npc.name, npc.hp_current)
                continue
            dist = state.pos.dist_to(npc.pos)
            if dist > CAMP_SIT_ENGAGE_RADIUS:
                continue
            if not is_valid_target(npc, state.level):
                log.debug("CampSit: skip '%s' at %.0fu -- not valid target", npc.name, dist)
                continue
            # Filter by camp mob_names if configured
            if camp_mob_names:
                from eq.strings import normalize_mob_name

                if normalize_mob_name(npc.name) not in camp_mob_names:
                    log.debug("CampSit: skip '%s' -- not in camp mob_names", npc.name)
                    continue
            if dist < best_dist:
                best_dist = dist
                best_mob = npc

        return best_mob, best_dist

    def _try_camp_sit_acquire(self, state: GameState) -> bool:
        """Proximity-based acquire for CAMP_SIT grind style.

        Scans spawns for nearest valid NPC within engage radius.
        No Tab targeting, no movement -- agent stays put.

        Returns True if a valid target was found and set.
        """
        from core.features import flags

        if flags.grind_style != GrindStyle.CAMP_SIT:
            return False

        camp_mob_names = self._get_camp_mob_names()
        best_mob, best_dist = self._find_nearest_camp_sit_npc(state, camp_mob_names)

        if best_mob is None:
            log.info("[TARGET] CampSit: no valid npc within %.0fu engage radius", CAMP_SIT_ENGAGE_RADIUS)
            return False

        # Guard proximity check: if a guard is near the target OR the player,
        # suppress acquire and wait for the guard to patrol away.
        if guard_nearby(best_mob, state):
            log.info("[TARGET] CampSit: WAITING -- guard near '%s', will retry", best_mob.name)
            return False

        log.info(
            "[TARGET] CampSit: proximity acquire '%s' id=%d at %.0fu lv=%d pos=(%.0f,%.0f)",
            best_mob.name,
            best_mob.spawn_id,
            best_dist,
            best_mob.level,
            best_mob.x,
            best_mob.y,
        )
        self._camp_sit_target = best_mob
        self._has_targets = True
        if self._ctx:
            self._ctx.metrics.acquire_modes["camp_sit"] = (
                self._ctx.metrics.acquire_modes.get("camp_sit", 0) + 1
            )
        return True

    def _do_fidget(self, state: GameState) -> None:
        """Random micro-movement between pulls to vary position."""
        fidget = random.choice(["forward", "turn_forward", "slight_turn"])
        log.info("[TARGET] Acquire: fidget '%s'", fidget)

        if fidget == "forward":
            walk_dist = random.uniform(3, 6)
            move_forward_start()
            t0 = time.time()
            while time.time() - t0 < 1.5:
                if interruptible_sleep(0.15, self._flee_check):
                    move_forward_stop()
                    return
                if self._read_state_fn:
                    ns = self._read_state_fn()
                    if state.pos.dist_to(ns.pos) >= walk_dist:
                        break
            move_forward_stop()

        elif fidget == "turn_forward":
            rsf = self._read_state_fn
            if rsf is not None:
                offset = random.uniform(15, 40) * random.choice([-1, 1])
                ns = rsf()
                new_hdg = (ns.heading + offset) % 512.0
                _rsf = rsf  # bind for lambda
                face_heading(new_hdg, lambda: _rsf().heading, tolerance=8.0)
            walk_dist = random.uniform(4, 8)
            move_forward_start()
            t0 = time.time()
            while time.time() - t0 < 2.0:
                if interruptible_sleep(0.15, self._flee_check):
                    move_forward_stop()
                    return
                if self._read_state_fn:
                    ns = self._read_state_fn()
                    if state.pos.dist_to(ns.pos) >= walk_dist:
                        break
            move_forward_stop()

        elif fidget == "slight_turn":
            rsf = self._read_state_fn
            if rsf:
                offset = random.uniform(20, 60) * random.choice([-1, 1])
                ns = rsf()
                new_hdg = (ns.heading + offset) % 512.0
                face_heading(new_hdg, lambda: rsf().heading, tolerance=8.0)

        interruptible_sleep(random.uniform(0.3, 0.8), self._flee_check)

    # ------------------------------------------------------------------
    # enter() helpers
    # ------------------------------------------------------------------

    def _enter_setup(self, state: GameState) -> None:
        """Initialize instance state and ensure player is standing."""
        self._tab_count = 0
        self._locked = False
        self._seen_ids = set()
        self._empty_tabs = 0
        self._approach_done = False
        self._has_targets = False
        self._best_target = None
        self._camp_sit_target = None

        if self._read_state_fn and self._ctx:
            self._flee_check = make_flee_predicate(self._read_state_fn, self._ctx)
        else:
            self._flee_check = None

        # Ensure standing -- trust internal _sitting flag (stand_state unreliable at L4)
        import motor.actions as _ma

        if _ma.is_sitting():
            log.info("[POSITION] Acquire: standing up (internal _sitting=True)")
            stand()
            interruptible_sleep(0.5, self._flee_check)

        log.info(
            "[POSITION] Acquire: player at (%.0f, %.0f) heading=%.0f camp_dist=%.0f",
            state.x,
            state.y,
            state.heading,
            self._ctx.camp.distance_to_camp(state) if self._ctx else 0,
        )

    def _enter_clear_stale_target(self, state: GameState) -> None:
        """Clear a stale invalid target so Tab starts fresh from nearest NPC."""
        if state.target and state.target.is_npc:
            if not is_valid_target(state.target, state.level):
                log.info(
                    "[TARGET] Acquire: clearing stale target '%s' lv=%d (not valid)",
                    state.target.name,
                    state.target.level,
                )
                clear_target()
                interruptible_sleep(0.2, self._flee_check)

    def _enter_cycle_init(self, state: GameState) -> None:
        """Start cycle timer and increment defeat cycle correlation ID."""
        # CycleTracker also maintains a cycle_id (incremented on ACQUIRE
        # success notification in decision.py). We sync from it in exit()
        # after decision.py has notified the tracker.
        if self._ctx:
            self._ctx.defeat_tracker.cycle_id += 1
            self._ctx.metrics.cycle_start_time = time.time()

        if state.is_sitting:
            stand()
            interruptible_sleep(0.5, self._flee_check)

        # Vary position after multiple stationary defeats
        if self._ctx and self._ctx.metrics.should_reposition():
            self._do_fidget(state)

    def _enter_world_model_path(self, state: GameState) -> None:
        """Use WorldModel pre-ranked targets (single scoring authority).

        Populates _has_targets, _best_target, and primes approach walk if needed.
        """
        world = self._ctx.world if self._ctx else None
        if not world or not world.targets:
            # Fallback: check if any valid npcs exist via find_targets
            valid = find_targets(state.spawns, state.x, state.y, state.level)
            if not valid:
                log.info("[TARGET] Acquire: no valid npcs within %.0f units", SCAN_RADIUS)
                if self._ctx and self._ctx.spatial_memory:
                    self._ctx.spatial_memory.record_empty_scan(state.pos)
                return
            # WorldModel not available but npcs exist -- proceed with tab
            self._has_targets = True
            log.info("[TARGET] Acquire: %d valid npcs (no WorldModel scores)", len(valid))
            return

        ranked = world.targets  # sorted by WorldModel score, descending
        if not ranked:
            log.info("[TARGET] Acquire: no valid npcs within %.0f units", SCAN_RADIUS)
            if self._ctx and self._ctx.spatial_memory:
                self._ctx.spatial_memory.record_empty_scan(state.pos)
            return

        ranked = self._enter_patrol_lookahead(state, ranked, world)

        self._has_targets = True
        self._best_target = ranked[0].spawn  # highest scored npc for approach fallback

        # Record npc sightings for spatial memory
        if self._ctx and self._ctx.spatial_memory:
            for p in ranked[:5]:
                self._ctx.spatial_memory.record_sighting(p.spawn.pos, p.spawn.name, p.spawn.level)

        # Log ranked targets with scores
        log.info("[TARGET] Acquire: %d scored targets:", len(ranked))
        from nav.movement import check_spell_los

        for p in ranked[:8]:
            dist = state.pos.dist_to(p.spawn.pos)
            flags_str = ""
            if p.social_npc_count > 0:
                flags_str += f" ADDS={p.social_npc_count}"
            if p.is_patrolling:
                flags_str += f" PATROL={p.patrol_period:.0f}s"
            if not check_spell_los(state.x, state.y, state.z, p.spawn.x, p.spawn.y, p.spawn.z):
                flags_str += " NO_LOS"
            log.info(
                "[TARGET]   - '%s' id=%d score=%.1f dist=%.0f con=%s "
                "pos=(%.0f,%.0f) lv=%d iso=%.2f fight=%.0fs%s",
                p.spawn.name,
                p.spawn.spawn_id,
                p.score,
                dist,
                p.con,
                p.spawn.x,
                p.spawn.y,
                p.spawn.level,
                p.isolation_score,
                p.fight_duration_est,
                flags_str,
            )

        if self._ctx:
            self._ctx.metrics.acquire_modes["tab"] = self._ctx.metrics.acquire_modes.get("tab", 0) + 1

        # Pre-emptive approach: if best target is beyond Tab range (~90u),
        # skip the 4-tab burn and go straight to approach-walk.
        if self._best_target and self._read_state_fn:
            best_dist = state.pos.dist_to(self._best_target.pos)
            if best_dist > 110:
                log.log(
                    VERBOSE,
                    "[TARGET] Acquire: best target '%s' at %.0fu -- "
                    "beyond tab range, pre-empting approach walk",
                    self._best_target.name,
                    best_dist,
                )
                self._empty_tabs = 4
                self._approach_done = False

    def _enter_patrol_lookahead(
        self, state: GameState, ranked: list[MobProfile], world: WorldModel
    ) -> list[MobProfile]:
        """Filter targets that will have patrol threats during the fight.

        Returns the filtered ranked list (or the original if all are unsafe).
        """
        if not world.patrolling_threats():
            return ranked

        safe_ranked = []
        for p in ranked:
            exposure = estimate_exposure(state, p, self._ctx)
            safe_window = world.patrol_safe_window(p.spawn.pos, exposure)
            if safe_window == 0.0:
                log.info(
                    "[TARGET] LOOKAHEAD REJECT '%s': patrol threat already in threat range", p.spawn.name
                )
            elif safe_window < exposure:
                rest_time = exposure - p.fight_duration_est
                if rest_time > 1.0:
                    log.info(
                        "[TARGET] LOOKAHEAD REJECT '%s': patrol arrives in %.0fs, "
                        "fight+rest est %.0fs (fight=%.0f rest=%.0f)",
                        p.spawn.name,
                        safe_window,
                        exposure,
                        p.fight_duration_est,
                        rest_time,
                    )
                else:
                    log.info(
                        "[TARGET] LOOKAHEAD REJECT '%s': patrol arrives in %.0fs, fight est %.0fs",
                        p.spawn.name,
                        safe_window,
                        p.fight_duration_est,
                    )
            else:
                safe_ranked.append(p)
        if safe_ranked:
            log.info("[TARGET] LOOKAHEAD: %d/%d targets safe from patrols", len(safe_ranked), len(ranked))
            return safe_ranked
        log.info(
            "[TARGET] LOOKAHEAD: all %d targets unsafe -- proceeding with best anyway (no safe alternative)",
            len(ranked),
        )
        # Don't reject ALL targets -- better to pull risky than idle
        return ranked

    @override
    def enter(self, state: GameState) -> None:
        self._enter_setup(state)
        self._enter_clear_stale_target(state)
        self._enter_cycle_init(state)

        # CAMP_SIT: proximity-based acquire -- no Tab, no movement
        from core.features import flags as _flags

        if _flags.grind_style == GrindStyle.CAMP_SIT:
            self._try_camp_sit_acquire(state)
            return  # never fall through to Tab-based acquire

        self._enter_world_model_path(state)

    # ------------------------------------------------------------------
    # tick() phase handlers
    # ------------------------------------------------------------------

    def _tick_no_targets(self) -> RoutineStatus:
        """Fast FAILURE when enter() found no valid npcs."""
        if self._ctx:
            self._ctx.metrics.consecutive_acquire_fails += 1
        log.info(
            "[TARGET] Acquire: FAILURE -- no valid targets (consecutive_fails=%d)",
            self._ctx.metrics.consecutive_acquire_fails if self._ctx else 0,
        )
        self.failure_reason = "no_targets"
        self.failure_category = FailureCategory.PRECONDITION
        return RoutineStatus.FAILURE

    def _tick_camp_sit(self) -> RoutineStatus:
        """CAMP_SIT path: proximity target already selected in enter().

        Must Tab-target the npc before returning SUCCESS -- PullRoutine
        validates that state.target.spawn_id matches pull_target_id.
        Without Tab, pull aborts with TARGET MISMATCH every cycle.
        """
        target = self._camp_sit_target
        assert target is not None

        # Tab-target the npc so PullRoutine sees it as the game target
        if self._read_state_fn:
            from motor.actions import face_heading
            from nav.geometry import heading_to

            rsf = self._read_state_fn
            ns = rsf()
            exact = heading_to(ns.x, ns.y, target.x, target.y)
            face_heading(exact, lambda: rsf().heading, tolerance=10.0)
            tab_target()
            interruptible_sleep(0.3, self._flee_check)
            # Verify we got the right npc
            check = self._read_state_fn()
            if not check.target or check.target.spawn_id != target.spawn_id:
                # Tab didn't land on the expected npc -- cycle through
                for _t in range(6):
                    cycle_target()
                    interruptible_sleep(0.2, self._flee_check)
                    check = self._read_state_fn()
                    if check.target and check.target.spawn_id == target.spawn_id:
                        break
                else:
                    log.warning(
                        "[TARGET] CampSit: could not Tab-target '%s' -- will retry next tick", target.name
                    )
                    self.failure_reason = "tab_failed"
                    self.failure_category = FailureCategory.EXECUTION
                    return RoutineStatus.FAILURE

        log.info(
            "[TARGET] CampSit: SUCCESS -- acquired '%s' id=%d lv=%d at (%.0f,%.0f)",
            target.name,
            target.spawn_id,
            target.level,
            target.x,
            target.y,
        )
        if self._ctx:
            self._ctx.combat.pull_target_id = target.spawn_id
            self._ctx.combat.pull_target_name = target.name
            self._ctx.metrics.consecutive_acquire_fails = 0
        return RoutineStatus.SUCCESS

    def _tick_tab_exhausted(self) -> RoutineStatus:
        """FAILURE when MAX_TABS reached with no valid target."""
        log.info("[TARGET] Acquire: FAILED  -  no valid target after %d tabs", self._tab_count)
        if self._ctx:
            self._ctx.metrics.consecutive_acquire_fails += 1
        self.failure_reason = "tab_exhausted"
        self.failure_category = FailureCategory.EXECUTION
        return RoutineStatus.FAILURE

    def _tick_approach_active(self) -> RoutineStatus:
        """Continue an in-progress approach walk started on a prior tick."""
        assert self._read_state_fn is not None
        ns2 = self._read_state_fn()
        walked = Point(self._approach_start_x, self._approach_start_y, 0.0).dist_to(ns2.pos)
        if walked >= self._approach_walk_dist or time.time() > self._approach_deadline:
            move_forward_stop()
            self._approach_active = False
            self._locked = False
            self._empty_tabs = 0
            log.info(
                "[POSITION] Acquire: approach walk complete (walked %.0fu/%.0fu)",
                walked,
                self._approach_walk_dist,
            )
            return RoutineStatus.RUNNING

        # Check: valid npc within Tab range? Stop early.
        for sp in ns2.spawns:
            if not sp.is_npc or sp.hp_current <= 0:
                continue
            if is_valid_target(sp, ns2.level):
                md = ns2.pos.dist_to(sp.pos)
                if md < 90:
                    log.info("[POSITION] Acquire: npc '%s' within %.0fu -- stopping approach", sp.name, md)
                    move_forward_stop()
                    self._approach_active = False
                    self._locked = False
                    self._empty_tabs = 0
                    return RoutineStatus.RUNNING

        # Still walking
        log.debug("Acquire: approach walk in progress (walked %.0fu/%.0fu)", walked, self._approach_walk_dist)
        return RoutineStatus.RUNNING

    def _tick_approach_init(self) -> RoutineStatus:
        """Fallback: walk toward closest valid npc when 4+ empty tabs occur."""
        # Npcs may be beyond Tab range -- start approach walk and retry
        self._approach_done = True
        self._locked = True
        assert self._read_state_fn is not None
        ns = self._read_state_fn()

        best_mob, best_dist = self._approach_find_target(ns)

        approach_stop = 100
        if best_mob and best_dist > approach_stop:
            self._approach_start_walk(ns, best_mob, best_dist, approach_stop)
            return RoutineStatus.RUNNING

        self._locked = False
        return RoutineStatus.RUNNING

    def _approach_find_target(self, ns: GameState) -> tuple[SpawnData | None, float]:
        """Find the best npc to approach toward during empty-tab fallback."""
        best_mob: SpawnData | None = None
        best_dist = 9999.0

        # Try WorldModel targets
        world = self._ctx.world if self._ctx else None
        if not best_mob and world and world.targets:
            wm_target = world.targets[0]
            for spawn in ns.spawns:
                if spawn.spawn_id == wm_target.spawn.spawn_id and spawn.hp_current > 0:
                    best_mob = spawn
                    best_dist = ns.pos.dist_to(spawn.pos)
                    break

        # Fallback: enter()-time best target
        if not best_mob and self._best_target:
            for spawn in ns.spawns:
                if spawn.spawn_id == self._best_target.spawn_id and spawn.hp_current > 0:
                    best_mob = spawn
                    best_dist = ns.pos.dist_to(spawn.pos)
                    break

        # Last resort: closest valid npc
        if not best_mob:
            for spawn in ns.spawns:
                if spawn.is_npc and spawn.hp_current > 0:
                    if is_valid_target(spawn, ns.level):
                        d = ns.pos.dist_to(spawn.pos)
                        if d < SCAN_RADIUS and d < best_dist:
                            best_dist = d
                            best_mob = spawn

        return best_mob, best_dist

    def _approach_start_walk(
        self, ns: GameState, best_mob: SpawnData, best_dist: float, approach_stop: int
    ) -> None:
        """Orient and start a forward walk toward best_mob."""
        walk_dist = min(best_dist - approach_stop, 150.0)
        exact = heading_to(ns.x, ns.y, best_mob.x, best_mob.y)
        # Use exact heading for scored targets (precise approach
        # maximizes chance Tab grabs it); jitter for fallback targets
        world = self._ctx.world if self._ctx else None
        wm_id = world.targets[0].spawn.spawn_id if (world and world.targets) else 0
        bt_id = self._best_target.spawn_id if self._best_target else 0
        is_best = best_mob.spawn_id in (wm_id, bt_id)
        if is_best:
            approach_heading = exact
            tol = 6.0  # tight tolerance for precise approach
        else:
            approach_heading = (exact + random.gauss(0, 15.0)) % 512.0
            tol = 12.0
        log.log(
            VERBOSE,
            "[POSITION] Acquire: %d empty tabs  -  %s '%s' at %.0fu, walking %.0fu closer (exact=%s)",
            self._empty_tabs,
            "best-scored" if is_best else "closest",
            best_mob.name,
            best_dist,
            walk_dist,
            is_best,
        )
        # Force-stand before face/move: character may have sat
        # between enter() and this tick (e.g. REST fired briefly).
        import motor.actions as _ma2

        if _ma2.is_sitting():
            log.info(
                "[POSITION] Acquire: standing before approach walk (internal _sitting=True at tick start)"
            )
            stand()
            interruptible_sleep(0.4, self._flee_check)
        rsf = self._read_state_fn
        assert rsf is not None
        face_heading(approach_heading, lambda: rsf().heading, tolerance=tol)
        self._approach_start_x = ns.x
        self._approach_start_y = ns.y
        self._approach_walk_dist = walk_dist
        self._approach_deadline = time.time() + 5.0
        self._approach_best_id = best_mob.spawn_id
        self._approach_is_best = is_best
        self._approach_active = True
        move_forward_start()
        self._locked = True

    def _tick_face_first_tab(self) -> None:
        """Face toward best visible target before pressing Tab on the first press."""
        if not self._read_state_fn:
            return
        ns = self._read_state_fn()
        face_target: SpawnData | None = None
        face_los = False
        face_dist = 9999.0
        target_cons = self._ctx.zone.target_cons if self._ctx else {Con.WHITE, Con.BLUE, Con.LIGHT_BLUE}
        if not face_target:
            for sp in ns.spawns:
                if not sp.is_npc or sp.hp_current <= 0:
                    continue
                if sp.hp_current < sp.hp_max:
                    continue  # skip damaged
                c = con_color(ns.level, sp.level)
                if c not in target_cons:
                    continue
                if not is_valid_target(sp, ns.level):
                    continue
                d = ns.pos.dist_to(sp.pos)
                if d < 200 and d < face_dist:
                    from nav.movement import check_spell_los

                    has_los = check_spell_los(ns.x, ns.y, ns.z, sp.x, sp.y, sp.z)
                    if has_los:
                        if face_target and not face_los:
                            log.debug(
                                "Acquire: switching face from '%s' (NO_LOS) to '%s' (LOS clear, %.0fu)",
                                face_target.name,
                                sp.name,
                                d,
                            )
                        face_target = sp
                        face_dist = d
                        face_los = True
                    elif face_target is None:
                        face_target = sp
                        face_dist = d
                        face_los = False
        if face_target:
            exact = heading_to(ns.x, ns.y, face_target.x, face_target.y)
            rsf2 = self._read_state_fn
            assert rsf2 is not None
            face_heading(exact, lambda: rsf2().heading, tolerance=8.0)
            los_str = "LOS clear" if face_los else "NO_LOS (best available)"
            log.info(
                "[TARGET] Acquire: facing '%s' at %.0fu before Tab (%s)", face_target.name, face_dist, los_str
            )

    def _tick_tab_and_validate(self, state: GameState) -> RoutineStatus:
        """Press Tab/cycle, read result, validate and score the tabbed target."""
        self._tab_count += 1
        if self._tab_count == 1:
            self._tick_face_first_tab()
            tab_target()
        else:
            cycle_target()
        interruptible_sleep(random.uniform(0.2, 0.4), self._flee_check)

        if not self._read_state_fn:
            return RoutineStatus.RUNNING

        check = self._read_state_fn()
        target_now: SpawnData | None = check.target
        if not target_now or not target_now.is_npc or target_now.hp_current <= 0:
            log.debug(
                "Acquire: tab %d  -  no NPC targeted (target=%s)",
                self._tab_count,
                target_now.name if target_now else "none",
            )
            if not target_now or not target_now.is_npc:
                self._empty_tabs += 1
            return RoutineStatus.RUNNING

        # Track seen IDs to detect cycling through same invalid npcs
        already_seen = target_now.spawn_id in self._seen_ids
        self._seen_ids.add(target_now.spawn_id)

        # After 2+ invalid tabs, try approaching a valid target instead of
        # cycling endlessly through YELLOW npcs we can't fight
        if already_seen and self._tab_count >= 3:
            return self._tick_cycling_fallback(check, target_now)

        tab_dist = check.pos.dist_to(target_now.pos)
        return self._tick_validate_target(check, target_now, tab_dist)

    def _tick_cycling_fallback(self, check: GameState, target: SpawnData) -> RoutineStatus:
        """Handle cycling-through-same-npcs case: walk toward a valid target instead."""
        if not self._approach_done and self._read_state_fn:
            ns = self._read_state_fn()
            # Find best valid target by con preference
            best_valid: SpawnData | None = None
            best_valid_dist = 9999.0
            for spawn in ns.spawns:
                if not spawn.is_npc or spawn.hp_current <= 0:
                    continue
                con = con_color(ns.level, spawn.level)
                if self._ctx and con not in self._ctx.zone.target_cons:
                    continue
                if not is_valid_target(spawn, ns.level):
                    continue
                d = ns.pos.dist_to(spawn.pos)
                if d < SCAN_RADIUS and d < best_valid_dist:
                    best_valid = spawn
                    best_valid_dist = d

            if best_valid:
                # Force approach walk toward valid target
                self._empty_tabs = 4  # triggers approach walk logic
                self._approach_done = False
                clear_target()
                log.info(
                    "[POSITION] Acquire: only invalid cons nearby, walking toward '%s' (%s) at %.0fu",
                    best_valid.name,
                    con_color(ns.level, best_valid.level).name,
                    best_valid_dist,
                )
                return RoutineStatus.RUNNING

        log.info(
            "[TARGET] Acquire: tab %d  -  cycling through same %d npcs, stopping",
            self._tab_count,
            len(self._seen_ids),
        )
        if self._ctx:
            self._ctx.metrics.consecutive_acquire_fails += 1
        self.failure_reason = "no_valid_target"
        self.failure_category = FailureCategory.PRECONDITION
        return RoutineStatus.FAILURE

    def _tick_validate_target(self, check: GameState, target: SpawnData, tab_dist: float) -> RoutineStatus:
        """Validate acceptability and WorldModel score of a freshly tabbed target."""
        # Validate in real-time
        if not is_acceptable_target(target, check, self._ctx):
            con = con_color(check.level, target.level)
            reason = "invalid"
            if target.hp_current < target.hp_max:
                reason = "damaged"
            elif guard_nearby(target, check):
                reason = "guard_near"
            elif con not in (
                self._ctx.zone.target_cons if self._ctx else {Con.WHITE, Con.BLUE, Con.LIGHT_BLUE}
            ):
                reason = f"con={con}"
            log.log(
                VERBOSE,
                "[TARGET] Acquire: tab %d got '%s' id=%d dist=%.0f lv=%d  -  %s, cycling",
                self._tab_count,
                target.name,
                target.spawn_id,
                tab_dist,
                target.level,
                reason,
            )
            if self._ctx:
                self._ctx.metrics.acquire_invalid_tabs += 1
            # Clear target to prevent Tab from grabbing the same npc again
            clear_target()
            interruptible_sleep(random.uniform(0.1, 0.3), self._flee_check)
            return RoutineStatus.RUNNING

        # Score via WorldModel (single scoring authority)
        world = self._ctx.world if self._ctx else None
        target_score = 0.0
        if world:
            profile = world.get_profile(target.spawn_id)
            if profile:
                target_score = profile.score
            else:
                # Npc not in WorldModel (just spawned?) -- accept if valid
                target_score = _WM_ACCEPT_THRESHOLD + 1
        else:
            # No WorldModel -- accept any valid npc
            target_score = _WM_ACCEPT_THRESHOLD + 1

        adds = social_npc_count(target, check, self._ctx)
        nearby = nearby_npc_count(target, check)

        # Reject npcs with social extra_npcs nearby (will get extra_npcs on pull)
        if adds > 0:
            log.info(
                "[TARGET] Acquire: REJECT '%s' - %d social extra_npcs within threat radius, cycling",
                target.name,
                adds,
            )
            if self._ctx:
                self._ctx.metrics.acquire_invalid_tabs += 1
            clear_target()
            interruptible_sleep(random.uniform(0.1, 0.3), self._flee_check)
            return RoutineStatus.RUNNING

        if target_score < _WM_ACCEPT_THRESHOLD:
            log.log(
                VERBOSE,
                "[TARGET] Acquire: tab %d '%s' score=%.1f (threshold=%.1f) "
                "adds=%d nearby=%d  -  too low, cycling",
                self._tab_count,
                target.name,
                target_score,
                _WM_ACCEPT_THRESHOLD,
                adds,
                nearby,
            )
            interruptible_sleep(random.uniform(0.1, 0.3), self._flee_check)
            return RoutineStatus.RUNNING

        log.log(
            EVENT,
            "[TARGET] Acquire: SUCCESS  -  tab %d got '%s' id=%d score=%.1f dist=%.0f "
            "pos=(%.0f,%.0f) lv=%d extra_npcs=%d",
            self._tab_count,
            target.name,
            target.spawn_id,
            target_score,
            tab_dist,
            target.x,
            target.y,
            target.level,
            adds,
        )
        if self._ctx:
            self._ctx.combat.pull_target_id = target.spawn_id
            self._ctx.combat.pull_target_name = target.name
            self._ctx.metrics.consecutive_acquire_fails = 0
            self._ctx.metrics.acquire_tab_totals.append(self._tab_count)
        return RoutineStatus.SUCCESS

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        if not self._has_targets:
            return self._tick_no_targets()

        # CAMP_SIT: proximity target already selected in enter()
        if self._camp_sit_target is not None:
            return self._tick_camp_sit()

        # Tab limit reached  -  give up
        if self._tab_count >= MAX_TABS:
            return self._tick_tab_exhausted()

        # Continue approach walk if one is in progress (must be checked
        # before tab logic -- the walk was started on a prior tick and
        # _approach_done is already True so the init block below won't
        # re-enter, but the agent is still moving forward).
        if self._approach_active and self._read_state_fn:
            return self._tick_approach_active()

        # Fallback: if 4+ empty tabs (no NPC at all), valid npcs may be
        # beyond Tab range  -  walk toward closest valid npc and retry
        if self._empty_tabs >= 4 and not self._approach_done and self._read_state_fn:
            return self._tick_approach_init()

        # --- Tab and validate ---
        return self._tick_tab_and_validate(state)

    @override
    def exit(self, state: GameState) -> None:
        self._locked = False
        # Stop approach walk if still active -- forward key may be held
        if self._approach_active:
            move_forward_stop()
            self._approach_active = False
            log.info("[POSITION] Acquire: exit() stopped approach walk")
        if self._ctx:
            got_target = self._ctx.combat.pull_target_id is not None and self._ctx.combat.pull_target_id != 0
            target_name = self._ctx.combat.pull_target_name if got_target else ""
            # Stash for cycle tracker
            self.last_acquire_summary = {
                "tabs": self._tab_count,
                "target": target_name,
                "entity_id": self._ctx.combat.pull_target_id or 0,
                "success": got_target,
            }
            log_event(
                log,
                "acquire_result",
                f"[TARGET] Acquire: {'SUCCESS' if got_target else 'FAIL'} tabs={self._tab_count} target='{target_name}'",
                **AcquireResultEvent(
                    success=got_target,
                    tabs=self._tab_count,
                    target=target_name,
                    consecutive_fails=self._ctx.metrics.consecutive_acquire_fails,
                    entity_id=self._ctx.combat.pull_target_id or 0,
                ),
                cycle_id=self._ctx.defeat_tracker.cycle_id,
            )
            if self._ctx.diag.metrics:
                self._ctx.diag.metrics.record_action("acquire", got_target)
