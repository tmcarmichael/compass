"""Pull routine: send pet to attack, wait for engage, cast DoT.

No approach needed  -  pet runs to the npc. Player stays put and casts from range.
Pet takes initial threat, then player follows up with a DoT.

If npc reaches the player (melee range), backstep to create space and re-send
pet. This is reactive  -  only backsteps when actually threatened, not preemptive.

Fear-pull strategy (fear_kite mode): Fear -> Pet -> DoT.
Cast Fear first to send npc running, then send pet + apply DoT while npc is feared.
Used for WHITE/YELLOW cons when fear-kite combat strategy is active.
"""

from __future__ import annotations

import logging
import random
import time
from enum import Enum, auto
from typing import TYPE_CHECKING, override

from core.constants import (
    BACKSTEP_RANGE,
    MELEE_RANGE,
    OPTIMAL_PULL_MAX,
    OPTIMAL_PULL_MIN,
    OPTIMAL_PULL_TARGET,
    PET_CLOSE_RANGE,
    PULL_ABORT_DISTANCE,
    SOCIAL_NPC_RADIUS,
)
from core.timing import interruptible_sleep, varying_sleep
from core.types import FailureCategory, Point, ReadStateFn, SpellOutcome
from eq.loadout import SpellRole, get_spell_by_role
from eq.strings import normalize_mob_name
from motor.actions import (
    face_heading,
    move_backward_start,
    move_backward_stop,
    move_forward_start,
    move_forward_stop,
    pet_attack,
    press_gem,
    stand,
)
from nav.geometry import heading_to
from perception.combat_eval import Con, con_color
from perception.queries import count_nearby_npcs, count_nearby_social
from perception.state import GameState, SpawnData
from routines.base import RoutineBase, RoutineStatus, make_flee_predicate
from util.event_schemas import PullResultEvent
from util.log_tiers import VERBOSE
from util.structured_log import log_event

if TYPE_CHECKING:
    from collections.abc import Callable

    from brain.context import AgentContext
    from eq.loadout import Spell
    from routines.casting import CastingPhase

log = logging.getLogger(__name__)

# Spells looked up by role via get_spell_by_role()  -  adapts to any level
MIN_MANA_DROP = 5  # Must see at least this much drop to confirm DoT landed
MAX_DOT_RETRIES = 4

# Heading jitter: approach angle is not pixel-perfect
FACE_JITTER_SIGMA = 6.0
FACE_TOLERANCE = 10.0


def verify_cast_landed(
    mana_before: int,
    mana_after: int,
    min_mana_drop: int,
    dot_retries: int,
    max_retries: int,
    los_blocked: bool,
) -> str:
    """Pure function: determine cast outcome from mana delta.

    Returns: "LANDED", "LOS_BLOCKED", "MAX_RETRIES", "FIZZLE_SKIP", "FIZZLE_RETRY".
    """
    mana_drop = mana_before - mana_after
    if mana_drop >= min_mana_drop:
        return "LANDED"
    # Cast failed
    if los_blocked:
        return "LOS_BLOCKED"
    if dot_retries >= max_retries:
        return "MAX_RETRIES"
    if dot_retries >= 2:
        return "FIZZLE_SKIP"
    return "FIZZLE_RETRY"


def choose_pull_strategy(
    tc: Con,
    nearby_count: int,
    learned_avg_adds: float | None,
    has_spell_candidates: bool,
    is_fear_kite: bool,
    has_fear: bool,
    fear_affordable: bool,
) -> str:
    """Pure function: select pull strategy from situational inputs.

    Returns one of: "FEAR_PULL", "SPELL_FIRST", "PET_ONLY", "PET_THEN_DOT".
    Extracted from PullRoutine._select_pull_approach for testability.
    """
    # Reject dangerous target
    if learned_avg_adds is not None and learned_avg_adds >= 3.0:
        return "ABORT"

    # Fear-pull for WHITE/YELLOW cons in fear_kite mode
    if is_fear_kite and tc in (Con.WHITE, Con.YELLOW) and has_fear and fear_affordable:
        return "FEAR_PULL"

    # Easy npc or no spells -> pet only
    if tc == Con.LIGHT_BLUE or not has_spell_candidates:
        return "PET_ONLY"

    # History shows frequent adds -> spell-first to separate
    if learned_avg_adds is not None and learned_avg_adds >= 2.0:
        return "SPELL_FIRST"

    # Clustered NPCs -> spell-first
    if nearby_count >= 2:
        return "SPELL_FIRST"

    # 20% chance spell-first for variety, 80% pet-first
    if random.random() < 0.20:
        return "SPELL_FIRST"

    return "PET_THEN_DOT"


class _Phase(Enum):
    APPROACH = auto()  # Walk to optimal pull range before engaging
    WAIT_PATROL = auto()  # Wait for patrolling threat to pass before pulling
    RANGED_PULL = auto()  # Cast spell first to separate npc from cluster
    WAIT_APPROACH = auto()  # Wait for pulled npc to run toward us
    SEND_PET = auto()  # Send pet to attack
    WAIT_PET = auto()  # Wait for pet to reach npc
    CAST_DOT = auto()  # Cast Disease Cloud
    WAIT_CAST = auto()  # Wait for cast to finish
    FEAR_CAST = auto()  # Cast Fear spell (fear-pull strategy)
    FEAR_WAIT = auto()  # Wait for Fear cast to complete
    FEAR_PET = auto()  # Send pet after feared npc
    FEAR_DOT = auto()  # Cast DoT on feared npc
    FEAR_DOT_WAIT = auto()  # Wait for DoT cast on feared npc
    ENGAGED = auto()  # Done, transition to combat


class PullRoutine(RoutineBase):
    """Send pet, wait, cast DoT. Simple and effective."""

    PET_ONLY = "pet only"
    PET_THEN_DOT = "Pet -> DC"
    SPELL_FIRST = "DC -> Pet"  # ranged pull for clustered npcs
    FEAR_PULL = "Fear -> Pet -> DoT"  # fear-kite: Fear first, then pet + DoT

    def __init__(self, ctx: AgentContext | None = None, read_state_fn: ReadStateFn | None = None) -> None:
        self._ctx = ctx
        self._read_state_fn = read_state_fn
        self._phase = _Phase.SEND_PET
        self._cast_start = 0.0
        self._mana_before_cast = 0
        self._pull_start = 0.0
        self._dot_retries = 0
        self._strategy = self.PET_THEN_DOT
        self._pull_spell: Spell | None = None  # random spell for this pull (DC or Lifetap)
        self._has_backstepped = False
        self._backstep_count = 0
        self._bs_active = False
        self._bs_target = 0.0
        self._bs_start_x = 0.0
        self._bs_start_y = 0.0
        self._bs_deadline = 0.0
        self._locked = False
        self._aborted = False
        self._approach_walking = False
        self._patrol_wait_deadline = 0.0
        self._pre_pull_mana = 0
        self._wait_pet_deadline: float = 0.0
        self._flee_check: Callable[[], bool] | None = None  # set in enter() via make_flee_predicate

    def _vitals(self, state: GameState) -> str:
        """Compact vitals string for log context."""
        pet_hp = ""
        if self._ctx and self._ctx.world:
            php = self._ctx.world.pet_hp_pct
            pet_hp = " Pet=%.0f%%" % (php * 100) if php >= 0 else " Pet=?"
        elif self._ctx and self._ctx.pet.alive:
            pet_hp = " Pet=alive"
        else:
            pet_hp = " Pet=dead"
        return f"HP={state.hp_pct * 100:.0f}% Mana={state.mana_pct * 100:.0f}%{pet_hp}"

    @override
    @property
    def locked(self) -> bool:
        return self._locked

    def _face_target(self, state: GameState, target: SpawnData) -> None:
        """Face toward target with gaussian heading jitter.

        No-op while a spell cast is active -- any turn key cancels casting in EQ.
        """
        if state.is_casting:
            log.debug("Pull: _face_target skipped -- casting in progress")
            return
        # Also do a fresh read to catch casts that started since state was snapped
        if self._read_state_fn:
            fresh = self._read_state_fn()
            if fresh.is_casting:
                log.debug("Pull: _face_target skipped -- casting confirmed by fresh read")
                return
            import motor.actions as _ma

            if _ma.is_sitting():
                log.warning("[COMBAT] Pull: _face_target -- player sitting -- forcing stand")
                stand()
                if interruptible_sleep(0.5, self._flee_check):
                    log.info("[DECISION] Pull: interrupted by FLEE urgency during stand in _face_target")
                    return
        # Use fresh player position for heading calculation (target may
        # have moved since state was snapped at tick start)
        if self._read_state_fn:
            fresh = self._read_state_fn()
            px, py = fresh.x, fresh.y
        else:
            px, py = state.x, state.y
        exact = heading_to(Point(px, py, 0.0), target.pos)
        desired = (exact + random.gauss(0, FACE_JITTER_SIGMA)) % 512.0
        rsf = self._read_state_fn
        if rsf:
            face_heading(desired, lambda: rsf().heading, tolerance=FACE_TOLERANCE)

    def _pet_recall_mob(self) -> None:
        """Call pet back to bring npc closer for LOS.

        Two /pet back off commands for reliability, then re-send attack.
        The npc follows the pet, closing the distance.
        """
        log.info("[POSITION] Pull: pet back off x2  -  bringing npc closer for LOS")
        from motor.actions import _pet_command

        _pet_command("back off")
        if interruptible_sleep(random.uniform(0.3, 0.5), self._flee_check):
            log.info("[DECISION] Pull: interrupted by FLEE urgency during pet recall (1st back off)")
            return
        _pet_command("back off")
        if interruptible_sleep(random.uniform(1.5, 2.5), self._flee_check):
            log.info("[DECISION] Pull: interrupted by FLEE urgency during pet recall (2nd back off)")
            return
        pet_attack()
        if interruptible_sleep(0.5, self._flee_check):
            log.info("[DECISION] Pull: interrupted by FLEE urgency during pet recall (re-attack)")
            return

    def _init_backstep(self) -> None:
        """Blocking backstep: hold backward for 0.4s then force stop.

        Previous approach (async start/stop) had persistent key-leak issues
        where EQ dropped the key-up event. This blocking approach holds the
        key for a fixed duration and aggressively stops.
        """
        if not self._read_state_fn:
            return
        log.info("[POSITION] Pull: BACKSTEP 0.4s  -  creating space")
        move_backward_start()
        varying_sleep(0.4, sigma=0.1)
        move_backward_stop()
        varying_sleep(0.05, sigma=0.1)
        move_backward_stop()
        # Forward tap to physically cancel any stuck backward input
        from motor.actions import move_forward_start, move_forward_stop

        move_forward_start()
        varying_sleep(0.05, sigma=0.1)
        move_forward_stop()
        self._bs_active = False

    def _tick_backstep(self) -> bool:
        """No-op: backstep is now blocking in _init_backstep."""
        return False

    def _count_nearby_npcs(self, target: SpawnData, state: GameState, radius: float = 35.0) -> int:
        """Count living NPCs near the target that could add on pull."""
        result: int = count_nearby_npcs(state, target.pos, radius, exclude_id=target.spawn_id)
        return result

    def _validate_pull_target(self, state: GameState) -> bool:
        """Validate pull target is correct, player is standing, and no social extra_npcs.

        Returns True if valid and ready to pull. Returns False and sets
        self._aborted if anything blocks the pull.
        """
        # Validate target identity: ensure current target matches what ACQUIRE selected
        if self._ctx and self._ctx.combat.pull_target_id:
            expected_id = self._ctx.combat.pull_target_id
            expected_name = self._ctx.combat.pull_target_name
            actual = state.target
            if not actual or actual.spawn_id != expected_id:
                actual_desc = f"'{actual.name}' id={actual.spawn_id}" if actual else "none"
                log.warning(
                    "[COMBAT] Pull: TARGET MISMATCH  -  expected '%s' id=%d but got %s  -  aborting pull",
                    expected_name,
                    expected_id,
                    actual_desc,
                )
                self._ctx.combat.pull_target_id = None
                self._ctx.combat.pull_target_name = ""
                self._aborted = True
                return False

        # Must be standing to pull -- trust internal _sitting flag
        import motor.actions as _ma

        if _ma.is_sitting():
            log.info("[POSITION] Pull: standing up first")
            stand()
            if interruptible_sleep(0.4, self._flee_check):
                log.info("[DECISION] Pull: interrupted by FLEE urgency while standing up")
                self._aborted = True
                return False

        target = state.target

        # Recheck social extra_npcs at pull time -- npc movement since acquire
        if target and self._ctx and self._ctx.zone.social_mob_group:
            social_npcs = count_nearby_social(
                target, state, self._ctx.zone.social_mob_group, SOCIAL_NPC_RADIUS
            )
            if social_npcs > 0:
                log.warning(
                    "[COMBAT] Pull: ABORT -- %d social extra_npcs near '%s' at pull time",
                    social_npcs,
                    target.name,
                )
                if self._ctx:
                    self._ctx.combat.pull_target_id = None
                self._aborted = True
                return False

        return True

    def _select_pull_approach(self, state: GameState, target: SpawnData | None, tc: Con) -> None:
        """Select pull strategy, spell, and initial phase.

        Delegates decision logic to choose_pull_strategy() for testability.
        """
        dot = get_spell_by_role(SpellRole.DOT)
        lifetap = get_spell_by_role(SpellRole.LIFETAP)
        fear = get_spell_by_role(SpellRole.FEAR)

        # Pick a random pull spell (DC or Lifetap)
        pull_spell_candidates = []
        if dot and state.mana_current >= dot.mana_cost:
            pull_spell_candidates.append(dot)
        if lifetap and state.mana_current >= lifetap.mana_cost:
            if getattr(lifetap, "spell_range", 200) >= 100:
                pull_spell_candidates.append(lifetap)
        self._pull_spell = random.choice(pull_spell_candidates) if pull_spell_candidates else dot

        # Gather inputs for pure decision function
        learned_avg_adds: float | None = None
        if target and self._ctx and self._ctx.fight_history:
            learned_avg_adds = self._ctx.fight_history.learned_adds(target.name)
            if learned_avg_adds is not None:
                log.info(
                    "[COMBAT] Pull: fight history for '%s': avg %.1f extra_npcs/fight",
                    target.name,
                    learned_avg_adds,
                )

        nearby = self._count_nearby_npcs(target, state) if target else 0

        _is_fear_kite: bool = bool(self._ctx and self._ctx.combat.active_strategy == "fear_kite")
        if not _is_fear_kite:
            try:
                from core.features import flags

                _is_fear_kite = getattr(flags, "grind_style", "") == "fear_kite"
            except Exception:
                pass

        decision = choose_pull_strategy(
            tc=tc,
            nearby_count=nearby,
            learned_avg_adds=learned_avg_adds,
            has_spell_candidates=bool(pull_spell_candidates),
            is_fear_kite=_is_fear_kite,
            has_fear=fear is not None,
            fear_affordable=fear is not None and state.mana_current >= fear.mana_cost,
        )

        if decision == "ABORT":
            log.warning(
                "[COMBAT] Pull: REJECTING '%s' (too many social adds)", target.name if target else "?"
            )
            if self._ctx:
                self._ctx.combat.pull_target_id = None
            self._aborted = True
            return

        _STRATEGY_MAP = {
            "FEAR_PULL": self.FEAR_PULL,
            "SPELL_FIRST": self.SPELL_FIRST,
            "PET_ONLY": self.PET_ONLY,
            "PET_THEN_DOT": self.PET_THEN_DOT,
        }
        _PHASE_MAP = {"FEAR_PULL": _Phase.FEAR_CAST, "SPELL_FIRST": _Phase.RANGED_PULL}
        self._strategy = _STRATEGY_MAP.get(decision, self.PET_THEN_DOT)
        if decision in _PHASE_MAP:
            self._phase = _PHASE_MAP[decision]

        dist = state.pos.dist_to(target.pos) if target else 0

        # Approach optimization: get to optimal range before pulling.
        # Only for pet-first strategies (PET_ONLY, PET_THEN_DOT).
        # SPELL_FIRST and FEAR_PULL have their own engagement logic.
        if self._phase == _Phase.SEND_PET and dist > OPTIMAL_PULL_MAX:
            self._phase = _Phase.APPROACH
            log.info("[POSITION] Pull: too far (%.0fu > %.0fu)  -  approaching first", dist, OPTIMAL_PULL_MAX)

    def _init_pet_send(self, state: GameState, target: SpawnData | None, tc: Con) -> None:
        """Record metrics and emit the pull-start log with nearby npc snapshot."""
        dist = state.pos.dist_to(target.pos) if target else 0.0

        self._pull_dist = dist
        if self._ctx:
            self._ctx.metrics.pull_distances.append(dist)
            if self._strategy == self.PET_ONLY:
                self._ctx.metrics.pull_pet_only_count += 1

        log.info(
            "[COMBAT] Pull: START strategy='%s' target='%s' id=%d con=%s "
            "dist=%.0f target_pos=(%.0f,%.0f) player_pos=(%.0f,%.0f) "
            "%s",
            self._strategy,
            target.name if target else "?",
            target.spawn_id if target else 0,
            tc,
            dist,
            target.x if target else 0,
            target.y if target else 0,
            state.x,
            state.y,
            self._vitals(state),
        )

        # Snapshot nearby npcs at pull time (once per pull, not per tick)
        nearby_mobs: list[tuple[str, int, float, float, float]] = []
        for sp in state.spawns:
            if not sp.is_npc or sp.hp_current <= 0:
                continue
            if sp.spawn_id == (target.spawn_id if target else 0):
                continue
            d = state.pos.dist_to(sp.pos)
            if d < 200:
                nearby_mobs.append((sp.name, sp.level, d, sp.x, sp.y))
        if nearby_mobs:
            nearby_mobs.sort(key=lambda x: x[2])
            parts = [f"{n[0]} lv{n[1]} {n[2]:.0f}u ({n[3]:.0f},{n[4]:.0f})" for n in nearby_mobs[:8]]
            log.log(VERBOSE, "[COMBAT] Pull: nearby npcs: %s", " | ".join(parts))

    @override
    def enter(self, state: GameState) -> None:
        self._phase = _Phase.SEND_PET
        self._dot_retries = 0
        self._pull_start = time.time()
        self._pull_dist = 0.0
        self._pet_engage_time = 0.0
        self._has_backstepped = False
        self._backstep_count = 0
        self._bs_active = False
        self._bs_target = 0.0
        self._bs_start_x = 0.0
        self._bs_start_y = 0.0
        self._bs_deadline = 0.0
        self._locked = False
        self._ranged_cast_phase: CastingPhase | None = None
        self._aborted = False

        # Build FLEE predicate for interruptible sleeps
        if self._read_state_fn and self._ctx:
            self._flee_check = make_flee_predicate(self._read_state_fn, self._ctx)
        else:
            self._flee_check = None

        if not self._validate_pull_target(state):
            return

        target = state.target
        tc = Con.WHITE
        if target and state.level > 0:
            tc = con_color(state.level, target.level)

        # Store target info on ctx NOW  -  npc may die during pull before
        # combat.py gets a chance to set these (pet one-shots weak npcs)
        if target and self._ctx:
            self._ctx.defeat_tracker.last_fight_name = target.name
            self._ctx.defeat_tracker.last_fight_id = target.spawn_id
            self._ctx.defeat_tracker.last_fight_x = target.x
            self._ctx.defeat_tracker.last_fight_y = target.y

        self._select_pull_approach(state, target, tc)
        if self._aborted:
            return

        self._init_pet_send(state, target, tc)

    def _handle_target_lost(self, state: GameState, target: SpawnData | None) -> RoutineStatus:
        """Handle target dead/gone during pull (pet defeated it or lost before send)."""
        if self._phase not in (_Phase.SEND_PET,):
            if self._ctx:
                # Use stored pull info  -  target may have despawned
                defeat_id = target.spawn_id if target else self._ctx.combat.pull_target_id or 0
                defeat_name = target.name if target else self._ctx.defeat_tracker.last_fight_name or ""
                defeat_pos = target.pos if target else state.pos
                log.info(
                    "[COMBAT] Pull: target dead/gone (pet defeated it)  -  "
                    "recording defeat '%s' id=%d at (%.0f,%.0f)",
                    defeat_name,
                    defeat_id,
                    defeat_pos.x,
                    defeat_pos.y,
                )
                self._ctx.record_kill(defeat_id, name=defeat_name, pos=defeat_pos)
                self._ctx.combat.engaged = False
                self._ctx.combat.pull_target_id = None
                self._ctx.combat.last_dot_time = time.time()
            # Post-defeat idle  -  let pet disengage before next pull
            idle = random.uniform(1.0, 2.0)
            log.info("[COMBAT] Pull: post-defeat idle %.1fs", idle)
            if interruptible_sleep(idle, self._flee_check):
                log.info("[DECISION] Pull: interrupted by FLEE urgency during post-defeat idle")
            return RoutineStatus.SUCCESS
        log.warning(
            "[COMBAT] Pull: LOST TARGET before pet sent (target=%s)", target.name if target else "none"
        )
        self.failure_reason = "target_lost"
        self.failure_category = FailureCategory.ENVIRONMENT
        if self._ctx:
            self._ctx.combat.pull_target_id = None
        return RoutineStatus.FAILURE

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        if self._aborted:
            log.info("[COMBAT] Pull: aborted (target mismatch in enter)")
            self.failure_reason = "target_mismatch"
            self.failure_category = FailureCategory.PRECONDITION
            return RoutineStatus.FAILURE
        elapsed = time.time() - self._pull_start

        # Safety timeout
        if elapsed > 25.0:
            log.warning("[COMBAT] Pull: TIMEOUT after %.1fs", elapsed)
            self.failure_reason = "timeout"
            self.failure_category = FailureCategory.TIMEOUT
            if self._ctx:
                self._ctx.combat.pull_target_id = None
            return RoutineStatus.FAILURE

        target = state.target
        if not target or not target.is_npc or target.hp_current <= 0:
            return self._handle_target_lost(state, target)

        dist = state.pos.dist_to(target.pos)

        # Non-blocking backstep: continue advancing if one is in progress
        if self._bs_active:
            if self._tick_backstep():
                return RoutineStatus.RUNNING
            # Backstep finished -- re-send pet
            pet_attack()
            if interruptible_sleep(0.3, self._flee_check):
                log.info("[DECISION] Pull: interrupted by FLEE urgency after backstep re-send")
            return RoutineStatus.RUNNING

        # Reactive backstep: npc in melee range  -  back up and re-send pet.
        # Allow up to 3 backsteps total (DC pull = player has initial threat,
        # npc will follow; pet needs time to build threat).
        backstep_limit = 3
        if (
            dist < MELEE_RANGE
            and self._backstep_count < backstep_limit
            and self._phase not in (_Phase.SEND_PET, _Phase.ENGAGED)
        ):
            log.info(
                "[POSITION] Pull: npc in melee (dist=%.0f)  -  backstep %d/%d + re-send pet "
                "player_pos=(%.0f,%.0f) mob_pos=(%.0f,%.0f)",
                dist,
                self._backstep_count + 1,
                backstep_limit,
                state.x,
                state.y,
                target.x,
                target.y,
            )
            self._init_backstep()
            self._backstep_count += 1
            self._has_backstepped = True
            if interruptible_sleep(0.3, self._flee_check):
                log.info("[DECISION] Pull: interrupted by FLEE urgency during reactive backstep")
                move_backward_stop()
                self._bs_active = False
            return RoutineStatus.RUNNING

        # Abort if target has moved too far away
        if dist > PULL_ABORT_DISTANCE:
            log.warning("[COMBAT] Pull: ABORT  -  target '%s' too far (dist=%.0f)", target.name, dist)
            self.failure_reason = "target_too_far"
            self.failure_category = FailureCategory.ENVIRONMENT
            if self._ctx:
                self._ctx.combat.pull_target_id = None
            return RoutineStatus.FAILURE

        # -- Phase dispatch --
        _PHASE_HANDLERS = {
            _Phase.APPROACH: self._tick_approach,
            _Phase.WAIT_PATROL: self._tick_wait_patrol,
            _Phase.RANGED_PULL: self._tick_ranged_pull,
            _Phase.WAIT_APPROACH: self._tick_wait_approach,
            _Phase.SEND_PET: self._tick_send_pet,
            _Phase.WAIT_PET: self._tick_wait_pet,
            _Phase.CAST_DOT: self._tick_cast_dot,
            _Phase.WAIT_CAST: self._tick_wait_cast,
            _Phase.FEAR_CAST: self._tick_fear_cast,
            _Phase.FEAR_WAIT: self._tick_fear_wait,
            _Phase.FEAR_PET: self._tick_fear_pet,
            _Phase.FEAR_DOT: self._tick_fear_dot,
            _Phase.FEAR_DOT_WAIT: self._tick_fear_dot_wait,
            _Phase.ENGAGED: self._tick_engaged,
        }
        handler = _PHASE_HANDLERS.get(self._phase)
        if handler:
            return handler(state, target, dist)
        self.failure_reason = "unknown_phase"
        self.failure_category = FailureCategory.UNKNOWN
        return RoutineStatus.FAILURE

    # -- Phase handlers (extracted from tick for readability) --

    def _check_patrol_safe(self, target: SpawnData, fight_est: float) -> float:
        """Check if patrolling threats will arrive during fight.

        Returns seconds until nearest patrol arrives (inf if safe).
        """
        if not self._ctx or not hasattr(self._ctx, "world") or not self._ctx.world:
            return float("inf")
        return float(self._ctx.world.patrol_safe_window(target.pos, fight_est))

    def _tick_approach(self, state: GameState, target: SpawnData, dist: float) -> RoutineStatus:
        """APPROACH: non-blocking walk to optimal pull range."""
        if OPTIMAL_PULL_MIN <= dist <= OPTIMAL_PULL_MAX:
            move_forward_stop()
            # Check patrol safety before pulling
            fight_est = 20.0  # default estimate
            if self._ctx and self._ctx.fight_history:
                mob_base = normalize_mob_name(target.name)
                learned = self._ctx.fight_history.learned_duration(mob_base)
                if learned is not None and learned > 0:
                    fight_est = learned
            safe_window = self._check_patrol_safe(target, fight_est)
            if safe_window < fight_est and safe_window > 0:
                log.info(
                    "[POSITION] Pull: APPROACH done but patrol arrives in %.0fs "
                    "(fight est %.0fs)  -  waiting for patrol to pass",
                    safe_window,
                    fight_est,
                )
                self._patrol_wait_deadline = time.time() + min(safe_window + 5.0, 15.0)
                self._phase = _Phase.WAIT_PATROL
                return RoutineStatus.RUNNING
            log.info("[POSITION] Pull: APPROACH done  -  dist=%.0f, proceeding to SEND_PET", dist)
            self._phase = _Phase.SEND_PET
            return RoutineStatus.RUNNING

        self._locked = True
        if dist > OPTIMAL_PULL_MAX:
            # Initialize approach walk on first tick
            if not self._approach_walking:
                self._face_target(state, target)
                move_forward_start()
                self._approach_walking = True
                self._approach_deadline = time.time() + 8.0
                self._approach_last_face = time.time()
                return RoutineStatus.RUNNING

            # Check arrival
            if dist <= OPTIMAL_PULL_TARGET:
                move_forward_stop()
                self._approach_walking = False
                self._phase = _Phase.SEND_PET
                return RoutineStatus.RUNNING

            # Check abort (npc fled)
            if dist > PULL_ABORT_DISTANCE:
                log.warning("[POSITION] Pull: APPROACH abort  -  target moved too far (dist=%.0f)", dist)
                move_forward_stop()
                self._approach_walking = False
                self.failure_reason = "approach_abort"
                self.failure_category = FailureCategory.ENVIRONMENT
                if self._ctx:
                    self._ctx.combat.pull_target_id = None
                return RoutineStatus.FAILURE

            # Check timeout
            if time.time() > self._approach_deadline:
                move_forward_stop()
                self._approach_walking = False
                self._phase = _Phase.SEND_PET
                return RoutineStatus.RUNNING

            # Re-face periodically (npc may have moved)
            if time.time() - self._approach_last_face > 2.0:
                move_forward_stop()
                self._face_target(state, target)
                move_forward_start()
                self._approach_last_face = time.time()

            return RoutineStatus.RUNNING  # still walking
        else:
            # Too close  -  backstep away (non-blocking)
            self._init_backstep()
            self._has_backstepped = True
            return RoutineStatus.RUNNING

    def _tick_wait_patrol(self, state: GameState, target: SpawnData, dist: float) -> RoutineStatus:
        """WAIT_PATROL: wait for patrolling threat to pass before pulling."""
        now = time.time()
        deadline = self._patrol_wait_deadline

        # Re-check: is the patrol still a problem?
        fight_est = 20.0
        if self._ctx and self._ctx.fight_history:
            mob_base = normalize_mob_name(target.name)
            learned = self._ctx.fight_history.learned_duration(mob_base)
            if learned is not None and learned > 0:
                fight_est = learned
        safe_window = self._check_patrol_safe(target, fight_est)

        if safe_window >= fight_est:
            log.info(
                "[COMBAT] Pull: patrol passed (safe_window=%.0fs, fight_est=%.0fs)"
                "  -  proceeding to SEND_PET",
                safe_window,
                fight_est,
            )
            self._phase = _Phase.SEND_PET
            return RoutineStatus.RUNNING

        if now > deadline:
            log.warning(
                "[COMBAT] Pull: patrol wait timeout (%.0fs)  -  pulling anyway (safe_window=%.0fs)",
                deadline - now + 15.0,
                safe_window,
            )
            self._phase = _Phase.SEND_PET
            return RoutineStatus.RUNNING

        # Still waiting -- FLEE can interrupt between ticks
        return RoutineStatus.RUNNING

    def _tick_ranged_pull(self, state: GameState, target: SpawnData, dist: float) -> RoutineStatus:
        """RANGED_PULL: cast pull spell (DC or Lifetap) to tag npc at range."""
        dot = get_spell_by_role(SpellRole.DOT)
        lifetap = get_spell_by_role(SpellRole.LIFETAP)
        ps = self._pull_spell or dot
        if not ps:
            log.warning("[CAST] Pull: RANGED_PULL -- no pull spell available, falling back to pet")
            self._phase = _Phase.SEND_PET
            return RoutineStatus.RUNNING
        # Non-blocking cast: first tick starts the cast, subsequent ticks poll
        if not hasattr(self, "_ranged_cast_phase") or self._ranged_cast_phase is None:
            # Terrain LOS pre-check: skip spell if terrain blocks the ray
            from nav.movement import check_spell_los

            if not check_spell_los(state.x, state.y, state.z, target.x, target.y, target.z):
                log.info(
                    "[CAST] Pull: RANGED_PULL skipped -- terrain blocks LOS to '%s' at dist=%.0f (dz=%.0f)",
                    target.name,
                    dist,
                    target.z - state.z,
                )
                # Fall through to pet-only
                if self._ctx:
                    self._ctx.metrics.total_casts += 0  # no cast attempted
                self._phase = _Phase.SEND_PET if self._strategy != self.SPELL_FIRST else _Phase.SEND_PET
                return RoutineStatus.RUNNING

            self._face_target(state, target)
            if interruptible_sleep(random.uniform(0.3, 0.5), self._flee_check):
                log.info("[DECISION] Pull: interrupted by FLEE urgency during ranged pull settle")
                return RoutineStatus.RUNNING
            pre = self._read_state_fn() if self._read_state_fn else state
            self._pre_pull_mana = pre.mana_current
            log.info(
                "[CAST] Pull: RANGED_PULL  -  casting %s gem=%d at '%s' dist=%.0f "
                "stand_state=%d casting_mode=%d %s",
                ps.name,
                ps.gem,
                target.name,
                dist,
                pre.stand_state,
                pre.casting_mode,
                self._vitals(pre),
            )
            press_gem(ps.gem)
            self._locked = True
            from routines.casting import CastingPhase

            self._ranged_cast_phase = CastingPhase(ps.cast_time, f"{ps.name}(pull)", self._read_state_fn)
            return RoutineStatus.RUNNING
        result = self._ranged_cast_phase.tick()
        from routines.casting import CastResult

        if result == CastResult.CASTING:
            return RoutineStatus.RUNNING
        # Cast complete -- verify mana dropped (spell actually landed)
        self._ranged_cast_phase = None
        ns = self._read_state_fn() if self._read_state_fn else state
        mana_cost = self._pre_pull_mana - ns.mana_current
        if mana_cost < 3:
            # Cast failed -- check if LOS blocked (skip retry) or fizzle (retry once)
            self._dot_retries += 1
            los_blocked = self._ctx and self._ctx.combat.last_cast_result == SpellOutcome.LOS_BLOCKED
            if los_blocked:
                log.warning("[CAST] Pull: %s LOS BLOCKED  -  sending pet (no retry from same spot)", ps.name)
            elif self._dot_retries <= 1:
                log.warning(
                    "[CAST] Pull: %s FAILED (mana %d->%d, no cost)  -  retrying (attempt %d)",
                    ps.name,
                    self._pre_pull_mana,
                    ns.mana_current,
                    self._dot_retries + 1,
                )
                self._ranged_cast_phase = None  # reset for retry
                return RoutineStatus.RUNNING  # will re-enter RANGED_PULL
            else:
                log.warning("[CAST] Pull: %s failed twice  -  sending pet anyway", ps.name)
        else:
            log.info("[CAST] Pull: %s LANDED (mana %d->%d)", ps.name, self._pre_pull_mana, ns.mana_current)
        if self._ctx:
            if ps is dot:
                self._ctx.combat.last_dot_time = time.time()
            elif ps is lifetap:
                self._ctx.combat.last_lifetap_time = time.time()
            self._ctx.metrics.total_casts += 1
        # Short random delay then send pet
        if interruptible_sleep(random.uniform(0.1, 0.5), self._flee_check):
            log.info("[DECISION] Pull: interrupted by FLEE urgency before sending pet after ranged pull")
            return RoutineStatus.RUNNING
        self._face_target(state, target)
        pet_attack()
        log.info("[COMBAT] Pull: pet sent after %s (dist=%.0f)", ps.name, dist)
        self._phase = _Phase.WAIT_PET
        self._locked = True
        return RoutineStatus.RUNNING

    def _tick_wait_approach(self, state: GameState, target: SpawnData, dist: float) -> RoutineStatus:
        """WAIT_APPROACH: npc is coming to us, wait until separated."""
        elapsed = time.time() - self._pull_start
        if dist < PET_CLOSE_RANGE:
            log.info("[COMBAT] Pull: npc approaching (dist=%.0f)  -  sending pet", dist)
            self._phase = _Phase.SEND_PET
        elif elapsed > 8.0:
            log.info("[COMBAT] Pull: npc didn't approach after %.1fs  -  sending pet", elapsed)
            self._phase = _Phase.SEND_PET
        else:
            # Backstep away from cluster while waiting (non-blocking)
            if dist < BACKSTEP_RANGE and not self._has_backstepped:
                self._init_backstep()
                self._has_backstepped = True
        return RoutineStatus.RUNNING

    def _tick_send_pet(self, state: GameState, target: SpawnData, dist: float) -> RoutineStatus:
        """SEND_PET: face npc, send pet."""
        self._face_target(state, target)
        log.info(
            "[COMBAT] Pull: SEND_PET -> '%s' dist=%.0f target_pos=(%.0f,%.0f)",
            target.name,
            dist,
            target.x,
            target.y,
        )
        pet_attack()
        self._phase = _Phase.WAIT_PET
        self._locked = True  # committed  -  pet is chasing
        return RoutineStatus.RUNNING

    def _tick_wait_pet(self, state: GameState, target: SpawnData, dist: float) -> RoutineStatus:
        """WAIT_PET: non-blocking poll for pet hit confirmation."""
        # First tick: initialize tracking
        if not hasattr(self, "_wait_pet_deadline") or self._wait_pet_deadline == 0:
            max_wait = max(3.0, min(dist / 35 + 1.5, 12.0))
            self._wait_pet_deadline = time.time() + max_wait
            self._wait_pet_initial_hp = target.hp_current
            self._wait_pet_initial_dist = dist
            self._wait_pet_logged = False
            log.log(
                VERBOSE,
                "[COMBAT] Pull: WAIT_PET polling HP (npc dist=%.0f, max_wait=%.1fs, "
                "npc HP=%d/%d) player_pos=(%.0f,%.0f) mob_pos=(%.0f,%.0f)",
                dist,
                max_wait,
                target.hp_current,
                target.hp_max,
                state.x,
                state.y,
                target.x,
                target.y,
            )

        now = time.time()

        # Check: pet hit confirmed (HP dropped)?
        if target.hp_current < self._wait_pet_initial_hp:
            engage_elapsed = now - (self._wait_pet_deadline - max(3.0, min(dist / 35 + 1.5, 12.0)))
            self._pet_engage_time = max(0.1, engage_elapsed)
            log.info(
                "[COMBAT] Pull: pet HIT confirmed (HP %d->%d) after %.1fs mob_dist=%.0f (was %.0f)",
                self._wait_pet_initial_hp,
                target.hp_current,
                self._pet_engage_time,
                dist,
                self._pull_dist,
            )
            if self._ctx:
                self._ctx.metrics.pull_engage_times.append(self._pet_engage_time)
            self._wait_pet_deadline = 0
            # Proceed to next phase
            if self._strategy == self.PET_ONLY:
                log.info("[COMBAT] Pull: pet-only strategy, skipping DoT")
                self._phase = _Phase.ENGAGED
            elif self._strategy == self.SPELL_FIRST:
                log.info(
                    "[COMBAT] Pull: spell-first, %s already cast, skipping CAST_DOT",
                    self._pull_spell.name if self._pull_spell else "spell",
                )
                self._phase = _Phase.ENGAGED
            else:
                # Skip DoT on easy npcs -- pet handles them without spell help
                tc = con_color(state.level, target.level) if target else Con.WHITE
                if tc in (Con.LIGHT_BLUE, Con.BLUE):
                    log.log(VERBOSE, "[COMBAT] Pull: skipping DoT on %s con (pet handles)", tc)
                    self._phase = _Phase.ENGAGED
                else:
                    self._phase = _Phase.CAST_DOT
            return RoutineStatus.RUNNING

        # Check: npc reached player (need backstep)?
        if dist < 25 and not self._has_backstepped:
            log.info(
                "[POSITION] Pull: npc reached player during WAIT_PET (dist=%.0f) "
                "player_pos=(%.0f,%.0f) mob_pos=(%.0f,%.0f)",
                dist,
                state.x,
                state.y,
                target.x,
                target.y,
            )
            self._init_backstep()
            pet_attack()
            self._has_backstepped = True
            self._wait_pet_deadline = 0
            return RoutineStatus.RUNNING

        # Check: timeout?
        if now > self._wait_pet_deadline:
            if not self._wait_pet_logged:
                direction = (
                    "closer"
                    if dist < self._wait_pet_initial_dist - 5
                    else "further"
                    if dist > self._wait_pet_initial_dist + 5
                    else "same"
                )
                log.info(
                    "[COMBAT] Pull: WAIT_PET timeout  -  npc moved %s (dist %.0f->%.0f)",
                    direction,
                    self._wait_pet_initial_dist,
                    dist,
                )
                self._wait_pet_logged = True

            # Re-send pet
            log.info("[COMBAT] Pull: pet didn't engage  -  re-sending")
            pet_attack()
            self._wait_pet_deadline = 0
            if self._strategy == self.PET_ONLY:
                self._phase = _Phase.ENGAGED
            elif self._strategy == self.SPELL_FIRST:
                log.info(
                    "[COMBAT] Pull: spell-first, %s already cast, skipping CAST_DOT",
                    self._pull_spell.name if self._pull_spell else "spell",
                )
                self._phase = _Phase.ENGAGED
            else:
                self._phase = _Phase.CAST_DOT
            return RoutineStatus.RUNNING

        # Still waiting -- return RUNNING, brain evaluates FLEE on next tick
        return RoutineStatus.RUNNING

    def _tick_cast_dot(self, state: GameState, target: SpawnData, dist: float) -> RoutineStatus:
        """CAST_DOT: cast follow-up spell after pet sent."""
        # Terrain LOS pre-check
        from nav.movement import check_spell_los

        if not check_spell_los(state.x, state.y, state.z, target.x, target.y, target.z):
            log.info("[CAST] Pull: CAST_DOT skipped -- terrain blocks LOS (dz=%.0f)", target.z - state.z)
            self._phase = _Phase.ENGAGED
            return RoutineStatus.RUNNING

        dot = get_spell_by_role(SpellRole.DOT)
        ps = self._pull_spell or dot
        if not ps:
            log.warning("[CAST] Pull: CAST_DOT -- no spell available, skipping to ENGAGED")
            self._phase = _Phase.ENGAGED
            return RoutineStatus.RUNNING
        self._face_target(state, target)
        if interruptible_sleep(random.uniform(0.3, 0.5), self._flee_check):
            log.info("[DECISION] Pull: interrupted by FLEE urgency during cast_dot settle")
            return RoutineStatus.RUNNING
        self._mana_before_cast = state.mana_current
        # Read fresh state for diagnostics
        pre = self._read_state_fn() if self._read_state_fn else state
        log.info(
            "[CAST] Pull: CAST %s gem=%d dist=%.0f mana=%d stand_state=%d casting_mode=%d target_HP=%d/%d",
            ps.name,
            ps.gem,
            dist,
            pre.mana_current,
            pre.stand_state,
            pre.casting_mode,
            target.hp_current,
            target.hp_max,
        )
        press_gem(ps.gem)
        self._cast_start = time.time()
        self._phase = _Phase.WAIT_CAST
        if self._ctx:
            self._ctx.metrics.total_casts += 1
        return RoutineStatus.RUNNING

    def _tick_wait_cast(self, state: GameState, target: SpawnData, dist: float) -> RoutineStatus:
        """WAIT_CAST: wait for pull spell to land."""
        dot = get_spell_by_role(SpellRole.DOT)
        ps = self._pull_spell or dot
        # Poll casting_mode for early exit, fall back to timer
        if self._read_state_fn:
            fresh = self._read_state_fn()
            if fresh.is_casting:
                if interruptible_sleep(0.1, self._flee_check):
                    log.info("[DECISION] Pull: interrupted by FLEE urgency during WAIT_CAST polling")
                return RoutineStatus.RUNNING
        else:
            cast_elapsed = time.time() - self._cast_start
            ps_cast = ps.cast_time if ps else 1.5
            if cast_elapsed < ps_cast + 0.3:
                if interruptible_sleep(0.3, self._flee_check):
                    log.info("[DECISION] Pull: interrupted by FLEE urgency during WAIT_CAST timer")
                return RoutineStatus.RUNNING

        # Check if mana decreased enough -- delegate to pure function
        if self._read_state_fn:
            new_state = self._read_state_fn()
            self._dot_retries += 1
            los_blocked = self._ctx and self._ctx.combat.last_cast_result == SpellOutcome.LOS_BLOCKED
            verdict = verify_cast_landed(
                self._mana_before_cast,
                new_state.mana_current,
                MIN_MANA_DROP,
                self._dot_retries,
                MAX_DOT_RETRIES,
                bool(los_blocked),
            )
            if verdict == "LANDED":
                log.info(
                    "[CAST] Pull: %s landed (mana %d->%d)",
                    ps.name if ps else "spell",
                    self._mana_before_cast,
                    new_state.mana_current,
                )
            elif verdict in ("LOS_BLOCKED", "MAX_RETRIES", "FIZZLE_SKIP"):
                log.info("[CAST] Pull: %s %s -- skipping to ENGAGED", ps.name if ps else "spell", verdict)
                self._phase = _Phase.ENGAGED
                return RoutineStatus.RUNNING
            else:  # FIZZLE_RETRY
                if self._ctx:
                    self._ctx.metrics.pull_dc_fizzles += 1
                log.info(
                    "[CAST] Pull: %s FIZZLE retry %d/%d",
                    ps.name if ps else "spell",
                    self._dot_retries,
                    MAX_DOT_RETRIES,
                )
                interruptible_sleep(0.5, self._flee_check)
                self._phase = _Phase.CAST_DOT
                return RoutineStatus.RUNNING

        self._phase = _Phase.ENGAGED
        return RoutineStatus.RUNNING

    def _tick_fear_cast(self, state: GameState, target: SpawnData, dist: float) -> RoutineStatus:
        """FEAR_CAST: cast Fear to send npc running."""
        fear = get_spell_by_role(SpellRole.FEAR)
        if not fear:
            log.warning("[CAST] Pull: FEAR_CAST but no Fear spell  -  falling back to pet-first")
            self._phase = _Phase.SEND_PET
            self._strategy = self.PET_THEN_DOT
            return RoutineStatus.RUNNING
        self._face_target(state, target)
        log.info(
            "[CAST] Pull: FEAR_CAST  -  casting '%s' at '%s' dist=%.0f mana=%d",
            fear.name,
            target.name,
            dist,
            state.mana_current,
        )
        press_gem(fear.gem)
        self._locked = True
        self._cast_start = time.time()
        self._mana_before_cast = state.mana_current
        self._phase = _Phase.FEAR_WAIT
        if self._ctx:
            self._ctx.metrics.total_casts += 1
        return RoutineStatus.RUNNING

    def _tick_fear_wait(self, state: GameState, target: SpawnData, dist: float) -> RoutineStatus:
        """FEAR_WAIT: wait for Fear cast to complete."""
        fear = get_spell_by_role(SpellRole.FEAR)
        cast_time = fear.cast_time if fear else 2.0
        if self._read_state_fn:
            fresh = self._read_state_fn()
            if fresh.is_casting:
                if interruptible_sleep(0.1, self._flee_check):
                    log.info("[DECISION] Pull: interrupted by FLEE urgency during FEAR_WAIT polling")
                return RoutineStatus.RUNNING
        else:
            cast_elapsed = time.time() - self._cast_start
            if cast_elapsed < cast_time + 0.3:
                if interruptible_sleep(0.3, self._flee_check):
                    log.info("[DECISION] Pull: interrupted by FLEE urgency during FEAR_WAIT timer")
                return RoutineStatus.RUNNING

        # Check if Fear landed (mana dropped)
        if self._read_state_fn:
            new_state = self._read_state_fn()
            mana_drop = self._mana_before_cast - new_state.mana_current
            if mana_drop < MIN_MANA_DROP:
                log.info(
                    "[CAST] Pull: Fear FIZZLE (mana %d->%d, drop=%d)  -  falling back to pet-first",
                    self._mana_before_cast,
                    new_state.mana_current,
                    mana_drop,
                )
                self._phase = _Phase.SEND_PET
                self._strategy = self.PET_THEN_DOT
                return RoutineStatus.RUNNING
            log.info(
                "[CAST] Pull: Fear LANDED (mana %d->%d)  -  npc should be running, sending pet",
                self._mana_before_cast,
                new_state.mana_current,
            )

        # Brief pause for Fear to take effect (npc starts running)
        if interruptible_sleep(0.5, self._flee_check):
            log.info("[DECISION] Pull: interrupted by FLEE urgency during fear take-effect pause")
            return RoutineStatus.RUNNING
        self._phase = _Phase.FEAR_PET
        return RoutineStatus.RUNNING

    def _tick_fear_pet(self, state: GameState, target: SpawnData, dist: float) -> RoutineStatus:
        """FEAR_PET: send pet after the feared npc."""
        dot = get_spell_by_role(SpellRole.DOT)
        self._face_target(state, target)
        log.info("[COMBAT] Pull: FEAR_PET  -  sending pet to feared '%s' dist=%.0f", target.name, dist)
        pet_attack()
        if interruptible_sleep(0.3, self._flee_check):
            log.info("[DECISION] Pull: interrupted by FLEE urgency during FEAR_PET (1st attack)")
            return RoutineStatus.RUNNING
        pet_attack()
        if interruptible_sleep(0.5, self._flee_check):
            log.info("[DECISION] Pull: interrupted by FLEE urgency during FEAR_PET (2nd attack)")
            return RoutineStatus.RUNNING

        if dot and state.mana_current >= dot.mana_cost:
            self._phase = _Phase.FEAR_DOT
        else:
            log.info("[COMBAT] Pull: fear-pull skipping DoT (no dot or low mana)")
            self._phase = _Phase.ENGAGED
        return RoutineStatus.RUNNING

    def _tick_fear_dot(self, state: GameState, target: SpawnData, dist: float) -> RoutineStatus:
        """FEAR_DOT: cast DoT on the feared npc."""
        dot = get_spell_by_role(SpellRole.DOT)
        if not dot:
            self._phase = _Phase.ENGAGED
            return RoutineStatus.RUNNING
        self._face_target(state, target)
        self._mana_before_cast = state.mana_current
        log.info(
            "[CAST] Pull: FEAR_DOT  -  casting '%s' on feared '%s' dist=%.0f mana=%d",
            dot.name,
            target.name,
            dist,
            state.mana_current,
        )
        press_gem(dot.gem)
        self._cast_start = time.time()
        self._phase = _Phase.FEAR_DOT_WAIT
        if self._ctx:
            self._ctx.metrics.total_casts += 1
        return RoutineStatus.RUNNING

    def _tick_fear_dot_wait(self, state: GameState, target: SpawnData, dist: float) -> RoutineStatus:
        """FEAR_DOT_WAIT: wait for DoT on feared npc."""
        dot = get_spell_by_role(SpellRole.DOT)
        if self._read_state_fn:
            fresh = self._read_state_fn()
            if fresh.is_casting:
                if interruptible_sleep(0.1, self._flee_check):
                    log.info("[DECISION] Pull: interrupted by FLEE urgency during FEAR_DOT_WAIT polling")
                return RoutineStatus.RUNNING
        else:
            cast_elapsed = time.time() - self._cast_start
            if cast_elapsed < (dot.cast_time if dot else 1.5) + 0.3:
                if interruptible_sleep(0.3, self._flee_check):
                    log.info("[DECISION] Pull: interrupted by FLEE urgency during FEAR_DOT_WAIT timer")
                return RoutineStatus.RUNNING
        # DoT cast done (landed or fizzled)  -  either way, proceed to engaged
        if self._read_state_fn:
            new_state = self._read_state_fn()
            mana_drop = self._mana_before_cast - new_state.mana_current
            if mana_drop >= MIN_MANA_DROP:
                log.info(
                    "[CAST] Pull: Fear-DoT landed (mana %d->%d)",
                    self._mana_before_cast,
                    new_state.mana_current,
                )
                if self._ctx:
                    self._ctx.combat.last_dot_time = time.time()
            else:
                log.info(
                    "[CAST] Pull: Fear-DoT fizzle (mana %d->%d)  -  proceeding anyway",
                    self._mana_before_cast,
                    new_state.mana_current,
                )
        self._phase = _Phase.ENGAGED
        return RoutineStatus.RUNNING

    def _tick_engaged(self, state: GameState, target: SpawnData, dist: float) -> RoutineStatus:
        """ENGAGED: pull complete, transition to combat."""
        pull_elapsed = time.time() - self._pull_start
        log.info(
            "[COMBAT] Pull: ENGAGED after %.1fs (dist=%.0f, pet_engage=%.1fs, dc_retries=%d, strategy='%s')",
            pull_elapsed,
            self._pull_dist,
            self._pet_engage_time,
            self._dot_retries,
            self._strategy,
        )
        if self._ctx:
            self._ctx.combat.engaged = True
            self._ctx.player.engagement_start = time.time()
            self._ctx.combat.last_dot_time = time.time()
        return RoutineStatus.SUCCESS

    @override
    def exit(self, state: GameState) -> None:
        self._locked = False
        # Stop any active backstep -- backward key may still be held if pull
        # was interrupted mid-backstep (e.g. by FLEE or brain transition)
        if self._bs_active:
            move_backward_stop()
            self._bs_active = False
            log.info("[COMBAT] Pull: exit -- stopped active backstep")
        pull_time = time.time() - self._pull_start if self._pull_start else 0
        target_name = self._ctx.combat.pull_target_name if self._ctx and self._ctx.combat else ""
        pull_success = not self.failure_reason  # any failure_reason means pull failed
        # Stash for cycle tracker
        self.last_pull_summary = {
            "strategy": self._strategy,
            "duration": round(pull_time, 1),
            "dot_retries": self._dot_retries,
        }
        log_event(
            log,
            "pull_result",
            "[COMBAT] Pull: {} '{}' in {:.1f}s".format(
                "SUCCESS" if pull_success else "FAIL", target_name, pull_time
            ),
            **PullResultEvent(
                success=pull_success,
                target=target_name,
                strategy=self._strategy,
                duration=round(pull_time, 1),
                dist=round(self._pull_dist),
                pet_engage=round(self._pet_engage_time, 1),
                dot_retries=self._dot_retries,
                pos_x=round(state.x),
                pos_y=round(state.y),
                entity_id=self._ctx.combat.pull_target_id or 0 if self._ctx else 0,
            ),
            cycle_id=self._ctx.defeat_tracker.cycle_id if self._ctx else 0,
        )
        if self._ctx and self._ctx.diag.metrics:
            self._ctx.diag.metrics.record_action("pull", pull_success)
