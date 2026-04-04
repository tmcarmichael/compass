"""Combat routine: adaptive spell rotation based on fight conditions.

Mana-efficient by default, escalates when needed. Spells resolved by
role at runtime so the same routine adapts across all levels.

Strategy selection based on level and context:
- PET_TANK (L1-7): Pet does damage, conserve mana
- PET_AND_DOT (L8-15): Pet + DoT + conditional lifetap/DD
- FEAR_KITE (L16-48): Fear -> DoT stack -> lifetap -> re-Fear cycle
- ENDGAME (L49-60): Full DoT stack, sustained lifetap rotation
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, override

from core.constants import (
    ADDS_DETECT_RANGE,
    LOST_PULL_DISTANCE,
    LOST_PULL_HP_THRESHOLD,
    MELEE_RANGE,
    SPELL_RANGE,
)
from core.timing import interruptible_sleep, varying_sleep
from core.types import FailureCategory, Point, ReadStateFn, SpellOutcome
from core.types import SpellOutcome as _SO
from eq.loadout import SpellRole, get_spell_by_role
from eq.strings import normalize_mob_name
from motor.actions import (
    clear_target,
    face_heading,
    move_backward_start,
    move_backward_stop,
    move_forward_start,
    move_forward_stop,
    pet_attack,
    pet_back_off,
    press_gem,
    redirect_pet,
    stand,
    tab_target,
    verified_stand,
)
from nav.geometry import heading_to
from perception.combat_eval import Con, con_color
from perception.queries import is_pet, nearby_live_npcs
from perception.state import GameState, SpawnData
from routines.base import RoutineBase, RoutineStatus, make_flee_predicate
from routines.combat_monitor import CombatMonitor
from routines.combat_phases import CombatPhase, CombatPhaseManager
from routines.pet_combat import PetCombatManager
from routines.strategies.base import CastContext, CastStrategy
from routines.strategies.endgame import EndgameStrategy
from routines.strategies.fear_kite import FearKiteStrategy
from routines.strategies.pet_and_dot import PetAndDotStrategy
from routines.strategies.pet_tank import PetTankStrategy
from routines.strategies.selection import CombatStrategy, select_strategy
from util.event_schemas import FightEndEvent
from util.forensics import compact_world
from util.log_tiers import EVENT, VERBOSE
from util.structured_log import log_event


def classify_dot_fizzle(
    cast_result: str,
    fizzle_count: int,
    dist: float,
) -> str:
    """Pure function: classify a DoT cast failure and recommend action.

    Returns: "LOS_SUPPRESS", "FIZZLE_RETRY", "MUST_STAND", "INTERRUPTED_BACKSTEP",
             "SILENT_SIDESTEP", "SILENT_REFACE".
    """
    if cast_result == _SO.LOS_BLOCKED:
        return "LOS_SUPPRESS"
    if cast_result == _SO.FIZZLE:
        return "FIZZLE_RETRY"
    if cast_result == _SO.MUST_STAND:
        return "MUST_STAND"
    if cast_result == _SO.INTERRUPTED:
        return "INTERRUPTED_BACKSTEP"
    # Silent fail
    if fizzle_count >= 3 and dist < 40:
        return "SILENT_SIDESTEP"
    return "SILENT_REFACE"


if TYPE_CHECKING:
    from brain.context import AgentContext

log = logging.getLogger(__name__)

# Spells looked up by role via get_spell_by_role()  -  adapts to any level

MAX_CAST_RANGE = SPELL_RANGE

# -- Domain-local constants (only used by this module) --
LOW_HP_POLL_THRESHOLD = 0.15  # poll faster when npc HP below this (near death)
MED_SAFE_DISTANCE = 30.0  # npc must be farther than this to sit/med
MED_MOB_HP_MIN = 0.20  # npc must have more HP than this to med safely
NEARBY_NPC_RANGE = 50.0  # NPC within this range during combat med = danger
SNARE_HP_THRESHOLD = 0.20  # snare npc when HP drops below this %


class _TickState:
    """Mutable container for values shared across tick sub-methods.

    Scoped to a single tick -- never persisted between ticks.
    Replaces local variables that flow between pipeline stages.
    """

    __slots__ = ("target", "dist", "target_hp", "now", "time_in_combat")

    def __init__(self, target: SpawnData | None, dist: float, target_hp: float, now: float) -> None:
        self.target = target
        self.dist = dist
        self.target_hp = target_hp
        self.now = now
        self.time_in_combat = 0.0


# _CombatPhase moved to combat_phases.py as CombatPhase
_CombatPhase = CombatPhase


class CombatRoutine(RoutineBase):
    """Adaptive combat: conserve mana when safe, escalate when threatened."""

    def _budget_sleep(self, base: float) -> bool:
        """interruptible_sleep clamped to remaining tick budget.

        Returns True if interrupted by flee check (same as interruptible_sleep).
        If tick budget is already spent, returns False immediately (no sleep).
        """
        remaining = self._tick_deadline - time.perf_counter()
        if remaining <= 0.01:
            return False  # budget spent, skip sleep
        clamped = min(base, remaining)
        interrupted: bool = interruptible_sleep(clamped, self._flee_check)
        return interrupted

    def __init__(self, ctx: AgentContext | None = None, read_state_fn: ReadStateFn | None = None) -> None:
        self._ctx = ctx
        self._read_state_fn = read_state_fn
        self._pet_mgr = PetCombatManager(ctx=ctx, read_state_fn=read_state_fn)
        self._phase_mgr = CombatPhaseManager(self)
        self._monitor = CombatMonitor(self)
        self._cast_end_time = 0.0
        self._combat_start = 0.0
        self._last_pb_time = 0.0
        self._last_snare_time = 0.0
        self._last_pet_cmd_time = 0.0
        self._last_combat_log = 0.0
        self._last_retarget_time = 0.0
        self._has_extra_npcs = False
        # Combat medding (sit while pet tanks)
        self._medding = False
        self._med_time = 0.0  # total seconds medding this fight
        # Fight summary tracking
        self._fight_casts = 0
        self._fight_mana_start = 0
        self._fight_backsteps = 0
        self._fight_retargets = 0
        self._fight_adds_seen: set[int] = set()  # spawn_ids of extra_npcs detected
        self._fight_target_name = ""
        self._fight_target_id = 0
        self._fight_target_level = 0
        self._fight_target_x = 0.0
        self._fight_target_y = 0.0
        self._fight_cast_time = 0.0  # total seconds spent casting
        self._fight_initial_dist = 0.0
        self._target_killed = False
        self._flee_logged = False
        self._combat_recalled = False
        self._aggro_logged = False
        self._combat_fizzle_count = 0
        self._los_blocked_until = 0.0
        # Stashed fight summary for cycle tracker (populated in exit())
        self.last_fight_summary: dict = {}
        # Strategy (enum + instance + dynamic switching)
        self._strategy = CombatStrategy.PET_AND_DOT
        self._strategy_impl: CastStrategy | None = None
        self._last_strategy_eval: float = 0.0
        assert self._ctx is not None
        self._strategies = {
            CombatStrategy.PET_TANK: PetTankStrategy(self, self._ctx),
            CombatStrategy.PET_AND_DOT: PetAndDotStrategy(self, self._ctx),
            CombatStrategy.FEAR_KITE: FearKiteStrategy(self, self._ctx),
            CombatStrategy.ENDGAME: EndgameStrategy(self, self._ctx),
        }
        # Flee-aware sleep predicate (initialized in enter(), None until then)
        self._flee_check: Callable[[], bool] | None = None
        # Tick-resumable walk state (close-to-pet / PET_SAVE tank).
        # When _walk_target_id != 0, _tick_walk_progress() polls each tick
        # instead of blocking in a while-loop.
        self._walk_target_id: int = 0  # spawn_id we're walking toward (0 = not walking)
        self._walk_close_dist: float = 0.0  # stop when within this distance
        self._walk_deadline: float = 0.0  # give up after this timestamp
        self._walk_settle_until: float = 0.0  # >0 = settle deadline timestamp
        self._walk_settle_duration: float = 0.0  # seconds to settle after arrival
        self._walk_update_enter_pos: bool = False  # update _enter_pos after settle
        # Pending cast verification: track mana before DoT cast to confirm
        # the spell actually landed (mana dropped) before updating last_dot_time.
        # This prevents fizzled/interrupted casts from resetting the recast timer.
        self._pending_dot_cast: tuple[float, int] | None = None  # (cast_time, pre_cast_mana)
        # After a fizzle, skip medding for 1 tick so the rotation can retry
        # the cast immediately instead of sitting down first.
        self._retry_after_fizzle: bool = False
        # Non-blocking phase state delegated to CombatPhaseManager.
        # Properties below proxy to self._phase_mgr for backward compat.

    @property
    def _nb_phase(self) -> str:
        phase: str = self._phase_mgr.phase
        return phase

    @_nb_phase.setter
    def _nb_phase(self, value: str) -> None:
        self._phase_mgr.phase = value

    @property
    def _nb_deadline(self) -> float:
        deadline: float = self._phase_mgr.deadline
        return deadline

    @_nb_deadline.setter
    def _nb_deadline(self, value: float) -> None:
        self._phase_mgr.deadline = value

    @property
    def _nb_data(self) -> dict:
        data: dict[str, object] = self._phase_mgr.data
        return data

    @_nb_data.setter
    def _nb_data(self, value: dict) -> None:
        self._phase_mgr.data = value

    @override
    def enter(self, state: GameState) -> None:
        # Auto-engage: consume candidate from pre-rule scan_auto_engage().
        # begin_engagement() was previously called inside _should_combat's
        # condition function, causing ghost-state when a higher-priority
        # rule won. Now it only fires when IN_COMBAT actually activates.
        if self._ctx and not self._ctx.combat.engaged and self._ctx.combat.auto_engage_candidate is not None:
            sid, name, x, y, level = self._ctx.combat.auto_engage_candidate
            log.info("[COMBAT] Auto-engage: beginning engagement with '%s' id=%d lv%d", name, sid, level)
            self._ctx.begin_engagement(sid, name=name, x=x, y=y, level=level)
            self._ctx.combat.auto_engaged = True
            self._ctx.combat.auto_engage_candidate = None

        target = state.target
        dist = state.pos.dist_to(target.pos) if target else 0

        # Select strategy based on level + context (con, danger, fear)
        mob_con = con_color(state.level, target.level) if target else None
        mob_danger = None
        mob_pet_death_rate = None
        fh = getattr(self._ctx, "fight_history", None) if self._ctx else None
        if fh and target:
            mob_danger = fh.learned_danger(target.name)
            mob_pet_death_rate = fh.learned_pet_death_rate(target.name)
        fear_spell = get_spell_by_role(SpellRole.FEAR)
        self._strategy = select_strategy(
            state.level,
            con=mob_con,
            danger=mob_danger,
            has_fear=bool(fear_spell),
            pet_death_rate=mob_pet_death_rate,
        )
        log.info(
            "[COMBAT] Combat: strategy=%s (con=%s, danger=%s, pet_death=%.0f%%)",
            self._strategy.value,
            mob_con if mob_con else "?",
            f"{mob_danger:.2f}" if mob_danger is not None else "none",
            (mob_pet_death_rate or 0) * 100,
        )
        self._strategy_impl = self._strategies.get(self._strategy)
        if self._strategy_impl:
            self._strategy_impl.reset()

        log.log(
            EVENT,
            "[COMBAT] Combat: START target='%s' id=%d dist=%.0f "
            "target_HP=%d/%d target_pos=(%.0f,%.0f) "
            "player_pos=(%.0f,%.0f) HP=%.0f%% Mana=%d",
            target.name if target else "none",
            target.spawn_id if target else 0,
            dist,
            target.hp_current if target else 0,
            target.hp_max if target else 0,
            target.x if target else 0,
            target.y if target else 0,
            state.x,
            state.y,
            state.hp_pct * 100,
            state.mana_current,
        )

        # Must be standing to fight. Use verified_stand() when possible so we
        # confirm the toggle landed before the first tick fires face_heading.
        # This is critical when entering combat from a sitting routine (e.g.
        # MEMORIZE_SPELLS) -- the character animation takes ~1s and heading
        # remains locked until actually standing.
        if state.is_sitting:
            log.info("[COMBAT] Combat: enter -- player sitting, standing up")
            if self._read_state_fn:
                verified_stand(self._read_state_fn)
            else:
                stand()
                if interruptible_sleep(0.5, self._flee_check):
                    log.info("[COMBAT] Combat: enter -- interrupted by FLEE urgency during stand")

        self._pending_dot_cast = None  # clear stale data from previous fight
        self._target_killed = False  # must reset per-fight (used in exit() for defeat bookkeeping)
        if self._ctx:
            self._ctx.combat.last_mob_hp_pct = 1.0  # reset for new fight
        self._flee_logged = False
        self._aggro_logged = False
        self._phase_mgr.reset()  # reset non-blocking phase on combat start
        self._combat_recalled = False  # pet recall used once per combat
        self._last_strategy_eval = 0.0  # reset strategy switch timer
        self._combat_fizzle_count = 0
        self._los_blocked_until = 0.0
        self._los_fail_x = 0.0
        self._los_fail_y = 0.0
        self._retry_after_fizzle = False
        self._combat_start = time.time()
        self._enter_pos = (state.x, state.y)  # track drift from combat start
        # Safety: stop any held movement keys from prior routine (pull backstep)
        move_backward_stop()
        move_forward_stop()
        self._walk_target_id = 0  # clear stale walk state from prior fight
        # Build flee predicate for interruptible sleeps
        if self._read_state_fn and self._ctx:
            self._flee_check = make_flee_predicate(self._read_state_fn, self._ctx)
        else:
            self._flee_check = None
        self._last_pb_time = 0.0
        self._last_snare_time = 0.0
        self._last_pet_cmd_time = 0.0
        self._last_combat_log = 0.0
        self._last_backstep_time = 0.0
        self._last_retarget_time = 0.0
        self._combat_backstep_count = 0  # total backsteps this fight (capped at 2)
        self._pet_save_engaged = False  # PET_SAVE melee: once per fight only
        self._backstep_active = False
        self._backstep_start_x = 0.0
        self._backstep_start_y = 0.0
        self._backstep_target = 0.0
        self._backstep_deadline = 0.0
        self._hp_at_start = state.hp_pct
        self._has_extra_npcs = False
        self._fight_casts = 0
        self._fight_mana_start = state.mana_current
        self._xp_at_start = self._ctx.defeat_tracker.xp_gains if self._ctx else 0
        self._fight_backsteps = 0
        self._fight_retargets = 0
        self._fight_adds_seen = set()
        self._fight_target_name = target.name if target else "?"
        self._fight_target_id = target.spawn_id if target else 0
        self._fight_target_level = target.level if target else 0
        self._fight_target_x = target.x if target else state.x
        self._fight_target_y = target.y if target else state.y
        # Store on ctx so brain can record defeat if target despawns between ticks
        if self._ctx and target:
            self._ctx.defeat_tracker.last_fight_name = target.name
            self._ctx.defeat_tracker.last_fight_id = target.spawn_id
            self._ctx.defeat_tracker.last_fight_x = target.x
            self._ctx.defeat_tracker.last_fight_y = target.y
            self._ctx.defeat_tracker.last_fight_level = target.level
        self._fight_cast_time = 0.0
        self._fight_initial_dist = state.pos.dist_to(target.pos) if target else 0

        self._medding = False
        self._med_time = 0.0
        self._med_start = 0.0
        self._pet_mgr.reset()

        # Store active strategy on ctx for brain diagnostics
        if self._ctx:
            self._ctx.combat.active_strategy = self._strategy.value

        # Pet should already be attacking from pull -- re-send if
        # entering combat via auto-engage or without a prior pull
        if self._ctx and (self._ctx.combat.auto_engaged or not self._ctx.combat.pull_target_id):
            if self._ctx.pet.alive:
                log.info("[COMBAT] Combat: sending pet (auto-engage or no prior pull)")
                pet_attack()
                if interruptible_sleep(0.3, self._flee_check):
                    log.info("[COMBAT] Combat: enter -- interrupted by FLEE urgency during pet send")
            else:
                log.info("[COMBAT] Combat: no pet alive -- skipping pet_attack")
            self._ctx.combat.auto_engaged = False
        self._last_pet_cmd_time = time.time()

    def _find_melee_attacker(self, state: GameState, current_target: SpawnData | None) -> SpawnData | None:
        """Find a non-target NPC in melee range (attacking player)."""
        exclude_id = current_target.spawn_id if current_target else 0
        for spawn, _dist in nearby_live_npcs(state, state.pos, MELEE_RANGE):
            if exclude_id and spawn.spawn_id == exclude_id:
                continue
            return spawn
        return None

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

    def _detect_adds(self, state: GameState) -> list[tuple[SpawnData, float, str]]:
        """Check for hostile NPCs near player or pet (not our target).

        Returns list of (spawn, dist, near_who) for each add detected.
        """
        target_id = state.target.spawn_id if state.target else 0
        adds = []
        # Near player
        for spawn, d in nearby_live_npcs(state, state.pos, ADDS_DETECT_RANGE):
            if spawn.spawn_id != target_id:
                adds.append((spawn, d, "player"))
        # Near pet (if pet is away from player)
        if self._ctx and self._ctx.pet.alive and self._ctx.pet.spawn_id:
            pet_x, pet_y = state.x, state.y
            for sp in state.spawns:
                if sp.spawn_id == self._ctx.pet.spawn_id:
                    pet_x, pet_y = sp.x, sp.y
                    break
            pet_pos = Point(pet_x, pet_y, 0.0)
            pet_far = state.pos.dist_to(pet_pos) > 30
            if pet_far:
                for spawn, d in nearby_live_npcs(state, pet_pos, ADDS_DETECT_RANGE):
                    if spawn.spawn_id != target_id:
                        # Don't double-count if already near player
                        if not any(a[0].spawn_id == spawn.spawn_id for a in adds):
                            adds.append((spawn, d, "pet"))
        return adds

    def _target_hp_pct(self, target: SpawnData) -> float:
        """Get target HP as a percentage."""
        if target.hp_max <= 0:
            return 1.0
        return float(target.hp_current / target.hp_max)

    def _stand_from_med(self) -> None:
        """Stand up from combat medding. Must call before casting or moving.

        Uses verified_stand for memory confirmation, then waits for the
        EQ stand animation (~1s) so heading unlocks before face_heading.
        """
        if not self._medding:
            return
        if self._read_state_fn:
            ok = verified_stand(self._read_state_fn)
            if not ok:
                stand()
                interruptible_sleep(0.8, self._flee_check)
                log.warning("[COMBAT] Combat: stand verification failed, forced retry")
        else:
            stand()
        self._medding = False
        if self._med_start > 0:
            self._med_time += time.time() - self._med_start
            self._med_start = 0.0
        # Wait for stand animation (heading locked until complete)
        interruptible_sleep(0.5, self._flee_check)
        import motor.actions as _ma

        _ma.force_standing()
        if self._read_state_fn:
            s = self._read_state_fn()
            if s.is_sitting:
                log.warning("[COMBAT] Combat: still sitting after stand -- forcing")
                stand()
                interruptible_sleep(0.8, self._flee_check)
                _ma.force_standing()
            log.info("[COMBAT] Combat: stood up from med (stand_state=%d)", s.stand_state)
        else:
            log.info("[COMBAT] Combat: stood up from med")

    def _face_target(self, state: GameState, target: SpawnData | None) -> None:
        """Face the target before casting. EQ requires facing to cast.

        No-op while a spell cast is active -- any turn key cancels casting in EQ.
        """
        if not target or not self._read_state_fn:
            return
        if state.is_casting:
            log.log(VERBOSE, "[COMBAT] Combat: _face_target skipped -- casting in progress")
            return
        exact = heading_to(state.pos, target.pos)
        desired = (exact + random.gauss(0, 2.0)) % 512.0
        current = state.heading
        diff = abs((desired - current + 256) % 512 - 256)
        if diff > 10:  # only turn if significantly off
            rsf = self._read_state_fn
            assert rsf is not None
            face_heading(desired, lambda: rsf().heading, tolerance=8.0)

    def _record_kill(self, target: SpawnData | None, fight_time: float) -> None:
        """Record a defeat in context and spatial memory."""
        if not self._ctx:
            return
        self._ctx.combat.engaged = False
        self._ctx.combat.pull_target_id = None
        defeat_name = target.name if target else self._fight_target_name
        defeat_id = target.spawn_id if target else self._fight_target_id
        defeat_pos = target.pos if target else Point(self._fight_target_x, self._fight_target_y, 0.0)
        defeat_level = target.level if target else self._fight_target_level
        if defeat_name:
            self._ctx.record_kill(defeat_id, name=defeat_name, pos=defeat_pos)
            if self._ctx.spatial_memory:
                self._ctx.spatial_memory.record_kill(
                    defeat_pos, defeat_name, defeat_level, fight_seconds=fight_time
                )

    def _cast_spell(
        self,
        gem: int,
        cast_time: float,
        now: float,
        state: GameState,
        target: SpawnData | None,
        *,
        is_dot: bool = False,
    ) -> None:
        """Stand, face, cast a spell, and update tracking.

        Args:
            is_dot: If True, record pre-cast mana for DoT landing verification.
                    The actual last_dot_time will only be set when the next tick
                    confirms mana dropped (cast succeeded), not on cast attempt.
        """
        # Skip if still at same position as last LOS failure
        if target and hasattr(self, "_los_fail_x") and now < self._los_blocked_until:
            moved = state.pos.dist_to(Point(self._los_fail_x, self._los_fail_y, 0.0))
            if moved < 10:
                log.log(
                    VERBOSE, "[CAST] Combat: skipping cast -- still at LOS fail position (moved %.0fu)", moved
                )
                return

        # Terrain LOS pre-check -- if blocked, suppress cast
        if target:
            from nav.movement import check_spell_los

            if not check_spell_los(state.x, state.y, state.z, target.x, target.y, target.z):
                log.info(
                    "[CAST] Combat: cast suppressed -- terrain LOS blocked to '%s' "
                    "(dist=%.0f dz=%.0f) -- waiting for pet to bring closer",
                    target.name,
                    state.pos.dist_to(target.pos),
                    target.z - state.z,
                )
                self._los_blocked_until = now + 5.0
                return

        self._stand_from_med()
        # Verify standing via internal flag
        import motor.actions as _ma

        if _ma.is_sitting():
            log.warning("[CAST] Combat: still sitting after _stand_from_med -- forcing stand")
            stand()
            if self._budget_sleep(0.3):
                log.info("[CAST] Combat: interrupted by FLEE urgency during forced stand in cast")
                return
        self._face_target(state, target)
        # settle before cast -- EQ needs stillness. Must NOT be
        # budget-clamped: game requires ~0.3s stillness before cast lands.
        if interruptible_sleep(random.uniform(0.3, 0.5), self._flee_check):
            log.info("[CAST] Combat: interrupted by FLEE urgency during pre-cast settle")
            return
        if is_dot:
            self._pending_dot_cast = (now, state.mana_current)
        press_gem(gem)
        self._cast_end_time = now + cast_time + 0.2
        self._fight_cast_time += cast_time + 0.2
        if self._ctx:
            self._ctx.metrics.total_casts += 1
        self._fight_casts += 1

    def _get_pet_status(self, state: GameState) -> tuple[float, float]:
        """Returns (pet_hp_pct, pet_dist) from spawn list."""
        result: tuple[float, float] = self._pet_mgr.get_pet_status(state)
        return result

    # -- Strategy delegation via self._strategy_impl.execute() ---------------
    # Spell rotation logic in strategies/*.py (pet_tank, pet_and_dot,
    # fear_kite, endgame). tick() builds CastContext and delegates.

    # -- Phase methods moved to combat_phases.py (CombatPhaseManager) --------
    # _tick_phase, _cleanup_phase, _tick_los_recall, _tick_los_walk,
    # _tick_pet_recall, _tick_backstep are now in self._phase_mgr

    # -- Dynamic strategy switching -------------------------------------------

    _STRATEGY_EVAL_INTERVAL = 10.0  # seconds between switch evaluations

    def _evaluate_strategy_switch(self, state: GameState, now: float) -> None:
        """Check if current strategy should switch mid-fight.

        Delegates decision logic to the pure function evaluate_mid_fight_switch()
        in routines.strategies.selection for testability.
        """
        if now - self._last_strategy_eval < self._STRATEGY_EVAL_INTERVAL:
            return
        self._last_strategy_eval = now

        if not self._ctx or not self._strategy_impl:
            return

        time_in_combat = now - self._combat_start
        target = state.target
        if not target:
            return

        pet_hp, _ = self._get_pet_status(state)
        fear = get_spell_by_role(SpellRole.FEAR)
        learned_dur = None
        pet_death_rate = None
        fight_count = 0
        if self._ctx.fight_history:
            learned_dur = self._ctx.fight_history.learned_duration(target.name)
            stats = self._ctx.fight_history.get_stats(target.name)
            if stats:
                pet_death_rate = stats.pet_death_rate
                fight_count = stats.fights

        from routines.strategies.selection import evaluate_mid_fight_switch

        result = evaluate_mid_fight_switch(
            current=self._strategy,
            time_in_combat=time_in_combat,
            pet_hp=pet_hp,
            pet_alive=self._ctx.pet.alive,
            fear_available=fear is not None,
            fear_mana_cost=fear.mana_cost if fear else 0,
            player_mana=state.mana_current,
            player_level=state.level,
            learned_duration=learned_dur,
            fear_phase=getattr(self._strategy_impl, "_fear_phase", None),
            pet_death_rate=pet_death_rate,
            fight_count=fight_count,
        )
        if result is not None:
            new_strat, reason = result
            self._switch_strategy(new_strat, reason)

    def _switch_strategy(self, new: CombatStrategy, reason: str) -> None:
        """Hot-swap combat strategy mid-fight."""
        old = self._strategy
        if new == old:
            return
        self._strategy = new
        self._strategy_impl = self._strategies.get(new)
        if self._strategy_impl:
            self._strategy_impl.reset()
        log.warning("[COMBAT] STRATEGY SWITCH: %s -> %s (%s)", old.value, new.value, reason)
        if self._ctx:
            self._ctx.combat.active_strategy = new.value
            self._ctx.combat.strategy_switches = getattr(self._ctx.combat, "strategy_switches", 0) + 1

    # -- Main tick -----------------------------------------------------------

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        now = time.time()

        # Non-blocking phase handler (LOS correction, pet recall, backstep)
        if self._phase_mgr.active:
            return self._phase_mgr.tick(state, now)

        self._evaluate_strategy_switch(state, now)

        # Build tick-local shared state
        ts = _TickState(
            target=state.target,
            dist=0.0,
            target_hp=1.0,
            now=now,
        )

        # Guard: if we're targeting our own pet, disengage -- BUT skip
        # this check if pet-heal is actively in progress (pet_target is
        # deliberately sent before Mend Bones, target returns to npc after).
        _healing_pet = (
            self._pet_mgr._cast_phase is not None or time.time() - self._pet_mgr._last_heal_time < 3.0
        )
        if (
            ts.target is not None
            and self._ctx
            and self._ctx.pet.spawn_id
            and ts.target.spawn_id == self._ctx.pet.spawn_id
            and not _healing_pet
        ):
            log.warning("[COMBAT] Combat: TARGETING OWN PET '%s' -- disengaging", ts.target.name)
            from motor.actions import clear_target

            clear_target()
            if self._ctx:
                self._ctx.combat.engaged = False
                self._ctx.combat.pull_target_id = None
            return RoutineStatus.FAILURE

        # Stale fight detection: npc at 100% HP, far away, nobody fighting it
        # This happens when pet heal retargets pet, npc resets and walks away.
        # Must compute dist here since ts.dist isn't populated yet.
        if ts.target is not None and ts.time_in_combat > 15.0:
            _stale_dist = state.pos.dist_to(ts.target.pos)
            _stale_hp = ts.target.hp_current / max(ts.target.hp_max, 1)
            if _stale_dist > 80 and _stale_hp >= 0.99:
                log.warning(
                    "[COMBAT] Combat: STALE FIGHT -- npc '%s' at %.0fu HP=100%% after %.0fs -- disengaging",
                    ts.target.name,
                    _stale_dist,
                    ts.time_in_combat,
                )
                if self._ctx:
                    self._ctx.combat.engaged = False
                    self._ctx.combat.pull_target_id = None
                return RoutineStatus.FAILURE

        # Pipeline: each step returns RoutineStatus (early exit) or None (continue).
        # Critical stages (cast poll, DoT verify, death check, distance update)
        # always run. Deferrable stages (face/melee, adds, pet heal, strategy,
        # medding) are skipped if the tick budget is exceeded -- they'll run
        # next tick. This caps combat tick time near the 200ms budget.
        _critical = (
            self._tick_casting_poll,
            self._tick_dot_verify,
            self._tick_retarget,
            self._monitor.tick_death_check,
            self._monitor.tick_distance_update,
            self._monitor.tick_pet_heal,  # pet survival IS critical
            self._tick_strategy,  # casting IS core combat
        )
        _deferrable = (
            self._tick_face_and_melee,
            self._monitor.tick_combat_log_and_adds,
        )
        for step in _critical:
            result = step(state, ts)
            if result is not None:
                return result
        for step in _deferrable:
            if time.perf_counter() > self._tick_deadline:
                return RoutineStatus.RUNNING  # budget spent, defer remaining
            result = step(state, ts)
            if result is not None:
                return result

        if time.perf_counter() > self._tick_deadline:
            return RoutineStatus.RUNNING
        return self._monitor.tick_medding(state, ts.now, ts.target, ts.dist, ts.target_hp)

    # -- Tick pipeline stages --------------------------------------------------

    def _tick_casting_poll(self, state: GameState, ts: _TickState) -> RoutineStatus | None:
        """Poll casting_mode for cast completion. Cancel if target died."""
        now = ts.now
        if now < self._cast_end_time:
            if self._read_state_fn:
                fresh = self._read_state_fn()
                if fresh.is_casting:
                    if fresh.target and fresh.target.hp_current <= 0:
                        from motor.actions import stop_cast

                        log.info("[COMBAT] Combat: target dead mid-cast  -  cancelling to save mana")
                        stop_cast()
                        self._cast_end_time = 0.0
                        if self._budget_sleep(0.15):
                            log.info(
                                "[COMBAT] Combat: interrupted by FLEE urgency after cast cancel (target dead)"
                            )
                            return RoutineStatus.RUNNING
                    elif not fresh.target:
                        from motor.actions import stop_cast

                        log.info("[COMBAT] Combat: target gone mid-cast  -  cancelling")
                        stop_cast()
                        self._cast_end_time = 0.0
                        if self._budget_sleep(0.15):
                            log.info(
                                "[COMBAT] Combat: interrupted by FLEE urgency after cast cancel (target gone)"
                            )
                            return RoutineStatus.RUNNING
                    else:
                        self._budget_sleep(0.1)
                        return RoutineStatus.RUNNING
                saved = self._cast_end_time - now
                if self._cast_end_time > 0:
                    log.debug("[CAST] Combat: cast complete (casting_mode=0, %.1fs early)", saved)
                self._cast_end_time = 0.0
                if saved > 0.15:
                    self._budget_sleep(random.uniform(0.1, 0.25))
            else:
                self._budget_sleep(0.3)
                return RoutineStatus.RUNNING
        return None

    def _tick_dot_verify(self, state: GameState, ts: _TickState) -> RoutineStatus | None:
        """Verify pending DoT cast landed (mana dropped). Handle fizzles."""
        now = ts.now
        if self._pending_dot_cast is None:
            return None
        cast_time_stamp, pre_mana = self._pending_dot_cast
        if now - cast_time_stamp < 2.5:
            return RoutineStatus.RUNNING
        self._pending_dot_cast = None
        mana_drop = pre_mana - state.mana_current
        if mana_drop >= 3:
            if self._ctx:
                self._ctx.combat.last_dot_time = cast_time_stamp
            log.info(
                "[CAST] Combat: DoT CONFIRMED (mana %d->%d, drop=%d)", pre_mana, state.mana_current, mana_drop
            )
            return None

        # Fizzle handling -- delegate classification to pure function
        self._combat_fizzle_count += 1
        target = state.target
        dist = state.pos.dist_to(target.pos) if target else 0
        cast_result = ""
        if self._ctx:
            cast_result = self._ctx.combat.last_cast_result
            self._ctx.combat.last_cast_result = SpellOutcome.NONE

        action = classify_dot_fizzle(cast_result, self._combat_fizzle_count, dist)

        if action == "LOS_SUPPRESS":
            log.warning(
                "[CAST] Combat: LOS BLOCKED at dist=%.0f -- suppressing 8s | %s", dist, self._vitals(state)
            )
            self._los_blocked_until = now + 8.0
            self._los_fail_x = state.x
            self._los_fail_y = state.y
        elif action == "FIZZLE_RETRY":
            log.info(
                "[CAST] Combat: FIZZLE #%d at dist=%.0f -- retry | %s",
                self._combat_fizzle_count,
                dist,
                self._vitals(state),
            )
        elif action == "MUST_STAND":
            log.warning("[CAST] Combat: MUST STAND at dist=%.0f -- forcing | %s", dist, self._vitals(state))
            import motor.actions as _ma

            _ma.force_standing()
            stand()
            self._budget_sleep(0.3)
        elif action == "INTERRUPTED_BACKSTEP":
            log.warning(
                "[CAST] Combat: INTERRUPTED #%d at dist=%.0f -- backstep | %s",
                self._combat_fizzle_count,
                dist,
                self._vitals(state),
            )
            if dist < 25:
                self._stand_from_med()
                move_backward_start()
                self._budget_sleep(random.uniform(0.5, 1.0))
                move_backward_stop()
        elif action == "SILENT_SIDESTEP":
            log.info(
                "[CAST] Combat: %d silent fails close range -- sidestep for LOS", self._combat_fizzle_count
            )
            self._stand_from_med()
            direction = random.choice([-1, 1])
            from motor.actions import _action_down, _action_up

            strafe = "strafe_left" if direction < 0 else "strafe_right"
            _action_down(strafe)
            interruptible_sleep(random.uniform(0.5, 0.8), self._flee_check)
            _action_up(strafe)
        else:  # SILENT_REFACE
            log.warning(
                "[CAST] Combat: SILENT FAIL #%d at dist=%.0f -- reface | %s",
                self._combat_fizzle_count,
                dist,
                self._vitals(state),
            )
            self._stand_from_med()
            self._face_target(state, target)

        dot_spell = get_spell_by_role(SpellRole.DOT)
        if self._ctx and dot_spell:
            self._ctx.combat.clear_spell_cast(dot_spell.spell_id)
        self._retry_after_fizzle = True
        log.info(
            "[CAST] Combat: DoT FIZZLE (mana %d->%d, drop=%d)  -  "
            "cleared cast record, will retry next tick (fizzles=%d)",
            pre_mana,
            state.mana_current,
            mana_drop,
            self._combat_fizzle_count,
        )
        return None

    def _tick_retarget(self, state: GameState, ts: _TickState) -> RoutineStatus | None:
        """Re-acquire target after pet heal cleared it."""
        ts.target = state.target
        if not ts.target and self._pet_mgr.heals > 0 and self._ctx and self._ctx.combat.pull_target_id:
            log.info("[COMBAT] Combat: retargeting npc after pet heal")
            for _attempt in range(2):
                tab_target()
                if self._budget_sleep(0.3):
                    log.info("[COMBAT] Combat: interrupted by FLEE during retarget")
                    return RoutineStatus.RUNNING
                if self._read_state_fn:
                    fresh = self._read_state_fn()
                    ts.target = fresh.target
                    # Skip own pet -- tab again
                    if ts.target and is_pet(ts.target):
                        log.log(VERBOSE, "[COMBAT] Combat: retarget skipping own pet")
                        ts.target = None
                        continue
                if ts.target:
                    break
        return None

    # _tick_death_check -> self._monitor.tick_death_check
    # _tick_distance_update -> self._monitor.tick_distance_update

    def _tick_face_and_melee(
        self,
        state: GameState,
        ts: _TickState,
    ) -> RoutineStatus | None:
        """Face target, handle melee retarget and reactive backstep."""
        target = ts.target
        if target is None:
            return None

        # Non-blocking walk in progress (close-to-pet / PET_SAVE tank)
        result = self._tick_walk_progress(state)
        if result is not None:
            return result

        # PET_SAVE emergency: stand from med if pet is dying
        self._check_pet_save_sitting(state)

        # Guard: no movement while casting (interrupts spells)
        if state.is_casting:
            return RoutineStatus.RUNNING

        # Guard: heading locked while sitting (keys have no effect)
        if state.is_sitting:
            if not getattr(self, "_sitting_logged", False):
                log.log(VERBOSE, "[COMBAT] Combat: _tick_face_and_melee skipped -- still sitting")
                self._sitting_logged = True
            return RoutineStatus.RUNNING
        self._sitting_logged = False

        # Close-to-pet: walk toward pet if >80u away
        result = self._tick_close_to_pet(state)
        if result is not None:
            return result

        # Melee retarget: add adoption + attacker switch
        melee_attacker = self._find_melee_attacker(state, target)
        has_confirmed_add = self._ctx is not None and self._ctx.pet.has_add
        result = self._tick_melee_retarget(state, ts, melee_attacker, has_confirmed_add)
        if result is not None:
            return result

        # Active backstep completion
        result = self._tick_backstep_completion(state, ts)
        if result is not None:
            return result

        # Reactive backstep + PET_SAVE tank
        return self._tick_reactive_backstep(state, ts, melee_attacker, has_confirmed_add)

    def _check_pet_save_sitting(self, state: GameState) -> None:
        """Stand from med if pet HP drops below 35% while sitting."""
        if not (state.is_sitting and self._ctx and self._ctx.pet.alive):
            return
        pet_hp = -1.0
        if self._ctx.pet.spawn_id:
            for sp in state.spawns:
                if sp.spawn_id == self._ctx.pet.spawn_id:
                    pet_hp = sp.hp_current / max(sp.hp_max, 1)
                    break
        if (
            pet_hp >= 0
            and pet_hp < 0.35
            and state.hp_pct > 0.80
            and not getattr(self, "_pet_save_engaged", False)
            and not self._has_extra_npcs
        ):
            log.info("[COMBAT] Combat: PET_SAVE emergency -- standing from med (pet=%.0f%%)", pet_hp * 100)
            self._stand_from_med()

    def _tick_close_to_pet(self, state: GameState) -> RoutineStatus | None:
        """Walk toward pet if >80u away during combat."""
        if not (
            self._ctx
            and self._ctx.pet.spawn_id
            and not self._medding
            and not self._backstep_active
            and not state.is_casting
            and not self._ctx.pet.has_add
        ):
            return None
        for sp in state.spawns:
            if sp.spawn_id == self._ctx.pet.spawn_id:
                pet_dist = state.pos.dist_to(sp.pos)
                if pet_dist > 80:
                    target_dist = random.uniform(40, 60)
                    walk_dist = pet_dist - target_dist
                    rsf = self._read_state_fn
                    if walk_dist > 15 and rsf is not None:
                        log.info(
                            "[COMBAT] Combat: closing to pet (pet_dist=%.0f, target_dist=%.0f)",
                            pet_dist,
                            target_dist,
                        )
                        exact = heading_to(state.pos, sp.pos)
                        _bound_rsf: ReadStateFn = rsf

                        def _get_heading(_reader: ReadStateFn = _bound_rsf) -> float:
                            return float(_reader().heading)

                        face_heading(exact, _get_heading, tolerance=10.0)
                        self._start_walk_toward(
                            sp.spawn_id, target_dist + 10, settle=0.5, update_enter_pos=True
                        )
                        return RoutineStatus.RUNNING
                break
        return None

    def _tick_melee_retarget(
        self,
        state: GameState,
        ts: _TickState,
        melee_attacker: SpawnData | None,
        has_confirmed_add: bool,
    ) -> RoutineStatus | None:
        """Handle add adoption and melee attacker retarget."""
        target = ts.target
        if target is None:
            return None
        now = ts.now
        retarget_cooldown = 1.0 if self._has_extra_npcs else 3.0

        # Fast path: target already IS the add (tab cycling landed on it).
        pull_id = self._ctx.combat.pull_target_id if self._ctx else 0
        if (
            has_confirmed_add
            and target.is_npc
            and target.hp_current > 0
            and target.spawn_id != (pull_id or 0)
            and ts.dist < 25
            and now - self._last_retarget_time > retarget_cooldown
        ):
            self._stand_from_med()
            if self._ctx:
                log.info(
                    "[COMBAT] Combat: adopting add '%s' id=%d as primary target (original target gone)",
                    target.name,
                    target.spawn_id,
                )
                self._ctx.combat.pull_target_id = target.spawn_id
                self._ctx.combat.pull_target_name = target.name
                self._ctx.pet.has_add = False
                self._fight_target_name = target.name
                self._fight_target_id = target.spawn_id
            log.debug("[COMBAT] redirect_pet -> '%s' id=%d dist=%.0f", target.name, target.spawn_id, ts.dist)
            redirect_pet()
            self._last_retarget_time = now
            self._last_pet_cmd_time = now
            self._hp_at_start = state.hp_pct
            self._fight_retargets += 1
            if self._combat_backstep_count < 2:
                self._start_backstep(state, now, random.uniform(20, 28))
                log.info("[COMBAT] Combat: BACKSTEP after add adopt  -  breaking melee (dist=%.0f)", ts.dist)
            return RoutineStatus.RUNNING

        # Melee attacker retarget
        if (
            melee_attacker
            and (state.hp_pct < self._hp_at_start or has_confirmed_add)
            and now - self._last_retarget_time > retarget_cooldown
        ):
            self._stand_from_med()
            attacker_dist = state.pos.dist_to(melee_attacker.pos)
            delay = random.uniform(0.2, 0.5)
            log.info(
                "[COMBAT] Combat: RETARGET  -  '%s' id=%d in melee (dist=%.0f), "
                "switching from '%s', reaction %.1fs",
                melee_attacker.name,
                melee_attacker.spawn_id,
                attacker_dist,
                target.name,
                delay,
            )
            if self._budget_sleep(delay):
                log.info("[COMBAT] Combat: interrupted by FLEE urgency during melee retarget reaction")
                return RoutineStatus.RUNNING
            tab_target()
            if self._budget_sleep(0.3):
                log.info("[COMBAT] Combat: interrupted by FLEE urgency during melee retarget tab")
                return RoutineStatus.RUNNING
            if self._read_state_fn:
                _rt = self._read_state_fn()
                if _rt.target and is_pet(_rt.target):
                    log.log(VERBOSE, "[COMBAT] Combat: melee retarget landed on pet, clearing")
                    clear_target()
                    return RoutineStatus.RUNNING
            if self._read_state_fn:
                _mrt = self._read_state_fn()
                log.debug(
                    "[COMBAT] melee redirect_pet -> '%s' id=%d",
                    _mrt.target.name if _mrt.target else "?",
                    _mrt.target.spawn_id if _mrt.target else 0,
                )
            redirect_pet()
            self._last_retarget_time = now
            self._last_pet_cmd_time = now
            self._hp_at_start = state.hp_pct
            self._fight_retargets += 1
            if self._combat_backstep_count < 2:
                self._start_backstep(state, now, random.uniform(20, 28))
                log.info(
                    "[COMBAT] Combat: BACKSTEP after retarget  -  breaking melee (dist=%.0f)",
                    attacker_dist,
                )
            return RoutineStatus.RUNNING
        return None

    def _tick_backstep_completion(self, state: GameState, ts: _TickState) -> RoutineStatus | None:
        """Check active backstep progress and complete or extend."""
        if not self._backstep_active:
            return None
        now = ts.now
        moved = Point(self._backstep_start_x, self._backstep_start_y, 0.0).dist_to(state.pos)
        if moved >= self._backstep_target or now > self._backstep_deadline:
            move_backward_stop()
            varying_sleep(0.05, sigma=0.1)
            move_backward_stop()
            if ts.dist < 15 and now <= self._backstep_deadline + 1.0:
                log.info("[COMBAT] Combat: backstep ended at %.0fu -- still in melee, extending", ts.dist)
                self._backstep_deadline = now + 1.5
                move_backward_start()
                return RoutineStatus.RUNNING
            self._backstep_active = False
            self._last_backstep_time = now
            self._hp_at_start = state.hp_pct
            self._fight_backsteps += 1
            self._combat_backstep_count += 1
            log.info("[COMBAT] Combat: backstep complete (moved %.0fu)", moved)
        return RoutineStatus.RUNNING

    def _tick_reactive_backstep(
        self,
        state: GameState,
        ts: _TickState,
        melee_attacker: SpawnData | None,
        has_confirmed_add: bool,
    ) -> RoutineStatus | None:
        """PET_SAVE tank closing + reactive backstep when hit in melee."""
        target = ts.target
        if target is None:
            return None
        now = ts.now

        target_melee = ts.dist < 20.0
        add_melee = melee_attacker is not None and (state.hp_pct < self._hp_at_start or has_confirmed_add)
        pet_dist, pet_hp_pct = self._get_pet_combat_status(state)

        # PET_SAVE melee tanking: close to melee once to share threat
        result = self._tick_pet_save_tank(state, ts, target, pet_hp_pct)
        if result is not None:
            return result

        _backstep_limit = 2
        # Suppress backstep when PET_SAVE is active and player HP is high
        _pet_save_suppress = (
            self._pet_save_engaged and state.hp_pct > 0.50 and pet_hp_pct >= 0 and pet_hp_pct < 0.50
        )
        if not (
            (target_melee or add_melee) and now - self._last_backstep_time > 5.0 and not _pet_save_suppress
        ):
            return None

        self._stand_from_med()
        if pet_dist > 30.0:
            if self._ctx and self._ctx.pet.alive:
                if has_confirmed_add:
                    log.info(
                        "[COMBAT] Combat: ADD in melee, pet at %.0fu  -  redirecting pet + backstep",
                        pet_dist,
                    )
                    log.debug(
                        "[COMBAT] add redirect_pet -> '%s' id=%d",
                        target.name if target else "?",
                        target.spawn_id if target else 0,
                    )
                    redirect_pet()
                    if self._combat_backstep_count < _backstep_limit:
                        self._start_backstep(state, now, random.uniform(20, 28))
                        return RoutineStatus.RUNNING
                else:
                    log.info(
                        "[COMBAT] Combat: npc in melee but pet at %.0fu  -  "
                        "re-sending pet (not backstepping)",
                        pet_dist,
                    )
                    pet_attack()
            else:
                log.info("[COMBAT] Combat: npc in melee, pet DEAD  -  skipping pet_attack")
            self._last_backstep_time = now
            self._hp_at_start = state.hp_pct
        elif self._combat_backstep_count >= _backstep_limit:
            if self._ctx and self._ctx.pet.alive:
                log.info(
                    "[COMBAT] Combat: npc in melee but backstep cap (%d/%d) reached  -  re-sending pet only",
                    self._combat_backstep_count,
                    _backstep_limit,
                )
                pet_attack()
            self._last_backstep_time = now
            self._hp_at_start = state.hp_pct
        else:
            backstep = random.uniform(20, 28)
            pet_attack()
            log.info(
                "[COMBAT] Combat: BACKSTEP %.0f (%d/%d)  -  hit in melee "
                "(HP %.0f%%->%.0f%%, dist=%.0f, pet=%.0fu)",
                backstep,
                self._combat_backstep_count + 1,
                _backstep_limit,
                self._hp_at_start * 100,
                state.hp_pct * 100,
                ts.dist,
                pet_dist,
            )
            self._start_backstep(state, now, backstep)
            return RoutineStatus.RUNNING
        return None

    def _get_pet_combat_status(self, state: GameState) -> tuple[float, float]:
        """Return (pet_distance, pet_hp_pct). Defaults: (9999, -1)."""
        if not (self._ctx and self._ctx.pet.spawn_id):
            return 9999.0, -1.0
        for sp in state.spawns:
            if sp.spawn_id == self._ctx.pet.spawn_id:
                return (
                    state.pos.dist_to(sp.pos),
                    sp.hp_current / max(sp.hp_max, 1),
                )
        return 9999.0, -1.0

    def _tick_pet_save_tank(
        self,
        state: GameState,
        ts: _TickState,
        target: SpawnData,
        pet_hp_pct: float,
    ) -> RoutineStatus | None:
        """Close to melee range once when pet is dying, to share threat."""
        _pet_save_engaged = getattr(self, "_pet_save_engaged", False)
        pet_save_tank = (
            pet_hp_pct >= 0
            and pet_hp_pct < 0.35
            and state.hp_pct > 0.80
            and not _pet_save_engaged
            and self._ctx
            and self._ctx.pet.alive
            and not self._has_extra_npcs
        )
        if not (pet_save_tank and ts.dist >= 12 and ts.dist < 100):
            return None

        log.info(
            "[COMBAT] Combat: PET_SAVE closing to melee (dist=%.0f player=%.0f%% pet=%.0f%%)",
            ts.dist,
            state.hp_pct * 100,
            pet_hp_pct * 100,
        )
        self._pet_save_engaged = True
        self._stand_from_med()
        exact = heading_to(state.pos, target.pos)
        assert self._read_state_fn is not None
        _rsf = self._read_state_fn
        face_heading(exact, lambda: _rsf().heading, tolerance=10.0)
        self._start_walk_toward(target.spawn_id, 12.0, settle=0.0, update_enter_pos=True)
        return RoutineStatus.RUNNING

    def _start_walk_toward(
        self,
        spawn_id: int,
        close_dist: float,
        *,
        settle: float = 0.0,
        update_enter_pos: bool = False,
    ) -> None:
        """Begin a non-blocking walk toward a spawn. Progress is checked
        each tick by _tick_walk_progress()."""
        self._walk_target_id = spawn_id
        self._walk_close_dist = close_dist
        self._walk_deadline = time.time() + 4.0
        self._walk_settle_until = 0.0  # 0 = walking phase; >0 = settle deadline
        self._walk_settle_duration = settle  # seconds to settle after arrival
        self._walk_update_enter_pos = update_enter_pos
        move_forward_start()

    def _tick_walk_progress(self, state: GameState) -> RoutineStatus | None:
        """Poll a non-blocking walk started by _start_walk_toward().

        Returns RUNNING while walking/settling, None when complete.
        Called at the top of _tick_face_and_melee every tick.
        """
        if self._walk_target_id == 0:
            return None

        # Post-walk settle phase (casts need stillness after movement)
        if self._walk_settle_until > 0:
            if time.time() < self._walk_settle_until:
                return RoutineStatus.RUNNING
            # Settle complete
            self._finish_walk()
            return RoutineStatus.RUNNING

        # Check if close enough or deadline exceeded
        now = time.time()
        arrived = False
        for sp in state.spawns:
            if sp.spawn_id == self._walk_target_id:
                if state.pos.dist_to(sp.pos) <= self._walk_close_dist:
                    arrived = True
                break

        if arrived or now > self._walk_deadline:
            move_forward_stop()
            if not arrived:
                log.info("[COMBAT] Combat: walk toward spawn %d timed out", self._walk_target_id)
            if self._walk_settle_duration > 0:
                self._walk_settle_until = now + self._walk_settle_duration
                return RoutineStatus.RUNNING
            self._finish_walk()
            return RoutineStatus.RUNNING

        return RoutineStatus.RUNNING

    def _finish_walk(self) -> None:
        """Complete a walk: update enter_pos if requested, clear walk state."""
        if self._walk_update_enter_pos and self._read_state_fn:
            ns = self._read_state_fn()
            self._enter_pos = (ns.x, ns.y)
        self._walk_target_id = 0

    def _start_backstep(self, state: GameState, now: float, distance: float) -> None:
        """Initialize a backstep: set tracking state and start backward movement."""
        self._backstep_start_x = state.x
        self._backstep_start_y = state.y
        self._backstep_target = distance
        self._backstep_deadline = now + 2.0
        self._backstep_active = True
        self._last_backstep_time = now
        move_backward_start()

    # _tick_combat_log_and_adds -> self._monitor.tick_combat_log_and_adds
    # _tick_pet_heal -> self._monitor.tick_pet_heal

    def _tick_strategy(self, state: GameState, ts: _TickState) -> RoutineStatus | None:
        """Delegate spell casting to strategy. Handle snare, fleeing, threat."""
        target = ts.target
        if target is None:
            return None
        now = ts.now
        dist = ts.dist
        target_hp = ts.target_hp

        out_of_range = dist > MAX_CAST_RANGE

        # Npc-on-player detection (needed for pet recall and cast suppression)
        mob_on_player = False
        if self._ctx and hasattr(self._ctx, "world") and self._ctx.world:
            mob_on_player = self._ctx.world.mob_targeting_player(state.name)

        # Pet recall: npc beyond spell range + targeting player
        if (
            dist > MAX_CAST_RANGE
            and target_hp < 0.99
            and self._ctx
            and self._ctx.pet.alive
            and mob_on_player
            and not self._combat_recalled
        ):
            self._stand_from_med()
            pet_back_off()
            wait_time = random.uniform(2.0, 3.0)
            self._nb_phase = _CombatPhase.PET_RECALL
            self._nb_deadline = now + wait_time
            self._nb_data = {}
            log.info(
                "[COMBAT] Combat: npc at %.0fu HP=%.0f%% (beyond spell range, "
                "npc on player)  -  entering pet recall phase (%.1fs)",
                dist,
                target_hp * 100,
                wait_time,
            )
            return RoutineStatus.RUNNING

        # Lost pull bailout
        if dist > LOST_PULL_DISTANCE and target_hp > LOST_PULL_HP_THRESHOLD:
            log.warning(
                "[COMBAT] Combat: ABORT  -  npc at %.0fu with %.0f%% HP, lost pull", dist, target_hp * 100
            )
            if self._ctx:
                self._ctx.combat.engaged = False
                self._ctx.combat.pull_target_id = None
            clear_target()
            self.failure_reason = "evade"
            self.failure_category = FailureCategory.ENVIRONMENT
            return RoutineStatus.FAILURE

        # Snare fleeing npc -- only if pet is dead or very low HP.
        # At low npc HP the pet will catch and finish it without snare.
        snare = get_spell_by_role(SpellRole.SNARE)
        time_since_snare = now - self._last_snare_time
        min_med_before_cast = 4.0
        med_elapsed = time.time() - self._med_start if self._med_start > 0 else 999
        recently_sat = self._medding and med_elapsed < min_med_before_cast
        pet_needs_help = self._ctx and (
            not self._ctx.pet.alive or (hasattr(ts, "pet_hp") and ts.pet_hp >= 0 and ts.pet_hp < 0.30)
        )
        if (
            snare
            and target_hp < SNARE_HP_THRESHOLD
            and target.speed > 0
            and state.mana_current >= snare.mana_cost
            and time_since_snare > 30.0
            and not out_of_range
            and not recently_sat
            and pet_needs_help
        ):
            log.info(
                "[CAST] Combat: CAST %s  -  npc fleeing (HP=%.0f%%, speed=%.1f) mana=%d dist=%.0f",
                snare.name,
                target_hp * 100,
                target.speed,
                state.mana_current,
                dist,
            )
            self._cast_spell(snare.gem, snare.cast_time, now, state, target)
            self._last_snare_time = now
            return RoutineStatus.RUNNING

        # Npc fleeing: save mana, let pet finish
        mob_fleeing = target_hp < SNARE_HP_THRESHOLD and target.speed > 0.2
        if mob_fleeing and state.hp_pct >= 0.50:
            if not self._flee_logged:
                log.info(
                    "[COMBAT] Combat: npc fleeing (HP=%.0f%%, speed=%.1f)  -  "
                    "letting pet finish, saving mana",
                    target_hp * 100,
                    target.speed,
                )
                self._flee_logged = True
            return None  # skip strategy, fall through to medding
        else:
            self._flee_logged = False

        # Threat awareness
        if mob_on_player and not self._aggro_logged:
            log.warning(
                "[COMBAT] Combat: NPC TARGETING PLAYER  -  suppressing casts until pet regains threat"
            )
            self._aggro_logged = True
        elif not mob_on_player:
            self._aggro_logged = False

        # LOS block
        los_blocked = now < self._los_blocked_until

        # Strategy delegation
        tc = con_color(state.level, target.level) if state.level > 0 else Con.WHITE
        undead_names = self._ctx.loot.undead_names if self._ctx else set()
        mob_base = normalize_mob_name(target.name) if target else ""
        is_undead = mob_base in undead_names

        cast_result = None
        can_cast = (not mob_fleeing or state.hp_pct < 0.50) and not los_blocked
        if mob_on_player and state.hp_pct >= 0.60:
            can_cast = False
        if can_cast and self._strategy_impl:
            pet_hp_pct, pet_d = self._get_pet_status(state)
            cc = CastContext(
                state=state,
                target=target,
                now=now,
                dist=dist,
                target_hp=target_hp,
                tc=tc,
                time_in_combat=ts.time_in_combat,
                out_of_range=out_of_range,
                recently_sat=recently_sat,
                is_undead=is_undead,
                has_adds=self._has_extra_npcs,
                mob_on_player=mob_on_player,
                pet_hp=pet_hp_pct,
                pet_dist=pet_d,
            )
            cast_result = self._strategy_impl.execute(cc)

        return cast_result  # None = fall through to medding, RoutineStatus = early exit

    # _tick_medding -> self._monitor.tick_medding

    def _record_fight_history(
        self,
        state: GameState,
        fight_time: float,
        mana_spent: int,
        hp_delta: float,
        adds_count: int,
        add_type_names: list[str],
    ) -> None:
        """Record fight data to fight history for learning."""
        fight_history = getattr(self._ctx, "fight_history", None) if self._ctx else None
        if not fight_history:
            return
        from brain.learning.scorecard import encounter_fitness
        from perception.combat_eval import con_color

        tc = con_color(state.level, self._fight_target_level).value if self._fight_target_level else ""
        expected_dur = fight_history.learned_duration(self._fight_target_name)
        fit = encounter_fitness(
            duration=fight_time,
            mana_spent=mana_spent,
            max_mana=state.mana_max if state.mana_max > 0 else 1,
            hp_delta=hp_delta,
            defeated=self._target_killed,
            expected_duration=expected_dur if expected_dur is not None else fight_time,
        )
        log.log(
            VERBOSE,
            "[COMBAT] Encounter fitness: %.3f (dur=%.1f mana=%d hp_d=%.2f %s)",
            fit,
            fight_time,
            mana_spent,
            hp_delta,
            "defeated" if self._target_killed else "fled",
        )
        fight_history.record(
            mob_name=self._fight_target_name,
            duration=fight_time,
            mana_spent=mana_spent,
            hp_delta=hp_delta,
            casts=self._fight_casts,
            pet_heals=self._pet_mgr.heals,
            pet_died=not self._ctx.pet.alive if self._ctx else False,
            defeated=self._target_killed,
            adds=adds_count,
            mob_level=self._fight_target_level,
            player_level=state.level,
            con=tc,
            strategy=self._strategy.value if self._strategy else "",
            mana_start=self._fight_mana_start,
            mana_end=state.mana_current,
            pet_hp_start=1.0,
            pet_hp_end=self._pet_mgr.last_hp_pct if hasattr(self._pet_mgr, "last_hp_pct") else 0.0,
            xp_gained=bool(
                self._ctx and self._ctx.defeat_tracker.xp_gains > getattr(self, "_xp_at_start", 0)
            ),
            cycle_time=(
                time.time() - self._ctx.metrics.cycle_start_time
                if self._ctx and self._ctx.metrics.cycle_start_time > 0
                else 0.0
            ),
            fitness=fit,
            extra_npc_types=tuple(add_type_names),
        )

    def _infer_fast_defeat(self, state: GameState, fight_time: float) -> None:
        """Infer defeat if target vanished without explicit detection.

        Checks whether the target took damage (casts, fight duration, or pull
        spell) and records or logs accordingly.
        """
        if not self._ctx or self._target_killed or not self._fight_target_name or state.target is not None:
            return
        pull_cast = self._ctx.combat.pull_target_id is not None or self._ctx.metrics.total_casts > 0
        target_took_damage = self._fight_casts > 0 or fight_time > 10.0 or pull_cast
        if fight_time > 0.5 and target_took_damage:
            log.info(
                "[COMBAT] Combat: exit -- target gone without defeat detection "
                "(fast death), recording defeat '%s' id=%d",
                self._fight_target_name,
                self._fight_target_id,
            )
            self._record_kill(None, fight_time)
            self._target_killed = True
        elif not target_took_damage:
            log.info(
                "[COMBAT] Combat: exit -- target gone but no damage dealt "
                "(npc walked away, not a defeat) '%s' id=%d",
                self._fight_target_name,
                self._fight_target_id,
            )

    @override
    def exit(self, state: GameState) -> None:
        # Stop any active walk (close-to-pet / PET_SAVE tank)
        if self._walk_target_id != 0:
            move_forward_stop()
            self._walk_target_id = 0
        # Stop any active backstep -- backward key may still be held
        if self._backstep_active:
            move_backward_stop()
            self._backstep_active = False
            log.info("[COMBAT] Combat: exit -- stopped active backstep")
        # Clean up any active non-blocking phase (stop movement, etc.)
        if self._phase_mgr.active:
            log.info("[COMBAT] Combat: exit during %s phase -- cleaning up", self._phase_mgr.phase)
            self._phase_mgr.cleanup()
            self._phase_mgr.reset()
        self._stand_from_med()  # ensure standing before next action
        self._pending_dot_cast = None  # clear stale cast data between fights
        # Cancel any in-progress pet heal cast (interrupted by FLEE, etc.)
        if self._pet_mgr._cast_phase is not None:
            from motor.actions import stop_cast

            stop_cast()
            self._pet_mgr._cast_phase = None
        fight_time = time.time() - self._combat_start

        self._infer_fast_defeat(state, fight_time)

        if self._ctx:
            self._ctx.combat.engaged = False
            self._ctx.combat.pull_target_id = None
            self._ctx.pet.has_add = False  # clear add flag when combat ends
            # Only record cycle time and stationary defeats if target actually died
            # (not when interrupted by FLEE with npc still alive)
            if self._target_killed:
                self._ctx.metrics.update_stationary_kills(state.pos.x, state.pos.y)

        # Record defeat cycle time (acquire -> defeat)  -  only on natural completion
        if self._ctx and self._ctx.metrics.cycle_start_time > 0:
            if self._target_killed:
                cycle_time = time.time() - self._ctx.metrics.cycle_start_time
                self._ctx.metrics.total_cycle_times.append(cycle_time)
            self._ctx.metrics.cycle_start_time = 0.0

        # Track combat timing
        if self._ctx:
            self._ctx.metrics.total_combat_time += fight_time
            self._ctx.metrics.total_cast_time += self._fight_cast_time

        # Fight summary
        mana_spent = max(0, self._fight_mana_start - state.mana_current)
        hp_delta = state.hp_pct - self._hp_at_start
        idle_time = max(0, fight_time - self._fight_cast_time - self._med_time)
        adds_count = len(self._fight_adds_seen)
        # Collect add entity type names for social graph learning
        add_type_names: list[str] = []
        if self._fight_adds_seen:
            spawn_lookup = {s.spawn_id: s.name for s in state.spawns}
            for add_id in self._fight_adds_seen:
                name = spawn_lookup.get(add_id)
                if name:
                    add_type_names.append(normalize_mob_name(name))
        defeats = self._ctx.defeat_tracker.defeats if self._ctx else 0
        # Stash for cycle tracker (read by decision.py after exit)
        self.last_fight_summary = {
            "duration": round(fight_time, 1),
            "casts": self._fight_casts,
            "mana_spent": mana_spent,
            "hp_delta": round(hp_delta, 3),
            "adds": adds_count,
            "strategy": self._strategy.value,
            "med_time": round(self._med_time, 1),
        }
        log_event(
            log,
            "fight_end",
            f"[COMBAT] Fight: {self._fight_target_name} in {fight_time:.1f}s, {self._fight_casts} casts, {mana_spent} mana",
            **FightEndEvent(
                npc=self._fight_target_name,
                duration=round(fight_time, 1),
                casts=self._fight_casts,
                mana_spent=mana_spent,
                backsteps=self._fight_backsteps,
                retargets=self._fight_retargets,
                pet_heals=self._pet_mgr.heals,
                adds=adds_count,
                hp_delta=round(hp_delta, 3),
                hp_start=round(self._hp_at_start, 3),
                hp_end=round(state.hp_pct, 3),
                mana_start=self._fight_mana_start,
                mana_end=state.mana_current,
                cast_time=round(self._fight_cast_time, 1),
                idle_time=round(idle_time, 1),
                med_time=round(self._med_time, 1),
                init_dist=round(self._fight_initial_dist),
                defeats=defeats,
                pos_x=round(state.x),
                pos_y=round(state.y),
                strategy=self._strategy.value,
                entity_id=self._ctx.combat.pull_target_id or 0 if self._ctx else 0,
                world=compact_world(state),
            ),
            cycle_id=self._ctx.defeat_tracker.cycle_id if self._ctx else 0,
        )
        # Feed fight data to world model and fight history for learning
        if self._ctx and hasattr(self._ctx, "world") and self._ctx.world:
            self._ctx.world.record_fight(self._fight_target_name, fight_time)
        self._record_fight_history(state, fight_time, mana_spent, hp_delta, adds_count, add_type_names)

        # Post-defeat idle -- interruptible so FLEE can fire if extra_npcs attack
        idle = random.uniform(1.0, 3.0)
        log.log(VERBOSE, "[COMBAT] Combat: post-defeat idle %.1fs", idle)
        # Guard: _flee_check may be None if enter() was never called
        flee_fn = self._flee_check if self._flee_check is not None else None
        if interruptible_sleep(idle, flee_fn):
            log.info(
                "[COMBAT] Combat: post-defeat idle interrupted by FLEE urgency -- skipping remaining exit"
            )
