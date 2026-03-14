"""Combat monitoring: death detection, distance tracking, log parsing, pet heal, medding.

Extracted from combat.py to keep the main CombatRoutine focused on
spell rotation and strategy. These are the "observation" stages of the
tick pipeline that detect state changes and react accordingly.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from core.constants import (
    COMBAT_LOG_INTERVAL,
)
from core.timing import interruptible_sleep, varying_sleep
from core.types import Point
from motor.actions import (
    clear_target,
    sit,
)
from perception.combat_eval import Con, con_color, is_pet
from perception.state import GameState, SpawnData
from routines.base import RoutineStatus
from util.log_tiers import VERBOSE

if TYPE_CHECKING:
    from routines.combat import CombatRoutine, _TickState

log = logging.getLogger(__name__)

# Domain-local constants (mirrored from combat.py for the methods that use them)
LOW_HP_POLL_THRESHOLD = 0.15
MED_SAFE_DISTANCE = 30.0
MED_MOB_HP_MIN = 0.20
NEARBY_NPC_RANGE = 50.0


def can_med_during_combat(
    fight_duration: float,
    mana_pct: float,
    dist: float,
    target_hp: float,
    hp_pct: float,
    hp_at_start: float,
    pet_alive: bool,
    pet_hp: float,
    has_extra_npcs: bool,
    nearby_npc: bool,
    time_since_cast: float,
) -> bool:
    """Pure function: decide whether to sit and med during combat.

    Extracted from CombatMonitor.tick_medding for testability.
    """
    return (
        fight_duration > 5.0
        and mana_pct < 0.75
        and dist > MED_SAFE_DISTANCE
        and target_hp > MED_MOB_HP_MIN
        and hp_pct >= hp_at_start - 0.02
        and pet_alive
        and pet_hp > 0.50
        and not has_extra_npcs
        and not nearby_npc
        and time_since_cast > 2.0
    )


class CombatMonitor:
    """Monitors combat state: death, distance, adds, pet heal, medding.

    Composed into CombatRoutine -- not a subclass. The parent routine
    passes itself so the monitor can access shared state (_combat_start,
    _medding, _has_extra_npcs, etc.) and call shared helpers (_stand_from_med,
    _record_kill, etc.).

    Args:
        combat: The parent CombatRoutine instance.
    """

    def __init__(self, combat: CombatRoutine) -> None:
        self._combat = combat
        self._med_next_poll: float = 0.0

    # -- Tick pipeline stages (called from CombatRoutine.tick) ----------------

    def tick_death_check(self, state: GameState, ts: _TickState) -> RoutineStatus | None:
        """Check if target is dead. Record defeat and return SUCCESS."""
        target = ts.target
        if not target or not target.is_npc or target.hp_current <= 0:
            self._combat._stand_from_med()
            self._combat._target_killed = True
            fight_time = ts.now - self._combat._combat_start
            log.info(
                "[COMBAT] Combat: TARGET DEAD  -  fight lasted %.1fs, player HP=%.0f%% Mana=%d "
                "pos=(%.0f,%.0f)",
                fight_time,
                state.hp_pct * 100,
                state.mana_current,
                state.x,
                state.y,
            )
            self._combat._record_kill(target, fight_time)
            return RoutineStatus.SUCCESS
        return None

    def tick_distance_update(self, state: GameState, ts: _TickState) -> RoutineStatus | None:
        """Update distance/HP tracking. Emergency stand from med if needed."""
        from routines.combat_phases import CombatPhase

        target = ts.target
        if target is None:
            return None
        ts.dist = state.pos.dist_to(target.pos)
        ts.target_hp = self._combat._target_hp_pct(target)
        ts.time_in_combat = ts.now - self._combat._combat_start
        self._combat._fight_target_x = target.x
        self._combat._fight_target_y = target.y
        if self._combat._ctx:
            self._combat._ctx.combat.last_mob_hp_pct = ts.target_hp
            self._combat._ctx.defeat_tracker.last_fight_x = target.x
            self._combat._ctx.defeat_tracker.last_fight_y = target.y

        if self._combat._medding:
            must_stand = False
            if ts.dist < 30:
                must_stand = True
                log.info("[COMBAT] Combat: npc close (%.0fu)  -  standing from med", ts.dist)
            elif state.hp_pct < self._combat._hp_at_start - 0.02:
                must_stand = True
                log.info(
                    "[COMBAT] Combat: taking damage (HP %.0f%%)  -  standing from med", state.hp_pct * 100
                )
            elif ts.target_hp < LOW_HP_POLL_THRESHOLD:
                must_stand = True
                log.info(
                    "[COMBAT] Combat: npc nearly dead (%.0f%%)  -  standing from med", ts.target_hp * 100
                )
            if must_stand:
                self._combat._stand_from_med()
                if ts.dist < 20:
                    from motor.actions import move_forward_start

                    log.info("[COMBAT] Combat: npc at %.0fu -- entering BACKSTEP phase", ts.dist)
                    move_forward_start()
                    self._combat._phase_mgr.phase = CombatPhase.BACKSTEP
                    self._combat._phase_mgr.deadline = ts.now + 0.5
                    return RoutineStatus.RUNNING
        return None

    def tick_combat_log_and_adds(self, state: GameState, ts: _TickState) -> RoutineStatus | None:
        """Periodic combat status logging and add detection."""
        target = ts.target
        if target is None:
            return None
        now = ts.now

        if now - self._combat._last_combat_log > COMBAT_LOG_INTERVAL:
            pet_info = ""
            if self._combat._ctx and self._combat._ctx.pet.spawn_id:
                for spawn in state.spawns:
                    if spawn.spawn_id == self._combat._ctx.pet.spawn_id:
                        pet_pct = spawn.hp_current / spawn.hp_max * 100 if spawn.hp_max > 0 else 0
                        pd = state.pos.dist_to(spawn.pos)
                        pet_info = (
                            f" | pet HP={spawn.hp_current}/{spawn.hp_max}"
                            f" ({pet_pct:.0f}%) pos=({spawn.x:.0f},{spawn.y:.0f})"
                            f" dist={pd:.0f}"
                        )
                        break
            log.log(
                VERBOSE,
                "[COMBAT] Combat: t=%.0fs | npc='%s' HP=%.0f%% (%d/%d) dist=%.0f "
                "mob_pos=(%.0f,%.0f) speed=%.1f | "
                "player HP=%.0f%% Mana=%d casts=%d pos=(%.0f,%.0f)%s "
                "| strategy=%s",
                ts.time_in_combat,
                target.name,
                ts.target_hp * 100,
                target.hp_current,
                target.hp_max,
                ts.dist,
                target.x,
                target.y,
                target.speed,
                state.hp_pct * 100,
                state.mana_current,
                self._combat._fight_casts,
                state.x,
                state.y,
                pet_info,
                self._combat._strategy.value,
            )
            self._combat._last_combat_log = now

            # Drift detection: if player moved >40u from combat start
            # without active backstep, a movement key is stuck
            enter_pos = getattr(self._combat, "_enter_pos", None)
            if enter_pos and not self._combat._backstep_active:
                drift = state.pos.dist_to(Point(enter_pos[0], enter_pos[1], 0.0))
                if drift > 40:
                    log.warning(
                        "[COMBAT] Combat: DRIFT %.0fu from combat start "
                        "(%.0f,%.0f)->(%.0f,%.0f) -- stuck key, "
                        "cancelling with forward tap",
                        drift,
                        enter_pos[0],
                        enter_pos[1],
                        state.x,
                        state.y,
                    )
                    # Key-up is being ignored by EQ. Send a brief forward
                    # press to physically cancel the backward movement.
                    from motor.actions import (
                        move_backward_stop as _mbs,
                    )
                    from motor.actions import (
                        move_forward_start as _mfs2,
                    )
                    from motor.actions import (
                        move_forward_stop as _mfs,
                    )

                    _mbs()
                    _mfs2()
                    varying_sleep(0.1, sigma=0.1)
                    _mfs()
                    _mbs()  # one more for good measure
                    self._combat._enter_pos = (state.x, state.y)

        # Add detection
        had_adds = self._combat._has_extra_npcs
        extra_npc_list = self._combat._detect_adds(state)
        self._combat._has_extra_npcs = bool(extra_npc_list)
        if self._combat._has_extra_npcs and not had_adds:
            for extra_npc_spawn, extra_npc_dist, near_who in extra_npc_list:
                add_con = con_color(state.level, extra_npc_spawn.level) if state.level > 0 else Con.WHITE
                log.warning(
                    "[COMBAT] Combat: ADD '%s' lv%d %s at %.0fu (near %s) HP=%d/%d",
                    extra_npc_spawn.name,
                    extra_npc_spawn.level,
                    add_con,
                    extra_npc_dist,
                    near_who,
                    extra_npc_spawn.hp_current,
                    extra_npc_spawn.hp_max,
                )
        for extra_npc_spawn, _ad, _nw in extra_npc_list:
            self._combat._fight_adds_seen.add(extra_npc_spawn.spawn_id)
        # Clear sticky has_add when no live extra_npcs remain
        if self._combat._ctx and self._combat._ctx.pet.has_add and not extra_npc_list and had_adds:
            log.info("[COMBAT] Combat: pet add cleared (no live extra_npcs nearby)")
            self._combat._ctx.pet.has_add = False
        return None

    def tick_pet_heal(self, state: GameState, ts: _TickState) -> RoutineStatus | None:
        """Priority 0: heal pet. Pet is our tank -- keeping it alive IS the DPS."""
        target = ts.target

        # Sit/stand jitter prevention -- bypass when PET_SAVE is active
        # (player already stood to tank, pet needs heal NOW)
        min_med_before_cast = 4.0
        med_elapsed = time.time() - self._combat._med_start if self._combat._med_start > 0 else 999
        pet_save = getattr(self._combat, "_pet_save_engaged", False)
        recently_sat = self._combat._medding and med_elapsed < min_med_before_cast and not pet_save

        # Check if we're mid-heal (non-blocking cast in progress)
        if self._combat._pet_mgr._cast_phase is not None:
            tick_result = self._combat._pet_mgr.tick_heal()
            if tick_result is None:
                return RoutineStatus.RUNNING
            if tick_result is True:
                self._combat._fight_casts += 1
                return RoutineStatus.RUNNING
            if tick_result is False:
                self._combat._target_killed = True
                fight_time = time.time() - self._combat._combat_start
                log.info(
                    "[COMBAT] Combat: npc died during Mend Bones  -  recording defeat '%s'",
                    self._combat._fight_target_name,
                )
                self._combat._record_kill(None, fight_time)
                clear_target()
                interruptible_sleep(0.3, self._combat._flee_check)
                return RoutineStatus.SUCCESS

        heal_result = self._combat._pet_mgr.try_heal(
            state,
            target,
            medding=self._combat._medding,
            recently_sat=recently_sat,
            stand_from_med_fn=self._combat._stand_from_med,
        )
        if heal_result == "healing":
            return RoutineStatus.RUNNING
        if heal_result is True:
            self._combat._fight_casts += 1
            return RoutineStatus.RUNNING
        if heal_result is False:
            self._combat._target_killed = True
            fight_time = time.time() - self._combat._combat_start
            log.info(
                "[COMBAT] Combat: npc died during Mend Bones  -  recording defeat '%s'",
                self._combat._fight_target_name,
            )
            self._combat._record_kill(None, fight_time)
            clear_target()
            interruptible_sleep(0.3, self._combat._flee_check)
            return RoutineStatus.SUCCESS
        return None

    def tick_medding(
        self, state: GameState, now: float, target: SpawnData | None, dist: float, target_hp: float
    ) -> RoutineStatus:
        """Sit to regen mana while pet tanks. Terminal block of tick().

        Only med if: pet alive, npc not close, not taking damage, npc not
        about to die, and no other NPCs nearby that could threat while sitting.
        """
        # Skip med for 1 tick after a fizzle to allow immediate retry
        if self._combat._retry_after_fizzle:
            self._combat._retry_after_fizzle = False
            log.info("[CAST] Combat: skipping med (retry after fizzle)  -  will retry cast next tick")
            interruptible_sleep(0.3, self._combat._flee_check)
            return RoutineStatus.RUNNING
        nearby_npc = False
        for spawn in state.spawns:
            if not spawn.is_npc or spawn.hp_current <= 0:
                continue
            if is_pet(spawn):
                continue
            if target and spawn.spawn_id == target.spawn_id:
                continue
            d = state.pos.dist_to(spawn.pos)
            if d < NEARBY_NPC_RANGE:
                nearby_npc = True
                break
        # Don't sit immediately after casting  -  add a human delay
        time_since_cast = now - self._combat._cast_end_time if self._combat._cast_end_time > 0 else 999
        # Break med if pet HP is critical -- need to cast to help
        pet_hp = (
            self._combat._ctx.world.pet_hp_pct if (self._combat._ctx and self._combat._ctx.world) else 1.0
        )
        # Don't med in short fights -- mana ticks are 6s, sitting for 3s is wasted
        fight_duration = now - self._combat._combat_start
        can_med = (
            fight_duration > 5.0  # sit after 5s (one mana tick = 6s)
            and state.mana_pct < 0.75  # don't sit if mana is healthy
            and dist > MED_SAFE_DISTANCE
            and target_hp > MED_MOB_HP_MIN
            and state.hp_pct >= self._combat._hp_at_start - 0.02
            and self._combat._ctx
            and self._combat._ctx.pet.alive
            and pet_hp > 0.50  # stand and help if pet is struggling
            and not self._combat._has_extra_npcs
            and not nearby_npc
            and time_since_cast > 2.0  # don't sit within 2s of casting
        )
        if can_med and not self._combat._medding:
            sit()
            self._combat._medding = True
            self._combat._med_start = time.time()
            log.info(
                "[COMBAT] Combat: medding while pet tanks (npc HP=%.0f%% dist=%.0f) %s",
                target_hp * 100,
                dist,
                self._combat._vitals(state),
            )
        # Check more frequently when npc is nearly dead so we catch the
        # death before the brain's stale-engaged safety net does
        if target_hp < LOW_HP_POLL_THRESHOLD:
            interruptible_sleep(0.15, self._combat._flee_check)
            # Re-check target after short sleep  -  npc may have died
            if self._combat._read_state_fn:
                fresh = self._combat._read_state_fn()
                ft = fresh.target
                if not ft or not ft.is_npc or ft.hp_current <= 0:
                    self._combat._stand_from_med()
                    self._combat._target_killed = True
                    fight_time = time.time() - self._combat._combat_start
                    log.info(
                        "[COMBAT] Combat: TARGET DEAD  -  fight lasted %.1fs, "
                        "player HP=%.0f%% Mana=%d pos=(%.0f,%.0f)",
                        fight_time,
                        fresh.hp_pct * 100,
                        fresh.mana_current,
                        fresh.x,
                        fresh.y,
                    )
                    self._combat._record_kill(ft, fight_time)
                    return RoutineStatus.SUCCESS
        else:
            # Non-blocking: throttle medding polls to ~1Hz instead of blocking 1s
            if time.time() >= self._med_next_poll:
                self._med_next_poll = time.time() + 1.0
        return RoutineStatus.RUNNING
