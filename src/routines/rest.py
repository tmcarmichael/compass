"""Rest routine: sit to regen HP/mana, stand when thresholds met."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, override

from core.timing import interruptible_sleep
from core.types import FailureCategory, ReadStateFn
from eq.loadout import SpellRole, get_spell_by_role
from motor.actions import pet_ready, pet_sit, press_gem, sit, stand, verified_sit
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus, make_flee_predicate
from util.event_schemas import RestEndEvent
from util.log_tiers import EVENT
from util.structured_log import log_event

if TYPE_CHECKING:
    from brain.context import AgentContext
    from routines.casting import CastingPhase

log = logging.getLogger(__name__)

# -- Domain-local constants --
MANA_REGEN_BUFF_THRESHOLD = 0.40  # cast mana regen buff below this mana %
REST_GRACE_PERIOD = 3.0  # ignore HP fluctuations for this long after sitting

# Spells looked up by role  -  adapts to any level


def should_cast_regen_buff(
    hp_pct: float,
    mana_pct: float,
    mana_current: int,
    regen_mana_cost: int,
    is_sitting: bool,
) -> bool:
    """Pure function: decide whether to cast mana regen buff before sitting.

    The lich buff (-2 HP/tick, +2 mana/tick) is only worth the HP cost when
    HP is at 100% and mana is below the threshold. Must have enough mana
    to cast and must not already be sitting (casting requires standing).
    """
    if hp_pct < 1.0:
        return False
    if mana_pct >= MANA_REGEN_BUFF_THRESHOLD:
        return False
    if mana_current < regen_mana_cost:
        return False
    if is_sitting:
        return False
    return True


def should_exit_rest(
    hp_pct: float,
    mana_pct: float,
    mana_max: int,
    hp_target: float,
    mana_target: float,
    pet_hp_pct: float | None,
) -> bool:
    """Pure function: decide whether rest is complete.

    All three conditions must be met: player HP, player mana, and pet HP.
    """
    hp_ok = hp_pct >= hp_target
    mana_ok = mana_pct >= mana_target or mana_max == 0
    pet_ok = pet_hp_pct is None or pet_hp_pct >= 0.90
    return hp_ok and mana_ok and pet_ok


class RestRoutine(RoutineBase):
    """Sit to regen. Stay seated until well recovered (80%+ mana)."""

    def __init__(
        self,
        hp_high: float = 0.95,
        mana_high: float = 0.80,
        ctx: AgentContext | None = None,
        read_state_fn: ReadStateFn | None = None,
    ) -> None:
        self._hp_target = hp_high
        self._mana_target = mana_high
        self._ctx = ctx
        self._read_state_fn = read_state_fn
        self._rest_start = 0.0
        self._hp_at_start = 0.0
        self._mana_at_start = 0
        self._pet_sat = False  # True if we told pet to sit this rest
        self._grace_until = 0.0  # suppress HP-drop fallback for first few seconds
        self._cast_phase: CastingPhase | None = None
        self._post_cast_action: str | None = None
        self._flee_check: Callable[[], bool] | None = None

    def _get_pet_hp_pct(self, state: GameState) -> float | None:
        """Find our pet's HP percentage from spawn list. Returns None if not found."""
        if not self._ctx or not self._ctx.pet.spawn_id:
            return None
        for spawn in state.spawns:
            if spawn.spawn_id == self._ctx.pet.spawn_id and spawn.is_npc:
                if spawn.hp_max > 0:
                    pct: float = spawn.hp_current / spawn.hp_max
                    return pct
        return None

    @override
    def enter(self, state: GameState) -> None:
        self._rest_start = time.time()
        self._hp_at_start = state.hp_pct
        self._mana_at_start = state.mana_current
        self._pet_sat = False
        self._cast_phase = None
        self._post_cast_action = None
        if self._read_state_fn and self._ctx:
            self._flee_check = make_flee_predicate(self._read_state_fn, self._ctx)
        else:
            self._flee_check = None
        # Grace period: ignore HP-drop fallback for 3s after sitting.
        # Catches: HP jitter from buff effects, CHARINFO read instability,
        # and sit-transition HP recalculations.
        self._grace_until = time.time() + REST_GRACE_PERIOD
        if self._ctx:
            self._ctx.player.rest_start_time = time.time()
            self._ctx.player.last_rest_hp = state.hp_pct
        log.log(
            EVENT,
            "[MANA] Rest: sitting (HP=%.0f%% [%d/%d], Mana=%d/%d [%.0f%%]) targets: HP>=%.0f%% Mana>=%.0f%%",
            state.hp_pct * 100,
            state.hp_current,
            state.hp_max,
            state.mana_current,
            state.mana_max,
            state.mana_pct * 100,
            self._hp_target * 100,
            self._mana_target * 100,
        )
        if self._ctx:
            self._ctx.metrics.rest_count += 1

            # Mana regen buff: cast before sitting if HP is full but mana is low.
            # The lich buff (-2 HP/tick, +2 mana/tick) stacks with med regen.
            # Only when HP >= 100% and mana < 40%  -  the HP cost is worth it.
            mana_regen = get_spell_by_role(SpellRole.MANA_REGEN)
            if (
                mana_regen
                and state.hp_pct >= 1.0
                and state.mana_max > 0
                and state.mana_pct < MANA_REGEN_BUFF_THRESHOLD
                and state.mana_current >= mana_regen.mana_cost
                and not state.is_sitting
            ):
                log.info(
                    "[CAST] Rest: casting %s before med (HP=100%%, Mana=%.0f%%)  -  mana regen active",
                    mana_regen.name,
                    state.mana_pct * 100,
                )
                press_gem(mana_regen.gem)
                from routines.casting import CastingPhase

                self._cast_phase = CastingPhase(mana_regen.cast_time, mana_regen.name, self._read_state_fn)
                self._post_cast_action = "mana_regen"
                if self._ctx:
                    self._ctx.metrics.total_casts += 1
                return  # tick() will poll the cast and sit when done

            # Sit if not already sitting. Check memory first: if the player is
            # already sitting from a prior med (e.g. combat med that didn't stand
            # before combat ended), skip the toggle to avoid standing back up.
            import motor.actions as _ma

            if state.is_sitting:
                # Memory confirms sitting -- just sync the internal tracker.
                # Sending sit() when already sitting would TOGGLE to standing.
                _ma.mark_sitting()
                log.info(
                    "[MANA] Rest: player already sitting (stand_state=%d) -- no toggle", state.stand_state
                )
            else:
                _ma.force_standing()  # reset so sit() won't skip
                log.info("[MANA] Rest: player sitting down (stand_state=%d)", state.stand_state)
                sit()
                interruptible_sleep(0.3, self._flee_check)
            # Only sit pet if it needs HP regen (below 80%)
            if self._ctx and self._ctx.pet.alive and not self._pet_sat:
                world = getattr(self._ctx, "world", None)
                pet_hp = world.pet_hp_pct if world else -1
                if 0 <= pet_hp < 0.80:
                    interruptible_sleep(0.3, self._flee_check)
                    pet_sit()
                    self._pet_sat = True
                    log.info("[MANA] Rest: pet sitting to regen (HP=%.0f%%)", pet_hp * 100)
                else:
                    log.debug(
                        "[MANA] Rest: pet HP OK (%.0f%%), not sitting pet",
                        pet_hp * 100 if pet_hp >= 0 else -1,
                    )

    def _handle_attack(self, state: GameState, reason: str) -> RoutineStatus:
        """Stand up and engage attacker. Shared by in_combat and HP-drop detection."""
        log.warning("[MANA] Rest: %s  -  standing + engaging attacker", reason)
        from motor.actions import pet_attack, stand

        stand()
        if self._ctx:
            # Find the attacker  -  any damaged NPC nearby
            from perception.combat_eval import is_pet

            for sp in state.spawns:
                if sp.is_npc and not is_pet(sp) and sp.hp_current > 0 and sp.hp_current < sp.hp_max:
                    d = state.pos.dist_to(sp.pos)
                    if d < 50:
                        log.warning(
                            "[MANA] Rest: attacker is '%s' lv%d at %.0fu  -  engaging regardless of con",
                            sp.name,
                            sp.level,
                            d,
                        )
                        self._ctx.combat.engaged = True
                        self._ctx.player.engagement_start = time.time()
                        self._ctx.defeat_tracker.last_fight_name = sp.name
                        self._ctx.defeat_tracker.last_fight_id = sp.spawn_id
                        self._ctx.defeat_tracker.last_fight_x = sp.x
                        self._ctx.defeat_tracker.last_fight_y = sp.y
                        pet_attack()
                        break
        self.failure_reason = "interrupted"
        self.failure_category = FailureCategory.ENVIRONMENT
        return RoutineStatus.FAILURE

    def _tick_cast_polling(self) -> RoutineStatus | None:
        """Poll an active cast (mana regen or pet heal).

        Returns RUNNING if still casting/post-cast, or None to continue tick.
        """
        if self._cast_phase is None:
            return None

        result = self._cast_phase.tick()
        from routines.casting import CastResult

        if result == CastResult.CASTING:
            return RoutineStatus.RUNNING
        # Cast done
        self._cast_phase = None
        if self._post_cast_action == "mana_regen":
            # Mana regen cast done -- now sit
            self._post_cast_action = None
            if self._read_state_fn:
                ns = self._read_state_fn()
                if not ns.is_sitting:
                    verified_sit(self._read_state_fn)
            if self._ctx and self._ctx.pet.alive and not self._pet_sat:
                world = getattr(self._ctx, "world", None)
                pet_hp = world.pet_hp_pct if world else -1
                if 0 <= pet_hp < 0.80:
                    interruptible_sleep(0.3, self._flee_check)
                    pet_sit()
                    self._pet_sat = True
            return RoutineStatus.RUNNING
        if self._post_cast_action == "pet_heal":
            # Pet heal done -- sit back down
            self._post_cast_action = None
            sit()
            interruptible_sleep(0.5, self._flee_check)
            return RoutineStatus.RUNNING
        return None

    def _tick_pet_heal(self, state: GameState, pet_hp: float | None) -> RoutineStatus | None:
        """Heal pet while resting if needed.

        Returns RUNNING if a heal cast was started, or None to continue tick.
        """
        pet_heal = get_spell_by_role(SpellRole.PET_HEAL)
        now = time.time()
        last_heal = self._ctx.pet.last_heal_time if self._ctx else 0
        if not (
            pet_heal
            and pet_hp is not None
            and pet_hp < 0.85
            and state.mana_current >= pet_heal.mana_cost
            and now - last_heal > pet_heal.recast + 0.5
        ):
            return None

        log.info(
            "[CAST] Rest: healing pet with %s (HP=%.0f%%, mana=%d)",
            pet_heal.name,
            pet_hp * 100,
            state.mana_current,
        )
        from motor.actions import stand

        stand()
        interruptible_sleep(1.0, self._flee_check)
        from motor.actions import pet_heal as _pet_heal_action

        _pet_heal_action()  # /pet target + cast gem 7 (no macro needed)
        from routines.casting import CastingPhase

        self._cast_phase = CastingPhase(pet_heal.cast_time, pet_heal.name, self._read_state_fn)
        self._post_cast_action = "pet_heal"
        if self._ctx:
            self._ctx.pet.last_heal_time = time.time()
        return RoutineStatus.RUNNING

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        # Non-blocking cast polling (mana regen or pet heal)
        cast_result = self._tick_cast_polling()
        if cast_result is not None:
            return cast_result

        # Primary: memory-based combat detection (instant, no threshold needed)
        # The in_combat flag reads entity struct no_regen_flag  -  set by the game
        # engine when a npc is attacking us. BUT also fires from DoTs ticking
        # on the player (Dark Pact, poison, disease). Only treat as attack if
        # there's an actual damaged NPC nearby (someone hitting us, not just DoT).
        if state.in_combat:
            pass

            attacker = None
            for sp in state.spawns:
                if sp.is_npc and sp.hp_current > 0 and sp.hp_current < sp.hp_max:
                    d = state.pos.dist_to(sp.pos)
                    if d < 50:
                        attacker = sp
                        break
            if attacker:
                return self._handle_attack(
                    state, f"in_combat + nearby npc '{attacker.name}' (HP={state.hp_pct * 100:.0f}%)"
                )
            else:
                log.debug("[MANA] Rest: in_combat flag but no attackers nearby  -  likely DoT tick, ignoring")

        # Fallback: HP-drop heuristic  -  catches attacks if no_regen_flag
        # has a delay or isn't set for certain damage types (e.g., DoTs).
        # Skip during grace period (first 3s) to avoid false triggers from
        # buff HP cost, sit-transition jitter, or CHARINFO read instability.
        if time.time() > self._grace_until:
            if state.hp_pct < self._hp_at_start - 0.03:
                return self._handle_attack(
                    state, f"HP dropped {self._hp_at_start * 100:.0f}%->{state.hp_pct * 100:.0f}% (fallback)"
                )

        # Heal pet while resting  -  a skilled player always heals between fights
        # Uses ctx shared timer to respect recast across routines
        pet_hp = self._get_pet_hp_pct(state)
        heal_result = self._tick_pet_heal(state, pet_hp)
        if heal_result is not None:
            return heal_result

        # Exit check: player HP + mana + pet HP all need to be OK
        hp_ok = state.hp_pct >= self._hp_target
        mana_ok = state.mana_pct >= self._mana_target or state.mana_max == 0
        pet_ok = pet_hp is None or pet_hp >= 0.90  # pet must be 90%+ to exit

        if hp_ok and mana_ok and pet_ok:
            return RoutineStatus.SUCCESS
        return RoutineStatus.RUNNING

    @override
    def exit(self, state: GameState) -> None:
        rest_duration = time.time() - self._rest_start if self._rest_start else 0
        pet_hp = self._get_pet_hp_pct(state)
        log_event(
            log,
            "rest_end",
            f"[MANA] Rest: {rest_duration:.1f}s HP={self._hp_at_start * 100:.0f}%->{state.hp_pct * 100:.0f}% mana={self._mana_at_start}->{state.mana_current}",
            **RestEndEvent(
                duration=round(rest_duration, 1),
                hp_start=round(self._hp_at_start, 3),
                hp_end=round(state.hp_pct, 3),
                mana_start=self._mana_at_start,
                mana_end=state.mana_current,
                pet_hp=round(pet_hp * 100) if pet_hp is not None else -1,
            ),
        )
        if self._ctx:
            # Stand pet first if we sat it
            if self._pet_sat:
                pet_ready()  # stand + debounce + follow
                log.info("[MANA] Rest: pet stand + follow")
                self._pet_sat = False
            if state.is_sitting:
                stand()
                interruptible_sleep(0.4, self._flee_check)
