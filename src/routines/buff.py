"""Buff routine: cast and maintain self-buffs.

Auto-detects which shielding spell is memorized (Minor/Lesser Shielding).
Buff expiration is detected via memory-read buff ticks (brain rule in
maintenance.py checks state.buff_ticks). Verifies cast by checking mana
decrease.

Non-blocking: uses phase state machine with deadline timestamps so tick()
never blocks >100ms. FLEE can fire between any two ticks.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from enum import Enum, auto
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from brain.context import AgentContext
    from eq.loadout import Spell
    from routines.casting import CastingPhase

from core.timing import interruptible_sleep
from core.types import ReadStateFn
from eq.loadout import SpellRole, get_spell_by_role
from motor.actions import press_gem, stand
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus, make_flee_predicate

log = logging.getLogger(__name__)

MAX_RETRIES = 3

# Non-blocking wait durations (seconds) -- same as old natural_sleep values
STAND_WAIT = 1.0  # wait after stand() for animation
SETTLE_WAIT = 1.0  # settle before casting (EQ needs time after standing)


class _Phase(Enum):
    """Phases for the non-blocking buff cast sequence."""

    INIT = auto()  # initial state: validate spell, stand, start wait
    STAND_WAIT = auto()  # waiting for stand animation to complete
    SETTLE_WAIT = auto()  # waiting for settle before pressing gem
    CASTING = auto()  # CastingPhase active, polling for completion


class BuffRoutine(RoutineBase):
    """Cast self-buff when memory-read buff ticks have expired."""

    def __init__(self, ctx: AgentContext | None = None, read_state_fn: ReadStateFn | None = None) -> None:
        self._ctx = ctx
        self._read_state_fn = read_state_fn
        self._retries = 0
        self._phase = _Phase.INIT
        self._phase_deadline = 0.0  # time.time() when current wait expires
        self._cast_phase: CastingPhase | None = None
        self._mana_before_cast = 0
        self._casting = False  # True while CastingPhase is active
        self._locked_for_cast = False  # True from first stand() until cast verified/failed
        self._flee_check: Callable[[], bool] | None = None

    @override
    @property
    def locked(self) -> bool:
        """Lock from stand-up through cast completion.

        Once we've stood up and started the buff sequence we must not be
        interrupted by a lower-priority rule (WANDER/ACQUIRE/REST) until the
        cast verifies or fails.  Without this lock the brain can deactivate
        BuffRoutine between STAND_WAIT and SETTLE_WAIT, leaving the character
        standing but with no cast having fired -- then the next tick sees
        is_sitting=False and the buff rule suppresses on 'Buff skip: sitting'.
        """
        return self._locked_for_cast

    def _get_buff_spell(self) -> Spell | None:
        return get_spell_by_role(SpellRole.SELF_BUFF)

    @override
    def enter(self, state: GameState) -> None:
        spell = self._get_buff_spell()
        if spell:
            log.info("[CAST] Buff: casting %s (gem %d)", spell.name, spell.gem)
        self._retries = 0
        self._phase = _Phase.INIT
        self._phase_deadline = 0.0
        self._cast_phase = None
        self._mana_before_cast = 0
        self._casting = False
        self._locked_for_cast = False
        if self._read_state_fn and self._ctx:
            self._flee_check = make_flee_predicate(self._read_state_fn, self._ctx)
        else:
            self._flee_check = None

        # Stand up if sitting (enter runs once, sleep is acceptable here)
        if state.is_sitting:
            stand()
            interruptible_sleep(0.5, self._flee_check)

    @override
    def tick(self, state: GameState) -> RoutineStatus:
        # -- Phase: CASTING (waiting for CastingPhase to complete) --
        if self._phase == _Phase.CASTING:
            if self._cast_phase is None:
                log.warning("[CAST] Buff: CASTING phase but no cast_phase object")
                self._phase = _Phase.INIT
                self._casting = False
                self._locked_for_cast = False
                return RoutineStatus.RUNNING
            result = self._cast_phase.tick()
            from routines.casting import CastResult

            if result == CastResult.CASTING:
                return RoutineStatus.RUNNING
            # Cast done or timed out -- verify
            log.debug("[CAST] Buff: cast phase complete, verifying")
            self._cast_phase = None
            self._casting = False
            self._locked_for_cast = False
            self._phase = _Phase.INIT
            return self._verify_cast()

        # -- Phase: STAND_WAIT (waiting for stand animation) --
        if self._phase == _Phase.STAND_WAIT:
            if time.time() < self._phase_deadline:
                return RoutineStatus.RUNNING
            log.debug("[CAST] Buff: stand wait complete, recording mana and settling")
            # Read fresh state for mana snapshot
            if self._read_state_fn:
                state = self._read_state_fn()
            self._mana_before_cast = state.mana_current

            # Record cast time on ctx for REST suppression and recast cooldown
            # Do this BEFORE casting so even if brain deactivates mid-cast,
            # the cooldown is set (prevents 30s rebuff loop)
            if self._ctx:
                self._ctx.player.last_buff_time = time.time()

            # Transition to settle wait
            self._phase = _Phase.SETTLE_WAIT
            self._phase_deadline = time.time() + SETTLE_WAIT
            return RoutineStatus.RUNNING

        # -- Phase: SETTLE_WAIT (waiting before pressing gem) --
        if self._phase == _Phase.SETTLE_WAIT:
            if time.time() < self._phase_deadline:
                return RoutineStatus.RUNNING
            log.debug("[CAST] Buff: settle complete, pressing gem")
            spell = self._get_buff_spell()
            if not spell or not spell.gem:
                log.warning("[CAST] Buff: spell disappeared during settle  -  aborting")
                return RoutineStatus.FAILURE
            press_gem(spell.gem)

            # Non-blocking: start cast phase and return RUNNING
            from routines.casting import CastingPhase

            self._cast_phase = CastingPhase(spell.cast_time, spell.name, self._read_state_fn)
            self._casting = True
            self._phase = _Phase.CASTING
            return RoutineStatus.RUNNING

        # -- Phase: INIT (validate and begin cast sequence) --
        spell = self._get_buff_spell()
        if not spell or not spell.gem:
            log.warning("[CAST] Buff: no self-buff spell memorized  -  skipping")
            return RoutineStatus.FAILURE

        if self._retries >= MAX_RETRIES:
            log.warning("[CAST] Buff: failed after %d attempts  -  skipping", self._retries)
            return RoutineStatus.FAILURE

        # Always read fresh state (brain state may be stale after memorize)
        if self._read_state_fn:
            state = self._read_state_fn()

        stand()
        self._locked_for_cast = True
        log.debug("[CAST] Buff: stood up, locking routine, waiting %.1fs for animation", STAND_WAIT)
        self._phase = _Phase.STAND_WAIT
        self._phase_deadline = time.time() + STAND_WAIT
        return RoutineStatus.RUNNING

    def _verify_cast(self) -> RoutineStatus:
        """Check mana delta to verify buff landed."""
        spell = self._get_buff_spell()
        if self._read_state_fn:
            ns = self._read_state_fn()
            mana_after = ns.mana_current
            cost = self._mana_before_cast - mana_after
            if cost >= 5:
                log.info(
                    "[CAST] Buff: %s LANDED (mana %d->%d, cost=%d, HP=%d/%d=%.0f%%)",
                    spell.name if spell else "?",
                    self._mana_before_cast,
                    mana_after,
                    cost,
                    ns.hp_current,
                    ns.hp_max,
                    ns.hp_pct * 100,
                )
                return RoutineStatus.SUCCESS
            elif cost == 0 and self._mana_before_cast == mana_after:
                log.info(
                    "[CAST] Buff: %s -- no mana spent (likely already active), treating as success",
                    spell.name if spell else "?",
                )
                return RoutineStatus.SUCCESS
            else:
                self._retries += 1
                log.warning(
                    "[CAST] Buff: cast failed (mana %d->%d, no cost) retry %d/%d",
                    self._mana_before_cast,
                    mana_after,
                    self._retries,
                    MAX_RETRIES,
                )
                return RoutineStatus.RUNNING

        log.info("[CAST] Buff: %s assumed applied (no verification)", spell.name if spell else "?")
        return RoutineStatus.SUCCESS

    @override
    def exit(self, state: GameState) -> None:
        if self._casting:
            log.warning("[CAST] Buff: INTERRUPTED mid-cast (deactivated by brain)")
            self._casting = False
        if self._locked_for_cast:
            log.warning("[CAST] Buff: INTERRUPTED pre-cast (deactivated during stand/settle)")
            self._locked_for_cast = False
