"""Casting state utilities shared across routines.

Instead of blind sleeping through spell cast times, poll the casting_mode
global (0x850CCC) for real-time cast completion. Falls back to timer-based
wait if casting_mode reads aren't available (offset returns 0).
"""

import logging
import time
from collections.abc import Callable
from enum import StrEnum

from core.timing import interruptible_sleep
from perception.state import GameState

log = logging.getLogger(__name__)


class CastResult(StrEnum):
    """Result of a CastingPhase tick or wait_for_cast."""

    CASTING = "casting"
    COMPLETE = "complete"
    TIMEOUT = "timeout"


# Polling interval while waiting for cast to complete
_POLL_INTERVAL = 0.1  # 100ms -- fast enough to catch fizzles, slow enough to not spam


class CastingPhase:
    """Non-blocking cast tracker for use inside routine tick() methods.

    Instead of blocking in a polling loop, the routine creates a CastingPhase
    at cast start and calls tick() each brain tick. The brain keeps evaluating
    emergency rules between ticks, so FLEE can interrupt mid-cast.

    Usage in a routine tick():
        if self._phase == Phase.CASTING:
            result = self._cast_phase.tick()
            if result == CastResult.CASTING:
                return RoutineStatus.RUNNING  # still casting, brain ticks normally
            elif result == CastResult.COMPLETE:
                self._phase = Phase.NEXT_STEP
            elif result == CastResult.TIMEOUT:
                # handle timeout
            return RoutineStatus.RUNNING
    """

    def __init__(
        self,
        cast_time: float,
        label: str,
        read_state_fn: Callable[[], GameState] | None,
        timeout_buffer: float = 0.5,
    ) -> None:
        self._deadline = time.time() + cast_time + timeout_buffer
        self._cast_time = cast_time
        self._label = label
        self._read_state_fn = read_state_fn
        self._started = time.time()
        self._saw_casting = False
        self._tick_count = 0
        # Record baseline casting_mode so we detect transitions, not absolute values.
        # casting_mode can be non-zero at rest (e.g., 6 = UI mode after spellbook).
        self._baseline_mode = 0
        if read_state_fn:
            try:
                self._baseline_mode = read_state_fn().casting_mode
            except (OSError, RuntimeError, AttributeError) as e:
                log.debug("[CAST] Failed to read baseline casting_mode: %s", e)

    def tick(self) -> CastResult:
        """Poll once. Returns CastResult.CASTING, COMPLETE, or TIMEOUT.

        Call this each brain tick from the routine's tick() method.
        Do NOT loop on this -- return RUNNING after each call.
        """
        now = time.time()
        elapsed = now - self._started
        self._tick_count += 1

        if self._read_state_fn is not None:
            state = self._read_state_fn()
            # Detect cast: mode==1 (active spell cast) OR mode changed from baseline
            actively_casting = state.casting_mode == 1 or (
                state.casting_mode != self._baseline_mode and state.casting_mode != 0
            )
            if actively_casting:
                self._saw_casting = True
            elif self._saw_casting:
                # Was casting, now stopped -- cast complete
                log.debug("[CAST] CastingPhase complete: %s (%.1fs)", self._label, elapsed)
                return CastResult.COMPLETE

        if now > self._deadline:
            if self._saw_casting:
                log.debug("[CAST] CastingPhase timeout: %s (saw casting, assuming complete)", self._label)
            else:
                log.debug("[CAST] CastingPhase timeout: %s (timer-based, %.1fs)", self._label, elapsed)
            return CastResult.COMPLETE  # trust the timer

        # Still within cast window
        return CastResult.CASTING

    @property
    def elapsed(self) -> float:
        return time.time() - self._started


def wait_for_cast(
    read_state_fn: Callable[[], GameState] | None,
    cast_time: float,
    timeout_buffer: float = 0.5,
    label: str = "spell",
) -> bool:
    """Wait for a spell cast to complete using casting_mode polling.

    Returns True if cast completed normally, False if it fizzled/interrupted early.
    Falls back to timer-based sleep if read_state_fn is None or casting_mode
    never activates (unverified offset).

    Args:
        read_state_fn: Function that returns current GameState snapshot.
        cast_time: Expected cast time in seconds (from spell data).
        timeout_buffer: Extra seconds to wait beyond cast_time before giving up.
        label: Spell name for logging.
    """
    deadline = time.time() + cast_time + timeout_buffer

    if read_state_fn is None:
        interruptible_sleep(cast_time + 0.3)
        return True

    # Brief initial delay -- casting_mode takes a tick to activate
    interruptible_sleep(0.15)

    # Check if casting_mode is active (confirms the offset works)
    state = read_state_fn()
    if not state.is_casting:
        # casting_mode never activated -- offset might not work yet,
        # fall back to timer-based wait
        remaining = cast_time - 0.15
        if remaining > 0:
            interruptible_sleep(remaining + 0.3)
        return True

    # Poll until cast completes or timeout
    while time.time() < deadline:
        state = read_state_fn()
        if not state.is_casting:
            log.debug(
                "[CAST] Cast complete: %s (casting_mode=0, %.1fs before timeout)",
                label,
                deadline - time.time(),
            )
            return True
        interruptible_sleep(_POLL_INTERVAL)

    log.warning("[CAST] Cast timeout: %s (%.1fs + %.1fs buffer expired)", label, cast_time, timeout_buffer)
    return True  # assume it completed -- timer ran out


def wait_for_cast_interruptible(
    read_state_fn: Callable[[], GameState] | None,
    cast_time: float,
    target_spawn_id: int,
    timeout_buffer: float = 0.5,
    label: str = "spell",
) -> str:
    """Wait for cast, but cancel if target dies or moves out of range.

    Returns:
        "complete" -- cast finished normally
        "cancelled_dead" -- target died, cast was interrupted
        "cancelled_range" -- target moved >200u, cast was interrupted
    """
    from motor.actions import stop_cast

    deadline = time.time() + cast_time + timeout_buffer

    if read_state_fn is None:
        interruptible_sleep(cast_time + 0.3)
        return CastResult.COMPLETE

    interruptible_sleep(0.15)

    state = read_state_fn()
    if not state.is_casting:
        remaining = cast_time - 0.15
        if remaining > 0:
            interruptible_sleep(remaining + 0.3)
        return CastResult.COMPLETE

    while time.time() < deadline:
        state = read_state_fn()
        if not state.is_casting:
            return CastResult.COMPLETE

        # Check if target died or moved out of range
        if state.target and state.target.spawn_id == target_spawn_id:
            if state.target.hp_current <= 0:
                log.info("[CAST] Cast interrupt: %s  -  target dead, cancelling", label)
                stop_cast()
                interruptible_sleep(0.1)
                return "cancelled_dead"
            d = state.pos.dist_to(state.target.pos)
            if d > 200:
                log.info("[CAST] Cast interrupt: %s  -  target out of range (%.0fu)", label, d)
                stop_cast()
                interruptible_sleep(0.1)
                return "cancelled_range"
        elif not state.target:
            # Target gone entirely (despawned)
            log.info("[CAST] Cast interrupt: %s  -  target gone, cancelling", label)
            stop_cast()
            interruptible_sleep(0.1)
            return "cancelled_dead"

        interruptible_sleep(_POLL_INTERVAL)

    return CastResult.COMPLETE
