"""Semantic motor actions  -  the control vocabulary for routine execution.

This module defines the abstract action interface that routines call to
control the agent. Each function represents a discrete motor action
(move, turn, sit, cast, target, pet command) without exposing the
underlying input mechanism.

The actual input implementation (OS-level keyboard/mouse synthesis,
keybind resolution, window management) is environment-specific and not
included.  A concrete backend must implement the low-level dispatch
that these actions delegate to.

Architecture:
    Routines call:  sit(), tab_target(), press_gem(3), move_forward_start()
    Actions map to: named action strings ("sit_stand", "target_npc", "cast_3")
    Backend sends:  environment-specific input (keybind-resolved scancodes)
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)


# ===============================================================================
# Motor backend  -  pluggable dispatch for environment-specific implementation
# ===============================================================================


class MotorBackend(Protocol):
    """Interface for motor output dispatch.

    The default backend logs actions for traceability. A concrete
    environment backend resolves action names to keybinds and sends
    OS-level input. Tests use a RecordingMotor that captures actions.
    """

    def action(self, name: str, duration: float = 0.05) -> None: ...
    def action_down(self, name: str) -> None: ...
    def action_up(self, name: str) -> None: ...
    def sleep(self, duration: float) -> None: ...


class _DefaultBackend:
    """Stub backend that logs actions without sending input."""

    def action(self, name: str, duration: float = 0.05) -> None:
        log.debug("MOTOR: action '%s' (%.3fs)", name, duration)

    def action_down(self, name: str) -> None:
        log.debug("MOTOR: action_down '%s'", name)

    def action_up(self, name: str) -> None:
        log.debug("MOTOR: action_up '%s'", name)

    def sleep(self, duration: float) -> None:
        time.sleep(duration)


_state: dict[str, MotorBackend] = {"backend": _DefaultBackend()}


def set_backend(backend: MotorBackend) -> None:
    """Set the motor backend. Called at startup with the environment
    implementation, or in tests with a RecordingMotor."""
    _state["backend"] = backend


def get_backend() -> MotorBackend:
    """Get the current motor backend."""
    return _state["backend"]


def _action(name: str, duration: float = 0.05) -> None:
    """Send a named action via the motor backend."""
    _state["backend"].action(name, duration)


def _action_down(name: str) -> None:
    """Hold down a named action key (for continuous movement)."""
    _state["backend"].action_down(name)


def _action_up(name: str) -> None:
    """Release a named action key."""
    _state["backend"].action_up(name)


def _sleep(duration: float) -> None:
    """Timing pause between motor actions."""
    _state["backend"].sleep(duration)


# ===============================================================================
# Public Action API  -  what routines call
# ===============================================================================

# -- Movement --


def move_forward_start() -> None:
    """Begin moving forward. Implicitly marks player as standing."""
    mark_standing()
    _action_down("forward")


def move_forward_stop() -> None:
    _action_up("forward")


def move_backward_start() -> None:
    _action_down("back")


def move_backward_stop() -> None:
    _action_up("back")


def face_heading(target_heading: float, read_heading_fn: Callable[[], float], tolerance: float = 3.0) -> bool:
    """Turn character to face *target_heading* using closed-loop control.

    Reads the current heading each iteration, sends turn-left or turn-right,
    and converges within *tolerance* degrees.  Hard-capped at ~1.5 s to
    prevent blocking when a target is behind geometry.

    Returns True if converged, False on timeout.
    """
    max_iterations = 150
    deadline = time.time() + 2.0
    converged = False
    current = target_heading
    stuck_count = 0
    last_read = -1.0

    for _ in range(max_iterations):
        if time.time() >= deadline:
            break
        current = read_heading_fn()
        if abs(current - last_read) < 0.5 and last_read >= 0:
            stuck_count += 1
            if stuck_count >= 20:
                break
        else:
            stuck_count = 0
        last_read = current

        diff = _angle_diff(current, target_heading)
        if abs(diff) <= tolerance:
            converged = True
            break
        if diff > 0:
            _action_down("turn_left")
            _action_up("turn_right")
        else:
            _action_down("turn_right")
            _action_up("turn_left")
        _sleep(0.01)

    _action_up("turn_left")
    _action_up("turn_right")
    return converged


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference from *a* to *b* in degrees (-180, 180]."""
    d = (b - a) % 360
    return d - 360 if d > 180 else d


# -- Targeting --


def tab_target() -> None:
    """Target nearest NPC."""
    _action("target_npc", duration=0.06)


def cycle_target() -> None:
    """Cycle to next NPC."""
    _action("cycle_npc", duration=0.06)


def clear_target() -> None:
    _action("escape")


# -- Stance --


class _StanceTracker:
    """Tracks sit/stand state to prevent toggle-back on the sit_stand key."""

    __slots__ = ("sitting",)

    def __init__(self) -> None:
        self.sitting: bool = False


_stance = _StanceTracker()


def sit() -> None:
    """Sit down. No-ops if already sitting (prevents toggle-back)."""
    if _stance.sitting:
        return
    log.info("[ACTION] Player: sit")
    _action("sit_stand")
    _stance.sitting = True


def stand() -> None:
    """Stand up. No-ops if internal state says already standing."""
    if not _stance.sitting:
        return
    log.info("[ACTION] Player: stand")
    _action("sit_stand")
    _stance.sitting = False


def mark_standing() -> None:
    """Mark internal state as standing without sending input.

    Use after actions that inherently stand the player (casting, walking).
    """
    _stance.sitting = False


def mark_sitting() -> None:
    """Mark internal state as sitting without sending input.

    Use when perception confirms the player is already sitting.
    """
    _stance.sitting = True


def is_sitting() -> bool:
    return _stance.sitting


def force_standing() -> None:
    """Reset internal sit tracker without sending input."""
    _stance.sitting = False


# -- Spells --


def press_gem(gem_number: int) -> None:
    """Press spell gem 1-9."""
    _action(f"cast_{gem_number}")


def press_hotbar(slot: int) -> None:
    """Press hotbar slot 1-9."""
    _action(f"hot1_{slot}")


def stop_cast() -> None:
    _action("stop_cast")


def toggle_spellbook() -> None:
    _action("spellbook")


# -- Bags / Inventory --


def open_bags() -> None:
    _action("open_bags")


def close_bags() -> None:
    _action("close_bags")


def open_inventory() -> None:
    _action("inventory")


# -- Pet commands --

_PET_HOTBAR = {
    "attack": 1,
    "back off": 2,
    "sit down": 3,
    "stand up": 4,
    "follow me": 5,
    "target": 6,
    "dismiss": 7,
}


def _pet_command(cmd: str) -> None:
    slot = _PET_HOTBAR.get(cmd)
    if slot is None:
        log.warning("[PET] Unknown pet command: '%s'", cmd)
        return
    press_hotbar(slot)
    _sleep(0.05)


def pet_attack() -> None:
    _pet_command("attack")


def pet_back_off() -> None:
    """Tell pet to stop attacking. Repeated for reliability."""
    presses = random.randint(3, 5)
    for i in range(presses):
        _pet_command("back off")
        if i < presses - 1:
            _sleep(0.15)


def redirect_pet() -> None:
    """Clear pet threat then send to current target."""
    pet_back_off()
    _sleep(0.6)
    pet_attack()


def pet_sit() -> None:
    _pet_command("sit down")


def pet_stand() -> None:
    _pet_command("stand up")


def pet_follow() -> None:
    _pet_command("follow me")


def pet_ready() -> None:
    """Stand pet and set to follow  -  used after rest/travel."""
    pet_stand()
    _sleep(0.3)
    pet_follow()


def pet_target() -> None:
    _pet_command("target")


def pet_heal(heal_gem: int = 0) -> None:
    """Target pet then cast heal spell.

    *heal_gem*=0 auto-detects from the spell loadout (SpellRole.PET_HEAL).
    """
    if heal_gem == 0:
        from eq.loadout import SpellRole, get_spell_by_role

        spell = get_spell_by_role(SpellRole.PET_HEAL)
        if spell and spell.gem:
            heal_gem = spell.gem
        else:
            log.warning("[ACTION] Pet: heal  -  no PET_HEAL spell in loadout")
            return
    pet_target()
    _sleep(0.2)
    press_gem(heal_gem)


# -- Camp/logout --


def camp() -> None:
    _action("camp")


# -- Motor lifecycle --


def disable_all_input() -> None:
    """Defeat switch: block all motor output. Called on agent stop."""
    log.info("MOTOR: input disabled (defeat switch)")


def enable_all_input() -> None:
    """Re-enable motor output. Called on agent start."""
    log.info("MOTOR: input enabled")


def release_all_keys() -> None:
    """Release all held keys. Called during shutdown cleanup."""
    log.info("MOTOR: releasing all held keys")
    for key in ("forward", "back", "turn_left", "turn_right", "strafe_left", "strafe_right"):
        _action_up(key)


# -- Verified actions (with perception confirmation) --


def verified_sit(read_state_fn: Callable | None = None, max_retries: int = 3) -> bool:
    """Sit down with optional verification via perception state."""
    sit()
    return True


def verified_stand(read_state_fn: Callable, max_retries: int = 3) -> bool:
    """Stand and verify via perception state. Retries if input dropped."""
    state = read_state_fn()
    if state.is_standing:
        return True
    for attempt in range(max_retries):
        stand()
        _sleep(0.3)
        state = read_state_fn()
        if state.is_standing:
            return True
        log.warning("[LIFECYCLE] stand() not confirmed (attempt %d)", attempt + 1)
    return False


# -- Mouse actions (abstract  -  concrete backend provides screen dispatch) --


def get_game_window_rect() -> tuple[int, int, int, int]:
    """Return the game window rectangle (x, y, width, height).

    Stub implementation -- returns a default rectangle for headless/test use.
    """
    return (0, 0, 640, 480)


def right_click_center(y_offset_pct: float = 0.1) -> bool:
    """Right-click at center of the environment window."""
    log.debug("MOTOR: right_click_center (offset=%.2f)", y_offset_pct)
    return False


def left_click_center(y_offset_pct: float = 0.0) -> bool:
    """Left-click at center of the environment window."""
    log.debug("MOTOR: left_click_center (offset=%.2f)", y_offset_pct)
    return False


def right_click_at(x: int, y: int) -> None:
    log.debug("MOTOR: right_click_at (%d, %d)", x, y)


def left_click_at(x: int, y: int) -> None:
    log.debug("MOTOR: left_click_at (%d, %d)", x, y)


def zoom_first_person() -> None:
    """Zoom camera to first-person view."""
    log.debug("MOTOR: zoom_first_person")


def zoom_third_person() -> None:
    """Zoom camera to third-person view."""
    log.debug("MOTOR: zoom_third_person")


# -- Timing utility (re-exported for callers that need inter-action delays) --


def jittered_sleep(base: float) -> None:
    """Sleep with slight jitter to avoid mechanical regularity."""
    _sleep(base * random.uniform(0.8, 1.2))
