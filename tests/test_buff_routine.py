"""Tests for BuffRoutine motor-coupled lifecycle.

Verifies enter/tick/exit produce the expected motor commands via
RecordingMotor. No sleeping, no input -- all actions recorded.
"""

from __future__ import annotations

from unittest.mock import patch

from brain.context import AgentContext
from eq.loadout import _ACTIVE_BY_ROLE, Spell, SpellRole
from motor.recording import RecordingMotor
from routines.base import RoutineStatus
from routines.buff import BuffRoutine
from tests.factories import make_game_state


def _noop_sleep(base: float, interrupt_fn=None, poll_interval=0.1, sigma=0.3) -> bool:
    """No-op replacement for interruptible_sleep in tests."""
    return False


def _install_self_buff(gem: int = 4) -> Spell:
    """Install a test self-buff spell into the active loadout."""
    spell = Spell(
        name="Minor Shielding",
        gem=gem,
        cast_time=2.5,
        mana_cost=10,
        spell_id=288,
        role=SpellRole.SELF_BUFF,
    )
    _ACTIVE_BY_ROLE[SpellRole.SELF_BUFF] = spell
    return spell


def _clear_loadout() -> None:
    """Remove test spells from active loadout."""
    _ACTIVE_BY_ROLE.pop(SpellRole.SELF_BUFF, None)


class TestBuffEnter:
    """BuffRoutine.enter() motor actions."""

    @patch("routines.buff.interruptible_sleep", _noop_sleep)
    def test_enter_stands_when_sitting(self, _recording_motor: RecordingMotor) -> None:
        """enter() sends sit_stand to stand up when player is sitting."""
        try:
            _install_self_buff()
            ctx = AgentContext()
            # Player is sitting (stand_state=1)
            state = make_game_state(mana_current=500, mana_max=500, stand_state=1)
            routine = BuffRoutine(ctx=ctx, read_state_fn=lambda: state)
            # Manually set the stance tracker to sitting so stand() fires
            import motor.actions as _ma

            _ma._stance.sitting = True
            routine.enter(state)
            assert "sit_stand" in _recording_motor.actions
        finally:
            _clear_loadout()

    @patch("routines.buff.interruptible_sleep", _noop_sleep)
    def test_enter_no_stand_when_already_standing(self, _recording_motor: RecordingMotor) -> None:
        """enter() does not send sit_stand if player is already standing."""
        try:
            _install_self_buff()
            ctx = AgentContext()
            state = make_game_state(mana_current=500, mana_max=500, stand_state=0)
            routine = BuffRoutine(ctx=ctx, read_state_fn=lambda: state)
            routine.enter(state)
            # No sit_stand because player is not sitting
            assert "sit_stand" not in _recording_motor.actions
        finally:
            _clear_loadout()

    @patch("routines.buff.interruptible_sleep", _noop_sleep)
    def test_enter_resets_phase(self, _recording_motor: RecordingMotor) -> None:
        """enter() resets the internal phase state machine."""
        try:
            _install_self_buff()
            ctx = AgentContext()
            state = make_game_state(mana_current=500, mana_max=500)
            routine = BuffRoutine(ctx=ctx, read_state_fn=lambda: state)
            routine.enter(state)
            from routines.buff import _Phase

            assert routine._phase == _Phase.INIT
            assert routine._retries == 0
            assert routine._casting is False
        finally:
            _clear_loadout()


class TestBuffTick:
    """BuffRoutine.tick() phase state machine."""

    @patch("routines.buff.interruptible_sleep", _noop_sleep)
    def test_tick_returns_failure_without_spell(self, _recording_motor: RecordingMotor) -> None:
        """tick() returns FAILURE when no self-buff spell is configured."""
        _clear_loadout()
        ctx = AgentContext()
        state = make_game_state(mana_current=500, mana_max=500)
        routine = BuffRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine.enter(state)
        status = routine.tick(state)
        assert status == RoutineStatus.FAILURE

    @patch("routines.buff.interruptible_sleep", _noop_sleep)
    def test_tick_init_stands_and_transitions(self, _recording_motor: RecordingMotor) -> None:
        """First tick in INIT phase calls stand() and moves to STAND_WAIT."""
        try:
            _install_self_buff()
            ctx = AgentContext()
            state = make_game_state(mana_current=500, mana_max=500)
            routine = BuffRoutine(ctx=ctx, read_state_fn=lambda: state)
            routine.enter(state)

            # The stance tracker needs to think we're sitting for stand() to send input
            import motor.actions as _ma

            _ma._stance.sitting = True

            _recording_motor.clear()
            status = routine.tick(state)
            assert status == RoutineStatus.RUNNING
            # stand() should have been called
            assert "sit_stand" in _recording_motor.actions
            from routines.buff import _Phase

            assert routine._phase == _Phase.STAND_WAIT
            assert routine._locked_for_cast is True
        finally:
            _clear_loadout()

    @patch("routines.buff.interruptible_sleep", _noop_sleep)
    def test_tick_settle_presses_gem(self, _recording_motor: RecordingMotor) -> None:
        """When SETTLE_WAIT expires, tick() presses the spell gem."""
        try:
            spell = _install_self_buff(gem=4)
            ctx = AgentContext()
            state = make_game_state(mana_current=500, mana_max=500)
            routine = BuffRoutine(ctx=ctx, read_state_fn=lambda: state)
            routine.enter(state)

            # Fast-forward through INIT -> STAND_WAIT -> SETTLE_WAIT -> press gem
            from routines.buff import _Phase

            routine._phase = _Phase.SETTLE_WAIT
            routine._phase_deadline = 0.0  # deadline already passed

            _recording_motor.clear()
            status = routine.tick(state)
            assert status == RoutineStatus.RUNNING
            # Should press the spell gem
            assert f"cast_{spell.gem}" in _recording_motor.actions
            assert routine._phase == _Phase.CASTING
        finally:
            _clear_loadout()

    @patch("routines.buff.interruptible_sleep", _noop_sleep)
    def test_tick_max_retries_fails(self, _recording_motor: RecordingMotor) -> None:
        """tick() returns FAILURE after MAX_RETRIES attempts."""
        try:
            _install_self_buff()
            ctx = AgentContext()
            state = make_game_state(mana_current=500, mana_max=500)
            routine = BuffRoutine(ctx=ctx, read_state_fn=lambda: state)
            routine.enter(state)
            routine._retries = 3  # MAX_RETRIES
            status = routine.tick(state)
            assert status == RoutineStatus.FAILURE
        finally:
            _clear_loadout()


class TestBuffExit:
    """BuffRoutine.exit() state cleanup."""

    @patch("routines.buff.interruptible_sleep", _noop_sleep)
    def test_exit_clears_cast_lock(self, _recording_motor: RecordingMotor) -> None:
        """exit() clears the locked_for_cast flag."""
        try:
            _install_self_buff()
            ctx = AgentContext()
            state = make_game_state(mana_current=500, mana_max=500)
            routine = BuffRoutine(ctx=ctx, read_state_fn=lambda: state)
            routine._locked_for_cast = True
            routine.exit(state)
            assert routine._locked_for_cast is False
        finally:
            _clear_loadout()

    @patch("routines.buff.interruptible_sleep", _noop_sleep)
    def test_exit_clears_casting_flag(self, _recording_motor: RecordingMotor) -> None:
        """exit() clears the casting flag if interrupted mid-cast."""
        try:
            _install_self_buff()
            ctx = AgentContext()
            state = make_game_state(mana_current=500, mana_max=500)
            routine = BuffRoutine(ctx=ctx, read_state_fn=lambda: state)
            routine._casting = True
            routine._locked_for_cast = True
            routine.exit(state)
            assert routine._casting is False
            assert routine._locked_for_cast is False
        finally:
            _clear_loadout()


class TestBuffFullCycle:
    """Full enter -> tick sequence."""

    @patch("routines.buff.interruptible_sleep", _noop_sleep)
    def test_buff_proceeds_through_phases(self, _recording_motor: RecordingMotor) -> None:
        """BuffRoutine progresses INIT -> STAND_WAIT -> SETTLE_WAIT -> CASTING."""
        try:
            spell = _install_self_buff(gem=4)
            ctx = AgentContext()
            state = make_game_state(mana_current=500, mana_max=500)
            routine = BuffRoutine(ctx=ctx, read_state_fn=lambda: state)
            routine.enter(state)

            # Tick 1: INIT -> stand + STAND_WAIT
            import motor.actions as _ma
            from routines.buff import _Phase

            _ma._stance.sitting = True
            status = routine.tick(state)
            assert status == RoutineStatus.RUNNING
            assert routine._phase == _Phase.STAND_WAIT

            # Force deadline to expire
            routine._phase_deadline = 0.0
            status = routine.tick(state)
            assert status == RoutineStatus.RUNNING
            assert routine._phase == _Phase.SETTLE_WAIT

            # Force settle deadline to expire
            routine._phase_deadline = 0.0
            _recording_motor.clear()
            status = routine.tick(state)
            assert status == RoutineStatus.RUNNING
            assert routine._phase == _Phase.CASTING
            assert f"cast_{spell.gem}" in _recording_motor.actions
        finally:
            _clear_loadout()
