"""Tests for routines.combat -- CombatRoutine lifecycle and tick pipeline.

Covers: __init__, enter(), tick() dispatch, _tick_face_and_melee guards,
_check_pet_save_sitting, _tick_melee_retarget, _tick_backstep_completion,
_tick_reactive_backstep, walk state machine, _start_backstep,
_get_pet_combat_status, exit() cleanup, and strategy selection.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from brain.context import AgentContext
from routines.base import RoutineStatus
from routines.combat import CombatRoutine
from routines.strategies.selection import CombatStrategy
from tests.factories import make_game_state, make_spawn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop_sleep(*a, **kw):
    return False


def _make_ctx(**overrides) -> AgentContext:
    """Build an AgentContext with pet alive by default."""
    ctx = AgentContext()
    ctx.pet.alive = overrides.pop("pet_alive", True)
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def _make_combat_state(target=None, **kw):
    """Build a GameState suitable for combat (HP/mana filled, target present)."""
    defaults = dict(
        hp_current=1000,
        hp_max=1000,
        mana_current=500,
        mana_max=500,
    )
    defaults.update(kw)
    if target is not None:
        defaults["target"] = target
    return make_game_state(**defaults)


def _target(**kw):
    """Build a live NPC target spawn at (50, 50)."""
    defaults = dict(spawn_id=100, name="a_skeleton", hp_current=100, hp_max=100, level=10)
    defaults.update(kw)
    return make_spawn(**defaults)


def _make_routine(ctx=None, state=None):
    """Build a CombatRoutine with a state holder for read_state_fn."""
    if ctx is None:
        ctx = _make_ctx()
    if state is None:
        state = _make_combat_state(target=_target())
    holder = [state]
    routine = CombatRoutine(ctx=ctx, read_state_fn=lambda: holder[0])
    return routine, holder


# ---------------------------------------------------------------------------
# 1. __init__ and enter()
# ---------------------------------------------------------------------------


class TestInitAndEnter:
    def test_init_sets_defaults(self):
        routine, _ = _make_routine()
        assert routine._combat_start == 0.0
        assert routine._fight_casts == 0
        assert routine._medding is False
        assert routine._walk_target_id == 0

    def test_enter_sets_combat_start(self):
        ctx = _make_ctx()
        target = _target()
        state = _make_combat_state(target=target)
        routine, holder = _make_routine(ctx=ctx, state=state)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        assert routine._combat_start > 0
        assert routine._fight_target_name == "a_skeleton"
        assert routine._fight_target_id == 100
        assert routine._fight_mana_start == 500

    def test_enter_selects_strategy_pet_tank_low_level(self):
        ctx = _make_ctx()
        target = _target(level=5)
        state = _make_combat_state(target=target, level=5)
        routine, _ = _make_routine(ctx=ctx, state=state)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        assert routine._strategy == CombatStrategy.PET_TANK

    def test_enter_selects_strategy_pet_and_dot_mid_level(self):
        ctx = _make_ctx()
        target = _target(level=10)
        state = _make_combat_state(target=target, level=10)
        routine, _ = _make_routine(ctx=ctx, state=state)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        assert routine._strategy == CombatStrategy.PET_AND_DOT

    def test_enter_stands_from_sit(self, _recording_motor):
        from motor.actions import mark_sitting

        ctx = _make_ctx()
        target = _target()
        sitting_state = _make_combat_state(target=target, stand_state=1)  # sitting
        standing_state = _make_combat_state(target=target, stand_state=0)

        # read_state_fn returns sitting first, then standing after stand() is called
        call_count = [0]

        def _read():
            call_count[0] += 1
            # First read inside verified_stand sees sitting, triggers stand()
            # Second read sees standing
            if call_count[0] <= 1:
                return sitting_state
            return standing_state

        routine = CombatRoutine(ctx=ctx, read_state_fn=_read)
        # Mark motor as sitting so stand() actually sends input
        mark_sitting()

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(sitting_state)

        # verified_stand calls stand() which sends "sit_stand" action
        assert _recording_motor.has_action("sit_stand")

    def test_enter_sends_pet_attack_on_auto_engage(self, _recording_motor):
        ctx = _make_ctx()
        ctx.combat.auto_engaged = True
        target = _target()
        state = _make_combat_state(target=target)
        routine, _ = _make_routine(ctx=ctx, state=state)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        # pet_attack presses hotbar slot 1 -> "hot1_1"
        assert _recording_motor.has_action("hot1_1")

    def test_enter_skips_pet_attack_when_pet_dead(self, _recording_motor):
        ctx = _make_ctx(pet_alive=False)
        ctx.combat.auto_engaged = True
        target = _target()
        state = _make_combat_state(target=target)
        routine, _ = _make_routine(ctx=ctx, state=state)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        # pet_attack would press hotbar slot 1; verify it didn't
        assert not _recording_motor.has_action("hot1_1")

    def test_enter_records_initial_distance(self):
        ctx = _make_ctx()
        target = _target(x=100.0, y=0.0)
        state = _make_combat_state(target=target, x=0.0, y=0.0)
        routine, _ = _make_routine(ctx=ctx, state=state)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        assert routine._fight_initial_dist > 90


# ---------------------------------------------------------------------------
# 2. tick() pipeline dispatch
# ---------------------------------------------------------------------------


class TestTickPipeline:
    def _enter_and_tick(self, routine, state, holder):
        """Enter combat and tick once, returning the status."""
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
        # Give ample tick budget
        routine._tick_deadline = time.perf_counter() + 10.0
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            return routine.tick(state)

    def test_tick_returns_running_during_combat(self):
        ctx = _make_ctx()
        target = _target()
        state = _make_combat_state(target=target)
        routine, holder = _make_routine(ctx=ctx, state=state)

        status = self._enter_and_tick(routine, state, holder)
        assert status == RoutineStatus.RUNNING

    def test_tick_returns_running_when_casting(self):
        ctx = _make_ctx()
        target = _target()
        state = _make_combat_state(target=target, casting_mode=1)
        routine, holder = _make_routine(ctx=ctx, state=state)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
        routine._tick_deadline = time.perf_counter() + 10.0
        routine._cast_end_time = time.time() + 5.0  # pretend we're casting
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            status = routine.tick(state)

        assert status == RoutineStatus.RUNNING

    def test_tick_failure_when_targeting_own_pet(self):
        """If target is our own pet, tick should disengage and fail."""
        ctx = _make_ctx()
        ctx.pet.spawn_id = 200
        pet_spawn = _target(spawn_id=200, name="pet_skeleton")
        state = _make_combat_state(target=pet_spawn)
        routine, holder = _make_routine(ctx=ctx, state=state)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
        routine._tick_deadline = time.perf_counter() + 10.0
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            status = routine.tick(state)

        assert status == RoutineStatus.FAILURE
        assert ctx.combat.engaged is False


# ---------------------------------------------------------------------------
# 3. _tick_face_and_melee dispatcher
# ---------------------------------------------------------------------------


class TestTickFaceAndMelee:
    def _setup(self, **state_kw):
        ctx = _make_ctx()
        target = _target()
        state = _make_combat_state(target=target, **state_kw)
        routine, holder = _make_routine(ctx=ctx, state=state)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
        return routine, state, holder

    def test_guard_casting_returns_running(self):
        """_tick_face_and_melee should return RUNNING when casting."""
        from routines.combat import _TickState

        routine, state, _ = self._setup(casting_mode=1)
        ts = _TickState(target=state.target, dist=30.0, target_hp=0.8, now=time.time())
        result = routine._tick_face_and_melee(state, ts)
        assert result == RoutineStatus.RUNNING

    def test_guard_sitting_returns_running(self):
        """_tick_face_and_melee should return RUNNING when sitting."""
        from routines.combat import _TickState

        routine, _, _ = self._setup()
        sitting_state = _make_combat_state(target=_target(), stand_state=1)
        ts = _TickState(target=sitting_state.target, dist=30.0, target_hp=0.8, now=time.time())
        result = routine._tick_face_and_melee(sitting_state, ts)
        assert result == RoutineStatus.RUNNING

    def test_no_target_returns_none(self):
        """_tick_face_and_melee returns None when target is None."""
        from routines.combat import _TickState

        routine, state, _ = self._setup()
        ts = _TickState(target=None, dist=0, target_hp=0, now=time.time())
        result = routine._tick_face_and_melee(state, ts)
        assert result is None

    def test_walk_progress_intercepts(self):
        """When a walk is active, _tick_face_and_melee returns RUNNING."""
        from routines.combat import _TickState

        routine, state, _ = self._setup()
        # Simulate active walk
        routine._walk_target_id = 100
        routine._walk_deadline = time.time() + 10.0
        routine._walk_close_dist = 10.0
        ts = _TickState(target=state.target, dist=50.0, target_hp=0.8, now=time.time())
        result = routine._tick_face_and_melee(state, ts)
        assert result == RoutineStatus.RUNNING


# ---------------------------------------------------------------------------
# 4. _check_pet_save_sitting
# ---------------------------------------------------------------------------


class TestCheckPetSaveSitting:
    def test_stand_from_med_when_pet_low(self, _recording_motor):
        """Standing from med when pet HP < 35% while sitting."""
        ctx = _make_ctx()
        ctx.pet.spawn_id = 200
        pet_spawn = make_spawn(spawn_id=200, name="pet", hp_current=20, hp_max=100)
        target = _target()
        state = _make_combat_state(
            target=target,
            stand_state=1,  # sitting
            hp_current=900,
            hp_max=1000,
            spawns=(pet_spawn,),
        )
        routine, _ = _make_routine(ctx=ctx, state=state)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
        routine._medding = True
        routine._med_start = time.time() - 5.0
        routine._pet_save_engaged = False
        routine._has_extra_npcs = False

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine._check_pet_save_sitting(state)

        # Should have stood up (medding cleared)
        assert routine._medding is False

    def test_no_stand_when_pet_healthy(self):
        """No action when pet HP is fine."""
        ctx = _make_ctx()
        ctx.pet.spawn_id = 200
        pet_spawn = make_spawn(spawn_id=200, name="pet", hp_current=90, hp_max=100)
        target = _target()
        state = _make_combat_state(
            target=target,
            stand_state=1,
            hp_current=900,
            hp_max=1000,
            spawns=(pet_spawn,),
        )
        routine, _ = _make_routine(ctx=ctx, state=state)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
        routine._medding = True
        routine._pet_save_engaged = False
        routine._has_extra_npcs = False

        routine._check_pet_save_sitting(state)

        assert routine._medding is True  # no change

    def test_no_stand_when_player_hp_low(self):
        """No pet save if player HP < 80%."""
        ctx = _make_ctx()
        ctx.pet.spawn_id = 200
        pet_spawn = make_spawn(spawn_id=200, name="pet", hp_current=20, hp_max=100)
        target = _target()
        state = _make_combat_state(
            target=target,
            stand_state=1,
            hp_current=500,
            hp_max=1000,  # 50% HP
            spawns=(pet_spawn,),
        )
        routine, _ = _make_routine(ctx=ctx, state=state)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
        routine._medding = True
        routine._pet_save_engaged = False
        routine._has_extra_npcs = False

        routine._check_pet_save_sitting(state)

        assert routine._medding is True  # no change


# ---------------------------------------------------------------------------
# 5. _tick_melee_retarget
# ---------------------------------------------------------------------------


class TestTickMeleeRetarget:
    def test_add_adoption_when_has_confirmed_add(self, _recording_motor):
        """If target IS the add (tab cycling), adopt it as primary."""
        from routines.combat import _TickState

        ctx = _make_ctx()
        ctx.pet.has_add = True
        ctx.combat.pull_target_id = 999  # different from current target
        add_target = _target(spawn_id=100, name="an_add")
        state = _make_combat_state(target=add_target)
        routine, _ = _make_routine(ctx=ctx, state=state)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        ts = _TickState(target=add_target, dist=15.0, target_hp=0.8, now=time.time())
        routine._last_retarget_time = 0  # cooldown expired

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            result = routine._tick_melee_retarget(state, ts, None, True)

        assert result == RoutineStatus.RUNNING
        assert ctx.combat.pull_target_id == 100
        assert ctx.pet.has_add is False
        assert routine._fight_retargets == 1

    def test_melee_attacker_switch(self, _recording_motor):
        """Switch to a melee attacker when player is taking damage."""
        from routines.combat import _TickState

        ctx = _make_ctx()
        target = _target()
        attacker = make_spawn(spawn_id=200, name="an_attacker", x=5.0, y=5.0, hp_current=80, hp_max=100)
        state = _make_combat_state(target=target, hp_current=800, hp_max=1000)
        routine, holder = _make_routine(ctx=ctx, state=state)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
        routine._hp_at_start = 1.0  # started at full HP
        routine._last_retarget_time = 0

        ts = _TickState(target=target, dist=30.0, target_hp=0.8, now=time.time())
        # hp_pct=0.8, hp_at_start=1.0 -> taking damage

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            result = routine._tick_melee_retarget(state, ts, attacker, False)

        assert result == RoutineStatus.RUNNING
        assert routine._fight_retargets == 1

    def test_no_retarget_during_cooldown(self):
        """No retarget within cooldown period."""
        from routines.combat import _TickState

        ctx = _make_ctx()
        target = _target()
        attacker = make_spawn(spawn_id=200, name="an_attacker", x=5.0, y=5.0)
        state = _make_combat_state(target=target, hp_current=800, hp_max=1000)
        routine, _ = _make_routine(ctx=ctx, state=state)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
        routine._hp_at_start = 1.0
        routine._last_retarget_time = time.time()  # just retargeted

        now = time.time()
        ts = _TickState(target=target, dist=30.0, target_hp=0.8, now=now)

        result = routine._tick_melee_retarget(state, ts, attacker, False)
        assert result is None


# ---------------------------------------------------------------------------
# 6. _tick_backstep_completion
# ---------------------------------------------------------------------------


class TestTickBackstepCompletion:
    def test_backstep_complete_when_target_reached(self, _recording_motor):
        """Backstep completes when we moved far enough."""
        from routines.combat import _TickState

        ctx = _make_ctx()
        target = _target()
        # Player moved 30 units from backstep start
        state = _make_combat_state(target=target, x=30.0, y=0.0)
        routine, _ = _make_routine(ctx=ctx, state=state)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        routine._backstep_active = True
        routine._backstep_start_x = 0.0
        routine._backstep_start_y = 0.0
        routine._backstep_target = 25.0
        routine._backstep_deadline = time.time() + 5.0
        now = time.time()
        ts = _TickState(target=target, dist=50.0, target_hp=0.8, now=now)

        result = routine._tick_backstep_completion(state, ts)

        assert result == RoutineStatus.RUNNING
        assert routine._backstep_active is False
        assert routine._fight_backsteps == 1
        # move_backward_stop sends "-back" action
        assert "-back" in _recording_motor.actions

    def test_backstep_not_active_returns_none(self):
        """No backstep active -> return None."""
        from routines.combat import _TickState

        routine, _ = _make_routine()
        state = _make_combat_state(target=_target())
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
        ts = _TickState(target=state.target, dist=30.0, target_hp=0.8, now=time.time())
        result = routine._tick_backstep_completion(state, ts)
        assert result is None

    def test_backstep_extends_when_still_in_melee(self, _recording_motor):
        """Backstep extends if we reached target distance but still close."""
        from routines.combat import _TickState

        ctx = _make_ctx()
        target = _target(x=8.0, y=0.0)  # target is only 8u away
        state = _make_combat_state(target=target, x=25.0, y=0.0)
        routine, _ = _make_routine(ctx=ctx, state=state)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        routine._backstep_active = True
        routine._backstep_start_x = 0.0
        routine._backstep_start_y = 0.0
        routine._backstep_target = 20.0  # moved 25u > 20u target
        now = time.time()
        routine._backstep_deadline = now + 5.0
        ts = _TickState(target=target, dist=10.0, target_hp=0.8, now=now)  # dist < 15

        result = routine._tick_backstep_completion(state, ts)

        assert result == RoutineStatus.RUNNING
        # Should still be active (extended)
        assert routine._backstep_active is True


# ---------------------------------------------------------------------------
# 7. _tick_reactive_backstep
# ---------------------------------------------------------------------------


class TestTickReactiveBackstep:
    def test_backstep_triggered_by_melee_hit(self, _recording_motor):
        """Reactive backstep fires when target is in melee range."""
        from routines.combat import _TickState

        ctx = _make_ctx()
        ctx.pet.spawn_id = 200
        pet_spawn = make_spawn(spawn_id=200, name="pet", hp_current=80, hp_max=100, x=20.0, y=0.0)
        target = _target(x=10.0, y=0.0)
        state = _make_combat_state(target=target, x=0.0, y=0.0, spawns=(pet_spawn,))
        routine, _ = _make_routine(ctx=ctx, state=state)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        routine._hp_at_start = 1.0
        routine._last_backstep_time = 0
        routine._combat_backstep_count = 0

        now = time.time()
        ts = _TickState(target=target, dist=10.0, target_hp=0.8, now=now)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            result = routine._tick_reactive_backstep(state, ts, None, False)

        assert result == RoutineStatus.RUNNING
        assert routine._backstep_active is True
        # move_backward_start sends "+back"
        assert "+back" in _recording_motor.actions

    def test_no_backstep_when_cap_reached(self, _recording_motor):
        """No backstep when backstep cap (2) is reached."""
        from routines.combat import _TickState

        ctx = _make_ctx()
        ctx.pet.spawn_id = 200
        pet_spawn = make_spawn(spawn_id=200, name="pet", hp_current=80, hp_max=100, x=20.0, y=0.0)
        target = _target(x=10.0, y=0.0)
        state = _make_combat_state(target=target, x=0.0, y=0.0, spawns=(pet_spawn,))
        routine, _ = _make_routine(ctx=ctx, state=state)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        routine._hp_at_start = 1.0
        routine._last_backstep_time = 0
        routine._combat_backstep_count = 2  # at cap

        now = time.time()
        ts = _TickState(target=target, dist=10.0, target_hp=0.8, now=now)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine._tick_reactive_backstep(state, ts, None, False)

        # Should NOT start a backstep
        assert routine._backstep_active is False


# ---------------------------------------------------------------------------
# 8. Walk state machine
# ---------------------------------------------------------------------------


class TestWalkStateMachine:
    def test_start_walk_toward(self, _recording_motor):
        """_start_walk_toward sets walk state and starts forward movement."""
        routine, _ = _make_routine()
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(_make_combat_state(target=_target()))
        routine._start_walk_toward(100, close_dist=20.0, settle=0.5)

        assert routine._walk_target_id == 100
        assert routine._walk_close_dist == 20.0
        assert routine._walk_settle_duration == 0.5
        assert routine._walk_deadline > time.time()
        # move_forward_start sends "+forward"
        assert "+forward" in _recording_motor.actions

    def test_tick_walk_progress_arrived(self, _recording_motor):
        """Walk completes when close enough to the target spawn."""
        routine, holder = _make_routine()
        target = _target(spawn_id=100, x=15.0, y=0.0)
        state = _make_combat_state(target=target, x=10.0, y=0.0, spawns=(target,))
        holder[0] = state
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        routine._start_walk_toward(100, close_dist=10.0)

        result = routine._tick_walk_progress(state)
        assert result == RoutineStatus.RUNNING
        # Walk should complete (distance ~5 < 10)
        # Without settle, walk_target_id resets to 0
        assert routine._walk_target_id == 0

    def test_tick_walk_progress_timeout(self, _recording_motor):
        """Walk times out when deadline exceeded."""
        routine, _ = _make_routine()
        state = _make_combat_state(target=_target())
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        routine._walk_target_id = 999  # spawn not in state.spawns
        routine._walk_close_dist = 10.0
        routine._walk_deadline = time.time() - 1.0  # already expired
        routine._walk_settle_duration = 0.0

        result = routine._tick_walk_progress(state)
        assert result == RoutineStatus.RUNNING
        assert routine._walk_target_id == 0  # cleared

    def test_tick_walk_not_active_returns_none(self):
        """No walk active -> return None."""
        routine, _ = _make_routine()
        state = _make_combat_state(target=_target())
        result = routine._tick_walk_progress(state)
        assert result is None

    def test_finish_walk_updates_enter_pos(self):
        """_finish_walk updates _enter_pos when flag set."""
        routine, holder = _make_routine()
        state = _make_combat_state(target=_target(), x=100.0, y=200.0)
        holder[0] = state
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
        routine._walk_target_id = 100
        routine._walk_update_enter_pos = True

        routine._finish_walk()

        assert routine._walk_target_id == 0
        assert routine._enter_pos == (100.0, 200.0)

    def test_walk_settle_phase(self, _recording_motor):
        """Walk enters settle phase after arrival when settle > 0."""
        routine, holder = _make_routine()
        target = _target(spawn_id=100, x=5.0, y=0.0)
        state = _make_combat_state(target=target, x=0.0, y=0.0, spawns=(target,))
        holder[0] = state
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        routine._start_walk_toward(100, close_dist=10.0, settle=0.5)
        # First tick: arrive, enter settle phase
        result = routine._tick_walk_progress(state)
        assert result == RoutineStatus.RUNNING
        # Should be in settle phase now (walk_settle_until > 0)
        assert routine._walk_settle_until > 0

        # Second tick before settle expires: still RUNNING
        result = routine._tick_walk_progress(state)
        assert result == RoutineStatus.RUNNING

        # Fast-forward past settle
        routine._walk_settle_until = time.time() - 0.1
        result = routine._tick_walk_progress(state)
        assert result == RoutineStatus.RUNNING
        assert routine._walk_target_id == 0  # finished


# ---------------------------------------------------------------------------
# 9. _start_backstep
# ---------------------------------------------------------------------------


class TestStartBackstep:
    def test_sets_backstep_state(self, _recording_motor):
        routine, _ = _make_routine()
        state = _make_combat_state(target=_target(), x=10.0, y=20.0)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        now = time.time()
        routine._start_backstep(state, now, 25.0)

        assert routine._backstep_active is True
        assert routine._backstep_start_x == 10.0
        assert routine._backstep_start_y == 20.0
        assert routine._backstep_target == 25.0
        assert routine._backstep_deadline > now
        # move_backward_start sends "+back"
        assert "+back" in _recording_motor.actions


# ---------------------------------------------------------------------------
# 10. _get_pet_combat_status
# ---------------------------------------------------------------------------


class TestGetPetCombatStatus:
    def test_returns_distance_and_hp(self):
        ctx = _make_ctx()
        ctx.pet.spawn_id = 200
        pet_spawn = make_spawn(spawn_id=200, name="pet", x=30.0, y=0.0, hp_current=60, hp_max=100)
        state = _make_combat_state(target=_target(), x=0.0, y=0.0, spawns=(pet_spawn,))
        routine, _ = _make_routine(ctx=ctx, state=state)

        dist, hp_pct = routine._get_pet_combat_status(state)

        assert abs(dist - 30.0) < 1.0
        assert abs(hp_pct - 0.6) < 0.01

    def test_no_pet_returns_defaults(self):
        ctx = _make_ctx()
        ctx.pet.spawn_id = 0  # no pet id
        state = _make_combat_state(target=_target())
        routine, _ = _make_routine(ctx=ctx, state=state)

        dist, hp_pct = routine._get_pet_combat_status(state)

        assert dist == 9999.0
        assert hp_pct == -1.0

    def test_pet_not_in_spawns_returns_defaults(self):
        ctx = _make_ctx()
        ctx.pet.spawn_id = 200
        state = _make_combat_state(target=_target(), spawns=())
        routine, _ = _make_routine(ctx=ctx, state=state)

        dist, hp_pct = routine._get_pet_combat_status(state)

        assert dist == 9999.0
        assert hp_pct == -1.0


# ---------------------------------------------------------------------------
# 11. exit() cleanup
# ---------------------------------------------------------------------------


class TestExit:
    def test_exit_clears_combat_state(self):
        ctx = _make_ctx()
        ctx.combat.engaged = True
        ctx.combat.pull_target_id = 100
        target = _target()
        state = _make_combat_state(target=target)
        routine, _ = _make_routine(ctx=ctx, state=state)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
            routine.exit(state)

        assert ctx.combat.engaged is False
        assert ctx.combat.pull_target_id is None
        assert ctx.pet.has_add is False

    def test_exit_stops_active_walk(self, _recording_motor):
        ctx = _make_ctx()
        target = _target()
        state = _make_combat_state(target=target)
        routine, _ = _make_routine(ctx=ctx, state=state)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
        routine._walk_target_id = 100

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.exit(state)

        assert routine._walk_target_id == 0
        # move_forward_stop sends "-forward"
        assert "-forward" in _recording_motor.actions

    def test_exit_stops_active_backstep(self, _recording_motor):
        ctx = _make_ctx()
        target = _target()
        state = _make_combat_state(target=target)
        routine, _ = _make_routine(ctx=ctx, state=state)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
        routine._backstep_active = True

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.exit(state)

        assert routine._backstep_active is False
        # move_backward_stop sends "-back"
        assert "-back" in _recording_motor.actions

    def test_exit_records_fight_summary(self):
        ctx = _make_ctx()
        target = _target()
        state = _make_combat_state(target=target)
        routine, _ = _make_routine(ctx=ctx, state=state)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
            routine.exit(state)

        summary = routine.last_fight_summary
        assert "duration" in summary
        assert "casts" in summary
        assert "strategy" in summary
        assert summary["casts"] == 0
        assert summary["strategy"] == routine._strategy.value

    def test_exit_infers_fast_defeat_when_target_gone(self):
        """If target vanished and fight had casts, exit infers a defeat."""
        ctx = _make_ctx()
        target = _target()
        state_with_target = _make_combat_state(target=target)
        routine, _ = _make_routine(ctx=ctx, state=state_with_target)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state_with_target)

        routine._fight_casts = 3  # we cast spells -> target took damage
        routine._combat_start = time.time() - 10.0  # fight lasted 10s

        # Exit with no target in state (it died between ticks)
        state_no_target = _make_combat_state()  # target=None
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.exit(state_no_target)

        assert routine._target_killed is True

    def test_exit_does_not_infer_defeat_when_no_damage(self):
        """If target vanished but no damage dealt, it's not a defeat."""
        ctx = _make_ctx()
        target = _target()
        state_with_target = _make_combat_state(target=target)
        routine, _ = _make_routine(ctx=ctx, state=state_with_target)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state_with_target)

        routine._fight_casts = 0
        routine._combat_start = time.time() - 0.2  # very short

        state_no_target = _make_combat_state()
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.exit(state_no_target)

        assert routine._target_killed is False

    def test_exit_cleans_up_active_phase(self):
        """Exit cleans up active non-blocking phase."""
        ctx = _make_ctx()
        target = _target()
        state = _make_combat_state(target=target)
        routine, _ = _make_routine(ctx=ctx, state=state)

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)
        routine._phase_mgr.phase = "PET_RECALL"

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.exit(state)

        assert routine._phase_mgr.phase == ""


# ---------------------------------------------------------------------------
# 12. _target_hp_pct
# ---------------------------------------------------------------------------


class TestTargetHpPct:
    def test_normal_hp(self):
        routine, _ = _make_routine()
        target = _target(hp_current=50, hp_max=100)
        assert abs(routine._target_hp_pct(target) - 0.5) < 0.01

    def test_zero_max_hp(self):
        routine, _ = _make_routine()
        target = _target(hp_current=0, hp_max=0)
        assert routine._target_hp_pct(target) == 1.0


# ---------------------------------------------------------------------------
# 13. _vitals format
# ---------------------------------------------------------------------------


class TestVitals:
    def test_vitals_with_pet_alive(self):
        ctx = _make_ctx()
        state = _make_combat_state(hp_current=500, hp_max=1000, mana_current=200, mana_max=500)
        routine, _ = _make_routine(ctx=ctx, state=state)
        v = routine._vitals(state)
        assert "HP=50%" in v
        assert "Mana=40%" in v
        assert "Pet=" in v

    def test_vitals_with_pet_dead(self):
        ctx = _make_ctx(pet_alive=False)
        state = _make_combat_state()
        routine, _ = _make_routine(ctx=ctx, state=state)
        v = routine._vitals(state)
        assert "Pet=dead" in v


# ---------------------------------------------------------------------------
# 14. _find_melee_attacker
# ---------------------------------------------------------------------------


class TestFindMeleeAttacker:
    def test_finds_npc_in_melee_range(self):
        ctx = _make_ctx()
        target = _target(spawn_id=100)
        attacker = make_spawn(spawn_id=200, name="attacker", x=5.0, y=0.0, hp_current=80, hp_max=100)
        state = _make_combat_state(
            target=target,
            x=0.0,
            y=0.0,
            spawns=(target, attacker),
        )
        routine, _ = _make_routine(ctx=ctx, state=state)

        found = routine._find_melee_attacker(state, target)
        assert found is not None
        assert found.spawn_id == 200

    def test_excludes_current_target(self):
        ctx = _make_ctx()
        target = _target(spawn_id=100, x=5.0, y=0.0)
        state = _make_combat_state(
            target=target,
            x=0.0,
            y=0.0,
            spawns=(target,),
        )
        routine, _ = _make_routine(ctx=ctx, state=state)

        found = routine._find_melee_attacker(state, target)
        assert found is None

    def test_returns_none_when_no_npcs_nearby(self):
        ctx = _make_ctx()
        target = _target(spawn_id=100, x=100.0, y=100.0)
        far_npc = make_spawn(spawn_id=200, name="far_npc", x=500.0, y=500.0, hp_current=100, hp_max=100)
        state = _make_combat_state(
            target=target,
            x=0.0,
            y=0.0,
            spawns=(target, far_npc),
        )
        routine, _ = _make_routine(ctx=ctx, state=state)

        found = routine._find_melee_attacker(state, target)
        assert found is None


# ---------------------------------------------------------------------------
# 15. _budget_sleep
# ---------------------------------------------------------------------------


class TestBudgetSleep:
    def test_skips_when_budget_spent(self):
        """Returns False immediately when tick budget is spent."""
        routine, _ = _make_routine()
        routine._tick_deadline = time.perf_counter() - 1.0  # expired
        routine._flee_check = None
        result = routine._budget_sleep(1.0)
        assert result is False

    def test_calls_interruptible_sleep_when_budget_remains(self):
        routine, _ = _make_routine()
        routine._tick_deadline = time.perf_counter() + 10.0
        routine._flee_check = None
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            result = routine._budget_sleep(0.5)
        assert result is False


# ---------------------------------------------------------------------------
# 16. _tick_dot_verify
# ---------------------------------------------------------------------------


class TestTickDotVerify:
    def test_no_pending_dot_returns_none(self):
        from routines.combat import _TickState

        routine, _ = _make_routine()
        state = _make_combat_state(target=_target())
        ts = _TickState(target=state.target, dist=30.0, target_hp=0.8, now=time.time())
        assert routine._tick_dot_verify(state, ts) is None

    def test_dot_confirmed_when_mana_dropped(self):
        from routines.combat import _TickState

        ctx = _make_ctx()
        target = _target()
        # Mana dropped from 500 to 450 (50 mana cost)
        state = _make_combat_state(target=target, mana_current=450, mana_max=500)
        routine, _ = _make_routine(ctx=ctx, state=state)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        cast_time = time.time() - 3.0  # 3s ago
        routine._pending_dot_cast = (cast_time, 500)

        ts = _TickState(target=target, dist=30.0, target_hp=0.8, now=time.time())
        result = routine._tick_dot_verify(state, ts)

        assert result is None  # confirmed, continue
        assert routine._pending_dot_cast is None
        assert ctx.combat.last_dot_time == cast_time

    def test_dot_fizzle_when_mana_unchanged(self):
        from routines.combat import _TickState

        ctx = _make_ctx()
        target = _target()
        state = _make_combat_state(target=target, mana_current=500, mana_max=500)
        routine, _ = _make_routine(ctx=ctx, state=state)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(state)

        cast_time = time.time() - 3.0
        routine._pending_dot_cast = (cast_time, 500)  # mana didn't drop

        ts = _TickState(target=target, dist=30.0, target_hp=0.8, now=time.time())

        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            result = routine._tick_dot_verify(state, ts)

        assert result is None  # fizzle handled, continues
        assert routine._pending_dot_cast is None
        assert routine._retry_after_fizzle is True
        assert routine._combat_fizzle_count == 1


# ---------------------------------------------------------------------------
# 17. Strategy switching
# ---------------------------------------------------------------------------


class TestStrategySwitch:
    def test_switch_strategy_changes_impl(self):
        ctx = _make_ctx()
        routine, _ = _make_routine(ctx=ctx)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(_make_combat_state(target=_target()))

        routine._switch_strategy(CombatStrategy.PET_TANK, "test reason")

        assert routine._strategy == CombatStrategy.PET_TANK
        assert ctx.combat.active_strategy == "pet_tank"

    def test_switch_strategy_noop_same(self):
        ctx = _make_ctx()
        routine, _ = _make_routine(ctx=ctx)
        with patch("routines.combat.interruptible_sleep", _noop_sleep):
            routine.enter(_make_combat_state(target=_target()))

        original = routine._strategy
        routine._switch_strategy(original, "same strategy")
        # No change
        assert routine._strategy == original
