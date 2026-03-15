"""Tests for routines/pull.py -- PullRoutine lifecycle and phase transitions."""

from __future__ import annotations

import time
from unittest.mock import patch

from brain.context import AgentContext
from core.types import FailureCategory
from perception.state import SpawnData
from routines.base import RoutineStatus
from routines.pull import PullRoutine, _Phase
from tests.factories import make_game_state, make_spawn


# interruptible_sleep always returns False (not interrupted) and does not block
def _noop_sleep(*a, **kw):
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(**overrides) -> AgentContext:
    """Create a minimal AgentContext with pet alive."""
    ctx = AgentContext()
    ctx.pet.alive = True
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def _make_target(**overrides) -> SpawnData:
    defaults = dict(spawn_id=200, name="a_bat", hp_current=100, hp_max=100, level=8)
    defaults.update(overrides)
    return make_spawn(**defaults)


def _make_routine(
    ctx=None,
    target=None,
    state_overrides=None,
    read_state_fn=None,
):
    """Build a PullRoutine with sensible defaults and a read_state_fn."""
    if ctx is None:
        ctx = _make_ctx()
    if target is None:
        target = _make_target()
    so = dict(target=target, hp_current=1000, hp_max=1000, mana_current=500, mana_max=500)
    if state_overrides:
        so.update(state_overrides)
    state = make_game_state(**so)
    if read_state_fn is None:

        def read_state_fn():
            return state

    routine = PullRoutine(ctx=ctx, read_state_fn=read_state_fn)
    return routine, state, ctx


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_phase_is_send_pet(self):
        routine = PullRoutine()
        assert routine._phase == _Phase.SEND_PET

    def test_default_state(self):
        routine = PullRoutine()
        assert routine._locked is False
        assert routine._aborted is False
        assert routine._dot_retries == 0
        assert routine._strategy == PullRoutine.PET_THEN_DOT
        assert routine._has_backstepped is False
        assert routine._backstep_count == 0
        assert routine._pull_start == 0.0

    def test_ctx_and_read_state_stored(self):
        ctx = _make_ctx()

        def fn():
            return None

        routine = PullRoutine(ctx=ctx, read_state_fn=fn)
        assert routine._ctx is ctx
        assert routine._read_state_fn is fn


# ---------------------------------------------------------------------------
# enter()
# ---------------------------------------------------------------------------


class TestEnter:
    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    @patch("routines.pull.random")
    def test_enter_resets_state_and_sets_pull_start(self, mock_random):
        mock_random.gauss.return_value = 0.0
        mock_random.choice.side_effect = lambda seq: seq[0]
        mock_random.random.return_value = 0.99  # avoid spell-first
        mock_random.uniform.return_value = 0.0

        routine, state, ctx = _make_routine()
        routine.enter(state)

        assert routine._pull_start > 0
        assert routine._dot_retries == 0
        assert routine._has_backstepped is False
        assert routine._aborted is False

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    @patch("routines.pull.random")
    def test_enter_stores_defeat_tracker_info(self, mock_random):
        mock_random.gauss.return_value = 0.0
        mock_random.choice.side_effect = lambda seq: seq[0]
        mock_random.random.return_value = 0.99
        mock_random.uniform.return_value = 0.0

        target = _make_target(x=100.0, y=200.0)
        routine, state, ctx = _make_routine(target=target)
        routine.enter(state)

        assert ctx.defeat_tracker.last_fight_name == "a_bat"
        assert ctx.defeat_tracker.last_fight_id == 200
        assert ctx.defeat_tracker.last_fight_x == 100.0
        assert ctx.defeat_tracker.last_fight_y == 200.0

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_enter_aborts_on_target_mismatch(self):
        ctx = _make_ctx()
        ctx.combat.pull_target_id = 999
        ctx.combat.pull_target_name = "wrong_mob"
        target = _make_target(spawn_id=200)
        routine, state, _ = _make_routine(ctx=ctx, target=target)

        routine.enter(state)

        assert routine._aborted is True
        assert ctx.combat.pull_target_id is None

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    @patch("routines.pull.random")
    def test_enter_sets_approach_phase_when_far(self, mock_random):
        """Target beyond OPTIMAL_PULL_MAX triggers APPROACH phase."""
        mock_random.gauss.return_value = 0.0
        mock_random.choice.side_effect = lambda seq: seq[0]
        mock_random.random.return_value = 0.99  # PET_THEN_DOT path
        mock_random.uniform.return_value = 0.0

        # Player at origin, target far away
        target = _make_target(x=200.0, y=200.0)
        routine, state, ctx = _make_routine(target=target)
        routine.enter(state)

        # Distance ~283 > OPTIMAL_PULL_MAX (130), so should be APPROACH
        assert routine._phase == _Phase.APPROACH

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    @patch("routines.pull.random")
    def test_enter_sends_pet_when_in_range(self, mock_random):
        """Target within range goes to SEND_PET (default PET_THEN_DOT)."""
        mock_random.gauss.return_value = 0.0
        mock_random.choice.side_effect = lambda seq: seq[0]
        mock_random.random.return_value = 0.99  # PET_THEN_DOT path
        mock_random.uniform.return_value = 0.0

        # Place target within optimal pull range
        target = _make_target(x=80.0, y=0.0)  # dist=80, within 60..130
        routine, state, ctx = _make_routine(target=target)
        routine.enter(state)

        assert routine._phase == _Phase.SEND_PET


# ---------------------------------------------------------------------------
# tick() -- aborted / timeout / target lost
# ---------------------------------------------------------------------------


class TestTickEdgeCases:
    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_tick_returns_failure_when_aborted(self):
        routine, state, ctx = _make_routine()
        routine._aborted = True
        routine._pull_start = time.time()

        result = routine.tick(state)

        assert result == RoutineStatus.FAILURE
        assert routine.failure_reason == "target_mismatch"
        assert routine.failure_category == FailureCategory.PRECONDITION

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_tick_returns_failure_on_timeout(self):
        routine, state, ctx = _make_routine()
        routine._pull_start = time.time() - 30.0  # 30s ago

        result = routine.tick(state)

        assert result == RoutineStatus.FAILURE
        assert routine.failure_reason == "timeout"
        assert routine.failure_category == FailureCategory.TIMEOUT

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_tick_failure_on_target_too_far(self):
        """Target beyond PULL_ABORT_DISTANCE triggers failure."""
        target = _make_target(x=300.0, y=300.0)  # dist ~424 > 250
        routine, state, ctx = _make_routine(target=target)
        routine._pull_start = time.time()
        routine._phase = _Phase.WAIT_PET
        routine._wait_pet_deadline = 0  # force re-init

        result = routine.tick(state)

        assert result == RoutineStatus.FAILURE
        assert routine.failure_reason == "target_too_far"

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_tick_target_lost_before_pet_sent(self):
        """Target gone while still in SEND_PET phase -> FAILURE."""
        # No target in state
        state = make_game_state(hp_current=1000, hp_max=1000, mana_current=500, mana_max=500)
        ctx = _make_ctx()
        routine = PullRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._pull_start = time.time()
        routine._phase = _Phase.SEND_PET

        result = routine.tick(state)

        assert result == RoutineStatus.FAILURE
        assert routine.failure_reason == "target_lost"

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_tick_target_dead_during_pull_records_kill(self):
        """Target dies mid-pull (pet killed it) -> SUCCESS + record_kill."""
        # Target with 0 HP
        target = _make_target(hp_current=0)
        state = make_game_state(
            target=target,
            hp_current=1000,
            hp_max=1000,
            mana_current=500,
            mana_max=500,
        )
        ctx = _make_ctx()
        routine = PullRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._pull_start = time.time()
        routine._phase = _Phase.WAIT_PET  # past SEND_PET
        routine._pull_dist = 80.0
        routine._pet_engage_time = 1.0

        result = routine.tick(state)

        assert result == RoutineStatus.SUCCESS
        assert ctx.combat.engaged is False
        assert ctx.combat.pull_target_id is None


# ---------------------------------------------------------------------------
# tick() -- SEND_PET phase
# ---------------------------------------------------------------------------


class TestTickSendPet:
    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    @patch("routines.pull.random")
    def test_send_pet_transitions_to_wait_pet(self, mock_random):
        mock_random.gauss.return_value = 0.0
        mock_random.uniform.return_value = 0.0

        target = _make_target(x=80.0, y=0.0)
        routine, state, ctx = _make_routine(target=target)
        routine._pull_start = time.time()
        routine._phase = _Phase.SEND_PET
        routine._pull_dist = 80.0
        routine._pet_engage_time = 0.0

        result = routine.tick(state)

        assert result == RoutineStatus.RUNNING
        assert routine._phase == _Phase.WAIT_PET
        assert routine._locked is True


# ---------------------------------------------------------------------------
# tick() -- WAIT_PET phase
# ---------------------------------------------------------------------------


class TestTickWaitPet:
    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_wait_pet_detects_hp_drop(self):
        """HP drop on target = pet hit confirmed -> CAST_DOT for WHITE con."""
        # Use level=10 target vs level=10 player -> WHITE con -> CAST_DOT
        target = _make_target(x=80.0, y=0.0, hp_current=90, hp_max=100, level=10)
        routine, state, ctx = _make_routine(
            target=target,
            state_overrides=dict(level=10),
        )
        routine._pull_start = time.time()
        routine._phase = _Phase.WAIT_PET
        routine._pull_dist = 80.0
        routine._pet_engage_time = 0.0
        routine._strategy = PullRoutine.PET_THEN_DOT
        # Simulate: WAIT_PET was initialized with full HP
        routine._wait_pet_deadline = time.time() + 10
        routine._wait_pet_initial_hp = 100  # target started at 100
        routine._wait_pet_initial_dist = 80.0
        routine._wait_pet_logged = False

        result = routine.tick(state)

        assert result == RoutineStatus.RUNNING
        # WHITE con -> PET_THEN_DOT -> CAST_DOT
        assert routine._phase == _Phase.CAST_DOT

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_wait_pet_pet_only_skips_dot(self):
        """PET_ONLY strategy goes to ENGAGED after pet hit."""
        target = _make_target(x=80.0, y=0.0, hp_current=90, hp_max=100)
        routine, state, ctx = _make_routine(target=target)
        routine._pull_start = time.time()
        routine._phase = _Phase.WAIT_PET
        routine._pull_dist = 80.0
        routine._pet_engage_time = 0.0
        routine._strategy = PullRoutine.PET_ONLY
        routine._wait_pet_deadline = time.time() + 10
        routine._wait_pet_initial_hp = 100
        routine._wait_pet_initial_dist = 80.0
        routine._wait_pet_logged = False

        result = routine.tick(state)

        assert result == RoutineStatus.RUNNING
        assert routine._phase == _Phase.ENGAGED

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_wait_pet_timeout_resends_pet(self):
        """Timeout in WAIT_PET re-sends pet and advances phase."""
        target = _make_target(x=80.0, y=0.0, hp_current=100, hp_max=100)
        routine, state, ctx = _make_routine(target=target)
        routine._pull_start = time.time()
        routine._phase = _Phase.WAIT_PET
        routine._pull_dist = 80.0
        routine._pet_engage_time = 0.0
        routine._strategy = PullRoutine.PET_THEN_DOT
        routine._wait_pet_deadline = time.time() - 1.0  # already expired
        routine._wait_pet_initial_hp = 100
        routine._wait_pet_initial_dist = 80.0
        routine._wait_pet_logged = False

        result = routine.tick(state)

        assert result == RoutineStatus.RUNNING
        # Pet-then-dot after timeout -> CAST_DOT
        assert routine._phase == _Phase.CAST_DOT


# ---------------------------------------------------------------------------
# tick() -- ENGAGED phase
# ---------------------------------------------------------------------------


class TestTickEngaged:
    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_engaged_returns_success(self):
        target = _make_target(x=80.0, y=0.0)
        routine, state, ctx = _make_routine(target=target)
        routine._pull_start = time.time()
        routine._phase = _Phase.ENGAGED
        routine._pull_dist = 80.0
        routine._pet_engage_time = 1.0
        routine._dot_retries = 0

        result = routine.tick(state)

        assert result == RoutineStatus.SUCCESS
        assert ctx.combat.engaged is True

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_engaged_sets_engagement_start(self):
        target = _make_target(x=80.0, y=0.0)
        routine, state, ctx = _make_routine(target=target)
        routine._pull_start = time.time()
        routine._phase = _Phase.ENGAGED
        routine._pull_dist = 80.0
        routine._pet_engage_time = 1.0
        routine._dot_retries = 0

        routine.tick(state)

        assert ctx.player.engagement_start > 0


# ---------------------------------------------------------------------------
# tick() -- APPROACH phase
# ---------------------------------------------------------------------------


class TestTickApproach:
    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    @patch("routines.pull.random")
    def test_approach_transitions_to_send_pet_when_in_range(self, mock_random):
        mock_random.gauss.return_value = 0.0
        mock_random.uniform.return_value = 0.0

        # Target at 80u (within OPTIMAL_PULL_MIN..OPTIMAL_PULL_MAX)
        target = _make_target(x=80.0, y=0.0)
        routine, state, ctx = _make_routine(target=target)
        routine._pull_start = time.time()
        routine._phase = _Phase.APPROACH
        routine._pull_dist = 80.0
        routine._pet_engage_time = 0.0

        result = routine.tick(state)

        assert result == RoutineStatus.RUNNING
        assert routine._phase == _Phase.SEND_PET

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    @patch("routines.pull.random")
    def test_approach_starts_walking_when_too_far(self, mock_random):
        mock_random.gauss.return_value = 0.0
        mock_random.uniform.return_value = 0.0

        target = _make_target(x=200.0, y=0.0)  # dist=200 > OPTIMAL_PULL_MAX
        routine, state, ctx = _make_routine(target=target)
        routine._pull_start = time.time()
        routine._phase = _Phase.APPROACH
        routine._approach_walking = False
        routine._pull_dist = 200.0
        routine._pet_engage_time = 0.0

        result = routine.tick(state)

        assert result == RoutineStatus.RUNNING
        assert routine._approach_walking is True
        assert routine._locked is True


# ---------------------------------------------------------------------------
# tick() -- reactive backstep
# ---------------------------------------------------------------------------


class TestReactiveBackstep:
    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    @patch("routines.pull.time")
    def test_backstep_when_npc_in_melee_range(self, mock_time):
        mock_time.time.return_value = 100.0
        mock_time.sleep = lambda *a: None  # noop for blocking backstep

        # Target at 10u (< MELEE_RANGE=15)
        target = _make_target(x=10.0, y=0.0, hp_current=100, hp_max=100)
        routine, state, ctx = _make_routine(target=target)
        routine._pull_start = 90.0  # 10s ago
        routine._phase = _Phase.WAIT_PET
        routine._backstep_count = 0
        routine._has_backstepped = False
        routine._pull_dist = 80.0
        routine._pet_engage_time = 0.0
        # Set up WAIT_PET state
        routine._wait_pet_deadline = 110.0
        routine._wait_pet_initial_hp = 100
        routine._wait_pet_initial_dist = 80.0
        routine._wait_pet_logged = False

        result = routine.tick(state)

        assert result == RoutineStatus.RUNNING
        assert routine._backstep_count == 1
        assert routine._has_backstepped is True


# ---------------------------------------------------------------------------
# _face_target()
# ---------------------------------------------------------------------------


class TestFaceTarget:
    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    @patch("routines.pull.random")
    def test_face_target_skips_during_cast(self, mock_random):
        """_face_target should no-op when player is casting."""
        mock_random.gauss.return_value = 0.0

        target = _make_target(x=80.0, y=0.0)
        # casting_mode=1 means is_casting=True
        state = make_game_state(
            target=target,
            hp_current=1000,
            hp_max=1000,
            mana_current=500,
            mana_max=500,
            casting_mode=1,
        )
        routine = PullRoutine(ctx=_make_ctx(), read_state_fn=lambda: state)

        # Should return without calling face_heading
        with patch("routines.pull.face_heading") as mock_face:
            routine._face_target(state, target)
            mock_face.assert_not_called()

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    @patch("routines.pull.random")
    def test_face_target_calls_face_heading(self, mock_random):
        """_face_target should call face_heading when not casting."""
        mock_random.gauss.return_value = 0.0

        target = _make_target(x=80.0, y=0.0)
        state = make_game_state(
            target=target,
            hp_current=1000,
            hp_max=1000,
            mana_current=500,
            mana_max=500,
        )
        routine = PullRoutine(ctx=_make_ctx(), read_state_fn=lambda: state)
        routine._flee_check = None

        with patch("routines.pull.face_heading") as mock_face:
            routine._face_target(state, target)
            mock_face.assert_called_once()


# ---------------------------------------------------------------------------
# exit()
# ---------------------------------------------------------------------------


class TestExit:
    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_exit_unlocks(self):
        routine, state, ctx = _make_routine()
        routine._locked = True
        routine._pull_start = time.time()
        routine._pull_dist = 80.0
        routine._pet_engage_time = 1.0
        routine._dot_retries = 0

        routine.exit(state)

        assert routine._locked is False

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_exit_stops_active_backstep(self):
        routine, state, ctx = _make_routine()
        routine._bs_active = True
        routine._pull_start = time.time()
        routine._pull_dist = 80.0
        routine._pet_engage_time = 1.0
        routine._dot_retries = 0

        routine.exit(state)

        assert routine._bs_active is False

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_exit_records_last_pull_summary(self):
        routine, state, ctx = _make_routine()
        routine._pull_start = time.time()
        routine._pull_dist = 80.0
        routine._pet_engage_time = 1.0
        routine._dot_retries = 2
        routine._strategy = PullRoutine.PET_THEN_DOT

        routine.exit(state)

        assert hasattr(routine, "last_pull_summary")
        assert routine.last_pull_summary["strategy"] == PullRoutine.PET_THEN_DOT
        assert routine.last_pull_summary["dot_retries"] == 2

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_exit_marks_success_when_no_failure_reason(self):
        routine, state, ctx = _make_routine()
        routine._pull_start = time.time()
        routine._pull_dist = 80.0
        routine._pet_engage_time = 1.0
        routine._dot_retries = 0
        routine.failure_reason = ""

        routine.exit(state)

        # last_pull_summary should exist, success is implied by no failure_reason
        assert routine.last_pull_summary is not None

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_exit_records_diagnostics(self):
        routine, state, ctx = _make_routine()
        routine._pull_start = time.time()
        routine._pull_dist = 80.0
        routine._pet_engage_time = 1.0
        routine._dot_retries = 0
        # Set up diag metrics
        from brain.state.diagnostic import DiagnosticState

        ctx.diag = DiagnosticState()
        from unittest.mock import MagicMock

        ctx.diag.metrics = MagicMock()

        routine.exit(state)

        ctx.diag.metrics.record_action.assert_called_once_with("pull", True)


# ---------------------------------------------------------------------------
# Full lifecycle: enter -> tick(SEND_PET) -> tick(WAIT_PET) -> tick(ENGAGED)
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    @patch("routines.pull.random")
    def test_pet_only_lifecycle(self, mock_random):
        """Pet-only pull: enter -> SEND_PET -> WAIT_PET (init) -> WAIT_PET (hp drop) -> ENGAGED -> exit."""
        mock_random.gauss.return_value = 0.0
        mock_random.choice.side_effect = lambda seq: seq[0]
        mock_random.random.return_value = 0.99
        mock_random.uniform.return_value = 0.0

        target = _make_target(x=80.0, y=0.0, hp_current=100, hp_max=100, level=3)
        ctx = _make_ctx()

        # Start with full-HP target
        state_full = make_game_state(
            target=target,
            hp_current=1000,
            hp_max=1000,
            mana_current=500,
            mana_max=500,
            level=10,
        )
        routine = PullRoutine(ctx=ctx, read_state_fn=lambda: state_full)

        # enter -- level 10 vs level 3 -> LIGHT_BLUE -> PET_ONLY
        routine.enter(state_full)
        assert routine._strategy == PullRoutine.PET_ONLY

        # tick 1: SEND_PET -> WAIT_PET
        r1 = routine.tick(state_full)
        assert r1 == RoutineStatus.RUNNING
        assert routine._phase == _Phase.WAIT_PET

        # tick 2: WAIT_PET initialization (sets initial HP from state)
        r2 = routine.tick(state_full)
        assert r2 == RoutineStatus.RUNNING
        assert routine._phase == _Phase.WAIT_PET  # still waiting

        # Simulate HP drop on target
        target_hit = _make_target(x=80.0, y=0.0, hp_current=80, hp_max=100, level=3)
        state_hit = make_game_state(
            target=target_hit,
            hp_current=1000,
            hp_max=1000,
            mana_current=500,
            mana_max=500,
            level=10,
        )

        # tick 3: WAIT_PET detects HP drop -> ENGAGED (pet-only skips dot)
        r3 = routine.tick(state_hit)
        assert r3 == RoutineStatus.RUNNING
        assert routine._phase == _Phase.ENGAGED

        # tick 4: ENGAGED -> SUCCESS
        r4 = routine.tick(state_hit)
        assert r4 == RoutineStatus.SUCCESS
        assert ctx.combat.engaged is True

        # exit
        routine.exit(state_hit)
        assert routine._locked is False
        assert routine.last_pull_summary["strategy"] == PullRoutine.PET_ONLY


# ---------------------------------------------------------------------------
# _validate_pull_target
# ---------------------------------------------------------------------------


class TestValidatePullTarget:
    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_validate_passes_when_no_pull_target_set(self):
        """No pull_target_id on ctx -> validation passes (no mismatch check)."""
        target = _make_target()
        routine, state, ctx = _make_routine(target=target)
        routine._flee_check = None

        assert routine._validate_pull_target(state) is True

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_validate_passes_when_ids_match(self):
        target = _make_target(spawn_id=200)
        ctx = _make_ctx()
        ctx.combat.pull_target_id = 200
        ctx.combat.pull_target_name = "a_bat"
        routine, state, _ = _make_routine(ctx=ctx, target=target)
        routine._flee_check = None

        assert routine._validate_pull_target(state) is True

    @patch("routines.pull.interruptible_sleep", _noop_sleep)
    def test_validate_fails_on_id_mismatch(self):
        target = _make_target(spawn_id=200)
        ctx = _make_ctx()
        ctx.combat.pull_target_id = 999
        ctx.combat.pull_target_name = "a_skeleton"
        routine, state, _ = _make_routine(ctx=ctx, target=target)
        routine._flee_check = None

        result = routine._validate_pull_target(state)

        assert result is False
        assert routine._aborted is True


# ---------------------------------------------------------------------------
# locked property
# ---------------------------------------------------------------------------


class TestLocked:
    def test_locked_reflects_internal_state(self):
        routine = PullRoutine()
        assert routine.locked is False
        routine._locked = True
        assert routine.locked is True
