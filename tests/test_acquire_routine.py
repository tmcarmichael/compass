"""Tests for routines.acquire -- AcquireRoutine target acquisition lifecycle.

Covers __init__, enter/exit lifecycle, tick dispatch, target validation,
cycling fallback, camp-sit acquire, and approach walk logic.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from brain.context import AgentContext
from core.types import FailureCategory, GrindStyle, Point
from perception.combat_eval import Con, set_avoid_names
from routines.acquire import (
    MAX_TABS,
    AcquireRoutine,
)
from routines.base import RoutineStatus
from routines.target_filter import (
    estimate_exposure,
    guard_nearby,
    is_acceptable_target,
    nearby_npc_count,
    social_npc_count,
)
from tests.factories import make_game_state, make_mob_profile, make_spawn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(**overrides) -> AgentContext:
    """Build an AgentContext with sensible test defaults."""
    ctx = AgentContext()
    ctx.pet.alive = True
    ctx.camp.camp_pos = Point(0.0, 0.0, 0.0)
    ctx.camp.roam_radius = 200.0
    # Include WHITE in target_cons so level-matched NPCs are valid targets
    ctx.zone.target_cons = frozenset({Con.WHITE, Con.BLUE, Con.LIGHT_BLUE})
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def _npc(spawn_id=100, name="a_skeleton", x=50.0, y=50.0, level=10, hp=100, **kw):
    """Shorthand for making an NPC spawn."""
    return make_spawn(
        spawn_id=spawn_id,
        name=name,
        x=x,
        y=y,
        level=level,
        hp_current=hp,
        hp_max=hp,
        **kw,
    )


def _state(spawns=(), target=None, x=0.0, y=0.0, level=10, **kw):
    """Shorthand for making a GameState."""
    return make_game_state(spawns=spawns, target=target, x=x, y=y, level=level, **kw)


def _make_routine(ctx=None, state=None, read_state_fn=None):
    """Build an AcquireRoutine with ctx and a fixed read_state_fn."""
    if ctx is None:
        ctx = _make_ctx()
    if state is None:
        state = _state()
    if read_state_fn is None:

        def read_state_fn():
            return state

    return AcquireRoutine(ctx=ctx, read_state_fn=read_state_fn)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_initial_state(self):
        routine = AcquireRoutine()
        assert routine._tab_count == 0
        assert routine._locked is False
        assert routine._seen_ids == set()
        assert routine._empty_tabs == 0
        assert routine._approach_done is False
        assert routine._has_targets is False
        assert routine._best_target is None
        assert routine._camp_sit_target is None
        assert routine._approach_active is False
        assert routine.last_acquire_summary == {}

    def test_init_with_ctx_and_read_fn(self):
        ctx = _make_ctx()
        state = _state()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
        assert routine._ctx is ctx
        assert routine._read_state_fn is not None

    def test_locked_property_default_false(self):
        routine = AcquireRoutine()
        assert routine.locked is False


# ---------------------------------------------------------------------------
# enter()
# ---------------------------------------------------------------------------


class TestEnter:
    def test_enter_resets_tab_count(self):
        """enter() resets internal counters from a previous run."""
        ctx = _make_ctx()
        npc = _npc(x=50.0, y=50.0)
        state = _state(spawns=(npc,))
        routine = _make_routine(ctx=ctx, state=state)
        # Simulate previous run's leftover state
        routine._tab_count = 5
        routine._seen_ids = {1, 2, 3}
        routine._empty_tabs = 3
        routine._approach_done = True

        routine.enter(state)

        assert routine._tab_count == 0
        assert routine._seen_ids == set()
        assert routine._empty_tabs == 0
        assert routine._approach_done is False

    def test_enter_sets_has_targets_when_valid_npcs(self):
        """enter() sets _has_targets when find_targets returns results."""
        npc = _npc(x=50.0, y=50.0, level=10)
        state = _state(spawns=(npc,), level=10)
        ctx = _make_ctx()
        routine = _make_routine(ctx=ctx, state=state)

        routine.enter(state)

        assert routine._has_targets is True

    def test_enter_no_targets_when_empty_spawns(self):
        """enter() leaves _has_targets=False when no valid NPCs exist."""
        state = _state(spawns=())
        ctx = _make_ctx()
        routine = _make_routine(ctx=ctx, state=state)

        routine.enter(state)

        assert routine._has_targets is False

    def test_enter_clears_stale_target(self, _recording_motor):
        """enter() clears a stale invalid target so Tab starts fresh."""
        dead_npc = _npc(spawn_id=99, name="a_skeleton", level=10, hp=0)
        state = _state(target=dead_npc, spawns=())
        ctx = _make_ctx()
        routine = _make_routine(ctx=ctx, state=state)

        routine.enter(state)

        # clear_target() sends "escape" action
        assert _recording_motor.has_action("escape")

    def test_enter_increments_cycle_id(self):
        """enter() increments defeat_tracker.cycle_id."""
        ctx = _make_ctx()
        old_cycle = ctx.defeat_tracker.cycle_id
        state = _state(spawns=())
        routine = _make_routine(ctx=ctx, state=state)

        routine.enter(state)

        assert ctx.defeat_tracker.cycle_id == old_cycle + 1

    def test_enter_sets_cycle_start_time(self):
        """enter() records cycle_start_time in metrics."""
        ctx = _make_ctx()
        state = _state(spawns=())
        routine = _make_routine(ctx=ctx, state=state)
        before = time.time()

        routine.enter(state)

        assert ctx.metrics.cycle_start_time >= before

    def test_enter_no_targets_with_only_green_con(self):
        """Green-con NPCs are not valid targets -- enter should find no targets."""
        # Level 10 player vs level 1 NPC = green con
        green_npc = _npc(x=50.0, y=50.0, level=1)
        state = _state(spawns=(green_npc,), level=10)
        ctx = _make_ctx()
        routine = _make_routine(ctx=ctx, state=state)

        routine.enter(state)

        assert routine._has_targets is False

    def test_enter_stands_if_sitting(self, _recording_motor):
        """enter() stands up if internal motor state says sitting."""
        import motor.actions as _ma

        _ma.mark_sitting()  # Set internal motor state to sitting
        npc = _npc(x=50.0, y=50.0)
        state = _state(spawns=(npc,))
        ctx = _make_ctx()
        routine = _make_routine(ctx=ctx, state=state)

        routine.enter(state)

        # stand() sends "sit_stand" action
        assert _recording_motor.has_action("sit_stand")

    def test_enter_finds_valid_npcs_without_world_model(self):
        """enter() sets _has_targets when valid NPCs exist (no WorldModel)."""
        npc = _npc(x=50.0, y=50.0, level=10)
        state = _state(spawns=(npc,), level=10)
        ctx = _make_ctx()
        ctx.world = None  # No WorldModel
        routine = _make_routine(ctx=ctx, state=state)

        routine.enter(state)

        # find_targets fallback should discover the NPC
        assert routine._has_targets is True


# ---------------------------------------------------------------------------
# tick() -- dispatch
# ---------------------------------------------------------------------------


class TestTickDispatch:
    def test_tick_no_targets_returns_failure(self):
        """tick() returns FAILURE immediately when no targets found in enter()."""
        state = _state(spawns=())
        ctx = _make_ctx()
        routine = _make_routine(ctx=ctx, state=state)
        routine.enter(state)

        result = routine.tick(state)

        assert result == RoutineStatus.FAILURE
        assert routine.failure_reason == "no_targets"
        assert routine.failure_category == FailureCategory.PRECONDITION

    def test_tick_increments_consecutive_fails_on_no_targets(self):
        """tick() increments consecutive_acquire_fails when no targets."""
        state = _state(spawns=())
        ctx = _make_ctx()
        routine = _make_routine(ctx=ctx, state=state)
        routine.enter(state)
        ctx.metrics.consecutive_acquire_fails = 2

        routine.tick(state)

        assert ctx.metrics.consecutive_acquire_fails == 3

    def test_tick_tab_exhausted_after_max_tabs(self):
        """tick() returns FAILURE when MAX_TABS reached."""
        npc = _npc(x=50.0, y=50.0, level=10)
        state = _state(spawns=(npc,), level=10)
        ctx = _make_ctx()
        routine = _make_routine(ctx=ctx, state=state)
        routine.enter(state)
        routine._tab_count = MAX_TABS  # Simulate exhaustion

        result = routine.tick(state)

        assert result == RoutineStatus.FAILURE
        assert routine.failure_reason == "tab_exhausted"
        assert routine.failure_category == FailureCategory.EXECUTION


# ---------------------------------------------------------------------------
# tick() -- tab and validate
# ---------------------------------------------------------------------------


class TestTickTabAndValidate:
    def test_first_tab_calls_tab_target(self, _recording_motor):
        """First tick presses Tab (not cycle_target)."""
        npc = _npc(spawn_id=200, x=50.0, y=50.0, level=10)
        # State after tab press returns the NPC as target
        state_with_target = _state(spawns=(npc,), target=npc, level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state_with_target)
        routine._has_targets = True

        routine.tick(state_with_target)

        assert routine._tab_count == 1
        # tab_target() records "target_npc"
        assert _recording_motor.has_action("target_npc")

    def test_second_tab_calls_cycle_target(self, _recording_motor):
        """Second tick presses Cycle (not tab_target)."""
        npc = _npc(spawn_id=200, x=50.0, y=50.0, level=10)
        state_no_target = _state(spawns=(npc,), level=10)

        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state_no_target)
        routine._has_targets = True

        # First tick -- tab_target (returns no target, RUNNING)
        routine.tick(state_no_target)
        assert routine._tab_count == 1

        # Second tick -- cycle_target (cycle_npc action)
        _recording_motor.clear()
        routine.tick(state_no_target)
        assert routine._tab_count == 2
        # cycle_target() records "cycle_npc"
        assert _recording_motor.has_action("cycle_npc")

    def test_valid_target_returns_success(self):
        """Valid NPC target above score threshold returns SUCCESS."""
        npc = _npc(spawn_id=200, x=50.0, y=50.0, level=10)
        state = _state(spawns=(npc,), target=npc, level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._has_targets = True

        result = routine.tick(state)

        assert result == RoutineStatus.SUCCESS
        assert ctx.combat.pull_target_id == 200
        assert ctx.combat.pull_target_name == "a_skeleton"
        assert ctx.metrics.consecutive_acquire_fails == 0

    def test_empty_tab_increments_empty_count(self):
        """Tab returning no NPC increments _empty_tabs."""
        state = _state(spawns=(), level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._has_targets = True

        routine.tick(state)

        assert routine._empty_tabs == 1

    def test_dead_target_not_accepted(self):
        """Tab returning a dead NPC (hp=0) is not accepted."""
        dead = _npc(spawn_id=200, x=50.0, y=50.0, level=10, hp=0)
        state = _state(spawns=(dead,), target=dead, level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._has_targets = True

        result = routine.tick(state)

        assert result == RoutineStatus.RUNNING
        # Should not have set pull_target_id
        assert ctx.combat.pull_target_id is None or ctx.combat.pull_target_id == 0


# ---------------------------------------------------------------------------
# _tick_validate_target
# ---------------------------------------------------------------------------


class TestTickValidateTarget:
    def test_reject_unacceptable_target_increments_invalid_tabs(self, _recording_motor):
        """Unacceptable target (e.g., red con) increments invalid_tabs."""
        # Level 10 player, level 20 NPC = red con
        red_npc = _npc(spawn_id=300, x=50.0, y=50.0, level=20)
        state = _state(spawns=(red_npc,), target=red_npc, level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._has_targets = True

        result = routine.tick(state)

        assert result == RoutineStatus.RUNNING
        assert ctx.metrics.acquire_invalid_tabs >= 1
        # clear_target() sends "escape" action after reject
        assert _recording_motor.has_action("escape")

    def test_reject_target_with_social_adds(self):
        """Target with social adds is rejected."""
        target = _npc(spawn_id=300, x=50.0, y=50.0, level=10, name="a_skeleton")
        add1 = _npc(spawn_id=301, x=55.0, y=55.0, level=10, name="a_skeleton")
        state = _state(
            spawns=(target, add1),
            target=target,
            level=10,
        )
        ctx = _make_ctx()
        # Social group key must match name.rstrip("0123456789") -- "a_skeleton"
        ctx.zone.social_mob_group = {"a_skeleton": frozenset({"a_skeleton"})}

        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._has_targets = True

        result = routine.tick(state)

        # With adds, should reject and keep RUNNING
        assert result == RoutineStatus.RUNNING

    def test_accept_target_without_world_model(self):
        """Target is accepted when no WorldModel is available."""
        npc = _npc(spawn_id=400, x=50.0, y=50.0, level=10)
        state = _state(spawns=(npc,), target=npc, level=10)
        ctx = _make_ctx()
        ctx.world = None  # No WorldModel
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._has_targets = True

        result = routine.tick(state)

        assert result == RoutineStatus.SUCCESS
        assert ctx.combat.pull_target_id == 400

    def test_records_tab_count_on_success(self):
        """Success appends tab count to acquire_tab_totals."""
        npc = _npc(spawn_id=400, x=50.0, y=50.0, level=10)
        state = _state(spawns=(npc,), target=npc, level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._has_targets = True

        routine.tick(state)

        assert 1 in ctx.metrics.acquire_tab_totals


# ---------------------------------------------------------------------------
# _tick_cycling_fallback
# ---------------------------------------------------------------------------


class TestTickCyclingFallback:
    def test_cycling_same_ids_triggers_fallback(self):
        """Seeing the same spawn_id again after 3+ tabs triggers cycling fallback."""
        red_npc = _npc(spawn_id=500, x=50.0, y=50.0, level=20)
        state = _state(spawns=(red_npc,), target=red_npc, level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._has_targets = True

        # First two ticks see the npc, third detects cycling
        for _ in range(3):
            routine.tick(state)

        # After seeing an already-seen ID on tab >= 3, it should either
        # set _empty_tabs=4 (approach walk) or fail with no_valid_target
        assert (routine._empty_tabs == 4) or (routine.failure_reason == "no_valid_target")

    def test_cycling_with_no_valid_targets_returns_failure(self):
        """Cycling when no valid targets exist returns FAILURE."""
        red_npc = _npc(spawn_id=500, x=50.0, y=50.0, level=20)
        state = _state(spawns=(red_npc,), target=red_npc, level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._has_targets = True
        routine._approach_done = True  # Already approached once

        # Pre-seed seen IDs so cycling is detected on first tick
        routine._seen_ids = {500}
        routine._tab_count = 3

        result = routine.tick(state)

        assert result == RoutineStatus.FAILURE
        assert routine.failure_reason == "no_valid_target"
        assert routine.failure_category == FailureCategory.PRECONDITION


# ---------------------------------------------------------------------------
# exit()
# ---------------------------------------------------------------------------


class TestExit:
    def test_exit_resets_locked(self):
        """exit() sets _locked to False."""
        routine = _make_routine()
        routine._locked = True

        routine.exit(_state())

        assert routine._locked is False

    def test_exit_stops_approach_walk(self, _recording_motor):
        """exit() stops forward movement if approach walk is active."""
        routine = _make_routine()
        routine._approach_active = True

        routine.exit(_state())

        assert routine._approach_active is False
        # move_forward_stop() sends "-forward" (action_up)
        assert "-forward" in _recording_motor.actions

    def test_exit_records_success_summary(self):
        """exit() populates last_acquire_summary on success."""
        ctx = _make_ctx()
        ctx.combat.pull_target_id = 200
        ctx.combat.pull_target_name = "a_skeleton"
        routine = _make_routine(ctx=ctx)

        routine.exit(_state())

        summary = routine.last_acquire_summary
        assert summary["success"] is True
        assert summary["target"] == "a_skeleton"
        assert summary["entity_id"] == 200

    def test_exit_records_failure_summary(self):
        """exit() records failure when no target was acquired."""
        ctx = _make_ctx()
        ctx.combat.pull_target_id = 0
        ctx.combat.pull_target_name = ""
        routine = _make_routine(ctx=ctx)
        routine._tab_count = 5

        routine.exit(_state())

        summary = routine.last_acquire_summary
        assert summary["success"] is False
        assert summary["tabs"] == 5

    def test_exit_records_tab_count(self):
        """exit() captures tab count in summary."""
        ctx = _make_ctx()
        ctx.combat.pull_target_id = 100
        ctx.combat.pull_target_name = "a_bat"
        routine = _make_routine(ctx=ctx)
        routine._tab_count = 3

        routine.exit(_state())

        assert routine.last_acquire_summary["tabs"] == 3

    def test_exit_without_ctx_is_safe(self):
        """exit() without AgentContext does not raise."""
        routine = AcquireRoutine(ctx=None, read_state_fn=None)
        routine.exit(_state())  # Should not raise
        assert routine.last_acquire_summary == {}


# ---------------------------------------------------------------------------
# Full lifecycle: enter -> tick -> exit
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_enter_tick_exit_success(self):
        """Full acquire cycle: enter finds targets, tick accepts, exit records."""
        npc = _npc(spawn_id=600, x=50.0, y=50.0, level=10)
        state = _state(spawns=(npc,), target=npc, level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)

        routine.enter(state)
        assert routine._has_targets is True

        result = routine.tick(state)
        assert result == RoutineStatus.SUCCESS

        routine.exit(state)
        assert routine.last_acquire_summary["success"] is True
        assert routine.last_acquire_summary["entity_id"] == 600

    def test_enter_tick_exit_failure_no_spawns(self):
        """Full acquire cycle: enter finds nothing, tick fails immediately."""
        state = _state(spawns=())
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)

        routine.enter(state)
        assert routine._has_targets is False

        result = routine.tick(state)
        assert result == RoutineStatus.FAILURE

        routine.exit(state)
        assert routine.last_acquire_summary["success"] is False

    def test_reuse_after_exit(self):
        """Routine can be reused after exit() -- enter() resets state."""
        npc = _npc(spawn_id=700, x=50.0, y=50.0, level=10)
        state = _state(spawns=(npc,), target=npc, level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)

        # First cycle
        routine.enter(state)
        routine.tick(state)
        routine.exit(state)
        assert routine.last_acquire_summary["success"] is True

        # Second cycle -- should reset and work again
        ctx.combat.pull_target_id = 0
        ctx.combat.pull_target_name = ""
        routine.enter(state)
        assert routine._tab_count == 0
        result = routine.tick(state)
        assert result == RoutineStatus.SUCCESS


# ---------------------------------------------------------------------------
# target_filter.py
# ---------------------------------------------------------------------------


class TestTargetFilter:
    def test_guard_nearby_true(self):
        """guard_nearby returns True when a guard is near the target."""
        # Ensure avoid names include "guard"
        set_avoid_names(frozenset({"guard"}))
        target = _npc(spawn_id=100, x=50.0, y=50.0)
        guard = _npc(spawn_id=200, name="a_guard", x=60.0, y=60.0, level=50)
        state = _state(spawns=(target, guard))
        assert guard_nearby(target, state) is True

    def test_guard_nearby_false_when_far(self):
        """guard_nearby returns False when guard is beyond check radius."""
        set_avoid_names(frozenset({"guard"}))
        target = _npc(spawn_id=100, x=50.0, y=50.0)
        guard = _npc(spawn_id=200, name="a_guard", x=500.0, y=500.0, level=50)
        state = _state(spawns=(target, guard))
        assert guard_nearby(target, state) is False

    def test_guard_nearby_false_no_guards(self):
        """guard_nearby returns False when no guards exist."""
        set_avoid_names(frozenset({"guard"}))
        target = _npc(spawn_id=100, x=50.0, y=50.0)
        other = _npc(spawn_id=200, name="a_bat", x=55.0, y=55.0)
        state = _state(spawns=(target, other))
        assert guard_nearby(target, state) is False

    def test_is_acceptable_rejects_invalid_target(self):
        """is_acceptable_target rejects a non-valid spawn (e.g., dead)."""
        dead = _npc(spawn_id=100, hp=0)
        state = _state(spawns=(dead,))
        assert is_acceptable_target(dead, state, None) is False

    def test_is_acceptable_rejects_wrong_con(self):
        """is_acceptable_target rejects a target with wrong con color."""
        # Red con NPC (level 20 vs player level 10)
        red_npc = _npc(spawn_id=100, level=20)
        state = _state(spawns=(red_npc,), level=10)
        ctx = _make_ctx()
        ctx.zone.target_cons = {Con.WHITE, Con.BLUE, Con.LIGHT_BLUE}
        assert is_acceptable_target(red_npc, state, ctx) is False

    def test_is_acceptable_rejects_far_target(self):
        """is_acceptable_target rejects targets beyond SCAN_RADIUS."""
        far_npc = _npc(spawn_id=100, x=5000.0, y=5000.0, level=10)
        state = _state(spawns=(far_npc,), level=10, x=0.0, y=0.0)
        assert is_acceptable_target(far_npc, state, None) is False

    def test_is_acceptable_rejects_damaged_target(self):
        """is_acceptable_target rejects badly damaged NPCs."""
        # HP below 90% threshold
        damaged = _npc(spawn_id=100, level=10, hp=50)
        damaged = make_spawn(spawn_id=100, level=10, hp_current=50, hp_max=100, x=50.0, y=50.0)
        state = _state(spawns=(damaged,), level=10)
        assert is_acceptable_target(damaged, state, None) is False

    def test_is_acceptable_accepts_valid_target(self):
        """is_acceptable_target accepts a healthy, close, valid-con NPC."""
        npc = _npc(spawn_id=100, x=50.0, y=50.0, level=10)
        state = _state(spawns=(npc,), level=10)
        assert is_acceptable_target(npc, state, None) is True

    def test_is_acceptable_rejects_player_proximity(self):
        """is_acceptable_target rejects NPCs near other players."""
        npc = _npc(spawn_id=100, x=50.0, y=50.0, level=10)
        player = make_spawn(
            spawn_id=200,
            name="OtherPlayer",
            x=52.0,
            y=52.0,
            level=10,
            spawn_type=0,
            hp_current=1000,
            hp_max=1000,
        )
        state = _state(spawns=(npc, player), level=10)
        assert is_acceptable_target(npc, state, None) is False

    def test_is_acceptable_rejects_guard_nearby(self):
        """is_acceptable_target rejects NPCs near guards."""
        set_avoid_names(frozenset({"guard"}))
        npc = _npc(spawn_id=100, x=50.0, y=50.0, level=10)
        guard = _npc(spawn_id=200, name="a_guard", x=60.0, y=60.0, level=50)
        state = _state(spawns=(npc, guard), level=10)
        assert is_acceptable_target(npc, state, None) is False

    def test_nearby_npc_count(self):
        """nearby_npc_count counts living NPCs near target."""
        target = _npc(spawn_id=100, x=50.0, y=50.0)
        nearby1 = _npc(spawn_id=101, x=55.0, y=55.0)
        nearby2 = _npc(spawn_id=102, x=52.0, y=52.0)
        far = _npc(spawn_id=103, x=500.0, y=500.0)
        state = _state(spawns=(target, nearby1, nearby2, far))
        count = nearby_npc_count(target, state)
        assert count == 2  # nearby1 and nearby2, not target itself, not far

    def test_social_npc_count_no_groups(self):
        """social_npc_count returns 0 when no social groups configured."""
        target = _npc(spawn_id=100)
        state = _state(spawns=(target,))
        ctx = _make_ctx()
        ctx.zone.social_mob_group = {}
        assert social_npc_count(target, state, ctx) == 0

    def test_estimate_exposure_fight_only(self):
        """estimate_exposure returns just fight time when mana is sufficient."""
        state = _state(mana_current=500, mana_max=500)
        profile = make_mob_profile(fight_duration_est=30.0, mana_cost_est=50)
        ctx = _make_ctx()
        exposure = estimate_exposure(state, profile, ctx)
        assert exposure == 30.0

    def test_estimate_exposure_includes_rest(self):
        """estimate_exposure adds rest time when post-fight mana drops below threshold."""
        # Set mana so that post-fight mana drops below rest_entry
        state = _state(mana_current=100, mana_max=500)
        profile = make_mob_profile(fight_duration_est=30.0, mana_cost_est=90)
        ctx = _make_ctx()
        ctx.rest_mana_entry = 0.40  # enter rest at 40%
        ctx.rest_mana_threshold = 0.60  # exit rest at 60%
        # Post-fight mana: 100 - 90 = 10 => 10/500 = 2% < 40%
        exposure = estimate_exposure(state, profile, ctx)
        assert exposure > 30.0  # Should include rest time


# ---------------------------------------------------------------------------
# CAMP_SIT mode
# ---------------------------------------------------------------------------


class TestCampSitAcquire:
    def test_camp_sit_enter_uses_proximity(self):
        """CAMP_SIT enter() uses proximity scan, not tab targeting."""
        npc = _npc(spawn_id=800, x=50.0, y=50.0, level=10)
        state = _state(spawns=(npc,), level=10)
        ctx = _make_ctx()

        with patch("core.features.flags") as mock_flags:
            mock_flags.grind_style = GrindStyle.CAMP_SIT
            routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
            routine.enter(state)

        assert routine._has_targets is True
        assert routine._camp_sit_target is not None
        assert routine._camp_sit_target.spawn_id == 800

    def test_camp_sit_rejects_far_npc(self):
        """CAMP_SIT rejects NPCs beyond engage radius."""
        far_npc = _npc(spawn_id=800, x=500.0, y=500.0, level=10)
        state = _state(spawns=(far_npc,), level=10)
        ctx = _make_ctx()

        with patch("core.features.flags") as mock_flags:
            mock_flags.grind_style = GrindStyle.CAMP_SIT
            routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
            routine.enter(state)

        assert routine._camp_sit_target is None
        assert routine._has_targets is False

    def test_camp_sit_rejects_dead_npc(self):
        """CAMP_SIT rejects dead NPCs."""
        dead = _npc(spawn_id=800, x=50.0, y=50.0, level=10, hp=0)
        state = _state(spawns=(dead,), level=10)
        ctx = _make_ctx()

        with patch("core.features.flags") as mock_flags:
            mock_flags.grind_style = GrindStyle.CAMP_SIT
            routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
            routine.enter(state)

        assert routine._camp_sit_target is None

    def test_camp_sit_selects_nearest(self):
        """CAMP_SIT selects the nearest valid NPC."""
        near = _npc(spawn_id=801, x=20.0, y=20.0, level=10)
        far = _npc(spawn_id=802, x=200.0, y=200.0, level=10)
        state = _state(spawns=(far, near), level=10)
        ctx = _make_ctx()

        with patch("core.features.flags") as mock_flags:
            mock_flags.grind_style = GrindStyle.CAMP_SIT
            routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
            routine.enter(state)

        assert routine._camp_sit_target is not None
        assert routine._camp_sit_target.spawn_id == 801

    def test_camp_sit_records_mode(self):
        """CAMP_SIT increments acquire_modes['camp_sit']."""
        npc = _npc(spawn_id=800, x=50.0, y=50.0, level=10)
        state = _state(spawns=(npc,), level=10)
        ctx = _make_ctx()

        with patch("core.features.flags") as mock_flags:
            mock_flags.grind_style = GrindStyle.CAMP_SIT
            routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
            routine.enter(state)

        assert ctx.metrics.acquire_modes.get("camp_sit", 0) >= 1

    def test_camp_sit_tick_success(self, _recording_motor):
        """CAMP_SIT tick returns SUCCESS after tab-targeting the npc."""
        npc = _npc(spawn_id=800, x=50.0, y=50.0, level=10)
        state_targeted = _state(spawns=(npc,), target=npc, level=10)
        ctx = _make_ctx()

        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state_targeted)
        # Manually set camp_sit state as if enter() ran in CAMP_SIT mode
        routine._has_targets = True
        routine._camp_sit_target = npc

        result = routine.tick(state_targeted)

        assert result == RoutineStatus.SUCCESS
        assert ctx.combat.pull_target_id == 800


# ---------------------------------------------------------------------------
# Approach walk
# ---------------------------------------------------------------------------


class TestApproachWalk:
    def test_empty_tabs_trigger_approach(self, _recording_motor):
        """4+ empty tabs trigger approach walk initialization."""
        npc = _npc(spawn_id=900, x=200.0, y=200.0, level=10)
        state = _state(spawns=(npc,), level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._has_targets = True
        routine._empty_tabs = 4  # Trigger threshold

        result = routine.tick(state)

        assert result == RoutineStatus.RUNNING
        assert routine._approach_done is True

    def test_approach_active_continues_running(self, _recording_motor):
        """Active approach walk returns RUNNING."""
        npc = _npc(spawn_id=900, x=200.0, y=200.0, level=10)
        state = _state(spawns=(npc,), level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._has_targets = True
        routine._approach_active = True
        routine._approach_start_x = 0.0
        routine._approach_start_y = 0.0
        routine._approach_walk_dist = 50.0
        routine._approach_deadline = time.time() + 10.0  # Not expired

        result = routine.tick(state)

        assert result == RoutineStatus.RUNNING

    def test_approach_walk_complete_resets_state(self, _recording_motor):
        """Approach walk completes when walked >= target distance."""
        npc = _npc(spawn_id=900, x=200.0, y=200.0, level=10)
        # Player has walked 60 units from start
        state = _state(spawns=(npc,), x=60.0, y=0.0, level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._has_targets = True
        routine._approach_active = True
        routine._approach_start_x = 0.0
        routine._approach_start_y = 0.0
        routine._approach_walk_dist = 50.0  # Target = 50 units, walked 60
        routine._approach_deadline = time.time() + 10.0
        routine._locked = True

        result = routine.tick(state)

        assert result == RoutineStatus.RUNNING
        assert routine._approach_active is False
        assert routine._locked is False
        assert routine._empty_tabs == 0

    def test_approach_stops_early_when_npc_in_range(self, _recording_motor):
        """Approach walk stops early if a valid NPC is within tab range."""
        close_npc = _npc(spawn_id=901, x=20.0, y=20.0, level=10)
        # Player at 10,0 -- NPC at 20,20 is ~22u away (< 90u tab range)
        state = _state(spawns=(close_npc,), x=10.0, y=0.0, level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)
        routine._has_targets = True
        routine._approach_active = True
        routine._approach_start_x = 0.0
        routine._approach_start_y = 0.0
        routine._approach_walk_dist = 100.0  # Far target
        routine._approach_deadline = time.time() + 10.0
        routine._locked = True

        result = routine.tick(state)

        assert result == RoutineStatus.RUNNING
        assert routine._approach_active is False
        assert routine._locked is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_ctx_enter_tick_exit(self):
        """Routine operates safely without AgentContext."""
        state = _state(spawns=())
        routine = AcquireRoutine(ctx=None, read_state_fn=lambda: state)

        routine.enter(state)
        result = routine.tick(state)
        routine.exit(state)

        assert result == RoutineStatus.FAILURE

    def test_no_read_state_fn(self):
        """Routine operates safely without read_state_fn."""
        npc = _npc(x=50.0, y=50.0, level=10)
        state = _state(spawns=(npc,), level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=None)

        routine.enter(state)
        # Has targets but can't read state -- tick will keep running
        if routine._has_targets:
            result = routine.tick(state)
            assert result == RoutineStatus.RUNNING

    def test_multiple_valid_targets_picks_one(self):
        """With multiple valid NPCs, tab should pick one and succeed."""
        npc1 = _npc(spawn_id=1001, x=30.0, y=30.0, level=10, name="a_skeleton")
        npc2 = _npc(spawn_id=1002, x=60.0, y=60.0, level=10, name="a_bat")
        state = _state(spawns=(npc1, npc2), target=npc1, level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)

        routine.enter(state)
        result = routine.tick(state)

        assert result == RoutineStatus.SUCCESS
        assert ctx.combat.pull_target_id == 1001

    def test_preemptive_approach_for_distant_best_target(self):
        """Best target beyond 110u triggers pre-emptive approach (empty_tabs=4)."""
        distant = _npc(spawn_id=1003, x=200.0, y=200.0, level=10)
        state = _state(spawns=(distant,), level=10)
        ctx = _make_ctx()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=lambda: state)

        routine.enter(state)

        # Best target at ~283u should trigger pre-emptive approach
        if routine._has_targets and routine._best_target:
            assert routine._empty_tabs == 4

    def test_recently_defeated_target_rejected(self):
        """is_acceptable_target rejects recently defeated spawns."""
        npc = _npc(spawn_id=100, x=50.0, y=50.0, level=10)
        state = _state(spawns=(npc,), level=10)
        ctx = _make_ctx()
        ctx.defeat_tracker.recent_kills = [(100, time.time())]
        assert is_acceptable_target(npc, state, ctx) is False
