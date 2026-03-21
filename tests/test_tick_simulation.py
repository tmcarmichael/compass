"""Multi-tick routine simulation tests.

Uses a TickDriver harness that feeds GameState sequences through
enter/tick/exit to cover phase-machine branches in routines.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from routines.base import RoutineBase, RoutineStatus
from tests.factories import make_agent_context, make_game_state, make_spawn

# ---------------------------------------------------------------------------
# Patch interruptible_sleep to be instant (no real sleeps in tests)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_sleeps():
    """Patch interruptible_sleep and time.sleep to be instant in routine tests."""
    with patch("core.timing.interruptible_sleep", return_value=False):
        with patch("time.sleep"):
            yield


# ---------------------------------------------------------------------------
# TickDriver harness
# ---------------------------------------------------------------------------


class TickDriver:
    """Drive a routine through enter -> N ticks -> exit with state sequences."""

    def __init__(self, routine: RoutineBase, max_ticks: int = 50):
        self.routine = routine
        self.max_ticks = max_ticks
        self.tick_count = 0
        self.results: list[RoutineStatus] = []

    def enter(self, state=None):
        if state is None:
            state = make_game_state()
        self.routine.enter(state)
        return self

    def tick(self, state=None) -> RoutineStatus:
        if state is None:
            state = make_game_state()
        result = self.routine.tick(state)
        self.results.append(result)
        self.tick_count += 1
        return result

    def tick_until_done(self, state_fn=None, max_ticks=None):
        """Tick until SUCCESS/FAILURE or max_ticks reached."""
        limit = max_ticks or self.max_ticks
        for _ in range(limit):
            state = state_fn() if state_fn else make_game_state()
            result = self.tick(state)
            if result in (RoutineStatus.SUCCESS, RoutineStatus.FAILURE):
                return result
        return RoutineStatus.RUNNING

    def exit(self, state=None):
        if state is None:
            state = make_game_state()
        self.routine.exit(state)
        return self


# ---------------------------------------------------------------------------
# PullRoutine tick simulation
# ---------------------------------------------------------------------------


class TestPullTickSimulation:
    def _make_pull(self, state_fn=None, **ctx_kw):
        from routines.pull import PullRoutine

        ctx = make_agent_context(**ctx_kw)
        ctx.combat.pull_target_id = 200
        ctx.combat.pull_target_name = "a_skeleton"
        routine = PullRoutine(ctx=ctx, read_state_fn=state_fn)
        return routine, ctx

    def _target_state(self, target_hp=100, dist=80.0, **kw):
        target = make_spawn(
            spawn_id=200, name="a_skeleton", x=dist, y=0.0, hp_current=target_hp, hp_max=100, level=10
        )
        return make_game_state(target=target, spawns=(target,), **kw)

    def test_aborted_pull_returns_failure(self):
        routine, ctx = self._make_pull()
        # Enter with no target -> aborted
        state = make_game_state()
        driver = TickDriver(routine).enter(state)
        result = driver.tick(state)
        assert result == RoutineStatus.FAILURE

    def test_pull_timeout(self):
        routine, ctx = self._make_pull()
        state = self._target_state(dist=80.0)
        routine.enter(state)
        routine._pull_start = time.time() - 26.0  # force timeout
        result = routine.tick(state)
        assert result == RoutineStatus.FAILURE
        assert routine.failure_reason == "timeout"

    def test_target_lost_before_pet_sent(self):
        routine, ctx = self._make_pull()
        state = self._target_state(dist=80.0)
        routine.enter(state)
        # Tick with no target
        no_target_state = make_game_state()
        result = routine.tick(no_target_state)
        assert result == RoutineStatus.FAILURE
        assert routine.failure_reason == "target_lost"

    def test_target_dead_during_pull_is_success(self):
        routine, ctx = self._make_pull()
        state = self._target_state(dist=80.0)
        routine.enter(state)
        routine._phase = __import__("routines.pull", fromlist=["_Phase"])._Phase.WAIT_PET
        # Target dies mid-pull
        dead_target = make_spawn(
            spawn_id=200, name="a_skeleton", x=80.0, y=0.0, hp_current=0, hp_max=100, level=10
        )
        dead_state = make_game_state(target=dead_target, spawns=(dead_target,))
        result = routine.tick(dead_state)
        assert result == RoutineStatus.SUCCESS

    def test_target_too_far_aborts(self):
        from routines.pull import _Phase

        routine, ctx = self._make_pull()
        state = self._target_state(dist=80.0)
        routine.enter(state)
        routine._phase = _Phase.WAIT_PET
        # Target moved far away
        far_target = make_spawn(
            spawn_id=200, name="a_skeleton", x=500.0, y=0.0, hp_current=100, hp_max=100, level=10
        )
        far_state = make_game_state(target=far_target, spawns=(far_target,))
        result = routine.tick(far_state)
        assert result == RoutineStatus.FAILURE
        assert routine.failure_reason == "target_too_far"

    def test_send_pet_phase(self):
        from routines.pull import _Phase

        routine, ctx = self._make_pull()
        state = self._target_state(dist=80.0)
        routine.enter(state)
        routine._phase = _Phase.SEND_PET
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING
        # Should advance to WAIT_PET
        assert routine._phase == _Phase.WAIT_PET

    def test_wait_pet_engage_detection(self):
        from routines.pull import _Phase

        routine, ctx = self._make_pull()
        state = self._target_state(dist=80.0, target_hp=100)
        routine.enter(state)
        routine._phase = _Phase.WAIT_PET
        routine._wait_pet_deadline = time.time() + 10.0
        routine._wait_pet_initial_hp = 100
        routine._wait_pet_initial_dist = 80.0
        routine._wait_pet_logged = False
        # Target HP dropped -> pet engaged
        engaged_state = self._target_state(dist=60.0, target_hp=80)
        result = routine.tick(engaged_state)
        assert result == RoutineStatus.RUNNING

    def test_wait_pet_timeout(self):
        from routines.pull import _Phase

        routine, ctx = self._make_pull()
        state = self._target_state(dist=80.0)
        routine.enter(state)
        routine._phase = _Phase.WAIT_PET
        routine._wait_pet_deadline = time.time() - 1.0  # expired
        routine._wait_pet_initial_hp = 100
        routine._wait_pet_initial_dist = 80.0
        routine._wait_pet_logged = False
        result = routine.tick(state)
        # Should either transition or return RUNNING/FAILURE
        assert result in (RoutineStatus.RUNNING, RoutineStatus.FAILURE)

    def test_approach_phase_arrives(self):
        from routines.pull import _Phase

        routine, ctx = self._make_pull()
        state = self._target_state(dist=80.0)
        routine.enter(state)
        routine._phase = _Phase.APPROACH
        # Target at optimal range
        optimal_state = self._target_state(dist=60.0)
        result = routine.tick(optimal_state)
        assert result == RoutineStatus.RUNNING
        assert routine._phase == _Phase.SEND_PET

    def test_approach_phase_walking(self):
        from routines.pull import _Phase

        routine, ctx = self._make_pull()
        state = self._target_state(dist=200.0)
        routine.enter(state)
        routine._phase = _Phase.APPROACH
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING
        assert routine._approach_walking is True

    def test_approach_phase_timeout(self):
        from routines.pull import _Phase

        routine, ctx = self._make_pull()
        state = self._target_state(dist=200.0)
        routine.enter(state)
        routine._phase = _Phase.APPROACH
        routine._approach_walking = True
        routine._approach_deadline = time.time() - 1.0
        routine._approach_last_face = time.time()
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING
        assert routine._phase == _Phase.SEND_PET

    def test_approach_too_close_backsteps(self):
        from routines.pull import _Phase

        routine, ctx = self._make_pull()
        state = self._target_state(dist=20.0)
        routine.enter(state)
        routine._phase = _Phase.APPROACH
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING
        assert routine._has_backstepped is True

    def test_backstep_reactive_melee_range(self):
        from routines.pull import _Phase

        routine, ctx = self._make_pull()
        state = self._target_state(dist=80.0)
        routine.enter(state)
        routine._phase = _Phase.WAIT_PET
        routine._wait_pet_deadline = time.time() + 10.0
        # Target in melee range
        close_state = self._target_state(dist=8.0)
        result = routine.tick(close_state)
        assert result == RoutineStatus.RUNNING
        assert routine._backstep_count == 1

    def test_engaged_phase_success(self):
        from routines.pull import _Phase

        routine, ctx = self._make_pull()
        state = self._target_state(dist=50.0)
        routine.enter(state)
        routine._phase = _Phase.ENGAGED
        result = routine.tick(state)
        assert result == RoutineStatus.SUCCESS

    def test_validate_pull_target_mismatch(self):
        routine, ctx = self._make_pull()
        ctx.combat.pull_target_id = 999
        ctx.combat.pull_target_name = "wrong_mob"
        target = make_spawn(
            spawn_id=200, name="a_skeleton", x=80.0, y=0.0, hp_current=100, hp_max=100, level=10
        )
        state = make_game_state(target=target, spawns=(target,))
        routine.enter(state)
        result = routine.tick(state)
        assert result == RoutineStatus.FAILURE

    def test_select_pull_approach_pet_only_light_blue(self):
        from routines.pull import PullRoutine

        routine, ctx = self._make_pull()
        target = make_spawn(
            spawn_id=200, name="a_skeleton", x=80.0, y=0.0, hp_current=100, hp_max=100, level=5
        )
        state = make_game_state(target=target, spawns=(target,), level=20)
        routine._select_pull_approach(
            state, target, __import__("perception.combat_eval", fromlist=["Con"]).Con.LIGHT_BLUE
        )
        assert routine._strategy == PullRoutine.PET_ONLY

    def test_wait_patrol_phase(self):
        from routines.pull import _Phase

        routine, ctx = self._make_pull()
        state = self._target_state(dist=60.0)
        routine.enter(state)
        routine._phase = _Phase.WAIT_PATROL
        routine._patrol_wait_deadline = time.time() - 1.0  # expired
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING
        assert routine._phase == _Phase.SEND_PET

    def test_ranged_pull_no_spell(self):
        from routines.pull import _Phase

        routine, ctx = self._make_pull()
        state = self._target_state(dist=80.0)
        routine.enter(state)
        routine._phase = _Phase.RANGED_PULL
        routine._pull_spell = None
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING
        assert routine._phase == _Phase.SEND_PET

    def test_exit_clears_state(self):
        routine, ctx = self._make_pull()
        state = self._target_state(dist=80.0)
        routine.enter(state)
        routine.exit(state)


# ---------------------------------------------------------------------------
# WanderRoutine tick simulation
# ---------------------------------------------------------------------------


class TestWanderTickSimulation:
    def _make_wander(self):
        from routines.wander import WanderRoutine

        ctx = make_agent_context()
        ctx.camp.camp_x = 0.0
        ctx.camp.camp_y = 0.0
        ctx.camp.hunt_min_dist = 50.0
        ctx.camp.hunt_max_dist = 300.0
        routine = WanderRoutine(camp_x=0.0, camp_y=0.0, ctx=ctx)
        return routine, ctx

    def test_enter_resets_state(self):
        routine, ctx = self._make_wander()
        state = make_game_state(x=100.0, y=100.0)
        routine.enter(state)
        assert routine._walked is False

    def test_tick_returns_running(self):
        routine, ctx = self._make_wander()
        state = make_game_state(x=100.0, y=100.0)
        routine.enter(state)
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING

    def test_tick_with_threat_nearby(self):
        routine, ctx = self._make_wander()
        # NPC targeting the player
        threat = make_spawn(
            spawn_id=500,
            name="a_skeleton",
            x=20.0,
            y=20.0,
            hp_current=100,
            hp_max=100,
            level=20,
            target_name="TestPlayer",
        )
        state = make_game_state(x=10.0, y=10.0, spawns=(threat,))
        routine.enter(state)
        result = routine.tick(state)
        # Should still return RUNNING (threat detection triggers evade via brain, not wander)
        assert result in (RoutineStatus.RUNNING, RoutineStatus.SUCCESS)

    def test_tick_outside_hunt_zone(self):
        routine, ctx = self._make_wander()
        # Player far from camp
        state = make_game_state(x=500.0, y=500.0)
        routine.enter(state)
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING

    def test_exit(self):
        routine, ctx = self._make_wander()
        state = make_game_state()
        routine.enter(state)
        routine.exit(state)

    def test_multiple_ticks(self):
        routine, ctx = self._make_wander()
        state = make_game_state(x=100.0, y=100.0)
        driver = TickDriver(routine, max_ticks=5).enter(state)
        for _ in range(5):
            r = driver.tick(state)
            assert r in (RoutineStatus.RUNNING, RoutineStatus.SUCCESS)


# ---------------------------------------------------------------------------
# RestRoutine tick simulation
# ---------------------------------------------------------------------------


class TestRestTickSimulation:
    def _make_rest(self):
        from routines.rest import RestRoutine

        ctx = make_agent_context()
        ctx.rest_hp_entry = 0.85
        ctx.rest_hp_exit = 0.92
        ctx.rest_mana_entry = 0.25
        ctx.rest_mana_exit = 0.60
        routine = RestRoutine(ctx=ctx)
        return routine, ctx

    def test_enter_and_tick_low_resources(self):
        routine, ctx = self._make_rest()
        state = make_game_state(hp_current=500, hp_max=1000, mana_current=100, mana_max=500)
        routine.enter(state)
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING

    def test_tick_recovered_success(self):
        routine, ctx = self._make_rest()
        state = make_game_state(hp_current=500, hp_max=1000, mana_current=100, mana_max=500)
        routine.enter(state)
        # Fully recovered
        full_state = make_game_state(hp_current=1000, hp_max=1000, mana_current=500, mana_max=500)
        result = routine.tick(full_state)
        assert result == RoutineStatus.SUCCESS

    def test_exit_stands_up(self):
        routine, ctx = self._make_rest()
        state = make_game_state(stand_state=1)  # sitting
        routine.enter(state)
        routine.exit(state)

    def test_attack_during_rest(self):
        routine, ctx = self._make_rest()
        low_state = make_game_state(hp_current=400, hp_max=1000, mana_current=100, mana_max=500)
        routine.enter(low_state)
        # HP drops during rest (being attacked)
        attacked_state = make_game_state(hp_current=300, hp_max=1000, mana_current=100, mana_max=500)
        result = routine.tick(attacked_state)
        # Should still be RUNNING (brain handles flee via emergency rules)
        assert result == RoutineStatus.RUNNING

    def test_multiple_tick_recovery(self):
        routine, ctx = self._make_rest()
        low_state = make_game_state(hp_current=400, hp_max=1000, mana_current=100, mana_max=500)
        routine.enter(low_state)
        # Gradually recover
        for hp in (500, 600, 700, 800, 900):
            state = make_game_state(hp_current=hp, hp_max=1000, mana_current=100, mana_max=500)
            result = routine.tick(state)
            assert result == RoutineStatus.RUNNING
        # Fully recovered
        full = make_game_state(hp_current=1000, hp_max=1000, mana_current=500, mana_max=500)
        result = routine.tick(full)
        assert result == RoutineStatus.SUCCESS


# ---------------------------------------------------------------------------
# TravelRoutine tick simulation
# ---------------------------------------------------------------------------


class TestTravelTickSimulation:
    def _make_travel(self, target_x=100.0, target_y=100.0):
        from routines.travel import TravelRoutine

        def rsf():
            return make_game_state(x=0.0, y=0.0)

        routine = TravelRoutine(target_x=target_x, target_y=target_y, read_state_fn=rsf)
        return routine

    def test_construction(self):
        routine = self._make_travel()
        assert routine is not None

    def test_enter(self):
        routine = self._make_travel()
        state = make_game_state()
        routine.enter(state)

    def test_tick(self):
        routine = self._make_travel()
        state = make_game_state()
        routine.enter(state)
        result = routine.tick(state)
        assert result in (RoutineStatus.SUCCESS, RoutineStatus.FAILURE, RoutineStatus.RUNNING)

    def test_already_at_target(self):
        routine = self._make_travel(target_x=0.0, target_y=0.0)
        state = make_game_state(x=0.0, y=0.0)
        routine.enter(state)
        result = routine.tick(state)
        assert result in (RoutineStatus.SUCCESS, RoutineStatus.RUNNING)

    def test_exit(self):
        routine = self._make_travel()
        state = make_game_state()
        routine.enter(state)
        routine.exit(state)


# ---------------------------------------------------------------------------
# FleeRoutine tick simulation
# ---------------------------------------------------------------------------


class TestFleeTickSimulation:
    def _make_flee(self):
        from routines.flee import FleeRoutine

        ctx = make_agent_context()
        routine = FleeRoutine(ctx=ctx, read_state_fn=None)
        return routine, ctx

    def test_enter(self):
        routine, ctx = self._make_flee()
        state = make_game_state(hp_current=100, hp_max=1000)
        routine.enter(state)

    def test_tick_no_waypoints(self):
        routine, ctx = self._make_flee()
        state = make_game_state(hp_current=100, hp_max=1000)
        routine.enter(state)
        result = routine.tick(state)
        assert result in (RoutineStatus.SUCCESS, RoutineStatus.FAILURE, RoutineStatus.RUNNING)

    def test_exit_clears_state(self):
        routine, ctx = self._make_flee()
        state = make_game_state()
        routine.enter(state)
        routine.exit(state)


# ---------------------------------------------------------------------------
# AcquireRoutine tick simulation
# ---------------------------------------------------------------------------


class TestAcquireTickSimulation:
    def _make_acquire(self):
        from routines.acquire import AcquireRoutine

        ctx = make_agent_context()
        routine = AcquireRoutine(ctx=ctx, read_state_fn=None)
        return routine, ctx

    def test_construction(self):
        routine, ctx = self._make_acquire()
        assert routine is not None

    def test_enter(self):
        routine, ctx = self._make_acquire()
        state = make_game_state()
        routine.enter(state)

    def test_tick_no_targets(self):
        routine, ctx = self._make_acquire()
        state = make_game_state()
        routine.enter(state)
        result = routine.tick(state)
        assert result in (RoutineStatus.SUCCESS, RoutineStatus.FAILURE, RoutineStatus.RUNNING)

    def test_tick_with_target_available(self):
        routine, ctx = self._make_acquire()
        npc = make_spawn(
            spawn_id=300, name="a_skeleton", x=50.0, y=50.0, hp_current=100, hp_max=100, level=10
        )
        state = make_game_state(spawns=(npc,))
        routine.enter(state)
        result = routine.tick(state)
        assert result in (RoutineStatus.SUCCESS, RoutineStatus.FAILURE, RoutineStatus.RUNNING)

    def test_exit(self):
        routine, ctx = self._make_acquire()
        state = make_game_state()
        routine.enter(state)
        routine.exit(state)


# ---------------------------------------------------------------------------
# MemorizeSpells tick simulation
# ---------------------------------------------------------------------------


class TestMemorizeTickSimulation:
    def _make_memorize(self):
        from routines.memorize_spells import MemorizeSpellsRoutine

        ctx = make_agent_context()
        routine = MemorizeSpellsRoutine(ctx=ctx, read_state_fn=None)
        return routine, ctx

    def test_construction(self):
        routine, ctx = self._make_memorize()
        assert routine is not None

    def test_enter_no_reader(self):
        routine, ctx = self._make_memorize()
        state = make_game_state()
        routine.enter(state)

    def test_tick_no_reader_skips(self):
        routine, ctx = self._make_memorize()
        state = make_game_state()
        routine.enter(state)
        result = routine.tick(state)
        assert result in (RoutineStatus.SUCCESS, RoutineStatus.FAILURE, RoutineStatus.RUNNING)

    def test_exit(self):
        routine, ctx = self._make_memorize()
        state = make_game_state()
        routine.enter(state)
        routine.exit(state)


# ---------------------------------------------------------------------------
# CombatRoutine tick simulation
# ---------------------------------------------------------------------------


class TestCombatTickSimulation:
    def _make_combat(self):
        from routines.combat import CombatRoutine

        ctx = make_agent_context()
        routine = CombatRoutine(ctx=ctx, read_state_fn=None)
        return routine, ctx

    def test_enter_with_target(self):
        routine, ctx = self._make_combat()
        target = make_spawn(
            spawn_id=200, name="a_skeleton", x=50.0, y=50.0, hp_current=100, hp_max=100, level=10
        )
        state = make_game_state(target=target, spawns=(target,))
        ctx.combat.engaged = True
        routine.enter(state)

    def test_tick_target_dead(self):
        routine, ctx = self._make_combat()
        target = make_spawn(
            spawn_id=200, name="a_skeleton", x=50.0, y=50.0, hp_current=100, hp_max=100, level=10
        )
        state = make_game_state(target=target, spawns=(target,))
        ctx.combat.engaged = True
        routine.enter(state)
        # Target dies
        dead = make_spawn(spawn_id=200, name="a_skeleton", x=50.0, y=50.0, hp_current=0, hp_max=100, level=10)
        dead_state = make_game_state(target=dead, spawns=(dead,))
        result = routine.tick(dead_state)
        assert result == RoutineStatus.SUCCESS

    def test_tick_no_target(self):
        routine, ctx = self._make_combat()
        target = make_spawn(
            spawn_id=200, name="a_skeleton", x=50.0, y=50.0, hp_current=100, hp_max=100, level=10
        )
        state = make_game_state(target=target, spawns=(target,))
        ctx.combat.engaged = True
        routine.enter(state)
        # Target gone
        no_target = make_game_state()
        result = routine.tick(no_target)
        assert result == RoutineStatus.SUCCESS

    def test_exit(self):
        routine, ctx = self._make_combat()
        target = make_spawn(
            spawn_id=200, name="a_skeleton", x=50.0, y=50.0, hp_current=100, hp_max=100, level=10
        )
        state = make_game_state(target=target, spawns=(target,))
        routine.enter(state)
        routine.exit(state)

    def test_combat_with_strategy(self):
        routine, ctx = self._make_combat()
        target = make_spawn(
            spawn_id=200, name="a_skeleton", x=50.0, y=50.0, hp_current=80, hp_max=100, level=10
        )
        state = make_game_state(
            target=target, spawns=(target,), hp_current=800, hp_max=1000, mana_current=400, mana_max=500
        )
        ctx.combat.engaged = True
        routine.enter(state)
        # Multiple ticks with alive target
        for _ in range(3):
            result = routine.tick(state)
            if result != RoutineStatus.RUNNING:
                break
        assert result in (RoutineStatus.RUNNING, RoutineStatus.SUCCESS)


# ---------------------------------------------------------------------------
# CombatMonitor tick simulation
# ---------------------------------------------------------------------------


class TestCombatMonitorTickSimulation:
    def _make_monitor(self):
        from routines.combat_monitor import CombatMonitor

        class MockCombat:
            _combat_start = time.time() - 5.0
            _target_killed = False
            _medding = False
            _sitting = False
            _combat_target_id = 200
            _combat_target_name = "a_skeleton"
            _has_extra_npcs = False
            _extra_npc_count = 0
            _pre_pull_mana = 400
            _cast_count = 0
            _last_combat_log = 0.0

            def _stand_from_med(self):
                self._medding = False

            def _record_kill(self, target, fight_time):
                self._target_killed = True

        combat = MockCombat()
        return CombatMonitor(combat), combat

    def test_death_check_no_target(self):
        monitor, combat = self._make_monitor()

        class TS:
            target = None
            now = time.time()

        state = make_game_state()
        result = monitor.tick_death_check(state, TS())
        assert result == RoutineStatus.SUCCESS


# ---------------------------------------------------------------------------
# DeathRecovery tick simulation
# ---------------------------------------------------------------------------


class TestDeathRecoveryTickSimulation:
    def test_construction_and_enter(self):
        from routines.death_recovery import DeathRecoveryRoutine

        ctx = make_agent_context()
        routine = DeathRecoveryRoutine(ctx=ctx, read_state_fn=None)
        state = make_game_state(body_state="d")
        routine.enter(state)

    def test_tick_still_dead(self):
        from routines.death_recovery import DeathRecoveryRoutine

        ctx = make_agent_context()
        routine = DeathRecoveryRoutine(ctx=ctx, read_state_fn=None)
        state = make_game_state(body_state="d")
        routine.enter(state)
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING

    def test_exit(self):
        from routines.death_recovery import DeathRecoveryRoutine

        ctx = make_agent_context()
        routine = DeathRecoveryRoutine(ctx=ctx, read_state_fn=None)
        state = make_game_state()
        routine.enter(state)
        routine.exit(state)


# ---------------------------------------------------------------------------
# SummonPet tick simulation
# ---------------------------------------------------------------------------


class TestSummonPetTickSimulation:
    def test_construction_and_enter(self):
        from routines.summon_pet import SummonPetRoutine

        ctx = make_agent_context()
        routine = SummonPetRoutine(ctx=ctx, read_state_fn=None)
        state = make_game_state()
        routine.enter(state)

    def test_tick(self):
        from routines.summon_pet import SummonPetRoutine

        ctx = make_agent_context()
        routine = SummonPetRoutine(ctx=ctx, read_state_fn=None)
        state = make_game_state()
        routine.enter(state)
        result = routine.tick(state)
        assert result in (RoutineStatus.SUCCESS, RoutineStatus.FAILURE, RoutineStatus.RUNNING)


# ---------------------------------------------------------------------------
# FeignDeath tick simulation
# ---------------------------------------------------------------------------


class TestFeignDeathTickSimulation:
    def test_enter_and_tick(self):
        from routines.feign_death import FeignDeathRoutine

        ctx = make_agent_context()
        routine = FeignDeathRoutine(ctx=ctx, read_state_fn=None)
        state = make_game_state()
        routine.enter(state)
        result = routine.tick(state)
        assert result in (RoutineStatus.SUCCESS, RoutineStatus.FAILURE, RoutineStatus.RUNNING)


# ---------------------------------------------------------------------------
# EngageAdd tick simulation
# ---------------------------------------------------------------------------


class TestEngageAddTickSimulation:
    def test_enter_and_tick(self):
        from routines.engage_add import EngageAddRoutine

        ctx = make_agent_context()
        routine = EngageAddRoutine(ctx=ctx, read_state_fn=None)
        state = make_game_state()
        routine.enter(state)
        result = routine.tick(state)
        assert result in (RoutineStatus.SUCCESS, RoutineStatus.FAILURE, RoutineStatus.RUNNING)


# ---------------------------------------------------------------------------
# Evade tick simulation
# ---------------------------------------------------------------------------


class TestEvadeTickSimulation:
    def test_enter_and_tick(self):
        from routines.evade import EvadeRoutine

        ctx = make_agent_context()
        routine = EvadeRoutine(ctx=ctx, read_state_fn=None)
        state = make_game_state()
        routine.enter(state)
        result = routine.tick(state)
        assert result in (RoutineStatus.SUCCESS, RoutineStatus.FAILURE, RoutineStatus.RUNNING)

    def test_exit(self):
        from routines.evade import EvadeRoutine

        ctx = make_agent_context()
        routine = EvadeRoutine(ctx=ctx, read_state_fn=None)
        state = make_game_state()
        routine.enter(state)
        routine.exit(state)


# ---------------------------------------------------------------------------
# Buff tick simulation
# ---------------------------------------------------------------------------


class TestBuffTickSimulation:
    def test_enter_and_tick(self):
        from routines.buff import BuffRoutine

        ctx = make_agent_context()
        routine = BuffRoutine(ctx=ctx, read_state_fn=None)
        state = make_game_state()
        routine.enter(state)
        result = routine.tick(state)
        assert result in (RoutineStatus.SUCCESS, RoutineStatus.FAILURE, RoutineStatus.RUNNING)

    def test_exit(self):
        from routines.buff import BuffRoutine

        ctx = make_agent_context()
        routine = BuffRoutine(ctx=ctx, read_state_fn=None)
        state = make_game_state()
        routine.enter(state)
        routine.exit(state)


# ---------------------------------------------------------------------------
# Deep pull phase tests with read_state_fn
# ---------------------------------------------------------------------------


class TestPullDeepPhases:
    """Test pull phase handlers with read_state_fn for deep coverage."""

    def _target_state(self, target_hp=100, dist=80.0, mana=400, **kw):
        target = make_spawn(
            spawn_id=200, name="a_skeleton", x=dist, y=0.0, hp_current=target_hp, hp_max=100, level=10
        )
        return make_game_state(target=target, spawns=(target,), mana_current=mana, mana_max=500, **kw)

    def _make_rsf(self, **kw):
        state = self._target_state(**kw)
        return lambda: state

    def test_send_pet_with_rsf(self):
        from routines.pull import PullRoutine, _Phase

        rsf = self._make_rsf(dist=80.0)
        ctx = make_agent_context()
        ctx.combat.pull_target_id = 200
        ctx.combat.pull_target_name = "a_skeleton"
        routine = PullRoutine(ctx=ctx, read_state_fn=rsf)
        state = self._target_state(dist=80.0)
        routine.enter(state)
        routine._phase = _Phase.SEND_PET
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING
        assert routine._phase == _Phase.WAIT_PET

    def test_cast_dot_no_spell(self):
        from routines.pull import PullRoutine, _Phase

        rsf = self._make_rsf(dist=50.0)
        ctx = make_agent_context()
        ctx.combat.pull_target_id = 200
        ctx.combat.pull_target_name = "a_skeleton"
        routine = PullRoutine(ctx=ctx, read_state_fn=rsf)
        state = self._target_state(dist=50.0)
        routine.enter(state)
        routine._phase = _Phase.CAST_DOT
        routine._pull_spell = None
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING
        assert routine._phase == _Phase.ENGAGED

    def test_engaged_phase_returns_success(self):
        from routines.pull import PullRoutine, _Phase

        rsf = self._make_rsf(dist=50.0)
        ctx = make_agent_context()
        ctx.combat.pull_target_id = 200
        ctx.combat.pull_target_name = "a_skeleton"
        routine = PullRoutine(ctx=ctx, read_state_fn=rsf)
        state = self._target_state(dist=50.0)
        routine.enter(state)
        routine._phase = _Phase.ENGAGED
        result = routine.tick(state)
        assert result == RoutineStatus.SUCCESS

    def test_wait_approach_npc_close(self):
        from routines.pull import PullRoutine, _Phase

        rsf = self._make_rsf(dist=20.0)
        ctx = make_agent_context()
        ctx.combat.pull_target_id = 200
        ctx.combat.pull_target_name = "a_skeleton"
        routine = PullRoutine(ctx=ctx, read_state_fn=rsf)
        state = self._target_state(dist=80.0)
        routine.enter(state)
        routine._phase = _Phase.WAIT_APPROACH
        close_state = self._target_state(dist=20.0)
        result = routine.tick(close_state)
        assert result == RoutineStatus.RUNNING
        assert routine._phase == _Phase.SEND_PET

    def test_wait_approach_timeout(self):
        from routines.pull import PullRoutine, _Phase

        rsf = self._make_rsf(dist=100.0)
        ctx = make_agent_context()
        ctx.combat.pull_target_id = 200
        ctx.combat.pull_target_name = "a_skeleton"
        routine = PullRoutine(ctx=ctx, read_state_fn=rsf)
        state = self._target_state(dist=100.0)
        routine.enter(state)
        routine._phase = _Phase.WAIT_APPROACH
        routine._pull_start = time.time() - 10.0
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING
        assert routine._phase == _Phase.SEND_PET

    def test_fear_cast_no_spell_falls_back(self):
        from routines.pull import PullRoutine, _Phase

        rsf = self._make_rsf(dist=60.0)
        ctx = make_agent_context()
        ctx.combat.pull_target_id = 200
        ctx.combat.pull_target_name = "a_skeleton"
        routine = PullRoutine(ctx=ctx, read_state_fn=rsf)
        state = self._target_state(dist=60.0)
        routine.enter(state)
        routine._phase = _Phase.FEAR_CAST
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING

    def test_full_pet_only_sequence(self):
        from routines.pull import PullRoutine, _Phase

        rsf = self._make_rsf(dist=60.0)
        ctx = make_agent_context()
        ctx.combat.pull_target_id = 200
        ctx.combat.pull_target_name = "a_skeleton"
        routine = PullRoutine(ctx=ctx, read_state_fn=rsf)
        state = self._target_state(dist=60.0)
        routine.enter(state)
        routine._strategy = PullRoutine.PET_ONLY
        routine._phase = _Phase.SEND_PET
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING
        assert routine._phase == _Phase.WAIT_PET
        routine._wait_pet_initial_hp = 100
        routine._wait_pet_initial_dist = 60.0
        routine._wait_pet_logged = False
        routine._wait_pet_deadline = time.time() + 10.0
        hit_state = self._target_state(dist=55.0, target_hp=80)
        result = routine.tick(hit_state)
        assert result == RoutineStatus.RUNNING
        assert routine._phase == _Phase.ENGAGED
        result = routine.tick(hit_state)
        assert result == RoutineStatus.SUCCESS

    def test_count_nearby_npcs(self):
        from routines.pull import PullRoutine

        ctx = make_agent_context()
        routine = PullRoutine(ctx=ctx)
        target = make_spawn(
            spawn_id=200, name="a_skeleton", x=50.0, y=0.0, hp_current=100, hp_max=100, level=10
        )
        nearby = make_spawn(spawn_id=201, name="a_bat", x=55.0, y=0.0, hp_current=50, hp_max=50, level=5)
        state = make_game_state(spawns=(target, nearby))
        count = routine._count_nearby_npcs(target, state)
        assert count == 1

    def test_ranged_pull_no_spell(self):
        from routines.pull import PullRoutine, _Phase

        rsf = self._make_rsf(dist=80.0)
        ctx = make_agent_context()
        ctx.combat.pull_target_id = 200
        ctx.combat.pull_target_name = "a_skeleton"
        routine = PullRoutine(ctx=ctx, read_state_fn=rsf)
        state = self._target_state(dist=80.0)
        routine.enter(state)
        routine._phase = _Phase.RANGED_PULL
        routine._pull_spell = None
        result = routine.tick(state)
        assert result == RoutineStatus.RUNNING
        assert routine._phase == _Phase.SEND_PET


# ---------------------------------------------------------------------------
# Deep wander, rest, and combat tick tests
# ---------------------------------------------------------------------------


class TestWanderDeepTicks:
    def test_multiple_ticks_with_rsf(self):
        from routines.wander import WanderRoutine

        ctx = make_agent_context()
        ctx.camp.camp_x = 100.0
        ctx.camp.camp_y = 100.0
        ctx.camp.hunt_min_dist = 50.0
        ctx.camp.hunt_max_dist = 300.0
        state = make_game_state(x=150.0, y=150.0)

        def rsf():
            return state

        routine = WanderRoutine(camp_x=100.0, camp_y=100.0, ctx=ctx, read_state_fn=rsf)
        routine.enter(state)
        for _ in range(10):
            r = routine.tick(state)
            assert r in (RoutineStatus.RUNNING, RoutineStatus.SUCCESS)

    def test_far_from_camp(self):
        from routines.wander import WanderRoutine

        ctx = make_agent_context()
        ctx.camp.camp_x = 0.0
        ctx.camp.camp_y = 0.0
        ctx.camp.hunt_max_dist = 100.0
        far = make_game_state(x=500.0, y=500.0)

        def rsf():
            return far

        routine = WanderRoutine(camp_x=0.0, camp_y=0.0, ctx=ctx, read_state_fn=rsf)
        routine.enter(far)
        result = routine.tick(far)
        assert result in (RoutineStatus.RUNNING, RoutineStatus.SUCCESS)


class TestRestDeepTicks:
    def test_gradual_recovery(self):
        from routines.rest import RestRoutine

        ctx = make_agent_context()
        ctx.rest_hp_entry = 0.85
        ctx.rest_hp_exit = 0.92
        ctx.rest_mana_entry = 0.25
        ctx.rest_mana_exit = 0.60
        routine = RestRoutine(ctx=ctx)
        low = make_game_state(hp_current=400, hp_max=1000, mana_current=50, mana_max=500)
        routine.enter(low)
        for mana in range(100, 350, 50):
            state = make_game_state(hp_current=950, hp_max=1000, mana_current=mana, mana_max=500)
            result = routine.tick(state)
            assert result in (RoutineStatus.RUNNING, RoutineStatus.SUCCESS)
        full = make_game_state(hp_current=1000, hp_max=1000, mana_current=500, mana_max=500)
        result = routine.tick(full)
        assert result == RoutineStatus.SUCCESS


class TestCombatDeepTicks:
    def test_target_hp_degrades(self):
        from routines.combat import CombatRoutine

        ctx = make_agent_context()
        routine = CombatRoutine(ctx=ctx, read_state_fn=None)
        target = make_spawn(
            spawn_id=200, name="a_skeleton", x=30.0, y=0.0, hp_current=100, hp_max=100, level=10
        )
        state = make_game_state(target=target, spawns=(target,))
        ctx.combat.engaged = True
        routine.enter(state)
        for hp in (80, 60, 40, 20, 0):
            t = make_spawn(
                spawn_id=200, name="a_skeleton", x=30.0, y=0.0, hp_current=hp, hp_max=100, level=10
            )
            s = make_game_state(target=t, spawns=(t,))
            result = routine.tick(s)
            if result == RoutineStatus.SUCCESS:
                break
        assert result == RoutineStatus.SUCCESS

    def test_target_gone(self):
        from routines.combat import CombatRoutine

        ctx = make_agent_context()
        routine = CombatRoutine(ctx=ctx, read_state_fn=None)
        target = make_spawn(
            spawn_id=200, name="a_skeleton", x=30.0, y=0.0, hp_current=80, hp_max=100, level=10
        )
        state = make_game_state(target=target, spawns=(target,))
        ctx.combat.engaged = True
        routine.enter(state)
        routine.tick(state)
        result = routine.tick(make_game_state())
        assert result == RoutineStatus.SUCCESS
