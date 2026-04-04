"""Tests for brain.rules.navigation -- wander, travel conditions.

Condition functions are called directly with GameState + AgentContext.
Feature flags are set per-fixture to isolate tests from global state.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from brain.context import AgentContext
from brain.rules.navigation import (
    _NavigationRuleState,
    _score_travel,
    _score_wander,
    _should_travel,
    _should_wander,
    register,
)
from core.features import flags
from core.types import CampType, PlanType, Point, TravelMode
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus
from tests.factories import make_game_state, make_spawn


@pytest.fixture(autouse=True)
def _enable_flags() -> None:
    """Enable navigation-relevant flags for all tests in this module."""
    flags.wander = True
    flags.pull = True


# ---------------------------------------------------------------------------
# _should_wander
# ---------------------------------------------------------------------------


class TestShouldWander:
    """WANDER condition: flag on, no plan, not engaged, pet nearby."""

    @pytest.mark.parametrize(
        "wander_flag, engaged, plan_active, pet_alive, expected",
        [
            pytest.param(True, False, None, True, True, id="all_clear_wander"),
            pytest.param(False, False, None, True, False, id="flag_off"),
            pytest.param(True, True, None, True, False, id="engaged"),
            pytest.param(True, False, PlanType.TRAVEL, True, False, id="travel_plan_active"),
            pytest.param(True, False, PlanType.NEEDS_MEMORIZE, True, False, id="memorize_plan_active"),
            pytest.param(True, False, None, False, True, id="no_pet_still_wanders"),
        ],
    )
    def test_basic_scenarios(
        self,
        wander_flag: bool,
        engaged: bool,
        plan_active: str | None,
        pet_alive: bool,
        expected: bool,
    ) -> None:
        flags.wander = wander_flag
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.engaged = engaged
        ctx.plan.active = plan_active
        ctx.pet.alive = pet_alive

        result = _should_wander(state, ctx)
        assert result is expected

    def test_pet_too_far_blocks_wander(self) -> None:
        """Pet > 200u away blocks wander."""
        pet_spawn = make_spawn(
            spawn_id=50,
            name="Kabaler`s_pet",
            x=300.0,
            y=300.0,
        )
        state = make_game_state(spawns=(pet_spawn,))
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.spawn_id = 50

        assert _should_wander(state, ctx) is False

    def test_pet_nearby_allows_wander(self) -> None:
        """Pet within 200u allows wander."""
        pet_spawn = make_spawn(
            spawn_id=50,
            name="Kabaler`s_pet",
            x=10.0,
            y=10.0,
        )
        state = make_game_state(spawns=(pet_spawn,))
        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.pet.spawn_id = 50

        assert _should_wander(state, ctx) is True

    def test_far_from_camp_triggers_travel(self) -> None:
        """Being > 400u from camp sets a TRAVEL plan instead of wandering."""
        state = make_game_state(x=500.0, y=500.0)
        ctx = AgentContext()
        ctx.camp.camp_pos = Point(10.0, 10.0, 0.0)  # non-zero so the camp distance check fires

        result = _should_wander(state, ctx)
        assert result is False
        assert ctx.plan.active == PlanType.TRAVEL


# ---------------------------------------------------------------------------
# _should_travel
# ---------------------------------------------------------------------------


class TestShouldTravel:
    """TRAVEL condition: travel plan active, not engaged."""

    @pytest.mark.parametrize(
        "plan_active, engaged, has_waypoint, expected",
        [
            pytest.param(None, False, False, False, id="no_plan"),
            pytest.param(PlanType.NEEDS_MEMORIZE, False, False, False, id="wrong_plan"),
            pytest.param(PlanType.TRAVEL, True, False, False, id="engaged"),
            pytest.param(PlanType.TRAVEL, False, True, True, id="travel_with_waypoint"),
        ],
    )
    def test_basic_scenarios(
        self,
        plan_active: str | None,
        engaged: bool,
        has_waypoint: bool,
        expected: bool,
    ) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.plan.active = plan_active
        ctx.combat.engaged = engaged
        if has_waypoint:
            ctx.plan.travel.waypoint = True
            ctx.plan.travel.target_x = 100.0
            ctx.plan.travel.target_y = 200.0

        rs = _NavigationRuleState()
        result = _should_travel(state, ctx, rs)
        assert result is expected

    def test_no_plan_clears_waypoint_travel(self) -> None:
        """When plan is not 'travel', waypoint_travel ref is cleared."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.plan.active = None

        rs = _NavigationRuleState(waypoint_travel=MagicMock(spec=RoutineBase))
        _should_travel(state, ctx, rs)
        assert rs.waypoint_travel is None

    def test_yields_to_acquire_for_close_target(self) -> None:
        """Travel yields to ACQUIRE when a valid target is very close (< 30u)."""
        close_npc = make_spawn(
            spawn_id=300,
            name="a_moss_snake",
            level=8,
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
            owner_id=0,
        )
        state = make_game_state(level=10, spawns=(close_npc,))
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.plan.travel.waypoint = True
        ctx.plan.travel.target_x = 500.0
        ctx.plan.travel.target_y = 500.0
        ctx.pet.alive = True

        rs = _NavigationRuleState()
        result = _should_travel(state, ctx, rs)
        assert result is False

    def test_direct_xy_target_converts_to_waypoint(self) -> None:
        """Travel with target_x/y but no waypoint flag converts to waypoint."""
        state = make_game_state(x=0.0, y=0.0)
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.plan.travel.waypoint = False
        ctx.plan.travel.target_x = 500.0
        ctx.plan.travel.target_y = 500.0

        rs = _NavigationRuleState()
        result = _should_travel(state, ctx, rs)
        assert result is True
        assert ctx.plan.travel.waypoint is True

    def test_direct_xy_target_arrived_clears_plan(self) -> None:
        """Travel with target_x/y close to player clears the plan."""
        state = make_game_state(x=95.0, y=95.0)
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.plan.travel.waypoint = False
        ctx.plan.travel.target_x = 100.0
        ctx.plan.travel.target_y = 100.0

        rs = _NavigationRuleState()
        result = _should_travel(state, ctx, rs)
        assert result is False
        assert ctx.plan.active is None

    def test_route_exhausted_clears_plan(self) -> None:
        """When route is present but hop_index >= len(route), plan clears."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.plan.travel.waypoint = False
        ctx.plan.travel.target_x = 0.0  # no direct xy
        ctx.plan.travel.route = ["zone_a", "zone_b"]
        ctx.plan.travel.hop_index = 2  # past end

        rs = _NavigationRuleState()
        result = _should_travel(state, ctx, rs)
        assert result is False
        assert ctx.plan.active is None

    def test_route_with_valid_hop_returns_true(self) -> None:
        """When route has remaining hops, travel should proceed."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.plan.travel.waypoint = False
        ctx.plan.travel.target_x = 0.0
        ctx.plan.travel.route = ["zone_a", "zone_b"]
        ctx.plan.travel.hop_index = 0

        rs = _NavigationRuleState()
        result = _should_travel(state, ctx, rs)
        assert result is True

    def test_empty_route_clears_plan(self) -> None:
        """Empty route list clears the plan."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.plan.travel.waypoint = False
        ctx.plan.travel.target_x = 0.0
        ctx.plan.travel.route = []
        ctx.plan.travel.hop_index = 0

        rs = _NavigationRuleState()
        result = _should_travel(state, ctx, rs)
        assert result is False
        assert ctx.plan.active is None

    def test_no_route_no_waypoint_no_xy_clears_plan(self) -> None:
        """Travel plan active but no waypoint, no target xy, no route -> clears."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.plan.travel.waypoint = False
        ctx.plan.travel.target_x = 0.0
        ctx.plan.travel.target_y = 0.0
        ctx.plan.travel.route = None
        ctx.plan.travel.hop_index = 0

        rs = _NavigationRuleState()
        result = _should_travel(state, ctx, rs)
        assert result is False
        assert ctx.plan.active is None

    def test_close_npc_with_pull_disabled_no_yield(self) -> None:
        """Travel should NOT yield to acquire when pull flag is disabled."""
        flags.pull = False
        close_npc = make_spawn(
            spawn_id=300,
            name="a_moss_snake",
            level=8,
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
            owner_id=0,
        )
        state = make_game_state(level=10, spawns=(close_npc,))
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.plan.travel.waypoint = True
        ctx.pet.alive = True

        rs = _NavigationRuleState()
        result = _should_travel(state, ctx, rs)
        assert result is True
        flags.pull = True

    def test_close_npc_pet_dead_no_yield(self) -> None:
        """Travel should NOT yield to acquire when pet is dead."""
        close_npc = make_spawn(
            spawn_id=300,
            name="a_moss_snake",
            level=8,
            x=5.0,
            y=5.0,
            hp_current=100,
            hp_max=100,
            owner_id=0,
        )
        state = make_game_state(level=10, spawns=(close_npc,))
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.plan.travel.waypoint = True
        ctx.pet.alive = False

        rs = _NavigationRuleState()
        result = _should_travel(state, ctx, rs)
        assert result is True


# ---------------------------------------------------------------------------
# _score_travel
# ---------------------------------------------------------------------------


class TestScoreTravel:
    """Score function for TRAVEL rule."""

    def test_travel_plan_active_returns_1(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL

        assert _score_travel(state, ctx) == 1.0

    def test_no_plan_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.plan.active = None

        assert _score_travel(state, ctx) == 0.0

    def test_wrong_plan_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.plan.active = PlanType.NEEDS_MEMORIZE

        assert _score_travel(state, ctx) == 0.0

    def test_engaged_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.combat.engaged = True

        assert _score_travel(state, ctx) == 0.0


# ---------------------------------------------------------------------------
# _score_wander
# ---------------------------------------------------------------------------


class TestScoreWander:
    """Score function for WANDER rule."""

    def test_wander_enabled_returns_1(self) -> None:
        state = make_game_state()
        ctx = AgentContext()

        assert _score_wander(state, ctx) == 1.0

    def test_wander_disabled_returns_0(self) -> None:
        flags.wander = False
        state = make_game_state()
        ctx = AgentContext()

        result = _score_wander(state, ctx)
        flags.wander = True
        assert result == 0.0

    def test_engaged_returns_0(self) -> None:
        state = make_game_state()
        ctx = AgentContext()
        ctx.combat.engaged = True

        assert _score_wander(state, ctx) == 0.0


# ---------------------------------------------------------------------------
# _should_wander  (additional branches)
# ---------------------------------------------------------------------------


class TestShouldWanderAdditional:
    """Additional branches for _should_wander not covered above."""

    def test_pet_fighting_blocks_wander(self) -> None:
        """When world model shows damaged NPCs near player, wander blocked."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True

        world = MagicMock()
        world.damaged_npcs_near.return_value = [MagicMock()]
        ctx.world = world

        assert _should_wander(state, ctx) is False

    def test_pet_not_fighting_allows_wander(self) -> None:
        """When world model shows no damaged NPCs, wander allowed."""
        state = make_game_state()
        ctx = AgentContext()
        ctx.pet.alive = True

        world = MagicMock()
        world.damaged_npcs_near.return_value = []
        ctx.world = world

        assert _should_wander(state, ctx) is True

    def test_linear_camp_type_returns_nearest_point(self) -> None:
        """LINEAR camp returns to nearest point on patrol path, not camp center."""
        from core.types import Point

        state = make_game_state(x=500.0, y=500.0)
        ctx = AgentContext()
        ctx.camp.camp_pos = Point(10.0, 10.0, 0.0)
        ctx.camp.camp_type = CampType.LINEAR
        # Set patrol path so nearest_point_on_path returns a point on the path
        ctx.camp.patrol_waypoints = [Point(0.0, 50.0, 0.0), Point(100.0, 50.0, 0.0)]

        result = _should_wander(state, ctx)
        assert result is False
        assert ctx.plan.active == PlanType.TRAVEL
        # Nearest point on horizontal line y=50 from (500,500) is (100,50)  -- the endpoint
        assert ctx.plan.travel.target_y == 50.0

    def test_circular_camp_returns_to_camp_center(self) -> None:
        """CIRCULAR (default) camp returns to camp_pos."""
        state = make_game_state(x=500.0, y=500.0)
        ctx = AgentContext()
        ctx.camp.camp_pos = Point(10.0, 10.0, 0.0)
        # Default camp_type is CIRCULAR

        result = _should_wander(state, ctx)
        assert result is False
        assert ctx.plan.active == PlanType.TRAVEL
        assert ctx.plan.travel.target_x == 10.0
        assert ctx.plan.travel.target_y == 10.0


# ---------------------------------------------------------------------------
# register + _TravelDispatcher
# ---------------------------------------------------------------------------


class TestRegister:
    """Integration tests for register() and the _TravelDispatcher."""

    def test_register_adds_travel_and_wander(self) -> None:
        """register() should add TRAVEL and WANDER rules."""
        brain = MagicMock()
        ctx = AgentContext()

        def read_state_fn() -> GameState:
            return make_game_state()

        register(brain, ctx, read_state_fn)

        assert brain.add_rule.call_count == 2
        rule_names = [call.args[0] for call in brain.add_rule.call_args_list]
        assert "TRAVEL" in rule_names
        assert "WANDER" in rule_names

    def test_travel_dispatcher_tick_success_clears_plan(self) -> None:
        """_TravelDispatcher.tick clears plan on SUCCESS."""
        brain = MagicMock()
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.plan.travel.waypoint = False
        state = make_game_state()

        def read_state_fn() -> GameState:
            return state

        register(brain, ctx, read_state_fn)

        # Get the TRAVEL rule's routine
        travel_call = brain.add_rule.call_args_list[0]
        dispatcher = travel_call.args[2]

        # Mock the underlying PlanTravelRoutine.tick to return SUCCESS
        with patch("routines.travel.PlanTravelRoutine.tick", return_value=RoutineStatus.SUCCESS):
            result = dispatcher.tick(state)

        assert result == RoutineStatus.SUCCESS
        assert ctx.plan.active is None

    def test_travel_dispatcher_tick_failure_clears_plan(self) -> None:
        """_TravelDispatcher.tick clears plan on FAILURE."""
        brain = MagicMock()
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.plan.travel.waypoint = False
        state = make_game_state()

        def read_state_fn() -> GameState:
            return state

        register(brain, ctx, read_state_fn)

        travel_call = brain.add_rule.call_args_list[0]
        dispatcher = travel_call.args[2]

        with patch("routines.travel.PlanTravelRoutine.tick", return_value=RoutineStatus.FAILURE):
            result = dispatcher.tick(state)

        assert result == RoutineStatus.FAILURE
        assert ctx.plan.active is None

    def test_travel_dispatcher_tick_running_keeps_plan(self) -> None:
        """_TravelDispatcher.tick keeps plan on RUNNING."""
        brain = MagicMock()
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.plan.travel.waypoint = False
        state = make_game_state()

        def read_state_fn() -> GameState:
            return state

        register(brain, ctx, read_state_fn)

        travel_call = brain.add_rule.call_args_list[0]
        dispatcher = travel_call.args[2]

        with patch("routines.travel.PlanTravelRoutine.tick", return_value=RoutineStatus.RUNNING):
            result = dispatcher.tick(state)

        assert result == RoutineStatus.RUNNING
        assert ctx.plan.active == PlanType.TRAVEL

    def test_travel_dispatcher_enter_delegates(self) -> None:
        """_TravelDispatcher.enter delegates to the underlying routine."""
        brain = MagicMock()
        ctx = AgentContext()
        ctx.plan.travel.waypoint = False
        state = make_game_state()

        def read_state_fn() -> GameState:
            return state

        register(brain, ctx, read_state_fn)

        travel_call = brain.add_rule.call_args_list[0]
        dispatcher = travel_call.args[2]

        with patch("routines.travel.PlanTravelRoutine.enter") as mock_enter:
            dispatcher.enter(state)
            mock_enter.assert_called_once_with(state)

    def test_travel_dispatcher_exit_delegates(self) -> None:
        """_TravelDispatcher.exit delegates to the underlying routine."""
        brain = MagicMock()
        ctx = AgentContext()
        ctx.plan.travel.waypoint = False
        state = make_game_state()

        def read_state_fn() -> GameState:
            return state

        register(brain, ctx, read_state_fn)

        travel_call = brain.add_rule.call_args_list[0]
        dispatcher = travel_call.args[2]

        with patch("routines.travel.PlanTravelRoutine.exit") as mock_exit:
            dispatcher.exit(state)
            mock_exit.assert_called_once_with(state)

    def test_travel_dispatcher_locked_property(self) -> None:
        """_TravelDispatcher.locked delegates to the underlying routine."""
        brain = MagicMock()
        ctx = AgentContext()
        ctx.plan.travel.waypoint = False
        state = make_game_state()

        def read_state_fn() -> GameState:
            return state

        register(brain, ctx, read_state_fn)

        travel_call = brain.add_rule.call_args_list[0]
        dispatcher = travel_call.args[2]

        with patch(
            "routines.travel.PlanTravelRoutine.locked", new_callable=lambda: property(lambda self: True)
        ):
            # locked delegates via getattr, so check it doesn't crash
            result = dispatcher.locked
            assert isinstance(result, bool)

    def test_travel_dispatcher_waypoint_simple_travel(self) -> None:
        """When waypoint=True and plan_travel_legs returns single A* leg, use TravelRoutine."""
        brain = MagicMock()
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.plan.travel.waypoint = True
        ctx.plan.travel.target_x = 500.0
        ctx.plan.travel.target_y = 500.0
        state = make_game_state()

        def read_state_fn() -> GameState:
            return state

        register(brain, ctx, read_state_fn)

        travel_call = brain.add_rule.call_args_list[0]
        dispatcher = travel_call.args[2]

        # Mock plan_travel_legs to return a single A* leg
        single_leg = SimpleNamespace(mode=TravelMode.PATHFIND)
        with patch("brain.rules.navigation.plan_travel_legs", return_value=[single_leg]):
            with patch("routines.travel.TravelRoutine.tick", return_value=RoutineStatus.RUNNING):
                result = dispatcher.tick(state)

        assert result == RoutineStatus.RUNNING

    def test_travel_dispatcher_waypoint_multi_leg(self) -> None:
        """When waypoint=True and plan_travel_legs returns multi-leg, use MultiLegTravelRoutine."""
        brain = MagicMock()
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.plan.travel.waypoint = True
        ctx.plan.travel.target_x = 500.0
        ctx.plan.travel.target_y = 500.0
        state = make_game_state()

        def read_state_fn() -> GameState:
            return state

        register(brain, ctx, read_state_fn)

        travel_call = brain.add_rule.call_args_list[0]
        dispatcher = travel_call.args[2]

        # Mock plan_travel_legs to return multiple legs
        leg1 = SimpleNamespace(mode=TravelMode.PATHFIND)
        leg2 = SimpleNamespace(mode=TravelMode.PATHFIND)
        with patch("brain.rules.navigation.plan_travel_legs", return_value=[leg1, leg2]):
            with patch("routines.travel.MultiLegTravelRoutine.tick", return_value=RoutineStatus.RUNNING):
                result = dispatcher.tick(state)

        assert result == RoutineStatus.RUNNING

    def test_travel_dispatcher_waypoint_manual_leg(self) -> None:
        """When waypoint=True and single leg is MANUAL, use MultiLegTravelRoutine."""
        brain = MagicMock()
        ctx = AgentContext()
        ctx.plan.active = PlanType.TRAVEL
        ctx.plan.travel.waypoint = True
        ctx.plan.travel.target_x = 500.0
        ctx.plan.travel.target_y = 500.0
        state = make_game_state()

        def read_state_fn() -> GameState:
            return state

        register(brain, ctx, read_state_fn)

        travel_call = brain.add_rule.call_args_list[0]
        dispatcher = travel_call.args[2]

        manual_leg = SimpleNamespace(mode=TravelMode.MANUAL)
        with patch("brain.rules.navigation.plan_travel_legs", return_value=[manual_leg]):
            with patch("routines.travel.MultiLegTravelRoutine.tick", return_value=RoutineStatus.SUCCESS):
                result = dispatcher.tick(state)

        assert result == RoutineStatus.SUCCESS
