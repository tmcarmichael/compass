"""Tests for brain.state.plan -- PlanState and TravelPlan.

Covers set_data syncing typed fields, clear(), and TravelPlan defaults.
"""

from __future__ import annotations

from brain.state.plan import PlanState, TravelPlan

# ---------------------------------------------------------------------------
# TravelPlan
# ---------------------------------------------------------------------------


class TestTravelPlan:
    def test_defaults(self) -> None:
        tp = TravelPlan()
        assert tp.destination == ""
        assert tp.route == []
        assert tp.hop_index == 0
        assert tp.waypoint is False
        assert tp.target_x == 0.0
        assert tp.target_y == 0.0
        assert tp.pre_hop_zone_id == 0


# ---------------------------------------------------------------------------
# PlanState.set_data
# ---------------------------------------------------------------------------


class TestPlanStateSetData:
    def test_set_data_with_route(self) -> None:
        ps = PlanState()
        ps.set_data(
            {
                "route": ["hop1", "hop2"],
                "hop_index": 1,
                "destination": "ecommons",
            }
        )
        assert ps.data["route"] == ["hop1", "hop2"]
        assert ps.travel.route == ["hop1", "hop2"]
        assert ps.travel.hop_index == 1
        assert ps.travel.destination == "ecommons"

    def test_set_data_with_waypoint(self) -> None:
        ps = PlanState()
        ps.set_data(
            {
                "waypoint": True,
                "target_x": 100.0,
                "target_y": 200.0,
                "destination": "camp",
            }
        )
        assert ps.travel.waypoint is True
        assert ps.travel.target_x == 100.0
        assert ps.travel.target_y == 200.0
        assert ps.travel.destination == "camp"

    def test_set_data_empty(self) -> None:
        ps = PlanState()
        ps.set_data({})
        assert ps.data == {}
        # Travel fields should remain defaults
        assert ps.travel.route == []
        assert ps.travel.waypoint is False

    def test_set_data_overwrites_previous(self) -> None:
        ps = PlanState()
        ps.set_data({"route": ["a"], "destination": "x"})
        ps.set_data({"route": ["b", "c"], "destination": "y"})
        assert ps.travel.route == ["b", "c"]
        assert ps.travel.destination == "y"

    def test_set_data_with_both_route_and_waypoint(self) -> None:
        ps = PlanState()
        ps.set_data(
            {
                "route": ["h"],
                "hop_index": 0,
                "waypoint": True,
                "target_x": 50.0,
                "target_y": 60.0,
                "destination": "dual",
            }
        )
        assert ps.travel.route == ["h"]
        assert ps.travel.waypoint is True
        assert ps.travel.target_x == 50.0
        assert ps.travel.destination == "dual"

    def test_set_data_partial_route_fields(self) -> None:
        """Only 'route' key triggers route sync; missing fields use defaults."""
        ps = PlanState()
        ps.set_data({"route": []})
        assert ps.travel.route == []
        assert ps.travel.hop_index == 0
        assert ps.travel.destination == ""


# ---------------------------------------------------------------------------
# PlanState.clear
# ---------------------------------------------------------------------------


class TestPlanStateClear:
    def test_clear_resets_all(self) -> None:
        ps = PlanState(active="travel")
        ps.set_data({"route": ["a"], "destination": "x"})
        ps.clear()
        assert ps.active is None
        assert ps.travel.destination == ""
        assert ps.travel.route == []
        assert ps.data == {}

    def test_clear_idempotent(self) -> None:
        ps = PlanState()
        ps.clear()
        assert ps.active is None
        assert ps.data == {}
