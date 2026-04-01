"""Tests for simulator result export and summary metadata."""

from __future__ import annotations

from simulator.results import SimulationResult


def test_to_dict_includes_simulator_assumptions() -> None:
    result = SimulationResult(scenario_name="camp_session")
    result.tick_times_ms = [0.1]
    result.simulator_assumptions = {
        "headless_scenario": True,
        "synthetic_encounter_learning": True,
    }

    data = result.to_dict()

    assert data["simulator_assumptions"] == {
        "headless_scenario": True,
        "synthetic_encounter_learning": True,
    }


def test_summary_surfaces_simulator_assumptions() -> None:
    result = SimulationResult(scenario_name="camp_session", total_ticks=1)
    result.tick_times_ms = [0.1]
    result.simulator_assumptions = {
        "headless_scenario": True,
        "synthetic_scorecard_inputs": True,
    }

    summary = result.summary()

    assert "Simulator Notes" in summary
    assert "headless_scenario" in summary
    assert "synthetic_scorecard_inputs" in summary
