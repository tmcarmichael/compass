"""Smoke tests for public simulator entrypoints."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import simulator.__main__ as simulator_main
from simulator.results import SimulationResult


def _result(name: str, overall: int) -> SimulationResult:
    result = SimulationResult(scenario_name=name, total_ticks=10)
    result.tick_times_ms = [0.1, 0.2, 0.3]
    result.scorecard = {"grade": "A", "overall": overall}
    result.fight_stats = {"a_skeleton": {"fights": 4, "avg_duration": 8.0}}
    result.simulator_assumptions = {"headless_scenario": True}
    return result


def test_simulator_main_converge_reuses_one_result_set(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, int] = {"run": 0, "run_convergence": 0}

    class _FakeRunner:
        def __init__(self, utility_phase: int, enable_goap: bool) -> None:
            assert utility_phase == 2
            assert enable_goap is True

        def run(self, *_args, **_kwargs):
            calls["run"] += 1
            raise AssertionError("converge mode should not fall back to replay run()")

        def run_convergence(self, scenario, sessions: int) -> list[SimulationResult]:
            calls["run_convergence"] += 1
            return [_result(f"{scenario.name} ({i + 1}/{sessions})", 90 + i) for i in range(sessions)]

    monkeypatch.setattr(simulator_main, "SimulationRunner", _FakeRunner)
    monkeypatch.setattr(
        simulator_main,
        "_load_scenario",
        lambda _name: SimpleNamespace(name="camp_session", tick_count=1280),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["simulator", "converge", "--sessions", "2", "--quiet", "--output", str(tmp_path / "out.json")],
    )

    simulator_main.main()

    assert calls == {"run": 0, "run_convergence": 1}
    data = json.loads((tmp_path / "out.json").read_text())
    assert len(data) == 2
    assert data[0]["learning"]["scorecard"]["overall"] == 90
    assert data[1]["learning"]["scorecard"]["overall"] == 91


def test_compass_module_entrypoint_delegates_to_simulator_main(monkeypatch) -> None:
    import compass.simulator

    called: list[str] = []
    monkeypatch.setattr(compass.simulator, "main", lambda: called.append("main"))

    runpy.run_module("compass.__main__", run_name="__main__")

    assert called == ["main"]
