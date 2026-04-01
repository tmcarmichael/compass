"""Focused tests for the headless simulator's GOAP handoff."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from simulator.__main__ import _parse_args
from simulator.runner import SimulationRunner
from tests.factories import make_game_state, make_spawn


class _FakeStep:
    def __init__(self, routine_name: str, ready: bool = True, *, apply_effects=None) -> None:
        self.routine_name = routine_name
        self._ready = ready
        self._apply_effects = apply_effects or (lambda ws: ws)
        self.seen_world_states: list[object] = []

    def preconditions_met(self, ws: object) -> bool:
        self.seen_world_states.append(ws)
        return self._ready

    def apply_effects(self, ws: object) -> object:
        return self._apply_effects(ws)


class _FakePlan:
    def summary(self) -> str:
        return "Plan(fake)"


class _FakePlanner:
    def __init__(self, steps: list[_FakeStep], *, has_plan: bool) -> None:
        self._steps = steps
        self._index = 0
        self.plan = _FakePlan() if has_plan else None
        self.generate_calls: list[tuple[object, object]] = []
        self.advance_calls: list[object] = []
        self.invalidations: list[str] = []
        self.start_calls = 0

    def has_plan(self) -> bool:
        return self.plan is not None and self._index < len(self._steps)

    @property
    def current_step(self) -> _FakeStep | None:
        if self.has_plan():
            return self._steps[self._index]
        return None

    def advance(self, ws: object) -> None:
        self.advance_calls.append(ws)
        if not self.has_plan():
            return
        self._index += 1
        if self._index >= len(self._steps):
            self.plan = None

    def start_step(self, _ctx: object) -> None:
        self.start_calls += 1

    def invalidate(self, reason: str) -> None:
        self.invalidations.append(reason)
        self.plan = None

    def generate(self, ws: object, ctx: object) -> _FakePlan | None:
        self.generate_calls.append((ws, ctx))
        if self.plan is None and self._steps:
            self._index = 0
            self.plan = _FakePlan()
        return self.plan


def test_simulation_runner_defaults_to_phase_two() -> None:
    runner = SimulationRunner()

    assert runner._utility_phase == 2
    assert runner._brain.utility_phase == 2


def test_simulator_cli_defaults_to_phase_two(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["simulator"])

    args = _parse_args()

    assert args.utility_phase == 2


def test_tick_goap_planner_generates_plan_and_sets_suggestion() -> None:
    runner = SimulationRunner(enable_goap=False)
    planner = _FakePlanner([_FakeStep("ACQUIRE")], has_plan=False)
    runner._planner = planner

    runner._tick_goap_planner(make_game_state())

    assert len(planner.generate_calls) == 1
    assert planner.start_calls == 1
    assert runner._ctx.diag.goap_suggestion == "ACQUIRE"


def test_tick_goap_planner_advances_to_next_step_before_suggesting() -> None:
    runner = SimulationRunner(enable_goap=False)
    planner = _FakePlanner([_FakeStep("REST"), _FakeStep("ACQUIRE")], has_plan=True)
    runner._planner = planner

    runner._tick_goap_planner(make_game_state())

    assert len(planner.advance_calls) == 1
    assert planner.start_calls == 1
    assert runner._ctx.diag.goap_suggestion == "ACQUIRE"


def test_tick_goap_planner_invalidates_unready_steps() -> None:
    runner = SimulationRunner(enable_goap=False)
    planner = _FakePlanner([_FakeStep("PULL", ready=False)], has_plan=True)
    runner._planner = planner
    runner._brain._active = object()

    runner._tick_goap_planner(make_game_state())

    assert planner.invalidations == ["preconditions_failed"]
    assert runner._ctx.diag.goap_suggestion == ""


def test_tick_goap_planner_advances_with_completed_step_effects() -> None:
    runner = SimulationRunner(enable_goap=False)
    planner = _FakePlanner(
        [
            _FakeStep("ACQUIRE", apply_effects=lambda ws: ws.with_changes(has_target=True)),
            _FakeStep("PULL"),
        ],
        has_plan=True,
    )
    runner._planner = planner
    runner._brain._ticked_routine_name = "ACQUIRE"

    runner._tick_goap_planner(make_game_state())

    assert len(planner.advance_calls) == 1
    advanced_ws = planner.advance_calls[0]
    assert advanced_ws.has_target is True
    assert runner._ctx.diag.goap_suggestion == "PULL"


def test_tick_goap_planner_seeds_pull_target_after_acquire_completion() -> None:
    runner = SimulationRunner(enable_goap=False)
    planner = _FakePlanner([_FakeStep("ACQUIRE"), _FakeStep("PULL")], has_plan=True)
    runner._planner = planner
    runner._brain._ticked_routine_name = "ACQUIRE"
    state = make_game_state(spawns=(make_spawn(spawn_id=777, x=10.0, y=10.0),))
    runner._world.update(state)

    runner._tick_goap_planner(state)

    assert runner._ctx.combat.pull_target_id == 777


def test_run_marks_synthetic_assumptions_when_post_processing_applies() -> None:
    runner = SimulationRunner(enable_goap=False)
    scenario = SimpleNamespace(name="mini", states=[(make_game_state(), "combat")])

    result = runner.run(scenario)

    assert result.simulator_assumptions["headless_scenario"] is True
    assert result.simulator_assumptions["synthetic_step_handoff"] is True
    assert result.simulator_assumptions["synthetic_encounter_learning"] is True
    assert result.simulator_assumptions["synthetic_scorecard_inputs"] is True


def test_run_convergence_does_not_double_apply_synthetic_learning(monkeypatch) -> None:
    runner = SimulationRunner(enable_goap=False)
    scenario = SimpleNamespace(name="mini", states=[(make_game_state(), "combat")])
    calls = {"count": 0}
    original = runner._simulate_encounter_learning

    def _wrapped(scenario_obj, result_obj) -> None:
        calls["count"] += 1
        original(scenario_obj, result_obj)

    monkeypatch.setattr(runner, "_simulate_encounter_learning", _wrapped)

    results = runner.run_convergence(scenario, sessions=1)

    assert calls["count"] == 1
    assert len(results) == 1
