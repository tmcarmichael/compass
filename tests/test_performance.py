"""Performance benchmarks: prove the 10 Hz tick budget with real measurements.

Uses pytest-benchmark for statistically rigorous timing. Each benchmark
proves a specific claim from the architecture docs:

- Brain tick (Phase 0, 2, 3): full rule evaluation + routine tick < 10ms
- GOAP plan generation: A* search completes within 50ms budget
- Target scoring: 20-entity scoring pass completes in < 5ms

Run benchmarks:     just benchmark
Run tests only:     uv run pytest tests/ --benchmark-disable
"""

from __future__ import annotations

import pytest

from brain.decision import Brain
from brain.goap.actions import build_action_set
from brain.goap.goals import build_goal_set
from brain.goap.planner import GOAPPlanner
from brain.goap.world_state import PlanWorldState
from brain.scoring.target import ScoringWeights, score_target
from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus
from tests.factories import make_game_state, make_mob_profile, make_spawn

# Skip module when pytest-benchmark plugin isn't registered (e.g. free-threaded
# 3.14t). CI uses --benchmark-disable which requires the plugin; this handles
# local runs where the plugin isn't loaded.
try:
    _pm = pytest.importorskip("pytest_benchmark.plugin")
    # Verify the plugin actually registered (the entry point may fail silently)
    if not hasattr(_pm, "pytest_benchmark_generate_machine_info"):
        raise ImportError("plugin not functional")
except Exception:
    pytest.skip("pytest-benchmark plugin not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _NoOpRoutine(RoutineBase):
    def enter(self, state: GameState) -> None:
        pass

    def tick(self, state: GameState) -> RoutineStatus:
        return RoutineStatus.RUNNING

    def exit(self, state: GameState) -> None:
        pass


def _cond_true(s: GameState) -> bool:
    return True


def _cond_false(s: GameState) -> bool:
    return False


def _score_high(s: GameState) -> float:
    return 0.8


def _score_low(s: GameState) -> float:
    return 0.1


def _brain_with_rules(n_rules: int = 14, *, phase: int = 0, first_matches: bool = True) -> Brain:
    """Build a Brain with n rules. If first_matches, rule 0 returns True (short-circuit)."""
    brain = Brain(utility_phase=phase)
    for i in range(n_rules):
        cond = _cond_true if (first_matches and i == 0) else _cond_false
        score = _score_high if (first_matches and i == 0) else _score_low
        brain.add_rule(
            f"RULE_{i}",
            cond,
            _NoOpRoutine(),
            score_fn=score,
            tier=i % 3,
            weight=1.0 + i * 0.1,
        )
    return brain


def _state_with_spawns(n: int = 20) -> GameState:
    """GameState with n NPC spawns at varied positions."""
    spawns = tuple(
        make_spawn(spawn_id=100 + i, x=float(i * 15), y=float(i * 10), level=8 + i % 5) for i in range(n)
    )
    return make_game_state(spawns=spawns, hp_current=800, hp_max=1000, mana_current=400, mana_max=500)


# ---------------------------------------------------------------------------
# Brain tick benchmarks
# ---------------------------------------------------------------------------


class TestBrainTickBenchmark:
    """Brain.tick() must complete well under 10ms (100ms / 10 Hz budget)."""

    def test_phase0_short_circuit(self, benchmark) -> None:
        """Phase 0: first rule matches, rest short-circuited. Best case."""
        brain = _brain_with_rules(14, phase=0, first_matches=True)
        state = _state_with_spawns(20)
        brain.tick(state)  # warm up

        benchmark(brain.tick, state)
        # Pedantic assertion: benchmark.stats available after run
        if benchmark.stats:
            assert benchmark.stats.stats.mean < 0.010  # 10ms

    def test_phase0_full_eval(self, benchmark) -> None:
        """Phase 0: no rule matches, all 14 evaluated. Worst case for Phase 0."""
        brain = _brain_with_rules(14, phase=0, first_matches=False)
        state = _state_with_spawns(20)
        brain.tick(state)

        benchmark(brain.tick, state)
        if benchmark.stats:
            assert benchmark.stats.stats.mean < 0.010  # 10ms

    def test_phase2_tier_scoring(self, benchmark) -> None:
        """Phase 2: score functions evaluated per tier. More work than Phase 0."""
        brain = _brain_with_rules(14, phase=2, first_matches=True)
        state = _state_with_spawns(20)
        brain.tick(state)

        benchmark(brain.tick, state)
        if benchmark.stats:
            assert benchmark.stats.stats.mean < 0.010  # 10ms

    def test_phase3_weighted_scoring(self, benchmark) -> None:
        """Phase 3: all rules scored and weighted. Maximum decision overhead."""
        brain = _brain_with_rules(14, phase=3, first_matches=True)
        state = _state_with_spawns(20)
        brain.tick(state)

        benchmark(brain.tick, state)
        if benchmark.stats:
            assert benchmark.stats.stats.mean < 0.010  # 10ms


# ---------------------------------------------------------------------------
# GOAP planner benchmarks
# ---------------------------------------------------------------------------


class TestGOAPBenchmark:
    """GOAP plan generation must complete within the 50ms budget."""

    def test_plan_from_low_resources(self, benchmark) -> None:
        """Plan generation from depleted state (worst case: longest search)."""
        planner = GOAPPlanner(goals=build_goal_set(), actions=build_action_set())
        ws = PlanWorldState(hp_pct=0.3, mana_pct=0.2, pet_alive=True, targets_available=5)

        def generate():
            planner._plan = None  # reset so it replans each iteration
            return planner.generate(ws)

        benchmark(generate)
        if benchmark.stats:
            assert benchmark.stats.stats.mean < 0.050  # 50ms budget

    def test_plan_from_ready_state(self, benchmark) -> None:
        """Plan generation from healthy state (typical case)."""
        planner = GOAPPlanner(goals=build_goal_set(), actions=build_action_set())
        ws = PlanWorldState(hp_pct=0.8, mana_pct=0.7, pet_alive=True, targets_available=3)

        def generate():
            planner._plan = None
            return planner.generate(ws)

        benchmark(generate)
        if benchmark.stats:
            assert benchmark.stats.stats.mean < 0.050  # 50ms budget


# ---------------------------------------------------------------------------
# Target scoring benchmarks
# ---------------------------------------------------------------------------


class TestScoringBenchmark:
    """Scoring 20 targets must complete in < 5ms (runs every tick at 10 Hz)."""

    def test_score_20_targets(self, benchmark) -> None:
        """Score 20 mob profiles with default weights. Typical tick workload."""
        weights = ScoringWeights()
        profiles = [
            make_mob_profile(
                spawn=make_spawn(spawn_id=100 + i, x=float(i * 15), y=float(i * 10), level=8 + i % 5),
                distance=20.0 + i * 5,
                camp_distance=30.0 + i * 3,
            )
            for i in range(20)
        ]
        players: list[object] = []

        def score_all():
            return [score_target(p, weights, profiles, players) for p in profiles]

        benchmark(score_all)
        if benchmark.stats:
            assert benchmark.stats.stats.mean < 0.005  # 5ms
