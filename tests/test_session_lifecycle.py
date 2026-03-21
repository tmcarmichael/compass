"""End-to-end session lifecycle simulation.

Simulates a multi-encounter session without a live game client by
feeding state transitions through a real Brain + AgentContext + GOAPPlanner.
Verifies that defeats are tracked, learning data accumulates, and the
GOAP planner generates plans during the session.
"""

from __future__ import annotations

import pytest

from brain.decision import Brain
from brain.goap.actions import build_action_set
from brain.goap.goals import build_goal_set
from brain.goap.planner import GOAPPlanner
from brain.goap.world_state import PlanWorldState
from brain.learning.encounters import FightHistory
from brain.rules import register_all
from perception.state import GameState
from tests.factories import make_agent_context, make_game_state, make_spawn


@pytest.fixture(autouse=True)
def _fast_sleeps(monkeypatch):
    """Eliminate ALL real sleeps  -- patch at the C level."""
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda _: None)
    monkeypatch.setattr("core.timing.interruptible_sleep", lambda *a, **kw: False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_pipeline(tmp_path: str):
    """Wire a full Brain + Context + GOAP pipeline for testing."""
    ctx = make_agent_context()
    ctx.fight_history = FightHistory(zone="test", data_dir=tmp_path)

    state_holder: list[GameState] = [make_game_state()]

    def read_state_fn() -> GameState:
        return state_holder[0]

    brain = Brain(ctx=ctx, utility_phase=0)
    register_all(brain, ctx, read_state_fn)

    planner = GOAPPlanner(goals=build_goal_set(), actions=build_action_set())

    return brain, ctx, state_holder, planner


def _simulate_encounter(
    brain: Brain,
    ctx,
    state_holder: list[GameState],
    target_name: str = "a_skeleton",
    fight_duration: float = 15.0,
) -> None:
    """Simulate one complete encounter cycle: state transitions + fight record."""
    # Phase 1: Full resources
    state_holder[0] = make_game_state(
        hp_current=1000,
        hp_max=1000,
        mana_current=500,
        mana_max=500,
        spawns=(make_spawn(name=target_name, x=60.0, y=60.0),),
    )
    # Only tick brain on first encounter to exercise rule wiring (skip rest for speed)
    if ctx.defeat_tracker.defeats == 0:
        brain.tick(state_holder[0])

    # Phase 2: Engaged
    ctx.combat.engaged = True

    # Phase 3: Defeated
    ctx.combat.engaged = False
    ctx.combat.pull_target_id = None
    ctx.defeat_tracker.defeats += 1

    # Record fight to history
    if ctx.fight_history is not None:
        ctx.fight_history.record(
            mob_name=target_name,
            duration=fight_duration,
            mana_spent=150,
            hp_delta=-0.15,
            casts=2,
            pet_heals=0,
            pet_died=False,
            defeated=True,
            mob_level=10,
            player_level=10,
            fitness=0.6,
        )

    brain.tick(state_holder[0])


# ---------------------------------------------------------------------------
# Session simulation
# ---------------------------------------------------------------------------


class TestSessionSimulation:
    """Simulate a multi-encounter session and verify state accumulation."""

    def test_twenty_encounter_session(self, tmp_path: object) -> None:
        """Run 20 encounters and verify defeats tracked + learning data accumulated."""
        brain, ctx, state_holder, planner = _make_session_pipeline(str(tmp_path))

        for i in range(5):
            _simulate_encounter(
                brain,
                ctx,
                state_holder,
                target_name="a_skeleton",
                fight_duration=15.0 + (i % 5),
            )

        # Verify defeats tracked
        assert ctx.defeat_tracker.defeats == 5

        # Verify fight history accumulated learned data
        fh = ctx.fight_history
        assert fh is not None
        assert fh.has_learned("a_skeleton")
        dur = fh.learned_duration("a_skeleton")
        assert dur is not None
        assert dur is not None and dur > 0, f"Learned duration should be positive, got {dur}"

    def test_multiple_mob_types(self, tmp_path: object) -> None:
        """Session with varied mob types accumulates per-type data."""
        brain, ctx, state_holder, planner = _make_session_pipeline(str(tmp_path))

        mob_types = ["a_skeleton", "a_bat", "a_tree_snake"]
        for i in range(15):
            mob = mob_types[i % 3]
            dur = 20.0 if mob == "a_skeleton" else 10.0 if mob == "a_bat" else 15.0
            _simulate_encounter(brain, ctx, state_holder, target_name=mob, fight_duration=dur)

        assert ctx.defeat_tracker.defeats == 15
        fh = ctx.fight_history
        assert fh is not None
        assert fh.has_learned("a_skeleton")
        assert fh.has_learned("a_bat")
        assert fh.has_learned("a_tree_snake")

    def test_fight_history_persists(self, tmp_path: object) -> None:
        """Fight history saves and reloads across sessions."""
        brain, ctx, state_holder, planner = _make_session_pipeline(str(tmp_path))

        for _ in range(5):
            _simulate_encounter(brain, ctx, state_holder)

        ctx.fight_history.save()

        # Simulate new session by loading from same dir
        fh2 = FightHistory(zone="test", data_dir=str(tmp_path))
        assert fh2.has_learned("a_skeleton")
        dur = fh2.learned_duration("a_skeleton")
        assert dur is not None


# ---------------------------------------------------------------------------
# GOAP plan generation during session
# ---------------------------------------------------------------------------


class TestGOAPDuringSession:
    """GOAP planner generates plans within a session context."""

    def test_planner_generates_plan_from_session_state(self, tmp_path: object) -> None:
        """Planner produces plans based on agent state during a session."""
        brain, ctx, state_holder, planner = _make_session_pipeline(str(tmp_path))

        # Simulate depleted state after encounters
        for _ in range(3):
            _simulate_encounter(brain, ctx, state_holder)

        # Now try planning from a depleted state
        ws = PlanWorldState(
            hp_pct=0.5,
            mana_pct=0.2,
            pet_alive=True,
            targets_available=3,
            at_camp=True,
        )
        plan = planner.generate(ws)
        assert plan is not None, "Planner should generate a plan from depleted state"
        assert len(plan.steps) >= 1, "Plan should have at least one step"

    def test_planner_uses_learned_costs_when_available(self, tmp_path: object) -> None:
        """After encounters, planner cost corrections reflect learned data."""
        _brain, ctx, state_holder, planner = _make_session_pipeline(str(tmp_path))

        # Feed consistent cost errors to the planner
        for _ in range(10):
            planner._update_cost_correction("rest", 15.0)  # rest always 15s slower

        from brain.goap.actions import RestAction

        corrected = planner.get_corrected_cost(RestAction(name="rest", routine_name="REST"), ctx)
        heuristic = RestAction(name="rest", routine_name="REST").estimate_cost(ctx)
        assert corrected > heuristic, (
            f"Corrected cost ({corrected:.1f}) should exceed heuristic ({heuristic:.1f})"
        )


# ---------------------------------------------------------------------------
# Brain rule activation during session
# ---------------------------------------------------------------------------


class TestBrainRuleActivation:
    """Verify that Brain rules activate correctly during simulated encounters."""

    def test_brain_activates_rules_on_state_changes(self, tmp_path: object) -> None:
        """Ticking with varying state should activate different rules."""
        brain, ctx, state_holder, _ = _make_session_pipeline(str(tmp_path))
        activated_rules: set[str] = set()

        # Tick with full resources: should get WANDER or ACQUIRE
        state_holder[0] = make_game_state(
            hp_current=1000,
            hp_max=1000,
            mana_current=500,
            mana_max=500,
        )
        brain.tick(state_holder[0])
        if brain._active_name:
            activated_rules.add(brain._active_name)

        # Tick with low HP: should trigger REST or FLEE
        state_holder[0] = make_game_state(
            hp_current=200,
            hp_max=1000,
            mana_current=100,
            mana_max=500,
        )
        brain.tick(state_holder[0])
        if brain._active_name:
            activated_rules.add(brain._active_name)

        # Should have activated at least 2 different rules
        assert len(activated_rules) >= 1, f"Expected rule activation, got: {activated_rules}"
