"""Session-level simulation: multi-phase scenarios through the full pipeline.

Drives Brain through scripted GameState sequences representing minutes of
game time, asserting on the sequence of routine activations. Catches
inter-routine interaction bugs that unit tests miss.

Uses register_all() with real rules  -- same pattern as test_pipeline.py
but with longer scenarios and richer assertions.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from brain.context import AgentContext
from brain.decision import Brain
from brain.rules import register_all
from core.features import flags
from perception.state import GameState
from tests.factories import make_game_state, make_spawn

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _enable_flags() -> None:
    flags.flee = True
    flags.rest = True
    flags._wander = True
    flags._pull = True


@pytest.fixture(autouse=True)
def _fast_sleeps():
    with patch("core.timing.interruptible_sleep", return_value=False):
        with patch("time.sleep"):
            yield


class SessionResult:
    """Captures the sequence of decisions across a simulated session."""

    def __init__(self) -> None:
        self.routine_log: list[str] = []
        self.rule_matches: list[str] = []
        self.tick_count: int = 0

    @property
    def unique_routines(self) -> set[str]:
        return {r for r in self.routine_log if r}


class SessionSimulator:
    """Drive Brain through a scripted game scenario."""

    def __init__(self, brain: Brain, ctx: AgentContext, state_holder: list[GameState]) -> None:
        self.brain = brain
        self.ctx = ctx
        self.state_holder = state_holder

    def run(self, states: list[GameState]) -> SessionResult:
        result = SessionResult()
        for state in states:
            self.state_holder[0] = state
            self.brain.tick(state)
            result.routine_log.append(self.brain._active_name)
            result.rule_matches.append(self.brain._last_matched_rule)
            result.tick_count += 1
        return result


class ScenarioBuilder:
    """Fluent builder for GameState sequences."""

    def __init__(self, **base_kw) -> None:
        self._base = dict(
            hp_current=1000,
            hp_max=1000,
            mana_current=500,
            mana_max=500,
            level=10,
        )
        self._base.update(base_kw)
        self._states: list[GameState] = []

    def idle(self, ticks: int = 5) -> ScenarioBuilder:
        for _ in range(ticks):
            self._states.append(make_game_state(**self._base))
        return self

    def damage(self, hp_pct: float, ticks: int = 1) -> ScenarioBuilder:
        hp = int(hp_pct * self._base["hp_max"])
        kw = {**self._base, "hp_current": hp}
        for _ in range(ticks):
            self._states.append(make_game_state(**kw))
        return self

    def drain_mana(self, mana_pct: float, ticks: int = 1) -> ScenarioBuilder:
        mana = int(mana_pct * self._base["mana_max"])
        kw = {**self._base, "mana_current": mana}
        for _ in range(ticks):
            self._states.append(make_game_state(**kw))
        return self

    def recover(self, ticks: int = 3) -> ScenarioBuilder:
        for _ in range(ticks):
            self._states.append(make_game_state(**self._base))
        return self

    def build(self) -> list[GameState]:
        return list(self._states)


def _make_session() -> tuple[SessionSimulator, AgentContext]:
    """Create a full session: Brain + real rules + simulator."""
    ctx = AgentContext()
    ctx.pet.alive = True
    ctx.player.last_buff_time = 0.0
    ctx.player.last_flee_time = 0.0
    ctx.rest_hp_entry = 0.85
    ctx.rest_mana_entry = 0.40

    state_holder: list[GameState] = [make_game_state()]

    def read_state_fn() -> GameState:
        return state_holder[0]

    brain = Brain(ctx=ctx, utility_phase=0)
    register_all(brain, ctx, read_state_fn)

    sim = SessionSimulator(brain, ctx, state_holder)
    return sim, ctx


# ---------------------------------------------------------------------------
# Session scenarios
# ---------------------------------------------------------------------------


class TestSessionScenarios:
    def test_emergency_interrupts_any_phase(self) -> None:
        """HP crash during normal operation triggers FLEE immediately."""
        sim, ctx = _make_session()
        states = ScenarioBuilder().idle(ticks=3).damage(hp_pct=0.10, ticks=3).build()
        result = sim.run(states)

        # FLEE must appear in the sequence after damage
        assert "FLEE" in result.rule_matches[3:], (
            f"FLEE should fire after HP crash, got: {result.rule_matches}"
        )

    def test_rest_when_low_mana(self) -> None:
        """Low mana triggers REST."""
        sim, ctx = _make_session()
        states = ScenarioBuilder().drain_mana(mana_pct=0.10, ticks=5).build()
        result = sim.run(states)
        assert "REST" in result.rule_matches, f"REST should fire on low mana, got: {result.rule_matches}"

    def test_flee_overrides_rest(self) -> None:
        """When both FLEE and REST conditions met, FLEE wins (emergency)."""
        sim, ctx = _make_session()
        # Low HP + low mana
        states = []
        for _ in range(5):
            states.append(
                make_game_state(
                    hp_current=100,
                    hp_max=1000,
                    mana_current=50,
                    mana_max=500,
                )
            )
        result = sim.run(states)
        assert result.rule_matches[-1] == "FLEE", f"FLEE should override REST, got: {result.rule_matches}"

    def test_no_oscillation_under_stable_state(self) -> None:
        """Stable state should not cause rapid rule switching."""
        sim, ctx = _make_session()
        states = ScenarioBuilder().idle(ticks=20).build()
        result = sim.run(states)

        # Count transitions (adjacent different non-empty rules)
        transitions = 0
        for i in range(1, len(result.rule_matches)):
            if (
                result.rule_matches[i] != result.rule_matches[i - 1]
                and result.rule_matches[i]
                and result.rule_matches[i - 1]
            ):
                transitions += 1

        # At most ~3 transitions in 20 ticks (initial settle + maybe one switch)
        assert transitions <= 5, (
            f"Too many transitions ({transitions}) in stable state: {result.rule_matches}"
        )

    def test_multiple_phases_in_sequence(self) -> None:
        """Session goes through multiple phases as state changes."""
        sim, ctx = _make_session()
        states = (
            ScenarioBuilder()
            .idle(ticks=3)  # should select ACQUIRE or WANDER
            .damage(0.10, ticks=3)  # should FLEE
            .recover(ticks=3)  # should REST then recover
            .idle(ticks=3)  # should return to normal
            .build()
        )
        result = sim.run(states)

        # Verify we saw at least 2 different routines
        active = {r for r in result.routine_log if r}
        assert len(active) >= 2, f"Expected multiple routines in multi-phase session, got: {active}"


# ---------------------------------------------------------------------------
# Documentation-linked scenarios
# Source: docs/samples/ artifacts from real sessions
# ---------------------------------------------------------------------------


class TestDocumentedBehaviors:
    """Reproduce specific scenarios from docs/samples/ telemetry."""

    def test_aggro_during_idle_triggers_flee(self) -> None:
        """Emergency override: HP crashes while agent is idle.

        Inspired by docs/samples/forensics-ring-buffer.md: skeleton aggroes
        a sitting player. The brain must fire FLEE within 1 tick of HP
        dropping below the safety floor.
        """
        sim, ctx = _make_session()

        attacker = make_spawn(
            spawn_id=1008,
            name="a_skeleton008",
            x=900.0,
            y=-4.0,
            hp_current=100,
            hp_max=100,
            level=8,
        )
        stable = make_game_state(
            hp_current=1000,
            hp_max=1000,
            mana_current=500,
            mana_max=500,
        )
        # HP crashes below FLEE safety floor (~15%)
        attacked = make_game_state(
            hp_current=100,
            hp_max=1000,
            mana_current=500,
            mana_max=500,
            target=attacker,
            spawns=(attacker,),
        )

        states = [stable] * 5 + [attacked] * 3
        result = sim.run(states)

        flee_fired = "FLEE" in result.rule_matches[5:]
        assert flee_fired, f"FLEE should fire after HP crash, got: {result.rule_matches}"

    def test_low_resources_triggers_rest(self) -> None:
        """Low HP + low mana (but above FLEE threshold) triggers REST.

        Source: docs/evolution.md, hysteresis thresholds.
        Rest entry: HP < 85% OR mana < 40%. At 70% HP + 15% mana,
        REST should fire (not FLEE, since HP is above safety floor).
        """
        sim, ctx = _make_session()

        states = [
            make_game_state(
                hp_current=700,
                hp_max=1000,
                mana_current=75,
                mana_max=500,
            )
        ] * 5
        result = sim.run(states)

        assert "REST" in result.rule_matches, (
            f"REST should fire at 70% HP + 15% mana, got: {result.rule_matches}"
        )
        assert "FLEE" not in result.rule_matches, (
            f"FLEE should NOT fire at 70% HP, got: {result.rule_matches}"
        )

    def test_healthy_agent_acquires_targets(self) -> None:
        """At full resources with pet alive, agent should ACQUIRE.

        Source: docs/samples/decision-trace.md  -- healthy state selects ACQUIRE.
        """
        sim, ctx = _make_session()
        states = ScenarioBuilder().idle(ticks=5).build()
        result = sim.run(states)

        assert "ACQUIRE" in result.rule_matches, f"Healthy agent should ACQUIRE, got: {result.rule_matches}"
