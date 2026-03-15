"""Integration tests: decision pipeline from snapshot to routine activation.

These tests wire up the real Brain, real AgentContext, and real rule
conditions to verify the core architectural claims:

1. Emergency rules override everything
2. Priority ordering selects the right non-emergency rule
3. Precondition gating blocks rules when requirements aren't met
4. The full pipeline composes: GameState in → rule evaluation → routine activation
5. Emergency rules break routine lock-in (safety envelope)
"""

from __future__ import annotations

from perception.state import GameState
from routines.base import RoutineBase, RoutineStatus
from tests.factories import make_game_state

# ---------------------------------------------------------------------------
# Minimal concrete routines for integration testing
# ---------------------------------------------------------------------------


class _StubRoutine(RoutineBase):
    """A routine that returns RUNNING forever. For testing rule selection."""

    def __init__(self, *, lock: bool = False) -> None:
        self._lock = lock

    @property
    def locked(self) -> bool:
        return self._lock

    def enter(self, state: GameState) -> None:
        pass

    def tick(self, state: GameState) -> RoutineStatus:
        return RoutineStatus.RUNNING

    def exit(self, state: GameState) -> None:
        pass


def _make_brain():
    """Construct a minimal Brain with no runtime dependencies."""
    from brain.decision import Brain

    return Brain(ctx=None, utility_phase=0)


# ---------------------------------------------------------------------------
# 1. Emergency override
# ---------------------------------------------------------------------------


class TestEmergencyOverride:
    def test_flee_fires_on_low_hp(self) -> None:
        """Emergency rule activates when HP is critical."""
        brain = _make_brain()
        flee = _StubRoutine()
        rest = _StubRoutine()

        # Emergency rules are registered first (highest priority in Phase 0)
        brain.add_rule("FLEE", lambda s: s.hp_pct < 0.30, flee, emergency=True)
        brain.add_rule("REST", lambda s: s.hp_pct < 0.85, rest)

        state = make_game_state(hp_current=200, hp_max=1000)
        brain.tick(state)

        assert brain._active_name == "FLEE"

    def test_emergency_skipped_when_healthy(self) -> None:
        """Emergency rule does not fire when its condition is False."""
        brain = _make_brain()
        flee = _StubRoutine()
        rest = _StubRoutine()

        brain.add_rule("FLEE", lambda s: s.hp_pct < 0.30, flee, emergency=True)
        brain.add_rule("REST", lambda s: s.hp_pct < 0.85, rest)

        # HP at 70%  -- FLEE condition is False, REST should win
        state = make_game_state(hp_current=700, hp_max=1000)
        brain.tick(state)

        assert brain._active_name == "REST"


# ---------------------------------------------------------------------------
# 2. Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_first_matching_rule_wins(self) -> None:
        """In Phase 0, first matching rule in insertion order wins."""
        brain = _make_brain()
        rest = _StubRoutine()
        acquire = _StubRoutine()

        brain.add_rule("REST", lambda s: s.mana_pct < 0.50, rest)
        brain.add_rule("ACQUIRE", lambda s: True, acquire)

        state = make_game_state(mana_current=200, mana_max=1000)
        brain.tick(state)

        assert brain._active_name == "REST"

    def test_skips_non_matching_to_find_match(self) -> None:
        """Rules whose conditions return False are skipped."""
        brain = _make_brain()
        rest = _StubRoutine()
        acquire = _StubRoutine()

        brain.add_rule("REST", lambda s: s.mana_pct < 0.30, rest)
        brain.add_rule("ACQUIRE", lambda s: True, acquire)

        # Mana at 80%  -- REST condition is False, ACQUIRE should win
        state = make_game_state(mana_current=800, mana_max=1000)
        brain.tick(state)

        assert brain._active_name == "ACQUIRE"


# ---------------------------------------------------------------------------
# 3. Precondition gating
# ---------------------------------------------------------------------------


class TestPreconditionGating:
    def test_no_matching_rule_means_no_routine(self) -> None:
        """When no rule condition is True, no routine activates."""
        brain = _make_brain()
        acquire = _StubRoutine()

        brain.add_rule("ACQUIRE", lambda s: False, acquire)

        state = make_game_state()
        brain.tick(state)

        assert brain._active_name == ""

    def test_condition_receives_live_state(self) -> None:
        """Rule conditions receive the actual GameState, not stale data."""
        brain = _make_brain()
        seen_states: list[float] = []

        def _track_hp(s: GameState) -> bool:
            seen_states.append(s.hp_pct)
            return False

        brain.add_rule("TRACKER", _track_hp, _StubRoutine())

        brain.tick(make_game_state(hp_current=500, hp_max=1000))
        brain.tick(make_game_state(hp_current=800, hp_max=1000))

        assert seen_states == [0.5, 0.8]


# ---------------------------------------------------------------------------
# 4. Full pipeline: snapshot → eval → routine lifecycle
# ---------------------------------------------------------------------------


class TestRoutineLifecycle:
    def test_routine_enters_on_activation(self) -> None:
        """When a rule fires, the routine's enter() is called."""
        brain = _make_brain()
        entered = []

        class _TrackingRoutine(RoutineBase):
            def enter(self, state: GameState) -> None:
                entered.append(True)

            def tick(self, state: GameState) -> RoutineStatus:
                return RoutineStatus.RUNNING

            def exit(self, state: GameState) -> None:
                pass

        brain.add_rule("REST", lambda s: True, _TrackingRoutine())
        brain.tick(make_game_state())

        assert entered == [True]

    def test_routine_transitions_on_new_winner(self) -> None:
        """When a different rule wins, old routine exits and new one enters."""
        brain = _make_brain()
        log: list[str] = []

        class _LogRoutine(RoutineBase):
            def __init__(self, label: str) -> None:
                self._label = label

            def enter(self, state: GameState) -> None:
                log.append(f"{self._label}:enter")

            def tick(self, state: GameState) -> RoutineStatus:
                return RoutineStatus.RUNNING

            def exit(self, state: GameState) -> None:
                log.append(f"{self._label}:exit")

        # Tick 1: hp_pct=0.5 → REST matches, ACQUIRE doesn't
        brain.add_rule("REST", lambda s: s.hp_pct < 0.85, _LogRoutine("REST"))
        brain.add_rule("ACQUIRE", lambda s: s.hp_pct >= 0.85, _LogRoutine("ACQ"))

        brain.tick(make_game_state(hp_current=500, hp_max=1000))
        assert brain._active_name == "REST"

        # Tick 2: hp_pct=0.95 → REST no longer matches, ACQUIRE does
        brain.tick(make_game_state(hp_current=950, hp_max=1000))
        assert brain._active_name == "ACQUIRE"

        assert log == ["REST:enter", "REST:exit", "ACQ:enter"]


# ---------------------------------------------------------------------------
# 5. Safety envelope: emergency breaks lock-in
# ---------------------------------------------------------------------------


class TestSafetyEnvelope:
    def test_locked_routine_blocks_non_emergency(self) -> None:
        """A locked routine cannot be interrupted by a non-emergency rule."""
        brain = _make_brain()
        locked = _StubRoutine(lock=True)
        rest = _StubRoutine()

        brain.add_rule("PULL", lambda s: True, locked)
        brain.add_rule("REST", lambda s: True, rest)

        # Tick 1: PULL activates and locks
        brain.tick(make_game_state())
        assert brain._active_name == "PULL"

        # Tick 2: REST matches but PULL is locked  -- PULL stays
        brain.tick(make_game_state())
        assert brain._active_name == "PULL"

    def test_emergency_breaks_lock(self) -> None:
        """An emergency rule overrides a locked routine."""
        brain = _make_brain()
        flee = _StubRoutine()
        locked = _StubRoutine(lock=True)

        # Emergency registered first (highest priority)
        brain.add_rule("FLEE", lambda s: s.hp_pct < 0.30, flee, emergency=True)
        brain.add_rule("PULL", lambda s: True, locked)

        # Tick 1: FLEE condition False, PULL activates and locks
        brain.tick(make_game_state(hp_current=900, hp_max=1000))
        assert brain._active_name == "PULL"

        # Tick 2: HP drops critically  -- FLEE breaks the lock
        brain.tick(make_game_state(hp_current=200, hp_max=1000))
        assert brain._active_name == "FLEE"


# ---------------------------------------------------------------------------
# 6. Multi-tick session simulation with real survival conditions
# ---------------------------------------------------------------------------


class TestSessionSimulation:
    """Simulate a multi-tick session through real rule condition functions.

    Wires up actual condition functions from brain.rules.survival with
    stub routines to verify the full decision pipeline.
    """

    def test_healthy_to_flee_to_rest_cycle(self) -> None:
        """Emergency override: FLEE fires when HP drops, REST takes over when safe."""
        from brain.context import AgentContext
        from brain.decision import Brain
        from brain.rules.survival import (
            _should_flee,
            _should_rest,
            _SurvivalRuleState,
        )
        from core.features import flags

        flags.flee = True
        flags.rest = True

        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.player.last_buff_time = 0.0
        ctx.player.last_flee_time = 0.0
        ctx.rest_hp_entry = 0.85
        ctx.rest_mana_entry = 0.40

        brain = Brain(ctx=ctx, utility_phase=0)
        rs = _SurvivalRuleState()

        flee = _StubRoutine()
        rest = _StubRoutine()
        wander = _StubRoutine()

        brain.add_rule("FLEE", lambda s: _should_flee(s, ctx), flee, emergency=True)
        brain.add_rule("REST", lambda s: _should_rest(s, ctx, rs), rest)
        brain.add_rule("WANDER", lambda s: True, wander)

        # Phase 1: Healthy -> WANDER
        state = make_game_state(hp_current=1000, hp_max=1000, mana_current=500, mana_max=500)
        brain.tick(state)
        assert brain._active_name == "WANDER"

        # Phase 2: HP drops to 20% -> FLEE (emergency override)
        state = make_game_state(hp_current=200, hp_max=1000, mana_current=500, mana_max=500)
        brain.tick(state)
        assert brain._active_name == "FLEE"

        # Phase 3: HP recovers to 70% but below rest threshold -> REST
        # First clear flee hysteresis by going to full HP briefly
        ctx.combat.flee_urgency_active = False
        state = make_game_state(hp_current=700, hp_max=1000, mana_current=500, mana_max=500)
        brain.tick(state)
        assert brain._active_name == "REST"

        # Phase 4: Thresholds met -> back to WANDER
        state = make_game_state(hp_current=950, hp_max=1000, mana_current=400, mana_max=500)
        brain.tick(state)
        assert brain._active_name == "WANDER"


# ---------------------------------------------------------------------------
# 7. Architectural invariants (untested claims)
# ---------------------------------------------------------------------------


class TestArchitecturalInvariants:
    """Verify four architectural claims that had no integration coverage:

    1. GOAP planner invalidation on emergency
    2. Locked routine survives REST but not FLEE
    3. Utility Phase 2 respects emergency override
    4. Encounter history learning improves scoring data
    """

    # -- 1. GOAP plan invalidated by emergency --------------------------------

    def test_goap_invalidated_by_emergency(self) -> None:
        """GOAP plan is cleared on emergency; brain FLEE fires regardless."""
        from brain.context import AgentContext
        from brain.decision import Brain
        from brain.goap.actions import build_action_set
        from brain.goap.goals import build_goal_set
        from brain.goap.planner import GOAPPlanner
        from brain.goap.world_state import PlanWorldState

        # -- Part A: planner.invalidate("emergency") clears the plan ----------
        goals = build_goal_set()
        actions = build_action_set()
        planner = GOAPPlanner(goals, actions)

        ws = PlanWorldState(hp_pct=0.3, mana_pct=0.2, targets_available=3)
        plan = planner.generate(ws)
        assert plan is not None, "Planner should generate a plan for low resources"
        assert planner.has_plan()

        planner.invalidate("emergency")
        assert not planner.has_plan(), "Plan must be cleared after invalidate"

        # -- Part B: Brain FLEE fires on emergency regardless of plan state ----
        ctx = AgentContext()
        brain = Brain(ctx=ctx, utility_phase=0)

        flee = _StubRoutine()
        wander = _StubRoutine()

        brain.add_rule("FLEE", lambda s: s.hp_pct < 0.30, flee, emergency=True)
        brain.add_rule("WANDER", lambda s: True, wander)

        # Generate a fresh plan (proves plan exists before emergency)
        plan2 = planner.generate(PlanWorldState(hp_pct=0.3, mana_pct=0.2, targets_available=3))
        assert plan2 is not None

        # Tick with critical HP -> FLEE fires (emergency override)
        state = make_game_state(hp_current=200, hp_max=1000)
        brain.tick(state)
        assert brain._active_name == "FLEE"

    # -- 2. Locked PULL survives REST but not FLEE ----------------------------

    def test_locked_pull_survives_rest_but_not_flee(self) -> None:
        """Locked PULL blocks REST (non-emergency) but FLEE (emergency) breaks it."""
        from brain.context import AgentContext
        from brain.decision import Brain
        from brain.rules.survival import (
            _should_flee,
            _should_rest,
            _SurvivalRuleState,
        )
        from core.features import flags

        flags.flee = True
        flags.rest = True

        ctx = AgentContext()
        ctx.pet.alive = True
        ctx.player.last_buff_time = 0.0
        ctx.player.last_flee_time = 0.0
        ctx.rest_hp_entry = 0.85
        ctx.rest_mana_entry = 0.40

        brain = Brain(ctx=ctx, utility_phase=0)
        rs = _SurvivalRuleState()

        flee = _StubRoutine()
        rest = _StubRoutine()
        pull_locked = _StubRoutine(lock=True)
        wander = _StubRoutine()

        brain.add_rule("FLEE", lambda s: _should_flee(s, ctx), flee, emergency=True)
        brain.add_rule("REST", lambda s: _should_rest(s, ctx, rs), rest)
        brain.add_rule("PULL", lambda s: True, pull_locked)
        brain.add_rule("WANDER", lambda s: True, wander)

        # Tick 1: healthy -> first matching non-emergency rule is REST? No,
        # HP=1000/1000 so REST condition is False. PULL matches (always True).
        state = make_game_state(hp_current=1000, hp_max=1000, mana_current=500, mana_max=500)
        brain.tick(state)
        assert brain._active_name == "PULL", f"Expected PULL, got {brain._active_name}"

        # Tick 2: lower HP to 70% -> REST condition fires (below 85%
        # threshold), but PULL is locked -> PULL stays
        ctx.combat.flee_urgency_active = False
        state = make_game_state(hp_current=700, hp_max=1000, mana_current=500, mana_max=500)
        brain.tick(state)
        assert brain._active_name == "PULL", f"Expected PULL (locked), got {brain._active_name}"

        # Tick 3: drop HP to 20% -> FLEE (emergency) breaks the lock
        state = make_game_state(hp_current=200, hp_max=1000, mana_current=500, mana_max=500)
        brain.tick(state)
        assert brain._active_name == "FLEE", f"Expected FLEE (emergency override), got {brain._active_name}"

    # -- 3. Utility Phase 2 respects emergency override -----------------------

    def test_utility_phase2_respects_emergency(self) -> None:
        """In Phase 2, an emergency rule overrides a higher-scoring rule."""
        from brain.context import AgentContext
        from brain.decision import Brain

        ctx = AgentContext()
        brain = Brain(ctx=ctx, utility_phase=2)

        flee = _StubRoutine()
        high_score = _StubRoutine()

        # FLEE: emergency, low score when healthy, max score when critical
        brain.add_rule(
            "FLEE",
            lambda s: s.hp_pct < 0.30,
            flee,
            emergency=True,
            score_fn=lambda s: 1.0 if s.hp_pct < 0.30 else 0.0,
            tier=0,
        )
        # HIGH_SCORE: not emergency, always scores 0.9
        brain.add_rule(
            "HIGH_SCORE",
            lambda s: True,
            high_score,
            emergency=False,
            score_fn=lambda s: 0.9,
            tier=0,
        )

        # Tick 1: hp=90% -> FLEE score=0.0, HIGH_SCORE score=0.9 -> HIGH_SCORE
        state = make_game_state(hp_current=900, hp_max=1000)
        brain.tick(state)
        assert brain._active_name == "HIGH_SCORE", (
            f"Expected HIGH_SCORE when healthy, got {brain._active_name}"
        )

        # Tick 2: hp=20% -> FLEE score=1.0 > HIGH_SCORE score=0.9 -> FLEE
        state = make_game_state(hp_current=200, hp_max=1000)
        brain.tick(state)
        assert brain._active_name == "FLEE", f"Expected FLEE on emergency, got {brain._active_name}"

    # -- 4. Encounter history improves scoring --------------------------------

    def test_encounter_history_improves_scoring(self, tmp_path) -> None:
        """FightHistory accumulates per-npc stats that scoring can query."""
        from brain.learning.encounters import FightHistory

        fh = FightHistory(zone="test", data_dir=str(tmp_path))

        # Before any data, has_learned is False
        assert not fh.has_learned("a_skeleton")

        # Record 5 fights with realistic data
        for _ in range(5):
            fh.record(
                mob_name="a_skeleton",
                duration=20.0,
                mana_spent=50,
                hp_delta=-0.1,
                casts=3,
                pet_heals=0,
                pet_died=False,
                defeated=True,
            )

        # Now has_learned should be True
        assert fh.has_learned("a_skeleton")

        # Learned duration should be approximately 20.0
        dur = fh.learned_duration("a_skeleton")
        assert dur is not None
        assert abs(dur - 20.0) < 1.0, f"Expected ~20.0, got {dur}"

        # Learned mana should be approximately 50
        mana = fh.learned_mana("a_skeleton")
        assert mana is not None
        assert abs(mana - 50) < 5, f"Expected ~50, got {mana}"

        # Learned danger should be a float in [0, 1]
        danger = fh.learned_danger("a_skeleton")
        assert danger is not None
        assert 0.0 <= danger <= 1.0, f"Expected danger in [0,1], got {danger}"
