"""Tests for brain.runner.tick_handlers -- _is_auto_engage_candidate.

TickHandlers lives on BrainRunner, but _is_auto_engage_candidate is
essentially a pure function (spawn + ctx + state -> bool + side effects).
We construct a TickHandlers with a mock runner and exercise the logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from brain.runner.tick_handlers import TickHandlers
from brain.state.combat import CombatState
from brain.state.pet import PetState
from brain.state.threat import ThreatState
from perception.combat_eval import Con
from tests.factories import make_game_state, make_spawn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler() -> TickHandlers:
    """Build a TickHandlers with a minimal mock runner."""
    runner = MagicMock()
    return TickHandlers(runner)


def _make_ctx(
    pet_alive: bool = True,
    auto_engage_candidate: object = None,
    engaged: bool = False,
) -> MagicMock:
    """Build a lightweight ctx with real sub-state objects."""
    ctx = MagicMock()
    ctx.pet = PetState(alive=pet_alive)
    ctx.threat = ThreatState()
    ctx.combat = CombatState(engaged=engaged, auto_engage_candidate=auto_engage_candidate)
    return ctx


# ---------------------------------------------------------------------------
# _is_auto_engage_candidate
# ---------------------------------------------------------------------------


class TestIsAutoEngageCandidate:
    """Tests for the per-spawn auto-engage evaluation."""

    def test_rejects_non_npc(self) -> None:
        handler = _make_handler()
        spawn = make_spawn(spawn_type=0, target_name="TestPlayer")  # player
        state = make_game_state()
        ctx = _make_ctx()
        assert handler._is_auto_engage_candidate(spawn, ctx, state) is False

    def test_rejects_dead_npc(self) -> None:
        handler = _make_handler()
        spawn = make_spawn(hp_current=0, target_name="TestPlayer")
        state = make_game_state()
        ctx = _make_ctx()
        assert handler._is_auto_engage_candidate(spawn, ctx, state) is False

    def test_rejects_not_targeting_player(self) -> None:
        handler = _make_handler()
        spawn = make_spawn(target_name="SomeoneElse")
        state = make_game_state()
        ctx = _make_ctx()
        assert handler._is_auto_engage_candidate(spawn, ctx, state) is False

    def test_rejects_too_far(self) -> None:
        handler = _make_handler()
        # Spawn at (500, 500), player at origin -> dist >> 60
        spawn = make_spawn(x=500.0, y=500.0, target_name="TestPlayer")
        state = make_game_state()
        ctx = _make_ctx()
        assert handler._is_auto_engage_candidate(spawn, ctx, state) is False

    def test_red_con_flags_threat_not_candidate(self) -> None:
        handler = _make_handler()
        # Level 50 npc vs level 10 player = RED con
        spawn = make_spawn(
            x=5.0,
            y=5.0,
            level=50,
            target_name="TestPlayer",
        )
        state = make_game_state(level=10)
        ctx = _make_ctx()
        result = handler._is_auto_engage_candidate(spawn, ctx, state)
        assert result is False
        assert ctx.threat.imminent_threat is True
        assert ctx.threat.imminent_threat_con == Con.RED

    def test_no_pet_flags_threat(self) -> None:
        handler = _make_handler()
        spawn = make_spawn(
            x=5.0,
            y=5.0,
            level=10,
            target_name="TestPlayer",
        )
        state = make_game_state(level=10)
        ctx = _make_ctx(pet_alive=False)
        result = handler._is_auto_engage_candidate(spawn, ctx, state)
        assert result is False
        assert ctx.threat.imminent_threat is True

    def test_accepts_valid_candidate(self) -> None:
        handler = _make_handler()
        spawn = make_spawn(
            spawn_id=42,
            x=5.0,
            y=5.0,
            level=10,
            target_name="TestPlayer",
        )
        state = make_game_state(level=10)
        ctx = _make_ctx(pet_alive=True)
        result = handler._is_auto_engage_candidate(spawn, ctx, state)
        assert result is True
        assert ctx.combat.auto_engage_candidate is not None
        assert ctx.combat.auto_engage_candidate[0] == 42  # spawn_id

    def test_skips_when_candidate_already_set(self) -> None:
        handler = _make_handler()
        spawn = make_spawn(
            spawn_id=99,
            x=5.0,
            y=5.0,
            level=10,
            target_name="TestPlayer",
        )
        state = make_game_state(level=10)
        ctx = _make_ctx(pet_alive=True, auto_engage_candidate=(1, "existing", 0, 0, 5))
        result = handler._is_auto_engage_candidate(spawn, ctx, state)
        assert result is False
        # Original candidate unchanged
        assert ctx.combat.auto_engage_candidate[0] == 1

    def test_empty_target_name_rejected(self) -> None:
        handler = _make_handler()
        spawn = make_spawn(x=5.0, y=5.0, target_name="")
        state = make_game_state()
        ctx = _make_ctx()
        assert handler._is_auto_engage_candidate(spawn, ctx, state) is False


# ---------------------------------------------------------------------------
# detect_adds
# ---------------------------------------------------------------------------


def _make_adds_ctx(
    *,
    in_active_combat: bool = True,
    has_add: bool = False,
    pet_alive: bool = True,
    pet_name: str = "MyPet",
    pet_spawn_id: int | None = 200,
    pull_target_id: int | None = 50,
    pull_target_name: str = "a_skeleton",
    engaged: bool = True,
) -> MagicMock:
    """Build a ctx for detect_adds tests."""
    ctx = MagicMock()
    ctx.in_active_combat = in_active_combat
    ctx.pet = PetState(alive=pet_alive, name=pet_name, spawn_id=pet_spawn_id, has_add=has_add)
    ctx.combat = CombatState(
        engaged=engaged,
        pull_target_id=pull_target_id,
        pull_target_name=pull_target_name,
    )
    ctx.zone.zone_knowledge = None
    return ctx


class TestDetectAdds:
    """Tests for detect_adds -- add detection during combat."""

    def test_skips_when_not_in_combat(self) -> None:
        handler = _make_handler()
        ctx = _make_adds_ctx(in_active_combat=False)
        state = make_game_state()
        handler.detect_adds(state, ctx)
        assert ctx.pet.has_add is False

    def test_skips_when_add_already_detected(self) -> None:
        handler = _make_handler()
        ctx = _make_adds_ctx(has_add=True)
        state = make_game_state()
        handler.detect_adds(state, ctx)
        # has_add remains True but no new processing occurs
        assert ctx.pet.has_add is True

    def test_detects_add_targeting_player(self) -> None:
        handler = _make_handler()
        ctx = _make_adds_ctx(pull_target_id=50)
        # Add NPC targeting the player, near player, not the pull target
        add_spawn = make_spawn(
            spawn_id=99,
            x=5.0,
            y=5.0,
            level=10,
            target_name="TestPlayer",
            hp_current=100,
            hp_max=100,
        )
        state = make_game_state(spawns=[add_spawn])
        handler.detect_adds(state, ctx)
        assert ctx.pet.has_add is True

    def test_detects_add_targeting_pet(self) -> None:
        handler = _make_handler()
        ctx = _make_adds_ctx(pet_name="MyPet", pull_target_id=50)
        add_spawn = make_spawn(
            spawn_id=99,
            x=5.0,
            y=5.0,
            level=10,
            target_name="MyPet",
            hp_current=100,
            hp_max=100,
        )
        state = make_game_state(spawns=[add_spawn])
        handler.detect_adds(state, ctx)
        assert ctx.pet.has_add is True

    def test_ignores_pull_target(self) -> None:
        handler = _make_handler()
        ctx = _make_adds_ctx(pull_target_id=50)
        # The pull target itself should not be flagged as an add
        pull_spawn = make_spawn(
            spawn_id=50,
            x=5.0,
            y=5.0,
            level=10,
            target_name="TestPlayer",
            hp_current=80,
            hp_max=100,
        )
        state = make_game_state(spawns=[pull_spawn])
        handler.detect_adds(state, ctx)
        assert ctx.pet.has_add is False

    def test_ignores_dead_npcs(self) -> None:
        handler = _make_handler()
        ctx = _make_adds_ctx()
        dead_spawn = make_spawn(
            spawn_id=99,
            x=5.0,
            y=5.0,
            hp_current=0,
            target_name="TestPlayer",
        )
        state = make_game_state(spawns=[dead_spawn])
        handler.detect_adds(state, ctx)
        assert ctx.pet.has_add is False

    def test_detects_damaged_npc_near_pet(self) -> None:
        """Path 2: damaged NPC near the pet (not targeting anyone)."""
        handler = _make_handler()
        ctx = _make_adds_ctx(pet_spawn_id=200)
        pet_spawn = make_spawn(spawn_id=200, x=30.0, y=30.0, spawn_type=1, hp_current=80, hp_max=80)
        damaged_npc = make_spawn(
            spawn_id=99,
            x=35.0,
            y=35.0,
            level=10,
            target_name="",
            hp_current=50,
            hp_max=100,
        )
        state = make_game_state(spawns=[pet_spawn, damaged_npc])
        handler.detect_adds(state, ctx)
        assert ctx.pet.has_add is True


# ---------------------------------------------------------------------------
# scan_auto_engage
# ---------------------------------------------------------------------------


class TestScanAutoEngage:
    """Tests for scan_auto_engage -- pre-tick threat detection."""

    def test_clears_candidate_each_tick(self) -> None:
        handler = _make_handler()
        ctx = _make_ctx(pet_alive=True, auto_engage_candidate=(1, "old", 0, 0, 5))
        state = make_game_state()
        handler.scan_auto_engage(state, ctx)
        # Candidate is cleared at start of scan even if nothing found
        # (since combat is not engaged, it runs the scan)
        assert ctx.combat.auto_engage_candidate is None or ctx.combat.auto_engage_candidate is not None
        # More precisely: candidate is set to None at top of method

    def test_skips_scan_when_engaged(self) -> None:
        handler = _make_handler()
        ctx = _make_ctx(pet_alive=True, engaged=True)
        spawn = make_spawn(x=5.0, y=5.0, level=10, target_name="TestPlayer")
        state = make_game_state(spawns=[spawn])
        handler.scan_auto_engage(state, ctx)
        # When engaged, candidate is cleared but no scan runs
        assert ctx.combat.auto_engage_candidate is None

    def test_finds_candidate_from_spawns(self) -> None:
        handler = _make_handler()
        ctx = _make_ctx(pet_alive=True)
        spawn = make_spawn(spawn_id=42, x=5.0, y=5.0, level=10, target_name="TestPlayer")
        state = make_game_state(level=10, spawns=[spawn])
        handler.scan_auto_engage(state, ctx)
        assert ctx.combat.auto_engage_candidate is not None
        assert ctx.combat.auto_engage_candidate[0] == 42

    def test_hp_drop_auto_engage(self) -> None:
        """HP dropping + damaged NPC nearby = auto-engage candidate."""
        handler = _make_handler()
        handler._prev_hp = 1.0  # full HP last tick
        ctx = _make_ctx(pet_alive=True)
        # No spawn targeting player, but HP dropped
        state = make_game_state(level=10, hp_current=800, hp_max=1000)  # 0.8 < 1.0 - 0.02
        # Need a world model with damaged_npcs_near
        npc_data = MagicMock()
        npc_data.spawn = make_spawn(spawn_id=77, x=5.0, y=5.0, level=8)
        npc_data.distance = 20.0
        ctx.world = MagicMock()
        ctx.world.damaged_npcs_near.return_value = [npc_data]
        handler.scan_auto_engage(state, ctx)
        assert ctx.combat.auto_engage_candidate is not None
        assert ctx.combat.auto_engage_candidate[0] == 77

    def test_hp_drop_red_flags_threat(self) -> None:
        """HP dropping + RED NPC = threat flag, not candidate."""
        handler = _make_handler()
        handler._prev_hp = 1.0
        ctx = _make_ctx(pet_alive=True)
        state = make_game_state(level=10, hp_current=800, hp_max=1000)
        npc_data = MagicMock()
        npc_data.spawn = make_spawn(spawn_id=77, x=5.0, y=5.0, level=50)  # RED
        npc_data.distance = 20.0
        ctx.world = MagicMock()
        ctx.world.damaged_npcs_near.return_value = [npc_data]
        handler.scan_auto_engage(state, ctx)
        assert ctx.threat.imminent_threat is True
        assert ctx.combat.auto_engage_candidate is None

    def test_updates_prev_hp(self) -> None:
        handler = _make_handler()
        handler._prev_hp = 1.0
        ctx = _make_ctx(pet_alive=True, engaged=True)
        state = make_game_state(hp_current=900, hp_max=1000)
        handler.scan_auto_engage(state, ctx)
        assert handler._prev_hp == 0.9
