"""Tests for AgentContext: mutable state shared across brain rules and routines."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from brain.context import AgentContext
from brain.session_summary import (
    format_cycle_stats as _format_cycle_stats,
)
from brain.session_summary import (
    format_pull_stats as _format_pull_stats,
)
from brain.session_summary import (
    format_routine_stats as _format_routine_stats,
)
from brain.state.camp import CampConfig
from brain.state.combat import CombatState
from brain.state.diagnostic import DiagnosticState
from brain.state.inventory import InventoryState
from brain.state.kill_tracker import DefeatInfo, DefeatTracker
from brain.state.loot_config import LootConfig
from brain.state.metrics import SessionMetrics
from brain.state.pet import PetState
from brain.state.plan import PlanState
from brain.state.player import PlayerState
from brain.state.threat import ThreatState
from brain.state.zone import ZoneState
from core.types import Point
from tests.factories import make_game_state, make_spawn

# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self) -> None:
        """AgentContext can be constructed with no arguments."""
        ctx = AgentContext()
        assert ctx.session is None
        assert ctx.reader is None
        assert ctx.world is None
        assert ctx.stop_event is None

    def test_rest_thresholds_defaults(self) -> None:
        ctx = AgentContext()
        assert ctx.rest_hp_entry == 0.85
        assert ctx.rest_mana_entry == 0.40
        assert ctx.rest_hp_threshold == 0.92
        assert ctx.rest_mana_threshold == 0.60

    def test_tunnel_routes_default_empty(self) -> None:
        ctx = AgentContext()
        assert ctx.tunnel_routes == []

    def test_infrastructure_refs_default_none(self) -> None:
        ctx = AgentContext()
        assert ctx.spatial_memory is None
        assert ctx.fight_history is None
        assert ctx.danger_memory is None
        assert ctx.waypoint_graph is None


# ---------------------------------------------------------------------------
# Sub-state initialization
# ---------------------------------------------------------------------------


class TestSubStateInit:
    def test_combat_is_combat_state(self) -> None:
        ctx = AgentContext()
        assert isinstance(ctx.combat, CombatState)

    def test_pet_is_pet_state(self) -> None:
        ctx = AgentContext()
        assert isinstance(ctx.pet, PetState)

    def test_camp_is_camp_config(self) -> None:
        ctx = AgentContext()
        assert isinstance(ctx.camp, CampConfig)

    def test_inventory_is_inventory_state(self) -> None:
        ctx = AgentContext()
        assert isinstance(ctx.inventory, InventoryState)

    def test_plan_is_plan_state(self) -> None:
        ctx = AgentContext()
        assert isinstance(ctx.plan, PlanState)

    def test_player_is_player_state(self) -> None:
        ctx = AgentContext()
        assert isinstance(ctx.player, PlayerState)

    def test_defeat_tracker_is_defeat_tracker(self) -> None:
        ctx = AgentContext()
        assert isinstance(ctx.defeat_tracker, DefeatTracker)

    def test_metrics_is_session_metrics(self) -> None:
        ctx = AgentContext()
        assert isinstance(ctx.metrics, SessionMetrics)

    def test_threat_is_threat_state(self) -> None:
        ctx = AgentContext()
        assert isinstance(ctx.threat, ThreatState)

    def test_loot_is_loot_config(self) -> None:
        ctx = AgentContext()
        assert isinstance(ctx.loot, LootConfig)

    def test_zone_is_zone_state(self) -> None:
        ctx = AgentContext()
        assert isinstance(ctx.zone, ZoneState)

    def test_diag_is_diagnostic_state(self) -> None:
        ctx = AgentContext()
        assert isinstance(ctx.diag, DiagnosticState)

    def test_each_instance_has_own_substates(self) -> None:
        """Two AgentContext instances have distinct sub-state objects."""
        a = AgentContext()
        b = AgentContext()
        assert a.combat is not b.combat
        assert a.pet is not b.pet
        assert a.metrics is not b.metrics


# ---------------------------------------------------------------------------
# pet_just_died()
# ---------------------------------------------------------------------------


class TestPetJustDied:
    def test_initially_false(self) -> None:
        ctx = AgentContext()
        assert ctx.pet.just_died() is False

    def test_returns_true_when_pet_was_alive_then_not(self) -> None:
        ctx = AgentContext()
        ctx.pet.prev_alive = True
        ctx.pet.alive = False
        assert ctx.pet.just_died() is True

    def test_returns_false_when_pet_still_alive(self) -> None:
        ctx = AgentContext()
        ctx.pet.prev_alive = True
        ctx.pet.alive = True
        assert ctx.pet.just_died() is False

    def test_returns_false_when_both_dead(self) -> None:
        ctx = AgentContext()
        ctx.pet.prev_alive = False
        ctx.pet.alive = False
        assert ctx.pet.just_died() is False


# ---------------------------------------------------------------------------
# in_active_combat property
# ---------------------------------------------------------------------------


class TestInActiveCombat:
    def test_false_by_default(self) -> None:
        ctx = AgentContext()
        assert ctx.in_active_combat is False

    def test_true_when_engaged(self) -> None:
        ctx = AgentContext()
        ctx.combat.engaged = True
        assert ctx.in_active_combat is True

    def test_true_when_pull_target_set(self) -> None:
        ctx = AgentContext()
        ctx.combat.pull_target_id = 42
        assert ctx.in_active_combat is True

    def test_true_when_both_engaged_and_pull(self) -> None:
        ctx = AgentContext()
        ctx.combat.engaged = True
        ctx.combat.pull_target_id = 42
        assert ctx.in_active_combat is True

    def test_false_after_clearing(self) -> None:
        ctx = AgentContext()
        ctx.combat.engaged = True
        ctx.combat.pull_target_id = 42
        ctx.combat.engaged = False
        ctx.combat.pull_target_id = None
        assert ctx.in_active_combat is False


# ---------------------------------------------------------------------------
# begin_engagement / clear_engagement
# ---------------------------------------------------------------------------


class TestEngagementTransitions:
    """Test begin_engagement() and clear_engagement() with thread guard disabled."""

    @pytest.fixture(autouse=True)
    def _bypass_thread_guard(self):
        """Bypass brain thread assertion for test thread."""
        with patch("brain.context.assert_brain_thread"):
            yield

    def test_begin_engagement_sets_combat_state(self) -> None:
        ctx = AgentContext()
        ctx.begin_engagement(target_id=100, name="a_skeleton", x=10.0, y=20.0, level=5)
        assert ctx.combat.engaged is True
        assert ctx.combat.pull_target_id == 100
        assert ctx.defeat_tracker.last_fight_name == "a_skeleton"
        assert ctx.defeat_tracker.last_fight_id == 100
        assert ctx.defeat_tracker.last_fight_x == 10.0
        assert ctx.defeat_tracker.last_fight_y == 20.0
        assert ctx.defeat_tracker.last_fight_level == 5
        assert ctx.player.engagement_start > 0

    def test_clear_engagement_resets_combat_state(self) -> None:
        ctx = AgentContext()
        ctx.begin_engagement(target_id=100, name="a_skeleton", x=10.0, y=20.0, level=5)
        ctx.clear_engagement()
        assert ctx.combat.engaged is False
        assert ctx.combat.pull_target_id is None
        assert ctx.defeat_tracker.last_fight_name == ""
        assert ctx.defeat_tracker.last_fight_id == 0
        assert ctx.defeat_tracker.last_fight_x == 0.0
        assert ctx.defeat_tracker.last_fight_y == 0.0
        assert ctx.player.engagement_start == 0.0

    def test_in_active_combat_after_begin(self) -> None:
        ctx = AgentContext()
        ctx.begin_engagement(target_id=99)
        assert ctx.in_active_combat is True

    def test_not_in_active_combat_after_clear(self) -> None:
        ctx = AgentContext()
        ctx.begin_engagement(target_id=99)
        ctx.clear_engagement()
        assert ctx.in_active_combat is False

    def test_begin_engagement_defaults(self) -> None:
        """begin_engagement with only target_id uses sensible defaults."""
        ctx = AgentContext()
        ctx.begin_engagement(target_id=1)
        assert ctx.defeat_tracker.last_fight_name == ""
        assert ctx.defeat_tracker.last_fight_x == 0.0
        assert ctx.defeat_tracker.last_fight_level == 0


# ---------------------------------------------------------------------------
# session_summary()
# ---------------------------------------------------------------------------


class TestSessionSummary:
    def test_returns_nonempty_string(self) -> None:
        ctx = AgentContext()
        summary = ctx.session_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_contains_session_header(self) -> None:
        ctx = AgentContext()
        summary = ctx.session_summary()
        assert "SESSION SUMMARY" in summary

    def test_contains_duration_info(self) -> None:
        ctx = AgentContext()
        summary = ctx.session_summary()
        assert "Duration:" in summary

    def test_summary_includes_defeat_count(self) -> None:
        ctx = AgentContext()
        ctx.defeat_tracker.defeats = 7
        summary = ctx.session_summary()
        assert "7" in summary


# ---------------------------------------------------------------------------
# _format_routine_stats
# ---------------------------------------------------------------------------


class TestFormatRoutineStats:
    def test_empty_collections(self) -> None:
        mc = {"routine_counts": {}, "routine_failures": {}, "acquire_modes": {}}
        assert _format_routine_stats(mc) == []

    def test_routine_counts(self) -> None:
        mc = {
            "routine_counts": {"pull": 5, "rest": 3},
            "routine_failures": {},
            "acquire_modes": {},
        }
        lines = _format_routine_stats(mc)
        assert len(lines) == 1
        assert "Routine activations:" in lines[0]
        assert "pull=5" in lines[0]
        assert "rest=3" in lines[0]

    def test_routine_failures(self) -> None:
        mc = {
            "routine_counts": {},
            "routine_failures": {"pull": 2},
            "acquire_modes": {},
        }
        lines = _format_routine_stats(mc)
        assert any("Routine failures:" in l for l in lines)

    def test_acquire_modes(self) -> None:
        mc = {
            "routine_counts": {},
            "routine_failures": {},
            "acquire_modes": {"tab": 10},
        }
        lines = _format_routine_stats(mc)
        assert any("Acquire modes:" in l for l in lines)


# ---------------------------------------------------------------------------
# _format_pull_stats
# ---------------------------------------------------------------------------


class TestFormatPullStats:
    def test_empty_stats(self) -> None:
        mc = {"pull_distances": [], "pull_engage_times": []}
        metrics = SessionMetrics()
        dt = DefeatTracker()
        assert _format_pull_stats(mc, metrics, dt) == []

    def test_mana_efficiency(self) -> None:
        mc = {"pull_distances": [], "pull_engage_times": []}
        metrics = SessionMetrics()
        metrics.total_casts = 30
        dt = DefeatTracker()
        dt.defeats = 10
        lines = _format_pull_stats(mc, metrics, dt)
        assert any("Mana efficiency:" in l for l in lines)
        assert any("3.0 casts/npc" in l for l in lines)

    def test_pull_distances(self) -> None:
        mc = {"pull_distances": [100.0, 200.0, 150.0], "pull_engage_times": []}
        metrics = SessionMetrics()
        dt = DefeatTracker()
        lines = _format_pull_stats(mc, metrics, dt)
        assert any("Pull distance:" in l for l in lines)
        assert any("avg=150" in l for l in lines)

    def test_pull_engage_times(self) -> None:
        mc = {"pull_distances": [], "pull_engage_times": [2.0, 4.0]}
        metrics = SessionMetrics()
        dt = DefeatTracker()
        lines = _format_pull_stats(mc, metrics, dt)
        assert any("Pet engage time:" in l for l in lines)

    def test_dc_fizzles(self) -> None:
        mc = {"pull_distances": [], "pull_engage_times": []}
        metrics = SessionMetrics()
        metrics.pull_dc_fizzles = 3
        dt = DefeatTracker()
        lines = _format_pull_stats(mc, metrics, dt)
        assert any("DC fizzles: 3" in l for l in lines)

    def test_pet_only_pulls(self) -> None:
        mc = {"pull_distances": [100.0], "pull_engage_times": []}
        metrics = SessionMetrics()
        metrics.pull_pet_only_count = 1
        dt = DefeatTracker()
        lines = _format_pull_stats(mc, metrics, dt)
        assert any("Pet-only pulls:" in l for l in lines)


# ---------------------------------------------------------------------------
# _format_cycle_stats
# ---------------------------------------------------------------------------


class TestFormatCycleStats:
    def _base_mc(self) -> dict:
        return {
            "total_cycle_times": [],
            "pull_distances": [],
            "pull_engage_times": [],
            "acquire_tab_totals": [],
            "routine_time": {},
        }

    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_empty_stats(self, _mock: object) -> None:
        mc = self._base_mc()
        metrics = SessionMetrics()
        dt = DefeatTracker()
        inv = InventoryState()
        lines = _format_cycle_stats(mc, metrics, dt, inv)
        assert lines == []

    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_cycle_times(self, _mock: object) -> None:
        mc = self._base_mc()
        mc["total_cycle_times"] = [60.0, 80.0, 100.0]
        metrics = SessionMetrics()
        dt = DefeatTracker()
        inv = InventoryState()
        lines = _format_cycle_stats(mc, metrics, dt, inv)
        assert any("Grind cycle:" in l for l in lines)
        assert any("avg=80.0s" in l for l in lines)

    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_acquire_tab_totals(self, _mock: object) -> None:
        mc = self._base_mc()
        mc["acquire_tab_totals"] = [3, 5]
        metrics = SessionMetrics()
        dt = DefeatTracker()
        inv = InventoryState()
        lines = _format_cycle_stats(mc, metrics, dt, inv)
        assert any("Acquire tabs/success:" in l for l in lines)

    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_wander_stats(self, _mock: object) -> None:
        mc = self._base_mc()
        metrics = SessionMetrics()
        metrics.wander_count = 5
        metrics.wander_total_distance = 500.0
        dt = DefeatTracker()
        inv = InventoryState()
        lines = _format_cycle_stats(mc, metrics, dt, inv)
        assert any("Wander:" in l for l in lines)

    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_combat_time_breakdown(self, _mock: object) -> None:
        mc = self._base_mc()
        metrics = SessionMetrics()
        metrics.total_combat_time = 100.0
        metrics.total_cast_time = 40.0
        dt = DefeatTracker()
        inv = InventoryState()
        lines = _format_cycle_stats(mc, metrics, dt, inv)
        assert any("Combat time:" in l for l in lines)
        assert any("casting 40%" in l for l in lines)

    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_xp_npcs(self, _mock: object) -> None:
        mc = self._base_mc()
        metrics = SessionMetrics()
        dt = DefeatTracker()
        dt.xp_gains = 8
        dt.defeats = 10
        inv = InventoryState()
        lines = _format_cycle_stats(mc, metrics, dt, inv)
        assert any("XP npcs:" in l for l in lines)

    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_loot_count(self, _mock: object) -> None:
        mc = self._base_mc()
        metrics = SessionMetrics()
        dt = DefeatTracker()
        inv = InventoryState()
        inv.loot_count = 12
        lines = _format_cycle_stats(mc, metrics, dt, inv)
        assert any("Corpses looted: 12" in l for l in lines)

    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_routine_time(self, _mock: object) -> None:
        mc = self._base_mc()
        mc["routine_time"] = {"pull": 60.0, "rest": 40.0}
        metrics = SessionMetrics()
        dt = DefeatTracker()
        inv = InventoryState()
        lines = _format_cycle_stats(mc, metrics, dt, inv)
        assert any("Time by routine:" in l for l in lines)

    @patch("nav.movement.get_stuck_event_count", return_value=0)
    def test_defeat_cycle_times(self, _mock: object) -> None:
        mc = self._base_mc()
        metrics = SessionMetrics()
        dt = DefeatTracker()
        dt.defeat_cycle_times = [30.0, 60.0, 90.0]
        inv = InventoryState()
        lines = _format_cycle_stats(mc, metrics, dt, inv)
        assert any("Time between npcs:" in l for l in lines)

    @patch("nav.movement.get_stuck_event_count", return_value=5)
    def test_stuck_events(self, _mock: object) -> None:
        mc = self._base_mc()
        metrics = SessionMetrics()
        dt = DefeatTracker()
        inv = InventoryState()
        lines = _format_cycle_stats(mc, metrics, dt, inv)
        assert any("Stuck events: 5" in l for l in lines)


# ---------------------------------------------------------------------------
# record_kill
# ---------------------------------------------------------------------------


class TestRecordKill:
    @pytest.fixture(autouse=True)
    def _bypass_thread_guard(self):
        with patch("brain.context.assert_brain_thread"):
            yield

    def test_record_kill_increments_defeats(self) -> None:
        ctx = AgentContext()
        ctx.record_kill(spawn_id=100, name="a_skeleton", pos=Point(10.0, 20.0, 0.0))
        assert ctx.defeat_tracker.defeats == 1

    def test_record_kill_adds_to_recent(self) -> None:
        ctx = AgentContext()
        ctx.record_kill(spawn_id=100, name="a_skeleton")
        assert len(ctx.defeat_tracker.recent_kills) == 1

    def test_record_kill_adds_to_history(self) -> None:
        ctx = AgentContext()
        ctx.record_kill(spawn_id=100, name="a_skeleton", pos=Point(10.0, 20.0, 0.0))
        assert len(ctx.defeat_tracker.defeat_history) == 1
        assert ctx.defeat_tracker.defeat_history[0].name == "a_skeleton"


# ---------------------------------------------------------------------------
# clear_recent_kills
# ---------------------------------------------------------------------------


class TestClearRecentKills:
    @pytest.fixture(autouse=True)
    def _bypass_thread_guard(self):
        with patch("brain.context.assert_brain_thread"):
            yield

    def test_removes_old_kills(self) -> None:
        ctx = AgentContext()
        old_time = time.time() - 120  # 2 minutes ago
        ctx.defeat_tracker.recent_kills = [(100, old_time)]
        ctx.defeat_tracker.clear_recent_kills()
        assert len(ctx.defeat_tracker.recent_kills) == 0

    def test_keeps_recent_kills(self) -> None:
        ctx = AgentContext()
        recent_time = time.time() - 30  # 30 seconds ago
        ctx.defeat_tracker.recent_kills = [(100, recent_time)]
        ctx.defeat_tracker.clear_recent_kills()
        assert len(ctx.defeat_tracker.recent_kills) == 1


# ---------------------------------------------------------------------------
# update_stationary_kills
# ---------------------------------------------------------------------------


class TestUpdateStationaryKills:
    @pytest.fixture(autouse=True)
    def _bypass_thread_guard(self):
        with patch("brain.context.assert_brain_thread"):
            yield

    def test_first_kill_sets_baseline(self) -> None:
        ctx = AgentContext()
        ctx.metrics.update_stationary_kills(100.0, 200.0)
        assert ctx.metrics.stationary_kills == 1
        assert ctx.metrics.last_kill_x == 100.0
        assert ctx.metrics.last_kill_y == 200.0

    def test_nearby_kill_increments(self) -> None:
        ctx = AgentContext()
        ctx.metrics.update_stationary_kills(100.0, 200.0)
        ctx.metrics.update_stationary_kills(110.0, 205.0)  # within 30 units
        assert ctx.metrics.stationary_kills == 2

    def test_far_kill_resets(self) -> None:
        ctx = AgentContext()
        ctx.metrics.update_stationary_kills(100.0, 200.0)
        ctx.metrics.update_stationary_kills(500.0, 600.0)  # far away
        assert ctx.metrics.stationary_kills == 1


# ---------------------------------------------------------------------------
# should_reposition
# ---------------------------------------------------------------------------


class TestShouldReposition:
    def test_false_when_few_stationary_kills(self) -> None:
        ctx = AgentContext()
        ctx.metrics.stationary_kills = 1
        assert ctx.metrics.should_reposition() is False

    def test_eventually_true_with_many_kills(self) -> None:
        """With high stationary kills (chance capped at 0.90), should eventually return True."""
        ctx = AgentContext()
        ctx.metrics.stationary_kills = 10
        # With 90% chance, in 50 tries at least one should be True
        _ = [ctx.metrics.should_reposition() for _ in range(50)]
        # Reset stationary_kills each time it returns True (method does that)
        # Just check that at least one call succeeded
        ctx.metrics.stationary_kills = 10
        found = False
        for _ in range(50):
            ctx.metrics.stationary_kills = 10
            if ctx.metrics.should_reposition():
                found = True
                break
        assert found

    def test_resets_on_reposition(self) -> None:
        """When should_reposition returns True, stationary_kills resets to 0."""
        ctx = AgentContext()
        for _ in range(100):
            ctx.metrics.stationary_kills = 10
            if ctx.metrics.should_reposition():
                assert ctx.metrics.stationary_kills == 0
                return
        pytest.skip("Probabilistic: did not trigger in 100 tries")


# ---------------------------------------------------------------------------
# distance_to_camp
# ---------------------------------------------------------------------------


class TestDistanceToCamp:
    def test_at_camp(self) -> None:
        ctx = AgentContext()
        ctx.camp.camp_x = 100.0
        ctx.camp.camp_y = 200.0
        state = make_game_state(x=100.0, y=200.0)
        assert ctx.camp.distance_to_camp(state) == pytest.approx(0.0)

    def test_away_from_camp(self) -> None:
        ctx = AgentContext()
        ctx.camp.camp_x = 0.0
        ctx.camp.camp_y = 0.0
        state = make_game_state(x=300.0, y=400.0)
        assert ctx.camp.distance_to_camp(state) > 0


# ---------------------------------------------------------------------------
# weight_gained / is_encumbered
# ---------------------------------------------------------------------------


class TestInventoryDelegation:
    def test_weight_gained_zero_baseline(self) -> None:
        ctx = AgentContext()
        state = make_game_state(weight=100)
        assert ctx.inventory.weight_gained(state) == 0

    def test_weight_gained_with_baseline(self) -> None:
        ctx = AgentContext()
        ctx.inventory.weight_baseline = 50
        state = make_game_state(weight=100)
        assert ctx.inventory.weight_gained(state) == 50

    def test_is_encumbered_false(self) -> None:
        ctx = AgentContext()
        ctx.inventory.weight_baseline = 50
        state = make_game_state(weight=60)
        assert ctx.inventory.is_encumbered(state) is False

    def test_is_encumbered_true(self) -> None:
        ctx = AgentContext()
        ctx.inventory.weight_baseline = 50
        ctx.inventory.weight_threshold = 80
        state = make_game_state(weight=200)
        assert ctx.inventory.is_encumbered(state) is True


# ---------------------------------------------------------------------------
# has_unlootable_corpse / find_unlootable_kill / clean_kill_history
# ---------------------------------------------------------------------------


class TestUnlootableCorpse:
    @pytest.fixture(autouse=True)
    def _bypass_thread_guard(self):
        with patch("brain.context.assert_brain_thread"):
            yield

    def test_no_corpses_returns_false(self) -> None:
        ctx = AgentContext()
        state = make_game_state(spawns=())
        assert ctx.has_unlootable_corpse(state) is False

    def test_corpse_matching_unlooted_defeat(self) -> None:
        ctx = AgentContext()
        now = time.time()
        ctx.defeat_tracker.defeat_history.append(
            DefeatInfo(spawn_id=100, name="a_skeleton", x=50.0, y=50.0, time=now)
        )
        corpse = make_spawn(
            spawn_id=100,
            name="a_skeleton's_corpse",
            x=50.0,
            y=50.0,
            spawn_type=2,  # NPC corpse
        )
        state = make_game_state(x=50.0, y=50.0, spawns=(corpse,))
        assert ctx.has_unlootable_corpse(state) is True

    def test_corpse_too_far_returns_false(self) -> None:
        ctx = AgentContext()
        now = time.time()
        ctx.defeat_tracker.defeat_history.append(
            DefeatInfo(spawn_id=100, name="a_skeleton", x=50.0, y=50.0, time=now)
        )
        corpse = make_spawn(
            spawn_id=100,
            name="a_skeleton's_corpse",
            x=5000.0,
            y=5000.0,
            spawn_type=2,
        )
        state = make_game_state(x=0.0, y=0.0, spawns=(corpse,))
        assert ctx.has_unlootable_corpse(state) is False

    def test_corpse_no_matching_defeat_logs_debug(self) -> None:
        """Non-matching unlooted defeat nearby triggers debug logging."""
        ctx = AgentContext()
        now = time.time()
        ctx.defeat_tracker.defeat_history.append(
            DefeatInfo(spawn_id=200, name="a_bat", x=50.0, y=50.0, time=now, looted=False)
        )
        corpse = make_spawn(
            spawn_id=300,
            name="a_skeleton's_corpse",
            x=55.0,
            y=55.0,
            spawn_type=2,
        )
        state = make_game_state(x=50.0, y=50.0, spawns=(corpse,))
        # Should not crash, returns False (name mismatch)
        assert ctx.has_unlootable_corpse(state) is False

    def test_clean_kill_history_delegation(self) -> None:
        ctx = AgentContext()
        now = time.time()
        ctx.defeat_tracker.defeat_history.append(
            DefeatInfo(spawn_id=100, name="a_skeleton", x=50.0, y=50.0, time=now - 600)
        )
        ctx.defeat_tracker.clean_kill_history()
        assert len(ctx.defeat_tracker.defeat_history) == 0

    def test_find_unlootable_kill_delegation(self) -> None:
        ctx = AgentContext()
        now = time.time()
        ctx.defeat_tracker.defeat_history.append(
            DefeatInfo(spawn_id=100, name="a_skeleton", x=50.0, y=50.0, time=now)
        )
        result = ctx.defeat_tracker.find_unlootable_kill(
            "a_skeleton's_corpse", Point(50.0, 50.0, 0.0), corpse_spawn_id=100
        )
        assert result is not None
        assert result.spawn_id == 100


# ---------------------------------------------------------------------------
# defeats_in_window / defeat_rate_window / last_kill_age
# ---------------------------------------------------------------------------


class TestDefeatStats:
    def test_defeats_in_window_empty(self) -> None:
        ctx = AgentContext()
        assert ctx.defeat_tracker.defeats_in_window(300) == 0

    def test_defeats_in_window_with_data(self) -> None:
        ctx = AgentContext()
        now = time.time()
        ctx.defeat_tracker.defeat_times = [now - 10, now - 60, now - 500]
        assert ctx.defeat_tracker.defeats_in_window(300) == 2

    def test_defeat_rate_window(self) -> None:
        ctx = AgentContext()
        now = time.time()
        ctx.defeat_tracker.defeat_times = [now - 10, now - 60]
        rate = ctx.defeat_tracker.defeat_rate_window(3600)
        # 2 defeats in 3600s = 2 per hour
        assert rate == pytest.approx(2.0, abs=0.1)

    def test_last_kill_age_no_kills(self) -> None:
        ctx = AgentContext()
        assert ctx.defeat_tracker.last_kill_age() == 9999.0

    def test_last_kill_age_with_kills(self) -> None:
        ctx = AgentContext()
        ctx.defeat_tracker.defeat_times = [time.time() - 5]
        age = ctx.defeat_tracker.last_kill_age()
        assert 4.0 < age < 7.0


# ---------------------------------------------------------------------------
# record_xp_sample / xp_per_hour / time_to_level
# ---------------------------------------------------------------------------


class TestXPTracking:
    def test_record_xp_ignores_zero(self) -> None:
        ctx = AgentContext()
        ctx.metrics.record_xp_sample(time.time(), 0)
        assert len(ctx.metrics.xp_history) == 0

    def test_record_xp_ignores_negative(self) -> None:
        ctx = AgentContext()
        ctx.metrics.record_xp_sample(time.time(), -10)
        assert len(ctx.metrics.xp_history) == 0

    def test_record_xp_appends(self) -> None:
        ctx = AgentContext()
        now = time.time()
        ctx.metrics.record_xp_sample(now, 100)
        assert len(ctx.metrics.xp_history) == 1
        assert ctx.metrics.xp_history[0] == (now, 100)

    def test_record_xp_trims_old(self) -> None:
        ctx = AgentContext()
        now = time.time()
        # Fill with 201 entries, all old
        for i in range(201):
            ctx.metrics.xp_history.append((now - 3600 + i, i))
        ctx.metrics.record_xp_sample(now, 250)
        # Should trim entries older than 1800s
        assert len(ctx.metrics.xp_history) <= 201

    def test_xp_per_hour_no_data(self) -> None:
        ctx = AgentContext()
        assert ctx.metrics.xp_per_hour() == 0.0

    def test_xp_per_hour_single_entry(self) -> None:
        ctx = AgentContext()
        ctx.metrics.xp_history = [(time.time(), 100)]
        assert ctx.metrics.xp_per_hour() == 0.0

    def test_xp_per_hour_positive(self) -> None:
        ctx = AgentContext()
        now = time.time()
        # 60 raw XP gained over 1 hour (from XP_SCALE_MAX=330)
        ctx.metrics.xp_history = [(now - 3600, 100), (now, 160)]
        rate = ctx.metrics.xp_per_hour(window_seconds=7200)
        # delta_pct = (160-100)/330*100 = ~18.18%/hr
        assert rate > 0

    def test_xp_per_hour_level_up_wrap(self) -> None:
        """When XP goes down (level up), use xp_gained_pct fallback."""
        ctx = AgentContext()
        now = time.time()
        ctx.metrics.xp_history = [(now - 3600, 300), (now, 50)]
        ctx.metrics.xp_gained_pct = 10.0
        rate = ctx.metrics.xp_per_hour(window_seconds=7200)
        assert rate > 0

    def test_xp_per_hour_too_close(self) -> None:
        """If first and last timestamps are <1s apart, return 0."""
        ctx = AgentContext()
        now = time.time()
        ctx.metrics.xp_history = [(now, 100), (now + 0.5, 200)]
        assert ctx.metrics.xp_per_hour() == 0.0

    def test_time_to_level_no_data(self) -> None:
        ctx = AgentContext()
        assert ctx.metrics.time_to_level() is None

    def test_time_to_level_no_rate(self) -> None:
        ctx = AgentContext()
        ctx.metrics.xp_history = [(time.time(), 100)]
        assert ctx.metrics.time_to_level() is None

    def test_time_to_level_at_max(self) -> None:
        """At max XP (330), remaining is 0 so TTL should be None."""
        ctx = AgentContext()
        now = time.time()
        ctx.metrics.xp_history = [(now - 3600, 100), (now, 330)]
        assert ctx.metrics.time_to_level() is None

    def test_time_to_level_positive(self) -> None:
        ctx = AgentContext()
        now = time.time()
        # Need enough data within 300s window
        ctx.metrics.xp_history = [(now - 300, 100), (now, 200)]
        ttl = ctx.metrics.time_to_level()
        assert ttl is not None
        assert ttl > 0


# ---------------------------------------------------------------------------
# nearest_player_dist / nearby_player_count
# ---------------------------------------------------------------------------


class TestPlayerProximity:
    def test_nearest_player_dist_no_players(self) -> None:
        ctx = AgentContext()
        state = make_game_state(x=0.0, y=0.0, spawns=())
        assert ctx.nearest_player_dist(state) == 9999.0

    def test_nearest_player_dist_with_player(self) -> None:
        ctx = AgentContext()
        player = make_spawn(spawn_id=50, name="OtherPlayer", spawn_type=0, x=30.0, y=40.0)
        state = make_game_state(x=0.0, y=0.0, name="TestPlayer", spawns=(player,))
        dist = ctx.nearest_player_dist(state)
        assert dist == pytest.approx(50.0, abs=1.0)

    def test_nearest_player_dist_ignores_self(self) -> None:
        ctx = AgentContext()
        self_spawn = make_spawn(spawn_id=1, name="TestPlayer", spawn_type=0, x=0.0, y=0.0)
        state = make_game_state(x=0.0, y=0.0, name="TestPlayer", spawns=(self_spawn,))
        assert ctx.nearest_player_dist(state) == 9999.0

    def test_nearby_player_count_empty(self) -> None:
        ctx = AgentContext()
        state = make_game_state(spawns=())
        assert ctx.nearby_player_count(state) == 0

    def test_nearby_player_count_within_radius(self) -> None:
        ctx = AgentContext()
        p1 = make_spawn(spawn_id=50, name="Player1", spawn_type=0, x=30.0, y=0.0)
        p2 = make_spawn(spawn_id=51, name="Player2", spawn_type=0, x=500.0, y=0.0)
        state = make_game_state(x=0.0, y=0.0, name="TestPlayer", spawns=(p1, p2))
        assert ctx.nearby_player_count(state, radius=200.0) == 1


# ---------------------------------------------------------------------------
# session_summary edge cases
# ---------------------------------------------------------------------------


class TestSessionSummaryEdgeCases:
    def test_summary_with_fight_history(self) -> None:
        ctx = AgentContext()
        ctx.fight_history = SimpleNamespace(summary=lambda: "Fight summary line")
        summary = ctx.session_summary()
        assert "Fight summary line" in summary

    def test_summary_with_fight_history_error(self) -> None:
        """fight_history.summary() raising doesn't crash session_summary."""
        ctx = AgentContext()

        def broken_summary() -> str:
            raise ValueError("boom")

        ctx.fight_history = SimpleNamespace(summary=broken_summary)
        summary = ctx.session_summary()
        assert "SESSION SUMMARY" in summary

    def test_summary_xp_stats_with_data(self) -> None:
        ctx = AgentContext()
        now = time.time()
        ctx.metrics.xp_history = [(now - 600, 50), (now, 100)]
        ctx.metrics.xp_gained_pct = 5.0
        summary = ctx.session_summary()
        assert "XP rate:" in summary

    def test_summary_perception_stats(self) -> None:
        """_format_xp_stats includes perception health when reader has stats."""
        ctx = AgentContext()
        stats_obj = SimpleNamespace(total=10, success=8, success_rate=0.8)
        ctx.reader = SimpleNamespace(_read_stats={"memory": stats_obj})
        # Need XP data to enter _format_xp_stats path
        now = time.time()
        ctx.metrics.xp_history = [(now - 600, 50), (now, 100)]
        ctx.metrics.xp_gained_pct = 5.0
        summary = ctx.session_summary()
        assert "Perception:" in summary
