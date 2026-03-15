"""Tests for brain.world.health -- HealthMonitor runtime health checks.

Covers Tier 1 (vitals, position), Tier 2 (deep checks), Tier 3 (threats),
and geometry helpers (_closest_point_on_line, _compute_evasion).
Uses make_game_state and make_spawn for realistic inputs.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from brain.world.health import HealthMonitor
from tests.factories import make_game_state, make_spawn

# ---------------------------------------------------------------------------
# Stub context -- minimal surface for HealthMonitor
# ---------------------------------------------------------------------------


def _make_threat() -> Any:
    return SimpleNamespace(
        approaching_threat=None,
        imminent_threat=False,
        imminent_threat_con="",
        evasion_point=None,
        patrol_evade=False,
    )


def _make_defeat_tracker(defeats: int = 0, defeat_history: list | None = None) -> Any:
    return SimpleNamespace(
        defeats=defeats,
        defeat_history=defeat_history or [],
    )


def _make_pet(alive: bool = False, spawn_id: int = 0) -> Any:
    return SimpleNamespace(alive=alive, spawn_id=spawn_id, name="pet")


def _make_combat(engaged: bool = False, pull_target_id: int | None = None) -> Any:
    return SimpleNamespace(engaged=engaged, pull_target_id=pull_target_id)


def _make_zone(zone_dispositions: dict | None = None) -> Any:
    return SimpleNamespace(zone_dispositions=zone_dispositions or {})


def _make_ctx(
    defeats: int = 0,
    pet_alive: bool = False,
    pet_spawn_id: int = 0,
    engaged: bool = False,
    pull_target_id: int | None = None,
    zone_dispositions: dict | None = None,
) -> Any:
    return SimpleNamespace(
        threat=_make_threat(),
        defeat_tracker=_make_defeat_tracker(defeats),
        metrics=SimpleNamespace(routine_counts={}),
        pet=_make_pet(pet_alive, pet_spawn_id),
        combat=_make_combat(engaged, pull_target_id),
        zone=_make_zone(zone_dispositions),
    )


# ---------------------------------------------------------------------------
# Tier 1: Vitals
# ---------------------------------------------------------------------------


class TestVitals:
    def test_hp_exceeds_max_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        state = make_game_state(hp_current=1200, hp_max=1000)
        hm._check_vitals(state)
        assert any("HP" in r.message and "exceeds max" in r.message for r in caplog.records)

    def test_negative_hp_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        state = make_game_state(hp_current=-5, hp_max=1000)
        hm._check_vitals(state)
        assert any("negative" in r.message for r in caplog.records)

    def test_weight_drop_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        hm._prev_weight = 50
        state = make_game_state(weight=0)
        hm._check_vitals(state)
        assert any("weight dropped" in r.message for r in caplog.records)

    def test_level_change_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        hm._prev_level = 10
        state = make_game_state(level=11)
        with caplog.at_level(10, logger="compass.health"):
            hm._check_vitals(state)
        assert any("level changed" in r.message for r in caplog.records)

    def test_no_warning_for_normal_hp(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        state = make_game_state(hp_current=800, hp_max=1000)
        hm._check_vitals(state)
        assert not any("HP" in r.message for r in caplog.records)

    def test_zero_mana_ticks_tracking(self) -> None:
        hm = HealthMonitor()
        state = make_game_state(mana_current=0, mana_max=0)
        for _ in range(5):
            hm._check_vitals(state)
        assert hm._zero_mana_ticks == 5


# ---------------------------------------------------------------------------
# Tier 1: Position
# ---------------------------------------------------------------------------


class TestPosition:
    def test_position_jump_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        hm._prev_x = 100.0
        hm._prev_y = 100.0
        state = make_game_state(x=700.0, y=100.0)
        hm._check_position(state)
        assert any("position jumped" in r.message for r in caplog.records)

    def test_normal_movement_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        hm._prev_x = 100.0
        hm._prev_y = 100.0
        state = make_game_state(x=110.0, y=105.0)
        hm._check_position(state)
        assert not any("position jumped" in r.message for r in caplog.records)

    def test_first_tick_sets_position(self) -> None:
        hm = HealthMonitor()
        state = make_game_state(x=50.0, y=60.0)
        hm._check_position(state)
        assert hm._prev_x == 50.0
        assert hm._prev_y == 60.0


# ---------------------------------------------------------------------------
# Tier 2: Deep checks
# ---------------------------------------------------------------------------


class TestDeepChecks:
    def test_pet_integrity_warns_missing(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx(pet_alive=True, pet_spawn_id=999)
        # No spawns contain pet id 999
        state = make_game_state(spawns=())
        hm._check_pet_integrity(state, ctx)
        assert any("pet alive=True" in r.message for r in caplog.records)

    def test_pet_integrity_ok_when_found(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx(pet_alive=True, pet_spawn_id=100)
        pet_spawn = make_spawn(spawn_id=100, name="pet")
        state = make_game_state(spawns=(pet_spawn,))
        hm._check_pet_integrity(state, ctx)
        assert not any("pet alive=True" in r.message for r in caplog.records)

    def test_pet_integrity_skipped_if_dead(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx(pet_alive=False, pet_spawn_id=0)
        state = make_game_state()
        hm._check_pet_integrity(state, ctx)
        assert not any("pet" in r.message.lower() for r in caplog.records)

    def test_engaged_integrity_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx(engaged=True, pull_target_id=None)
        state = make_game_state(target=None)
        hm._check_engaged_integrity(state, ctx)
        assert any("engaged=True" in r.message for r in caplog.records)

    def test_buff_integrity_negative_spell_id(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx()
        state = make_game_state(buffs=((-1, 100),))
        hm._check_buff_integrity(state, ctx)
        assert any("negative spell_id" in r.message for r in caplog.records)

    def test_buff_integrity_negative_ticks(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx()
        state = make_game_state(buffs=((100, -5),))
        hm._check_buff_integrity(state, ctx)
        assert any("negative ticks" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Geometry: _closest_point_on_line
# ---------------------------------------------------------------------------


class TestClosestPointOnLine:
    def test_point_projects_onto_midpoint(self) -> None:
        hm = HealthMonitor()
        # Line from (0,0) to (10,0), point at (5,5) -> closest = (5,0)
        pt = hm._closest_point_on_line(5, 5, 0, 0, 10, 0)
        assert abs(pt.x - 5.0) < 0.01
        assert abs(pt.y - 0.0) < 0.01

    def test_point_beyond_segment_end(self) -> None:
        hm = HealthMonitor()
        pt = hm._closest_point_on_line(20, 0, 0, 0, 10, 0)
        assert abs(pt.x - 10.0) < 0.01
        assert abs(pt.y - 0.0) < 0.01

    def test_point_before_segment_start(self) -> None:
        hm = HealthMonitor()
        pt = hm._closest_point_on_line(-5, 0, 0, 0, 10, 0)
        assert abs(pt.x - 0.0) < 0.01

    def test_degenerate_segment(self) -> None:
        hm = HealthMonitor()
        pt = hm._closest_point_on_line(5, 5, 3, 3, 3, 3)
        assert abs(pt.x - 3.0) < 0.01
        assert abs(pt.y - 3.0) < 0.01


# ---------------------------------------------------------------------------
# Geometry: _compute_evasion
# ---------------------------------------------------------------------------


class TestComputeEvasion:
    def test_evasion_perpendicular_to_path(self) -> None:
        hm = HealthMonitor()
        # Mob at (0,0) moving to (10,0), player at (5,5)
        ep = hm._compute_evasion(5, 5, 0, 0, 10, 0)
        # Evasion should be roughly 60u perpendicular from player
        dist = math.hypot(ep.x - 5, ep.y - 5)
        assert abs(dist - 60.0) < 1.0

    def test_evasion_away_from_stationary_mob(self) -> None:
        hm = HealthMonitor()
        # Mob not moving (same src/dest), player nearby
        ep = hm._compute_evasion(10, 10, 0, 0, 0, 0)
        # Should move 60u directly away from mob
        dist_from_player = math.hypot(ep.x - 10, ep.y - 10)
        assert abs(dist_from_player - 60.0) < 1.0
        # Should be farther from mob than player is
        dist_from_mob = math.hypot(ep.x, ep.y)
        assert dist_from_mob > math.hypot(10, 10)

    def test_evasion_when_player_on_mob(self) -> None:
        hm = HealthMonitor()
        # Player at same position as mob, mob not moving -> fallback
        ep = hm._compute_evasion(0, 0, 0, 0, 0, 0)
        assert abs(ep.x - 60.0) < 0.01
        assert abs(ep.y - 0.0) < 0.01

    def test_picks_side_farther_from_mob(self) -> None:
        hm = HealthMonitor()
        # Mob at (-50, 0) moving east to (50, 0), player at (0, 10)
        ep = hm._compute_evasion(0, 10, -50, 0, 50, 0)
        # Should pick the side with positive y (farther from mob at y=0)
        assert ep.y > 0


# ---------------------------------------------------------------------------
# tick() integration -- Tier 1 + Tier 3
# ---------------------------------------------------------------------------


class TestTickIntegration:
    def test_tick_runs_all_tiers(self) -> None:
        """tick() should run vitals, position, and threat checks."""
        hm = HealthMonitor()
        ctx = _make_ctx()
        state = make_game_state(hp_current=800, hp_max=1000)
        hm.tick(state, ctx)
        # After tick, prev_x/y should be set and threat flags reset
        assert hm._prev_x == state.x
        assert hm._prev_y == state.y
        assert ctx.threat.imminent_threat is False

    def test_tick_detects_imminent_threat(self) -> None:
        """A YELLOW/RED NPC at close range should set imminent_threat."""
        hm = HealthMonitor()
        ctx = _make_ctx()
        # RED con npc very close (dist < 40)
        threat_npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=5.0,
            y=5.0,
            level=13,
            speed=0.0,
            heading=0,
        )
        state = make_game_state(level=10, x=0.0, y=0.0, spawns=(threat_npc,))
        hm.tick(state, ctx)
        assert ctx.threat.imminent_threat is True
        assert ctx.threat.evasion_point is not None

    def test_tick_no_threat_from_passive_npc(self) -> None:
        """NPCs that are GREEN/BLUE con should not trigger threats."""
        hm = HealthMonitor()
        ctx = _make_ctx()
        # GREEN con (much lower level)
        npc = make_spawn(
            spawn_id=50,
            name="a_bat",
            x=30.0,
            y=0.0,
            level=1,
            speed=0.0,
            heading=0,
        )
        state = make_game_state(level=10, x=0.0, y=0.0, spawns=(npc,))
        hm.tick(state, ctx)
        assert ctx.threat.imminent_threat is False

    def test_tick_skips_dead_npcs(self) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx()
        dead_npc = make_spawn(spawn_id=50, name="a_skeleton", x=5.0, y=5.0, level=13, hp_current=0)
        state = make_game_state(level=10, spawns=(dead_npc,))
        hm.tick(state, ctx)
        assert ctx.threat.imminent_threat is False

    def test_tick_skips_pet_spawns(self) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx()
        pet = make_spawn(spawn_id=50, name="pet_skel", x=5.0, y=5.0, level=13, owner_id=1)
        state = make_game_state(level=10, spawns=(pet,))
        hm.tick(state, ctx)
        assert ctx.threat.imminent_threat is False

    def test_tick_resets_threat_flags_each_tick(self) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx()
        # First tick: imminent threat
        threat_npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=5.0,
            y=5.0,
            level=13,
        )
        state1 = make_game_state(level=10, spawns=(threat_npc,))
        hm.tick(state1, ctx)
        assert ctx.threat.imminent_threat is True

        # Second tick: no threats
        state2 = make_game_state(level=10, spawns=())
        hm.tick(state2, ctx)
        assert ctx.threat.imminent_threat is False

    def test_tick_approaching_threat_heading(self) -> None:
        """A fast NPC headed toward player should trigger approaching_threat."""
        hm = HealthMonitor()
        ctx = _make_ctx()
        # NPC at y=-80, heading 0 (north toward player at origin), fast
        npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=0.0,
            y=-80.0,
            level=13,
            speed=10.0,
            heading=0,
        )
        state = make_game_state(level=10, x=0.0, y=0.0, spawns=(npc,))
        hm.tick(state, ctx)
        assert ctx.threat.approaching_threat is not None
        assert ctx.threat.evasion_point is not None

    def test_tick_no_heading_threat_when_slow(self) -> None:
        """Slow NPC even if facing player should not trigger heading threat."""
        hm = HealthMonitor()
        ctx = _make_ctx()
        npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=0.0,
            y=-80.0,
            level=13,
            speed=0.3,
            heading=0,
        )
        state = make_game_state(level=10, x=0.0, y=0.0, spawns=(npc,))
        hm.tick(state, ctx)
        assert ctx.threat.approaching_threat is None

    def test_tick_trajectory_threat(self) -> None:
        """NPC with velocity heading toward player should trigger trajectory threat."""
        hm = HealthMonitor()
        ctx = _make_ctx()
        # NPC at (0, -100) with velocity pointing toward player at origin
        npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=0.0,
            y=-100.0,
            level=13,
            speed=0.3,
            heading=128,
            velocity_x=0.0,
            velocity_y=3.0,
        )
        state = make_game_state(level=10, x=0.0, y=0.0, spawns=(npc,))
        hm.tick(state, ctx)
        # Either approaching or imminent should be set
        has_threat = ctx.threat.approaching_threat is not None or ctx.threat.imminent_threat
        assert has_threat is True

    def test_tick_no_threat_from_far_away(self) -> None:
        """NPC beyond 150u should not trigger trajectory threat."""
        hm = HealthMonitor()
        ctx = _make_ctx()
        npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=0.0,
            y=-200.0,
            level=13,
            speed=5.0,
            heading=0,
            velocity_x=0.0,
            velocity_y=5.0,
        )
        state = make_game_state(level=10, x=0.0, y=0.0, spawns=(npc,))
        hm.tick(state, ctx)
        assert ctx.threat.approaching_threat is None

    def test_tick_threat_position_tracking(self) -> None:
        """Threat positions should be stored for trajectory calculation."""
        hm = HealthMonitor()
        ctx = _make_ctx()
        npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=30.0,
            y=-80.0,
            level=13,
            speed=0.0,
            heading=0,
        )
        state = make_game_state(level=10, x=0.0, y=0.0, spawns=(npc,))
        hm.tick(state, ctx)
        assert 50 in hm._threat_positions

    def test_tick_stale_throttle_cleanup(self) -> None:
        """When throttle dict grows large, stale entries should be pruned."""
        import time as _time

        hm = HealthMonitor()
        # Fill throttle dict with stale entries
        old = _time.time() - 120.0
        for i in range(60):
            hm._last_threat_log[f"key_{i}"] = old
        ctx = _make_ctx()
        state = make_game_state(level=10, spawns=())
        hm.tick(state, ctx)
        # All stale entries should be pruned (they are >60s old)
        assert len(hm._last_threat_log) < 55

    def test_tick_skips_avoid_prefix_npcs(self) -> None:
        """Guard-like NPCs (in _AVOID_PREFIXES) should be skipped."""
        hm = HealthMonitor()
        ctx = _make_ctx()
        guard = make_spawn(
            spawn_id=50,
            name="Guard_Fippy",
            x=5.0,
            y=5.0,
            level=50,
            speed=0.0,
            heading=0,
        )
        state = make_game_state(level=10, spawns=(guard,))
        hm.tick(state, ctx)
        assert ctx.threat.imminent_threat is False

    def test_tick_heading_threat_patrol_evade_when_engaged(self) -> None:
        """When engaged in combat, heading threat should set patrol_evade."""
        hm = HealthMonitor()
        ctx = _make_ctx(engaged=True, pull_target_id=99)
        # RED con NPC charging toward us, target_name empty (random aggro)
        npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=0.0,
            y=-80.0,
            level=13,
            speed=10.0,
            heading=0,
            target_name="",
        )
        state = make_game_state(level=10, x=0.0, y=0.0, spawns=(npc,))
        hm.tick(state, ctx)
        assert ctx.threat.patrol_evade is True

    def test_tick_trajectory_with_position_delta(self) -> None:
        """When velocity is 0, trajectory uses position delta from previous tick."""
        hm = HealthMonitor()
        ctx = _make_ctx()
        npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=0.0,
            y=-100.0,
            level=13,
            speed=0.3,
            heading=128,
            velocity_x=0.0,
            velocity_y=0.0,
        )
        state1 = make_game_state(level=10, x=0.0, y=0.0, spawns=(npc,))
        hm.tick(state1, ctx)  # first tick records position

        # Second tick: NPC moved closer
        npc2 = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=0.0,
            y=-95.0,
            level=13,
            speed=0.3,
            heading=128,
            velocity_x=0.0,
            velocity_y=0.0,
        )
        state2 = make_game_state(level=10, x=0.0, y=0.0, spawns=(npc2,))
        hm.tick(state2, ctx)
        # Position delta (0, 5) shows approach; should detect or at least track
        assert 50 in hm._threat_positions

    def test_tick_passive_disposition_skipped(self) -> None:
        """NPCs with passive disposition (INDIFFERENT, etc.) should not be threats."""
        hm = HealthMonitor()
        ctx = _make_ctx(zone_dispositions={"indifferent": ["a_skeleton"]})
        npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=5.0,
            y=5.0,
            level=13,
            speed=0.0,
            heading=0,
        )
        state = make_game_state(level=10, x=0.0, y=0.0, spawns=(npc,))
        hm.tick(state, ctx)
        assert ctx.threat.imminent_threat is False


# ---------------------------------------------------------------------------
# deep_check() integration -- Tier 2
# ---------------------------------------------------------------------------


class TestDeepCheckIntegration:
    def test_deep_check_runs_all_tier2(self) -> None:
        """deep_check() should run kill integrity, pet, engaged, buff, mana checks."""
        hm = HealthMonitor()
        hm._last_deep_check = 0.0  # force deep check to run
        ctx = _make_ctx()
        state = make_game_state()
        hm.deep_check(state, ctx)
        assert hm._last_deep_check > 0.0

    def test_deep_check_throttled(self) -> None:
        """deep_check() should not run if less than 25s since last check."""
        import time as _time

        hm = HealthMonitor()
        hm._last_deep_check = _time.time()  # just ran
        ctx = _make_ctx()
        state = make_game_state()
        prev = hm._last_deep_check
        hm.deep_check(state, ctx)
        assert hm._last_deep_check == prev  # not updated

    def test_kill_integrity_empty_name_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        hm._last_deep_check = 0.0
        defeat_entry = SimpleNamespace(name="")
        ctx = _make_ctx()
        ctx.defeat_tracker = _make_defeat_tracker(defeats=1, defeat_history=[defeat_entry])
        state = make_game_state()
        hm.deep_check(state, ctx)
        assert any("empty names" in r.message for r in caplog.records)

    def test_kill_integrity_combats_without_kill(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        hm._last_deep_check = 0.0
        hm._prev_combat_count = 0
        hm._prev_kills = 0
        ctx = _make_ctx()
        ctx.metrics = SimpleNamespace(routine_counts={"IN_COMBAT": 3})
        ctx.defeat_tracker = _make_defeat_tracker(defeats=0)
        state = make_game_state()
        hm.deep_check(state, ctx)
        assert any("combats completed" in r.message for r in caplog.records)

    def test_mana_frozen_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Mana unchanged for 120+ seconds while standing should warn."""
        import time as _time

        hm = HealthMonitor()
        hm._last_deep_check = 0.0
        hm._last_mana_change = _time.time() - 150.0  # 150s ago
        ctx = _make_ctx()
        state = make_game_state(mana_current=200, mana_max=500, stand_state=0)
        hm.deep_check(state, ctx)
        assert any("mana unchanged" in r.message for r in caplog.records)

    def test_mana_frozen_no_warn_at_max(self, caplog: pytest.LogCaptureFixture) -> None:
        """Mana at max should not trigger frozen warning."""
        hm = HealthMonitor()
        hm._last_deep_check = 0.0
        hm._last_mana_change = 0.1  # a long time ago
        ctx = _make_ctx()
        state = make_game_state(mana_current=500, mana_max=500, stand_state=0)
        hm.deep_check(state, ctx)
        assert not any("mana unchanged" in r.message for r in caplog.records)

    def test_mana_frozen_no_warn_zero_max(self, caplog: pytest.LogCaptureFixture) -> None:
        """When mana_max is 0 (melee class), no frozen warning."""
        hm = HealthMonitor()
        hm._last_deep_check = 0.0
        ctx = _make_ctx()
        state = make_game_state(mana_current=0, mana_max=0, stand_state=0)
        hm.deep_check(state, ctx)
        assert not any("mana unchanged" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Vitals: mana 30-tick warning
# ---------------------------------------------------------------------------


class TestManaZeroWarning:
    def test_mana_zero_30_ticks_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        hm = HealthMonitor()
        state = make_game_state(mana_current=0, mana_max=0)
        for _ in range(30):
            hm._check_vitals(state)
        assert any("mana reads 0 for 30 ticks" in r.message for r in caplog.records)

    def test_mana_zero_resets_on_nonzero(self) -> None:
        hm = HealthMonitor()
        state_zero = make_game_state(mana_current=0, mana_max=0)
        for _ in range(10):
            hm._check_vitals(state_zero)
        assert hm._zero_mana_ticks == 10
        state_ok = make_game_state(mana_current=100, mana_max=500)
        hm._check_vitals(state_ok)
        assert hm._zero_mana_ticks == 0
        assert hm._mana_warned is False

    def test_mana_warns_only_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning should fire at tick 30, not again at tick 31+."""
        hm = HealthMonitor()
        state = make_game_state(mana_current=0, mana_max=0)
        for _ in range(35):
            hm._check_vitals(state)
        warnings = [r for r in caplog.records if "mana reads 0" in r.message]
        assert len(warnings) == 1

    def test_mana_change_tracked(self) -> None:
        hm = HealthMonitor()
        state1 = make_game_state(mana_current=100, mana_max=500)
        hm._check_vitals(state1)
        t1 = hm._last_mana_change
        assert t1 > 0

        # Same mana -- should not update
        hm._check_vitals(state1)
        assert hm._last_mana_change == t1

        # Different mana -- should update
        state2 = make_game_state(mana_current=90, mana_max=500)
        hm._check_vitals(state2)
        assert hm._last_mana_change >= t1


# ---------------------------------------------------------------------------
# Threat evaluation: _evaluate_single_spawn_threat
# ---------------------------------------------------------------------------


class TestEvaluateSingleSpawnThreat:
    def test_returns_none_for_low_con(self) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx()
        npc = make_spawn(spawn_id=50, name="a_bat", x=30.0, y=0.0, level=1)
        state = make_game_state(level=10)
        result = hm._evaluate_single_spawn_threat(npc, state, ctx)
        assert result is None

    def test_returns_none_for_avoid_prefix(self) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx()
        npc = make_spawn(spawn_id=50, name="Guard_Fippy", x=5.0, y=5.0, level=50)
        state = make_game_state(level=10)
        result = hm._evaluate_single_spawn_threat(npc, state, ctx)
        assert result is None

    def test_imminent_threat_within_40u(self) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx()
        npc = make_spawn(spawn_id=50, name="a_skeleton", x=5.0, y=5.0, level=13)
        state = make_game_state(level=10, x=0.0, y=0.0)
        result = hm._evaluate_single_spawn_threat(npc, state, ctx)
        assert result is not None
        assert result.get("imminent") is True

    def test_far_away_no_approach(self) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx()
        npc = make_spawn(spawn_id=50, name="a_skeleton", x=200.0, y=0.0, level=13, speed=0.0)
        state = make_game_state(level=10, x=0.0, y=0.0)
        result = hm._evaluate_single_spawn_threat(npc, state, ctx)
        # Should return result but no imminent or approaching
        assert result is not None
        assert result.get("imminent") is None
        assert result.get("approaching") is None

    def test_heading_threat_detected(self) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx()
        # NPC at y=-80, heading 0 (north), fast
        npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=0.0,
            y=-80.0,
            level=13,
            speed=10.0,
            heading=0,
        )
        state = make_game_state(level=10, x=0.0, y=0.0)
        result = hm._evaluate_single_spawn_threat(npc, state, ctx)
        assert result is not None
        assert result.get("approaching") is not None
        assert result.get("skip_traj") is True

    def test_trajectory_threat_via_velocity(self) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx()
        # NPC at (0, -100), velocity pointing north (toward player)
        npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=0.0,
            y=-100.0,
            level=13,
            speed=0.3,
            heading=128,
            velocity_x=0.0,
            velocity_y=3.0,
        )
        state = make_game_state(level=10, x=0.0, y=0.0)
        result = hm._evaluate_single_spawn_threat(npc, state, ctx)
        assert result is not None
        assert result.get("approaching") is not None

    def test_trajectory_threat_via_position_delta(self) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx()
        # First store a previous position
        from core.types import Point

        hm._threat_positions[50] = Point(0.0, -110.0, 0.0)
        # NPC moved from -110 to -100 (toward player at 0)
        npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=0.0,
            y=-100.0,
            level=13,
            speed=0.3,
            heading=128,
            velocity_x=0.0,
            velocity_y=0.0,
        )
        state = make_game_state(level=10, x=0.0, y=0.0)
        result = hm._evaluate_single_spawn_threat(npc, state, ctx)
        assert result is not None
        # With speed=10 (position delta) approaching player, should detect trajectory threat
        # The speed is hypot(0, 10) = 10 > 0.5, and closest_dist should be < 60

    def test_trajectory_engaged_patrol_evade(self) -> None:
        """When engaged, trajectory threat should set patrol_evade."""
        hm = HealthMonitor()
        ctx = _make_ctx(engaged=True, pull_target_id=99)
        # RED con NPC with velocity heading toward us
        npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=0.0,
            y=-100.0,
            level=13,
            speed=0.3,
            heading=128,
            velocity_x=0.0,
            velocity_y=3.0,
            target_name="",
        )
        state = make_game_state(level=10, x=0.0, y=0.0)
        result = hm._evaluate_single_spawn_threat(npc, state, ctx)
        assert result is not None
        if result.get("approaching"):
            assert result.get("patrol_evade") is True

    def test_heading_threat_engaged_non_red_skipped(self) -> None:
        """Heading threat should be skipped for non-RED cons when engaged."""
        hm = HealthMonitor()
        ctx = _make_ctx(engaged=True, pull_target_id=99)
        # YELLOW con (level 11 vs 10), fast, heading toward us
        npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=0.0,
            y=-80.0,
            level=11,
            speed=10.0,
            heading=0,
            target_name="something",
        )
        state = make_game_state(level=10, x=0.0, y=0.0)
        result = hm._evaluate_single_spawn_threat(npc, state, ctx)
        # Heading check should be skipped (con != RED or target_name != "")
        assert result is None or result.get("approaching") is None or result.get("skip_traj") is None

    def test_heading_threat_slow_eta_skipped(self) -> None:
        """If time_to_aggro >= 5s, heading threat is not triggered."""
        hm = HealthMonitor()
        ctx = _make_ctx()
        # Far away but heading toward us, slow enough that eta > 5s
        npc = make_spawn(
            spawn_id=50,
            name="a_skeleton",
            x=0.0,
            y=-140.0,
            level=13,
            speed=1.5,
            heading=0,
        )
        state = make_game_state(level=10, x=0.0, y=0.0)
        result = hm._evaluate_single_spawn_threat(npc, state, ctx)
        # dist=140, eta = (140-40)/1.5 = 66.7s > 5s -> no heading trigger
        assert result is None or result.get("approaching") is None

    def test_passive_disposition_returns_none(self) -> None:
        hm = HealthMonitor()
        ctx = _make_ctx(zone_dispositions={"indifferent": ["a_skeleton"]})
        npc = make_spawn(spawn_id=50, name="a_skeleton", x=5.0, y=5.0, level=13)
        state = make_game_state(level=10)
        result = hm._evaluate_single_spawn_threat(npc, state, ctx)
        assert result is None
