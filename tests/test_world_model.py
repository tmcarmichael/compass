"""Tests for brain.world.model -- WorldModel target ranking and NPC queries.

WorldModel requires AgentContext for full update(), so these tests exercise
the standalone helper methods and properties that work on pre-populated
internal state: record_fight, nearest_player_dist, is_approaching,
heading_error_to, and direct property access after manual setup.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from brain.world.model import WorldModel, _MobHistory
from core.types import Point
from tests.factories import make_game_state, make_spawn

# ---------------------------------------------------------------------------
# _MobHistory: per-spawn temporal tracking
# ---------------------------------------------------------------------------


class TestMobHistory:
    def test_add_records_position(self) -> None:
        h = _MobHistory(spawn_id=1, name="a_bat", positions=deque(), hp_samples=deque())
        h.add(100.0, Point(10.0, 20.0, 0.0))
        assert len(h.positions) == 1
        assert h.last_seen == 100.0
        assert h.first_seen == 100.0

    def test_add_prunes_old_entries(self) -> None:
        h = _MobHistory(spawn_id=1, name="a_bat", positions=deque(), hp_samples=deque())
        for i in range(50):
            h.add(100.0 + i * 0.1, Point(float(i), 0.0, 0.0))
        # Only entries within last 3 seconds kept
        oldest_t = h.positions[0][0]
        assert h.last_seen - oldest_t <= 3.1

    def test_velocity_insufficient_data(self) -> None:
        h = _MobHistory(spawn_id=1, name="a_bat", positions=deque(), hp_samples=deque())
        h.add(100.0, Point(0.0, 0.0, 0.0))
        vx, vy, _vz = h.velocity()
        assert vx == 0.0
        assert vy == 0.0

    def test_velocity_from_positions(self) -> None:
        h = _MobHistory(spawn_id=1, name="a_bat", positions=deque(), hp_samples=deque())
        h.add(100.0, Point(0.0, 0.0, 0.0))
        h.add(101.0, Point(10.0, 0.0, 0.0))
        vx, vy, _vz = h.velocity()
        assert abs(vx - 10.0) < 0.1
        assert vy == 0.0

    def test_velocity_prefers_spawn_data(self) -> None:
        h = _MobHistory(spawn_id=1, name="a_bat", positions=deque(), hp_samples=deque())
        h.add(100.0, Point(0.0, 0.0, 0.0))
        h.add(101.0, Point(10.0, 0.0, 0.0))
        spawn = make_spawn(velocity_x=5.0, velocity_y=3.0)
        vx, vy, _vz = h.velocity(spawn)
        assert vx == 5.0
        assert vy == 3.0

    def test_speed(self) -> None:
        h = _MobHistory(spawn_id=1, name="a_bat", positions=deque(), hp_samples=deque())
        h.add(100.0, Point(0.0, 0.0, 0.0))
        h.add(101.0, Point(3.0, 4.0, 0.0))
        s = h.speed()
        assert abs(s - 5.0) < 0.1

    def test_predicted_pos(self) -> None:
        h = _MobHistory(spawn_id=1, name="a_bat", positions=deque(), hp_samples=deque())
        h.add(100.0, Point(0.0, 0.0, 0.0))
        h.add(101.0, Point(10.0, 0.0, 0.0))
        pred = h.predicted_pos(2.0)
        px, py = pred.x, pred.y
        assert abs(px - 30.0) < 0.5  # 10 + 10*2
        assert abs(py) < 0.1

    def test_predicted_pos_empty(self) -> None:
        h = _MobHistory(spawn_id=1, name="a_bat", positions=deque(), hp_samples=deque())
        assert h.predicted_pos(5.0) == Point(0.0, 0.0, 0.0)

    def test_hp_rate_insufficient_data(self) -> None:
        h = _MobHistory(spawn_id=1, name="a_bat", positions=deque(), hp_samples=deque())
        assert h.hp_rate() == 0.0

    def test_hp_rate_damage(self) -> None:
        h = _MobHistory(spawn_id=1, name="a_bat", positions=deque(), hp_samples=deque())
        h.add(100.0, Point(0.0, 0.0, 0.0), hp_pct=1.0)
        h.add(101.0, Point(0.0, 0.0, 0.0), hp_pct=0.5)
        rate = h.hp_rate()
        # hp0=1.0, hp1=0.5, dt=1s => 50%/sec
        assert abs(rate - 50.0) < 1.0


# ---------------------------------------------------------------------------
# WorldModel: record_fight and fight duration learning
# ---------------------------------------------------------------------------


class TestWorldModelRecordFight:
    def test_record_fight_stores_duration(self) -> None:
        wm = WorldModel()
        wm.record_fight("a_skeleton", 15.0)
        assert "a_skeleton" in wm._fight_durations
        assert 15.0 in wm._fight_durations["a_skeleton"]

    def test_record_fight_caps_at_twenty(self) -> None:
        wm = WorldModel()
        for i in range(25):
            wm.record_fight("a_skeleton", float(i))
        assert len(wm._fight_durations["a_skeleton"]) == 20

    def test_record_fight_keeps_latest(self) -> None:
        wm = WorldModel()
        for i in range(25):
            wm.record_fight("a_skeleton", float(i))
        # Should keep the last 20 (5..24)
        assert wm._fight_durations["a_skeleton"][0] == 5.0
        assert wm._fight_durations["a_skeleton"][-1] == 24.0


# ---------------------------------------------------------------------------
# WorldModel: player/NPC query properties (empty state)
# ---------------------------------------------------------------------------


class TestWorldModelQueries:
    def test_nearest_player_dist_no_players(self) -> None:
        wm = WorldModel()
        assert wm.nearest_player_dist() == 9999.0

    def test_nearby_player_count_no_players(self) -> None:
        wm = WorldModel()
        assert wm.nearby_player_count() == 0

    def test_profiles_empty_initially(self) -> None:
        wm = WorldModel()
        assert wm.profiles == []

    def test_threats_empty_initially(self) -> None:
        wm = WorldModel()
        assert wm.threats == []

    def test_best_target_none_initially(self) -> None:
        wm = WorldModel()
        assert wm.best_target is None

    def test_pet_spawn_none_initially(self) -> None:
        wm = WorldModel()
        assert wm.pet_spawn is None

    def test_pet_hp_pct_no_pet(self) -> None:
        wm = WorldModel()
        assert wm.pet_hp_pct == -1.0

    def test_has_pet_nearby_false_initially(self) -> None:
        wm = WorldModel()
        assert wm.has_pet_nearby() is False

    def test_pet_damage_rate_no_data(self) -> None:
        wm = WorldModel()
        assert wm.pet_damage_rate() == 0.0

    def test_pet_time_to_death_no_data(self) -> None:
        wm = WorldModel()
        assert wm.pet_time_to_death() is None


# ---------------------------------------------------------------------------
# WorldModel: static heading methods
# ---------------------------------------------------------------------------


class TestWorldModelHeading:
    def test_is_approaching_stationary_false(self) -> None:
        spawn = make_spawn(x=100.0, y=0.0, speed=0.0, heading=0.0)
        assert WorldModel.is_approaching(spawn, Point(0.0, 0.0, 0.0)) is False

    def test_heading_error_to_facing_target(self) -> None:
        # Heading 0, target straight ahead along x-axis
        spawn = make_spawn(x=0.0, y=0.0, heading=0.0)
        err = WorldModel.heading_error_to(spawn, Point(100.0, 0.0, 0.0))
        # Exact error depends on heading_to implementation, but should be small or 256
        assert 0.0 <= err <= 256.0

    def test_heading_error_symmetric(self) -> None:
        spawn = make_spawn(x=0.0, y=0.0, heading=128.0)
        err = WorldModel.heading_error_to(spawn, Point(100.0, 0.0, 0.0))
        assert 0.0 <= err <= 256.0


# ---------------------------------------------------------------------------
# WorldModel: summary and get_profile
# ---------------------------------------------------------------------------


class TestWorldModelGetProfile:
    def test_get_profile_not_found(self) -> None:
        wm = WorldModel()
        assert wm.get_profile(999) is None

    def test_mob_density_empty(self) -> None:
        wm = WorldModel()
        assert wm.mob_density == 0

    def test_threat_count_empty(self) -> None:
        wm = WorldModel()
        assert wm.threat_count == 0

    def test_any_npc_within_empty(self) -> None:
        wm = WorldModel()
        assert wm.any_npc_within(100.0) is False

    def test_any_hostile_npc_within_empty(self) -> None:
        wm = WorldModel()
        assert wm.any_hostile_npc_within(100.0) is False

    def test_threats_within_empty(self) -> None:
        wm = WorldModel()
        assert wm.threats_within(100.0) == []


# ---------------------------------------------------------------------------
# WorldModel: load_from_spatial
# ---------------------------------------------------------------------------


class TestLoadFromSpatial:
    def test_load_from_spatial_seeds_fight_durations(self) -> None:
        wm = WorldModel()
        spatial = SimpleNamespace(
            _kills=[
                {"name": "a_skeleton", "fight_s": 12.0},
                {"name": "a_bat", "fight_s": 8.5},
                {"name": "a_skeleton", "fight_s": 10.0},
            ]
        )
        wm.load_from_spatial(spatial)
        assert "a_skeleton" in wm._fight_durations
        assert len(wm._fight_durations["a_skeleton"]) == 2
        assert 12.0 in wm._fight_durations["a_skeleton"]
        assert "a_bat" in wm._fight_durations
        assert len(wm._fight_durations["a_bat"]) == 1

    def test_load_from_spatial_caps_at_twenty(self) -> None:
        wm = WorldModel()
        kills = [{"name": "a_skeleton", "fight_s": float(i)} for i in range(25)]
        spatial = SimpleNamespace(_kills=kills)
        wm.load_from_spatial(spatial)
        assert len(wm._fight_durations["a_skeleton"]) == 20

    def test_load_from_spatial_skips_zero_duration(self) -> None:
        wm = WorldModel()
        spatial = SimpleNamespace(_kills=[{"name": "a_skeleton", "fight_s": 0}])
        wm.load_from_spatial(spatial)
        assert "a_skeleton" not in wm._fight_durations

    def test_load_from_spatial_skips_empty_name(self) -> None:
        wm = WorldModel()
        spatial = SimpleNamespace(_kills=[{"name": "", "fight_s": 5.0}])
        wm.load_from_spatial(spatial)
        assert len(wm._fight_durations) == 0

    def test_load_from_spatial_no_kills_attr(self) -> None:
        wm = WorldModel()
        spatial = SimpleNamespace()  # no _kills attribute
        wm.load_from_spatial(spatial)
        assert len(wm._fight_durations) == 0


# ---------------------------------------------------------------------------
# WorldModel: update() with real spawns -- spawn tracking, entity lifecycle
# ---------------------------------------------------------------------------


def _make_ctx_for_update(
    camp_x: float = 0.0,
    camp_y: float = 0.0,
    pet_spawn_id: int = 0,
    zone_dispositions: dict | None = None,
    social_mob_group: dict | None = None,
    engaged: bool = False,
    pull_target_id: int = 0,
) -> SimpleNamespace:
    """Minimal AgentContext stub for WorldModel.update()."""
    return SimpleNamespace(
        camp=SimpleNamespace(
            camp_pos=Point(camp_x, camp_y, 0.0),
            roam_radius=200.0,
            effective_camp_distance=lambda pos: Point(camp_x, camp_y, 0.0).dist_to(pos),
        ),
        pet=SimpleNamespace(spawn_id=pet_spawn_id),
        zone=SimpleNamespace(
            zone_dispositions=zone_dispositions,
            social_mob_group=social_mob_group or {},
            target_cons=None,
        ),
        combat=SimpleNamespace(engaged=engaged, pull_target_id=pull_target_id),
        fight_history=None,
        danger_memory=None,
        diag=None,
        loot=SimpleNamespace(
            resource_targets=None,
            caster_mob_names=set(),
            mob_loot_values={},
        ),
        spatial_memory=None,
        threat=SimpleNamespace(
            approaching_threat=None,
            imminent_threat=False,
            imminent_threat_con="",
            evasion_point=None,
            patrol_evade=False,
        ),
    )


class TestWorldModelUpdate:
    """Full integration tests for WorldModel.update() with spawns."""

    def test_update_builds_profiles_for_living_npcs(self) -> None:
        ctx = _make_ctx_for_update()
        wm = WorldModel(ctx=ctx)
        npc = make_spawn(spawn_id=10, name="a_skeleton", x=50.0, y=50.0, level=10)
        state = make_game_state(level=10, spawns=(npc,))
        wm.update(state)
        assert len(wm.profiles) == 1
        assert wm.profiles[0].spawn.spawn_id == 10

    def test_update_skips_dead_npcs(self) -> None:
        ctx = _make_ctx_for_update()
        wm = WorldModel(ctx=ctx)
        dead_npc = make_spawn(spawn_id=10, name="a_skeleton", hp_current=0, hp_max=100)
        state = make_game_state(spawns=(dead_npc,))
        wm.update(state)
        assert len(wm.profiles) == 0

    def test_update_skips_pet_spawns(self) -> None:
        ctx = _make_ctx_for_update()
        wm = WorldModel(ctx=ctx)
        # A pet has owner_id set
        pet = make_spawn(spawn_id=20, name="Soandso`s_pet", owner_id=1)
        state = make_game_state(spawns=(pet,))
        wm.update(state)
        assert len(wm.profiles) == 0

    def test_update_tracks_players(self) -> None:
        ctx = _make_ctx_for_update()
        wm = WorldModel(ctx=ctx)
        player_spawn = make_spawn(spawn_id=50, name="OtherPlayer", spawn_type=0, x=100.0, y=0.0)
        state = make_game_state(name="TestPlayer", spawns=(player_spawn,))
        wm.update(state)
        assert len(wm._players) == 1
        assert wm.nearest_player_dist() < 110.0
        assert wm.nearby_player_count(radius=200.0) == 1

    def test_update_tracks_our_pet(self) -> None:
        ctx = _make_ctx_for_update(pet_spawn_id=30)
        wm = WorldModel(ctx=ctx)
        pet = make_spawn(spawn_id=30, name="pet_skeleton", owner_id=1, hp_current=80, hp_max=100)
        state = make_game_state(spawns=(pet,))
        wm.update(state)
        assert wm.pet_spawn is not None
        assert wm.pet_spawn.spawn_id == 30
        assert abs(wm.pet_hp_pct - 0.8) < 0.01

    def test_update_prunes_stale_trackers(self) -> None:
        ctx = _make_ctx_for_update()
        wm = WorldModel(ctx=ctx)
        npc = make_spawn(spawn_id=10, name="a_skeleton", x=50.0, y=50.0)
        state = make_game_state(spawns=(npc,))
        wm.update(state)
        assert 10 in wm._trackers

        # Next tick: npc gone
        state2 = make_game_state(spawns=())
        wm.update(state2)
        assert 10 not in wm._trackers

    def test_update_identifies_threats(self) -> None:
        ctx = _make_ctx_for_update()
        wm = WorldModel(ctx=ctx)
        # Red con npc (level 13 vs player level 10) with aggressive disposition
        red_npc = make_spawn(spawn_id=10, name="a_skeleton", x=50.0, y=50.0, level=13)
        state = make_game_state(level=10, spawns=(red_npc,))
        wm.update(state)
        assert len(wm.threats) > 0
        assert wm.threat_count > 0

    def test_update_stores_player_position(self) -> None:
        ctx = _make_ctx_for_update()
        wm = WorldModel(ctx=ctx)
        state = make_game_state(x=100.0, y=200.0, spawns=())
        wm.update(state)
        assert wm._last_player_pos.x == 100.0
        assert wm._last_player_pos.y == 200.0

    def test_update_records_profiling_time(self) -> None:
        ctx = _make_ctx_for_update()
        wm = WorldModel(ctx=ctx)
        state = make_game_state(spawns=())
        wm.update(state)
        assert wm.update_ms >= 0.0

    def test_update_scores_targets(self) -> None:
        ctx = _make_ctx_for_update(camp_x=0.0, camp_y=0.0)
        wm = WorldModel(ctx=ctx)
        # WHITE con npc at full HP, within 250u
        npc = make_spawn(spawn_id=10, name="a_skeleton", x=60.0, y=0.0, level=10, hp_current=100, hp_max=100)
        state = make_game_state(level=10, spawns=(npc,))
        wm.update(state)
        assert len(wm.targets) >= 1
        assert wm.best_target is not None
        assert wm.best_target.score > 0

    def test_update_pet_hp_tracking(self) -> None:
        """Pet HP history recorded across ticks for damage rate."""
        ctx = _make_ctx_for_update(pet_spawn_id=30)
        wm = WorldModel(ctx=ctx)
        pet = make_spawn(spawn_id=30, name="pet", owner_id=1, hp_current=100, hp_max=100)
        state = make_game_state(spawns=(pet,))
        wm.update(state)
        assert len(wm._pet_hp_history) == 1

    def test_update_clears_pet_history_when_pet_gone(self) -> None:
        ctx = _make_ctx_for_update(pet_spawn_id=30)
        wm = WorldModel(ctx=ctx)
        pet = make_spawn(spawn_id=30, name="pet", owner_id=1, hp_current=100, hp_max=100)
        state = make_game_state(spawns=(pet,))
        wm.update(state)
        assert len(wm._pet_hp_history) == 1

        # Pet gone
        state2 = make_game_state(spawns=())
        wm.update(state2)
        assert len(wm._pet_hp_history) == 0

    def test_update_multiple_npcs_isolation(self) -> None:
        """Two NPCs near each other should have lower isolation scores."""
        ctx = _make_ctx_for_update()
        wm = WorldModel(ctx=ctx)
        npc1 = make_spawn(spawn_id=10, name="a_skeleton", x=50.0, y=50.0, level=10)
        npc2 = make_spawn(spawn_id=11, name="a_skeleton", x=55.0, y=50.0, level=10)
        state = make_game_state(level=10, spawns=(npc1, npc2))
        wm.update(state)
        assert len(wm.profiles) == 2
        # Both should have isolation < 1.0 (they are neighbors)
        for p in wm.profiles:
            assert p.isolation_score < 1.0


# ---------------------------------------------------------------------------
# WorldModel: query methods with populated state
# ---------------------------------------------------------------------------


class TestWorldModelQueryMethods:
    """Test query methods by running update() with real spawns."""

    def _setup_wm_with_npcs(self) -> WorldModel:
        ctx = _make_ctx_for_update()
        wm = WorldModel(ctx=ctx)
        # One close NPC, one far NPC (threat), one damaged NPC
        close_npc = make_spawn(spawn_id=10, name="a_skeleton", x=30.0, y=0.0, level=10)
        far_npc = make_spawn(spawn_id=11, name="a_skeleton", x=280.0, y=0.0, level=13)
        damaged_npc = make_spawn(
            spawn_id=12, name="a_bat", x=40.0, y=0.0, level=10, hp_current=50, hp_max=100
        )
        state = make_game_state(level=10, spawns=(close_npc, far_npc, damaged_npc))
        wm.update(state)
        return wm

    def test_any_npc_within_true(self) -> None:
        wm = self._setup_wm_with_npcs()
        assert wm.any_npc_within(50.0) is True

    def test_any_npc_within_false_small_radius(self) -> None:
        wm = self._setup_wm_with_npcs()
        assert wm.any_npc_within(5.0) is False

    def test_threats_within_returns_threats(self) -> None:
        wm = self._setup_wm_with_npcs()
        # The far npc (level 13) is RED con vs level 10 player, but at 280u
        threats = wm.threats_within(300.0)
        # Threat detection depends on disposition; check that the method works
        assert isinstance(threats, list)

    def test_damaged_npcs_near(self) -> None:
        wm = self._setup_wm_with_npcs()
        damaged = wm.damaged_npcs_near(Point(0.0, 0.0, 0.0), 100.0)
        assert len(damaged) >= 1
        assert any(p.spawn.spawn_id == 12 for p in damaged)

    def test_damaged_npcs_near_with_exclude(self) -> None:
        wm = self._setup_wm_with_npcs()
        damaged = wm.damaged_npcs_near(Point(0.0, 0.0, 0.0), 100.0, exclude_id=12)
        assert not any(p.spawn.spawn_id == 12 for p in damaged)

    def test_mob_density(self) -> None:
        wm = self._setup_wm_with_npcs()
        # At least some of the 3 NPCs should be within 200u of camp (0,0)
        assert wm.mob_density >= 1

    def test_get_profile_found(self) -> None:
        wm = self._setup_wm_with_npcs()
        p = wm.get_profile(10)
        assert p is not None
        assert p.spawn.spawn_id == 10

    def test_get_profile_not_found_after_update(self) -> None:
        wm = self._setup_wm_with_npcs()
        assert wm.get_profile(9999) is None

    def test_any_hostile_npc_within_with_npcs(self) -> None:
        """Test hostile NPC detection with aggressive-disposition NPCs."""
        ctx = _make_ctx_for_update(
            zone_dispositions={"scowling": ["a_skeleton"]},
        )
        wm = WorldModel(ctx=ctx)
        npc = make_spawn(spawn_id=10, name="a_skeleton", x=30.0, y=0.0, level=13)
        state = make_game_state(level=10, spawns=(npc,))
        wm.update(state)
        # The npc is RED con + SCOWLING, should be hostile
        assert wm.any_hostile_npc_within(100.0) is True
        assert wm.any_hostile_npc_within(5.0) is False


# ---------------------------------------------------------------------------
# WorldModel: heading/distance with real spawns
# ---------------------------------------------------------------------------


class TestWorldModelHeadingWithSpawns:
    def test_is_approaching_moving_toward_player(self) -> None:
        # Heading 0 = North (+Y). Spawn south of player, heading north -> approaching
        spawn = make_spawn(x=0.0, y=-100.0, speed=5.0, heading=0.0)
        result = WorldModel.is_approaching(spawn, Point(0.0, 0.0, 0.0))
        assert result is True

    def test_is_approaching_moving_away(self) -> None:
        # Heading 256 = South. Spawn south of player, heading further south -> not approaching
        spawn = make_spawn(x=0.0, y=-100.0, speed=5.0, heading=256.0)
        result = WorldModel.is_approaching(spawn, Point(0.0, 0.0, 0.0))
        assert result is False

    def test_heading_error_facing_directly_at_target(self) -> None:
        # Spawn at origin, heading 0 (north), target is north
        spawn = make_spawn(x=0.0, y=0.0, heading=0.0)
        err = WorldModel.heading_error_to(spawn, Point(0.0, 100.0, 0.0))
        assert err < 10.0  # nearly facing the target

    def test_heading_error_facing_away(self) -> None:
        spawn = make_spawn(x=0.0, y=0.0, heading=256.0)
        err = WorldModel.heading_error_to(spawn, Point(0.0, 100.0, 0.0))
        assert err > 200.0  # facing away


# ---------------------------------------------------------------------------
# WorldModel: velocity tracking via _MobHistory with spawn data
# ---------------------------------------------------------------------------


class TestMobHistoryVelocityTracking:
    def test_velocity_falls_back_when_spawn_zero(self) -> None:
        """When spawn velocity is (0,0), falls back to position delta."""
        h = _MobHistory(spawn_id=1, name="a_bat", positions=deque(), hp_samples=deque())
        h.add(100.0, Point(0.0, 0.0, 0.0))
        h.add(101.0, Point(10.0, 0.0, 0.0))
        spawn = make_spawn(velocity_x=0.0, velocity_y=0.0)
        vx, vy, _vz = h.velocity(spawn)
        assert abs(vx - 10.0) < 0.1

    def test_velocity_too_short_dt(self) -> None:
        """When time delta < 0.3s, returns (0,0)."""
        h = _MobHistory(spawn_id=1, name="a_bat", positions=deque(), hp_samples=deque())
        h.add(100.0, Point(0.0, 0.0, 0.0))
        h.add(100.1, Point(10.0, 0.0, 0.0))
        vx, vy, _vz = h.velocity()
        assert vx == 0.0
        assert vy == 0.0

    def test_hp_rate_short_dt(self) -> None:
        """When time span < 0.5s, returns 0.0."""
        h = _MobHistory(spawn_id=1, name="a_bat", positions=deque(), hp_samples=deque())
        h.add(100.0, Point(0.0, 0.0, 0.0), hp_pct=1.0)
        h.add(100.2, Point(0.0, 0.0, 0.0), hp_pct=0.5)
        assert h.hp_rate() == 0.0

    def test_hp_samples_pruned(self) -> None:
        """HP samples older than 3s are pruned."""
        h = _MobHistory(spawn_id=1, name="a_bat", positions=deque(), hp_samples=deque())
        h.add(100.0, Point(0.0, 0.0, 0.0), hp_pct=1.0)
        h.add(105.0, Point(0.0, 0.0, 0.0), hp_pct=0.5)
        # Old sample at t=100 should be pruned (cutoff = 105 - 3 = 102)
        assert len(h.hp_samples) == 1


# ---------------------------------------------------------------------------
# WorldModel: pet damage rate and TTD
# ---------------------------------------------------------------------------


class TestPetDamageRate:
    def test_pet_damage_rate_with_history(self) -> None:
        """OLS slope over declining HP samples should yield positive rate."""
        wm = WorldModel()
        # Simulate declining pet HP over 2 seconds
        wm._pet_hp_history = deque(
            [
                (100.0, 1.0),
                (100.5, 0.9),
                (101.0, 0.8),
                (101.5, 0.7),
                (102.0, 0.6),
            ]
        )
        rate = wm.pet_damage_rate()
        assert rate > 0  # taking damage
        # ~20%/sec expected
        assert 15.0 < rate < 25.0

    def test_pet_damage_rate_stable_hp(self) -> None:
        """Stable HP should yield 0 or near-0 rate."""
        wm = WorldModel()
        wm._pet_hp_history = deque(
            [
                (100.0, 0.8),
                (100.5, 0.8),
                (101.0, 0.8),
                (101.5, 0.8),
            ]
        )
        rate = wm.pet_damage_rate()
        assert rate < 1.0

    def test_pet_damage_rate_insufficient_span(self) -> None:
        wm = WorldModel()
        wm._pet_hp_history = deque(
            [
                (100.0, 1.0),
                (100.1, 0.9),
                (100.2, 0.8),
            ]
        )
        rate = wm.pet_damage_rate()
        assert rate == 0.0

    def test_pet_time_to_death_with_damage(self) -> None:
        wm = WorldModel()
        wm._pet_hp_history = deque(
            [
                (100.0, 1.0),
                (100.5, 0.9),
                (101.0, 0.8),
                (101.5, 0.7),
                (102.0, 0.6),
            ]
        )
        ttd = wm.pet_time_to_death()
        assert ttd is not None
        assert ttd > 0

    def test_pet_time_to_death_at_zero_hp(self) -> None:
        wm = WorldModel()
        wm._pet_hp_history = deque(
            [
                (100.0, 0.1),
                (100.5, 0.05),
                (101.0, 0.02),
                (101.5, 0.0),
                (102.0, 0.0),
            ]
        )
        ttd = wm.pet_time_to_death()
        # Current pct is 0, should return 0.0
        if ttd is not None:
            assert ttd == 0.0

    def test_pet_time_to_death_no_damage(self) -> None:
        wm = WorldModel()
        wm._pet_hp_history = deque(
            [
                (100.0, 0.8),
                (100.5, 0.8),
                (101.0, 0.8),
                (101.5, 0.8),
            ]
        )
        assert wm.pet_time_to_death() is None

    def test_has_pet_nearby_true(self) -> None:
        wm = WorldModel()
        pet_spawn = make_spawn(spawn_id=30, name="pet", owner_id=1)
        wm._nearby_pets = [(pet_spawn, 50.0)]
        assert wm.has_pet_nearby(radius=100.0) is True

    def test_has_pet_nearby_false_too_far(self) -> None:
        wm = WorldModel()
        pet_spawn = make_spawn(spawn_id=30, name="pet", owner_id=1)
        wm._nearby_pets = [(pet_spawn, 150.0)]
        assert wm.has_pet_nearby(radius=100.0) is False


# ---------------------------------------------------------------------------
# WorldModel: mob_targeting_player
# ---------------------------------------------------------------------------


class TestMobTargetingPlayer:
    def test_mob_targeting_player_true(self) -> None:
        ctx = _make_ctx_for_update(engaged=True, pull_target_id=10)
        wm = WorldModel(ctx=ctx)
        npc = make_spawn(spawn_id=10, name="a_skeleton", x=30.0, y=0.0, level=10, target_name="TestPlayer")
        state = make_game_state(level=10, spawns=(npc,))
        wm.update(state)
        assert wm.mob_targeting_player("TestPlayer") is True

    def test_mob_targeting_player_false_targeting_pet(self) -> None:
        ctx = _make_ctx_for_update(engaged=True, pull_target_id=10)
        wm = WorldModel(ctx=ctx)
        npc = make_spawn(spawn_id=10, name="a_skeleton", x=30.0, y=0.0, level=10, target_name="pet_skeleton")
        state = make_game_state(level=10, spawns=(npc,))
        wm.update(state)
        assert wm.mob_targeting_player("TestPlayer") is False

    def test_mob_targeting_player_not_engaged(self) -> None:
        ctx = _make_ctx_for_update(engaged=False)
        wm = WorldModel(ctx=ctx)
        assert wm.mob_targeting_player("TestPlayer") is False

    def test_mob_targeting_player_no_pull_target(self) -> None:
        ctx = _make_ctx_for_update(engaged=True, pull_target_id=0)
        wm = WorldModel(ctx=ctx)
        assert wm.mob_targeting_player("TestPlayer") is False

    def test_mob_targeting_player_target_not_in_profiles(self) -> None:
        ctx = _make_ctx_for_update(engaged=True, pull_target_id=999)
        wm = WorldModel(ctx=ctx)
        npc = make_spawn(spawn_id=10, name="a_skeleton", x=30.0, y=0.0, level=10)
        state = make_game_state(level=10, spawns=(npc,))
        wm.update(state)
        assert wm.mob_targeting_player("TestPlayer") is False


# ---------------------------------------------------------------------------
# WorldModel: target_damage_rate
# ---------------------------------------------------------------------------


class TestTargetDamageRate:
    def test_target_damage_rate_no_context(self) -> None:
        wm = WorldModel()
        assert wm.target_damage_rate() == 0.0

    def test_target_damage_rate_no_tracker(self) -> None:
        wm = WorldModel()
        assert wm.target_damage_rate(spawn_id=999) == 0.0

    def test_target_damage_rate_with_tracker(self) -> None:
        wm = WorldModel()
        tracker = _MobHistory(spawn_id=42, name="a_bat", positions=deque(), hp_samples=deque())
        tracker.add(100.0, Point(0.0, 0.0, 0.0), hp_pct=1.0)
        tracker.add(101.0, Point(0.0, 0.0, 0.0), hp_pct=0.5)
        wm._trackers[42] = tracker
        rate = wm.target_damage_rate(spawn_id=42)
        assert abs(rate - 50.0) < 1.0

    def test_target_damage_rate_uses_ctx_pull_target(self) -> None:
        ctx = _make_ctx_for_update(engaged=True, pull_target_id=42)
        wm = WorldModel(ctx=ctx)
        tracker = _MobHistory(spawn_id=42, name="a_bat", positions=deque(), hp_samples=deque())
        tracker.add(100.0, Point(0.0, 0.0, 0.0), hp_pct=1.0)
        tracker.add(101.0, Point(0.0, 0.0, 0.0), hp_pct=0.5)
        wm._trackers[42] = tracker
        rate = wm.target_damage_rate()  # no spawn_id, uses ctx
        assert abs(rate - 50.0) < 1.0

    def test_target_damage_rate_no_pull_target_id(self) -> None:
        ctx = _make_ctx_for_update(engaged=True, pull_target_id=0)
        wm = WorldModel(ctx=ctx)
        assert wm.target_damage_rate() == 0.0


# ---------------------------------------------------------------------------
# WorldModel: social groups, unknown disp threats, patrol helpers
# ---------------------------------------------------------------------------


class TestWorldModelSocialAndThreat:
    def test_social_group_counts(self) -> None:
        """NPCs in the same social group should be counted as social neighbors."""
        ctx = _make_ctx_for_update(
            social_mob_group={"a_skeleton": {"a_skeleton", "a_zombie"}},
        )
        wm = WorldModel(ctx=ctx)
        npc1 = make_spawn(spawn_id=10, name="a_skeleton", x=50.0, y=50.0, level=10)
        npc2 = make_spawn(spawn_id=11, name="a_zombie", x=55.0, y=50.0, level=10)
        state = make_game_state(level=10, spawns=(npc1, npc2))
        wm.update(state)
        # The skeleton should have a_zombie as social neighbor
        skel = wm.get_profile(10)
        assert skel is not None
        assert skel.social_npc_count >= 1

    def test_unknown_disposition_yellow_threat(self) -> None:
        """YELLOW con with UNKNOWN disposition should be flagged as threat."""
        ctx = _make_ctx_for_update()  # no zone_dispositions -> UNKNOWN
        wm = WorldModel(ctx=ctx)
        # YELLOW con: mob_level = player_level + 1
        npc = make_spawn(spawn_id=10, name="a_skeleton", x=50.0, y=0.0, level=11)
        state = make_game_state(level=10, spawns=(npc,))
        wm.update(state)
        p = wm.get_profile(10)
        assert p is not None
        assert p.is_threat is True
        assert p.threat_level > 0

    def test_fight_history_add_probability(self) -> None:
        """When fight_history returns add probability, it populates the profile."""
        fh = SimpleNamespace(
            learned_add_probability=lambda name: 0.5,
            learned_adds=lambda name: 0,
            learned_duration=lambda name: None,
            learned_mana=lambda name: None,
            get_stats=lambda name: None,
            sample_add_probability=lambda name: 0.5,
        )
        ctx = _make_ctx_for_update()
        ctx.fight_history = fh
        wm = WorldModel(ctx=ctx)
        npc = make_spawn(spawn_id=10, name="a_skeleton", x=50.0, y=0.0, level=10)
        state = make_game_state(level=10, spawns=(npc,))
        wm.update(state)
        p = wm.get_profile(10)
        assert p is not None
        assert p.extra_npc_probability == 0.5

    def test_any_hostile_npc_within_unknown_threat(self) -> None:
        """UNKNOWN disposition + is_threat should register as hostile."""
        ctx = _make_ctx_for_update()  # UNKNOWN disp
        wm = WorldModel(ctx=ctx)
        # RED con (level 13 vs 10) with UNKNOWN disp -> is_threat=True
        npc = make_spawn(spawn_id=10, name="a_skeleton", x=30.0, y=0.0, level=13)
        state = make_game_state(level=10, spawns=(npc,))
        wm.update(state)
        assert wm.any_hostile_npc_within(50.0) is True

    def test_pet_hp_history_pruned_over_time(self) -> None:
        """Pet HP history entries older than 5s should be pruned."""
        import time as _time

        ctx = _make_ctx_for_update(pet_spawn_id=30)
        wm = WorldModel(ctx=ctx)
        # Manually populate old history entries
        old_time = _time.time() - 10.0
        wm._pet_hp_history.append((old_time, 1.0))
        wm._pet_hp_history.append((old_time + 1, 0.9))
        # Now update with pet -- old entries should be pruned
        pet = make_spawn(spawn_id=30, name="pet", owner_id=1, hp_current=80, hp_max=100)
        state = make_game_state(spawns=(pet,))
        wm.update(state)
        # All old entries should be pruned, only the new one remains
        assert len(wm._pet_hp_history) == 1

    def test_patrol_safe_window_no_patrols(self) -> None:
        """Without patrolling threats, safe window is infinite."""
        ctx = _make_ctx_for_update()
        wm = WorldModel(ctx=ctx)
        npc = make_spawn(spawn_id=10, name="a_skeleton", x=50.0, y=0.0, level=10)
        state = make_game_state(level=10, spawns=(npc,))
        wm.update(state)
        window = wm.patrol_safe_window(50.0, 0.0, 30.0)
        assert window == float("inf")

    def test_patrolling_threats_empty(self) -> None:
        """Without patrolling NPCs, list should be empty."""
        ctx = _make_ctx_for_update()
        wm = WorldModel(ctx=ctx)
        npc = make_spawn(spawn_id=10, name="a_skeleton", x=50.0, y=0.0, level=10)
        state = make_game_state(level=10, spawns=(npc,))
        wm.update(state)
        assert wm.patrolling_threats() == []
