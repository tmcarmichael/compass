"""Focused tests for simulator scenario definitions and JSON loading."""

from __future__ import annotations

import json

from perception.state import GameState
from simulator.scenarios import Scenario


# -- Built-in scenario shape --------------------------------------------------


class TestCampSession:
    def test_tick_count_scales_with_cycles(self) -> None:
        s1 = Scenario.camp_session(cycles=1)
        s4 = Scenario.camp_session(cycles=4)
        assert s1.tick_count > 0
        assert s4.tick_count == s1.tick_count * 4

    def test_phase_order_repeats_per_cycle(self) -> None:
        s = Scenario.camp_session(cycles=2)
        phases = _unique_phases_in_order(s)
        # Each cycle: idle -> pull -> combat -> rest
        assert phases == ["idle", "pull", "combat", "rest", "idle", "pull", "combat", "rest"]

    def test_all_states_are_game_states(self) -> None:
        s = Scenario.camp_session(cycles=1)
        for gs, label in s.states:
            assert isinstance(gs, GameState)
            assert isinstance(label, str) and label

    def test_combat_ticks_have_target(self) -> None:
        s = Scenario.camp_session(cycles=1)
        for gs, label in s.states:
            if label == "combat":
                assert gs.target is not None

    def test_name(self) -> None:
        assert Scenario.camp_session().name == "camp_session"


class TestSurvivalStress:
    def test_phase_progression(self) -> None:
        s = Scenario.survival_stress()
        phases = _unique_phases_in_order(s)
        assert phases == ["healthy", "damage_ramp", "critical", "recovery", "adds"]

    def test_critical_phase_has_low_hp(self) -> None:
        s = Scenario.survival_stress()
        for gs, label in s.states:
            if label == "critical":
                assert gs.hp_pct < 0.30

    def test_adds_phase_has_multiple_spawns(self) -> None:
        s = Scenario.survival_stress()
        for gs, label in s.states:
            if label == "adds":
                assert len(gs.spawns) >= 2
                break

    def test_name(self) -> None:
        assert Scenario.survival_stress().name == "survival_stress"


class TestExploration:
    def test_phase_progression(self) -> None:
        s = Scenario.exploration()
        phases = _unique_phases_in_order(s)
        # Starts with wander, then repeating approach/combat/rest batches
        assert phases[0] == "wander_empty"
        assert set(phases[1:]) == {"approach", "combat", "rest"}

    def test_wander_has_no_targets(self) -> None:
        s = Scenario.exploration()
        for gs, label in s.states:
            if label == "wander_empty":
                assert gs.target is None
                assert len(gs.spawns) == 0

    def test_name(self) -> None:
        assert Scenario.exploration().name == "exploration"


# -- JSON loading --------------------------------------------------------------


class TestFromJson:
    def test_basic_load(self, tmp_path) -> None:
        data = {
            "name": "json_test",
            "phases": [
                {"label": "idle", "ticks": 5, "state": {}},
                {"label": "fight", "ticks": 3, "state": {"in_combat": True}},
            ],
        }
        p = tmp_path / "scenario.json"
        p.write_text(json.dumps(data))
        s = Scenario.from_json(str(p))
        assert s.name == "json_test"
        assert s.tick_count == 8
        phases = _unique_phases_in_order(s)
        assert phases == ["idle", "fight"]

    def test_hp_pct_convenience(self, tmp_path) -> None:
        data = {
            "phases": [
                {"label": "hurt", "ticks": 1, "state": {"hp_pct": 0.5, "hp_max": 200}},
            ],
        }
        p = tmp_path / "s.json"
        p.write_text(json.dumps(data))
        s = Scenario.from_json(str(p))
        gs, _ = s.states[0]
        assert gs.hp_current == 100
        assert gs.hp_max == 200

    def test_mana_pct_convenience(self, tmp_path) -> None:
        data = {
            "phases": [
                {"label": "oom", "ticks": 1, "state": {"mana_pct": 0.2}},
            ],
        }
        p = tmp_path / "s.json"
        p.write_text(json.dumps(data))
        s = Scenario.from_json(str(p))
        gs, _ = s.states[0]
        assert gs.mana_current == 100  # 0.2 * 500 default

    def test_spawns_in_json(self, tmp_path) -> None:
        data = {
            "phases": [
                {
                    "label": "pull",
                    "ticks": 2,
                    "state": {
                        "spawns": [{"spawn_id": 42, "name": "orc", "level": 5}],
                    },
                },
            ],
        }
        p = tmp_path / "s.json"
        p.write_text(json.dumps(data))
        s = Scenario.from_json(str(p))
        gs, _ = s.states[0]
        assert len(gs.spawns) == 1
        assert gs.spawns[0].spawn_id == 42
        assert gs.spawns[0].name == "orc"

    def test_target_in_json(self, tmp_path) -> None:
        data = {
            "phases": [
                {
                    "label": "fight",
                    "ticks": 1,
                    "state": {
                        "target": {"spawn_id": 99, "name": "bat"},
                    },
                },
            ],
        }
        p = tmp_path / "s.json"
        p.write_text(json.dumps(data))
        s = Scenario.from_json(str(p))
        gs, _ = s.states[0]
        assert gs.target is not None
        assert gs.target.spawn_id == 99

    def test_default_ticks(self, tmp_path) -> None:
        """Phases without explicit ticks default to 10."""
        data = {"phases": [{"label": "x", "state": {}}]}
        p = tmp_path / "s.json"
        p.write_text(json.dumps(data))
        s = Scenario.from_json(str(p))
        assert s.tick_count == 10

    def test_name_defaults_to_path(self, tmp_path) -> None:
        data = {"phases": [{"label": "a", "ticks": 1, "state": {}}]}
        p = tmp_path / "s.json"
        p.write_text(json.dumps(data))
        s = Scenario.from_json(str(p))
        assert s.name == str(p)


# -- tick_count property -------------------------------------------------------


def test_tick_count_matches_len() -> None:
    s = Scenario(name="empty", states=[])
    assert s.tick_count == 0
    s2 = Scenario.camp_session(cycles=1)
    assert s2.tick_count == len(s2.states)


# -- Helpers -------------------------------------------------------------------


def _unique_phases_in_order(s: Scenario) -> list[str]:
    """Return phase labels in order, collapsing consecutive duplicates."""
    result: list[str] = []
    for _, label in s.states:
        if not result or result[-1] != label:
            result.append(label)
    return result
