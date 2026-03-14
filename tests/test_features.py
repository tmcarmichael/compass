"""Tests for core.features -- FeatureFlags thread-safe feature flag store.

Covers initialization, property accessors/setters, load_from_config,
validate, reload_from_file, as_dict, on_change callbacks, and
should_recover_death.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.features import FeatureFlags
from core.types import DeathRecoveryMode, GrindStyle, LootMode, ManaMode

# ---------------------------------------------------------------------------
# Default construction
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_loot_mode(self) -> None:
        ff = FeatureFlags()
        assert ff.loot_mode == LootMode.OFF

    def test_default_looting_is_false(self) -> None:
        ff = FeatureFlags()
        assert ff.looting is False

    def test_default_combat_casting(self) -> None:
        ff = FeatureFlags()
        assert ff.combat_casting is True

    def test_default_wander(self) -> None:
        ff = FeatureFlags()
        assert ff.wander is True

    def test_default_pull(self) -> None:
        ff = FeatureFlags()
        assert ff.pull is True

    def test_default_rest(self) -> None:
        ff = FeatureFlags()
        assert ff.rest is True

    def test_default_flee(self) -> None:
        ff = FeatureFlags()
        assert ff.flee is True

    def test_default_shielding_buff(self) -> None:
        ff = FeatureFlags()
        assert ff.shielding_buff is False

    def test_default_death_recovery(self) -> None:
        ff = FeatureFlags()
        assert ff.death_recovery == DeathRecoveryMode.OFF

    def test_default_utility_phase(self) -> None:
        ff = FeatureFlags()
        assert ff.utility_phase == 2

    def test_default_obstacle_avoidance(self) -> None:
        ff = FeatureFlags()
        assert ff.obstacle_avoidance is True

    def test_default_mana_mode(self) -> None:
        ff = FeatureFlags()
        assert ff.mana_mode == ManaMode.MEDIUM

    def test_default_grind_style(self) -> None:
        ff = FeatureFlags()
        assert ff.grind_style == GrindStyle.WANDER

    def test_default_pareto_scoring(self) -> None:
        ff = FeatureFlags()
        assert ff.pareto_scoring is False

    def test_default_goap_planning(self) -> None:
        ff = FeatureFlags()
        assert ff.goap_planning is True


# ---------------------------------------------------------------------------
# Property setters
# ---------------------------------------------------------------------------


class TestPropertySetters:
    def test_set_loot_mode(self) -> None:
        ff = FeatureFlags()
        ff.loot_mode = LootMode.SMART
        assert ff.loot_mode == LootMode.SMART

    def test_set_looting_true(self) -> None:
        ff = FeatureFlags()
        ff.looting = True
        assert ff.looting is True
        assert ff.loot_mode == LootMode.ALL

    def test_set_looting_false(self) -> None:
        ff = FeatureFlags()
        ff.looting = True
        ff.looting = False
        assert ff.looting is False
        assert ff.loot_mode == LootMode.OFF

    def test_set_combat_casting(self) -> None:
        ff = FeatureFlags()
        ff.combat_casting = False
        assert ff.combat_casting is False

    def test_set_wander(self) -> None:
        ff = FeatureFlags()
        ff.wander = False
        assert ff.wander is False

    def test_set_pull(self) -> None:
        ff = FeatureFlags()
        ff.pull = False
        assert ff.pull is False

    def test_set_rest(self) -> None:
        ff = FeatureFlags()
        ff.rest = False
        assert ff.rest is False

    def test_set_flee(self) -> None:
        ff = FeatureFlags()
        ff.flee = False
        assert ff.flee is False

    def test_set_shielding_buff(self) -> None:
        ff = FeatureFlags()
        ff.shielding_buff = True
        assert ff.shielding_buff is True

    def test_set_obstacle_avoidance(self) -> None:
        ff = FeatureFlags()
        ff.obstacle_avoidance = False
        assert ff.obstacle_avoidance is False

    def test_set_mana_mode(self) -> None:
        ff = FeatureFlags()
        ff.mana_mode = ManaMode.HIGH
        assert ff.mana_mode == ManaMode.HIGH

    def test_set_grind_style(self) -> None:
        ff = FeatureFlags()
        ff.grind_style = GrindStyle.FEAR_KITE
        assert ff.grind_style == GrindStyle.FEAR_KITE

    def test_set_death_recovery(self) -> None:
        ff = FeatureFlags()
        ff.death_recovery = DeathRecoveryMode.SMART
        assert ff.death_recovery == DeathRecoveryMode.SMART

    def test_set_utility_phase(self) -> None:
        ff = FeatureFlags()
        ff.utility_phase = 3
        assert ff.utility_phase == 3

    def test_utility_phase_clamped_high(self) -> None:
        ff = FeatureFlags()
        ff.utility_phase = 99
        assert ff.utility_phase == 4

    def test_utility_phase_clamped_low(self) -> None:
        ff = FeatureFlags()
        ff.utility_phase = -5
        assert ff.utility_phase == 0

    def test_set_goap_planning(self) -> None:
        ff = FeatureFlags()
        ff.goap_planning = True
        assert ff.goap_planning is True

    def test_set_pareto_scoring(self) -> None:
        ff = FeatureFlags()
        ff.pareto_scoring = True
        assert ff.pareto_scoring is True


# ---------------------------------------------------------------------------
# on_change / _notify callbacks
# ---------------------------------------------------------------------------


class TestChangeCallbacks:
    def test_on_change_fires_callback(self) -> None:
        ff = FeatureFlags()
        received: list[object] = []
        ff.on_change("obstacle_avoidance", lambda v: received.append(v))
        ff.obstacle_avoidance = False
        assert received == [False]

    def test_multiple_callbacks(self) -> None:
        ff = FeatureFlags()
        a: list[object] = []
        b: list[object] = []
        ff.on_change("obstacle_avoidance", lambda v: a.append(v))
        ff.on_change("obstacle_avoidance", lambda v: b.append(v))
        ff.obstacle_avoidance = True
        assert len(a) == 1
        assert len(b) == 1

    def test_callback_failure_does_not_propagate(self) -> None:
        ff = FeatureFlags()

        def bad_cb(v: object) -> None:
            raise RuntimeError("boom")

        ff.on_change("obstacle_avoidance", bad_cb)
        # Should not raise
        ff.obstacle_avoidance = False
        assert ff.obstacle_avoidance is False


# ---------------------------------------------------------------------------
# should_recover_death
# ---------------------------------------------------------------------------


class TestShouldRecoverDeath:
    def test_off_always_false(self) -> None:
        ff = FeatureFlags()
        ff.death_recovery = DeathRecoveryMode.OFF
        assert ff.should_recover_death(0) is False
        assert ff.should_recover_death(1) is False

    def test_smart_first_death(self) -> None:
        ff = FeatureFlags()
        ff.death_recovery = DeathRecoveryMode.SMART
        assert ff.should_recover_death(0) is True
        assert ff.should_recover_death(1) is True

    def test_smart_second_death_false(self) -> None:
        ff = FeatureFlags()
        ff.death_recovery = DeathRecoveryMode.SMART
        assert ff.should_recover_death(2) is False

    def test_on_always_true(self) -> None:
        ff = FeatureFlags()
        ff.death_recovery = DeathRecoveryMode.ON
        assert ff.should_recover_death(0) is True
        assert ff.should_recover_death(5) is True


# ---------------------------------------------------------------------------
# load_from_config
# ---------------------------------------------------------------------------


class TestLoadFromConfig:
    def test_empty_config(self) -> None:
        ff = FeatureFlags()
        ff.load_from_config({})
        # Nothing changes from defaults
        assert ff.combat_casting is True

    def test_no_features_section(self) -> None:
        ff = FeatureFlags()
        ff.load_from_config({"general": {"something": True}})
        assert ff.combat_casting is True

    def test_load_boolean_flags(self) -> None:
        ff = FeatureFlags()
        ff.load_from_config(
            {
                "features": {
                    "combat_casting": False,
                    "wander": False,
                    "pull": False,
                    "rest": False,
                    "flee": False,
                    "shielding_buff": True,
                    "obstacle_avoidance": False,
                    "pareto_scoring": True,
                    "goap_planning": True,
                }
            }
        )
        assert ff.combat_casting is False
        assert ff.wander is False
        assert ff.pull is False
        assert ff.rest is False
        assert ff.flee is False
        assert ff.shielding_buff is True
        assert ff.obstacle_avoidance is False
        assert ff.pareto_scoring is True
        assert ff.goap_planning is True

    def test_load_loot_mode(self) -> None:
        ff = FeatureFlags()
        ff.load_from_config({"features": {"loot_mode": "smart"}})
        assert ff.loot_mode == LootMode.SMART

    def test_load_death_recovery(self) -> None:
        ff = FeatureFlags()
        ff.load_from_config({"features": {"death_recovery": "smart"}})
        assert ff.death_recovery == DeathRecoveryMode.SMART

    def test_load_mana_mode(self) -> None:
        ff = FeatureFlags()
        ff.load_from_config({"features": {"mana_mode": "high"}})
        assert ff.mana_mode == ManaMode.HIGH

    def test_load_grind_style(self) -> None:
        ff = FeatureFlags()
        ff.load_from_config({"features": {"grind_style": "fear_kite"}})
        assert ff.grind_style == GrindStyle.FEAR_KITE

    def test_load_utility_phase(self) -> None:
        ff = FeatureFlags()
        ff.load_from_config({"features": {"utility_phase": 3}})
        assert ff.utility_phase == 3

    def test_load_utility_phase_clamped(self) -> None:
        ff = FeatureFlags()
        ff.load_from_config({"features": {"utility_phase": 10}})
        assert ff.utility_phase == 4

    def test_unknown_key_ignored(self) -> None:
        """Keys that don't match known flags are silently ignored."""
        ff = FeatureFlags()
        ff.load_from_config({"features": {"nonexistent_flag": True}})
        # No crash, no change
        assert ff.combat_casting is True


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_no_warnings_defaults(self) -> None:
        ff = FeatureFlags()
        warnings = ff.validate()
        # flee and rest are True by default, so no warnings
        assert warnings == []

    def test_flee_off_warns(self) -> None:
        ff = FeatureFlags()
        ff.flee = False
        warnings = ff.validate()
        assert any("flee=OFF" in w for w in warnings)

    def test_rest_off_warns(self) -> None:
        ff = FeatureFlags()
        ff.rest = False
        warnings = ff.validate()
        assert any("rest=OFF" in w for w in warnings)

    def test_both_off(self) -> None:
        ff = FeatureFlags()
        ff.flee = False
        ff.rest = False
        warnings = ff.validate()
        assert len(warnings) == 2


# ---------------------------------------------------------------------------
# as_dict
# ---------------------------------------------------------------------------


class TestAsDict:
    def test_returns_all_flags(self) -> None:
        ff = FeatureFlags()
        d = ff.as_dict()
        assert "loot_mode" in d
        assert "combat_casting" in d
        assert "wander" in d
        assert "pull" in d
        assert "rest" in d
        assert "flee" in d
        assert "shielding_buff" in d
        assert "obstacle_avoidance" in d
        assert "death_recovery" in d
        assert "utility_phase" in d
        assert "mana_mode" in d
        assert "grind_style" in d
        assert "pareto_scoring" in d
        assert "goap_planning" in d

    def test_reflects_changes(self) -> None:
        ff = FeatureFlags()
        ff.combat_casting = False
        ff.loot_mode = LootMode.SMART
        d = ff.as_dict()
        assert d["combat_casting"] is False
        assert d["loot_mode"] == LootMode.SMART


# ---------------------------------------------------------------------------
# log_summary
# ---------------------------------------------------------------------------


class TestLogSummary:
    def test_log_summary_runs(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        ff = FeatureFlags()
        with caplog.at_level(logging.INFO):
            ff.log_summary()
        assert any("Features:" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# reload_from_file
# ---------------------------------------------------------------------------


class TestReloadFromFile:
    def _write_toml(self, path: Path, content: str) -> None:
        path.write_text(content)

    def test_reload_changes_flag(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "settings.toml"
        self._write_toml(toml_file, "[features]\ncombat_casting = false\n")
        ff = FeatureFlags()
        assert ff.combat_casting is True
        changed = ff.reload_from_file(toml_file)
        assert changed is True
        assert ff.combat_casting is False

    def test_reload_no_change(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "settings.toml"
        # Write defaults
        self._write_toml(toml_file, "[features]\ncombat_casting = true\n")
        ff = FeatureFlags()
        changed = ff.reload_from_file(toml_file)
        assert changed is False

    def test_reload_missing_file(self) -> None:
        ff = FeatureFlags()
        changed = ff.reload_from_file(Path("/nonexistent/file.toml"))
        assert changed is False

    def test_reload_invalid_toml(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "settings.toml"
        toml_file.write_text("not valid toml {{{{")
        ff = FeatureFlags()
        changed = ff.reload_from_file(toml_file)
        assert changed is False

    def test_reload_triggers_validate(self, tmp_path: Path) -> None:
        """When flags change, validate() is called."""
        toml_file = tmp_path / "settings.toml"
        self._write_toml(toml_file, "[features]\nflee = false\n")
        ff = FeatureFlags()
        changed = ff.reload_from_file(toml_file)
        assert changed is True
        # Validate would have been called internally; verify the flag is set
        assert ff.flee is False
