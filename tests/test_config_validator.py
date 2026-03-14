"""Tests for core.config_validator -- settings.toml and zone config validation.

Covers validate_settings, validate_zone_config, and log_config_warnings
with valid configs, unknown keys, threshold range violations, and zone edge cases.
"""

from __future__ import annotations

import logging

import pytest

from core.config_validator import (
    log_config_warnings,
    validate_settings,
    validate_zone_config,
)

# ---------------------------------------------------------------------------
# validate_settings
# ---------------------------------------------------------------------------


class TestValidateSettings:
    def test_empty_config_no_warnings(self) -> None:
        assert validate_settings({}) == []

    def test_valid_sections_no_warnings(self) -> None:
        config = {
            "general": {"client_path": "/foo", "tick_rate_hz": 10},
            "thresholds": {"rest_hp_low": 0.3, "flee_hp": 0.1},
            "features": {"loot_mode": "smart", "combat_casting": True},
            "stuck": {"check_seconds": 5},
            "log_levels": {"compass.brain_loop": "DEBUG"},
        }
        assert validate_settings(config) == []

    def test_unknown_section_warns(self) -> None:
        config = {"banana": {"x": 1}}
        warnings = validate_settings(config)
        assert len(warnings) == 1
        assert "unknown section [banana]" in warnings[0]

    def test_unknown_key_in_general_warns(self) -> None:
        config = {"general": {"client_path": "/foo", "typo_key": "oops"}}
        warnings = validate_settings(config)
        assert len(warnings) == 1
        assert "unknown key 'typo_key'" in warnings[0]
        assert "[general]" in warnings[0]

    def test_unknown_key_in_thresholds_warns(self) -> None:
        config = {"thresholds": {"rest_hp_low": 0.3, "fake_threshold": 99}}
        warnings = validate_settings(config)
        assert any("fake_threshold" in w for w in warnings)

    def test_unknown_key_in_features_warns(self) -> None:
        config = {"features": {"looting": True, "nonexistent_flag": False}}
        warnings = validate_settings(config)
        assert any("nonexistent_flag" in w for w in warnings)

    def test_unknown_key_in_stuck_warns(self) -> None:
        config = {"stuck": {"check_seconds": 3, "bad_key": True}}
        warnings = validate_settings(config)
        assert any("bad_key" in w for w in warnings)

    def test_log_levels_section_freeform_no_warnings(self) -> None:
        """log_levels allows arbitrary keys (logger names)."""
        config = {"log_levels": {"compass.brain_loop": "DEBUG", "root": "INFO"}}
        assert validate_settings(config) == []

    def test_threshold_below_zero(self) -> None:
        config = {"thresholds": {"rest_hp_low": -0.1}}
        warnings = validate_settings(config)
        assert any("outside valid range" in w for w in warnings)

    def test_threshold_above_one(self) -> None:
        config = {"thresholds": {"flee_hp": 1.5}}
        warnings = validate_settings(config)
        assert any("outside valid range" in w for w in warnings)

    def test_threshold_at_boundary_valid(self) -> None:
        config = {"thresholds": {"rest_hp_low": 0.0, "rest_hp_high": 1.0}}
        assert validate_settings(config) == []

    def test_multiple_bad_thresholds(self) -> None:
        config = {"thresholds": {"rest_hp_low": -1.0, "rest_mana_low": 2.0, "flee_hp": 5.0}}
        warnings = validate_settings(config)
        assert len(warnings) == 3

    def test_non_dict_section_skipped(self) -> None:
        """If a section value is not a dict, skip key validation."""
        config = {"general": "not a dict"}
        warnings = validate_settings(config)
        # No crash, and no unknown-key warnings for 'general'
        assert all("unknown key" not in w for w in warnings)

    def test_threshold_none_not_validated(self) -> None:
        """Thresholds set to None are not range-checked."""
        config = {"thresholds": {"rest_hp_low": None}}
        assert validate_settings(config) == []

    @pytest.mark.parametrize("val", [0.0, 0.5, 1.0])
    def test_valid_threshold_values(self, val: float) -> None:
        config = {"thresholds": {"rest_hp_low": val}}
        assert validate_settings(config) == []


# ---------------------------------------------------------------------------
# validate_zone_config
# ---------------------------------------------------------------------------


class TestValidateZoneConfig:
    def test_empty_zone_config_no_warnings(self) -> None:
        assert validate_zone_config({}) == []

    def test_valid_zone_sections(self) -> None:
        config = {"zone": {"name": "gfay"}, "camps": [], "waypoints": []}
        assert validate_zone_config(config) == []

    def test_unknown_top_level_key_warns(self) -> None:
        config = {"zone": {}, "bogus_section": {}}
        warnings = validate_zone_config(config)
        assert any("unknown top-level key 'bogus_section'" in w for w in warnings)

    def test_camp_missing_name_warns(self) -> None:
        config = {"camps": [{"center": [100, 200]}]}
        warnings = validate_zone_config(config)
        assert any("missing 'name'" in w for w in warnings)

    def test_camp_missing_center_warns(self) -> None:
        config = {"camps": [{"name": "east_camp"}]}
        warnings = validate_zone_config(config)
        assert any("missing 'center'" in w for w in warnings)

    def test_camp_unknown_key_warns(self) -> None:
        config = {"camps": [{"name": "camp1", "center": [0, 0], "bogus_key": True}]}
        warnings = validate_zone_config(config)
        assert any("unknown key 'bogus_key'" in w for w in warnings)

    def test_valid_camp_no_warnings(self) -> None:
        config = {
            "camps": [
                {
                    "name": "east_camp",
                    "center": [100, 200],
                    "safe_spot": [90, 190],
                    "roam_radius": 100,
                    "mob_names": ["a_skeleton"],
                }
            ]
        }
        assert validate_zone_config(config) == []

    def test_multiple_camps_validated(self) -> None:
        config = {
            "camps": [
                {"name": "camp1", "center": [0, 0]},
                {"name": "camp2"},  # missing center
                {"center": [1, 1]},  # missing name
            ]
        }
        warnings = validate_zone_config(config)
        assert any("camp2" in w and "missing 'center'" in w for w in warnings)
        assert any("missing 'name'" in w for w in warnings)

    def test_camps_not_a_list_skipped(self) -> None:
        """If camps is not a list, skip camp validation without crashing."""
        config = {"camps": "not a list"}
        warnings = validate_zone_config(config)
        # Should not crash and not produce camp warnings
        assert all("camp" not in w.lower() for w in warnings)

    def test_camp_not_a_dict_skipped(self) -> None:
        config = {"camps": ["string_not_dict", {"name": "real", "center": [0, 0]}]}
        warnings = validate_zone_config(config)
        # Only the valid camp is checked; the string element is skipped
        assert all("camp[0]" not in w for w in warnings)

    def test_linear_camp_keys_valid(self) -> None:
        config = {
            "camps": [
                {
                    "name": "patrol",
                    "center": [0, 0],
                    "camp_type": "linear",
                    "patrol_waypoints": [[0, 0], [100, 100]],
                    "corridor_width": 50,
                }
            ]
        }
        assert validate_zone_config(config) == []


# ---------------------------------------------------------------------------
# log_config_warnings
# ---------------------------------------------------------------------------


class TestLogConfigWarnings:
    def test_returns_zero_for_valid_configs(self) -> None:
        config = {"general": {"client_path": "/foo"}}
        zone = {"zone": {"name": "gfay"}, "camps": []}
        assert log_config_warnings(config, zone) == 0

    def test_returns_count_of_warnings(self) -> None:
        config = {"bad_section": {}, "also_bad": {}}
        count = log_config_warnings(config)
        assert count == 2

    def test_zone_config_none_skips_zone_validation(self) -> None:
        config = {"general": {"client_path": "/foo"}}
        assert log_config_warnings(config, None) == 0

    def test_combines_settings_and_zone_warnings(self) -> None:
        config = {"bogus": {}}
        zone = {"invalid_key": {}}
        count = log_config_warnings(config, zone)
        assert count == 2

    def test_logs_warnings(self, caplog: pytest.LogCaptureFixture) -> None:
        config = {"bogus_section": {}}
        with caplog.at_level(logging.WARNING):
            log_config_warnings(config)
        assert any("CONFIG" in r.message for r in caplog.records)

    def test_logs_passed_on_clean(self, caplog: pytest.LogCaptureFixture) -> None:
        config = {"general": {"client_path": "/foo"}}
        with caplog.at_level(logging.INFO):
            log_config_warnings(config)
        assert any("validation passed" in r.message for r in caplog.records)
