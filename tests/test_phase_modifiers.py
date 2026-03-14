"""Tests for session phase score modifiers (src/brain/phase_modifiers.py)."""

from __future__ import annotations

from typing import Any

import pytest

from brain.phase_modifiers import (
    _RULE_MODIFIER_MAP,
    PHASE_PROFILES,
    PhaseModifiers,
    get_phase_modifier,
)

# ---------------------------------------------------------------------------
# PhaseModifiers dataclass
# ---------------------------------------------------------------------------


class TestPhaseModifiersDataclass:
    def test_default_values_are_1(self) -> None:
        pm = PhaseModifiers()
        assert pm.acquire_mult == 1.0
        assert pm.rest_mult == 1.0
        assert pm.wander_mult == 1.0
        assert pm.safety_mult == 1.0
        assert pm.combat_mult == 1.0
        assert pm.maintenance_mult == 1.0

    def test_is_frozen(self) -> None:
        pm: Any = PhaseModifiers()
        with pytest.raises(AttributeError):
            pm.acquire_mult = 2.0

    def test_custom_values(self) -> None:
        pm = PhaseModifiers(acquire_mult=0.5, safety_mult=1.5)
        assert pm.acquire_mult == 0.5
        assert pm.safety_mult == 1.5
        assert pm.rest_mult == 1.0  # unchanged


# ---------------------------------------------------------------------------
# PHASE_PROFILES registry
# ---------------------------------------------------------------------------


class TestPhaseProfiles:
    EXPECTED_PHASES = ["startup", "grinding", "resting", "incident", "idle"]

    def test_all_expected_phases_exist(self) -> None:
        for phase in self.EXPECTED_PHASES:
            assert phase in PHASE_PROFILES, f"Missing phase: {phase}"

    def test_profiles_are_phase_modifiers_instances(self) -> None:
        for name, profile in PHASE_PROFILES.items():
            assert isinstance(profile, PhaseModifiers), f"{name} is not PhaseModifiers"

    def test_grinding_is_all_defaults(self) -> None:
        """Grinding phase should have all 1.0 modifiers (steady state)."""
        grinding = PHASE_PROFILES["grinding"]
        assert grinding.acquire_mult == 1.0
        assert grinding.rest_mult == 1.0
        assert grinding.wander_mult == 1.0
        assert grinding.safety_mult == 1.0
        assert grinding.combat_mult == 1.0
        assert grinding.maintenance_mult == 1.0

    def test_startup_is_cautious(self) -> None:
        startup = PHASE_PROFILES["startup"]
        assert startup.acquire_mult < 1.0  # less aggressive
        assert startup.safety_mult > 1.0  # more cautious
        assert startup.wander_mult > 1.0  # more exploratory

    def test_incident_is_very_cautious(self) -> None:
        incident = PHASE_PROFILES["incident"]
        assert incident.acquire_mult < 1.0
        assert incident.safety_mult > 1.0
        assert incident.rest_mult > 1.0

    def test_idle_is_exploratory(self) -> None:
        idle = PHASE_PROFILES["idle"]
        assert idle.wander_mult > 1.0
        assert idle.acquire_mult > 1.0


# ---------------------------------------------------------------------------
# get_phase_modifier()
# ---------------------------------------------------------------------------


class TestGetPhaseModifier:
    def test_unknown_phase_returns_1(self) -> None:
        assert get_phase_modifier("nonexistent_phase", "FLEE") == 1.0

    def test_unknown_rule_returns_1(self) -> None:
        assert get_phase_modifier("startup", "UNKNOWN_RULE") == 1.0

    def test_unknown_phase_and_rule_returns_1(self) -> None:
        assert get_phase_modifier("xxx", "yyy") == 1.0

    def test_grinding_phase_always_returns_1(self) -> None:
        """Grinding is all-default, so every mapped rule returns 1.0."""
        for rule_name in _RULE_MODIFIER_MAP:
            assert get_phase_modifier("grinding", rule_name) == 1.0

    @pytest.mark.parametrize("rule_name", ["FLEE", "EVADE", "FEIGN_DEATH", "DEATH_RECOVERY"])
    def test_survival_rules_map_to_safety(self, rule_name: str) -> None:
        # In incident phase, safety_mult > 1.0
        val = get_phase_modifier("incident", rule_name)
        assert val > 1.0

    @pytest.mark.parametrize("rule_name", ["ACQUIRE", "PULL"])
    def test_acquire_rules_suppressed_in_incident(self, rule_name: str) -> None:
        val = get_phase_modifier("incident", rule_name)
        assert val < 1.0

    @pytest.mark.parametrize("rule_name", ["TRAVEL", "WANDER"])
    def test_wander_rules_boosted_in_idle(self, rule_name: str) -> None:
        val = get_phase_modifier("idle", rule_name)
        assert val > 1.0

    @pytest.mark.parametrize("rule_name", ["IN_COMBAT", "ENGAGE_ADD"])
    def test_combat_rules_in_grinding(self, rule_name: str) -> None:
        # Grinding defaults: combat_mult=1.0
        val = get_phase_modifier("grinding", rule_name)
        assert val == 1.0

    @pytest.mark.parametrize("rule_name", ["MEMORIZE_SPELLS", "SUMMON_PET", "BUFF"])
    def test_maintenance_rules_boosted_in_startup(self, rule_name: str) -> None:
        val = get_phase_modifier("startup", rule_name)
        assert val > 1.0


# ---------------------------------------------------------------------------
# All defined phases are valid
# ---------------------------------------------------------------------------


class TestPhaseValidity:
    def test_all_rule_modifier_fields_exist_on_phase_modifiers(self) -> None:
        """Every field referenced in _RULE_MODIFIER_MAP exists on PhaseModifiers."""
        pm = PhaseModifiers()
        for rule, field_name in _RULE_MODIFIER_MAP.items():
            assert hasattr(pm, field_name), f"Field {field_name} (rule={rule}) not on PhaseModifiers"

    def test_all_profiles_have_valid_modifier_values(self) -> None:
        """All modifier values should be positive floats."""
        for phase, profile in PHASE_PROFILES.items():
            for field_name in _RULE_MODIFIER_MAP.values():
                val = getattr(profile, field_name)
                assert isinstance(val, float), f"{phase}.{field_name} is not float"
                assert val > 0.0, f"{phase}.{field_name} is non-positive"
