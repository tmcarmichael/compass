"""Tests for runtime/agent.py module.

Tests requiring the live EQ client or Windows admin check are marked
with @pytest.mark.live so they can be skipped in CI.
"""

from __future__ import annotations

import sys

import pytest

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

live = pytest.mark.skipif(sys.platform != "win32", reason="Requires Windows platform")


# ---------------------------------------------------------------------------
# ThresholdConfig tests (pure logic, no live process needed)
# ---------------------------------------------------------------------------


class TestThresholdConfig:
    """Test ThresholdConfig dataclass and factory method."""

    def test_defaults(self) -> None:
        from runtime.agent import ThresholdConfig

        tc = ThresholdConfig()
        assert tc.rest_hp_high == 0.92
        assert tc.rest_mana_high == 0.70
        assert tc.rest_hp_low == 0.30
        assert tc.rest_mana_low == 0.20

    def test_from_toml_empty(self) -> None:
        from runtime.agent import ThresholdConfig

        tc = ThresholdConfig.from_toml({})
        assert tc.rest_hp_high == 0.92
        assert tc.rest_mana_high == 0.70
        assert tc.rest_hp_low == 0.30
        assert tc.rest_mana_low == 0.20

    def test_from_toml_partial(self) -> None:
        from runtime.agent import ThresholdConfig

        raw = {"thresholds": {"rest_hp_high": 0.85}}
        tc = ThresholdConfig.from_toml(raw)
        assert tc.rest_hp_high == 0.85
        assert tc.rest_mana_high == 0.70  # default

    def test_from_toml_full(self) -> None:
        from runtime.agent import ThresholdConfig

        raw = {
            "thresholds": {
                "rest_hp_high": 0.80,
                "rest_mana_high": 0.60,
                "rest_hp_low": 0.25,
                "rest_mana_low": 0.15,
            }
        }
        tc = ThresholdConfig.from_toml(raw)
        assert tc.rest_hp_high == 0.80
        assert tc.rest_mana_high == 0.60
        assert tc.rest_hp_low == 0.25
        assert tc.rest_mana_low == 0.15


# ---------------------------------------------------------------------------
# _is_admin tests
# ---------------------------------------------------------------------------


@live
class TestIsAdmin:
    """Test _is_admin() returns a bool without crashing."""

    def test_returns_bool(self) -> None:
        from runtime.agent import _is_admin

        result = _is_admin()
        assert isinstance(result, bool)
