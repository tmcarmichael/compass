"""Tests for core/timing.py: timing variation utilities.

Covers varying_sleep distribution bounds, interruptible_sleep interrupt
behavior and duration bounds, and jittered_value gaussian noise.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.timing import interruptible_sleep, jittered_value, varying_sleep

# ---------------------------------------------------------------------------
# varying_sleep
# ---------------------------------------------------------------------------


class TestVaryingSleep:
    def test_sleeps_at_least_60_pct(self) -> None:
        """The minimum multiplier is 0.6, so sleep >= 0.6 * base."""
        base = 0.05
        start = time.monotonic()
        varying_sleep(base, sigma=0.3)
        elapsed = time.monotonic() - start
        # Allow small timing tolerance
        assert elapsed >= base * 0.6 - 0.01

    def test_completes_in_reasonable_time(self) -> None:
        """With sigma=0.3, sleep should almost never exceed 5x base."""
        base = 0.02
        start = time.monotonic()
        varying_sleep(base, sigma=0.3)
        elapsed = time.monotonic() - start
        assert elapsed < base * 10  # generous upper bound

    @patch("core.timing.random.lognormvariate", return_value=0.3)
    @patch("core.timing.time.sleep")
    def test_clamps_to_min_multiplier(self, mock_sleep, mock_lognorm) -> None:
        """When lognormvariate returns < 0.6, multiplier is clamped to 0.6."""
        varying_sleep(1.0, sigma=0.3)
        mock_sleep.assert_called_once_with(pytest.approx(0.6, abs=0.01))

    @patch("core.timing.random.lognormvariate", return_value=1.5)
    @patch("core.timing.time.sleep")
    def test_applies_multiplier(self, mock_sleep, mock_lognorm) -> None:
        varying_sleep(2.0, sigma=0.3)
        mock_sleep.assert_called_once_with(pytest.approx(3.0, abs=0.01))


# ---------------------------------------------------------------------------
# interruptible_sleep
# ---------------------------------------------------------------------------


class TestInterruptibleSleep:
    def test_returns_false_when_no_interrupt(self) -> None:
        result = interruptible_sleep(0.05, interrupt_fn=None, poll_interval=0.01, sigma=0.0)
        assert result is False

    def test_returns_true_when_interrupted(self) -> None:
        result = interruptible_sleep(5.0, interrupt_fn=lambda: True, poll_interval=0.01, sigma=0.0)
        assert result is True

    def test_interrupt_cuts_sleep_short(self) -> None:
        """When interrupt fires immediately, sleep should end well before base."""
        start = time.monotonic()
        interruptible_sleep(5.0, interrupt_fn=lambda: True, poll_interval=0.01, sigma=0.0)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0  # way less than 5s

    def test_interrupt_fn_exception_ignored(self) -> None:
        """If interrupt_fn raises, it is swallowed and sleep continues."""
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("boom")
            return True  # interrupt on 3rd call

        result = interruptible_sleep(5.0, interrupt_fn=flaky, poll_interval=0.01, sigma=0.0)
        assert result is True

    def test_sigma_zero_gives_exact_base(self) -> None:
        """With sigma=0, lognormvariate(0,0)=1.0, so sleep = base exactly."""
        base = 0.05
        start = time.monotonic()
        interruptible_sleep(base, interrupt_fn=None, poll_interval=0.01, sigma=0.0)
        elapsed = time.monotonic() - start
        assert elapsed >= base - 0.02
        assert elapsed < base + 0.1


# ---------------------------------------------------------------------------
# jittered_value
# ---------------------------------------------------------------------------


class TestJitteredValue:
    def test_zero_sigma_returns_base(self) -> None:
        assert jittered_value(10.0, 0.0) == pytest.approx(10.0)

    @given(base=st.floats(-1000, 1000), sigma=st.floats(0.01, 10))
    @settings(max_examples=50)
    def test_jitter_is_finite(self, base: float, sigma: float) -> None:
        import math

        result = jittered_value(base, sigma)
        assert math.isfinite(result)

    def test_statistical_mean(self) -> None:
        """Over many samples, mean should be close to base."""
        import random

        random.seed(42)
        samples = [jittered_value(100.0, 1.0) for _ in range(10000)]
        mean = sum(samples) / len(samples)
        assert mean == pytest.approx(100.0, abs=0.5)
