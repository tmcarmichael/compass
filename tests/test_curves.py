"""Tests for brain.scoring.curves  -- response curves for utility scoring.

All curves map a value to [0, 1]. Property-based tests verify range and
monotonicity invariants; parametrized tests verify boundary values.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from brain.scoring.curves import (
    bell,
    inverse_linear,
    inverse_logistic,
    linear,
    logistic,
    polynomial,
)

unit_float = st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
positive_float = st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False)

# Ordered pair strategy: generates (lo, hi) with hi - lo > 0.01
_ordered_range = (
    st.tuples(unit_float, unit_float).map(lambda t: (min(t), max(t))).filter(lambda t: t[1] - t[0] > 0.01)
)
# Ordered value pair: generates (v1, v2) with v1 <= v2
_ordered_values = st.tuples(unit_float, unit_float).map(lambda t: (min(t), max(t)))


# ---------------------------------------------------------------------------
# linear / inverse_linear
# ---------------------------------------------------------------------------


class TestLinear:
    @given(v=unit_float, bounds=_ordered_range)
    def test_output_range(self, v: float, bounds: tuple[float, float]) -> None:
        lo, hi = bounds
        assert 0.0 <= linear(v, lo, hi) <= 1.0

    @given(vals=_ordered_values, bounds=_ordered_range)
    def test_monotonic(self, vals: tuple[float, float], bounds: tuple[float, float]) -> None:
        v1, v2 = vals
        lo, hi = bounds
        assert linear(v1, lo, hi) <= linear(v2, lo, hi)

    def test_at_lo(self) -> None:
        assert linear(0.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_at_hi(self) -> None:
        assert linear(1.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_below_lo_clamps(self) -> None:
        assert linear(-5.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_above_hi_clamps(self) -> None:
        assert linear(5.0, 0.0, 1.0) == pytest.approx(1.0)


class TestInverseLinear:
    @given(v=unit_float, bounds=_ordered_range)
    def test_complement_of_linear(self, v: float, bounds: tuple[float, float]) -> None:
        lo, hi = bounds
        assert inverse_linear(v, lo, hi) == pytest.approx(1.0 - linear(v, lo, hi))


# ---------------------------------------------------------------------------
# logistic / inverse_logistic
# ---------------------------------------------------------------------------


class TestLogistic:
    @given(v=unit_float, mid=unit_float)
    def test_output_range(self, v: float, mid: float) -> None:
        assert 0.0 <= logistic(v, mid) <= 1.0

    def test_midpoint_is_half(self) -> None:
        assert logistic(5.0, 5.0) == pytest.approx(0.5)

    @given(vals=_ordered_values, mid=unit_float)
    def test_monotonic(self, vals: tuple[float, float], mid: float) -> None:
        v1, v2 = vals
        assert logistic(v1, mid) <= logistic(v2, mid) + 1e-9

    def test_extreme_value_no_overflow(self) -> None:
        assert logistic(1000.0, 0.0) == pytest.approx(1.0)
        assert logistic(-1000.0, 0.0) == pytest.approx(0.0)


class TestInverseLogistic:
    @given(v=unit_float, mid=unit_float)
    def test_complement_of_logistic(self, v: float, mid: float) -> None:
        assert inverse_logistic(v, mid) == pytest.approx(1.0 - logistic(v, mid))


# ---------------------------------------------------------------------------
# polynomial
# ---------------------------------------------------------------------------


class TestPolynomial:
    @given(
        v=unit_float,
        bounds=_ordered_range,
        exp=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    def test_output_range(self, v: float, bounds: tuple[float, float], exp: float) -> None:
        lo, hi = bounds
        assert 0.0 <= polynomial(v, lo, hi, exp) <= 1.0

    def test_exp_1_equals_linear(self) -> None:
        for v in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert polynomial(v, 0, 1, exp=1.0) == pytest.approx(linear(v, 0, 1))


# ---------------------------------------------------------------------------
# bell
# ---------------------------------------------------------------------------


class TestBell:
    @given(v=unit_float, center=unit_float, width=positive_float)
    def test_output_range(self, v: float, center: float, width: float) -> None:
        assert 0.0 <= bell(v, center, width) <= 1.0

    @given(center=unit_float, width=positive_float)
    def test_peak_at_center(self, center: float, width: float) -> None:
        assert bell(center, center, width) == pytest.approx(1.0)

    def test_far_from_center_near_zero(self) -> None:
        assert bell(100.0, 0.0, 1.0) < 0.01
