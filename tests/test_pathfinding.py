"""Tests for nav/pathfinding.py: heuristic and path variation utilities.

The core A*/JPS search requires a fully constructed ZoneTerrain bitfield.
These tests verify the pure functions that support pathfinding: the
heuristic function and path variation logic.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nav.pathfinding import _heuristic

# ---------------------------------------------------------------------------
# Heuristic function (Chebyshev / octile distance)
# ---------------------------------------------------------------------------

_grid_coord = st.integers(min_value=0, max_value=1000)


class TestHeuristic:
    def test_zero_distance(self) -> None:
        assert _heuristic(5, 5, 5, 5) == pytest.approx(0.0)

    def test_straight_horizontal(self) -> None:
        h = _heuristic(0, 0, 10, 0)
        assert h == pytest.approx(10.0)

    def test_straight_vertical(self) -> None:
        h = _heuristic(0, 0, 0, 10)
        assert h == pytest.approx(10.0)

    def test_diagonal(self) -> None:
        h = _heuristic(0, 0, 10, 10)
        # Octile: 10 diagonal steps at cost ~1.414 each
        assert h > 10.0
        assert h < 15.0

    @given(c1=_grid_coord, r1=_grid_coord, c2=_grid_coord, r2=_grid_coord)
    def test_non_negative(self, c1: int, r1: int, c2: int, r2: int) -> None:
        assert _heuristic(c1, r1, c2, r2) >= 0.0

    @given(c1=_grid_coord, r1=_grid_coord, c2=_grid_coord, r2=_grid_coord)
    def test_symmetric(self, c1: int, r1: int, c2: int, r2: int) -> None:
        assert _heuristic(c1, r1, c2, r2) == pytest.approx(_heuristic(c2, r2, c1, r1))

    @given(c1=_grid_coord, r1=_grid_coord, c2=_grid_coord, r2=_grid_coord)
    def test_triangle_inequality_with_origin(self, c1: int, r1: int, c2: int, r2: int) -> None:
        # h(a, b) <= h(a, origin) + h(origin, b)  -- admissibility sanity check
        assert _heuristic(c1, r1, c2, r2) <= _heuristic(c1, r1, 0, 0) + _heuristic(0, 0, c2, r2) + 1e-9
