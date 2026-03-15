"""Tests for SpawnPredictor: Poisson-based respawn prediction from defeat history."""

from __future__ import annotations

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from brain.goap.spawn_predictor import (
    CELL_SIZE,
    MAX_PREDICTION_HORIZON,
    SpawnPredictor,
    _cell_center,
    _cell_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeSpatialMemory:
    """Minimal spatial memory stub with _kills list."""

    def __init__(self, kills: list[dict] | None = None) -> None:
        self._kills = kills or []


def _make_kills(x: float, y: float, count: int, start_time: float, interval: float) -> list[dict]:
    """Generate evenly-spaced defeat records at a position."""
    return [{"x": x, "y": y, "time": start_time + i * interval} for i in range(count)]


# ---------------------------------------------------------------------------
# Cell key and center
# ---------------------------------------------------------------------------


class TestCellKey:
    def test_same_cell_for_nearby_points(self) -> None:
        assert _cell_key(10.0, 10.0) == _cell_key(15.0, 15.0)

    def test_different_cell_for_distant_points(self) -> None:
        assert _cell_key(0.0, 0.0) != _cell_key(200.0, 200.0)

    def test_negative_coords(self) -> None:
        key = _cell_key(-100.0, -100.0)
        assert isinstance(key, tuple)
        assert len(key) == 2

    @given(
        x=st.floats(min_value=-5000, max_value=5000, allow_nan=False, allow_infinity=False),
        y=st.floats(min_value=-5000, max_value=5000, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_cell_key_returns_int_pair(self, x: float, y: float) -> None:
        cx, cy = _cell_key(x, y)
        assert isinstance(cx, int)
        assert isinstance(cy, int)


class TestCellCenter:
    def test_center_of_origin_cell(self) -> None:
        p = _cell_center(0, 0)
        assert p.x == pytest.approx(CELL_SIZE * 0.5)
        assert p.y == pytest.approx(CELL_SIZE * 0.5)

    def test_center_of_offset_cell(self) -> None:
        p = _cell_center(2, 3)
        assert p.x == pytest.approx(2.5 * CELL_SIZE)
        assert p.y == pytest.approx(3.5 * CELL_SIZE)


# ---------------------------------------------------------------------------
# Recording defeats and predicting spawns
# ---------------------------------------------------------------------------


class TestRecordAndPredict:
    def test_no_prediction_with_insufficient_data(self) -> None:
        """Fewer than MIN_DEFEATS_FOR_PREDICTION defeats returns None."""
        predictor = SpawnPredictor()
        kills = _make_kills(100.0, 100.0, count=2, start_time=time.time() - 300, interval=60)
        mem = FakeSpatialMemory(kills)
        predictor.update_from_memory(mem)

        result = predictor.predict_next(100.0, 100.0, time.time())
        assert result is None

    def test_prediction_with_sufficient_data(self) -> None:
        """With enough defeats, predict_next returns a non-None float."""
        now = time.time()
        predictor = SpawnPredictor()
        kills = _make_kills(100.0, 100.0, count=5, start_time=now - 600, interval=100)
        mem = FakeSpatialMemory(kills)
        predictor.update_from_memory(mem)

        result = predictor.predict_next(100.0, 100.0, now)
        assert result is not None
        assert isinstance(result, float)
        assert result >= 0.0

    def test_prediction_in_different_cell_returns_none(self) -> None:
        """Prediction for a cell with no data returns None."""
        now = time.time()
        predictor = SpawnPredictor()
        kills = _make_kills(100.0, 100.0, count=5, start_time=now - 600, interval=100)
        mem = FakeSpatialMemory(kills)
        predictor.update_from_memory(mem)

        # Far away cell
        result = predictor.predict_next(5000.0, 5000.0, now)
        assert result is None

    def test_overdue_spawn_returns_zero(self) -> None:
        """When elapsed > expected_interval, prediction returns 0.0."""
        now = time.time()
        predictor = SpawnPredictor()
        # 5 kills over 500s, interval ~100s each. last kill at now-100.
        # Rate = 5/500 = 0.01/s, expected_interval=100s.
        # If we query at now (100s after last kill), it should be overdue.
        kills = _make_kills(100.0, 100.0, count=5, start_time=now - 500, interval=100)
        mem = FakeSpatialMemory(kills)
        predictor.update_from_memory(mem)

        # Now is 500s - (start + 500) = at the last kill. Advance past interval.
        result = predictor.predict_next(100.0, 100.0, now + 200)
        assert result == 0.0

    def test_prediction_capped_at_max_horizon(self) -> None:
        """Prediction is capped at MAX_PREDICTION_HORIZON."""
        now = time.time()
        predictor = SpawnPredictor()
        # Very sparse kills: 3 kills over 600s = rate of 0.005/s
        # expected_interval = 200s. If last kill just happened, remaining = 200s.
        # With only 3 kills at threshold, the prediction should be <= MAX_PREDICTION_HORIZON.
        kills = _make_kills(100.0, 100.0, count=3, start_time=now - 600, interval=200)
        mem = FakeSpatialMemory(kills)
        predictor.update_from_memory(mem)

        result = predictor.predict_next(100.0, 100.0, now)
        if result is not None:
            assert result <= MAX_PREDICTION_HORIZON


# ---------------------------------------------------------------------------
# Decay / observation window
# ---------------------------------------------------------------------------


class TestDecayBehavior:
    def test_short_observation_window_rejected(self) -> None:
        """Defeats within 60s window are rejected (insufficient observation)."""
        now = time.time()
        kills = _make_kills(100.0, 100.0, count=5, start_time=now - 30, interval=5)
        mem = FakeSpatialMemory(kills)
        predictor = SpawnPredictor()
        predictor.update_from_memory(mem)
        assert predictor.cell_count == 0

    def test_long_observation_window_accepted(self) -> None:
        now = time.time()
        kills = _make_kills(100.0, 100.0, count=5, start_time=now - 600, interval=100)
        mem = FakeSpatialMemory(kills)
        predictor = SpawnPredictor()
        predictor.update_from_memory(mem)
        assert predictor.cell_count == 1

    def test_update_clears_previous_data(self) -> None:
        """Calling update_from_memory replaces old cell data."""
        now = time.time()
        predictor = SpawnPredictor()

        # First update: 5 kills at (100, 100)
        kills1 = _make_kills(100.0, 100.0, count=5, start_time=now - 600, interval=100)
        predictor.update_from_memory(FakeSpatialMemory(kills1))
        assert predictor.cell_count == 1

        # Second update: empty kills
        predictor.update_from_memory(FakeSpatialMemory([]))
        assert predictor.cell_count == 0


# ---------------------------------------------------------------------------
# Best cells selection
# ---------------------------------------------------------------------------


class TestBestCells:
    def test_best_cells_returns_sorted_by_soonest(self) -> None:
        now = time.time()
        predictor = SpawnPredictor()

        # Cell A: 5 kills, last kill long ago (overdue)
        kills_a = _make_kills(100.0, 100.0, count=5, start_time=now - 1000, interval=100)
        # Cell B: 5 kills, last kill recent (not yet due)
        kills_b = _make_kills(500.0, 500.0, count=5, start_time=now - 300, interval=50)

        mem = FakeSpatialMemory(kills_a + kills_b)
        predictor.update_from_memory(mem)

        cells = predictor.best_cells(5, now)
        assert len(cells) >= 1
        # Should be sorted by soonest first
        times = [c[1] for c in cells]
        assert times == sorted(times)

    def test_best_cells_limits_n(self) -> None:
        now = time.time()
        predictor = SpawnPredictor()

        all_kills = []
        for i in range(10):
            all_kills.extend(
                _make_kills(
                    i * 200.0,
                    i * 200.0,
                    count=5,
                    start_time=now - 600,
                    interval=100,
                )
            )

        mem = FakeSpatialMemory(all_kills)
        predictor.update_from_memory(mem)

        cells = predictor.best_cells(3, now)
        assert len(cells) <= 3

    def test_best_cells_empty_when_no_data(self) -> None:
        predictor = SpawnPredictor()
        cells = predictor.best_cells(5, time.time())
        assert cells == []

    def test_cell_count_property(self) -> None:
        predictor = SpawnPredictor()
        assert predictor.cell_count == 0
