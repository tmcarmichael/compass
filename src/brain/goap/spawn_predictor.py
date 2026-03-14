"""Spawn prediction via Poisson process from defeat timestamps.

Models respawn rate per spatial cell from historical defeat data. The planner
uses predictions to position the agent where targets will appear, converting
random wandering into directed positioning.

Confidence threshold: 3+ defeats per cell before predictions are trusted.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from core.types import Point

if TYPE_CHECKING:
    from brain.learning.spatial import SpatialMemory

log = logging.getLogger(__name__)

# Grid cell size matching SpatialMemory
CELL_SIZE = 50.0
# Minimum defeats per cell to trust predictions
MIN_DEFEATS_FOR_PREDICTION = 3
# Maximum prediction horizon (seconds)
MAX_PREDICTION_HORIZON = 600.0  # 10 minutes


def _cell_key(x: float, y: float) -> tuple[int, int]:
    """Quantize position to grid cell."""
    return (int(x // CELL_SIZE), int(y // CELL_SIZE))


def _cell_center(cx: int, cy: int) -> Point:
    """Return the center point of a grid cell."""
    return Point(
        x=(cx + 0.5) * CELL_SIZE,
        y=(cy + 0.5) * CELL_SIZE,
        z=0.0,
    )


class SpawnPredictor:
    """Predicts entity respawn times from defeat history via Poisson process.

    Usage:
        predictor = SpawnPredictor()
        predictor.update_from_memory(spatial_memory)
        next_spawn = predictor.predict_next(x, y, now)
        best = predictor.best_cells(5, now)
    """

    def __init__(self) -> None:
        # cell_key -> (lambda_rate, last_defeat_time)
        # lambda_rate = defeats_per_second (Poisson rate parameter)
        self._cells: dict[tuple[int, int], tuple[float, float]] = {}

    def update_from_memory(self, memory: SpatialMemory) -> None:
        """Rebuild predictions from spatial memory defeat data.

        Called periodically (every 60s or on zone entry), not every tick.
        """
        self._cells.clear()
        now = time.time()

        # Access the raw defeat records from spatial memory
        defeats = getattr(memory, "_kills", [])
        if not defeats:
            return

        # Group defeats by cell
        cell_times: dict[tuple[int, int], list[float]] = {}
        for record in defeats:
            x = record.get("x", 0.0) if isinstance(record, dict) else getattr(record, "x", 0.0)
            y = record.get("y", 0.0) if isinstance(record, dict) else getattr(record, "y", 0.0)
            t = record.get("time", 0.0) if isinstance(record, dict) else getattr(record, "time", 0.0)
            key = _cell_key(x, y)
            if key not in cell_times:
                cell_times[key] = []
            cell_times[key].append(t)

        # Fit Poisson rate per cell
        for key, times in cell_times.items():
            if len(times) < MIN_DEFEATS_FOR_PREDICTION:
                continue
            times.sort()
            # Observation window: first defeat to now
            window = now - times[0]
            if window < 60:  # need at least 60s of observation
                continue
            # Lambda = defeats / observation_time
            rate = len(times) / window
            last_defeat = times[-1]
            self._cells[key] = (rate, last_defeat)

    def predict_next(self, x: float, y: float, now: float) -> float | None:
        """Seconds until next predicted respawn at this location.

        Returns None if insufficient data for this cell.
        """
        key = _cell_key(x, y)
        cell_data = self._cells.get(key)
        if cell_data is None:
            return None

        rate, last_defeat = cell_data
        if rate <= 0:
            return None

        # Expected inter-arrival time = 1 / lambda
        expected_interval = 1.0 / rate
        elapsed = now - last_defeat
        remaining = expected_interval - elapsed

        if remaining < 0:
            # Overdue -- respawn expected any moment
            return 0.0
        return min(remaining, MAX_PREDICTION_HORIZON)

    def best_cells(self, n: int, now: float) -> list[tuple[Point, float]]:
        """Top N cells by predicted imminent respawn.

        Returns [(position, seconds_until_respawn)] sorted by soonest first.
        Excludes cells with insufficient data.
        """
        predictions: list[tuple[Point, float]] = []

        for key, (rate, last_defeat) in self._cells.items():
            if rate <= 0:
                continue
            expected_interval = 1.0 / rate
            elapsed = now - last_defeat
            remaining = max(0.0, expected_interval - elapsed)
            if remaining <= MAX_PREDICTION_HORIZON:
                center = _cell_center(*key)
                predictions.append((center, remaining))

        # Sort by soonest respawn
        predictions.sort(key=lambda p: p[1])
        return predictions[:n]

    @property
    def cell_count(self) -> int:
        """Number of cells with sufficient data for prediction."""
        return len(self._cells)
