"""Tests for nav.stuck -- displacement and speed-based stuck detection.

StuckDetector uses three methods: zero-speed ticks, position stagnation over
recent samples, and displacement over a time window. Time is controlled via
constructor-injected clock.
"""

from __future__ import annotations

from nav.stuck import StuckDetector


class TestStuckDetector:
    def test_initial_not_stuck(self) -> None:
        sd = StuckDetector()
        assert sd.is_stuck is False

    def test_stationary_becomes_stuck(self) -> None:
        """Same position over 3+ samples triggers position-based stuck."""
        t = [100.0]
        sd = StuckDetector(clock=lambda: t[0])
        # 3 identical positions -> stuck via recent-position check
        assert sd.check(100.0, 100.0) is False
        t[0] = 100.1
        assert sd.check(100.0, 100.0) is False
        t[0] = 100.2
        result = sd.check(100.0, 100.0)
        assert result is True

    def test_moving_clears_stuck(self) -> None:
        t = [100.0]
        sd = StuckDetector(clock=lambda: t[0])
        # Become stuck via 4 identical positions
        for i in range(4):
            t[0] = 100.0 + i * 0.1
            sd.check(100.0, 100.0)
        assert sd.is_stuck is True
        # Move far enough to clear
        t[0] = 102.0
        result = sd.check(200.0, 200.0)
        assert result is False

    def test_reset_clears_state(self) -> None:
        t = [100.0]
        sd = StuckDetector(clock=lambda: t[0])
        for i in range(4):
            t[0] = 100.0 + i * 0.1
            sd.check(100.0, 100.0)
        assert sd.is_stuck is True
        sd.reset()
        assert sd.is_stuck is False

    def test_zero_speed_detection(self) -> None:
        """Two consecutive zero-speed ticks trigger stuck."""
        t = [100.0]
        sd = StuckDetector(clock=lambda: t[0])
        sd.check(100.0, 100.0, speed=0.0)
        t[0] = 100.1
        result = sd.check(100.0, 100.0, speed=0.0)
        assert result is True

    def test_negative_speed_skips_speed_check(self) -> None:
        """speed=-1.0 (default) disables speed-based detection."""
        t = [100.0]
        sd = StuckDetector(clock=lambda: t[0])
        # Single check with default speed should not trigger
        result = sd.check(100.0, 100.0, speed=-1.0)
        assert result is False

    def test_is_stuck_property(self) -> None:
        t = [100.0]
        sd = StuckDetector(clock=lambda: t[0])
        assert sd.is_stuck is False
        # Trigger zero-speed stuck
        sd.check(50.0, 50.0, speed=0.0)
        t[0] = 100.1
        sd.check(50.0, 50.0, speed=0.0)
        assert sd.is_stuck is True

    def test_slow_movement_below_threshold(self) -> None:
        """Displacement below min_distance over check window is stuck."""
        t = [0.0]
        sd = StuckDetector(check_seconds=1.0, min_distance=3.0, clock=lambda: t[0])
        # Build enough samples within the window for position-based detection
        # 3 nearly-identical positions -> position-based stuck triggers
        positions = [(100.0, 100.0), (100.2, 100.2), (100.4, 100.4), (100.5, 100.5)]
        result = False
        for i, (x, y) in enumerate(positions):
            t[0] = 100.0 + i * 0.2
            result = sd.check(x, y)
        # Total displacement from first to last < 1.5 -> recent-position stuck
        assert result is True

    def test_speed_nonzero_resets_counter(self) -> None:
        """Non-zero speed resets the consecutive zero-speed counter."""
        t = [100.0]
        sd = StuckDetector(clock=lambda: t[0])
        sd.check(100.0, 100.0, speed=0.0)
        t[0] = 100.1
        # Inject a non-zero speed tick to reset the counter
        sd.check(120.0, 120.0, speed=5.0)
        t[0] = 100.2
        # Next zero-speed tick is only the first again, so not stuck via speed
        sd.check(140.0, 140.0, speed=0.0)
        # Not stuck: only 1 consecutive zero-speed tick
        assert sd._zero_speed_count == 1
