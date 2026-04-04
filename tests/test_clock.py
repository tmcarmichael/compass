"""Tests for util.clock -- tick timing and dt tracking."""

from __future__ import annotations

import threading
import time

from util.clock import TickClock


class TestTickClockInit:
    def test_default_hz(self) -> None:
        c = TickClock()
        assert c._interval == 0.1  # 10 Hz default

    def test_custom_hz(self) -> None:
        c = TickClock(tick_rate_hz=50.0)
        assert abs(c._interval - 0.02) < 1e-9

    def test_initial_state(self) -> None:
        c = TickClock()
        assert c.dt == 0.0
        assert c.tick_count == 0


class TestWaitForNextTick:
    def test_increments_tick_count(self) -> None:
        c = TickClock(tick_rate_hz=1000.0)  # fast to avoid sleeping
        c.wait_for_next_tick()
        assert c.tick_count == 1
        c.wait_for_next_tick()
        assert c.tick_count == 2

    def test_dt_is_positive(self) -> None:
        c = TickClock(tick_rate_hz=1000.0)
        dt = c.wait_for_next_tick()
        assert dt > 0.0
        assert c.dt == dt

    def test_returns_dt(self) -> None:
        c = TickClock(tick_rate_hz=1000.0)
        result = c.wait_for_next_tick()
        assert isinstance(result, float)
        assert result == c.dt

    def test_stop_event_interrupts_sleep(self) -> None:
        """A slow clock (1 Hz = 1s sleep) should return almost immediately when stop_event fires."""
        stop = threading.Event()
        c = TickClock(tick_rate_hz=1.0, stop_event=stop)
        c.wait_for_next_tick()  # first tick, sets _last_tick

        # Fire stop_event after 50ms -- wait_for_next_tick should unblock well before 1s
        threading.Timer(0.05, stop.set).start()
        t0 = time.perf_counter()
        c.wait_for_next_tick()
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.3, f"expected fast wakeup, took {elapsed:.3f}s"
