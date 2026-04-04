"""Tick timing and dt tracking."""

from __future__ import annotations

import threading
import time


class TickClock:
    """Regulates the main loop to a target tick rate and tracks delta time."""

    def __init__(self, tick_rate_hz: float = 10.0, stop_event: threading.Event | None = None) -> None:
        self._interval = 1.0 / tick_rate_hz
        self._last_tick = time.perf_counter()
        self.dt: float = 0.0
        self.tick_count: int = 0
        self._stop_event = stop_event

    def wait_for_next_tick(self) -> float:
        """Sleep until the next tick is due.  Returns dt since last tick.

        If a *stop_event* was provided at construction, the sleep is
        interruptible: ``stop_event.wait(timeout)`` is used instead of
        ``time.sleep`` so the thread wakes immediately on shutdown.
        """
        now = time.perf_counter()
        elapsed = now - self._last_tick
        sleep_time = self._interval - elapsed
        if sleep_time > 0:
            if self._stop_event is not None:
                self._stop_event.wait(sleep_time)
            else:
                time.sleep(sleep_time)

        now = time.perf_counter()
        self.dt = now - self._last_tick
        self._last_tick = now
        self.tick_count += 1
        return self.dt
