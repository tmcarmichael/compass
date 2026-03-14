"""StuckDetector: fast displacement check with instant speed-based detection."""

import math
import time
from collections import deque
from collections.abc import Callable


class StuckDetector:
    """Detect if the character is stuck using displacement + speed.

    Three detection methods:
    1. Speed-based (instant): if game-reported speed is ~0 while we expect
       movement, trigger after 2 consecutive zero-speed ticks.
    2. Position-based (fast): if position unchanged over 3 samples (~0.5s),
       trigger immediately.
    3. Displacement window (backup): if total displacement over check_seconds
       is below threshold, trigger.
    """

    def __init__(
        self,
        check_seconds: float = 1.0,
        min_distance: float = 3.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._check_seconds = check_seconds
        self._min_distance = min_distance
        self._clock = clock

        self._history: deque[tuple[float, float, float]] = deque()  # (time, x, y)
        self._stuck_since: float | None = None
        # Fast detection: track last few positions
        self._recent: deque[tuple[float, float]] = deque(maxlen=4)  # last 4 (x, y)
        self._recent_time: float = 0.0
        # Speed-based detection: consecutive zero-speed ticks
        self._zero_speed_count: int = 0

    def reset(self) -> None:
        """Clear stuck state and history."""
        self._history.clear()
        self._recent.clear()
        self._stuck_since = None
        self._zero_speed_count = 0

    def check(self, x: float, y: float, speed: float = -1.0) -> bool:
        """Record position and return True if stuck.

        Args:
            x: current X position.
            y: current Y position.
            speed: game-reported movement speed (optional). When >= 0,
                   enables fast zero-speed stuck detection. Pass -1.0
                   (default) to skip speed-based checks.
        """
        now = self._clock()
        self._history.append((now, x, y))

        # -- Speed-based fast path: zero speed while we expect movement --
        if speed >= 0 and speed < 0.1:
            self._zero_speed_count += 1
            if self._zero_speed_count >= 2:  # 2 consecutive zero-speed ticks
                if self._stuck_since is None:
                    self._stuck_since = now
                return True
        else:
            self._zero_speed_count = 0

        # -- Fast check: last 3 positions nearly identical --
        self._recent.append((x, y))

        if len(self._recent) >= 3:
            x0, y0 = self._recent[0]
            total_disp = math.hypot(x - x0, y - y0)
            if total_disp < 1.5:
                # Barely moved in 3+ samples (~0.5s)  -  stuck
                if self._stuck_since is None:
                    self._stuck_since = now
                return True

        # -- Window check: displacement over check_seconds --
        cutoff = now - self._check_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if len(self._history) < 2:
            return False

        t0, x0, y0 = self._history[0]
        elapsed = now - t0
        if elapsed < self._check_seconds * 0.4:
            return False

        dist = math.hypot(x - x0, y - y0)
        if dist < self._min_distance:
            if self._stuck_since is None:
                self._stuck_since = now
            return True
        else:
            self._stuck_since = None
            return False

    @property
    def is_stuck(self) -> bool:
        return self._stuck_since is not None
