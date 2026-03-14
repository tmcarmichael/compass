"""Defeat tracking: history, recent defeats, last fight info."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from core.types import Point


@dataclass(slots=True, kw_only=True)
class DefeatInfo:
    spawn_id: int = 0
    name: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    time: float = 0.0
    looted: bool = False

    @property
    def pos(self) -> Point:
        return Point(self.x, self.y, self.z)


@dataclass(slots=True, kw_only=True)
class DefeatTracker:
    """Tracks defeats, corpses, and last-fight info for loot matching."""

    defeats: int = 0
    xp_gains: int = 0
    recent_kills: list[tuple[int, float]] = field(default_factory=list)
    defeat_history: list[DefeatInfo] = field(default_factory=list)
    defeat_times: list[float] = field(default_factory=list)

    # Last fight target (for defeat recording when target despawns)
    last_fight_name: str = ""
    last_fight_id: int = 0
    last_fight_x: float = 0.0
    last_fight_y: float = 0.0
    last_fight_level: int = 0

    # Defeat cycle correlation -- monotonically increasing per acquire->loot cycle
    cycle_id: int = 0

    # Defeat cycle timing
    _last_kill_time: float = 0.0
    defeat_cycle_times: list[float] = field(default_factory=list)

    def record_kill(self, spawn_id: int, name: str = "", pos: Point | None = None) -> None:
        now = time.time()
        _pos = pos or Point(0.0, 0.0, 0.0)
        # Dedup
        for k in self.defeat_history:
            if k.spawn_id == spawn_id and now - k.time < 30:
                return
        self.recent_kills.append((spawn_id, now))
        self.defeats += 1
        self.defeat_times.append(now)
        if name:
            self.defeat_history.append(
                DefeatInfo(
                    spawn_id=spawn_id,
                    name=name,
                    x=_pos.x,
                    y=_pos.y,
                    z=_pos.z,
                    time=now,
                )
            )
        # Defeat cycle timing
        if self._last_kill_time > 0:
            cycle = now - self._last_kill_time
            self.defeat_cycle_times.append(cycle)
            if len(self.defeat_cycle_times) > 500:
                self.defeat_cycle_times = self.defeat_cycle_times[-250:]
        self._last_kill_time = now
        # Cleanup
        if len(self.defeat_times) > 500:
            cutoff = now - 3600
            self.defeat_times = [t for t in self.defeat_times if t > cutoff]
        if len(self.defeat_history) > 100:
            self.defeat_history = self.defeat_history[-50:]
        if len(self.recent_kills) > 100:
            self.recent_kills = self.recent_kills[-50:]

    def find_unlootable_kill(
        self, corpse_name: str, pos: Point, corpse_spawn_id: int = 0, max_dist: float = 100.0
    ) -> DefeatInfo | None:
        """Find unlooted defeat matching a corpse."""
        if corpse_spawn_id:
            for defeat in self.defeat_history:
                if defeat.spawn_id == corpse_spawn_id and defeat.looted:
                    return None
        base = corpse_name.split("'s_corpse")[0] if "'s_corpse" in corpse_name else corpse_name
        base = base.rstrip("0123456789")
        for defeat in reversed(self.defeat_history):
            if defeat.looted:
                continue
            defeat_base = defeat.name.rstrip("0123456789")
            if defeat_base != base:
                continue
            if corpse_spawn_id and defeat.spawn_id == corpse_spawn_id:
                return defeat
            if defeat.pos.dist_to(pos) < max_dist:
                return defeat
        return None

    def clear_recent_kills(self) -> None:
        """Remove defeats older than 60s from recent_kills."""
        now = time.time()
        self.recent_kills = [(sid, t) for sid, t in self.recent_kills if now - t < 60]

    def clean_kill_history(self) -> None:
        """Prune old defeats (>5min) from defeat_history."""
        now = time.time()
        self.defeat_history = [k for k in self.defeat_history if now - k.time < 300]

    def defeats_in_window(self, window_seconds: float = 300.0) -> int:
        now = time.time()
        return sum(1 for t in self.defeat_times if now - t < window_seconds)

    def defeat_rate_window(self, window_seconds: float) -> float:
        """Defeats per hour over the given window."""
        count = self.defeats_in_window(window_seconds)
        return count / (window_seconds / 3600)

    def last_kill_age(self) -> float:
        if not self.defeat_times:
            return 9999.0
        return time.time() - self.defeat_times[-1]
