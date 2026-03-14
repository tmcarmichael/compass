"""Session memory: cross-session performance tracking with trend detection.

Persists a compressed summary of each session. On startup, compares to
previous sessions to detect improvement, regression, and stagnation.
This is the survival curve infrastructure -- the proof that the agent
improves over time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from util.log_tiers import EVENT

log = logging.getLogger(__name__)

MAX_SESSIONS = 100
REGRESSION_THRESHOLD = 0.10  # 10% below rolling average = regression


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """Compressed summary of one session for cross-session analysis."""

    timestamp: float = 0.0
    duration_minutes: float = 0.0
    defeats_per_hour: float = 0.0
    deaths: int = 0
    flees: int = 0
    survival_score: int = 0
    overall_score: int = 0
    overall_grade: str = "F"
    goap_completion_pct: float = 0.0
    goap_avg_cost_error: float = 0.0
    weight_drift: float = 0.0
    zone: str = ""


class SessionMemory:
    """Cross-session performance tracking with trend detection."""

    def __init__(self, zone: str = "", data_dir: str = "") -> None:
        self._zone = zone
        self._data_dir = data_dir or str(Path("data") / "memory")
        self._sessions: list[SessionRecord] = []
        self._load()

    def record(self, session: SessionRecord) -> None:
        """Append a session record and persist."""
        self._sessions.append(session)
        if len(self._sessions) > MAX_SESSIONS:
            self._sessions = self._sessions[-MAX_SESSIONS:]
        self._save()
        log.log(
            EVENT,
            "[LIFECYCLE] Session recorded: %.1f defeats/hr, %d deaths, grade %s (session %d for %s)",
            session.defeats_per_hour,
            session.deaths,
            session.overall_grade,
            len(self._sessions),
            self._zone,
        )

    def trend(self, n: int = 5) -> dict[str, float]:
        """Rolling average of last N sessions."""
        recent = self._sessions[-n:] if self._sessions else []
        if not recent:
            return {"defeats_per_hour": 0, "survival_score": 0, "overall_score": 0, "goap_completion_pct": 0}
        return {
            "defeats_per_hour": sum(s.defeats_per_hour for s in recent) / len(recent),
            "survival_score": sum(s.survival_score for s in recent) / len(recent),
            "overall_score": sum(s.overall_score for s in recent) / len(recent),
            "goap_completion_pct": sum(s.goap_completion_pct for s in recent) / len(recent),
        }

    def is_regressing(self) -> bool:
        """True if last session scored >10% below the 5-session rolling average."""
        if len(self._sessions) < 3:
            return False
        last = self._sessions[-1]
        avg = self.trend(5)
        if avg["overall_score"] <= 0:
            return False
        return last.overall_score < avg["overall_score"] * (1 - REGRESSION_THRESHOLD)

    def best_session(self) -> SessionRecord | None:
        """Highest-scoring session for this zone."""
        if not self._sessions:
            return None
        return max(self._sessions, key=lambda s: s.overall_score)

    def sessions_since_improvement(self) -> int:
        """Count of sessions since a new best overall score was set."""
        if not self._sessions:
            return 0
        best_score = max(s.overall_score for s in self._sessions)
        for i in range(len(self._sessions) - 1, -1, -1):
            if self._sessions[i].overall_score >= best_score:
                return len(self._sessions) - 1 - i
        return len(self._sessions)

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    def startup_summary(self) -> str:
        """Human-readable summary for session startup log."""
        n = len(self._sessions)
        if n == 0:
            return f"Session history: first session in {self._zone}"
        t = self.trend(5)
        best = self.best_session()
        lines = [
            f"Session history: {n} sessions in {self._zone}",
            f"  Last 5 avg: {t['defeats_per_hour']:.1f} defeats/hr, "
            f"survival {t['survival_score']:.0f}, "
            f"overall {t['overall_score']:.0f}",
        ]
        if best:
            lines.append(
                f"  Best: {best.defeats_per_hour:.1f} defeats/hr, "
                f"grade {best.overall_grade} "
                f"(session {self._sessions.index(best) + 1})"
            )
        if n >= 5:
            # Compute trend direction
            first_half = self._sessions[: n // 2]
            second_half = self._sessions[n // 2 :]
            first_avg = sum(s.defeats_per_hour for s in first_half) / len(first_half)
            second_avg = sum(s.defeats_per_hour for s in second_half) / len(second_half)
            delta = second_avg - first_avg
            if abs(delta) > 0.5:
                direction = "improving" if delta > 0 else "declining"
                lines.append(f"  Trend: {direction} ({delta:+.1f} defeats/hr)")
            else:
                lines.append("  Trend: stable")
        if self.is_regressing():
            lines.append("  WARNING: last session regressed vs rolling average")
        return "\n".join(lines)

    # -- Persistence -----------------------------------------------------------

    def _path(self) -> str:
        return str(Path(self._data_dir) / f"{self._zone}_sessions.json")

    def _load(self) -> None:
        path = self._path()
        if not Path(path).exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            records = data.get("sessions", [])
            for r in records[-MAX_SESSIONS:]:
                self._sessions.append(
                    SessionRecord(
                        timestamp=r.get("ts", 0),
                        duration_minutes=r.get("dur", 0),
                        defeats_per_hour=r.get("dph", 0),
                        deaths=r.get("deaths", 0),
                        flees=r.get("flees", 0),
                        survival_score=r.get("surv", 0),
                        overall_score=r.get("overall", 0),
                        overall_grade=r.get("grade", "F"),
                        goap_completion_pct=r.get("goap_pct", 0),
                        goap_avg_cost_error=r.get("goap_err", 0),
                        weight_drift=r.get("drift", 0),
                        zone=self._zone,
                    )
                )
            if self._sessions:
                log.info(
                    "[LIFECYCLE] SessionMemory: loaded %d sessions for %s", len(self._sessions), self._zone
                )
        except (OSError, json.JSONDecodeError, TypeError) as e:
            log.warning("[LIFECYCLE] SessionMemory: load failed: %s", e)

    def _save(self) -> None:
        path = self._path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        records = []
        for s in self._sessions[-MAX_SESSIONS:]:
            records.append(
                {
                    "ts": round(s.timestamp, 0),
                    "dur": round(s.duration_minutes, 1),
                    "dph": round(s.defeats_per_hour, 1),
                    "deaths": s.deaths,
                    "flees": s.flees,
                    "surv": s.survival_score,
                    "overall": s.overall_score,
                    "grade": s.overall_grade,
                    "goap_pct": round(s.goap_completion_pct, 1),
                    "goap_err": round(s.goap_avg_cost_error, 1),
                    "drift": round(s.weight_drift, 3),
                }
            )
        try:
            with open(path, "w") as f:
                json.dump({"v": 1, "sessions": records}, f, separators=(",", ":"))
        except OSError as e:
            log.warning("[LIFECYCLE] SessionMemory: save failed: %s", e)
