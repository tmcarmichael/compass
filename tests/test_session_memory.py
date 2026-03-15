"""Tests for brain.learning.session_memory -- cross-session performance tracking.

Covers SessionRecord creation, SessionMemory recording, trend calculation,
regression detection, persistence to disk (tmp_path), and startup summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from brain.learning.session_memory import (
    MAX_SESSIONS,
    SessionMemory,
    SessionRecord,
)


def _record(
    overall_score: int = 50,
    defeats_per_hour: float = 10.0,
    deaths: int = 0,
    survival_score: int = 50,
    overall_grade: str = "C",
    goap_completion_pct: float = 80.0,
    **kw: Any,
) -> SessionRecord:
    return SessionRecord(
        overall_score=overall_score,
        defeats_per_hour=defeats_per_hour,
        deaths=deaths,
        survival_score=survival_score,
        overall_grade=overall_grade,
        goap_completion_pct=goap_completion_pct,
        **kw,
    )


# ---------------------------------------------------------------------------
# SessionRecord
# ---------------------------------------------------------------------------


class TestSessionRecord:
    def test_defaults(self) -> None:
        r = SessionRecord()
        assert r.timestamp == 0.0
        assert r.overall_grade == "F"
        assert r.zone == ""

    def test_frozen(self) -> None:
        r: Any = SessionRecord(overall_score=100)
        with pytest.raises(AttributeError):
            r.overall_score = 200


# ---------------------------------------------------------------------------
# SessionMemory -- basic operations
# ---------------------------------------------------------------------------


class TestSessionMemory:
    def test_empty_memory(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        assert mem.session_count == 0
        assert mem.best_session() is None
        assert mem.sessions_since_improvement() == 0

    def test_record_and_count(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        mem.record(_record(overall_score=50))
        assert mem.session_count == 1
        mem.record(_record(overall_score=60))
        assert mem.session_count == 2

    def test_best_session(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        mem.record(_record(overall_score=30))
        mem.record(_record(overall_score=90))
        mem.record(_record(overall_score=50))
        best = mem.best_session()
        assert best is not None
        assert best.overall_score == 90

    def test_max_sessions_cap(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        for i in range(MAX_SESSIONS + 10):
            mem.record(_record(overall_score=i))
        assert mem.session_count == MAX_SESSIONS


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


class TestTrend:
    def test_empty_trend(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        t = mem.trend()
        assert t["defeats_per_hour"] == 0
        assert t["overall_score"] == 0

    def test_trend_averages_last_n(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        for score in [10, 20, 30, 40, 50]:
            mem.record(_record(overall_score=score))
        t = mem.trend(n=3)
        # Last 3: 30, 40, 50 -> avg = 40
        assert t["overall_score"] == pytest.approx(40.0)

    def test_trend_with_fewer_sessions(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        mem.record(_record(defeats_per_hour=12.0))
        t = mem.trend(n=5)
        assert t["defeats_per_hour"] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


class TestRegression:
    def test_not_regressing_with_few_sessions(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        mem.record(_record(overall_score=50))
        mem.record(_record(overall_score=10))
        assert not mem.is_regressing()

    def test_not_regressing_when_score_is_close(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        for _ in range(5):
            mem.record(_record(overall_score=100))
        assert not mem.is_regressing()

    def test_regressing_when_last_drops(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        for _ in range(5):
            mem.record(_record(overall_score=100))
        # Last session drops to below 90% of rolling avg (100)
        # Threshold is 10%, so score < 90 triggers regression
        mem.record(_record(overall_score=80))
        assert mem.is_regressing()

    def test_not_regressing_with_zero_average(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        for _ in range(5):
            mem.record(_record(overall_score=0))
        assert not mem.is_regressing()


# ---------------------------------------------------------------------------
# sessions_since_improvement
# ---------------------------------------------------------------------------


class TestSessionsSinceImprovement:
    def test_just_set_best(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        mem.record(_record(overall_score=50))
        mem.record(_record(overall_score=100))
        assert mem.sessions_since_improvement() == 0

    def test_stagnation(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        mem.record(_record(overall_score=100))
        mem.record(_record(overall_score=50))
        mem.record(_record(overall_score=60))
        # Best is at index 0, current is index 2 -> 2 sessions since
        assert mem.sessions_since_improvement() == 2


# ---------------------------------------------------------------------------
# Persistence (tmp_path)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_reload(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        mem.record(_record(overall_score=42, defeats_per_hour=8.5, deaths=1))
        mem.record(_record(overall_score=55, defeats_per_hour=12.0, deaths=0))

        # Reload from same directory
        mem2 = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        assert mem2.session_count == 2
        best = mem2.best_session()
        assert best is not None
        assert best.overall_score == 55

    def test_file_structure(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="testzone", data_dir=str(tmp_path))
        mem.record(_record())
        path = tmp_path / "testzone_sessions.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["v"] == 1
        assert len(data["sessions"]) == 1

    def test_corrupted_file_handled(self, tmp_path: Path) -> None:
        path = tmp_path / "badzone_sessions.json"
        path.write_text("NOT VALID JSON")
        mem = SessionMemory(zone="badzone", data_dir=str(tmp_path))
        assert mem.session_count == 0

    def test_missing_file_starts_empty(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="nofile", data_dir=str(tmp_path))
        assert mem.session_count == 0


# ---------------------------------------------------------------------------
# Startup summary
# ---------------------------------------------------------------------------


class TestStartupSummary:
    def test_first_session_summary(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        s = mem.startup_summary()
        assert "first session" in s
        assert "gfay" in s

    def test_summary_with_sessions(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        for i in range(6):
            mem.record(_record(overall_score=50 + i * 5, defeats_per_hour=10.0 + i))
        s = mem.startup_summary()
        assert "6 sessions" in s
        assert "Best" in s
        # With 6 sessions, trend info should appear
        assert "Trend" in s

    def test_summary_shows_regression_warning(self, tmp_path: Path) -> None:
        mem = SessionMemory(zone="gfay", data_dir=str(tmp_path))
        for _ in range(5):
            mem.record(_record(overall_score=100))
        mem.record(_record(overall_score=50))
        s = mem.startup_summary()
        assert "regressed" in s.lower()
