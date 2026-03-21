"""Tests for util.incident_reporter -- composite incident reports."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from tests.factories import make_game_state, make_spawn
from util.incident_reporter import IncidentReporter


def _make_ctx(cycle_id: int = 1, defeats: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        diag=SimpleNamespace(
            forensics=SimpleNamespace(snapshot=lambda: []),
            cycle_tracker=SimpleNamespace(cycle_id=cycle_id),
        ),
        defeat_tracker=SimpleNamespace(defeats=defeats),
    )


def _make_buffer(
    n: int = 10,
    hp_start: int = 1000,
    hp_end: int = 0,
    hp_max: int = 1000,
    x_start: float = 0.0,
    y_start: float = 0.0,
    x_end: float = 100.0,
    y_end: float = 0.0,
    target: str = "a_skeleton",
    target_hp_start: int = 100,
    target_hp_end: int = 50,
    mana: int = 300,
) -> list[dict]:
    """Build a synthetic forensics buffer with linear interpolation."""
    buf: list[dict] = []
    for i in range(n):
        frac = i / max(n - 1, 1)
        hp = int(hp_start + (hp_end - hp_start) * frac)
        x = x_start + (x_end - x_start) * frac
        y = y_start + (y_end - y_start) * frac
        tgt_hp = int(target_hp_start + (target_hp_end - target_hp_start) * frac)
        buf.append(
            {
                "tick": i * 10,
                "hp": hp,
                "hp_max": hp_max,
                "mana": mana,
                "x": x,
                "y": y,
                "tgt": target,
                "tgt_hp": tgt_hp,
            }
        )
    return buf


# ---------------------------------------------------------------------------
# _build_report
# ---------------------------------------------------------------------------


class TestBuildReport:
    def test_empty_buffer_returns_defaults(self) -> None:
        r = IncidentReporter()
        state = make_game_state()
        ctx = _make_ctx()
        report = r._build_report("player_death", [], state, ctx)
        assert report["hp_sequence"] == []
        assert report["mob_hp_sequence"] == []
        assert report["flee_distance"] == 0
        assert report["flee_duration_s"] == 0
        assert report["trigger_mob"] == ""
        assert report["guards_nearby"] is False

    def test_hp_sequence_deduplicated(self) -> None:
        r = IncidentReporter()
        # Buffer where HP stays at 1000 for 5 ticks then drops
        buf = [{"tick": i, "hp": 1000 if i < 5 else 500, "hp_max": 1000, "x": 0, "y": 0} for i in range(10)]
        state = make_game_state()
        ctx = _make_ctx()
        report = r._build_report("test", buf, state, ctx)
        assert report["hp_sequence"] == [1.0, 0.5]

    def test_hp_sequence_capped_at_20(self) -> None:
        r = IncidentReporter()
        buf = [{"tick": i, "hp": 1000 - i * 10, "hp_max": 1000, "x": 0, "y": 0} for i in range(50)]
        state = make_game_state()
        ctx = _make_ctx()
        report = r._build_report("test", buf, state, ctx)
        assert len(report["hp_sequence"]) <= 20

    def test_flee_distance_calculated(self) -> None:
        r = IncidentReporter()
        buf = _make_buffer(n=5, x_start=0.0, y_start=0.0, x_end=300.0, y_end=400.0)
        state = make_game_state()
        ctx = _make_ctx()
        report = r._build_report("flee", buf, state, ctx)
        assert report["flee_distance"] == 500  # 3-4-5 triangle

    def test_flee_duration_from_ticks(self) -> None:
        r = IncidentReporter()
        buf = _make_buffer(n=5)  # ticks 0, 10, 20, 30, 40
        state = make_game_state()
        ctx = _make_ctx()
        report = r._build_report("flee", buf, state, ctx)
        assert report["flee_duration_s"] == 4.0  # 40 ticks / 10 tps

    def test_mob_hp_sequence(self) -> None:
        r = IncidentReporter()
        buf = _make_buffer(n=5, target="a_bat", target_hp_start=100, target_hp_end=60)
        state = make_game_state()
        ctx = _make_ctx()
        report = r._build_report("test", buf, state, ctx)
        assert report["trigger_mob"] == "a_bat"
        assert len(report["mob_hp_sequence"]) > 0
        assert report["mob_hp_sequence"][0] == 100

    def test_mana_at_trigger(self) -> None:
        r = IncidentReporter()
        buf = _make_buffer(n=3, mana=250)
        state = make_game_state()
        ctx = _make_ctx()
        report = r._build_report("test", buf, state, ctx)
        assert report["mana_at_trigger"] == 250

    def test_guards_nearby_detected(self) -> None:
        r = IncidentReporter()
        guard = make_spawn(name="Guard_Tylfos", x=10.0, y=10.0)
        state = make_game_state(x=10.0, y=10.0, spawns=(guard,))
        ctx = _make_ctx()
        report = r._build_report("test", _make_buffer(n=2), state, ctx)
        assert report["guards_nearby"] is True

    def test_guards_not_nearby_when_far(self) -> None:
        r = IncidentReporter()
        guard = make_spawn(name="Guard_Tylfos", x=1000.0, y=1000.0)
        state = make_game_state(x=0.0, y=0.0, spawns=(guard,))
        ctx = _make_ctx()
        report = r._build_report("test", _make_buffer(n=2), state, ctx)
        assert report["guards_nearby"] is False


# ---------------------------------------------------------------------------
# _build_summary
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def test_death_summary_basic(self) -> None:
        r = IncidentReporter()
        report = {
            "trigger_mob": "a_skeleton",
            "trigger_reason": "",
            "flee_distance": 10,
            "guards_nearby": False,
        }
        s = r._build_summary("death", report)
        assert "Died fighting a_skeleton" in s

    def test_death_summary_with_flee(self) -> None:
        r = IncidentReporter()
        report = {
            "trigger_mob": "a_bat",
            "trigger_reason": "hp_low",
            "flee_distance": 200,
            "flee_duration_s": 5.0,
            "guards_nearby": False,
        }
        s = r._build_summary("death", report)
        assert "fled 200u" in s
        assert "trigger=hp_low" in s

    def test_death_summary_with_guards(self) -> None:
        r = IncidentReporter()
        report = {
            "trigger_mob": "a_skeleton",
            "trigger_reason": "",
            "flee_distance": 10,
            "guards_nearby": True,
        }
        s = r._build_summary("death", report)
        assert "guards nearby" in s

    def test_flee_summary_with_hp(self) -> None:
        r = IncidentReporter()
        report = {"trigger_mob": "a_bat", "trigger_reason": "low_hp", "hp_sequence": [0.8, 0.4]}
        s = r._build_summary("flee", report)
        assert "Fled from a_bat" in s
        assert "HP" in s


# ---------------------------------------------------------------------------
# Debouncing
# ---------------------------------------------------------------------------


class TestDebounce:
    @patch("util.incident_reporter.log_event")
    def test_death_debounced_within_5s(self, mock_log) -> None:
        r = IncidentReporter()
        state = make_game_state()
        ctx = _make_ctx()
        r.report_death(state, ctx, buffer=[])
        r.report_death(state, ctx, buffer=[])  # within 5s
        assert mock_log.call_count == 1

    @patch("util.incident_reporter.log_event")
    def test_flee_debounced_within_30s(self, mock_log) -> None:
        r = IncidentReporter()
        state = make_game_state()
        ctx = _make_ctx()
        r.report_flee(state, ctx)
        r.report_flee(state, ctx)  # within 30s
        assert mock_log.call_count == 1


# ---------------------------------------------------------------------------
# Full report_death / report_flee
# ---------------------------------------------------------------------------


class TestReportDeath:
    @patch("util.incident_reporter.log_event")
    def test_emits_incident_event(self, mock_log) -> None:
        r = IncidentReporter()
        state = make_game_state()
        ctx = _make_ctx(cycle_id=3, defeats=12)
        r.report_death(state, ctx, source="hp_zero", buffer=_make_buffer(n=5))
        assert mock_log.called
        _, kwargs = mock_log.call_args
        assert kwargs["type"] == "player_death"
        assert kwargs["source"] == "hp_zero"
        assert kwargs["cycle_id"] == 3
        assert kwargs["defeats_before_incident"] == 12


class TestReportFlee:
    @patch("util.incident_reporter.log_event")
    def test_emits_incident_event(self, mock_log) -> None:
        r = IncidentReporter()
        state = make_game_state()
        ctx = _make_ctx(cycle_id=2)
        r.report_flee(state, ctx, trigger_reason="hp_critical")
        assert mock_log.called
        _, kwargs = mock_log.call_args
        assert kwargs["type"] == "flee"
        assert kwargs["trigger_reason"] == "hp_critical"
        assert kwargs["cycle_id"] == 2
