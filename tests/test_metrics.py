"""Tests for util.metrics -- percentile tracking, success rates, and aggregation.

PercentileTracker maintains a rolling sorted window. SuccessRateTracker counts
outcomes. MetricsEngine ties them together for session reporting.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from util.metrics import MetricsEngine, PercentileTracker, SuccessRateTracker

# ---------------------------------------------------------------------------
# PercentileTracker
# ---------------------------------------------------------------------------


class TestPercentileTracker:
    def test_empty_tracker_returns_zero(self) -> None:
        pt = PercentileTracker()
        assert pt.p50() == 0.0

    def test_single_value(self) -> None:
        pt = PercentileTracker()
        pt.add(42.0)
        assert pt.p50() == 42.0

    @pytest.mark.parametrize(
        "pct, expected",
        [(0.5, 50), (0.95, 95), (0.99, 99)],
    )
    def test_known_percentiles(self, pct: float, expected: int) -> None:
        pt = PercentileTracker(window=200)
        for v in range(1, 101):
            pt.add(float(v))
        assert pt.percentile(pct) == pytest.approx(expected, abs=1)

    def test_window_overflow(self) -> None:
        pt = PercentileTracker(window=10)
        for v in range(100):
            pt.add(float(v))
        assert pt.count == 10
        # Only the last 10 values (90..99) remain
        assert pt.p50() >= 90.0

    @given(
        values=st.lists(
            st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100,
        ),
    )
    def test_percentile_monotonic_with_pct(self, values: list[float]) -> None:
        """percentile(p) is monotonically non-decreasing as p increases."""
        pt = PercentileTracker(window=200)
        for v in values:
            pt.add(v)
        prev = pt.percentile(0.0)
        for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0):
            cur = pt.percentile(p)
            assert cur >= prev - 1e-9
            prev = cur

    @given(
        values=st.lists(
            st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100,
        ),
    )
    def test_percentile_within_range(self, values: list[float]) -> None:
        """Any percentile is in [min_added, max_added]."""
        pt = PercentileTracker(window=200)
        for v in values:
            pt.add(v)
        lo, hi = min(values), max(values)
        for p in (0.0, 0.25, 0.5, 0.75, 1.0):
            val = pt.percentile(p)
            assert lo <= val <= hi

    def test_as_dict_keys(self) -> None:
        pt = PercentileTracker()
        d = pt.as_dict()
        assert set(d.keys()) == {"p50", "p95", "p99", "count"}


# ---------------------------------------------------------------------------
# SuccessRateTracker
# ---------------------------------------------------------------------------


class TestSuccessRateTracker:
    def test_empty_rate_returns_one(self) -> None:
        """Empty tracker returns 1.0 (documented default for no data)."""
        sr = SuccessRateTracker()
        assert sr.rate() == pytest.approx(1.0)

    def test_all_success_rate_100(self) -> None:
        sr = SuccessRateTracker()
        for _ in range(10):
            sr.record(True)
        assert sr.rate() == pytest.approx(1.0)

    def test_all_failure_rate_zero(self) -> None:
        sr = SuccessRateTracker()
        for _ in range(10):
            sr.record(False)
        assert sr.rate() == pytest.approx(0.0)

    def test_mixed_rate(self) -> None:
        sr = SuccessRateTracker()
        for _ in range(7):
            sr.record(True)
        for _ in range(3):
            sr.record(False)
        assert sr.rate() == pytest.approx(0.7)

    def test_window_rolling(self) -> None:
        sr = SuccessRateTracker(window=5)
        for _ in range(5):
            sr.record(False)
        # Now push 5 successes -- all failures should be evicted
        for _ in range(5):
            sr.record(True)
        assert sr.rate() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MetricsEngine
# ---------------------------------------------------------------------------


class TestMetricsEngine:
    def test_record_tick_duration(self) -> None:
        eng = MetricsEngine()
        eng.record_tick(15.0)
        eng.record_tick(25.0)
        assert eng.tick_duration.count == 2

    def test_record_routine_outcome(self) -> None:
        eng = MetricsEngine()
        eng.record_routine_outcome("REST", True)
        eng.record_routine_outcome("REST", False)
        assert eng._routine_success["REST"].count == 2
        assert eng._routine_success["REST"].rate() == pytest.approx(0.5)

    def test_record_action(self) -> None:
        eng = MetricsEngine()
        eng.record_action("pull", True)
        eng.record_action("pull", True)
        eng.record_action("pull", False)
        assert eng._action_success["pull"].rate() == pytest.approx(2 / 3, abs=0.01)

    def test_summary_format(self) -> None:
        eng = MetricsEngine()
        eng.record_tick(10.0)
        eng.record_routine_outcome("REST", True)
        eng.record_action("loot", True)
        s = eng.summary()
        assert set(s.keys()) == {"tick_duration", "routine_success", "action_success"}
        routine_success = s["routine_success"]
        action_success = s["action_success"]
        assert isinstance(routine_success, dict)
        assert isinstance(action_success, dict)
        assert "REST" in routine_success
        assert "loot" in action_success


# ---------------------------------------------------------------------------
# SessionMetrics -- brain/state/metrics.py
# ---------------------------------------------------------------------------


class TestSessionMetrics:
    def test_trim_lists_short_lists_unchanged(self) -> None:
        from brain.state.metrics import SessionMetrics

        m = SessionMetrics()
        m.pull_distances = [1.0] * 10
        m.trim_lists()
        assert len(m.pull_distances) == 10

    def test_trim_lists_caps_pull_distances(self) -> None:
        from brain.state.metrics import SessionMetrics

        m = SessionMetrics()
        m.pull_distances = [1.0] * 600
        m.trim_lists()
        assert len(m.pull_distances) == 250

    def test_trim_lists_caps_pull_engage_times(self) -> None:
        from brain.state.metrics import SessionMetrics

        m = SessionMetrics()
        m.pull_engage_times = [1.0] * 600
        m.trim_lists()
        assert len(m.pull_engage_times) == 250

    def test_trim_lists_caps_total_cycle_times(self) -> None:
        from brain.state.metrics import SessionMetrics

        m = SessionMetrics()
        m.total_cycle_times = [1.0] * 600
        m.trim_lists()
        assert len(m.total_cycle_times) == 250

    def test_trim_lists_caps_acquire_tab_totals(self) -> None:
        from brain.state.metrics import SessionMetrics

        m = SessionMetrics()
        m.acquire_tab_totals = [1] * 600
        m.trim_lists()
        assert len(m.acquire_tab_totals) == 250

    def test_trim_lists_caps_xp_history(self) -> None:
        from brain.state.metrics import SessionMetrics

        m = SessionMetrics()
        m.xp_history = [(float(i), i) for i in range(600)]
        m.trim_lists()
        assert len(m.xp_history) == 250

    def test_snapshot_collections_returns_copies(self) -> None:
        from brain.state.metrics import SessionMetrics

        m = SessionMetrics()
        m.routine_time["REST"] = 10.5
        m.routine_counts["REST"] = 5
        snap = m.snapshot_collections()
        # Mutating the snapshot shouldn't affect the original
        snap["routine_time"]["REST"] = 999
        assert m.routine_time["REST"] == 10.5
