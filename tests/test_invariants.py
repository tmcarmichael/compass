"""Tests for util.invariants -- runtime invariant checker.

Covers InvariantChecker registration, check dispatch at correct tick
intervals, violation counting, rate limiting (cooldown), summary format,
and resilience to exceptions in check functions.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from tests.factories import make_game_state
from util.invariants import InvariantChecker


def _make_stub_ctx() -> MagicMock:
    """Minimal AgentContext stub with attributes the checker accesses."""
    ctx = MagicMock()
    ctx.combat.engaged = False
    ctx.combat.pull_target_id = 0
    ctx.diag = None
    return ctx


# ---------------------------------------------------------------------------
# Basic registration and check
# ---------------------------------------------------------------------------


class TestInvariantCheckerBasics:
    def test_empty_checker_zero_violations(self) -> None:
        ic = InvariantChecker()
        assert ic.violation_count == 0

    def test_passing_check_no_violations(self) -> None:
        t = [1000.0]
        ic = InvariantChecker(clock=lambda: t[0])
        ic.register("always_ok", lambda state, ctx: None, every_n_ticks=1)
        state = make_game_state()
        ctx = _make_stub_ctx()
        ic.check(0, state, ctx)
        assert ic.violation_count == 0

    def test_failing_check_records_violation(self) -> None:
        t = [1000.0]
        ic = InvariantChecker(clock=lambda: t[0])
        ic.register("always_fail", lambda state, ctx: "something broke", every_n_ticks=1)
        state = make_game_state()
        ctx = _make_stub_ctx()
        ic.check(0, state, ctx)
        assert ic.violation_count == 1


# ---------------------------------------------------------------------------
# Tick interval dispatch
# ---------------------------------------------------------------------------


class TestInvariantTickInterval:
    def test_check_fires_at_interval(self) -> None:
        t = [1000.0]
        ic = InvariantChecker(clock=lambda: t[0])
        call_count = 0

        def counting_check(state: object, ctx: object) -> str | None:
            nonlocal call_count
            call_count += 1
            return None

        ic.register("counter", counting_check, every_n_ticks=5)
        state = make_game_state()
        ctx = _make_stub_ctx()

        for tick in range(20):
            ic.check(tick, state, ctx)

        # Ticks 0, 5, 10, 15 are divisible by 5 => 4 calls
        assert call_count == 4

    def test_check_skips_non_matching_ticks(self) -> None:
        t = [1000.0]
        ic = InvariantChecker(clock=lambda: t[0])
        ic.register("every10", lambda s, c: "fail", every_n_ticks=10)
        state = make_game_state()
        ctx = _make_stub_ctx()

        # Tick 7 is not divisible by 10
        ic.check(7, state, ctx)
        assert ic.violation_count == 0


# ---------------------------------------------------------------------------
# Rate limiting (cooldown)
# ---------------------------------------------------------------------------


class TestInvariantCooldown:
    def test_same_invariant_rate_limited(self) -> None:
        t = [1000.0]
        ic = InvariantChecker(clock=lambda: t[0])
        ic.register("flaky", lambda s, c: "broken", every_n_ticks=1)
        state = make_game_state()
        ctx = _make_stub_ctx()

        # First violation at tick 0
        ic.check(0, state, ctx)
        assert ic.violation_count == 1

        # Second violation at tick 1 -- within 30s cooldown
        t[0] = 1001.0
        ic.check(1, state, ctx)
        assert ic.violation_count == 1  # still 1, rate-limited

    def test_fires_again_after_cooldown(self) -> None:
        t = [1000.0]
        ic = InvariantChecker(clock=lambda: t[0])
        ic.register("flaky", lambda s, c: "broken", every_n_ticks=1)
        state = make_game_state()
        ctx = _make_stub_ctx()

        ic.check(0, state, ctx)
        assert ic.violation_count == 1

        # Advance past cooldown (default 30s)
        t[0] = 1031.0
        ic.check(1, state, ctx)
        assert ic.violation_count == 2


# ---------------------------------------------------------------------------
# Exception resilience
# ---------------------------------------------------------------------------


class TestInvariantExceptionHandling:
    def test_exception_in_check_fn_doesnt_crash(self) -> None:
        t = [1000.0]
        ic = InvariantChecker(clock=lambda: t[0])

        def explosive(state: object, ctx: object) -> str | None:
            raise RuntimeError("kaboom")

        ic.register("bomb", explosive, every_n_ticks=1)
        state = make_game_state()
        ctx = _make_stub_ctx()

        # Should not raise
        ic.check(0, state, ctx)
        assert ic.violation_count == 0

    def test_exception_doesnt_block_other_checks(self) -> None:
        t = [1000.0]
        ic = InvariantChecker(clock=lambda: t[0])

        ic.register("bomb", lambda s, c: (_ for _ in ()).throw(ValueError("boom")), every_n_ticks=1)
        ic.register("ok_fail", lambda s, c: "valid failure", every_n_ticks=1)
        state = make_game_state()
        ctx = _make_stub_ctx()

        ic.check(0, state, ctx)
        # The second check should still fire and record its violation
        assert ic.violation_count == 1


# ---------------------------------------------------------------------------
# Summary format
# ---------------------------------------------------------------------------


class TestInvariantSummary:
    def test_empty_summary(self) -> None:
        ic = InvariantChecker()
        s = ic.summary()
        assert s["total"] == 0
        assert s["by_category"] == {}
        assert s["by_name"] == {}

    def test_summary_with_violations(self) -> None:
        t = [1000.0]
        ic = InvariantChecker(clock=lambda: t[0])
        ic.register("state_check", lambda s, c: "bad state", every_n_ticks=1, category="state")
        ic.register("timing_check", lambda s, c: "slow tick", every_n_ticks=1, category="timing")
        state = make_game_state()
        ctx = _make_stub_ctx()

        ic.check(0, state, ctx)
        s = ic.summary()
        assert s["total"] == 2
        assert s["by_category"]["state"] == 1
        assert s["by_category"]["timing"] == 1
        assert s["by_name"]["state_check"] == 1
        assert s["by_name"]["timing_check"] == 1


# ---------------------------------------------------------------------------
# Violation count property
# ---------------------------------------------------------------------------


class TestViolationCount:
    def test_increments_per_unique_fire(self) -> None:
        t = [1000.0]
        ic = InvariantChecker(clock=lambda: t[0])
        ic.register("a", lambda s, c: "fail_a", every_n_ticks=1, category="state")
        ic.register("b", lambda s, c: "fail_b", every_n_ticks=1, category="state")
        state = make_game_state()
        ctx = _make_stub_ctx()

        ic.check(0, state, ctx)
        assert ic.violation_count == 2


# ---------------------------------------------------------------------------
# Violation list trimming
# ---------------------------------------------------------------------------


class TestViolationListTrimming:
    def test_violations_trimmed_above_200(self) -> None:
        """Violation list is trimmed to 150 when it exceeds 200 entries."""
        t = [1000.0]
        ic = InvariantChecker(clock=lambda: t[0])
        ic._cooldown = 0.0  # disable cooldown for this test
        ic.register("spam", lambda s, c: "fail", every_n_ticks=1)
        state = make_game_state()
        ctx = _make_stub_ctx()

        for tick in range(210):
            t[0] += 1.0
            ic.check(tick, state, ctx)

        assert len(ic._violations) <= 200


class TestForensicsFlush:
    def test_violation_triggers_forensics_flush(self) -> None:
        """When flush_forensics=True and forensics buffer exists, flush is called."""
        t = [1000.0]
        ic = InvariantChecker(clock=lambda: t[0])
        ic.register("bad", lambda s, c: "broken", every_n_ticks=1, flush_forensics=True)
        state = make_game_state()
        ctx = _make_stub_ctx()
        mock_forensics = MagicMock()
        ctx.diag = MagicMock()
        ctx.diag.forensics = mock_forensics

        ic.check(0, state, ctx)

        mock_forensics.flush.assert_called_once_with("invariant:bad")

    def test_no_flush_when_flush_disabled(self) -> None:
        """When flush_forensics=False, forensics buffer is NOT flushed."""
        t = [1000.0]
        ic = InvariantChecker(clock=lambda: t[0])
        ic.register("noisy", lambda s, c: "slow", every_n_ticks=1, flush_forensics=False)
        state = make_game_state()
        ctx = _make_stub_ctx()
        mock_forensics = MagicMock()
        ctx.diag = MagicMock()
        ctx.diag.forensics = mock_forensics

        ic.check(0, state, ctx)

        mock_forensics.flush.assert_not_called()


# ---------------------------------------------------------------------------
# Built-in invariant check functions
# ---------------------------------------------------------------------------


class TestCheckEngagedHasTarget:
    def test_engaged_no_target_violates(self) -> None:
        from util.invariants import _check_engaged_has_target

        state = make_game_state()
        ctx = _make_stub_ctx()
        ctx.combat.engaged = True
        ctx.combat.pull_target_id = 0

        result = _check_engaged_has_target(state, ctx)
        assert result is not None
        assert "pull_target_id" in result

    def test_engaged_with_target_ok(self) -> None:
        from util.invariants import _check_engaged_has_target

        state = make_game_state()
        ctx = _make_stub_ctx()
        ctx.combat.engaged = True
        ctx.combat.pull_target_id = 42

        result = _check_engaged_has_target(state, ctx)
        assert result is None

    def test_not_engaged_ok(self) -> None:
        from util.invariants import _check_engaged_has_target

        state = make_game_state()
        ctx = _make_stub_ctx()
        ctx.combat.engaged = False

        result = _check_engaged_has_target(state, ctx)
        assert result is None

    def test_no_combat_attr_ok(self) -> None:
        from util.invariants import _check_engaged_has_target

        state = make_game_state()
        ctx = MagicMock(spec=[])  # no combat attr
        result = _check_engaged_has_target(state, ctx)
        assert result is None


class TestCheckManaBounded:
    def test_negative_mana_violates(self) -> None:
        from util.invariants import _check_mana_bounded

        state = make_game_state(mana_current=-10, mana_max=500)
        ctx = _make_stub_ctx()

        result = _check_mana_bounded(state, ctx)
        assert result is not None
        assert "negative" in result

    def test_mana_exceeds_max_violates(self) -> None:
        from util.invariants import _check_mana_bounded

        state = make_game_state(mana_current=600, mana_max=500)
        ctx = _make_stub_ctx()

        result = _check_mana_bounded(state, ctx)
        assert result is not None
        assert "mana_max" in result

    def test_mana_within_tolerance_ok(self) -> None:
        from util.invariants import _check_mana_bounded

        # 10% tolerance: 500 * 1.1 = 550, so 540 is OK
        state = make_game_state(mana_current=540, mana_max=500)
        ctx = _make_stub_ctx()

        result = _check_mana_bounded(state, ctx)
        assert result is None

    def test_normal_mana_ok(self) -> None:
        from util.invariants import _check_mana_bounded

        state = make_game_state(mana_current=300, mana_max=500)
        ctx = _make_stub_ctx()

        result = _check_mana_bounded(state, ctx)
        assert result is None


class TestCheckPositionFinite:
    def test_out_of_bounds_violates(self) -> None:
        from util.invariants import _check_position_finite

        state = make_game_state(x=60000.0, y=0.0)
        ctx = _make_stub_ctx()

        result = _check_position_finite(state, ctx)
        assert result is not None
        assert "out of bounds" in result

    def test_normal_position_ok(self) -> None:
        from util.invariants import _check_position_finite

        state = make_game_state(x=1000.0, y=-2000.0)
        ctx = _make_stub_ctx()

        result = _check_position_finite(state, ctx)
        assert result is None


class TestCheckTickBudget:
    def test_no_diag_ok(self) -> None:
        from util.invariants import _check_tick_budget

        state = make_game_state()
        ctx = _make_stub_ctx()
        ctx.diag = None

        result = _check_tick_budget(state, ctx)
        assert result is None

    def test_no_metrics_ok(self) -> None:
        from util.invariants import _check_tick_budget

        state = make_game_state()
        ctx = _make_stub_ctx()
        ctx.diag = MagicMock()
        ctx.diag.metrics = None

        result = _check_tick_budget(state, ctx)
        assert result is None

    def test_p99_above_budget_violates(self) -> None:
        from util.invariants import _check_tick_budget

        state = make_game_state()
        ctx = _make_stub_ctx()
        tracker = MagicMock()
        tracker.count = 10
        tracker.p99.return_value = 600.0
        ctx.diag = MagicMock()
        ctx.diag.metrics = MagicMock()
        ctx.diag.metrics.tick_duration = tracker

        result = _check_tick_budget(state, ctx)
        assert result is not None
        assert "500ms" in result

    def test_p99_within_budget_ok(self) -> None:
        from util.invariants import _check_tick_budget

        state = make_game_state()
        ctx = _make_stub_ctx()
        tracker = MagicMock()
        tracker.count = 10
        tracker.p99.return_value = 100.0
        ctx.diag = MagicMock()
        ctx.diag.metrics = MagicMock()
        ctx.diag.metrics.tick_duration = tracker

        result = _check_tick_budget(state, ctx)
        assert result is None


class TestCheckNoZombieEngagement:
    def test_not_engaged_ok(self) -> None:
        from util.invariants import _check_no_zombie_engagement

        state = make_game_state()
        ctx = _make_stub_ctx()
        ctx.combat.engaged = False

        result = _check_no_zombie_engagement(state, ctx)
        assert result is None

    def test_engaged_short_duration_ok(self) -> None:
        import time

        from util.invariants import _check_no_zombie_engagement

        state = make_game_state()
        ctx = _make_stub_ctx()
        ctx.combat.engaged = True
        ctx.player = MagicMock()
        ctx.player.engagement_start = time.time() - 30.0  # 30s, under 120

        result = _check_no_zombie_engagement(state, ctx)
        assert result is None

    def test_engaged_long_with_combat_active_ok(self) -> None:
        import time

        from util.invariants import _check_no_zombie_engagement

        state = make_game_state()
        ctx = _make_stub_ctx()
        ctx.combat.engaged = True
        ctx.player = MagicMock()
        ctx.player.engagement_start = time.time() - 200.0
        ctx.diag = MagicMock()
        ctx.diag.last_rule_evaluation = {"IN_COMBAT": "YES"}

        result = _check_no_zombie_engagement(state, ctx)
        assert result is None

    def test_engaged_long_no_combat_rule_violates(self) -> None:
        import time

        from util.invariants import _check_no_zombie_engagement

        state = make_game_state()
        ctx = _make_stub_ctx()
        ctx.combat.engaged = True
        ctx.player = MagicMock()
        ctx.player.engagement_start = time.time() - 200.0
        ctx.diag = MagicMock()
        ctx.diag.last_rule_evaluation = {"REST": "YES"}

        result = _check_no_zombie_engagement(state, ctx)
        assert result is not None
        assert "engaged for" in result

    def test_engaged_long_with_flee_active_ok(self) -> None:
        import time

        from util.invariants import _check_no_zombie_engagement

        state = make_game_state()
        ctx = _make_stub_ctx()
        ctx.combat.engaged = True
        ctx.player = MagicMock()
        ctx.player.engagement_start = time.time() - 200.0
        ctx.diag = MagicMock()
        ctx.diag.last_rule_evaluation = {"FLEE": "YES"}

        result = _check_no_zombie_engagement(state, ctx)
        assert result is None

    def test_engagement_start_zero_ok(self) -> None:
        from util.invariants import _check_no_zombie_engagement

        state = make_game_state()
        ctx = _make_stub_ctx()
        ctx.combat.engaged = True
        ctx.player = MagicMock()
        ctx.player.engagement_start = 0

        result = _check_no_zombie_engagement(state, ctx)
        assert result is None


class TestCheckThreadOwnership:
    def test_no_violations_ok(self) -> None:
        from util.invariants import _check_thread_ownership

        state = make_game_state()
        ctx = _make_stub_ctx()
        result = _check_thread_ownership(state, ctx, violation_fn=lambda: 0)
        assert result is None

    def test_violations_reported(self) -> None:
        from util.invariants import _check_thread_ownership

        state = make_game_state()
        ctx = _make_stub_ctx()
        result = _check_thread_ownership(state, ctx, violation_fn=lambda: 3)
        assert result is not None
        assert "3" in result


class TestRegisterBuiltinInvariants:
    def test_registers_all_expected_invariants(self) -> None:
        from util.invariants import register_builtin_invariants

        ic = InvariantChecker()
        register_builtin_invariants(ic)

        names = [name for name, _, _, _, _ in ic._checks]
        assert "engaged_has_target" in names
        assert "mana_bounded" in names
        assert "position_finite" in names
        assert "tick_budget" in names
        assert "no_zombie_engagement" in names
        assert "thread_ownership" in names
        assert len(names) == 6
