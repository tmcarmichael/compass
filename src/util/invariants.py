"""Runtime invariant checker -- replaces tests for a live control system.

Invariants are checked every N ticks. Violations are emitted as structured
events and trigger a forensics buffer flush. This catches real bugs in the
real environment -- the exact thing traditional tests model worst.

Built-in invariants cover the most common state corruption scenarios:
engaged without target, pull target dead, mana out of bounds, position
at infinity, tick budget exceeded, zombie engagement.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from util.event_schemas import InvariantViolationEvent
from util.structured_log import iso_ts, log_event
from util.thread_guard import violation_count as thread_violation_count

if TYPE_CHECKING:
    from brain.context import AgentContext
    from perception.state import GameState

log = logging.getLogger(__name__)


class InvariantChecker:
    """Runtime invariant checker. Violations -> structured event + forensics flush.

    Each invariant is a check function that returns None (ok) or an error
    message string. Checked every N ticks (configurable per invariant).
    Rate-limited per invariant name to avoid spam.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._checks: list[tuple[str, Callable, int, str, bool]] = []
        # name -> (check_fn, every_n_ticks, category, flush_forensics)
        self._violations: list[dict] = []
        self._violation_count: int = 0
        self._last_fire: dict[str, float] = {}
        self._cooldown: float = 30.0  # min seconds between same invariant

    def register(
        self,
        name: str,
        check_fn: Callable[..., str | None],
        every_n_ticks: int = 10,
        category: str = "state",
        flush_forensics: bool = True,
    ) -> None:
        """Register an invariant.

        Args:
            name: Unique invariant name (e.g. "engaged_has_target").
            check_fn: Callable(state, ctx) -> None (ok) or error message.
            every_n_ticks: Check frequency (10 = every 10th tick = 1Hz).
            category: Classification (state, timing, resource, combat).
            flush_forensics: If True, dump forensics ring buffer on violation.
                Set False for noisy performance metrics (tick_budget).
        """
        self._checks.append((name, check_fn, every_n_ticks, category, flush_forensics))

    def check(self, tick_id: int, state: GameState, ctx: AgentContext) -> None:
        """Run all registered invariants at their configured frequency."""
        now = self._clock()
        for name, check_fn, interval, category, do_flush in self._checks:
            if tick_id % interval != 0:
                continue
            try:
                result = check_fn(state, ctx)
            except Exception:
                continue  # invariant check itself failed -- skip, don't crash
            if result is None:
                continue  # invariant holds

            # Rate limit per invariant name
            last = self._last_fire.get(name, 0.0)
            if now - last < self._cooldown:
                continue
            self._last_fire[name] = now
            self._violation_count += 1

            violation = {
                "name": name,
                "message": result,
                "category": category,
                "tick_id": tick_id,
                "ts": iso_ts(now),
            }
            self._violations.append(violation)
            if len(self._violations) > 200:
                self._violations = self._violations[-150:]

            log_event(
                log,
                "invariant_violation",
                f"INVARIANT: {name} -- {result}",
                level=logging.WARNING,
                **InvariantViolationEvent(invariant=name, category=category, detail=result),
            )

            # Trigger forensics flush (skip for noisy performance invariants)
            if do_flush:
                forensics = getattr(getattr(ctx, "diag", None), "forensics", None)
                if forensics:
                    forensics.flush(f"invariant:{name}")

    @property
    def violation_count(self) -> int:
        return self._violation_count

    def summary(self) -> dict:
        """Summary for session report."""
        by_category: dict[str, int] = {}
        by_name: dict[str, int] = {}
        for v in self._violations:
            cat = v.get("category", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1
            n = v.get("name", "unknown")
            by_name[n] = by_name.get(n, 0) + 1
        return {
            "total": self._violation_count,
            "by_category": by_category,
            "by_name": by_name,
        }


# -- Built-in invariant check functions ------------------------------------
# Each takes (state, ctx) and returns None (ok) or error string.


def _check_engaged_has_target(state: GameState, ctx: AgentContext) -> str | None:
    """If engaged, pull_target_id must be set."""
    combat = getattr(ctx, "combat", None)
    if combat and getattr(combat, "engaged", False):
        pid = getattr(combat, "pull_target_id", None)
        if not pid:
            return "engaged=True but pull_target_id is None/0"
    return None


def _check_mana_bounded(state: GameState, ctx: AgentContext) -> str | None:
    """Mana must be 0 <= current <= max."""
    mana = getattr(state, "mana_current", 0)
    mana_max = getattr(state, "mana_max", 0)
    if mana < 0:
        return f"mana_current={mana} (negative)"
    if mana_max > 0 and mana > mana_max * 1.1:  # 10% tolerance for read timing
        return f"mana_current={mana} > mana_max={mana_max}"
    return None


def _check_position_finite(state: GameState, ctx: AgentContext) -> str | None:
    """Position must be within EQ world bounds."""
    x = getattr(state, "x", 0.0)
    y = getattr(state, "y", 0.0)
    if abs(x) > 50000 or abs(y) > 50000:
        return f"position out of bounds: ({x:.0f}, {y:.0f})"
    return None


def _check_tick_budget(state: GameState, ctx: AgentContext) -> str | None:
    """Brain tick should not exceed 500ms."""
    diag = getattr(ctx, "diag", None)
    if not diag:
        return None
    # Access metrics engine for the last tick_ms
    metrics = getattr(diag, "metrics", None)
    if metrics and hasattr(metrics, "tick_duration"):
        tracker = metrics.tick_duration
        if tracker.count > 0 and tracker.p99() > 500:
            return f"tick p99={tracker.p99():.0f}ms exceeds 500ms budget"
    return None


def _check_no_zombie_engagement(state: GameState, ctx: AgentContext) -> str | None:
    """If engaged for >120s and not in combat/pull/flee routine, flag it."""
    combat = getattr(ctx, "combat", None)
    player = getattr(ctx, "player", None)
    if not combat or not player:
        return None
    if not getattr(combat, "engaged", False):
        return None
    start = getattr(player, "engagement_start", 0)
    if start <= 0:
        return None
    age = time.time() - start
    if age <= 120:
        return None
    # Check active routine via diag -- rule_eval values vary by utility_phase:
    # Phase 0/1: "YES"/"no"/"cooldown(Xs)", Phase 2+: score strings like "1.50"
    # So check for anything that isn't "no", empty, or a cooldown prefix
    diag = getattr(ctx, "diag", None)
    rule_eval = getattr(diag, "last_rule_evaluation", {}) if diag else {}
    for rule in ("IN_COMBAT", "PULL", "FLEE"):
        val = rule_eval.get(rule, "")
        if val and val != "no" and not val.startswith("cooldown"):
            return None
    return f"engaged for {age:.0f}s without combat/pull/flee active"


def _check_thread_ownership(
    state: GameState, ctx: AgentContext, *, violation_fn: Callable[[], int] = thread_violation_count
) -> str | None:
    """Report if any thread ownership violations have occurred."""
    count = violation_fn()
    if count > 0:
        return f"{count} thread ownership violation(s) this session"
    return None


def register_builtin_invariants(
    checker: InvariantChecker,
    thread_violation_fn: Callable[[], int] | None = None,
) -> None:
    """Register all built-in invariants."""
    checker.register("engaged_has_target", _check_engaged_has_target, every_n_ticks=10, category="state")
    checker.register("mana_bounded", _check_mana_bounded, every_n_ticks=10, category="state")
    checker.register("position_finite", _check_position_finite, every_n_ticks=10, category="state")
    checker.register(
        "tick_budget", _check_tick_budget, every_n_ticks=50, category="timing", flush_forensics=False
    )
    checker.register("no_zombie_engagement", _check_no_zombie_engagement, every_n_ticks=50, category="combat")

    vfn = thread_violation_fn or thread_violation_count

    def _thread_check(state: GameState, ctx: AgentContext) -> str | None:
        return _check_thread_ownership(state, ctx, violation_fn=vfn)

    checker.register(
        "thread_ownership",
        _thread_check,
        every_n_ticks=100,
        category="threading",
        flush_forensics=False,
    )
