"""Real-time anomaly detector: streaming issue detection with runtime correction.

Runs every 30s during the periodic snapshot cycle. Detects behavioral anomalies
(camp drift, acquire loops, heading lock, mana depletion, defeat drought) and
optionally applies lightweight corrections to resume normal operation.

Usage:
    detector = AnomalyDetector(ctx)
    # Every 30s in brain_runner:
    issues = detector.check(state)
    # issues is a list of Issue dataclasses with type, message, severity
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.context import AgentContext
    from perception.state import GameState

log = logging.getLogger(__name__)


class IssueSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class IssueType(StrEnum):
    CAMP_DRIFT = "camp_drift"
    ACQUIRE_LOOP = "acquire_loop"
    HEADING_LOCK = "heading_lock"
    MANA_DEPLETION = "mana_depletion"
    DEFEAT_DROUGHT = "defeat_drought"
    FIZZLE_STREAK = "fizzle_streak"
    STUCK_EVENTS = "stuck_events"


@dataclass(frozen=True, slots=True)
class Issue:
    """A detected anomaly with type, message, and severity."""

    type: IssueType
    message: str
    severity: IssueSeverity
    timestamp: float


@dataclass(slots=True)
class AnomalyDetector:
    """Streaming anomaly detector called every 30s during periodic snapshot.

    Detects behavioral anomalies and applies lightweight runtime corrections.
    Issues are returned to the caller and logged via the brain logger.
    """

    ctx: AgentContext
    _active_issues: list[Issue] = field(default_factory=list)
    _issue_history: list[Issue] = field(default_factory=list)
    _last_heading: float = -1.0
    _heading_stuck_count: int = 0
    _corrections_applied: int = 0
    _max_history: int = 200

    # Debounce: don't re-fire the same issue type within this window
    _last_fire_time: dict[str, float] = field(default_factory=dict)
    _debounce_seconds: float = 90.0  # 3 snapshot cycles

    def check(self, state: GameState) -> list[Issue]:
        """Run all detectors, return new issues, apply any runtime corrections.

        Called every 30s from brain_runner's periodic snapshot cycle.
        """
        now = time.time()
        issues: list[Issue] = []

        self._check_camp_drift(state, now, issues)
        self._check_acquire_loop(state, now, issues)
        self._check_heading_lock(state, now, issues)
        self._check_mana_depletion(state, now, issues)
        self._check_kill_drought(state, now, issues)
        self._check_fizzle_streak(state, now, issues)
        self._check_stuck_events(state, now, issues)

        # Update active issues and history
        self._active_issues = issues
        for issue in issues:
            self._issue_history.append(issue)
        if len(self._issue_history) > self._max_history:
            self._issue_history = self._issue_history[-self._max_history // 2 :]

        # Log all issues
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                log.warning("[PERCEPTION] ANOMALY [%s] %s", issue.type.value, issue.message)
            else:
                log.info("[PERCEPTION] ANOMALY [%s] %s", issue.type.value, issue.message)

        return issues

    @property
    def active_issues(self) -> list[Issue]:
        """Current active issues (from last check)."""
        return list(self._active_issues)

    @property
    def correction_count(self) -> int:
        return self._corrections_applied

    def _debounced(self, issue_type: str, now: float) -> bool:
        """Return True if this issue type was recently fired (should skip)."""
        last = self._last_fire_time.get(issue_type, 0.0)
        if now - last < self._debounce_seconds:
            return True
        self._last_fire_time[issue_type] = now
        return False

    # -- Detectors --

    def _check_camp_drift(self, state: GameState, now: float, issues: list[Issue]) -> None:
        """Detect when agent has drifted too far from camp center."""
        ctx = self.ctx
        if not ctx.camp.camp_pos.x and not ctx.camp.camp_pos.y:
            return
        dist = ctx.camp.distance_to_camp(state)
        if dist > 300 and not self._debounced(IssueType.CAMP_DRIFT, now):
            severity = IssueSeverity.CRITICAL if dist > 600 else IssueSeverity.WARNING
            issues.append(
                Issue(
                    type=IssueType.CAMP_DRIFT,
                    message=f"Camp drift {dist:.0f}u (camp at {ctx.camp.camp_pos.x:.0f},{ctx.camp.camp_pos.y:.0f})",
                    severity=severity,
                    timestamp=now,
                )
            )
            # Runtime correction: set travel plan back to camp if severely drifted
            if dist > 600 and not ctx.combat.engaged:
                from core.types import PlanType

                if ctx.plan.active not in (PlanType.TRAVEL, PlanType.NEEDS_MEMORIZE):
                    ctx.plan.active = PlanType.TRAVEL
                    ctx.plan.travel.target_x = ctx.camp.camp_pos.x
                    ctx.plan.travel.target_y = ctx.camp.camp_pos.y
                    self._corrections_applied += 1
                    log.warning(
                        "[PERCEPTION] ANOMALY correction: set TRAVEL plan to camp (drift=%.0fu)", dist
                    )

    def _check_acquire_loop(self, state: GameState, now: float, issues: list[Issue]) -> None:
        """Detect repeated acquire failures (no valid targets found)."""
        ctx = self.ctx
        fails = ctx.metrics.consecutive_acquire_fails
        if fails >= 5 and not self._debounced(IssueType.ACQUIRE_LOOP, now):
            severity = IssueSeverity.CRITICAL if fails >= 10 else IssueSeverity.WARNING
            issues.append(
                Issue(
                    type=IssueType.ACQUIRE_LOOP,
                    message=f"{fails}x consecutive acquire failures",
                    severity=severity,
                    timestamp=now,
                )
            )
            # Runtime correction: temporarily expand roam radius
            if fails >= 5 and hasattr(ctx.camp, "roam_radius"):
                original = ctx.camp.roam_radius
                expanded = min(original + 50, original * 1.5, 300)
                if expanded > original:
                    ctx.camp.roam_radius = expanded
                    self._corrections_applied += 1
                    log.warning(
                        "[PERCEPTION] ANOMALY correction: expanded roam_radius "
                        "%.0f -> %.0f (acquire fails=%d)",
                        original,
                        expanded,
                        fails,
                    )

    def _check_heading_lock(self, state: GameState, now: float, issues: list[Issue]) -> None:
        """Detect stuck heading (same heading for multiple snapshots)."""
        heading = round(state.heading, 0)
        if self._last_heading >= 0 and heading == self._last_heading:
            self._heading_stuck_count += 1
        else:
            self._heading_stuck_count = 0
        self._last_heading = heading

        # 3+ consecutive snapshots (90s+) with same heading = stuck
        if self._heading_stuck_count >= 3 and not self._debounced(IssueType.HEADING_LOCK, now):
            issues.append(
                Issue(
                    type=IssueType.HEADING_LOCK,
                    message=f"Heading locked at {heading:.0f} for {self._heading_stuck_count * 30}s+",
                    severity=IssueSeverity.WARNING,
                    timestamp=now,
                )
            )

    def _check_mana_depletion(self, state: GameState, now: float, issues: list[Issue]) -> None:
        """Detect dangerously low mana with no rest activity."""
        ctx = self.ctx
        mana_pct = state.mana_pct
        if (
            mana_pct < 0.15
            and ctx.metrics.rest_count == 0
            and not ctx.combat.engaged
            and not self._debounced(IssueType.MANA_DEPLETION, now)
        ):
            issues.append(
                Issue(
                    type=IssueType.MANA_DEPLETION,
                    message=f"Mana at {mana_pct * 100:.0f}%% with 0 rests",
                    severity=IssueSeverity.CRITICAL,
                    timestamp=now,
                )
            )

    def _check_kill_drought(self, state: GameState, now: float, issues: list[Issue]) -> None:
        """Detect extended periods without defeats."""
        ctx = self.ctx
        defeat_age = ctx.defeat_tracker.last_kill_age()
        session_age = now - ctx.metrics.session_start
        # Only fire after agent has been running 2+ minutes
        if (
            defeat_age > 120
            and session_age > 120
            and not ctx.combat.engaged
            and not self._debounced(IssueType.DEFEAT_DROUGHT, now)
        ):
            severity = IssueSeverity.CRITICAL if defeat_age > 300 else IssueSeverity.WARNING
            issues.append(
                Issue(
                    type=IssueType.DEFEAT_DROUGHT,
                    message=f"No defeats for {defeat_age:.0f}s",
                    severity=severity,
                    timestamp=now,
                )
            )

    def _check_fizzle_streak(self, state: GameState, now: float, issues: list[Issue]) -> None:
        """Detect high fizzle rate indicating level/skill issues."""
        ctx = self.ctx
        fizzles = ctx.metrics.pull_dc_fizzles
        if fizzles >= 4 and not self._debounced(IssueType.FIZZLE_STREAK, now):
            issues.append(
                Issue(
                    type=IssueType.FIZZLE_STREAK,
                    message=f"{fizzles}x DC fizzles this session",
                    severity=IssueSeverity.WARNING,
                    timestamp=now,
                )
            )

    def _check_stuck_events(self, state: GameState, now: float, issues: list[Issue]) -> None:
        """Detect recurring stuck events."""
        try:
            from nav.movement import get_stuck_event_count

            stuck = get_stuck_event_count()
        except (
            ImportError,
            AttributeError,
        ):
            return
        if stuck >= 3 and not self._debounced(IssueType.STUCK_EVENTS, now):
            issues.append(
                Issue(
                    type=IssueType.STUCK_EVENTS,
                    message=f"{stuck}x stuck events this session",
                    severity=IssueSeverity.WARNING,
                    timestamp=now,
                )
            )
