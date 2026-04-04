"""Brain: priority state machine that picks which routine to run each tick.

Each tick: read state -> evaluate rules -> tick active routine -> issue motor commands.
Emergency rules (FLEE, DEATH_RECOVERY) evaluate first and can override any locked
routine. Non-emergency rules participate in utility scoring if enabled.

Utility scoring phases implement a "validate before influence" escalation:
  Phase 0: Binary conditions, insertion-order priority (original system).
            Scores are computed but ignored. Conservative baseline.
  Phase 1: Score functions run in parallel for divergence logging.
            When score-based selection would differ, the divergence is logged.
            No behavior change  -- observation mode for validating score functions.
  Phase 2: Score-based selection within priority tiers.
            Within a tier, highest score wins. Between tiers, higher priority wins.
            Safety hierarchy preserved while gaining flexibility.
  Phase 3: Weighted cross-tier scoring.
            Emergency rules retain hard priority. Non-emergency rules compete
            by weight * score. Full utility AI  -- decisions are value-compared.
  Phase 4: Consideration-based scoring.
            Rules with considerations use weighted geometric mean of
            (input_fn -> curve -> weight) components. Rules without
            considerations fall back to score_fn. Otherwise same as Phase 3.

The escalation path allows each phase to be validated in production before
the next phase changes behavior. Phase 1 ran for weeks before Phase 2
was enabled, catching score functions that produced pathological results.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from brain.circuit_breaker import CircuitBreaker
from brain.completion import tick_active_routine
from brain.profiling import tick_profiling
from brain.rule_def import Consideration, RuleDef
from brain.scoring_phases import (
    build_phase_selector,
    compute_divergence,
)
from brain.transitions import handle_transition
from perception.state import GameState
from routines.base import RoutineBase
from util.log_tiers import VERBOSE

if TYPE_CHECKING:
    from brain.context import AgentContext

log = logging.getLogger(__name__)


class Brain:
    """Priority-based decision engine with optional utility scoring.

    Each tick, evaluates conditions top-to-bottom and activates the
    highest-priority routine whose condition is met. Handles routine
    lifecycle (enter/tick/exit).

    Supports per-rule cooldowns: after a routine finishes with FAILURE,
    the rule is suppressed for a configurable duration.

    Utility scoring (phases 1-4) is additive and backward-compatible:
      Phase 0-1: Binary conditions (original system / divergence logging).
      Phase 2: Score-based selection within priority tiers.
      Phase 3: Weighted cross-tier scoring.
      Phase 4: Consideration-based scoring with weighted geometric mean.
    Set utility_phase=0 (default) for original binary behavior.
    """

    def __init__(
        self,
        ctx: AgentContext | None = None,
        utility_phase: int = 0,
        perf_clock: Callable[[], float] = time.perf_counter,
        shutdown_hook: Callable[[], None] | None = None,
    ) -> None:
        self.perf_clock = perf_clock
        self._shutdown_hook = shutdown_hook
        self._rules: list[RuleDef] = []
        self._active: RoutineBase | None = None
        self._active_name: str = ""
        self._cooldowns: dict[str, float] = {}  # rule_name -> cooldown_until
        self._ctx = ctx  # for session tracking
        self._last_diag_time: float = 0.0
        self._active_start_time: float = 0.0  # when current routine activated
        self._last_lock_blocked: str = ""  # suppress repeated lock messages
        self.utility_phase: int = utility_phase
        self._phase_selector = build_phase_selector(utility_phase)

        # Per-tick profiling (rule eval + routine tick times in ms)
        self.rule_times: dict[str, float] = {}  # rule_name -> eval ms
        self.tick_total_ms: float = 0.0  # total tick() wall time
        self.routine_tick_ms: float = 0.0  # routine.tick() time

        # Last matched rule name (set each tick, consumed by brain_runner for replay log)
        self._last_matched_rule: str = ""

        # Decision receipt data (consumed by brain_runner for structured logging)
        self.last_rule_eval: dict[str, str] = {}  # rule_name -> YES/no/cooldown
        # Utility scoring diagnostics (Phase 1+)
        self.rule_scores: dict[str, float] = {}  # rule_name -> last score
        self._score_winner: str = ""  # what score-based selection would pick
        self._ticked_routine_name: str = ""  # name captured before completion clears it

        # Per-rule circuit breakers (Phase 3 hardening)
        self._breakers: dict[str, CircuitBreaker] = {}

    def add_rule(
        self,
        name: str,
        condition: Callable[[GameState], bool],
        routine: RoutineBase,
        failure_cooldown: float = 0.0,
        emergency: bool = False,
        max_lock_seconds: float = 0.0,
        score_fn: Callable[[GameState], float] | None = None,
        tier: int = 0,
        weight: float = 1.0,
        considerations: list[Consideration] | None = None,
        breaker_max_failures: int = 5,
        breaker_window: float = 300.0,
        breaker_recovery: float = 120.0,
    ) -> None:
        """Add a priority rule. Rules are evaluated in insertion order.

        Args:
            name: Human-readable name for logging.
            condition: Callable(GameState) -> bool.
            routine: The RoutineBase instance to activate.
            failure_cooldown: Seconds to suppress this rule after FAILURE.
            emergency: If True, can interrupt locked routines.
            max_lock_seconds: If > 0, force-exit locked routine after this
                many seconds. Prevents indefinite stuck states. 0 = no limit.
            score_fn: Callable(GameState) -> float (Phase 1+). Utility score.
            tier: Priority tier (Phase 2+). Lower = higher priority.
            weight: Base weight (Phase 3+). Multiplied by score for selection.
            considerations: List of Consideration objects (Phase 4). When
                non-empty, Phase 4 uses these instead of score_fn.
            breaker_max_failures: Trip circuit after N failures in window (0=disabled).
            breaker_window: Failure counting window in seconds.
            breaker_recovery: Seconds to stay open before half-open probe.
        """
        self._rules.append(
            RuleDef(
                name=name,
                condition=condition,
                routine=routine,
                failure_cooldown=failure_cooldown,
                emergency=emergency,
                max_lock_seconds=max_lock_seconds,
                score_fn=score_fn or (lambda s: 0.0),
                tier=tier,
                weight=weight,
                considerations=considerations or [],
                breaker_max_failures=breaker_max_failures,
                breaker_window=breaker_window,
                breaker_recovery=breaker_recovery,
            )
        )
        # Create circuit breaker (emergency rules exempt -- agent must always flee)
        if breaker_max_failures > 0 and not emergency:
            self._breakers[name] = CircuitBreaker(
                name=name,
                max_failures=breaker_max_failures,
                window_seconds=breaker_window,
                recovery_seconds=breaker_recovery,
            )

    # -- Public accessors (used by BrainRunner, simulator) --

    @property
    def active_routine(self) -> RoutineBase | None:
        """The currently running routine, or None."""
        return self._active

    @property
    def active_routine_name(self) -> str:
        """Name of the currently running routine."""
        return self._active_name

    @property
    def last_matched_rule(self) -> str:
        """Name of the rule that matched on the most recent tick."""
        return self._last_matched_rule

    def tick(self, state: GameState) -> None:
        """Evaluate rules and run the appropriate routine."""
        tick_start = self.perf_clock()
        now = time.time()
        selected, selected_name, selected_emergency = self._evaluate_rules(state, now)
        handle_transition(self, state, selected, selected_name, selected_emergency, now)
        tick_active_routine(self, state, now)
        tick_profiling(self, tick_start)

    def _evaluate_rules(
        self,
        state: GameState,
        now: float,
    ) -> tuple[RoutineBase | None, str, bool]:
        """Evaluate all rules and return (selected, selected_name, selected_emergency).

        Also updates self.rule_times, self.last_rule_eval, self._last_matched_rule,
        and ctx.diag.last_rule_evaluation. Logs the evaluation summary on transitions
        and every 30 s.
        """
        selected: RoutineBase | None = None
        selected_name = ""
        selected_emergency = False

        do_diag = now - self._last_diag_time > 30.0
        diag_results: list[str] = []
        rule_eval: dict[str, str] = {}
        rule_times: dict[str, float] = {}

        selected, selected_name, selected_emergency = self._phase_selector.select(
            self, state, now, rule_eval, diag_results, rule_times
        )

        # Phase 1+: compute scores in parallel for divergence logging
        if self.utility_phase >= 1:
            compute_divergence(self, state, now, selected_name)

        self.rule_times = rule_times
        self._last_matched_rule = selected_name
        self.last_rule_eval = rule_eval  # exposed for decision receipt logging

        # Expose rule evaluation to diagnostic state (thread-safe: atomic dict ref assignment)
        if self._ctx:
            self._ctx.diag.last_rule_evaluation = rule_eval

        # Log full rule evaluation on actual transitions or every 30s
        # Don't count "blocked by lock" as a transition (causes 10Hz spam)
        is_lock_blocked = (
            selected is not self._active
            and self._active is not None
            and self._active.locked
            and not selected_emergency
        )
        is_transition = (selected is not self._active) and not is_lock_blocked
        if is_transition or do_diag:
            log.log(
                VERBOSE,
                "[DECISION] Brain eval: [%s] -> %s",
                " | ".join(diag_results),
                selected_name or "NONE",
            )
            self._last_diag_time = now

        return selected, selected_name, selected_emergency

    def shutdown(self, state: GameState) -> None:
        """Cleanly deactivate the running routine and release motor keys.

        Called by brain_runner during shutdown to prevent ghost input.
        """
        if self._active is not None:
            log.info("[LIFECYCLE] Brain shutdown: exiting active routine %s", self._active_name)
            try:
                self._active.exit(state)
            except Exception as e:
                log.warning("[LIFECYCLE] Brain shutdown: exit(%s) failed: %s", self._active_name, e)
            self._active = None
            self._active_name = ""
            self._active_start_time = 0.0

        # Release any held keys (caller provides hook to avoid brain->motor import)
        if self._shutdown_hook:
            try:
                self._shutdown_hook()
            except Exception as e:
                log.warning("[LIFECYCLE] Brain shutdown: shutdown_hook failed: %s", e)
