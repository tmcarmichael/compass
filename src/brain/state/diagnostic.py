"""Diagnostic state: rule evaluation, structured logging, routine timing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from util.cycle_tracker import CycleTracker
    from util.forensics import ForensicsBuffer
    from util.incident_reporter import IncidentReporter
    from util.invariants import InvariantChecker
    from util.metrics import MetricsEngine
    from util.phase_detector import PhaseDetector
    from util.structured_log import DecisionThrottle, StructuredHandler


@dataclass(slots=True, kw_only=True)
class DiagnosticState:
    """Brain decision diagnostics and structured logging handles.

    Read by external diagnostics consumers (scalars are GIL-atomic, safe without lock).
    structured_handler ref used by brain_runner to update tick_id each tick.
    """

    last_rule_evaluation: dict[str, str] = field(default_factory=dict)
    # Phase 1+: utility scores per rule (thread-safe via atomic ref assignment)
    rule_scores: dict[str, float] = field(default_factory=dict)
    # Monotonic tick counter -- set by brain_runner each tick
    current_tick_id: int = 0
    # Ref to StructuredHandler for tick_id updates
    structured_handler: StructuredHandler | None = None
    # Ref to DecisionThrottle for 2Hz decision receipts
    decision_throttle: DecisionThrottle | None = None
    # Ref to ForensicsBuffer for context-on-failure ring buffer
    forensics: ForensicsBuffer | None = None
    # Ref to MetricsEngine for percentile/success rate tracking
    metrics: MetricsEngine | None = None
    # Ref to InvariantChecker for runtime invariants
    invariants: InvariantChecker | None = None
    # Ref to CycleTracker for defeat cycle narrative events
    cycle_tracker: CycleTracker | None = None
    # Ref to IncidentReporter for death/flee composite events
    incident_reporter: IncidentReporter | None = None
    # Ref to PhaseDetector for session phase transitions
    phase_detector: PhaseDetector | None = None
    # GOAP planner suggestion (routine name, empty if no plan)
    goap_suggestion: str = ""
    # Tick budget tracking (Phase 2a hardening)
    tick_overbudget_count: int = 0
    tick_overbudget_max_ms: float = 0.0
    tick_overbudget_last_routine: str = ""
    # Circuit breaker states (Phase 3 hardening) -- dict[name, state_str]
    # Atomic ref swap: brain thread writes, consumers read
    breaker_states: dict[str, str] = field(default_factory=dict)
