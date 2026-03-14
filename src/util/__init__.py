"""Observability and correctness infrastructure.

Structured logging:
    structured_log.py    -  JSONL event emission integrated with stdlib logging
    event_schemas.py     -  Typed event dataclasses (kills, deaths, level-ups, ...)
    log_tiers.py         -  4-tier graduated log levels (EVENT/INFO/VERBOSE/DEBUG)

Session analysis:
    cycle_tracker.py     -  Composite cycle narratives (acquire→pull→combat→defeat)
    phase_detector.py    -  Session phase detection for long-horizon analysis
    session_reporter.py  -  End-of-session performance summary
    analyze_session.py   -  Offline JSONL log analysis and session comparison

Failure forensics:
    forensics.py         -  300-tick ring buffer, flushes on death/violation
    incident_reporter.py -  Causal chain reconstruction from incident data

Correctness:
    invariants.py        -  Runtime invariant checking with structured violation events
    thread_guard.py      -  Thread-ownership assertions for shared state

Infrastructure:
    metrics.py           -  Lightweight percentile/rate metrics
    clock.py             -  Tick-rate clock with drift compensation
"""

__all__: list[str] = []
