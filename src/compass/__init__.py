"""Public facade package for Compass.

This package gives the repo a stable import surface without renaming the
existing top-level modules (`brain`, `runtime`, `simulator`, ...).
"""

from __future__ import annotations

from core import __version__

__all__ = [
    "__version__",
    "AgentOrchestrator",
    "build_brain",
    "build_context",
    "find_config",
    "load_zone_config",
]


def __getattr__(name: str):
    """Lazily expose the most useful runtime entry points."""
    if name == "AgentOrchestrator":
        from runtime.orchestrator import AgentOrchestrator

        return AgentOrchestrator
    if name in {"build_brain", "build_context", "find_config", "load_zone_config"}:
        from runtime import agent as _agent

        return getattr(_agent, name)
    raise AttributeError(f"module 'compass' has no attribute {name!r}")
