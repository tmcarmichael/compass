"""RuleDef: structured rule definition replacing the 6-tuple in Brain._rules.

Supports all five utility scoring phases:
- Phase 0: condition() only (current binary system)
- Phase 1: score_fn() runs in parallel for divergence logging
- Phase 2: score-based selection within priority tiers
- Phase 3: weighted cross-tier scoring
- Phase 4: consideration-based scoring with online tuning
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from perception.state import GameState
from routines.base import RoutineBase

if TYPE_CHECKING:
    from brain.context import AgentContext


@dataclass(slots=True)
class RuleDef:
    """A single brain rule with condition, scoring, and metadata."""

    name: str
    condition: Callable[[GameState], bool]
    routine: RoutineBase
    failure_cooldown: float = 0.0
    emergency: bool = False
    max_lock_seconds: float = 0.0
    # Phase 1+: utility scoring
    score_fn: Callable[[GameState], float] = lambda s: 0.0
    # Phase 2+: priority tier (lower = higher priority)
    tier: int = 0
    # Phase 3+: weight for cross-tier scoring
    weight: float = 1.0
    # Phase 4: considerations (list of Consideration objects)
    considerations: list = field(default_factory=list)
    # Circuit breaker: max failures in window before tripping (0 = disabled)
    breaker_max_failures: int = 5
    breaker_window: float = 300.0
    breaker_recovery: float = 120.0


@dataclass(slots=True)
class Consideration:
    """A named input -> curve -> weight component of a rule's score.

    Used in Phase 4 for declarative scoring. The scoring engine computes
    weighted geometric mean of all consideration outputs.
    """

    name: str
    input_fn: Callable  # (GameState, ctx) -> float
    curve: Callable[[float], float]  # maps raw input to 0.0-1.0
    weight: float = 1.0


def score_from_considerations(
    considerations: list[Consideration], state: GameState, ctx: AgentContext
) -> float:
    """Compute weighted geometric mean of consideration outputs.

    A zero from any consideration acts as a hard gate (returns 0.0).
    Used by Phase 4 for declarative scoring.
    """
    if not considerations:
        return 0.0
    total_log = 0.0
    weight_sum = 0.0
    for c in considerations:
        raw = c.input_fn(state, ctx)
        mapped = c.curve(raw)
        if mapped <= 0.0:
            return 0.0  # hard gate
        total_log += c.weight * math.log(mapped)
        weight_sum += c.weight
    return math.exp(total_log / weight_sum) if weight_sum > 0 else 0.0
