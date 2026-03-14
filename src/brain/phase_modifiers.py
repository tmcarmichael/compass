"""Session phase score modifiers for context-aware utility scoring.

Maps the PhaseDetector's 5 operational phases (startup, grinding, resting,
incident, idle) to per-rule score multipliers. Applied in the decision
engine's scoring path so the agent behaves differently during startup
(cautious, exploratory), steady grinding (default), incident recovery
(very cautious), and idle periods (strongly exploratory).

Also provides an inventory pressure modifier that gradually boosts
maintenance urgency as inventory weight fills up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PhaseModifiers:
    """Multiplicative score adjustments per session phase.

    Each field corresponds to a rule category. Values >1.0 boost
    the category, <1.0 suppress it. Default 1.0 = no change.
    """

    acquire_mult: float = 1.0  # target acquisition eagerness
    rest_mult: float = 1.0  # rest urgency
    wander_mult: float = 1.0  # exploration vs. camping
    safety_mult: float = 1.0  # flee/evade sensitivity
    combat_mult: float = 1.0  # in-combat commitment
    maintenance_mult: float = 1.0  # buff/pet/memorize


# Phase profiles: how each operational phase modifies scoring
PHASE_PROFILES: dict[str, PhaseModifiers] = {
    "startup": PhaseModifiers(
        acquire_mult=0.8,  # less aggressive (building up)
        wander_mult=1.3,  # explore more (find good targets)
        safety_mult=1.2,  # more cautious (don't die early)
        maintenance_mult=1.2,  # prioritize buffing/pet setup
    ),
    "grinding": PhaseModifiers(),  # all 1.0 (steady state)
    "resting": PhaseModifiers(
        rest_mult=1.1,  # slightly prefer longer rest
    ),
    "incident": PhaseModifiers(
        acquire_mult=0.6,  # very conservative after death/flee
        safety_mult=1.5,  # heightened safety awareness
        rest_mult=1.3,  # prefer full recovery before engaging
        maintenance_mult=1.3,  # rebuff, resummon if needed
    ),
    "idle": PhaseModifiers(
        wander_mult=1.5,  # strongly explore (camp is dry)
        acquire_mult=1.2,  # grab any target available
    ),
}

# Rule name -> modifier field mapping
_RULE_MODIFIER_MAP: dict[str, str] = {
    # Survival rules
    "FLEE": "safety_mult",
    "EVADE": "safety_mult",
    "FEIGN_DEATH": "safety_mult",
    "DEATH_RECOVERY": "safety_mult",
    "REST": "rest_mult",
    # Combat rules
    "IN_COMBAT": "combat_mult",
    "ENGAGE_ADD": "combat_mult",
    "ACQUIRE": "acquire_mult",
    "PULL": "acquire_mult",
    # Maintenance rules
    "MEMORIZE_SPELLS": "maintenance_mult",
    "SUMMON_PET": "maintenance_mult",
    "BUFF": "maintenance_mult",
    # Navigation rules
    "TRAVEL": "wander_mult",
    "WANDER": "wander_mult",
}


def get_phase_modifier(phase: str, rule_name: str) -> float:
    """Return the score multiplier for a rule in the current session phase.

    Returns 1.0 (no modification) if the phase or rule is unknown.
    """
    profile = PHASE_PROFILES.get(phase)
    if profile is None:
        return 1.0
    field_name = _RULE_MODIFIER_MAP.get(rule_name)
    if field_name is None:
        return 1.0
    return getattr(profile, field_name, 1.0)
