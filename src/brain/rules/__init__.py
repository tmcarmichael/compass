"""Brain rule modules  -  each registers rules with the Brain.

Separates rule conditions from agent.py's monolithic _build_brain.
Each module exports register(brain, ctx, read_state_fn).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from brain.rules.combat import register as register_combat
from brain.rules.maintenance import register as register_maintenance
from brain.rules.navigation import register as register_navigation
from brain.rules.survival import register as register_survival

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.decision import Brain
    from core.types import ReadStateFn


def register_all(brain: Brain, ctx: AgentContext, read_state_fn: ReadStateFn) -> None:
    """Register all rule modules in priority order.

    Order (14 rules): DEATH_RECOVERY > FEIGN_DEATH > FLEE > REST >
           EVADE > IN_COMBAT > ENGAGE_ADD >
           ACQUIRE > PULL > MEMORIZE_SPELLS > SUMMON_PET >
           BUFF > TRAVEL > WANDER

    Key: FEIGN_DEATH fires BEFORE FLEE  -  try FD first, flee if FD fails/unavailable.
    Key: EVADE fires BEFORE combat/acquire  -  sidestep threats before engaging.
    """
    register_survival(brain, ctx, read_state_fn)  # DEATH_RECOVERY, FEIGN_DEATH, FLEE, REST, EVADE
    register_combat(brain, ctx, read_state_fn)  # IN_COMBAT, ENGAGE_ADD, ACQUIRE, PULL
    register_maintenance(brain, ctx, read_state_fn)  # MEMORIZE_SPELLS, SUMMON_PET, BUFF
    register_navigation(brain, ctx, read_state_fn)  # TRAVEL, WANDER


__all__: list[str] = []
