"""Routines: state machines implementing the grinding loop.

Each routine extends RoutineBase (base.py) with enter/tick/exit lifecycle.
tick() returns RUNNING, SUCCESS, or FAILURE each brain tick (10 Hz).

Routines by priority (highest first):
  death_recovery, feign_death, flee, rest, (evade),
  combat, engage_extra_npc, loot, acquire, pull,
  memorize_spells, summon_pet, buff, travel, wander

Support routines (no brain rule): casting, pet_combat.
"""

from routines.base import RoutineBase, RoutineStatus

__all__ = ["RoutineBase", "RoutineStatus"]
