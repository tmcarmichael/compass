"""Combat spell rotation strategies.

Four CastStrategy implementations that define how the combat routine
prioritizes abilities: pet_tank, pet_and_dot, fear_kite, endgame.
Strategy selection uses level, difficulty rating, and learned data.
"""

from routines.strategies.base import CastContext, CastStrategy
from routines.strategies.selection import CombatStrategy, select_strategy

__all__ = [
    "CastContext",
    "CastStrategy",
    "CombatStrategy",
    "select_strategy",
]
