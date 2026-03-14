"""Combat engagement state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.types import SpellOutcome

if TYPE_CHECKING:
    from eq.loadout import Spell
    from routines.strategies.fear_kite import FearPullTracker


@dataclass(slots=True, kw_only=True)
class CombatState:
    """Tracks live combat engagement with a single NPC target."""

    engaged: bool = False
    pull_target_id: int | None = None
    pull_target_name: str = ""  # name of npc when pull_target_id was set
    active_strategy: str = ""  # CombatStrategy value (pet_tank, fear_kite, etc.)
    strategy_switches: int = 0  # count of mid-fight strategy switches

    # Per-spell recast tracking (spell_id -> last cast timestamp)
    # Replaces the old single last_dot_time / last_lifetap_time
    spell_cast_times: dict[int, float] = field(default_factory=dict)

    # Backward-compatible aliases (used by existing combat.py / pull.py)
    last_dot_time: float = 0.0
    last_lifetap_time: float = 0.0

    # True when combat was initiated by auto-engage (npc attacked us)
    # rather than by pull routine.  CombatRoutine.enter() uses this to
    # know it must send pet_attack (pull routine already did).
    auto_engaged: bool = False

    # Last spell cast result from game log (set by brain_tick_handlers each tick)
    last_cast_result: SpellOutcome = SpellOutcome.NONE

    # Flee urgency hysteresis (was module-level global in survival.py)
    flee_urgency_active: bool = False

    # Pre-rule auto-engage candidate: set by scan_auto_engage() each tick,
    # consumed by CombatRoutine.enter(). Cleared every tick before scan.
    # (spawn_id, name, x, y, level) or None.
    auto_engage_candidate: tuple[int, str, float, float, int] | None = None

    # Last observed HP fraction of the fight target (updated per tick by
    # CombatRoutine). Used by brain stale-engaged clearing to infer defeats
    # when the npc despawns before HP=0 is read.
    last_mob_hp_pct: float = 1.0

    # Fear-pull success/extra_npcs tracking (initialized by FearKiteStrategy)
    fear_tracker: FearPullTracker | None = None

    def time_since_spell(self, spell_id: int) -> float:
        """Seconds since a specific spell was last cast."""
        return time.time() - self.spell_cast_times.get(spell_id, 0)

    def record_spell_cast(self, spell_id: int) -> None:
        """Record that a spell was just cast."""
        self.spell_cast_times[spell_id] = time.time()

    def clear_spell_cast(self, spell_id: int) -> None:
        """Clear cast record for a spell (e.g. after fizzle)."""
        self.spell_cast_times.pop(spell_id, None)

    def should_recast_dot(self, spell: Spell | None) -> bool:
        """Check if a DoT needs reapplication based on its duration.

        Reapply ~6s (1 tick) before the DoT expires.
        Uses the spell's actual duration instead of a hardcoded interval.
        """
        if not spell or not spell.spell_id:
            return False
        elapsed = self.time_since_spell(spell.spell_id)
        # SpellData.duration_seconds = ticks * 6. Reapply 1 tick early.
        # Fallback to 30s if spell has no duration data.
        from eq.loadout import get_spell_db

        sd = get_spell_db().get(spell.spell_id)
        if sd and sd.duration_ticks > 0:
            recast_at = sd.duration_seconds - 6.0
        else:
            recast_at = 30.0  # safe fallback
        needs_recast: bool = elapsed > recast_at
        return needs_recast
