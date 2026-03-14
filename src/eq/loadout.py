"""Spell loadout management  -  gem assignment, role lookup, runtime configuration.

Class-specific spell priority tables removed from public release.

Bridges the spell data model (eq.spells) to agent behavior. Routines and rules
import get_spell_by_role() to check spell availability; the runtime calls
configure_loadout() at startup and on level-up to populate the gem mapping.

This is automation support, not data parsing  -  separated from eq.spells
(the data model layer) to keep that boundary explicit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from eq.spells import SpellData, SpellDB, SpellRole

log = logging.getLogger(__name__)

# EQ always has 8 spell gem slots at every level
NUM_SPELL_GEMS = 8


def gems_for_level(level: int) -> int:
    """How many spell gem slots a character has. Always 8 in EQ."""
    return NUM_SPELL_GEMS


# -- Class priority tables ----------------------------------------------------
# Maps class_id -> ordered list of (spell_name, role).
# GemPlanner assigns gems in this priority order: first available spell wins
# within each role. Roles listed first get gems first.

_CLASS_SPELL_PRIORITIES: dict[int, list[tuple[str, str]]] = {}


# -- Spell reference (what routines import) -----------------------------------


@dataclass(slots=True)
class Spell:
    """A spell in the loadout. Mutable  -  gem updates on level-up."""

    name: str
    gem: int  # gem slot (1-8), 0 = not memorized
    cast_time: float  # seconds
    mana_cost: int
    spell_id: int = 0
    recast: float = 0.0
    role: str = ""
    est_damage: float = 0.0  # estimated total damage (for efficiency ranking)
    spell_range: int = 200  # EQ range units (0=self, 10=touch, 200=ranged)

    def __bool__(self) -> bool:
        """Truthy when memorized (gem > 0). Lets routines do `if DISEASE_CLOUD:`"""
        return self.gem > 0

    @property
    def mana_efficiency(self) -> float:
        """Damage per mana point. Higher = more efficient."""
        if self.mana_cost <= 0 or self.est_damage <= 0:
            return 0.0
        return self.est_damage / self.mana_cost

    @property
    def dps_per_mana(self) -> float:
        """DPS per mana point (factors cast time). Higher = faster damage per mana."""
        if self.mana_cost <= 0 or self.est_damage <= 0 or self.cast_time <= 0:
            return 0.0
        dps = self.est_damage / self.cast_time
        return dps / self.mana_cost


# -- Active loadout (module-level, updated by configure_loadout) --------------

LESSER_SHIELDING = Spell(
    "Lesser Shielding", gem=0, cast_time=2.5, mana_cost=25, spell_id=246, role=SpellRole.SELF_BUFF
)
MINOR_SHIELDING = Spell(
    "Minor Shielding", gem=0, cast_time=2.5, mana_cost=10, spell_id=288, role=SpellRole.SELF_BUFF
)
LIFETAP = Spell(
    "Lifetap", gem=0, cast_time=1.5, mana_cost=9, spell_id=341, role=SpellRole.LIFETAP, est_damage=20.0
)
LIFESPIKE = Spell(
    "Lifespike", gem=0, cast_time=1.75, mana_cost=18, spell_id=502, role=SpellRole.LIFETAP, est_damage=45.0
)
CAVORTING_BONES = Spell(
    "Cavorting Bones", gem=0, cast_time=5.0, mana_cost=15, spell_id=338, role=SpellRole.PET_SUMMON
)
BONE_WALK = Spell("Bone Walk", gem=0, cast_time=7.0, mana_cost=80, spell_id=351, role=SpellRole.PET_SUMMON)
CONVOKE_SHADOW = Spell(
    "Convoke Shadow", gem=0, cast_time=8.0, mana_cost=120, spell_id=362, role=SpellRole.PET_SUMMON
)
DISEASE_CLOUD = Spell(
    "Disease Cloud", gem=0, cast_time=1.5, mana_cost=10, spell_id=340, role=SpellRole.DOT, est_damage=36.0
)
POISON_BOLT = Spell(
    "Poison Bolt", gem=0, cast_time=1.75, mana_cost=30, spell_id=348, role=SpellRole.DD, est_damage=42.0
)
DARK_PACT = Spell("Dark Pact", gem=0, cast_time=3.0, mana_cost=5, spell_id=641, role=SpellRole.MANA_REGEN)
MEND_BONES = Spell(
    "Mend Bones", gem=0, cast_time=3.5, mana_cost=25, spell_id=353, recast=7.0, role=SpellRole.PET_HEAL
)
GATHER_SHADOWS = Spell(
    "Gather Shadows", gem=0, cast_time=5.0, mana_cost=35, spell_id=522, role=SpellRole.INVIS
)

# All spell objects by name (lowercase) for loadout updates
_ALL_SPELLS: dict[str, Spell] = {
    s.name.lower(): s
    for s in [
        LESSER_SHIELDING,
        MINOR_SHIELDING,
        LIFETAP,
        LIFESPIKE,
        CAVORTING_BONES,
        BONE_WALK,
        CONVOKE_SHADOW,
        DISEASE_CLOUD,
        POISON_BOLT,
        DARK_PACT,
        MEND_BONES,
        GATHER_SHADOWS,
    ]
}

# Role-based lookup for routines that need "give me the active pet summon"
_ACTIVE_BY_ROLE: dict[str, Spell] = {}


def get_spell_by_role(role: str) -> Spell | None:
    """Get the currently active spell for a role, or None."""
    return _ACTIVE_BY_ROLE.get(role)


def rank_damage_spells() -> list[Spell]:
    """Return memorized damage spells sorted by mana efficiency (best first)."""
    damage_roles = {SpellRole.DOT, SpellRole.DD, SpellRole.LIFETAP}
    spells = [s for s in _ALL_SPELLS.values() if s.gem > 0 and s.est_damage > 0 and s.role in damage_roles]
    return sorted(spells, key=lambda s: s.mana_efficiency, reverse=True)


def _upsert_spell(spell_name: str, sd: SpellData, role: str) -> Spell:
    """Get or create a Spell in _ALL_SPELLS and sync fields from SpellData."""
    spell_obj = _ALL_SPELLS.get(spell_name.lower())
    if spell_obj is None:
        spell_obj = Spell(
            name=sd.name,
            gem=0,
            cast_time=sd.cast_time,
            mana_cost=sd.mana_cost,
            spell_id=sd.id,
            recast=sd.recast_time,
            role=role,
        )
        _ALL_SPELLS[sd.name.lower()] = spell_obj
    spell_obj.cast_time = sd.cast_time
    spell_obj.mana_cost = sd.mana_cost
    spell_obj.spell_id = sd.id
    spell_obj.recast = sd.recast_time
    spell_obj.spell_range = sd.range
    spell_obj.role = role
    return spell_obj


def _find_gate_entry(
    priorities: list[tuple[str, str]],
    available: dict[str, SpellData],
    scribed_ids: set[int] | None,
) -> tuple[str, SpellData] | None:
    """Find the gate spell entry from priorities, if available and scribed."""
    for spell_name, role in priorities:
        if role == SpellRole.GATE:
            sd = available.get(spell_name.lower())
            if sd and (scribed_ids is None or sd.id in scribed_ids):
                return (spell_name, sd)
            break
    return None


def configure_loadout(
    class_id: int, level: int, db: SpellDB | None = None, scribed_ids: set[int] | None = None
) -> dict[int, str]:
    """Auto-configure spell gems for a class + level.

    Returns: {gem_slot: spell_name} for logging.
    """
    if db is None:
        db = get_spell_db()

    for spell in _ALL_SPELLS.values():
        spell.gem = 0
    _ACTIVE_BY_ROLE.clear()

    priorities = _CLASS_SPELL_PRIORITIES.get(class_id, [])
    if not priorities:
        log.warning("[CAST] No spell priorities for class %d  -  no gems assigned", class_id)
        return {}

    num_gems = gems_for_level(level)
    available = {s.name.lower(): s for s in db.available_for(class_id, level)}

    assigned: dict[int, str] = {}
    used_roles: set[str] = set()
    skipped_for_role: dict[str, str] = {}
    next_gem = 1

    # Gate always in gem 8 (reserved last slot)
    gate_entry = _find_gate_entry(priorities, available, scribed_ids)
    combat_gem_limit = num_gems - 1 if gate_entry else num_gems

    for spell_name, role in priorities:
        if next_gem > combat_gem_limit:
            break
        if role == SpellRole.GATE:
            continue
        if role in used_roles:
            continue

        sd = available.get(spell_name.lower())
        if sd is None:
            continue

        if scribed_ids is not None and sd.id not in scribed_ids:
            if role not in skipped_for_role:
                skipped_for_role[role] = sd.name
            continue

        spell_obj = _upsert_spell(spell_name, sd, role)
        spell_obj.gem = next_gem

        assigned[next_gem] = sd.name
        used_roles.add(role)
        _ACTIVE_BY_ROLE[role] = spell_obj
        next_gem += 1

    if gate_entry:
        gate_name, gate_sd = gate_entry
        gate_obj = _upsert_spell(gate_name, gate_sd, SpellRole.GATE)
        gate_obj.gem = num_gems
        assigned[num_gems] = gate_sd.name
        _ACTIVE_BY_ROLE[SpellRole.GATE] = gate_obj

    log.info(
        "[CAST] Loadout for class=%d level=%d (%d gems): %s",
        class_id,
        level,
        num_gems,
        ", ".join(f"gem{g}={n}" for g, n in sorted(assigned.items())),
    )

    for role, ideal_name in skipped_for_role.items():
        actual = _ACTIVE_BY_ROLE.get(role)
        if actual:
            log.warning(
                "[CAST] \033[93mMISSING SPELL: '%s' (%s) not scribed  -  using '%s' instead\033[0m",
                ideal_name,
                role,
                actual.name,
            )
        else:
            log.warning(
                "[CAST] \033[93mMISSING SPELL: '%s' (%s) not scribed  -  "
                "no fallback available for this role!\033[0m",
                ideal_name,
                role,
            )

    return assigned


def compute_desired_loadout(
    class_id: int, level: int, db: SpellDB | None = None, scribed_ids: set[int] | None = None
) -> dict[int, int]:
    """Compute desired spell gems without modifying global state.

    Returns: {gem_slot: spell_id} for the achievable loadout.
    """
    if db is None:
        db = get_spell_db()

    priorities = _CLASS_SPELL_PRIORITIES.get(class_id, [])
    if not priorities:
        return {}

    num_gems = gems_for_level(level)
    available = {s.name.lower(): s for s in db.available_for(class_id, level)}

    result: dict[int, int] = {}
    used_roles: set[str] = set()
    next_gem = 1

    gate_sd = None
    for spell_name, role in priorities:
        if role == SpellRole.GATE:
            sd = available.get(spell_name.lower())
            if sd:
                gate_sd = sd
            break
    combat_gem_limit = num_gems - 1 if gate_sd else num_gems

    for spell_name, role in priorities:
        if next_gem > combat_gem_limit:
            break
        if role == SpellRole.GATE:
            continue
        if role in used_roles:
            continue
        sd = available.get(spell_name.lower())
        if sd is None:
            continue
        if scribed_ids is not None and sd.id not in scribed_ids:
            continue
        result[next_gem] = sd.id
        used_roles.add(role)
        next_gem += 1

    if gate_sd:
        result[num_gems] = gate_sd.id

    return result


def check_spell_loadout(
    memorized: dict[int, int], class_id: int, level: int, db: SpellDB | None = None
) -> list[dict]:
    """Compare current memorized spells to desired loadout.

    Returns list of changes needed. Empty list = loadout is correct.
    """
    if db is None:
        db = get_spell_db()

    desired = compute_desired_loadout(class_id, level, db)
    actual_ids = set(memorized.values())
    desired_ids = set(desired.values())

    missing = desired_ids - actual_ids
    if not missing:
        log.info("[CAST] Spell check: loadout OK  -  all %d desired spells memorized", len(desired_ids))
        return []

    extra = actual_ids - desired_ids
    changes: list[dict] = []

    log.info("[CAST] Spell check: MISMATCH detected")
    log.info(
        "[CAST]   Current gems: %s",
        ", ".join(
            f"gem{g}={sd.name if (sd := db.get(sid)) else sid}" for g, sid in sorted(memorized.items())
        ),
    )
    log.info(
        "[CAST]   Desired gems: %s",
        ", ".join(f"gem{g}={sd.name if (sd := db.get(sid)) else sid}" for g, sid in sorted(desired.items())),
    )

    for spell_id in missing:
        sd = db.get(spell_id)
        name = sd.name if sd else f"spell_{spell_id}"
        log.info("[CAST]   MISSING: %s (id=%d)", name, spell_id)
        changes.append({"gem": 0, "action": "memorize", "spell_id": spell_id, "spell_name": name})

    for spell_id in extra:
        sd = db.get(spell_id)
        name = sd.name if sd else f"spell_{spell_id}"
        gem = next((g for g, sid in memorized.items() if sid == spell_id), 0)
        log.info("[CAST]   EXTRA: %s (id=%d) in gem %d", name, spell_id, gem)
        changes.append({"gem": gem, "action": "clear", "spell_id": spell_id, "spell_name": name})

    return changes


def configure_from_memory(
    memorized: dict[int, int], class_id: int, db: SpellDB | None = None
) -> dict[int, str]:
    """Configure spell loadout from actual memorized gems read from memory.

    Returns: {gem_slot: spell_name} for logging.
    """
    if db is None:
        db = get_spell_db()

    for spell in _ALL_SPELLS.values():
        spell.gem = 0
    _ACTIVE_BY_ROLE.clear()

    priorities = _CLASS_SPELL_PRIORITIES.get(class_id, [])
    name_to_role = {name.lower(): role for name, role in priorities}

    assigned: dict[int, str] = {}

    for gem_slot, spell_id in sorted(memorized.items()):
        sd = db.get(spell_id)
        if sd is None:
            log.warning("[CAST] Gem %d: unknown spell ID %d", gem_slot, spell_id)
            continue

        spell_obj = _ALL_SPELLS.get(sd.name.lower())
        if spell_obj is None:
            role = name_to_role.get(sd.name.lower(), SpellRole.UTILITY)
            spell_obj = Spell(
                name=sd.name,
                gem=0,
                cast_time=sd.cast_time,
                mana_cost=sd.mana_cost,
                spell_id=sd.id,
                recast=sd.recast_time,
                role=role,
            )
            _ALL_SPELLS[sd.name.lower()] = spell_obj

        spell_obj.cast_time = sd.cast_time
        spell_obj.mana_cost = sd.mana_cost
        spell_obj.spell_id = sd.id
        spell_obj.recast = sd.recast_time
        spell_obj.spell_range = sd.range
        spell_obj.gem = gem_slot
        if not spell_obj.role:
            spell_obj.role = name_to_role.get(sd.name.lower(), SpellRole.UTILITY)

        if spell_obj.role and spell_obj.role not in _ACTIVE_BY_ROLE:
            _ACTIVE_BY_ROLE[spell_obj.role] = spell_obj

        assigned[gem_slot] = sd.name

    log.info(
        "[CAST] Loadout from memory (%d gems): %s",
        len(assigned),
        ", ".join(f"gem{g}={n}" for g, n in sorted(assigned.items())),
    )

    return assigned


# -- Global spell database ----------------------------------------------------

_db: SpellDB | None = None


def get_spell_db(client_path: str | None = None) -> SpellDB:
    """Get the global spell database, loading from client_path or defaults."""
    global _db
    if _db is None:
        _db = SpellDB()
        if client_path:
            spell_file = Path(client_path) / "spells_us.txt"
            if spell_file.exists():
                _db.load(spell_file)
        if not _db._by_id:
            log.warning("[LIFECYCLE] Spell database is empty (no spells_us.txt found)")
    return _db
