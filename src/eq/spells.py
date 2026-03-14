"""Spell data model  -  parsed from spells_us.txt.

Spell data field mappings removed from public release.

SpellData    -  frozen dataclass per spell (8,088 entries, class-level requirements,
                effect slots, resist types, target types)
SpellDB      -  in-memory database with class/level/effect queries
SpellRole    -  class-agnostic spell categories
"""

import logging
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path

log = logging.getLogger(__name__)

# -- Class-level field positions in spells_us.txt -----------------------------
# Fields 104-119 (0-indexed) = min level per class. 255 = can't use.
# Class order: WAR CLR PAL RNG SHD DRU MNK BRD ROG SHM NEC WIZ MAG ENC BST BER
_CLASS_LEVEL_FIELD_START = 0
_CLASS_ID_TO_FIELD_OFFSET: dict[int, int] = {}

# -- Spell roles (class-agnostic categories for gem planning) -----------------


class SpellRole(StrEnum):
    PET_SUMMON = "pet_summon"
    DOT = "dot"  # Primary DoT (highest damage available)
    DOT_2 = "dot_2"  # Secondary DoT (stacked on harder npcs)
    LIFETAP = "lifetap"
    DD = "dd"
    SELF_BUFF = "self_buff"
    PET_HEAL = "pet_heal"
    MANA_REGEN = "mana_regen"
    INVIS = "invis"
    FEAR = "fear"
    SNARE = "snare"  # Snare DoTs (Clinging/Dooming/Cascading Darkness)
    GATE = "gate"  # Gate -- always last gem slot (L4+, emergency escape)
    UTILITY = "utility"  # Feign Death, etc.


# -- Resist types (field[85] in spells_us.txt) ---------------------------------


class ResistType(IntEnum):
    UNRESISTABLE = 0
    MAGIC = 1
    FIRE = 2
    COLD = 3
    POISON = 4
    DISEASE = 5
    CHROMATIC = 6
    PRISMATIC = 7
    PHYSICAL = 8
    CORRUPTION = 9

    @classmethod
    def label_for(cls, resist_id: int) -> str:
        try:
            return cls(resist_id)._name_.lower()
        except ValueError:
            return f"unknown({resist_id})"


# -- Target types (field[98] in spells_us.txt) ---------------------------------


class TargetType(IntEnum):
    LINE_OF_SIGHT = 0
    AE_PLAYER_CENTER = 1
    GROUP_V1 = 2
    CASTER_AE = 3
    AE_PLAYER_CENTER_2 = 4
    SINGLE = 5
    SELF = 6
    TARGETED_AE = 8
    PET = 10
    UNDEAD = 11
    LIFETAP = 13
    PET_SUMMON = 14
    CORPSE = 38
    GROUP_V2 = 41

    @classmethod
    def label_for(cls, target_id: int) -> str:
        override = _TARGET_TYPE_LABELS.get(target_id)
        if override:
            return override
        try:
            return cls(target_id)._name_.lower()
        except ValueError:
            return f"unknown({target_id})"


# Display name overrides for TargetType values that share a label
_TARGET_TYPE_LABELS: dict[int, str] = {
    1: "ae_player",
    4: "ae_player",
    2: "group",
    41: "group",
}


# -- Spell Power Areas (SPA) -- effect type IDs --------------------------------
# These identify WHAT each effect slot does. Up to 12 slots per spell.
# 254 = unused slot.


class SPA:
    CURRENT_HP = 0
    ARMOR_CLASS = 1
    ATK = 2
    MOVEMENT_SPEED = 3
    STR = 5
    DEX = 6
    AGI = 7
    STA = 8
    INT = 9
    WIS = 10
    CHA = 11
    ATTACK_SPEED = 12
    INVISIBILITY = 13
    SEE_INVIS = 14
    MANA_REGEN = 15
    DISEASE_COUNTER = 35
    POISON_COUNTER = 36
    MAX_HP = 69
    UNUSED = 254


@dataclass(frozen=True, slots=True)
class SpellData:
    """Parsed spell from spells_us.txt."""

    id: int
    name: str
    range: int  # 0=self, 100=target, 200=ranged
    cast_time_ms: int  # milliseconds
    recovery_ms: int  # recovery time after cast
    recast_ms: int  # recast delay
    duration_ticks: int  # duration in ticks (~6s per tick), 0=instant
    mana_cost: int
    cast_message: str
    cast_on_other: str
    fade_message: str
    class_levels: tuple  # tuple of 16 ints: min level per class (255=can't use)
    # New fields -- verified against reference client spells_us.txt (2026-03-17)
    beneficial: bool  # True=beneficial, False=detrimental (field[83])
    resist_type: int  # ResistType enum (field[85])
    target_type: int  # TargetType enum (field[98])
    effect_ids: tuple  # SPA IDs per slot, 12 ints, 254=unused (fields[86-97])
    base_values: tuple  # base effect values per slot, 12 ints (fields[20-31])
    max_values: tuple  # max effect values per slot, 12 ints (fields[44-55])
    duration_formula: int  # duration calc formula ID (field[16])
    aoe_range: int  # AoE radius (field[10])
    pushback: float  # knockback distance (field[11])

    @property
    def cast_time(self) -> float:
        return self.cast_time_ms / 1000.0

    @property
    def recast_time(self) -> float:
        return self.recast_ms / 1000.0

    @property
    def duration_seconds(self) -> float:
        return self.duration_ticks * 6.0

    @property
    def resist_name(self) -> str:
        return ResistType.label_for(self.resist_type)

    @property
    def target_name(self) -> str:
        return TargetType.label_for(self.target_type)

    @property
    def is_detrimental(self) -> bool:
        return not self.beneficial

    @property
    def is_dot(self) -> bool:
        """True if spell deals damage over time."""
        return (
            self.is_detrimental and self.duration_ticks > 0 and any(v < 0 for v in self.base_values if v != 0)
        )

    @property
    def is_dd(self) -> bool:
        """True if spell deals direct damage (instant, no duration)."""
        return (
            self.is_detrimental
            and self.duration_ticks == 0
            and any(v < 0 for v in self.base_values if v != 0)
        )

    @property
    def is_heal(self) -> bool:
        """True if beneficial spell with positive HP effect."""
        return (
            self.beneficial
            and SPA.CURRENT_HP in self.effect_ids
            and any(v > 0 for v in self.base_values if v != 0)
        )

    @property
    def active_effects(self) -> list[tuple[int, int, int]]:
        """List of (spa_id, base_value, max_value) for non-empty effect slots."""
        results = []
        for i in range(12):
            spa = self.effect_ids[i] if i < len(self.effect_ids) else 254
            if spa == 254:
                continue
            base = self.base_values[i] if i < len(self.base_values) else 0
            maxv = self.max_values[i] if i < len(self.max_values) else 0
            results.append((spa, base, maxv))
        return results

    def min_level_for_class(self, class_id: int) -> int:
        """Min level required for a given class. 255 = can't use."""
        offset = _CLASS_ID_TO_FIELD_OFFSET.get(class_id)
        if offset is None or offset >= len(self.class_levels):
            return 255
        return int(self.class_levels[offset])


class SpellDB:
    """In-memory spell database loaded from spells_us.txt."""

    def __init__(self) -> None:
        self._by_id: dict[int, SpellData] = {}
        self._by_name: dict[str, SpellData] = {}

    def load(self, path: str | Path) -> int:
        """Load spells from spells_us.txt. Returns number loaded.

        Stubbed in public release -- returns 0 (no spells loaded).
        """
        return 0

    def available_for(self, class_id: int, level: int) -> list[SpellData]:
        """All spells a class can use at a given level, sorted by name."""
        offset = _CLASS_ID_TO_FIELD_OFFSET.get(class_id)
        if offset is None:
            return []
        results = []
        for spell in self._by_id.values():
            min_lvl = spell.class_levels[offset] if offset < len(spell.class_levels) else 255
            if 0 < min_lvl <= level:
                results.append(spell)
        results.sort(key=lambda s: s.name)
        return results

    def get(self, spell_id: int) -> SpellData | None:
        return self._by_id.get(spell_id)

    def find(self, name: str) -> SpellData | None:
        return self._by_name.get(name.lower())

    def by_resist_type(self, resist_type: int) -> list[SpellData]:
        """All spells of a given resist type."""
        return [s for s in self._by_id.values() if s.resist_type == resist_type]

    def by_target_type(self, target_type: int) -> list[SpellData]:
        """All spells of a given target type."""
        return [s for s in self._by_id.values() if s.target_type == target_type]

    def by_effect(self, spa_id: int) -> list[SpellData]:
        """All spells that have a given SPA effect."""
        return [s for s in self._by_id.values() if spa_id in s.effect_ids]

    def dots(self) -> list[SpellData]:
        """All DoT spells."""
        return [s for s in self._by_id.values() if s.is_dot]

    def heals(self) -> list[SpellData]:
        """All heal spells."""
        return [s for s in self._by_id.values() if s.is_heal]

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, spell_id: int) -> bool:
        return spell_id in self._by_id
