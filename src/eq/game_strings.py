"""Parse EQ game string databases (eqstr_us.txt, dbstr_us.txt).

Game string databases removed from public release.

These files contain every text string the game client displays -- combat messages,
UI text, spell descriptions, race/class names, error messages, etc. Parsing them
gives us the exact message templates the game uses, enabling bulletproof log parsing
and spell/item description lookups.

File formats:
  eqstr_us.txt -- "EQST0002" header, then "ID MessageText" per line (5,900+ entries)
  dbstr_us.txt -- "ID^Type^Value" caret-delimited, type codes for categories (8,300+ entries)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# -- eqstr_us.txt: Game Message Strings ----------------------------------------
# These are the templates for all game messages. Some use %1, %2, @1 for
# substitution (player names, damage values, spell names, etc.)


# Well-known message IDs (verified against eqstr_us.txt)
class MsgID:
    """Known game message IDs from eqstr_us.txt."""

    pass


@dataclass(slots=True)
class GameStrings:
    """All game message strings from eqstr_us.txt."""

    _by_id: dict[int, str] = field(default_factory=dict)

    def get(self, msg_id: int) -> str | None:
        return self._by_id.get(msg_id)

    def search(self, text: str) -> list[tuple[int, str]]:
        """Search for messages containing text (case-insensitive)."""
        text_lower = text.lower()
        return [(mid, msg) for mid, msg in self._by_id.items() if text_lower in msg.lower()]

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, msg_id: int) -> bool:
        return msg_id in self._by_id


def load_game_strings(path: str | Path) -> GameStrings:
    """Load game message strings from eqstr_us.txt.

    Stubbed in public release -- returns an empty GameStrings.
    """
    return GameStrings()


# -- dbstr_us.txt: Database Strings --------------------------------------------
# Format: "ID^Type^Value" -- type codes determine the category.
#
# Type distribution (from actual file):
#   Type 1  (866 entries): AA ability names
#   Type 2  (474 entries): AA short descriptions
#   Type 3  (395 entries): AA category labels
#   Type 4  (865 entries): AA long descriptions
#   Type 5  (130 entries): Zone/expansion names, spell category names
#   Type 6  (4606 entries): Spell descriptions (by spell ID)
#   Type 7  (2 entries): Special labels ("Epic Weapons", "Wayfarer's Emblems")
#   Type 11 (475 entries): Race names (singular)
#   Type 12 (475 entries): Race names (plural) / class names
#   Type 13 (17 entries): Class names


class DBStrType:
    """Type codes in dbstr_us.txt."""

    AA_NAME = 1
    AA_SHORT_DESC = 2
    AA_CATEGORY = 3
    AA_LONG_DESC = 4
    CATEGORY_NAME = 5
    SPELL_DESCRIPTION = 6
    SPECIAL = 7
    RACE_SINGULAR = 11
    RACE_PLURAL = 12
    CLASS_NAME = 13


@dataclass(frozen=True, slots=True)
class DBStrEntry:
    """A single entry from dbstr_us.txt."""

    id: int
    type: int
    value: str


@dataclass(slots=True)
class DatabaseStrings:
    """All database strings from dbstr_us.txt."""

    _entries: dict[tuple[int, int], str] = field(default_factory=dict)  # (id, type) -> value
    _by_type: dict[int, dict[int, str]] = field(default_factory=dict)  # type -> {id: value}

    def get(self, entry_id: int, entry_type: int) -> str | None:
        return self._entries.get((entry_id, entry_type))

    def by_type(self, entry_type: int) -> dict[int, str]:
        return self._by_type.get(entry_type, {})

    def spell_description(self, spell_id: int) -> str | None:
        """Get spell description by spell ID."""
        return self._entries.get((spell_id, DBStrType.SPELL_DESCRIPTION))

    def race_name(self, race_id: int) -> str | None:
        """Get race name by race ID."""
        return self._entries.get((race_id, DBStrType.RACE_SINGULAR))

    def class_name(self, class_id: int) -> str | None:
        """Get class name by class ID."""
        return self._entries.get((class_id, DBStrType.CLASS_NAME))

    def aa_name(self, aa_id: int) -> str | None:
        return self._entries.get((aa_id, DBStrType.AA_NAME))

    def search(self, text: str, entry_type: int | None = None) -> list[DBStrEntry]:
        """Search entries by text (case-insensitive), optionally filtered by type."""
        text_lower = text.lower()
        results = []
        for (eid, etype), value in self._entries.items():
            if entry_type is not None and etype != entry_type:
                continue
            if text_lower in value.lower():
                results.append(DBStrEntry(id=eid, type=etype, value=value))
        return results

    def __len__(self) -> int:
        return len(self._entries)


def load_database_strings(path: str | Path) -> DatabaseStrings:
    """Load database strings from dbstr_us.txt.

    Stubbed in public release -- returns an empty DatabaseStrings.
    """
    return DatabaseStrings()
