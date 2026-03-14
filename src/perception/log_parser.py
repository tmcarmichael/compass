"""Game event log tailing and parsing.

Watches the client log file for:
- Faction standing messages from /consider
- Zone change messages ("You have entered ...")
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import TextIO

from core.types import Disposition

log = logging.getLogger(__name__)


# /consider faction message patterns -> Disposition
# Format: "[timestamp] <npc name> <faction phrase> -- <con phrase>"
FACTION_PATTERNS: list[tuple[re.Pattern[str], Disposition]] = [
    (re.compile(r"^(.+?) regards you as an ally"), Disposition.ALLY),
    (re.compile(r"^(.+?) looks upon you warmly"), Disposition.WARMLY),
    (re.compile(r"^(.+?) kindly considers you"), Disposition.KINDLY),
    (re.compile(r"^(.+?) judges you amiably"), Disposition.AMIABLE),
    (re.compile(r"^(.+?) regards you indifferently"), Disposition.INDIFFERENT),
    (re.compile(r"^(.+?) looks your way apprehensively"), Disposition.APPREHENSIVE),
    (re.compile(r"^(.+?) glowers at you dubiously"), Disposition.DUBIOUS),
    (re.compile(r"^(.+?) glares at you threateningly"), Disposition.THREATENING),
    (re.compile(r"^(.+?) scowls at you, ready to attack"), Disposition.READY_TO_ATTACK),
    (re.compile(r"^(.+?) scowls at you, what would you like your tombstone to say"), Disposition.SCOWLING),
]

# Zone change: "You have entered Nektulos Forest."
ZONE_RE = re.compile(r"^You have entered (.+)\.$")

# XP gain: "You gain experience!!" or "You gain party experience!!"
XP_RE = re.compile(r"^You gain (?:party )?experience!!")

# Death: "You have been slain by <npc>!" or "You died."
DEATH_RE = re.compile(r"^(?:You have been slain by .+!|You died\.)")

# Unconscious / low HP warning (EQ emits this at ~20% HP)
STUNNED_RE = re.compile(r"^You have been stunned\.")

# Food/drink warnings
FOOD_DRINK_RE = re.compile(r"^You (?:are (low on|out of) (food|drink))\.")

# Spell fizzle: "Your spell fizzles!"
FIZZLE_RE = re.compile(r"^Your spell fizzles!")

# LOS block: "You cannot see your target."
LOS_BLOCK_RE = re.compile(r"^You (?:cannot|can't) see your target")

# Spell interrupted: "Your spell is interrupted." (movement or hit during cast)
SPELL_INTERRUPTED_RE = re.compile(r"^Your (?:spell is interrupted|casting is interrupted)")

# Must be standing: "You must be standing to cast a spell."
MUST_STAND_RE = re.compile(r"^You must be standing")

# Npc attacking pet: "A spiderling tries to bite Varn, but misses!"
# or "A spiderling hits Varn for 3 points of damage."
# Captures (attacker_name, pet_name)
MOB_ATTACKS_PET_RE = re.compile(
    r"^(.+?) (?:tries to \w+|hits|crushes|slashes|bashes|claws|bites|pierces|kicks|strikes|mauls|gores|stings) (.+?)(?:,| for )"
)

# Strip the timestamp prefix: [Day Mon DD HH:MM:SS YYYY]
TIMESTAMP_RE = re.compile(r"^\[.*?\] ")


@dataclass(slots=True)
class LogEvents:
    """Parsed events from a single poll of the log file."""

    dispositions: dict[str, Disposition] = field(default_factory=dict)
    zone_name: str | None = None  # display name (e.g. "Nektulos Forest")
    zone_short: str | None = None  # short name (e.g. "nektulos")
    xp_gained: int = 0  # number of XP messages seen
    pet_attackers: set[str] = field(default_factory=set)  # npc names attacking our pet
    player_died: bool = False  # "You have been slain" detected
    player_stunned: bool = False  # "You have been stunned" detected
    food_drink_warnings: list[str] = field(default_factory=list)  # e.g. "low on food"
    spell_fizzled: bool = False  # "Your spell fizzles!"
    los_blocked: bool = False  # "You cannot see your target."
    spell_interrupted: bool = False  # "Your spell is interrupted."
    must_be_standing: bool = False  # "You must be standing to cast a spell."


class LogParser:
    """Tails the game log file and parses events."""

    def __init__(self, log_path: str, pet_names: set[str] | None = None) -> None:
        self._path = log_path
        self._file: TextIO | None = None
        # Known pet name prefixes for detecting "npc attacks pet" messages
        self._pet_names: set[str] = pet_names or set()
        self._open()

    def _open(self) -> None:
        """Open the log file and seek to the end (only read new lines)."""
        try:
            f = open(self._path, encoding="utf-8", errors="replace")
            self._file = f
            f.seek(0, os.SEEK_END)
            log.info("[PERCEPTION] LogParser tailing %s (pos=%d)", self._path, f.tell())
        except OSError as e:
            log.warning("[PERCEPTION] Could not open log file %s: %s", self._path, e)
            self._file = None

    def _parse_line(self, body: str, events: LogEvents) -> None:
        """Dispatch a single log line body into the appropriate event field."""
        # Zone change
        if zm := ZONE_RE.match(body):
            events.zone_name = zm.group(1)
            from nav.zone_graph import normalize_zone_name

            events.zone_short = normalize_zone_name(zm.group(1))
            return

        # Simple flag matches
        if XP_RE.match(body):
            events.xp_gained += 1
            return
        if DEATH_RE.match(body):
            events.player_died = True
            return
        if STUNNED_RE.match(body):
            events.player_stunned = True
            return
        if FIZZLE_RE.match(body):
            events.spell_fizzled = True
            return
        if LOS_BLOCK_RE.match(body):
            events.los_blocked = True
            return
        if SPELL_INTERRUPTED_RE.match(body):
            events.spell_interrupted = True
            return
        if MUST_STAND_RE.match(body):
            events.must_be_standing = True
            return

        # Food/drink warnings
        if fd := FOOD_DRINK_RE.match(body):
            events.food_drink_warnings.append(f"{fd.group(1)} {fd.group(2)}")
            return

        # Npc attacking pet (social threat detection)
        if self._pet_names:
            if am := MOB_ATTACKS_PET_RE.match(body):
                attacker = am.group(1)
                victim = am.group(2)
                if victim in self._pet_names:
                    events.pet_attackers.add(attacker)

        # Faction /consider
        for pattern, disposition in FACTION_PATTERNS:
            if fm := pattern.match(body):
                mob_name = fm.group(1).strip()
                events.dispositions[mob_name] = disposition
                break

    def poll(self) -> LogEvents:
        """Read new log lines and return parsed events.

        Returns a LogEvents with any /consider dispositions and zone changes
        found since the last poll.
        """
        events = LogEvents()
        if self._file is None:
            return events

        try:
            for line in self._file:
                line = line.rstrip("\n\r")
                if not line:
                    continue

                # Strip timestamp
                if not (m := TIMESTAMP_RE.match(line)):
                    continue
                self._parse_line(line[m.end() :], events)
        except OSError:
            self._file = None

        return events

    def poll_dispositions(self) -> dict[str, Disposition]:
        """Convenience: poll and return only dispositions."""
        return self.poll().dispositions

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
