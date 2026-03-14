"""Persistent per-zone knowledge  -  dispositions and social threat groups.

Learns NPC interaction outcomes (disposition, social relationships) and
persists them across sessions to ``data/knowledge/<zone>.json``.
Zone config files become optional seeds rather than the only source.

On load, **config seed data is merged underneath** learned data:
learned dispositions override config for the same npc, but config entries
for npcs never encountered at runtime are preserved.  Same for social
groups  -  learned associations augment config groups.
"""

import json
import logging
import time
from pathlib import Path

from core.types import Disposition
from core.types import normalize_entity_name as normalize_mob_name

log = logging.getLogger(__name__)

# Learned dispositions older than this are pruned on save.
_DISPOSITION_MAX_AGE_S = 7 * 24 * 3600  # 7 days


class ZoneKnowledge:
    """Persistent knowledge about a zone, learned from gameplay."""

    def __init__(
        self,
        zone_name: str,
        data_dir: str | Path = "data/knowledge",
        toml_dispositions: dict[str, list[str]] | None = None,
        toml_social_groups: list[list[str]] | None = None,
    ) -> None:
        self.zone_name = zone_name
        self._data_dir = Path(data_dir)

        # Learned dispositions: {mob_display_name: {disp, time}}
        self._dispositions: dict[str, dict] = {}

        # Learned social groups: list of sets of npc base names
        self._social_groups: list[set[str]] = []

        # TOML seed data (read-only reference)
        self._toml_dispositions = toml_dispositions or {}
        self._toml_social_groups = toml_social_groups or []

        self._dirty = False
        self._load()

    # -- Disposition API ------------------------------------------------

    def record_disposition(self, mob_name: str, disp: Disposition) -> bool:
        """Record a learned disposition.  Returns True if this is new/changed."""
        existing = self._dispositions.get(mob_name)
        if existing and existing["disp"] == disp:
            # Same disposition  -  just refresh timestamp
            existing["time"] = time.time()
            return False

        self._dispositions[mob_name] = {
            "disp": disp,
            "time": time.time(),
        }
        self._dirty = True
        log.info("[ZONE] %s -> %s (learned)", mob_name, disp)
        return True

    def get_merged_dispositions(self) -> dict[str, list[str]]:
        """Return disposition dict merging TOML seed + learned data.

        Format matches zone TOML: ``{disposition_name: [mob_prefixes]}``.
        Learned entries override TOML for the same npc.
        """
        # Start with a copy of TOML seed
        merged: dict[str, list[str]] = {}
        for disp_name, prefixes in self._toml_dispositions.items():
            merged[disp_name] = list(prefixes)

        # Collect npcs that have learned dispositions
        learned_mobs: set[str] = set()
        for mob_name, entry in self._dispositions.items():
            disp_name = entry["disp"].lower()
            learned_mobs.add(mob_name)
            if disp_name not in merged:
                merged[disp_name] = []
            if mob_name not in merged[disp_name]:
                merged[disp_name].append(mob_name)

        # Remove learned npcs from TOML categories if they've moved
        # (e.g., TOML says indifferent but we learned scowling)
        for mob_name in learned_mobs:
            learned_disp = self._dispositions[mob_name]["disp"].lower()
            for disp_name, prefixes in merged.items():
                if disp_name != learned_disp and mob_name in prefixes:
                    prefixes.remove(mob_name)

        return merged

    @property
    def learned_disposition_count(self) -> int:
        return len(self._dispositions)

    # -- Social Threat API -----------------------------------------------

    def record_social_add(self, pull_target: str, add_mob: str) -> bool:
        """Record that *add_mob* assisted *pull_target* during combat.

        Merges into existing groups or creates a new one.
        Returns True if a new association was learned.
        """
        a = _base_name(pull_target)
        b = _base_name(add_mob)
        if a == b:
            return False  # same npc type, not a social add

        # Check if both are already in the same group
        for group in self._social_groups:
            if a in group and b in group:
                return False  # already known

        # Find groups containing either npc
        group_a = next((g for g in self._social_groups if a in g), None)
        group_b = next((g for g in self._social_groups if b in g), None)

        if group_a is not None and group_b is not None:
            # Merge two groups
            group_a.update(group_b)
            self._social_groups.remove(group_b)
        elif group_a is not None:
            group_a.add(b)
        elif group_b is not None:
            group_b.add(a)
        else:
            # New group
            self._social_groups.append({a, b})

        self._dirty = True
        log.info("[ZONE] social add learned: %s + %s", a, b)
        return True

    def get_merged_social_groups(self) -> list[list[str]]:
        """Return social groups merging TOML seed + learned data.

        Each group is a list of npc base names.
        """
        # Start with learned groups
        merged: list[set[str]] = [set(g) for g in self._social_groups]

        # Merge TOML groups
        for toml_group in self._toml_social_groups:
            toml_set = set(toml_group)
            # Find if any learned group overlaps
            found = False
            for learned in merged:
                if learned & toml_set:
                    learned.update(toml_set)
                    found = True
                    break
            if not found:
                merged.append(toml_set)

        return [sorted(g) for g in merged]

    def build_social_mob_group(self) -> dict[str, frozenset[str]]:
        """Build the lookup dict used by acquire/combat: name -> all members."""
        groups = self.get_merged_social_groups()
        lookup: dict[str, frozenset[str]] = {}
        for group in groups:
            members = frozenset(group)
            for name in group:
                lookup[name] = members
        return lookup

    @property
    def learned_social_group_count(self) -> int:
        return len(self._social_groups)

    # -- Persistence ----------------------------------------------------

    def save(self) -> None:
        """Save learned knowledge to disk."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        path = self._data_dir / f"{self.zone_name}.json"

        # Prune old dispositions
        cutoff = time.time() - _DISPOSITION_MAX_AGE_S
        self._dispositions = {k: v for k, v in self._dispositions.items() if v["time"] > cutoff}

        data = {
            "v": 1,
            "zone": self.zone_name,
            "saved": time.time(),
            "dispositions": self._dispositions,
            "social_groups": [sorted(g) for g in self._social_groups],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info(
            "[ZONE] saved %d dispositions, %d social groups to %s",
            len(self._dispositions),
            len(self._social_groups),
            path.name,
        )
        self._dirty = False

    def _load(self) -> None:
        """Load learned knowledge from disk."""
        path = self._data_dir / f"{self.zone_name}.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            if "v" not in data:
                log.info("[ZONE] no schema version in %s (pre-v1)", path.name)
            self._dispositions = data.get("dispositions", {})
            raw_groups = data.get("social_groups", [])
            self._social_groups = [set(g) for g in raw_groups]
            log.info(
                "[ZONE] loaded %d dispositions, %d social groups from %s",
                len(self._dispositions),
                len(self._social_groups),
                path.name,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning("[ZONE] failed to load %s: %s", path, e)

    @property
    def dirty(self) -> bool:
        return self._dirty


def _base_name(spawn_name: str) -> str:
    """Strip trailing digits and normalize: 'a_spiderling017' -> 'a_spiderling'."""
    result: str = normalize_mob_name(spawn_name)
    return result
