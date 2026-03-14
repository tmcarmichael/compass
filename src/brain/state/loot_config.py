"""NPC knowledge base and configurable resource target policy."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, kw_only=True)
class LootConfig:
    """NPC knowledge and resource target state. Mostly write-once at startup."""

    undead_names: set[str] = field(default_factory=set)
    caster_mob_names: set[str] = field(default_factory=set)
    mob_loot_values: dict[str, float] = field(default_factory=dict)
    resource_targets: dict[str, str] = field(default_factory=dict)
    resource_item_count: int = 0
    resource_item_target: int = 0
    resource_item_session_start: int = 0  # set at startup from inventory scan
