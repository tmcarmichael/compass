"""Zone configuration state: dispositions, social groups, camps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.types import Con

if TYPE_CHECKING:
    from brain.learning.zone import ZoneKnowledge


@dataclass(slots=True, kw_only=True)
class ZoneState:
    """Per-zone config loaded at startup or zone change."""

    zone_config: dict[str, Any] = field(default_factory=dict)
    zone_dispositions: dict[str, list[str]] | None = None
    social_mob_group: dict[str, frozenset[str]] = field(default_factory=dict)
    zone_camps: list[dict[str, Any]] = field(default_factory=list)
    active_camp_name: str = ""
    zone_knowledge: ZoneKnowledge | None = None
    target_cons: frozenset[Con] = field(default_factory=lambda: frozenset({Con.BLUE, Con.LIGHT_BLUE}))
