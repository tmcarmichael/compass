"""Inventory tracking (weight-based + item scan)."""

from __future__ import annotations

from dataclasses import dataclass, field

from perception.state import GameState


@dataclass(slots=True, kw_only=True)
class InventoryState:
    """Weight-based inventory tracking for encumbrance and item accounting."""

    weight_baseline: int = 0
    weight_threshold: int = 80
    loot_count: int = 0

    # Item tracking from periodic read_inventory() scans
    items: list[tuple[str, int, int]] = field(default_factory=list)
    last_scan_time: float = 0.0

    def weight_gained(self, state: GameState) -> int:
        if self.weight_baseline == 0:
            return 0
        gained: int = max(0, state.weight - self.weight_baseline)
        return gained

    def is_encumbered(self, state: GameState) -> bool:
        return self.weight_gained(state) >= self.weight_threshold

    def count_items_matching(self, name: str) -> int:
        """Count inventory items whose name contains the given substring."""
        return sum(count for item_name, iid, count in self.items if name.lower() in item_name.lower())

    def slots_used(self) -> int:
        """Number of occupied inventory slots."""
        return len(self.items)

    def update_items(self, items: list[tuple[str, int, int]], timestamp: float) -> None:
        """Update inventory from reader scan."""
        self.items = list(items)
        self.last_scan_time = timestamp
