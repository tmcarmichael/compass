"""InventoryReader: loot window and inventory bag memory reads.

Extracted from reader.py to keep module sizes manageable.
All methods access the parent MemoryReader via self._r for
low-level read primitives (_read_bytes, _read_int32, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from perception import offsets

if TYPE_CHECKING:
    from perception.reader import MemoryReader

log = logging.getLogger(__name__)


class InventoryReader:
    """Reads loot window and inventory items from target process memory.

    Requires a reference to the parent MemoryReader for read primitives.
    """

    def __init__(self, reader: MemoryReader) -> None:
        self._r = reader

    # -- Loot window reads ----------------------------------------------------

    def read_loot_items(self) -> tuple[int, ...]:
        """Read metadata values from CLootWnd+0x164 (NOT actual item IDs).

        These are icon/type indices, not database item IDs. Use
        read_loot_items_deep() for actual item identification.
        Returns non-negative values. () if window closed.
        """
        try:
            loot_wnd = self._r._read_pointer(offsets.LOOT_WND_PTR)
            if loot_wnd == 0:
                return ()
            items = []
            for slot in range(offsets.LOOT_WND_ITEM_SLOTS):
                off = offsets.LOOT_WND_METADATA_OFFSET + slot * 4
                val = self._r._read_int32(loot_wnd + off)
                if val >= 0:  # -1 = empty, 0+ = valid icon index
                    items.append(val)
            return tuple(items)
        except OSError:
            return ()

    def read_loot_items_deep(self) -> list[tuple[int, str, int]]:
        """Read actual item data from loot window via CONTENTS->ITEMINFO chain.

        Returns list of (slot_index, item_name, item_id) for occupied slots.
        Empty list if window closed or pointer chain fails.
        """
        try:
            loot_wnd = self._r._read_pointer(offsets.LOOT_WND_PTR)
            if loot_wnd == 0:
                return []

            items = []
            slot_failures = 0
            for slot in range(offsets.LOOT_WND_ITEM_SLOTS):
                # Read CONTENTS pointer from array
                contents_addr = loot_wnd + offsets.LOOT_WND_CONTENTS_OFFSET + slot * 4
                try:
                    contents_ptr = self._r._read_pointer(contents_addr)
                except OSError:
                    slot_failures += 1
                    continue
                if contents_ptr == 0 or contents_ptr < 0x10000:
                    continue

                # Follow CONTENTS+0x1C -> ITEMINFO pointer
                try:
                    iteminfo_ptr = self._r._read_pointer(contents_ptr + offsets.CONTENTS_ITEMINFO_PTR)
                except OSError:
                    slot_failures += 1
                    continue
                if iteminfo_ptr == 0 or iteminfo_ptr < 0x10000:
                    continue

                # Read item name and ID from ITEMINFO
                try:
                    name = self._r._read_string(iteminfo_ptr + offsets.ITEMINFO_NAME, 64)
                    item_id = self._r._read_uint32(iteminfo_ptr + offsets.ITEMINFO_ITEM_NUMBER)
                    items.append((slot, name, item_id))
                except OSError:
                    slot_failures += 1
                    continue

            if slot_failures > 0:
                log.warning(
                    "[PERCEPTION] Loot chain: %d/%d slots failed to read",
                    slot_failures,
                    offsets.LOOT_WND_ITEM_SLOTS,
                )
            return items
        except OSError:
            return []

    # -- Inventory bag reads --------------------------------------------------

    def _read_bag_items(self, bag_ptr: int) -> list[tuple[str, int, int]]:
        """Read items from a single bag CONTENTS pointer.

        Returns list of (item_name, item_id, stack_count).
        """
        items = []
        for item_idx in range(offsets.CHARINFO_BAG_SLOTS):
            item_off = offsets.CONTENTS_BAG_ITEMS_START + item_idx * 4
            try:
                item_ptr = self._r._read_pointer(bag_ptr + item_off)
            except OSError:
                continue
            if item_ptr == 0 or item_ptr < 0x10000:
                continue

            # Validate item CONTENTS vtable
            try:
                ivt = self._r._read_pointer(item_ptr)
                if ivt != offsets.CONTENTS_VTABLE:
                    continue
            except OSError:
                continue

            try:
                iteminfo = self._r._read_pointer(item_ptr + offsets.CONTENTS_ITEMINFO_PTR)
                if iteminfo == 0 or iteminfo < 0x10000:
                    continue
                name = self._r._read_string(iteminfo + offsets.ITEMINFO_NAME, 64)
                item_id = self._r._read_uint32(iteminfo + offsets.ITEMINFO_ITEM_NUMBER)
                stack = self._r._read_int32(item_ptr + offsets.CONTENTS_STACK_COUNT)
                if stack <= 0:
                    stack = 1
                items.append((name, item_id, stack))
            except OSError:
                continue
        return items

    def read_inventory(self) -> list[tuple[str, int, int]]:
        """Read all inventory items from bag contents via profile chain.

        Returns list of (item_name, item_id, stack_count) for every item
        inside equipped bags.
        """
        items = []
        pb = self._r._resolve_profile_base()
        if pb is None:
            log.debug("[PERCEPTION] Inventory: profile chain unavailable -- returning empty")
            return []

        for bag in range(offsets.PROFILE_BAG_COUNT):
            bag_off = offsets.PROFILE_BAG_START + bag * offsets.PROFILE_BAG_STRIDE
            try:
                bag_ptr = self._r._read_pointer(pb + bag_off)
            except OSError:
                continue
            if bag_ptr == 0 or bag_ptr < 0x10000:
                continue
            try:
                vt = self._r._read_pointer(bag_ptr)
                if vt != offsets.CONTENTS_VTABLE:
                    continue
            except OSError:
                continue
            bag_items = self._read_bag_items(bag_ptr)
            if bag_items:
                items.extend(bag_items)
            else:
                # No sub-items: this slot might be a loose item (not a bag).
                # Read the slot's own ITEMINFO to check.
                try:
                    iteminfo = self._r._read_pointer(bag_ptr + offsets.CONTENTS_ITEMINFO_PTR)
                    if iteminfo and iteminfo > 0x10000:
                        name = self._r._read_string(iteminfo + offsets.ITEMINFO_NAME, 64)
                        item_id = self._r._read_uint32(iteminfo + offsets.ITEMINFO_ITEM_NUMBER)
                        stack = self._r._read_int32(bag_ptr + offsets.CONTENTS_STACK_COUNT)
                        if stack <= 0:
                            stack = 1
                        if name:
                            items.append((name, item_id, stack))
                except OSError:
                    pass
        return items

    def count_item(self, item_name: str) -> int:
        """Count total quantity of a named item across all bags.

        Case-insensitive match on item name.
        """
        total = 0
        for name, item_id, stack in self.read_inventory():
            if name.lower() == item_name.lower():
                total += stack
        return total
