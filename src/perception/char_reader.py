"""CharReader: character data reads via the profile pointer chain.

All character data (mana, money, buffs, spellbook, spell gems) is read
via the profile pointer chain:
  character info+0x0108 -> intermediate+0x04 -> profile_base

Weight is the only field still read from character info directly (+0x0048).

Extracted from reader.py to keep module sizes manageable.
All methods access the parent MemoryReader via self._r for
low-level read primitives (_read_bytes, _read_int32, etc.)
and shared state (_observed_mana_max, _profile_base_cache, etc.).
"""

from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING

from perception import offsets

if TYPE_CHECKING:
    from perception.reader import MemoryReader

log = logging.getLogger(__name__)


class CharReader:
    """Reads character info, buffs, and spellbook from target process memory.

    Requires a reference to the parent MemoryReader for read primitives
    and shared instance state (mana max tracking, profile cache, etc.).
    """

    def __init__(self, reader: MemoryReader) -> None:
        self._r = reader

    # -- Mana calculation (CalcBaseMana, pre-SoF formula) ---------------

    # INT casters: SHD(5), BRD(8), NEC(11), WIZ(12), MAG(13), ENC(14)
    _INT_CASTER_CLASSES = frozenset({5, 8, 11, 12, 13, 14})
    # WIS casters: CLR(2), PAL(3), RNG(4), DRU(6), SHM(10), BST(15)
    _WIS_CASTER_CLASSES = frozenset({2, 3, 4, 6, 10, 15})

    @staticmethod
    def calc_base_mana(wis_or_int: int, level: int) -> int:
        """Calculate base max mana from primary caster stat and level.

        Uses the CalcBaseMana formula (pre-SoF). Same formula for both INT
        and WIS casters -- caller passes the appropriate stat. Does NOT
        include item/spell/AA bonuses.
        """
        if level <= 0 or wis_or_int <= 0:
            return 0
        # MindLesserFactor: diminishing returns above 199
        mind_lesser = max(0, (wis_or_int - 199) // 2)
        mind_factor = wis_or_int - mind_lesser
        if wis_or_int > 100:
            return ((5 * (mind_factor + 20)) // 2) * 3 * level // 40
        else:
            return ((5 * (mind_factor + 200)) // 2) * 3 * level // 100

    def _calc_max_mana(self) -> int | None:
        """Calculate max mana from profile stats (INT/WIS, level, class).

        Returns the calculated max mana, or None if profile is unavailable.
        """
        pb = self._r._resolve_profile_base()
        if pb is None:
            return None
        try:
            class_id = self._r._read_int32(pb + offsets.PROFILE_CLASS)
            level = self._r._read_int32(pb + offsets.PROFILE_LEVEL)

            if class_id in self._INT_CASTER_CLASSES:
                stat = self._r._read_int32(pb + offsets.PROFILE_INT)
            elif class_id in self._WIS_CASTER_CLASSES:
                stat = self._r._read_int32(pb + offsets.PROFILE_WIS)
            else:
                return 0  # non-caster

            if not (1 <= level <= 65 and 1 <= stat <= 500):
                log.warning("[PERCEPTION] Mana calc: bad stats (level=%d, stat=%d) -- skipping", level, stat)
                return None

            return self.calc_base_mana(stat, level)
        except OSError:
            log.debug("[PERCEPTION] Mana calc: profile read failed")
            return None

    # -- Mana reads -----------------------------------------------------------

    def _read_profile_mana(self) -> int | None:
        """Read current mana from profile_base + PROFILE_MANA.

        Returns the mana value, or None if profile chain is unavailable.
        """
        pb = self._r._resolve_profile_base()
        if pb is None:
            log.debug("[PERCEPTION] Profile mana: profile base unavailable -- skipping")
            return None
        try:
            mana = self._r._read_int32(pb + offsets.PROFILE_MANA)
            if 0 <= mana <= 50000:
                return int(mana)
            log.debug("[PERCEPTION] Profile mana: value %d out of range -- ignoring", mana)
            return None
        except OSError:
            log.debug("[PERCEPTION] Profile mana: read failed at profile_base+0x%04X", offsets.PROFILE_MANA)
            return None

    def _read_charinfo_mana(self) -> int:
        """Read current mana via profile pointer chain. Returns 0 on failure."""
        profile_mana = self._read_profile_mana()
        if profile_mana is not None:
            return profile_mana
        log.debug("[PERCEPTION] Mana: profile chain unavailable -- returning 0")
        return 0

    # -- Weight reads ---------------------------------------------------------

    def _read_charinfo_weight(self) -> int:
        """Read current carry weight from character info+0x0048. Returns 0 on failure."""
        ci = self._r._get_charinfo_base()
        if ci is None:
            return 0
        try:
            weight = self._r._read_int32(ci + offsets.CHARINFO_WEIGHT)
            if 0 <= weight <= 10000:
                return int(weight)
            if not getattr(self._r, "_weight_garbage_warned", False):
                log.warning(
                    "[PERCEPTION] Weight: character info+0x%04X returned garbage (%d)",
                    offsets.CHARINFO_WEIGHT,
                    weight,
                )
                self._r._weight_garbage_warned = True
            return 0
        except OSError:
            return 0

    # -- Money reads ----------------------------------------------------------

    def _read_profile_money(self) -> tuple[int, int, int, int] | None:
        """Read PP/GP/SP/CP from profile_base + PROFILE_PP/GP/SP/CP.

        Returns (pp, gp, sp, cp) or None if profile chain is unavailable.
        """
        pb = self._r._resolve_profile_base()
        if pb is None:
            log.debug("[PERCEPTION] Profile money: profile base unavailable -- skipping")
            return None
        try:
            money_buf = self._r._read_bytes(pb + offsets.PROFILE_PP, 16)
            pp, gp, sp, cp = struct.unpack_from("<4i", money_buf, 0)
            if any(v < 0 or v > 10_000_000 for v in (pp, gp, sp, cp)):
                log.debug(
                    "[PERCEPTION] Profile money: values out of range (pp=%d gp=%d sp=%d cp=%d) -- ignoring",
                    pp,
                    gp,
                    sp,
                    cp,
                )
                return None
            return (pp, gp, sp, cp)
        except OSError:
            log.debug("[PERCEPTION] Profile money: read failed at profile_base+0x%04X", offsets.PROFILE_PP)
            return None

    def read_money(self) -> tuple[int, int, int, int]:
        """Read money (plat, gold, silver, copper) via profile chain.

        Returns (0,0,0,0) on failure.
        """
        profile_money = self._read_profile_money()
        if profile_money is not None:
            return profile_money
        log.debug("[PERCEPTION] Money: profile chain unavailable -- returning zeros")
        return (0, 0, 0, 0)

    # -- Batch character info read --------------------------------------------------

    def _read_charinfo_batch(self, ci_ptr: int) -> tuple:
        """Batch-read all character fields needed per tick.

        Returns (mana_current, mana_max, weight, (pp, gp, sp, cp), buffs).
        All reads via profile pointer chain except weight (character info+0x0048).
        """
        if ci_ptr == 0:
            return (0, 0, 0, (0, 0, 0, 0), ())

        # Weight: direct character info read (only field not in profile)
        weight = self._read_charinfo_weight()

        # Money: profile chain
        money = self._read_profile_money()
        if money is None:
            money = (0, 0, 0, 0)

        # Mana: profile chain (current) + calculated (max)
        mana = self._read_charinfo_mana()
        mana_max = self._calc_max_mana()
        if mana_max is None:
            mana_max = 0

        # Buffs: profile chain
        buffs = self._read_profile_buffs()
        if buffs is None:
            buffs = ()

        return (mana, mana_max, weight, money, buffs)

    # -- Buff reads -----------------------------------------------------------

    def _read_profile_buffs(self) -> tuple[tuple[int, int], ...] | None:
        """Read buff array from profile pointer chain.

        Returns tuple of (spell_id, ticks) for active buffs, or None if
        the profile chain is unavailable.
        """
        pb = self._r._resolve_profile_base()
        if pb is None:
            return None
        try:
            # Read all 25 buff slots in one bulk read (25 * 20 = 500 bytes)
            buf = self._r._read_bytes(
                pb + offsets.PROFILE_BUFF_BASE, offsets.PROFILE_BUFF_COUNT * offsets.PROFILE_BUFF_SLOT_SIZE
            )
            active = []
            for slot in range(offsets.PROFILE_BUFF_COUNT):
                base = slot * offsets.PROFILE_BUFF_SLOT_SIZE
                spell_id = struct.unpack_from("<i", buf, base + offsets.PROFILE_BUFF_SPELL_ID_OFF)[0]
                if 0 < spell_id <= 10000:
                    ticks = struct.unpack_from("<i", buf, base + offsets.PROFILE_BUFF_TICKS_OFF)[0]
                    active.append((spell_id, ticks))
            return tuple(active)
        except OSError:
            return None

    def read_buffs(self) -> tuple[tuple[int, int], ...]:
        """Read active buff slots via profile chain."""
        profile_buffs = self._read_profile_buffs()
        if profile_buffs is not None:
            return profile_buffs
        log.debug("[PERCEPTION] Buffs: profile chain unavailable -- returning empty")
        return ()

    def get_buff_ticks(self, spell_id: int) -> int:
        """Read the remaining ticks for a buff by spell ID.

        Returns the ticks value (positive = active, 0/-1 = expired/empty),
        or -1 if the spell is not found in any buff slot.
        """
        profile_buffs = self._read_profile_buffs()
        if profile_buffs is not None:
            for sid, ticks in profile_buffs:
                if sid == spell_id:
                    return ticks
            return -1
        return -1

    def is_buff_active(self, spell_id: int) -> bool:
        """Check if a buff is in the buff array with positive ticks."""
        ticks = self.get_buff_ticks(spell_id)
        return ticks > 0

    # -- Spellbook reads ------------------------------------------------------

    def _read_profile_gems(self) -> dict[int, int] | None:
        """Read memorized spell gems from profile_base + PROFILE_SPELL_GEMS.

        Returns {gem_slot: spell_id} (1-indexed) for occupied slots,
        or None if profile chain is unavailable.
        """
        pb = self._r._resolve_profile_base()
        if pb is None:
            log.debug("[PERCEPTION] Profile gems: profile base unavailable -- skipping")
            return None
        try:
            gem_count = offsets.PROFILE_SPELL_GEM_COUNT
            gem_buf = self._r._read_bytes(pb + offsets.PROFILE_SPELL_GEMS, gem_count * 4)
            result = {}
            for i in range(gem_count):
                spell_id = struct.unpack_from("<i", gem_buf, i * 4)[0]
                if 0 < spell_id < 10000:
                    result[i + 1] = spell_id  # 1-indexed gem slots
            return result
        except OSError:
            log.debug(
                "[PERCEPTION] Profile gems: read failed at profile_base+0x%04X", offsets.PROFILE_SPELL_GEMS
            )
            return None

    def read_memorized_spells(self) -> dict[int, int]:
        """Read memorized spell gem array via profile chain.

        Returns: {gem_slot: spell_id} for occupied slots (1-indexed).
        Empty slots (-1) are omitted.
        """
        profile_gems = self._read_profile_gems()
        if profile_gems is not None:
            return profile_gems
        log.debug("[PERCEPTION] Spell gems: profile chain unavailable -- returning empty")
        return {}

    def _read_spellbook_raw(self) -> bytes | None:
        """Read the 400-slot spellbook array via profile chain.

        Returns 1600 bytes (400 INT32s) or None on failure.
        """
        pb = self._r._resolve_profile_base()
        if pb is not None:
            try:
                return bytes(
                    self._r._read_bytes(pb + offsets.PROFILE_SPELLBOOK, offsets.PROFILE_SPELLBOOK_SIZE * 4)
                )
            except OSError:
                log.debug("[PERCEPTION] Spellbook: profile read failed")
        return None

    def read_spellbook(self) -> set[int]:
        """Read ALL scribed spell IDs from the spellbook array.

        Returns: set of spell IDs that are scribed (available to memorize).
        The spellbook is a 400-slot uint32 array. Empty slots are 0xFFFFFFFF (-1).
        """
        raw = self._read_spellbook_raw()
        if raw is None:
            return set()
        scribed = set()
        slot_count = len(raw) // 4
        for i in range(slot_count):
            spell_id = struct.unpack_from("<i", raw, i * 4)[0]
            if 0 < spell_id < 10000:
                scribed.add(spell_id)
        return scribed

    def spellbook_slot_for(self, spell_id: int) -> int | None:
        """Find which spellbook slot contains a spell ID.

        Returns: slot index (0-based) or None if not found.
        Use: page = slot // 8, position = slot % 8.
        """
        raw = self._read_spellbook_raw()
        if raw is None:
            return None
        slot_count = len(raw) // 4
        for i in range(slot_count):
            sid = struct.unpack_from("<i", raw, i * 4)[0]
            if sid == spell_id:
                return i
        return None
