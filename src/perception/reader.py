"""MemoryReader: reads target process memory via ctypes ReadProcessMemory.

Core class that owns the process handle and low-level read primitives.
Domain-specific reads are delegated to sub-reader modules:
  - CharReader   (char_reader.py)  -- CHARINFO, buffs, spellbook
  - InventoryReader (inventory_reader.py) -- loot window, bags
  - SpawnReader  (spawn_reader.py) -- spawn list, target

Public API is unchanged -- all methods remain accessible on MemoryReader.
"""

from __future__ import annotations

import ctypes
import logging
import struct
import time

from core.constants import XP_SCALE_MAX
from core.exceptions import MemoryReadError, ProcessNotFoundError
from perception import offsets
from perception._win32 import (
    PROCESS_QUERY_INFORMATION,
    PROCESS_VM_READ,
    kernel32,
)
from perception._win32 import (
    get_last_error as _get_last_error,
)
from perception.char_reader import CharReader
from perception.inventory_reader import InventoryReader
from perception.spawn_reader import (
    _SPAWN_POS_FMT,
    _SPAWN_POS_OFFSET,
    _VALID_BODY_STATES,
    MAX_SPAWNS,
    SPAWN_BUF_SIZE,
    SpawnReader,
)
from perception.state import GameState, SpawnData

log = logging.getLogger(__name__)

__all__ = [
    "MemoryReader",
    "SPAWN_BUF_SIZE",
    "_SPAWN_POS_FMT",
    "_SPAWN_POS_OFFSET",
    "MAX_SPAWNS",
    "_VALID_BODY_STATES",
]


class ReadStats:
    """Per-source error rate tracking for perception watchdog."""

    __slots__ = ("success", "fail", "consecutive_fail", "total")

    def __init__(self) -> None:
        self.success = 0
        self.fail = 0
        self.consecutive_fail = 0
        self.total = 0

    def record_ok(self) -> None:
        self.success += 1
        self.total += 1
        self.consecutive_fail = 0

    def record_fail(self) -> None:
        self.fail += 1
        self.total += 1
        self.consecutive_fail += 1

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 1.0
        return self.success / self.total


class MemoryReader:
    """Reads game state from target process memory each tick."""

    _char: CharReader
    _inv: InventoryReader
    _spawn: SpawnReader

    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._handle = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
        if not self._handle:
            raise ProcessNotFoundError(f"Failed to open process {pid}: error {_get_last_error()}")
        log.info("[PERCEPTION] Opened target process (PID %d), handle=0x%X", pid, self._handle)
        self._observed_mana_max = 0  # Track peak mana as surrogate for max
        self._profile_base_cache: int | None = None  # Cached profile_base (stable per session)
        self._profile_chain_failed = False  # True if chain was attempted and failed
        self._profile_retry_count = 0  # Number of cache-clear retries
        self._profile_retry_after = 0.0  # Timestamp: don't retry before this time
        self._weight_garbage_warned = False  # Suppress repeated weight-garbage warnings
        self._vbase_cache: int | None = None  # Game engine vbase address (stable across zones)

        # Perception watchdog: per-source error tracking
        self._read_stats: dict[str, ReadStats] = {
            "state": ReadStats(),
            "profile_chain": ReadStats(),
            "mana": ReadStats(),
            "spawns": ReadStats(),
        }
        self._stats_warned: set[str] = set()  # avoid repeated warnings

        # Sub-readers (initialized after shared state is set up)
        self._char = CharReader(self)
        self._inv = InventoryReader(self)
        self._spawn = SpawnReader(self)

    def __getattr__(self, name: str) -> object:
        """Lazy-init sub-readers on first access when not yet initialized."""
        if name in ("_char", "_inv", "_spawn"):
            obj: CharReader | InventoryReader | SpawnReader
            if name == "_char":
                obj = CharReader(self)
            elif name == "_inv":
                obj = InventoryReader(self)
            else:
                obj = SpawnReader(self)
            object.__setattr__(self, name, obj)
            return obj
        if name == "_read_stats":
            stats = {
                "state": ReadStats(),
                "profile_chain": ReadStats(),
                "mana": ReadStats(),
                "spawns": ReadStats(),
            }
            object.__setattr__(self, "_read_stats", stats)
            return stats
        if name == "_stats_warned":
            warned: set[str] = set()
            object.__setattr__(self, "_stats_warned", warned)
            return warned
        # Profile chain state (set in __init__, needed by tests using object.__new__)
        _profile_defaults = {
            "_observed_mana_max": 0,
            "_profile_base_cache": None,
            "_profile_chain_failed": False,
            "_profile_retry_count": 0,
            "_profile_retry_after": 0.0,
        }
        if name in _profile_defaults:
            val = _profile_defaults[name]
            object.__setattr__(self, name, val)
            return val
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

    def close(self) -> None:
        if self._handle:
            kernel32.CloseHandle(self._handle)
            self._handle = None

    # -- Health check ---------------------------------------------------------

    def health_check(self, state: GameState | None = None) -> dict:
        """Run a perception health check. Returns dict of stat->status.

        Call after first successful read_state() to report what's working.
        Each entry is (value, status) where status is 'ok', 'zero', 'garbage', or 'error'.
        """
        if state is None:
            state = self.read_state(include_spawns=False)

        results: dict[str, tuple[str | int, str]] = {}

        # entity struct-based (stable)
        results["hp"] = (f"{state.hp_current}/{state.hp_max}", "ok" if state.hp_max > 0 else "zero")
        results["level"] = (state.level, "ok" if 1 <= state.level <= 65 else "garbage")
        results["position"] = (
            f"({state.x:.0f}, {state.y:.0f}, {state.z:.0f})",
            "ok" if (state.x != 0 or state.y != 0) else "zero",
        )

        # Mana/Money source: profile chain (stable) or CHARINFO (fragile)
        profile_ok = self._profile_base_cache is not None
        mana_source = "profile" if profile_ok else "charinfo"
        results["mana"] = (
            f"{state.mana_current}/{state.mana_max} ({mana_source}, calc)",
            "ok" if state.mana_current > 0 else "zero",
        )
        results["weight"] = (state.weight, "ok" if state.weight > 0 else "zero")

        # Money
        money = (state.money_pp, state.money_gp, state.money_sp, state.money_cp)
        all_zero = all(v == 0 for v in money)
        any_huge = any(v > 100000 for v in money)
        money_source = "profile" if profile_ok else "charinfo"
        results["money"] = (
            f"{money[0]}p {money[1]}g {money[2]}s {money[3]}c ({money_source})",
            "garbage" if any_huge else ("unknown" if all_zero else "ok"),
        )

        # XP
        results["xp"] = (
            f"{state.xp_pct * 100:.1f}%",
            "ok" if 0 <= state.xp_pct_raw <= XP_SCALE_MAX else "garbage",
        )

        # Buffs
        buff_status = "ok" if state.buffs else "empty"
        results["buffs"] = (f"{len(state.buffs)} active", buff_status)

        # Casting
        results["casting_mode"] = (state.casting_mode, "ok")

        return results

    def validate_structs(self) -> bool:
        """Run struct compatibility validation. Returns True if all checks pass."""
        from perception.struct_validator import StructValidator

        validator = StructValidator(self)
        result = validator.validate()
        compatible: bool = result.compatible
        return compatible

    def log_health_check(self, state: GameState | None = None) -> None:
        """Log a startup health report showing what's reading correctly."""
        checks = self.health_check(state)
        lines = []
        problems = []
        for stat, (value, status) in checks.items():
            if status in ("ok", "primary", "empty"):
                lines.append(f"  {stat}: {value}")
            elif status == "scan":
                lines.append(f"  {stat}: {value} (auto-discovered)")
            elif status == "zero":
                lines.append(f"  {stat}: {value} [NOT READING]")
                problems.append(stat)
            elif status == "garbage":
                lines.append(f"  {stat}: {value} [GARBAGE]")
                problems.append(stat)
            elif status == "unknown":
                lines.append(f"  {stat}: {value} [unknown - may be correct]")
            else:
                lines.append(f"  {stat}: {value} [{status}]")

        header = "Perception health check"
        if problems:
            header += f" ({len(problems)} issue(s): {', '.join(problems)})"
        else:
            header += " (all OK)"
        log.info("[PERCEPTION] %s\n%s", header, "\n".join(lines))

    # -- Perception watchdog: error rate monitoring --------------------------

    def _check_read_health(self) -> None:
        """Warn if any read source drops below 90% success rate.

        Called every tick from read_state(). Checks are cheap (dict lookup).
        Auto-retries profile chain on 3 consecutive failures.
        """
        for source, stats in self._read_stats.items():
            if stats.total < 100:
                continue  # need enough samples
            if stats.success_rate < 0.90 and source not in self._stats_warned:
                log.warning(
                    "[PERCEPTION] PERCEPTION: '%s' success rate %.0f%% (%d/%d) -- reads may be degraded",
                    source,
                    stats.success_rate * 100,
                    stats.success,
                    stats.total,
                )
                self._stats_warned.add(source)

        # Auto-retry profile chain on 3 consecutive failures (with backoff)
        pc = self._read_stats.get("profile_chain")
        if pc and pc.consecutive_fail >= 3 and self._profile_base_cache is not None:
            self._profile_retry_count += 1
            if self._profile_retry_count > 5:
                log.error(
                    "[PERCEPTION] PERCEPTION: profile chain failed %d retries -- "
                    "giving up (watchdog will reattach)",
                    self._profile_retry_count,
                )
                pc.consecutive_fail = 0
                return
            backoff = min(2**self._profile_retry_count, 30)
            self._profile_retry_after = time.time() + backoff
            log.warning(
                "[PERCEPTION] PERCEPTION: profile chain 3x consecutive fail -- "
                "clearing cache (retry %d/5, backoff %ds)",
                self._profile_retry_count,
                backoff,
            )
            self._profile_base_cache = None
            self._profile_chain_failed = False
            pc.consecutive_fail = 0

    def health_stats(self) -> dict[str, dict[str, int | float]]:
        """Return per-source read statistics for dashboard/logging."""
        result = {}
        stats_snapshot = dict(self._read_stats)  # snapshot before cross-thread iteration
        for source, stats in stats_snapshot.items():
            result[source] = {
                "success": stats.success,
                "fail": stats.fail,
                "total": stats.total,
                "success_rate": round(stats.success_rate, 3),
                "consecutive_fail": stats.consecutive_fail,
            }
        return result

    # -- Low-level read primitives -------------------------------------------

    def _read_bytes(self, address: int, size: int) -> bytes:
        buf = ctypes.create_string_buffer(size)
        bytes_read = ctypes.c_size_t(0)
        ok = kernel32.ReadProcessMemory(
            self._handle, ctypes.c_void_p(address), buf, size, ctypes.byref(bytes_read)
        )
        if not ok:
            raise MemoryReadError(f"ReadProcessMemory failed at 0x{address:08X}: error {_get_last_error()}")
        return buf.raw

    def _read_float(self, address: int) -> float:
        return float(struct.unpack("<f", self._read_bytes(address, 4))[0])

    def _read_int32(self, address: int) -> int:
        return int(struct.unpack("<i", self._read_bytes(address, 4))[0])

    def _read_uint32(self, address: int) -> int:
        return int(struct.unpack("<I", self._read_bytes(address, 4))[0])

    def _read_byte(self, address: int) -> int:
        return self._read_bytes(address, 1)[0]

    def _read_string(self, address: int, max_len: int = 64) -> str:
        raw = self._read_bytes(address, max_len)
        end = raw.find(b"\x00")
        if end != -1:
            raw = raw[:end]
        from eq.strings import decode_eq_string

        decoded: str = decode_eq_string(raw)
        return decoded

    def _read_pointer(self, address: int) -> int:
        return self._read_uint32(address)

    # -- Base pointer resolution ---------------------------------------------

    def _get_spawn_base(self) -> int:
        """Dereference the static pointer to get the player entity struct base address."""
        base = self._read_pointer(offsets.PLAYER_SPAWN_PTR)
        if base == 0:
            raise RuntimeError("Player spawn pointer is null  -  character may not be in world")
        return base

    def read_char_name(self) -> str:
        """Read the player character name from entity struct+NAME offset.

        Lightweight read -- does not require full read_state().
        Used during startup to derive log file paths dynamically.
        """
        base = self._get_spawn_base()
        return self._read_string(base + offsets.NAME)

    def _get_charinfo_base(self) -> int | None:
        """Dereference entity struct+CHARINFO_PTR to get CHARINFO base. Returns None on failure."""
        if offsets.CHARINFO_PTR == 0:
            return None
        try:
            spawn_base = self._get_spawn_base()
            ptr = self._read_pointer(spawn_base + offsets.CHARINFO_PTR)
            return ptr if ptr != 0 else None
        except OSError:
            return None

    def read_window_pos(self, wnd_ptr_addr: int) -> tuple[int, int] | None:
        """Read (x, y) position of a UI window from its CXWnd base fields.

        Args:
            wnd_ptr_addr: Static address of the window pointer
                (e.g., SPELL_BOOK_WND_PTR, CAST_SPELL_WND_PTR).

        Returns:
            (x, y) in pixels relative to game client area, or None if
            the pointer is null, read fails, or values are garbage.
        """
        try:
            wnd = self._read_pointer(wnd_ptr_addr)
            if not wnd:
                return None
            x = self._read_int32(wnd + offsets.CXWND_X)
            y = self._read_int32(wnd + offsets.CXWND_Y)
            if not (-100 <= x <= 4096 and -100 <= y <= 4096):
                return None
            return (x, y)
        except OSError:
            return None

    def _resolve_vbase(self, engine_ptr: int) -> int | None:
        """Resolve game engine vbptr chain to vbase address.

        Chain: engine+0x08 -> vbptr, *(vbptr+4) -> vb_offset, vbase = engine + vb_offset.
        Cached because engine_ptr and vbase survive zone transitions.
        """
        if self._vbase_cache is not None:
            return self._vbase_cache
        try:
            vbptr = self._read_pointer(engine_ptr + offsets.GAME_ENGINE_VBPTR_OFFSET)
            if not vbptr:
                return None
            vb_offset = self._read_int32(vbptr + 4)
            if vb_offset is None or vb_offset <= 0 or vb_offset > 0xC4D0:
                return None
            self._vbase_cache = engine_ptr + vb_offset
            return self._vbase_cache
        except OSError:
            return None

    def _validate_profile_guardrails(self, profile_base: int) -> tuple[int, int, int, int] | None:
        """Read and validate sentinel fields at the candidate profile_base.

        Returns (level, class, race, mana) if all values are sane,
        or None if validation fails (with retry/backoff side-effects).
        """
        try:
            g_level = self._read_int32(profile_base + offsets.PROFILE_LEVEL)
            g_class = self._read_int32(profile_base + offsets.PROFILE_CLASS)
            g_race = self._read_int32(profile_base + offsets.PROFILE_RACE)
            g_mana = self._read_int32(profile_base + offsets.PROFILE_MANA)
        except OSError as ge:
            log.warning(
                "[PERCEPTION] Profile chain: guardrail reads failed at profile_base=0x%08X -- %s",
                profile_base,
                ge,
            )
            self._profile_chain_failed = True
            return None

        guardrail_ok = True
        if not (1 <= g_level <= 60):
            log.warning("[PERCEPTION] Profile chain guardrail FAILED: level=%d (expected 1-60)", g_level)
            guardrail_ok = False
        if not (1 <= g_class <= 16):
            log.warning("[PERCEPTION] Profile chain guardrail FAILED: class=%d (expected 1-16)", g_class)
            guardrail_ok = False
        if not (1 <= g_race <= 330):
            log.warning("[PERCEPTION] Profile chain guardrail FAILED: race=%d (expected 1-330)", g_race)
            guardrail_ok = False
        if not (0 <= g_mana <= 10000):
            log.warning("[PERCEPTION] Profile chain guardrail FAILED: mana=%d (expected 0-10000)", g_mana)
            guardrail_ok = False

        if not guardrail_ok:
            self._profile_retry_count += 1
            backoff = min(2**self._profile_retry_count, 30)
            self._profile_retry_after = time.time() + backoff
            log.warning(
                "[PERCEPTION] Profile chain: guardrail validation failed -- "
                "not caching profile_base=0x%08X "
                "(level=%d class=%d race=%d mana=%d) "
                "retry %d in %ds",
                profile_base,
                g_level,
                g_class,
                g_race,
                g_mana,
                self._profile_retry_count,
                backoff,
            )
            if self._profile_retry_count > 10:
                log.error(
                    "[PERCEPTION] Profile chain: guardrail failed %d times -- "
                    "giving up until cache cleared externally",
                    self._profile_retry_count,
                )
                self._profile_chain_failed = True
            self._read_stats["profile_chain"].record_fail()
            return None

        return g_level, g_class, g_race, g_mana

    def _resolve_profile_base(self) -> int | None:
        """Walk the profile pointer chain to get the stable profile base.

        Chain: CHARINFO+0x0108 -> intermediate_ptr
               intermediate_ptr+0x0004 -> profile_base

        The profile_base is cached because it does not change during a session.
        Returns None if any pointer in the chain is null or unreadable.
        """
        # Return cached value if already resolved
        if self._profile_base_cache is not None:
            return self._profile_base_cache
        # Don't retry if we already know the chain is broken
        if self._profile_chain_failed:
            return None
        # Respect backoff delay after cache-clear retry
        if self._profile_retry_after > 0 and time.time() < self._profile_retry_after:
            return None

        ci = self._get_charinfo_base()
        if ci is None:
            log.debug("[PERCEPTION] Profile chain: CHARINFO base is null -- cannot resolve")
            self._profile_chain_failed = True
            return None

        try:
            intermediate = self._read_pointer(ci + offsets.CHARINFO_PROFILE_INDIR)
            if intermediate == 0:
                log.warning(
                    "[PERCEPTION] Profile chain: CHARINFO+0x%04X -> null intermediate pointer",
                    offsets.CHARINFO_PROFILE_INDIR,
                )
                self._profile_chain_failed = True
                return None

            profile_base = self._read_pointer(intermediate + offsets.PROFILE_PTR_OFFSET)
            if profile_base == 0:
                log.warning(
                    "[PERCEPTION] Profile chain: intermediate+0x%04X -> null profile_base",
                    offsets.PROFILE_PTR_OFFSET,
                )
                self._profile_chain_failed = True
                return None

            # -- Guardrails: validate sentinel fields before caching --------
            guardrail = self._validate_profile_guardrails(profile_base)
            if guardrail is None:
                return None
            g_level, g_class, g_race, g_mana = guardrail

            log.info(
                "[PERCEPTION] Profile chain resolved: CHARINFO=0x%08X -> "
                "intermediate=0x%08X -> profile_base=0x%08X "
                "(guardrails OK: level=%d class=%d race=%d mana=%d)",
                ci,
                intermediate,
                profile_base,
                g_level,
                g_class,
                g_race,
                g_mana,
            )
            self._profile_base_cache = profile_base
            self._read_stats["profile_chain"].record_ok()
            # Reset retry state on success
            if self._profile_retry_count > 0:
                log.info("[PERCEPTION] Profile chain: recovered after %d retries", self._profile_retry_count)
                self._profile_retry_count = 0
                self._profile_retry_after = 0.0
            return profile_base

        except OSError as e:
            log.warning("[PERCEPTION] Profile chain: read failed -- %s", e)
            self._profile_chain_failed = True
            self._read_stats["profile_chain"].record_fail()
            return None

    # -- Typed sub-reader accessors -------------------------------------------
    # __getattr__ returns ``object`` which makes mypy infer Any for attribute
    # access on self._char / self._inv / self._spawn, even though they are
    # assigned concrete types in __init__. These helpers recover the types.

    def _get_char(self) -> CharReader:
        return self._char

    def _get_inv(self) -> InventoryReader:
        return self._inv

    def _get_spawn(self) -> SpawnReader:
        return self._spawn

    # -- Delegation to sub-readers --------------------------------------------

    def _read_profile_mana(self) -> int | None:
        return self._get_char()._read_profile_mana()

    def _read_profile_money(self) -> tuple[int, int, int, int] | None:
        return self._get_char()._read_profile_money()

    def _read_profile_gems(self) -> dict[int, int] | None:
        return self._get_char()._read_profile_gems()

    def _read_charinfo_mana(self) -> int:
        return self._get_char()._read_charinfo_mana()

    def _read_charinfo_weight(self) -> int:
        return self._get_char()._read_charinfo_weight()

    def read_money(self) -> tuple[int, int, int, int]:
        return self._get_char().read_money()

    def _read_charinfo_batch(self, ci_ptr: int) -> tuple:
        return self._get_char()._read_charinfo_batch(ci_ptr)

    def read_buffs(self) -> tuple[tuple[int, int], ...]:
        return self._get_char().read_buffs()

    def get_buff_ticks(self, spell_id: int) -> int:
        return self._get_char().get_buff_ticks(spell_id)

    def is_buff_active(self, spell_id: int) -> bool:
        return self._get_char().is_buff_active(spell_id)

    def read_memorized_spells(self) -> dict[int, int]:
        return self._get_char().read_memorized_spells()

    def _read_spellbook_raw(self) -> bytes | None:
        return self._get_char()._read_spellbook_raw()

    def read_spellbook(self) -> set[int]:
        return self._get_char().read_spellbook()

    def spellbook_slot_for(self, spell_id: int) -> int | None:
        return self._get_char().spellbook_slot_for(spell_id)

    # -- Delegation to InventoryReader ----------------------------------------

    def read_loot_items(self) -> tuple[int, ...]:
        return self._get_inv().read_loot_items()

    def read_loot_items_deep(self) -> list[tuple[int, str, int]]:
        return self._get_inv().read_loot_items_deep()

    def _read_bag_items(self, bag_ptr: int) -> list[tuple[str, int, int]]:
        return self._get_inv()._read_bag_items(bag_ptr)

    def read_inventory(self) -> list[tuple[str, int, int]]:
        return self._get_inv().read_inventory()

    def count_item(self, item_name: str) -> int:
        return self._get_inv().count_item(item_name)

    # -- Delegation to SpawnReader -------------------------------------------

    def _read_target(self) -> SpawnData | None:
        return self._get_spawn()._read_target()

    def _parse_spawn_from_buffer(self, buf: bytes) -> dict:
        return self._get_spawn()._parse_spawn_from_buffer(buf)

    @staticmethod
    def _spawn_from_parsed(fields: dict) -> SpawnData:
        return SpawnReader._spawn_from_parsed(fields)

    def _bulk_read_spawn_node(self, base: int) -> tuple[SpawnData, dict]:
        return self._get_spawn()._bulk_read_spawn_node(base)

    def read_spawns(self) -> list[SpawnData]:
        return self._get_spawn().read_spawns()

    # -- Sub-readers for state assembly ----------------------------------------

    def _read_casting_state(self) -> tuple[int, int]:
        """Read casting mode and spell ID. Returns (casting_mode, spell_id)."""
        try:
            casting = self._read_int32(offsets.CASTING_MODE_PTR)
            if casting < 0 or casting > 10:
                casting = 0
        except OSError:
            casting = 0

        casting_spell = -1
        if offsets.CASTING_SPELL_ID_PTR != 0 and casting > 0:
            try:
                casting_spell = self._read_int32(offsets.CASTING_SPELL_ID_PTR)
                if casting_spell < 0 or casting_spell > 10000:
                    casting_spell = -1
            except OSError:
                casting_spell = -1
        return casting, casting_spell

    def _read_engine_state(self) -> tuple[int, int, int, int]:
        """Read engine globals. Returns (game_mode, xp_raw, defeat_count, engine_zone_id)."""
        game_mode = 5
        try:
            engine_ptr = self._read_pointer(offsets.ENGINE_STATE_PTR)
            if engine_ptr:
                game_mode = self._read_int32(engine_ptr + offsets.ENGINE_GAME_MODE_OFFSET)
                if game_mode < 0 or game_mode > 0xFF:
                    game_mode = 5
        except OSError:
            pass

        xp_raw = 0
        defeat_count = 0
        engine_zone_id = 0
        try:
            engine = self._read_pointer(offsets.GAME_ENGINE_PTR)
            if engine:
                xp_raw = self._read_uint32(engine + offsets.GAME_ENGINE_XP_PCT)
                if xp_raw > XP_SCALE_MAX:
                    xp_raw = 0
                defeat_count = self._read_uint32(engine + offsets.GAME_ENGINE_KILL_COUNT)
                vbase = self._resolve_vbase(engine)
                if vbase:
                    engine_zone_id = self._read_uint32(vbase + offsets.GAME_ENGINE_VB_ZONE_ID)
                    if engine_zone_id > 500:
                        engine_zone_id = 0
        except OSError:
            pass
        return game_mode, xp_raw, defeat_count, engine_zone_id

    # -- State assembly (orchestrates sub-readers) ---------------------------

    def read_state(self, include_spawns: bool = False) -> GameState:
        """Read a complete GameState snapshot from memory.

        Uses bulk reads to minimize ReadProcessMemory calls and reduce the
        window for client/server reconciliation inconsistency.

        Before:  ~930 RPM calls per tick (~100ms window, guaranteed jitter)
        After:   ~60 RPM calls per tick (~5ms window, rare jitter)
        """
        base = self._get_spawn_base()  # 1 RPM

        # -- Bulk read player's entity struct in one call (1 RPM) --
        player_buf = self._read_bytes(base, SPAWN_BUF_SIZE)
        pf = self._spawn._parse_spawn_from_buffer(player_buf)

        # -- Consistency sentinel: snapshot HP before batch reads --
        sentinel_hp = pf["hp_current"]

        # -- Spawns (bulk read per node) --
        spawns: tuple[SpawnData, ...] = ()
        if include_spawns:
            try:
                spawns = tuple(self._spawn.read_spawns(player_x=pf["x"], player_y=pf["y"]))
                self._read_stats["spawns"].record_ok()
            except OSError:
                log.debug("[PERCEPTION] Failed to read spawn list")
                self._read_stats["spawns"].record_fail()

        # -- Target (bulk read: 2-3 RPM) --
        target = self._spawn._read_target()

        # -- Global reads --
        casting, casting_spell = self._read_casting_state()
        game_mode, xp_raw, defeat_count, engine_zone_id = self._read_engine_state()

        # -- CHARINFO batch (pre-resolved pointer from player buffer) --
        ci_ptr = pf["_charinfo_ptr"]
        mana, mana_max, weight, money, buffs = self._char._read_charinfo_batch(ci_ptr)
        pp, gp, sp, cp = money

        # Track mana read success
        if mana > 0:
            self._read_stats["mana"].record_ok()
        else:
            self._read_stats["mana"].record_fail()

        # Update observed max (tracks peak mana including gear/buff bonuses)
        if mana > self._observed_mana_max:
            self._observed_mana_max = mana
        # Use the higher of calc and observed -- calc uses base INT from
        # profile (no gear bonuses), observed tracks actual peak mana
        if mana_max <= 0:
            mana_max = self._observed_mana_max
        elif self._observed_mana_max > mana_max:
            mana_max = self._observed_mana_max

        # -- Consistency sentinel: re-read HP after all reads --
        try:
            sentinel_hp_after = self._read_int32(base + offsets.HP_CURRENT)
            if sentinel_hp != sentinel_hp_after:
                log.debug(
                    "[PERCEPTION] Snapshot sentinel: HP changed %d -> %d (straddled tick)",
                    sentinel_hp,
                    sentinel_hp_after,
                )
        except OSError:
            pass

        # -- Player-specific fields from buffer --
        player_state = 0
        if offsets.PLAYER_STATE:
            player_state = struct.unpack_from("<i", player_buf, offsets.PLAYER_STATE)[0]
        speed_heading = 0.0
        if offsets.SPEED_HEADING:
            speed_heading = struct.unpack_from("<f", player_buf, offsets.SPEED_HEADING)[0]

        # -- Sit/stand from ActorClient --
        # AC_ACTIVITY_STATE: 32=standing, 38=sitting/spellbook, 24=ducking, 26=casting, 44=combat
        ac_ptr = pf["_actorclient_ptr"]
        activity_state = 32  # default: standing
        if ac_ptr:
            try:
                activity_state = self._read_int32(ac_ptr + offsets.AC_ACTIVITY_STATE)
            except OSError:
                pass
        # Map to stand_state for backward compat: 0=standing, 1=sitting
        # 38=sitting, 26=casting/memorizing (player remains seated during memorize)
        stand_state = 1 if activity_state in (38, 26) else 0
        # Derive in_combat from activity_state (replaces unverified NO_REGEN_FLAG)
        # Combat states: 44=combat, 5=melee, 13=hit reaction, 18=chasing
        in_combat = activity_state in (44, 5, 13, 18)

        self._read_stats["state"].record_ok()
        self._check_read_health()

        return GameState(
            x=pf["x"],
            y=pf["y"],
            z=pf["z"],
            heading=pf["heading"],
            hp_current=pf["hp_current"],
            hp_max=pf["hp_max"],
            mana_current=mana,
            mana_max=mana_max,
            level=pf["level"],
            name=pf["name"],
            spawn_type=pf["spawn_type"],
            stand_state=stand_state,
            player_state=player_state,
            spawn_id=pf["spawn_id"],
            class_id=pf["mob_class"],
            body_state=pf["body_state"],
            zone_id=pf["_zone_id"],
            engine_zone_id=engine_zone_id,
            speed_run=pf["speed"],
            speed_heading=speed_heading,
            weight=weight,
            xp_pct_raw=xp_raw,
            defeat_count=defeat_count,
            money_pp=pp,
            money_gp=gp,
            money_sp=sp,
            money_cp=cp,
            casting_mode=casting,
            casting_spell_id=casting_spell,
            in_combat=in_combat,
            game_mode=game_mode,
            buffs=buffs,
            target=target,
            spawns=spawns,
        )
