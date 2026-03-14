"""SpawnReader: spawn list and target memory reads.

Extracted from reader.py to keep module sizes manageable.
All methods access the parent MemoryReader via self._r for
low-level read primitives (_read_bytes, _read_int32, etc.).
"""

from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING

from perception import offsets
from perception.state import SpawnData

if TYPE_CHECKING:
    from perception.reader import MemoryReader

log = logging.getLogger(__name__)

# Maximum number of spawns to read per tick (safety limit)
MAX_SPAWNS = 500

# Valid entity struct body_state byte values (0x0251)
_VALID_BODY_STATES = frozenset({ord("d"), ord("f"), ord("n"), ord("o"), ord("i")})

# -- Bulk read constants -------------------------------------------------------
# Read entire entity struct in one RPM call instead of 18+ individual reads.
SPAWN_BUF_SIZE = 0x02F0  # 752 bytes

# Pre-compiled struct for contiguous float block: Y,X,Z,velY,velX,velZ,speed,heading
_SPAWN_POS_FMT = struct.Struct("<8f")
_SPAWN_POS_OFFSET = offsets.Y


class SpawnReader:
    """Reads spawn list, target, and individual entity struct from target process memory.

    Requires a reference to the parent MemoryReader for read primitives.
    """

    def __init__(self, reader: MemoryReader) -> None:
        self._r = reader
        self._consecutive_spawn_errors: int = 0

    def _read_target(self) -> SpawnData | None:
        """Read the current target's SpawnData. Returns None if no target.

        Uses bulk read: 1 RPM for pointer + 1 RPM for entity struct buffer
        (+ 1 for NPC target name) instead of ~19 individual reads.
        """
        try:
            target_base = self._r._read_pointer(offsets.TARGET_PTR)
            if target_base == 0:
                return None
            spawn, _ = self._bulk_read_spawn_node(target_base)
            return spawn
        except OSError:
            return None

    def _parse_spawn_from_buffer(self, buf: bytes) -> dict:
        """Parse all entity struct fields from a pre-read byte buffer.

        Returns dict with SpawnData kwargs plus internal keys (prefixed '_'):
          _next_ptr, _prev_ptr, _actorclient_ptr, _charinfo_ptr,
          _no_regen_flag, _stand_state_raw, _zone_id
        """
        # Contiguous float block: y, x, z, vel_y, vel_x, vel_z, speed, heading
        y, x, z, vel_y, vel_x, vel_z, speed, heading = _SPAWN_POS_FMT.unpack_from(buf, _SPAWN_POS_OFFSET)

        # Name string: 64 bytes at offsets.NAME
        name_end = buf.find(b"\x00", offsets.NAME, offsets.NAME + 64)
        if name_end == -1:
            name_end = offsets.NAME + 64
        from eq.strings import decode_eq_string

        name = decode_eq_string(buf[offsets.NAME : name_end])

        # Byte fields
        spawn_type = buf[offsets.TYPE]
        level = buf[offsets.LEVEL]
        hide = buf[offsets.HIDE]
        body_raw = buf[offsets.BODY_STATE]
        body_char = chr(body_raw) if body_raw in _VALID_BODY_STATES else "n"
        mob_class = buf[offsets.CLASS]
        no_regen = buf[offsets.NO_REGEN_FLAG] if offsets.NO_REGEN_FLAG else 0
        stand_state = buf[offsets.STAND_STATE] if offsets.STAND_STATE else 0

        # Multi-byte fields (uint32 / int32)
        spawn_id = struct.unpack_from("<I", buf, offsets.SPAWN_ID)[0]
        owner_id = struct.unpack_from("<I", buf, offsets.OWNER_SPAWN_ID)[0]
        race = struct.unpack_from("<I", buf, offsets.RACE)[0]
        hp_current = struct.unpack_from("<i", buf, offsets.HP_CURRENT)[0]
        hp_max = struct.unpack_from("<i", buf, offsets.HP_MAX)[0]

        # Pointers (for caller use)
        ac_ptr = struct.unpack_from("<I", buf, offsets.ACTORCLIENT_PTR)[0]
        ci_ptr = struct.unpack_from("<I", buf, offsets.CHARINFO_PTR)[0]
        next_ptr = struct.unpack_from("<I", buf, offsets.NEXT)[0]
        prev_ptr = struct.unpack_from("<I", buf, offsets.PREV)[0]
        zone_id = struct.unpack_from("<i", buf, offsets.ZONE_ID)[0]

        return {
            # SpawnData fields
            "spawn_id": spawn_id,
            "name": name,
            "x": x,
            "y": y,
            "z": z,
            "heading": heading,
            "speed": speed,
            "level": level,
            "spawn_type": spawn_type,
            "race": race,
            "mob_class": mob_class,
            "hide": hide,
            "hp_current": hp_current,
            "hp_max": hp_max,
            "owner_id": owner_id,
            "body_state": body_char,
            "target_name": "",
            "velocity_y": vel_y,
            "velocity_x": vel_x,
            "velocity_z": vel_z,
            # Internal keys (stripped before SpawnData construction)
            "_next_ptr": next_ptr,
            "_prev_ptr": prev_ptr,
            "_actorclient_ptr": ac_ptr,
            "_charinfo_ptr": ci_ptr,
            "_no_regen_flag": no_regen,
            "_stand_state_raw": stand_state,
            "_zone_id": zone_id,
        }

    @staticmethod
    def _spawn_from_parsed(fields: dict) -> SpawnData:
        """Construct SpawnData from parsed dict, stripping internal keys."""
        return SpawnData(**{k: v for k, v in fields.items() if not k.startswith("_")})

    def _bulk_read_spawn_node(
        self,
        base: int,
        player_x: float = 0.0,
        player_y: float = 0.0,
        skip_distant_ac: bool = False,
    ) -> tuple[SpawnData, dict]:
        """Bulk-read one spawn node: 1 RPM for entity struct + 1 RPM for NPC target name.

        Returns (SpawnData, parsed_fields_dict) so caller can access internal keys.
        Raises OSError if the bulk read fails.

        If skip_distant_ac=True, skips the AC target_name RPM call for NPCs
        beyond 200u from (player_x, player_y). Threat range is ~40u so distant
        NPC targeting is irrelevant. Saves ~40-60% RPM in full zones.
        """
        buf = self._r._read_bytes(base, SPAWN_BUF_SIZE)  # 1 RPM
        fields = self._parse_spawn_from_buffer(buf)

        # ActorClient target_name: 1 extra RPM per NPC
        ac_ptr = fields["_actorclient_ptr"]
        if fields["spawn_type"] == 1 and ac_ptr and offsets.AC_TARGET_NAME != 0:
            # Skip distant NPCs to reduce RPM calls
            if skip_distant_ac:
                dx = fields["x"] - player_x
                dy = fields["y"] - player_y
                dist_sq = dx * dx + dy * dy
                if dist_sq > 40000:  # 200u squared
                    return self._spawn_from_parsed(fields), fields
            try:
                fields["target_name"] = self._r._read_string(ac_ptr + offsets.AC_TARGET_NAME, max_len=64)
            except OSError:
                pass

        return self._spawn_from_parsed(fields), fields

    def read_spawns(self, player_x: float = 0.0, player_y: float = 0.0) -> list[SpawnData]:
        """Walk the spawn linked list and return all spawns in the zone.

        Uses bulk reads: 1 RPM per spawn node (+ 1 for NPC target name
        within 200u). NEXT/PREV pointers are extracted from the same
        buffer -- zero extra RPM calls for list walk.

        Pass player_x/player_y to enable distance-based AC read filtering.
        """
        spawns: list[SpawnData] = []
        seen: set[int] = set()
        has_pos = player_x != 0.0 or player_y != 0.0

        player_base = self._r._get_spawn_base()
        if player_base == 0:
            return spawns

        # Walk forward from player via NEXT
        current = player_base
        player_prev_ptr = 0  # cached from first node for backward walk
        for _ in range(MAX_SPAWNS):
            if current == 0 or current in seen:
                break
            seen.add(current)
            try:
                spawn, fields = self._bulk_read_spawn_node(
                    current, player_x, player_y, skip_distant_ac=has_pos
                )
                spawns.append(spawn)
                if current == player_base:
                    player_prev_ptr = fields["_prev_ptr"]
                current = fields["_next_ptr"]
            except OSError:
                self._consecutive_spawn_errors += 1
                if self._consecutive_spawn_errors >= 5:
                    log.warning(
                        "[PERCEPTION] Spawn list forward walk: %d consecutive read failures",
                        self._consecutive_spawn_errors,
                    )
                break

        # Walk backward from player via PREV (pointer cached from first node)
        current = player_prev_ptr

        for _ in range(MAX_SPAWNS):
            if current == 0 or current in seen:
                break
            seen.add(current)
            try:
                spawn, fields = self._bulk_read_spawn_node(
                    current, player_x, player_y, skip_distant_ac=has_pos
                )
                spawns.append(spawn)
                current = fields["_prev_ptr"]
            except OSError:
                self._consecutive_spawn_errors += 1
                if self._consecutive_spawn_errors >= 5:
                    log.warning(
                        "[PERCEPTION] Spawn list backward walk: %d consecutive read failures",
                        self._consecutive_spawn_errors,
                    )
                break

        self._consecutive_spawn_errors = 0  # reset on successful walk
        return spawns
