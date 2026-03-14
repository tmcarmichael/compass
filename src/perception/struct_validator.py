"""Struct layout validation for environment compatibility.

Runs at startup to detect target binary mismatches, shifted offsets, or
corrupted pointer chains before the agent reads garbage values. Each check
reads a canary value at a known offset and validates against expected
ranges or patterns.

Design:
  - Each check is independent and returns a CheckResult
  - All checks run even if earlier ones fail (full diagnostic report)
  - Checks use the same _read_* primitives as MemoryReader
  - No side effects -- read-only validation

Usage:
    validator = StructValidator(reader)
    result = validator.validate()
    if not result.compatible:
        log.error("[PERCEPTION] Struct validation FAILED: %s", result.summary)
"""

import logging
import struct
from dataclasses import dataclass, field
from typing import Any

from perception import offsets

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CheckResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(slots=True)
class ValidationResult:
    """Aggregate result of all struct validation checks."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def compatible(self) -> bool:
        """True if all checks passed."""
        return all(c.passed for c in self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def summary(self) -> str:
        total = len(self.checks)
        if self.compatible:
            return f"{total}/{total} checks passed -- client structs compatible"
        failed = [c for c in self.checks if not c.passed]
        names = ", ".join(c.name for c in failed)
        return f"{self.failed_count}/{total} checks FAILED ({names}) -- client may be incompatible"


class StructValidator:
    """Validates target environment struct layouts at startup.

    Reads canary values at known stable offsets to detect:
    - Binary version mismatch (incompatible binary with shifted offsets)
    - Struct layout changes
    - Broken pointer chains (wrong process, corrupted memory)
    - Wrong target process attached
    """

    def __init__(self, reader: Any) -> None:
        """Accept a MemoryReader instance for raw reads."""
        self._r = reader

    def validate(self) -> ValidationResult:
        """Run all struct compatibility checks and return results."""
        checks = [
            self._check_static_pointers(),
            self._check_spawninfo_layout(),
            self._check_spawninfo_name(),
            self._check_profile_chain(),
            self._check_profile_identity(),
            self._check_actorclient_reachable(),
            self._check_contents_vtable(),
            self._check_engine_state(),
        ]
        result = ValidationResult(checks=checks)

        # Log full report
        for c in checks:
            if c.passed:
                log.info("[PERCEPTION] [OK]   %s: %s", c.name, c.detail)
            else:
                log.warning("[PERCEPTION] [FAIL] %s: %s", c.name, c.detail)

        if result.compatible:
            log.info("[PERCEPTION] Struct validation: %s", result.summary)
        else:
            log.warning("[PERCEPTION] Struct validation: %s", result.summary)

        return result

    # -- Individual checks ---------------------------------------------------

    def _check_static_pointers(self) -> CheckResult:
        """Verify root pointers resolve to non-null addresses.

        If they resolve to null, the binary differs from the expected build.
        """
        name = "static_pointers"
        try:
            player_ptr = self._r._read_pointer(offsets.PLAYER_SPAWN_PTR)
            zone_ptr = self._r._read_pointer(offsets.ZONE_PTR)
            engine_ptr = self._r._read_pointer(offsets.GAME_ENGINE_PTR)

            # Player and game engine must be non-null when character is in-world
            if player_ptr == 0:
                return CheckResult(
                    name, False, f"PLAYER_SPAWN_PTR (0x{offsets.PLAYER_SPAWN_PTR:08X}) -> null"
                )
            if engine_ptr == 0:
                return CheckResult(name, False, f"GAME_ENGINE_PTR (0x{offsets.GAME_ENGINE_PTR:08X}) -> null")

            # Validate pointer range (should be in user-mode address space)
            for label, ptr in [("player", player_ptr), ("engine", engine_ptr)]:
                if ptr < 0x10000 or ptr > 0x7FFFFFFF:
                    return CheckResult(name, False, f"{label} pointer 0x{ptr:08X} outside valid range")

            return CheckResult(
                name, True, f"player=0x{player_ptr:08X} engine=0x{engine_ptr:08X} zone=0x{zone_ptr:08X}"
            )
        except OSError as e:
            return CheckResult(name, False, f"read failed: {e}")

    def _check_spawninfo_layout(self) -> CheckResult:
        """Verify entity struct layout by reading player's known fields.

        Reads level, race, class, type from known offsets and validates
        they contain sane values. If these are garbage, offsets have shifted.
        """
        name = "spawninfo_layout"
        try:
            base = self._r._read_pointer(offsets.PLAYER_SPAWN_PTR)
            if base == 0:
                return CheckResult(name, False, "player spawn null")

            # Bulk read the spawn struct once
            buf = self._r._read_bytes(base, 0x02F0)

            # Extract canary fields
            level = buf[offsets.LEVEL]
            race = struct.unpack_from("<I", buf, offsets.RACE)[0]
            cls = buf[offsets.CLASS]
            spawn_type = buf[offsets.TYPE]
            hp_max = struct.unpack_from("<i", buf, offsets.HP_MAX)[0]

            issues = []
            if not (1 <= level <= 65):
                issues.append(f"level={level} (expected 1-65)")
            if not (1 <= race <= 330):
                issues.append(f"race={race} (expected 1-330)")
            if not (1 <= cls <= 16):
                issues.append(f"class={cls} (expected 1-16)")
            if spawn_type != 0:
                issues.append(f"type={spawn_type} (expected 0=Player)")
            if hp_max <= 0:
                issues.append(f"hp_max={hp_max} (expected >0)")

            if issues:
                return CheckResult(name, False, "; ".join(issues))

            return CheckResult(
                name, True, f"lv{level} race={race} class={cls} type={spawn_type} hp_max={hp_max}"
            )
        except OSError as e:
            return CheckResult(name, False, f"read failed: {e}")

    def _check_spawninfo_name(self) -> CheckResult:
        """Verify player name is a readable ASCII string at expected offset."""
        name = "spawninfo_name"
        try:
            base = self._r._read_pointer(offsets.PLAYER_SPAWN_PTR)
            if base == 0:
                return CheckResult(name, False, "player spawn null")

            char_name = self._r._read_string(base + offsets.NAME, 64)

            if not char_name or len(char_name) < 3:
                return CheckResult(name, False, f"name too short: '{char_name}' (expected >=3 chars)")
            if not char_name.isascii():
                return CheckResult(name, False, f"name not ASCII: '{char_name}'")
            # Character names are alphabetic only
            if not char_name.isalpha():
                return CheckResult(name, False, f"name not alphabetic: '{char_name}'")

            return CheckResult(name, True, f"'{char_name}'")
        except OSError as e:
            return CheckResult(name, False, f"read failed: {e}")

    def _check_profile_chain(self) -> CheckResult:
        """Verify the profile pointer chain resolves and guardrails pass.

        Chain: entity struct+0x218 -> character info -> +0x108 -> +0x04 -> profile_base
        Then read level/class/race/mana as guardrails.
        """
        name = "profile_chain"
        try:
            base = self._r._read_pointer(offsets.PLAYER_SPAWN_PTR)
            if base == 0:
                return CheckResult(name, False, "player spawn null")

            # Step 1: CHARINFO
            ci = self._r._read_pointer(base + offsets.CHARINFO_PTR)
            if ci == 0:
                return CheckResult(
                    name, False, f"CHARINFO_PTR (entity struct+0x{offsets.CHARINFO_PTR:03X}) -> null"
                )

            # Step 2: intermediate
            intermediate = self._r._read_pointer(ci + offsets.CHARINFO_PROFILE_INDIR)
            if intermediate == 0:
                return CheckResult(name, False, f"character info+0x{offsets.CHARINFO_PROFILE_INDIR:03X} -> null")

            # Step 3: profile base
            profile_base = self._r._read_pointer(intermediate + offsets.PROFILE_PTR_OFFSET)
            if profile_base == 0:
                return CheckResult(name, False, f"intermediate+0x{offsets.PROFILE_PTR_OFFSET:03X} -> null")

            # Guardrail reads
            g_level = self._r._read_int32(profile_base + offsets.PROFILE_LEVEL)
            g_class = self._r._read_int32(profile_base + offsets.PROFILE_CLASS)
            g_race = self._r._read_int32(profile_base + offsets.PROFILE_RACE)
            g_mana = self._r._read_int32(profile_base + offsets.PROFILE_MANA)

            issues = []
            if not (1 <= g_level <= 60):
                issues.append(f"level={g_level}")
            if not (1 <= g_class <= 16):
                issues.append(f"class={g_class}")
            if not (1 <= g_race <= 330):
                issues.append(f"race={g_race}")
            if not (0 <= g_mana <= 10000):
                issues.append(f"mana={g_mana}")

            if issues:
                return CheckResult(
                    name,
                    False,
                    "guardrails failed: {} at profile_base=0x{:08X}".format(", ".join(issues), profile_base),
                )

            return CheckResult(
                name, True, f"0x{profile_base:08X} -> lv{g_level} class={g_class} race={g_race} mana={g_mana}"
            )
        except OSError as e:
            return CheckResult(name, False, f"read failed: {e}")

    def _check_profile_identity(self) -> CheckResult:
        """Cross-validate entity struct identity fields against profile chain.

        entity struct and profile should agree on level, race, class. Disagreement
        means one struct's offsets have shifted.
        """
        name = "profile_cross_validate"
        try:
            base = self._r._read_pointer(offsets.PLAYER_SPAWN_PTR)
            if base == 0:
                return CheckResult(name, False, "player spawn null")

            # entity struct values
            buf = self._r._read_bytes(base, 0x02F0)
            si_level = buf[offsets.LEVEL]
            si_race = struct.unpack_from("<I", buf, offsets.RACE)[0]
            si_class = buf[offsets.CLASS]

            # Profile values (via chain)
            pb = self._r._resolve_profile_base()
            if pb is None:
                return CheckResult(name, False, "profile chain not resolved")

            p_level = self._r._read_int32(pb + offsets.PROFILE_LEVEL)
            p_race = self._r._read_int32(pb + offsets.PROFILE_RACE)
            p_class = self._r._read_int32(pb + offsets.PROFILE_CLASS)

            mismatches = []
            if si_level != p_level:
                mismatches.append(f"level: spawn={si_level} profile={p_level}")
            if si_race != p_race:
                mismatches.append(f"race: spawn={si_race} profile={p_race}")
            if si_class != p_class:
                mismatches.append(f"class: spawn={si_class} profile={p_class}")

            if mismatches:
                return CheckResult(
                    name, False, "entity struct/profile mismatch: {}".format("; ".join(mismatches))
                )

            return CheckResult(
                name, True, f"entity struct matches profile (lv{si_level} race={si_race} class={si_class})"
            )
        except OSError as e:
            return CheckResult(name, False, f"read failed: {e}")

    def _check_actorclient_reachable(self) -> CheckResult:
        """Verify ActorClient sub-struct pointer resolves and activity_state is sane."""
        name = "actorclient"
        try:
            base = self._r._read_pointer(offsets.PLAYER_SPAWN_PTR)
            if base == 0:
                return CheckResult(name, False, "player spawn null")

            ac_ptr = self._r._read_pointer(base + offsets.ACTORCLIENT_PTR)
            if ac_ptr == 0:
                return CheckResult(
                    name, False, f"ACTORCLIENT_PTR (entity struct+0x{offsets.ACTORCLIENT_PTR:03X}) -> null"
                )

            # Validate pointer range
            if ac_ptr < 0x10000 or ac_ptr > 0x7FFFFFFF:
                return CheckResult(name, False, f"AC pointer 0x{ac_ptr:08X} outside valid range")

            # Read activity state (should be a small integer)
            activity = self._r._read_int32(ac_ptr + offsets.AC_ACTIVITY_STATE)
            # Known values: 32=idle, 5=melee, 13=hit, 18=chase
            if not (0 <= activity <= 100):
                return CheckResult(name, False, f"activity_state={activity} (expected 0-100)")

            # Read combat flag (should be 0 or 1)
            combat_flag = self._r._read_int32(ac_ptr + offsets.AC_COMBAT_FLAG)
            if combat_flag not in (0, 1):
                return CheckResult(name, False, f"combat_flag={combat_flag} (expected 0 or 1)")

            return CheckResult(name, True, f"0x{ac_ptr:08X} activity={activity} combat={combat_flag}")
        except OSError as e:
            return CheckResult(name, False, f"read failed: {e}")

    def _check_contents_vtable(self) -> CheckResult:
        """Verify CONTENTS vtable pointer matches the expected value.

        If this differs, the binary is a different build.
        """
        name = "contents_vtable"
        try:
            pb = self._r._resolve_profile_base()
            if pb is None:
                return CheckResult(name, False, "profile chain not resolved")

            # Read first equipment slot CONTENTS pointer
            equip_ptr = self._r._read_pointer(pb + offsets.PROFILE_EQUIP_START)
            if equip_ptr == 0:
                # No item equipped in first slot -- not a failure, just skip
                return CheckResult(name, True, "slot empty (skipped)")

            # Read vtable from CONTENTS object
            vtable = self._r._read_pointer(equip_ptr)
            if vtable == offsets.CONTENTS_VTABLE:
                return CheckResult(name, True, f"vtable=0x{vtable:08X} (matches expected)")

            return CheckResult(
                name,
                False,
                f"vtable=0x{vtable:08X} (expected 0x{offsets.CONTENTS_VTABLE:08X}) -- "
                "possible client version mismatch",
            )
        except OSError as e:
            return CheckResult(name, False, f"read failed: {e}")

    def _check_engine_state(self) -> CheckResult:
        """Verify engine state indicates character is in-game."""
        name = "engine_state"
        try:
            engine_ptr = self._r._read_pointer(offsets.ENGINE_STATE_PTR)
            if engine_ptr == 0:
                return CheckResult(
                    name, False, f"ENGINE_STATE_PTR (0x{offsets.ENGINE_STATE_PTR:08X}) -> null"
                )

            game_mode = self._r._read_int32(engine_ptr + offsets.ENGINE_GAME_MODE_OFFSET)
            # 0=charselect, 5=ingame, 0xFF=shutdown
            if game_mode == 5:
                return CheckResult(name, True, f"game_mode={game_mode} (in-game)")
            if game_mode == 0:
                return CheckResult(name, False, f"game_mode={game_mode} (char select -- not in world)")

            return CheckResult(name, False, f"game_mode={game_mode} (unexpected value)")
        except OSError as e:
            return CheckResult(name, False, f"read failed: {e}")
