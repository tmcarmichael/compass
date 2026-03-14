"""Difficulty evaluation and npc classification for target selection and threat assessment.

Relative difficulty is determined by level difference between the agent and target.
At low levels the ranges are tighter; this uses the classic level-difference formula.

Disposition (faction standing) determines whether a npc will attack on sight.
This is separate from difficulty level  -  a dangerous npc can be passive (won't threat),
while an easy npc can be hostile (will attack on sight).
"""

from typing import Any

from core.types import Con, Disposition, Point
from eq.strings import normalize_mob_name
from perception.queries import (
    is_pet,
)
from perception.state import SpawnData

__all__ = [
    "Con",
    "Disposition",
    "con_color",
    "FIGHTABLE_CONS",
    "THREAT_CONS",
    "AGGRESSIVE_DISPOSITIONS",
    "PASSIVE_DISPOSITIONS",
    "get_avoid_names",
    "set_avoid_names",
    "get_zone_avoid_mobs",
    "set_zone_avoid_mobs",
    "configure_avoid_names",
    "get_disposition",
    "is_valid_target",
    "is_threat",
    "find_targets",
    "is_pet",
]


# Con is re-exported from core.types for backward compatibility.
# Brain modules should import from core.types directly.


def _con_from_diff(diff: int, yellow_floor: int, blue_floor: int, light_blue_floor: int) -> Con:
    """Map a level difference to a con color given band thresholds."""
    if diff >= 3:
        return Con.RED
    if diff >= yellow_floor:
        return Con.YELLOW
    if diff == 0:
        return Con.WHITE
    if diff >= blue_floor:
        return Con.BLUE
    if diff >= light_blue_floor:
        return Con.LIGHT_BLUE
    return Con.GREEN


def con_color(player_level: int, mob_level: int) -> Con:
    """Determine the difficulty level of a npc relative to the player.

    Uses the level-difference formula. At very low levels (1-7), the ranges
    are compressed  -  even a 1-2 level difference matters a lot.
    """
    diff = mob_level - player_level
    if player_level < 8:
        return _con_from_diff(diff, yellow_floor=1, blue_floor=-2, light_blue_floor=-3)
    if player_level < 20:
        return _con_from_diff(diff, yellow_floor=1, blue_floor=-3, light_blue_floor=-5)
    return _con_from_diff(diff, yellow_floor=1, blue_floor=-5, light_blue_floor=-9)


# Con colors that are safe to fight
FIGHTABLE_CONS = frozenset({Con.YELLOW, Con.WHITE, Con.BLUE, Con.LIGHT_BLUE})

# Con colors that represent threats
THREAT_CONS = frozenset({Con.YELLOW, Con.RED})

# Faction standings that mean the npc will threat on sight (KOS)
# DUBIOUS is the lowest NON-hostile disposition
AGGRESSIVE_DISPOSITIONS = frozenset(
    {
        Disposition.SCOWLING,
        Disposition.READY_TO_ATTACK,
        Disposition.THREATENING,
    }
)

# Faction standings that mean the npc won't threat on sight
PASSIVE_DISPOSITIONS = frozenset(
    {
        Disposition.ALLY,
        Disposition.WARMLY,
        Disposition.KINDLY,
        Disposition.AMIABLE,
        Disposition.INDIFFERENT,
        Disposition.APPREHENSIVE,
        Disposition.DUBIOUS,  # lowest non-hostile standing
    }
)

# Default npc name prefixes to always avoid (non-combat or too dangerous).
# Overridden by zone config [avoid_npcs]; these are fallback examples only.
_DEFAULT_AVOID_PREFIXES = frozenset(
    {
        "Guard",
        "Captain",
        "Merchant",
        "Shopkeeper",
        "Banker",
        "Guildmaster",
    }
)
_avoid_names: frozenset[str] = frozenset(_DEFAULT_AVOID_PREFIXES)

# Zone-specific avoid npcs  -  populated from zone config at startup.
# Rebuilt atomically on config change; GIL-safe reference swap.
_zone_avoid_mobs: frozenset[str] = frozenset()


def get_avoid_names() -> frozenset[str]:
    """Return the current avoid-names frozenset (thread-safe read)."""
    return _avoid_names


def set_avoid_names(names: frozenset[str]) -> None:
    """Atomically replace the avoid-names set (GIL-safe reference swap)."""
    global _avoid_names
    _avoid_names = names


def get_zone_avoid_mobs() -> frozenset[str]:
    """Return the current zone-avoid-npcs frozenset (thread-safe read)."""
    return _zone_avoid_mobs


def set_zone_avoid_mobs(names: frozenset[str]) -> None:
    """Atomically replace the zone-avoid-npcs set (GIL-safe reference swap)."""
    global _zone_avoid_mobs
    _zone_avoid_mobs = names


def configure_avoid_names(zone_config: dict[str, Any]) -> None:
    """Update avoid names from zone config [avoid_npcs] section.

    Called at startup from agent.py. Falls back to defaults if no config.
    Atomically swaps the frozenset reference (GIL-safe).
    """
    avoid_cfg = zone_config.get("avoid_npcs", {})
    names: set[str] = set()
    names.update(avoid_cfg.get("global_prefixes", _DEFAULT_AVOID_PREFIXES))
    names.update(avoid_cfg.get("zone_specific", []))
    set_avoid_names(frozenset(names))


def get_disposition(
    name: str,
    zone_dispositions: dict[str, list[str]] | None,
) -> Disposition:
    """Look up a npc's disposition from zone config data.

    zone_dispositions maps disposition names (lowercase) to lists of name prefixes.
    Returns Disposition.UNKNOWN if no match found.
    """
    if not zone_dispositions:
        return Disposition.UNKNOWN
    for disp_name, prefixes in zone_dispositions.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                try:
                    return Disposition[disp_name.upper()]
                except KeyError:
                    continue
    return Disposition.UNKNOWN


def is_valid_target(spawn: SpawnData, player_level: int) -> bool:
    """Check if a spawn is a valid combat target."""
    if not spawn.is_npc:
        return False
    # body_state offset is only reliable for the player character.
    # NPC body_state bytes contain unrelated data (e.g. 'd' for alive npcs).
    # Cross-reference with HP: a "dead" NPC at full HP is clearly alive.
    if spawn.hp_current <= 0:
        return False
    if is_pet(spawn):
        return False
    if spawn.level == 0:
        return False  # Likely a non-combat NPC or object
    # Check if the name suggests a guard/merchant or zone-avoided npc
    for avoid in _avoid_names:
        if avoid in spawn.name:
            return False
    mob_base = normalize_mob_name(spawn.name)
    if mob_base in _zone_avoid_mobs:
        return False
    # Must be a fightable con
    con = con_color(player_level, spawn.level)
    return con in FIGHTABLE_CONS


def is_threat(
    spawn: SpawnData,
    player_level: int,
    zone_dispositions: dict[str, list[str]] | None = None,
) -> bool:
    """Check if a spawn is a threat (will aggro AND is too dangerous to fight).

    A npc is only a threat if it is both aggressive and above fightable difficulty.
    Aggressive npcs at fightable difficulty levels are targets, not threats.

    Uses disposition data if available:
    - Known passive npcs are NOT threats regardless of difficulty
    - Known aggressive npcs are threats only if YELLOW/RED difficulty
    - Unknown npcs fall back to difficulty-based assessment
    """
    if not spawn.is_npc:
        return False
    if is_pet(spawn):
        return False

    con = con_color(player_level, spawn.level)
    disp = get_disposition(spawn.name, zone_dispositions)

    if disp in PASSIVE_DISPOSITIONS:
        return False  # Won't threat, not a threat

    if disp in AGGRESSIVE_DISPOSITIONS:
        # Aggressive but fightable difficulty = target, not threat
        return con in THREAT_CONS

    # UNKNOWN: fall back to con-based assessment
    return con in THREAT_CONS


def find_targets(
    spawns: tuple[SpawnData, ...],
    player_x: float,
    player_y: float,
    player_level: int,
    max_distance: float = 200.0,
    zone_dispositions: dict[str, list[str]] | None = None,
    resource_mob_names: set[str] | None = None,
) -> list[tuple[SpawnData, float, Con, Disposition]]:
    """Find viable combat targets from the spawn list.

    Returns list of (spawn, distance, con, disposition) sorted by preference:
    - White con first, then blue, then light blue
    - Within same con, closer targets first

    Only includes npcs within max_distance.
    resource_mob_names: set of base npc names (e.g. {"a_decaying_skeleton", "a_skeleton"})
        that should be included regardless of con color (for resource collection).
    """
    candidates = []
    for spawn in spawns:
        # Check if this is a resource target (bypass con filter)
        is_resource = False
        if resource_mob_names and spawn.is_npc and not is_pet(spawn):
            mob_base = normalize_mob_name(spawn.name)
            if mob_base in resource_mob_names:
                is_resource = True

        if not is_resource and not is_valid_target(spawn, player_level):
            continue
        dist = Point(player_x, player_y, 0.0).dist_to(spawn.pos)
        if dist > max_distance:
            continue
        con = con_color(player_level, spawn.level)
        disp = get_disposition(spawn.name, zone_dispositions)
        candidates.append((spawn, dist, con, disp))

    # Sort: white first, then blue, then light blue. Within each, by distance.
    con_priority = {Con.WHITE: 0, Con.BLUE: 1, Con.LIGHT_BLUE: 2}
    candidates.sort(key=lambda t: (con_priority.get(t[2], 99), t[1]))
    return candidates


# ======================================================================
# Dynamic LOS (npc collision detection)
# ======================================================================

# Approximate collision radius by NPC size. Small npcs are ~2-3u,
# large npcs up to ~20u. Most humanoid npcs are ~3-5u. Conservative default.
_DEFAULT_COLLISION_RADIUS = 3.5


def check_mob_blocked_los(
    caster_x: float,
    caster_y: float,
    caster_z: float,
    target_x: float,
    target_y: float,
    target_z: float,
    spawns: tuple[SpawnData, ...],
    target_id: int = 0,
    collision_radius: float = _DEFAULT_COLLISION_RADIUS,
) -> SpawnData | None:
    """Check if any NPC's collision sphere blocks the line between caster and target.

    Tests the 3D line from caster to target against the collision sphere
    of every nearby NPC (excluding the target itself). Returns the first
    blocking spawn, or None if LOS is clear.

    This catches cases where terrain LOS is clear but a large npc
    (dragon, giant) is physically between the caster and target,
    blocking spells.

    Args:
        target_id: spawn_id of intended target (excluded from checks).
        collision_radius: assumed collision sphere radius for NPCs.

    Returns:
        The SpawnData of the blocking npc, or None if clear.
    """
    dx = target_x - caster_x
    dy = target_y - caster_y
    dz = target_z - caster_z
    line_len_sq = dx * dx + dy * dy + dz * dz
    if line_len_sq < 1.0:
        return None  # caster and target nearly overlapping

    inv_len_sq = 1.0 / line_len_sq

    for spawn in spawns:
        if spawn.spawn_id == target_id:
            continue
        if not spawn.is_npc or spawn.is_corpse:
            continue

        # Vector from caster to this npc
        mx = spawn.x - caster_x
        my = spawn.y - caster_y
        mz = spawn.z - caster_z

        # Project npc center onto caster->target line
        t = (mx * dx + my * dy + mz * dz) * inv_len_sq
        if t < 0.05 or t > 0.95:
            continue  # npc is behind caster or beyond target

        # Closest point on line to npc center
        cx = caster_x + dx * t - spawn.x
        cy = caster_y + dy * t - spawn.y
        cz = caster_z + dz * t - spawn.z
        dist_sq = cx * cx + cy * cy + cz * cz

        if dist_sq < collision_radius * collision_radius:
            return spawn

    return None
