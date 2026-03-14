"""Spawn query helpers and spawn classification.

Centralizes spawn iteration patterns (filter-distance logic) and pet
detection heuristics. Both are perception-layer concerns: they classify
raw spawn data without making brain-level decisions.
"""

from __future__ import annotations

from collections.abc import Iterator

from core.types import Point
from perception.state import GameState, SpawnData

# ---------------------------------------------------------------------------
#  Pet detection (spawn classification, re-exported by combat_eval.py)
# ---------------------------------------------------------------------------


def _looks_like_pet_name(name: str) -> bool:
    """Heuristic: pet names are a single capitalized word followed by 3 digits.
    NPC names use underscores (a_fire_beetle) or start lowercase."""
    if len(name) < 7:
        return False
    if not name[-3:].isdigit():
        return False
    prefix = name[:-3]
    return prefix.isalpha() and prefix[0].isupper() and "_" not in name


def is_pet(spawn: SpawnData, player_spawn_id: int = 0) -> bool:
    """Check if a spawn is a player pet.

    Primary: owner_id field. Definitive when non-zero.
    Fallback: name pattern heuristic.
    """
    if not spawn.is_npc:
        return False

    # Definitive: owner_id links pet to owner.
    if spawn.owner_id != 0:
        return True

    return _looks_like_pet_name(spawn.name)


def is_our_pet(spawn: SpawnData, player_spawn_id: int) -> bool:
    """Check if a spawn is OUR pet (not another player's).

    Uses owner_id for definitive matching. Falls back to is_pet() name
    heuristic when owner_id is unavailable (0).
    """
    if spawn.owner_id != 0:
        result: bool = spawn.owner_id == player_spawn_id
        return result
    # owner_id not available -- can't distinguish, fall back to generic check
    return is_pet(spawn)


# ---------------------------------------------------------------------------
#  Spawn list queries
# ---------------------------------------------------------------------------


def live_npcs(state: GameState, *, exclude_pets: bool = True) -> Iterator[SpawnData]:
    """Yield living NPCs (excludes corpses, players, dead/feigning npcs, optionally pets)."""
    for spawn in state.spawns:
        if not spawn.is_npc or spawn.hp_current <= 0:
            continue
        if spawn.is_feigning or spawn.is_dead_body:
            continue
        if exclude_pets and is_pet(spawn):
            continue
        yield spawn


def nearby_live_npcs(
    state: GameState,
    pos: Point,
    radius: float,
    *,
    exclude_pets: bool = True,
) -> list[tuple[SpawnData, float]]:
    """Return (spawn, distance) pairs for live NPCs within radius, sorted by distance."""
    results: list[tuple[SpawnData, float]] = []
    for spawn in live_npcs(state, exclude_pets=exclude_pets):
        d = pos.dist_to(spawn.pos)
        if d <= radius:
            results.append((spawn, d))
    results.sort(key=lambda pair: pair[1])
    return results


def count_nearby_npcs(
    state: GameState,
    pos: Point,
    radius: float,
    *,
    exclude_id: int = 0,
    exclude_pets: bool = True,
) -> int:
    """Count living NPCs within radius of a point."""
    count = 0
    for spawn in live_npcs(state, exclude_pets=exclude_pets):
        if exclude_id and spawn.spawn_id == exclude_id:
            continue
        d = pos.dist_to(spawn.pos)
        if d < radius:
            count += 1
    return count


def count_nearby_social(
    target: SpawnData,
    state: GameState,
    social_groups: dict[str, frozenset[str]],
    radius: float = 50.0,
) -> int:
    """Count social-group members near a target npc."""
    if not social_groups:
        return 0
    base = target.name.rstrip("0123456789")
    group = social_groups.get(base)
    if not group:
        return 0
    count = 0
    for spawn in live_npcs(state, exclude_pets=False):
        if spawn.spawn_id == target.spawn_id:
            continue
        spawn_base = spawn.name.rstrip("0123456789")
        if spawn_base in group:
            d = target.pos.dist_to(spawn.pos)
            if d < radius:
                count += 1
    return count
