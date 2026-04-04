"""Shared type definitions, Protocols, and enums for the compass package.

Use these instead of bare callables and magic strings.
"""

from __future__ import annotations

import math
from enum import StrEnum, unique
from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    from perception.state import GameState

__all__ = [
    "Con",
    "PerceptionProvider",
    "ReadStateFn",
    "PlanType",
    "LootMode",
    "DeathRecoveryMode",
    "CampType",
    "ManaMode",
    "GrindStyle",
    "Disposition",
    "FailureCategory",
    "SpellOutcome",
    "TravelMode",
    "Point",
    "DangerZone",
    "TravelWaypoint",
    "Waypoints",
    "DispositionMap",
    "normalize_entity_name",
]


class PerceptionProvider(Protocol):
    """Interface contract between perception and brain layers.

    Any perception implementation (live memory reader, replay file,
    test mock) must satisfy this protocol. The brain never imports
    from perception directly -- it depends on this contract.
    """

    def read_state(self) -> GameState: ...


class ReadStateFn(Protocol):
    """Callable that returns the current game state snapshot."""

    def __call__(self) -> GameState: ...


class PlanType(StrEnum):
    """Active plan types for ctx.plan.active."""

    TRAVEL = "travel"
    NEEDS_MEMORIZE = "needs_memorize"


class LootMode(StrEnum):
    """Loot filtering modes."""

    OFF = "off"
    SMART = "smart"
    ALL = "all"


class DeathRecoveryMode(StrEnum):
    """Death recovery behavior modes."""

    OFF = "off"
    SMART = "smart"
    ON = "on"


class CampType(StrEnum):
    """Camp geometry type.

    CIRCULAR -- center + roam_radius (default, existing behavior)
    LINEAR   -- 2+ waypoint polyline + corridor_width
    """

    CIRCULAR = "circular"
    LINEAR = "linear"


class ManaMode(StrEnum):
    """Mana usage aggressiveness during combat.

    LOW    -- pet solos everything, cast only emergency lifetap (HP < 50%)
    MEDIUM -- cast on WHITE+ cons, skip BLUE/LIGHT_BLUE (default, existing)
    HIGH   -- cast on all targets, maximum DPS, most rest time
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GrindStyle(StrEnum):
    """Grinding behavior profiles.

    WANDER    -- roam hunting zone, tab-target, pull, fight (default)
    FEAR_KITE -- pull with Fear, stack DoTs while npc runs (L16+)
    CAMP_SIT  -- fixed position, engage only npcs within radius (semi-AFK)
    """

    WANDER = "wander"
    FEAR_KITE = "fear_kite"
    CAMP_SIT = "camp_sit"


@unique
class Disposition(StrEnum):
    """EQ faction standing -- determines if a npc will attack on sight.

    Ordered from friendliest to most hostile. Npcs at THREATENING or worse
    will threat on sight. Configure per-zone in config/zones/<zone>.toml.
    """

    ALLY = "ally"
    WARMLY = "warmly"
    KINDLY = "kindly"
    AMIABLE = "amiable"
    INDIFFERENT = "indifferent"
    APPREHENSIVE = "apprehensive"
    DUBIOUS = "dubious"
    THREATENING = "threatening"
    READY_TO_ATTACK = "ready_to_attack"
    SCOWLING = "scowling"
    UNKNOWN = "unknown"


@unique
class FailureCategory(StrEnum):
    """Failure classification taxonomy for routine failures.

    Every routine FAILURE must classify its failure_category so that
    post-session analysis can distinguish perception bugs from execution
    bugs from environmental interference.
    """

    PERCEPTION = "perception"  # memory read failed, stale pointer, garbage data
    SCORING = "scoring"  # target scored wrong, bad evaluation
    PLANNING = "planning"  # wrong rule fired, plan conflict
    PRECONDITION = "precondition"  # not in range, no mana, pet dead, no targets
    EXECUTION = "execution"  # cast fizzle, movement stuck, window not opened
    ENVIRONMENT = "environment"  # add, player nearby, camp drift, npc despawn
    DESYNC = "desync"  # state mismatch (thought sitting, was standing)
    TIMEOUT = "timeout"  # deadline exceeded without resolution
    UNKNOWN = "unknown"  # unclassified


class SpellOutcome(StrEnum):
    """Spell cast outcome from EQ combat log.

    Set on ctx.combat.last_cast_result each tick by brain_tick_handlers.
    Consumed by combat, pull, and strategy routines to decide retry behavior.
    """

    NONE = ""  # no event (success or not yet cast)
    FIZZLE = "fizzle"  # spell failed randomly -- retry
    LOS_BLOCKED = "los_blocked"  # target not in line of sight -- reposition
    INTERRUPTED = "interrupted"  # hit or moved during cast -- wait/retry
    MUST_STAND = "must_stand"  # tried to cast while sitting -- stand first


class TravelMode(StrEnum):
    """Travel leg mode -- how to traverse this segment."""

    PATHFIND = "pathfind"
    MANUAL = "manual"


@unique
class Con(StrEnum):
    """Relative difficulty level. Environment-neutral; computed from level delta.

    Ordered from easiest to most dangerous. The perception layer maps
    game-specific level calculations into these categories; the brain
    consumes them without knowing how they were derived.
    """

    GREEN = "green"
    LIGHT_BLUE = "light_blue"
    BLUE = "blue"
    WHITE = "white"
    YELLOW = "yellow"
    RED = "red"


def normalize_entity_name(name: str) -> str:
    """Strip trailing instance digits and underscores, lowercase.

    Entity names often carry instance suffixes (e.g. 'a_black_bear007').
    This produces a canonical base name for history keys, disposition
    lookups, and loot matching.
    """
    return name.rstrip("0123456789").rstrip("_").lower()


# -- Location types --


class Point(NamedTuple):
    """World-space position in EQ state coordinates (x, y, z).

    The canonical position type. Every world position is a Point.
    Subsystems that operate in 2D (A*, polyline, minimap) use .x and .y --
    the XY projection of a 3D point. 2D is an operation, not a type.

    EQ /loc displays Y, X, Z (not X, Y, Z). Use from_loc() to convert.
    Prefer .x/.y/.z attribute access over tuple unpacking or indexing.
    """

    x: float
    y: float
    z: float

    @classmethod
    def from_loc(cls, loc_y: float, loc_x: float, loc_z: float) -> Point:
        """Convert /loc format (Y, X, Z) to state coords (x, y, z)."""
        return cls(x=loc_x, y=loc_y, z=loc_z)

    def dist_to(self, other: Point) -> float:
        """3D world-space distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def dist_2d(self, other: Point) -> float:
        """2D distance (XY plane) to another point."""
        return math.hypot(self.x - other.x, self.y - other.y)


class DangerZone(NamedTuple):
    """Named cylindrical danger area in state coordinates.

    Radius is 2D (horizontal distance). The z coordinate anchors the
    danger zone vertically so threats on different floors are excluded.
    """

    x: float
    y: float
    z: float
    radius: float
    name: str

    @property
    def pos(self) -> Point:
        return Point(self.x, self.y, self.z)


class TravelWaypoint(NamedTuple):
    """Tunnel/travel waypoint with optional action."""

    x: float
    y: float
    z: float
    action: str = ""

    @property
    def pos(self) -> Point:
        return Point(self.x, self.y, self.z)


# -- Type aliases for widely-used compound types --
type Waypoints = list[Point]
type DispositionMap = dict[str, list[str]]
