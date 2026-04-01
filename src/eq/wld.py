"""Parse zone geometry files.

Geometry format parser stubbed in public release. Data classes define
the schema for terrain mesh, BSP, and placement data.

Geometry files live inside compressed archives and contain zone terrain
meshes, BSP spatial partitions, material definitions, and region markers
(water, lava, zone lines).
"""

from dataclasses import dataclass

# ======================================================================
# Data classes
# ======================================================================


@dataclass(frozen=True, slots=True)
class MeshVertex:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class MeshTriangle:
    """A triangle referencing vertex indices + its material."""

    v1: int
    v2: int
    v3: int
    flags: int
    material_idx: int


@dataclass(slots=True)
class Mesh:
    """Zone terrain mesh extracted from a 0x36 DmSpriteDef2 fragment."""

    name: str
    vertices: list[MeshVertex]
    triangles: list[MeshTriangle]
    center: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class BSPNode:
    """One node in the zone's BSP tree (fragment 0x21)."""

    normal_x: float
    normal_y: float
    normal_z: float
    split_distance: float
    region: int  # 1-based region index (0 = internal/branch node)
    front: int  # 1-based child index (0 = none)
    back: int  # 1-based child index (0 = none)


@dataclass(frozen=True, slots=True)
class ObjectPlacement:
    """A placed object instance in the zone (fragment 0x15).

    Coordinates are in WLD space (wld_x = game_y, wld_y = game_x).
    The heightmap converts internally when applying obstacles.
    """

    name: str  # fragment name
    model_name: str  # resolved model name (stripped _ACTORDEF suffix)
    x: float  # WLD x coordinate
    y: float  # WLD y coordinate
    z: float  # WLD z coordinate
    heading: float  # raw 0-512 heading
    scale: float  # uniform scale factor (from scale.y)


class RegionType:
    """Region type constants from fragment 0x29 name prefixes.

    Prefixes: WT/WTN=water, LA/LAN=lava, DRNTP=zoneline, DRP=pvp,
    SLN=waterBlockLos, VWN=freezingWater, DRN+_S_=slippery.
    """

    NORMAL = 0
    WATER = 1
    LAVA = 2
    ZONELINE = 3
    PVP = 4
    WATER_BLOCK_LOS = 5  # underwater blocking LOS
    FREEZING_WATER = 6  # freezing water
    SLIPPERY = 7  # slippery surfaces (ice)
    VWATER = 8  # visual water surface (no swim)

    # Backward compat aliases
    SLIME = WATER_BLOCK_LOS
    ICE = SLIPPERY

    @classmethod
    def from_name(cls, name: str) -> int | None:
        """Determine region type from a 0x29 fragment name prefix.

        Prefix matching order matters  -  longer prefixes checked first.
        Region string parsing based on known prefix conventions.
        """
        lower = name.lower()
        # Zoneline variants (water+zoneline, lava+zoneline, drain+zoneline)
        if lower.startswith("wtntp") or lower.startswith("drntp") or lower.startswith("lantp"):
            return cls.ZONELINE
        # Water: wt_, wtn_
        if lower.startswith("wt_") or lower.startswith("wtn_"):
            return cls.WATER
        # Lava: la_, lan_
        if lower.startswith("la_") or lower.startswith("lan_"):
            return cls.LAVA
        # WaterBlockLos: sln_
        if lower.startswith("sln_"):
            return cls.WATER_BLOCK_LOS
        # FreezingWater: vwn_
        if lower.startswith("vwn_"):
            return cls.FREEZING_WATER
        # PvP: drp_
        if lower.startswith("drp_"):
            return cls.PVP
        # Slippery: drn_ with _s_ in name
        if lower.startswith("drn_") and "_s_" in lower:
            return cls.SLIPPERY
        # Visual water: vwa
        if lower.startswith("vwa"):
            return cls.VWATER
        # Drain (generic): drn_ without _s_  -  treat as normal
        if lower.startswith("drn_"):
            return cls.NORMAL
        return None


# ======================================================================
# WLD parser
# ======================================================================


class WLDFile:
    """Parse a WLD file and extract terrain geometry + spatial data."""

    MAGIC = 0x54503D02
    VERSION_OLD = 0x00015500
    VERSION_NEW = 0x1000C800

    def __init__(self, data: bytes) -> None:
        raise NotImplementedError(
            "WLDFile parser stubbed in public release. Provide an environment-specific implementation."
        )

    def extract_meshes(self) -> list[Mesh]:
        """Extract all terrain meshes (fragment 0x36)."""
        raise NotImplementedError("WLDFile parser stubbed in public release.")

    def extract_bsp_nodes(self) -> list[BSPNode]:
        """Extract BSP tree nodes (fragment 0x21)."""
        raise NotImplementedError("WLDFile parser stubbed in public release.")

    def extract_region_types(self) -> dict[int, int]:
        """Map region_id -> RegionType from fragment 0x29 entries."""
        raise NotImplementedError("WLDFile parser stubbed in public release.")

    def extract_placements(self) -> list[ObjectPlacement]:
        """Extract object placement instances (fragment 0x15)."""
        raise NotImplementedError("WLDFile parser stubbed in public release.")

    def extract_actor_mesh_refs(self) -> dict[str, int]:
        """Map actor name -> 0x36 mesh fragment index from 0x14 ActorDefs."""
        raise NotImplementedError("WLDFile parser stubbed in public release.")

    def extract_actor_meshes(self) -> dict[str, Mesh]:
        """Map actor model name -> resolved Mesh object."""
        raise NotImplementedError("WLDFile parser stubbed in public release.")

    def extract_mesh_material_names(self) -> dict[str, list[str]]:
        """Map lowercased mesh name -> [texture filename per material_idx]."""
        raise NotImplementedError("WLDFile parser stubbed in public release.")

    def fragment_type_summary(self) -> dict[int, int]:
        """Return {fragment_type: count} for diagnostic inspection."""
        raise NotImplementedError("WLDFile parser stubbed in public release.")
