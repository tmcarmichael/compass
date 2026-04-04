"""Build a grid heightmap from EQ zone mesh data.

The heightmap stores ground Z and surface type (walkable, water, lava, steep)
for each cell in a 2D grid covering the zone. This enables O(1) terrain
queries: "what's the ground height at (x, y)?" and "is this point water?"

Coordinate convention:
    WLD mesh uses (wld_x, wld_y) = (game_state.y, game_state.x).
    All public methods accept game coordinates (state.x, state.y) and
    convert internally. Callers never need to think about WLD axes.
"""

from __future__ import annotations

import logging
import math
import struct
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from core.types import Point
from eq.wld import BSPNode, Mesh, MeshVertex, RegionType

if TYPE_CHECKING:
    from eq.wld import ObjectPlacement

log = logging.getLogger(__name__)

# Surface type flags (uint16 bitfield -- v3 cache expanded from uint8)
SURFACE_NONE = 0x0000  # no mesh data at this cell
SURFACE_WALKABLE = 0x0001  # solid ground, slope < 45 deg
SURFACE_STEEP = 0x0002  # too steep to walk (> 45 deg slope)
SURFACE_WATER = 0x0004  # underwater region
SURFACE_LAVA = 0x0008  # lava region
SURFACE_ZONELINE = 0x0010  # zone transition area
SURFACE_CLIFF = 0x0020  # large Z drop to adjacent cell
SURFACE_OBSTACLE = 0x0040  # static object (tree, rock, building, pillar)
SURFACE_BRIDGE = 0x0080  # bridge deck (walkable over water, auto-detected)

# Max Z difference between agent and cell surface to consider walkable.
# Shared with pathfinding (which defines its own copy for module independence).
_WALKABLE_CLIMB = 15.0
_NAN = float("nan")

# Material IDs (uint8, stored per cell in v3 cache)
MAT_UNKNOWN = 0
MAT_STONE = 1
MAT_WOOD = 2
MAT_DIRT = 3
MAT_GRASS = 4
MAT_WATER = 5
MAT_METAL = 6
MAT_BARK = 7
MAT_LAVA = 8

# Texture name patterns -> material ID.
# Checked in order; first match wins. Patterns are substrings of the
# lowercased texture filename. Ordered from most specific to least
# to avoid false positives (e.g., "stone" in "limestone_cave_ceiling"
# is acceptable -- bridge detection has normal + Z-gap guards).
_MAT_PATTERNS: list[tuple[int, tuple[str, ...]]] = [
    (MAT_WATER, ("water", "river", "lake", "pond", "ocean", "stream", "swamp")),
    (MAT_LAVA, ("lava", "magma")),
    (MAT_WOOD, ("wood", "plank", "board", "lumber", "timber", "dock", "pier", "bridge", "brid")),
    (MAT_STONE, ("stone", "brick", "cobble", "marble", "granite", "slate", "flagstone", "paver")),
    (MAT_DIRT, ("dirt", "sand", "ground", "mud", "earth", "soil")),
    (MAT_GRASS, ("grass", "turf", "lawn", "moss")),
    (MAT_METAL, ("metal", "iron", "steel", "copper", "bronze", "chain")),
    (MAT_BARK, ("bark", "trunk", "root", "stump")),
]

# Materials that indicate a walkable bridge deck over water
_BRIDGE_MATERIALS = frozenset({MAT_STONE, MAT_WOOD, MAT_METAL})


def classify_material(texture_name: str) -> int:
    """Classify a WLD texture filename to a material ID.

    EQ water textures follow a consistent naming convention: 'w1', 'w2',
    'w3', 'w4' (short names, always lowercase). These are the actual
    water surface mesh materials -- far more precise than BSP volumes.
    """
    if not texture_name:
        return MAT_UNKNOWN
    t = texture_name.lower()

    # EQ water convention: 'w' followed by 1-2 digits, nothing else
    if len(t) <= 3 and t[0] == "w" and t[1:].isdigit():
        return MAT_WATER

    for mat_id, patterns in _MAT_PATTERNS:
        for p in patterns:
            if p in t:
                return mat_id
    return MAT_UNKNOWN


class ZoneTerrain:
    """Grid-based terrain heightmap with hazard detection.

    Built from zone mesh data (WLD fragment 0x36) and BSP region flags.
    Provides O(1) terrain queries in game coordinates.
    """

    # Cache file format
    _CACHE_MAGIC = b"EQTM"
    _CACHE_VERSION = 3

    def __init__(self, cell_size: float = 1.0) -> None:
        self.cell_size = cell_size
        # Grid bounds in WLD coordinates
        self._min_x: float = 0.0
        self._min_y: float = 0.0
        self._cols: int = 0
        self._rows: int = 0
        # Grid data: flat arrays indexed by row * cols + col
        self._z: list[float] = []  # ground height (highest upward surface)
        self._z_ceiling: list[float] = []  # upper Z for multi-level cells (NaN=single)
        self._flags: list[int] = []  # SURFACE_* bitfield (uint16 in v3)
        self._normal_z: list[float] = []  # steepness
        self._material_id: list[int] = []  # surface material (MAT_* constants)
        self._region_id: list[int] = []  # BSP region ID per cell

        # BSP tree for water/lava point queries
        self._bsp_nodes: list[BSPNode] = []
        self._region_types: dict[int, int] = {}

        # Per-instance override/avoidance lists (must NOT be class-level
        # or all ZoneTerrain instances share the same mutable lists)
        self._walkable_overrides: list[tuple[float, float, float, float]] = []
        self._water_overrides: list[tuple[float, float, float, float]] = []
        self._avoidance_zones: list[tuple[float, float, float]] = []
        self._dynamic_avoidance: list[tuple[float, float, float]] = []

        # Precomputed walkability bitfield for fast JPS scanning.
        # One bit per cell (row-major, LSB = lowest column in byte).
        # Eliminates coord conversion + override iteration from hot path.
        # NOTE: bitfield always treats SURFACE_OBSTACLE as blocked
        # regardless of the obstacle_avoidance feature flag. If the flag
        # is toggled at runtime, call invalidate_walk_bits() to rebuild.
        self._walk_bits: bytearray = bytearray()
        self._walk_byte_cols: int = 0

    # ------------------------------------------------------------------
    # Public API (all in GAME coordinates: state.x, state.y)
    # ------------------------------------------------------------------

    def get_z(self, game_x: float, game_y: float) -> float:
        """Ground height at game position. Returns NaN if unknown."""
        col, row = self._game_to_grid(game_x, game_y)
        idx = self._grid_idx(col, row)
        if idx < 0:
            return float("nan")
        return self._z[idx]

    def get_flags(self, game_x: float, game_y: float) -> int:
        """Surface flags at game position."""
        col, row = self._game_to_grid(game_x, game_y)
        idx = self._grid_idx(col, row)
        if idx < 0:
            return SURFACE_NONE
        return self._flags[idx]

    def add_walkable_override(self, x_min: float, y_min: float, x_max: float, y_max: float) -> None:
        """Mark a rectangular region as walkable (e.g. bridge over water)."""
        self._walkable_overrides.append(
            (min(x_min, x_max), min(y_min, y_max), max(x_min, x_max), max(y_min, y_max))
        )
        self._patch_walk_bits_rect(x_min, y_min, x_max, y_max, walkable=True)

    def add_water_override(self, x_min: float, y_min: float, x_max: float, y_max: float) -> None:
        """Mark a rectangular region as water (e.g. river banks that look dry)."""
        self._water_overrides.append(
            (min(x_min, x_max), min(y_min, y_max), max(x_min, x_max), max(y_min, y_max))
        )
        self._patch_walk_bits_rect(x_min, y_min, x_max, y_max, walkable=False)

    def add_avoidance_zone(self, x: float, y: float, radius: float) -> None:
        """Mark a circular area as dangerous  -  A* will route around it."""
        self._avoidance_zones.append((x, y, radius))

    def update_dynamic_avoidance(self, zones: list[tuple[float, float, float]]) -> None:
        """Replace all dynamic avoidance zones (from live threats).

        Called each tick by brain_runner to feed active threat positions
        into A* pathfinding. Static zones (from config) are unchanged.
        """
        old_count = len(self._dynamic_avoidance)
        self._dynamic_avoidance = zones
        if len(zones) != old_count and zones:
            log.debug("[NAV] Avoidance zones updated: %d threats active", len(zones))

    def avoidance_cost(self, game_x: float, game_y: float) -> float:
        """Return extra cost for being near an avoidance zone. 0 if clear.

        Uses exponential falloff: high cost near center, smooth taper toward
        edge. This prevents the hard cost wall that can trap A* in local
        minima with the previous linear formula.
        """
        best = 0.0
        # sigma = radius/3 gives ~95% decay at the edge (e^-3 ~= 0.05)
        for ax, ay, ar in self._avoidance_zones:
            d = math.hypot(game_x - ax, game_y - ay)
            if d < ar:
                sigma = ar / 3.0
                cost = 1000.0 * math.exp(-d / sigma)
                if cost > best:
                    best = cost
        for ax, ay, ar in self._dynamic_avoidance:
            d = math.hypot(game_x - ax, game_y - ay)
            if d < ar:
                sigma = ar / 3.0
                cost = 1000.0 * math.exp(-d / sigma)
                if cost > best:
                    best = cost
        return best

    def _in_override(self, game_x: float, game_y: float) -> bool:
        for x0, y0, x1, y1 in self._walkable_overrides:
            if x0 <= game_x <= x1 and y0 <= game_y <= y1:
                return True
        return False

    # Z height threshold for detecting unmarked water.
    # BSP water flags miss some river sections. Any cell with Z below this
    # threshold is treated as water even without the BSP flag.
    # No Z-threshold water detection  -  BSP flags + water_overrides are the source of truth.
    # EQ has caves, tunnels, and multi-level terrain where low Z is valid ground.

    def set_water_z_threshold(self, z: float) -> None:
        """Deprecated  -  no-op. Water detection uses BSP flags + overrides only."""
        pass

    # ------------------------------------------------------------------
    # Walkability bitfield (fast JPS scanning)
    # ------------------------------------------------------------------

    def _build_walk_bits(self) -> None:
        """Build packed walkability bitfield from grid flags.

        One bit per cell (row-major, LSB = lowest column in byte).
        Used by JPS for O(1) walkability lookups instead of the full
        is_walkable() chain (coord conversion + overrides + feature flags).

        Called after build(), apply_obstacles(), and load_cache().
        """
        if not self._flags:
            self._walk_bits = bytearray()
            self._walk_byte_cols = 0
            return

        cols = self._cols
        rows = self._rows
        byte_cols = (cols + 7) >> 3
        wb = bytearray(rows * byte_cols)
        blocked = SURFACE_WATER | SURFACE_LAVA | SURFACE_CLIFF | SURFACE_OBSTACLE

        for row in range(rows):
            rb = row * byte_cols
            rf = row * cols
            for col in range(cols):
                f = self._flags[rf + col]
                # Bridge cells are always walkable (override water)
                if f & SURFACE_BRIDGE:
                    wb[rb + (col >> 3)] |= 1 << (col & 7)
                elif (f & SURFACE_WALKABLE) and not (f & blocked):
                    wb[rb + (col >> 3)] |= 1 << (col & 7)

        self._walk_bits = wb
        self._walk_byte_cols = byte_cols

        walkable = sum(bin(b).count("1") for b in wb)
        log.debug("[NAV] Walk bitfield built: %d walkable bits, %d bytes", walkable, len(wb))

    def build_walk_bits_z(self, near_z: float) -> tuple[bytearray, int]:
        """Build a Z-filtered walkability bitfield for pathfinding.

        Like ``_build_walk_bits`` but rejects cells whose surface Z is
        too far from *near_z* (the agent's current level).  This prevents
        paths from routing across bridges or floors the agent is not on.

        Returns ``(bitfield, byte_cols)`` without mutating the cached
        ``_walk_bits`` (which is Z-agnostic and used elsewhere).
        """
        if not self._flags:
            return bytearray(), 0

        cols = self._cols
        rows = self._rows
        byte_cols = (cols + 7) >> 3
        wb = bytearray(rows * byte_cols)
        blocked = SURFACE_WATER | SURFACE_LAVA | SURFACE_CLIFF | SURFACE_OBSTACLE
        has_ceil = bool(self._z_ceiling)
        _z = self._z
        _zc = self._z_ceiling
        climb = _WALKABLE_CLIMB

        for row in range(rows):
            rb = row * byte_cols
            rf = row * cols
            for col in range(cols):
                idx = rf + col
                f = self._flags[idx]

                if has_ceil:
                    z_ceil = _zc[idx]
                else:
                    z_ceil = _NAN

                is_multi = z_ceil == z_ceil  # not NaN → multi-level cell

                if is_multi:
                    # Multi-level cell (bridge, multi-floor): decide
                    # walkability based on which surface the agent is on.
                    z_ground = _z[idx]
                    mid = (z_ground + z_ceil) / 2.0
                    on_upper = near_z > mid
                    if on_upper:
                        # Agent on the upper surface (bridge deck) — walkable
                        # only if BRIDGE flag is set.
                        walkable = bool(f & SURFACE_BRIDGE)
                    else:
                        # Agent at ground level — walkable only if the
                        # ground surface itself is walkable (not water/lava
                        # under a bridge).
                        walkable = bool((f & SURFACE_WALKABLE) and not (f & blocked))
                        # Bridge-only cells are not ground-walkable
                        if (f & SURFACE_BRIDGE) and (f & (SURFACE_WATER | SURFACE_LAVA)):
                            walkable = False
                    level_z = z_ceil if on_upper else z_ground
                    if abs(level_z - near_z) > climb:
                        walkable = False
                else:
                    # Single-level cell: standard walkability check
                    if f & SURFACE_BRIDGE:
                        walkable = True
                    elif (f & SURFACE_WALKABLE) and not (f & blocked):
                        walkable = True
                    else:
                        walkable = False

                if walkable:
                    wb[rb + (col >> 3)] |= 1 << (col & 7)

        return wb, byte_cols

    def invalidate_walk_bits(self) -> None:
        """Force rebuild of the walkability bitfield.

        Call after toggling feature flags (e.g. obstacle_avoidance) that
        change which cells are considered walkable. The next find_path()
        call will rebuild automatically if the bitfield is empty.
        """
        self._walk_bits = bytearray()
        self._walk_byte_cols = 0

    def _patch_walk_bits_rect(
        self,
        gx_min: float,
        gy_min: float,
        gx_max: float,
        gy_max: float,
        walkable: bool,
    ) -> None:
        """Patch walk_bits for a game-coordinate rectangle.

        Called when walkable/water overrides are added so the bitfield
        stays in sync without a full rebuild.
        """
        if not self._walk_bits:
            return
        c1, r1 = self._game_to_grid(gx_min, gy_min)
        c2, r2 = self._game_to_grid(gx_max, gy_max)
        c_lo = max(0, min(c1, c2))
        c_hi = min(self._cols - 1, max(c1, c2))
        r_lo = max(0, min(r1, r2))
        r_hi = min(self._rows - 1, max(r1, r2))
        byte_cols = self._walk_byte_cols

        for row in range(r_lo, r_hi + 1):
            ro = row * byte_cols
            for col in range(c_lo, c_hi + 1):
                bi = ro + (col >> 3)
                bit = 1 << (col & 7)
                if walkable:
                    self._walk_bits[bi] |= bit
                else:
                    self._walk_bits[bi] &= ~bit

    def _in_water_override(self, game_x: float, game_y: float) -> bool:
        """Check if position is in a forced-water region."""
        for x0, y0, x1, y1 in self._water_overrides:
            if x0 <= game_x <= x1 and y0 <= game_y <= y1:
                return True
        return False

    def is_walkable(self, game_x: float, game_y: float) -> bool:
        """True if the terrain at this position is safe to walk on."""
        # Walkable overrides (bridges) take priority over water overrides
        if self._in_override(game_x, game_y):
            return True
        # Water overrides (river banks) block walking
        if self._in_water_override(game_x, game_y):
            return False
        f = self.get_flags(game_x, game_y)
        # Auto-detected bridges are always walkable
        if f & SURFACE_BRIDGE:
            return True
        # Check non-obstacle hazards unconditionally
        if f & (SURFACE_WATER | SURFACE_LAVA | SURFACE_CLIFF):
            return False
        # Obstacle check gated by feature flag (allows runtime bypass)
        if f & SURFACE_OBSTACLE:
            from core.features import flags

            if flags.obstacle_avoidance:
                return False
        if not (f & SURFACE_WALKABLE):
            return False
        return True

    def is_water(self, game_x: float, game_y: float) -> bool:
        # Walkable override = not water (bridge)
        if self._in_override(game_x, game_y):
            return False
        # Water override = forced water
        if self._in_water_override(game_x, game_y):
            return True
        f = self.get_flags(game_x, game_y)
        return bool(f & SURFACE_WATER)

    def is_lava(self, game_x: float, game_y: float) -> bool:
        f = self.get_flags(game_x, game_y)
        return bool(f & SURFACE_LAVA)

    def is_cliff(self, game_x: float, game_y: float) -> bool:
        f = self.get_flags(game_x, game_y)
        return bool(f & SURFACE_CLIFF)

    def is_hazard(self, game_x: float, game_y: float) -> bool:
        """True if water, lava, cliff, steep, or unknown terrain."""
        f = self.get_flags(game_x, game_y)
        if f == SURFACE_NONE:
            return True  # unknown = hazard
        return bool(f & (SURFACE_WATER | SURFACE_LAVA | SURFACE_CLIFF | SURFACE_STEEP | SURFACE_OBSTACLE))

    def is_obstacle(self, game_x: float, game_y: float) -> bool:
        """True if there is a static object (tree, rock, etc.) at this position."""
        f = self.get_flags(game_x, game_y)
        return bool(f & SURFACE_OBSTACLE)

    def is_zoneline(self, game_x: float, game_y: float) -> bool:
        """True if this position is in a BSP zoneline transition region."""
        f = self.get_flags(game_x, game_y)
        return bool(f & SURFACE_ZONELINE)

    def get_zoneline_centers(self) -> list[Point]:
        """Return game-coordinate centers of all zoneline cell clusters.

        Groups adjacent SURFACE_ZONELINE cells into clusters via flood
        fill, returns the centroid of each cluster. Useful for
        auto-discovering zone transition points.
        """
        if not self._flags:
            return []

        visited: set[int] = set()
        centers: list[Point] = []
        cols = self._cols
        rows = self._rows

        for i in range(len(self._flags)):
            if i in visited or not (self._flags[i] & SURFACE_ZONELINE):
                continue
            # Flood fill this cluster
            cluster: list[int] = []
            stack = [i]
            while stack:
                idx = stack.pop()
                if idx in visited:
                    continue
                if not (self._flags[idx] & SURFACE_ZONELINE):
                    continue
                visited.add(idx)
                cluster.append(idx)
                r, c = divmod(idx, cols)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append(nr * cols + nc)
            # Compute centroid in game coords
            if cluster:
                sum_x = 0.0
                sum_y = 0.0
                for idx in cluster:
                    c = idx % cols
                    r = idx // cols
                    wx = self._min_x + (c + 0.5) * self.cell_size
                    wy = self._min_y + (r + 0.5) * self.cell_size
                    sum_x += wy  # WLD y -> game x
                    sum_y += wx  # WLD x -> game y
                cx = sum_x / len(cluster)
                cy = sum_y / len(cluster)
                centers.append(Point(cx, cy, self.get_z(cx, cy)))

        if centers:
            log.debug("[NAV] Found %d zoneline clusters", len(centers))
        return centers

    def get_z_ceiling(self, game_x: float, game_y: float) -> float:
        """Upper surface Z at position (bridge deck, upper floor).

        Returns NaN for single-level cells (no overhead geometry).
        Only set for cells where two upward surfaces are separated by > 3u.
        """
        col, row = self._game_to_grid(game_x, game_y)
        idx = self._grid_idx(col, row)
        if idx < 0 or not self._z_ceiling:
            return float("nan")
        return self._z_ceiling[idx]

    def is_multi_level(self, game_x: float, game_y: float) -> bool:
        """True if this cell has two distinct vertical levels (bridge, etc.)."""
        return not math.isnan(self.get_z_ceiling(game_x, game_y))

    def get_level_z(self, game_x: float, game_y: float, agent_z: float) -> float:
        """Return the Z of the level the agent is on.

        For multi-level cells (bridges, multi-floor), returns the surface
        closest to agent_z. For single-level cells, returns the ground Z.
        """
        z_ground = self.get_z(game_x, game_y)
        z_ceil = self.get_z_ceiling(game_x, game_y)
        if math.isnan(z_ceil):
            return z_ground
        # Agent above midpoint -> on upper level (bridge deck)
        midpoint = (z_ground + z_ceil) / 2.0
        if agent_z > midpoint:
            return z_ceil
        return z_ground

    def check_path(
        self, x1: float, y1: float, x2: float, y2: float, step: float = 5.0
    ) -> tuple[float, float] | None:
        """Walk along path and return first hazard point, or None if clear."""
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 0.01:
            return None

        steps = max(1, int(dist / step))
        for i in range(steps + 1):
            t = i / steps
            px, py = x1 + dx * t, y1 + dy * t
            if self.is_hazard(px, py):
                return (px, py)
        return None

    def check_los(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
        margin: float = 3.0,
    ) -> bool:
        """Check line-of-sight between two 3D points using DDA grid traversal.

        Amanatides-Woo algorithm: walks every grid cell the ray crosses,
        exactly once. At each cell, compares the ray's interpolated Z against
        terrain Z. Faster and more precise than parametric stepping (no
        sampling gaps, no wasted samples).

        Args:
            x1,y1,z1: source position (player, game coords)
            x2,y2,z2: target position (npc, game coords)
            margin: terrain must be this much above the ray to block

        Returns: True if LOS is clear, False if blocked.
        """
        cs = self.cell_size
        # game -> WLD: swap x/y
        fx1 = (y1 - self._min_x) / cs
        fy1 = (x1 - self._min_y) / cs
        fx2 = (y2 - self._min_x) / cs
        fy2 = (x2 - self._min_y) / cs

        dx = fx2 - fx1
        dy = fy2 - fy1
        dz = z2 - z1

        col = int(fx1)
        row = int(fy1)

        step_col = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_row = 1 if dy > 0 else (-1 if dy < 0 else 0)

        abs_dx = abs(dx)
        abs_dy = abs(dy)
        t_delta_col = (1.0 / abs_dx) if abs_dx > 1e-10 else 1e30
        t_delta_row = (1.0 / abs_dy) if abs_dy > 1e-10 else 1e30

        if abs_dx > 1e-10:
            if dx > 0:
                t_max_col = (math.floor(fx1) + 1.0 - fx1) / abs_dx
            else:
                frac = fx1 - math.floor(fx1)
                t_max_col = (frac / abs_dx) if frac > 1e-10 else t_delta_col
        else:
            t_max_col = 1e30

        if abs_dy > 1e-10:
            if dy > 0:
                t_max_row = (math.floor(fy1) + 1.0 - fy1) / abs_dy
            else:
                frac = fy1 - math.floor(fy1)
                t_max_row = (frac / abs_dy) if frac > 1e-10 else t_delta_row
        else:
            t_max_row = 1e30

        cols = self._cols
        rows = self._rows
        _z = self._z
        _flags = self._flags

        from core.features import flags as _flags_mod

        check_obstacles = _flags_mod.obstacle_avoidance

        while True:
            if t_max_col < t_max_row:
                t = t_max_col
                if t > 1.0:
                    break
                col += step_col
                t_max_col += t_delta_col
            else:
                t = t_max_row
                if t > 1.0:
                    break
                row += step_row
                t_max_row += t_delta_row

            if col < 0 or col >= cols or row < 0 or row >= rows:
                continue

            idx = row * cols + col

            if check_obstacles and (_flags[idx] & SURFACE_OBSTACLE):
                return False

            ray_z = z1 + dz * t
            terrain_z = _z[idx]
            if terrain_z != terrain_z:  # NaN guard
                continue
            if terrain_z > ray_z + margin:
                return False

        return True

    def find_path(
        self,
        start: Point,
        goal: Point,
        max_nodes: int = 50000,
        jitter: float = 12.0,
    ) -> list[Point] | None:
        """A* pathfind from start to goal through walkable terrain.

        The start point's z coordinate is used for multi-layer terrain
        selection (bridges, multi-floor buildings).

        Args:
            start: Start position as Point (x, y, z in game coordinates).
            goal: Goal position as Point (x, y, z in game coordinates).
            jitter: path humanization amount (0=straight, 12=natural wandering)
        Returns list of Point(x, y, z) waypoints, or None if no path.
        """
        from nav.pathfinding import find_path as _find_path

        return _find_path(self, start, goal, max_nodes, jitter=jitter)

    def check_cliff_ahead(
        self, game_x: float, game_y: float, heading: float, look_dist: float = 20.0, max_drop: float = 100.0
    ) -> bool:
        """Check if there's a cliff drop in the direction we're facing."""
        # Convert EQ heading to direction vector
        rad = heading * 2.0 * math.pi / 512.0
        dx = math.sin(rad)
        dy = math.cos(rad)

        z_here = self.get_z(game_x, game_y)
        if math.isnan(z_here):
            return False  # unknown, don't block

        for d in (5.0, 10.0, 15.0, look_dist):
            px, py = game_x + dx * d, game_y + dy * d
            z_there = self.get_z(px, py)
            if not math.isnan(z_there) and (z_here - z_there) > max_drop:
                return True
        return False

    def query_bsp(self, game_x: float, game_y: float, game_z: float) -> int:
        """Query BSP tree for region type at a 3D point.

        Returns RegionType constant (WATER, LAVA, NORMAL, etc.).
        Falls back to NORMAL if no BSP data.
        """
        if not self._bsp_nodes:
            return int(RegionType.NORMAL)
        # Convert game -> WLD coords
        wx, wy, wz = game_y, game_x, game_z
        return self._bsp_classify(wx, wy, wz)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (min_game_x, min_game_y, max_game_x, max_game_y)."""
        # WLD bounds -> game coords (swap back)
        max_game_x = self._min_y + self._rows * self.cell_size
        max_game_y = self._min_x + self._cols * self.cell_size
        return (self._min_y, self._min_x, max_game_x, max_game_y)

    @property
    def stats(self) -> dict:
        """Diagnostic statistics."""
        total = self._cols * self._rows
        walkable = sum(1 for f in self._flags if f & SURFACE_WALKABLE)
        water = sum(1 for f in self._flags if f & SURFACE_WATER)
        lava = sum(1 for f in self._flags if f & SURFACE_LAVA)
        cliff = sum(1 for f in self._flags if f & SURFACE_CLIFF)
        obstacle = sum(1 for f in self._flags if f & SURFACE_OBSTACLE)
        bridge = sum(1 for f in self._flags if f & SURFACE_BRIDGE)
        empty = sum(1 for f in self._flags if f == SURFACE_NONE)
        return {
            "grid": f"{self._cols}x{self._rows}",
            "cell_size": self.cell_size,
            "total_cells": total,
            "walkable": walkable,
            "water": water,
            "lava": lava,
            "cliff": cliff,
            "obstacle": obstacle,
            "bridge": bridge,
            "empty": empty,
            "bsp_nodes": len(self._bsp_nodes),
            "region_types": len(self._region_types),
        }

    # ------------------------------------------------------------------
    # Build from mesh data
    # ------------------------------------------------------------------

    def build(
        self,
        meshes: list[Mesh],
        bsp_nodes: list[BSPNode],
        region_types: dict[int, int],
        mesh_materials: dict[str, list[str]] | None = None,
        margin: float = 50.0,
        sky_z_threshold: float = 200.0,
    ) -> None:
        """Build heightmap from extracted zone mesh data.

        Args:
            mesh_materials: Optional mapping from mesh name to per-material-idx
                texture filenames (from WLDFile.extract_mesh_material_names).
                Enables material-based bridge detection.
            sky_z_threshold: Triangles with ALL vertices above this Z
                are treated as sky dome / visibility geometry and skipped.
        """
        if not meshes:
            log.warning("[NAV] No meshes provided  -  terrain will be empty")
            return

        # Compute WLD bounds from all vertices
        all_x = [v.x for m in meshes for v in m.vertices]
        all_y = [v.y for m in meshes for v in m.vertices]

        self._min_x = min(all_x) - margin
        self._min_y = min(all_y) - margin
        max_x = max(all_x) + margin
        max_y = max(all_y) + margin

        self._cols = int(math.ceil((max_x - self._min_x) / self.cell_size))
        self._rows = int(math.ceil((max_y - self._min_y) / self.cell_size))

        total = self._cols * self._rows
        self._z = [float("nan")] * total
        self._z_ceiling = [float("nan")] * total
        self._flags = [SURFACE_NONE] * total
        self._normal_z = [0.0] * total
        self._material_id = [MAT_UNKNOWN] * total
        self._region_id = [0] * total

        # Temporary: track second-highest Z for multi-level detection
        z_lower: list[float] = [float("nan")] * total

        self._bsp_nodes = bsp_nodes
        self._region_types = region_types

        log.info(
            "[NAV] Building heightmap: %dx%d grid (%.0f unit cells), "
            "%d meshes, %d BSP nodes, %d region types",
            self._cols,
            self._rows,
            self.cell_size,
            len(meshes),
            len(bsp_nodes),
            len(region_types),
        )

        # Rasterize mesh triangles onto the grid
        tri_count = 0
        sky_skipped = 0
        mat_resolved = 0
        for mesh in meshes:
            verts = mesh.vertices
            mat_names = mesh_materials.get(mesh.name.lower(), []) if mesh_materials else []
            for tri in mesh.triangles:
                if tri.v1 >= len(verts) or tri.v2 >= len(verts) or tri.v3 >= len(verts):
                    continue
                v1, v2, v3 = verts[tri.v1], verts[tri.v2], verts[tri.v3]
                if v1.z > sky_z_threshold and v2.z > sky_z_threshold and v3.z > sky_z_threshold:
                    sky_skipped += 1
                    continue
                # Resolve material from texture name
                mat_id = MAT_UNKNOWN
                if tri.material_idx < len(mat_names):
                    mat_id = classify_material(mat_names[tri.material_idx])
                    if mat_id != MAT_UNKNOWN:
                        mat_resolved += 1
                self._rasterize_triangle(v1, v2, v3, mat_id, z_lower)
                tri_count += 1

        log.info(
            "[NAV] Rasterized %d triangles (%d sky dome skipped, %d material-classified)",
            tri_count,
            sky_skipped,
            mat_resolved,
        )

        # Post-rasterization: detect multi-level cells.
        # Gap must be 8-50u to be a real level separation (bridge, floor).
        # < 8u = hilly terrain noise. > 50u = underground geometry.
        multi_level = 0
        for i in range(total):
            z_hi = self._z[i]
            z_lo = z_lower[i]
            if not math.isnan(z_hi) and not math.isnan(z_lo):
                gap = z_hi - z_lo
                if 8.0 < gap < 50.0:
                    self._z_ceiling[i] = z_hi
                    multi_level += 1
        if multi_level:
            log.info("[NAV] Multi-level cells detected: %d", multi_level)

        # Apply BSP region types (water/lava/zoneline)
        self._apply_bsp_regions()

        # Detect bridges: walkable material over BSP water region
        self._detect_bridges()

        # Detect cliffs (large Z drops between adjacent cells)
        self._detect_cliffs()

        # Build walkability bitfield for fast JPS scanning
        self._build_walk_bits()

        stats = self.stats
        log.info(
            "[NAV] Terrain built: %d walkable, %d water, %d lava, %d cliff, %d bridge, %d empty cells",
            stats["walkable"],
            stats["water"],
            stats["lava"],
            stats["cliff"],
            stats.get("bridge", 0),
            stats["empty"],
        )

    # ------------------------------------------------------------------
    # Obstacle helpers
    # ------------------------------------------------------------------

    def _resolve_obstacle_cells(
        self,
        p: ObjectPlacement,
        model_key: str,
        object_meshes: dict[str, Mesh] | None,
        mesh_radii: dict[str, float] | None,
        compute_mesh_footprint_cells_fn: Callable[..., list | None],
        compute_obstacle_cells_fn: Callable[..., list],
        get_model_radius_fn: Callable[[str], float | None],
    ) -> tuple[list | None, str]:
        """Resolve obstacle cells for a placement using tiered resolution.

        Returns (cells, tier) where tier is "mesh" or "circular",
        or (None, "") if no cells could be computed.
        """
        # Tier 3: mesh-accurate triangle footprint
        mesh = object_meshes.get(model_key) if object_meshes else None
        if mesh is not None:
            cells = compute_mesh_footprint_cells_fn(
                mesh,
                p,
                self._min_x,
                self._min_y,
                self._cols,
                self._rows,
                self.cell_size,
            )
            if cells:
                return cells, "mesh"

        # Tier 2/1: circular radius fallback
        radius: float | None = None
        if mesh_radii:
            radius = mesh_radii.get(model_key)
        if radius is None:
            radius = get_model_radius_fn(p.model_name)
        if radius is None or radius <= 0.0:
            return None, ""
        cells = compute_obstacle_cells_fn(
            p,
            self._min_x,
            self._min_y,
            self._cols,
            self._rows,
            self.cell_size,
            mesh_radius=radius,
        )
        return cells, "circular"

    def _flag_obstacle_cells(
        self,
        cells: list,
        p: ObjectPlacement,
        z_tolerance: float,
    ) -> tuple[int, int, int]:
        """Flag walkable cells as obstacles, applying Z-filter.

        Returns (flagged, skipped_z, skipped_nodata) counts.
        """
        flagged = 0
        skipped_z = 0
        skipped_nodata = 0
        for col, row in cells:
            idx = row * self._cols + col
            if idx < 0 or idx >= len(self._flags):
                continue
            terrain_z = self._z[idx]
            if not math.isnan(terrain_z):
                if abs(p.z - terrain_z) > z_tolerance:
                    skipped_z += 1
                    continue
            else:
                skipped_nodata += 1
                continue
            f = self._flags[idx]
            if not (f & SURFACE_WALKABLE):
                continue
            if f & (SURFACE_WATER | SURFACE_LAVA | SURFACE_CLIFF):
                continue
            self._flags[idx] |= SURFACE_OBSTACLE
            flagged += 1
        return flagged, skipped_z, skipped_nodata

    # ------------------------------------------------------------------
    # Obstacle application (placed objects: trees, rocks, buildings)
    # ------------------------------------------------------------------

    def apply_obstacles(
        self,
        placements: list[ObjectPlacement],
        mesh_radii: dict[str, float] | None = None,
        object_meshes: dict[str, Mesh] | None = None,
        buffer_cells: int = 1,
        z_tolerance: float = 50.0,
    ) -> None:
        """Flag grid cells blocked by placed objects (trees, rocks, etc.).

        Called after build() during terrain cache generation. Marks cells
        as SURFACE_OBSTACLE so A* pathfinding routes around them.

        Uses three tiers of obstacle resolution:
        - Tier 3: Mesh-accurate triangle rasterization (from object_meshes)
        - Tier 2: Circular radius from mesh AABB (from mesh_radii)
        - Tier 1: Name-based radius lookup (fallback)

        Args:
            placements: Object placement list from WLDFile.extract_placements().
                        Coordinates are in WLD space (matching internal grid).
            mesh_radii: Optional dict mapping lowercased model name to
                        footprint radius from mesh AABB (Tier 2). Falls back
                        to name-based radius lookup when not available.
            object_meshes: Optional dict mapping lowercased model name to
                           Mesh object for exact triangle rasterization (Tier 3).
            buffer_cells: Number of cells to expand around each obstacle
                          (1 cell = 1u clearance at default cell_size).
            z_tolerance: Max Z difference between object and terrain for
                         flagging. Prevents cross-level blocking in dungeons.
        """
        from eq.placeables import (
            compute_mesh_footprint_cells,
            compute_obstacle_cells,
            get_model_radius,
        )

        if not self._flags:
            log.warning("[NAV] apply_obstacles called on empty terrain")
            return

        flagged = 0
        skipped_walkable = 0
        skipped_small = 0
        skipped_z = 0
        skipped_nodata = 0
        mesh_used = 0
        circular_used = 0

        # At coarse resolution, small objects (< half cell size) don't
        # meaningfully block a cell. Skip them to prevent over-blocking.
        min_radius = self.cell_size * 0.4

        for p in placements:
            model_key = p.model_name.lower()

            # Check if this is a walkable/passthrough object by name
            name_radius = get_model_radius(p.model_name)
            if name_radius <= 0.0:
                skipped_walkable += 1
                continue

            # Skip small objects at coarse resolution
            effective_r = name_radius * p.scale
            if mesh_radii:
                mr = mesh_radii.get(model_key)
                if mr is not None:
                    effective_r = mr * p.scale
            if effective_r < min_radius:
                skipped_small += 1
                continue

            cells, tier = self._resolve_obstacle_cells(
                p,
                model_key,
                object_meshes,
                mesh_radii,
                compute_mesh_footprint_cells,
                compute_obstacle_cells,
                get_model_radius,
            )
            if cells is None:
                skipped_walkable += 1
                continue
            if tier == "mesh":
                mesh_used += 1
            else:
                circular_used += 1

            f_count, z_count, nd_count = self._flag_obstacle_cells(cells, p, z_tolerance)
            flagged += f_count
            skipped_z += z_count
            skipped_nodata += nd_count

        buffer_count = self._expand_obstacle_buffer(buffer_cells)
        total_obstacle = flagged + buffer_count

        # Density safety check: strip buffer if obstacles block >40% of walkable cells
        block_pct = self._obstacle_block_pct()
        if block_pct > 40.0 and buffer_count > 0:
            self._rollback_obstacle_buffer(placements, mesh_radii, z_tolerance)
            total_obstacle = sum(1 for f in self._flags if f & SURFACE_OBSTACLE)
            buffer_count = 0
            block_pct = self._obstacle_block_pct()

        log.info(
            "[NAV] Obstacles applied: %d cells flagged + %d buffer = %d total "
            "(%d placements, %d mesh-accurate, %d circular, "
            "%d walkable-skip, %d small-skip, %d z-skip, %d nodata-skip, "
            "%.0f%% walkable blocked)",
            flagged,
            buffer_count,
            total_obstacle,
            len(placements),
            mesh_used,
            circular_used,
            skipped_walkable,
            skipped_small,
            skipped_z,
            skipped_nodata,
            block_pct,
        )

        # Rebuild walkability bitfield with obstacle data baked in
        self._build_walk_bits()

    # ------------------------------------------------------------------
    # Triangle rasterization
    # ------------------------------------------------------------------

    def _rasterize_triangle(
        self,
        v1: MeshVertex,
        v2: MeshVertex,
        v3: MeshVertex,
        material_id: int = 0,
        z_lower: list[float] | None = None,
    ) -> None:
        """Rasterize one triangle onto the heightmap grid."""
        # Compute triangle normal for slope detection
        e1x, e1y, e1z = v2.x - v1.x, v2.y - v1.y, v2.z - v1.z
        e2x, e2y, e2z = v3.x - v1.x, v3.y - v1.y, v3.z - v1.z
        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x
        n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
        if n_len < 1e-10:
            return  # degenerate triangle
        nz /= n_len

        # EQ uses clockwise winding -> our cross product gives inverted normals.
        # Flip so +nz = upward-facing (ground), -nz = downward (ceiling).
        nz = -nz

        # Skip downward-facing triangles (ceilings)
        if nz < 0.0:
            return

        is_steep = abs(nz) < 0.09  # > ~85 degree slope (nearly vertical)

        # 2D bounding box in grid cells
        xs = [v1.x, v2.x, v3.x]
        ys = [v1.y, v2.y, v3.y]
        min_col = max(0, int((min(xs) - self._min_x) / self.cell_size))
        max_col = min(self._cols - 1, int((max(xs) - self._min_x) / self.cell_size))
        min_row = max(0, int((min(ys) - self._min_y) / self.cell_size))
        max_row = min(self._rows - 1, int((max(ys) - self._min_y) / self.cell_size))

        # Precompute barycentric denominator
        denom = (v2.y - v3.y) * (v1.x - v3.x) + (v3.x - v2.x) * (v1.y - v3.y)
        if abs(denom) < 1e-10:
            return  # degenerate

        inv_denom = 1.0 / denom

        for row in range(min_row, max_row + 1):
            py = self._min_y + (row + 0.5) * self.cell_size
            for col in range(min_col, max_col + 1):
                px = self._min_x + (col + 0.5) * self.cell_size

                # Barycentric coordinates
                w1 = ((v2.y - v3.y) * (px - v3.x) + (v3.x - v2.x) * (py - v3.y)) * inv_denom
                w2 = ((v3.y - v1.y) * (px - v3.x) + (v1.x - v3.x) * (py - v3.y)) * inv_denom
                w3 = 1.0 - w1 - w2

                # Point inside triangle?
                if w1 < -0.001 or w2 < -0.001 or w3 < -0.001:
                    continue

                # Interpolate Z
                z = w1 * v1.z + w2 * v2.z + w3 * v3.z

                idx = row * self._cols + col

                # Keep the HIGHEST upward-facing surface below the sky threshold.
                # Sky dome is already filtered in the caller. Of the remaining
                # triangles, the highest Z is the ground surface (lower Z values
                # are underground geometry like water bottoms and caves).
                current_z = self._z[idx]
                if math.isnan(current_z) or z > current_z:
                    # Track displaced Z for multi-level detection.
                    # z_lower keeps the LOWEST valid surface so the full
                    # gap (highest - lowest) is measured correctly even
                    # when triangles arrive out of Z-order.
                    if z_lower is not None and not math.isnan(current_z):
                        cur_lo = z_lower[idx]
                        if math.isnan(cur_lo) or current_z < cur_lo:
                            z_lower[idx] = current_z
                    self._z[idx] = z
                    self._normal_z[idx] = nz
                    self._material_id[idx] = material_id
                    if material_id == MAT_WATER:
                        self._flags[idx] = SURFACE_WATER
                    elif is_steep:
                        self._flags[idx] = SURFACE_STEEP
                    else:
                        self._flags[idx] = SURFACE_WALKABLE
                elif z_lower is not None:
                    # Lower surface: keep the minimum
                    cur_lo = z_lower[idx]
                    if math.isnan(cur_lo) or z < cur_lo:
                        z_lower[idx] = z

    # ------------------------------------------------------------------
    # BSP region application
    # ------------------------------------------------------------------

    def _apply_bsp_regions(self) -> None:
        """Apply water/lava/zoneline flags from BSP tree."""
        if not self._bsp_nodes or not self._region_types:
            return

        applied = 0
        for row in range(self._rows):
            wy = self._min_y + (row + 0.5) * self.cell_size
            for col in range(self._cols):
                wx = self._min_x + (col + 0.5) * self.cell_size
                idx = row * self._cols + col
                z = self._z[idx]
                if math.isnan(z):
                    z = 0.0

                rtype = self._bsp_classify(wx, wy, z)
                if rtype in (RegionType.WATER, RegionType.WATER_BLOCK_LOS, RegionType.FREEZING_WATER):
                    self._flags[idx] |= SURFACE_WATER
                    self._region_id[idx] = rtype
                    applied += 1
                elif rtype == RegionType.LAVA:
                    self._flags[idx] |= SURFACE_LAVA
                    self._region_id[idx] = rtype
                    applied += 1
                elif rtype == RegionType.ZONELINE:
                    self._flags[idx] |= SURFACE_ZONELINE
                    self._region_id[idx] = rtype
                    applied += 1
                elif rtype != RegionType.NORMAL:
                    self._region_id[idx] = rtype

        log.info("[NAV] BSP regions applied: %d cells flagged", applied)

        # Filter false-positive water from BSP over-coverage
        self._filter_bsp_water()

    def _strip_water_by_z(self) -> int:
        """Strip water cells above the estimated surface Z. Returns count stripped."""
        water_zs: list[float] = []
        for i in range(len(self._flags)):
            if (self._flags[i] & SURFACE_WATER) and not math.isnan(self._z[i]):
                water_zs.append(self._z[i])

        if not water_zs:
            return 0

        water_zs.sort()
        surface_z = water_zs[int(len(water_zs) * 0.75)]
        cutoff = surface_z + 8.0

        stripped = 0
        for i in range(len(self._flags)):
            if not (self._flags[i] & SURFACE_WATER):
                continue
            if not math.isnan(self._z[i]) and self._z[i] > cutoff:
                self._flags[i] &= ~SURFACE_WATER
                stripped += 1
        return stripped

    def _strip_small_water_bodies(self, min_body_cells: int) -> tuple[int, int]:
        """Flood-fill water cells and strip bodies smaller than min_body_cells.

        Returns (cells_stripped, num_small_bodies).
        """
        cols = self._cols
        rows = self._rows
        visited: set[int] = set()
        small_bodies: list[list[int]] = []

        for i in range(len(self._flags)):
            if i in visited or not (self._flags[i] & SURFACE_WATER):
                continue
            body: list[int] = []
            stack = [i]
            while stack:
                idx = stack.pop()
                if idx in visited:
                    continue
                if not (self._flags[idx] & SURFACE_WATER):
                    continue
                visited.add(idx)
                body.append(idx)
                r = idx // cols
                c = idx % cols
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        ni = nr * cols + nc
                        if ni not in visited:
                            stack.append(ni)
            if len(body) < min_body_cells:
                small_bodies.append(body)

        conn_stripped = 0
        for body in small_bodies:
            for idx in body:
                self._flags[idx] &= ~SURFACE_WATER
                conn_stripped += 1
        return conn_stripped, len(small_bodies)

    def _filter_bsp_water(self, min_body_cells: int = 5000) -> None:
        """Remove false water flags from BSP over-coverage.

        BSP water regions are coarse 3D volumes that extend well beyond
        actual water surfaces -- onto hillsides (vertical over-coverage)
        and across dry land (horizontal over-coverage). Two filters:

        1. Z filter: estimate water surface Z from cell distribution,
           strip cells clearly above it (catches hillside false positives).
        2. Connectivity filter: flood-fill water cells into connected
           bodies, strip bodies smaller than min_body_cells (catches
           isolated BSP artifact patches far from actual water).
        """
        total_before = sum(1 for f in self._flags if f & SURFACE_WATER)
        if total_before < 100:
            return

        z_stripped = self._strip_water_by_z()
        conn_stripped, num_small = self._strip_small_water_bodies(min_body_cells)

        total_after = sum(1 for f in self._flags if f & SURFACE_WATER)
        total_stripped = z_stripped + conn_stripped
        if total_stripped:
            log.info(
                "[NAV] Water filter: %d Z-stripped + %d connectivity-stripped "
                "= %d removed (%d -> %d water cells, %d small bodies)",
                z_stripped,
                conn_stripped,
                total_stripped,
                total_before,
                total_after,
                num_small,
            )

    def _detect_bridges(self) -> None:
        """Detect bridges by spatial pattern: bridge material adjacent to water.

        A bridge is WOOD/STONE/METAL cells with WATER cells nearby.
        The bridge deck and water surface are at different Z levels
        (bridge ~Z=-11, river ~Z=-30) so they occupy ADJACENT grid
        cells, not the same cell. Detection checks a 2-cell radius
        around each bridge-material cell for water neighbors.

        Also clears SURFACE_OBSTACLE on bridge cells (placed object
        footprints may have flagged the bridge structure).
        """
        if not self._material_id:
            return

        cols = self._cols
        rows = self._rows
        search = 2  # search radius in cells

        bridge_count = 0
        for i in range(len(self._flags)):
            if self._material_id[i] not in _BRIDGE_MATERIALS:
                continue
            if self._normal_z[i] < 0.7:
                continue

            # Check for water in nearby cells
            r = i // cols
            c = i % cols
            water_nearby = False
            for dr in range(-search, search + 1):
                if water_nearby:
                    break
                for dc in range(-search, search + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if self._flags[nr * cols + nc] & SURFACE_WATER:
                            water_nearby = True
                            break

            if water_nearby:
                self._flags[i] |= SURFACE_BRIDGE | SURFACE_WALKABLE
                self._flags[i] &= ~(SURFACE_OBSTACLE | SURFACE_WATER)
                bridge_count += 1

        if bridge_count:
            log.info(
                "[NAV] Bridge detection: %d cells auto-detected (bridge material adjacent to water)",
                bridge_count,
            )

    def _bsp_classify(self, wx: float, wy: float, wz: float) -> int:
        """Traverse BSP tree to find region type at WLD point."""
        normal: int = int(RegionType.NORMAL)
        nodes = self._bsp_nodes
        if not nodes:
            return normal

        # Start at root (index 0, which is 1-based index 1)
        node_idx = 0
        max_depth = 64  # prevent infinite loops

        for _ in range(max_depth):
            if node_idx < 0 or node_idx >= len(nodes):
                return normal

            node = nodes[node_idx]

            # Leaf node: has region reference
            if node.front == 0 and node.back == 0:
                if node.region > 0:
                    return self._region_types.get(node.region, normal)
                return normal

            # Split plane test
            dist = wx * node.normal_x + wy * node.normal_y + wz * node.normal_z + node.split_distance

            if dist >= 0:
                if node.front == 0:
                    return normal
                node_idx = node.front - 1  # convert 1-based to 0-based
            else:
                if node.back == 0:
                    return normal
                node_idx = node.back - 1

        return normal

    # ------------------------------------------------------------------
    # Obstacle buffer / density helpers
    # ------------------------------------------------------------------

    def _expand_obstacle_buffer(self, buffer_cells: int) -> int:
        """Expand SURFACE_OBSTACLE to adjacent walkable cells."""
        if buffer_cells <= 0:
            return 0
        count = 0
        obstacle_indices = {i for i in range(len(self._flags)) if self._flags[i] & SURFACE_OBSTACLE}
        for _ in range(buffer_cells):
            new_indices: set[int] = set()
            for idx in obstacle_indices:
                r = idx // self._cols
                c = idx % self._cols
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self._rows and 0 <= nc < self._cols:
                        nidx = nr * self._cols + nc
                        f = self._flags[nidx]
                        if (
                            f & SURFACE_WALKABLE
                            and not (f & SURFACE_OBSTACLE)
                            and not (f & (SURFACE_WATER | SURFACE_LAVA | SURFACE_CLIFF))
                        ):
                            self._flags[nidx] |= SURFACE_OBSTACLE
                            count += 1
                            new_indices.add(nidx)
            obstacle_indices = new_indices
        return count

    def _obstacle_block_pct(self) -> float:
        """Percentage of walkable cells blocked by obstacles."""
        walkable = sum(1 for f in self._flags if f & SURFACE_WALKABLE)
        if walkable == 0:
            return 0.0
        free = sum(
            1
            for f in self._flags
            if (f & SURFACE_WALKABLE)
            and not (f & (SURFACE_WATER | SURFACE_LAVA | SURFACE_CLIFF | SURFACE_OBSTACLE))
        )
        return (1.0 - free / walkable) * 100.0

    def _rollback_obstacle_buffer(
        self,
        placements: list,
        mesh_radii: dict[str, float] | None,
        z_tolerance: float,
    ) -> None:
        """Remove all obstacle flags and re-flag core obstacles only (no buffer)."""
        from eq.placeables import compute_obstacle_cells, get_model_radius

        log.warning("[NAV] Obstacles block >40%% of walkable cells -- removing buffer to reduce blockage")
        for i in range(len(self._flags)):
            if self._flags[i] & SURFACE_OBSTACLE:
                self._flags[i] &= ~SURFACE_OBSTACLE
        for p in placements:
            obj_r: float | None = None
            if mesh_radii:
                obj_r = mesh_radii.get(p.model_name.lower())
            if obj_r is None:
                obj_r = get_model_radius(p.model_name)
            if obj_r <= 0.0:
                continue
            cells = compute_obstacle_cells(
                p,
                self._min_x,
                self._min_y,
                self._cols,
                self._rows,
                self.cell_size,
                mesh_radius=obj_r,
            )
            for col, row in cells:
                idx = row * self._cols + col
                if idx < 0 or idx >= len(self._flags):
                    continue
                terrain_z = self._z[idx]
                if math.isnan(terrain_z):
                    continue
                if abs(p.z - terrain_z) > z_tolerance:
                    continue
                f = self._flags[idx]
                if f & SURFACE_WALKABLE and not (f & (SURFACE_WATER | SURFACE_LAVA | SURFACE_CLIFF)):
                    self._flags[idx] |= SURFACE_OBSTACLE

    # ------------------------------------------------------------------
    # Cliff detection
    # ------------------------------------------------------------------

    def _detect_cliffs(
        self,
        gradient_threshold: float = 15.0,
        sustained_slope_window: int = 5,
        sustained_slope_total: float = 40.0,
    ) -> None:
        """Mark cliff and steep-slope cells as unwalkable.

        Two detection methods:
        1. Gradient: any cell with abs(z-neighbor) > gradient_threshold
        2. Sustained slope: cells where z rises > sustained_slope_total
           over sustained_slope_window consecutive cells in any cardinal
           direction. Catches zone walls (smooth 45-degree ramps in mesh).

        Both sides of detected cliffs are flagged (approach + cliff face).
        """
        cliff_count = self._detect_gradient_cliffs(gradient_threshold)
        slope_count = self._detect_sustained_slopes(sustained_slope_window, sustained_slope_total)
        buffer_count = self._expand_cliff_buffer()

        log.info(
            "[NAV] Cliff detection: %d gradient + %d slope + %d buffer = %d total",
            cliff_count,
            slope_count,
            buffer_count,
            cliff_count + slope_count + buffer_count,
        )

    def _detect_gradient_cliffs(self, gradient_threshold: float) -> int:
        """Phase 1: Flag cells with sharp height drops to neighbors."""
        count = 0
        for row in range(self._rows):
            for col in range(self._cols):
                idx = row * self._cols + col
                z = self._z[idx]
                if math.isnan(z):
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self._rows and 0 <= nc < self._cols:
                        nidx = nr * self._cols + nc
                        nz = self._z[nidx]
                        if not math.isnan(nz) and abs(z - nz) > gradient_threshold:
                            self._flags[idx] |= SURFACE_CLIFF
                            count += 1
                            break
        return count

    def _detect_sustained_slopes(self, window: int, total_threshold: float) -> int:
        """Phase 2: Flag cells on smooth ramps (zone walls) where Z rises
        more than total_threshold over window consecutive cells."""
        count = 0
        # Scan horizontal runs
        for row in range(self._rows):
            for col in range(self._cols - window):
                idx_start = row * self._cols + col
                idx_end = row * self._cols + col + window
                z_start = self._z[idx_start]
                z_end = self._z[idx_end]
                if math.isnan(z_start) or math.isnan(z_end):
                    continue
                if abs(z_end - z_start) > total_threshold:
                    for c in range(col, col + window + 1):
                        sidx = row * self._cols + c
                        if not (self._flags[sidx] & SURFACE_CLIFF):
                            self._flags[sidx] |= SURFACE_CLIFF
                            count += 1
        # Scan vertical runs
        for col in range(self._cols):
            for row in range(self._rows - window):
                idx_start = row * self._cols + col
                idx_end = (row + window) * self._cols + col
                z_start = self._z[idx_start]
                z_end = self._z[idx_end]
                if math.isnan(z_start) or math.isnan(z_end):
                    continue
                if abs(z_end - z_start) > total_threshold:
                    for r in range(row, row + window + 1):
                        sidx = r * self._cols + col
                        if not (self._flags[sidx] & SURFACE_CLIFF):
                            self._flags[sidx] |= SURFACE_CLIFF
                            count += 1
        return count

    def _expand_cliff_buffer(self) -> int:
        """Phase 3: Expand cliff flags to adjacent cells so the agent
        stays ~5u away from edges."""
        cliff_buf = max(1, round(5.0 / self.cell_size))
        count = 0
        cliff_indices = {i for i in range(len(self._flags)) if self._flags[i] & SURFACE_CLIFF}
        for _ in range(cliff_buf):
            new_cliff: set[int] = set()
            for idx in cliff_indices:
                row = idx // self._cols
                col = idx % self._cols
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self._rows and 0 <= nc < self._cols:
                        nidx = nr * self._cols + nc
                        if not (self._flags[nidx] & SURFACE_CLIFF):
                            nz = self._z[nidx]
                            if not math.isnan(nz):
                                self._flags[nidx] |= SURFACE_CLIFF
                                count += 1
                                new_cliff.add(nidx)
            cliff_indices = new_cliff
        return count

    def redetect_cliffs(self, threshold: float) -> None:
        """Re-run cliff detection with a new threshold.

        Clears existing SURFACE_CLIFF flags and re-detects with the
        given threshold. Use after loading from cache to apply
        zone-specific cliff sensitivity (e.g., desert dunes vs canyons).
        """
        # Clear existing cliff flags
        for i in range(len(self._flags)):
            self._flags[i] &= ~SURFACE_CLIFF
        # Re-detect with new threshold
        self._detect_cliffs(threshold)
        stats = self.stats
        log.info("[NAV] Cliff redetection: %d cliff cells (threshold=%.1f)", stats["cliff"], threshold)

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _game_to_wld(self, game_x: float, game_y: float) -> tuple[float, float]:
        """Convert game coordinates to WLD coordinates."""
        return (game_y, game_x)  # WLD x = state.y, WLD y = state.x

    def _game_to_grid(self, game_x: float, game_y: float) -> tuple[int, int]:
        """Convert game coordinates to grid (col, row)."""
        wx, wy = self._game_to_wld(game_x, game_y)
        col = int((wx - self._min_x) / self.cell_size)
        row = int((wy - self._min_y) / self.cell_size)
        return (col, row)

    def _grid_idx(self, col: int, row: int) -> int:
        """Return flat array index, or -1 if out of bounds."""
        if 0 <= col < self._cols and 0 <= row < self._rows:
            return row * self._cols + col
        return -1

    # ------------------------------------------------------------------
    # Cache save/load
    # ------------------------------------------------------------------

    def save_cache(self, path: str | Path) -> None:
        """Save terrain data to a binary cache file (EQTM v3)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            # Header
            f.write(self._CACHE_MAGIC)
            f.write(
                struct.pack(
                    "<BdddII",
                    self._CACHE_VERSION,
                    self.cell_size,
                    self._min_x,
                    self._min_y,
                    self._cols,
                    self._rows,
                )
            )

            # Grid data: 16 bytes/cell (v3)
            total = self._cols * self._rows
            for i in range(total):
                f.write(
                    struct.pack(
                        "<ffHBBf",
                        self._z[i],
                        self._z_ceiling[i] if self._z_ceiling else float("nan"),
                        self._flags[i] & 0xFFFF,
                        self._material_id[i] if self._material_id else 0,
                        self._region_id[i] if self._region_id else 0,
                        self._normal_z[i],
                    )
                )

            # BSP nodes
            f.write(struct.pack("<I", len(self._bsp_nodes)))
            for n in self._bsp_nodes:
                f.write(
                    struct.pack(
                        "<ffffIII",
                        n.normal_x,
                        n.normal_y,
                        n.normal_z,
                        n.split_distance,
                        n.region,
                        n.front,
                        n.back,
                    )
                )

            # Region types
            f.write(struct.pack("<I", len(self._region_types)))
            for rid, rtype in self._region_types.items():
                f.write(struct.pack("<II", rid, rtype))

        size_kb = path.stat().st_size / 1024
        log.info("[NAV] Saved terrain cache: %s (%.0f KB)", path.name, size_kb)

    def load_cache(self, path: str | Path) -> bool:
        """Load terrain from cache file. Returns True if successful."""
        path = Path(path)
        if not path.exists():
            return False

        try:
            with open(path, "rb") as f:
                magic = f.read(4)
                if magic != self._CACHE_MAGIC:
                    return False

                version, cell_size, min_x, min_y, cols, rows = struct.unpack(
                    "<BdddII", f.read(1 + 8 * 3 + 4 * 2)
                )

                if version not in (2, 3):
                    return False

                # Reject cache if cell_size doesn't match requested resolution
                if abs(cell_size - self.cell_size) > 0.01:
                    log.warning(
                        "[NAV] Cache cell_size %.1f != requested %.1f -- rejecting", cell_size, self.cell_size
                    )
                    return False

                self.cell_size = cell_size
                self._min_x = min_x
                self._min_y = min_y
                self._cols = cols
                self._rows = rows

                total = cols * rows
                self._z = [0.0] * total
                self._z_ceiling = [float("nan")] * total
                self._flags = [0] * total
                self._normal_z = [0.0] * total
                self._material_id = [0] * total
                self._region_id = [0] * total

                if version == 3:
                    # v3: 16 bytes/cell
                    for i in range(total):
                        z, zc, flags, mat, reg, nz = struct.unpack("<ffHBBf", f.read(16))
                        self._z[i] = z
                        self._z_ceiling[i] = zc
                        self._flags[i] = flags
                        self._material_id[i] = mat
                        self._region_id[i] = reg
                        self._normal_z[i] = nz
                else:
                    # v2: 9 bytes/cell (backward compat)
                    for i in range(total):
                        z, flags, nz = struct.unpack("<fBf", f.read(9))
                        self._z[i] = z
                        self._flags[i] = flags
                        self._normal_z[i] = nz

                # BSP nodes
                bsp_count = struct.unpack("<I", f.read(4))[0]
                self._bsp_nodes = []
                for _ in range(bsp_count):
                    nx, ny, nz_val, sd, reg, front, back = struct.unpack("<ffffIII", f.read(28))
                    self._bsp_nodes.append(BSPNode(nx, ny, nz_val, sd, reg, front, back))

                # Region types
                rt_count = struct.unpack("<I", f.read(4))[0]
                self._region_types = {}
                for _ in range(rt_count):
                    rid, rtype = struct.unpack("<II", f.read(8))
                    self._region_types[rid] = rtype

            # Build walkability bitfield for fast JPS scanning
            self._build_walk_bits()

            log.info(
                "[NAV] Loaded terrain cache: %s (%dx%d, %.0f unit cells)",
                path.name,
                self._cols,
                self._rows,
                self.cell_size,
            )
            return True

        except (struct.error, ValueError, EOFError) as e:
            log.warning("[NAV] Failed to load terrain cache %s: %s", path, e)
            return False


# ======================================================================
# Convenience: build terrain for a zone
# ======================================================================


def _load_object_meshes(
    zone_path: Path,
    zone_name: str,
    S3DArchive: type,
    WLDFile: type,
) -> tuple[dict[str, float] | None, dict[str, Mesh] | None]:
    """Load object meshes and radii from the zone's _obj.s3d file.

    Returns (mesh_radii, object_meshes), both None if unavailable.
    """
    mesh_radii: dict[str, float] | None = None
    object_meshes: dict[str, Mesh] | None = None
    obj_s3d_path = zone_path.parent / f"{zone_name}_obj.s3d"
    if not obj_s3d_path.exists():
        return mesh_radii, object_meshes

    try:
        from eq.placeables import compute_mesh_footprint_radius, get_model_radius

        obj_arc = S3DArchive(obj_s3d_path)
        obj_wld_name = f"{zone_name}_obj.wld"
        obj_wld_files = [f for f in obj_arc.list_files() if f.endswith(".wld")]
        if obj_wld_name not in obj_arc and obj_wld_files:
            obj_wld_name = obj_wld_files[0]
        if obj_wld_name in obj_arc:
            obj_wld = WLDFile(obj_arc.extract(obj_wld_name))
            object_meshes = obj_wld.extract_actor_meshes()

            mesh_radii = {}
            for model_name, m in object_meshes.items():
                r = compute_mesh_footprint_radius(m)
                if r > 0.0:
                    mesh_radii[model_name] = r

            log.info(
                "[NAV] Loaded %d mesh objects, %d radii from %s",
                len(object_meshes),
                len(mesh_radii),
                obj_s3d_path.name,
            )

            mismatches = 0
            for mname, mrad in mesh_radii.items():
                guessed = get_model_radius(mname)
                if guessed > 0.0 and abs(mrad - guessed) > guessed * 0.5:
                    mismatches += 1
                    log.debug("[NAV] Radius mismatch %s: mesh=%.1f guess=%.1f", mname, mrad, guessed)
            if mismatches:
                log.info("[NAV] %d models have >50%% radius discrepancy (mesh vs name-guess)", mismatches)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
        log.warning("[NAV] Failed to load object meshes from %s: %s", obj_s3d_path.name, e)

    return mesh_radii, object_meshes


def build_zone_terrain(
    zone_s3d_path: str | Path, cache_dir: str | Path | None = None, cell_size: float = 1.0
) -> ZoneTerrain:
    """Extract terrain from a zone's asset archive.

    If a cache file exists, loads from cache. Otherwise builds from
    the archive geometry data and saves the cache for next time.

    Args:
        zone_s3d_path: Path to zone archive file
        cache_dir: Directory for cache files (default: data/terrain/)
        cell_size: Grid resolution in game units (default: 1.0)

    Returns:
        ZoneTerrain with heightmap and hazard data ready for queries.
    """
    from eq.s3d import S3DArchive

    zone_path = Path(zone_s3d_path)
    zone_name = zone_path.stem  # e.g. "nektulos"

    if cache_dir is None:
        # Navigate from compass/nav/terrain/ up to project root
        cache_dir = Path(__file__).parents[3] / "data" / "terrain"
    cache_path = Path(cache_dir) / f"{zone_name}.terrain"

    terrain = ZoneTerrain(cell_size=cell_size)

    # Try cache first
    if terrain.load_cache(cache_path):
        return terrain

    # Build from S3D
    log.info("[NAV] Building terrain for %s from %s", zone_name, zone_path)

    arc = S3DArchive(zone_path)

    # Find the zone WLD (usually {zonename}.wld)
    wld_name = f"{zone_name}.wld"
    if wld_name not in arc:
        # Fall back to first .wld file
        wld_files = [f for f in arc.list_files() if f.endswith(".wld")]
        if not wld_files:
            log.error("[NAV] No WLD files found in %s", zone_path)
            return terrain
        wld_name = wld_files[0]

    from eq.wld import WLDFile

    wld = WLDFile(arc.extract(wld_name))
    meshes = wld.extract_meshes()
    bsp_nodes = wld.extract_bsp_nodes()
    region_types = wld.extract_region_types()

    # Extract material names for bridge detection + surface classification
    mesh_materials = wld.extract_mesh_material_names()

    terrain.build(meshes, bsp_nodes, region_types, mesh_materials=mesh_materials)

    # Extract object placements from objects.wld inside the main S3D.
    # Placements live in a separate WLD file (objects.wld), NOT in the
    # main {zone}.wld which only contains terrain mesh + BSP + regions.
    placements = []
    if "objects.wld" in arc:
        obj_placement_wld = WLDFile(arc.extract("objects.wld"))
        placements = obj_placement_wld.extract_placements()
    else:
        # Fallback: check main WLD (some zones embed placements)
        placements = wld.extract_placements()

    mesh_radii, object_meshes = _load_object_meshes(zone_path, zone_name, S3DArchive, WLDFile)

    # Apply obstacle footprints to terrain grid.
    # Mesh-accurate footprints already cover the exact object shape --
    # zero buffer needed (the polygon IS the collision boundary).
    # Without mesh data, circular radius is already conservative (1u buf).
    buf = 0 if object_meshes else 1
    if placements:
        terrain.apply_obstacles(placements, mesh_radii, object_meshes, buffer_cells=buf)

    # Save cache
    terrain.save_cache(cache_path)

    return terrain
