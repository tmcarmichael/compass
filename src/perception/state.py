"""GameState and SpawnData: frozen snapshots of the world each tick."""

from dataclasses import dataclass

from core.constants import XP_SCALE_MAX
from core.types import Point


@dataclass(frozen=True, slots=True)
class DoorData:
    """State of a single door/switch in the zone.

    Used to dynamically mark open doors as walkable terrain so the
    agent can navigate through doorways and zone transitions.
    """

    door_id: int
    name: str
    x: float
    y: float
    z: float
    heading: float
    is_open: bool
    # Approximate door width for terrain patching (default 10u)
    width: float = 10.0


@dataclass(frozen=True, slots=True, kw_only=True)
class SpawnData:
    """Data for a single spawn (NPC, player, or corpse)."""

    spawn_id: int
    name: str
    x: float
    y: float
    z: float
    heading: float
    speed: float
    level: int
    spawn_type: int  # 0=Player, 1=NPC, 2=NPC Corpse, 3=Player Corpse
    race: int
    mob_class: int
    hide: int
    hp_current: int = 0
    hp_max: int = 0
    owner_id: int = 0  # spawn_id of owner (pet/corpse link). 0=no owner.
    body_state: str = "n"  # 'd'=dead 'f'=feign 'n'=normal 'o'=mounted 'i'=invis
    target_name: str = ""  # who this NPC is attacking. empty=idle.
    velocity_y: float = 0.0  # Y velocity component. instant, no lag.
    velocity_x: float = 0.0  # X velocity component.
    velocity_z: float = 0.0  # Z velocity component.

    @property
    def pos(self) -> Point:
        """World-space position as Point."""
        return Point(self.x, self.y, self.z)

    @property
    def is_npc(self) -> bool:
        return self.spawn_type == 1

    @property
    def is_player(self) -> bool:
        return self.spawn_type == 0

    @property
    def is_corpse(self) -> bool:
        return self.spawn_type in (2, 3)

    @property
    def is_player_corpse(self) -> bool:
        return self.spawn_type == 3

    @property
    def is_owned(self) -> bool:
        return self.owner_id != 0

    @property
    def is_feigning(self) -> bool:
        return self.body_state == "f"

    @property
    def is_dead_body(self) -> bool:
        return self.body_state == "d"

    @property
    def is_invisible(self) -> bool:
        return self.body_state == "i"


@dataclass(frozen=True, slots=True, kw_only=True)
class GameState:
    """Immutable snapshot of game state read from memory each tick."""

    # Player position
    x: float
    y: float
    z: float
    heading: float  # 0-512 range

    # Player vitals (0 if offset not yet verified)
    hp_current: int
    hp_max: int
    mana_current: int
    mana_max: int
    level: int

    # Player status
    name: str
    spawn_type: int  # 0=Player, 1=NPC, 2=NPC Corpse, 3=Player Corpse
    stand_state: int  # sit/stand/feign
    player_state: int  # 0=Idle, 4=Aggressive, etc.
    spawn_id: int

    # Speeds
    speed_run: float
    speed_heading: float

    # Character class
    class_id: int = 0  # class ID (1=WAR..11=NEC..16=BER)

    # Player body state (same as SpawnData.body_state)
    body_state: str = "n"  # 'd'=dead 'f'=feign 'n'=normal 'o'=mounted 'i'=invis

    # Inventory
    weight: int = 0  # current carry weight

    # Money (personal)
    money_pp: int = 0  # platinum
    money_gp: int = 0  # gold
    money_sp: int = 0  # silver
    money_cp: int = 0  # copper

    # Experience
    xp_pct_raw: int = 0  # 0-330 scale (330 = 100% of current level)
    defeat_count: int = 0  # total defeats this session (increments by 1 per defeat)

    # Casting state
    casting_mode: int = 0  # 0=idle, 1=spell cast, 2+=other UI modes (spellbook, mem)
    casting_spell_id: int = -1  # Spell ID being cast. -1 = unknown.

    # Combat state
    in_combat: bool = False  # True when combat engagement flag is set

    # Zone
    zone_id: int = 0  # zone ID from spawn struct
    engine_zone_id: int = 0  # zone ID from engine object (earliest zone signal)

    # Game mode (from engine state object)
    game_mode: int = 5  # 0=char_select, 3=zoning, 4=loading, 5=in_game, 253=pre-ingame

    # Active buffs: tuple of (spell_id, ticks_remaining)
    buffs: tuple[tuple[int, int], ...] = ()

    # Current target (None if no target)
    target: SpawnData | None = None

    # Nearby spawns
    spawns: tuple[SpawnData, ...] = ()

    # Doors in current zone (empty until door reading is implemented)
    doors: tuple[DoorData, ...] = ()

    @property
    def pos(self) -> Point:
        """World-space position as Point."""
        return Point(self.x, self.y, self.z)

    @property
    def hp_pct(self) -> float:
        if self.hp_max == 0:
            return 1.0  # Assume full if we can't read HP
        return self.hp_current / self.hp_max

    @property
    def mana_pct(self) -> float:
        if self.mana_max == 0:
            return 1.0  # Assume full if we can't read mana
        return self.mana_current / self.mana_max

    @property
    def xp_pct(self) -> float:
        """XP percentage through current level (0.0 to 1.0)."""
        if self.xp_pct_raw <= 0:
            return 0.0
        return min(self.xp_pct_raw / float(XP_SCALE_MAX), 1.0)

    @property
    def has_target(self) -> bool:
        return self.target is not None

    @property
    def is_sitting(self) -> bool:
        return self.stand_state in (1, 2)

    @property
    def is_medding(self) -> bool:
        return self.stand_state == 2

    @property
    def is_casting(self) -> bool:
        return self.casting_mode == 1

    @property
    def is_in_game(self) -> bool:
        return self.game_mode == 5

    @property
    def is_zoning(self) -> bool:
        """True during zone transition (game_mode not charselect or ingame)."""
        return self.game_mode not in (0, 5)

    @property
    def is_at_char_select(self) -> bool:
        return self.game_mode == 0

    @property
    def money_total_cp(self) -> int:
        """Total personal money in copper pieces."""
        return self.money_pp * 1000 + self.money_gp * 100 + self.money_sp * 10 + self.money_cp

    @property
    def is_dead(self) -> bool:
        """True when the player body_state indicates death."""
        return self.body_state == "d"

    @property
    def is_standing(self) -> bool:
        return self.stand_state not in (1, 2)

    def has_buff(self, spell_id: int) -> bool:
        """Check if a specific buff is active."""
        return any(sid == spell_id for sid, _ticks in self.buffs)

    def buff_ticks(self, spell_id: int) -> int:
        """Get remaining ticks for a buff. Returns 0 if not active."""
        for sid, ticks in self.buffs:
            if sid == spell_id:
                return ticks
        return 0

    @property
    def nearby_npcs(self) -> tuple[SpawnData, ...]:
        return tuple(s for s in self.spawns if s.is_npc)

    @property
    def nearby_players(self) -> tuple[SpawnData, ...]:
        return tuple(s for s in self.spawns if s.is_player)
