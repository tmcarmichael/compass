"""Perception layer memory layout constants.

Defines the binding contract between the perception layer and the target
environment's process memory. Two categories:

  Root pointers  -- absolute addresses into the target binary. Zeroed in
                   this public release so the repository serves as an
                   architecture reference, not a turnkey tool.

  Struct offsets -- field positions within named structs. Values here are
                   synthetic placeholders that preserve the schema and keep
                   tests self-consistent. They do not correspond to any
                   real client binary.
"""

# ── Root pointers ──────────────────────────────────────────────────────────────

PLAYER_SPAWN_PTR = 0x0
TARGET_PTR = 0x0
ZONE_PTR = 0x0
ITEMS_PTR = 0x0

CASTING_MODE_PTR = 0x0
ANIMATION_FLAG_PTR = 0x0

IN_COMBAT_GLOBAL = 0x0
AUTOATTACK_TIME = 0x0


# ── Entity struct field offsets ────────────────────────────────────────────────
# Relative to the base of any entity node (player or NPC).

# Linked list traversal
NEXT = 0x0004  # DWORD
PREV = 0x0008  # DWORD

# Position
Y = 0x0010  # FLOAT
X = 0x0014  # FLOAT
Z = 0x0018  # FLOAT

# Velocity
VELOCITY_Y = 0x001C  # FLOAT
VELOCITY_X = 0x0020  # FLOAT
VELOCITY_Z = 0x0024  # FLOAT

# Movement
SPEED = 0x0028  # FLOAT
HEADING = 0x002C  # FLOAT

# Identity
NAME = 0x0030  # CHAR[64]

# Zone
ZONE_ID = 0x0074  # INT32

# Classification
TYPE = 0x0078  # BYTE
LEVEL = 0x0079  # BYTE
HIDE = 0x007A  # BYTE
BODY_STATE = 0x007B  # BYTE
CLASS = 0x007C  # BYTE
OWNER_SPAWN_ID = 0x0080  # DWORD
SPAWN_ID = 0x0084  # DWORD
RACE = 0x0088  # DWORD

# Equipment
PRIMARY = 0x0090  # DWORD
OFFHAND = 0x0094  # DWORD

# Vitals
HP_CURRENT = 0x0098  # INT32
HP_MAX = 0x009C  # INT32
NO_REGEN_FLAG = 0x00A0  # BYTE

# Casting state
CASTING_STATE = 0x00A4  # INT32
CASTING_SPELL_ID = 0x00A8  # UINT32
CASTING_TARGET_ID = 0x00AC  # UINT32

# Social / guild
GUILD_ID = 0x00B0  # INT32
LASTNAME = 0x00B4  # CHAR[32]
TITLE = 0x00D4  # CHAR[32]

STAND_STATE = 0x0000
PLAYER_STATE = 0x0000


# ── Animation/physics sub-struct ───────────────────────────────────────────────

ACTORCLIENT_PTR = 0x00F8  # PTR -> animation/physics sub-struct

# Offsets relative to sub-struct base
AC_TARGET_NAME = 0x0100  # CHAR[64]
AC_ACTIVITY_STATE = 0x0144  # INT32
AC_COMBAT_FLAG = 0x0148  # INT32


# ── Player profile chain ───────────────────────────────────────────────────────

CHARINFO_PTR = 0x00FC  # entity -> character info struct pointer
CHARINFO_PROFILE_INDIR = 0x0010  # character info -> intermediate pointer
PROFILE_PTR_OFFSET = 0x0004  # intermediate -> profile base

# Inventory
PROFILE_EQUIP_START = 0x0018  # first equipment slot pointer
PROFILE_BAG_START = 0x0058  # first bag slot pointer
PROFILE_BAG_COUNT = 8
PROFILE_BAG_STRIDE = 4

# Buff array: 25 slots at 20-byte stride
PROFILE_BUFF_BASE = 0x0080
PROFILE_BUFF_SLOT_SIZE = 20
PROFILE_BUFF_SPELL_ID_OFF = 0x04
PROFILE_BUFF_TICKS_OFF = 0x08
PROFILE_BUFF_COUNT = 25

# Spellbook and spell gems
PROFILE_SPELLBOOK = 0x0300  # INT32[400]
PROFILE_SPELLBOOK_SIZE = 400
PROFILE_SPELL_GEMS = 0x0940  # INT32[10]
PROFILE_SPELL_GEM_COUNT = 10

# Progression
PROFILE_PRACTICE_POINTS = 0x0970  # INT32
PROFILE_LEVEL = 0x0974  # INT32

# Identity
PROFILE_RACE = 0x0978  # INT32
PROFILE_CLASS = 0x097C  # INT32

# Resources
PROFILE_MANA = 0x0980  # INT32
PROFILE_HP_MAX_TOTAL = 0x0984  # INT32
PROFILE_HP_CURRENT = 0x0988  # INT32

# Base stats
PROFILE_STR = 0x0990  # INT32
PROFILE_STA = 0x0994  # INT32
PROFILE_CHA = 0x0998  # INT32
PROFILE_DEX = 0x099C  # INT32
PROFILE_INT = 0x09A0  # INT32
PROFILE_AGI = 0x09A4  # INT32
PROFILE_WIS = 0x09A8  # INT32

# Currency
PROFILE_PP = 0x09B0  # INT32
PROFILE_GP = 0x09B4  # INT32
PROFILE_SP = 0x09B8  # INT32
PROFILE_CP = 0x09BC  # INT32

# Maintenance
PROFILE_HUNGER = 0x09C0  # INT32
PROFILE_THIRST = 0x09C4  # INT32
PROFILE_UNKNOWN_LEVEL_SCALE = 0x09C8  # INT32

# Resists
PROFILE_MR = 0x09D0  # INT32
PROFILE_FR = 0x09D4  # INT32
PROFILE_CR = 0x09D8  # INT32
PROFILE_DR = 0x09DC  # INT32
PROFILE_PR = 0x09E0  # INT32

# Character info direct field
CHARINFO_WEIGHT = 0x0040  # INT32


# ── UI window layout ───────────────────────────────────────────────────────────

CXWND_X = 0x0010  # INT32
CXWND_Y = 0x0014  # INT32
CXWND_RIGHT = 0x0018  # INT32
CXWND_BOTTOM = 0x001C  # INT32

WND_VIS = 0x0004  # UINT32
WND_X = 0x0010  # UINT32
WND_Y = 0x0014  # UINT32

# UI window root pointers
SPELL_BOOK_WND_PTR = 0x0
SPELL_BOOK_WND_SIZE = 0x0100
CAST_SPELL_WND_PTR = 0x0
TRADE_WND_PTR = 0x0
INVENTORY_WND_PTR = 0x0
CURSOR_ITEM_PTR = 0x0
LOOT_WND_PTR = 0x0

# Visibility offset
TRADE_WND_VISIBLE_OFFSET = 0x0004
INVENTORY_WND_VISIBLE_OFFSET = 0x0004
CURSOR_ITEM_VISIBLE_OFFSET = 0x0004
LOOT_WND_VISIBLE_OFFSET = 0x0004

# Loot window item data
LOOT_WND_METADATA_OFFSET = 0x0020
LOOT_WND_ITEM_IDS_OFFSET = LOOT_WND_METADATA_OFFSET
LOOT_WND_ITEM_SLOTS = 31
LOOT_WND_CONTENTS_OFFSET = 0x00A0


# ── Container window manager ───────────────────────────────────────────────────

CONTAINER_MGR_VT = 0x0
CONTAINER_WND_VT = 0x0
CONTAINER_MGR_ARRAY_OFF = 0x04


# ── CONTENTS struct (item instance) ────────────────────────────────────────────

CONTENTS_ITEMINFO_PTR = 0x0010  # PTR -> item definition struct
CONTENTS_STACK_COUNT = 0x0014  # INT32
CONTENTS_BAG_ITEMS_START = 0x0018  # first sub-item pointer (bags only)
CONTENTS_VTABLE = 0x0

INV_BAG_COUNT = 8
CHARINFO_BAG_SLOTS = 10


# ── Item definition struct ──────────────────────────────────────────────────────

ITEMINFO_NAME = 0x00  # CHAR[64]
ITEMINFO_ITEM_NUMBER = 0x0044  # DWORD


# ── Game engine object ──────────────────────────────────────────────────────────

GAME_ENGINE_PTR = 0x0
GAME_ENGINE_XP_PCT = 0x0010  # UINT32
GAME_ENGINE_KILL_COUNT = 0x0014  # UINT32

# Virtual base pointer chain
GAME_ENGINE_VBPTR_OFFSET = 0x0008
GAME_ENGINE_VB_ZONE_ID = 0x0020  # UINT32
GAME_ENGINE_VB_STATE_FLAG = 0x0024  # BYTE
GAME_ENGINE_VB_VEHICLE_Y = 0x0028  # FLOAT
GAME_ENGINE_VB_VEHICLE_X = 0x002C  # FLOAT
GAME_ENGINE_VB_VEHICLE_Z = 0x0030  # FLOAT


# ── Engine state ───────────────────────────────────────────────────────────────

ENGINE_STATE_PTR = 0x0
ENGINE_GAME_MODE_OFFSET = 0x0010  # INT32

CASTING_SPELL_ID_PTR = 0x0000
SPEED_RUN = SPEED
SPEED_HEADING = 0x0000
