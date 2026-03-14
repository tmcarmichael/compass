"""Shared constants used by 2+ modules across routines and brain rules.

Single-consumer constants belong in their domain module, not here.
Distance units are EQ world units.
"""

# -- Distances (EQ world units) -----------------------------------------------

MELEE_RANGE = 15.0  # NPC in melee range (attacking player)
BACKSTEP_RANGE = 80.0  # npc close enough to trigger backstep during pull
SCAN_RADIUS = 200.0  # Tab targeting / acquire search radius
SPELL_RANGE = 200.0  # max spell cast range
PULL_ABORT_DISTANCE = 250.0  # abort pull if target beyond this
LOST_PULL_DISTANCE = 300.0  # target fled too far with high HP
GUARD_CHECK_RADIUS = 150.0  # skip npcs near guards
SOCIAL_NPC_RADIUS = 100.0  # social threat scan radius (EQ social threat ~100u)
PET_CLOSE_RANGE = 60.0  # pet close enough to not need recall
RECALL_DISTANCE = 150.0  # recall pet only for far pulls with confirmed hit
ADDS_DETECT_RANGE = 50.0  # hostile NPC this close is probably attacking us
PLAYER_MOB_PROXIMITY = 30.0  # skip npcs within this range of another player
OPTIMAL_PULL_MIN = 60.0  # don't pull closer than this
OPTIMAL_PULL_MAX = 130.0  # don't pull farther than this
OPTIMAL_PULL_TARGET = 100.0  # ideal pull distance

# -- Target Filtering (used by target_filter.py + acquire.py) -----------------

NEARBY_AGGRO_RADIUS = 30.0  # NPC clustering check radius for add risk
DAMAGED_TARGET_HP_THRESHOLD = 0.9  # reject targets below this HP fraction
SITTING_MANA_REGEN_RATE = 3.0  # approximate mana/sec while sitting (for rest estimates)
THREAT_RADIUS_PER_LEVEL = 3.0  # threat avoidance radius = level * this + base
THREAT_RADIUS_BASE = 40.0  # threat avoidance radius base offset

# -- Combat Thresholds (used by combat.py + strategies + pet_combat) ----------

PET_HEAL_THRESHOLD = 0.85  # heal pet when HP drops to this %
MEND_BONES_RECAST = 7.0  # Mend Bones recast (6.5s + safety margin)
LOST_PULL_HP_THRESHOLD = 0.90  # target HP above this = lost pull
COMBAT_LOG_INTERVAL = 2.0  # seconds between detailed combat state logs

# -- XP Scale ------------------------------------------------------------------

XP_SCALE_MAX = 330  # EQ XP raw value: 0-330 maps to 0%-100% of current level

# -- Camp Selection ---------------------------------------------------------------

LEVEL_RANGE_PENALTY = 5_000  # camp scoring: distance penalty per level outside range
