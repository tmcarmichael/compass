"""Test factories and hypothesis strategies for typed state objects.

Factory functions construct frozen dataclasses with sensible defaults.
Callers override only the fields they care about. Hypothesis strategies
compose these factories for property-based testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hypothesis import strategies as st

from brain.goap.world_state import PlanWorldState
from core.types import Point
from perception.state import GameState, SpawnData

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.scoring.target import MobProfile

# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def make_spawn(**overrides: Any) -> SpawnData:
    """Construct a SpawnData with sensible NPC defaults."""
    defaults: dict[str, Any] = dict(
        spawn_id=100,
        name="a_skeleton",
        x=50.0,
        y=50.0,
        z=0.0,
        heading=0.0,
        speed=0.0,
        level=10,
        spawn_type=1,  # NPC
        race=0,
        mob_class=0,
        hide=0,
        hp_current=100,
        hp_max=100,
    )
    defaults.update(overrides)
    return SpawnData(**defaults)


def make_game_state(**overrides: Any) -> GameState:
    """Construct a GameState with sensible defaults for all required fields."""
    defaults: dict[str, Any] = dict(
        x=0.0,
        y=0.0,
        z=0.0,
        heading=0.0,
        hp_current=1000,
        hp_max=1000,
        mana_current=500,
        mana_max=500,
        level=10,
        name="TestPlayer",
        spawn_type=0,  # Player
        stand_state=0,
        player_state=0,
        spawn_id=1,
        speed_run=0.7,
        speed_heading=0.5,
    )
    defaults.update(overrides)
    return GameState(**defaults)


def make_plan_world_state(**overrides: Any) -> PlanWorldState:
    """Construct a PlanWorldState with all-default fields, overrides applied."""
    if not overrides:
        return PlanWorldState()
    defaults: dict[str, Any] = dict(
        hp_pct=1.0,
        mana_pct=1.0,
        pet_alive=True,
        engaged=False,
        has_target=False,
        corpse_nearby=False,
        buffs_active=True,
        spells_ready=True,
        inventory_pct=0.0,
        at_camp=True,
        targets_available=0,
        nearby_threats=0,
    )
    defaults.update(overrides)
    return PlanWorldState(**defaults)


def make_mob_profile(**overrides: Any) -> MobProfile:
    """Construct a MobProfile with sensible defaults for testing."""
    from brain.scoring.target import MobProfile
    from core.types import Disposition
    from perception.combat_eval import Con

    spawn = overrides.pop("spawn", make_spawn())
    defaults: dict[str, Any] = dict(
        spawn=spawn,
        con=Con.WHITE,
        disposition=Disposition.SCOWLING,
        distance=50.0,
        camp_distance=40.0,
        isolation_score=80.0,
        nearby_npc_count=0,
        social_npc_count=0,
        is_moving=False,
        speed=0.0,
        velocity=(0.0, 0.0, 0.0),
        predicted_pos_5s=Point(spawn.x, spawn.y, spawn.z),
        fight_duration_est=30.0,
        mana_cost_est=200,
        threat_level=0.0,
        is_threat=False,
    )
    defaults.update(overrides)
    return MobProfile(**defaults)


def make_agent_context(**overrides: Any) -> AgentContext:
    """Construct an AgentContext with sensible test defaults.

    Supports nested field access via double-underscore syntax:
        make_agent_context(pet__alive=True, camp__roam_radius=300.0)
    Also supports flat overrides for top-level fields:
        make_agent_context(rest_hp_entry=0.90)
    """
    from brain.context import AgentContext
    from perception.combat_eval import Con

    ctx = AgentContext()
    ctx.pet.alive = True
    ctx.zone.target_cons = frozenset({Con.WHITE, Con.BLUE, Con.LIGHT_BLUE})
    for k, v in overrides.items():
        if "__" in k:
            obj_name, field = k.split("__", 1)
            setattr(getattr(ctx, obj_name), field, v)
        else:
            setattr(ctx, k, v)
    return ctx


def make_fight_record(**overrides: Any) -> dict[str, Any]:
    """Return kwargs suitable for FightHistory.record().

    Provides sensible defaults for a successful encounter.
    """
    defaults: dict[str, Any] = dict(
        mob_name="a_skeleton",
        duration=20.0,
        mana_spent=50,
        hp_delta=-0.05,
        casts=2,
        pet_heals=0,
        pet_died=False,
        defeated=True,
        adds=0,
        mob_level=10,
        player_level=10,
        con="white",
        strategy="pet_and_dot",
        mana_start=500,
        mana_end=450,
        pet_hp_start=1.0,
        pet_hp_end=0.8,
        xp_gained=True,
        cycle_time=25.0,
        fitness=0.6,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
_coord = st.floats(min_value=-5000, max_value=5000, allow_nan=False, allow_infinity=False)
_positive_int = st.integers(min_value=0, max_value=10_000)

st_plan_world_state = st.builds(
    PlanWorldState,
    hp_pct=_unit,
    mana_pct=_unit,
    pet_alive=st.booleans(),
    engaged=st.booleans(),
    has_target=st.booleans(),
    targets_available=st.integers(min_value=0, max_value=20),
    inventory_pct=_unit,
    nearby_threats=st.integers(min_value=0, max_value=10),
)

st_spawn = st.builds(
    make_spawn,
    spawn_id=st.integers(min_value=1, max_value=10_000),
    name=st.sampled_from(["a_skeleton", "a_moss_snake", "a_bat", "a_fire_beetle"]),
    x=_coord,
    y=_coord,
    level=st.integers(min_value=1, max_value=60),
    hp_current=_positive_int,
    hp_max=st.integers(min_value=1, max_value=10_000),
)

st_game_state = st.builds(
    make_game_state,
    x=_coord,
    y=_coord,
    heading=st.floats(min_value=0, max_value=512, allow_nan=False),
    hp_current=_positive_int,
    hp_max=st.integers(min_value=1, max_value=10_000),
    mana_current=_positive_int,
    mana_max=st.integers(min_value=1, max_value=10_000),
    level=st.integers(min_value=1, max_value=60),
)
