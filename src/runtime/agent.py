"""Agent library: config loading, context building, brain wiring."""

from __future__ import annotations

import ctypes
import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from perception.state import GameState

from brain.context import AgentContext
from brain.decision import Brain
from perception.reader import MemoryReader
from runtime.camp_selector import apply_camp as _apply_camp
from runtime.camp_selector import select_camp

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Typed config dataclass (replaces raw TOML dict access)
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    """Rest-threshold overrides from settings.toml [thresholds]."""

    rest_hp_high: float = 0.92
    rest_mana_high: float = 0.70
    rest_hp_low: float = 0.30
    rest_mana_low: float = 0.20

    @classmethod
    def from_toml(cls, raw: dict) -> ThresholdConfig:
        section = raw.get("thresholds", {})
        return cls(
            rest_hp_high=section.get("rest_hp_high", 0.92),
            rest_mana_high=section.get("rest_mana_high", 0.70),
            rest_hp_low=section.get("rest_hp_low", 0.30),
            rest_mana_low=section.get("rest_mana_low", 0.20),
        )


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------


def find_config() -> Path:
    candidates = [
        Path(__file__).resolve().parents[1] / "config" / "settings.toml",
        Path.cwd() / "config" / "settings.toml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"settings.toml not found: {[str(c) for c in candidates]}")


def load_zone_config(zone_name: str, player_level: int = 0) -> dict[str, Any]:
    candidates = [
        Path(__file__).resolve().parents[1] / "config" / "zones" / f"{zone_name}.toml",
        Path.cwd() / "config" / "zones" / f"{zone_name}.toml",
    ]
    for p in candidates:
        if p.exists():
            log.info("[LIFECYCLE] Loading zone config from TOML: %s", p)
            with open(p, "rb") as f:
                return tomllib.load(f)

    # No TOML found -- return empty config
    log.warning("[LIFECYCLE] No TOML config for '%s' -- using empty config", zone_name)
    return {}


def _is_admin() -> bool:
    try:
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return False
        return bool(windll.shell32.IsUserAnAdmin() != 0)
    except (
        OSError,
        AttributeError,
        ValueError,
    ):
        return False


# ------------------------------------------------------------------
# Context building (orchestrator)
# ------------------------------------------------------------------


def build_context(
    config: dict[str, Any],
    zone_config: dict[str, Any],
    player_x: float = 0.0,
    player_y: float = 0.0,
    player_level: int = 0,
) -> AgentContext:
    """Build AgentContext from config files.

    Auto-selects the best camp by level fit + proximity.
    Falls back to settings.toml active_camp if position unknown.
    """
    # -- Camp selection + application --
    camps = zone_config.get("camps", [])
    fallback = config.get("general", {}).get("active_camp", "")
    from core.types import Point

    camp = select_camp(camps, Point(player_x, player_y, 0.0), player_level, fallback_name=fallback)

    ctx = AgentContext()
    ctx.zone.zone_camps = camps
    ctx.zone.zone_config = zone_config
    _apply_camp(ctx, camp)

    # -- Zone knowledge --
    _configure_zone_knowledge(ctx, zone_config, player_level)

    # -- Thresholds (typed config) --
    thresholds = ThresholdConfig.from_toml(config)
    ctx.rest_hp_threshold = thresholds.rest_hp_high
    ctx.rest_mana_threshold = thresholds.rest_mana_high
    ctx.rest_hp_entry = thresholds.rest_hp_low
    ctx.rest_mana_entry = thresholds.rest_mana_low

    # -- Resources + npc types --
    _configure_resources(ctx, zone_config)

    # -- NPC avoidance --
    from perception.combat_eval import configure_avoid_names

    configure_avoid_names(zone_config)

    # -- Zone intel defaults --
    ctx.loot.caster_mob_names = set()

    # -- Spatial memory + fight history --
    from brain.learning.encounters import FightHistory
    from brain.learning.spatial import SpatialMemory

    zone_short = zone_config.get("zone", {}).get("short_name", "unknown")
    ctx.spatial_memory = SpatialMemory(zone_short)
    ctx.fight_history = FightHistory(zone=zone_short)

    # -- Navigation --
    from nav.waypoint_graph import parse_waypoint_graph
    from routines.travel import parse_tunnel_routes

    ctx.tunnel_routes = parse_tunnel_routes(zone_config)
    ctx.waypoint_graph = parse_waypoint_graph(zone_config)

    return ctx


# ------------------------------------------------------------------
# Helpers (called by build_context)
# ------------------------------------------------------------------


def _configure_zone_knowledge(ctx: AgentContext, zone_config: dict[str, Any], player_level: int = 0) -> None:
    """Merge TOML seed + persisted learned data into zone knowledge."""
    from brain.learning.zone import ZoneKnowledge

    toml_dispositions = zone_config.get("disposition")
    toml_social = zone_config.get("social", {}).get("groups", [])
    zone_short = zone_config.get("zone", {}).get("short_name", "unknown")
    zk = ZoneKnowledge(
        zone_name=zone_short,
        toml_dispositions=toml_dispositions,
        toml_social_groups=toml_social,
    )
    ctx.zone.zone_knowledge = zk
    ctx.zone.zone_dispositions = zk.get_merged_dispositions()
    # Scale target cons with level: WHITE cons safe at level 8+ with new pet spell
    from perception.combat_eval import Con

    if ctx.zone.target_cons == frozenset({Con.BLUE, Con.LIGHT_BLUE}):
        if player_level >= 8:
            ctx.zone.target_cons = frozenset({Con.WHITE, Con.BLUE, Con.LIGHT_BLUE})
            log.info("Target cons: added WHITE (level %d)", player_level)
    ctx.zone.social_mob_group = zk.build_social_mob_group()


def _configure_resources(ctx: AgentContext, zone_config: dict[str, Any]) -> None:
    """Resource targets + undead npc names from zone config."""
    resources = zone_config.get("resources", [])
    if resources:
        ctx.loot.resource_targets = {}
        for res in resources:
            for npc in res.get("source_mobs", []):
                ctx.loot.resource_targets[npc] = res.get("item_name", "")
            ctx.loot.resource_item_target = res.get("target_count", 0)
        log.info("[LIFECYCLE] Resource targets from config: %s", ctx.loot.resource_targets)
    else:
        ctx.loot.resource_targets = {}

    mob_types = zone_config.get("mob_types", {})
    ctx.loot.undead_names = set(mob_types.get("undead", []))


# ------------------------------------------------------------------
# Brain wiring
# ------------------------------------------------------------------


def build_brain(ctx: AgentContext, reader: MemoryReader) -> Brain:
    """Wire all brain rules via modular rule registration."""
    from brain.rules import register_all
    from core.features import flags
    from motor.actions import release_all_keys

    brain = Brain(ctx=ctx, utility_phase=flags.utility_phase, shutdown_hook=release_all_keys)
    ctx.reader = reader  # make reader available to routines

    # Initialize world model
    from brain.world.model import WorldModel

    ctx.world = WorldModel(ctx=ctx)
    # Seed fight durations from persisted spatial memory
    if ctx.spatial_memory:
        ctx.world.load_from_spatial(ctx.spatial_memory)

    def read_state() -> GameState:
        return reader.read_state(include_spawns=True)

    register_all(brain, ctx, read_state)
    return brain
