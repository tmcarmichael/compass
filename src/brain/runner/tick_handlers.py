"""Per-tick event handlers: level-up, adds, auto-engage scan.

Extracted from brain_runner.py to separate per-tick event detection
and handling from brain loop orchestration.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from core.types import Con, PlanType, Point
from core.types import normalize_entity_name as normalize_mob_name
from eq.loadout import configure_loadout
from perception.combat_eval import con_color
from perception.queries import is_pet
from util.event_schemas import LevelUpEvent
from util.structured_log import log_event

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.runner.loop import BrainRunner
    from perception.state import GameState, SpawnData

brain_log = logging.getLogger("compass.brain_loop")


class TickHandlers:
    """Per-tick event detection and handling.

    Composed into BrainRunner -- not a subclass. The runner passes
    itself so the handler can access shared state.

    Args:
        runner: The parent BrainRunner instance.
    """

    def __init__(self, runner: BrainRunner) -> None:
        self._runner = runner
        self._prev_hp: float = 1.0  # HP tracking for auto-engage damage detection
        # Rate-limit auto-engage candidate log per spawn_id
        self._auto_engage_logged: dict[int, float] = {}  # spawn_id -> time.time()

    def handle_level_up(self, state: GameState, ctx: AgentContext) -> None:
        """Handle player level-up: reset mana, reconfigure spells."""
        old_level = self._runner._prev_level
        log_event(
            brain_log,
            "level_up",
            f"[LIFECYCLE] LEVEL UP: {old_level} -> {state.level} (mana={state.mana_current}, defeats={ctx.defeat_tracker.defeats})",
            **LevelUpEvent(
                old_level=old_level,
                new_level=state.level,
                mana=state.mana_current,
                mana_max_old=self._runner._reader._observed_mana_max,
                pos_x=round(state.x),
                pos_y=round(state.y),
                defeats=ctx.defeat_tracker.defeats,
            ),
        )

        # Snapshot BEFORE any resets -- capture the pre-level-up memory state
        brain_log.info("[LIFECYCLE] LEVEL UP: === PRE-LEVEL STATE SNAPSHOT ===")
        self._runner._reader.log_health_check(state)

        # Don't reset mana max here -- we'll read the real value after profile re-resolve
        brain_log.info(
            "[LIFECYCLE] LEVEL UP: mana at level-up: %d (old max %d)",
            state.mana_current,
            self._runner._reader._observed_mana_max,
        )
        self._runner._reader._profile_base_cache = None
        self._runner._reader._profile_chain_failed = False
        brain_log.info("[LIFECYCLE] LEVEL UP: profile chain cache cleared for re-resolve")

        old_gems = configure_loadout(state.class_id, old_level)
        new_gems = configure_loadout(state.class_id, state.level)
        if new_gems:
            brain_log.info(
                "[CAST] SPELL LOADOUT updated for level %d: %s",
                state.level,
                ", ".join(f"gem{g}={n}" for g, n in sorted(new_gems.items())),
            )
        if new_gems != old_gems:
            ctx.plan.active = PlanType.NEEDS_MEMORIZE
            brain_log.info("[LIFECYCLE] LEVEL UP: new spells available -- triggering memorize")

        # Check if pet summon spell upgraded -- if so, resummon for stronger pet
        from eq.loadout import SpellRole, get_spell_by_role

        new_pet_spell = get_spell_by_role(SpellRole.PET_SUMMON)
        if new_pet_spell and ctx.pet.alive:
            # Compare with what we had before -- if spell changed, force resummon
            configure_loadout(state.class_id, old_level)
            old_pet_spell = get_spell_by_role(SpellRole.PET_SUMMON)
            configure_loadout(state.class_id, state.level)  # restore new loadout
            if old_pet_spell and new_pet_spell.name != old_pet_spell.name:
                brain_log.info(
                    "[LIFECYCLE] LEVEL UP: pet spell upgraded %s -> %s -- marking pet for resummon",
                    old_pet_spell.name,
                    new_pet_spell.name,
                )
                ctx.pet.alive = False
                ctx.pet.spawn_id = None
                ctx.pet.name = ""

        # Post-level snapshot: re-read state with fresh profile chain
        try:
            post_state = self._runner._reader.read_state(include_spawns=False)
            brain_log.info("[LIFECYCLE] LEVEL UP: === POST-LEVEL STATE SNAPSHOT ===")
            self._runner._reader.log_health_check(post_state)
            # Fix mana max from fresh profile chain read
            if post_state.mana_max > 0:
                self._runner._reader._observed_mana_max = post_state.mana_max
                brain_log.info(
                    "[LIFECYCLE] LEVEL UP: mana max set to %d from profile chain", post_state.mana_max
                )
            brain_log.info(
                "[LIFECYCLE] LEVEL UP: level=%d HP=%d/%d Mana=%d/%d XP=%.1f%% buffs=%d casting_mode=%d",
                post_state.level,
                post_state.hp_current,
                post_state.hp_max,
                post_state.mana_current,
                post_state.mana_max,
                post_state.xp_pct * 100,
                len(post_state.buffs),
                post_state.casting_mode,
            )
        except (OSError, RuntimeError, ValueError) as e:
            brain_log.warning("[LIFECYCLE] LEVEL UP: post-level state read failed: %s", e)

    @staticmethod
    def _record_add(ctx: AgentContext, sp_name: str) -> None:
        """Mark an add detected and record social link if zone knowledge is available."""
        if ctx.zone.zone_knowledge and ctx.combat.pull_target_name:
            ctx.zone.zone_knowledge.record_social_add(ctx.combat.pull_target_name, sp_name)
        ctx.pet.has_add = True

    def _check_spawn_is_add(
        self,
        sp: SpawnData,
        state: GameState,
        ctx: AgentContext,
        pet_name: str,
        pet_x: float,
        pet_y: float,
    ) -> bool:
        """Check whether *sp* qualifies as an add via targeting or proximity.

        Returns True (and sets ctx.pet.has_add) when an add is confirmed.
        """
        # Path 1: npc targeting player or pet (catches full-HP adds)
        if sp.target_name and state.pos.dist_to(sp.pos) < 100:
            targeting_player = sp.target_name == state.name
            targeting_pet = pet_name and sp.target_name == pet_name
            if targeting_player or targeting_pet:
                who = "player" if targeting_player else "pet"
                brain_log.warning(
                    "[COMBAT] ADD detected (target_name): '%s' lv%d targeting %s at %.0fu",
                    sp.name,
                    sp.level,
                    who,
                    state.pos.dist_to(sp.pos),
                )
                self._record_add(ctx, sp.name)
                return True

        # Path 2: damaged NPC near pet (original detection)
        if sp.hp_current < sp.hp_max:
            d = sp.pos.dist_to(Point(pet_x, pet_y, 0.0))
            if d < 50:
                brain_log.info(
                    "[COMBAT] ADD detected (memory): '%s' damaged near pet (dist=%.0f)", sp.name, d
                )
                self._record_add(ctx, sp.name)
                return True

        return False

    def detect_adds(self, state: GameState, ctx: AgentContext) -> None:
        """Detect social extra_npcs from spawn list.

        Two detection paths:
          1. Damaged NPCs near pet (original -- catches npcs already in combat)
          2. Npcs targeting player or pet via target_name (catches full-HP
             adds that just threat'd -- previously in _should_combat)
        """
        if not ctx.in_active_combat:
            return
        if ctx.pet.has_add:
            return  # already detected, skip scan

        pull_id = ctx.combat.pull_target_id or 0
        pet_name = ctx.pet.name if ctx.pet.alive else ""

        # Locate pet for proximity check
        pet_x, pet_y = state.x, state.y  # fallback to player pos
        if ctx.pet.alive and ctx.pet.spawn_id:
            for sp in state.spawns:
                if sp.spawn_id == ctx.pet.spawn_id:
                    pet_x, pet_y = sp.x, sp.y
                    break

        from perception.combat_eval import get_zone_avoid_mobs

        zone_avoid = get_zone_avoid_mobs()

        for sp in state.spawns:
            if not sp.is_npc or sp.hp_current <= 0:
                continue
            if sp.spawn_id == pull_id:
                continue
            if is_pet(sp):
                continue
            # Skip zone-avoided npcs (e.g. will-o-wisp) -- don't retarget
            mob_base = normalize_mob_name(sp.name)
            if mob_base in zone_avoid:
                continue

            if self._check_spawn_is_add(sp, state, ctx, pet_name, pet_x, pet_y):
                return

    def _is_auto_engage_candidate(self, spawn: SpawnData, ctx: AgentContext, state: GameState) -> bool:
        """Evaluate whether a single spawn qualifies for auto-engagement.

        Side-effects when the spawn is relevant:
          - Sets ``ctx.threat.imminent_threat`` for RED or no-pet threats.
          - Sets ``ctx.combat.auto_engage_candidate`` for engageable NPCs.
          - Rate-limited logging of candidate detection.

        Returns True when the spawn was accepted as an auto-engage
        candidate (i.e. stored on ``ctx.combat``), False otherwise.
        """
        if not spawn.is_npc or spawn.hp_current <= 0:
            return False
        if spawn.target_name != state.name:
            return False
        d = state.pos.dist_to(spawn.pos)
        if d >= 60:
            return False

        tc = con_color(state.level, spawn.level)

        if tc == Con.RED:
            brain_log.warning(
                "[COMBAT] Pre-scan: RED npc '%s' lv%d targeting player -- flagging threat",
                spawn.name,
                spawn.level,
            )
            ctx.threat.imminent_threat = True
            ctx.threat.imminent_threat_con = Con.RED
            return False  # threat only, not an engage candidate

        if not ctx.pet.alive:
            brain_log.warning(
                "[COMBAT] Pre-scan: '%s' lv%d %s targeting player but NO PET -- flagging threat",
                spawn.name,
                spawn.level,
                tc,
            )
            ctx.threat.imminent_threat = True
            return False

        # Already found a candidate earlier in the loop -- skip.
        if ctx.combat.auto_engage_candidate is not None:
            return False

        now_ae = time.time()
        if now_ae - self._auto_engage_logged.get(spawn.spawn_id, 0.0) > 5.0:
            brain_log.warning(
                "[COMBAT] Pre-scan: '%s' lv%d %s targeting player at %.0fu -- auto-engage candidate",
                spawn.name,
                spawn.level,
                tc,
                d,
            )
            self._auto_engage_logged[spawn.spawn_id] = now_ae
        ctx.combat.auto_engage_candidate = (spawn.spawn_id, spawn.name, spawn.x, spawn.y, spawn.level)
        return True

    def scan_auto_engage(self, state: GameState, ctx: AgentContext) -> None:
        """Pre-rule scan: detect npcs targeting player, set threat flags.

        Runs every tick BEFORE brain.tick(). Replaces the detection logic
        that was previously embedded inside _should_combat's condition
        function, which caused ghost-state mutations when the condition
        ran for diagnostics but a higher-priority rule won.

        Sets:
          ctx.threat.imminent_threat -- for RED/no-pet threats (consumed by FLEE)
          ctx.combat.auto_engage_candidate -- for engageable npcs (consumed by IN_COMBAT)
        """
        # Clear previous tick's candidate (fresh scan each tick)
        ctx.combat.auto_engage_candidate = None

        # Skip auto-engage detection when already in combat
        if ctx.combat.engaged:
            self._prev_hp = state.hp_pct
            return

        # -- Scan 1: npc targeting player (target_name memory read) --
        for sp in state.spawns:
            self._is_auto_engage_candidate(sp, ctx, state)

        # -- Scan 2: HP-drop auto-engage (damaged NPC nearby when HP drops) --
        hp_dropped = state.hp_pct < self._prev_hp - 0.02
        if hp_dropped and ctx.combat.auto_engage_candidate is None:
            world = ctx.world
            if world:
                damaged = world.damaged_npcs_near(state.pos, 30)
                if damaged:
                    npc = damaged[0]
                    tc = con_color(state.level, npc.spawn.level)
                    if tc == Con.RED:
                        brain_log.warning(
                            "[COMBAT] Pre-scan: RED npc '%s' lv%d at %.0fu -- HP dropping, flagging threat",
                            npc.spawn.name,
                            npc.spawn.level,
                            npc.distance,
                        )
                        ctx.threat.imminent_threat = True
                    elif tc == Con.YELLOW and (not ctx.pet.alive or state.hp_pct < 0.50):
                        brain_log.warning(
                            "[COMBAT] Pre-scan: YELLOW npc '%s' lv%d -- "
                            "pet=%s HP=%.0f%% -- too risky, flagging threat",
                            npc.spawn.name,
                            npc.spawn.level,
                            "alive" if ctx.pet.alive else "dead",
                            state.hp_pct * 100,
                        )
                        ctx.threat.imminent_threat = True
                    elif not ctx.pet.alive:
                        brain_log.warning(
                            "[COMBAT] Pre-scan: taking damage from '%s' lv%d %s "
                            "but NO PET -- flagging threat",
                            npc.spawn.name,
                            npc.spawn.level,
                            tc,
                        )
                        ctx.threat.imminent_threat = True
                    else:
                        brain_log.warning(
                            "[COMBAT] Pre-scan: taking damage! '%s' lv%d %s at %.0fu -- "
                            "auto-engage candidate",
                            npc.spawn.name,
                            npc.spawn.level,
                            tc,
                            npc.distance,
                        )
                        ctx.combat.auto_engage_candidate = (
                            npc.spawn.spawn_id,
                            npc.spawn.name,
                            npc.spawn.x,
                            npc.spawn.y,
                            npc.spawn.level,
                        )

        self._prev_hp = state.hp_pct
