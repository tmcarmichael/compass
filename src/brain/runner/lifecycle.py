"""Brain lifecycle: process recovery, zone loading, death handling, startup.

Extracted from brain_runner.py to separate process lifecycle management
from brain loop orchestration.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

from core.exceptions import MemoryReadError
from core.features import flags
from core.types import PlanType
from motor.actions import (
    camp as do_camp,
)
from motor.actions import (
    mark_standing,
    release_all_keys,
    sit,
)
from util.event_schemas import PlayerDeathEvent
from util.log_tiers import EVENT
from util.structured_log import log_event

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.runner.loop import BrainRunner
    from perception.state import GameState

brain_log = logging.getLogger("compass.brain_loop")

# -- Domain-local constants --
ZONE_LOAD_TIMEOUT = 30.0  # max seconds to wait for zone to stabilize
ZONE_LOAD_POLL_INTERVAL = 0.2  # seconds between zone stability checks
ZONE_LOAD_STABLE_READS = 3  # consecutive good reads before zone is "stable"
NO_PROGRESS_TIMEOUT = 900.0  # 15 minutes with no defeats -> safety camp-out


class LifecycleHandler:
    """Manages process recovery, zone loading, death handling, and startup.

    Composed into BrainRunner -- not a subclass. The runner passes
    itself so the handler can access shared state.

    Args:
        runner: The parent BrainRunner instance.
    """

    def __init__(self, runner: BrainRunner) -> None:
        self._runner = runner
        # Zoning state machine fields (moved from BrainRunner)
        self._zoning: bool = False  # True while zone transition in progress
        self._zoning_start: float = 0.0  # monotonic time when zoning began
        self._zoning_prev_zone_id: int = 0  # zone_id at time of zoning detection

    def check_watchdog_restart(self) -> bool:
        """Check for an external restart signal and re-attach to the target process.

        Returns True if re-attachment succeeded (tick loop should continue).
        Returns False if no signal present.
        Raises RuntimeError if recovery fails and the agent should stop.
        """
        # Override in environment-specific subclasses to implement recovery.
        return False

    def enter_zoning(self, source: str) -> None:
        """Immediately suppress all brain activity for a zone transition.

        Called when zoning is first detected (from log message or game_mode
        change). Releases held keys, deactivates the active routine, and
        prevents motor spam during the client freeze.

        Args:
            source: What triggered detection ("log", "game_mode", "zone_id").
        """
        runner = self._runner
        brain_log.info("[TRAVEL] ZONING: detected (source=%s) -- suppressing brain", source)

        # Release all held movement/modifier keys immediately
        try:
            release_all_keys()
        except (OSError, RuntimeError) as e:
            brain_log.warning("[TRAVEL] ZONING: release_all_keys failed: %s", e)

        # Deactivate the current routine cleanly
        brain = runner._brain
        if brain._active is not None:
            brain_log.info("[TRAVEL] ZONING: deactivating routine %s", brain._active_name)
            try:
                # Read one last state for exit() -- may be stale but better
                # than no state
                try:
                    st = runner._reader.read_state(include_spawns=False)
                except (
                    MemoryReadError,
                    OSError,
                    RuntimeError,
                ):
                    st = None
                if st is not None:
                    brain._active.exit(st)
                else:
                    brain_log.warning("[TRAVEL] ZONING: no state for routine exit")
            except (OSError, RuntimeError, AttributeError) as e:
                brain_log.warning("[TRAVEL] ZONING: routine exit failed: %s", e)
            brain._active = None
            brain._active_name = ""
            brain._active_start_time = 0.0

    def post_zone_recovery(self, ctx: AgentContext) -> None:
        """Reset state after a zone transition completes.

        Called after wait_for_zone_load() succeeds. Clears stale combat
        and engagement state, resets motor tracking, and invalidates the
        profile chain cache (may shift across zones).
        """
        brain_log.info("[TRAVEL] ZONING: post-zone recovery -- clearing stale state")

        # Release keys again (belt and suspenders)
        try:
            release_all_keys()
        except (OSError, RuntimeError) as e:
            brain_log.debug("[LIFECYCLE] Post-zone key release failed: %s", e)

        # Reset motor sit/stand tracking (player starts standing after zone)
        mark_standing()

        # Clear stale combat/engagement state
        ctx.clear_engagement()
        ctx.combat.pull_target_id = None
        ctx.combat.pull_target_name = ""
        ctx.pet.has_add = False
        ctx.threat.imminent_threat = False

        # Invalidate profile chain cache -- layout may differ across zones
        reader = self._runner._reader
        reader._profile_base_cache = None
        reader._profile_chain_failed = False

        # Load terrain heightmap for A* pathfinding and obstacle avoidance
        self._load_zone_terrain(ctx)

        brain_log.info("[TRAVEL] ZONING: recovery complete -- brain resuming")

    def _load_zone_terrain(self, ctx: AgentContext) -> None:
        """Load terrain heightmap into the movement controller.

        Called once per zone load. If no terrain cache exists, logs an
        informative message with the rebuild command (no spam).
        """
        from nav.movement import set_terrain
        from nav.terrain.heightmap import ZoneTerrain

        zone_name = ctx.zone.zone_config.get("zone", {}).get("short_name", "")
        if not zone_name:
            brain_log.debug("[TRAVEL] TERRAIN: no zone name in config, skipping")
            set_terrain(None)
            return

        cache_dir = Path(__file__).parents[3] / "data" / "terrain"
        cache_path = cache_dir / f"{zone_name}.terrain"

        terrain: ZoneTerrain | None
        _zt = ZoneTerrain()
        if _zt.load_cache(cache_path):
            terrain = _zt
            set_terrain(terrain)
            # Invalidate walk bitfield when obstacle_avoidance flag toggles
            from core.features import flags as _flags

            _flags.on_change("obstacle_avoidance", lambda _v: _zt.invalidate_walk_bits())
            brain_log.info(
                "[TRAVEL] TERRAIN: loaded %s (%s, %d obstacles)",
                zone_name,
                _zt.stats.get("grid", "?"),
                _zt.stats.get("obstacle", 0),
            )
        else:
            terrain = None
            set_terrain(None)
            brain_log.info("[TRAVEL] TERRAIN: no cache for %s", zone_name)

        # Sync terrain into session if available
        if hasattr(ctx, "session") and ctx.session is not None:
            ctx.session.terrain = terrain

    def wait_for_zone_load(self, new_zone_id: int) -> bool:
        """Poll until zone is fully loaded. Returns True if zone loaded OK.

        Checks: zone_id stable across 3 consecutive reads, game_mode is
        in_game (5), and spawn list is populated. Times out after 30s.
        """
        poll_interval = ZONE_LOAD_POLL_INTERVAL
        timeout = ZONE_LOAD_TIMEOUT
        stable_required = ZONE_LOAD_STABLE_READS
        stable_count = 0
        start = time.time()

        brain_log.info("[TRAVEL] ZONE LOAD: waiting for zone %d to stabilize...", new_zone_id)

        while time.time() - start < timeout:
            if self._runner._stop_event.is_set():
                return False
            time.sleep(poll_interval)
            try:
                st = self._runner._reader.read_state(include_spawns=True)
            except (
                MemoryReadError,
                OSError,
                RuntimeError,
            ):
                stable_count = 0
                continue

            # Zone ID must match and game must be in-world
            if st.zone_id == new_zone_id and st.is_in_game:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= stable_required and st.spawns:
                elapsed = time.time() - start
                brain_log.info(
                    "[TRAVEL] ZONE LOAD: zone %d ready in %.1fs (%d spawns)",
                    new_zone_id,
                    elapsed,
                    len(st.spawns),
                )
                return True

        elapsed = time.time() - start
        brain_log.warning(
            "[TRAVEL] ZONE LOAD: timeout after %.1fs waiting for zone %d (stable=%d, spawns=%d)",
            elapsed,
            new_zone_id,
            stable_count,
            len(st.spawns) if "st" in locals() else 0,
        )
        return False

    def handle_death(self, ctx: AgentContext, source: str) -> bool:
        """Handle player death detection. Returns True if agent should stop.

        Centralizes death handling for log-based, HP-based, and body_state
        detection paths so they share the same logic.
        """
        if ctx.player.dead:
            return False  # already handled
        ctx.player.dead = True
        ctx.player.deaths += 1
        recover = flags.should_recover_death(ctx.player.deaths)
        routine_name = self._runner._brain._active_name if self._runner._brain else ""
        log_event(
            brain_log,
            "player_death",
            f"[LIFECYCLE] DEATH: source={source} routine={routine_name} deaths={ctx.player.deaths} recover={recover}",
            level=logging.CRITICAL,
            **PlayerDeathEvent(
                source=source,
                routine=routine_name,
                pos_x=round(ctx.player.last_known_x),
                pos_y=round(ctx.player.last_known_y),
                pos_z=round(ctx.player.last_known_z),
                deaths=ctx.player.deaths,
                recover=recover,
            ),
        )
        # Capture forensics snapshot BEFORE flushing (flush clears the buffer)
        pre_death_snapshot = None
        if ctx.diag.forensics:
            pre_death_snapshot = ctx.diag.forensics.snapshot()
            ctx.diag.forensics.flush("death")
        # Emit composite incident report using the pre-flush snapshot
        if ctx.diag.incident_reporter:
            try:
                last_state = self._runner._reader.read_state(include_spawns=True)
                ctx.diag.incident_reporter.report_death(last_state, ctx, source, buffer=pre_death_snapshot)
            except (MemoryReadError, OSError, RuntimeError, AttributeError, TypeError, IndexError) as e:
                brain_log.debug("[LIFECYCLE] Incident reporter failed: %s", e)
        if not recover:
            # Always sit + camp on death (regardless of detection source).
            # Wait for respawn at bind point before sending commands.
            brain_log.log(EVENT, "[LIFECYCLE] Death (source=%s): waiting 10s for respawn...", source)
            time.sleep(10.0)
            brain_log.log(EVENT, "[LIFECYCLE] Death: sitting then camping out")
            try:
                sit()
                time.sleep(2.0)
                do_camp()
                brain_log.log(EVENT, "[LIFECYCLE] Death: /camp sent")
            except (OSError, RuntimeError) as e:
                brain_log.warning("[LIFECYCLE] Death: camp failed: %s", e)
            self._runner._stop_event.set()
            return True
        return False

    def check_no_progress_safety(self, ctx: AgentContext) -> bool:
        """Safety net: camp out if no defeats for 15 minutes.

        Catches all stuck scenarios: failed death detection, stuck pathing,
        empty zones, broken state. Called every 30s from periodic snapshot.

        Returns True if safety camp was triggered (caller should stop).
        """
        session_age = time.time() - ctx.metrics.session_start
        if session_age < NO_PROGRESS_TIMEOUT:
            return False

        defeat_age = ctx.defeat_tracker.last_kill_age()
        if defeat_age < NO_PROGRESS_TIMEOUT:
            return False

        # Exempt plans where no defeats are expected
        if ctx.plan.active in (PlanType.TRAVEL, PlanType.NEEDS_MEMORIZE):
            brain_log.debug(
                "[LIFECYCLE] Safety check: no defeats for %.0fs but plan=%s -- exempt",
                defeat_age,
                ctx.plan.active,
            )
            return False

        # Exempt if currently in combat (mid-fight, defeats imminent)
        if ctx.combat.engaged:
            brain_log.debug(
                "[LIFECYCLE] Safety check: no defeats for %.0fs but in combat -- exempt", defeat_age
            )
            return False

        # No progress for 15+ minutes -- safety camp-out
        brain_log.error(
            "[LIFECYCLE] SAFETY: no defeats for %.0fs (%.1f min) -- camping out to prevent AFK death loop",
            defeat_age,
            defeat_age / 60.0,
        )
        try:
            sit()
            time.sleep(2.0)
            do_camp()
            brain_log.info("[LIFECYCLE] SAFETY: /camp sent")
        except (OSError, RuntimeError) as e:
            brain_log.warning("[LIFECYCLE] SAFETY: camp failed: %s", e)
        self._runner._stop_event.set()
        return True

    def startup_warmup(self, ctx: AgentContext) -> None:
        """Clear stale state, verify perception reads, set baselines."""
        warmup_time = random.uniform(2.0, 4.0)
        brain_log.info("[LIFECYCLE] Startup warmup: %.1fs (checking pet, state...)", warmup_time)

        # Helper: bail early if stop was requested during warmup.
        def _stopped() -> bool:
            stopped: bool = self._runner._stop_event.is_set()
            return stopped

        # Interruptible warmup sleep -- check stop_event every 100ms
        # instead of one long blocking sleep.
        deadline = time.monotonic() + warmup_time
        while time.monotonic() < deadline:
            if _stopped():
                brain_log.info("[LIFECYCLE] Startup warmup interrupted by stop_event")
                return
            time.sleep(0.1)
        self._runner._prev_level = 0
        try:
            state = self._runner._reader.read_state(include_spawns=True)
            ctx.update_pet_status(state)
            self._runner._prev_level = state.level
            if state.mana_current == 0:
                brain_log.info("[LIFECYCLE] Warmup: mana reads 0, retrying CHARINFO...")
                for _retry in range(5):
                    if _stopped():
                        brain_log.info("[LIFECYCLE] Warmup mana retry interrupted by stop_event")
                        return
                    time.sleep(1.0)
                    state = self._runner._reader.read_state(include_spawns=True)
                    if state.mana_current > 0:
                        brain_log.info(
                            "[LIFECYCLE] Warmup: mana now reads %d (retry %d)", state.mana_current, _retry + 1
                        )
                        break
                else:
                    brain_log.warning("[LIFECYCLE] Warmup: mana still 0 after retries")
            if state.weight > 0 and ctx.inventory.weight_baseline == 0:
                ctx.inventory.weight_baseline = state.weight
            # Seed XP baseline so track_xp detects gains from first defeat
            if state.xp_pct_raw > 0:
                ctx.metrics.xp_last_raw = state.xp_pct_raw
                ctx.metrics.record_xp_sample(time.time(), state.xp_pct_raw)
            brain_log.info(
                "[LIFECYCLE] Warmup complete: HP=%.0f%% Mana=%d/%d Pet=%s Pos=(%.0f,%.0f) Weight=%d",
                state.hp_pct * 100,
                state.mana_current,
                self._runner._reader._observed_mana_max,
                f"alive (id={ctx.pet.spawn_id})" if ctx.pet.alive else "NONE",
                state.x,
                state.y,
                state.weight,
            )
            self._runner._reader.log_health_check(state)
            self._runner._reader.validate_structs()
        except (
            OSError,
            RuntimeError,
        ):
            pass

        # Load terrain heightmap for A* pathfinding and obstacle avoidance.
        # Also runs on zone transitions via post_zone_recovery().
        self._load_zone_terrain(ctx)

        # Initial inventory scan -- populate inventory state before first tick.
        try:
            items = self._runner._reader.read_inventory()
            if hasattr(ctx, "inventory") and items:
                ctx.inventory.update_items(items, time.time())
        except (MemoryReadError, OSError, RuntimeError) as e:
            brain_log.debug("[LIFECYCLE] Startup inventory scan failed: %s", e)

    # -- Zoning / not-in-game checks (moved from BrainRunner) ----------------

    def check_not_in_game(self, state: GameState) -> str | None:
        """Handle game-mode transitions: char select stop, zoning suppression.

        Returns "break" to stop the loop, "continue" to skip this tick, or
        None to proceed normally.
        """
        runner = self._runner
        if state.is_at_char_select:
            brain_log.warning("[LIFECYCLE] Game mode: char select -- stopping agent")
            runner._stop_event.set()
            return "break"
        # Zoning: release keys and deactivate routine on first detection
        if not self._zoning:
            self._zoning = True
            self._zoning_start = time.monotonic()
            self._zoning_prev_zone_id = runner._prev_zone_id
            self.enter_zoning("game_mode")
        runner._last_heartbeat = time.monotonic()
        time.sleep(1.0)
        return "continue"

    def check_zoning_recovery(self, state: GameState, ctx: AgentContext) -> str | None:
        """Drive zone transition state machine once we are back in-game.

        Covers three zone-change detectors (zoning-state recovery,
        engine_zone_id earliest detection, memory zone_id silent change)
        and the log-based early-detection path.

        Returns "break" / "continue" / None for loop control.
        """
        runner = self._runner

        # -- Zoning recovery: waiting for zone_id to stabilize after transition
        if self._zoning:
            zoning_age = time.monotonic() - self._zoning_start
            zone_changed = state.zone_id != self._zoning_prev_zone_id and 1 <= state.zone_id <= 500
            if zone_changed:
                brain_log.log(
                    EVENT,
                    "[TRAVEL] ZONING: zone_id changed %d -> %d -- waiting for stabilization",
                    self._zoning_prev_zone_id,
                    state.zone_id,
                )
                self.wait_for_zone_load(state.zone_id)
                self.post_zone_recovery(ctx)
                self._zoning = False
                # Re-read state after zone load for fresh data
                try:
                    state = runner._reader.read_state(include_spawns=True)
                except (
                    MemoryReadError,
                    OSError,
                    RuntimeError,
                ):
                    pass
                runner._prev_zone_id = state.zone_id
                runner._last_heartbeat = time.monotonic()
                return "continue"
            if zoning_age > 15.0:
                # Timeout: zone_id never changed -- false alarm or
                # same-zone reload (e.g. /rewind)
                brain_log.warning(
                    "[TRAVEL] ZONING: timeout after %.0fs with no zone_id change -- clearing zoning flag",
                    zoning_age,
                )
                self._zoning = False
            else:
                # Still waiting for zone_id to change -- suppress brain
                runner._last_heartbeat = time.monotonic()
                return "continue"

        # -- Game engine vbase zone_id: earliest detection (350ms before
        # base_ptr goes null). Fires while game_mode is still 5.
        if (
            not self._zoning
            and runner._prev_zone_id > 0
            and state.engine_zone_id > 0
            and state.engine_zone_id != runner._prev_zone_id
            and 1 <= state.engine_zone_id <= 500
        ):
            brain_log.log(
                EVENT,
                "[TRAVEL] ZONING: engine_zone_id changed %d -> %d -- earliest detection, suppressing brain",
                runner._prev_zone_id,
                state.engine_zone_id,
            )
            self._zoning = True
            self._zoning_start = time.monotonic()
            self._zoning_prev_zone_id = runner._prev_zone_id
            self.enter_zoning("engine_zone_id")
            runner._last_heartbeat = time.monotonic()
            return "continue"

        # -- Track zone changes from memory (no game_mode transition)
        # This catches zones where game_mode stays 5 throughout
        if runner._prev_zone_id > 0:
            if state.zone_id != runner._prev_zone_id and 1 <= state.zone_id <= 500:
                brain_log.log(
                    EVENT,
                    "[TRAVEL] ZONE CHANGE (memory): zone_id %d -> %d",
                    runner._prev_zone_id,
                    state.zone_id,
                )
                self.enter_zoning("zone_id")
                self.wait_for_zone_load(state.zone_id)
                self.post_zone_recovery(ctx)
                # Re-read state for fresh data
                try:
                    state = runner._reader.read_state(include_spawns=True)
                except (
                    MemoryReadError,
                    OSError,
                    RuntimeError,
                ):
                    pass
                runner._prev_zone_id = state.zone_id
                runner._last_heartbeat = time.monotonic()
                return "continue"
        runner._prev_zone_id = state.zone_id

        return None
