"""Brain loop runner -- no GUI dependency.

The BrainRunner owns the main tick loop that reads game state,
evaluates brain rules, and drives routines. Delegates to focused
handler classes for lifecycle (process recovery, death, zone load),
per-tick events (level-up, adds, auto-engage), world state updates,
session reporting, and XP tracking.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from core import __version__

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.decision import Brain
    from perception.reader import MemoryReader
    from perception.state import GameState
from brain.goap import (
    GOAPPlanner,
    build_action_set,
    build_goal_set,
)
from brain.goap.spawn_predictor import SpawnPredictor
from brain.learning.scorecard import (
    apply_tuning,
    compute_scorecard,
    load_tuning,
)
from brain.learning.session_memory import SessionMemory, SessionRecord
from brain.runner.learning import LearningTickHandler
from brain.runner.lifecycle import LifecycleHandler
from brain.runner.tick_handlers import TickHandlers
from brain.scoring.weight_learner import (
    GradientTuner,
    load_learned_weights,
    save_learned_weights,
)
from brain.world.health import HealthMonitor
from brain.world.tracker import StateChangeTracker
from brain.world.updater import WorldStateUpdater
from core.exceptions import MemoryReadError
from core.features import flags
from nav.movement import clear_movement_cancel
from perception.reader import MemoryReader
from util.clock import TickClock
from util.log_tiers import EVENT
from util.session_reporter import SessionReporter
from util.thread_guard import set_brain_thread

brain_log = logging.getLogger("compass.brain_loop")


class TickSignal(StrEnum):
    """Typed sentinel for tick-helper loop control (replaces raw strings)."""

    BREAK = "break"
    CONTINUE = "continue"
    PROCEED = "proceed"


def _save_goap_costs(planner: GOAPPlanner, zone: str) -> None:
    """Persist GOAP cost corrections to disk.

    Module-level to keep _run_cleanup complexity low.
    """
    costs_path = Path("data/memory") / f"{zone}_goap_costs.json"
    costs_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(costs_path, "w") as f:
            json.dump({"v": 1, "corrections": planner.cost_corrections}, f)
    except OSError as e:
        brain_log.debug("[LIFECYCLE] GOAP cost save failed: %s", e)


class BrainRunner:
    """Runs the brain loop in a background thread. No GUI dependency."""

    def __init__(
        self,
        reader: MemoryReader,
        ctx: AgentContext,
        brain: Brain,
        stop_event: threading.Event,
        config: dict,
        current_zone: str = "",
        log_path: str = "",
        session_id: str = "",
    ) -> None:
        self._reader = reader
        self._ctx = ctx
        self._brain = brain
        self._stop_event = stop_event
        self._config = config
        self._current_zone = current_zone
        self._log_path = log_path
        self._session_id = session_id
        self._paused = False
        self._prev_zone_id = 0
        self._prev_level = 0
        # Callback for status updates: fn(routine_name: str, defeats: int)
        self.on_display_update: Callable[[str, int], object] | None = None
        # Watchdog: heartbeat + crash tracking
        self._last_heartbeat: float = 0.0
        self._last_exception: Exception | None = None
        self._death_time: float = 0.0
        # Per-tick crash restart (non-fatal errors resume next tick)
        self._crash_count: int = 0
        self._crash_window_start: float = 0.0
        # Composed handlers (each owns one concern)
        self._reporter = SessionReporter(self)
        self._world_updater = WorldStateUpdater(self)
        self._lifecycle = LifecycleHandler(self)
        self._tick_handlers = TickHandlers(self)
        self._learning = LearningTickHandler(self)

    @property
    def seconds_since_heartbeat(self) -> float:
        if self._last_heartbeat == 0.0:
            return float("inf")
        return time.monotonic() - self._last_heartbeat

    @property
    def last_exception(self) -> Exception | None:
        return self._last_exception

    @property
    def brain_healthy(self) -> bool:
        """True if brain thread is ticking normally."""
        return self.seconds_since_heartbeat < 10.0

    @property
    def paused(self) -> bool:
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        if value and not self._paused:
            # Entering pause -- release any held movement/modifier keys
            try:
                from motor.actions import move_backward_stop, move_forward_stop

                move_forward_stop()
                move_backward_stop()
            except (ImportError, OSError, RuntimeError) as e:
                brain_log.debug("[LIFECYCLE] Pause key release failed: %s", e)
        self._paused = value

    # -- Delegate accessors for WorldStateUpdater / SessionReporter --------

    def _handle_death(self, ctx: AgentContext, source: str) -> bool:
        """Delegate to lifecycle handler."""
        result: bool = self._lifecycle.handle_death(ctx, source)
        return result

    def _write_session_report(self, ctx: AgentContext, session_dir: str) -> None:
        """Write machine-readable session report JSON at shutdown."""
        self._reporter.write_session_report(ctx, session_dir)

    def _log_session_ready(self, ctx: AgentContext) -> None:
        """Log one-time health report of all subsystems after init.

        Enables 'one grep on SESSION READY' diagnosis of wiring failures.
        Called once after startup_warmup, before the main loop.
        """
        from core.features import flags
        from nav.movement import get_terrain

        terrain = get_terrain()
        terrain_status = "None (no cache)"
        if terrain:
            s = terrain.stats
            terrain_status = f"{s.get('grid', '?')}, {s.get('obstacle', 0)} obstacles"

        zone_name = ctx.zone.zone_config.get("zone", {}).get("short_name", "?")
        camp_name = ctx.zone.active_camp_name or "none"

        flag_summary = []
        for name in (
            "obstacle_avoidance",
            "loot_mode",
            "combat_casting",
            "flee",
            "rest",
            "wander",
            "death_recovery",
        ):
            val = getattr(flags, name, "?")
            flag_summary.append(f"{name}={val}")

        brain_log.log(
            EVENT,
            "[LIFECYCLE] SESSION READY: zone=%s camp=%s\n  terrain: %s\n  flags: %s",
            zone_name,
            camp_name,
            terrain_status,
            ", ".join(flag_summary),
        )

    # -- Main loop ---------------------------------------------------------

    def _init_monitoring(self) -> None:
        """Create health monitoring, state tracking, and anomaly detection."""
        from brain.world.anomaly import AnomalyDetector

        self._health_monitor = HealthMonitor()
        self._state_tracker = StateChangeTracker()
        self._anomaly_detector = AnomalyDetector(ctx=self._ctx)

    def _init_learning(self, ctx: AgentContext) -> None:
        """Load persisted tuning and initialize gradient weight learner."""
        self._tuning = load_tuning(self._current_zone)
        apply_tuning(self._tuning, ctx)

        self._gradient_tuner: GradientTuner | None = None
        if hasattr(ctx, "world") and ctx.world is not None and hasattr(ctx.world, "_weights"):
            self._gradient_tuner = GradientTuner(ctx.world._weights)
            saved_weights, saved_lr = load_learned_weights(self._current_zone)
            if saved_weights:
                applied = self._gradient_tuner.load_learned_weights(saved_weights)
                brain_log.info(
                    "[LIFECYCLE] Loaded %d learned scoring weights for %s", applied, self._current_zone
                )
            if saved_lr:
                self._gradient_tuner.load_learning_rates(saved_lr)

    def _init_goap(self, ctx: AgentContext) -> None:
        """Initialize GOAP planner, spawn predictor, and load persisted costs."""
        self._goap_planner: GOAPPlanner | None = None
        self._spawn_predictor: SpawnPredictor | None = None
        if not flags.goap_planning:
            return
        self._goap_planner = GOAPPlanner(goals=build_goal_set(), actions=build_action_set())
        self._spawn_predictor = SpawnPredictor()
        ctx.spawn_predictor = self._spawn_predictor
        brain_log.info(
            "[LIFECYCLE] GOAP planner initialized with %d goals, %d actions",
            len(build_goal_set()),
            len(build_action_set()),
        )
        if ctx.spatial_memory:
            self._spawn_predictor.update_from_memory(ctx.spatial_memory)
        goap_costs_path = Path("data/memory") / f"{self._current_zone}_goap_costs.json"
        if goap_costs_path.exists():
            try:
                with open(goap_costs_path) as f:
                    cost_data = json.load(f)
                self._goap_planner.load_cost_corrections(cost_data.get("corrections", {}))
                brain_log.info("[LIFECYCLE] Loaded GOAP cost corrections for %s", self._current_zone)
            except (OSError, json.JSONDecodeError, KeyError) as e:
                brain_log.debug("[LIFECYCLE] GOAP cost load failed: %s", e)

    def _run_setup(self) -> None:
        """Initialize all loop-lifetime state before the main tick loop."""
        ctx = self._ctx
        ctx.stop_event = self._stop_event

        brain_log.log(EVENT, "[LIFECYCLE] Brain loop started  -  v%s", __version__)
        brain_log.info("[LIFECYCLE] Feature flags: %s", flags.as_dict())

        self._init_monitoring()

        tick_rate = self._config["general"].get("tick_rate_hz", 10)
        self._clock = TickClock(tick_rate)
        self._next_snapshot = time.time() + 30.0
        self._next_tuning_eval = time.time() + 1800.0

        self._init_learning(ctx)
        self._init_goap(ctx)
        self._next_spawn_update = time.time() + 60.0

        self._session_memory = SessionMemory(zone=self._current_zone)
        brain_log.info("[LIFECYCLE] %s", self._session_memory.startup_summary())

        self._session_dir = str(Path(__file__).parent.parent.parent / "logs" / "sessions")

        self._lifecycle.startup_warmup(ctx)
        self._log_session_ready(ctx)

    def _run_cleanup(self, state: GameState | None) -> None:
        """Post-loop cleanup: save state, close resources, write session report.

        Called from run()'s finally block -- always executes even on crash.
        Uses ExitStack so each callback runs independently.
        """
        ctx = self._ctx

        # Shutdown active routine + release held keys FIRST to stop
        # ghost input (movement keys, /pet attack spam) immediately.
        try:
            if state is not None:
                self._brain.shutdown(state)
            else:
                # No state read yet -- just release keys
                from motor.actions import release_all_keys

                release_all_keys()
        except Exception as e:
            brain_log.warning("[LIFECYCLE] Brain shutdown failed: %s", e)
            # Fallback: try releasing keys even if routine exit failed
            try:
                from motor.actions import release_all_keys

                release_all_keys()
            except Exception as e2:
                brain_log.debug("[LIFECYCLE] Fallback key release failed: %s", e2)

        elapsed = time.time() - ctx.metrics.session_start
        brain_log.log(
            EVENT,
            "[LIFECYCLE] Brain loop cleanup: ran %.1f min, %d npcs, %d deaths, %d flees",
            elapsed / 60,
            ctx.defeat_tracker.defeats,
            ctx.player.deaths,
            ctx.metrics.flee_count,
        )

        # ExitStack ensures each cleanup runs even if a prior one throws
        with ExitStack() as cleanup:
            cleanup.callback(lambda: brain_log.log(EVENT, "[LIFECYCLE] Brain loop ended"))
            if ctx.zone.zone_knowledge and hasattr(ctx.zone.zone_knowledge, "save"):
                cleanup.callback(ctx.zone.zone_knowledge.save)
            if ctx.fight_history:
                fh = ctx.fight_history
                cleanup.callback(lambda: brain_log.info(fh.summary()))
                cleanup.callback(fh.save)
            if self._gradient_tuner is not None:
                tuner = self._gradient_tuner
                zone = self._current_zone
                cleanup.callback(
                    lambda: save_learned_weights(
                        tuner.get_weight_snapshot(), zone, learning_rates=tuner.get_learning_rates()
                    )
                )
            if self._goap_planner is not None:
                planner = self._goap_planner
                zone = self._current_zone
                cleanup.callback(lambda: brain_log.info("[LIFECYCLE] %s", planner.stats_summary()))
                cleanup.callback(lambda: _save_goap_costs(planner, zone))
            if ctx.spatial_memory:
                cleanup.callback(ctx.spatial_memory.save)
            cleanup.callback(lambda: brain_log.info(ctx.session_summary()))
            # Session memory: record this session's performance
            sm = self._session_memory
            cleanup.callback(lambda: self._record_session_to_memory(ctx, sm))
            # Session report JSON
            cleanup.callback(lambda: self._write_session_report(ctx, self._session_dir))

    @staticmethod
    def _record_session_to_memory(ctx: AgentContext, sm: SessionMemory) -> None:
        """Build a SessionRecord from context and persist to session memory."""
        try:
            elapsed = time.time() - ctx.metrics.session_start
            hours = max(elapsed / 3600, 0.01)
            scores = compute_scorecard(ctx)
            goap_pct = 0.0
            goap_err = 0.0
            # (GOAP stats would come from planner if available)
            sm.record(
                SessionRecord(
                    timestamp=time.time(),
                    duration_minutes=round(elapsed / 60, 1),
                    defeats_per_hour=round(ctx.defeat_tracker.defeats / hours, 1),
                    deaths=ctx.player.deaths,
                    flees=ctx.metrics.flee_count,
                    survival_score=int(scores.get("survival", 0)),
                    overall_score=int(scores.get("overall", 0)),
                    overall_grade=str(scores.get("grade", "F")),
                    goap_completion_pct=goap_pct,
                    goap_avg_cost_error=goap_err,
                    zone=sm._zone,
                )
            )
        except (AttributeError, TypeError, ValueError, OSError) as e:
            brain_log.warning("[LIFECYCLE] Session memory record failed: %s", e)

    # -- Tick helpers (each owns one concern, return TickSignal for loop control)
    # Signal values: BREAK = exit loop, CONTINUE = skip to next tick,
    # PROCEED = proceed to next helper in the tick sequence.

    def _tick_pre_state(self, ctx: AgentContext) -> tuple[TickSignal, GameState | None]:
        """Watchdog check, pause gate, and memory read.

        Returns (signal, state):
          - (BREAK, None)    loop must stop (watchdog recovery failed)
          - (CONTINUE, None) skip this tick (watchdog recovered / paused /
                              memory read failed)
          - (PROCEED, state) proceed with the fresh state snapshot
        """
        # Check for watchdog restart flag (crash/freeze/disconnect)
        try:
            if self._lifecycle.check_watchdog_restart():
                # Recovery succeeded: skip this tick, read fresh next time
                return TickSignal.CONTINUE, None
        except RuntimeError:
            # Recovery failed, stop_event already set
            return TickSignal.BREAK, None

        if self._paused:
            return TickSignal.CONTINUE, None

        clear_movement_cancel()

        try:
            state = self._reader.read_state(include_spawns=True)
        except (MemoryReadError, OSError, RuntimeError) as e:
            brain_log.warning("[PERCEPTION] Memory read failed: %s", e)
            time.sleep(1.0)
            return TickSignal.CONTINUE, None

        return TickSignal.PROCEED, state

    def _tick_periodic_snapshot(self, state: GameState, ctx: AgentContext, now: float) -> bool:
        """Run the every-30s snapshot, anomaly check, config hot-reload, and
        no-progress safety check.

        Returns True if the loop should break (no-progress safety triggered).
        """
        if now <= self._next_snapshot:
            return False
        self._reporter.periodic_snapshot(state, ctx, now, self._health_monitor)
        ctx.metrics.trim_lists()
        if ctx.spatial_memory:
            ctx.spatial_memory.trim_lists()
        # Phase detection (grinding, resting, incident, idle)
        if ctx.diag.phase_detector:
            try:
                ctx.diag.phase_detector.check(state, ctx, now)
            except (AttributeError, TypeError) as e:
                brain_log.debug("[LIFECYCLE] Phase detector error: %s", e)
        # Real-time anomaly detection + self-healing
        try:
            self._anomaly_detector.check(state)
        except (AttributeError, TypeError, KeyError, ValueError, ZeroDivisionError) as e:
            brain_log.debug("[LIFECYCLE] Anomaly detector error: %s", e)
        # No-progress safety: camp out if no defeats for 10 minutes
        if self._lifecycle.check_no_progress_safety(ctx):
            return True
        self._next_snapshot = now + 30.0
        return False

    def _tick_brain(self, state: GameState, ctx: AgentContext) -> TickSignal:
        """Run one brain tick with crash-resilient error handling.

        Returns BREAK if crash rate exceeded and loop must stop,
        CONTINUE if a recoverable crash occurred and tick should restart,
        or PROCEED on success.
        """
        try:
            self._brain.tick(state)
        except (
            TypeError,
            ValueError,
            AttributeError,
            NameError,
            KeyError,
            IndexError,
            ZeroDivisionError,
            AssertionError,
            RuntimeError,
        ) as tick_err:
            self._last_exception = tick_err
            self._crash_count += 1
            brain_log.error(
                "[LIFECYCLE] Brain tick crashed (#%d): %s", self._crash_count, tick_err, exc_info=True
            )
            if ctx.diag.forensics:
                ctx.diag.forensics.flush("tick_crash")
            # Safety: release held keys and exit active routine
            try:
                from motor.actions import move_backward_stop, move_forward_stop

                move_forward_stop()
                move_backward_stop()
                if self._brain._active:
                    self._brain.shutdown(state)
            except Exception as e:
                brain_log.debug("[LIFECYCLE] Crash recovery key release failed: %s", e)
            # Rate limit: >3 crashes in 60s = give up
            if time.monotonic() - self._crash_window_start > 60.0:
                self._crash_window_start = time.monotonic()
                self._crash_count = 1
            if self._crash_count > 3:
                brain_log.error(
                    "[LIFECYCLE] Brain tick crashed %d times in 60s -- stopping", self._crash_count
                )
                self._stop_event.set()
                return TickSignal.BREAK
            brain_log.warning("[LIFECYCLE] Brain tick: recovering from crash (resuming next tick)")
            time.sleep(1.0)
            return TickSignal.CONTINUE
        return TickSignal.PROCEED

    def _tick_record_diag(self, state: GameState, ctx: AgentContext) -> None:
        """Record decision receipt, forensics ring buffer, metrics, invariants,
        and push display update. All zero-disk-I/O per-tick observability."""
        if ctx.diag.decision_throttle:
            ctx.diag.decision_throttle.record(
                tick_id=self._clock.tick_count,
                state=state,
                rule_eval=self._brain.last_rule_eval,
                rule_scores=self._brain.rule_scores,
                selected=self._brain._last_matched_rule,
                active=self._brain._active_name,
                locked=self._brain._active.locked if self._brain._active else False,
                tick_ms=self._brain.tick_total_ms,
                routine_ms=self._brain.routine_tick_ms,
                engaged=ctx.combat.engaged,
                pet_alive=ctx.pet.alive,
            )
        if ctx.diag.forensics:
            ctx.diag.forensics.record_tick(
                tick_id=self._clock.tick_count,
                state=state,
                active_routine=self._brain._active_name,
                engaged=ctx.combat.engaged,
            )
        if ctx.diag.metrics:
            ctx.diag.metrics.record_tick(self._brain.tick_total_ms)
        if ctx.diag.invariants:
            ctx.diag.invariants.check(self._clock.tick_count, state, ctx)
        if self.on_display_update:
            self.on_display_update(self._brain._active_name or "", ctx.defeat_tracker.defeats)

    def _tick_one(self, ctx: AgentContext) -> tuple[TickSignal, GameState | None]:
        """Execute one iteration of the brain loop.

        Returns (signal, state) where signal tells the caller whether
        to break, continue, or proceed to the next tick.
        """
        self._clock.wait_for_next_tick()
        self._ctx.diag.current_tick_id = self._clock.tick_count
        if self._ctx.diag.structured_handler:
            self._ctx.diag.structured_handler.set_tick_id(self._clock.tick_count)

        pre_signal, state_or_none = self._tick_pre_state(ctx)
        if pre_signal is TickSignal.BREAK:
            return TickSignal.BREAK, None
        if pre_signal is TickSignal.CONTINUE:
            return TickSignal.CONTINUE, None
        assert state_or_none is not None
        state = state_or_none

        if not state.is_in_game:
            if self._lifecycle.check_not_in_game(state) == TickSignal.BREAK:
                return TickSignal.BREAK, state
            return TickSignal.CONTINUE, state

        self._world_updater.update_world_state(state, ctx, self._health_monitor, self._state_tracker)

        if (
            state.level != self._prev_level
            and self._prev_level > 0
            and abs(state.level - self._prev_level) == 1
        ):
            self._tick_handlers.handle_level_up(state, ctx)
        self._prev_level = state.level

        now = time.time()
        self._reporter.track_xp(state, ctx, now)
        self._tick_handlers.detect_adds(state, ctx)
        self._tick_handlers.scan_auto_engage(state, ctx)

        zone_signal = self._lifecycle.check_zoning_recovery(state, ctx)
        if zone_signal == TickSignal.BREAK:
            return TickSignal.BREAK, state
        if zone_signal == TickSignal.CONTINUE:
            return TickSignal.CONTINUE, state

        if self._world_updater.check_player_status(state, ctx):
            return TickSignal.BREAK, state

        if self._tick_periodic_snapshot(state, ctx, now):
            return TickSignal.BREAK, state

        self._learning.tick_tuning_eval(ctx, now)
        self._learning.tick_gradient_learning(ctx)

        # GOAP planner: suggest next action if plan exists
        goap_suggestion = self._learning.tick_goap_planner(state, ctx, now)
        ctx.diag.goap_suggestion = goap_suggestion if goap_suggestion else ""

        brain_signal = self._tick_brain(state, ctx)
        if brain_signal is TickSignal.BREAK:
            return TickSignal.BREAK, state
        if brain_signal is TickSignal.CONTINUE:
            return TickSignal.CONTINUE, state
        self._last_heartbeat = time.monotonic()
        self._tick_record_diag(state, ctx)

        return TickSignal.PROCEED, state

    def run(self) -> None:
        """Main brain loop. Call from a background thread."""
        set_brain_thread()
        self._last_heartbeat = time.monotonic()
        self._run_setup()

        ctx = self._ctx
        state = None  # track last state for shutdown cleanup
        try:
            while not self._stop_event.is_set():
                signal, tick_state = self._tick_one(ctx)
                if tick_state is not None:
                    state = tick_state
                if signal is TickSignal.BREAK:
                    break

            if self._stop_event.is_set():
                brain_log.log(
                    EVENT, "[LIFECYCLE] Brain loop stopped (stop_event set -- user/shutdown/death/resource)"
                )
            else:
                brain_log.warning(
                    "[LIFECYCLE] Brain loop exited unexpectedly (while loop ended without stop_event)"
                )

        except Exception as e:
            self._last_exception = e
            self._death_time = time.monotonic()
            brain_log.error("[LIFECYCLE] Brain loop crashed: %s", e, exc_info=True)
            if ctx.diag.forensics:
                ctx.diag.forensics.flush("crash")
        finally:
            self._run_cleanup(state)
