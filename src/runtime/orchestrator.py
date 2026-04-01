"""Agent orchestrator -- client connection, agent lifecycle, state management.

Owns the MemoryReader, BrainRunner lifecycle, and log ring buffer.
"""

from __future__ import annotations

import logging
import threading
import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from brain.runner import BrainRunner
from core import __version__
from core.features import flags
from core.types import DeathRecoveryMode, GrindStyle, LootMode, ManaMode, PlanType, Point
from perception.reader import MemoryReader
from runtime.agent import (
    build_brain,
    build_context,
    find_config,
    load_zone_config,
)
from runtime.session_logging import (
    LogCaptureHandler,
    SessionLoggingHandles,
    prune_session_logs,
    setup_session_logging,
)

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.decision import Brain
    from eq.spells import SpellDB
    from nav.waypoint_graph import WaypointGraph
    from nav.zone_graph import ZoneGraph

log = logging.getLogger(__name__)

_prune_session_logs = prune_session_logs


class AgentOrchestrator:
    """Manages client connection, agent lifecycle, and log ring buffer."""

    def __init__(self) -> None:
        # Connection
        self.reader: MemoryReader | None = None
        self._char_name = "Player"
        self.client_path = ""
        self.current_zone = ""
        self.zone_display = ""
        self._zone_config: dict = {}
        self._config: dict = {}
        self._spell_db: SpellDB | None = None

        # Agent state (guarded by _agent_lock for cross-thread access)
        self._agent_lock = threading.Lock()
        self.agent_running = False
        self.agent_paused = False
        self.agent_ctx: AgentContext | None = None
        self._standalone_travel_dest: str | None = None
        self.brain: Brain | None = None
        self._runner: BrainRunner | None = None
        self.stop_event = threading.Event()
        self.agent_thread: threading.Thread | None = None
        self.agent_start_time = 0.0
        self.agent_routine_name = ""
        self.agent_defeats = 0

        # Log ring buffer
        self._log_buffer: list[dict] = []
        self._log_lock = threading.Lock()
        self._log_index = 0

        # Logging handlers
        self._log_handler: LogCaptureHandler | None = None
        self._session_logging: SessionLoggingHandles | None = None

        # Travel
        self._zone_graph: ZoneGraph | None = None
        self._zone_waypoints: dict[str, Point] = {}
        self._travel_destinations: list[dict] = []
        self._tunnel_routes: list = []
        self._waypoint_graph: WaypointGraph | None = None

        # Brain crash auto-recovery
        self._brain_crash_restarts = 0
        self._brain_unhealthy_since: float = 0.0

        # Watchdog warning throttle
        self._last_watchdog_warn = 0.0

    # -- Connect to EQ --

    def connect(self, pid: int | None = None) -> None:
        """Connect to target process, create reader, detect zone, load SpellDB."""
        if pid is None:
            raise RuntimeError("Target process PID required")
        self.reader = MemoryReader(pid)

        # Load config
        try:
            config_path = find_config()
            with open(config_path, "rb") as f:
                self._config = tomllib.load(f)
        except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
            log.warning("[LIFECYCLE] Config load issue: %s", e)
            self._config = {}

        self.client_path = self._config.get("general", {}).get("client_path", "")

        # Character name
        try:
            self._char_name = self.reader.read_char_name()
        except (
            OSError,
            RuntimeError,
        ):
            pass

        # Zone detection
        self.current_zone = self._detect_zone()
        self.zone_display = self.current_zone.replace("_", " ").title()
        self._zone_config = load_zone_config(self.current_zone) or {}
        self._zone_dispositions = self._zone_config.get("disposition", {})

        # Build travel destinations at connect (available before agent starts)
        self._build_travel_data()

        # Spell DB for buff/cast names
        try:
            from eq.loadout import get_spell_db

            self._spell_db = get_spell_db(self.client_path)
        except (OSError, ValueError, KeyError, ImportError) as e:
            log.warning("[LIFECYCLE] SpellDB not loaded: %s", e)

        # Install log capture
        self._log_handler = LogCaptureHandler(self.add_log)
        logging.getLogger("compass").addHandler(self._log_handler)

        self.add_log(f"Connected: {self._char_name} in {self.zone_display}")
        self.add_log(f"Agent v{__version__}")

    def _detect_zone(self) -> str:
        """Detect the current zone from memory-mapped zone ID."""
        from eq.zone_ids import ZONE_ID_MAP
        from perception import offsets

        fallback = self._config.get("general", {}).get("current_zone", "unknown")
        try:
            assert self.reader is not None
            base = self.reader._get_spawn_base()
            zone_id = self.reader._read_int32(base + offsets.ZONE_ID)
            name: str | None = ZONE_ID_MAP.get(zone_id)
            if name:
                return name
        except (
            OSError,
            RuntimeError,
        ):
            pass
        return str(fallback)

    def _spell_name(self, spell_id: int) -> str:
        """Resolve spell_id -> name via SpellDB. Falls back to ID string."""
        if self._spell_db and spell_id > 0:
            spell = self._spell_db.get(spell_id)
            if spell:
                return str(spell.name)
        return f"Spell#{spell_id}" if spell_id > 0 else ""

    # -- Travel --

    def _build_travel_data(self) -> None:
        """Build zone graph + waypoints + tunnel routes for travel."""
        try:
            from nav.travel_planner import parse_tunnel_routes
            from nav.waypoint_graph import parse_waypoint_graph
            from nav.zone_graph import build_zone_graph

            self._zone_graph = build_zone_graph(f"{self.client_path}/maps")
            self._zone_waypoints = {}
            for wp in self._zone_config.get("waypoints", []):
                self._zone_waypoints[wp["name"]] = Point(wp["x"], wp["y"], wp.get("z", 0.0))
            self._tunnel_routes = parse_tunnel_routes(self._zone_config)
            self._waypoint_graph = parse_waypoint_graph(self._zone_config)
        except (OSError, KeyError, ValueError, TypeError, ImportError) as e:
            log.warning("[TRAVEL] Zone graph build failed: %s", e)
            self._zone_graph = None
            self._travel_destinations = []

    def travel_to(self, dest_name: str) -> dict:
        """Initiate travel to a waypoint or zone."""
        if not self.agent_running or not self.agent_ctx:
            return {"error": "Agent not running"}
        from brain.state.plan import TravelPlan

        if dest_name in self._zone_waypoints:
            _wp = self._zone_waypoints[dest_name]
            wx, wy = _wp.x, _wp.y
            self.agent_ctx.plan.active = PlanType.TRAVEL
            self.agent_ctx.plan.travel = TravelPlan(
                destination=dest_name, waypoint=True, target_x=wx, target_y=wy
            )
            return {"ok": True, "destination": dest_name}
        if not self._zone_graph:
            return {"error": "Zone graph not loaded"}
        route = self._zone_graph.find_route(self.current_zone, dest_name)
        if not route:
            return {"error": f"No route to {dest_name}"}
        self.agent_ctx.plan.active = PlanType.TRAVEL
        self.agent_ctx.plan.travel = TravelPlan(destination=dest_name, route=route, hop_index=0)
        return {"ok": True, "destination": dest_name, "hops": len(route)}

    def goto_corpse(self, pos: Point, name: str = "corpse") -> dict:
        """Navigate the agent to a corpse location."""
        if not self.agent_running or not self.agent_ctx:
            return {"error": "Start agent first"}
        self.agent_ctx.plan.active = PlanType.TRAVEL
        self.agent_ctx.plan.travel.hop_index = 0
        self.agent_ctx.plan.set_data(
            {
                "destination": f"corpse: {name}",
                "waypoint": True,
                "target_x": pos.x,
                "target_y": pos.y,
            }
        )
        self.add_log(f"Corpse locate: walking to {name}'s corpse at ({pos.x:.0f}, {pos.y:.0f})")
        return {"ok": True, "destination": name, "x": pos.x, "y": pos.y}

    _MAX_BRAIN_RESTARTS = 3  # max auto-restarts per session
    _BRAIN_UNHEALTHY_TIMEOUT = 30.0  # seconds before auto-restart

    def _check_brain_crash_recovery(self) -> None:
        """Auto-restart brain thread if unhealthy for 30s. Max 3 per session."""
        if not self.agent_running or not self._runner:
            self._brain_unhealthy_since = 0.0
            return

        if self._runner.brain_healthy:
            self._brain_unhealthy_since = 0.0
            return

        now = time.time()
        if self._brain_unhealthy_since == 0.0:
            self._brain_unhealthy_since = now
            return

        unhealthy_duration = now - self._brain_unhealthy_since
        if unhealthy_duration < self._BRAIN_UNHEALTHY_TIMEOUT:
            return

        # Brain has been unhealthy for 30s+ -- attempt restart
        if self._brain_crash_restarts >= self._MAX_BRAIN_RESTARTS:
            log.error(
                "[LIFECYCLE] BRAIN CRASH: max restarts (%d) reached -- not restarting",
                self._MAX_BRAIN_RESTARTS,
            )
            self._brain_unhealthy_since = 0.0  # stop checking
            return

        self._brain_crash_restarts += 1
        exc = self._runner.last_exception
        log.critical(
            "[LIFECYCLE] BRAIN CRASH RECOVERY: brain unhealthy %.0fs, restart %d/%d (error: %s)",
            unhealthy_duration,
            self._brain_crash_restarts,
            self._MAX_BRAIN_RESTARTS,
            exc if exc else "unresponsive",
        )
        self.add_log(
            f"Brain crashed -- auto-restart {self._brain_crash_restarts}/{self._MAX_BRAIN_RESTARTS}", "ERROR"
        )

        # Stop current brain thread
        try:
            self.stop_agent()
        except Exception as e:
            log.warning("[LIFECYCLE] Brain crash recovery: stop_agent error: %s", e)

        # Brief pause then restart
        time.sleep(2.0)
        try:
            result = self.start_agent()
            if "error" in result:
                log.error("[LIFECYCLE] Brain crash recovery: restart failed: %s", result["error"])
                self.add_log(f"Brain restart FAILED: {result['error']}", "ERROR")
            else:
                log.info("[LIFECYCLE] Brain crash recovery: restarted successfully")
                self.add_log("Brain restarted successfully", "WARNING")
        except Exception as e:
            log.error("[LIFECYCLE] Brain crash recovery: restart exception: %s", e)
            self.add_log(f"Brain restart FAILED: {e}", "ERROR")

        self._brain_unhealthy_since = 0.0

    # -- Log ring buffer --

    def add_log(self, msg: str, level: str = "INFO") -> None:
        with self._log_lock:
            self._log_buffer.append(
                {
                    "msg": msg,
                    "level": level,
                    "ts": time.time(),
                    "idx": self._log_index,
                }
            )
            self._log_index += 1
            if len(self._log_buffer) > 500:
                self._log_buffer = self._log_buffer[-400:]

    def get_logs_since(self, since_idx: int) -> list[dict]:
        with self._log_lock:
            return [e for e in self._log_buffer if e["idx"] > since_idx]

    # -- Agent lifecycle --

    def start_agent(self) -> dict:
        assert self.reader is not None, "connect() must be called before start_agent()"
        reader = self.reader
        if self.agent_running:
            return {"error": "Already running"}
        # Guard: ensure previous brain thread is fully dead before starting new one
        if self.agent_thread and self.agent_thread.is_alive():
            log.warning("[LIFECYCLE] start_agent: previous brain thread still alive -- waiting")
            self.stop_event.set()
            self.agent_thread.join(timeout=10)
            if self.agent_thread.is_alive():
                log.error(
                    "[LIFECYCLE] start_agent: old brain thread refuses to die -- "
                    "aborting start to prevent duplicate runners"
                )
                return {"error": "Previous brain thread still running"}
            self.agent_thread = None

        try:
            config_path = find_config()
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
        except FileNotFoundError:
            return {"error": "Config not found"}
        except tomllib.TOMLDecodeError as e:
            return {"error": f"Config load failed: {e}"}

        # Preserve ALL dashboard-toggled flags across config reload.
        # load_from_config resets everything to TOML defaults, which wipes
        # any runtime toggles the user set before clicking Start.
        pre = flags.as_dict()

        flags.load_from_config(config)
        flags.validate()

        # Restore any flags that differ from what was loaded
        post = flags.as_dict()
        restored = []
        for key, pre_val in pre.items():
            if pre_val != post.get(key):
                setattr(flags, key, pre_val)
                restored.append(f"{key}={pre_val}")
        if restored:
            log.info("[LIFECYCLE] start_agent: preserved dashboard toggles: %s", ", ".join(restored))

        # Read initial state for auto-camp
        state = reader.read_state()
        zone_config = load_zone_config(self.current_zone, player_level=state.level) or {}

        # Auto-configure spell loadout
        from eq.loadout import (
            check_spell_loadout,
            configure_from_memory,
            configure_loadout,
        )

        pcls = state.class_id
        plvl = state.level
        if pcls > 0 and plvl > 0:
            memorized = reader.read_memorized_spells()
            if memorized:
                configure_from_memory(memorized, pcls)
                changes = check_spell_loadout(memorized, pcls, plvl)
                if changes:
                    self.add_log(f"Spell mismatch: {len(changes)} changes needed")
                else:
                    self.add_log("Spell loadout OK")
            else:
                configure_loadout(pcls, plvl)
                self.add_log(f"Spell loadout configured for class={pcls} level={plvl}")

        # Validate configs before building context (catches typos early)
        from core.config_validator import log_config_warnings

        log_config_warnings(config, zone_config)

        ctx = build_context(config, zone_config, state.x, state.y, state.level)

        # Create explicit session wiring layer
        from runtime.agent_session import AgentSession

        session = AgentSession.from_globals(config)
        session.zone_config = zone_config
        session.reader = reader
        ctx.session = session

        brain = build_brain(ctx, reader)

        # Force spell memorization check on startup
        ctx.plan.active = PlanType.NEEDS_MEMORIZE

        # Log path for combat detection (set externally via config or CLI)
        log_path = ""

        # Session log file
        session_dir = Path(__file__).parent.parent.parent / "logs" / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        _prune_session_logs(session_dir)
        session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        self._session_logging = setup_session_logging(ctx, session_dir, session_id)

        self.stop_event.clear()
        runner = BrainRunner(
            reader=reader,
            ctx=ctx,
            brain=brain,
            stop_event=self.stop_event,
            config=config,
            current_zone=self.current_zone,
            log_path=log_path,
            session_id=session_id,
        )
        runner.on_display_update = self._on_display_update

        self.agent_ctx = ctx
        self.brain = brain
        self._runner = runner
        # Seed heartbeat so watchdog doesn't spam during startup/warmup
        runner._last_heartbeat = time.monotonic()
        with self._agent_lock:
            self.agent_running = True
            self.agent_paused = False
            self.agent_start_time = time.time()
            self.agent_defeats = 0
            self.agent_routine_name = ""

        # Build travel destinations
        self._build_travel_data()

        self.agent_thread = threading.Thread(target=self._run_brain, daemon=True)
        self.agent_thread.start()

        camp_name = ctx.zone.active_camp_name or "?"
        self.add_log(f"Agent STARTED in {self.zone_display} -- camp: {camp_name}")
        return {"running": True, "zone": self.current_zone}

    def _run_brain(self) -> None:
        assert self._runner is not None, "start_agent() must set _runner before _run_brain()"
        try:
            self._runner.run()
        except Exception as e:
            self.add_log(f"Brain crashed: {e}", "ERROR")
            log.exception("[LIFECYCLE] Brain thread crashed")
        finally:
            with self._agent_lock:
                self.agent_running = False
            self.add_log("Agent stopped")
            if self._session_logging:
                try:
                    self._session_logging.close()
                except Exception:
                    log.exception("[LIFECYCLE] Session logging teardown failed")
                finally:
                    self._session_logging = None

    def _on_display_update(self, routine_name: str, defeats: int) -> None:
        with self._agent_lock:
            self.agent_routine_name = routine_name
            self.agent_defeats = defeats

    def stop_agent(self) -> dict:
        if not self.agent_running:
            return {"error": "Not running"}
        self.stop_event.set()
        if self.agent_thread:
            self.agent_thread.join(timeout=10)
            if self.agent_thread.is_alive():
                log.warning(
                    "[LIFECYCLE] stop_agent: brain thread did not stop within 10s -- it will be orphaned"
                )
            self.agent_thread = None
        with self._agent_lock:
            self.agent_running = False
            defeats = self.agent_defeats
        self.add_log(f"Agent stopped -- {defeats} defeats")
        return {"running": False, "defeats": defeats}

    def pause_agent(self) -> dict:
        with self._agent_lock:
            if self._runner:
                self.agent_paused = not self.agent_paused
                self._runner.paused = self.agent_paused
            paused = self.agent_paused
        action = "PAUSED" if paused else "RESUMED"
        self.add_log(f"Agent {action}")
        return {"paused": paused}

    def toggle_feature(self, name: str) -> dict:
        if name == "loot_mode":
            loot_cycle: dict[LootMode, LootMode] = {
                LootMode.OFF: LootMode.SMART,
                LootMode.SMART: LootMode.ALL,
                LootMode.ALL: LootMode.OFF,
            }
            flags.loot_mode = loot_cycle.get(LootMode(flags.loot_mode), LootMode.SMART)
            self.add_log(f"Loot: {flags.loot_mode.upper()}")
            return {"loot_mode": flags.loot_mode}
        elif name == "death_recovery":
            dr_cycle: dict[DeathRecoveryMode, DeathRecoveryMode] = {
                DeathRecoveryMode.OFF: DeathRecoveryMode.SMART,
                DeathRecoveryMode.SMART: DeathRecoveryMode.ON,
                DeathRecoveryMode.ON: DeathRecoveryMode.OFF,
            }
            flags.death_recovery = dr_cycle.get(flags.death_recovery, DeathRecoveryMode.OFF)
            self.add_log(f"Death recovery: {flags.death_recovery.upper()}")
            return {"death_recovery": flags.death_recovery}
        elif name == "mana_mode":
            mm_cycle: dict[ManaMode, ManaMode] = {
                ManaMode.LOW: ManaMode.MEDIUM,
                ManaMode.MEDIUM: ManaMode.HIGH,
                ManaMode.HIGH: ManaMode.LOW,
            }
            flags.mana_mode = mm_cycle.get(ManaMode(flags.mana_mode), ManaMode.MEDIUM)
            self.add_log(f"Mana: {flags.mana_mode.upper()}")
            return {"mana_mode": flags.mana_mode}
        elif name == "grind_style":
            gs_cycle: dict[GrindStyle, GrindStyle] = {
                GrindStyle.WANDER: GrindStyle.FEAR_KITE,
                GrindStyle.FEAR_KITE: GrindStyle.CAMP_SIT,
                GrindStyle.CAMP_SIT: GrindStyle.WANDER,
            }
            flags.grind_style = gs_cycle.get(GrindStyle(flags.grind_style), GrindStyle.WANDER)
            self.add_log(f"Grind style: {flags.grind_style.upper()}")
            return {"grind_style": flags.grind_style}
        else:
            current = getattr(flags, name, None)
            if current is not None:
                setattr(flags, name, not current)
                new = getattr(flags, name)
                self.add_log(f"{name}: {'ON' if new else 'OFF'}")
                return {name: new}
            return {"error": f"Unknown flag: {name}"}

    def reload_config(self) -> dict:
        """Force immediate reload of feature flags from settings.toml."""
        try:
            from runtime.agent import find_config

            config_path = find_config()
            changed = flags.reload_from_file(config_path)
            if changed:
                self.add_log("Config reloaded (flags changed)")
                return {"reloaded": True, "flags": flags.as_dict()}
            return {"reloaded": False, "message": "No changes detected"}
        except (OSError, ValueError, KeyError) as e:
            return {"error": f"Config reload failed: {e}"}

    def shutdown(self) -> None:
        if self.agent_running:
            self.stop_agent()
        if self.reader:
            self.reader.close()
            self.reader = None
        if self._log_handler:
            logging.getLogger("compass").removeHandler(self._log_handler)
