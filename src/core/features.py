"""Central feature flags  -  single source of truth for toggleable behaviors.

Loaded from config at startup. Runtime updates modify live flags without
touching the file.

Usage in brain rules / routines:
    from core.features import flags
    if not flags.looting:
        return False  # skip loot rule entirely
"""

from __future__ import annotations

import logging
import threading
import tomllib
from pathlib import Path

from core.types import DeathRecoveryMode, GrindStyle, LootMode, ManaMode
from util.log_tiers import EVENT

log = logging.getLogger(__name__)

__all__ = ["FeatureFlags", "flags"]


class FeatureFlags:
    """Thread-safe feature flag store.

    All flags default to True (enabled). Config can override.
    Brain rules check flags each tick.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._change_callbacks: dict[str, list] = {}

        # -- Core behavior flags --
        self._loot_mode: str = LootMode.OFF
        self._combat_casting = True
        self._wander = True
        self._pull = True
        self._rest = True
        self._flee = True
        self._shielding_buff = False
        self._death_recovery = DeathRecoveryMode.OFF
        # Utility scoring phase: 0=binary, 1=divergence log,
        # 2=tier-based, 3=weighted, 4=consideration-based
        self._utility_phase: int = 2
        # Obstacle avoidance: A* respects SURFACE_OBSTACLE cells
        self._obstacle_avoidance: bool = True
        # Mana usage mode: LOW | MEDIUM | HIGH
        self._mana_mode: str = ManaMode.MEDIUM
        # Grinding style: WANDER | FEAR_KITE | CAMP_SIT
        self._grind_style: str = GrindStyle.WANDER
        # Pareto multi-objective target selection
        self._pareto_scoring: bool = False
        # GOAP goal-oriented action planning
        self._goap_planning: bool = True

    # -- Change notification --

    def on_change(self, flag_name: str, callback: object) -> None:
        """Register a callback for when a flag changes.

        Used by terrain to invalidate walk_bits when obstacle_avoidance
        toggles, without features.py importing from nav/.
        """
        self._change_callbacks.setdefault(flag_name, []).append(callback)

    def _notify(self, flag_name: str, value: object) -> None:
        for cb in self._change_callbacks.get(flag_name, []):
            try:
                cb(value)
            except Exception:
                log.warning("[LIFECYCLE] Flag change callback failed: %s", flag_name)

    # -- Thread-safe properties --

    @property
    def loot_mode(self) -> str:
        with self._lock:
            return self._loot_mode

    @loot_mode.setter
    def loot_mode(self, value: str) -> None:
        with self._lock:
            self._loot_mode = value
        log.info("[LIFECYCLE] Feature flag: loot_mode = %s", value)

    @property
    def looting(self) -> bool:
        with self._lock:
            result: bool = self._loot_mode != LootMode.OFF
            return result

    @looting.setter
    def looting(self, value: bool) -> None:
        with self._lock:
            self._loot_mode = LootMode.ALL if value else LootMode.OFF
        log.info("[LIFECYCLE] Feature flag: loot_mode = %s", self._loot_mode)

    @property
    def combat_casting(self) -> bool:
        with self._lock:
            return self._combat_casting

    @combat_casting.setter
    def combat_casting(self, value: bool) -> None:
        with self._lock:
            self._combat_casting = value
        log.info("[LIFECYCLE] Feature flag: combat_casting = %s", value)

    @property
    def wander(self) -> bool:
        with self._lock:
            return self._wander

    @wander.setter
    def wander(self, value: bool) -> None:
        with self._lock:
            self._wander = value
        log.info("[LIFECYCLE] Feature flag: wander = %s", value)

    @property
    def pull(self) -> bool:
        with self._lock:
            return self._pull

    @pull.setter
    def pull(self, value: bool) -> None:
        with self._lock:
            self._pull = value
        log.info("[LIFECYCLE] Feature flag: pull = %s", value)

    @property
    def rest(self) -> bool:
        with self._lock:
            return self._rest

    @rest.setter
    def rest(self, value: bool) -> None:
        with self._lock:
            self._rest = value
        log.info("[LIFECYCLE] Feature flag: rest = %s", value)

    @property
    def flee(self) -> bool:
        with self._lock:
            return self._flee

    @flee.setter
    def flee(self, value: bool) -> None:
        with self._lock:
            self._flee = value
        log.info("[LIFECYCLE] Feature flag: flee = %s", value)

    @property
    def shielding_buff(self) -> bool:
        with self._lock:
            return self._shielding_buff

    @shielding_buff.setter
    def shielding_buff(self, value: bool) -> None:
        with self._lock:
            self._shielding_buff = bool(value)
        log.info("[LIFECYCLE] Feature flag: shielding_buff = %s", value)

    @property
    def obstacle_avoidance(self) -> bool:
        """Whether A* pathfinding respects SURFACE_OBSTACLE cells."""
        with self._lock:
            return self._obstacle_avoidance

    @obstacle_avoidance.setter
    def obstacle_avoidance(self, value: bool) -> None:
        with self._lock:
            self._obstacle_avoidance = value
        log.info("[LIFECYCLE] Feature flag: obstacle_avoidance = %s", value)
        self._notify("obstacle_avoidance", value)

    @property
    def mana_mode(self) -> str:
        with self._lock:
            return self._mana_mode

    @mana_mode.setter
    def mana_mode(self, value: str) -> None:
        with self._lock:
            self._mana_mode = str(value)
        log.info("[LIFECYCLE] Feature flag: mana_mode = %s", value)

    @property
    def grind_style(self) -> str:
        with self._lock:
            return self._grind_style

    @grind_style.setter
    def grind_style(self, value: str) -> None:
        with self._lock:
            self._grind_style = str(value)
        log.info("[LIFECYCLE] Feature flag: grind_style = %s", value)

    @property
    def death_recovery(self) -> DeathRecoveryMode:
        with self._lock:
            return self._death_recovery

    @death_recovery.setter
    def death_recovery(self, value: str) -> None:
        with self._lock:
            self._death_recovery = DeathRecoveryMode(str(value))
        log.info("[LIFECYCLE] Feature flag: death_recovery = %s", value)

    @property
    def utility_phase(self) -> int:
        """Utility scoring phase: 0=binary, 1=divergence, 2=tier, 3=weighted, 4=considerations."""
        with self._lock:
            return self._utility_phase

    @utility_phase.setter
    def utility_phase(self, value: int) -> None:
        with self._lock:
            self._utility_phase = max(0, min(4, int(value)))
        log.info("[LIFECYCLE] Feature flag: utility_phase = %d", self._utility_phase)

    @property
    def goap_planning(self) -> bool:
        """Goal-Oriented Action Planning for multi-step sequencing."""
        with self._lock:
            return self._goap_planning

    @goap_planning.setter
    def goap_planning(self, value: bool) -> None:
        with self._lock:
            self._goap_planning = bool(value)
        log.info("[LIFECYCLE] Feature flag: goap_planning = %s", value)

    @property
    def pareto_scoring(self) -> bool:
        """Multi-objective Pareto target selection."""
        with self._lock:
            return self._pareto_scoring

    @pareto_scoring.setter
    def pareto_scoring(self, value: bool) -> None:
        with self._lock:
            self._pareto_scoring = bool(value)
        log.info("[LIFECYCLE] Feature flag: pareto_scoring = %s", value)

    def should_recover_death(self, deaths: int) -> bool:
        """Check if recovery should be attempted given the death count."""
        with self._lock:
            if self._death_recovery == DeathRecoveryMode.OFF:
                return False
            if self._death_recovery == DeathRecoveryMode.SMART:
                return deaths <= 1
            return True

    def load_from_config(self, config: dict) -> None:
        """Load flags from [features] section of config."""
        features = config.get("features", {})
        with self._lock:
            for key, value in features.items():
                if key == "loot_mode":
                    self._loot_mode = str(value)
                elif key == "death_recovery":
                    self._death_recovery = DeathRecoveryMode(str(value))
                elif key == "mana_mode":
                    self._mana_mode = str(value)
                elif key == "grind_style":
                    self._grind_style = str(value)
                elif key == "utility_phase":
                    self._utility_phase = max(0, min(4, int(value)))
                elif hasattr(self, f"_{key}"):
                    setattr(self, f"_{key}", bool(value))
        self.log_summary()

    def log_summary(self) -> None:
        """Log all feature flags as a single compact line."""
        d = self.as_dict()
        parts = []
        for k, v in d.items():
            if isinstance(v, bool):
                parts.append(f"{k}={'on' if v else 'off'}")
            else:
                parts.append(f"{k}={v}")
        log.info("[LIFECYCLE] Features: %s", " ".join(parts))

    def validate(self) -> list[str]:
        """Check for invalid or suspicious flag combinations."""
        warnings: list[str] = []
        with self._lock:
            if not self._flee:
                warnings.append("flee=OFF -- agent cannot escape threat situations")
            if not self._rest:
                warnings.append("rest=OFF -- agent will not recover HP/mana between fights")
        for w in warnings:
            log.warning("[LIFECYCLE] Flag validation: %s", w)
        return warnings

    def reload_from_file(self, path: Path) -> bool:
        """Reload [features] section from TOML file. Returns True if changed."""
        try:
            with open(path, "rb") as f:
                raw = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            log.warning("[LIFECYCLE] Config reload failed: %s", e)
            return False
        old = self.as_dict()
        self.load_from_config(raw)
        new = self.as_dict()
        changed = {k for k in new if old.get(k) != new.get(k)}
        if changed:
            log.log(EVENT, "[LIFECYCLE] CONFIG RELOAD: changed=%s", changed)
            self.validate()
            return True
        return False

    def as_dict(self) -> dict:
        """Return all flags as a dict (for logging/display)."""
        with self._lock:
            return {
                "loot_mode": self._loot_mode,
                "combat_casting": self._combat_casting,
                "wander": self._wander,
                "pull": self._pull,
                "rest": self._rest,
                "flee": self._flee,
                "shielding_buff": self._shielding_buff,
                "obstacle_avoidance": self._obstacle_avoidance,
                "death_recovery": self._death_recovery,
                "utility_phase": self._utility_phase,
                "mana_mode": self._mana_mode,
                "grind_style": self._grind_style,
                "pareto_scoring": self._pareto_scoring,
                "goap_planning": self._goap_planning,
            }


# Global singleton
flags = FeatureFlags()
