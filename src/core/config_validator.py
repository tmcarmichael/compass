"""Config validation -- catches typos and missing keys at startup.

Warns loudly on unknown keys (likely typos) and missing required keys.
Does NOT crash -- the agent continues with defaults, and the session log
shows exactly what is misconfigured.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# -- Known keys per section (settings.toml) --------------------------------

_GENERAL_KEYS = {
    "client_path",
    "current_zone",
    "active_camp",
    "tick_rate_hz",
    "log_level",
    "character_name",
    "server_name",
}

_THRESHOLD_KEYS = {
    "rest_hp_low",
    "rest_hp_high",
    "rest_mana_low",
    "rest_mana_high",
    "flee_hp",
    "arrival_tolerance",
}

_FEATURE_KEYS = {
    "loot_mode",
    "combat_casting",
    "wander",
    "pull",
    "rest",
    "flee",
    "shielding_buff",
    "death_recovery",
    "utility_phase",
    "grind_style",
    "obstacle_avoidance",
    "mana_mode",
    "pareto_scoring",
    "goap_planning",
    "looting",
}

_STUCK_KEYS = {"check_seconds", "min_distance", "try_jump"}

_SETTINGS_SECTIONS = {
    "general": _GENERAL_KEYS,
    "thresholds": _THRESHOLD_KEYS,
    "features": _FEATURE_KEYS,
    "stuck": _STUCK_KEYS,
    "log_levels": None,  # free-form (logger names -> levels)
}

# -- Known keys for zone config (zones/*.toml) -----------------------------

_ZONE_SECTIONS = {
    "zone",
    "camps",
    "waypoints",
    "waypoint_edges",
    "walkable_overrides",
    "water_overrides",
    "disposition",
    "social",
    "water_z_threshold",
}

_CAMP_KEYS = {
    "name",
    "center",
    "safe_spot",
    "flee_spot",
    "hunt_min_dist",
    "hunt_max_dist",
    "roam_radius",
    "mob_names",
    "avoid_mobs",
    "level_range",
    "danger_points",
    "flee_waypoints",
    "pull_distance",
    "description",
    # LINEAR camp fields
    "camp_type",
    "patrol_waypoints",
    "corridor_width",
    # Bounds
    "bounds_x_min",
    "bounds_x_max",
    "bounds_y_min",
    "bounds_y_max",
}


def validate_settings(config: dict[str, object]) -> list[str]:
    """Validate settings.toml structure. Returns list of warnings."""
    warnings: list[str] = []

    for section in config:
        if section not in _SETTINGS_SECTIONS:
            warnings.append(f"settings.toml: unknown section [{section}]")

    for section, known_keys in _SETTINGS_SECTIONS.items():
        if known_keys is None:
            continue
        data = config.get(section, {})
        if not isinstance(data, dict):
            continue
        for key in data:
            if key not in known_keys:
                warnings.append(
                    f"settings.toml [{section}]: unknown key '{key}'"
                    f" (typo? known: {', '.join(sorted(known_keys))})"
                )

    thresholds_raw = config.get("thresholds", {})
    thresholds = thresholds_raw if isinstance(thresholds_raw, dict) else {}
    for key in ("rest_hp_low", "rest_hp_high", "rest_mana_low", "rest_mana_high", "flee_hp"):
        val = thresholds.get(key)
        if val is not None and not (0.0 <= val <= 1.0):
            warnings.append(f"settings.toml [thresholds]: {key}={val} outside valid range 0.0-1.0")

    return warnings


def validate_zone_config(zone_config: dict[str, object]) -> list[str]:
    """Validate zone config structure. Returns list of warnings."""
    warnings: list[str] = []

    for key in zone_config:
        if key not in _ZONE_SECTIONS:
            warnings.append(f"zone config: unknown top-level key '{key}'")

    camps_raw = zone_config.get("camps", [])
    camps = camps_raw if isinstance(camps_raw, list) else []
    for i, camp in enumerate(camps):
        if not isinstance(camp, dict):
            continue
        name = camp.get("name", f"camp[{i}]")
        if "name" not in camp:
            warnings.append(f"zone config: camp[{i}] missing 'name'")
        if "center" not in camp:
            warnings.append(f"zone config: camp '{name}' missing 'center'")
        for key in camp:
            if key not in _CAMP_KEYS:
                warnings.append(f"zone config: camp '{name}' unknown key '{key}'")

    return warnings


def log_config_warnings(config: dict[str, object], zone_config: dict[str, object] | None = None) -> int:
    """Validate configs and log warnings. Returns warning count."""
    warnings = validate_settings(config)
    if zone_config:
        warnings.extend(validate_zone_config(zone_config))

    for w in warnings:
        log.warning("[LIFECYCLE] CONFIG: %s", w)

    if not warnings:
        log.info("[LIFECYCLE] CONFIG: validation passed (no issues)")

    return len(warnings)
