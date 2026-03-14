"""TypedDict schemas for structured events emitted via log_event().

Each TypedDict corresponds to one event type in the _events.jsonl log.
These serve as living documentation and enable static type checking
at call sites that adopt the **fields unpacking pattern.

The 'decision' event (emitted by DecisionThrottle) is excluded -- it is
internal, high-frequency, and its schema lives in structured_log.py.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class FightEndEvent(TypedDict):
    """Emitted when a combat routine completes (routines/combat.py)."""

    npc: str
    duration: float
    casts: int
    mana_spent: int
    backsteps: int
    retargets: int
    pet_heals: int
    adds: int
    hp_delta: float
    hp_start: float
    hp_end: float
    mana_start: int
    mana_end: int
    cast_time: float
    idle_time: float
    med_time: float
    init_dist: int
    defeats: int
    pos_x: int
    pos_y: int
    strategy: str
    entity_id: int
    world: dict[str, Any]
    cycle_id: NotRequired[int]


class PullResultEvent(TypedDict):
    """Emitted when a pull routine completes (routines/pull.py)."""

    success: bool
    target: str
    strategy: str
    duration: float
    dist: int
    pet_engage: float
    dot_retries: int
    pos_x: int
    pos_y: int
    entity_id: int
    cycle_id: NotRequired[int]


class AcquireResultEvent(TypedDict):
    """Emitted when an acquire routine completes (routines/acquire.py)."""

    success: bool
    tabs: int
    target: str
    consecutive_fails: int
    entity_id: int
    cycle_id: NotRequired[int]


class FleeTriggerEvent(TypedDict):
    """Emitted when the flee routine activates (routines/flee.py)."""

    hp_pct: float
    mana: int
    pet: str
    pos_x: int
    pos_y: int
    target: str
    target_dist: int
    entity_id: int
    world: dict[str, Any]


class RestEndEvent(TypedDict):
    """Emitted when the rest routine completes (routines/rest.py)."""

    duration: float
    hp_start: float
    hp_end: float
    mana_start: int
    mana_end: int
    pet_hp: int


class SummonPetResultEvent(TypedDict):
    """Emitted when a pet is accepted (routines/summon_pet.py)."""

    pet_name: str
    pet_level: int
    pet_id: int
    spell_name: str
    spell_id: int
    range_min: int
    range_max: int
    resummons: int
    was_optimal: bool


class LevelUpEvent(TypedDict):
    """Emitted on player level-up (brain_tick_handlers.py)."""

    old_level: int
    new_level: int
    mana: int
    mana_max_old: int
    pos_x: int
    pos_y: int
    defeats: int


class PlayerDeathEvent(TypedDict):
    """Emitted on player death (brain_lifecycle.py)."""

    source: str
    routine: str
    pos_x: int
    pos_y: int
    pos_z: int
    deaths: int
    recover: bool


class RoutineEndEvent(TypedDict):
    """Emitted when any routine completes (brain/decision.py)."""

    routine: str
    result: str
    hp_pct: float
    mana_pct: float
    reason: str
    failure_category: str
    entity_id: int
    cycle_id: NotRequired[int]


class InvariantViolationEvent(TypedDict):
    """Emitted on runtime invariant failure (util/invariants.py)."""

    invariant: str
    category: str
    detail: str


class SessionEndEvent(TypedDict):
    """Emitted at session shutdown (util/session_reporter.py)."""

    duration: float
    defeats: int
    deaths: int
    flees: int
    dph: float
    xp_pct: float
    grade: str
    loot_count: int


class SnapshotEvent(TypedDict):
    """Emitted every 30s as a periodic health pulse (util/session_reporter.py)."""

    hp_pct: float
    mana_pct: float
    pos_x: int
    pos_y: int
    defeats: int
    dph: float
    engaged: bool
    routine: str
    pet_alive: bool
    npcs_nearby: int
    camp_dist: int
    players: int
    acq_fails: int
    last_kill_age: int
    # Derivative fields (rate of change since last snapshot)
    mana_rate: NotRequired[float]  # mana/min delta (negative = draining)
    kph_5min: NotRequired[float]  # defeats/hr over last 5 min
    kph_delta: NotRequired[float]  # change from session average
    camp_drift_rate: NotRequired[float]  # units/min movement from camp
    avg_cycle_s: NotRequired[float]  # rolling average cycle time
    phase: NotRequired[str]  # current operational phase


class CycleCompleteEvent(TypedDict):
    """Emitted when an acquire->pull->combat defeat cycle finishes (util/cycle_tracker.py)."""

    cycle_id: int
    npc: str
    entity_id: int
    acquire_tabs: int
    pull_strategy: str
    pull_duration: float
    pull_dot_retries: int
    fight_duration: float
    fight_casts: int
    fight_mana_spent: int
    fight_hp_delta: float
    fight_adds: int
    fight_strategy: str
    fight_med_time: float
    mana_start: int
    mana_end: int
    cycle_total_s: float
    defeats_so_far: int
    kph_rolling: float
    pos_x: int
    pos_y: int


class IncidentEvent(TypedDict):
    """Emitted on death or flee with causal chain (util/incident_reporter.py)."""

    type: str  # "player_death" or "flee"
    summary: str  # human-readable one-line summary
    trigger_mob: str  # npc that caused the incident
    trigger_reason: str  # why it happened (pet_died, hp_low, etc.)
    source: NotRequired[str]  # death source (hp_zero, etc.)
    hp_sequence: list[float]  # HP% drain over time (deduplicated)
    mob_hp_sequence: list[int]  # chasing npc HP over time
    flee_distance: int  # distance from start to end position
    flee_duration_s: float  # seconds from trigger to end
    mana_at_trigger: int  # mana when incident began
    guards_nearby: bool  # whether guards were within 300u
    cycle_id: NotRequired[int]  # defeat cycle that led to this
    defeats_before_incident: NotRequired[int]
    pos_start_x: NotRequired[int]
    pos_start_y: NotRequired[int]
    pos_end_x: NotRequired[int]
    pos_end_y: NotRequired[int]


class PhaseChangeEvent(TypedDict):
    """Emitted when operational phase transitions (util/phase_detector.py)."""

    old_phase: str  # startup, grinding, resting, incident, idle
    new_phase: str
    phase_duration_s: float  # how long we were in the old phase
    defeats_in_phase: int
    kph_in_phase: float
    elapsed: float  # total session elapsed seconds
    total_kills: int


class SpellResultEvent(TypedDict):
    """Emitted on spell cast outcome (brain_tick_handlers.py)."""

    result: str  # fizzle, interrupt, los_blocked, must_stand
    engaged: bool  # whether in combat when result occurred
