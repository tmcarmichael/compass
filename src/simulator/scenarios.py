"""Scenario definitions for headless simulation.

A scenario is a sequence of (GameState, phase_label) pairs that drive
the brain through a realistic progression of world states.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from perception.state import GameState, SpawnData


def _spawn(**kw: Any) -> SpawnData:
    defaults: dict[str, Any] = dict(
        spawn_id=100,
        name="a_skeleton",
        x=50.0,
        y=50.0,
        z=0.0,
        heading=0.0,
        speed=0.0,
        level=10,
        spawn_type=1,
        race=0,
        mob_class=0,
        hide=0,
        hp_current=100,
        hp_max=100,
    )
    defaults.update(kw)
    return SpawnData(**defaults)


def _state(**kw: Any) -> GameState:
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
        spawn_type=0,
        stand_state=0,
        player_state=0,
        spawn_id=1,
        speed_run=0.7,
        speed_heading=0.5,
    )
    defaults.update(kw)
    return GameState(**defaults)


@dataclass
class Scenario:
    """Sequence of (GameState, phase_label) pairs for simulation."""

    name: str
    states: list[tuple[GameState, str]]

    @property
    def tick_count(self) -> int:
        return len(self.states)

    @classmethod
    def from_json(cls, path: str) -> Scenario:
        """Load a scenario from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        name = data.get("name", path)
        states: list[tuple[GameState, str]] = []
        for phase in data["phases"]:
            label = phase["label"]
            ticks = phase.get("ticks", 10)
            sd = phase.get("state", {})
            spawns_raw = sd.pop("spawns", [])
            target_raw = sd.pop("target", None)
            # Convert convenience fields
            if "hp_pct" in sd:
                pct = sd.pop("hp_pct")
                sd.setdefault("hp_current", int(pct * sd.get("hp_max", 1000)))
            if "mana_pct" in sd:
                pct = sd.pop("mana_pct")
                sd.setdefault("mana_current", int(pct * sd.get("mana_max", 500)))
            spawns = tuple(_spawn(**s) for s in spawns_raw)
            target = _spawn(**target_raw) if target_raw else None
            gs = _state(spawns=spawns, target=target, **sd)
            for _ in range(ticks):
                states.append((gs, label))
        return cls(name=name, states=states)

    # -- Built-in scenarios --

    @classmethod
    def camp_session(cls, cycles: int = 8) -> Scenario:
        """5-minute camp grind: pull/combat/rest cycles.

        Exercises target scoring, combat routines, rest rules, encounter
        learning, weight tuner, and GOAP planning.
        """
        states: list[tuple[GameState, str]] = []
        npc_id = 100

        for i in range(cycles):
            npc_id += 1
            npc = _spawn(
                spawn_id=npc_id, name="a_skeleton", x=80.0, y=0.0, level=10, hp_current=500, hp_max=500
            )

            # Idle at camp (targets visible)
            for _ in range(30):
                states.append(
                    (
                        _state(
                            hp_current=1000,
                            mana_current=500,
                            spawns=(npc,),
                        ),
                        "idle",
                    )
                )

            # Acquire + pull phase
            for _ in range(20):
                states.append(
                    (
                        _state(
                            hp_current=1000,
                            mana_current=480,
                            target=npc,
                            spawns=(npc,),
                        ),
                        "pull",
                    )
                )

            # Combat: HP/mana drain, target HP drops
            for t in range(60):
                frac = t / 60
                target_hp = max(1, int(500 * (1 - frac)))
                dmg_npc = _spawn(
                    spawn_id=npc_id,
                    name="a_skeleton",
                    x=60.0,
                    y=0.0,
                    level=10,
                    hp_current=target_hp,
                    hp_max=500,
                )
                states.append(
                    (
                        _state(
                            hp_current=int(1000 * (1 - frac * 0.15)),
                            mana_current=int(500 * (1 - frac * 0.40)),
                            in_combat=True,
                            target=dmg_npc,
                            spawns=(dmg_npc,),
                        ),
                        "combat",
                    )
                )

            # Victory: target dead, rest needed
            mana_after = max(50, 500 - i * 30)
            for t in range(50):
                frac = t / 50
                states.append(
                    (
                        _state(
                            hp_current=int(850 + 150 * frac),
                            mana_current=int(mana_after + (500 - mana_after) * frac),
                            stand_state=1 if frac < 0.8 else 0,  # sitting to rest
                        ),
                        "rest",
                    )
                )

        return cls(name="camp_session", states=states)

    @classmethod
    def survival_stress(cls) -> Scenario:
        """Escalating damage to test safety envelope and flee behavior.

        Exercises flee urgency axes, emergency rules, danger memory.
        """
        states: list[tuple[GameState, str]] = []
        npc = _spawn(spawn_id=200, name="a_ghoul", x=40.0, y=0.0, level=12, hp_current=800, hp_max=800)

        # Healthy start
        for _ in range(30):
            states.append(
                (
                    _state(
                        hp_current=1000,
                        mana_current=500,
                        spawns=(npc,),
                    ),
                    "healthy",
                )
            )

        # Gradual damage ramp
        for t in range(80):
            frac = t / 80
            hp = int(1000 * (1 - frac * 0.7))
            states.append(
                (
                    _state(
                        hp_current=hp,
                        mana_current=300,
                        in_combat=True,
                        target=npc,
                        spawns=(npc,),
                    ),
                    "damage_ramp",
                )
            )

        # Critical: should trigger FLEE
        for _ in range(30):
            states.append(
                (
                    _state(
                        hp_current=250,
                        mana_current=100,
                        in_combat=True,
                        target=npc,
                        spawns=(npc,),
                    ),
                    "critical",
                )
            )

        # Fleeing: out of combat, recovering
        for t in range(40):
            frac = t / 40
            states.append(
                (
                    _state(
                        hp_current=int(250 + 400 * frac),
                        mana_current=int(100 + 200 * frac),
                    ),
                    "recovery",
                )
            )

        # Second wave: adds appear
        add = _spawn(spawn_id=201, name="a_skeleton", x=30.0, y=10.0, level=11, hp_current=400, hp_max=400)
        for _ in range(40):
            states.append(
                (
                    _state(
                        hp_current=650,
                        mana_current=300,
                        in_combat=True,
                        target=npc,
                        spawns=(npc, add),
                    ),
                    "adds",
                )
            )

        return cls(name="survival_stress", states=states)

    @classmethod
    def exploration(cls) -> Scenario:
        """Sparse spawns over large area. Tests GOAP and spatial learning.

        Exercises GOAP planner, spawn prediction, spatial memory, wander.
        """
        states: list[tuple[GameState, str]] = []

        # Long wander phase: no targets nearby
        for t in range(200):
            x = float(t * 2)
            states.append(
                (
                    _state(
                        x=x,
                        y=0.0,
                        hp_current=1000,
                        mana_current=500,
                    ),
                    "wander_empty",
                )
            )

        # Discover sparse targets
        for batch in range(4):
            npc_id = 300 + batch
            npc = _spawn(
                spawn_id=npc_id,
                name="a_bat",
                x=400.0 + batch * 100,
                y=50.0,
                level=9,
                hp_current=300,
                hp_max=300,
            )

            # Approach
            for t in range(20):
                dist = 120 - t * 5
                states.append(
                    (
                        _state(
                            x=npc.x - dist,
                            y=0.0,
                            hp_current=1000,
                            mana_current=500,
                            spawns=(npc,),
                        ),
                        "approach",
                    )
                )

            # Combat
            for t in range(40):
                frac = t / 40
                dmg = _spawn(
                    spawn_id=npc_id,
                    name="a_bat",
                    x=npc.x,
                    y=50.0,
                    level=9,
                    hp_current=max(1, int(300 * (1 - frac))),
                    hp_max=300,
                )
                states.append(
                    (
                        _state(
                            x=npc.x - 20,
                            y=0.0,
                            hp_current=int(1000 * (1 - frac * 0.05)),
                            mana_current=int(500 * (1 - frac * 0.15)),
                            in_combat=True,
                            target=dmg,
                            spawns=(dmg,),
                        ),
                        "combat",
                    )
                )

            # Rest between encounters
            for t in range(30):
                frac = t / 30
                states.append(
                    (
                        _state(
                            x=npc.x,
                            y=0.0,
                            hp_current=int(950 + 50 * frac),
                            mana_current=int(425 + 75 * frac),
                        ),
                        "rest",
                    )
                )

        return cls(name="exploration", states=states)
