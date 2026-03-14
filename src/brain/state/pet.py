"""Pet tracking state."""

from __future__ import annotations

from dataclasses import dataclass

from perception.queries import is_pet
from perception.state import GameState


@dataclass(slots=True, kw_only=True)
class PetState:
    """Tracks our summoned pet by spawn_id."""

    alive: bool = False
    prev_alive: bool = False
    spawn_id: int | None = None
    name: str = ""
    has_add: bool = False
    last_heal_time: float = 0.0

    def update(self, state: GameState) -> None:
        """Detect our pet in the spawn list. Called every tick."""
        self.prev_alive = self.alive
        self.alive = False

        if self.spawn_id is not None:
            for spawn in state.spawns:
                if spawn.spawn_id == self.spawn_id and spawn.is_npc:
                    dist = state.pos.dist_to(spawn.pos)
                    if dist < 1000 and spawn.hp_current > 0:
                        self.alive = True
                        return
            self.spawn_id = None
            self.name = ""
            return

        # No tracked pet  -  find closest pet-named NPC
        best_dist = 9999.0
        best_spawn = None
        for spawn in state.spawns:
            if is_pet(spawn) and spawn.is_npc:
                dist = state.pos.dist_to(spawn.pos)
                if dist < 100 and spawn.level <= state.level + 3 and dist < best_dist:
                    best_dist = dist
                    best_spawn = spawn
        if best_spawn:
            self.alive = True
            self.spawn_id = best_spawn.spawn_id
            self.name = best_spawn.name

    def just_died(self) -> bool:
        """True if pet was alive last tick but gone now."""
        return self.prev_alive and not self.alive
