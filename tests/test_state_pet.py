"""Tests for brain.state.pet -- PetState tracking.

Covers PetState.update (tracking by spawn_id, discovery of new pet,
pet death detection) and just_died().
"""

from __future__ import annotations

from brain.state.pet import PetState
from tests.factories import make_game_state, make_spawn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pet_spawn(**overrides):
    """Create a spawn that looks like a pet (NPC, close, alive)."""
    defaults = dict(
        spawn_id=200,
        name="Xantik123",  # pet name pattern: Capital + digits
        x=10.0,
        y=10.0,
        z=0.0,
        level=10,
        spawn_type=1,  # NPC
        hp_current=100,
        hp_max=100,
        owner_id=1,  # owned by player
    )
    defaults.update(overrides)
    return make_spawn(**defaults)


# ---------------------------------------------------------------------------
# update: known pet tracking
# ---------------------------------------------------------------------------


class TestPetUpdateKnown:
    def test_tracks_known_pet(self) -> None:
        pet = PetState(spawn_id=200, name="Xantik123")
        spawn = _pet_spawn()
        state = make_game_state(x=10.0, y=10.0, spawns=(spawn,))
        pet.update(state)
        assert pet.alive is True
        assert pet.spawn_id == 200

    def test_known_pet_too_far_away(self) -> None:
        """Pet >1000 units away is considered lost."""
        pet = PetState(spawn_id=200, name="Xantik123")
        spawn = _pet_spawn(x=2000.0, y=2000.0)
        state = make_game_state(x=0.0, y=0.0, spawns=(spawn,))
        pet.update(state)
        assert pet.alive is False
        assert pet.spawn_id is None
        assert pet.name == ""

    def test_known_pet_dead(self) -> None:
        """Pet with hp_current=0 is considered dead."""
        pet = PetState(spawn_id=200, name="Xantik123")
        spawn = _pet_spawn(hp_current=0)
        state = make_game_state(x=10.0, y=10.0, spawns=(spawn,))
        pet.update(state)
        assert pet.alive is False
        assert pet.spawn_id is None

    def test_known_pet_despawned(self) -> None:
        """Pet not in spawn list is considered lost."""
        pet = PetState(spawn_id=200, name="Xantik123")
        state = make_game_state(x=0.0, y=0.0, spawns=())
        pet.update(state)
        assert pet.alive is False
        assert pet.spawn_id is None

    def test_known_pet_not_npc_ignored(self) -> None:
        """Non-NPC with same spawn_id is not matched."""
        pet = PetState(spawn_id=200, name="Xantik123")
        spawn = _pet_spawn(spawn_type=0)  # Player, not NPC
        state = make_game_state(x=10.0, y=10.0, spawns=(spawn,))
        pet.update(state)
        assert pet.alive is False
        assert pet.spawn_id is None


# ---------------------------------------------------------------------------
# update: discovery of new pet
# ---------------------------------------------------------------------------


class TestPetDiscovery:
    def test_discovers_new_pet_nearby(self) -> None:
        pet = PetState()
        spawn = _pet_spawn(x=5.0, y=5.0, level=10)
        state = make_game_state(x=0.0, y=0.0, level=10, spawns=(spawn,))
        pet.update(state)
        assert pet.alive is True
        assert pet.spawn_id == 200
        assert pet.name == "Xantik123"

    def test_ignores_pet_too_far(self) -> None:
        """Untracked pet >100 units away is not discovered."""
        pet = PetState()
        spawn = _pet_spawn(x=200.0, y=200.0, level=10)
        state = make_game_state(x=0.0, y=0.0, level=10, spawns=(spawn,))
        pet.update(state)
        assert pet.alive is False
        assert pet.spawn_id is None

    def test_ignores_pet_too_high_level(self) -> None:
        """Untracked pet with level > player+3 is not discovered."""
        pet = PetState()
        spawn = _pet_spawn(x=5.0, y=5.0, level=20)
        state = make_game_state(x=0.0, y=0.0, level=10, spawns=(spawn,))
        pet.update(state)
        assert pet.alive is False

    def test_picks_closest_pet(self) -> None:
        """When multiple pets nearby, picks the closest."""
        pet = PetState()
        far_pet = _pet_spawn(spawn_id=201, name="Farbot123", x=50.0, y=50.0, level=10)
        close_pet = _pet_spawn(spawn_id=202, name="Closey123", x=5.0, y=5.0, level=10)
        state = make_game_state(x=0.0, y=0.0, level=10, spawns=(far_pet, close_pet))
        pet.update(state)
        assert pet.spawn_id == 202
        assert pet.name == "Closey123"

    def test_no_spawns_no_discovery(self) -> None:
        pet = PetState()
        state = make_game_state(x=0.0, y=0.0, spawns=())
        pet.update(state)
        assert pet.alive is False
        assert pet.spawn_id is None


# ---------------------------------------------------------------------------
# just_died
# ---------------------------------------------------------------------------


class TestJustDied:
    def test_just_died_transition(self) -> None:
        pet = PetState(alive=True, prev_alive=True, spawn_id=200, name="Xantik123")
        # Tick where pet disappears
        state = make_game_state(x=0.0, y=0.0, spawns=())
        pet.update(state)
        assert pet.just_died() is True

    def test_not_died_if_still_alive(self) -> None:
        pet = PetState(alive=True, prev_alive=True, spawn_id=200, name="Xantik123")
        spawn = _pet_spawn()
        state = make_game_state(x=10.0, y=10.0, spawns=(spawn,))
        pet.update(state)
        assert pet.just_died() is False

    def test_not_died_if_never_alive(self) -> None:
        pet = PetState(alive=False, prev_alive=False)
        assert pet.just_died() is False
