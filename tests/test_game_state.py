"""Tests for perception.state  -- frozen snapshot contracts.

GameState and SpawnData are the concurrency boundary: frozen dataclasses
that cannot be mutated after creation. These tests verify the contract
and the computed properties that every layer depends on.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from hypothesis import given

from perception.state import GameState
from tests.factories import make_game_state, make_spawn, st_game_state

# ---------------------------------------------------------------------------
# GameState  -- frozen / slots contract
# ---------------------------------------------------------------------------


class TestGameStateFrozen:
    def test_cannot_assign_field(self, game_state: GameState) -> None:
        mutable: Any = game_state
        with pytest.raises(dataclasses.FrozenInstanceError):
            mutable.hp_current = 999

    def test_has_slots(self, game_state: GameState) -> None:
        assert not hasattr(game_state, "__dict__")


# ---------------------------------------------------------------------------
# GameState  -- property invariants (hypothesis)
# ---------------------------------------------------------------------------


class TestGameStateInvariants:
    @given(state=st_game_state)
    def test_hp_pct_non_negative(self, state: GameState) -> None:
        assert state.hp_pct >= 0.0

    @given(state=st_game_state)
    def test_mana_pct_non_negative(self, state: GameState) -> None:
        assert state.mana_pct >= 0.0

    @given(state=st_game_state)
    def test_zero_max_safe(self, state: GameState) -> None:
        """hp_max=0 and mana_max=0 don't crash, return safe defaults."""
        s = make_game_state(hp_max=0, mana_max=0)
        assert s.hp_pct == 1.0
        assert s.mana_pct == 1.0


# ---------------------------------------------------------------------------
# GameState  -- computed properties
# ---------------------------------------------------------------------------


class TestGameStateProperties:
    def test_hp_pct_full(self) -> None:
        s = make_game_state(hp_current=1000, hp_max=1000)
        assert s.hp_pct == pytest.approx(1.0)

    def test_hp_pct_half(self) -> None:
        s = make_game_state(hp_current=500, hp_max=1000)
        assert s.hp_pct == pytest.approx(0.5)

    def test_hp_pct_zero_max_safe(self) -> None:
        s = make_game_state(hp_current=0, hp_max=0)
        assert s.hp_pct == pytest.approx(1.0)

    def test_mana_pct_zero_max_safe(self) -> None:
        s = make_game_state(mana_current=0, mana_max=0)
        assert s.mana_pct == pytest.approx(1.0)

    def test_has_target_with_spawn(self) -> None:
        s = make_game_state(target=make_spawn())
        assert s.has_target is True

    def test_has_target_none(self) -> None:
        s = make_game_state(target=None)
        assert s.has_target is False

    @pytest.mark.parametrize(
        "stand_state, expected",
        [
            (0, False),
            (1, True),
            (2, True),
            (3, False),
        ],
    )
    def test_is_sitting(self, stand_state: int, expected: bool) -> None:
        s = make_game_state(stand_state=stand_state)
        assert s.is_sitting is expected

    def test_money_total_cp(self) -> None:
        s = make_game_state(money_pp=1, money_gp=2, money_sp=3, money_cp=4)
        assert s.money_total_cp == 1234

    def test_has_buff(self) -> None:
        s = make_game_state(buffs=((101, 5), (202, 10)))
        assert s.has_buff(101) is True
        assert s.has_buff(999) is False

    def test_buff_ticks(self) -> None:
        s = make_game_state(buffs=((101, 5),))
        assert s.buff_ticks(101) == 5
        assert s.buff_ticks(999) == 0

    def test_nearby_npcs_filters(self) -> None:
        npc = make_spawn(spawn_id=1, spawn_type=1)
        player = make_spawn(spawn_id=2, spawn_type=0)
        corpse = make_spawn(spawn_id=3, spawn_type=2)
        s = make_game_state(spawns=(npc, player, corpse))
        assert len(s.nearby_npcs) == 1
        assert s.nearby_npcs[0].spawn_id == 1


# ---------------------------------------------------------------------------
# SpawnData
# ---------------------------------------------------------------------------


class TestSpawnData:
    def test_frozen(self) -> None:
        s: Any = make_spawn()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.name = "changed"

    @pytest.mark.parametrize(
        "spawn_type, prop, expected",
        [
            (0, "is_player", True),
            (1, "is_npc", True),
            (2, "is_corpse", True),
            (3, "is_player_corpse", True),
        ],
    )
    def test_type_properties(self, spawn_type: int, prop: str, expected: bool) -> None:
        s = make_spawn(spawn_type=spawn_type)
        assert getattr(s, prop) is expected

    @pytest.mark.parametrize(
        "body_state, prop",
        [
            ("f", "is_feigning"),
            ("d", "is_dead_body"),
            ("i", "is_invisible"),
        ],
    )
    def test_body_state_properties(self, body_state: str, prop: str) -> None:
        s = make_spawn(body_state=body_state)
        assert getattr(s, prop) is True
