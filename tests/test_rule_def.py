"""Tests for brain.rule_def -- RuleDef, Consideration, score_from_considerations.

RuleDef is a structured rule definition. Consideration + score_from_considerations
implement Phase 4 declarative scoring via weighted geometric mean.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

from brain.rule_def import Consideration, RuleDef, score_from_considerations
from tests.factories import make_game_state

# ---------------------------------------------------------------------------
# RuleDef dataclass
# ---------------------------------------------------------------------------


class TestRuleDef:
    def test_default_fields(self) -> None:
        r = RuleDef(
            name="test_rule",
            condition=lambda s: True,
            routine=MagicMock(),
        )
        assert r.name == "test_rule"
        assert r.failure_cooldown == 0.0
        assert r.emergency is False
        assert r.max_lock_seconds == 0.0
        assert r.tier == 0
        assert r.weight == 1.0
        assert r.considerations == []
        assert r.breaker_max_failures == 5
        assert r.breaker_window == 300.0
        assert r.breaker_recovery == 120.0

    def test_score_fn_default_returns_zero(self) -> None:
        r = RuleDef(
            name="test",
            condition=lambda s: True,
            routine=MagicMock(),
        )
        state = make_game_state()
        assert r.score_fn(state) == 0.0

    def test_custom_score_fn(self) -> None:
        r = RuleDef(
            name="test",
            condition=lambda s: True,
            routine=MagicMock(),
            score_fn=lambda s: 0.75,
        )
        state = make_game_state()
        assert r.score_fn(state) == 0.75


# ---------------------------------------------------------------------------
# Consideration dataclass
# ---------------------------------------------------------------------------


class TestConsideration:
    def test_fields(self) -> None:
        c = Consideration(
            name="mana",
            input_fn=lambda s, ctx: s.mana_pct,
            curve=lambda x: x,
            weight=2.0,
        )
        assert c.name == "mana"
        assert c.weight == 2.0

    def test_default_weight(self) -> None:
        c = Consideration(
            name="hp",
            input_fn=lambda s, ctx: s.hp_pct,
            curve=lambda x: x,
        )
        assert c.weight == 1.0


# ---------------------------------------------------------------------------
# score_from_considerations
# ---------------------------------------------------------------------------


class TestScoreFromConsiderations:
    def _make_ctx(self) -> object:
        return SimpleNamespace()

    def test_empty_considerations_returns_zero(self) -> None:
        state = make_game_state()
        assert score_from_considerations([], state, self._make_ctx()) == 0.0

    def test_single_consideration(self) -> None:
        state = make_game_state(hp_current=800, hp_max=1000)
        c = Consideration(
            name="hp",
            input_fn=lambda s, ctx: s.hp_pct,
            curve=lambda x: x,  # identity
        )
        score = score_from_considerations([c], state, self._make_ctx())
        assert abs(score - 0.8) < 0.01

    def test_hard_gate_zero(self) -> None:
        """A consideration returning 0 acts as a hard gate."""
        state = make_game_state()
        c1 = Consideration(name="always_zero", input_fn=lambda s, ctx: 0.0, curve=lambda x: x)
        c2 = Consideration(name="always_one", input_fn=lambda s, ctx: 1.0, curve=lambda x: 1.0)
        score = score_from_considerations([c1, c2], state, self._make_ctx())
        assert score == 0.0

    def test_weighted_geometric_mean(self) -> None:
        """Two considerations with different weights produce weighted geometric mean."""
        state = make_game_state()
        c1 = Consideration(
            name="a",
            input_fn=lambda s, ctx: 0.0,
            curve=lambda x: 0.5,
            weight=1.0,
        )
        c2 = Consideration(
            name="b",
            input_fn=lambda s, ctx: 0.0,
            curve=lambda x: 0.8,
            weight=1.0,
        )
        score = score_from_considerations([c1, c2], state, self._make_ctx())
        # Geometric mean of 0.5 and 0.8 with equal weights:
        expected = math.exp((1.0 * math.log(0.5) + 1.0 * math.log(0.8)) / 2.0)
        assert abs(score - expected) < 0.001

    def test_unequal_weights(self) -> None:
        state = make_game_state()
        c1 = Consideration(
            name="high_weight",
            input_fn=lambda s, ctx: 0.0,
            curve=lambda x: 0.9,
            weight=3.0,
        )
        c2 = Consideration(
            name="low_weight",
            input_fn=lambda s, ctx: 0.0,
            curve=lambda x: 0.1,
            weight=1.0,
        )
        score = score_from_considerations([c1, c2], state, self._make_ctx())
        expected = math.exp((3.0 * math.log(0.9) + 1.0 * math.log(0.1)) / 4.0)
        assert abs(score - expected) < 0.001

    def test_all_ones_returns_one(self) -> None:
        state = make_game_state()
        considerations = [
            Consideration(name="a", input_fn=lambda s, ctx: 0, curve=lambda x: 1.0),
            Consideration(name="b", input_fn=lambda s, ctx: 0, curve=lambda x: 1.0),
        ]
        score = score_from_considerations(considerations, state, self._make_ctx())
        assert abs(score - 1.0) < 0.001

    def test_zero_weight_sum_returns_zero(self) -> None:
        """All zero-weight considerations -> weight_sum is 0 -> return 0.0."""
        state = make_game_state()
        c = Consideration(
            name="zero_weight",
            input_fn=lambda s, ctx: 0.0,
            curve=lambda x: 0.5,
            weight=0.0,
        )
        score = score_from_considerations([c], state, self._make_ctx())
        assert score == 0.0
