"""Unit tests for pure functions extracted from routines.

These functions are testable without motor, state machines, or mocking.
Each test is pure input/output with parametrized edge cases.
"""

from __future__ import annotations

import pytest

from routines.combat import classify_dot_fizzle
from routines.flee import should_attempt_gate
from routines.pull import choose_pull_strategy, verify_cast_landed
from routines.rest import should_cast_regen_buff, should_exit_rest

# ---------------------------------------------------------------------------
# classify_dot_fizzle (combat.py)
# ---------------------------------------------------------------------------


class TestClassifyDotFizzle:
    @pytest.mark.parametrize(
        "cast_result, fizzle_count, dist, expected",
        [
            ("los_blocked", 0, 30.0, "LOS_SUPPRESS"),
            ("fizzle", 0, 50.0, "FIZZLE_RETRY"),
            ("must_stand", 0, 50.0, "MUST_STAND"),
            ("interrupted", 0, 50.0, "INTERRUPTED_BACKSTEP"),
            ("", 3, 30.0, "SILENT_SIDESTEP"),  # 3+ fizzles + close range
            ("", 3, 50.0, "SILENT_REFACE"),  # 3+ fizzles but far
            ("", 0, 30.0, "SILENT_REFACE"),  # no fizzles
            ("", 2, 30.0, "SILENT_REFACE"),  # <3 fizzles
        ],
        ids=[
            "los_blocked",
            "fizzle_retry",
            "must_stand",
            "interrupted_backstep",
            "silent_sidestep_close",
            "silent_reface_far",
            "silent_reface_no_fizzles",
            "silent_reface_few_fizzles",
        ],
    )
    def test_classification(self, cast_result: str, fizzle_count: int, dist: float, expected: str) -> None:
        assert classify_dot_fizzle(cast_result, fizzle_count, dist) == expected


# ---------------------------------------------------------------------------
# verify_cast_landed (pull.py)
# ---------------------------------------------------------------------------


class TestVerifyCastLanded:
    def test_landed_when_mana_dropped(self) -> None:
        assert verify_cast_landed(500, 480, 5, 0, 4, False) == "LANDED"

    def test_los_blocked(self) -> None:
        assert verify_cast_landed(500, 500, 5, 0, 4, True) == "LOS_BLOCKED"

    def test_max_retries(self) -> None:
        assert verify_cast_landed(500, 500, 5, 4, 4, False) == "MAX_RETRIES"

    def test_fizzle_skip_after_2(self) -> None:
        assert verify_cast_landed(500, 500, 5, 2, 4, False) == "FIZZLE_SKIP"

    def test_fizzle_retry_first_attempt(self) -> None:
        assert verify_cast_landed(500, 500, 5, 0, 4, False) == "FIZZLE_RETRY"

    def test_fizzle_retry_second_attempt(self) -> None:
        assert verify_cast_landed(500, 500, 5, 1, 4, False) == "FIZZLE_RETRY"

    @pytest.mark.parametrize("drop", [5, 10, 50, 200])
    def test_any_sufficient_drop_is_landed(self, drop: int) -> None:
        assert verify_cast_landed(500, 500 - drop, 5, 0, 4, False) == "LANDED"

    def test_insufficient_drop_is_fizzle(self) -> None:
        assert verify_cast_landed(500, 497, 5, 0, 4, False) == "FIZZLE_RETRY"


# ---------------------------------------------------------------------------
# choose_pull_strategy (pull.py)
# ---------------------------------------------------------------------------


class TestChoosePullStrategy:
    def test_abort_on_high_adds(self) -> None:
        result = choose_pull_strategy(
            tc=__import__("perception.combat_eval", fromlist=["Con"]).Con.WHITE,
            nearby_count=0,
            learned_avg_adds=3.0,
            has_spell_candidates=True,
            is_fear_kite=False,
            has_fear=False,
            fear_affordable=False,
        )
        assert result == "ABORT"

    def test_pet_only_for_light_blue(self) -> None:
        from perception.combat_eval import Con

        result = choose_pull_strategy(
            tc=Con.LIGHT_BLUE,
            nearby_count=0,
            learned_avg_adds=None,
            has_spell_candidates=True,
            is_fear_kite=False,
            has_fear=False,
            fear_affordable=False,
        )
        assert result == "PET_ONLY"

    def test_pet_only_when_no_spells(self) -> None:
        from perception.combat_eval import Con

        result = choose_pull_strategy(
            tc=Con.WHITE,
            nearby_count=0,
            learned_avg_adds=None,
            has_spell_candidates=False,
            is_fear_kite=False,
            has_fear=False,
            fear_affordable=False,
        )
        assert result == "PET_ONLY"

    def test_fear_pull_in_fear_kite_mode(self) -> None:
        from perception.combat_eval import Con

        result = choose_pull_strategy(
            tc=Con.WHITE,
            nearby_count=0,
            learned_avg_adds=None,
            has_spell_candidates=True,
            is_fear_kite=True,
            has_fear=True,
            fear_affordable=True,
        )
        assert result == "FEAR_PULL"

    def test_spell_first_when_clustered(self) -> None:
        from perception.combat_eval import Con

        result = choose_pull_strategy(
            tc=Con.WHITE,
            nearby_count=3,
            learned_avg_adds=None,
            has_spell_candidates=True,
            is_fear_kite=False,
            has_fear=False,
            fear_affordable=False,
        )
        assert result == "SPELL_FIRST"

    def test_spell_first_when_learned_adds_high(self) -> None:
        from perception.combat_eval import Con

        result = choose_pull_strategy(
            tc=Con.WHITE,
            nearby_count=0,
            learned_avg_adds=2.5,
            has_spell_candidates=True,
            is_fear_kite=False,
            has_fear=False,
            fear_affordable=False,
        )
        assert result == "SPELL_FIRST"


# ---------------------------------------------------------------------------
# should_attempt_gate (flee.py)
# ---------------------------------------------------------------------------


class TestShouldAttemptGate:
    def test_gate_when_pet_dead_and_mana_sufficient(self) -> None:
        assert (
            should_attempt_gate(
                pet_alive=False,
                has_gate_spell=True,
                gate_gem_set=True,
                mana_current=200,
                gate_mana_cost=150,
            )
            is True
        )

    def test_no_gate_when_pet_alive(self) -> None:
        assert (
            should_attempt_gate(
                pet_alive=True,
                has_gate_spell=True,
                gate_gem_set=True,
                mana_current=200,
                gate_mana_cost=150,
            )
            is False
        )

    def test_no_gate_when_no_spell(self) -> None:
        assert (
            should_attempt_gate(
                pet_alive=False,
                has_gate_spell=False,
                gate_gem_set=True,
                mana_current=200,
                gate_mana_cost=150,
            )
            is False
        )

    def test_no_gate_when_gem_not_set(self) -> None:
        assert (
            should_attempt_gate(
                pet_alive=False,
                has_gate_spell=True,
                gate_gem_set=False,
                mana_current=200,
                gate_mana_cost=150,
            )
            is False
        )

    def test_no_gate_when_mana_insufficient(self) -> None:
        assert (
            should_attempt_gate(
                pet_alive=False,
                has_gate_spell=True,
                gate_gem_set=True,
                mana_current=100,
                gate_mana_cost=150,
            )
            is False
        )

    def test_gate_at_exact_mana_cost(self) -> None:
        assert (
            should_attempt_gate(
                pet_alive=False,
                has_gate_spell=True,
                gate_gem_set=True,
                mana_current=150,
                gate_mana_cost=150,
            )
            is True
        )


# ---------------------------------------------------------------------------
# should_cast_regen_buff (rest.py)
# ---------------------------------------------------------------------------


class TestShouldCastRegenBuff:
    def test_cast_when_full_hp_low_mana(self) -> None:
        assert (
            should_cast_regen_buff(
                hp_pct=1.0,
                mana_pct=0.20,
                mana_current=100,
                regen_mana_cost=50,
                is_sitting=False,
            )
            is True
        )

    def test_no_cast_when_hp_not_full(self) -> None:
        assert (
            should_cast_regen_buff(
                hp_pct=0.95,
                mana_pct=0.20,
                mana_current=100,
                regen_mana_cost=50,
                is_sitting=False,
            )
            is False
        )

    def test_no_cast_when_mana_above_threshold(self) -> None:
        assert (
            should_cast_regen_buff(
                hp_pct=1.0,
                mana_pct=0.50,
                mana_current=250,
                regen_mana_cost=50,
                is_sitting=False,
            )
            is False
        )

    def test_no_cast_when_insufficient_mana(self) -> None:
        assert (
            should_cast_regen_buff(
                hp_pct=1.0,
                mana_pct=0.10,
                mana_current=30,
                regen_mana_cost=50,
                is_sitting=False,
            )
            is False
        )

    def test_no_cast_when_already_sitting(self) -> None:
        assert (
            should_cast_regen_buff(
                hp_pct=1.0,
                mana_pct=0.20,
                mana_current=100,
                regen_mana_cost=50,
                is_sitting=True,
            )
            is False
        )


# ---------------------------------------------------------------------------
# should_exit_rest (rest.py)
# ---------------------------------------------------------------------------


class TestShouldExitRest:
    def test_exit_when_all_targets_met(self) -> None:
        assert (
            should_exit_rest(
                hp_pct=0.95,
                mana_pct=0.80,
                mana_max=500,
                hp_target=0.92,
                mana_target=0.60,
                pet_hp_pct=0.95,
            )
            is True
        )

    def test_stay_when_hp_low(self) -> None:
        assert (
            should_exit_rest(
                hp_pct=0.50,
                mana_pct=0.80,
                mana_max=500,
                hp_target=0.92,
                mana_target=0.60,
                pet_hp_pct=0.95,
            )
            is False
        )

    def test_stay_when_mana_low(self) -> None:
        assert (
            should_exit_rest(
                hp_pct=0.95,
                mana_pct=0.30,
                mana_max=500,
                hp_target=0.92,
                mana_target=0.60,
                pet_hp_pct=0.95,
            )
            is False
        )

    def test_stay_when_pet_hp_low(self) -> None:
        assert (
            should_exit_rest(
                hp_pct=0.95,
                mana_pct=0.80,
                mana_max=500,
                hp_target=0.92,
                mana_target=0.60,
                pet_hp_pct=0.70,
            )
            is False
        )

    def test_exit_when_no_pet(self) -> None:
        assert (
            should_exit_rest(
                hp_pct=0.95,
                mana_pct=0.80,
                mana_max=500,
                hp_target=0.92,
                mana_target=0.60,
                pet_hp_pct=None,
            )
            is True
        )

    def test_exit_when_mana_max_zero(self) -> None:
        """Melee class with no mana should exit when HP is ok."""
        assert (
            should_exit_rest(
                hp_pct=0.95,
                mana_pct=0.0,
                mana_max=0,
                hp_target=0.92,
                mana_target=0.60,
                pet_hp_pct=None,
            )
            is True
        )

    @pytest.mark.parametrize(
        "hp,mana,pet",
        [
            (0.92, 0.60, 0.90),  # exact thresholds
            (1.0, 1.0, 1.0),  # fully recovered
            (0.93, 0.61, 0.91),  # just above
        ],
    )
    def test_exit_at_boundary_values(self, hp: float, mana: float, pet: float) -> None:
        assert (
            should_exit_rest(
                hp_pct=hp,
                mana_pct=mana,
                mana_max=500,
                hp_target=0.92,
                mana_target=0.60,
                pet_hp_pct=pet,
            )
            is True
        )
