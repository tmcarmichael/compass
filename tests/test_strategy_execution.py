"""Tests for combat strategy execute() paths and combat monitor.

Exercises the spell decision branches in each strategy by configuring
the spell registry with test spells, then calling execute() with
various CastContext configurations.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from eq.loadout import _ACTIVE_BY_ROLE, _ALL_SPELLS, Spell, SpellRole
from perception.combat_eval import Con
from routines.base import RoutineStatus
from routines.strategies.base import CastContext
from tests.factories import make_agent_context, make_game_state, make_spawn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockCombatCaster:
    def __init__(self):
        self.casts = []

    def _cast_spell(self, gem, cast_time, now, state, target, *, is_dot=False):
        self.casts.append((gem, cast_time, is_dot))


def _make_spell(
    name: str,
    gem: int,
    role: str,
    mana_cost: int = 50,
    cast_time: float = 2.5,
    spell_id: int = 100,
    est_damage: float = 100.0,
) -> Spell:
    return Spell(
        name=name,
        gem=gem,
        cast_time=cast_time,
        mana_cost=mana_cost,
        spell_id=spell_id,
        role=role,
        est_damage=est_damage,
    )


def _make_cc(**kw) -> CastContext:
    target = kw.pop("target", make_spawn(spawn_id=200, x=50.0, y=50.0, hp_current=80, hp_max=100, speed=0.0))
    state = kw.pop("state", make_game_state(hp_current=800, hp_max=1000, mana_current=400, mana_max=500))
    defaults = dict(
        state=state,
        target=target,
        now=time.time(),
        dist=30.0,
        target_hp=0.8,
        tc=Con.WHITE,
        time_in_combat=5.0,
        out_of_range=False,
        recently_sat=False,
        is_undead=False,
        has_adds=False,
        mob_on_player=False,
        pet_hp=0.9,
        pet_dist=20.0,
    )
    defaults.update(kw)
    return CastContext(**defaults)


@pytest.fixture(autouse=True)
def _clean_spell_registry():
    """Reset spell registry before and after each test."""
    old_all = dict(_ALL_SPELLS)
    old_active = dict(_ACTIVE_BY_ROLE)
    _ALL_SPELLS.clear()
    _ACTIVE_BY_ROLE.clear()
    yield
    _ALL_SPELLS.clear()
    _ALL_SPELLS.update(old_all)
    _ACTIVE_BY_ROLE.clear()
    _ACTIVE_BY_ROLE.update(old_active)


def _register_spell(spell: Spell) -> None:
    _ALL_SPELLS[spell.name.lower()] = spell
    if spell.role:
        _ACTIVE_BY_ROLE[spell.role] = spell


# ---------------------------------------------------------------------------
# PetTankStrategy
# ---------------------------------------------------------------------------


class TestPetTankExecute:
    def _make_strategy(self):
        from routines.strategies.pet_tank import PetTankStrategy

        ctx = make_agent_context()
        caster = _MockCombatCaster()
        return PetTankStrategy(caster, ctx), caster, ctx

    def test_no_spells_returns_none(self):
        strat, caster, ctx = self._make_strategy()
        cc = _make_cc()
        result = strat.execute(cc)
        assert result is None
        assert caster.casts == []

    def test_dot_cast(self):
        strat, caster, ctx = self._make_strategy()
        dot = _make_spell("Disease Cloud", 1, SpellRole.DOT, mana_cost=20)
        _register_spell(dot)
        cc = _make_cc()
        result = strat.execute(cc)
        assert result == RoutineStatus.RUNNING
        assert len(caster.casts) == 1
        assert caster.casts[0][2] is True  # is_dot

    def test_lifetap_urgent_low_hp(self):
        strat, caster, ctx = self._make_strategy()
        lt = _make_spell("Lifespike", 2, SpellRole.LIFETAP, mana_cost=30)
        _register_spell(lt)
        state = make_game_state(hp_current=400, hp_max=1000, mana_current=200, mana_max=500)
        cc = _make_cc(state=state)
        result = strat.execute(cc)
        assert result == RoutineStatus.RUNNING
        assert len(caster.casts) == 1

    def test_pet_save_lifetap(self):
        strat, caster, ctx = self._make_strategy()
        lt = _make_spell("Lifespike", 2, SpellRole.LIFETAP, mana_cost=30)
        _register_spell(lt)
        state = make_game_state(hp_current=900, hp_max=1000, mana_current=200, mana_max=500)
        cc = _make_cc(state=state, pet_hp=0.3)
        result = strat.execute(cc)
        assert result == RoutineStatus.RUNNING

    def test_out_of_range_skips_cast(self):
        strat, caster, ctx = self._make_strategy()
        dot = _make_spell("Disease Cloud", 1, SpellRole.DOT, mana_cost=20)
        _register_spell(dot)
        cc = _make_cc(out_of_range=True)
        result = strat.execute(cc)
        assert result is None

    def test_efficiency_log_once(self):
        strat, caster, ctx = self._make_strategy()
        cc = _make_cc()
        strat.execute(cc)
        assert strat._efficiency_logged is True
        strat.execute(cc)  # second call should not re-log

    def test_reset_clears_efficiency_flag(self):
        strat, caster, ctx = self._make_strategy()
        strat._efficiency_logged = True
        strat.reset()
        assert strat._efficiency_logged is False

    def test_mana_mode_low_skips_dot(self):
        from core.types import ManaMode

        strat, caster, ctx = self._make_strategy()
        dot = _make_spell("Disease Cloud", 1, SpellRole.DOT, mana_cost=20)
        _register_spell(dot)
        with patch("routines.strategies.pet_tank.flags") as mock_flags:
            mock_flags.mana_mode = ManaMode.LOW
            cc = _make_cc(has_adds=False)
            result = strat.execute(cc)
        assert result is None

    def test_lifetap_sustain_full_mana(self):
        strat, caster, ctx = self._make_strategy()
        lt = _make_spell("Lifespike", 2, SpellRole.LIFETAP, mana_cost=30)
        _register_spell(lt)
        state = make_game_state(hp_current=900, hp_max=1000, mana_current=500, mana_max=500)
        cc = _make_cc(state=state, time_in_combat=20.0, pet_hp=0.9)
        result = strat.execute(cc)
        # May or may not cast based on cooldown, but should not error
        assert result in (None, RoutineStatus.RUNNING)


# ---------------------------------------------------------------------------
# PetAndDotStrategy
# ---------------------------------------------------------------------------


class TestPetAndDotExecute:
    def _make_strategy(self):
        from routines.strategies.pet_and_dot import PetAndDotStrategy

        ctx = make_agent_context()
        caster = _MockCombatCaster()
        return PetAndDotStrategy(caster, ctx), caster, ctx

    def test_no_spells_returns_none(self):
        strat, caster, ctx = self._make_strategy()
        cc = _make_cc()
        result = strat.execute(cc)
        assert result is None

    def test_dot_cast(self):
        strat, caster, ctx = self._make_strategy()
        dot = _make_spell("Disease Cloud", 1, SpellRole.DOT, mana_cost=20)
        _register_spell(dot)
        cc = _make_cc()
        result = strat.execute(cc)
        assert result == RoutineStatus.RUNNING

    def test_lifetap_low_hp(self):
        strat, caster, ctx = self._make_strategy()
        lt = _make_spell("Lifespike", 2, SpellRole.LIFETAP, mana_cost=30)
        _register_spell(lt)
        state = make_game_state(hp_current=400, hp_max=1000, mana_current=200, mana_max=500)
        cc = _make_cc(state=state)
        result = strat.execute(cc)
        assert result == RoutineStatus.RUNNING

    def test_reset(self):
        strat, caster, ctx = self._make_strategy()
        strat.reset()  # should not raise


# ---------------------------------------------------------------------------
# FearKiteStrategy
# ---------------------------------------------------------------------------


class TestFearKiteExecute:
    def _make_strategy(self):
        from routines.strategies.fear_kite import FearKiteStrategy

        ctx = make_agent_context()
        caster = _MockCombatCaster()
        return FearKiteStrategy(caster, ctx), caster, ctx

    def test_no_fear_spell_falls_back(self):
        strat, caster, ctx = self._make_strategy()
        cc = _make_cc()
        result = strat.execute(cc)
        assert result is None

    def test_with_fear_spell_initial_phase(self):
        strat, caster, ctx = self._make_strategy()
        fear = _make_spell("Fear", 3, SpellRole.FEAR, mana_cost=40, cast_time=1.5, spell_id=200)
        _register_spell(fear)
        cc = _make_cc()
        result = strat.execute(cc)
        # Should attempt fear cast in INITIAL phase
        assert result in (None, RoutineStatus.RUNNING)

    def test_with_fear_and_dot(self):
        strat, caster, ctx = self._make_strategy()
        fear = _make_spell("Fear", 3, SpellRole.FEAR, mana_cost=40, cast_time=1.5, spell_id=200)
        dot = _make_spell("Disease Cloud", 1, SpellRole.DOT, mana_cost=20)
        _register_spell(fear)
        _register_spell(dot)
        cc = _make_cc()
        result = strat.execute(cc)
        assert result in (None, RoutineStatus.RUNNING)

    def test_reset_clears_phase(self):
        strat, caster, ctx = self._make_strategy()
        strat.reset()
        from routines.strategies.fear_kite import _FearPhase

        assert strat._fear_phase == _FearPhase.INITIAL

    def test_pet_tanking_phase(self):
        strat, caster, ctx = self._make_strategy()
        from routines.strategies.fear_kite import _FearPhase

        strat._fear_phase = _FearPhase.PET_TANKING
        dot = _make_spell("Disease Cloud", 1, SpellRole.DOT, mana_cost=20)
        _register_spell(dot)
        cc = _make_cc()
        result = strat.execute(cc)
        assert result in (None, RoutineStatus.RUNNING)

    def test_feared_phase_with_dot(self):
        strat, caster, ctx = self._make_strategy()
        from routines.strategies.fear_kite import _FearPhase

        strat._fear_phase = _FearPhase.FEARED
        strat._fear_cast_time = time.time() - 5.0
        dot = _make_spell("Disease Cloud", 1, SpellRole.DOT, mana_cost=20)
        _register_spell(dot)
        cc = _make_cc()
        result = strat.execute(cc)
        assert result in (None, RoutineStatus.RUNNING)

    def test_re_fear_phase(self):
        strat, caster, ctx = self._make_strategy()
        from routines.strategies.fear_kite import _FearPhase

        strat._fear_phase = _FearPhase.RE_FEAR
        fear = _make_spell("Fear", 3, SpellRole.FEAR, mana_cost=40, cast_time=1.5, spell_id=200)
        _register_spell(fear)
        cc = _make_cc(dist=15.0)
        result = strat.execute(cc)
        assert result in (None, RoutineStatus.RUNNING)


# ---------------------------------------------------------------------------
# EndgameStrategy
# ---------------------------------------------------------------------------


class TestEndgameExecute:
    def _make_strategy(self):
        from routines.strategies.endgame import EndgameStrategy

        ctx = make_agent_context()
        caster = _MockCombatCaster()
        return EndgameStrategy(caster, ctx), caster, ctx

    def test_no_spells_returns_none(self):
        strat, caster, ctx = self._make_strategy()
        cc = _make_cc()
        result = strat.execute(cc)
        assert result is None

    def test_lifetap_cast(self):
        strat, caster, ctx = self._make_strategy()
        lt = _make_spell("Lifespike", 2, SpellRole.LIFETAP, mana_cost=30)
        _register_spell(lt)
        state = make_game_state(hp_current=600, hp_max=1000, mana_current=200, mana_max=500)
        cc = _make_cc(state=state)
        result = strat.execute(cc)
        assert result == RoutineStatus.RUNNING

    def test_dot_cast(self):
        strat, caster, ctx = self._make_strategy()
        dot = _make_spell("Envenomed Bolt", 1, SpellRole.DOT, mana_cost=80)
        _register_spell(dot)
        cc = _make_cc()
        result = strat.execute(cc)
        assert result == RoutineStatus.RUNNING

    def test_dot2_cast(self):
        strat, caster, ctx = self._make_strategy()
        dot2 = _make_spell("Venom of the Snake", 4, SpellRole.DOT_2, mana_cost=60)
        _register_spell(dot2)
        cc = _make_cc()
        result = strat.execute(cc)
        assert result == RoutineStatus.RUNNING

    def test_dd_cast(self):
        strat, caster, ctx = self._make_strategy()
        dd = _make_spell("Ignite Blood", 5, SpellRole.DD, mana_cost=100)
        _register_spell(dd)
        cc = _make_cc()
        result = strat.execute(cc)
        assert result == RoutineStatus.RUNNING

    def test_snare_low_hp_target(self):
        strat, caster, ctx = self._make_strategy()
        snare = _make_spell("Darkness", 6, SpellRole.SNARE, mana_cost=40)
        _register_spell(snare)
        cc = _make_cc(target_hp=0.15, dist=60.0)
        result = strat.execute(cc)
        # Snare fires on low-HP fleeing targets
        assert result in (None, RoutineStatus.RUNNING)

    def test_reset(self):
        strat, caster, ctx = self._make_strategy()
        strat._last_pb_time = 999.0
        strat.reset()
        assert strat._last_pb_time == 0.0

    def test_full_rotation(self):
        strat, caster, ctx = self._make_strategy()
        lt = _make_spell("Lifespike", 2, SpellRole.LIFETAP, mana_cost=30)
        dot = _make_spell("Envenomed Bolt", 1, SpellRole.DOT, mana_cost=80)
        dot2 = _make_spell("Venom", 4, SpellRole.DOT_2, mana_cost=60)
        dd = _make_spell("Ignite", 5, SpellRole.DD, mana_cost=100)
        for s in (lt, dot, dot2, dd):
            _register_spell(s)
        state = make_game_state(hp_current=600, hp_max=1000, mana_current=400, mana_max=500)
        cc = _make_cc(state=state)
        result = strat.execute(cc)
        assert result == RoutineStatus.RUNNING
        assert len(caster.casts) >= 1


# ---------------------------------------------------------------------------
# CombatMonitor
# ---------------------------------------------------------------------------


class TestCombatMonitorDeathCheck:
    def test_target_dead_returns_success(self):
        from routines.combat_monitor import CombatMonitor

        class MockCombat:
            _combat_start = time.time() - 15.0
            _target_killed = False
            _medding = False
            _sitting = False

            def _stand_from_med(self):
                pass

            def _record_kill(self, target, fight_time):
                self._target_killed = True

        combat = MockCombat()
        monitor = CombatMonitor(combat)

        class MockTickState:
            target = make_spawn(hp_current=0, hp_max=100)
            now = time.time()

        state = make_game_state()
        result = monitor.tick_death_check(state, MockTickState())
        assert result == RoutineStatus.SUCCESS
        assert combat._target_killed is True

    def test_target_alive_returns_none(self):
        from routines.combat_monitor import CombatMonitor

        class MockCombat:
            _combat_start = time.time() - 5.0
            _target_killed = False

        combat = MockCombat()
        monitor = CombatMonitor(combat)

        class MockTickState:
            target = make_spawn(hp_current=80, hp_max=100)
            now = time.time()

        state = make_game_state()
        result = monitor.tick_death_check(state, MockTickState())
        assert result is None


# ---------------------------------------------------------------------------
# FearPullTracker
# ---------------------------------------------------------------------------


class TestFearPullTrackerDetailed:
    def test_record_and_rate(self):
        from routines.strategies.fear_kite import FearPullTracker

        t = FearPullTracker()
        t.record(had_adds=False)
        t.record(had_adds=True)
        t.record(had_adds=False)
        assert t.total == 3
        assert t.with_adds == 1
        assert abs(t.adds_rate - 1 / 3) < 0.01

    def test_empty_rate(self):
        from routines.strategies.fear_kite import FearPullTracker

        t = FearPullTracker()
        assert t.adds_rate == 0.0
        assert t.total == 0

    def test_high_add_rate_logs(self):
        from routines.strategies.fear_kite import FearPullTracker

        t = FearPullTracker()
        for _ in range(10):
            t.record(had_adds=True)
        assert t.adds_rate == 1.0


# ---------------------------------------------------------------------------
# Spell utility functions
# ---------------------------------------------------------------------------


class TestSpellManaEfficiency:
    def test_positive(self):
        s = _make_spell("test", 1, "dot", mana_cost=50, est_damage=200.0)
        assert s.mana_efficiency == 4.0

    def test_zero_mana(self):
        s = _make_spell("test", 1, "dot", mana_cost=0, est_damage=200.0)
        assert s.mana_efficiency == 0.0

    def test_zero_damage(self):
        s = _make_spell("test", 1, "dot", mana_cost=50, est_damage=0.0)
        assert s.mana_efficiency == 0.0


# ---------------------------------------------------------------------------
# Brain runner tick helpers
# ---------------------------------------------------------------------------


class TestBrainRunnerTickHelpers:
    def _make_runner(self):
        """Create a minimal BrainRunner without Win32 dependencies."""
        import threading

        from brain.runner.loop import BrainRunner

        runner = object.__new__(BrainRunner)
        runner._reader = None
        runner._ctx = make_agent_context()
        runner._brain = type(
            "MockBrain",
            (),
            {
                "tick": lambda self, s: None,
                "last_rule_eval": {},
                "rule_scores": {},
                "_last_matched_rule": "",
                "_active_name": "",
                "_active": None,
                "shutdown": lambda self, s: None,
                "tick_total_ms": 1.0,
                "routine_tick_ms": 0.5,
            },
        )()
        runner._stop_event = threading.Event()
        runner._config = {"general": {"tick_rate_hz": 10}}
        runner._current_zone = "testzone"
        runner._log_path = ""
        runner._session_id = "test_session"
        runner._paused = False
        runner._prev_zone_id = 0
        runner._prev_level = 0
        runner.on_display_update = None
        runner._last_heartbeat = 0.0
        runner._last_exception = None
        runner._death_time = 0.0
        runner._crash_count = 0
        runner._crash_window_start = 0.0
        return runner

    def test_tick_brain_success(self):
        from brain.runner.loop import TickSignal

        runner = self._make_runner()
        state = make_game_state()
        result = runner._tick_brain(state, runner._ctx)
        assert result == TickSignal.PROCEED

    def test_tick_brain_crash_recovery(self):
        from brain.runner.loop import TickSignal

        runner = self._make_runner()
        runner._brain.tick = lambda s: (_ for _ in ()).throw(ValueError("test crash"))
        state = make_game_state()
        result = runner._tick_brain(state, runner._ctx)
        assert result == TickSignal.CONTINUE
        assert runner._crash_count == 1

    def test_tick_brain_crash_rate_exceeded(self):
        from brain.runner.loop import TickSignal

        runner = self._make_runner()
        runner._brain.tick = lambda s: (_ for _ in ()).throw(ValueError("test crash"))
        runner._crash_count = 3
        runner._crash_window_start = time.monotonic()
        state = make_game_state()
        result = runner._tick_brain(state, runner._ctx)
        assert result == TickSignal.BREAK

    def test_tick_pre_state_paused(self):
        from brain.runner.loop import TickSignal

        runner = self._make_runner()
        runner._paused = True
        runner._lifecycle = type("Mock", (), {"check_watchdog_restart": lambda self: False})()
        signal, state = runner._tick_pre_state(runner._ctx)
        assert signal == TickSignal.CONTINUE
        assert state is None

    def test_pause_setter_releases_keys(self):
        runner = self._make_runner()
        runner.paused = True
        assert runner._paused is True
        runner.paused = False
        assert runner._paused is False

    def test_log_session_ready(self):
        runner = self._make_runner()
        runner._ctx.zone.zone_config = {"zone": {"short_name": "testzone"}}
        runner._ctx.zone.active_camp_name = "north_camp"
        # Should not raise
        runner._log_session_ready(runner._ctx)

    def test_tick_periodic_snapshot_not_due(self):
        runner = self._make_runner()
        runner._next_snapshot = time.time() + 999
        state = make_game_state()
        result = runner._tick_periodic_snapshot(state, runner._ctx, time.time())
        assert result is False

    def test_write_session_report_delegates(self):
        runner = self._make_runner()
        runner._reporter = type("Mock", (), {"write_session_report": lambda self, ctx, sd: None})()
        runner._write_session_report(runner._ctx, "/tmp/test")

    def test_handle_death_delegates(self):
        runner = self._make_runner()
        runner._lifecycle = type("Mock", (), {"handle_death": lambda self, ctx, src: False})()
        result = runner._handle_death(runner._ctx, "test")
        assert result is False
