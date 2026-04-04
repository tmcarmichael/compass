"""Survival rules: DEATH_RECOVERY, FEIGN_DEATH, FLEE, REST, EVADE."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from brain.rule_def import Consideration
from brain.rules.skip_log import SkipLog
from brain.scoring.curves import inverse_linear, inverse_logistic
from core.features import flags
from eq.loadout import SpellRole, get_spell_by_role
from perception.state import GameState
from routines.death_recovery import DeathRecoveryRoutine
from routines.feign_death import FeignDeathRoutine
from routines.flee import FleeRoutine
from routines.rest import RestRoutine

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.context_views import SurvivalView
    from brain.decision import Brain
    from brain.world.model import WorldModel
    from core.types import ReadStateFn
    from eq.loadout import Spell

# Helper functions are typed against SurvivalView to enforce that survival
# rules only access the subset of AgentContext they need.  See context_views.py.

log = logging.getLogger(__name__)
_skip = SkipLog(log)


@dataclass
class _SurvivalRuleState:
    """Mutable state shared across survival rule condition functions."""

    resting: bool = False
    learned_mana_logged: bool = False
    evade_logged: bool = False
    last_patrol_evade: float = 0.0


# -- Flee urgency thresholds (hysteresis) --
FLEE_URGENCY_ENTER = 0.65  # start fleeing at this urgency
FLEE_URGENCY_EXIT = 0.35  # stop fleeing below this


def _check_core_safety_floors(ctx: SurvivalView, state: GameState, label: str) -> bool | None:
    """Check the three core safety floors shared by FLEE and FEIGN_DEATH.

    Returns True if a floor is triggered, None if no floor fires.
    The three floors: HP critical, pet died in unwinnable fight, RED threat.
    """
    if state.hp_pct < 0.40:
        log.info("[DECISION] %s: HP %.0f%% < 40%% (safety floor)", label, state.hp_pct * 100)
        return True
    if ctx.pet.just_died() and ctx.in_active_combat:
        if not _fight_winnable(state):
            log.info("[DECISION] %s: pet died mid-combat (safety floor)", label)
            return True
    if ctx.threat.imminent_threat and ctx.threat.imminent_threat_con == "red":
        log.info("[DECISION] %s: RED threat imminent (safety floor)", label)
        return True
    return None


def _fight_winnable(state: GameState) -> bool:
    """True if player can finish the current fight without a pet.

    Conditions: player HP > 60% AND target npc HP < 50%.
    At these thresholds a few Lifespikes will finish the npc.
    """
    t = state.target
    if not t or t.hp_max <= 0:
        log.debug("[DECISION] _fight_winnable: no valid target")
        return False
    mob_hp: float = t.hp_current / max(t.hp_max, 1)
    winnable: bool = state.hp_pct > 0.60 and mob_hp < 0.50
    return winnable


def _count_damaged_npcs(ctx: SurvivalView, state: GameState) -> int:
    """Count damaged NPCs within 40u (for add detection)."""
    from brain.rules.skip_log import damaged_npcs_near

    return len(damaged_npcs_near(ctx, state, state.pos, 40))


def _mob_attacking_player(state: GameState) -> bool:
    """Return True if any NPC within 30u is targeting the player."""
    for sp in state.spawns:
        if (
            sp.is_npc
            and sp.hp_current > 0
            and sp.target_name == state.name
            and state.pos.dist_to(sp.pos) < 30
        ):
            return True
    return False


def _urgency_hp(hp_pct: float) -> float:
    """HP axis: power curve scaled so ~35% HP yields ~0.65."""
    result: float = ((1.0 - hp_pct) ** 1.8) * 1.45
    return result


def _urgency_pet_died(ctx: SurvivalView, state: GameState, in_combat: bool) -> float:
    """Pet died mid-combat with unwinnable fight: +0.4."""
    if in_combat and ctx.pet.just_died() and not _fight_winnable(state):
        return 0.4
    return 0.0


def _urgency_adds(ctx: SurvivalView, state: GameState, in_combat: bool) -> float:
    """Extra damaged NPCs nearby: +0.15 per add."""
    if not in_combat:
        return 0.0
    count = _count_damaged_npcs(ctx, state)
    return 0.15 * max(0, count - 1)


def _urgency_target_dying(state: GameState) -> float:
    """Target nearly dead (<15% HP): -0.25 to finish the defeat."""
    t = state.target
    if t and t.is_npc and t.hp_max > 0 and t.hp_current > 0:
        if t.hp_current / t.hp_max < 0.15:
            return -0.25
    return 0.0


def _urgency_learned_danger(ctx: SurvivalView, in_combat: bool) -> float:
    """Learned danger from FightHistory (>0.7): +0.2."""
    fh = ctx.fight_history
    if not fh or not in_combat:
        return 0.0
    name = ctx.defeat_tracker.last_fight_name
    if not name:
        return 0.0
    danger = fh.learned_danger(name)
    return 0.2 if danger is not None and danger > 0.7 else 0.0


def _urgency_pet_hp_low(ctx: SurvivalView) -> float:
    """Pet HP below 30%: +0.1."""
    world = ctx.world
    if ctx.pet.alive and world:
        pet_hp = world.pet_hp_pct
        if 0 <= pet_hp < 0.30:
            return 0.1
    return 0.0


def _urgency_red_threat(ctx: SurvivalView) -> float:
    """RED imminent threat: +0.5."""
    if ctx.threat.imminent_threat and ctx.threat.imminent_threat_con == "red":
        return 0.5
    return 0.0


def _urgency_mob_on_player(ctx: SurvivalView, state: GameState) -> float:
    """No pet + NPC attacking player: +0.5."""
    if not ctx.pet.alive and not ctx.combat.engaged and _mob_attacking_player(state):
        return 0.5
    return 0.0


def compute_flee_urgency(ctx: SurvivalView, state: GameState) -> float:
    """Composite flee urgency score (0.0 = safe, 1.0 = critical).

    Eight axes contribute additively, then clamp to [0, 1].
    """
    in_combat = ctx.in_active_combat

    hp = _urgency_hp(state.hp_pct)
    pet_died = _urgency_pet_died(ctx, state, in_combat)
    adds = _urgency_adds(ctx, state, in_combat)
    target_dying = _urgency_target_dying(state)
    danger = _urgency_learned_danger(ctx, in_combat)
    pet_low = _urgency_pet_hp_low(ctx)
    red = _urgency_red_threat(ctx)
    mob_on = _urgency_mob_on_player(ctx, state)

    result: float = max(0.0, min(1.0, hp + pet_died + adds + target_dying + danger + pet_low + red + mob_on))

    if result >= FLEE_URGENCY_ENTER:
        factors = []
        if hp > 0.01:
            factors.append(f"hp={hp:.2f}")
        if pet_died > 0:
            factors.append("pet_died")
        if adds > 0:
            factors.append(f"extra_npcs={adds / 0.15:.0f}")
        if red > 0:
            factors.append("RED_threat")
        if mob_on > 0:
            factors.append("mob_on_player")
        log.info(
            "[DECISION] Flee urgency: %.3f [%s] combat=%s",
            result,
            " ".join(factors) if factors else "hp_only",
            in_combat,
        )

    return result


def _next_pull_mana_estimate(ctx: SurvivalView) -> int | None:
    """Estimate mana cost for next pull from FightHistory.

    Returns average learned mana cost across all learned npc types,
    or None if insufficient data.
    """
    fh = ctx.fight_history
    if not fh:
        return None
    costs = []
    for k in fh.get_all_stats():
        if fh.has_learned(k):
            m = fh.learned_mana(k)
            if m is not None:
                costs.append(m)
    if not costs:
        return None
    return int(sum(costs) / len(costs))


def reset_flee_hysteresis(ctx: SurvivalView) -> None:
    """Reset flee hysteresis state on ctx.combat (for testing)."""
    ctx.combat.flee_urgency_active = False


def _flee_safety_floors(ctx: SurvivalView, state: GameState) -> bool:
    """Binary safety floors that always fire regardless of urgency score.

    These are structural guarantees: the agent cannot learn its way out
    of responding to lethal conditions.
    """
    # Core floors shared with feign death
    if _check_core_safety_floors(ctx, state, "flee_condition"):
        return True
    # Imminent threat with no pet to intercept
    if ctx.threat.imminent_threat and not ctx.pet.alive:
        log.info(
            "[DECISION] flee_condition: %s imminent threat + no pet (safety floor)",
            ctx.threat.imminent_threat_con,
        )
        return True
    # NPC attacking player with no pet (including mid-pull)
    if not ctx.pet.alive and (not ctx.combat.engaged or ctx.combat.pull_target_id is not None):
        if _mob_attacking_player(state):
            log.info("[DECISION] flee_condition: npc attacking with NO PET (safety floor)")
            return True
    # Multiple adds overwhelming the agent
    if ctx.in_active_combat:
        from brain.rules.skip_log import damaged_npcs_near

        damaged = damaged_npcs_near(ctx, state, state.pos, 40)
        if len(damaged) >= 3:
            log.info("[DECISION] flee_condition: %d damaged npcs (train, safety floor)", len(damaged))
            return True
        if len(damaged) >= 2:
            if not ctx.pet.alive:
                log.info("[DECISION] flee_condition: %d extra_npcs + pet DEAD (safety floor)", len(damaged))
                return True
            world = ctx.world
            pet_hp = world.pet_hp_pct if world else -1
            if pet_hp >= 0 and pet_hp < 0.70:
                log.info(
                    "[DECISION] flee_condition: %d extra_npcs + pet HP %.0f%% (safety floor)",
                    len(damaged),
                    pet_hp * 100,
                )
                return True
    return False


def flee_condition(ctx: SurvivalView, state: GameState) -> bool:
    """Standalone flee condition: urgency hysteresis + binary safety floors.

    Used by tests and can be called outside of the brain rule system.
    """
    urgency = compute_flee_urgency(ctx, state)

    # Hysteresis: enter at >= ENTER, exit at < EXIT
    if ctx.combat.flee_urgency_active:
        if urgency < FLEE_URGENCY_EXIT:
            log.info(
                "[DECISION] flee_condition: urgency %.3f < %.2f exit -- disengaging",
                urgency,
                FLEE_URGENCY_EXIT,
            )
            ctx.combat.flee_urgency_active = False
            return False
        return True
    if urgency >= FLEE_URGENCY_ENTER:
        log.info("[DECISION] flee_condition: urgency %.3f >= %.2f -- engaging", urgency, FLEE_URGENCY_ENTER)
        ctx.combat.flee_urgency_active = True
        return True

    # Binary safety floors (always fire, independent of urgency score)
    if _flee_safety_floors(ctx, state):
        ctx.combat.flee_urgency_active = True
        return True
    return False


def feign_death_condition(ctx: SurvivalView, state: GameState, fd_spell: Spell | None = None) -> bool:
    """Standalone feign death condition.

    Check if FD should trigger. Conditions:
      - fd_spell is not None, has 'Feign' in name, gem > 0
      - mana >= mana_cost
      - HP < 40% OR pet just died mid-combat OR RED threat imminent

    Args:
        ctx: Agent context with combat/pet/threat state.
        state: Current game state snapshot.
        fd_spell: Spell object for Feign Death, or None.

    Returns True if feign death should be attempted.
    """
    if fd_spell is None:
        log.debug("[DECISION] feign_death_condition: no FD spell")
        return False
    if not fd_spell.gem or "Feign" not in fd_spell.name:
        log.debug("[DECISION] feign_death_condition: FD not memorized or wrong spell")
        return False
    if state.mana_current < fd_spell.mana_cost:
        log.debug("[DECISION] feign_death_condition: mana %d < %d", state.mana_current, fd_spell.mana_cost)
        return False
    floor = _check_core_safety_floors(ctx, state, "feign_death_condition")
    if floor:
        return True
    return False


def rest_needs_check(
    ctx: SurvivalView, state: GameState, world: WorldModel | None = None
) -> tuple[bool, bool, bool]:
    """Check if rest is needed -- returns (hp_low, mana_low, pet_low).

    Standalone version of the rest entry logic from should_rest().
    Does NOT check combat state, threats, or hysteresis -- just the
    resource deficit checks.

    Args:
        ctx: Agent context with rest thresholds.
        state: Current game state snapshot.
        world: Optional world model for pet HP.

    Returns:
        (hp_low, mana_low, pet_low) booleans.
    """
    hp_deficit = state.hp_max - state.hp_current
    hp_low = state.hp_pct < ctx.rest_hp_entry and hp_deficit >= 5

    pet_low = False
    pet_hp = 1.0
    if world is None:
        world = ctx.world
    if ctx.pet.alive and world:
        pet_hp = world.pet_hp_pct if world.pet_hp_pct >= 0 else 1.0
        if pet_hp < 0.60:
            pet_low = True

    mana_low = state.mana_pct < ctx.rest_mana_entry if state.mana_max > 0 else False
    if mana_low and state.mana_pct > 0.20 and state.hp_pct >= 0.85 and pet_hp >= 0.70:
        mana_low = False  # pet can grind without mana

    return (hp_low, mana_low, pet_low)


# -- Module-level extracted condition/score functions --


def _should_death_recover(state: GameState, ctx: SurvivalView) -> bool:
    if not ctx.player.dead:
        _skip("DeathRecovery", "not dead")
        return False
    should: bool = flags.should_recover_death(ctx.player.deaths)
    return should


def _score_death_recovery(state: GameState, ctx: SurvivalView) -> float:
    if not ctx.player.dead:
        return 0.0
    return 1.0 if flags.should_recover_death(ctx.player.deaths) else 0.0


def _should_feign_death(state: GameState, ctx: SurvivalView) -> bool:
    if not flags.flee:
        _skip("FeignDeath", "flee disabled")
        return False
    fd_spell = get_spell_by_role(SpellRole.UTILITY)
    if not fd_spell or not fd_spell.gem or "Feign" not in fd_spell.name:
        _skip("FeignDeath", "no FD spell available")
        return False
    if state.mana_current < fd_spell.mana_cost:
        log.debug("[DECISION] FeignDeath skip: mana %d < %d", state.mana_current, fd_spell.mana_cost)
        return False
    floor = _check_core_safety_floors(ctx, state, "FeignDeath")
    if floor:
        return True
    _skip("FeignDeath", "no trigger condition met")
    return False


def _score_feign_death(state: GameState, ctx: SurvivalView) -> float:
    if not flags.flee:
        return 0.0
    fd_spell = get_spell_by_role(SpellRole.UTILITY)
    if not fd_spell or not fd_spell.gem or "Feign" not in fd_spell.name:
        return 0.0
    if state.mana_current < fd_spell.mana_cost:
        return 0.0
    if state.hp_pct < 0.40:
        urgency: float = inverse_logistic(state.hp_pct, 0.40, 12)
        return urgency
    if ctx.pet.just_died() and ctx.in_active_combat:
        if not _fight_winnable(state):
            return 1.0
    if ctx.threat.imminent_threat and ctx.threat.imminent_threat_con == "red":
        return 1.0
    return 0.0


def _should_flee(state: GameState, ctx: SurvivalView) -> bool:
    if not flags.flee:
        ctx.combat.flee_urgency_active = False
        _skip("Flee", "flee disabled")
        return False
    result = flee_condition(ctx, state)
    if not result:
        _skip("Flee", "no trigger conditions met")
    return result


def _score_flee(state: GameState, ctx: SurvivalView) -> float:
    if not flags.flee:
        return 0.0
    if _flee_safety_floors(ctx, state):
        return 1.0

    urgency = compute_flee_urgency(ctx, state)
    if ctx.combat.flee_urgency_active:
        return urgency if urgency >= FLEE_URGENCY_EXIT else 0.0
    return urgency if urgency >= FLEE_URGENCY_ENTER else 0.0


def _in_combat_dot_suppresses_rest(ctx: SurvivalView, state: GameState, rs: _SurvivalRuleState) -> bool:
    """Handle in_combat flag when not engaged (likely DoT/lich buff ticks).

    Returns True to suppress rest, False to allow it through.
    When pet is critically low, rest is allowed so it can receive heals.
    """
    if not (state.in_combat and not ctx.combat.engaged):
        return False
    # Allow rest when pet is critically low and no real attacker nearby.
    # The in_combat flag fires from DoT/lich buff ticks on the player --
    # suppressing rest here starves the pet of heals indefinitely.
    # RestRoutine.tick() handles actual attackers (stands + engages).
    pet_critical = False
    if ctx.pet.alive:
        _w = ctx.world
        if _w and 0 <= _w.pet_hp_pct < 0.50:
            pet_critical = True
    if not pet_critical:
        rs.resting = False
        _skip("Rest", "in_combat but not engaged")
        return True
    _skip("Rest", "in_combat (likely DoT) but pet critical -- allowing")
    return False


def _rest_hp_dropping(ctx: SurvivalView, state: GameState, rs: _SurvivalRuleState) -> bool:
    """Return True if HP is dropping while resting (possible attack)."""
    if not (rs.resting and ctx.player.rest_start_time > 0):
        return False
    rest_age = time.time() - ctx.player.rest_start_time
    if rest_age > 5.0 and state.hp_pct < ctx.player.last_rest_hp - 0.10:
        log.info(
            "[STATE] Rest: HP dropping while resting (entry=%.0f%% now=%.0f%%) -- standing (possible attack)",
            ctx.player.last_rest_hp * 100,
            state.hp_pct * 100,
        )
        rs.resting = False
        return True
    return False


def _rest_suppressed(ctx: SurvivalView, state: GameState, rs: _SurvivalRuleState) -> bool:
    """Return True if rest should be suppressed (disabled, combat, threats, etc.)."""
    if not flags.rest:
        rs.resting = False
        rs.learned_mana_logged = False
        _skip("Rest", "rest disabled")
        return True
    if ctx.in_active_combat:
        rs.resting = False
        _skip("Rest", "in active combat")
        return True
    now = time.time()
    # Suppress rest briefly after buff cast -- buff raises max HP,
    # making current HP% look low before natural regen catches up
    if now - ctx.player.last_buff_time < 10.0 and not rs.resting:
        _skip("Rest", "recently buffed (suppressed 10s)")
        return True
    # Suppress rest after flee -- npcs may still be nearby at guard area
    if now - ctx.player.last_flee_time < 10.0 and not rs.resting:
        _skip("Rest", "recently fled")
        return True
    if _in_combat_dot_suppresses_rest(ctx, state, rs):
        return True
    if _rest_hp_dropping(ctx, state, rs):
        return True
    if ctx.threat.imminent_threat:
        rs.resting = False
        _skip("Rest", "imminent threat")
        return True
    # Yield to EVADE when an evasion point is set -- EVADE is lower
    # priority than REST, so without this check REST blocks it
    if ctx.threat.evasion_point is not None:
        rs.resting = False
        _skip("Rest", "evasion active")
        return True
    world = ctx.world
    if world:
        if world.any_hostile_npc_within(20):
            rs.resting = False
            _skip("Rest", "hostile NPC within 20u")
            return True
        if world.threats_within(50):
            rs.resting = False
            _skip("Rest", "threat within 50u")
            return True
    # Suppress rest if engaged or targeting a damaged NPC (active fight).
    # Exclude our own pet -- pet heal targets it, and its HP < max is normal.
    target_is_fight = (
        state.target
        and state.target.is_npc
        and state.target.hp_current > 0
        and state.target.hp_current < state.target.hp_max
        and state.target.spawn_id != ctx.pet.spawn_id
    )
    if ctx.combat.engaged or target_is_fight:
        rs.resting = False
        _skip("Rest", "engaged or fighting NPC targeted")
        return True
    return False


def _rest_exit_check(ctx: SurvivalView, state: GameState, rs: _SurvivalRuleState) -> bool:
    """Check if rest thresholds are met and we should exit rest.

    Returns True if rest should end (thresholds met), False to keep resting.
    """
    hp_ok = state.hp_pct >= ctx.rest_hp_threshold
    # D-4: Learned mana rest exit -- exit rest earlier when we know
    # the next pull costs less mana than the default threshold.
    # Only rest SHORTER, never longer.
    mana_threshold = ctx.rest_mana_threshold
    if state.mana_max > 0:
        learned_cost = _next_pull_mana_estimate(ctx)
        if learned_cost is not None and state.mana_max > 0:
            # Need enough mana for pull + 20% safety buffer
            needed_pct = (learned_cost * 1.2) / state.mana_max
            # Clamp: minimum 40% mana (pet-tank needs mana for
            # lifetap/dot on next pull), never exceed default threshold
            needed_pct = max(0.40, min(needed_pct, mana_threshold))
            if needed_pct < mana_threshold and not rs.learned_mana_logged:
                log.info(
                    "[MANA] Rest: learned mana exit %.0f%% (cost=%d, default=%.0f%%)",
                    needed_pct * 100,
                    learned_cost,
                    mana_threshold * 100,
                )
                rs.learned_mana_logged = True
            mana_threshold = needed_pct
    mana_ok = state.mana_pct >= mana_threshold if state.mana_max > 0 else True
    pet_ok = True
    world = ctx.world
    if ctx.pet.alive and world:
        php = world.pet_hp_pct
        pet_ok = php < 0 or php >= 0.90
    if hp_ok and mana_ok and pet_ok:
        log.info(
            "[MANA] Rest: thresholds met (HP=%.0f%% Mana=%.0f%% thresh=%.0f%%) -- exiting",
            state.hp_pct * 100,
            state.mana_pct * 100,
            mana_threshold * 100,
        )
        rs.resting = False
        rs.learned_mana_logged = False
        return True
    return False


def _should_rest(
    state: GameState,
    ctx: SurvivalView,
    rs: _SurvivalRuleState,
) -> bool:
    if _rest_suppressed(ctx, state, rs):
        return False

    hp_low, mana_low, pet_low = rest_needs_check(ctx, state)

    if hp_low or mana_low or pet_low:
        if not rs.resting:
            reasons = []
            if hp_low:
                reasons.append(f"HP={state.hp_pct * 100:.0f}%")
            if mana_low:
                reasons.append(f"Mana={state.mana_pct * 100:.0f}%")
            if pet_low:
                reasons.append("pet_low")
            log.info("[MANA] Rest: entering (%s)", ", ".join(reasons))
        rs.resting = True

    if rs.resting:
        _rest_exit_check(ctx, state, rs)

    return rs.resting


def _score_rest(state: GameState, ctx: SurvivalView) -> float:
    if not flags.rest:
        return 0.0
    if ctx.in_active_combat:
        return 0.0
    hp_score: float = inverse_linear(state.hp_pct, ctx.rest_hp_entry, ctx.rest_hp_threshold)
    mana_score: float = 0.0
    if state.mana_max > 0:
        mana_score = inverse_linear(state.mana_pct, ctx.rest_mana_entry, ctx.rest_mana_threshold)
    return max(hp_score, mana_score)


def _should_evade(
    state: GameState,
    ctx: SurvivalView,
    rs: _SurvivalRuleState,
) -> bool:
    if ctx.combat.engaged:
        # During combat: only evade RED patrol on collision course
        if (
            ctx.threat.evasion_point is not None
            and ctx.threat.patrol_evade
            and time.time() - rs.last_patrol_evade > 8.0
        ):
            log.warning("[POSITION] Evade: RED patrol collision during combat -- sidestepping")
            rs.last_patrol_evade = time.time()
            return True
        rs.evade_logged = False
        return False
    if ctx.threat.evasion_point is not None:
        if not rs.evade_logged:
            log.info("[STATE] Evade: threat detected, evasion point set")
            rs.evade_logged = True
        return True
    rs.evade_logged = False
    return False


def _score_evade(state: GameState, ctx: SurvivalView) -> float:
    if ctx.combat.engaged:
        return 0.0
    return 1.0 if ctx.threat.evasion_point is not None else 0.0


def register(brain: Brain, ctx: AgentContext, read_state_fn: ReadStateFn) -> None:
    """Register survival rules (highest priority)."""

    # DEATH_RECOVERY - highest priority, recover from death
    death_recovery = DeathRecoveryRoutine(ctx=ctx, read_state_fn=read_state_fn)

    brain.add_rule(
        "DEATH_RECOVERY",
        lambda s: _should_death_recover(s, ctx),
        death_recovery,
        score_fn=lambda s: _score_death_recovery(s, ctx),
        tier=0,
        weight=100,
    )

    # FEIGN_DEATH - try FD before fleeing
    feign_death = FeignDeathRoutine(ctx=ctx, read_state_fn=read_state_fn)

    brain.add_rule(
        "FEIGN_DEATH",
        lambda s: _should_feign_death(s, ctx),
        feign_death,
        failure_cooldown=30.0,
        emergency=True,
        score_fn=lambda s: _score_feign_death(s, ctx),
        tier=0,
        weight=100,
    )

    # FLEE - run to guards/zoneline
    flee = FleeRoutine(ctx=ctx, read_state_fn=read_state_fn)

    brain.add_rule(
        "FLEE",
        lambda s: _should_flee(s, ctx),
        flee,
        emergency=True,
        max_lock_seconds=120.0,
        score_fn=lambda s: _score_flee(s, ctx),
        tier=0,
        weight=100,
    )

    rest = RestRoutine(
        hp_high=ctx.rest_hp_threshold,
        mana_high=ctx.rest_mana_threshold,
        ctx=ctx,
        read_state_fn=read_state_fn,
    )

    _state = _SurvivalRuleState()

    rest_considerations = [
        Consideration(
            name="hp_deficit",
            input_fn=lambda s, _ctx: 1.0 - s.hp_pct,
            curve=lambda v: inverse_linear(1.0 - v, ctx.rest_hp_entry, ctx.rest_hp_threshold),
            weight=1.5,
        ),
        Consideration(
            name="mana_deficit",
            input_fn=lambda s, _ctx: 1.0 - s.mana_pct if s.mana_max > 0 else 0.0,
            curve=lambda v: (
                inverse_linear(1.0 - v, ctx.rest_mana_entry, ctx.rest_mana_threshold) if v > 0 else 0.01
            ),
            weight=1.0,
        ),
    ]

    brain.add_rule(
        "REST",
        lambda s: _should_rest(s, ctx, _state),
        rest,
        score_fn=lambda s: _score_rest(s, ctx),
        considerations=rest_considerations,
        tier=0,
        weight=50,
    )

    # EVADE - sidestep approaching YELLOW/RED threat
    from routines.evade import EvadeRoutine

    evade = EvadeRoutine(ctx=ctx, read_state_fn=read_state_fn)

    brain.add_rule(
        "EVADE",
        lambda s: _should_evade(s, ctx, _state),
        evade,
        emergency=True,
        score_fn=lambda s: _score_evade(s, ctx),
        tier=0,
        weight=100,
    )
