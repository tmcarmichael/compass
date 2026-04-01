"""GOAP planner: forward A* search with Monte Carlo plan evaluation.

Searches from the current world state through action effects to find the
cheapest sequence that satisfies the target goal. Plans are short (3-8 steps)
and represent one operational cycle (rest -> acquire -> pull -> defeat -> loot).

Candidate plans are evaluated via Monte Carlo rollouts: action effects are
sampled stochastically from learned posterior distributions (Thompson Sampling
on encounter data) to estimate expected plan value under uncertainty. A plan
that performs well across sampled outcomes is more robust than one that looks
optimal under point estimates alone.

The planner never interrupts a running routine. It composes routines by
selecting which one to activate next, advancing the plan when the current
step completes.
"""

from __future__ import annotations

import heapq
import logging
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from brain.goap.actions import PlanAction
from brain.goap.goals import Goal
from brain.goap.world_state import PlanWorldState
from util.log_tiers import VERBOSE
from util.structured_log import log_event

if TYPE_CHECKING:
    from brain.context import AgentContext

log = logging.getLogger(__name__)

# Planning constraints
MAX_PLAN_STEPS = 8
MAX_SEARCH_NODES = 500
SATISFACTION_THRESHOLD = 0.70  # goal is "achieved enough" at this level
PLAN_BUDGET_MS = 50.0  # max time for plan generation
MC_ROLLOUTS = 20  # Monte Carlo rollouts per candidate plan
MC_NOISE_SIGMA = 0.15  # fallback noise when no learned variance available
MC_ROBUSTNESS_THRESHOLD = 0.50  # reject plans below this MC satisfaction


@dataclass(slots=True)
class Plan:
    """An ordered sequence of actions to achieve a goal."""

    goal: Goal
    steps: list[PlanAction]
    expected_cost: float  # total estimated seconds
    expected_satisfaction: float  # goal satisfaction at plan end
    step_index: int = 0

    @property
    def current_step(self) -> PlanAction | None:
        if self.step_index < len(self.steps):
            return self.steps[self.step_index]
        return None

    @property
    def completed(self) -> bool:
        return self.step_index >= len(self.steps)

    def advance(self) -> None:
        """Move to the next step in the plan."""
        self.step_index += 1

    def summary(self) -> str:
        """Human-readable plan summary."""
        step_names = [s.name for s in self.steps]
        return (
            f"Plan({self.goal.name}: {' -> '.join(step_names)}, "
            f"cost={self.expected_cost:.1f}s, "
            f"sat={self.expected_satisfaction:.2f})"
        )


# -- A* Search Node -----------------------------------------------------------


@dataclass(slots=True)
class _Node:
    """A* search node in the plan state space."""

    state: PlanWorldState
    g_cost: float  # accumulated cost (seconds)
    h_cost: float  # heuristic: estimated remaining cost
    actions: list[PlanAction]  # actions taken to reach this state
    depth: int

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost

    def __lt__(self, other: _Node) -> bool:
        return self.f_cost < other.f_cost


# -- GOAP Planner -------------------------------------------------------------


class GOAPPlanner:
    """Goal-Oriented Action Planner using forward A* search.

    Usage:
        planner = GOAPPlanner(goals, actions)
        plan = planner.generate(world_state, ctx)
        # Each tick:
        if planner.has_plan():
            step = planner.current_step
            ...  # activate step.routine_name
            # When routine completes:
            planner.advance(new_world_state)
    """

    def __init__(self, goals: list[Goal], actions: list[PlanAction]) -> None:
        self._goals = goals
        self._actions = actions
        self._plan: Plan | None = None
        self._plans_generated = 0
        self._plans_completed = 0
        self._plans_invalidated = 0
        # Cost accuracy tracking: estimated vs actual per step
        self._step_start_time: float = 0.0
        self._step_estimated_cost: float = 0.0
        self._cost_errors: list[float] = []  # (actual - estimated) per step
        # Cost self-correction: per-action exponential moving average of error
        self._cost_corrections: dict[str, float] = {}
        self._cost_correction_counts: dict[str, int] = {}

    @property
    def plan(self) -> Plan | None:
        return self._plan

    def has_plan(self) -> bool:
        return self._plan is not None and not self._plan.completed

    @property
    def current_step(self) -> PlanAction | None:
        if self._plan:
            return self._plan.current_step
        return None

    def generate(self, ws: PlanWorldState, ctx: AgentContext | None = None) -> Plan | None:
        """Generate a plan for the most insistent goal.

        Returns None if no plan found within budget.
        """
        goal = self._most_insistent_goal(ws)
        if goal is None:
            log.log(VERBOSE, "[GOAP] No insistent goal found")
            return None

        # Log all goal states for debugging goal dynamics
        log.log(
            VERBOSE,
            "[GOAP] Goal evaluation: %s | world: hp=%.0f%% mana=%.0f%% pet=%s targets=%d inv=%.0f%%",
            " ".join(f"{g.name}={g.insistence(ws):.2f}(sat={g.satisfaction(ws):.2f})" for g in self._goals),
            ws.hp_pct * 100,
            ws.mana_pct * 100,
            ws.pet_alive,
            ws.targets_available,
            ws.inventory_pct * 100,
        )

        # Already satisfied?
        sat = goal.satisfaction(ws)
        if sat >= SATISFACTION_THRESHOLD:
            log.log(
                VERBOSE,
                "[GOAP] Goal '%s' already satisfied (%.2f >= %.2f)",
                goal.name,
                sat,
                SATISFACTION_THRESHOLD,
            )
            return None

        t0 = time.perf_counter()
        plan = self._search(ws, goal, ctx)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if plan:
            self._plan = plan
            self._plans_generated += 1
            log_event(
                log,
                "goap_plan",
                f"[GOAP] Generated: {plan.summary()} in {elapsed_ms:.1f}ms",
                goal=goal.name,
                steps=len(plan.steps),
                cost=round(plan.expected_cost, 1),
                satisfaction=round(plan.expected_satisfaction, 2),
                plan_ms=round(elapsed_ms, 1),
            )
        else:
            log.log(
                VERBOSE,
                "[GOAP] No plan found for '%s' (sat=%.2f, %d actions, %.1fms)",
                goal.name,
                sat,
                len(self._actions),
                elapsed_ms,
            )

        return plan

    def start_step(self, ctx: AgentContext | None = None) -> None:
        """Called when a plan step's routine begins executing.

        Records the start time and estimated cost for accuracy tracking.
        """
        self._step_start_time = time.time()
        step = self.current_step
        if step:
            self._step_estimated_cost = step.estimate_cost(ctx)

    def advance(self, ws: PlanWorldState) -> None:
        """Advance the plan after current step completes.

        Tracks cost accuracy, checks if the goal is satisfied (early
        completion), and validates next step's preconditions.
        """
        if not self._plan:
            return

        # Cost accuracy: compare estimated vs actual duration
        step = self._plan.current_step
        if step and self._step_start_time > 0:
            actual = time.time() - self._step_start_time
            estimated = self._step_estimated_cost
            error = actual - estimated
            self._cost_errors.append(error)
            # Update self-correction model (EMA of error per action)
            self._update_cost_correction(step.name, error)
            if abs(error) > 5.0:  # >5s discrepancy is notable
                log.info(
                    "[GOAP] Cost accuracy: step '%s' estimated=%.1fs "
                    "actual=%.1fs error=%+.1fs correction=%+.1fs",
                    step.name,
                    estimated,
                    actual,
                    error,
                    self._cost_corrections.get(step.name, 0.0),
                )
            else:
                log.log(
                    VERBOSE,
                    "[GOAP] Cost accuracy: step '%s' estimated=%.1fs actual=%.1fs error=%+.1fs",
                    step.name,
                    estimated,
                    actual,
                    error,
                )
            self._step_start_time = 0.0

        self._plan.advance()

        # Goal satisfied early?
        sat = self._plan.goal.satisfaction(ws)
        if sat >= SATISFACTION_THRESHOLD:
            log.info(
                "[GOAP] Goal '%s' satisfied early at step %d/%d (sat=%.2f)",
                self._plan.goal.name,
                self._plan.step_index,
                len(self._plan.steps),
                sat,
            )
            self.complete()
            return

        # Plan completed?
        if self._plan.completed:
            self.complete()
            return

        # Next step's preconditions still valid?
        next_step = self._plan.current_step
        if next_step and not next_step.preconditions_met(ws):
            log.info(
                "[GOAP] Preconditions failed at step %d/%d (%s) -- "
                "invalidating plan | world: hp=%.0f%% mana=%.0f%% "
                "pet=%s engaged=%s targets=%d threats=%d",
                self._plan.step_index,
                len(self._plan.steps),
                next_step.name,
                ws.hp_pct * 100,
                ws.mana_pct * 100,
                ws.pet_alive,
                ws.engaged,
                ws.targets_available,
                ws.nearby_threats,
            )
            self.invalidate("preconditions_failed")

    def complete(self) -> None:
        """Mark the current plan as completed."""
        if self._plan:
            self._plans_completed += 1
            log_event(
                log,
                "goap_complete",
                f"[GOAP] Plan completed: {self._plan.goal.name} "
                f"({self._plan.step_index}/{len(self._plan.steps)} steps)",
                goal=self._plan.goal.name,
                steps_total=len(self._plan.steps),
                steps_executed=self._plan.step_index,
            )
        self._plan = None

    def invalidate(self, reason: str = "") -> None:
        """Invalidate the current plan (emergency, world change, etc.)."""
        if self._plan:
            self._plans_invalidated += 1
            log_event(
                log,
                "goap_invalidate",
                f"[GOAP] Plan invalidated: {self._plan.goal.name} at "
                f"step {self._plan.step_index}/{len(self._plan.steps)} "
                f"(reason: {reason or 'unspecified'})",
                goal=self._plan.goal.name,
                step=self._plan.step_index,
                steps_total=len(self._plan.steps),
                reason=reason or "unspecified",
            )
        self._plan = None

    def _update_cost_correction(self, action_name: str, error: float) -> None:
        """Exponential moving average of cost error per action type."""
        alpha = 0.3  # weight of new observation
        old = self._cost_corrections.get(action_name, 0.0)
        self._cost_corrections[action_name] = old * (1 - alpha) + error * alpha
        self._cost_correction_counts[action_name] = self._cost_correction_counts.get(action_name, 0) + 1

    def get_corrected_cost(self, action: PlanAction, ctx: AgentContext | None) -> float:
        """Action cost adjusted by self-correction model."""
        base: float = action.estimate_cost(ctx)
        correction: float = self._cost_corrections.get(action.name, 0.0)
        # Only apply correction after 3+ observations (avoid noise)
        count: int = self._cost_correction_counts.get(action.name, 0)
        if count < 3:
            return base
        # Floor at 10% of base (not 1.0) to allow learning of fast steps
        corrected: float = max(base * 0.1, base + correction)
        return corrected

    @property
    def cost_corrections(self) -> dict[str, float]:
        """Current cost corrections per action (for persistence)."""
        return dict(self._cost_corrections)

    def load_cost_corrections(self, data: dict[str, float]) -> None:
        """Load previously saved cost corrections."""
        self._cost_corrections = dict(data)
        # Assume at least 3 observations for each loaded correction
        for k in data:
            self._cost_correction_counts[k] = max(3, self._cost_correction_counts.get(k, 0))

    def stats_summary(self) -> str:
        """Return a summary of planner statistics."""
        completion_rate = 0.0
        if self._plans_generated > 0:
            completion_rate = self._plans_completed / self._plans_generated * 100
        avg_cost_error = 0.0
        if self._cost_errors:
            avg_cost_error = sum(self._cost_errors) / len(self._cost_errors)
        return (
            f"GOAP: {self._plans_generated} generated, "
            f"{self._plans_completed} completed, "
            f"{self._plans_invalidated} invalidated "
            f"({completion_rate:.0f}% completion rate, "
            f"avg cost error: {avg_cost_error:+.1f}s)"
        )

    # -- Internal: Goal Selection -----------------------------------------------

    def _most_insistent_goal(self, ws: PlanWorldState) -> Goal | None:
        """Select the goal with the highest insistence."""
        if not self._goals:
            return None
        best = max(self._goals, key=lambda g: g.insistence(ws))
        if best.insistence(ws) <= 0.01:
            return None  # all goals effectively satisfied
        log.log(
            VERBOSE,
            "[GOAP] Goal insistence: %s",
            ", ".join(f"{g.name}={g.insistence(ws):.2f}" for g in self._goals),
        )
        return best

    # -- Internal: Monte Carlo Plan Evaluation -----------------------------------

    def _mc_evaluate(
        self,
        plan_actions: list[PlanAction],
        start: PlanWorldState,
        goal: Goal,
        ctx: AgentContext | None,
    ) -> float:
        """Evaluate a candidate plan via Monte Carlo rollouts.

        Runs MC_ROLLOUTS stochastic simulations of the plan. In each rollout,
        action effects are perturbed with noise drawn from learned posterior
        variance (encounter history) when available, or fixed sigma as
        fallback.  Returns the mean goal satisfaction across rollouts.

        A plan that achieves high satisfaction across noisy rollouts is robust
        to the inherent uncertainty in combat outcomes, rest durations, etc.
        """
        if not plan_actions:
            return goal.satisfaction(start)

        # Derive noise sigma from learned posterior variance when available.
        # Wider posteriors (less data) produce more noise, naturally penalising
        # plans that depend on uncertain outcomes.
        hp_sigma, mana_sigma = self._learned_mc_sigma(ctx)

        total_sat = 0.0
        for _ in range(MC_ROLLOUTS):
            ws = start
            for action in plan_actions:
                ws = action.apply_effects(ws)
                hp_noise = random.gauss(0, hp_sigma)
                mana_noise = random.gauss(0, mana_sigma)
                ws = ws.with_changes(
                    hp_pct=max(0.0, min(1.0, ws.hp_pct + hp_noise)),
                    mana_pct=max(0.0, min(1.0, ws.mana_pct + mana_noise)),
                )
            total_sat += goal.satisfaction(ws)

        return total_sat / MC_ROLLOUTS

    @staticmethod
    def _learned_mc_sigma(ctx: AgentContext | None) -> tuple[float, float]:
        """Derive MC noise sigma from encounter posterior variance.

        When fight history has enough data, the posterior variance on HP loss
        and mana cost reflects actual outcome uncertainty.  Wider posteriors
        (fewer observations) produce larger sigma, so plans that depend on
        poorly-known actions are penalised more heavily.

        Falls back to MC_NOISE_SIGMA when no learned data is available.
        """
        if not ctx or not ctx.fight_history:
            return MC_NOISE_SIGMA, MC_NOISE_SIGMA
        all_stats = ctx.fight_history.get_all_stats()
        if not all_stats:
            return MC_NOISE_SIGMA, MC_NOISE_SIGMA
        # Average posterior std across known entity types
        hp_vars: list[float] = []
        mana_vars: list[float] = []
        for stats in all_stats.values():
            if stats.danger_post_var > 0:
                hp_vars.append(stats.danger_post_var)
            if stats.mana_post_var > 0:
                mana_vars.append(stats.mana_post_var)
        hp_sigma = (sum(hp_vars) / len(hp_vars)) ** 0.5 if hp_vars else MC_NOISE_SIGMA
        mana_sigma = (sum(mana_vars) / len(mana_vars)) ** 0.5 if mana_vars else MC_NOISE_SIGMA
        # Clamp to reasonable range
        hp_sigma = max(0.02, min(0.40, hp_sigma))
        mana_sigma = max(0.02, min(0.40, mana_sigma))
        return hp_sigma, mana_sigma

    # -- Internal: A* Search ----------------------------------------------------

    def _search(self, start: PlanWorldState, goal: Goal, ctx: AgentContext | None) -> Plan | None:
        """Forward A* search from current state to goal satisfaction."""
        deadline = time.perf_counter() + PLAN_BUDGET_MS / 1000.0

        start_node = _Node(
            state=start,
            g_cost=0.0,
            h_cost=self._heuristic(start, goal),
            actions=[],
            depth=0,
        )

        # Priority queue: (f_cost, tiebreaker, node)
        open_list: list[tuple[float, int, _Node]] = []
        heapq.heappush(open_list, (start_node.f_cost, 0, start_node))
        counter = 1
        visited = 0

        while open_list:
            if visited >= MAX_SEARCH_NODES:
                log.log(VERBOSE, "[GOAP] Search exhausted: %d nodes", visited)
                break
            if time.perf_counter() > deadline:
                log.log(VERBOSE, "[GOAP] Search budget exceeded: %.1fms", PLAN_BUDGET_MS)
                break

            _, _, node = heapq.heappop(open_list)
            visited += 1

            # Goal test: deterministic satisfaction check
            sat = goal.satisfaction(node.state)
            if sat >= SATISFACTION_THRESHOLD:
                # Monte Carlo robustness gate: reject plans that don't hold
                # under stochastic action outcomes.  Uses learned posterior
                # variance when available, fixed sigma as fallback.
                mc_sat = self._mc_evaluate(node.actions, start, goal, ctx)
                if mc_sat < MC_ROBUSTNESS_THRESHOLD:
                    log.log(
                        VERBOSE,
                        "[GOAP] Plan rejected (mc_sat=%.2f < %.2f): %d steps, cost=%.1f",
                        mc_sat,
                        MC_ROBUSTNESS_THRESHOLD,
                        len(node.actions),
                        node.g_cost,
                    )
                    continue  # keep searching for a more robust plan
                log.log(
                    VERBOSE,
                    "[GOAP] Plan found: %d steps, %d nodes, cost=%.1f, sat=%.2f, mc_sat=%.2f",
                    len(node.actions),
                    visited,
                    node.g_cost,
                    sat,
                    mc_sat,
                )
                return Plan(
                    goal=goal,
                    steps=node.actions,
                    expected_cost=node.g_cost,
                    expected_satisfaction=mc_sat,
                )

            # Depth limit
            if node.depth >= MAX_PLAN_STEPS:
                continue

            # Expand: try each action
            for action in self._actions:
                if not action.preconditions_met(node.state):
                    continue
                # Avoid duplicate actions in sequence (rest -> rest)
                if node.actions and node.actions[-1].name == action.name:
                    continue

                new_state = action.apply_effects(node.state)
                new_cost = node.g_cost + self.get_corrected_cost(action, ctx)
                new_actions = node.actions + [action]

                child = _Node(
                    state=new_state,
                    g_cost=new_cost,
                    h_cost=self._heuristic(new_state, goal),
                    actions=new_actions,
                    depth=node.depth + 1,
                )
                heapq.heappush(open_list, (child.f_cost, counter, child))
                counter += 1

        return None  # no plan found

    def _heuristic(self, ws: PlanWorldState, goal: Goal) -> float:
        """Estimated remaining cost to satisfy goal from this state.

        Uses goal dissatisfaction as a proxy. Lower satisfaction = higher
        estimated remaining cost. Scaled by a typical action cost (15s)
        so the heuristic is in the same units as g_cost (seconds).
        """
        sat: float = goal.satisfaction(ws)
        if sat >= SATISFACTION_THRESHOLD:
            return 0.0
        # Remaining dissatisfaction * estimated seconds per unit of progress
        remaining: float = SATISFACTION_THRESHOLD - sat
        return remaining * 30.0  # ~30s per unit of goal progress (tunable)
