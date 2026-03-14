"""Learning and GOAP tick helpers.

Extracted from loop.py to separate learning subsystem ticks
(scorecard tuning, gradient weight learning, GOAP planning) from
brain loop orchestration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from brain.goap import build_world_state
from brain.learning.scorecard import (
    apply_tuning,
    compute_scorecard,
    evaluate_and_tune,
    save_tuning,
)
from brain.scoring.weight_learner import save_learned_weights
from core.features import flags

if TYPE_CHECKING:
    from brain.context import AgentContext
    from brain.runner.loop import BrainRunner
    from perception.state import GameState

brain_log = logging.getLogger("compass.brain_loop")


class LearningTickHandler:
    """Learning subsystem tick helpers: tuning, gradient learning, GOAP planning.

    Composed into BrainRunner (not a subclass). The runner passes
    itself so the handler can access shared state.

    Args:
        runner: The parent BrainRunner instance.
    """

    def __init__(self, runner: BrainRunner) -> None:
        self._runner = runner

    def tick_tuning_eval(self, ctx: AgentContext, now: float) -> None:
        """Run scorecard feedback loop every 30 minutes."""
        runner = self._runner
        if now <= runner._next_tuning_eval:
            return
        try:
            scores = compute_scorecard(ctx)
            brain_log.info(
                "[TUNING] eval: dph=%d surv=%d pull=%d mana=%d overall=%d(%s)",
                scores.get("defeat_rate", 0),
                scores.get("survival", 0),
                scores.get("pull_success", 0),
                scores.get("mana_efficiency", 0),
                scores.get("overall", 0),
                scores.get("grade", "?"),
            )
            runner._tuning = evaluate_and_tune(dict(scores), runner._tuning)
            apply_tuning(runner._tuning, ctx)
            save_tuning(runner._tuning, runner._current_zone)
        except (AttributeError, TypeError, ZeroDivisionError, KeyError, OSError, ValueError) as e:
            brain_log.warning("[TUNING] evaluation failed: %s", e)
        runner._next_tuning_eval = now + 1800.0

    def tick_gradient_learning(self, ctx: AgentContext) -> None:
        """Feed encounter fitness to gradient tuner and step when ready.

        Drains recent fitness data from fight_history, feeds to the gradient
        tuner, and triggers a weight update step when enough observations
        have accumulated (~15 encounters). Lightweight: runs every brain tick
        but returns immediately unless new data exists.
        """
        tuner = self._runner._gradient_tuner
        if tuner is None:
            return
        fh = getattr(ctx, "fight_history", None)
        if fh is None:
            return
        # Drain new encounter fitness pairs
        recent = fh.drain_recent_fitness()
        if not recent:
            return
        # Get the latest target breakdown from world model (captured at
        # target selection time for the best target each tick)
        breakdown: dict[str, float] = {}
        world = getattr(ctx, "world", None)
        if world is not None:
            breakdown = getattr(world, "_last_target_breakdown", {})
        for fitness, _mob_name in recent:
            tuner.observe(fitness, breakdown)
        if tuner.ready_to_step():
            tuner.step()
            save_learned_weights(
                tuner.get_weight_snapshot(),
                self._runner._current_zone,
                learning_rates=tuner.get_learning_rates(),
            )

    def tick_goap_planner(self, state: GameState, ctx: AgentContext, now: float) -> str | None:
        """GOAP planner tick: generate/advance plans, activate next step.

        Returns the routine name to activate, or None if no plan step ready.
        Called between emergency rule check and utility scoring fallback.
        """
        runner = self._runner
        planner = runner._goap_planner
        if planner is None or not flags.goap_planning:
            return None

        ws = build_world_state(state, ctx)

        # Refresh spawn predictions periodically
        if runner._spawn_predictor and now > runner._next_spawn_update:
            if ctx.spatial_memory:
                runner._spawn_predictor.update_from_memory(ctx.spatial_memory)
            runner._next_spawn_update = now + 60.0

        # If active routine just completed, advance the plan
        if runner._brain._active is None and planner.has_plan():
            planner.advance(ws)

        # If plan exists and current step is ready, return its routine
        if planner.has_plan():
            step = planner.current_step
            if step is not None and step.preconditions_met(ws):
                planner.start_step(ctx)  # begin cost accuracy tracking
                name: str = step.routine_name
                return name
            elif step is not None:
                # Preconditions failed: invalidate and fall through to replan
                planner.invalidate("preconditions_failed")

        # No plan: try to generate one
        if not planner.has_plan():
            planner.generate(ws, ctx)
            if planner.has_plan():
                step = planner.current_step
                if step is not None:
                    planner.start_step(ctx)  # begin cost accuracy tracking
                    name2: str = step.routine_name
                    return name2

        return None  # no plan available, fall through to utility scoring
