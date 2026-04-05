# Evaluation Guide

Reference architecture. Public extraction. Not intended to run standalone.

This document exists to set the evaluation contract for code reviewers, automated agents, and anyone assessing this repository. The repo is a cleaned extraction from a private working codebase, published as an architecture reference for layered agent decision-making in a real-time 3D environment.

## What this repo is

A control architecture that layers priority rules, utility scoring, and goal-oriented planning over a forward-only pipeline: perception, brain, routines, motor. The architecture is enforced in code, not just documented in prose. Import-boundary tests verify that routines cannot import the decision engine, motor cannot import brain or routines, and perception cannot import upper layers.

The codebase includes:

- A decomposed Brain and BrainRunner (no god object)
- Extracted transition, completion, and profiling logic
- GOAP planner with A* search, Monte Carlo robustness gating, and cost self-correction
- Bayesian encounter learning with Thompson Sampling
- Utility scoring phases (0--4) with consideration-based selection
- Geometry-aware 3D navigation with stuck detection and terrain recovery
- World model with velocity tracking, patrol detection, and Pareto target selection
- Pipeline, integration, adversarial, and boundary tests
- CI running lint, typecheck, dead-code scan, and coverage enforcement

## What is intentionally omitted

Live runtime configuration, environment-specific assets, memory-reading glue, and operational deployment tooling. These are private and environment-specific. Their absence is by design, not oversight.

The simulator (`src/simulator/`) runs headless scenarios against recorded state sequences to exercise the full decision stack without a live environment. Sample session output in `docs/samples/` demonstrates real telemetry from the working system.

## What should be scored

| Dimension | Where to look |
|---|---|
| Module boundaries and import DAG | `tests/test_import_dag.py`, package structure under `src/` |
| Decision-stack decomposition | `brain/decision.py`, `brain/scoring_phases.py`, `brain/runner/loop.py` |
| Planner/routine separation | `brain/goap/planner.py`, `brain/runner/learning.py`, `brain/completion.py` |
| World-model design | `brain/world/model.py`, `brain/world/patrol.py`, `brain/scoring/target.py` |
| Navigation stack | `nav/movement.py`, `nav/stuck_recovery.py`, `nav/geometry.py` |
| Learning subsystems | `brain/learning/encounters.py`, `brain/learning/scorecard.py`, `brain/scoring/weight_learner.py` |
| Test coverage and invariants | `tests/`, `util/invariants.py`, CI coverage threshold |
| Extensibility seams | Rule registration (`brain/rules/__init__.py`), action/goal sets (`brain/goap/actions.py`, `brain/goap/goals.py`) |

## What should not be scored

Missing deployment, runtime, or environment assets. The repo documents their absence explicitly and provides simulation and sample output as substitutes. Grading a reference architecture on whether it boots a live client is like grading a textbook compiler on whether it ships an OS.
