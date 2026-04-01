<!-- last_modified: 2026-03-25 -->

# Retrospective

Engineering reflection on what worked, what was hard, and what generalizes.

This architecture was developed and validated against Classic EverQuest: a persistent 3D world with real-time combat, resource management, and spatial reasoning.

---

## What This Project Demonstrates

- The [four-stage pipeline](architecture.md#system-overview) has been additive since the pipeline decomposition (see [Evolution](evolution.md#the-invariant-each-stage-is-additive) for the full record).

- Classical priority rules and goal-oriented planning coexist cleanly when layered correctly. Priority rules guarantee safety; GOAP optimizes throughput. Neither system needs to know the other's internals.

- An agent that learns from its own experience (encounter history with Thompson Sampling, spatial memory, finite-difference weight tuning, spawn prediction) measurably outperforms the same agent with hand-tuned heuristics. One signal (encounter outcomes) drives all four learning systems.

- Direct state reading (process memory) outperforms indirect observation (screen parsing, log file tailing) on every axis. If the environment knows a value, reading it directly is faster, more reliable, and more complete.

- Zero-dependency Python handles real-time 10 Hz control loops with GOAP planning, utility scoring, and finite-difference gradient tuning. The entire agent runs on the Python 3.14 standard library.

---

## What Worked Well

**The four-stage pipeline never needed rethinking.** The [forward-only pipeline](architecture.md#system-overview) was established at the pipeline stage and has never required structural change. Data flows down. Nothing imports upward.

**The safety envelope principle (see Architecture) proved out.** Having an inviolable safety layer allowed every other system to optimize aggressively without risk.

**Frozen dataclasses for perception snapshots eliminated concurrency bugs.** Before snapshots, the brain and secondary threads raced on mutable state. After switching to frozen GameState/SpawnData, every concurrency bug in the perception layer disappeared. The pattern is simple: produce immutable data, hand it off, never touch it again.

**The non-blocking tick contract caught every bad pattern early.** The rule that `tick()` must never block for more than ~200ms prevented every attempt to add polling loops, long sleeps, or synchronous waits inside routines. Phase state machines are more code than a while loop, but they keep the brain responsive to emergencies.

**Encounter history produced genuine cross-session improvement.** Once [learned data takes over](architecture.md#encounter-learning-per-encounter), the agent selects better strategies, avoids unwinnable encounters, and provides accurate cost estimates to the GOAP planner. The improvement is measurable in defeats/hour.

**GOAP planning eliminated reactive sequencing waste.** The most visible improvement: the agent rests to the right mana level before pulling (not a fixed threshold), positions toward predicted respawns instead of wandering randomly, and reasons about multi-step sequences rather than evaluating each rule independently.

**4-tier logging made debugging overnight sessions tractable.** The EVENT tier (~50 lines/hour) gives an immediate summary: defeats, deaths, zone changes. The VERBOSE tier (~5,000 lines/hour) explains every decision. Scanning the event log first, then drilling into verbose for a specific incident, is dramatically faster than searching a single monolithic log.

**The strategy pattern allowed adding combat approaches independently.** Four strategies exist today. Each is a self-contained class. Adding a fifth means implementing one class and updating the selection function. No existing strategy code is touched.

---

## What Was Difficult

**Reverse engineering the client's memory layout.** The client has no symbols, no documentation, and struct layouts that were never intended for external consumption. Discovering the profile pointer chain required multiple sessions of Ghidra analysis and before/after memory diffing. Each new offset is a mini research project.

**Routine interactions.** Many state machines sharing context through 12 sub-states create ordering dependencies that are hard to reason about. When one routine modifies inventory state and another rule reads it, the timing matters. Most bugs in the priority-rule era were interaction bugs, not individual routine bugs.

**Navigation in 3D terrain.** Cliffs, water, lava, bridges, tunnels, and variable elevation make 2D pathfinding insufficient but full 3D pathfinding prohibitively expensive. The hybrid approach (1-unit heightmap + waypoint graph) works but required significant iteration to handle edge cases: ramps that project as walkable, water with walkable floors, bridges that overlap lower terrain.

**Logging discipline.** The [decision coverage contract](architecture.md#decision-coverage) must be maintained manually, every branch, every time. A single omission creates a gap that can hide for hours in an autonomous session.

**Tuning score functions.** Utility scoring required every rule to produce a meaningful float score. "How valuable is resting right now?" is not a natural question for a condition that was originally boolean. The divergence logging phase (Phase 1) was essential. It revealed which score functions produced pathological results before any scores influenced decisions.

**GOAP plan horizon.** Short plans (3-5 steps) work well. Longer plans (6-8 steps) are more likely to be invalidated by world changes before completion, wasting the planning budget. The current sweet spot is planning one "cycle" ahead: rest -> acquire -> pull -> combat -> loot. Plans beyond that are too speculative.

**Cost function convergence.** The GOAP planner is only as good as its cost estimates. Early in a session (before encounter history accumulates), heuristic costs produce suboptimal plans. The transition from heuristic to learned costs (roughly 50-100 defeats) is a period of mixed-quality planning. This is acceptable but visible in the survival curve.

---

## Architecture Evolution

The architecture progressed through six stages, each solving a specific failure mode of the previous one: monolith, pipeline, priority rules, utility scoring, learning loops, and GOAP. See [Evolution](evolution.md) for the full stage-by-stage history. The architecture has been [additive since the pipeline](evolution.md#the-invariant-each-stage-is-additive). Nothing was replaced. Everything composes.

---

## Transferability

Classic EverQuest provided a rich environment for validating autonomous agent design: complex 3D terrain, real-time combat, a persistent economy, and unpredictable NPC behavior. The patterns that emerged are intentionally reusable; the environment-specific bindings (memory offsets, game structs, terrain formats) are the only coupling points.

### The Hybrid Architecture

The central transferable contribution is the three-layer decision stack:

1. **Classical safety layer**: priority rules with emergency override, circuit breakers, oscillation prevention. Provides hard guarantees.
2. **Learned optimization layer**: utility scoring with finite-difference gradient-tuned weights, Thompson Sampling on encounter posteriors, threshold auto-tuning. Improves with experience.
3. **Anticipatory planning layer**: GOAP with Monte Carlo plan evaluation, learned cost functions, spawn prediction. Reasons about sequences under uncertainty.

Classic EverQuest was the concrete proof point, but this architecture applies to any domain where an agent must operate autonomously: safe by construction, improving by experience, planning ahead when possible.

### Specific Transferable Patterns

- **Other game environments**: any application with readable process memory can be perceived the same way. The perception layer is the only part that changes.

- **Autonomous testing agents**: navigating complex UIs, handling error states, recovering from unexpected conditions. The routine state machine pattern handles multi-step UI flows cleanly.

- **Any long-running autonomous system**: the 4-tier logging pipeline, the forensic buffer, and the scorecard system are useful wherever an agent operates unattended and needs post-hoc debugging.

### The Core Insight

An autonomous agent needs the same things regardless of domain: reliable perception, safe prioritized decisions, non-blocking execution, learned optimization, anticipatory planning, and deep observability. Classic EverQuest exercised all of these under real-world conditions. The specific sensors, actuators, and domain knowledge change. The architecture does not.
