<!-- last_modified: 2026-03-25 -->

# Design Decisions

Rationale for major architectural choices and rejected alternatives, grouped by subsystem.

---

## Architecture

### Functions over classes where dispatch isn't needed

Rules are functions, not classes, because rule evaluation is a linear scan with no polymorphic dispatch. Adding a rule means writing a condition function and registering it. Scoring is functions over dataclasses because the data flows one way: state in, float out. Learning systems are plain classes with dicts because their job is accumulation and persistence, not behavioral variation.

Classes are used where polymorphism earns its keep: routines hold phase state across ticks and share an enter/tick/exit lifecycle (RoutineBase). Combat strategies vary behavior per encounter (CastStrategy). GOAP goals and actions have preconditions and effects that differ per subclass. These hierarchies are small and shallow, never more than one level deep.

The result is a codebase where most logic is directly readable as functions, and the few class hierarchies exist because the problem requires dispatch.

### Thread-safe sharing via frozen dataclass snapshots

GameState and SpawnData are frozen dataclasses. Once the perception layer creates a snapshot, it is immutable. The brain thread produces snapshots; any secondary reader consumes them. No locks, no races, no defensive copies. The cost is one allocation per tick, negligible at 10 Hz.

### Sub-state objects on shared context

AgentContext carries 12 focused sub-state dataclasses (CombatState, PetState, CampConfig, InventoryState, PlanState, PlayerState, DefeatTracker, SessionMetrics, ThreatState, LootConfig, ZoneState, DiagnosticState). Each owns a narrow slice of agent state. This replaced an early monolithic context where `ctx.player_dead`, `ctx.player_x`, `ctx.target_id` sat as flat attributes: hard to grep, easy to collide, impossible to reason about ownership.

### Modular rule registration

Each of the four rule modules (survival, combat, maintenance, navigation) registers its rules independently. Adding a new rule means adding a function in the relevant module and one line in `register_all()`. No existing rule code is touched. This has scaled cleanly from the initial rule set to the current system.

### Strict forward-only data flow

The [forward-only import rule](architecture.md#system-overview) eliminates circular dependencies and keeps each layer independently understandable. A routine cannot import from `brain.decision`. The brain cannot import from `runtime.orchestrator`. Motor cannot make decisions. Enforced by convention and caught quickly by import errors.

### Additive architecture evolution

The rationale for strict forward-only data flow is validated by its track record: the pipeline has absorbed every subsequent capability without structural change. See [Evolution](evolution.md) for the full stage-by-stage progression.

---

## Perception

### Direct memory reads over screen parsing or log parsing

If the environment knows a value, read it directly from process memory. Screen parsing is fragile (resolution-dependent, occluded by windows, laggy). Log parsing is incomplete (not all state changes produce log lines) and introduces latency. A memory read takes microseconds, is always current, and cannot be occluded.

### Observed peak values for unknown maxima

Some values (like maximum HP) are not directly readable at a known offset. Rather than hardcoding formulas or parsing UI text, the agent tracks the highest observed value and uses it as the working maximum. This converges to the true value within seconds of the first full-health observation and self-corrects after level-ups or equipment changes.

### Internal pointer chains over absolute addresses

Character and entity state is accessed by following the environment's own internal pointer chains rather than absolute offsets into dynamically allocated regions. Absolute offsets into heap-allocated structures can shift between client sessions; following the client's own indirection is stable regardless of runtime state. This eliminated an entire class of session-dependent read bugs.

### Frozen snapshots for thread-safe reads

Perception produces a new frozen GameState every tick. The alternative, mutable state protected by locks, would require lock acquisition on every field access in both threads. Frozen snapshots are simpler, faster, and impossible to misuse.

---

## Decision Architecture

### Classical rules guarantee safety; learned scoring optimizes performance

The [core design principle](architecture.md#decision-architecture) separates safety from optimization structurally. The rationale: keeping classical safety and learned optimization at different levels means they never conflict.

### Utility scoring within priority tiers

Pure priority rules are correct for safety but overly rigid for optimization. REST always outranking ACQUIRE means the agent sits down when a trivial target is 20 units away and would cost 15% mana. Utility scoring resolves this: each rule produces a score, and within a priority tier, the highest-scoring rule wins. Between tiers, higher priority wins. Safety is preserved while optimization gains flexibility.

All non-trivial rules have score functions. Five selection phases allow gradual trust escalation: Phase 0 ignores scores (baseline), Phase 1 logs divergences (observation), Phase 2 uses scores within tiers (conservative), Phase 3 uses weighted cross-tier scoring (full utility AI), Phase 4 uses consideration-based scoring where rules declare (input, curve, weight) components and the engine computes weighted geometric mean. Phase 4 falls back to score functions for rules without considerations, so activation is incremental.

### GOAP proposes, priorities dispose

The GOAP planner generates multi-step action sequences, but the [priority system remains the safety envelope](architecture.md#layer-3-goap-planner). The planner can be aggressive because it never needs to worry about safety. The priority system handles safety independently. Neither system needs to know the other's internals.

### Reactive safety with anticipatory planning

Early versions were purely reactive: respond to observed state, never predict. This was correct at the time. At 10 Hz evaluation frequency, the next tick observes the actual outcome before most predictions would matter.

But reactive decisions produce suboptimal sequences. The agent rests to 60% mana, then acquires a target that costs 50%. A planner rests to 80% knowing the encounter will be expensive. The agent wanders randomly. A planner moves toward predicted respawns.

The resolution is layered: reactive safety (emergency rules evaluate every tick based on observed state) with anticipatory planning (GOAP planner generates sequences based on learned cost models and spawn predictions). The reactive layer handles what prediction cannot: unexpected threats, memory read errors, desync. The planning layer handles what reaction cannot: resource anticipation, positioning, multi-step sequencing.

### Narrow GOAP action set by design

The planner operates over 8 actions, not because the search space is too expensive for more, but because each action maps 1:1 to a tested routine with known enter/tick/exit behavior. Adding actions means adding routines, each of which must handle interruption, failure recovery, and motor cleanup. Untested routines would degrade the safety guarantee that makes the planning layer viable in the first place. The planner's value is in *sequencing* existing behaviors, not in expanding the behavior repertoire.

### Bounded learning over unbounded optimization

The gradient learner uses Pearson correlation with +-20% drift bounds, not neural networks or unconstrained reinforcement learning. In an autonomous system where a divergent policy means a dead character and a lost session, a bounded optimizer that converges in ~100 encounters is preferable to a powerful optimizer that might not converge at all. The learning rate adapts per-weight (oscillation detection dampens, stagnation boosts) but the drift ceiling is fixed. This is a deliberate choice: the system trades optimal final performance for guaranteed safe convergence.

---

## Combat and Action

### Strategy pattern with runtime selection

Four CastStrategy implementations handle different combat approaches. The active strategy is selected based on character level, current threat assessment, and per-entity encounter history. This decouples "what to do in combat" from "when to be in combat." The combat routine orchestrates timing and target management while the strategy decides which abilities to use.

Adding a new strategy means implementing one class. No switch statements, no modification of existing strategies.

### Lock-in semantics for multi-step sequences

Some actions span multiple ticks and must not be interrupted by lower-priority rules. The [lock-in mechanism](architecture.md#layer-1-priority-rules-safety-envelope) solves this: without it, the rule engine would abandon a half-completed pull because the rest rule momentarily fired.

### Mana and resource conservation scaling with threat level

Resource spending scales with the situation. Against a weak target, use the minimum effective action set. Against a dangerous target or with extra npcs present, spend aggressively. This emerges naturally from the strategy selection (different strategies have different resource profiles) but is also enforced within strategies via threat-aware conditionals.

---

## Learning

### Learned data over hand-tuned heuristics

Every heuristic in the system has a [learned override](architecture.md#encounter-learning-per-encounter) that activates after sufficient data. The heuristics are the bootstrap; the learned data is the steady state.

This means the agent improves at any camp over multiple sessions without human intervention. It also means the agent handles novel environments gracefully, since heuristics provide reasonable behavior while data accumulates.

### Encounter outcome as the universal training signal

Every learning system in the agent trains on the same signal: encounter outcomes. Duration, resource spent, HP lost, survival, pet death, extra npcs encountered. This single event drives scoring weight updates, threshold tuning, strategy classification, and GOAP cost estimation.

One signal, many consumers. No separate reward functions, no synthetic objectives. The agent's actual experience is the training data.

### Finite-difference projected gradient descent with safety floors

Scoring weights tune via numerical gradient estimation: each weight is perturbed independently by a small epsilon, the effect on encounter fitness is measured through centered finite differences, and the update is projected back into the bounded region (+/-20% of defaults). This is true gradient descent -- not correlation-as-proxy -- with the bounded region acting as a projection step. Adaptive per-weight learning rates dampen oscillation and escape stagnation. Thresholds tune from outcome data, bounded by safety floors and efficiency ceilings. The agent cannot learn to never rest, never flee, or ignore isolation. Bounds prevent catastrophic drift while allowing meaningful adaptation.

### Thompson Sampling for target exploration

Encounter posteriors (Normal conjugate for duration/mana/danger, Beta conjugate for add probability and pet death rate) are maintained per entity type and sampled via Thompson Sampling in the target scoring path. Unknown targets have wide posteriors and occasionally score high, driving natural exploration. As observations accumulate, posteriors tighten and exploitation dominates. Per-encounter regret (chosen vs best-available fitness) is tracked; sublinear cumulative regret growth validates convergence. This replaces the fixed 5-encounter threshold with a smooth Bayesian transition from prior to learned data.

### Monte Carlo plan evaluation

GOAP candidate plans are evaluated via stochastic rollouts: action effects are perturbed with Gaussian noise on resource fields (HP, mana) to simulate outcome uncertainty. A plan that achieves high goal satisfaction across noisy rollouts is robust to the inherent variance in combat outcomes, rest durations, and resource costs. This is planning under uncertainty -- the deterministic A* search finds candidates, and Monte Carlo evaluation selects the most robust one.

### Spawn prediction from encounter history

Defeat timestamps and locations feed a per-cell Poisson process. The planner uses predicted respawn times to position the agent before targets appear, converting random wandering into directed positioning. Prediction confidence requires 3+ defeats per cell; stale data decays with a 4-hour half-life.

---

## Navigation

### Terrain parsed from game asset files

The terrain heightmaps are derived from the game client's own 3D mesh data. This is ground truth: the same geometry the client renders. Alternatives like sampling movement outcomes or building maps from exploration data would be slow, incomplete, and error-prone.

### 1-unit resolution heightmaps

Each cell in the terrain grid represents 1 game unit (roughly half a meter in world scale). At ~80 MB per zone, this is expensive in storage but provides the fidelity needed for precise pathfinding near cliffs, water edges, and narrow passages. An earlier 10-unit grid missed critical gaps and produced paths that walked off ledges.

### Waypoint graph for complex terrain

Pure grid-based A* fails in areas with bridges, tunnels, spiral ramps, and other 3D structures that project ambiguously onto a 2D heightmap. Pre-recorded waypoint graphs provide known-safe routes through these areas. The agent switches to waypoint following when entering a complex region and returns to A* pathfinding when back on open terrain.

### Danger zones with cost inflation

Areas with aggressive high-level NPCs or environmental hazards are marked as danger zones with a configurable radius. The A* cost function inflates path cost within these radii, causing the pathfinder to route around them when possible. This is cheaper than maintaining a dynamic threat map and works well for static hazards. Dynamic threats (patrols, hostile players) are handled by the GOAP planner's trajectory forecasting.

---

## Survival

### Hysteresis on all threshold-based decisions

Every threshold that triggers a state change uses hysteresis (different entry and exit thresholds). Rest entry at HP <85% / mana <25%, rest exit at HP >=92% / mana >=60%. Without hysteresis, the agent oscillates: rest one tick, pull the next, rest again, pull again. Hysteresis was added after observing exactly this failure mode in early versions. Thresholds themselves auto-tune from outcome data (see Learning section above).

### Attack detection during rest

The agent monitors for incoming damage while resting. If HP drops or an NPC targets the player, the rest routine exits immediately and the survival rules take over. Early versions would sit and meditate while being attacked because the rest routine held lock-in. Now rest never holds lock-in, and the survival rules have higher priority.

### Emergency override hierarchy

FLEE outranks everything. It can interrupt locked routines, override combat, cancel travel, and invalidate GOAP plans. The priority order is absolute: flee > death recovery > combat > maintenance > navigation. This hierarchy has never needed rethinking. It is the safety guarantee that allows every other system to optimize freely.

---

## Observability

### Decision branch coverage

The [logging contract](architecture.md#decision-coverage) treats a silent `return False` as a defect. The driving question: can we reconstruct exactly what the agent decided and why from the logs alone?

### Rate-limited hot-path logging

Scoring and profiling functions run per-entity per-tick (potentially 10+ calls per second per entity). Unthrottled logging in these paths would produce millions of lines per session. Hot-path logs use `emit_throttled()` or periodic summaries instead.

---

## Timing and Input

### Input pacing across multiple timescales

Motor output operates across three timing scales: sub-50ms for individual key press and release cadence; 100-500ms for tactical action sequencing to allow game state to settle between commands; and session-scale variation in movement patterns and rest durations. Timing follows log-normal rather than uniform distributions, which produces natural clustering around expected durations rather than flat-random spread. The three scales are independently tunable and compose into the agent's overall action cadence.

---

## Why Classic EverQuest

Classic EverQuest concentrates most of the hard problems in autonomous agent design into a single domain:

- **Partial observability**: world state is never fully known
- **Real-time pressure**: decisions have millisecond-scale consequences
- **Competing goals**: combat, rest, flee, and recover all contend for the same action slot
- **Complex 3D terrain**: cliffs, water, lava, bridges, tunnels, zone boundaries
- **Social threat**: engaging one entity can trigger others to join
- **Resource constraints**: health, mana, inventory, cooldowns
- **Failure recovery**: death, obstacle negotiation, state desync
- **Long-horizon autonomy**: multi-hour sessions where compounding errors surface
