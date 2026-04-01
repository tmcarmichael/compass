# Single-Tick Walkthrough

What happens in one tick of the 10 Hz brain loop, traced through the actual source code. This walkthrough follows tick 1147 from [`docs/samples/decision-trace.md`](samples/decision-trace.md), the moment the agent transitions from WANDER to ACQUIRE after spotting a valid target.

---

## 1. Clock gate

[`brain/runner/loop.py:541`](../src/brain/runner/loop.py) -- The `TickClock` blocks until the next 100ms boundary. When it releases, the tick counter increments and the loop begins.

```
_clock.wait_for_next_tick()
```

The clock measures `dt` (actual elapsed time since last tick). At 10 Hz with no contention, dt is ~100ms. If the previous tick overran, dt is larger and the clock skips the sleep entirely.

## 2. Perception snapshot

[`brain/runner/loop.py:546`](../src/brain/runner/loop.py) -- `_tick_pre_state` calls `reader.read_state(include_spawns=True)`, which traverses the game client's internal pointer chains via `ReadProcessMemory` and returns a frozen `GameState` dataclass. This snapshot is immutable after creation: the brain thread produces it, every subsequent step consumes it, no locks needed.

The `GameState` contains: player position (x, y, z), HP/mana current and max, target info, heading, zone ID, combat flag, sitting flag, and a tuple of `SpawnData` for every visible entity.

If the memory read fails (process not responding, pointer chain broken), the tick is skipped entirely -- the brain never evaluates stale state.

## 3. World state update

[`brain/runner/loop.py:559`](../src/brain/runner/loop.py) -- `WorldStateUpdater.update_world_state()` refreshes the derived world model: NPC tracking (who appeared, who moved, who despawned), pet status, threat detection (approaching hostile entities), and health monitoring. This runs before rule evaluation so that rules see current-tick derived state.

For tick 1147: the world model registers a level-appropriate NPC at 85 units distance that was not present on the previous tick. The threat scanner classifies it as non-hostile (not approaching). The entity enters the world model's tracking table.

## 4. Pre-rule event detection

[`brain/runner/loop.py:570-572`](../src/brain/runner/loop.py) -- Three pre-rule handlers run: XP tracking (records any XP change), add detection (checks if a new NPC is attacking the pet), and auto-engage scanning (detects if the current target changed). These update `AgentContext` sub-state objects before the decision engine reads them.

## 5. Rule evaluation (the decision)

[`brain/runner/loop.py:593`](../src/brain/runner/loop.py) -- `_tick_brain` calls `brain.tick(state)`, entering the decision engine. Inside [`brain/decision.py:173-180`](../src/brain/decision.py):

```python
def tick(self, state: GameState) -> None:
    tick_start = self.perf_clock()
    now = time.time()
    selected, selected_name, selected_emergency = self._evaluate_rules(state, now)
    handle_transition(self, state, selected, selected_name, selected_emergency, now)
    tick_active_routine(self, state, now)
    tick_profiling(self, tick_start)
```

### 5a. Rule scan

[`brain/decision.py:182`](../src/brain/decision.py) -- At Phase 0 (binary conditions, insertion-order priority), rules evaluate top to bottom. The first rule whose condition returns `True` wins. After a winner is found, remaining rules are skipped.

For tick 1147, the evaluation cascade looks like this (from the decision receipt in the sample data):

| Rule | Result | Why |
|------|--------|-----|
| DEATH_RECOVERY | no | player is not dead |
| FEIGN_DEATH | no | flee disabled check passes, but no safety floor fires |
| FLEE | no | urgency 0.000 < 0.65 threshold |
| REST | no | HP 100%, mana 78%, pet HP 95% -- all above entry thresholds |
| EVADE | no | no evasion point set |
| BUFF | no | buff recently cast |
| COMBAT_MONITOR | no | not engaged |
| ACQUIRE | **YES** | valid target within scan radius, not recently defeated, level-appropriate |
| PULL | skip | already have a winner |
| IN_COMBAT | skip | already have a winner |
| WANDER | skip | already have a winner |

Emergency rules (DEATH_RECOVERY, FEIGN_DEATH, FLEE) always evaluate, even when a locked routine is active. The agent cannot learn its way into ignoring a lethal threat.

### 5b. Transition

[`brain/transitions.py`](../src/brain/transitions.py) -- The selected routine (ACQUIRE) differs from the currently active routine (WANDER). Since WANDER is not locked, the transition proceeds: `wander.exit(state)` is called, then `acquire.enter(state)` begins the new routine.

## 6. Routine tick

[`brain/completion.py`](../src/brain/completion.py) -- After the transition, `tick_active_routine` calls `acquire.tick(state)`. The acquire routine's first tick initializes target selection: it reads visible spawns from the `GameState`, filters by level range and disposition, scores candidates using the 15-factor utility function, and issues a `tab_target()` motor command toward the highest-scoring entity.

The routine returns `RUNNING`. It will continue ticking on subsequent cycles until the target is acquired (returns `SUCCESS`) or the attempt fails (returns `FAILURE`).

## 7. Motor output

Motor commands issued during `acquire.tick()` flow through [`motor/actions.py`](../src/motor/actions.py) to the pluggable `MotorBackend`. In production, this sends OS-level keyboard input to the game client window. In tests, `RecordingMotor` captures the command sequence without side effects.

For this tick: `tab_target()` sends a Tab keypress. The next perception snapshot (tick 1148) will reflect whether the game client acquired the target.

## 8. Observability

[`brain/runner/loop.py:599`](../src/brain/runner/loop.py) -- `_tick_record_diag` runs after the brain tick completes:

- **Decision receipt**: a structured record of which rules fired, their scores, the selected routine, lock state, and tick timing. This is the data in [`docs/samples/decision-trace.md`](samples/decision-trace.md).
- **Forensics ring buffer**: the last 300 ticks of brain state, continuously overwritten. On death or crash, this buffer flushes to disk -- 30 seconds of pre-incident telemetry.
- **Tick metrics**: wall time for the full tick and for the routine's tick() call, recorded for performance monitoring.
- **Invariant checks**: structural assertions (e.g., engaged flag consistent with target state) that log warnings on violation.

## 9. Heartbeat

[`brain/runner/loop.py:598`](../src/brain/runner/loop.py) -- The heartbeat timestamp updates. A secondary thread monitors this value; if it goes stale for 10+ seconds, the watchdog triggers recovery (process reconnection or graceful shutdown).

---

## Timing budget

The entire sequence -- perception read, world update, rule evaluation, routine tick, motor output, diagnostics -- targets a 100ms loop cadence. A typical decision tick takes 2-8ms. Routine authors treat 200ms as a cooperative soft budget, longer waits use interruptible sleeps that can break for emergencies, and any single `tick()` that runs past 5 seconds is force-exited as hung.

If a tick overruns, the clock compensates by shortening the next sleep. The agent never drops ticks; it runs them late rather than skipping them.
