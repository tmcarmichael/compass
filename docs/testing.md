<!-- last_modified: 2026-03-30 -->

# Testing

How the test suite is structured, what it covers, and what it deliberately omits.

## Running tests

```bash
uv sync               # install dev dependencies
uv run pytest          # run all tests
uv run pytest --cov    # run with coverage report
just check             # lint + format + typecheck + test (mirrors CI)
```

Hypothesis profiles: `dev` (50 examples, fast) and `ci` (200 examples, thorough).

## Coverage strategy

Coverage is measured in CI with `pytest-cov` and enforced with a `fail_under` floor of 70% in `pyproject.toml`. The measured surface is the full source with zero omissions or exclusion pragmas. Three categories of code are structurally difficult to cover in CI:

- **Perception layer** (`src/perception/reader.py`, `spawn_reader.py`): Win32 `ReadProcessMemory` calls require a live game client on Windows. Struct parsing and state assembly are tested via crafted buffers; the I/O boundary is not.
- **Multi-phase routine ticks** (`src/routines/pull.py`, `combat_phases.py`, `casting.py`, etc.): These are motor-coupled state machines that advance through 5-10 phases with real timing and game state feedback. Core logic is extracted into pure functions and tested directly; the phase orchestration is validated through runtime invariants and forensics.
- **Binary asset parsers** (`src/eq/`, `nav/terrain/heightmap.py`): Full parsing requires game asset files not included in the repo. Synthetic binary fixtures cover format parsing; geometry construction and terrain cache building are tested with crafted grids.

### What is covered

The test suite exercises the full architecture from decision logic through routine execution:

- **Brain decision engine**: rule evaluation, priority ordering, cooldowns, circuit breakers, utility scoring phases 0–4, routine lifecycle (enter/tick/exit)
- **GOAP planner**: plan generation, advancement, invalidation, cost correction, goal satisfaction
- **Learning systems**: encounter history with Thompson Sampling posteriors, spatial memory, session scorecard, finite-difference weight tuning, danger memory, regret tracking
- **Scoring**: target scoring with 15-factor utility curves, Pareto efficiency, weight learner convergence, Monte Carlo plan evaluation
- **Rules**: survival (flee, rest, death recovery, evade), combat (acquire, pull, engage), maintenance (buff, summon, memorize), navigation (travel, wander)
- **Routine execution**: rest, flee, buff routines tested with RecordingMotor backend (real enter/tick/exit lifecycle, real motor commands captured)
- **Navigation**: A\* and JPS pathfinding on synthetic grids, zone graph routing, waypoint graphs, geometry, heightmap construction and cliff detection
- **State models**: GameState, SpawnData, PlanWorldState, AgentContext sub-states
- **Observability**: forensics buffer, event schemas, metrics, invariant checks, session analysis
- **Pipeline integration**: `register_all()` with real rules and real routines, verifying perception → brain → routines → motor data flow
- **Infrastructure**: camp selection, zone progression, tick handlers, session lifecycle

### Motor backend architecture

Routines call motor functions (`sit()`, `tab_target()`, `press_gem()`) which delegate to a pluggable `MotorBackend`. In production, the backend sends OS-level keyboard input. In tests, a `RecordingMotor` captures all actions in a list without sleeping or sending input:

```python
# conftest.py installs this automatically for all tests
from motor.actions import set_backend
from motor.recording import RecordingMotor

recorder = RecordingMotor()
set_backend(recorder)
```

This makes every routine testable without mocking, monkeypatching, or environment access. Tests assert on `recorder.actions` to verify motor commands.

### Previously omitted modules

Seven modules that were previously excluded from coverage measurement are now fully measured and tested via mock-based testing:

- **Perception layer** (`src/perception/*`): Win32 `ReadProcessMemory` calls are tested through mock readers that return crafted entity struct buffers. `read_state()` assembly, profile chain resolution, struct validation, log parsing, and all sub-readers (char, spawn, inventory) are covered.
- **Binary asset parsers** (`src/eq/s3d.py`, `wld.py`, `zone_chr.py`): tested with synthetic binary fixtures via `struct.pack`.
- **Runtime orchestration** (`src/runtime/orchestrator.py`, `agent.py`): config building, feature toggles, log buffer, session pruning tested with mock dependencies.

### Compensating controls for deeply-coupled code

The remaining uncovered lines are concentrated in multi-phase routine tick handlers (motor-coupled state machines) and the 10 Hz brain runner loop (threading, I/O). These are validated through:

1. **Runtime invariants** (`src/util/invariants.py`): assertions that fire during operation and flush diagnostics on violation
2. **Forensics buffer** (`src/util/forensics.py`): 300-tick ring buffer that dumps to disk on death or crash, providing post-hoc debugging for the perception/routine layers
3. **4-tier logging**: every decision branch logs its reasoning, so failures in production can be reconstructed from session logs
4. **Session scorecard**: automated performance grading every 30 minutes flags regressions that unit tests cannot catch

## Test patterns

**Factory-first testing.** The test suite uses factory functions and dependency injection as the primary testing strategy. `unittest.mock.patch` is used sparingly and only at system boundaries (perception I/O, file system). All domain logic is tested through constructor-injected dependencies, not mocks.

**Factory functions** (`tests/factories.py`): `make_game_state()`, `make_spawn()`, `make_plan_world_state()`, `make_mob_profile()` construct frozen dataclasses with sensible defaults. Tests override only the fields they care about.

**Dependency injection**: time-dependent classes accept a `clock` parameter (`CircuitBreaker`, `SpatialMemory`, `DangerMemory`, `InvariantChecker`, `StuckDetector`). Rule modules accept `spell_provider` and `buff_routine` parameters. Tests pass controlled implementations without monkeypatching module state.

**Hypothesis strategies**: `st_game_state`, `st_spawn`, `st_plan_world_state` compose factory functions for property-based testing.

**Parametrized scenarios**: rule condition tests use `@pytest.mark.parametrize` with named test IDs for readable output.

**Pipeline integration** (`tests/test_pipeline.py`): wires `register_all()` with real rule conditions, real routines, and real `AgentContext`, feeding scripted `GameState` sequences. Verifies that brain decisions produce motor commands through the full four-stage pipeline.

**Stateful property tests** (`tests/test_safety_envelope.py`): a `hypothesis.stateful.RuleBasedStateMachine` exercises the Brain's safety envelope under random state sequences, verifying that emergency rules always override, locked routines are respected, and cooldowns are honored across arbitrary tick sequences.

**Session simulation** (`tests/test_session_simulation.py`): a `SessionSimulator` drives the Brain through multi-phase `GameState` sequences using `register_all()` with real rules. `ScenarioBuilder` constructs scripted scenarios (idle, damage, drain mana, recover). Tests reproduce documented behaviors from `docs/samples/` telemetry, including the forensics ring buffer skeleton-aggro incident.

**Learning invariants** (`tests/test_learning_invariants.py`): property-based tests verifying structural guarantees: finite-difference gradient weight drift stays within +-20% of defaults under arbitrary fitness sequences, scorecard tuning parameters stay within declared bounds after repeated evaluation, and encounter history respects `MIN_FIGHTS_FOR_LEARNED` and `MAX_SAMPLES` contracts.

**Headless simulator** (`src/simulator/`): runs the full decision stack through synthetic perception with three modes. `benchmark` verifies tick timing at 10 Hz. `replay` drives built-in scenarios (camp session, survival stress, exploration) and reports decision traces. `converge` runs multiple sessions with preserved learning state to verify improvement over time. Run via `python3 -m simulator`.

**Adversarial sequences** (`tests/test_adversarial.py`): Hypothesis-driven tests feeding arbitrary `GameState` sequences through the Brain to verify it never crashes. Includes edge cases (zero HP max, negative HP, 200-spawn lists) and timing budget assertions (single tick under 100ms).

**Observability contracts** (`tests/test_observability_contracts.py`): verifies that decision receipts cover all registered rules, forensics buffer respects capacity and eviction order, and structured events carry expected fields.

**Architecture enforcement** (`tests/test_import_dag.py`): AST-based tests verifying the forward-only import DAG: routines cannot import brain.decision, motor cannot import brain or routines, perception cannot import upper layers. Also verifies all `Point()` constructors include the z coordinate.

**Pure function tests** (`tests/test_routine_pure_functions.py`): unit tests for decision functions extracted from routines (`classify_dot_fizzle`, `verify_cast_landed`, `choose_pull_strategy`, `should_attempt_gate`, `should_cast_regen_buff`, `should_exit_rest`). No mocks, no motor, no state machines.

**RecordingMotor** (`src/motor/recording.py`): installed by conftest for all tests. Captures motor action sequences for assertion. Eliminates artificial `time.sleep()` delays so the full suite runs in tens of seconds instead of spending minutes blocked on routine timing.
