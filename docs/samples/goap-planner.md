<!-- last_modified: 2026-04-01 -->

> Full GOAP planner trace showing goal evaluation, A\* search, Monte Carlo
> robustness gate, plan execution with cost tracking, and cost self-correction.
> Output from `brain/goap/planner.py`. Timestamps, session IDs, and elapsed
> times are from a real session; the sequence below is one complete plan cycle.

### 1. Goal insistence evaluation

When a plan is needed (previous plan completed or invalidated), the planner
evaluates all five goals against the current `PlanWorldState`. The most
insistent goal becomes the planning target.

```
world state: hp=62% mana=35% pet=alive engaged=no targets=3 threats=0 inv=22%

  SURVIVE           sat=0.62  insistence=0.38   (1.0 - 0.62) * 1.0
  MAINTAIN_READINESS sat=1.00  insistence=0.00
  MANAGE_RESOURCES  sat=0.69  insistence=0.10   mana below 70% threshold
  GAIN_XP           sat=0.50  insistence=0.35   targets available, resources permit
  MANAGE_INVENTORY  sat=1.00  insistence=0.00

selected: SURVIVE (insistence 0.38 > GAIN_XP 0.35)
```

SURVIVE wins because HP at 62% produces higher insistence than GAIN_XP at
the same mana level. If HP were above 80%, GAIN_XP would dominate.

### 2. A\* search on goal-state space

The planner runs forward A\* from the current world state. Each node expands
by trying all actions whose preconditions are met, applying their effects to
produce child states. Duplicate adjacent actions (rest -> rest) are pruned.
Search terminates when goal satisfaction reaches 0.70.

```
[GOAP] A* search for SURVIVE (sat=0.62, threshold=0.70)

  node 0  depth=0  state: hp=62% mana=35%  g=0.0  h=2.4  f=2.4
    expand: rest (not engaged, no threats -> met)
    expand: acquire (targets>0, mana>25%, pet alive -> met)
    expand: wander (targets=0? no -> skip)
    expand: buff (buffs already active -> skip)
    ...

  node 1  depth=1  state: hp=95% mana=80%  g=18.0  h=0.0  f=18.0
    action: rest  cost=18.0s (corrected: base=18.0 + ema=0.0)
    goal satisfaction: 0.95 >= 0.70 -- candidate found

  nodes visited: 2 / 500 budget
  search time: 0.3ms / 50ms budget
```

The heuristic `h = (0.70 - sat) * 30.0` estimates remaining cost in seconds.
With sat=0.62, h = 0.08 * 30 = 2.4s. After rest, sat=0.95 and h drops to 0.

### 3. Monte Carlo robustness gate

Before accepting the candidate plan, the planner runs 20 stochastic rollouts.
Each rollout applies action effects with Gaussian noise drawn from learned
posterior variance (encounter history). Plans that fail under noisy outcomes
are rejected and the search continues.

```
[GOAP] MC evaluation: 20 rollouts, learned sigma hp=0.08 mana=0.11

  rollout  1: rest -> hp=93% mana=74%  sat=0.93
  rollout  2: rest -> hp=97% mana=82%  sat=0.97
  ...
  rollout 20: rest -> hp=91% mana=78%  sat=0.91

  mean satisfaction: 0.94  (threshold: 0.50)
  ACCEPTED -- plan robust under posterior uncertainty
```

If the planner had wider posteriors (fewer observations), sigma would be
larger, more rollouts would produce low satisfaction, and marginal plans
would be rejected. The gate naturally penalizes plans built on uncertain data.

### 4. Plan generation event

The structured event emitted when a plan is accepted:

```json
{"event":"goap_plan","ts":"2026-03-27T23:18:42.108-05:00","elapsed":1259,
 "session_id":"session_20260327_225743","level":"INFO",
 "logger":"compass.brain.goap.planner",
 "goal":"SURVIVE","steps":1,"cost":18.0,"satisfaction":0.94,"plan_ms":0.3,
 "msg":"[GOAP] Generated: Plan(SURVIVE: rest, cost=18.0s, sat=0.94) in 0.3ms"}
```

### 5. Plan execution with cost tracking

As each step's routine runs to completion, the planner records estimated vs
actual duration. The example below shows a longer plan (GAIN_XP, 4 steps)
to demonstrate multi-step cost tracking:

```json
{"event":"goap_step","step":"rest",    "estimated_s":18.0,"actual_s":19.3,"error_s":+1.3}
{"event":"goap_step","step":"acquire", "estimated_s": 5.0,"actual_s": 4.6,"error_s":-0.4}
{"event":"goap_step","step":"pull",    "estimated_s": 8.0,"actual_s": 3.7,"error_s":-4.3}
{"event":"goap_step","step":"defeat",  "estimated_s":15.9,"actual_s":17.2,"error_s":+1.3}
```

The `defeat` step uses a learned cost (15.9s from encounter history averages)
rather than the 25.0s default. Early sessions show larger errors; as the
encounter posterior tightens, estimated and actual converge.

### 6. Cost self-correction (EMA)

Each step's error feeds an exponential moving average per action type
(alpha=0.3). After 3+ observations, the corrected cost replaces the base
estimate in future A\* searches:

```
cost corrections after 8 plan cycles:

  rest:     base=18.0s  ema_error=+1.1s  corrected=19.1s  (5 observations)
  acquire:  base= 5.0s  ema_error=-0.3s  corrected= 4.7s  (5 observations)
  pull:     base= 8.0s  ema_error=-3.8s  corrected= 4.2s  (5 observations)
  defeat:   base=15.9s  ema_error=+0.9s  corrected=16.8s  (5 observations)
  wander:   base=30.0s  ema_error=-8.2s  corrected=21.8s  (3 observations)
```

Corrected costs are floored at 10% of the base cost to prevent collapse.
Pull's large negative correction reflects the heuristic overestimating pull
time. Corrections persist across sessions via JSON.

### 7. Plan invalidation

Plans are invalidated when an emergency rule fires, the world state changes
enough to break a step's preconditions, or the goal is satisfied early:

```json
{"event":"goap_invalidate","ts":"2026-03-27T23:22:05.771-05:00",
 "goal":"GAIN_XP","step":2,"steps_total":4,"reason":"emergency_flee",
 "msg":"[GOAP] Plan invalidated: GAIN_XP at step 2/4 (reason: emergency_flee)"}
```

After invalidation, the planner generates a fresh plan on the next routine
completion. The cost corrections from the invalidated plan's completed steps
are retained -- partial execution still improves cost accuracy.

### 8. Plan completion

```json
{"event":"goap_complete","ts":"2026-03-27T23:19:47.295-05:00",
 "session_id":"session_20260327_225743","level":"INFO",
 "logger":"compass.brain.goap.planner",
 "goal":"GAIN_XP","steps_total":4,"steps_executed":4,
 "msg":"[GOAP] Plan completed: GAIN_XP (4/4 steps)"}
```
