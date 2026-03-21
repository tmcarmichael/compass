<!-- last_modified: 2026-03-31 -->

> Headless simulator output: 10 sessions of `camp_session` with learning state preserved across sessions.
> Run with `just simulate converge --sessions 10`.
> Fight duration drops 53% as encounter history accumulates and cost functions self-correct.
> No game client required.

```
Convergence: camp_session x 10 sessions

 Session  Grade   Ticks   p99 ms  Fights   Avg Dur   Drift  GOAP %
----------------------------------------------------------------------
       1      A    1280     1.1       8     22.2s    0.0%      0%
       2      A    1280     0.1      16     20.5s    0.0%      0%
       3      A    1280     0.1      24     19.4s    0.0%      0%
       4      A    1280     0.1      30     18.3s    0.0%      0%
       5      A    1280     0.1      30     16.6s    0.0%      0%
       6      A    1280     0.1      30     15.3s    0.0%      0%
       7      A    1280     0.1      30     14.1s    0.0%      0%
       8      A    1280     0.1      30     12.8s    0.0%      0%
       9      A    1280     0.1      30     11.5s    0.0%      0%
      10      A    1280     0.1      30     10.4s    0.0%      0%

Fight duration: 22.2s -> 10.4s (53% improvement)
Grade: A -> A
```

**What's happening across sessions.** Each session runs 8 pull/combat/rest cycles against synthetic skeletons. After each session, encounter outcomes feed back into the learning stack: per-NPC duration and mana posteriors tighten (Normal-Normal conjugate updates), GOAP cost estimates self-correct via exponential moving average, and the fight history sliding window accumulates data. By session 4-5, `learned_duration()` returns values and the planner's cost function reflects actual combat time rather than the initial 25-second prior. Duration converges toward the 8-second floor imposed by the synthetic scenario's minimum combat length.

**Columns.** `Grade` is the session scorecard (7 weighted categories: throughput, survival, pull success, uptime, pathing, mana efficiency, targeting). `Fights` is cumulative encounter count (capped at the 30-sample sliding window). `Avg Dur` is the learned average fight duration in seconds. `Drift` is the maximum weight deviation from defaults (0% here because the synthetic scenario doesn't produce full target-scoring breakdowns needed for gradient tuning). `GOAP %` is plan completion rate. `p99 ms` is 99th-percentile tick time, confirming the decision stack stays within the 100ms real-time budget.
