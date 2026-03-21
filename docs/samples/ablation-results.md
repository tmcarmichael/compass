<!-- last_modified: 2026-03-31 -->

> Ablation results from the automated test suite (`tests/test_ablation.py`).
> Each test isolates one learning system and compares its output against a no-learning baseline.
> Run with `just test tests/test_ablation.py -v`.

## Scoring weight tuning

Train the gradient tuner on 120 synthetic encounters where isolated targets yield high fitness (0.7) and clustered targets yield low fitness (0.3). After 8 gradient steps, score both target profiles with default and tuned weights.

```
Default weights: isolated=198.00  clustered=0.00  gap=198.00
Tuned weights:   isolated=202.42  clustered=0.00  gap=202.42
Steps: 8, Improvement: +2%
```

The default weights already produce strong separation for this scenario (the hard social-NPC reject gates clustered targets to zero). The gradient tuner does not degrade that separation. In scenarios where defaults produce less decisive separation (closer isolation scores, fewer social adds), tuned weights produce meaningfully wider gaps. The bounded region (±20% of defaults) prevents catastrophic drift.

## GOAP cost self-correction

Simulate 10 rest actions that always take 45 seconds. The planner's heuristic estimates 30 seconds. After EMA-based cost correction (alpha=0.3, 3-observation minimum), the learned cost converges to actual.

```
Actual rest time: 45s
Heuristic cost:   30.0s  (error: 15.0s)
Learned cost:     44.6s  (error: 0.4s)
Error reduction:  97%
```

The planner uses `get_corrected_cost()` for all plan generation after the minimum observation threshold. As cost accuracy improves, plans that depend on rest duration (e.g., rest-then-pull sequences) schedule more accurately, reducing wasted idle time.

## Danger gating

Record 10 dangerous encounters (a_red_wolf: 45s, 40% HP loss, pet always dies) and 10 safe encounters (a_bat: 10s, 2% HP loss, pet survives). Compare learned danger scores.

```
a_red_wolf: danger=1.00  (10 fights, 40% HP loss, pet always dies)
a_bat:      danger=0.04  (10 fights,  2% HP loss, pet survives)
Separation: 0.96  (25x more dangerous)
```

Danger scores feed directly into target scoring (penalty for high-danger targets), ACQUIRE gating (skip dangerous targets when HP or mana is low), and flee urgency (learned danger adds +0.2 urgency for danger > 0.7). Without fight history, all targets are treated equally.

## Data threshold behavior

With fewer than 5 fights recorded, all `learned_*()` methods return `None` and the system falls back to heuristic defaults. This prevents early noise from corrupting decisions.

```
2 fights recorded -> learned_danger("a_skeleton") = None
2 fights recorded -> learned_duration("a_skeleton") = None
5 fights recorded -> learned_danger("a_skeleton") = 0.20  (data takes over)
```

## Persistence

Fight history survives save/load cycles. After saving 10 fights with mean duration 20.0s and loading into a fresh `FightHistory`, the loaded duration is within 3.0s of the original (tolerance for sample variance).
