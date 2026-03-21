<!-- last_modified: 2026-03-28 -->

> Example GOAP planner output showing the structured event format produced by
> `brain/goap/planner.py`. The planner runs forward A\* on the goal-state space
> to find the cheapest action sequence satisfying the most insistent goal.
> Costs are learned from encounter history and self-corrected via EMA.

### Plan generation (low resources, rest-first)

```json
{"event":"goap_plan","ts":"2026-03-27T23:18:42.108-05:00","elapsed":1259,"session_id":"session_20260327_225743","level":"INFO","logger":"compass.brain.goap.planner",
 "goal":"MANAGE_RESOURCES","steps":4,"cost":62.4,"satisfaction":0.82,"plan_ms":1.3,
 "msg":"[GOAP] Generated: Plan(MANAGE_RESOURCES: rest -> acquire -> pull -> defeat, cost=62.4s, sat=0.82) in 1.3ms"}
```

### Cost accuracy tracking (per-step estimated vs actual)

```json
{"event":"goap_step","ts":"2026-03-27T23:19:11.482-05:00","elapsed":1288,"step":"rest","estimated_s":29.4,"actual_s":28.7,"error_s":-0.7}
{"event":"goap_step","ts":"2026-03-27T23:19:16.103-05:00","elapsed":1293,"step":"acquire","estimated_s":5.0,"actual_s":4.6,"error_s":-0.4}
{"event":"goap_step","ts":"2026-03-27T23:19:19.817-05:00","elapsed":1296,"step":"pull","estimated_s":8.0,"actual_s":3.7,"error_s":-4.3}
{"event":"goap_step","ts":"2026-03-27T23:19:47.294-05:00","elapsed":1324,"step":"defeat","estimated_s":20.0,"actual_s":27.5,"error_s":7.5}
```

### Plan completion

```json
{"event":"goap_complete","ts":"2026-03-27T23:19:47.295-05:00","elapsed":1324,"session_id":"session_20260327_225743","level":"INFO","logger":"compass.brain.goap.planner",
 "goal":"MANAGE_RESOURCES","steps_total":4,"steps_executed":4,
 "msg":"[GOAP] Plan completed: MANAGE_RESOURCES (4/4 steps)"}
```
