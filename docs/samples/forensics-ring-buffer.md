<!-- last_modified: 2026-03-28 -->

> Real excerpt from `session_20260327_220547_forensics.jsonl`, ticks 41-55.
> Shows emergency interrupt: player sitting at full HP memorizing spells, skeleton aggroes,
> HP drops 136 -> 126 in one tick, brain fires FLEE and overrides the locked routine.
> The 300-entry ring buffer captures ~30 seconds of per-tick state at 10 Hz (~150 bytes/entry,
> ~45 KB max). Flushed to disk on death, flee, or shutdown. No deaths exist in the current
> session archive -- this is the closest real incident (aggro interrupt -> flee).
> Character name redacted.

```jsonl
{"tick":41,"ts":"2026-03-28T03:05:56.268+00:00","hp":136,"hp_max":136,"mana":259,"x":894.3,"y":-4.2,"hdg":277,"tgt":"","tgt_hp":0,"engaged":false,"routine":"MEMORIZE_SPELLS"}
{"tick":42,"ts":"2026-03-28T03:05:56.370+00:00","hp":136,"hp_max":136,"mana":259,"x":894.3,"y":-4.2,"hdg":277,"tgt":"","tgt_hp":0,"engaged":false,"routine":"MEMORIZE_SPELLS"}
{"tick":43,"ts":"2026-03-28T03:05:56.470+00:00","hp":136,"hp_max":136,"mana":259,"x":894.3,"y":-4.2,"hdg":277,"tgt":"","tgt_hp":0,"engaged":false,"routine":"MEMORIZE_SPELLS"}
{"tick":44,"ts":"2026-03-28T03:05:56.569+00:00","hp":136,"hp_max":136,"mana":259,"x":894.3,"y":-4.2,"hdg":277,"tgt":"","tgt_hp":0,"engaged":false,"routine":"MEMORIZE_SPELLS"}
{"tick":45,"ts":"2026-03-28T03:05:56.669+00:00","hp":136,"hp_max":136,"mana":259,"x":894.3,"y":-4.2,"hdg":277,"tgt":"","tgt_hp":0,"engaged":false,"routine":"MEMORIZE_SPELLS"}
{"tick":46,"ts":"2026-03-28T03:05:56.771+00:00","hp":136,"hp_max":136,"mana":259,"x":894.3,"y":-4.2,"hdg":277,"tgt":"","tgt_hp":0,"engaged":false,"routine":"MEMORIZE_SPELLS"}
{"tick":47,"ts":"2026-03-28T03:05:56.870+00:00","hp":136,"hp_max":136,"mana":259,"x":894.3,"y":-4.2,"hdg":277,"tgt":"","tgt_hp":0,"engaged":false,"routine":"MEMORIZE_SPELLS"}
{"tick":48,"ts":"2026-03-28T03:05:57.107+00:00","hp":136,"hp_max":136,"mana":259,"x":894.3,"y":-4.2,"hdg":277,"tgt":"","tgt_hp":0,"engaged":false,"routine":"MEMORIZE_SPELLS"}
{"tick":49,"ts":"2026-03-28T03:05:57.416+00:00","hp":126,"hp_max":136,"mana":259,"x":894.4,"y":-4.2,"hdg":277,"tgt":"a_skeleton008","tgt_hp":100,"engaged":false,"routine":"FLEE"}
{"tick":50,"ts":"2026-03-28T03:05:57.423+00:00","hp":126,"hp_max":136,"mana":259,"x":894.4,"y":-4.2,"hdg":277,"tgt":"a_skeleton008","tgt_hp":100,"engaged":false,"routine":"FLEE"}
{"tick":51,"ts":"2026-03-28T03:05:57.522+00:00","hp":126,"hp_max":136,"mana":259,"x":894.4,"y":-4.2,"hdg":277,"tgt":"a_skeleton008","tgt_hp":100,"engaged":false,"routine":"FLEE"}
{"tick":52,"ts":"2026-03-28T03:05:57.624+00:00","hp":126,"hp_max":136,"mana":259,"x":894.4,"y":-4.2,"hdg":277,"tgt":"a_skeleton008","tgt_hp":100,"engaged":false,"routine":"FLEE"}
{"tick":53,"ts":"2026-03-28T03:05:57.722+00:00","hp":126,"hp_max":136,"mana":259,"x":894.4,"y":-4.2,"hdg":277,"tgt":"a_skeleton008","tgt_hp":100,"engaged":false,"routine":"FLEE"}
{"tick":54,"ts":"2026-03-28T03:05:57.822+00:00","hp":126,"hp_max":136,"mana":259,"x":894.4,"y":-4.2,"hdg":277,"tgt":"a_skeleton008","tgt_hp":100,"engaged":false,"routine":"FLEE"}
{"tick":55,"ts":"2026-03-28T03:05:57.923+00:00","hp":126,"hp_max":136,"mana":259,"x":894.4,"y":-4.2,"hdg":277,"tgt":"a_skeleton008","tgt_hp":100,"engaged":false,"routine":"FLEE"}
```
