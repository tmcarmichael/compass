<!-- last_modified: 2026-03-28 -->

> Raw `fight_end` event from `session_20260327_225743_events.jsonl`.
> One complete combat outcome: duration, casts, mana, HP delta, strategy,
> world snapshot with 6 nearby NPCs, and cycle tracking. Character name redacted.

```json
{
  "event": "fight_end",
  "ts": "2026-03-28T03:58:44.074+00:00",
  "elapsed": 61.05,
  "session_id": "session_20260327_225743",
  "tick_id": 422,
  "level": "INFO",
  "logger": "compass.routines.combat",
  "npc": "a_skeleton002",
  "duration": 16.7,
  "casts": 0,
  "mana_spent": 0,
  "backsteps": 0,
  "retargets": 0,
  "pet_heals": 0,
  "adds": 0,
  "hp_delta": 0.0,
  "hp_start": 1.0,
  "hp_end": 1.0,
  "mana_start": 259,
  "mana_end": 259,
  "cast_time": 0.0,
  "idle_time": 15.1,
  "med_time": 0.0,
  "init_dist": 17,
  "defeats": 1,
  "pos_x": 568,
  "pos_y": 123,
  "strategy": "pet_and_dot",
  "entity_id": 1407,
  "world": {
    "npcs": [
      {"name": "Garann000", "id": 1016, "hp": 97, "dist": 41},
      {"name": "a_skeleton002", "id": 1407, "hp": 0, "dist": 84},
      {"name": "a_spiderling003", "id": 1344, "hp": 100, "dist": 296},
      {"name": "a_spiderling004", "id": 1381, "hp": 100, "dist": 345},
      {"name": "a_fire_beetle009", "id": 997, "hp": 100, "dist": 371},
      {"name": "a_tree_snake004", "id": 1392, "hp": 100, "dist": 413}
    ],
    "players": 5
  },
  "cycle_id": 2
}
```
