<!-- last_modified: 2026-03-28 -->

> Real data from session reports and persisted memory files. Nektulos Forest, 233 defeats across
> multiple sessions. Shows session grade progression B -> A -> A, auto-tuned parameters drifting
> from defaults, and fight duration improving for the same mob type over time.
> Character name redacted.

### Session grade progression

```json
{"session":"session_20260326_222138","version":"v1.31.9","duration_min":2.8,"defeats":4,"dph":84.8,
 "scorecard":{"pathing":0,"defeat_rate":100,"pull_success":100,"targeting":100,"survival":100,"mana_efficiency":83,"uptime":100,"overall":88,"grade":"B"},
 "mob_stats":{"a_tree_snake":{"avg_dur":29.5,"avg_mana":23,"pet_death_rate":0.07,"danger":0.04}}}

{"session":"session_20260327_213213","version":"v1.33.2","duration_min":18.0,"defeats":24,"dph":77.8,
 "scorecard":{"pathing":100,"defeat_rate":100,"pull_success":95,"targeting":91,"survival":100,"mana_efficiency":100,"uptime":100,"overall":98,"grade":"A"},
 "mob_stats":{"a_tree_snake":{"avg_dur":15.9,"avg_mana":3,"pet_death_rate":0.0,"danger":0.0}}}

{"session":"session_20260327_221817","version":"v1.37.2","duration_min":38.0,"defeats":49,"dph":76.7,
 "scorecard":{"pathing":100,"defeat_rate":100,"pull_success":100,"targeting":86,"survival":100,"mana_efficiency":100,"uptime":100,"overall":99,"grade":"A"},
 "mob_stats":{"a_skeleton":{"avg_dur":15.3,"avg_mana":1},"casts_per_defeat":0.55,"avg_cycle_s":24.8}}
```

### Persisted auto-tuning (drifted from defaults)

```json
// data/memory/nektulos_tuning.json
{ "v": 1, "roam_radius_mult": 1.2999999999999998, "social_add_limit": 5, "mana_conserve_level": 0 }
```

Defaults: `roam_radius_mult=1.0`, `social_add_limit=3`, `mana_conserve_level=0`.
High defeat rate + survival triggers the relaxation path in `evaluate_and_tune()`:
search radius expanded 30%, social pull aggression increased from 3 to 5.

### Fight duration: same mob type, first session vs latest

```json
// data/memory/nektulos.json (233 defeats, 500 sightings)
{"x":607,"y":71,"name":"a_tree_snake003","level":5,"fight_s":23.9,"_context":"first session defeat"}
{"x":694,"y":-6,"name":"a_tree_snake013","level":4,"fight_s":9.0,"_context":"latest session defeat"}
```

### Scorecard weights (how the grade is computed)

```python
# From brain/scorecard.py -- weighted average, each category 0-100
weights = {"defeat_rate": 25, "survival": 20, "pull_success": 15,
           "uptime": 15, "pathing": 10, "mana_efficiency": 10, "targeting": 5}
```
