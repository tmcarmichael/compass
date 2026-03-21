<!-- last_modified: 2026-03-28 -->

> Raw log excerpts from `session_20260327_225743` (Nektulos Forest, 34 minutes, 49 defeats).
> Same session viewed through four log tiers. Character name redacted.

### T1 -- EVENT: session arc in 15 lines

> State changes only. Defeats, zone entry, snapshots. The reader gets the full session story in 10 seconds.

```
2026-03-27 22:57:45,027 EVENT compass.brain_loop: [LIFECYCLE] +2s Brain loop started  -  v1.37.2
2026-03-27 22:57:47,817 EVENT compass.brain_loop: [LIFECYCLE] +5s SESSION READY: zone=nektulos camp=central_camp
  terrain: 669x415, 21386 obstacles
  flags: obstacle_avoidance=True, loot_mode=off, combat_casting=True, flee=True, rest=True, wander=True, death_recovery=off
2026-03-27 22:57:55,757 EVENT compass.routines.acquire: [TARGET] +13s Acquire: SUCCESS  -  tab 1 got 'a_skeleton000' id=1371 score=294.3 dist=70 pos=(622,110) lv=5 adds=0
2026-03-27 22:58:00,096 EVENT compass.routines.combat: [COMBAT] +17s Combat: START target='a_skeleton000' id=1371 dist=17 target_HP=72/100 target_pos=(606,116) player_pos=(590,121) HP=96% Mana=259
2026-03-27 22:58:23,420 EVENT compass.routines.acquire: [TARGET] +40s Acquire: SUCCESS  -  tab 1 got 'a_skeleton002' id=1407 score=294.7 dist=87 pos=(710,331) lv=5 adds=0
2026-03-27 22:58:27,389 EVENT compass.routines.combat: [COMBAT] +44s Combat: START target='a_skeleton002' id=1407 dist=87 target_HP=73/100 target_pos=(710,331) player_pos=(656,264) HP=100% Mana=259
2026-03-27 22:58:45,984 EVENT compass.brain_loop: [SNAPSHOT] +63s SNAPSHOT | HP=100% Mana=100% Pos=(656,264) Hdg=52 Pet=yes(id=1016,'Garann000') Engaged=no Mobs=1(57.0/hr) 5min=1(12.0/hr) XP=1 NPCs=199 Camp_dist=348 Players=0(nearest=549) Routine=none Pose=standing AcqFails=0 LastKill=2s XP_rate=18.7%/hr TTL=141min
2026-03-27 23:01:16,268 EVENT compass.brain_loop: [SNAPSHOT] +213s SNAPSHOT | HP=100% Mana=98% Pos=(166,126) Hdg=446 Pet=yes(id=1016,'Garann000') Engaged=no Mobs=3(50.6/hr) 5min=3(36.0/hr) XP=2 NPCs=202 Camp_dist=169 Players=0(nearest=200) Routine=WANDER Pose=standing AcqFails=0 LastKill=8s XP_rate=20.9%/hr TTL=123min
2026-03-27 23:02:25,681 WARNING compass.brain_loop: [COMBAT] +283s ADD detected (target_name): 'a_skeleton005' lv6 targeting pet at 99u
2026-03-27 23:04:18,756 EVENT compass.brain_loop: [SNAPSHOT] +396s SNAPSHOT | HP=99% Mana=100% Pos=(355,601) Hdg=114 Pet=yes(id=1016,'Garann000') Engaged=no Mobs=8(72.7/hr) 5min=7(84.0/hr) XP=5 NPCs=194 Camp_dist=473 Players=0(nearest=1544) Routine=ACQUIRE Pose=standing AcqFails=0 LastKill=7s LOCKED=ACQUIRE XP_rate=36.3%/hr TTL=50min
2026-03-27 23:05:19,780 EVENT compass.brain_loop: [SNAPSHOT] +457s SNAPSHOT | HP=100% Mana=100% Pos=(433,799) Hdg=300 Pet=yes(id=1016,'Garann000') Engaged=no Mobs=10(78.8/hr) 5min=9(108.0/hr) XP=6 NPCs=215 Camp_dist=678 Players=0(nearest=770) Routine=ACQUIRE Pose=standing AcqFails=0 LastKill=5s XP_rate=38.6%/hr TTL=40min
2026-03-27 23:30:38,681 EVENT compass.brain_loop: [SNAPSHOT] +1976s SNAPSHOT | HP=100% Mana=76% Pos=(245,-162) Hdg=31 Pet=yes(id=1016,'Garann000') Engaged=no Mobs=44(80.2/hr) 5min=9(108.0/hr) XP=26 NPCs=229 Camp_dist=323 Players=0(nearest=1076) Routine=ACQUIRE Pose=standing AcqFails=0 LastKill=6s LOCKED=ACQUIRE XP_rate=32.0%/hr TTL=32min
2026-03-27 23:31:53,836 EVENT compass.brain_loop: [LIFECYCLE] +2051s Brain loop stopped (stop_event set -- user stop)
2026-03-27 23:31:55,957 EVENT compass.brain_loop: [LIFECYCLE] +2053s /camp sent (brain_exit, attempt 1)
```

### T2 -- INFO: one complete acquire -> pull -> combat -> defeat -> wander cycle

> Routine transitions, target names, engagement state. The operational "what happened" layer.

```
2026-03-27 22:58:16,710 INFO compass.brain_loop: [STATE] +34s STATE: disengaged
2026-03-27 22:58:16,711 INFO compass.brain.decision: [DECISION] +34s Brain: -> WANDER | HP=97% Mana=100% Pos=(568,123) Pet=yes
2026-03-27 22:58:19,111 INFO compass.brain.decision: [DECISION] +36s Brain: deactivating WanderRoutine
2026-03-27 22:58:19,111 INFO compass.brain.decision: [DECISION] +36s Brain: -> ACQUIRE | HP=98% Mana=100% Pos=(595,170) Pet=yes
2026-03-27 22:58:19,111 INFO compass.routines.acquire: [POSITION] +36s Acquire: player at (595, 170) heading=43 camp_dist=263
2026-03-27 22:58:19,111 INFO compass.routines.acquire: [TARGET] +36s Acquire: 5 scored targets:
2026-03-27 22:58:19,111 INFO compass.routines.acquire: [TARGET] +36s   - 'a_skeleton002' id=1407 score=212.0 dist=198 con=light_blue pos=(710,331) lv=5 iso=1.00 fight=15s
2026-03-27 22:58:19,115 INFO compass.routines.acquire: [POSITION] +36s Acquire: approach 'a_skeleton002' at 198u -- MovementPhase to (710,331) timeout=10s
2026-03-27 22:58:23,029 INFO compass.routines.acquire: [POSITION] +40s Acquire: mob 'a_skeleton002' within 87u -- stopping approach
2026-03-27 22:58:23,420 EVENT compass.routines.acquire: [TARGET] +40s Acquire: SUCCESS  -  tab 1 got 'a_skeleton002' id=1407 score=294.7 dist=87 pos=(710,331) lv=5 adds=0
2026-03-27 22:58:23,420 INFO compass.brain.decision: [DECISION] +40s Routine ACQUIRE success (HP=98% mana=100%)
2026-03-27 22:58:23,427 INFO compass.brain_loop: [STATE] +40s STATE: pull target set -> 'a_skeleton002' id=1407
2026-03-27 22:58:23,429 INFO compass.brain.decision: [DECISION] +40s Brain: -> PULL | HP=98% Mana=100% Pos=(656,264) Pet=yes Target='a_skeleton002' id=1407
2026-03-27 22:58:23,429 INFO compass.routines.pull: [COMBAT] +40s Pull: START strategy='pet only' target='a_skeleton002' id=1407 con=light_blue dist=87 target_pos=(710,331) player_pos=(656,264) HP=98% Mana=100% Pet=100%
2026-03-27 22:58:23,515 INFO compass.routines.pull: [COMBAT] +40s Pull: SEND_PET -> 'a_skeleton002' dist=87 target_pos=(710,331)
2026-03-27 22:58:27,188 INFO compass.routines.pull: [COMBAT] +44s Pull: pet HIT confirmed (HP 100->73) after 3.5s mob_dist=87 (was 87)
2026-03-27 22:58:27,288 INFO compass.routines.pull: [COMBAT] +44s Pull: ENGAGED after 3.9s (dist=87, pet_engage=3.5s, dc_retries=0, strategy='pet only')
2026-03-27 22:58:27,289 INFO compass.brain.decision: [DECISION] +44s Routine PULL success (HP=100% mana=100%)
2026-03-27 22:58:27,389 INFO compass.brain_loop: [STATE] +44s STATE: ENGAGED -> 'a_skeleton002'
2026-03-27 22:58:27,389 INFO compass.brain.decision: [DECISION] +44s Brain: -> IN_COMBAT | HP=100% Mana=100% Pos=(656,264) Pet=yes Target='a_skeleton002' id=1407
2026-03-27 22:58:27,389 INFO compass.routines.combat: [COMBAT] +44s Combat: strategy=pet_and_dot (con=light_blue, danger=0.01, pet_death=0%)
2026-03-27 22:58:27,389 EVENT compass.routines.combat: [COMBAT] +44s Combat: START target='a_skeleton002' id=1407 dist=87 target_HP=73/100 target_pos=(710,331) player_pos=(656,264) HP=100% Mana=259
2026-03-27 22:58:27,390 INFO compass.strategies.pet_and_dot: [CAST] +44s SPELL EFFICIENCY: Disease Cloud=3.6dmg/mana > Lifespike=2.5dmg/mana
2026-03-27 22:58:44,073 INFO compass.brain_loop: [LIFECYCLE] +61s XP gain detected (memory): delta=1, mob='a_skeleton002'
2026-03-27 22:58:44,074 INFO compass.routines.combat_monitor: [COMBAT] +61s Combat: TARGET DEAD  -  fight lasted 16.7s, player HP=100% Mana=259 pos=(656,264)
2026-03-27 22:58:44,074 INFO compass.brain.decision: [DECISION] +61s Routine IN_COMBAT success (HP=100% mana=100%)
2026-03-27 22:58:44,074 INFO compass.routines.combat: [COMBAT] +61s Fight: a_skeleton002 in 16.7s, 0 casts, 0 mana
2026-03-27 22:58:45,977 INFO compass.util.cycle_tracker: Cycle #2: a_skeleton002 | 17s fight, 0 casts, 0 mana | 23s total
2026-03-27 22:58:45,985 INFO compass.brain_loop: [STATE] +63s STATE: disengaged
2026-03-27 22:58:45,986 INFO compass.brain.decision: [DECISION] +63s Brain: -> WANDER | HP=100% Mana=100% Pos=(656,264) Pet=yes
```

### T3 -- VERBOSE: rule evaluation cascade + combat monitor + NPC radar

> Decision branches explaining *why* each rule won or lost. The 14-rule cascade and nearby mob table.

```
2026-03-27 22:58:02,061 VERBOSE compass.routines.combat_monitor: [COMBAT] +19s Combat: t=2s | mob='a_skeleton000' HP=67% (67/100) dist=38 mob_pos=(606,116) speed=0.0 | player HP=96% Mana=259 casts=0 pos=(568,123) | pet HP=95/100 (95%) pos=(590,116) dist=23 | strategy=pet_and_dot
2026-03-27 22:58:04,065 VERBOSE compass.routines.combat_monitor: [COMBAT] +21s Combat: t=4s | mob='a_skeleton000' HP=67% (67/100) dist=38 mob_pos=(606,116) speed=0.0 | player HP=99% Mana=259 casts=0 pos=(568,123) | pet HP=98/100 (98%) pos=(590,116) dist=23 | strategy=pet_and_dot
2026-03-27 22:58:06,071 VERBOSE compass.routines.combat_monitor: [COMBAT] +23s Combat: t=6s | mob='a_skeleton000' HP=48% (48/100) dist=38 mob_pos=(606,116) speed=0.0 | player HP=96% Mana=259 casts=0 pos=(568,123) | pet HP=94/100 (94%) pos=(590,116) dist=23 | strategy=pet_and_dot
2026-03-27 22:58:08,079 VERBOSE compass.routines.combat_monitor: [COMBAT] +25s Combat: t=8s | mob='a_skeleton000' HP=31% (31/100) dist=38 mob_pos=(606,116) speed=0.0 | player HP=96% Mana=259 casts=0 pos=(568,123) | pet HP=94/100 (94%) pos=(590,116) dist=23 | strategy=pet_and_dot
2026-03-27 22:58:10,084 VERBOSE compass.routines.combat_monitor: [COMBAT] +27s Combat: t=10s | mob='a_skeleton000' HP=31% (31/100) dist=38 mob_pos=(606,116) speed=0.0 | player HP=99% Mana=259 casts=0 pos=(568,123) | pet HP=97/100 (97%) pos=(590,116) dist=23 | strategy=pet_and_dot
2026-03-27 22:58:12,395 VERBOSE compass.routines.combat_monitor: [COMBAT] +29s Combat: t=12s | mob='a_skeleton000' HP=22% (22/100) dist=36 mob_pos=(603,116) speed=0.0 | player HP=97% Mana=259 casts=0 pos=(568,123) | pet HP=97/100 (97%) pos=(590,116) dist=23 | strategy=pet_and_dot
2026-03-27 22:58:15,226 VERBOSE compass.brain_loop: [RADAR] +32s 11 NPCs within 500u | player=(568,123) lv9 range=4-8 routine=IN_COMBAT:
2026-03-27 22:58:15,226 VERBOSE compass.brain_loop: [RADAR] +32s   'a_skeleton002' lv5 LIGHT_BLUE SCOWLING dist=251 pos=(710,331) HP=100/100 [INRANGE]
2026-03-27 22:58:15,226 VERBOSE compass.brain_loop: [RADAR] +32s   'a_spiderling003' lv2 GREEN INDIFFERENT dist=296 pos=(514,414) HP=100/100 [UNDER MOV:1]
2026-03-27 22:58:15,226 VERBOSE compass.brain_loop: [RADAR] +32s   'a_tree_snake004' lv4 LIGHT_BLUE INDIFFERENT dist=413 pos=(584,-289) HP=100/100 [INRANGE]
2026-03-27 22:58:15,226 VERBOSE compass.brain_loop: [RADAR] +32s   'a_tree_snake001' lv6 BLUE INDIFFERENT dist=433 pos=(1000,83) HP=100/100 [INRANGE]
2026-03-27 22:58:15,226 VERBOSE compass.brain_loop: [RADAR] +32s   'a_black_bear012' lv4 LIGHT_BLUE SCOWLING dist=437 pos=(200,358) HP=100/100 [INRANGE MOV:1]
2026-03-27 22:57:47,826 VERBOSE compass.brain.decision: [DECISION] +5s Brain eval: [DEATH_RECOVERY=no | FEIGN_DEATH=no | FLEE=no | REST=no | EVADE=no | IN_COMBAT=no | ENGAGE_ADD=no | ACQUIRE=no | PULL=no | MEMORIZE_SPELLS=YES | SUMMON_PET=skip | BUFF=skip | TRAVEL=skip | WANDER=skip] -> MEMORIZE_SPELLS
2026-03-27 22:57:47,924 VERBOSE compass.brain.decision: [DECISION] +5s Brain eval: [DEATH_RECOVERY=no | FEIGN_DEATH=no | FLEE=no | REST=no | EVADE=no | IN_COMBAT=no | ENGAGE_ADD=no | ACQUIRE=YES | PULL=skip | MEMORIZE_SPELLS=skip | SUMMON_PET=skip | BUFF=skip | TRAVEL=skip | WANDER=skip] -> ACQUIRE
2026-03-27 22:57:55,765 VERBOSE compass.brain.decision: [DECISION] +13s Brain eval: [DEATH_RECOVERY=no | FEIGN_DEATH=no | FLEE=no | REST=no | EVADE=no | IN_COMBAT=no | ENGAGE_ADD=no | ACQUIRE=no | PULL=YES | MEMORIZE_SPELLS=skip | SUMMON_PET=skip | BUFF=skip | TRAVEL=skip | WANDER=skip] -> PULL
2026-03-27 22:57:55,928 VERBOSE compass.routines.pull: [COMBAT] +13s Pull: WAIT_PET polling HP (mob dist=66, max_wait=3.4s, mob HP=100/100) player_pos=(558,124) mob_pos=(622,110)
```

### T4 -- DEBUG: raw motor commands, tick budget, NPC entity table

> Implementation-level telemetry. Motor key presses, tick timing breakdown, spawn table dump.

```
2026-03-27 22:57:48,568 DEBUG compass.motor.actions: MOTOR: forward START
2026-03-27 22:57:55,102 DEBUG compass.motor.actions: MOTOR: forward STOP
2026-03-27 22:57:55,773 DEBUG compass.motor.actions: [ACTION] +13s Pet: /pet attack (hotbar 1)
2026-03-27 22:57:57,334 DEBUG compass.motor.actions: MOTOR: backward START
2026-03-27 22:57:57,735 DEBUG compass.motor.actions: MOTOR: backward STOP
2026-03-27 22:57:57,786 DEBUG compass.motor.actions: MOTOR: forward START
2026-03-27 22:57:57,837 DEBUG compass.motor.actions: MOTOR: forward STOP
2026-03-27 22:57:57,837 DEBUG compass.motor.actions: [ACTION] +15s Pet: /pet attack (hotbar 1)
2026-03-27 22:58:15,225 DEBUG compass.brain_loop: [SNAPSHOT] +32s   NPC: 'Garann000' id=1016 dist=16 pos=(584,118) HP=100/100 lv=9
2026-03-27 22:58:15,225 DEBUG compass.brain_loop: [SNAPSHOT] +32s   NPC: 'a_skeleton002' id=1407 dist=251 pos=(710,331) HP=100/100 lv=5
2026-03-27 22:58:15,225 DEBUG compass.brain_loop: [SNAPSHOT] +32s   NPC: 'a_spiderling003' id=1344 dist=296 pos=(514,414) HP=100/100 lv=2
2026-03-27 22:58:15,225 DEBUG compass.brain_loop: [SNAPSHOT] +32s   NPC: 'a_fire_beetle009' id=997 dist=371 pos=(217,5) HP=100/100 lv=2
2026-03-27 22:58:15,225 DEBUG compass.brain_loop: [SNAPSHOT] +32s   TICK: total=1321.1ms rules=0.0ms routine=1321.0ms world=2.8ms
2026-03-27 22:58:15,225 DEBUG compass.brain_loop: [SNAPSHOT] +32s Inventory scan: 17 items
2026-03-27 22:57:45,027 DEBUG compass.util.thread_guard: [THREAD] +2s Brain thread registered: id=9892
```
