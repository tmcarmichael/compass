"""Brain: decision engine with priority rules and sub-state management.

Key components:
  Brain (decision.py)        -- priority rule engine, cooldowns, lock-in
  AgentContext (context.py)    -- mutable session state with typed sub-states
  WorldModel (world_model.py) -- NPC tracking, temporal profiles, threat detection
  mob_scoring (mob_scoring.py)     -- MobProfile, ScoringWeights, 15-factor scorer
  patrol_detector (patrol_detector.py) -- geometric patrol cycle detection
  combat_eval (combat_eval.py) -- con color, disposition, threat assessment
  rules/ (5 modules)         -- modular rule registration (see rules/__init__.py)
  state/ (13 dataclasses)    -- focused sub-state objects
  strategies/ (4 classes)    -- combat strategy pattern (CastStrategy ABC)
"""

__all__: list[str] = []
