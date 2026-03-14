"""Perception layer: memory reading, state snapshots, log parsing.

Key exports:
  MemoryReader (reader.py)        -- reads target process memory
  GameState, SpawnData (state.py) -- frozen per-tick snapshots
  offsets (offsets.py)            -- struct field offset definitions
  LogParser (log_parser.py)       -- game log tailing for events
  queries (queries.py)            -- spawn list query helpers
"""

__all__ = ["offsets"]
