"""Motor layer: semantic action interface for agent control.

actions.py defines the control vocabulary that routines call  -  movement,
targeting, stance, casting, pet commands, and lifecycle management.  The
interface is environment-agnostic; a concrete backend provides the actual
input dispatch.
"""

__all__: list[str] = []
