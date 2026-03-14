"""Recording motor backend for testing.

Captures all motor actions in a list without sending input or sleeping.
Use with motor.actions.set_backend() in test fixtures.
"""

from __future__ import annotations


class RecordingMotor:
    """Motor backend that records actions without side effects.

    No input is sent. No sleeps are executed. All actions are captured
    in the ``actions`` list for assertion in tests.
    """

    def __init__(self) -> None:
        self.actions: list[str] = []
        self.held: set[str] = set()
        self.total_sleep: float = 0.0

    def action(self, name: str, duration: float = 0.05) -> None:
        self.actions.append(name)

    def action_down(self, name: str) -> None:
        self.actions.append(f"+{name}")
        self.held.add(name)

    def action_up(self, name: str) -> None:
        self.actions.append(f"-{name}")
        self.held.discard(name)

    def sleep(self, duration: float) -> None:
        self.total_sleep += duration

    def clear(self) -> None:
        """Reset recorded state."""
        self.actions.clear()
        self.held.clear()
        self.total_sleep = 0.0

    def has_action(self, name: str) -> bool:
        """Check if a specific action was recorded."""
        return name in self.actions or f"+{name}" in self.actions

    @property
    def action_names(self) -> list[str]:
        """Action names without +/- prefixes."""
        return [a.lstrip("+-") for a in self.actions]
