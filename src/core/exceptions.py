"""Custom exception hierarchy for the compass agent."""


class CompassError(Exception):
    """Base exception for all agent errors."""


class MemoryReadError(CompassError):
    """Failed to read target process memory.

    Raised by MemoryReader when ReadProcessMemory fails.
    Callers should retry after a brief delay.
    """


class ProcessNotFoundError(CompassError):
    """Target process not found or not accessible.

    Raised during startup when the target application isn't running or
    the agent lacks admin privileges for ReadProcessMemory.
    """
