"""Custom log levels for the tiered logging system.

Registers two levels between the stdlib defaults:
  VERBOSE = 15  (between DEBUG=10 and INFO=20)  -- decision branches, scoring
  EVENT   = 25  (between INFO=20 and WARNING=30) -- defeats, deaths, level-ups

Usage: ``log.log(VERBOSE, "message", ...)`` and ``log.log(EVENT, "message", ...)``.
Import this module before any logging happens to ensure levels are registered.
"""

from __future__ import annotations

import logging
from enum import IntEnum

__all__ = ["LogLevel", "VERBOSE", "EVENT"]


class LogLevel(IntEnum):
    """Named log level constants for the tiered logging system."""

    DEBUG = logging.DEBUG  # 10
    VERBOSE = 15  # decision branches, per-tick scoring
    INFO = logging.INFO  # 20
    EVENT = 25  # defeats, deaths, level-ups
    WARNING = logging.WARNING  # 30
    ERROR = logging.ERROR  # 40
    CRITICAL = logging.CRITICAL  # 50


VERBOSE: int = LogLevel.VERBOSE
EVENT: int = LogLevel.EVENT


def register() -> None:
    """Register custom level names so they display as VERBOSE/EVENT in logs.

    Safe to call multiple times -- idempotent.
    """
    logging.addLevelName(VERBOSE, "VERBOSE")
    logging.addLevelName(EVENT, "EVENT")


# Auto-register on import
register()
