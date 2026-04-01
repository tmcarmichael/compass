"""Facade re-exports for the headless simulator."""

from simulator import Scenario, SimulationResult, SimulationRunner
from simulator.__main__ import main

__all__ = ["SimulationRunner", "Scenario", "SimulationResult", "main"]
