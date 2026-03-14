"""Headless simulator for the Compass agent architecture.

Run scenarios through the full decision stack without a game client.
Usage: python3 -m simulator [benchmark|replay|converge] [options]
"""

from simulator.results import SimulationResult
from simulator.runner import SimulationRunner
from simulator.scenarios import Scenario

__all__ = ["SimulationRunner", "Scenario", "SimulationResult"]
