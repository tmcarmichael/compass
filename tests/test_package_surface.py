"""Public package facade tests."""

from __future__ import annotations


def test_compass_facade_exposes_runtime_entry_points() -> None:
    import compass
    from runtime.orchestrator import AgentOrchestrator

    assert isinstance(compass.__version__, str) and compass.__version__
    assert compass.AgentOrchestrator is AgentOrchestrator
    assert callable(compass.build_context)
    assert callable(compass.build_brain)
    assert callable(compass.find_config)
    assert callable(compass.load_zone_config)


def test_compass_runtime_module_reexports_runtime_objects() -> None:
    from compass.runtime import AgentOrchestrator, build_context
    from runtime.orchestrator import AgentOrchestrator as RuntimeAgentOrchestrator

    assert AgentOrchestrator is RuntimeAgentOrchestrator
    assert callable(build_context)


def test_compass_simulator_module_reexports_simulator_objects() -> None:
    from compass.simulator import Scenario, SimulationRunner, main

    assert Scenario.__name__ == "Scenario"
    assert SimulationRunner.__name__ == "SimulationRunner"
    assert callable(main)
