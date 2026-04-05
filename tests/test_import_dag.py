"""Enforce key architectural import constraints as tests.

The architecture documents specific import prohibitions
(see docs/design-decisions.md "Strict forward-only data flow"):

  - Routines cannot import brain.decision (the rule engine)
  - Brain cannot import runtime.orchestrator
  - Motor cannot import from brain/ or routines/ (motor doesn't decide)
  - No circular imports between decision layer and execution layer

These constraints preserve the pipeline separation where data flows
P -> B -> R -> M. Support modules (core, util, eq, nav) and shared
types (brain.context, perception.combat_eval) are accessible cross-layer.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"


def _collect_imports(filepath: Path) -> list[str]:
    """Parse a Python file and return all absolute imported module paths."""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
    except SyntaxError:
        return []

    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                modules.append(node.module)
    return modules


def _imports_for_package(package: str) -> dict[str, list[str]]:
    """Collect all imports from files in a source package."""
    pkg_dir = _SRC / package
    if not pkg_dir.is_dir():
        return {}
    result: dict[str, list[str]] = {}
    for py_file in pkg_dir.rglob("*.py"):
        rel = str(py_file.relative_to(_SRC))
        result[rel] = _collect_imports(py_file)
    return result


# ---------------------------------------------------------------------------
# Core constraint: routines cannot import brain.decision
# ---------------------------------------------------------------------------


def test_routines_do_not_import_brain_decision() -> None:
    """Routines must not import the rule evaluation engine.

    Routines can import brain.context (shared state), perception.combat_eval
    (shared types), and brain.learning (data). They MUST NOT import
    brain.decision, brain.rules, or brain.runner, which are the
    decision-making layer that invokes routines.
    """
    forbidden = {"brain.decision", "brain.rules", "brain.runner"}
    violations: list[str] = []

    for filepath, imports in _imports_for_package("routines").items():
        for imp in imports:
            for prefix in forbidden:
                if imp == prefix or imp.startswith(prefix + "."):
                    violations.append(f"{filepath}: imports {imp}")

    assert not violations, (
        f"Routines must not import the decision engine ({len(violations)} violations):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# Core constraint: motor cannot import brain or routines
# ---------------------------------------------------------------------------


def test_motor_does_not_import_brain_or_routines() -> None:
    """Motor is the lowest execution layer; it must not make decisions.

    Motor can import core, util, eq, nav, and perception types.
    It must NOT import brain/ or routines/.
    """
    forbidden_prefixes = {"brain", "routines"}
    violations: list[str] = []

    for filepath, imports in _imports_for_package("motor").items():
        for imp in imports:
            top = imp.split(".")[0]
            if top in forbidden_prefixes:
                violations.append(f"{filepath}: imports {imp}")

    assert not violations, (
        f"Motor must not import brain or routines ({len(violations)} violations):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# Core constraint: brain cannot import runtime orchestrator
# ---------------------------------------------------------------------------


def test_brain_does_not_import_runtime_orchestrator() -> None:
    """Brain must not depend on session lifecycle or orchestrator modules.

    brain.context may import runtime.agent_session (a shared type),
    but brain must not import runtime.orchestrator.
    """
    forbidden = {"runtime.orchestrator"}
    violations: list[str] = []

    for filepath, imports in _imports_for_package("brain").items():
        for imp in imports:
            for prefix in forbidden:
                if imp == prefix or imp.startswith(prefix + "."):
                    violations.append(f"{filepath}: imports {imp}")

    assert not violations, (
        f"Brain must not import orchestrator ({len(violations)} violations):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# Structural: no circular imports between decision and execution
# ---------------------------------------------------------------------------


def test_no_circular_decision_execution_imports() -> None:
    """brain.decision must not import from routines/, and vice versa.

    brain.rules/ registers routines but only via references passed
    during registration. brain.decision itself must not import
    routine implementations.
    """
    # Check brain/decision.py does not import routine implementations
    # (importing routines.base for the RoutineBase interface is allowed)
    decision_file = _SRC / "brain" / "decision.py"
    if decision_file.exists():
        violations = [
            imp
            for imp in _collect_imports(decision_file)
            if imp.startswith("routines") and imp != "routines.base"
        ]
        assert not violations, f"brain/decision.py imports routine implementations: {violations}"

    # Check brain/completion.py, brain/transitions.py, brain/profiling.py
    # (importing routines.base for RoutineBase/RoutineStatus interface is allowed)
    for name in ("completion.py", "transitions.py", "profiling.py"):
        filepath = _SRC / "brain" / name
        if filepath.exists():
            violations = [
                f"{name}: {imp}"
                for imp in _collect_imports(filepath)
                if imp.startswith("routines") and imp != "routines.base"
            ]
            assert not violations, f"Decision support imports routine implementations: {violations}"


# ---------------------------------------------------------------------------
# Perception layer import constraints
# ---------------------------------------------------------------------------


def test_perception_imports_only_core_and_eq() -> None:
    """Perception must not import brain, routines, runtime, or nav.

    Perception is the lowest pipeline layer: it reads state and produces
    immutable snapshots. It may import core/ (types, constants) and eq/
    (game data). It must not import decision-making or execution layers.
    """
    forbidden_prefixes = {"brain", "routines", "runtime"}
    # nav.zone_graph is used by log_parser for zone name resolution  -- accepted coupling
    allowed_exceptions = {("perception/log_parser.py", "nav.zone_graph")}
    violations: list[str] = []

    for filepath, imports in _imports_for_package("perception").items():
        for imp in imports:
            top = imp.split(".")[0]
            if top in forbidden_prefixes:
                if (filepath, imp) not in allowed_exceptions:
                    violations.append(f"{filepath}: imports {imp}")

    assert not violations, (
        f"Perception must not import upper layers ({len(violations)} violations):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# 3D migration: Point constructors must have z coordinate
# ---------------------------------------------------------------------------


def test_brain_analytical_modules_do_not_import_eq() -> None:
    """Brain scoring, GOAP, learning, and world modules must not import eq/.

    These analytical modules must use core.types abstractions (Con,
    normalize_entity_name, Disposition) instead of environment-specific
    eq/ modules. brain/rules/ may import eq.loadout for spell lookups.
    brain/runner/ may import eq/ for startup configuration.
    """
    pure_dirs = {"brain/scoring", "brain/goap", "brain/learning", "brain/world"}
    violations: list[str] = []

    for filepath, imports in _imports_for_package("brain").items():
        in_pure = any(filepath.startswith(d) for d in pure_dirs)
        if not in_pure:
            continue
        for imp in imports:
            if imp.split(".")[0] == "eq":
                violations.append(f"{filepath}: imports {imp}")

    assert not violations, (
        f"Brain analytical modules must not import eq/ ({len(violations)} violations):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# 3D migration: Point constructors must have z coordinate
# ---------------------------------------------------------------------------


def test_point_constructors_have_three_args() -> None:
    """All Point() calls in src/ must pass 3 positional args (x, y, z).

    After the 3D migration, any Point(x, y) without z is a bug.
    """
    import ast

    violations: list[str] = []

    for py_file in _SRC.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue

        rel = str(py_file.relative_to(_SRC))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Match Point(...) calls
                func = node.func
                is_point_call = False
                if isinstance(func, ast.Name) and func.id == "Point":
                    is_point_call = True
                elif isinstance(func, ast.Attribute) and func.attr == "Point":
                    is_point_call = True

                if is_point_call:
                    # Count positional args (exclude keyword args)
                    n_pos = len(node.args)
                    # Point.from_loc is a classmethod that takes 3 args too
                    if n_pos == 2:
                        violations.append(f"{rel}:{node.lineno}: Point() with 2 args (missing z)")

    assert not violations, f"Point() calls missing z coordinate ({len(violations)}):\n" + "\n".join(
        f"  {v}" for v in violations
    )
