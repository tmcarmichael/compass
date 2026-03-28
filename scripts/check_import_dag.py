#!/usr/bin/env python3
"""Standalone import DAG enforcement -- runs in CI alongside lint and typecheck.

Verifies the architectural invariant: data flows forward through the pipeline
(perception -> brain -> routines -> motor) and no module imports upward.
Brain decision modules must not import environment-specific code (eq/).

Exit code 0 = all constraints satisfied.
Exit code 1 = violations found (printed to stderr).

Usage:
    python3 scripts/check_import_dag.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


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
    pkg_dir = SRC / package
    if not pkg_dir.is_dir():
        return {}
    result: dict[str, list[str]] = {}
    for py_file in pkg_dir.rglob("*.py"):
        rel = str(py_file.relative_to(SRC))
        result[rel] = _collect_imports(py_file)
    return result


def check_routines_no_decision() -> list[str]:
    """Routines must not import brain.decision, brain.rules, or brain.runner."""
    forbidden = {"brain.decision", "brain.rules", "brain.runner"}
    violations = []
    for filepath, imports in _imports_for_package("routines").items():
        for imp in imports:
            for prefix in forbidden:
                if imp == prefix or imp.startswith(prefix + "."):
                    if "base.py" in filepath and "brain.rules.survival" in imp:
                        continue
                    violations.append(f"{filepath}: imports {imp}")
    return violations


def check_motor_no_brain() -> list[str]:
    """Motor must not import brain or routines."""
    forbidden_prefixes = {"brain", "routines"}
    violations = []
    for filepath, imports in _imports_for_package("motor").items():
        for imp in imports:
            if imp.split(".")[0] in forbidden_prefixes:
                violations.append(f"{filepath}: imports {imp}")
    return violations


def check_brain_no_runtime() -> list[str]:
    """Brain must not import runtime.orchestrator."""
    forbidden = {"runtime.orchestrator"}
    violations = []
    for filepath, imports in _imports_for_package("brain").items():
        for imp in imports:
            for prefix in forbidden:
                if imp == prefix or imp.startswith(prefix + "."):
                    violations.append(f"{filepath}: imports {imp}")
    return violations


def check_brain_decision_no_eq() -> list[str]:
    """Brain analytical modules must not import eq/ (environment-specific).

    brain/runner/ is the orchestration layer and may import eq/ for
    startup configuration. brain/rules/ bridges environment concepts to
    decision logic and may import eq.loadout (spell lookups). All other
    brain subdirectories (scoring, goap, learning, world) must use
    core.types abstractions only.
    """
    # Analytical subdirectories that must be environment-free
    pure_dirs = {"brain/scoring", "brain/goap", "brain/learning", "brain/world"}
    violations = []
    for filepath, imports in _imports_for_package("brain").items():
        in_pure = any(filepath.startswith(d) for d in pure_dirs)
        if not in_pure:
            continue
        for imp in imports:
            if imp.split(".")[0] == "eq":
                violations.append(f"{filepath}: imports {imp}")
    return violations


def check_perception_no_upper() -> list[str]:
    """Perception must not import brain, routines, or runtime."""
    forbidden_prefixes = {"brain", "routines", "runtime"}
    allowed_exceptions = {("perception/log_parser.py", "nav.zone_graph")}
    violations = []
    for filepath, imports in _imports_for_package("perception").items():
        for imp in imports:
            if imp.split(".")[0] in forbidden_prefixes:
                if (filepath, imp) not in allowed_exceptions:
                    violations.append(f"{filepath}: imports {imp}")
    return violations


def main() -> int:
    checks = [
        ("routines -> decision", check_routines_no_decision),
        ("motor -> brain/routines", check_motor_no_brain),
        ("brain -> runtime", check_brain_no_runtime),
        ("brain decision -> eq", check_brain_decision_no_eq),
        ("perception -> upper layers", check_perception_no_upper),
    ]

    all_violations: list[str] = []
    for label, check_fn in checks:
        violations = check_fn()
        if violations:
            all_violations.append(f"\n{label} ({len(violations)} violations):")
            for v in violations:
                all_violations.append(f"  {v}")

    if all_violations:
        print("Import DAG violations found:", file=sys.stderr)
        for line in all_violations:
            print(line, file=sys.stderr)
        return 1

    print(f"Import DAG: all {len(checks)} constraints satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
