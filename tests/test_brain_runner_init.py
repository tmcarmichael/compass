"""Tests for brain.runner package -- import and re-export smoke test.

brain.runner.__init__ re-exports BrainRunner from brain.runner.loop.
"""

from __future__ import annotations


def test_brain_runner_package_exports_brain_runner() -> None:
    from brain.runner import BrainRunner

    assert BrainRunner is not None


def test_brain_runner_all_contains_brain_runner() -> None:
    import brain.runner

    assert "BrainRunner" in brain.runner.__all__
