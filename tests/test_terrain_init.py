"""Tests for nav.terrain package -- import smoke test.

nav.terrain.__init__ re-exports nothing (__all__ = []) but must be
importable without side effects.
"""

from __future__ import annotations


def test_terrain_package_imports() -> None:
    import nav.terrain

    assert hasattr(nav.terrain, "__all__")
    assert nav.terrain.__all__ == []
