"""Tests for eq.zone_ids -- zone ID-to-name mapping constant.

ZONE_ID_MAP is a dict[int, str] used to resolve memory-mapped zone IDs
to human-readable zone short names. Stubbed to empty in public release.
"""

from __future__ import annotations

from eq.zone_ids import ZONE_ID_MAP


class TestZoneIdMap:
    def test_is_dict(self) -> None:
        assert isinstance(ZONE_ID_MAP, dict)

    def test_importable(self) -> None:
        """ZONE_ID_MAP is importable and accessible."""
        assert ZONE_ID_MAP is not None
