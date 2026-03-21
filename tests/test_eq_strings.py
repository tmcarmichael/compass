"""Tests for eq.strings  -- name normalization and byte decoding."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from eq.strings import decode_eq_string, normalize_mob_name


class TestNormalizeMobName:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("a_black_bear007", "a_black_bear"),
            ("a_skeleton_009", "a_skeleton"),
            ("A_Fire_Beetle", "a_fire_beetle"),
            ("a_skeleton", "a_skeleton"),
            ("", ""),
            ("a_moss_snake01", "a_moss_snake"),
        ],
    )
    def test_known_names(self, raw: str, expected: str) -> None:
        assert normalize_mob_name(raw) == expected

    @given(name=st.from_regex(r"[a-z][a-z_]{0,19}", fullmatch=True))
    def test_idempotent_on_clean_names(self, name: str) -> None:
        """A name with no trailing digits normalizes to itself."""
        result = normalize_mob_name(name)
        # Idempotent: normalizing again gives the same result
        assert normalize_mob_name(result) == result


class TestDecodeEqString:
    def test_basic_ascii(self) -> None:
        assert decode_eq_string(b"Hello") == "Hello"

    def test_empty(self) -> None:
        assert decode_eq_string(b"") == ""
