"""EQ string utilities: npc name normalization, cp1252 decoding."""

from __future__ import annotations


def normalize_mob_name(name: str) -> str:
    """Strip trailing digits and underscores, lowercase.

    EQ npc names have instance suffixes: 'a_black_bear007' -> 'a_black_bear'.
    Delegates to core.types.normalize_entity_name. This wrapper preserves
    the eq.strings import path for perception and environment code.
    """
    from core.types import normalize_entity_name

    return normalize_entity_name(name)


def decode_eq_string(raw: bytes) -> str:
    """Decode EQ string from cp1252/latin-1 with right-single-quote normalization.

    EQ uses cp1252 encoding where 0x92 is a right single quote. Normalize
    to ASCII apostrophe for safe logging and string comparison.
    """
    return raw.decode("latin-1").replace("\x92", "'")
