"""Parse zone _chr.txt files -- NPC model/spawn reference per zone.

Zone model reference data removed from public release.

Format: First line is a count, then "model_code,chr_reference" pairs.
These map NPC model short codes to their _chr.txt model definition files.

Example (nektulos_chr.txt):
  24              <- number of entries
  dke,dke         <- race model -> dke_chr.txt
  seg,seg         <- skeleton model -> seg_chr.txt
  snk,snk         <- snake model -> snk_chr.txt
  zom,ecommons_chr <- zombie model, shared from ecommons zone

This tells us what NPC model types are present in a zone -- a proxy for
what npc types can spawn there. The model codes map to race/creature types.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Model code -> creature type mapping (derived from EQ model naming conventions)
# These are the 3-letter model codes used in _chr.txt files.
MODEL_CREATURE_TYPES: set[str] = set()


@dataclass(frozen=True, slots=True)
class ZoneChrEntry:
    """A model reference in a zone's _chr.txt file."""

    model_code: str  # 3-letter model code (e.g., "seg")
    chr_reference: str  # chr file reference (e.g., "seg" or "ecommons_chr")
    creature_type: str  # human-readable type from MODEL_CREATURE_TYPES

    @property
    def is_shared(self) -> bool:
        """Whether this model is referenced from another zone's chr file."""
        return "_chr" in self.chr_reference


@dataclass(slots=True)
class ZoneChrData:
    """All model references for a zone, parsed from <zone>_chr.txt."""

    zone_name: str
    entries: list[ZoneChrEntry] = field(default_factory=list)

    @property
    def model_codes(self) -> list[str]:
        return [e.model_code for e in self.entries]

    @property
    def creature_types(self) -> list[str]:
        return sorted(set(e.creature_type for e in self.entries))

    def has_model(self, code: str) -> bool:
        return any(e.model_code == code for e in self.entries)

    def has_creature_type(self, creature_type: str) -> bool:
        return any(e.creature_type.lower() == creature_type.lower() for e in self.entries)


def load_zone_chr(path: str | Path) -> ZoneChrData:
    """Load zone NPC model data from a <zone>_chr.txt file.

    Stubbed in public release -- returns an empty ZoneChrData.
    """
    path = Path(path)
    zone_name = path.stem.replace("_chr", "")
    return ZoneChrData(zone_name=zone_name)


def load_all_zone_chr(eq_dir: str | Path) -> dict[str, ZoneChrData]:
    """Load all zone _chr.txt files from the EQ directory.

    Stubbed in public release -- returns an empty dict.
    """
    return {}
