"""Parse EQ RaceData.txt -- race/model/animation mapping table.

Race/model data removed from public release. The data classes define
the schema; populate from environment-specific sources.

Format: caret-delimited, one row per race ID.
Fields: RaceID ^ 3 unknowns ^ male_model_ids[14] ^ female_model_ids[14]

Each row has 37 fields total:
  [0]  Race ID (0-based, 0=Human/Unknown, 1=Human, 2=Barbarian, etc.)
  [1-3]  Unknown flags (always 0 in observed data)
  [4-17]  Male model/animation IDs (14 fields: base model, skeleton, animation sets)
  [18-21]  Padding (always 0, 4 fields)
  [22-35]  Female model/animation IDs (14 fields, same layout as male)
  [36]  Trailing padding (always 0)

The model IDs reference model asset files. The animation IDs (3001, 3002, etc.)
reference shared animation sets. Race 0 and 1 both map to Human models.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

RACE_NAMES: dict[int, str] = {}


@dataclass(frozen=True, slots=True)
class RaceModelData:
    """Model and animation IDs for a single race."""

    race_id: int
    male_model_ids: tuple[int, ...]  # 14 IDs: base model, skeleton, animation sets
    female_model_ids: tuple[int, ...]  # 14 IDs: same layout

    @property
    def name(self) -> str:
        return RACE_NAMES.get(self.race_id, f"Race_{self.race_id}")

    @property
    def male_base_model(self) -> int:
        return self.male_model_ids[0] if self.male_model_ids else 0

    @property
    def female_base_model(self) -> int:
        return self.female_model_ids[0] if self.female_model_ids else 0


@dataclass(slots=True)
class RaceDB:
    """All race data from RaceData.txt."""

    _by_id: dict[int, RaceModelData] = field(default_factory=dict)

    def get(self, race_id: int) -> RaceModelData | None:
        return self._by_id.get(race_id)

    def name(self, race_id: int) -> str:
        rd = self._by_id.get(race_id)
        if rd:
            return rd.name
        return RACE_NAMES.get(race_id, f"Race_{race_id}")

    def all_races(self) -> list[RaceModelData]:
        return sorted(self._by_id.values(), key=lambda r: r.race_id)

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, race_id: int) -> bool:
        return race_id in self._by_id


def load_race_data(path: str | Path) -> RaceDB:
    """Load race/model data from RaceData.txt.

    Format: caret-delimited, 37 fields per line.
    Field 0 = race ID, fields 4-17 = male models, fields 21-34 = female models.

    Stubbed in public release -- returns an empty RaceDB.
    """
    return RaceDB()
