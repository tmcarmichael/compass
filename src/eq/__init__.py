"""Game environment data  -  asset parsers, world geometry, spell database.

Modules:
    s3d          -  Compressed archive reader (zlib-compressed game assets)
    wld          -  Zone geometry parser (meshes, BSP, regions, materials)
    placeables   -  Object placement footprints for obstacle baking
    strings      -  String normalization and decode helpers
    race_data    -  Race/model/animation metadata
    zone_chr     -  Per-zone model reference parsing
    zone_ids     -  Zone name/ID mapping
    spells       -  Spell data model (SpellData, SpellDB, 8k+ entries)
    loadout      -  Spell loadout management (gem assignment, role lookup)
    game_strings -  Game message database parser
"""

__all__: list[str] = []
