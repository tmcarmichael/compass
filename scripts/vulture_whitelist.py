# Vulture whitelist: false positives for unused function parameters.
# These are interface contracts or keyword arguments required by callers.

wnd_name  # _read_wnd_pos parameter (routines/memorize_spells.py)
medding  # try_heal keyword argument (routines/pet_combat.py)
filename  # S3DArchive.extract / __contains__ parameter (eq/s3d.py)
eq_dir  # load_all_zone_chr parameter (eq/zone_chr.py)
repeat_count  # _compute_detour parameter (nav/movement.py)
