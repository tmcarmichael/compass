"""Tests for brain.learning.zone -- persistent per-zone knowledge (dispositions & social).

Covers ZoneKnowledge creation, disposition recording, social group learning,
merged disposition/social output, persistence (tmp_path), and TOML seed merging.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from brain.learning.zone import _DISPOSITION_MAX_AGE_S, ZoneKnowledge, _base_name
from core.types import Disposition

# ---------------------------------------------------------------------------
# _base_name helper
# ---------------------------------------------------------------------------


class TestBaseName:
    def test_strips_trailing_digits(self) -> None:
        assert _base_name("a_spiderling017") == "a_spiderling"

    def test_lowercase(self) -> None:
        assert _base_name("A_Skeleton") == "a_skeleton"

    def test_no_suffix(self) -> None:
        assert _base_name("a_bat") == "a_bat"

    def test_empty_string(self) -> None:
        assert _base_name("") == ""


# ---------------------------------------------------------------------------
# Disposition API
# ---------------------------------------------------------------------------


class TestDispositions:
    def test_record_new_disposition(self, tmp_path: Path) -> None:
        zk = ZoneKnowledge("gfay", data_dir=tmp_path)
        changed = zk.record_disposition("a_bat", Disposition.SCOWLING)
        assert changed is True
        assert zk.learned_disposition_count == 1

    def test_record_same_disposition_returns_false(self, tmp_path: Path) -> None:
        zk = ZoneKnowledge("gfay", data_dir=tmp_path)
        zk.record_disposition("a_bat", Disposition.SCOWLING)
        changed = zk.record_disposition("a_bat", Disposition.SCOWLING)
        assert changed is False

    def test_record_changed_disposition(self, tmp_path: Path) -> None:
        zk = ZoneKnowledge("gfay", data_dir=tmp_path)
        zk.record_disposition("a_bat", Disposition.SCOWLING)
        changed = zk.record_disposition("a_bat", Disposition.INDIFFERENT)
        assert changed is True

    def test_merged_dispositions_from_learned(self, tmp_path: Path) -> None:
        zk = ZoneKnowledge("gfay", data_dir=tmp_path)
        zk.record_disposition("a_bat", Disposition.SCOWLING)
        merged = zk.get_merged_dispositions()
        assert "scowling" in merged
        assert "a_bat" in merged["scowling"]

    def test_merged_dispositions_with_toml_seed(self, tmp_path: Path) -> None:
        toml_disp = {"indifferent": ["a_fairy", "a_pixie"]}
        zk = ZoneKnowledge("gfay", data_dir=tmp_path, toml_dispositions=toml_disp)
        merged = zk.get_merged_dispositions()
        assert "a_fairy" in merged["indifferent"]

    def test_learned_overrides_toml(self, tmp_path: Path) -> None:
        """If TOML says indifferent but we learned scowling, mob moves category."""
        toml_disp = {"indifferent": ["a_bat"]}
        zk = ZoneKnowledge("gfay", data_dir=tmp_path, toml_dispositions=toml_disp)
        zk.record_disposition("a_bat", Disposition.SCOWLING)
        merged = zk.get_merged_dispositions()
        # a_bat should be in scowling, NOT in indifferent
        assert "a_bat" in merged.get("scowling", [])
        assert "a_bat" not in merged.get("indifferent", [])


# ---------------------------------------------------------------------------
# Social threat groups
# ---------------------------------------------------------------------------


class TestSocialGroups:
    def test_record_social_add(self, tmp_path: Path) -> None:
        zk = ZoneKnowledge("gfay", data_dir=tmp_path)
        result = zk.record_social_add("a_skeleton017", "a_zombie003")
        assert result is True
        assert zk.learned_social_group_count == 1

    def test_same_base_name_ignored(self, tmp_path: Path) -> None:
        """Same NPC type assisting itself is not a social add."""
        zk = ZoneKnowledge("gfay", data_dir=tmp_path)
        result = zk.record_social_add("a_bat001", "a_bat002")
        assert result is False

    def test_duplicate_add_ignored(self, tmp_path: Path) -> None:
        zk = ZoneKnowledge("gfay", data_dir=tmp_path)
        zk.record_social_add("a_skeleton001", "a_zombie001")
        result = zk.record_social_add("a_skeleton002", "a_zombie003")
        assert result is False  # already known

    def test_merge_groups(self, tmp_path: Path) -> None:
        """A adds B, C adds D, then B adds C -> all merge into one group."""
        zk = ZoneKnowledge("gfay", data_dir=tmp_path)
        zk.record_social_add("a_skeleton001", "a_zombie001")
        zk.record_social_add("a_bat001", "a_spider001")
        # Now link the two groups
        zk.record_social_add("a_zombie002", "a_bat003")
        assert zk.learned_social_group_count == 1
        groups = zk.get_merged_social_groups()
        assert len(groups) == 1
        assert set(groups[0]) == {"a_skeleton", "a_zombie", "a_bat", "a_spider"}

    def test_merged_social_with_toml_seed(self, tmp_path: Path) -> None:
        toml_groups = [["a_fire_beetle", "a_fire_beetle_queen"]]
        zk = ZoneKnowledge("gfay", data_dir=tmp_path, toml_social_groups=toml_groups)
        groups = zk.get_merged_social_groups()
        assert any("a_fire_beetle" in g for g in groups)

    def test_build_social_mob_group_lookup(self, tmp_path: Path) -> None:
        zk = ZoneKnowledge("gfay", data_dir=tmp_path)
        zk.record_social_add("a_skeleton001", "a_zombie001")
        lookup = zk.build_social_mob_group()
        assert "a_skeleton" in lookup
        assert "a_zombie" in lookup["a_skeleton"]


# ---------------------------------------------------------------------------
# Persistence (tmp_path)
# ---------------------------------------------------------------------------


class TestZonePersistence:
    def test_save_and_reload(self, tmp_path: Path) -> None:
        zk = ZoneKnowledge("gfay", data_dir=tmp_path)
        zk.record_disposition("a_bat", Disposition.SCOWLING)
        zk.record_social_add("a_skeleton001", "a_zombie001")
        zk.save()

        zk2 = ZoneKnowledge("gfay", data_dir=tmp_path)
        assert zk2.learned_disposition_count == 1
        assert zk2.learned_social_group_count == 1

    def test_save_creates_file(self, tmp_path: Path) -> None:
        zk = ZoneKnowledge("testzone", data_dir=tmp_path)
        zk.record_disposition("a_bat", Disposition.INDIFFERENT)
        zk.save()
        path = tmp_path / "testzone.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["v"] == 1
        assert data["zone"] == "testzone"

    def test_dirty_flag(self, tmp_path: Path) -> None:
        zk = ZoneKnowledge("gfay", data_dir=tmp_path)
        assert not zk.dirty
        zk.record_disposition("a_bat", Disposition.SCOWLING)
        assert zk.dirty
        zk.save()
        assert not zk.dirty

    def test_corrupted_file_handled(self, tmp_path: Path) -> None:
        path = tmp_path / "badzone.json"
        path.write_text("{invalid json")
        zk = ZoneKnowledge("badzone", data_dir=tmp_path)
        assert zk.learned_disposition_count == 0

    def test_prune_old_dispositions_on_save(self, tmp_path: Path) -> None:
        zk = ZoneKnowledge("gfay", data_dir=tmp_path)
        zk.record_disposition("a_bat", Disposition.SCOWLING)

        # Manually set the time to be very old
        zk._dispositions["a_bat"]["time"] = time.time() - _DISPOSITION_MAX_AGE_S - 100
        zk.save()

        zk2 = ZoneKnowledge("gfay", data_dir=tmp_path)
        assert zk2.learned_disposition_count == 0
