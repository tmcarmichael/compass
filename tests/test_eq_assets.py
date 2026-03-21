"""Tests for eq/ parsers requiring game asset files on disk.

All tests are marked with @pytest.mark.assets and will be skipped
when the EQ game directory is not present. Designed for local
Windows testing with EQ installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Asset directory lookup
# ---------------------------------------------------------------------------

_eq_dir_env = os.environ.get("EQ_DIR", "")
EQ_DIR = Path(_eq_dir_env) if _eq_dir_env else Path("__nonexistent__")

_assets_available = EQ_DIR.is_dir()

assets = pytest.mark.skipif(not _assets_available, reason="Requires EQ game assets directory")


# ---------------------------------------------------------------------------
# S3D Archive tests
# ---------------------------------------------------------------------------


@assets
class TestS3DArchive:
    """Test S3DArchive with nektulos.s3d."""

    @pytest.fixture(autouse=True)
    def _archive(self):
        from eq.s3d import S3DArchive

        self.archive = S3DArchive(EQ_DIR / "nektulos.s3d")

    def test_list_files_non_empty(self):
        files = self.archive.list_files()
        assert isinstance(files, list)
        assert len(files) > 0

    def test_list_files_contains_wld(self):
        files = self.archive.list_files()
        wld_files = [f for f in files if f.endswith(".wld")]
        assert len(wld_files) > 0, f"No .wld files found. Files: {files[:10]}"

    def test_extract_returns_bytes(self):
        files = self.archive.list_files()
        assert len(files) > 0
        data = self.archive.extract(files[0])
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_contains_check(self):
        files = self.archive.list_files()
        assert len(files) > 0
        # First file should pass membership check
        assert files[0] in self.archive
        # Non-existent file should fail
        assert "definitely_not_a_real_file.xyz" not in self.archive

    def test_extract_wld_file(self):
        files = self.archive.list_files()
        wld_files = [f for f in files if f.endswith(".wld")]
        assert len(wld_files) > 0
        wld_data = self.archive.extract(wld_files[0])
        assert isinstance(wld_data, bytes)
        assert len(wld_data) > 100  # WLD files are substantial

    def test_extract_nonexistent_raises(self):
        with pytest.raises(KeyError):
            self.archive.extract("nonexistent_file.txt")

    def test_repr(self):
        r = repr(self.archive)
        assert "S3DArchive" in r
        assert "nektulos.s3d" in r


# ---------------------------------------------------------------------------
# WLD File tests
# ---------------------------------------------------------------------------


@assets
class TestWLDFile:
    """Test WLDFile parsing of zone geometry from nektulos.s3d."""

    @pytest.fixture(autouse=True)
    def _wld(self):
        from eq.s3d import S3DArchive
        from eq.wld import WLDFile

        archive = S3DArchive(EQ_DIR / "nektulos.s3d")
        files = archive.list_files()
        wld_files = [f for f in files if f.endswith(".wld")]
        assert len(wld_files) > 0, "No .wld file in nektulos.s3d"
        # Use the zone geometry WLD (named after the zone), not lights.wld
        zone_wld = [f for f in wld_files if "nektulos" in f.lower()]
        wld_name = zone_wld[0] if zone_wld else wld_files[0]
        wld_data = archive.extract(wld_name)
        self.wld = WLDFile(wld_data)

    def test_extract_meshes(self):
        meshes = self.wld.extract_meshes()
        assert isinstance(meshes, list)
        assert len(meshes) > 0

    def test_mesh_has_vertices(self):
        meshes = self.wld.extract_meshes()
        assert len(meshes) > 0
        for mesh in meshes[:5]:  # check first 5
            assert len(mesh.vertices) > 0
            v = mesh.vertices[0]
            assert hasattr(v, "x")
            assert hasattr(v, "y")
            assert hasattr(v, "z")
            assert isinstance(v.x, float)
            assert isinstance(v.y, float)
            assert isinstance(v.z, float)

    def test_mesh_has_triangles(self):
        meshes = self.wld.extract_meshes()
        # At least some meshes should have triangles
        meshes_with_tris = [m for m in meshes if m.triangles]
        assert len(meshes_with_tris) > 0

    def test_mesh_has_name(self):
        meshes = self.wld.extract_meshes()
        assert len(meshes) > 0
        for mesh in meshes[:5]:
            assert isinstance(mesh.name, str)

    def test_mesh_has_center(self):
        meshes = self.wld.extract_meshes()
        assert len(meshes) > 0
        for mesh in meshes[:5]:
            assert isinstance(mesh.center, tuple)
            assert len(mesh.center) == 3

    def test_extract_bsp_nodes(self):
        nodes = self.wld.extract_bsp_nodes()
        assert isinstance(nodes, list)
        assert len(nodes) > 0

    def test_bsp_node_fields(self):
        from eq.wld import BSPNode

        nodes = self.wld.extract_bsp_nodes()
        assert len(nodes) > 0
        node = nodes[0]
        assert isinstance(node, BSPNode)
        assert isinstance(node.normal_x, float)
        assert isinstance(node.normal_y, float)
        assert isinstance(node.normal_z, float)
        assert isinstance(node.split_distance, float)
        assert isinstance(node.region, int)
        assert isinstance(node.front, int)
        assert isinstance(node.back, int)

    def test_extract_region_types(self):
        region_types = self.wld.extract_region_types()
        assert isinstance(region_types, dict)
        # nektulos should have at least some region types (water, etc.)

    def test_extract_placements(self):
        placements = self.wld.extract_placements()
        assert isinstance(placements, list)
        # Zone geometry WLD may or may not have placements

    def test_placement_fields(self):
        from eq.wld import ObjectPlacement

        placements = self.wld.extract_placements()
        for p in placements[:5]:
            assert isinstance(p, ObjectPlacement)
            assert isinstance(p.name, str)
            assert isinstance(p.model_name, str)
            assert isinstance(p.x, float)
            assert isinstance(p.y, float)
            assert isinstance(p.z, float)
            assert isinstance(p.heading, float)
            assert isinstance(p.scale, float)
            assert p.scale > 0

    def test_fragment_type_summary(self):
        summary = self.wld.fragment_type_summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0
        # Should contain 0x36 (mesh) fragments
        assert 0x36 in summary
        assert summary[0x36] > 0


# ---------------------------------------------------------------------------
# WLD objects_wld (second WLD file in S3D, if present)
# ---------------------------------------------------------------------------


@assets
class TestWLDObjectsFile:
    """Test the objects WLD file from nektulos.s3d if it contains one."""

    @pytest.fixture(autouse=True)
    def _wld(self):
        from eq.s3d import S3DArchive
        from eq.wld import WLDFile

        archive = S3DArchive(EQ_DIR / "nektulos.s3d")
        files = archive.list_files()
        wld_files = [f for f in files if f.endswith(".wld")]
        if len(wld_files) < 2:
            pytest.skip("No objects WLD in nektulos.s3d")
        # Objects WLD is typically the one named "objects.wld"
        obj_wld_name = None
        for name in wld_files:
            if "obj" in name.lower():
                obj_wld_name = name
                break
        if obj_wld_name is None:
            # fallback: just use the second WLD
            obj_wld_name = wld_files[1]
        wld_data = archive.extract(obj_wld_name)
        self.wld = WLDFile(wld_data)

    def test_fragment_summary_non_empty(self):
        summary = self.wld.fragment_type_summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0


# ---------------------------------------------------------------------------
# Zone CHR tests
# ---------------------------------------------------------------------------


@assets
class TestZoneChr:
    """Test zone_chr.py parsing with nektulos_chr.txt."""

    def test_load_zone_chr(self):
        from eq.zone_chr import load_zone_chr

        data = load_zone_chr(EQ_DIR / "nektulos_chr.txt")
        assert data.zone_name == "nektulos"
        assert len(data.entries) > 0

    def test_model_codes(self):
        from eq.zone_chr import load_zone_chr

        data = load_zone_chr(EQ_DIR / "nektulos_chr.txt")
        codes = data.model_codes
        assert isinstance(codes, list)
        assert len(codes) > 0
        for code in codes:
            assert isinstance(code, str)
            assert len(code) > 0

    def test_creature_types(self):
        from eq.zone_chr import load_zone_chr

        data = load_zone_chr(EQ_DIR / "nektulos_chr.txt")
        types = data.creature_types
        assert isinstance(types, list)
        assert len(types) > 0
        # Should be sorted with no duplicates
        assert types == sorted(set(types))

    def test_has_model(self):
        from eq.zone_chr import load_zone_chr

        data = load_zone_chr(EQ_DIR / "nektulos_chr.txt")
        assert len(data.entries) > 0
        first_code = data.entries[0].model_code
        assert data.has_model(first_code) is True
        assert data.has_model("xyznonexistent") is False

    def test_entry_fields(self):
        from eq.zone_chr import ZoneChrEntry, load_zone_chr

        data = load_zone_chr(EQ_DIR / "nektulos_chr.txt")
        assert len(data.entries) > 0
        entry = data.entries[0]
        assert isinstance(entry, ZoneChrEntry)
        assert isinstance(entry.model_code, str)
        assert isinstance(entry.chr_reference, str)
        assert isinstance(entry.creature_type, str)

    def test_is_shared_property(self):
        from eq.zone_chr import load_zone_chr

        data = load_zone_chr(EQ_DIR / "nektulos_chr.txt")
        # At least verify the property doesn't crash and returns bool
        for entry in data.entries:
            assert isinstance(entry.is_shared, bool)

    def test_load_nonexistent_returns_empty(self):
        from eq.zone_chr import load_zone_chr

        data = load_zone_chr(EQ_DIR / "definitely_not_a_zone_chr.txt")
        assert data.entries == []

    def test_has_creature_type(self):
        from eq.zone_chr import load_zone_chr

        data = load_zone_chr(EQ_DIR / "nektulos_chr.txt")
        types = data.creature_types
        if types:
            assert data.has_creature_type(types[0]) is True
        assert data.has_creature_type("NonExistentCreature9999") is False
