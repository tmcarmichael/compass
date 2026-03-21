"""Tests for eq/ binary parsers: zone_chr, wld, s3d.

Unit tests using synthetic data -- no game assets required.
Tests data classes, stub behavior (NotImplementedError), and zone_chr parsing.
"""

from __future__ import annotations

import pytest

from eq.wld import (
    BSPNode,
    Mesh,
    MeshTriangle,
    MeshVertex,
    ObjectPlacement,
    RegionType,
    WLDFile,
)
from eq.zone_chr import (
    MODEL_CREATURE_TYPES,
    ZoneChrData,
    ZoneChrEntry,
    load_all_zone_chr,
    load_zone_chr,
)


# ===========================================================================
# zone_chr.py -- data classes (not stubbed)
# ===========================================================================


class TestZoneChrEntry:
    def test_construction(self):
        entry = ZoneChrEntry(model_code="seg", chr_reference="seg", creature_type="Skeleton")
        assert entry.model_code == "seg"
        assert entry.chr_reference == "seg"
        assert entry.creature_type == "Skeleton"

    def test_is_shared_false(self):
        entry = ZoneChrEntry(model_code="seg", chr_reference="seg", creature_type="Skeleton")
        assert entry.is_shared is False

    def test_is_shared_true(self):
        entry = ZoneChrEntry(model_code="zom", chr_reference="ecommons_chr", creature_type="Zombie")
        assert entry.is_shared is True

    def test_frozen(self):
        entry = ZoneChrEntry(model_code="seg", chr_reference="seg", creature_type="Skeleton")
        with pytest.raises(AttributeError):
            entry.model_code = "xxx"


class TestZoneChrData:
    def _make_data(self) -> ZoneChrData:
        return ZoneChrData(
            zone_name="nektulos",
            entries=[
                ZoneChrEntry("seg", "seg", "Skeleton"),
                ZoneChrEntry("zom", "ecommons_chr", "Zombie"),
                ZoneChrEntry("bat", "bat", "Bat"),
                ZoneChrEntry("seg", "seg", "Skeleton"),  # duplicate type
            ],
        )

    def test_model_codes(self):
        data = self._make_data()
        assert data.model_codes == ["seg", "zom", "bat", "seg"]

    def test_creature_types_sorted_unique(self):
        data = self._make_data()
        types = data.creature_types
        assert types == ["Bat", "Skeleton", "Zombie"]

    def test_has_model_found(self):
        data = self._make_data()
        assert data.has_model("seg") is True
        assert data.has_model("bat") is True

    def test_has_model_not_found(self):
        data = self._make_data()
        assert data.has_model("dra") is False

    def test_has_creature_type_case_insensitive(self):
        data = self._make_data()
        assert data.has_creature_type("skeleton") is True
        assert data.has_creature_type("SKELETON") is True
        assert data.has_creature_type("Skeleton") is True

    def test_has_creature_type_not_found(self):
        data = self._make_data()
        assert data.has_creature_type("Dragon") is False

    def test_empty_data(self):
        data = ZoneChrData(zone_name="empty")
        assert data.model_codes == []
        assert data.creature_types == []
        assert data.has_model("seg") is False
        assert data.has_creature_type("Skeleton") is False


class TestLoadZoneChr:
    """load_zone_chr is stubbed -- returns empty ZoneChrData with zone_name from path."""

    def test_nonexistent_path_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent_chr.txt"
        data = load_zone_chr(path)
        assert data.zone_name == "nonexistent"
        assert data.entries == []

    def test_zone_name_extracted_from_stem(self, tmp_path):
        path = tmp_path / "gukbottom_chr.txt"
        data = load_zone_chr(path)
        assert data.zone_name == "gukbottom"

    def test_returns_empty_even_with_file(self, tmp_path):
        """Stubbed load_zone_chr ignores file content."""
        content = "3\nseg,seg\nzom,ecommons_chr\nbat,bat\n"
        path = tmp_path / "nektulos_chr.txt"
        path.write_text(content)
        data = load_zone_chr(path)
        assert data.entries == []


class TestLoadAllZoneChr:
    """load_all_zone_chr is stubbed -- returns empty dict."""

    def test_empty_directory(self, tmp_path):
        result = load_all_zone_chr(tmp_path)
        assert result == {}

    def test_returns_empty_even_with_files(self, tmp_path):
        """Stubbed load_all_zone_chr ignores directory content."""
        (tmp_path / "nektulos_chr.txt").write_text("1\nseg,seg\n")
        result = load_all_zone_chr(tmp_path)
        assert result == {}


class TestModelCreatureTypes:
    """MODEL_CREATURE_TYPES is stubbed to an empty set."""

    def test_is_set(self):
        assert isinstance(MODEL_CREATURE_TYPES, set)


# ===========================================================================
# wld.py -- data classes (not stubbed)
# ===========================================================================


class TestMeshVertex:
    def test_construction(self):
        v = MeshVertex(x=1.0, y=2.0, z=3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_frozen(self):
        v = MeshVertex(x=1.0, y=2.0, z=3.0)
        with pytest.raises(AttributeError):
            v.x = 99.0


class TestMeshTriangle:
    def test_construction(self):
        t = MeshTriangle(v1=0, v2=1, v3=2, flags=0x10, material_idx=3)
        assert t.v1 == 0
        assert t.v2 == 1
        assert t.v3 == 2
        assert t.flags == 0x10
        assert t.material_idx == 3

    def test_frozen(self):
        t = MeshTriangle(v1=0, v2=1, v3=2, flags=0, material_idx=0)
        with pytest.raises(AttributeError):
            t.v1 = 99


class TestMesh:
    def test_construction(self):
        verts = [MeshVertex(0.0, 0.0, 0.0), MeshVertex(1.0, 0.0, 0.0)]
        tris = [MeshTriangle(0, 1, 0, 0, 0)]
        m = Mesh(name="test_mesh", vertices=verts, triangles=tris, center=(0.0, 0.0, 0.0))
        assert m.name == "test_mesh"
        assert len(m.vertices) == 2
        assert len(m.triangles) == 1
        assert m.center == (0.0, 0.0, 0.0)


class TestBSPNode:
    def test_construction(self):
        node = BSPNode(
            normal_x=1.0,
            normal_y=0.0,
            normal_z=0.0,
            split_distance=100.0,
            region=5,
            front=2,
            back=3,
        )
        assert node.normal_x == 1.0
        assert node.split_distance == 100.0
        assert node.region == 5
        assert node.front == 2
        assert node.back == 3

    def test_frozen(self):
        node = BSPNode(0.0, 1.0, 0.0, 50.0, 0, 0, 0)
        with pytest.raises(AttributeError):
            node.region = 99


class TestObjectPlacement:
    def test_construction(self):
        p = ObjectPlacement(
            name="OBJ_TREE01",
            model_name="TREE01",
            x=100.0,
            y=200.0,
            z=50.0,
            heading=128.0,
            scale=1.5,
        )
        assert p.name == "OBJ_TREE01"
        assert p.model_name == "TREE01"
        assert p.x == 100.0
        assert p.y == 200.0
        assert p.z == 50.0
        assert p.heading == 128.0
        assert p.scale == 1.5

    def test_frozen(self):
        p = ObjectPlacement("a", "b", 0.0, 0.0, 0.0, 0.0, 1.0)
        with pytest.raises(AttributeError):
            p.x = 99.0


# ===========================================================================
# wld.py -- RegionType.from_name (algorithmic, not stubbed)
# ===========================================================================


class TestRegionType:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("WT_Zone_Water_01", RegionType.WATER),
            ("wt_water_test", RegionType.WATER),
            ("WTN_Pool_01", RegionType.WATER),
            ("LA_Zone_Lava_01", RegionType.LAVA),
            ("la_lava_pool", RegionType.LAVA),
            ("LAN_Lava_01", RegionType.LAVA),
            ("WTNTP_Zoneline_01", RegionType.ZONELINE),
            ("DRNTP_Zoneline_02", RegionType.ZONELINE),
            ("LANTP_Zoneline_03", RegionType.ZONELINE),
            ("SLN_Underwater_01", RegionType.WATER_BLOCK_LOS),
            ("VWN_Freeze_01", RegionType.FREEZING_WATER),
            ("DRP_Arena_01", RegionType.PVP),
            ("DRN_Ice_S_01", RegionType.SLIPPERY),
            ("VWA_Surface_01", RegionType.VWATER),
            ("DRN_Normal_01", RegionType.NORMAL),
        ],
    )
    def test_known_prefixes(self, name: str, expected: int):
        assert RegionType.from_name(name) == expected

    def test_unknown_prefix_returns_none(self):
        assert RegionType.from_name("UNKNOWN_Something") is None
        assert RegionType.from_name("") is None

    def test_case_insensitive(self):
        assert RegionType.from_name("wt_water") == RegionType.WATER
        assert RegionType.from_name("WT_WATER") == RegionType.WATER

    def test_backward_compat_aliases(self):
        assert RegionType.SLIME == RegionType.WATER_BLOCK_LOS
        assert RegionType.ICE == RegionType.SLIPPERY


# ===========================================================================
# wld.py -- WLDFile raises NotImplementedError (stubbed)
# ===========================================================================


class TestWLDFileStubbed:
    def test_constructor_raises(self):
        with pytest.raises(NotImplementedError, match="stubbed"):
            WLDFile(b"\x00" * 100)

    def test_constructor_raises_with_any_data(self):
        with pytest.raises(NotImplementedError, match="stubbed"):
            WLDFile(b"")


# ===========================================================================
# s3d.py -- S3DArchive raises NotImplementedError (stubbed)
# ===========================================================================


class TestS3DArchiveStubbed:
    def test_constructor_raises(self, tmp_path):
        from eq.s3d import S3DArchive

        path = tmp_path / "test.s3d"
        path.write_bytes(b"\x00" * 100)
        with pytest.raises(NotImplementedError, match="stubbed"):
            S3DArchive(path)

    def test_constructor_raises_nonexistent(self, tmp_path):
        from eq.s3d import S3DArchive

        with pytest.raises(NotImplementedError, match="stubbed"):
            S3DArchive(tmp_path / "nonexistent.s3d")
