"""Tests for deckard.voxel_exec -- MineBench voxel.exec builds as sigma-ground matter."""
import shutil

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")

from sigma_ground.deckard.voxel_exec import (  # noqa: E402
    UngroundedMaterialError, field_from_voxel_exec, run_voxel_exec,
)
from sigma_ground.field.interface.resolve import material_profile  # noqa: E402


def test_slab_mass_is_blocks_times_sourced_density():
    field, report = field_from_voxel_exec('box(0,0,0,9,0,9,"iron")', block_m=0.5)
    iron = material_profile("iron")["density"]
    assert report["blocks"] == 100
    assert report["materials"]["iron"]["source"] == iron.source
    assert field.mass_kg == pytest.approx(100 * 0.5 ** 3 * iron.value)
    assert field.volume_m3 == pytest.approx(100 * 0.5 ** 3)


def test_minebench_y_up_becomes_sigma_z_up():
    # a 5-block column standing on the ground at MineBench y = 0..4
    field, _ = field_from_voxel_exec('box(3,0,7,3,4,7,"copper")')
    occ = np.argwhere(field.label_grid > 0)
    extent = occ.max(0) - occ.min(0) + 1
    assert tuple(extent) == (1, 1, 5)
    assert field.com_m[2] == pytest.approx(2.5)        # ground at z = 0, 1 m blocks


def test_ungrounded_material_is_refused_with_counts():
    with pytest.raises(UngroundedMaterialError) as err:
        field_from_voxel_exec('box(0,0,0,1,0,1,"unobtainium"); block(5,0,0,"iron")')
    assert err.value.unknown == {"unobtainium": 4}


def test_air_carves_and_a_fully_carved_name_needs_no_grounding():
    field, report = field_from_voxel_exec(
        'box(0,0,0,2,2,2,"iron"); block(1,1,1,"air");'
        'block(9,9,9,"unobtainium"); block(9,9,9,"air")')
    assert report["blocks"] == 26
    assert list(report["materials"]) == ["iron"]
    assert field.materials == ["air", "iron"]


def test_names_with_one_canonical_material_share_a_label():
    field, report = field_from_voxel_exec('block(0,0,0,"ice"); block(1,0,0,"water_ice")')
    assert field.materials == ["air", "water_ice"]
    assert sorted(report["materials"]["water_ice"]["written_as"]) == ["ice", "water_ice"]


def test_contact_face_between_materials_is_scanned():
    field, _ = field_from_voxel_exec('block(0,0,0,"copper"); block(0,1,0,"iron")', block_m=0.1)
    area = {tuple(sorted(k)): v for k, v in field.interfaces.items()}
    assert area[("copper", "iron")] == pytest.approx(0.01)


def test_model_code_has_no_file_access(tmp_path):
    target = tmp_path / "escaped.txt"
    with pytest.raises(RuntimeError, match="ERR_ACCESS_DENIED"):
        run_voxel_exec(f'require("fs").writeFileSync({str(target)!r}, "x")')
    assert not target.exists()
