"""Tests for radiance.voxel_scene -- a multi-material voxel build, one leaf per material, staged and rendered."""
import shutil

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")

from sigma_ground.deckard.voxel_exec import field_from_voxel_exec  # noqa: E402
from sigma_ground.kernel.vec import Vec3  # noqa: E402
from sigma_ground.radiance.scene_export import scene_spec_to_sdf  # noqa: E402
from sigma_ground.radiance.voxel_scene import render_voxel_field, scene_from_voxel_field  # noqa: E402

TANK = ('box(0,0,0,7,0,7,"granite"); box(1,1,1,6,4,6,"glass"); box(2,1,2,5,4,5,"air");'
        'box(2,1,2,5,2,5,"mercury"); block(3,3,3,"gold")')


@pytest.fixture(scope="module")
def tank():
    field, _ = field_from_voxel_exec(TANK, block_m=0.05)
    return field


def _world(field, idx):
    lo = np.asarray(field.center) - np.asarray(field.label_grid.shape) * field.voxel_size / 2
    return Vec3(*(lo + (np.asarray(idx) + 0.5) * field.voxel_size))


def test_one_leaf_per_material_and_air_is_none(tank):
    spec = scene_from_voxel_field(tank)
    assert [leaf["material"] for leaf in spec["csg_leaves"]] == ["granite", "glass", "mercury", "gold"]
    assert set(spec["materials"]) == {"granite", "glass", "mercury", "gold"}
    assert spec["materials"]["gold"]["color_rgb"] != spec["materials"]["mercury"]["color_rgb"]
    assert spec["physics"]["mass_kg"] == pytest.approx(tank.mass_kg)


def test_each_cell_draws_as_its_own_material_and_air_is_empty(tank):
    sdf, material_at = scene_spec_to_sdf(scene_from_voxel_field(tank))
    for k, mat in enumerate(tank.materials[1:], start=1):
        p = _world(tank, np.argwhere(tank.label_grid == k)[0])
        assert sdf(p) < 0 and material_at(p) == mat
    gold = np.argwhere(tank.label_grid == tank.materials.index("gold"))[0]
    above = gold + np.array([0, 0, 1])                  # the air over the gold block, inside the tank
    assert tank.label_grid[tuple(above)] == 0
    assert sdf(_world(tank, above)) > 0


def test_renders_in_a_theater(tank, tmp_path):
    path = render_voxel_field(tank, str(tmp_path / "tank.png"), width=48, height=36)
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 200
