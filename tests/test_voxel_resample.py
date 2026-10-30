"""Tests for theater resolution: deckard.voxel_resample and VoxelWorld.resample."""
import shutil

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")

from sigma_ground.deckard.voxel_exec import field_from_voxel_exec  # noqa: E402
from sigma_ground.deckard.voxel_resample import coarsen_labels, parse_factor, resample_field  # noqa: E402
from sigma_ground.materia.voxel_world import VoxelWorld  # noqa: E402

TANK = ('box(0,0,0,7,0,7,"granite"); box(1,1,1,6,5,6,"glass"); box(2,1,2,5,5,5,"air");'
        'box(2,1,2,5,2,5,"mercury"); block(3,3,3,"iron")')


@pytest.fixture(scope="module")
def tank():
    return field_from_voxel_exec(TANK, block_m=0.04)[0]


def test_factor_is_n_or_one_over_n():
    assert parse_factor(2) == ("refine", 2) and parse_factor(0.5) == ("coarsen", 2)
    assert parse_factor(1 / 3) == ("coarsen", 3) and parse_factor(4) == ("refine", 4)
    with pytest.raises(ValueError):
        parse_factor(1.5)


def test_double_resolution_is_exact(tank):
    fine, rep = resample_field(tank, 2)
    assert fine.voxel_size == pytest.approx(tank.voxel_size / 2)
    assert rep["blocks_after"] == 8 * rep["blocks_before"] and rep["mass_change_kg"] == {}
    assert fine.mass_kg == pytest.approx(tank.mass_kg, rel=1e-12)
    assert fine.volume_m3 == pytest.approx(tank.volume_m3, rel=1e-12)
    assert np.allclose(fine.com_m, tank.com_m)
    for pair, area in tank.interfaces.items():
        assert fine.interfaces[pair] == pytest.approx(area)


def test_double_then_half_round_trips(tank):
    back, rep = resample_field(resample_field(tank, 2)[0], 0.5)
    assert np.array_equal(back.label_grid, tank.label_grid)
    assert back.mass_kg == pytest.approx(tank.mass_kg, rel=1e-12) and rep["mass_change_kg"] == {}


def test_half_resolution_drops_lone_blocks_and_says_so():
    field, _ = field_from_voxel_exec('box(0,0,0,3,1,3,"granite"); block(0,2,0,"gold")', block_m=0.1)
    coarse, rep = resample_field(field, 0.5)
    assert "gold" not in [field.materials[k] for k in np.unique(coarse.label_grid) if k]
    assert rep["mass_change_kg"]["gold"] == pytest.approx(-rep["materials_before"]["gold"]["mass_kg"])
    assert coarse.mass_kg == pytest.approx(rep["mass_after_kg"])


def test_merge_takes_the_majority_and_a_tie_goes_to_the_denser():
    label = np.zeros((2, 2, 2), dtype=np.int32)
    label[0] = 1                                         # 4 granite
    label[1] = 2                                         # 4 gold: a tie
    out, _ = coarsen_labels(label, 2, [0.0, 2650.0, 19300.0])
    assert out[0, 0, 0] == 2
    label[1, 1, 1] = 1                                   # 5 granite, 3 gold
    assert coarsen_labels(label, 2, [0.0, 2650.0, 19300.0])[0][0, 0, 0] == 1
    label[:] = 0
    label[0, 0, :] = 1                                   # 2 of 8: air wins
    assert coarsen_labels(label, 2, [0.0, 2650.0, 19300.0])[0][0, 0, 0] == 0


def test_world_refines_exactly_and_keeps_running(tank):
    T = np.where(tank.label_grid == tank.materials.index("iron"), 500.0, 293.15)
    w = VoxelWorld(tank, temperature_k=T, env="ISM")
    m0, E0, floor = w.mass_kg(), w.energy_J(), w.floor_z
    rep = w.resample(2)
    assert rep["mass_change_kg"] == pytest.approx(0.0, abs=1e-12 * m0)
    assert rep["energy_change_J"] == pytest.approx(0.0, abs=1e-9 * E0)
    assert w.floor_z == 2 * floor and w.T.max() == 500.0
    w.run(3, 0.01)
    assert w.mass_kg() == pytest.approx(m0, rel=1e-12)


def test_world_coarsens_with_heat_capacity_weighted_temperature():
    field, _ = field_from_voxel_exec('box(0,0,0,1,1,1,"copper")', block_m=0.01)
    T = np.full(field.label_grid.shape, 293.15)
    cu = field.label_grid == field.materials.index("copper")
    T[cu] = np.where(np.arange(cu.sum()) % 2, 400.0, 300.0)   # half the copper hot, half cool
    w = VoxelWorld(field, temperature_k=T, env="ISM")
    E0 = w.energy_J()
    w.resample(0.5)
    cells = w.label == w.materials.index("copper")
    assert cells.sum() >= 1 and np.allclose(w.T[cells], 350.0)
    assert w.energy_J() == pytest.approx(E0, rel=1e-9)        # one material, all kept: no energy lost
