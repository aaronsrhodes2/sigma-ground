"""Tests for theater resolution (set once, at build time): deckard.voxel_resample."""
import shutil

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")

from sigma_ground.deckard.voxel_exec import field_from_voxel_exec  # noqa: E402
from sigma_ground.deckard.voxel_resample import coarsen_labels, parse_factor, resample_field  # noqa: E402

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


def test_shape_mode_takes_the_majority_and_a_tie_goes_to_the_denser():
    d = [0.0, 2650.0, 19300.0]
    label = np.zeros((2, 2, 2), dtype=np.int32)
    label[0] = 1                                         # 4 granite
    label[1] = 2                                         # 4 gold: a tie
    assert coarsen_labels(label, 2, d, "shape")[0][0, 0, 0] == 2
    label[1, 1, 1] = 1                                   # 5 granite, 3 gold
    assert coarsen_labels(label, 2, d, "shape")[0][0, 0, 0] == 1
    label[:] = 0
    label[0, 0, :] = 1                                   # 2 of 8: air wins
    assert coarsen_labels(label, 2, d, "shape")[0][0, 0, 0] == 0


def test_mass_mode_turns_a_thin_floor_into_an_even_checkerboard():
    label = np.zeros((8, 8, 2), dtype=np.int32)
    label[:, :, 0] = 1                                   # a one-block granite floor: half of every merged cell
    coarse, _ = coarsen_labels(label, 2, [0.0, 2650.0])
    assert coarse.sum() == 8                             # 64 blocks / 8 = 8 merged blocks: the mass, exactly
    layer = coarse[:, :, 0]
    assert ((np.indices(layer.shape).sum(0) % 2 == 0) == (layer == 1)).all() or            ((np.indices(layer.shape).sum(0) % 2 == 1) == (layer == 1)).all()
    shape, _ = coarsen_labels(label, 2, [0.0, 2650.0], "shape")
    assert shape.sum() == 16                             # the per-block vote doubles it


def test_mass_mode_fills_the_fullest_cells_first():
    label = np.zeros((4, 2, 2), dtype=np.int32)
    label[0:2] = 1                                       # one merged cell full of iron
    label[2, 0, 0] = 1                                   # and a stray block in the next
    coarse, _ = coarsen_labels(label, 2, [0.0, 7870.0])
    assert coarse[0, 0, 0] == 1 and coarse[1, 0, 0] == 0


def test_mass_is_kept_to_within_half_a_merged_block_per_material(tank):
    coarse, rep = resample_field(tank, 0.5)
    merged_m3 = coarse.voxel_size ** 3
    for m, b in rep["materials_before"].items():
        rho = b["mass_kg"] / (b["blocks"] * tank.voxel_size ** 3)
        assert abs(rep["materials_after"][m]["mass_kg"] - b["mass_kg"]) <= 0.5 * rho * merged_m3 + 1e-9


def test_resolution_is_set_when_the_theater_is_built():
    field, rep = field_from_voxel_exec('box(0,0,0,3,3,3,"copper")', block_m=0.1, resolution=2)
    assert field.voxel_size == pytest.approx(0.05) and rep["block_m"] == pytest.approx(0.05)
    assert field.mass_kg == pytest.approx(rep["resolution"]["mass_before_kg"], rel=1e-12)
    coarse, rep = field_from_voxel_exec('box(0,0,0,3,3,3,"copper")', block_m=0.1, resolution=0.5)
    assert coarse.voxel_size == pytest.approx(0.2) and rep["resolution"]["keep"] == "mass"
