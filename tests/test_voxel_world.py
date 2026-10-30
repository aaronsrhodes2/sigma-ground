"""Tests for materia.voxel_world -- a material voxel build that runs: heat, phase, settling."""
import shutil

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")

from sigma_ground.deckard.voxel_exec import field_from_voxel_exec  # noqa: E402
from sigma_ground.field.interface.phase_transition import PHASE_DATA  # noqa: E402
from sigma_ground.materia.voxel_world import LIQUID, SOLID, VoxelWorld, material_card  # noqa: E402

TANK = ('box(0,0,0,7,0,7,"granite"); box(1,1,1,6,5,6,"glass"); box(2,1,2,5,5,5,"air");'
        'box(2,1,2,5,2,5,"mercury"); block(3,3,3,"iron"); block(4,5,4,"gold")')
LEAD_ON_PLATE = 'box(0,0,0,3,0,3,"tungsten"); box(1,1,1,2,2,2,"lead")'


def _z(world, name):
    return sorted(set(np.argwhere(world.label == world.materials.index(name))[:, 2].tolist()))


def _hot_plate(plate_T, lead_T=300.0):
    field, _ = field_from_voxel_exec(LEAD_ON_PLATE, block_m=0.01)
    T = np.where(field.label_grid == field.materials.index("tungsten"), plate_T, lead_T)
    return VoxelWorld(field, temperature_k=T, env="ISM")


def test_denser_sinks_lighter_floats_and_mass_is_kept():
    field, _ = field_from_voxel_exec(TANK, block_m=0.05)
    w = VoxelWorld(field)
    m0, hg_floor = w.mass_kg(), _z(w, "mercury")[0]
    w.run(8, 1.0)
    assert _z(w, "gold") == [hg_floor]                   # gold (19300) sinks to the bottom of the mercury
    assert _z(w, "iron")[0] > hg_floor                   # iron (7870) stays on it
    assert w.mass_kg() == pytest.approx(m0, rel=1e-12)


def test_the_build_rests_on_the_ground():
    field, _ = field_from_voxel_exec(TANK, block_m=0.05)
    w = VoxelWorld(field)
    floor = _z(w, "granite")
    w.run(3, 1.0)
    assert _z(w, "granite") == floor


def test_melting_takes_latent_heat_and_energy_balances_in_vacuum():
    w = _hot_plate(1500.0)
    E0 = w.energy_J()
    lead = w.materials.index("lead")
    saw_partial = False
    for _ in range(60):
        w.step(0.05)
        f = w.melt[w.label == lead]
        saw_partial |= bool(((f > 0) & (f < 1)).any())
        # every partly melted lead cell sits exactly at the melting point
        part = (w.label == lead) & (w.melt > 0) & (w.melt < 1)
        assert np.allclose(w.T[part], PHASE_DATA["lead"]["T_melt_K"])
    assert saw_partial
    assert (w.phase()[w.label == lead] == LIQUID).any()
    assert w.energy_J() + w.ledger["radiated_J"] == pytest.approx(E0, rel=1e-9)


def test_cold_liquid_freezes_and_gives_the_heat_back():
    w = _hot_plate(250.0, lead_T=700.0)                  # lead starts molten on a cold plate
    lead = w.materials.index("lead")
    assert (w.phase()[w.label == lead] == LIQUID).all()
    E0 = w.energy_J()
    w.run(200, 0.05)
    assert (w.phase()[w.label == lead] == SOLID).all()
    assert w.energy_J() + w.ledger["radiated_J"] == pytest.approx(E0, rel=1e-9)


def test_boiled_cells_vent_with_their_mass_and_energy():
    w = _hot_plate(250.0, lead_T=3.0e5)                  # far past lead's boiling point
    m0, E0 = w.mass_kg(), w.energy_J()
    w.step(1e-4)
    assert "lead" not in [w.materials[k] for k in np.unique(w.label) if k]
    assert w.mass_kg() + w.ledger["vented_kg"] == pytest.approx(m0, rel=1e-12)
    assert w.energy_J() + w.ledger["radiated_J"] + w.ledger["vented_J"] == pytest.approx(E0, rel=1e-9)


def test_liquid_falls_and_spreads():
    field, _ = field_from_voxel_exec('box(0,0,0,6,0,6,"granite"); box(3,1,3,3,4,3,"mercury")', block_m=0.05)
    w = VoxelWorld(field)
    w.run(10, 1.0)
    hg = np.argwhere(w.label == w.materials.index("mercury"))
    assert len(hg) == 4 and set(hg[:, 2]) == {_z(w, "granite")[0] + 1}   # one layer on the floor


def test_gaps_are_named_not_filled():
    hg = material_card("mercury", 13534.0)
    assert hg["rho_cp"] == 0 and any("no thermal data" in g for g in hg["gaps"])
    ice = material_card("water_ice", 917.0)
    assert ice["L_fus"] is None and any("mole basis" in g for g in ice["gaps"])
    lead = material_card("lead", 11340.0)
    assert lead["gaps"] == [] and lead["L_fus"] > 0


def test_to_field_carries_temperature():
    w = _hot_plate(1500.0)
    w.step(0.05)
    f = w.to_field()
    assert f.scalar_fields["temperature_k"].shape == f.label_grid.shape
    assert f.mass_kg == pytest.approx(w.mass_kg())


def test_extent_sets_the_block_size():
    field, report = field_from_voxel_exec('block(0,0,0,"iron")', extent_m=12.8, grid_size=128)
    assert report["block_m"] == pytest.approx(0.1)
    assert field.voxel_size == pytest.approx(0.1)
