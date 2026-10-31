"""Live cards (SKIPPYCRAFT_PLAN 10b, P1): every property a VoxelWorld cell uses is sigma-ground's value at THAT cell's
temperature, never a room-temperature card frozen for the run."""
import numpy as np
import pytest

from sigma_ground.deckard.voxel_exec import field_from_voxel_exec
from sigma_ground.field.interface import thermal
from sigma_ground.field.interface.block import block
from sigma_ground.materia.voxel_world import VoxelWorld


def _world(T_tungsten=1500.0):
    field, _ = field_from_voxel_exec('box(0,0,0,3,0,3,"granite"); box(1,1,1,2,1,2,"tungsten")', block_m=0.01)
    T = np.where(field.label_grid == field.materials.index("tungsten"), T_tungsten, 293.15)
    return VoxelWorld(field, temperature_k=T, env="ISM")


def test_hot_tungsten_radiates_with_its_hot_emissivity():
    w = _world(1500.0)
    eps = w._live("emissivity")[w.label == w.materials.index("tungsten")]
    assert eps == pytest.approx(thermal.emissivity("tungsten", 1500.0), rel=1e-3)     # 0.113, not the 0.022 at 293 K
    assert eps.min() > 4 * thermal.emissivity("tungsten", 293.15)


@pytest.mark.parametrize("key, prop", [("k", "thermal_conductivity"), ("cp", "specific_heat"), ("emissivity", "emissivity"),
                                       ("rho_solid", "density_solid")])
@pytest.mark.parametrize("T", [300.0, 950.0, 1500.0, 2600.0])
def test_every_live_value_is_block_at_the_cells_temperature(key, prop, T):
    """On the table's grid the live value IS block(material, T); between grid points, its linear interpolation."""
    w = _world(T)
    k = w.materials.index("tungsten")
    got = w._live(key)[w.label == k][0]
    want = block("tungsten", T)["properties"][prop]["value"]
    assert got == pytest.approx(want, rel=2e-3)


def test_the_card_follows_the_cell_as_it_cools():
    w = _world(2000.0)
    k = w.materials.index("tungsten")
    eps0 = w._live("emissivity")[w.label == k].mean()
    w.run(20, 1.0)                                         # radiating into space and into the granite
    T1 = w.T[w.label == k].mean()
    assert T1 < 2000.0
    assert w._live("emissivity")[w.label == k].mean() < eps0
    assert w._live("emissivity")[w.label == k].mean() == pytest.approx(thermal.emissivity("tungsten", T1), rel=0.05)


def test_energy_is_the_enthalpy_and_balances():
    w = _world(2500.0)
    E0 = w.energy_J()
    w.run(50, 0.5)
    assert w.energy_J() + w.ledger["radiated_J"] == pytest.approx(E0, rel=1e-9)


def test_enthalpy_round_trips_past_the_table_ends():
    w = _world(1500.0)
    for T in (50.0, 293.15, 1234.5, 3999.0, 8000.0):
        Tg = np.full(w.label.shape, T)
        back = w._T_from_H(w._H(Tg))
        assert back[w.label > 0] == pytest.approx(T, rel=1e-9)
