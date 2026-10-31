"""FLIP liquid (dynamics.fluid.flip) and the wave-machine voxel world (materia.voxel_fluid): momentum and forces add up."""
import math

import numpy as np
import pytest

from sigma_ground.dynamics.fluid import flip
from sigma_ground.dynamics.fluid.flip import Flip
from sigma_ground.field.interface.fluid_surface import STANDARD_GRAVITY, sloshing_angular_frequency

RHO, MU = 998.2, 1.002e-3


def pool(shape=(16, 8, 16), depth=6, h=0.02):
    liquid = np.zeros(shape, bool)
    liquid[:, :, :depth] = True
    return Flip(shape, h, density=RHO, viscosity=MU, gravity=STANDARD_GRAVITY, solid=np.zeros(shape, bool),
                liquid=liquid)


@pytest.fixture(params=["compiled", "numpy"])
def path(request, monkeypatch):
    if request.param == "numpy":
        monkeypatch.setattr(flip, "_fast", None)
    elif flip._fast is None:
        pytest.skip("numba not installed")
    return request.param


def test_still_water_stays_still_and_the_floor_carries_its_weight(path):
    f = pool()
    for _ in range(10):
        f.step(0.02)
    assert np.abs(f.vel).max() < 1e-4
    weight = f.mass_kg() * STANDARD_GRAVITY
    assert f.ledger["solid_force_N"][2] == pytest.approx(weight, rel=1e-4)
    # hydrostatic: the floor cell's pressure is ρ g × its depth below the first air cell's centre (p = 0 there)
    assert f.pressure[8, 4, 0] == pytest.approx(RHO * STANDARD_GRAVITY * 6 * 0.02, rel=1e-3)


def test_falling_blob_gains_exactly_m_g_t(path):
    shape = (16, 16, 32)
    liquid = np.zeros(shape, bool)
    liquid[6:10, 6:10, 20:24] = True
    f = Flip(shape, 0.02, density=RHO, viscosity=MU, gravity=STANDARD_GRAVITY, solid=np.zeros(shape, bool),
             liquid=liquid)
    for _ in range(10):
        f.step(0.02)
    p = f.momentum()
    assert p[2] == pytest.approx(-f.mass_kg() * STANDARD_GRAVITY * f.t, rel=1e-6)
    assert abs(p[0]) < 1e-9 and abs(p[1]) < 1e-9


def test_mass_is_exact_and_a_dam_break_runs_downstream():
    shape = (32, 4, 16)
    liquid = np.zeros(shape, bool)
    liquid[:8, :, :10] = True
    f = Flip(shape, 0.02, density=RHO, viscosity=MU, gravity=STANDARD_GRAVITY, solid=np.zeros(shape, bool),
             liquid=liquid)
    m0 = f.mass_kg()
    for _ in range(10):
        f.step(0.02)
    assert f.mass_kg() == m0
    assert f.momentum()[0] > 0                                       # the column collapses toward +x
    assert f.pos[:, 0].max() > 8 * 0.02                              # past where the dam stood


def test_sloshing_frequency_limits():
    g, L = STANDARD_GRAVITY, 1.0
    k = math.pi / L
    deep = sloshing_angular_frequency(L, 100.0)
    assert deep == pytest.approx(math.sqrt(g * k))
    shallow = sloshing_angular_frequency(L, 1e-3)
    assert shallow == pytest.approx(k * math.sqrt(g * 1e-3), rel=1e-5)


def test_wave_machine_puts_momentum_and_work_into_the_water():
    from sigma_ground.deckard.voxel_exec import field_from_voxel_exec
    from sigma_ground.materia.voxel_fluid import FluidVoxelWorld

    code = 'box(0,0,0,9,4,3,"glass"); box(1,1,1,8,4,2,"air"); box(2,1,1,2,4,2,"aluminum"); box(3,1,1,8,2,2,"water")'
    field, _ = field_from_voxel_exec(code, block_m=0.05, resolution=2)
    w = FluidVoxelWorld(field, paddle={"material": "aluminum", "axis": 0, "amplitude_m": 0.02, "period_s": None})
    m0 = w.mass_kg()
    assert w.paddle["period_s"] == pytest.approx(w.paddle["resonant_s"])
    for _ in range(8):
        w.step(0.04)
    assert w.mass_kg() == m0
    assert w.ledger["paddle_work_J"] > 0                             # the first push forward does work on the water
    assert abs(w.ledger["momentum_Ns"][0]) > 1e-3
    lab = w.label
    assert (lab == w.materials.index("water")).sum() > 0
    assert (lab == w.materials.index("aluminum")).sum() == (field.label_grid == field.materials.index("aluminum")).sum()
    assert any("surface tension" in g for g in w.gaps())
