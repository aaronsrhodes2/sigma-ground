"""Rigid voxel bodies (SKIPPYCRAFT_PLAN 10b, P3): contact, friction, toppling, strength at temperature.

Done when: a stack topples; a heated beam sags and breaks where hot. Every check here is against a closed-form answer:
rest height, the sliding distance v^2 / (2 mu g), the tipping angle atan(width / height), the plastic moment of a section,
and a loaded beam that holds cold and hinges, sags and tears hot."""
import math

import numpy as np
import pytest

from sigma_ground.dynamics.voxel_rigid import (RigidWorld, VoxelBody, _plastic_moment, _subfaces, quat_rotvec,
                                              quat_to_R, strength_table)

G = 9.80665
H = 0.01


def _floor(shape):
    static = np.zeros(shape, bool)
    static[:, :, 0:2] = True                                       # floor top at z = 2 cm
    return static


def _block(lo, size, material="granite", rho=2700.0, T=293.15):
    cells = np.argwhere(np.ones(size, bool)) + lo
    return VoxelBody(material, 1, cells, H, np.full(len(cells), rho * H ** 3), np.full(len(cells), T))


def _world(shape, static):
    return RigidWorld(shape, H, static, lambda c: "granite", G)


def test_a_dropped_block_comes_to_rest_on_the_floor_and_its_energy_is_heat():
    w = _world((30, 30, 30), _floor((30, 30, 30)))
    b = _block((10, 10, 5), (4, 4, 4))
    w.bodies.append(b)
    E0 = w.mechanical_J()
    w.step(1.0)
    assert b.x[2] == pytest.approx(0.04, abs=2e-4)               # 2 cm floor + half the 4 cm block
    assert np.linalg.norm(b.v) < 1e-6 and np.linalg.norm(b.w) < 1e-6
    lost = E0 - w.mechanical_J()
    L = w.ledger
    assert L["contact_heat_J"] + L["integration_J"] == pytest.approx(lost, rel=1e-9)
    assert L["contact_heat_J"] == pytest.approx(lost, rel=0.1)   # nearly all of it went into the contact as heat
    heat = sum(float(np.sum(J)) for _, _, J in w.collect_heat())
    assert heat == pytest.approx(L["contact_heat_J"], rel=1e-12)


def test_a_sliding_block_stops_where_kinetic_friction_says():
    """Granite on granite: Byerlee's 0.85 (measured; no separate sliding value)."""
    w = _world((60, 30, 20), _floor((60, 30, 20)))
    b = _block((10, 10, 2), (4, 4, 4))
    w.bodies.append(b)
    w.step(0.05)
    b.v = np.array([0.5, 0.0, 0.0])
    x0 = b.x[0]
    w.step(0.5)
    assert np.linalg.norm(b.v) < 1e-6
    assert b.x[0] - x0 == pytest.approx(0.5 ** 2 / (2 * 0.85 * G), rel=0.05)
    assert any("no measured sliding value" in f for f in w.gaps())


@pytest.mark.parametrize("deg,falls", [(14, False), (22, True)])
def test_a_tilted_slab_falls_past_its_tipping_angle(deg, falls):
    """A 4 x 4 x 12 cm slab tips over past atan(4/12) = 18.4 degrees and rights itself below it."""
    w = _world((40, 40, 30), _floor((40, 40, 30)))
    b = _block((18, 18, 2), (4, 4, 12))
    b.q = quat_rotvec(np.array([math.radians(deg), 0.0, 0.0]))
    b.x = b.x + np.array([0.0, 0.0, 0.02 - b.world(b.shell)[:, 2].min() + 1e-4])
    w.bodies.append(b)
    w.step(1.5)
    tilt = math.degrees(math.acos(min(1.0, quat_to_R(b.q)[2, 2])))
    assert (tilt > 80) == falls and (tilt < 1 or falls)


def test_a_leaning_stack_topples_and_a_straight_one_stands():
    def run(step):
        w = _world((60, 30, 40), _floor((60, 30, 40)))
        for k in range(4):
            w.bodies.append(_block((20 + step * k, 12, 2 + 4 * k), (4, 4, 4)))
        top0 = w.bodies[-1].x.copy()
        w.step(1.0)
        return np.linalg.norm(w.bodies[-1].x - top0)
    assert run(0) < 1e-3                                         # a straight stack stays put (settles < 0.1 cell)
    assert run(2) > 0.03                                          # each block 2 cm out: its top falls


def test_the_plastic_moment_of_a_section_is_exact():
    sy = 250e6
    one = _plastic_moment(*_subfaces(np.zeros(1), np.array([sy * H * H]), H), 0.0)
    assert one == pytest.approx(sy * H ** 3 / 4, rel=1e-12)       # b h^2 / 4 for a single cell
    two = _plastic_moment(*_subfaces(np.array([-0.5, 0.5]) * H, np.full(2, sy * H * H), H), 0.0)
    assert two == pytest.approx(sy * H * (2 * H) ** 2 / 4, rel=1e-12)


def test_hot_steel_yields_by_the_eurocode_table():
    st = strength_table("steel")
    assert st["ductile"] and "EN 1993-1-2" in st["source"]
    y = lambda T: float(np.interp(T, st["T"], st["yield"]))
    assert y(293.15) == pytest.approx(250e6, rel=1e-6)
    assert y(773.15) == pytest.approx(0.78 * 250e6, rel=5e-3)    # 500 C (the live table is every 25 K)
    assert y(1173.15) == pytest.approx(0.06 * 250e6, rel=5e-2)   # 900 C
    assert y(1600.0) == 0.0


def _beam_world(T_beam):
    """A 60 x 2 x 1 cm steel beam on granite piers 50 cm apart, a 2.8 kg tungsten block on its middle."""
    shape = (70, 30, 50)
    static = _floor(shape)
    static[3:8, 12:18, 2:20] = True
    static[57:62, 12:18, 2:20] = True
    w = _world(shape, static)
    beam = _block((2, 14, 20), (60, 2, 1), "steel", 7850.0, T_beam)
    load = _block((29, 14, 21), (6, 2, 12), "tungsten", 19300.0)
    w.bodies += [beam, load]
    return w, load


def test_a_loaded_beam_holds_when_cold():
    w, load = _beam_world(900.0)                                  # 627 C: still 0.52 of its yield
    z0 = load.x[2]
    for _ in range(25):
        w.step(0.02)
        w.check_strength()
    assert w.ledger["hinges"] == 0 and load.x[2] == pytest.approx(z0, abs=5e-4)


def test_a_hot_beam_hinges_sags_and_tears():
    """At 1400 K steel keeps 1.5% of its yield: under the block the plastic moment (1.8 N m) is less than the load's
    (3.4 N m), so a hinge forms; it bends, its work becomes heat, and it tears past the steel's elongation."""
    w, load = _beam_world(1400.0)
    z0 = load.x[2]
    bends = []
    for _ in range(100):
        w.step(0.02)
        w.check_strength()
        bends += [j["bend"] for j in w.joints]
        if w.ledger["tears"]:
            break
    L = w.ledger
    assert L["hinges"] >= 1 and L["tears"] >= 1
    assert max(bends) > math.radians(5)                          # it sagged before it tore
    assert L["plastic_heat_J"] > 0
    assert load.x[2] < z0 - 0.02


def test_the_liquid_torque_matches_its_force():
    """FLIP's face forces: force_torque_on's force is force_on's; torque about the point of application is zero."""
    from sigma_ground.dynamics.fluid.flip import Flip
    shape = (8, 8, 10)
    solid = np.zeros(shape, bool)
    solid[3:5, 3:5, 2:4] = True
    liquid = np.zeros(shape, bool)
    liquid[:, :, :8] = True
    liquid &= ~solid
    f = Flip(shape, H, density=1000.0, viscosity=0.0, gravity=G, solid=solid, liquid=liquid)
    f.step(0.01)
    F1 = f.force_on(solid)
    F2, T = f.force_torque_on(solid, np.array([4, 4, 3]) * H)
    assert np.allclose(F1, F2)
    assert F1[2] > 0                                              # buoyancy
    assert abs(T[0]) < 1e-3 * abs(F1[2]) * H and abs(T[1]) < 1e-3 * abs(F1[2]) * H   # symmetric block: no twist
