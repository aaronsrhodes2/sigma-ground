"""Full windmill drivetrain capstone -- Arc A Phase 4: rotor -> gearset ->
slider-crank -> reciprocating pump, all driven by real wind, nothing
prescribed. This is where Phase 0 (slider-crank kinematics), Phase 1
(bearing->gearset coupling), Phase 2 (spur gearset), and Phase 3 (real-wind
wiring) all compose into one scene.

The gearset's final arbor (previously a "crank stub" with zero lever arm
in Phase 2/3) now IS the crank: its own CoM is reconstructed offset by
r_crank from its RevoluteJoint anchor -- the same body, same gear-derived
mass/inertia, just carrying a crank arm. Everything downstream (rod,
piston, PrismaticJoint) is Phase 0's exact linkage, relabeled onto the
shared hinge axis +x (AXIS = the whole drivetrain's spin axis) with the
piston sliding along +y instead of Phase 0's +x/+z choice -- same relative
geometry, rotated, so the closed form s(theta) = r*cos(theta) +
sqrt(l^2 - r^2*sin(theta)^2) applies unchanged (now against
piston.position.y - pivot.y).
"""
import math

import pytest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.parcel import PhysicsParcel
from sigma_ground.dynamics.scene import PhysicsScene
from sigma_ground.dynamics.stepper import step
from sigma_ground.dynamics.joints import RevoluteJoint, GearCouplingJoint, PrismaticJoint
from sigma_ground.dynamics.quat import twist_angle
from sigma_ground.dynamics.mechanisms.wind import (RotorWind, build_rotor_blades,
                                                    terminal_omega)
from sigma_ground.dynamics.mechanisms.bearing import RigidBearing
from sigma_ground.dynamics.mechanisms.bearing_gear_coupling import BearingGearCoupling
from sigma_ground.dynamics.mechanisms.pump import ReciprocatingPumpState
from sigma_ground.kernel.gear import InvoluteGear
from sigma_ground.materia.engine import _material_density, _DensityMaterial


class _Mat:
    density_kg_m3 = 1000.0
    restitution = 0.5

    def density_at_sigma(self, s):
        return 1000.0


AXIS = Vec3(1.0, 0.0, 0.0)
SLIDE_AXIS = Vec3(0.0, 1.0, 0.0)
MODULE_M = 0.002
PRESSURE_ANGLE = math.radians(20.0)
FACE_WIDTH_M = 0.006
TEETH_P0 = 12
TEETH_W1, TEETH_P1 = 60, 15
TEETH_W2 = 45
COUPLING_RATIO = 4.0


def _gear(teeth):
    return InvoluteGear(module=MODULE_M, teeth=teeth,
                        pressure_angle=PRESSURE_ANGLE,
                        face_width=FACE_WIDTH_M, grid_resolution=36)


def _cd(na, nb):
    return MODULE_M * (na + nb) / 2.0


def _closed_form_s(theta, r, l):
    return r * math.cos(theta) + math.sqrt(l * l - (r * math.sin(theta)) ** 2)


def _build_scene(r_crank=0.03, l_rod=0.09, piston_radius_m=0.01,
                 wind_speed_m_s=10.0, n_blades=4, blade_length_m=0.5,
                 blade_chord_m=0.15, blade_thickness_m=0.004,
                 blade_pitch_deg=45.0, radius_centroid_m=0.6,
                 blade_material="aluminum"):
    # -- rotor: mirrors record_windmill_spinup's construction --
    beta = math.radians(blade_pitch_deg)
    dens_blade, _ = _material_density(blade_material, 288.15)
    dens_hub, _ = _material_density("iron", 288.15)
    blades, area = build_rotor_blades(n_blades, radius_centroid_m,
                                      blade_length_m, blade_chord_m, beta)
    m_blade = dens_blade * blade_length_m * blade_chord_m * blade_thickness_m
    t2, L2, c2 = blade_thickness_m ** 2, blade_length_m ** 2, blade_chord_m ** 2
    I_own_x = m_blade * (L2 + c2) / 12.0
    I_own_z = m_blade * (t2 + L2) / 12.0
    I_own_pitched = (math.cos(beta) ** 2 * I_own_x
                     + math.sin(beta) ** 2 * I_own_z)
    hub_r, hub_h = 0.08, 0.1
    m_hub = dens_hub * math.pi * hub_r ** 2 * hub_h
    I_axis = (n_blades * (m_blade * radius_centroid_m ** 2 + I_own_pitched)
              + 0.5 * m_hub * hub_r ** 2)
    mass = n_blades * m_blade + m_hub
    hub_pos = Vec3(0.0, 0.9, 0.0)
    rotor = PhysicsParcel(0.05, _DensityMaterial(dens_blade), mass=mass,
                          position=hub_pos,
                          inertia_body=(I_axis, I_axis, I_axis), label="rotor")
    bearing = RigidBearing(rotor, hub_pos, AXIS)
    wind = RotorWind(rotor, blades, Vec3(wind_speed_m_s, 0.0, 0.0))
    w_star = terminal_omega(wind_speed_m_s, beta, radius_centroid_m)

    # -- gearset (Phase 2/3 layout) --
    dens_iron, _ = _material_density("iron", 288.15)
    g_p0, g_w1, g_p1, g_w2 = (_gear(TEETH_P0), _gear(TEETH_W1),
                              _gear(TEETH_P1), _gear(TEETH_W2))
    y0 = -0.6
    y1 = y0 + _cd(TEETH_P0, TEETH_W1)
    y2 = y1 + _cd(TEETH_P1, TEETH_W2)
    P = Vec3(0.0, y2, 0.0)                      # crank pivot: arbor2's own anchor

    def _arbor(y, gears, label):
        m = sum(dens_iron * g.volume() for g in gears)
        iz = sum(dens_iron * g.volume() * g.inertia_factor("z") for g in gears)
        pos = Vec3(0.0, y, 0.0)
        return PhysicsParcel(0.01, _DensityMaterial(dens_iron), mass=m,
                             position=pos, inertia_body=(iz, iz, iz), label=label)

    arbor0 = _arbor(y0, [g_p0], "arbor0")
    arbor1 = _arbor(y1, [g_w1, g_p1], "arbor1")

    # -- arbor2 IS the crank now: same gear-derived mass/inertia, CoM offset
    # by r_crank from its own joint anchor P (Phase 2/3 had zero offset) --
    crank_tip0 = P + Vec3(0.0, r_crank, 0.0)
    m_w2 = dens_iron * g_w2.volume()
    iz_w2 = dens_iron * g_w2.volume() * g_w2.inertia_factor("z")
    arbor2 = PhysicsParcel(0.01, _DensityMaterial(dens_iron), mass=m_w2,
                           position=crank_tip0,
                           inertia_body=(iz_w2, iz_w2, iz_w2), label="crank")

    j0 = RevoluteJoint(arbor0, None, arbor0.position, AXIS)
    j1 = RevoluteJoint(arbor1, None, arbor1.position, AXIS)
    j2 = RevoluteJoint(arbor2, None, P, AXIS)
    ratio01 = TEETH_W1 / TEETH_P0
    ratio12 = TEETH_W2 / TEETH_P1
    c01 = GearCouplingJoint(j0, j1, ratio01)
    c12 = GearCouplingJoint(j1, j2, ratio12)
    coupling = BearingGearCoupling(bearing, j0, COUPLING_RATIO,
                                   motor_max_torque=50.0)

    # -- Phase 0's linkage, relabeled onto AXIS=+x / SLIDE_AXIS=+y --
    piston0 = P + Vec3(0.0, r_crank + l_rod, 0.0)
    rod_mid0 = P + Vec3(0.0, r_crank + 0.5 * l_rod, 0.0)
    rod = PhysicsParcel(0.005, _Mat(), position=rod_mid0, mass=0.01, label="rod")
    piston = PhysicsParcel(0.005, _Mat(), position=piston0, mass=0.02,
                           label="piston")
    j_rod_crank = RevoluteJoint(rod, arbor2, crank_tip0, AXIS)
    j_rod_piston = RevoluteJoint(rod, piston, piston0, AXIS)
    j_slide = PrismaticJoint(piston, None, piston0, SLIDE_AXIS)

    all_parcels = [rotor, arbor0, arbor1, arbor2, rod, piston]
    all_constraints = [j0, j1, j2, c01, c12, j_rod_crank, j_rod_piston, j_slide]
    scene = PhysicsScene(all_parcels, gravity=Vec3(0.0, -9.80665, 0.0),
                         ground=False, constraints=all_constraints)
    scene.solver_iterations = 20

    piston_area = math.pi * piston_radius_m ** 2
    stroke_length = 2.0 * r_crank                # s(0)-s(pi) = (r+l)-(l-r) = 2r
    pump = ReciprocatingPumpState(j_slide, piston_area, stroke_length)

    return dict(scene=scene, bearing=bearing, wind=wind, coupling=coupling,
               arbor2=arbor2, piston=piston, j_slide=j_slide, pump=pump,
               r_crank=r_crank, l_rod=l_rod, P=P, w_star=w_star,
               piston_area=piston_area, stroke_length=stroke_length)


def _unwrap(raw):
    out = [raw[0]]
    offset = 0.0
    prev = raw[0]
    for a in raw[1:]:
        d = a - prev
        if d > math.pi:
            offset -= 2.0 * math.pi
        elif d < -math.pi:
            offset += 2.0 * math.pi
        prev = a
        out.append(a + offset)
    return out


def test_capstone_piston_and_pump():
    b = _build_scene()
    dt = 1.0 / 960.0
    t_max = 10.0
    n_steps = int(t_max / dt)

    raw_thetas, piston_ys, piston_zs = [], [], []
    for i in range(n_steps):
        b["coupling"].step()
        step(b["scene"], dt=dt, sub_steps=1, external_forces=b["wind"])
        b["bearing"].project()
        b["pump"].step()
        if i % 20 == 0:
            raw_thetas.append(twist_angle(b["arbor2"].orientation, (1.0, 0.0, 0.0)))
            piston_ys.append(b["piston"].position.y)
            piston_zs.append(b["piston"].position.z)

    thetas = _unwrap(raw_thetas)
    r, l, P = b["r_crank"], b["l_rod"], b["P"]

    # piston motion matches the closed form at EVERY sample, against the
    # ACTUAL spin-up-then-terminal theta(t) -- not a constant assumed rate
    for th, y in zip(thetas, piston_ys):
        expected = P.y + _closed_form_s(th, r, l)
        assert y == pytest.approx(expected, abs=2e-4)

    # PrismaticJoint's perpendicular lock: piston never leaves the slide axis
    for z in piston_zs:
        assert abs(z - P.z) < 2e-4

    # the crank actually turned a meaningful amount (not stalled)
    total_revolutions = (thetas[-1] - thetas[0]) / (2.0 * math.pi)
    assert total_revolutions > 1.0

    # reservoir volume jumps discretely once per detected reversal (piston
    # starts AT a maximum, theta=0, so the first counted reversal lands
    # after ~one full revolution, then once per revolution thereafter)
    assert b["pump"].strokes >= 1
    assert abs(b["pump"].strokes - math.floor(total_revolutions)) <= 1
    expected_volume = b["pump"].strokes * b["piston_area"] * b["stroke_length"]
    assert b["pump"].volume_m3 == pytest.approx(expected_volume, rel=1e-9)
