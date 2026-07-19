"""Rotor -> gearset, wired together with REAL wind -- windmill drivetrain
roadmap, Arc A Phase 3. No rod/piston/pump yet (Phase 4).

Combines three already-gated pieces into one scene: the real aero rotor
(dynamics/mechanisms/wind.py's RotorWind, same construction as
record_windmill_spinup in radiance/trajectory.py -- duplicated here
deliberately rather than refactored, since a pytest-only spike isn't a
second real production consumer yet; Phase 5's final assembly is where
extraction becomes worth it), the Phase 1 BearingGearCoupling, and the
Phase 2 in-plane spur gearset. Proves two things under REAL aero load
(not Phase 1's synthetic constant torque):

  1. The rotor still reaches the standalone closed-form terminal_omega --
     confirms the load-blindness claim (KNOWN_GAPS.md) holds beyond a
     synthetic drive, not just in the isolated Phase 1 spike.
  2. The gearset's final ("crank-stub") arbor rate tracks the full
     cumulative ratio (BearingGearCoupling.ratio * the two mesh ratios).
"""
import math

import pytest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.parcel import PhysicsParcel
from sigma_ground.dynamics.scene import PhysicsScene
from sigma_ground.dynamics.stepper import step
from sigma_ground.dynamics.joints import RevoluteJoint, GearCouplingJoint
from sigma_ground.dynamics.mechanisms.wind import (RotorWind, build_rotor_blades,
                                                    terminal_omega)
from sigma_ground.dynamics.mechanisms.bearing import RigidBearing
from sigma_ground.dynamics.mechanisms.bearing_gear_coupling import BearingGearCoupling
from sigma_ground.kernel.gear import InvoluteGear
from sigma_ground.materia.engine import _material_density, _DensityMaterial

AXIS = Vec3(1.0, 0.0, 0.0)
MODULE_M = 0.002
PRESSURE_ANGLE = math.radians(20.0)
FACE_WIDTH_M = 0.006
TEETH_P0 = 12
TEETH_W1, TEETH_P1 = 60, 15
TEETH_W2 = 45
COUPLING_RATIO = 4.0                    # BearingGearCoupling's own step


def _gear(teeth):
    return InvoluteGear(module=MODULE_M, teeth=teeth,
                        pressure_angle=PRESSURE_ANGLE,
                        face_width=FACE_WIDTH_M, grid_resolution=36)


def _cd(na, nb):
    return MODULE_M * (na + nb) / 2.0


def _build_scene(wind_speed_m_s=10.0, n_blades=4, blade_length_m=0.5,
                 blade_chord_m=0.15, blade_thickness_m=0.004,
                 blade_pitch_deg=45.0, radius_centroid_m=0.6,
                 blade_material="aluminum"):
    # -- rotor: mirrors record_windmill_spinup's construction exactly --
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

    # -- gearset: same layout as Phase 2, offset away from the rotor/mast --
    dens_iron, _ = _material_density("iron", 288.15)
    g_p0, g_w1, g_p1, g_w2 = (_gear(TEETH_P0), _gear(TEETH_W1),
                              _gear(TEETH_P1), _gear(TEETH_W2))
    y0 = -0.6
    y1 = y0 + _cd(TEETH_P0, TEETH_W1)
    y2 = y1 + _cd(TEETH_P1, TEETH_W2)

    def _arbor(y, gears, label):
        m = sum(dens_iron * g.volume() for g in gears)
        iz = sum(dens_iron * g.volume() * g.inertia_factor("z") for g in gears)
        pos = Vec3(0.0, y, 0.0)
        return PhysicsParcel(0.01, _DensityMaterial(dens_iron), mass=m,
                             position=pos, inertia_body=(iz, iz, iz), label=label)

    arbor0 = _arbor(y0, [g_p0], "arbor0")
    arbor1 = _arbor(y1, [g_w1, g_p1], "arbor1")
    arbor2 = _arbor(y2, [g_w2], "arbor2")           # the future crank stub

    j0 = RevoluteJoint(arbor0, None, arbor0.position, AXIS)   # motor set by coupling
    j1 = RevoluteJoint(arbor1, None, arbor1.position, AXIS)
    j2 = RevoluteJoint(arbor2, None, arbor2.position, AXIS)
    ratio01 = TEETH_W1 / TEETH_P0
    ratio12 = TEETH_W2 / TEETH_P1
    c01 = GearCouplingJoint(j0, j1, ratio01)
    c12 = GearCouplingJoint(j1, j2, ratio12)

    coupling = BearingGearCoupling(bearing, j0, COUPLING_RATIO,
                                   motor_max_torque=50.0)

    scene = PhysicsScene([rotor, arbor0, arbor1, arbor2],
                         gravity=Vec3(0.0, -9.80665, 0.0), ground=False,
                         constraints=[j0, j1, j2, c01, c12])
    scene.solver_iterations = 20
    return dict(scene=scene, rotor=rotor, bearing=bearing, wind=wind,
               coupling=coupling, arbor0=arbor0, arbor2=arbor2,
               w_star=w_star, ratio01=ratio01, ratio12=ratio12)


def test_rotor_still_reaches_closed_form_terminal_omega_under_real_load():
    b = _build_scene()
    dt = 1.0 / 960.0
    n_steps = int(6.0 / dt)
    trace = []
    for i in range(n_steps):
        b["coupling"].step()
        step(b["scene"], dt=dt, sub_steps=1, external_forces=b["wind"])
        b["bearing"].project()
        if i % 40 == 0:
            trace.append(b["bearing"].omega())

    w_star = b["w_star"]
    assert len(trace) > 50
    assert all(w <= w_star * 1.01 for w in trace)          # never overshoots
    assert trace[-1] > 0.4 * w_star                          # really moving
    # this IS the measured load-blindness claim under real aero load, not
    # just Phase 1's synthetic torque: the bound above is identical to
    # test_trajectory_windmill.py's own unloaded-rotor gate, and it still
    # holds here with a loaded gearset attached.


def test_crank_stub_tracks_cumulative_ratio_under_real_wind():
    b = _build_scene()
    dt = 1.0 / 960.0
    n_steps = int(4.0 / dt)
    cumulative = COUPLING_RATIO / (b["ratio01"] * b["ratio12"])
    for i in range(n_steps):
        b["coupling"].step()
        step(b["scene"], dt=dt, sub_steps=1, external_forces=b["wind"])
        b["bearing"].project()
        if i % 400 == 0 and i > 0:
            expected = cumulative * b["bearing"].omega()
            assert b["arbor2"].angular_velocity.x == pytest.approx(
                expected, rel=0.02, abs=1e-3)
