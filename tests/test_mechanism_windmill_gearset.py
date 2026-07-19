"""In-plane spur gearset on the rotor's own spin axis -- windmill drivetrain
roadmap, Arc A Phase 2.

Standalone: 3 arbors / 2 mesh stages, reusing record_clock's _gear()/_cd()/
mass-from-InvoluteGear pattern near-verbatim (sigma_ground/radiance/
trajectory.py's record_clock), but with axis=(1,0,0) -- matching the
windmill rotor's fixed +x spin axis (dynamics/mechanisms/wind.py) -- and
arbors offset along Y (off-axis) instead of along the shared spin axis.
This is a relabeling of the clock's already-proven layout, not new physics:
GearCouplingJoint enforces the SAME kinematic rate constraint regardless of
which world axis the shared hinge direction points along.

Tooth counts here are ARBITRARY GEOMETRY CHOICES for this synthetic test
gearset (no cited blueprint source, unlike the clock's Kelly-1944 data) --
flagged [estimated] wherever this gearset eventually reaches a rendered
scene (Phase 3+), same convention as every InvoluteGear-based demo.
"""
import math

import pytest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.parcel import PhysicsParcel
from sigma_ground.dynamics.scene import PhysicsScene
from sigma_ground.dynamics.stepper import step
from sigma_ground.dynamics.joints import RevoluteJoint, GearCouplingJoint
from sigma_ground.kernel.gear import InvoluteGear
from sigma_ground.materia.engine import _material_density, _DensityMaterial


AXIS = Vec3(1.0, 0.0, 0.0)
MODULE_M = 0.002
PRESSURE_ANGLE = math.radians(20.0)
FACE_WIDTH_M = 0.006

# arbor0: driven pinion only. arbor1: wheel (meshes arbor0) + its own pinion
# (meshes arbor2). arbor2: wheel only (output).
TEETH_P0 = 12
TEETH_W1, TEETH_P1 = 60, 15
TEETH_W2 = 45


def _gear(teeth):
    return InvoluteGear(module=MODULE_M, teeth=teeth,
                        pressure_angle=PRESSURE_ANGLE,
                        face_width=FACE_WIDTH_M, grid_resolution=36)


def _cd(na, nb):
    return MODULE_M * (na + nb) / 2.0


def _build_gearset(motor_speed=-2.0, motor_max_torque=5.0):
    dens_iron, _ = _material_density("iron", 288.15)

    g_p0 = _gear(TEETH_P0)
    g_w1 = _gear(TEETH_W1)
    g_p1 = _gear(TEETH_P1)
    g_w2 = _gear(TEETH_W2)

    y0 = 0.0
    y1 = y0 + _cd(TEETH_P0, TEETH_W1)
    y2 = y1 + _cd(TEETH_P1, TEETH_W2)

    def _arbor(y, gears, label):
        mass = sum(dens_iron * g.volume() for g in gears)
        iz = sum(dens_iron * g.volume() * g.inertia_factor("z") for g in gears)
        pos = Vec3(0.0, y, 0.0)
        p = PhysicsParcel(0.01, _DensityMaterial(dens_iron), mass=mass,
                          position=pos, inertia_body=(iz, iz, iz), label=label)
        return p

    arbor0 = _arbor(y0, [g_p0], "arbor0")
    arbor1 = _arbor(y1, [g_w1, g_p1], "arbor1")
    arbor2 = _arbor(y2, [g_w2], "arbor2")

    j0 = RevoluteJoint(arbor0, None, arbor0.position, AXIS,
                       motor_speed=motor_speed, motor_max_torque=motor_max_torque)
    j1 = RevoluteJoint(arbor1, None, arbor1.position, AXIS)
    j2 = RevoluteJoint(arbor2, None, arbor2.position, AXIS)

    ratio01 = TEETH_W1 / TEETH_P0
    ratio12 = TEETH_W2 / TEETH_P1
    c01 = GearCouplingJoint(j0, j1, ratio01)
    c12 = GearCouplingJoint(j1, j2, ratio12)

    scene = PhysicsScene([arbor0, arbor1, arbor2], ground=False,
                         constraints=[j0, j1, j2, c01, c12])
    # arbor1 carries TWO coupling rows simultaneously (c01 and c12) -- more
    # Gauss-Seidel sweeps than the 10-iteration default to stay crisp, same
    # reasoning as record_clock's 5-deep chain (trajectory.py)
    scene.solver_iterations = 20
    return scene, dict(arbor0=arbor0, arbor1=arbor1, arbor2=arbor2,
                       ratio01=ratio01, ratio12=ratio12)


def test_ratio_holds_through_the_two_stage_chain():
    scene, b = _build_gearset()
    dt = 1.0 / 960.0
    for k in range(int(2.0 / dt)):
        step(scene, dt=dt, sub_steps=1)
        if k % 200 == 0:
            assert b["arbor1"].angular_velocity.x == pytest.approx(
                -b["arbor0"].angular_velocity.x / b["ratio01"], rel=1e-3)
            assert b["arbor2"].angular_velocity.x == pytest.approx(
                -b["arbor1"].angular_velocity.x / b["ratio12"], rel=1e-3)

    # overall cumulative ratio: two meshes compound multiplicatively (same
    # sign flips each mesh -- a spur train alternates rotation direction
    # stage to stage, exactly like the clock's cited going train)
    cumulative = b["ratio01"] * b["ratio12"]
    assert b["arbor2"].angular_velocity.x == pytest.approx(
        b["arbor0"].angular_velocity.x / cumulative, rel=1e-3)


def test_teeth_flagged_estimated_when_rendered():
    """Not-yet-rendered here (Phase 2 is pytest-only) -- this just pins the
    tooth counts as the values Phase 3+ must reuse and flag [estimated],
    same convention as every InvoluteGear-based demo in trajectory.py."""
    teeth = (TEETH_P0, TEETH_W1, TEETH_P1, TEETH_W2)
    assert all(isinstance(t, int) and t > 0 for t in teeth)
