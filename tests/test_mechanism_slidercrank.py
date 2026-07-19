"""Slider-crank kinematics — windmill drivetrain roadmap, Arc A Phase 0.

Standalone spike: crank -> connecting rod -> piston, driven only by a
strong-torque crank motor. No wind/bearing/gearset involved yet — this
proves the RevoluteJoint -> RevoluteJoint -> (RevoluteJoint + PrismaticJoint)
COMPOSITION is solver-stable and matches the classical slider-crank
displacement closed form

    s(theta) = r*cos(theta) + sqrt(l**2 - r**2*sin(theta)**2)

`PrismaticJoint` (dynamics/joints.py) has zero production/chain usage
anywhere else in the repo (only one isolated incline test) -- this is
genuinely untested joint-composition territory, the same risk class as the
RevoluteJoint-under-external-torque gap already in KNOWN_GAPS.md. If this
proves unstable, the documented fallback is an ideal-projection
RigidSliderCrank (a la dynamics/mechanisms/bearing.py's RigidBearing)
computed directly from this same closed form.

Layout (all three hinges share ONE axis, +z, so the whole linkage stays in
the XY plane -- a DOF well-posedness requirement, not a detail): crank
pivots at the origin, piston slides along world +x THROUGH the origin (the
"in-line"/centered slider-crank the closed form above assumes), crank tip
starts at theta=0 (fully extended along +x, collinear with rod and slide
axis -- momentarily degenerate but not solver-singular).
"""
import math

import pytest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.parcel import PhysicsParcel
from sigma_ground.dynamics.scene import PhysicsScene
from sigma_ground.dynamics.stepper import step
from sigma_ground.dynamics.joints import RevoluteJoint, PrismaticJoint
from sigma_ground.dynamics.quat import twist_angle


class _Mat:
    density_kg_m3 = 1000.0
    restitution = 0.5

    def density_at_sigma(self, s):
        return 1000.0


AXIS = Vec3(0.0, 0.0, 1.0)
SLIDE_AXIS = Vec3(1.0, 0.0, 0.0)


def _closed_form_s(theta, r, l):
    return r * math.cos(theta) + math.sqrt(l * l - (r * math.sin(theta)) ** 2)


def _build_scene(r=0.05, l=0.15, omega=4.0, torque_cap=5.0):
    pivot = Vec3(0.0, 0.0, 0.0)
    crank_tip0 = pivot + Vec3(r, 0.0, 0.0)           # theta=0: extended +x
    piston0 = Vec3(_closed_form_s(0.0, r, l), 0.0, 0.0)
    rod_mid0 = Vec3(0.5 * (crank_tip0.x + piston0.x), 0.0, 0.0)

    crank = PhysicsParcel(0.01, _Mat(), position=crank_tip0, mass=0.02,
                          label="crank")
    rod = PhysicsParcel(0.01, _Mat(), position=rod_mid0, mass=0.03,
                        label="rod")
    piston = PhysicsParcel(0.01, _Mat(), position=piston0, mass=0.05,
                           label="piston")

    # sign convention (established this session, record_motor_spin): with
    # b=None the world plays the joint's "child", so motor_speed=-omega
    # drives the BODY's own world-frame spin to +omega about AXIS.
    j_crank = RevoluteJoint(crank, None, pivot, AXIS,
                            motor_speed=-omega, motor_max_torque=torque_cap)
    j_rod_crank = RevoluteJoint(rod, crank, crank_tip0, AXIS)
    j_rod_piston = RevoluteJoint(rod, piston, piston0, AXIS)
    j_slide = PrismaticJoint(piston, None, piston0, SLIDE_AXIS)

    scene = PhysicsScene([crank, rod, piston], ground=False,
                         constraints=[j_crank, j_rod_crank, j_rod_piston,
                                      j_slide])
    scene.solver_iterations = 20
    return scene, dict(crank=crank, rod=rod, piston=piston, r=r, l=l,
                       omega=omega)


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


def test_piston_matches_closed_form_over_two_revolutions():
    scene, b = _build_scene()
    r, l, omega = b["r"], b["l"], b["omega"]
    crank, piston = b["crank"], b["piston"]

    dt = 1.0 / 960.0
    t_total = 2.2 * (2.0 * math.pi / omega)           # >= 2 full revolutions
    n_steps = int(t_total / dt)

    raw_thetas = []
    piston_xs, piston_yz = [], []
    for i in range(n_steps):
        step(scene, dt=dt, sub_steps=1)
        if i % 8 == 0:
            raw_thetas.append(twist_angle(crank.orientation, (0.0, 0.0, 1.0)))
            piston_xs.append(piston.position.x)
            piston_yz.append((piston.position.y, piston.position.z))

    thetas = _unwrap(raw_thetas)

    # >= 2 full revolutions actually elapsed (no stall, no wrap-seam bug)
    assert thetas[-1] - thetas[0] >= 2.0 * (2.0 * math.pi) - 0.1
    # monotone: a strong-torque-capped motor should never reverse the crank
    assert all(b_ >= a_ - 1e-6 for a_, b_ in zip(thetas, thetas[1:]))

    # piston position matches the classical closed form at EVERY sample --
    # checking every step (not just the final one) means any drift/
    # instability shows up as a hard failure, not silent divergence
    for th, x in zip(thetas, piston_xs):
        expected = _closed_form_s(th, r, l)
        assert x == pytest.approx(expected, abs=1e-4)

    # PrismaticJoint's perpendicular lock: piston never leaves the slide axis
    for y, z in piston_yz:
        assert abs(y) < 1e-4
        assert abs(z) < 1e-4


def test_crank_position_matches_its_own_orientation():
    """Redundant physical cross-check: the crank BODY's world position must
    trace pivot + r*(cos theta, sin theta, 0) as a direct consequence of its
    own solved orientation and the point-constraint's local anchor offset --
    independent of the piston/rod chain entirely."""
    scene, b = _build_scene()
    r, crank = b["r"], b["crank"]
    dt = 1.0 / 960.0
    for i in range(400):
        step(scene, dt=dt, sub_steps=1)
    theta = twist_angle(crank.orientation, (0.0, 0.0, 1.0))
    expected = Vec3(r * math.cos(theta), r * math.sin(theta), 0.0)
    assert crank.position.x == pytest.approx(expected.x, abs=1e-4)
    assert crank.position.y == pytest.approx(expected.y, abs=1e-4)
