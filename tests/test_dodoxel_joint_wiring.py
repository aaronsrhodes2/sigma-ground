"""Inference-to-solver wiring gates -- hinge arc Phase D, the arc's
capstone: a multi-part DodoxelField becomes a live PhysicsScene whose
inferred welds/hinges are REAL dynamics/joints.py constraints exercised by
the actual stepper -- the Captain's standing condition ("exercising our
physics chains"), not a data-graph inspection.
"""
import math

import pytest

from sigma_ground.deckard.dodoxelize import (dodoxelize_parts,
                                             dodoxel_parts_to_scene)
from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.scene import _G_STANDARD
from sigma_ground.dynamics.stepper import step

_SQRT2 = math.sqrt(2.0)
PITCH = 0.008
G = PITCH / _SQRT2
DENSITY = 5000.0


def _density_of(name):
    return DENSITY


class _Mat:
    density_kg_m3 = DENSITY
    restitution = 0.3

    def density_at_sigma(self, s):
        return DENSITY


def _mat_of(name):
    return _Mat()


def _slab_pair_field():
    split = 3.5 * G

    def left(x, y, z):
        return x < split

    def right(x, y, z):
        return x >= split

    span = 8 * G
    return dodoxelize_parts(
        [("left", "iron", left), ("right", "iron", right)],
        PITCH, (0.0, 0.0, 0.0), (span, span, span), _density_of)


def _line_pair_field():
    def line_a(x, y, z):
        return abs(x - 2 * G) < 0.35 * G and abs(y - 2 * G) < 0.35 * G

    def line_b(x, y, z):
        return abs(x - 3 * G) < 0.35 * G and abs(y - 3 * G) < 0.35 * G

    return dodoxelize_parts(
        [("a", "iron", line_a), ("b", "iron", line_b)],
        PITCH, (0.0, 0.0, 0.0), (5 * G, 5 * G, 14 * G), _density_of)


def test_welded_pair_free_falls_as_one_rigid_body():
    """The slab pair infers a weld; under gravity with no floor, the two
    parcels must fall TOGETHER (relative velocity ~ 0, the weld is
    workless) and match the free-fall closed form."""
    out = dodoxel_parts_to_scene(_slab_pair_field(), _mat_of, _density_of)
    (ja,) = out["inference"]["joints"]
    assert ja["type"] == "weld"

    pa, pb = out["parcels"]
    y0a, y0b = pa.position.y, pb.position.y
    offset0 = (pa.position - pb.position).length()
    dt = 1.0 / 960.0
    t = 0.0
    while t < 0.3:
        step(out["scene"], dt=dt, sub_steps=1)
        t += dt

    expected_drop = 0.5 * _G_STANDARD * t * t
    assert pa.position.y == pytest.approx(y0a - expected_drop, rel=1e-3)
    assert pb.position.y == pytest.approx(y0b - expected_drop, rel=1e-3)
    rel_v = (pa.velocity - pb.velocity).length()
    assert rel_v < 1e-6 * max(1.0, pa.velocity.length())
    # rigid: the inter-part offset is preserved through the fall
    assert (pa.position - pb.position).length() == pytest.approx(
        offset0, rel=1e-6)


def test_hinged_pair_spins_freely_about_the_inferred_axis():
    """The line pair infers a revolute about z through the interface
    centroid. Pin part A to the WORLD (an earlier draft left A dynamic and
    measured B's orbit against the ORIGINAL anchor position -- wrong
    physics, not a solver bug: B circling the hinge needs centripetal
    constraint force through the anchor, which reacts on A, so the whole
    free-floating assembly drifts and the hinge line moves with it; the
    observed 0.004->0.0064 m "drift" was that real, correct recoil). With
    A world-welded the hinge line is genuinely fixed: B keeps spinning
    about the axis at its set rate and its CoM stays on its circle."""
    from sigma_ground.dynamics.joints import WeldJoint

    field = _line_pair_field()
    out = dodoxel_parts_to_scene(field, _mat_of, _density_of,
                                 gravity=Vec3(0.0, 0.0, 0.0))
    (jinfo,) = out["inference"]["joints"]
    assert jinfo["type"] == "revolute"
    axis = Vec3(*jinfo["axis"])
    anchor = Vec3(*jinfo["anchor_m"])

    pa, pb = out["parcels"]
    out["scene"].constraints.append(WeldJoint(pa, None, pa.position))
    # default 10 Gauss-Seidel sweeps bleed ~11%/0.5s of spin here (the
    # double-pendulum test documents this same iteration-count energy
    # characteristic); 20 is the repo's own established remedy
    # (record_clock uses exactly this for its constraint chains)
    out["scene"].solver_iterations = 20
    omega = 3.0
    r = pb.position - anchor
    pb.angular_velocity = axis * omega
    pb.velocity = (axis * omega).cross(r)

    def dist_to_hinge_line(p):
        d = p - anchor
        along = d.dot(axis)
        perp = d - axis * along
        return perp.length()

    r_perp0 = dist_to_hinge_line(pb.position)
    dt = 1.0 / 960.0
    t = 0.0
    while t < 0.5:
        step(out["scene"], dt=dt, sub_steps=1)
        t += dt

    # B still spins about ~the hinge axis at ~the set rate
    spin = pb.angular_velocity
    assert spin.length() == pytest.approx(omega, rel=0.05)
    assert spin.dot(axis) / spin.length() == pytest.approx(1.0, abs=0.02)
    # B's CoM stays on its circle around the (now genuinely fixed) hinge
    assert dist_to_hinge_line(pb.position) == pytest.approx(r_perp0, rel=0.02)
    # A pinned: stays put
    assert pa.velocity.length() < 1e-6


def test_weld_contrast_kills_the_same_relative_spin():
    """Same initial relative spin, but forced weld typing (threshold too
    high for anything to read as a hinge... impossible for a perfect line,
    so use the slab pair instead, which genuinely welds): the weld's
    angular lock must rapidly kill relative rotation -- the joint TYPE
    decision has real dynamical consequences."""
    out = dodoxel_parts_to_scene(_slab_pair_field(), _mat_of, _density_of,
                                 gravity=Vec3(0.0, 0.0, 0.0))
    (jinfo,) = out["inference"]["joints"]
    assert jinfo["type"] == "weld"
    pa, pb = out["parcels"]
    pb.angular_velocity = Vec3(0.0, 0.0, 3.0)

    dt = 1.0 / 960.0
    for _ in range(200):
        step(out["scene"], dt=dt, sub_steps=1)

    rel_spin = (pa.angular_velocity - pb.angular_velocity).length()
    assert rel_spin < 0.05                          # locked back together