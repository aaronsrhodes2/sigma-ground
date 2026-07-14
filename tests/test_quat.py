"""dynamics.quat — the quaternion kernel (M0 of the actuation epic).

The functions were lifted verbatim from the proven tumble recorder; these
tests pin the algebra the constraint solver will lean on: unit-norm stability
over long integrations, rotation composition, axis-angle round-trips, the
twist decomposition (joint limits), and the 3×3 block solve.
"""
import math
import random

import pytest

from sigma_ground.dynamics.quat import (
    qrot, qrot_inv, quat_step, quat_mul, quat_conj, quat_normalize,
    quat_from_axis_angle, twist_angle, solve3,
)


def test_unit_norm_preserved_over_1e5_steps():
    q = [0.0, 0.0, 0.0, 1.0]
    w = (0.3, -1.1, 0.7)
    for _ in range(100_000):
        q = quat_step(q, w, 1e-3)
    n = math.sqrt(sum(v * v for v in q))
    assert n == pytest.approx(1.0, abs=1e-12)


def test_rotation_composition_associates():
    rng = random.Random(8128)
    for _ in range(50):
        a = quat_normalize([rng.uniform(-1, 1) for _ in range(4)])
        b = quat_normalize([rng.uniform(-1, 1) for _ in range(4)])
        v = [rng.uniform(-2, 2) for _ in range(3)]
        lhs = qrot(quat_mul(a, b), v)
        rhs = qrot(a, qrot(b, v))
        assert all(abs(x - y) < 1e-12 for x, y in zip(lhs, rhs))


def test_qrot_inv_inverts():
    q = quat_from_axis_angle((1.0, 2.0, -0.5), 1.234)
    v = (0.3, -0.7, 1.9)
    back = qrot_inv(q, qrot(q, v))
    assert all(abs(x - y) < 1e-12 for x, y in zip(back, v))


def test_axis_angle_round_trip():
    axis = (0.0, 0.0, 1.0)
    for ang in (-2.5, -0.4, 0.0, 0.7, 3.0):
        q = quat_from_axis_angle(axis, ang)
        # rotate x̂ and measure the turn in the xy plane
        x, y, _ = qrot(q, (1.0, 0.0, 0.0))
        assert math.atan2(y, x) == pytest.approx(
            math.atan2(math.sin(ang), math.cos(ang)), abs=1e-12)


def test_twist_angle_measures_the_hinge_turn():
    axis = (0.0, 1.0, 0.0)
    for ang in (-1.5, -0.2, 0.0, 0.9, 2.8):
        q = quat_from_axis_angle(axis, ang)
        assert twist_angle(q, axis) == pytest.approx(ang, abs=1e-12)
    # rotation about a PERPENDICULAR axis has zero twist about this one
    q_perp = quat_from_axis_angle((1.0, 0.0, 0.0), 1.0)
    assert twist_angle(q_perp, axis) == pytest.approx(0.0, abs=1e-12)


def test_solve3_matches_hand_inverse():
    K = ((4.0, 1.0, 0.2), (1.0, 3.0, -0.5), (0.2, -0.5, 5.0))
    x = (0.7, -1.2, 0.4)
    b = tuple(sum(K[i][j] * x[j] for j in range(3)) for i in range(3))
    got = solve3(K, b)
    assert all(abs(g - e) < 1e-12 for g, e in zip(got, x))
    # degenerate K falls back without exploding
    y = solve3(((0.0,) * 3,) * 3, (1.0, 2.0, 3.0))
    assert all(math.isfinite(v) for v in y)


def test_parcel_angular_state_and_matrix_free_inertia():
    from sigma_ground.dynamics.vec import Vec3
    from sigma_ground.dynamics.parcel import PhysicsParcel
    from sigma_ground.shapes import Sphere

    class _M:
        density_kg_m3 = 1000.0
        restitution = 0.5
        def density_at_sigma(self, s):
            return 1000.0

    p = PhysicsParcel(Sphere(0.1), _M())
    assert p.orientation == [0.0, 0.0, 0.0, 1.0]
    # sphere: isotropic principal inertia == the legacy moment_of_inertia
    assert p.inertia_body[0] == pytest.approx(p.moment_of_inertia('x'))
    # I⁻¹(I·ω) == ω through the matrix-free path, at a non-identity orientation
    p.orientation = list(quat_from_axis_angle((1.0, 1.0, 0.0), 0.8))
    p.angular_velocity = Vec3(0.4, -0.2, 1.1)
    L = p.angular_momentum()
    w = p.inv_inertia_apply(L)
    assert (w.x, w.y, w.z) == pytest.approx(
        (p.angular_velocity.x, p.angular_velocity.y, p.angular_velocity.z),
        abs=1e-12)
    # rotational KE at identity == the old world-axis formula
    p.orientation = [0.0, 0.0, 0.0, 1.0]
    ke_old = 0.5 * (p.moment_of_inertia('x') * 0.4 ** 2
                    + p.moment_of_inertia('y') * 0.2 ** 2
                    + p.moment_of_inertia('z') * 1.1 ** 2)
    assert p.rotational_ke() == pytest.approx(ke_old, rel=1e-12)
