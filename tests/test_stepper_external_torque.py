"""Gate for the stepper's external-torque path — the extension that lets a
distributed force (wind on pitched blades) SPIN a body, not just push it.

Closed form: a constant pure torque tau via the (force, torque) callback,
half-kicked around the drift like gravity, gives EXACTLY the continuous
sampled solution (the leapfrog second-order property):
    omega_n = alpha * n * dt
    theta_n = 1/2 * alpha * (n * dt)^2
(unlike the motor row's full-impulse-before-drift placement, which carries
the n(n+1)/2 discrete bias gated in test_mechanism_spring.py — the two
placements are different, both exact against their own recursions).
Also gated: a plain Vec3 return still works (backwards contract), and a
pure torque does not translate the body.
"""
import math

import pytest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.parcel import PhysicsParcel
from sigma_ground.dynamics.scene import PhysicsScene
from sigma_ground.dynamics.stepper import step


class _Mat:
    density_kg_m3 = 1000.0
    restitution = 0.5

    def density_at_sigma(self, s):
        return 1000.0


def test_constant_external_torque_matches_leapfrog_closed_form():
    p = PhysicsParcel(0.05, _Mat(), position=Vec3(0, 0, 0), mass=1.0)
    scene = PhysicsScene([p], gravity=Vec3(0, 0, 0), ground=False)
    tau = 0.02
    I = 0.4 * 1.0 * 0.05 ** 2
    alpha = tau / I

    def cb(parcel):
        return (Vec3(0, 0, 0), Vec3(0, 0, tau))

    dt = 1.0 / 960.0
    n = 0
    from sigma_ground.dynamics.quat import twist_angle
    for _ in range(96):
        step(scene, dt=dt, sub_steps=1, external_forces=cb)
        n += 1
        t = n * dt
        assert p.angular_velocity.z == pytest.approx(alpha * t, rel=1e-9)
        theta = twist_angle(p.orientation, (0, 0, 1))
        assert theta == pytest.approx(0.5 * alpha * t * t, rel=1e-6)
    # a pure torque must not translate the body
    assert p.position.length() < 1e-12
    assert p.velocity.length() < 1e-12


def test_plain_force_return_still_works_unchanged():
    p = PhysicsParcel(0.05, _Mat(), position=Vec3(0, 0, 0), mass=2.0)
    scene = PhysicsScene([p], gravity=Vec3(0, 0, 0), ground=False)

    def cb(parcel):
        return Vec3(0.4, 0, 0)          # the ORIGINAL Vec3-only contract

    dt = 1.0 / 960.0
    for _ in range(96):
        step(scene, dt=dt, sub_steps=1, external_forces=cb)
    t = 96 * dt
    assert p.velocity.x == pytest.approx(0.2 * t, rel=1e-9)      # a = F/m
    assert p.angular_velocity.length() < 1e-12                    # no spin
