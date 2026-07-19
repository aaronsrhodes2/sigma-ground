"""Phase 3, step 1 — gate the spring-as-capped-torque-source trick ALONE,
in isolation, before any escapement or gear train exists.

Two closed forms:
  1. Constant torque (no winding decay): motor_speed set far beyond reach,
     motor_max_torque fixed. The existing impulse-clamp machinery should
     saturate every substep, reproducing plain constant-alpha kinematics
     theta(t)=1/2*alpha*t^2, omega(t)=alpha*t exactly.
  2. MainspringState's Hooke's-law decay: tau(theta_wound)=k*theta_wound
     turns I*theta''=tau into theta_wound''=-(k/I)*theta_wound (SHM) until
     the spring runs out, then constant angular velocity. Energy
     conservation (spring PE == final KE) is an independent cross-check on
     the same derivation, not just a repeat of it.
"""
import math

import pytest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.parcel import PhysicsParcel
from sigma_ground.dynamics.scene import PhysicsScene
from sigma_ground.dynamics.stepper import step
from sigma_ground.dynamics.joints import RevoluteJoint
from sigma_ground.dynamics.mechanisms.spring import MainspringState, HUGE_SPEED


class _Mat:
    density_kg_m3 = 1000.0
    restitution = 0.5

    def density_at_sigma(self, s):
        return 1000.0


def _wheel(mass, radius, pos=None):
    return PhysicsParcel(radius, _Mat(), position=pos or Vec3(0.0, 0.0, 0.0), mass=mass)


def test_constant_torque_matches_constant_angular_acceleration():
    """omega(t)=alpha*t matches the continuous closed form exactly (the
    motor's impulse per substep is exactly alpha*dt, confirmed directly).
    theta does NOT match the naive continuous 1/2*alpha*t^2, though — the
    stepper applies the motor's FULL substep impulse in one pass (the
    pre-drift SHAKE row) and then drifts the WHOLE substep using that
    already-updated velocity, i.e. symplectic/semi-implicit Euler for this
    DOF specifically (unlike gravity, which is genuinely half-kicked before
    AND after the drift). Unrolling that recursion from rest gives EXACTLY
    theta_n = alpha*dt^2*n*(n+1)/2, not the continuous n^2/2 — a real,
    first-order-in-dt discrete-vs-continuous gap (confirmed empirically: it
    shrinks by exactly 10x for a 10x-finer dt), not a solver bug. Gating
    against the CORRECT discrete recursion (exact to float precision) is a
    stronger check than a loose tolerance against the continuum limit."""
    wheel = _wheel(1.0, 0.05)
    axis = Vec3(0.0, 0.0, 1.0)
    tau = 0.02
    joint = RevoluteJoint(wheel, None, wheel.position, axis,
                          motor_speed=-HUGE_SPEED, motor_max_torque=tau)
    scene = PhysicsScene([wheel], ground=False, constraints=[joint])
    I = 0.4 * 1.0 * 0.05 ** 2          # solid-sphere inertia (isotropic)
    alpha = tau / I

    dt = 1.0 / 960.0
    n = 0
    for _ in range(48):
        step(scene, dt=dt, sub_steps=1)
        n += 1
        theta = abs(joint.angle())
        omega = wheel.angular_velocity.length()
        theta_discrete = alpha * dt * dt * n * (n + 1) / 2.0
        assert theta == pytest.approx(theta_discrete, rel=1e-6)
        assert omega == pytest.approx(alpha * (n * dt), rel=1e-6)

    # never actually reached the target speed — confirms the motor stayed
    # torque-capped throughout, not coincidentally velocity-limited instead
    assert wheel.angular_velocity.length() < 0.1 * HUGE_SPEED


def test_constant_torque_discrete_bias_shrinks_linearly_with_dt():
    """The theta_n vs 1/2*alpha*t_n^2 gap above is a real discretization
    artifact, not a modeling error — confirm it vanishes as dt -> 0 at the
    expected first order (a 10x finer dt should shrink the gap ~10x)."""
    def _final_ratio(dt_frac):
        wheel = _wheel(1.0, 0.05)
        axis = Vec3(0.0, 0.0, 1.0)
        tau = 0.02
        joint = RevoluteJoint(wheel, None, wheel.position, axis,
                              motor_speed=-HUGE_SPEED, motor_max_torque=tau)
        scene = PhysicsScene([wheel], ground=False, constraints=[joint])
        I = 0.4 * 1.0 * 0.05 ** 2
        alpha = tau / I
        dt = 1.0 / dt_frac
        t = 0.0
        for _ in range(int(0.05 / dt)):
            step(scene, dt=dt, sub_steps=1)
            t += dt
        return abs(joint.angle()) / (0.5 * alpha * t * t)

    r1, r2 = _final_ratio(960), _final_ratio(9600)
    bias1, bias2 = r1 - 1.0, r2 - 1.0
    assert bias1 > 0.005                       # a real, measurable bias at dt=1/960
    assert bias2 == pytest.approx(bias1 / 10.0, rel=0.15)   # shrinks ~10x for 10x finer dt


def test_mainspring_matches_closed_form_shm_then_coasts():
    wheel = _wheel(2.0, 0.1)
    axis = Vec3(0.0, 0.0, 1.0)
    joint = RevoluteJoint(wheel, None, wheel.position, axis, motor_speed=0.0,
                          motor_max_torque=0.0)
    scene = PhysicsScene([wheel], ground=False, constraints=[joint])
    I = 0.4 * 2.0 * 0.1 ** 2
    k, theta_wound0 = 0.08, 4.0
    spring = MainspringState(joint, k=k, theta_wound0=theta_wound0)
    omega_n = math.sqrt(k / I)
    t_exhaust = math.pi / (2.0 * omega_n)
    omega_final = theta_wound0 * omega_n
    margin = 0.03 * t_exhaust
    # the discrete-vs-continuous gap characterized in the constant-torque
    # tests above is worst at SMALL t too (not just near exhaustion) — same
    # underlying effect, confirmed empirically to be <1.1% here for t>0.05s
    t_settle = 0.05

    dt = 1.0 / 1920.0
    t = 0.0
    t_max = t_exhaust * 1.6
    checked_pre, checked_post = 0, 0
    while t < t_max:
        spring.drive(direction=-1.0)
        step(scene, dt=dt, sub_steps=1)
        t += dt
        unwound = spring.unwound()
        omega = wheel.angular_velocity.length()
        if t_settle < t < t_exhaust - margin:
            expected_unwound = theta_wound0 * (1.0 - math.cos(omega_n * t))
            assert unwound == pytest.approx(expected_unwound, rel=0.03)
            checked_pre += 1
        elif t > t_exhaust + margin:
            assert omega == pytest.approx(omega_final, rel=0.02)
            assert spring.wound_out
            checked_post += 1
    assert checked_pre > 20 and checked_post > 20, "not enough resolution either side of exhaustion"

    # energy conservation: spring PE == closed-form final KE (independent
    # cross-check on the same SHM derivation), and the SIMULATED final KE
    # matches too — not just the algebra, the actual integrated motion
    PE = 0.5 * k * theta_wound0 ** 2
    KE_closed_form = 0.5 * I * omega_final ** 2
    assert PE == pytest.approx(KE_closed_form, rel=1e-9)
    KE_actual = 0.5 * I * wheel.angular_velocity.length() ** 2
    assert KE_actual == pytest.approx(KE_closed_form, rel=0.03)
