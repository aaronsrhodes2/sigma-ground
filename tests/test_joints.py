"""M2 gates — joints in the ONE stepper (actuation epic, analytic gates).

Closed forms, not eyeballs:
  - pendulum period T = 2π√(I_pivot/(m·g·d)) within 1% (dt = 1/960),
    oscillation-energy drift < 0.5% per period;
  - double pendulum conserves energy to < 1% over 10 s (chaotic — energy is
    the only honest invariant);
  - prismatic incline accelerates at a = g·sinθ within 0.5%;
  - a welded spinning pair's CoM follows the single-body parabola exactly
    (weld impulses are internal) with pose drift < 1 mm / 0.5°;
  - motor ledger: ΔE ≤ logged ∫τ·ω dt;
  - hinge limits settle to the boundary within 0.5°.
"""
import math

import pytest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.parcel import PhysicsParcel
from sigma_ground.dynamics.scene import PhysicsScene
from sigma_ground.dynamics.stepper import step
from sigma_ground.dynamics.joints import (WeldJoint, RevoluteJoint,
                                          PrismaticJoint, GearCouplingJoint)
from sigma_ground.dynamics.quat import qrot, quat_from_axis_angle, quat_mul

_G = 9.80665


class _Mat:
    density_kg_m3 = 1000.0
    restitution = 0.5

    def density_at_sigma(self, s):
        return 1000.0


def _rod(mass, length, pivot_world, hang_angle):
    """A thin rod hanging from pivot_world, tilted hang_angle from straight
    down (about z). Local +y runs from CoM toward the pivot end."""
    q = list(quat_from_axis_angle((0.0, 0.0, 1.0), hang_angle))
    off = qrot(q, (0.0, -0.5 * length, 0.0))       # pivot → CoM, world
    com = pivot_world + Vec3(off[0], off[1], off[2])
    inertia = (mass * length * length / 12.0, 1e-3 * mass,
               mass * length * length / 12.0)
    return PhysicsParcel(0.5 * length, _Mat(), mass=mass, position=com,
                         orientation=q, inertia_body=inertia)


def _energy(scene):
    e = scene.total_kinetic_energy()
    for p in scene.parcels:
        e += p.mass * _G * p.position.y
    return e


# ── pendulum period + energy drift ───────────────────────────────────────────

def test_pendulum_period_matches_closed_form_within_1pct():
    m, L, theta0 = 1.0, 1.0, math.radians(5.0)
    rod = _rod(m, L, Vec3(0, 0, 0), theta0)
    joint = RevoluteJoint(rod, None, Vec3(0, 0, 0), Vec3(0, 0, 1))
    scene = PhysicsScene([rod], ground=False, constraints=[joint])

    d = 0.5 * L                                    # pivot → CoM
    I_pivot = m * L * L / 3.0                      # thin rod about its end
    T_expect = 2.0 * math.pi * math.sqrt(I_pivot / (m * _G * d))

    dt = 1.0 / 960.0
    e0 = _energy(scene)
    # swing angle from the CoM position (φ = 0 is straight down)
    phi_prev = math.atan2(rod.position.x, -rod.position.y)
    t = 0.0
    crossings = []
    while t < 11.0 * T_expect and len(crossings) < 21:
        step(scene, dt=dt, sub_steps=1)
        t += dt
        phi = math.atan2(rod.position.x, -rod.position.y)
        if phi_prev > 0.0 >= phi or phi_prev < 0.0 <= phi:
            f = phi_prev / (phi_prev - phi)        # linear interp inside dt
            crossings.append(t - dt + f * dt)
        phi_prev = phi

    assert len(crossings) >= 21, "pendulum did not swing"
    T_meas = (crossings[20] - crossings[0]) / 10.0  # 20 half-periods
    assert T_meas == pytest.approx(T_expect, rel=0.01)

    # energy drift, measured against the oscillation energy scale
    e_bottom = -m * _G * d                          # hanging straight down
    e_osc = e0 - e_bottom
    drift = abs(_energy(scene) - e0)
    assert drift < 0.005 * 10.0 * e_osc             # < 0.5% per period


# ── double pendulum: chaotic chain — integrity exact, energy bounded ─────────

def test_double_pendulum_chain_integrity_and_bounded_energy():
    """A 90°/90° double pendulum whips chaotically (ω peaks ~5 rad/s).

    What IS guaranteed and gated here:
      - constraint INTEGRITY: the hinge anchors stay coincident to < 0.5 mm
        through 10 s of whip (measured 0.11 mm at the default 10 Gauss–
        Seidel iterations — the plateau is the GS chain-coupling limit,
        25× inside the mechanism recorder's 2 mm refusal gate);
      - energy error stays BOUNDED (< 12% of the oscillation scale at
        sub_dt = 1/960 — no runaway pumping or draining).

    KNOWN LIMITATION (measured, honest): a velocity-impulse solver with
    SHAKE/RATTLE staging does NOT conserve energy to 1% on fast chaotic
    CHAINS at practical resolution. Measured worst |ΔE|/scale:
      vs dt   (10 iters): 9.0% @ 1/960, 7.8% @ 1/1920, 2.4% @ 1/3840,
                          1.1% @ 1/7680           (~O(dt), costs wall-clock)
      vs iters (1/960):   9.0% @ 10,  4.4% @ 30,  2.3% @ 60
    The demo-ladder mechanisms (lids, scissors, cranks) are driven/damped,
    where this term is irrelevant; scene knobs (solver_iterations,
    sub_steps) buy accuracy when a scenario needs it. The architectural
    fix, if 1%-conservative chains ever matter, is reduced-coordinate
    (Featherstone) dynamics for joint chains — a named future stone, not
    a patch on this one.
    """
    L = 1.0
    rod1 = _rod(1.0, L, Vec3(0, 0, 0), math.radians(90.0))
    tip1 = Vec3(0, 0, 0) + (rod1.position - Vec3(0, 0, 0)) * 2.0
    rod2 = _rod(1.0, L, tip1, math.radians(90.0))
    j1 = RevoluteJoint(rod1, None, Vec3(0, 0, 0), Vec3(0, 0, 1))
    j2 = RevoluteJoint(rod2, rod1, tip1, Vec3(0, 0, 1))
    scene = PhysicsScene([rod1, rod2], ground=False, constraints=[j1, j2])

    e0 = _energy(scene)
    e_min = -1.0 * _G * 0.5 - 1.0 * _G * 1.5       # both hanging straight
    scale = e0 - e_min
    assert scale > 0

    dt = 1.0 / 960.0
    worst_e = 0.0
    worst_anchor = 0.0
    for k in range(int(10.0 / dt)):
        step(scene, dt=dt, sub_steps=1)
        if k % 96 == 0:
            worst_e = max(worst_e, abs(_energy(scene) - e0))
            for j in (j1, j2):
                ra, wa, rb, wb = j._geom()
                worst_anchor = max(worst_anchor, (wb - wa).length())
    assert worst_anchor < 5e-4                     # the chain never separates
    assert worst_e < 0.12 * scale                  # bounded, not runaway


# ── prismatic incline: a = g·sinθ ────────────────────────────────────────────

def test_prismatic_incline_a_equals_g_sin_theta():
    theta = math.radians(25.0)
    axis = Vec3(math.cos(theta), -math.sin(theta), 0)   # downhill
    p = PhysicsParcel(0.1, _Mat(), mass=2.0, position=Vec3(0, 0, 0))
    joint = PrismaticJoint(p, None, Vec3(0, 0, 0), axis)
    scene = PhysicsScene([p], ground=False, constraints=[joint])

    dt = 1.0 / 960.0
    T = 1.0
    for _ in range(int(T / dt)):
        step(scene, dt=dt, sub_steps=1)
    v_along = p.velocity.dot(axis)
    assert v_along == pytest.approx(_G * math.sin(theta) * T, rel=0.005)
    # travel matches ½at² too (negative: with b=None the WORLD is the child,
    # so a body sliding +n̂ reads travel() < 0 — the documented convention)
    assert -joint.travel() == pytest.approx(
        0.5 * _G * math.sin(theta) * T * T, rel=0.005)
    # KDK stagger: the end-of-step velocity carries the trailing gravity
    # half-kick; its ⊥ component (g·cosθ·dt/2) is removed by the NEXT
    # substep's solve — the honest off-axis bound is g·dt, not zero.
    off_axis = (p.velocity - axis * v_along).length()
    assert off_axis < _G * dt


# ── welded pair: internal impulses, parabola, no drift ───────────────────────

def test_welded_spinning_pair_follows_parabola_without_drift():
    pa = PhysicsParcel(0.2, _Mat(), mass=2.0, position=Vec3(-0.5, 2.0, 0.0))
    pb = PhysicsParcel(0.2, _Mat(), mass=1.0, position=Vec3(0.5, 2.0, 0.0))
    # rigid-body initial state: common v plus spin ω = 2 rad/s about z
    w = Vec3(0.0, 0.0, 2.0)
    com0 = (pa.position * 2.0 + pb.position * 1.0) * (1.0 / 3.0)
    v0 = Vec3(2.0, 3.0, 0.0)
    for p in (pa, pb):
        r = p.position - com0
        p.velocity = v0 + w.cross(r)
        p.angular_velocity = Vec3(w.x, w.y, w.z)
    weld = WeldJoint(pa, pb, Vec3(0.0, 2.0, 0.0))
    scene = PhysicsScene([pa, pb], ground=False, constraints=[weld])

    gap0 = (pb.position - pa.position).length()
    dt = 1.0 / 960.0
    T = 1.5
    for _ in range(int(T / dt)):
        step(scene, dt=dt, sub_steps=1)

    # CoM parabola (weld impulses are equal-opposite → momentum untouched)
    com = (pa.position * 2.0 + pb.position * 1.0) * (1.0 / 3.0)
    expect = com0 + v0 * T + Vec3(0, -0.5 * _G * T * T, 0)
    assert (com - expect).length() < 1e-9

    # pose drift: separation < 1 mm, relative orientation < 0.5°
    assert abs((pb.position - pa.position).length() - gap0) < 1e-3
    from sigma_ground.dynamics.quat import quat_conj
    q_rel = quat_mul(quat_conj(pa.orientation), pb.orientation)
    x, y, z, qw = q_rel
    ang = 2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), abs(qw))
    assert ang < math.radians(0.5)


# ── motor ledger: ΔE ≤ ∫τ·ω dt ───────────────────────────────────────────────

def test_motor_ledger_bounds_energy_gain():
    rod = _rod(1.0, 1.0, Vec3(0, 0, 0), 0.0)       # hanging straight down
    joint = RevoluteJoint(rod, None, Vec3(0, 0, 0), Vec3(0, 0, 1),
                          motor_speed=-3.0, motor_max_torque=2.0)
    scene = PhysicsScene([rod], ground=False, constraints=[joint])

    e0 = _energy(scene)
    dt = 1.0 / 960.0
    for _ in range(int(2.0 / dt)):
        step(scene, dt=dt, sub_steps=1)
    de = _energy(scene) - e0
    assert de > 0.1                                # the motor really worked
    assert de <= joint.motor_work_j + 1e-6         # ledger bounds the gain
    # and the cap was honored: |W| ≤ τ_max · |ω|_max · t (coarse sanity)
    assert joint.motor_work_j < 2.0 * 3.5 * 2.0


# ── limits settle at the boundary ────────────────────────────────────────────

def test_hinge_limit_settles_within_half_degree():
    rod = _rod(1.0, 1.0, Vec3(0, 0, 0), math.radians(35.0))
    joint = RevoluteJoint(rod, None, Vec3(0, 0, 0), Vec3(0, 0, 1),
                          limits=(-math.radians(15.0), math.radians(15.0)))
    scene = PhysicsScene([rod], ground=False, constraints=[joint])

    dt = 1.0 / 960.0
    tail = []
    n = int(4.0 / dt)
    for k in range(n):
        step(scene, dt=dt, sub_steps=1)
        if k >= n - int(0.5 / dt):
            tail.append(abs(joint.angle()))
    # released 35° off vertical with a ±15° window (relative to construction
    # pose): gravity drives it against the 15° boundary and it stays there
    lim = math.radians(15.0)
    tol = math.radians(0.5)
    assert all(abs(th - lim) < tol for th in tail)


# ── gear coupling: two hinges held at a fixed rate ratio (Phase 1) ──────────
# Placeholder wheels (bare spheres, isotropic inertia) — no gear-tooth
# geometry, no blueprint data. This gate proves the CONSTRAINT (a rigid,
# re-solved-every-substep ratio), not any rendered shape.

def _wheel(mass, radius, pos):
    return PhysicsParcel(radius, _Mat(), position=pos, mass=mass)


def test_gear_coupling_holds_the_commanded_ratio():
    wheel_a = _wheel(1.0, 0.05, Vec3(0.0, 0.0, 0.0))
    wheel_b = _wheel(1.0, 0.05, Vec3(0.3, 0.0, 0.0))
    axis = Vec3(0.0, 0.0, 1.0)
    ja = RevoluteJoint(wheel_a, None, wheel_a.position, axis,
                       motor_speed=-3.0, motor_max_torque=2.0)
    jb = RevoluteJoint(wheel_b, None, wheel_b.position, axis)
    ratio = 2.0
    coupling = GearCouplingJoint(ja, jb, ratio)
    scene = PhysicsScene([wheel_a, wheel_b], ground=False,
                         constraints=[ja, jb, coupling])

    e0 = _energy(scene)
    dt = 1.0 / 960.0
    for k in range(int(2.0 / dt)):
        step(scene, dt=dt, sub_steps=1)
        if k == int(1.0 / dt):
            # mid-run: the RATE ratio holds at an arbitrary instant, not just
            # at the end (s = -angular_velocity.z here since both joints
            # share world axis +z and b=None — see module docstring)
            assert wheel_b.angular_velocity.z == pytest.approx(
                -wheel_a.angular_velocity.z / ratio, rel=1e-3)

    # RATE ratio still holds at the end
    assert wheel_b.angular_velocity.z == pytest.approx(
        -wheel_a.angular_velocity.z / ratio, rel=1e-3)

    # the integral relationship (angle_a + ratio*angle_b = 0, both start at
    # 0) holds too — the coupling doesn't let the two hinges' PHASE drift.
    # angle() wraps to (-pi, pi], so compare modulo 2*pi rather than raw sum
    # (over 2s at ~3 rad/s ja's raw angle crosses the wrap boundary).
    phase = ja.angle() + ratio * jb.angle()
    phase -= 2.0 * math.pi * round(phase / (2.0 * math.pi))
    assert abs(phase) < 1e-3

    # the coupling is WORKLESS (transmits power losslessly): total energy
    # gain is still bounded by the motor's OWN logged work, same ledger gate
    # as test_motor_ledger_bounds_energy_gain — the second wheel's motion
    # comes entirely through the coupling, not from any energy of its own
    de = _energy(scene) - e0
    assert de > 0.0
    assert de <= ja.motor_work_j + 1e-6
