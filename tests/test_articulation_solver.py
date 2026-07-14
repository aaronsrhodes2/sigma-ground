"""M1 gates — torque-aware contacts in the ONE stepper (actuation epic).

The sequential-impulse solver (dynamics/solver.py) replaced the stepper's
immediate resolve_* calls. These gates hold it to the plan's contract:

  (a) LEGACY PARITY — sphere scenes reproduce the old resolve-in-place
      stepper to 1e-12 (center-line rows reduce exactly; skip-when-
      separating and the single 0.8·pen position pass are mirrored).
  (b) FIRST-IMPACT TORQUE — a tilted rod's endpoint impact changes angular
      momentum by exactly r×J (closed form, 1e-9).
  (c) RESTING MANIFOLD — a flat box on 4 corner contacts settles level and
      does not jitter.
  (d) COULOMB INCLINE — static holds when μ > tanθ; kinetic slides at
      a = g(sinθ − μcosθ) within 2% (the analytic friction gates).
  (e) LEDGER — contacts only dissipate; free-pair momentum conserved 1e-9.
"""
import math

import pytest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.parcel import PhysicsParcel
from sigma_ground.dynamics.scene import PhysicsScene, GroundPlane
from sigma_ground.dynamics.stepper import step
from sigma_ground.dynamics.collision import (
    resolve_sphere_sphere, resolve_sphere_plane)
from sigma_ground.dynamics.quat import qrot, quat_from_axis_angle

_G = 9.80665


class _Mat:
    density_kg_m3 = 1000.0
    restitution = 0.5

    def density_at_sigma(self, s):
        return 1000.0


def _sphere(radius, pos, vel, mass=None):
    return PhysicsParcel(radius, _Mat(), position=pos, velocity=vel,
                         mass=mass)


# ── (a) legacy parity ────────────────────────────────────────────────────────

def _legacy_substep(scene, dt):
    """The pre-M1 stepper body, verbatim (resolve-in-place semantics)."""
    g = scene.gravity
    for p in scene.parcels:
        if not p.is_static:
            p.velocity = p.velocity + g * (dt * 0.5)
    for p in scene.parcels:
        if not p.is_static:
            p.position = p.position + p.velocity * dt
    parcels = scene.parcels
    n = len(parcels)
    for i in range(n):
        for j in range(i + 1, n):
            resolve_sphere_sphere(parcels[i], parcels[j])
    if scene.ground is not None:
        gnd = scene.ground
        for p in parcels:
            if not p.is_static:
                resolve_sphere_plane(p, gnd.point, gnd.normal,
                                     gnd.restitution)
    for p in scene.parcels:
        if not p.is_static:
            p.velocity = p.velocity + g * (dt * 0.5)


def _max_state_diff(sa, sb):
    d = 0.0
    for pa, pb in zip(sa.parcels, sb.parcels):
        for u, v in ((pa.position, pb.position), (pa.velocity, pb.velocity)):
            d = max(d, abs(u.x - v.x), abs(u.y - v.y), abs(u.z - v.z))
    return d


def test_bouncing_sphere_matches_legacy_stepper_1e12():
    new = PhysicsScene([_sphere(0.5, Vec3(0, 2.0, 0), Vec3(0.3, 0, 0))])
    old = PhysicsScene([_sphere(0.5, Vec3(0, 2.0, 0), Vec3(0.3, 0, 0))])
    dt, subs = 0.005, 4
    for _ in range(300):                       # 1.5 s — several real bounces
        step(new, dt=dt, sub_steps=subs)
        for _ in range(subs):
            _legacy_substep(old, dt / subs)
    assert _max_state_diff(new, old) <= 1e-12


def test_two_sphere_collision_matches_legacy_stepper_1e12():
    def build():
        return PhysicsScene(
            [_sphere(0.3, Vec3(-1.0, 0, 0), Vec3(1.0, 0, 0), mass=2.0),
             _sphere(0.3, Vec3(1.0, 0, 0), Vec3(-1.0, 0, 0), mass=1.0)],
            gravity=Vec3(0, 0, 0), ground=False)
    new, old = build(), build()
    dt, subs = 0.005, 4
    for _ in range(300):
        step(new, dt=dt, sub_steps=subs)
        for _ in range(subs):
            _legacy_substep(old, dt / subs)
    assert _max_state_diff(new, old) <= 1e-12


# ── (b) first-impact torque: ΔL = r×J ────────────────────────────────────────

def test_tilted_rod_first_impact_dL_equals_r_cross_J():
    tilt = 0.3
    q0 = list(quat_from_axis_angle((0.0, 0.0, 1.0), tilt))
    lo = qrot(q0, (0.0, -0.5, 0.0))            # lower endpoint, world offset
    rod = PhysicsParcel(0.6, _Mat(), mass=3.0,
                        position=Vec3(0, -lo[1] + 1e-3, 0),
                        velocity=Vec3(0, -1.0, 0),
                        orientation=q0,
                        inertia_body=(0.25, 0.01, 0.25),
                        contact_offsets=[(0, -0.5, 0), (0, 0.5, 0)])
    scene = PhysicsScene([rod], gravity=Vec3(0, 0, 0))
    dt = 1e-3
    for _ in range(200):
        v_pre = Vec3(rod.velocity.x, rod.velocity.y, rod.velocity.z)
        L_pre = rod.angular_momentum()
        step(scene, dt=dt, sub_steps=1)
        dv = rod.velocity - v_pre
        if dv.length() > 1e-9:                 # the impact step
            J = dv * rod.mass                  # gravity-free: Δp is the impulse
            r = Vec3(*qrot(q0, (0.0, -0.5, 0.0)))   # ω was 0 → q unchanged
            dL = rod.angular_momentum() - L_pre
            rxJ = r.cross(J)
            assert abs(dL.x - rxJ.x) < 1e-9
            assert abs(dL.y - rxJ.y) < 1e-9
            assert abs(dL.z - rxJ.z) < 1e-9
            assert dL.length() > 1e-4          # a REAL torque was applied
            return
    pytest.fail("rod never hit the ground")


# ── (c) resting manifold: flat box settles level, no jitter ─────────────────

def _box(mass, hx, hy, hz, pos, tilt_z=0.0):
    inertia = (mass / 3.0 * (hy * hy + hz * hz),
               mass / 3.0 * (hx * hx + hz * hz),
               mass / 3.0 * (hx * hx + hy * hy))
    p = PhysicsParcel(math.sqrt(hx * hx + hy * hy + hz * hz), _Mat(),
                      mass=mass, position=pos,
                      orientation=list(quat_from_axis_angle((0, 0, 1),
                                                            tilt_z)),
                      inertia_body=inertia,
                      contact_offsets=[(sx * hx, -hy, sz * hz)
                                       for sx in (-1, 1) for sz in (-1, 1)])
    return p


def test_flat_box_settles_level_without_jitter():
    box = _box(2.0, 0.5, 0.05, 0.5, Vec3(0, 0.15, 0), tilt_z=0.02)
    box.restitution = 0.0
    scene = PhysicsScene([box], friction_fn=lambda a, b: 0.4)
    dt = 1.0 / 240.0
    ys = []
    n = int(5.0 / dt)
    for k in range(n):
        step(scene, dt=dt, sub_steps=4)
        if k >= n - int(1.0 / dt):             # record the last second
            ys.append(box.position.y)
    # KDK stagger: the end-of-step velocity carries the trailing gravity
    # half-kick (g·sub_dt/2 into the floor), removed by the next pre-solve.
    # The honest resting bound is therefore g·sub_dt, not zero.
    sub_dt = dt / 4.0
    assert box.velocity.length() < _G * sub_dt
    assert box.angular_velocity.length() < 1e-3
    up = qrot(box.orientation, (0.0, 1.0, 0.0))
    assert up[1] > 0.9999                      # level (tilt fully damped)
    assert 0.05 - 0.0015 < box.position.y < 0.05 + 0.001
    assert max(ys) - min(ys) < 1e-4            # no jitter
    # ledger: essentially all the drop energy was dissipated by contacts
    # (the residual is exactly the ½m(g·sub_dt/2)² stagger term)
    assert (box.kinetic_energy() + box.rotational_ke()) < 1e-4


# ── (d) Coulomb incline (analytic) ───────────────────────────────────────────

def _incline_scene(theta, mu):
    s, c = math.sin(theta), math.cos(theta)
    normal = Vec3(-s, c, 0)
    box = _box(2.0, 0.5, 0.05, 0.5, normal * (0.05 + 1e-4), tilt_z=theta)
    box.restitution = 0.0
    scene = PhysicsScene([box], ground=GroundPlane(y=0.0, normal=normal),
                         friction_fn=lambda a, b: mu)
    downhill = Vec3(-c, -s, 0)
    return scene, box, downhill


def test_static_friction_holds_when_mu_exceeds_tan_theta():
    theta = math.radians(30.0)                 # tanθ = 0.577
    scene, box, downhill = _incline_scene(theta, mu=0.8)
    p0 = Vec3(box.position.x, box.position.y, box.position.z)
    dt = 1.0 / 960.0
    for _ in range(int(2.0 / dt)):
        step(scene, dt=dt, sub_steps=1)
    slid = (box.position - p0).dot(downhill)
    assert abs(slid) < 2e-3                    # < 2 mm over 2 s: it holds
    # tangential velocity, net of the KDK stagger (the g·dt/2 trailing
    # half-kick lives in the NORMAL direction; downhill picks up s·g·dt/2)
    assert abs(box.velocity.dot(downhill)) < 3e-3


def test_kinetic_friction_gives_g_sin_minus_mu_cos():
    theta = math.radians(30.0)
    mu = 0.2                                   # < tanθ → slides from rest
    scene, box, downhill = _incline_scene(theta, mu)
    dt = 1.0 / 960.0
    for _ in range(int(0.5 / dt)):
        step(scene, dt=dt, sub_steps=1)
    v1 = box.velocity.dot(downhill)
    for _ in range(int(1.0 / dt)):
        step(scene, dt=dt, sub_steps=1)
    v2 = box.velocity.dot(downhill)
    a_meas = (v2 - v1) / 1.0
    a_expect = _G * (math.sin(theta) - mu * math.cos(theta))
    assert a_meas == pytest.approx(a_expect, rel=0.02)


# ── (e) ledger: dissipation + momentum ──────────────────────────────────────

def test_free_pair_conserves_momentum_and_dissipates():
    scene = PhysicsScene(
        [_sphere(0.3, Vec3(-1.0, 0, 0), Vec3(1.0, 0, 0), mass=2.0),
         _sphere(0.3, Vec3(1.0, 0, 0), Vec3(-1.0, 0, 0), mass=1.0)],
        gravity=Vec3(0, 0, 0), ground=False)
    mom0 = scene.total_momentum()
    ke0 = scene.total_kinetic_energy()
    dt = 1.0 / 480.0
    for _ in range(int(1.5 / dt)):
        step(scene, dt=dt, sub_steps=1)
    mom1 = scene.total_momentum()
    ke1 = scene.total_kinetic_energy()
    assert (mom1 - mom0).length() < 1e-9
    assert ke1 <= ke0 + 1e-12                  # contacts only dissipate
    assert ke1 < ke0                           # e=0.5 head-on: strictly so
