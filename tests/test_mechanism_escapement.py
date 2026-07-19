"""Phase 3, step 2 — gate the escapement as a STANDALONE toy: one escape
wheel + one pendulum, placeholder shapes, no gear train. Per the plan's
de-risking order, this must pass on its own before Escapement is ever
wired into a real gear-train recorder.

Closed-form gate: the pendulum's period is ALREADY validated independently
(test_joints.py::test_pendulum_period_matches_closed_form_within_1pct,
T=2*pi*sqrt(I_pivot/(m*g*d))); this test checks the escapement's tick
spacing against that SAME closed form (one tick per pendulum half-period,
the classic anchor/deadbeat cadence — see escapement.py's module docstring
for why this is NOT the lever escapement's double-impulse mechanics), plus
exact tooth-count accounting (N ticks == N tooth pitches, no more, no less)
and the same motor-work energy ledger bound used throughout dynamics/joints.py.
"""
import math

import pytest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.parcel import PhysicsParcel
from sigma_ground.dynamics.scene import PhysicsScene
from sigma_ground.dynamics.stepper import step
from sigma_ground.dynamics.joints import RevoluteJoint
from sigma_ground.dynamics.quat import qrot, quat_from_axis_angle
from sigma_ground.dynamics.mechanisms.escapement import Escapement

_G = 9.80665


class _Mat:
    density_kg_m3 = 1000.0
    restitution = 0.5

    def density_at_sigma(self, s):
        return 1000.0


def _rod(mass, length, pivot_world, hang_angle):
    """Same construction as test_joints.py's own pendulum helper — a thin
    rod hanging from pivot_world, tilted hang_angle from straight down."""
    q = list(quat_from_axis_angle((0.0, 0.0, 1.0), hang_angle))
    off = qrot(q, (0.0, -0.5 * length, 0.0))
    com = pivot_world + Vec3(off[0], off[1], off[2])
    inertia = (mass * length * length / 12.0, 1e-3 * mass,
              mass * length * length / 12.0)
    return PhysicsParcel(0.5 * length, _Mat(), mass=mass, position=com,
                         orientation=q, inertia_body=inertia)


def _release(rod, pivot_world, length, hang_angle):
    """Displace an ALREADY-CONSTRUCTED rod (built at vertical, so its
    joint's angle()==0 is TRUE vertical — see the test docstring) to
    hang_angle from vertical at rest — exactly like pulling a pendulum
    aside and letting go, without moving the joint's own zero reference."""
    q = list(quat_from_axis_angle((0.0, 0.0, 1.0), hang_angle))
    off = qrot(q, (0.0, -0.5 * length, 0.0))
    rod.position = pivot_world + Vec3(off[0], off[1], off[2])
    rod.orientation = q


def _wheel(mass, radius, pos=None):
    return PhysicsParcel(radius, _Mat(), position=pos or Vec3(0.3, 0.0, 0.0), mass=mass)


def test_escapement_ticks_at_half_pendulum_period_with_exact_tooth_accounting():
    # Built at VERTICAL so the joint's angle()==0 is TRUE vertical (the
    # escapement's zero-crossing trigger), then displaced to the release
    # angle — angle() reads 0 at CONSTRUCTION, so building it already
    # tilted would put "zero" at the release point, not at the swing's
    # true center, and the pendulum would never cross it.
    L, m_pend = 1.0, 1.0
    pivot = Vec3(0.0, 0.0, 0.0)
    rod = _rod(m_pend, L, pivot, 0.0)
    pend_joint = RevoluteJoint(rod, None, pivot, Vec3(0.0, 0.0, 1.0))
    _release(rod, pivot, L, math.radians(5.0))

    d = 0.5 * L
    I_pivot = m_pend * L * L / 3.0
    T_expect = 2.0 * math.pi * math.sqrt(I_pivot / (m_pend * _G * d))

    teeth = 30
    wheel = _wheel(0.05, 0.02)
    tau = 0.001
    esc_joint = RevoluteJoint(wheel, None, wheel.position, Vec3(0.0, 0.0, 1.0),
                              motor_speed=-100.0, motor_max_torque=tau)

    scene = PhysicsScene([rod, wheel], ground=False,
                         constraints=[pend_joint, esc_joint])
    escapement = Escapement(esc_joint, pend_joint, teeth=teeth, direction=-1.0)

    dt = 1.0 / 1920.0
    t = 0.0
    # 10 periods, NOT 4: at teeth=30 the cumulative travel crosses pi after
    # ~15 ticks — the window where a walking-stop implementation silently
    # loses its brake to angle()'s (-pi, pi] wrap (the bug that made the
    # full clock's minute hand run ~145x fast). This window is the
    # regression coverage for that; the re-anchoring design keeps the
    # working angle in [0, pitch] so there is no seam to cross.
    t_max = 10.0 * T_expect
    while t < t_max:
        step(scene, dt=dt, sub_steps=1)
        t += dt
        escapement.step(t)

    assert escapement.ticks >= 20, "not enough ticks in the observation window"

    # tick spacing == half the (independently, already-validated) pendulum
    # period, for EVERY gap (the first crossing happens a quarter-period
    # after release, but the GAPS between crossings are uniformly T/2 from
    # the first one on — a released pendulum still swings symmetrically)
    assert escapement.tick_times[0] == pytest.approx(T_expect / 4.0, rel=0.03)
    gaps = [b - a for a, b in zip(escapement.tick_times, escapement.tick_times[1:])]
    for g in gaps:
        assert g == pytest.approx(T_expect / 2.0, rel=0.03)

    # exact tooth accounting: N ticks -> EXACTLY N tooth pitches of REAL,
    # measured wheel travel (travel_rad integrates actual angle deltas,
    # wrap-corrected, re-anchor shifts excluded — not inferred from the
    # tick count), never more (missed relock) or fewer (missed release)
    pitch = 2.0 * math.pi / teeth
    expected_travel = escapement.direction * escapement.ticks * pitch
    assert escapement.travel_rad == pytest.approx(expected_travel, abs=0.02 * pitch)

    # energy ledger: same bound used throughout dynamics/joints.py — the
    # wheel's current KE (mid-traverse or freshly locked) is bounded by the
    # motor's own logged work, regardless of how much was already
    # dissipated at prior lock events
    I_wheel = 0.4 * wheel.mass * (0.02 ** 2)      # solid-sphere inertia
    KE_wheel = 0.5 * I_wheel * wheel.angular_velocity.length() ** 2
    assert esc_joint.motor_work_j > 0.0
    assert KE_wheel <= esc_joint.motor_work_j + 1e-9


def test_escapement_holds_locked_between_ticks():
    """Between releases the wheel must sit DEAD STILL at its locked stop —
    not creep, not oscillate — proving the brake-only limit row (already
    gated by test_joints.py::test_hinge_limit_settles_within_half_degree)
    genuinely holds under this module's zero-width-window locking scheme."""
    pivot = Vec3(0.0, 0.0, 0.0)
    rod = _rod(1.0, 1.0, pivot, 0.0)          # built at vertical — see the
    pend_joint = RevoluteJoint(rod, None, pivot, Vec3(0.0, 0.0, 1.0))  # other test's docstring note
    _release(rod, pivot, 1.0, math.radians(5.0))
    wheel = _wheel(0.05, 0.02)
    esc_joint = RevoluteJoint(wheel, None, wheel.position, Vec3(0.0, 0.0, 1.0),
                              motor_speed=-100.0, motor_max_torque=0.001)
    scene = PhysicsScene([rod, wheel], ground=False,
                         constraints=[pend_joint, esc_joint])
    escapement = Escapement(esc_joint, pend_joint, teeth=30, direction=-1.0)

    dt = 1.0 / 1920.0
    t = 0.0
    settle_until = 0.0
    settle_grace = 0.2     # generously more than the wheel's own traverse
                           # time for one pitch under this torque/inertia
                           # (~0.06s, computed the same way the module
                           # docstring's parameters were sized) — a real
                           # per-tick "still in flight" window, not creep
    max_creep = 0.0
    for _ in range(int(1.0 / dt)):        # well under one tick period — at most one tick fires
        step(scene, dt=dt, sub_steps=1)
        t += dt
        ticked = escapement.step(t)
        if ticked:
            settle_until = t + settle_grace
        elif t > settle_until:
            max_creep = max(max_creep, abs(esc_joint.angle() - escapement.locked_angle))
    pitch = 2.0 * math.pi / 30
    assert max_creep < 0.02 * pitch, "wheel crept while nominally locked"
