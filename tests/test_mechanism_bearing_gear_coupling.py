"""BearingGearCoupling -- windmill drivetrain roadmap, Arc A Phase 1.

Standalone, synthetic drive: a RigidBearing-mounted rotor spun by a simple
KNOWN CONSTANT TORQUE (not the real wind model -- that's Phase 3), coupled
to a RevoluteJoint follower via BearingGearCoupling. Two things to prove:

  1. The follower's actual world-frame spin tracks ratio*bearing.omega().
  2. The coupling is LOAD-BLIND (the flagged KNOWN_GAPS.md claim, measured
     directly): the bearing's own omega() trace is identical whether or not
     a loaded follower is attached.
"""
import math

import pytest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.parcel import PhysicsParcel
from sigma_ground.dynamics.scene import PhysicsScene
from sigma_ground.dynamics.stepper import step
from sigma_ground.dynamics.joints import RevoluteJoint
from sigma_ground.dynamics.mechanisms.bearing import RigidBearing
from sigma_ground.dynamics.mechanisms.bearing_gear_coupling import BearingGearCoupling


class _Mat:
    density_kg_m3 = 1000.0
    restitution = 0.5

    def density_at_sigma(self, s):
        return 1000.0


AXIS = Vec3(1.0, 0.0, 0.0)
TAU = 0.02                                  # N*m, constant torque on the rotor
ROTOR_R = 0.05
ROTOR_M = 1.0
I_ROTOR = 0.4 * ROTOR_M * ROTOR_R ** 2      # uniform sphere, I = (2/5)m*r^2


def _const_torque(rotor):
    def f(p):
        if p is not rotor:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(0.0, 0.0, 0.0), AXIS * TAU
    return f


def _spin_up_isolated(dt, n_steps):
    """Rotor + bearing ALONE -- no follower, no coupling: the baseline."""
    rotor = PhysicsParcel(ROTOR_R, _Mat(), position=Vec3(0, 0, 0), mass=ROTOR_M)
    bearing = RigidBearing(rotor, rotor.position, AXIS)
    scene = PhysicsScene([rotor], gravity=Vec3(0, 0, 0), ground=False)
    torque = _const_torque(rotor)
    trace = []
    for i in range(n_steps):
        step(scene, dt=dt, sub_steps=1, external_forces=torque)
        bearing.project()
        trace.append(bearing.omega())
    return trace


def _spin_up_with_follower(dt, n_steps, ratio):
    rotor = PhysicsParcel(ROTOR_R, _Mat(), position=Vec3(0, 0, 0), mass=ROTOR_M)
    bearing = RigidBearing(rotor, rotor.position, AXIS)
    follower = PhysicsParcel(0.01, _Mat(), position=Vec3(1.0, 0, 0), mass=0.02)
    j_follower = RevoluteJoint(follower, None, follower.position, AXIS)
    coupling = BearingGearCoupling(bearing, j_follower, ratio,
                                   motor_max_torque=50.0)
    scene = PhysicsScene([rotor, follower], gravity=Vec3(0, 0, 0),
                         ground=False, constraints=[j_follower])
    torque = _const_torque(rotor)
    bearing_trace, follower_trace = [], []
    for i in range(n_steps):
        coupling.step()
        step(scene, dt=dt, sub_steps=1, external_forces=torque)
        bearing.project()
        bearing_trace.append(bearing.omega())
        follower_trace.append(follower.angular_velocity.dot(AXIS))
    return bearing_trace, follower_trace


def test_follower_tracks_commanded_ratio():
    dt = 1.0 / 960.0
    n_steps = 2000
    ratio = 3.0
    bearing_trace, follower_trace = _spin_up_with_follower(dt, n_steps, ratio)
    # coupling.step() sets THIS step's target from the PRIOR step's bearing
    # omega (a one-substep control lag) -- compare against that, not the
    # same-index value
    for i in range(50, n_steps, 50):
        expected = ratio * bearing_trace[i - 1]
        assert follower_trace[i] == pytest.approx(expected, abs=1e-3)


def test_coupling_is_load_blind_matching_the_flagged_gap():
    """The measured form of the KNOWN_GAPS.md claim: the bearing's own omega
    trace is UNAFFECTED by whether a loaded follower is coupled to it."""
    dt = 1.0 / 960.0
    n_steps = 2000
    baseline = _spin_up_isolated(dt, n_steps)
    loaded, _ = _spin_up_with_follower(dt, n_steps, ratio=3.0)
    for a, b in zip(baseline[::50], loaded[::50]):
        assert a == pytest.approx(b, rel=1e-6, abs=1e-9)


def test_bearing_omega_matches_constant_torque_closed_form():
    dt = 1.0 / 960.0
    n_steps = 2000
    trace = _spin_up_isolated(dt, n_steps)
    alpha = TAU / I_ROTOR
    for i in range(0, n_steps, 200):
        t = (i + 1) * dt
        assert trace[i] == pytest.approx(alpha * t, rel=1e-3)
