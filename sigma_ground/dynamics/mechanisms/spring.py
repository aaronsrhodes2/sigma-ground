"""Mainspring — a torque-capped RevoluteJoint motor whose cap decays as the
spring unwinds. Reuses dynamics/joints.py's RevoluteJoint motor AS-IS with
no new solver math: set ``motor_speed`` far beyond anything the joint could
ever reach, so the motor's own impulse-clamp logic (``joints.py``'s
``cap = motor_max_torque*dt``) saturates every substep — the existing
"drive toward a target speed, capped" row degenerates into a plain constant-
at-each-instant torque source for free. ``motor_max_torque`` is then
recomputed from the winding state once per outer step, entirely from
Python; the solver never sees anything but the motor row it already has.

MVP torque law: tau(theta_wound) = k * theta_wound (linear / Hooke's-law).
NOT a real mainspring curve — real clock/watch mainsprings are markedly
NONLINEAR (a documented future source, see misc/DATASETS.md's mechanism
blueprint queue). Flagged here explicitly, not hidden behind a plausible-
looking constant.

Closed form this unlocks (tests/test_mechanism_spring.py): with the wheel
pinned at its own centre (gravity fully absorbed by the point constraint,
same construction as record_motor_spin) and I*theta'' = tau, substituting
u = theta_wound0 - |Delta theta| gives u'' = -(k/I)*u — SIMPLE HARMONIC
MOTION, u(t) = theta_wound0*cos(wt), w = sqrt(k/I), valid until the spring
runs out at wt = pi/2. Energy conservation (spring PE = (1/2)k*theta_wound0^2
exactly equals the wheel's KE at that instant) falls out of the same
algebra as an independent cross-check.
"""
import math

from ..quat import twist_angle

HUGE_SPEED = 1.0e4     # rad/s target, far beyond anything reachable — turns
                        # the existing motor primitive into a pure constant-
                        # torque source (see module docstring)


class MainspringState:
    """theta_wound0: total wound angle at full wind (rad, > 0). Any real
    mainspring unwinds through several full turns, so theta_wound0 is
    routinely >> pi.
    k: torque constant (N*m/rad, Hooke's-law MVP).
    joint: the RevoluteJoint this spring drives — theta_wound is read off
    the joint's RAW relative twist (twist_angle(joint._q_rel(),
    joint.axis_a)), UNWRAPPED. Two deliberate choices there:
      - unwrapped, because the raw twist wraps to (-pi, pi] and a naive
        abs(angle-angle0) reading corrupts the instant cumulative rotation
        exceeds pi (caught by the closed-form energy gate in
        tests/test_mechanism_spring.py going visibly wrong past that point);
      - the RAW twist rather than joint.angle(), because angle() measures
        against q0_rel — a REFERENCE other code may legitimately move:
        mechanisms/escapement.py re-anchors q0_rel by one tooth pitch per
        tick (its wrap-safe locking scheme), which exactly cancels the
        unwind angle() would report when a spring and an escapement share
        one joint (caught by the Phase-3 capstone's wind-down gate: the
        spring read zero unwind over 27 real ticks). The raw twist depends
        only on the two bodies' actual orientations — nothing any
        bookkeeping does to the joint's reference can touch it.
    No separate accumulator duplicates the physics — the unwrap tracker is
    fed the bodies' own state each call, so it can never drift from what
    the joint actually reports."""

    def __init__(self, joint, k, theta_wound0):
        self.joint = joint
        self.k = float(k)
        self.theta_wound0 = float(theta_wound0)
        self._theta0 = self._raw_twist()
        self._prev_raw = self._theta0
        self._unwrapped = self._theta0

    def _raw_twist(self) -> float:
        """The joint's relative twist about its own axis, measured directly
        from the two bodies' orientations — independent of q0_rel."""
        return twist_angle(self.joint._q_rel(), self.joint.axis_a)

    def _sync(self) -> None:
        """Advance the unwrapped angle by the WRAPPED delta since the last
        call, correcting only when that delta exceeds pi — valid as long as
        consecutive calls are close in time (true for one call/physics
        step; the fastest speed in this module's own gate test moves the
        joint by ~0.007 rad/step, far under the pi-radian assumption)."""
        raw = self._raw_twist()
        d = raw - self._prev_raw
        if d > math.pi:
            d -= 2.0 * math.pi
        elif d < -math.pi:
            d += 2.0 * math.pi
        self._unwrapped += d
        self._prev_raw = raw

    def unwound(self) -> float:
        """How far (rad, >= 0) the joint has turned since full wind."""
        self._sync()
        return abs(self._unwrapped - self._theta0)

    def theta_wound(self) -> float:
        """Remaining wound angle (rad, >= 0) — floored at 0: a real spring
        cannot overwind past empty or drive the train backward once spent."""
        return max(0.0, self.theta_wound0 - self.unwound())

    def torque(self) -> float:
        return self.k * self.theta_wound()

    @property
    def wound_out(self) -> bool:
        return self.theta_wound() <= 0.0

    def drive(self, direction: float = -1.0) -> None:
        """Update the joint's motor for THIS instant. Call once per outer
        step, before dynamics.stepper.step(). ``direction`` only needs to
        stay CONSTANT across calls (which way "unwinding" is is arbitrary —
        see joints.py's b=None sign convention); it must never flip mid-run,
        or the wheel would be driven back toward full wind."""
        sign = 1.0 if direction >= 0 else -1.0
        self.joint.motor_speed = sign * HUGE_SPEED
        self.joint.motor_max_torque = self.torque()
