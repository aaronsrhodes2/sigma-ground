"""Escapement — releases an escape-wheel RevoluteJoint by exactly one tooth
pitch each time an oscillator (pendulum/foliot/balance) RevoluteJoint's
angle crosses zero, gating the going train's free rotation into a real,
periodic tick.

Built ENTIRELY from existing joints.py primitives — no new solver math.
The escape wheel is normally LOCKED at its joint's zero reference: its
``limits`` window pins angle()==0 as a hard floor (or ceiling, per
``direction``), so RevoluteJoint's own brake-only limit rows (already gated
by tests/test_joints.py::test_hinge_limit_settles_within_half_degree) hold
it dead still under the spring's motor torque. On each oscillator
zero-crossing this class RE-ANCHORS the joint's zero: it rotates
``q0_rel`` (the joint's construction-pose reference) by one tooth pitch in
the travel direction, so the parked wheel suddenly reads one pitch AWAY
from the stop, accelerates under its motor torque, and the SAME limit row
brakes it at the new zero — locking, releasing and relocking are all just
consequences of the existing RevoluteJoint machinery.

WHY re-anchoring instead of a walking stop value: angle() wraps to
(-pi, pi] (twist_angle's own convention). A stop that walks monotonically
crosses the wrap seam after ~pi/pitch ticks, after which the limit
comparison can never fire again and the wheel silently FREE-RUNS — a real
bug caught when the full clock's minute hand ran ~145x fast while every
gear coupling held exactly (the diagnostic that isolated it) and the
escape wheel measured 4.3 rad/s instead of parking. Re-anchoring keeps the
working angle bounded in [0, pitch] forever, so there is no seam to cross.
(Rotations about the joint's own axis commute, so composing q0_rel with a
pitch twist shifts angle() by exactly that pitch.)

``travel_rad`` integrates the wheel's ACTUAL angle deltas (wrap-corrected,
re-anchor shifts excluded), so tooth accounting in tests is measured from
real motion, not inferred from the tick count.

SCOPE, stated plainly:
  - This is the classic ANCHOR/DEADBEAT escapement's cadence — one tooth
    released per oscillator HALF-PERIOD. It is NOT the lever escapement's
    double-impulse-per-tooth mechanics (Kelly's CTF2E/tfe beats/hour
    formula is lever-specific; an anchor escapement's beats/hour is
    escape_wheel_turns_per_hour * teeth, no factor of 2).
  - This does NOT model a sustaining "kick" back to the oscillator. A real
    pendulum needs one to counter air/pivot friction; this simulator's
    pendulum joint already conserves energy to <0.5%/period
    (test_joints.py), so there is no friction here to counter — an honest
    simplification given what the solver actually models, not a hidden gap.
"""
import math

from ..quat import quat_mul, quat_normalize, quat_from_axis_angle


class Escapement:
    def __init__(self, escape_joint, oscillator_joint, teeth: int, direction: float = -1.0):
        self.escape_joint = escape_joint
        self.oscillator_joint = oscillator_joint
        self.teeth = int(teeth)
        self.pitch = 2.0 * math.pi / self.teeth
        self.direction = 1.0 if direction >= 0 else -1.0
        # re-anchor so the CURRENT pose reads angle()==0: the initial stop
        escape_joint.q0_rel = quat_normalize(escape_joint._q_rel())
        self._apply_limits()
        self._prev_osc_angle = oscillator_joint.angle()
        self._prev_raw = escape_joint.angle()
        self.travel_rad = 0.0
        self.ticks = 0
        self.tick_times = []

    @property
    def locked_angle(self) -> float:
        """The stop, in the joint's CURRENT (re-anchored) frame — always 0:
        the wheel is braked wherever angle() reads zero."""
        return 0.0

    def _apply_limits(self):
        if self.direction < 0:
            self.escape_joint.limits = (0.0, 1.0e9)     # floor at the anchor
        else:
            self.escape_joint.limits = (-1.0e9, 0.0)    # ceiling at the anchor

    def _rebase(self):
        """Shift the joint's zero by one pitch in the travel direction: the
        parked wheel (angle 0) now reads -direction*pitch and is free to
        travel back to 0, where the unchanged limit row brakes it."""
        j = self.escape_joint
        r = quat_from_axis_angle(j.axis_a, self.direction * self.pitch)
        j.q0_rel = quat_normalize(quat_mul(j.q0_rel, r))

    def step(self, t_sim: float) -> bool:
        """Call once per outer step, AFTER dynamics.stepper.step() has
        advanced the oscillator. Returns True on a tick (release) this call."""
        # integrate REAL wheel motion (wrap-corrected) before any re-anchor
        raw = self.escape_joint.angle()
        d = raw - self._prev_raw
        if d > math.pi:
            d -= 2.0 * math.pi
        elif d < -math.pi:
            d += 2.0 * math.pi
        self.travel_rad += d
        self._prev_raw = raw

        cur = self.oscillator_joint.angle()
        prev = self._prev_osc_angle
        self._prev_osc_angle = cur
        crossed = (prev < 0.0) != (cur < 0.0)
        if not crossed:
            return False
        self._rebase()
        self._prev_raw = self.escape_joint.angle()   # exclude the anchor shift
        self.ticks += 1
        self.tick_times.append(t_sim)
        return True
