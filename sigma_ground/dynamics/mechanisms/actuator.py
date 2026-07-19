"""OscillatingRevoluteActuator -- drives a limited RevoluteJoint back and
forth between its own ``limits`` window, reversing the motor at each end.

Built entirely from existing joints.py primitives, same discipline as
Escapement: RevoluteJoint's limit rows are brake-only (inelastic stop, no
restitution) -- they hold the joint AT a boundary under a driving motor but
never reverse it. Reversal is orchestration, not new solver math: this class
just flips ``motor_speed``'s sign once the joint's own ``angle()`` reads at
or past either bound, exactly the "read state, mutate a plain attribute"
pattern BearingGearCoupling and MainspringState already use.

General-purpose: any RevoluteJoint with a real ``limits`` window can be
driven this way -- pliers/tongs/scissors jaws, a valve, a lever -- this
module has no tool-specific knowledge.
"""
from __future__ import annotations


class OscillatingRevoluteActuator:
    """Reversing motor driver for one RevoluteJoint.

    ``joint.limits`` must already be set (lo, hi), lo < hi, both finite.
    Starts driving toward ``hi``. Call ``.step()`` once per outer loop,
    AFTER ``dynamics.stepper.step()`` has advanced the joint -- mirrors
    Escapement's own calling convention.

    ``eps_rad`` is the boundary-approach tolerance: the limit rows are a
    Gauss-Seidel brake, not an exact clamp, so ``angle()`` may sit a small
    residual short of the true bound under load; reversing only once within
    ``eps_rad`` avoids a reversal that never triggers on tiny numerical
    shortfall.
    """

    def __init__(self, joint, speed_rad_s: float, eps_rad: float = 0.02):
        if joint.limits is None:
            raise ValueError("OscillatingRevoluteActuator needs joint.limits set")
        lo, hi = joint.limits
        if not (lo < hi):
            raise ValueError(f"limits must satisfy lo < hi, got {joint.limits}")
        self.joint = joint
        self.speed_rad_s = abs(float(speed_rad_s))
        self.eps_rad = float(eps_rad)
        self._dir = 1.0                              # +1 → driving toward hi
        joint.motor_speed = self.speed_rad_s
        self.cycles_completed = 0.0
        self.reversals = 0

    def step(self, t_sim: float = None) -> bool:
        """Advance the actuator's own state; returns True on a reversal this
        call. Call once per outer loop after the physics step."""
        lo, hi = self.joint.limits
        th = self.joint.angle()
        if self._dir > 0.0 and th >= hi - self.eps_rad:
            self._dir = -1.0
            self.joint.motor_speed = -self.speed_rad_s
            self.reversals += 1
            self.cycles_completed += 0.5
            return True
        if self._dir < 0.0 and th <= lo + self.eps_rad:
            self._dir = 1.0
            self.joint.motor_speed = self.speed_rad_s
            self.reversals += 1
            self.cycles_completed += 0.5
            return True
        return False


__all__ = ["OscillatingRevoluteActuator"]
