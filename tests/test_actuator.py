"""OscillatingRevoluteActuator gates -- standalone, disconnected from any
tool/mechanism demo, per this project's risk-first discipline for a new
orchestration primitive."""
import math

import pytest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.parcel import PhysicsParcel
from sigma_ground.dynamics.scene import PhysicsScene
from sigma_ground.dynamics.joints import RevoluteJoint
from sigma_ground.dynamics.stepper import step
from sigma_ground.dynamics.mechanisms.actuator import OscillatingRevoluteActuator
from sigma_ground.kernel.shapes import Box


class _ConstDensity:
    def density_at_sigma(self, s):
        return 2700.0


def _leaf_scene(limits=(-0.4, 0.4)):
    leaf = PhysicsParcel(Box(0.02, 0.1, 0.02), _ConstDensity(),
                         position=Vec3(0.0, 0.0, 0.0))
    axis = Vec3(0.0, 0.0, 1.0)
    joint = RevoluteJoint(leaf, None, leaf.position, axis, limits=limits)
    scene = PhysicsScene([leaf], gravity=Vec3(0.0, 0.0, 0.0), ground=False,
                         constraints=[joint])
    return scene, joint


def test_oscillates_between_the_set_limits_for_many_cycles():
    scene, joint = _leaf_scene()
    joint.motor_max_torque = 5.0
    act = OscillatingRevoluteActuator(joint, speed_rad_s=6.0)
    dt = 1.0 / 960.0
    lo, hi = joint.limits
    seen_min, seen_max = 0.0, 0.0
    t = 0.0
    while t < 6.0 and act.reversals < 12:
        step(scene, dt=dt, sub_steps=1)
        act.step(t)
        th = joint.angle()
        seen_min, seen_max = min(seen_min, th), max(seen_max, th)
        t += dt
    assert act.reversals >= 12                        # genuinely cycled, not stuck
    # never blew through either wall (limits are the hard geometric bound)
    assert seen_min >= lo - 0.05
    assert seen_max <= hi + 0.05
    # actually swept close to both ends, not just twitching in the middle
    assert seen_min <= lo + 3 * act.eps_rad
    assert seen_max >= hi - 3 * act.eps_rad


def test_rejects_a_joint_with_no_limits():
    scene, joint = _leaf_scene()
    joint.limits = None
    with pytest.raises(ValueError):
        OscillatingRevoluteActuator(joint, speed_rad_s=1.0)


def test_rejects_an_inverted_limit_window():
    scene, joint = _leaf_scene(limits=(0.4, -0.4))
    with pytest.raises(ValueError):
        OscillatingRevoluteActuator(joint, speed_rad_s=1.0)


def test_cycle_period_is_consistent_with_speed_and_span():
    """A rough closed-form sanity check: with the motor torque high enough to
    dominate inertia quickly, one half-cycle's wall-clock time should be on
    the order of span/speed (the time to sweep the window at the commanded
    rate), not wildly off (e.g. off by a factor >3, which would mean the
    actuator is not really driving the joint)."""
    scene, joint = _leaf_scene()
    joint.motor_max_torque = 50.0                      # torque-rich: near-ideal tracking
    speed = 4.0
    act = OscillatingRevoluteActuator(joint, speed_rad_s=speed)
    lo, hi = joint.limits
    # the joint starts at angle() == 0 (construction pose), driving toward
    # hi first -- so the FIRST leg only covers hi - 0, not the full span
    expected_half_period = (hi - joint.angle()) / speed
    dt = 1.0 / 960.0
    t = 0.0
    first_reversal_t = None
    while t < 4.0 and act.reversals < 1:
        step(scene, dt=dt, sub_steps=1)
        act.step(t)
        t += dt
        if act.reversals >= 1 and first_reversal_t is None:
            first_reversal_t = t
    assert first_reversal_t is not None
    assert first_reversal_t == pytest.approx(expected_half_period, rel=0.5)
