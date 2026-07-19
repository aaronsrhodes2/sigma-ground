"""BearingGearCoupling -- drives a RevoluteJoint's motor from a RigidBearing's
own spin rate each step. Orchestration-level, same layer as MainspringState/
Escapement/RigidBearing: no new solver math, just Python reading/writing
existing joint fields once per outer step.

WHY this exists instead of a GearCouplingJoint: GearCouplingJoint enforces
its ratio between two RevoluteJoints' own solver rows (dynamics/joints.py,
GearCouplingJoint._solve). A wind rotor lives on a RigidBearing, not a
RevoluteJoint (see bearing.py's own docstring for why) -- it has no solver
row for a gear constraint to attach to at all. This class is the bridge:
read the bearing's PHYSICAL spin rate (RigidBearing.omega(), no joint-
convention sign flip involved) and set the follower's motor_speed toward
`ratio * bearing.omega()` each step, with the sign flip a RevoluteJoint
built with b=None always needs (established this session, record_motor_spin:
motor_speed=-omega drives the body's own world-frame spin to +omega).

# PHYSICS_GAP, stated plainly (see KNOWN_GAPS.md): this coupling is
LOAD-BLIND. The follower's own motor manufactures whatever torque it needs
(up to its cap) to track the commanded rate -- nothing subtracts that
torque from the bearing side, so the rotor spins up to the exact same
closed-form terminal speed whether or not anything downstream is "loaded".
An energy-honest version would feed a reaction torque back through
RigidBearing.project() (which already discards transverse KE but has no
path to ACCEPT an axial reaction) -- real new solver-composition work, not
built here. Ship this simplification measured, not silent: every recorder
using this class asserts the rotor still hits the standalone closed-form
terminal_omega under load, proving the claim rather than merely stating it.
"""


class BearingGearCoupling:
    def __init__(self, bearing, follower_joint, ratio, motor_max_torque=1.0):
        self.bearing = bearing
        self.follower_joint = follower_joint
        self.ratio = float(ratio)
        follower_joint.motor_max_torque = float(motor_max_torque)

    def step(self) -> None:
        self.follower_joint.motor_speed = -self.ratio * self.bearing.omega()


__all__ = ["BearingGearCoupling"]
