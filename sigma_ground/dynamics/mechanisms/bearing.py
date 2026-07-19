"""RigidBearing — an ideal rigid bearing: constrains a rotor body to pure
twist about a fixed world axis through a fixed point, by projection.

Orchestration-level, same layer as Escapement (no new solver math): call
``project()`` once per outer step, after dynamics.stepper.step(). It
projects the body's orientation onto its twist component about the axis,
zeroes transverse spin, and re-pins the CoM to the anchor.

WHY this exists instead of a RevoluteJoint: mounting a wind-loaded rotor
on a RevoluteJoint exposed an exponential instability in the joint's
swing-correction loop under ORIENTATION-COUPLED external torque (the wind
torque responds to axis tilt; the SHAKE/RATTLE swing rows correct with a
one-substep delay; the loop gain exceeds 1 in a mid-omega band). Isolated
decisively: the identical aero model on a hand-projected "perfect bearing"
spins up monotonically to the closed-form terminal speed (0 drops over
4800 steps), while every joint-mounted variant tumbles regardless of dt
(1/960..1/3840) or solver iterations (10..30). Documented as a
# PHYSICS_GAP in misc/KNOWN_GAPS.md with an xfail regression test
(tests/test_mechanism_wind.py) so fixing the solver loop un-xfails it.

NOT_PHYSICS, stated plainly: ideal-constraint projection (a numerical
choice, same doctrine as collision.py's slop/beta constants). It is also
the HONEST abstraction for this machine: a real nacelle holds its shaft in
a stiff bearing assembly — far closer to an ideal constraint than to a
marginally-corrected point hinge. The energy the projection removes
(transverse KE, second-order small in the tilt error) is ACCOUNTED, not
hidden: ``absorbed_energy_j`` accumulates it, and the gates assert it
stays negligible relative to the spin energy.
"""
import math

from ..vec import Vec3
from ..quat import twist_angle, quat_from_axis_angle


class RigidBearing:
    def __init__(self, parcel, anchor_world, axis_world):
        self.parcel = parcel
        self.anchor = Vec3(anchor_world.x, anchor_world.y, anchor_world.z)
        l = axis_world.length()
        self.axis = Vec3(axis_world.x / l, axis_world.y / l, axis_world.z / l)
        self.absorbed_energy_j = 0.0

    def angle(self) -> float:
        """Twist about the bearing axis, wrapped (-pi, pi]."""
        return twist_angle(self.parcel.orientation,
                           (self.axis.x, self.axis.y, self.axis.z))

    def omega(self) -> float:
        """Signed spin rate about the bearing axis."""
        return self.parcel.angular_velocity.dot(self.axis)

    def project(self) -> None:
        p = self.parcel
        ke_before = p.rotational_ke()
        # orientation -> its twist component about the axis
        th = self.angle()
        p.orientation = list(quat_from_axis_angle(
            (self.axis.x, self.axis.y, self.axis.z), th))
        # angular velocity -> axial component only (ledger the removed KE)
        w = p.angular_velocity
        p.angular_velocity = self.axis * w.dot(self.axis)
        self.absorbed_energy_j += max(0.0, ke_before - p.rotational_ke())
        # CoM pinned at the anchor
        p.position = Vec3(self.anchor.x, self.anchor.y, self.anchor.z)
        p.velocity = Vec3(0.0, 0.0, 0.0)


__all__ = ["RigidBearing"]
