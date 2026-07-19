"""ReciprocatingPumpState -- positive-displacement volume bookkeeping for a
piston driven by a PrismaticJoint. Orchestration-level, same layer as
MainspringState/Escapement/RigidBearing/BearingGearCoupling: no new solver
math, just Python reading PrismaticJoint.travel() once per outer step.

PrismaticJoint.travel() is READ-ONLY here (nothing writes a force back) --
unlike BearingGearCoupling this can't corrupt any energy ledger; it's a pure
observer.

The one real subtlety: naively integrating dV += A*d(travel) over a FULL
back-and-forth cycle nets to ~0 (the piston returns to where it started). A
real reciprocating pump has check valves -- volume accumulates on only ONE
stroke per cycle. Fixed the same way Escapement detects a tooth release
(dynamics/mechanisms/escapement.py: sign of consecutive angle deltas): watch
for travel() to stop increasing and start decreasing (a local MAXIMUM --
the discharge stroke's end), and add one full stroke's swept volume
(piston_area * stroke_length) exactly once per detected reversal.

SIMPLIFIED_MODEL, stated plainly: no fluid inertia, viscosity, or back-
pressure, and no load feeds back onto the crank (this stays a pure
observer, same discipline as wind.py's flat-plate flag). See KNOWN_GAPS.md.
"""


class ReciprocatingPumpState:
    def __init__(self, joint, piston_area_m2, stroke_length_m):
        self.joint = joint
        self.piston_area_m2 = float(piston_area_m2)
        self.stroke_length_m = float(stroke_length_m)
        self.volume_m3 = 0.0
        self.strokes = 0
        self._prev_travel = joint.travel()
        self._prev_delta = 0.0

    def step(self) -> None:
        t = self.joint.travel()
        delta = t - self._prev_travel
        if self._prev_delta > 0.0 and delta < 0.0:      # local max: one full stroke
            self.volume_m3 += self.piston_area_m2 * self.stroke_length_m
            self.strokes += 1
        if delta != 0.0:
            self._prev_delta = delta
        self._prev_travel = t


__all__ = ["ReciprocatingPumpState"]
