"""Plug — force provenance for artificial stand-ins (the force-side twin of
the Fact/citation system: Facts give every NUMBER provenance, Plugs give
every MOTION provenance; same no-silent-fabrication doctrine).

Doctrine (Captain, 2026-07-15): motion requires a mover. A gear in space
shouldn't move unless something is making it move. When a scene drives a
body with machinery that ISN'T modeled — "rotate the propeller as if a
20hp engine were attached, but without the engine" — that stand-in must be
flagged as plugged, with the substituted forces/variables briefly stated,
all the way into the rendered scene's own metadata.

What is and is not a plug:
  - A RevoluteJoint motor spinning a demo disc IS a plug (kind "drive"):
    the motor stands in for absent machinery (an engine, a hand, a river)
    that the scene does not model.
  - A world-anchored joint holding gears in space IS a plug (kind
    "support"): real gears would fall under gravity and fly apart without
    a frame; if no frame/plates are modeled, the invisible anchor is
    "held as if by a frame, without the frame" and must say so. "Nothing
    plugged" is only claimable when every mover AND every holder is
    either modeled or natural.
  - The clock's MainspringState is NOT a plug: the spring is the real,
    present mover, modeled as an energy store that winds down — its
    CONSTANTS are estimated (flagged as such), but the machinery itself is
    in the scene. Estimated parameters of present machinery are the Fact
    system's [estimated] lane; absent machinery is the Plug lane.
  - Wind pushing on blades is NOT a plug: it's a natural force computed by
    the physics (the preferred mode).

A drive Plug is LOAD-BEARING, not decorative: constructing one is what
actually configures the joint's motor (there is no way to describe a
drive plug without also being the code path that applies it, so the flag
can never silently drift from the physics). Support plugs annotate
constraints that already exist structurally (the world-anchored joint IS
the support), so they are declarations by construction.

Adjustability: plugged variables are artificial anyway, so they surface
as adjustable controls by default (``adjustable``, on by default for
drives) — realized as parameter sweeps of frozen recordings (scene-spec
``variants`` the viewer offers as a slider), never live physics in the
renderer.
"""
from .spring import HUGE_SPEED


class Plug:
    """An artificial stand-in for absent machinery: a driver (kind
    "drive") or a holder (kind "support" — build via ``Plug.support``).

    Drive modes, mirroring the two honest ways to substitute a mover:
      - speed plug (``motor_speed_rad_s`` + a generous torque cap): "spin it
        AT this rate, as the absent machinery would have";
      - torque plug (``torque_nm`` alone): "push it WITH this torque" — the
        motor targets an unreachable speed so the existing impulse clamp
        degenerates into a pure torque source (same trick, and same
        machinery, as MainspringState.drive()).
    """

    @classmethod
    def support(cls, description: str, **variables):
        """A support-kind plug: annotates an artificial HOLDER (a world-
        anchored joint or bearing standing in for an unmodeled frame,
        plate, mount...). The constraint itself already exists in the
        scene's joint list — this is its honest citation."""
        p = cls.__new__(cls)
        p.joint = None
        p.kind = "support"
        p.description = str(description)
        p.equivalent = str(variables.pop("equivalent", ""))
        p.adjustable = bool(variables.pop("adjustable", False))
        p.variables = dict(variables)
        if p.equivalent:
            p.variables["equivalent"] = p.equivalent
        return p

    def __init__(self, joint, description: str, *, motor_speed_rad_s=None,
                 motor_max_torque_nm=None, torque_nm=None, equivalent: str = "",
                 direction: float = -1.0, adjustable: bool = True):
        if (torque_nm is None) == (motor_speed_rad_s is None):
            raise ValueError("a Plug is EITHER a speed plug (motor_speed_rad_s"
                             " [+ motor_max_torque_nm]) OR a torque plug"
                             " (torque_nm) — exactly one")
        self.joint = joint
        self.kind = "drive"
        self.description = str(description)
        self.equivalent = str(equivalent)
        self.adjustable = bool(adjustable)
        self.variables = {}
        if torque_nm is not None:
            sign = 1.0 if direction >= 0 else -1.0
            joint.motor_speed = sign * HUGE_SPEED
            joint.motor_max_torque = float(torque_nm)
            self.variables["torque_nm"] = float(torque_nm)
        else:
            joint.motor_speed = float(motor_speed_rad_s)
            cap = float(motor_max_torque_nm if motor_max_torque_nm is not None
                        else 1.0)
            joint.motor_max_torque = cap
            self.variables["motor_speed_rad_s"] = float(motor_speed_rad_s)
            self.variables["motor_max_torque_nm"] = cap
        if equivalent:
            self.variables["equivalent"] = self.equivalent

    def to_dict(self) -> dict:
        """The scene-spec entry: goes into scene_spec["plugs"] so the flag
        travels with the render bundle, machine-readably."""
        return {"kind": self.kind, "description": self.description,
                "adjustable": self.adjustable,
                "variables": dict(self.variables)}

    def cite(self) -> str:
        """Human-readable one-liner for source strings."""
        vs = ", ".join(f"{k}={v}" for k, v in self.variables.items())
        tag = "PLUGGED" if self.kind == "drive" else "PLUGGED SUPPORT"
        return f"{tag}: {self.description}" + (f" ({vs})" if vs else "")


__all__ = ["Plug"]
