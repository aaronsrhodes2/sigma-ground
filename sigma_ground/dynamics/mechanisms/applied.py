"""The closed taxonomy of artificial forces — Golden Rule 7's whole vocabulary.

Rule 7 says only artificial forces may be adjusted during a simulation. The
Captain's follow-up question (2026-07-30): with that rule, the set of
artificial forces we would ever wish to apply must be FINITE — define them
once, so a new application is data flowing through one pipeline, never a new
hand-coded verb.

Classical mechanics guarantees the set is closed. Everything an operator's
hand can do to a rigid body is a wrench (a force + a torque) delivered on
some time profile, a kinematic constraint, or a non-mechanical power
injection. That yields exactly SIX kinds:

  IMPULSE — an instantaneous wrench: Δmomentum (and, applied off the centre
            of mass or with a twist component, Δangular-momentum) delivered
            at a point at one instant. The push, the shove, the knock, the
            strike, the spin-up, the tip. duration == 0 by definition.
  FORCE   — a sustained wrench: force and/or torque applied over a time
            window [t0, t0+duration]. The press, the pull, the steady wind,
            the thrust, the held twist.
  DRIVE   — a servo at a joint: a velocity target with a torque cap, or a
            pure torque, maintained by feedback. The motor. (This is the
            existing drive Plug, absorbed into the taxonomy unchanged.)
  HOLD    — an artificial kinematic constraint: pinning matter against the
            world where no modeled structure holds it. The existing support
            Plug / world-weld, absorbed unchanged. Its "force" is whatever
            reaction the constraint demands — disclosed precisely because
            nature didn't provide the bracket.
  SOURCE  — non-mechanical injection: power delivered into matter that is
            not a wrench at all. Heat (a plugged torch, W into a body or
            cell) first; current/light later. The thermal sibling of DRIVE.
  TILT    — re-orienting the scene relative to gravity over a window: the
            operator's hands tilting the whole tray ("tilt the table until
            the plate slides"). Admitted by the Captain (2026-07-30,
            "orientation to gravity") through exactly the doctrine-amendment
            process this module prescribes. The DIRECTION of gravity
            relative to the scene is an operator action — nobody models the
            hand holding the tray — but its MAGNITUDE stays a world-fact
            (Moon vs Earth is a between-runs Choice, not a slider).

The Captain's candidate list (2026-07-30) — "movement, heat, rotation,
orientation, location, gravity, time" — sorts into a TRICHOTOMY, and the
sort is the doctrine:

  FORCES (the six above — adjustable DURING a run, always disclosed):
    movement    → IMPULSE / FORCE (linear wrench)
    rotation    → IMPULSE / FORCE with a twist component, or DRIVE
    heat        → SOURCE
    orientation-to-gravity → TILT (the hands tilting the tray)
  SETUP (initial conditions — edited BETWEEN runs, never mid-run):
    location    → placement is scenario setup; "teleport it" mid-run would
                  silently break the energy/momentum accounting Phase A
                  ships in every frame, so it is deliberately NOT a force
    orientation-as-pose → how a body starts oriented is placement too;
                  only re-aiming GRAVITY mid-run is a force (TILT)
    gravity MAGNITUDE → a world-fact Choice (Moon vs Earth), corrected
                  between runs like any other fact
  OBSERVATION (always free, touches no physics):
    time        → scrubbing/pausing playback is looking, not acting; the
                  recorded frames are immutable. (Re-running from an
                  earlier state is a new run — setup again.)

Nothing else exists to apply. Changing the ambient temperature is not a
force either — a world-fact (Choice), governed by Rule 7's other half:
corrected between runs, never dialed during one.

Every AppliedForce IS a plug (kind + description + variables) — the
disclosure and the specification are the same object, so an artificial
input cannot exist undisclosed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# kind -> the parameters that kind requires (beyond the common ones).
# THIS dict is the closed set. Adding a sixth kind is a doctrine decision
# (amend Golden Rule 7's commentary), not a routine code change.
KINDS = {
    "impulse": {"vector"},           # N*s   (linear); optional twist: N*m*s
    "force":   {"vector", "duration"},   # N, s; optional twist: N*m
    "drive":   {"joint"},            # motor_speed_rad_s+torque cap | torque_nm
    "hold":    {"point"},            # world anchor
    "source":  {"power_w", "duration"},  # W into the target
    "tilt":    {"axis", "angle_deg", "duration"},  # re-orient gravity vs the scene
}


@dataclass
class AppliedForce:
    """One artificial input to a simulation — spec and disclosure in one.

    Common fields:
      kind      one of KINDS
      target    the body label it acts on ("plate", "Doughnut")
      label     short human phrasing for the HUD ("the push")
    Mechanical fields (impulse/force):
      vector    [x, y, z] — N*s for impulse, N for force
      twist     [x, y, z] torque component — N*m*s / N*m; default none
      point     [x, y, z] application point; None = centre of mass (a pure
                wrench through the CoM produces no rotation; an off-CoM
                point is how "tip it" differs from "push it")
    Timing:
      t0        seconds into the simulation (default 0 = at the start)
      duration  seconds; impulse is 0 by definition
    DRIVE/HOLD carry their existing plug parameters via ``variables``.
    SOURCE carries power_w. TILT carries axis + angle_deg in ``variables``
    (the target is the whole scene — target "world").
    """
    kind: str
    target: str
    label: str = ""
    vector: list | None = None
    twist: list | None = None
    point: list | None = None
    t0: float = 0.0
    duration: float = 0.0
    power_w: float | None = None
    variables: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(
                f"unknown artificial-force kind {self.kind!r} — the set is "
                f"CLOSED: {sorted(KINDS)}. A new kind is a Golden Rule 7 "
                f"doctrine amendment, not a parameter.")
        if self.kind == "impulse":
            if self.duration:
                raise ValueError("an impulse is instantaneous by definition "
                                 "(duration must be 0); a lasting wrench is "
                                 "kind='force'")
            if not self.vector and not self.twist:
                raise ValueError("an impulse needs a vector and/or a twist")
        if self.kind == "force" and not self.vector and not self.twist:
            raise ValueError("a force needs a vector and/or a twist")
        if self.kind == "source" and not self.power_w:
            raise ValueError("a source needs power_w")
        if self.kind == "tilt" and "angle_deg" not in self.variables:
            raise ValueError("a tilt needs variables['angle_deg'] (and "
                             "optionally 'axis', default the push-transverse "
                             "horizontal)")

    # ── the plug disclosure: same object, same JSON, no second bookkeeping ──
    def to_plug(self) -> dict:
        v = dict(self.variables)
        if self.vector is not None:
            v["vector"] = [round(float(x), 6) for x in self.vector]
        if self.twist is not None:
            v["twist"] = [round(float(x), 6) for x in self.twist]
        if self.point is not None:
            v["point"] = [round(float(x), 6) for x in self.point]
        if self.power_w is not None:
            v["power_w"] = float(self.power_w)
        if self.duration:
            v["duration_s"] = float(self.duration)
        if self.t0:
            v["t0_s"] = float(self.t0)
        return {"kind": self.kind, "target": self.target,
                "description": self.label or f"{self.kind} on the {self.target}",
                "variables": v, "adjustable": True}

    @classmethod
    def from_dict(cls, d: dict) -> "AppliedForce":
        v = dict(d.get("variables") or {})
        return cls(kind=d["kind"], target=d["target"],
                   label=d.get("description", d.get("label", "")),
                   vector=v.pop("vector", d.get("vector")),
                   twist=v.pop("twist", d.get("twist")),
                   point=v.pop("point", d.get("point")),
                   t0=float(v.pop("t0_s", d.get("t0", 0.0))),
                   duration=float(v.pop("duration_s", d.get("duration", 0.0))),
                   power_w=v.pop("power_w", d.get("power_w", None)),
                   variables=v)


__all__ = ["AppliedForce", "KINDS"]
