"""Region protocol + the first adapter: StackSimRegion (engine phase E1).

A Region is the seam between the ECS (cold identity/provenance) and ONE
physics backend (hot state). The adapter holds and steps the ACTUAL backend
object — StackSim is wrapped VERBATIM, physics stays in one place (the
plan's non-negotiable: no second integrator, no mirrored physics state).
The golden-replay gate (test_region_stacksim.py) proves it: the same
scenario through the engine and through bare StackSim must be
byte-identical, and anything less means this layer perturbed physics.

Golden Rule 7 routing: every world perturbation arrives as an AppliedForce
from the CLOSED taxonomy (dynamics.mechanisms.applied.KINDS) and is routed
onto the backend's own entry points. Kinds the backend has no state for
(SOURCE — thermal; DRIVE — joints) are REFUSED honestly with the reason,
matching the /live loop's established behaviour — never faked.

DESIGN CALLS (disclosed):
* Single module (protocol + adapter together) — a ``regions/`` package
  waits for E5's second adapter; one adapter does not justify a package.
* ``STACK_LIVE_KINDS`` deliberately DUPLICATES serve.py's LIVE_KINDS rather
  than importing it: a web server must never enter a region's import graph
  (serve.py carries a grandfathered tier-4 edge). E2's cockpit unifies them.
* extract() returns dynamics.seeding.SeedBody verbatim (tier 1, frozen,
  plain floats, already consumed by to_nbody/to_continuum — free E5 spawn
  compatibility) instead of a new neutral type. Its radius_m is the bbox
  half-diagonal — a disclosed Deduction: the bounding-sphere analogue of a
  box body. The support relation and the dx/dz decomposition are
  deliberately NOT neutral state; full-fidelity reads stay on the held
  ``.sim``.
* apply() with kind="tilt" takes entity_id=None: StackSim.apply_tilt is
  scene-global (the operator's hands tilting the whole tray — there is no
  per-body target in the backend, and pretending otherwise would invent
  physics the model does not have).
* Body 0 is the base (the floor reference): never a force target, matching
  record_push_off's "the base cannot be pushed" and the /live loop's base
  refusal. Reading it (extract) is always allowed — observation is free.
"""
from __future__ import annotations

import math
from typing import Protocol

from ..dynamics.mechanisms.applied import AppliedForce, KINDS
from ..dynamics.seeding import SeedBody
from ..radiance.push_off import StackSim
from .ecs import Binding, EntityId, GameTags, Geometry, World, \
    matter_from_construct

# The kinds this backend has live state for. Deliberately duplicated from
# radiance.web.serve's LIVE_KINDS, NOT imported — see the module docstring.
STACK_LIVE_KINDS = frozenset({"impulse", "force", "hold", "tilt"})


class RegionRefusal(RuntimeError):
    """Honest backend-capability refusal — the exception form of the /live
    loop's ``{"type": "refusal"}`` message.

    Reserved for capability honesty ("this backend cannot represent that").
    Caller errors stay TypeError/ValueError; dead handles stay
    StaleEntityError. The three never blur: a refusal is the backend being
    honest about physics, not the caller being wrong about types.
    """


class Region(Protocol):
    """The Region contract — every backend adapter (E5's orbital/SPH included)
    presents exactly this face. Zero runtime cost (typing.Protocol).
    """

    def step(self, dt: float) -> None:
        """Advance the backend by one fixed step (fixed dt is the
        determinism contract — a Region may refuse a changed dt)."""
        ...

    def apply(self, entity_id: "EntityId | None", applied_force) -> None:
        """Route one closed-taxonomy AppliedForce onto the backend's own
        entry points. ``entity_id`` is None ONLY for scene-scoped kinds
        (tilt); a body handle otherwise.

        CONTRACT (binding on E2's WorldHost): apply() is SYNCHRONOUS at a
        step boundary — no queueing, no reordering. StackSim.apply_force
        computes t_end = t + duration and apply_tilt captures t0 = t at
        CALL time; deferral would decouple command time from sim time and
        break replay byte-identity.
        """
        ...

    def extract(self, entity_id) -> "SeedBody":
        """Engine-neutral state read: plain floats, no engine types
        (dynamics.seeding.SeedBody — the established neutral format)."""
        ...

    def spawn(self, neutral) -> "EntityId":
        """Materialize a neutral body mid-run, where the backend genuinely
        supports it. Backends that cannot raise RegionRefusal — honesty is
        part of the contract, not an optional extra."""
        ...


class StackSimRegion:
    """The StackSim adapter: wraps the planar stack integrator VERBATIM.

    ``self.sim`` is the actual StackSim, held as a plain public attribute
    (house style — no property ceremony). Mutating it around apply() is a
    Golden Rule 7 violation (an undisclosed world perturbation); the golden
    replay gate is the tripwire that would catch a layer doing so.
    """

    BACKEND = "stacksim"

    def __init__(self, world: World, built, *, mu_lookup, e_lookup):
        # The VERBATIM constructor call — the only construction logic.
        self.sim = StackSim(built, mu_lookup=mu_lookup, e_lookup=e_lookup)
        self.world = world
        eids = []
        for i, (name, construct) in enumerate(built):
            eids.append(world.create(
                name,
                matter_from_construct(name, construct),
                geometry=Geometry(construct, construct.bbox),
                binding=Binding(self.BACKEND, i),
                tags=GameTags()))
        self.entity_ids = tuple(eids)       # index-aligned with sim bodies
        self._eids = frozenset(eids)
        # The routing table over the WHOLE closed taxonomy. The assert is
        # doctrine: if a 7th kind ever enters applied.KINDS, this adapter
        # fails LOUDLY at construction instead of silently ignoring it.
        self._route = {
            "impulse": "sim.apply_impulse — dv from variables"
                       "['push_speed_m_s'] (the wire convention), else the "
                       "Deduction vector[0]/mass",
            "force":   "sim.apply_force — fx_n = vector[0], duration verbatim",
            "hold":    "sim.hold — on from variables['on'], default True",
            "tilt":    "sim.apply_tilt — angle_deg + duration; scene-scoped "
                       "(entity_id=None)",
            "source":  "REFUSED — no thermal state in this planar stack model",
            "drive":   "REFUSED — no joint state in this planar stack model",
        }
        assert set(self._route) == set(KINDS), (
            "route table must cover the closed taxonomy exactly — "
            f"missing {set(KINDS) - set(self._route)}, "
            f"extra {set(self._route) - set(KINDS)}")
        self._dt = None                     # fixed-dt guard state

    # ── stepping ──────────────────────────────────────────────────────────
    def step(self, dt: float) -> None:
        """Exactly one ``sim.step(dt)`` — nothing before, nothing after,
        forever. The engine layer performs ZERO arithmetic on hot state;
        the guard below is control flow only (no float math)."""
        if not dt > 0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        if self._dt is None:
            self._dt = float(dt)
        elif float(dt) != self._dt:
            raise ValueError(
                f"fixed dt is the determinism contract: this region was "
                f"first stepped at dt={self._dt!r}; refusing dt={dt!r}")
        self.sim.step(dt)

    # ── the closed-taxonomy routing ───────────────────────────────────────
    def apply(self, entity_id, af) -> None:
        """Route one AppliedForce onto StackSim's own entry points.

        Check order is part of the contract (tests depend on it):
        type -> t0 -> live-kind -> tilt handling -> handle presence ->
        world.get (StaleEntityError propagates BEFORE region membership,
        so any stale handle dies as StaleEntityError) -> membership ->
        binding -> base body -> target label -> planar honesty -> dispatch.

        Dispatch is exact float pass-through — the byte-identity table.
        IMPULSE routing (disclosed): dv comes verbatim from
        variables["push_speed_m_s"] when present (the wire convention
        serve.py and _push_af established); the fallback vector[0]/mass is
        an honest Deduction but NOT bit-equal to an equivalent direct-dv
        call ((m*v)/m != v in floats) — replay-critical callers supply
        push_speed_m_s.
        """
        if not isinstance(af, AppliedForce):
            raise TypeError(
                f"apply() takes an AppliedForce, got {type(af).__name__} — "
                f"wire dicts go through AppliedForce.from_dict first "
                f"(routing never reads to_plug()'s rounded values)")
        if af.t0 != 0.0:
            raise RegionRefusal(
                "E1 apply is immediate — scheduled t0 arrives with E2's "
                "host loop; apply at the step boundary you mean")
        if af.kind not in STACK_LIVE_KINDS:
            noun = "thermal" if af.kind == "source" else "joint"
            raise RegionRefusal(
                f"{af.kind!r} is in the closed set, but this planar stack "
                f"model has no {noun} state — live kinds: "
                f"{sorted(STACK_LIVE_KINDS)}")
        if af.kind == "tilt":
            if entity_id is not None:
                raise RegionRefusal(
                    "tilt acts on the whole scene (the operator's hands "
                    "tilting the tray) — pass entity_id=None")
            if "axis" in af.variables:
                raise RegionRefusal(
                    "this backend has exactly one tilt axis — the "
                    "push-transverse horizontal; omit 'axis' (no axis "
                    "representation convention exists anywhere in the repo, "
                    "and accepting one here would invent it silently)")
            self.sim.apply_tilt(float(af.variables["angle_deg"]),
                                float(af.duration))
            return
        if entity_id is None:
            raise RegionRefusal(
                f"{af.kind!r} needs a body handle — only tilt is "
                f"scene-scoped")
        ent = self.world.get(entity_id)     # StaleEntityError propagates
        if entity_id not in self._eids:
            raise RegionRefusal(
                f"entity {ent.label!r} is not bound to this region")
        if ent.binding is None or ent.binding.backend != self.BACKEND:
            raise RegionRefusal(            # defensive — should be unreachable
                f"entity {ent.label!r} carries no {self.BACKEND!r} binding")
        idx = ent.binding.index
        if idx == 0:
            raise RegionRefusal(
                f"the {self.sim.names[0]} is the base — the floor "
                f"reference, never a force target")
        if af.target != self.sim.names[idx]:
            raise RegionRefusal(
                f"the AppliedForce is the disclosure — its target must name "
                f"the body it acts on (target={af.target!r}, body="
                f"{self.sim.names[idx]!r})")
        # Planar honesty: refuse what the model cannot represent — never
        # silently project it away.
        if af.twist is not None:
            raise RegionRefusal(
                "planar model: no rotational DOF — a twist is "
                "unrepresentable here")
        if af.point is not None:
            raise RegionRefusal(
                "this backend applies at the centre of mass only — an "
                "application point (off-CoM wrench, hold anchor) is "
                "unrepresentable; omit 'point'")
        if af.vector and (af.vector[1] or af.vector[2]):
            raise RegionRefusal(
                "planar +x model: no y/z DOF — off-plane vector components "
                "are refused, not silently projected")
        if af.kind == "impulse":
            if "push_speed_m_s" in af.variables:
                dv = float(af.variables["push_speed_m_s"])
            else:
                dv = float(af.vector[0]) / self.sim.mass[idx]
            self.sim.apply_impulse(idx, dv)
        elif af.kind == "force":
            if af.duration <= 0:
                raise RegionRefusal(
                    "a sustained wrench needs a positive window; an "
                    "instantaneous one is kind='impulse' — the backend "
                    "would silently no-op a zero-length force")
            self.sim.apply_force(idx, float(af.vector[0]),
                                 float(af.duration))
        else:                               # hold — the only kind left
            self.sim.hold(idx, bool(af.variables.get("on", True)))

    # ── neutral reads ─────────────────────────────────────────────────────
    def extract(self, entity_id) -> SeedBody:
        """Engine-neutral state of one body: plain floats, no engine types.

        Base body 0 IS extractable — reading is observation, always free
        (only forcing the base is refused). radius_m is the bbox
        half-diagonal (Deduction: bounding-sphere analogue of a box body).
        """
        ent = self.world.get(entity_id)     # StaleEntityError propagates
        if entity_id not in self._eids:
            raise RegionRefusal(
                f"entity {ent.label!r} is not bound to this region")
        if ent.binding is None or ent.binding.backend != self.BACKEND:
            raise RegionRefusal(
                f"entity {ent.label!r} carries no {self.BACKEND!r} binding")
        i = ent.binding.index
        s = self.sim
        (x0, x1), (y0, y1), (z0, z1) = s.bbox[i]
        half_diag = 0.5 * math.sqrt(
            (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        return SeedBody(
            mass_kg=float(s.mass[i]),
            position_m=(s.com[i][0] + s.dx[i], s.com[i][1],
                        s.com[i][2] + s.dz[i]),
            velocity_m_s=(s.vx[i], 0.0, s.vz[i]),
            radius_m=half_diag,
            label=s.names[i])

    # ── honest non-capability ─────────────────────────────────────────────
    def spawn(self, neutral) -> EntityId:
        """StackSim cannot spawn mid-run — refused honestly, never faked."""
        raise RegionRefusal(
            "StackSim's body set is fixed at construction — the support "
            "chain is index-based; mid-run spawn is not physically supported "
            "by this backend, and a rebuild-in-place would be a second "
            "integrator wearing a mask. Build a new stack instead. (Same "
            "honesty as the /live loop's source/drive refusal.)")


__all__ = ["STACK_LIVE_KINDS", "RegionRefusal", "Region", "StackSimRegion"]
