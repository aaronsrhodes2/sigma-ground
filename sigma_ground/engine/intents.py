"""The closed game-verb registry — E3's input layer, server-side reduction.

Golden Rule 7 IS the input layer: "All world perturbations — from any
source, including a game controller — map onto the closed AppliedForce
taxonomy (dynamics/mechanisms/applied.py: impulse/force/drive/hold/
source/tilt). An analog stick held down becomes a short-duration force
re-issued per input tick so a release decays naturally, never a new
seventh kind. Camera and UI state are engine commands that cannot touch
physics."

WHY the reduction lives SERVER-SIDE (this module, tier 3), not in the
browser: an honest impulse wire dict must disclose momentum J = m·v in its
vector (applied.py documents the impulse vector as N·s; push_off._push_af
is the established producer writing vector=[m·v0, 0, 0] plus
variables.push_speed_m_s; cockpit._apply_force documents that "the viewer
can't know J = m·v0 because matter lives server-side"). A client-side
reduction is therefore not merely drift-prone — it cannot be honest. One
closed verb set, one refusal vocabulary, one reduction; input.js maps
device chords to verbs and NEVER constructs force dicts.

Two channels, hard split:
* WORLD verbs reduce to canonical AppliedForce wire dicts (exact floats,
  host._canonical's conventions) and enter physics ONLY through
  host.submit → _apply_at_boundary → the command log — replay-exactness
  is inherited from E2, not re-proven here.
* ENGINE verbs reduce to protocol ops / client-local camera deltas. The
  EngineVerb dataclass has NO reduce field: the force path is structurally
  absent, not merely untested.

DISCLOSED Choices (design calls, each honest and revisable):
* PRESS_TICK_S = 1/60 s — one input tick's sustained-force window: exactly
  16 sim steps at host.SIM_DT = 1/960, so held re-issues at the 60 Hz cap
  tile without overlap-summing (StackSim._forces SUMS overlapping
  windows). Duration is deliberately NOT a client parameter: a client
  could otherwise claim longer windows than it actually held.
* pace-scaled press windows — FIXED IN E4: reduce(pace=host.pace) scales
  the press window (duration = PRESS_TICK_S * pace) so held re-issues tile
  in wall time at any pace — one 60 Hz wall tick advances pace/60 sim-s
  and the window now covers exactly that. The command log records the
  ACTUAL scaled duration, so replay is unaffected either way — the fix is
  FEEL, not physics. Residual (disclosed): sub-60fps clients re-issue at
  frame rate and still overlap-sum slightly; a pace change racing between
  reduce and boundary-apply scales by the reduce-time value (the log
  records what actually applied).
* shove speed range (0, 20] m/s — disclosed reuse of the frozen /apply
  route's knob (cockpit._apply_force's "in (0, 20]" check); the two ranges
  cross-reference each other and must move together.
* press force range [-50, 50] N nonzero — sized for tabletop-stack bodies.
* tilt range [-45, 45] deg nonzero, duration (0, 10] s — the sign IS the
  direction (one verb, never tilt_left/tilt_right closed-set inflation).
* Server-side rate limiting is deliberately ABSENT: the 60 Hz cap is
  client politeness (input.js), and the log records what actually applied
  — an uncapped flood applies more honest, logged force. E4's lane.
* cam_move / cam_mode / follow (R2) are client-local engine verbs
  (server_side=False — cockpit's generic not-server_side arm refuses
  them with the existing "client-local" message, zero cockpit edits).
  Their tuning constants (speed scale k, distance floor, step cap, hold
  window) live as disclosed Choices in radiance/web/camera.js, not here
  — the server has no camera. follow is deliberately PARAM-LESS: Param's
  validator is numeric-or-choices only, a free-string body name can't be
  expressed (and no static chord could usefully carry a scene-dependent
  body anyway) — named follow is the controller API
  (window._camctl.follow(label)); a string-Param extension is the
  documented E4 path if verb-level named follow is ever wanted.

The verb set is CLOSED (the applied.KINDS pattern): 14 members, and
magnitude/sign variants are binding DATA (input_bindings.json), never new
members. Adding a verb is a doctrine decision, not a parameter.

Regenerate the static mirror after ANY registry change:
    python -m sigma_ground.engine.intents
writes radiance/web/data/intent_verbs.json (committed, never hand-edited;
test_intents.test_intent_verbs_mirror_is_fresh is the tripwire).
"""
from __future__ import annotations

import math
import reprlib
from dataclasses import dataclass
from typing import Callable

from ..dynamics.mechanisms.applied import AppliedForce
from .host import PACE_MAX, PACE_MIN

RATE_CAP_HZ = 60                    # client-side intent cap (politeness)
PRESS_TICK_S = 1.0 / 60.0           # one input tick's force window — see docstring

# Refusal messages QUOTE the offending value, and the value is hostile
# input: an attacker controls its size and its nesting depth. A bounded,
# depth-limited repr keeps a refusal a refusal — an unbounded f"{v!r}" turns
# a 1 MB string (or a deeply nested object, whose repr recurses) into a
# 1 MB reply (or a RecursionError escaping reduce()).
_SHOW = reprlib.Repr()
_SHOW.maxlevel, _SHOW.maxstring, _SHOW.maxother = 4, 48, 48
_SHOW.maxdict, _SHOW.maxlist, _SHOW.maxtuple, _SHOW.maxlong = 4, 6, 6, 48
_show = _SHOW.repr


class IntentRefusal(ValueError):
    """Honest game-verb refusal — the caller asked for something the closed
    verb set / parameter schema cannot express.

    ValueError family per region.py's taxonomy note: a bad intent is the
    CALLER being wrong; RegionRefusal stays reserved for backend capability
    honesty. Over the wire both become {"type": "refusal"}. reduce()
    raises IntentRefusal for EVERY invalid input and nothing else (pinned
    by the fuzz test); host._apply_at_boundary's ValueError arm is defense
    in depth behind it.
    """


@dataclass(frozen=True)
class Param:
    """One declared verb parameter. default=None means REQUIRED (no shipped
    verb requires one — defaults everywhere is deliberate, so a bare chord
    with no params block still reduces; the validator enforces it anyway)."""
    name: str
    default: object = None          # None => required
    lo: "float | None" = None
    hi: "float | None" = None
    lo_open: bool = False           # lo is an OPEN bound (value must be > lo)
    nonzero: bool = False
    choices: "tuple | None" = None  # str-valued enum; passes through as str


@dataclass(frozen=True)
class WorldVerb:
    """A world-channel verb: reduces to a canonical AppliedForce wire dict."""
    name: str
    params: tuple
    doc: str
    continuous: bool                # True: re-issued while held (the cap tiles)
    needs_target: bool
    needs_mass: bool
    reduce: Callable                # (validated_params, target, mass_kg) -> dict
    tick_scaled: bool = False       # True: the per-tick window scales by the
    #                                 host pace at reduce time (press only —
    #                                 shove has no duration; tilt's duration
    #                                 is a USER-chosen sim-time ramp and
    #                                 deliberately does NOT scale). Declarative
    #                                 server-side data: NOT exported by
    #                                 verbs_json (duration was never a client
    #                                 parameter).


@dataclass(frozen=True)
class EngineVerb:
    """An engine-channel verb: a protocol op or a client-local camera delta.

    Deliberately NO ``reduce`` field — an engine verb producing a force
    dict is structurally impossible, and test_intents pins this via
    dataclasses.fields().
    """
    name: str
    params: tuple
    doc: str
    continuous: bool
    op: str
    server_side: bool               # False: camera — the server has no camera


# ── world reductions (exact floats; host._canonical's wire conventions) ───

def _reduce_shove(p, target, mass_kg):
    # The vector IS the honest momentum disclosure: exact float m*s, never
    # rounded (_push_af's 4dp rounding is display-side; _canonical never
    # rounds). dv authority is variables.push_speed_m_s, which region.apply
    # reads VERBATIM — replay-critical. Sign flows through: negative dv is
    # legal at region level (only the frozen HTTP /apply route is (0, 20]).
    s = p["speed_m_s"] if p["direction"] == "+x" else -p["speed_m_s"]
    return {"kind": "impulse", "target": target,
            "vector": [mass_kg * s, 0.0, 0.0],
            "variables": {"push_speed_m_s": s}}


def _reduce_press(p, target, mass_kg):
    # THE analog-stick doctrine: a short-duration force re-issued per input
    # tick while held; release decays naturally — never a seventh kind.
    return {"kind": "force", "target": target,
            "vector": [p["force_n"], 0.0, 0.0],
            "duration": PRESS_TICK_S}


def _reduce_grab(p, target, mass_kg):
    # NO point key ever — the planar backend refuses application points.
    return {"kind": "hold", "target": target, "variables": {"on": True}}


def _reduce_release(p, target, mass_kg):
    return {"kind": "hold", "target": target, "variables": {"on": False}}


def _reduce_tilt(p, target, mass_kg):
    # Scene-scoped: force target "world" (applied.py's tilt convention),
    # envelope target None. NO axis key ever (region refuses it); duration
    # top-level per host._canonical's wire form.
    return {"kind": "tilt", "target": "world",
            "variables": {"angle_deg": p["angle_deg"]},
            "duration": p["duration_s"]}


# ── the CLOSED registries ─────────────────────────────────────────────────

WORLD_VERBS = {
    "shove": WorldVerb(
        "shove",
        (Param("speed_m_s", default=2.2, lo=0.0, hi=20.0, lo_open=True),
         Param("direction", default="+x", choices=("+x", "-x"))),
        "impulse-class: an instantaneous push at speed_m_s (m/s of dv)",
        continuous=False, needs_target=True, needs_mass=True,
        reduce=_reduce_shove),
    "press": WorldVerb(
        "press",
        (Param("force_n", default=5.0, lo=-50.0, hi=50.0, nonzero=True),),
        "sustained-force-class: force_n newtons for one input tick — "
        "re-issued at the cap while held (the analog-stick doctrine)",
        continuous=True, needs_target=True, needs_mass=False,
        reduce=_reduce_press, tick_scaled=True),
    "grab": WorldVerb(
        "grab", (),
        "hold on: pin the body against the world (the hold plug)",
        continuous=False, needs_target=True, needs_mass=False,
        reduce=_reduce_grab),
    "release": WorldVerb(
        "release", (),
        "hold off: release a grabbed body",
        continuous=False, needs_target=True, needs_mass=False,
        reduce=_reduce_release),
    "tilt": WorldVerb(
        "tilt",
        (Param("angle_deg", default=6.0, lo=-45.0, hi=45.0, nonzero=True),
         Param("duration_s", default=2.0, lo=0.0, hi=10.0, lo_open=True)),
        "scene-scoped: tilt the whole tray by angle_deg over duration_s "
        "(the sign is the direction — one verb, not two)",
        continuous=False, needs_target=False, needs_mass=False,
        reduce=_reduce_tilt),
}

ENGINE_VERBS = {
    "pace": EngineVerb(
        "pace", (Param("factor", default=1.0, lo=PACE_MIN, hi=PACE_MAX),),
        "set the host's wall-time pace factor",
        continuous=False, op="pace", server_side=True),
    "pause": EngineVerb(
        "pause", (), "freeze wall-time consumption (sim.t untouched)",
        continuous=False, op="pause", server_side=True),
    "resume": EngineVerb(
        "resume", (), "resume the paced loop",
        continuous=False, op="resume", server_side=True),
    "cam_orbit": EngineVerb(
        "cam_orbit",
        (Param("daz", default=0.0, lo=-0.5, hi=0.5),
         Param("delv", default=0.0, lo=-0.5, hi=0.5)),
        "orbit the client camera by daz/delv radians per emission",
        continuous=True, op="cam_orbit", server_side=False),
    "cam_zoom": EngineVerb(
        "cam_zoom", (Param("factor", default=1.0, lo=0.5, hi=2.0),),
        "multiply the client camera radius per emission",
        continuous=True, op="cam_zoom", server_side=False),
    "cam_pan": EngineVerb(
        "cam_pan",
        (Param("dx", default=0.0, lo=-1.0, hi=1.0),
         Param("dy", default=0.0, lo=-1.0, hi=1.0)),
        "pan the client camera target in the view plane (registry member "
        "only in E3: no default binding and not wired in the viewer — the "
        "view-plane math duplicates pickMaterialAt's frame math, deferred)",
        continuous=True, op="cam_pan", server_side=False),
    "cam_move": EngineVerb(
        "cam_move",
        (Param("dx", default=0.0, lo=-1.0, hi=1.0),
         Param("dy", default=0.0, lo=-1.0, hi=1.0),
         Param("dz", default=0.0, lo=-1.0, hi=1.0)),
        "translate the client fly camera along its look basis (dx=right, "
        "dy=up, dz=forward), normalized stick deflection in [-1,1]; the "
        "client controller scales by its distance-proportional speed law "
        "(camera.js's disclosed Choices) — ignored outside fly mode",
        continuous=True, op="cam_move", server_side=False),
    "cam_mode": EngineVerb(
        "cam_mode",
        (Param("mode", default="cycle",
               choices=("cycle", "orbit", "follow", "fly")),),
        "set or cycle the client camera mode; every transition preserves "
        "the exact eye position (no teleports)",
        continuous=False, op="cam_mode", server_side=False),
    "follow": EngineVerb(
        "follow", (),
        "lock the client camera target to the nearest body's live pose — "
        "choosing what to look at never touches physics; named follow is "
        "the controller API (window._camctl.follow(label)), since a static "
        "bindings table cannot name scene-dependent bodies",
        continuous=False, op="follow", server_side=False),
}

# Route-table doctrine (region.py's construction-assert pattern): the two
# channels never overlap, and every registry key names its own spec.
assert not (set(WORLD_VERBS) & set(ENGINE_VERBS)), \
    "a verb cannot live on both channels"
assert all(v.name == k for k, v in WORLD_VERBS.items())
assert all(v.name == k for k, v in ENGINE_VERBS.items())


def verb_names() -> list:
    """The closed set, world first — the refusal vocabulary."""
    return sorted(WORLD_VERBS) + sorted(ENGINE_VERBS)


def _validated(verb_name, param_specs, params) -> dict:
    """Validate + default-fill one verb's params. IntentRefusal on ANY
    problem, with the param name and the honest bound in the text. Every
    accepted numeric value passes through float() (exact-float doctrine);
    str-choice params pass through as str."""
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise IntentRefusal(
            f"{verb_name!r} params must be an object of named values, got "
            f"{type(params).__name__}")
    declared = {p.name for p in param_specs}
    # str() every key before sorting: JSON keys are always strings, but
    # reduce() is a public API and a mixed-type key set would make sorted()
    # raise TypeError — an escape from the IntentRefusal-only contract.
    unknown = sorted(str(k) for k in set(params) - declared)
    if unknown:
        shown = unknown if len(unknown) <= 8 else unknown[:8] + ["…"]
        raise IntentRefusal(
            f"{verb_name!r} has no parameter {shown} — its parameters: "
            f"{sorted(declared)}")
    out = {}
    for p in param_specs:
        if p.name not in params:
            if p.default is None:
                raise IntentRefusal(
                    f"{verb_name!r} requires parameter {p.name!r}")
            out[p.name] = p.default
            continue
        v = params[p.name]
        if p.choices is not None:
            if v not in p.choices:
                raise IntentRefusal(
                    f"{verb_name}.{p.name} must be one of "
                    f"{list(p.choices)}, got {_show(v)}")
            out[p.name] = v
            continue
        # bool is EXPLICITLY refused — it subclasses int and True would
        # otherwise slip through as 1.0 (a silent coercion, not honesty).
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise IntentRefusal(
                f"{verb_name}.{p.name} must be a finite number, got {_show(v)}")
        try:
            v = float(v)
        except OverflowError:
            # JSON integers have NO width limit: {"speed_m_s": 999…} (400
            # digits) parses as an arbitrary-precision Python int, and
            # float() on one raises OverflowError — an ArithmeticError, not
            # a refusal. Unconverted it escaped reduce()'s "IntentRefusal
            # and nothing else" contract, sailed past cockpit's one
            # IntentRefusal arm, and killed the /world connection instead of
            # answering with a refusal envelope.
            # bit_length, never len(str(v)): CPython refuses int->str above
            # 4300 digits (sys.set_int_max_str_digits), so counting digits
            # here would trade one escaping exception for another.
            raise IntentRefusal(
                f"{verb_name}.{p.name} is not a finite number: a "
                f"{v.bit_length()}-bit integer does not fit a double "
                f"(|value| must be under 1.8e308)") from None
        if not math.isfinite(v):
            raise IntentRefusal(
                f"{verb_name}.{p.name} is not a finite number: {v!r}")
        if p.nonzero and v == 0.0:
            raise IntentRefusal(
                f"{verb_name}.{p.name} must be nonzero — a zero "
                f"{p.name} is a no-op wearing a mask")
        if p.lo is not None:
            if p.lo_open:
                if not v > p.lo:
                    raise IntentRefusal(
                        f"{verb_name}.{p.name} must be > {p.lo} "
                        f"(got {v})")
            elif v < p.lo:
                raise IntentRefusal(
                    f"{verb_name}.{p.name} must be >= {p.lo} (got {v})")
        if p.hi is not None and v > p.hi:
            raise IntentRefusal(
                f"{verb_name}.{p.name} must be <= {p.hi} (got {v})")
        out[p.name] = v
    return out


def _validated_pace(pace) -> float:
    """The reduce-time pace factor (E4's press-window fix). None → 1.0
    (headless/no-host callers). Anything else must be a finite number
    within [PACE_MIN, PACE_MAX] — IntentRefusal otherwise (totality: the
    fuzz contract covers hostile pace values too)."""
    if pace is None:
        return 1.0
    if isinstance(pace, bool) or not isinstance(pace, (int, float)):
        raise IntentRefusal(
            f"pace must be a finite number in [{PACE_MIN}, {PACE_MAX}], "
            f"got {_show(pace)}")
    try:
        f = float(pace)
    except OverflowError:
        raise IntentRefusal(
            f"pace is not a finite number: a {pace.bit_length()}-bit "
            f"integer does not fit a double") from None
    if not math.isfinite(f) or not (PACE_MIN <= f <= PACE_MAX):
        raise IntentRefusal(
            f"pace must be a finite number in [{PACE_MIN}, {PACE_MAX}], "
            f"got {_show(pace)}")
    return f


def reduce(verb, params=None, *, target=None, masses=None,
           pace=None) -> dict:
    """Reduce one {verb, params, target} game intent to its channel form.

    Returns {"channel": "world", "verb", "target", "force": {wire dict}}
    or {"channel": "engine", "verb", "op", "server_side", "args": {...}}.
    Raises IntentRefusal — and ONLY IntentRefusal — for every invalid
    input. ``masses`` is the server-side label→mass_kg table (needed only
    by needs_mass verbs: the impulse vector discloses J = m·v). ``pace``
    is the host's current pace factor at reduce time (the cockpit passes
    host.pace): tick_scaled verbs (press) get duration = PRESS_TICK_S *
    pace so held re-issues tile in WALL time at any pace — the E3
    deferral, fixed here. The command log records the ACTUAL scaled
    duration, so replay stays exact.
    """
    pace_f = _validated_pace(pace)      # validation FIRST — totality
    if not isinstance(verb, str) or (verb not in WORLD_VERBS
                                     and verb not in ENGINE_VERBS):
        raise IntentRefusal(
            f"unknown game verb {_show(verb)} — the set is CLOSED: "
            f"{verb_names()}. A new verb is a doctrine decision "
            f"(Golden Rule 7's input layer), not a parameter.")
    if verb in WORLD_VERBS:
        spec = WORLD_VERBS[verb]
        p = _validated(verb, spec.params, params)
        if spec.needs_target:
            if not isinstance(target, str):
                raise IntentRefusal(
                    f"{verb!r} needs a target body — pass its label; "
                    f"bodies live server-side")
        elif target is not None:
            raise IntentRefusal(
                f"{verb!r} is scene-scoped — pass target null (it acts "
                f"on the whole scene, not one body)")
        mass = None
        if spec.needs_mass:
            # The ONE deliberate second target check (all other membership
            # stays the host's single refusal source): reduce cannot build
            # an honest J = m·v vector without the mass.
            if not isinstance(masses, dict) or target not in masses:
                known = sorted(masses) if isinstance(masses, dict) else []
                raise IntentRefusal(
                    f"{verb!r} discloses momentum J = m*v in its vector "
                    f"and mass lives server-side — target must be one of "
                    f"{known}, got {_show(target)}")
            try:
                mass = float(masses[target])
            except (TypeError, ValueError, OverflowError):
                # totality again: the caller owns ``masses``, so a
                # non-numeric entry must refuse, never raise past reduce()
                raise IntentRefusal(
                    f"{verb!r} cannot disclose J = m*v: the mass table entry "
                    f"for {target!r} is not a number "
                    f"({_show(masses[target])})") from None
        force = spec.reduce(p, target, mass)
        if spec.tick_scaled:
            # E4: one wall tick at the cap advances pace/60 sim-s; the
            # window covers exactly that → duty cycle 1 at every pace.
            force["duration"] = PRESS_TICK_S * pace_f
        AppliedForce.from_dict(force)   # SELF-CHECK: a registry bug dies
        #                                 here, not at the region
        return {"channel": "world", "verb": verb,
                "target": target if spec.needs_target else None,
                "force": force}
    spec = ENGINE_VERBS[verb]
    p = _validated(verb, spec.params, params)
    if target is not None:
        raise IntentRefusal(
            f"{verb!r} is an engine command — engine commands cannot "
            f"touch physics; no world target")
    # This branch NEVER references AppliedForce — see EngineVerb.
    return {"channel": "engine", "verb": verb, "op": spec.op,
            "server_side": spec.server_side, "args": p}


def _param_json(p: Param) -> dict:
    return {"name": p.name, "default": p.default, "lo": p.lo, "hi": p.hi,
            "lo_open": p.lo_open, "nonzero": p.nonzero,
            "choices": list(p.choices) if p.choices is not None else None}


def verbs_json() -> dict:
    """The static mirror's content — strict JSON, the client's registry
    view (input.js validates bindings against it). Regenerate with
    ``python -m sigma_ground.engine.intents``; never hand-edit the file."""
    verbs = {}
    for name, s in WORLD_VERBS.items():
        verbs[name] = {"channel": "world", "continuous": s.continuous,
                       "doc": s.doc, "needs_target": s.needs_target,
                       "needs_mass": s.needs_mass,
                       "params": [_param_json(p) for p in s.params]}
    for name, s in ENGINE_VERBS.items():
        verbs[name] = {"channel": "engine", "continuous": s.continuous,
                       "doc": s.doc, "op": s.op,
                       "server_side": s.server_side,
                       "params": [_param_json(p) for p in s.params]}
    return {"format": "sg-intent-verbs", "version": 1,
            "_generated_by": "python -m sigma_ground.engine.intents",
            "verbs": verbs}


__all__ = ["IntentRefusal", "Param", "WorldVerb", "EngineVerb",
           "PRESS_TICK_S", "RATE_CAP_HZ", "WORLD_VERBS", "ENGINE_VERBS",
           "verb_names", "reduce", "verbs_json"]


if __name__ == "__main__":
    # Regenerate the committed static mirror next to the web assets. The
    # radiance import is same-tier (3→3, precedent: host.py imports
    # radiance.push_off) and lives INSIDE this block so importing intents
    # never drags the web server in.
    import json
    import os

    from ..radiance.web import serve
    path = os.path.join(os.path.dirname(os.path.abspath(serve.__file__)),
                        "data", "intent_verbs.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(verbs_json(), indent=2, sort_keys=True) + "\n")
    print(path)
