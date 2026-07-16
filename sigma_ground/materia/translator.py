"""Materia translator — natural language → Simulation Spec → answer.

The front door. Two paths, mirroring the Q&A switchboard:

  1. Deterministic intent + slot extraction (keywords + regex) — fast, free,
     and correct for the common phrasings. Most What-Ifs never touch the LLM.
  2. qwen residual (ollama, 7b) — for novel babble the keywords miss. The model
     emits ONLY a JSON {verb, params}; we validate it against the manifest and
     run it. The model routes; it never computes.

If neither yields a grounded, runnable spec, we ASK FOR CLARIFICATION rather
than guess — the same never-confidently-wrong discipline as the Q&A layer.

This is "compile-then-run": translate() produces a SimulationSpec; run_spec()
executes it. The infinity of possible questions is absorbed by the translator;
the finite, validated verbs do the physics.
"""
from __future__ import annotations

import json
import re
import urllib.request

from .manifest import (VERB_MANIFEST, manifest_for_prompt,
                       SPEED_TRIGGERS, HEAT_TRIGGERS)
from .spec import SimulationSpec, SpecStep, run_spec

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

# Intent trigger words now live in the manifest (SPEED_TRIGGERS / HEAT_TRIGGERS)
# so routing is manifest-driven — a new verb is routable purely by its entry.
# The launch-chain composition template below imports them for its cue checks.

# Domains we have NO verb for yet. If one of these appears, DECLINE rather than
# grab a falling-object verb on a stray "speed"/"hot" — groundedness over
# coverage (a wrong answer is worse than "not yet modeled"). This list shrinks
# as families (statics, rigid-body, fluids) ship.
_OUT_OF_SCOPE = [
    # structural / statics
    "beam", "cantilever", "tripod", "truss", "girder", "buckl", "catenary",
    "safe hangs", "hollow leg", "pole sticks", "tapers",
    # rigid-body dynamics
    "pendulum", "swinging stick", "dangling", "rocket", "sled", "billiard",
    "pool ball", "cue ball", "flywheel", "gyroscope",
    # fluids / continuum / PDE
    "viscometer", "couette", "slime", "gooey", "pipe", "conduit", "valve",
    "nozzle", "water hammer", "cylinder", "boiling",
    # abstract / metaphor domains — "fall"/"rise"/"crash"/"hot" aren't physics
    "stock", "market", "economy", "inflation", "gdp", "interest rate",
    "currency", "the dollar", "shares", "portfolio", "ratings", "approval",
    # collisions / momentum transfer — we drop ONE body, not interactions
    "collide", "collision", "collides", "head-on", "elastic collision",
    "inelastic", "ricochet", "newton's cradle", "rebound off", "recoil",
    # static / free-surface / fluid-interaction — a body IN or AROUND a fluid is
    # not a single falling object (no verb, no solver). Refuse, don't force-fit.
    "sitting in", "sits in", "submerged", "half-submerged", "floating in",
    "floats in", "splash", "splashes", "wake", "ripple", "ripples",
    "flows around", "flowing around", "running water", "trough",
]

# A Materia simulation is about an OBJECT in motion. A trigger word
# (speed/hot/how hard) routes ONLY when the question also names a falling/thrown
# body or such an action — otherwise a stray "how fast"/"hot" is a non-physics
# or fact question ("how fast is my internet?", "speed of sound at sea level?")
# and we DECLINE rather than fake a falling-object answer.
_OBJECT_NOUNS = {
    "ball", "sphere", "marble", "cannonball", "projectile", "rock", "stone",
    "weight", "pellet", "bullet", "boulder", "ball bearing", "meteor",
    "meteorite", "asteroid", "raindrop", "hailstone", "brick", "anvil", "egg",
    "watermelon", "piano", "steak", "coin", "penny", "feather", "apple",
    "hammer", "anchor", "dumbbell", "skydiver", "parachutist", "parachute",
    "payload", "capsule",
}
# Strong falling-scenario phrases that imply a body even without a noun.
_SCENARIO_CUES = ("off a cliff", "off a building", "off a bridge",
                  "off the roof", "off a tower", "off a ledge", "off a balcony")


def _has_object_context(low: str) -> bool:
    """A Materia sim needs an OBJECT. Require a falling/thrown body NOUN (or a
    strong 'off a cliff'-type phrase). A bare action verb like 'fall'/'drop' is
    NOT enough — it metaphors too easily (prices fall, drop the mic, waterfall)
    and would fake a physics answer for a non-physics question.
    """
    return (any(n in low for n in _OBJECT_NOUNS)
            or any(c in low for c in _SCENARIO_CUES))


# A drop/fall of an object FROM a height — the cues that mark a terminal-velocity
# scenario even with no explicit "how fast" (impact speed is the implied
# question). The caller gates this on object context so "prices fall" can't trip
# it. "from <number>" catches "from 12 feet / from 10 km".
_DROP_CUES = ("drop", "dropped", "fall", "falls", "fell", "falling", "plummet",
              "off a ", "off the ", "thrown off", "tossed off", "released from",
              "from a height")


def _is_drop(low: str) -> bool:
    return (any(c in low for c in _DROP_CUES)
            or re.search(r"\bfrom\s+\d", low) is not None)


# Named NON-sphere objects whose shape Materia doesn't carry: a drop/fall of one
# of these asks the DECKARD shape researcher for its real mass/shape (the
# Materia↔Deckard seam). Spheres (ball/marble/…) keep the fast analytic path.
_SHAPE_OBJECTS = {
    "anvil", "piano", "hammer", "anchor", "watermelon", "brick", "steak",
    "dumbbell", "toaster", "frying pan", "skillet", "chair", "laptop",
    "bottle", "coffee cup", "mug", "wrench", "axe", "log", "kettle", "teapot",
    "vase", "bucket", "ladder", "fridge", "refrigerator", "microwave",
    "feather",
}


def _named_shape_object(low: str):
    """The named non-sphere object in the question (longest match), or None."""
    for obj in sorted(_SHAPE_OBJECTS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(obj) + r"\b", low):
            return obj
    return None


def _candidate_object(low: str):
    """The object-ish word the user named that we could NOT ground to a shape
    (longest match over the known object nouns). Used ONLY to name it honestly in
    a refusal — never to run a sim."""
    for noun in sorted(_OBJECT_NOUNS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(noun) + r"\b", low):
            return noun
    return None


_LEN_UNIT_M = {  # → metres
    "mm": 1e-3, "millimeter": 1e-3, "millimetre": 1e-3, "millimeters": 1e-3,
    "cm": 1e-2, "centimeter": 1e-2, "centimetre": 1e-2, "centimeters": 1e-2,
    "m": 1.0, "meter": 1.0, "metre": 1.0, "meters": 1.0,
    "km": 1e3, "kilometer": 1e3, "kilometre": 1e3, "kilometers": 1e3,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
}
_BIG_UNITS = {"km", "kilometer", "kilometre", "kilometers", "mi", "mile",
              "miles"}
_SMALL_UNITS = {"mm", "millimeter", "millimetre", "millimeters",
                "cm", "centimeter", "centimetre", "centimeters",
                "in", "inch", "inches"}

# No true zero. A material object is at least ~1 atom across, so we clamp a
# degenerate radius to this physical floor — "0 cm" becomes the smallest
# possible ball (a real, tiny answer) instead of a div/zero crash. The engine
# refines this to the material's exact atomic minimum (lattice spacing); for
# baryonic energy the floor is 3 quarks, and for σ-measurements the Planck floor.
_ATOMIC_FLOOR_M = 1e-10

_CLARIFY = ("I can simulate a falling object's *impact speed* or *how much it "
            "heats up* from drag. Try e.g. \"how fast does a 5 cm steel ball "
            "hit the ground from 10 km?\" or \"does an iron sphere heat up "
            "falling from 30 km?\"")


# ── Slot extraction ─────────────────────────────────────────────────────
def _material_synonyms() -> dict:
    from ..field.interface.surface import MATERIALS
    syn = {}
    for key, m in MATERIALS.items():
        syn[key.replace("_", " ")] = key
        syn[m.get("name", key).lower()] = key
    # hand aliases for the common informal words
    syn.update({"steel": "steel_mild", "mild steel": "steel_mild",
                "aluminium": "aluminum", "ice": "water_ice", "wood": "wood_oak",
                "oak": "wood_oak", "uranium": "depleted_uranium",
                "plastic": "plastic_abs", "alumina": "ceramic_alumina",
                "ceramic": "ceramic_alumina", "carbon fibre": "carbon_fiber"})
    return syn


def _extract_material(q: str) -> str:
    low = q.lower()
    syn = _material_synonyms()
    # prefer the longest phrase match (so "mild steel" beats "steel")
    for phrase in sorted(syn, key=len, reverse=True):
        if re.search(r"\b" + re.escape(phrase) + r"\b", low):
            return syn[phrase]
    return "iron"


# Generic material-CLASS words: naming a class ("a metal", "a polymer") still
# makes it a material query — a representative answer is fair. Naming NO material
# at all is the danger: a material-property verb would fall back to the iron
# DEFAULT and answer confidently about the wrong substance.
_MATERIAL_CLASS_WORDS = (
    "metal", "metals", "material", "alloy", "polymer", "ceramic", "composite",
    "semiconductor", "dielectric", "magnet", "glass", "plastic", "fluid",
    "liquid", "gas", "crystal", "insulator", "conductor", "elastomer", "rubber",
    "water",          # a common acoustics/optics medium the synonym table omits
)


def _named_material(low: str) -> bool:
    """Does the question NAME a material — a specific one OR a material class?

    A pure material-property verb needs this. Without it the verb defaults to
    iron and answers confidently about the WRONG substance — e.g. "speed of sound
    at sea level" would route to acoustics and report "iron 5942 m/s". No material
    named ⇒ the verb must DECLINE, not guess iron (dial-1: never confidently
    wrong). A named material ("in steel", "of a metal") routes as before.
    """
    syn = _material_synonyms()
    for phrase in syn:
        if re.search(r"\b" + re.escape(phrase) + r"\b", low):
            return True
    return any(re.search(r"\b" + re.escape(w) + r"\b", low)
               for w in _MATERIAL_CLASS_WORDS)


def _extract_lengths(q: str) -> dict:
    """Pull radius and altitude (metres) from the question, with defaults.

    Units disambiguate most cases: mm/cm → a sphere radius, km/mi → a drop
    altitude. Only bare metres/feet need context (a size word → radius, else
    altitude). Crucially the verb "drop" is NOT an altitude cue — "drop a 2 cm
    ball" is a size, not a height.
    """
    low = q.lower()
    radius, altitude = 0.05, 10000.0
    radius_found = altitude_found = False
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", low):
        unit = m.group(2)
        if unit not in _LEN_UNIT_M:
            continue
        if low[m.end():m.end() + 1] == "/":   # m/s, km/h — a rate, not a length
            continue
        metres = float(m.group(1)) * _LEN_UNIT_M[unit]
        # Wide enough to catch "5 inches IN DIAMETER" (the qualifier trails the
        # number by ~12 chars) and "DIAMETER of 5 inches" (it leads).
        win = low[max(0, m.start() - 16):min(len(low), m.end() + 24)]
        is_diam = "diameter" in win
        if unit in _SMALL_UNITS or is_diam:
            radius, radius_found = (metres / 2.0 if is_diam else metres), True
        elif unit in _BIG_UNITS:
            altitude, altitude_found = metres, True
        elif any(w in win for w in ("radius", "wide", "across")):
            radius, radius_found = metres, True
        else:
            altitude, altitude_found = metres, True
    return {"radius_m": radius, "drop_altitude_m": altitude,
            "radius_found": radius_found, "altitude_found": altitude_found}


def _extract_mach(q: str):
    low = q.lower()
    m = (re.search(r"mach\s*(\d+(?:\.\d+)?)", low) or
         re.search(r"(\d+(?:\.\d+)?)\s*(?:times\s+)?(?:the\s+)?"
                   r"(?:speed of sound|mach)", low))
    return float(m.group(1)) if m else None


_ENERGY_UNIT_J = {"megaton": 4.184e15, "megatons": 4.184e15, "mt": 4.184e15,
                  "kiloton": 4.184e12, "kilotons": 4.184e12, "kt": 4.184e12,
                  "joule": 1.0, "joules": 1.0}


def _extract_energy_j(q: str):
    """An energy with a TNT/SI unit → joules (for the reverse E=mc² direction)."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*"
                  r"(megatons?|kilotons?|mt|kt|joules?)\b", q.lower())
    if m:
        return float(m.group(1)) * _ENERGY_UNIT_J.get(m.group(2).strip(), 1.0)
    return None


_TIME_UNIT_S = {  # → seconds. day/week are exact; a "year" is the project's
    # 3.15e7 s ≈ 1 yr convention (matches corrosion's default exposure) and a
    # month is year/12.
    "second": 1.0, "sec": 1.0,
    "minute": 60.0, "min": 60.0,
    "hour": 3600.0, "hr": 3600.0,
    "day": 86400.0,
    "week": 604800.0,
    "month": 3.15e7 / 12.0,
    "year": 3.15e7, "yr": 3.15e7,
    "decade": 3.15e8,
    "century": 3.15e9, "centuries": 3.15e9,
}
# Longest-first so "months" can't half-match "min"/"mon" style prefixes.
_TIME_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(" +
    "|".join(sorted(_TIME_UNIT_S, key=len, reverse=True)) + r")s?\b")
_TIME_WORD_RE = re.compile(
    r"\b(?:a|an|one)\s+(" +
    "|".join(sorted(_TIME_UNIT_S, key=len, reverse=True)) + r")\b")


def _extract_duration_s(q: str):
    """A duration phrase → seconds: "over 5 years", "for 10 days", "after 3
    months", "in 2 weeks", also "a year"/"an hour". None when the question
    names no duration — the verb's default then applies."""
    low = q.lower()
    m = _TIME_UNIT_RE.search(low)
    if m:
        return float(m.group(1)) * _TIME_UNIT_S[m.group(2)]
    m = _TIME_WORD_RE.search(low)
    if m:
        return _TIME_UNIT_S[m.group(1)]
    return None


# Environment cues (for verbs declaring an "environment" slot — corrosion).
# Word-boundary regexes: "ground" must not fire on "background", "acid" not
# on "acidity of the joke"… well, it will — but the slot only exists on
# corrosion verbs, so a stray match can't hijack routing, only annotate it.
_ENV_CUES = (
    ("acidic", r"\bacid(ic|ified)?\b"),
    ("alkaline", r"\b(alkaline|alkali|caustic|lye)\b"),
    ("soil", r"\b(soil|buried|underground|dirt|ground)\b"),
    ("seawater", r"\b(seawater|sea\s+water|saltwater|salt\s+water|marine|"
                 r"ocean|brine)\b"),
    ("immersed", r"\b(immersed|submerged|underwater|in\s+water)\b"),
    ("aerated", r"\b(aerated|oxygenated|oxidizing|oxidising)\b"),
    ("deaerated", r"\b(deaerated|de-aerated|anaerobic|oxygen-free|"
                  r"waterlogged|stagnant)\b"),
)


def _extract_environment(q: str):
    """Corrosion-environment words → a compact label ("alkaline soil, "
    "aerated"). None when the question names no environment."""
    low = q.lower()
    found = [name for name, pat in _ENV_CUES if re.search(pat, low)]
    if "deaerated" in found and "aerated" in found:
        found.remove("aerated")        # "deaerated" contains "aerated"
    if not found:
        return None
    # Readable ordering: pH word, then medium, then aeration.
    order = ("acidic", "alkaline", "soil", "seawater", "immersed",
             "aerated", "deaerated")
    found.sort(key=order.index)
    medium_ph = [f for f in found if f not in ("aerated", "deaerated")]
    aeration = [f for f in found if f in ("aerated", "deaerated")]
    label = " ".join(medium_ph)
    if aeration:
        label = (label + ", " + aeration[0]) if label else aeration[0]
    return label


def _params_for(verb: str, q: str) -> dict:
    """Fill ONLY the slots the chosen verb declares (verb-aware extraction)."""
    slots = VERB_MANIFEST[verb]["slots"]
    L = _extract_lengths(q)
    p = {}
    if "material_key" in slots:
        p["material_key"] = _extract_material(q)
    if "radius_m" in slots:
        p["radius_m"] = max(L["radius_m"], _ATOMIC_FLOOR_M)  # ≥ ~1 atom; no true zero
    if "drop_altitude_m" in slots:
        p["drop_altitude_m"] = L["drop_altitude_m"]
    if "start_altitude_m" in slots and L["altitude_found"]:
        p["start_altitude_m"] = L["drop_altitude_m"]   # else scenario default (35 km)
    if "altitude_m" in slots and L["altitude_found"]:
        p["altitude_m"] = L["drop_altitude_m"]
    if "central_body" in slots:
        body, sma_au = _extract_orbit(q)
        p["central_body"] = body
        if sma_au is not None:
            p["semimajor_axis_au"] = sma_au
    if "body" in slots:                       # planetary_surface: gravity OF Mars
        b = _extract_body(q)
        if b:
            p["body"] = b
    if "object_name" in slots:                # drop_object: the named object
        obj = _named_shape_object(q.lower())
        if obj:
            p["object_name"] = obj
    if "mass_kg" in slots:
        mm = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)\b", q.lower())
        if mm:
            p["mass_kg"] = float(mm.group(1))
    if "energy_j" in slots:
        ej = _extract_energy_j(q)
        if ej is not None:
            p["energy_j"] = ej
    if "launch_mach" in slots:
        mach = _extract_mach(q)
        if mach is not None:
            p["launch_mach"] = mach
    if "launch_speed_m_s" in slots:
        spd = _extract_speed(q)
        if spd is not None:
            p["launch_speed_m_s"] = spd
    if "duration_s" in slots:
        dur = _extract_duration_s(q)
        if dur is not None:
            p["duration_s"] = dur
    if "environment" in slots:
        envl = _extract_environment(q)
        if envl:
            p["environment"] = envl
    return p


def _extract_speed(q: str):
    low = q.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m/s|mps|meters? per second)\b", low)
    if m:
        return float(m.group(1))
    m = re.search(r"mach\s*(\d+(?:\.\d+)?)", low)
    if m:
        return float(m.group(1)) * 340.3
    return None


_BODIES = ("earth", "moon", "mars", "sun", "jupiter", "saturn", "venus",
           "mercury", "neptune", "uranus", "pluto")


def _extract_body(q: str):
    low = q.lower()
    for b in _BODIES:
        if re.search(r"\b" + b + r"\b", low):
            return b
    return None


_PLANET_AU = {"mercury": 0.387, "venus": 0.723, "earth": 1.0, "mars": 1.524,
              "jupiter": 5.203, "saturn": 9.537, "uranus": 19.19,
              "neptune": 30.07, "pluto": 39.48}


def _extract_orbit(q: str):
    """(central_body, semimajor_axis_au | None).

    A planet orbiting the Sun → heliocentric (central='sun', the planet's AU);
    otherwise a satellite-altitude orbit around the named body (default earth).
    """
    low = q.lower()
    if "sun" in low and ("orbit" in low or "around" in low):
        for planet, au in _PLANET_AU.items():
            if re.search(r"\b" + planet + r"\b", low):
                return "sun", au
    return _extract_body(q) or "earth", None


def _auto_bind(steps):
    """Thread each step's slots from prior steps — MANIFEST-DRIVEN.

    A slot declaring "bind_from": <output_field> in the verb manifest is wired
    to the nearest EARLIER step whose verb outputs that field. This is the whole
    chaining contract: verb authors declare bind_from, the translator threads it
    — zero per-verb code here, so any verb the Mentat track registers chains for
    free. The binding overrides whatever default the slot had.
    """
    for n, step in enumerate(steps):
        slots = VERB_MANIFEST.get(step.verb, {}).get("slots", {})
        for slot, meta in slots.items():
            src_field = meta.get("bind_from")
            if not src_field or slot in step.bindings:
                continue
            for m in range(n - 1, -1, -1):
                if src_field in VERB_MANIFEST.get(steps[m].verb, {}).get("outputs", []):
                    step.bindings[slot] = (m, src_field)
                    break
    return steps


# ── Intent → verb(s) ────────────────────────────────────────────────────
def _matches(low: str, triggers) -> bool:
    r"""A trigger starting with \b is a word-boundary regex (so 'mach' doesn't
    fire on 'machine'); any other trigger is a plain substring test."""
    return _match_len(low, triggers) > 0


def _match_len(low: str, triggers) -> int:
    r"""Length of the LONGEST matching trigger (0 if none). A trigger containing
    a backslash is a regex (\b word-boundary etc.); else a plain substring. The
    length lets the router pick the MOST SPECIFIC verb — 'shear modulus of'
    (elastic_solid) beats 'shear modulus' (material_profile), 'all properties of'
    beats 'properties of' — so routing is specificity-ordered, not dict-ordered.
    """
    best = 0
    for t in triggers:
        if "\\" in t:                                   # a regex trigger
            try:
                mm = re.search(t, low)
                if mm:
                    best = max(best, len(mm.group(0)))
            except re.error:
                if t in low:
                    best = max(best, len(t))
        elif t in low:
            best = max(best, len(t))
    return best


def _classify_verbs(q: str):
    """Route to verb(s) — MANIFEST-DRIVEN, gated on OBJECT CONTEXT.

    A simulation needs an object in motion, so a trigger only routes when the
    question also has object/action context. That keeps a stray 'speed'/'hot'
    in a non-physics or fact question from faking a falling-object answer.
    """
    low = q.lower()
    ctx = _has_object_context(low)
    is_drop = ctx and _is_drop(low)
    # 0. A drop/fall of a NAMED non-sphere object → ask Deckard for its real
    #    shape (Materia carries only spheres natively). Spheres fall through.
    obj = _named_shape_object(low)
    if obj and not any(k in low for k in _OUT_OF_SCOPE) and (
            any(c in low for c in ("drop", "fall", "fell", "off a ",
                                   "off the ", "thrown off", "tossed off"))
            or _matches(low, SPEED_TRIGGERS)):
        return ["drop_object"]
    # 1. Launch-then-descend template — needs an object NOUN (so "throw a ball
    #    up" routes, but "I might throw up" does not).
    if (any(n in low for n in _OBJECT_NOUNS)
            and any(t in low for t in ("throw", "hurl", "launch", "fling",
                                       "toss", "lob"))
            and (re.search(r"\bup(ward|wards)?\b", low) or "into the air" in low
                 or "skyward" in low or "straight up" in low)):
        chain = ["vertical_launch"]
        if (_matches(low, SPEED_TRIGGERS) or "land" in low or "come down" in low
                or "comes down" in low or "comes back" in low
                or "falls back" in low):
            chain.append("terminal_velocity_drop")
        if _matches(low, HEAT_TRIGGERS):
            chain.append("drag_heating_drop")
        return chain
    # 2. Exclusive verbs route on their (specific) triggers alone — no generic
    #    object-context gate. "orbital velocity"/"supersonic"/"parachute" are
    #    already unambiguous physics; only the generic speed/heat verbs need the
    #    object gate. Among all matching exclusive verbs the MOST SPECIFIC wins
    #    (longest matched trigger), so a precise domain phrase beats a generic
    #    one regardless of registration order.
    mat_named = _named_material(low)
    is_action = is_drop or bool(re.search(
        r"\b(fire[sd]?|firing|shoot(s|ing)?|shot|launch(es|ed|ing)?|"
        r"throw(s|n|ing)?|threw|toss(es|ed)?|hurl(s|ed)?|jump(s|ed|ing)?)\b", low))
    best_verb, best_len, matched = None, 0, set()
    cands = []
    for verb, m in VERB_MANIFEST.items():
        if not m.get("exclusive"):
            continue
        # A verb flagged material_required has the MATERIAL as its subject; with
        # none named it would default to iron and answer confidently about the
        # WRONG substance ("speed of sound at sea level" → "iron 5942 m/s"). So
        # it is not a candidate unless a material is named — it declines instead.
        # (Opt-in per verb: structure verbs like transmission_line keep their
        # sensible default; the flag is only for material-IS-the-answer verbs.)
        if m.get("material_required") and not mat_named:
            continue
        # An ambient REPORT (atmospheric_profile) must not hijack a falling-object
        # sim that only mentions the setting ("... in standard atmosphere").
        if is_drop and m.get("ambient_report"):
            continue
        ml = _match_len(low, m.get("triggers", []))
        if ml > 0:
            matched.add(verb)
            cands.append((verb, ml, bool(m.get("motion"))))
    # An ACTION question (fired/thrown/dropped/jumps...) with a motion-verb match:
    # property REPORTS leave the contest — "Fire a slug at Mach 2" is a
    # supersonic_projectile sim even though acoustics' "speed of sound" is the
    # longer literal trigger. ("speed of sound in steel" has no action cue and
    # still routes to acoustics.) Generalizes the old skydiver-only tiebreak.
    if is_action and any(c[2] for c in cands):
        cands = [c for c in cands if c[2]]
    for verb, ml, _motion in cands:
        if ml > best_len:
            best_verb, best_len = verb, ml
    # A faller going supersonic (skydiver, re-entry) is a high-altitude DESCENT,
    # not a fired projectile — descent owns that framing even though "sound
    # barrier" is a longer literal match than "skydiver".
    if best_verb == "supersonic_projectile" and "high_altitude_descent" in matched:
        best_verb = "high_altitude_descent"
    # A projectile IN THE MACH REGIME is the transonic-drag verb, not the
    # ballistic-arc one — the mach cue is decisive even when a range phrase
    # ("how far") is the longer literal match.
    if best_verb == "projectile_motion" and "supersonic_projectile" in matched:
        best_verb = "supersonic_projectile"
    # A material actively corroding/rusting is corrosion_attack, even when a
    # generic acid/base cue ("alkaline soil") is a longer literal match than
    # "corrode" -- the corrosion cue is decisive (bug found live: "zinc rod
    # ... alkaline soil ... corrode" was routing to chemistry_lab's default
    # acetic-acid demo instead of corrosion_attack, on an 8-vs-7-char
    # tiebreak with no topical relevance to the actual match length).
    if best_verb == "chemistry_lab" and "corrosion_attack" in matched:
        best_verb = "corrosion_attack"
    if best_verb:
        return [best_verb]
    # 3. A family we don't model yet → decline regardless.
    if any(k in low for k in _OUT_OF_SCOPE):
        return []
    # 3.5 A drop/fall of an object from a height IS a terminal-velocity sim — even
    #     with no "how fast" cue, and even amid ambient words ("in standard
    #     atmosphere", "onto a concrete floor"): those are the setting, not the
    #     question. (Named non-sphere objects were already taken by step 0; this
    #     is the generic-sphere drop.) Impact speed is the default question; add
    #     heating only when a heat cue is also present.
    if is_drop:
        speed = _matches(low, SPEED_TRIGGERS)
        heat = _matches(low, HEAT_TRIGGERS)
        chain = []
        if speed or not heat:
            chain.append("terminal_velocity_drop")
        if heat:
            chain.append("drag_heating_drop")
        return chain
    # 4. Generic verbs route only WITH object context (else it isn't a sim;
    #    speed + heat both present → both drop verbs, terminal first).
    if ctx:
        return [verb for verb, m in VERB_MANIFEST.items()
                if not m.get("exclusive") and m.get("triggers")
                and _matches(low, m["triggers"])]
    return []


# ── qwen residual ───────────────────────────────────────────────────────
def _ollama_chat(question: str, url=OLLAMA_URL, model=OLLAMA_MODEL,
                 timeout=30) -> str | None:
    sys_prompt = (
        "You route a physics what-if to ONE Materia simulation verb, or to "
        "none. Output ONLY JSON: {\"verb\": <name or \"none\">, \"params\": "
        "{<slot>: <number or material>}}. Fill params only from the chosen "
        "verb's slots.\nVerbs:\n" + manifest_for_prompt() +
        "\nIf NONE of these verbs fits the question, output {\"verb\": "
        "\"none\"}. material_key is a common solid (iron, copper, lead, "
        "steel_mild, aluminum, gold, titanium, tungsten). Lengths in metres, "
        "Mach as a number. Do NOT compute physics — only pick the verb and "
        "fill its slots.")
    body = json.dumps({
        "model": model, "stream": False, "format": "json",
        "options": {"temperature": 0},
        "messages": [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": question}],
    }).encode()
    try:
        req = urllib.request.Request(url + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["message"]["content"]
    except Exception:
        return None


def _qwen_translate(question: str) -> SimulationSpec | None:
    raw = _ollama_chat(question)
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        verb = obj["verb"]
        if verb not in VERB_MANIFEST:
            return None
        declared = VERB_MANIFEST[verb]["slots"]
        p_in = obj.get("params", {}) or {}
        params = {}
        for slot in declared:
            if slot not in p_in or p_in[slot] is None:
                continue
            if slot == "material_key":
                params[slot] = _coerce_material(p_in[slot])
            elif declared[slot].get("unit") in ("object", "body", "environment"):
                # STRING slots survive (manifest-driven) — same rule as
                # _qwen_plan: float("alkaline soil") must not delete the slot.
                v = str(p_in[slot]).strip()
                if v:
                    params[slot] = v
            else:
                try:
                    params[slot] = float(p_in[slot])
                except (TypeError, ValueError):
                    pass
        # Fix C — same guard as _qwen_plan: no force-fit drop onto a static/fluid scene.
        low = question.lower()
        if verb == "drop_object" and (
                any(k in low for k in _OUT_OF_SCOPE) or not _is_drop(low)):
            return None
        return SimulationSpec(question, [SpecStep(verb, params)], source="qwen",
                              note="routed by qwen2.5:7b")
    except Exception:
        return None


def _coerce_material(name: str) -> str:
    from ..field.interface.surface import MATERIALS
    if name in MATERIALS:
        return name
    return _extract_material(str(name))


def _route_consistent(question: str, n: int = 2, temperature: float = 0.5,
                      use_qwen: bool = True) -> bool:
    """Route-consistency cross-check (gate 2 for the LLM residual): sample the
    router `n` times at non-zero temperature. If it agrees with ITSELF on the
    verb sequence every time, the routing is confident → trust it. If the
    samples diverge, the LLM is uncertain — and an uncertain route is exactly
    how a question gets sent to the WRONG tool — so we refuse instead of guess.
    The model's own disagreement is the second opinion.
    """
    if not use_qwen:
        return True
    seen = None
    for _ in range(max(2, n)):
        raw = _ollama_plan(question, temperature=temperature)
        if not raw:
            return False                     # router unavailable → can't verify
        try:
            verbs = tuple(s.get("verb") for s in json.loads(raw).get("steps", []))
        except Exception:
            return False
        if seen is None:
            seen = verbs
        elif verbs != seen:
            return False                     # diverged → uncertain → refuse
    return True


def _ollama_plan(question: str, url=OLLAMA_URL, model=OLLAMA_MODEL,
                 timeout=45, temperature: float = 0.0) -> str | None:
    sys_prompt = (
        "You break a physics what-if into an ORDERED LIST of Materia verbs (a "
        "plan). Output ONLY JSON: {\"steps\":[{\"verb\":<name>,\"params\":"
        "{<slot>:<value>}}]}. One verb per step; fill that verb's slots. If one "
        "verb's result should feed another (throw it UP, then how it comes "
        "DOWN), just list them in order — the engine threads the altitude "
        "through automatically, so don't wire it yourself. A single verb → one "
        "step. NONE fit → {\"steps\":[]}.\nVerbs:\n" + manifest_for_prompt() +
        "\nmaterial_key is a common solid (iron, copper, lead, steel_mild, "
        "aluminum, titanium). Lengths in metres, speeds in m/s, Mach a number. "
        "Do NOT compute physics — only choose verbs and fill slots.")
    body = json.dumps({
        "model": model, "stream": False, "format": "json",
        "options": {"temperature": temperature},
        "messages": [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": question}],
    }).encode()
    try:
        req = urllib.request.Request(url + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["message"]["content"]
    except Exception:
        return None


def _qwen_plan(question: str) -> SimulationSpec | None:
    """qwen → an ordered, auto-bound multi-step plan (1+ verbs)."""
    raw = _ollama_plan(question)
    if not raw:
        return None
    try:
        raw_steps = json.loads(raw).get("steps", [])
        if not isinstance(raw_steps, list) or not raw_steps:
            return None
        steps = []
        for rs in raw_steps:
            verb = rs.get("verb")
            if verb not in VERB_MANIFEST:
                return None
            declared = VERB_MANIFEST[verb]["slots"]
            p_in = rs.get("params", {}) or {}
            params = {}
            for slot in declared:
                if slot in p_in and p_in[slot] is not None:
                    if slot == "material_key":
                        params[slot] = _coerce_material(p_in[slot])
                    elif declared[slot].get("unit") in ("object", "body",
                                                        "environment"):
                        # STRING slots survive (manifest-driven): float("wine
                        # glass") used to silently delete drop_object's
                        # object_name, orphaning the verb qwen chose.
                        v = str(p_in[slot]).strip()
                        if v:
                            params[slot] = v
                    else:
                        try:
                            params[slot] = float(p_in[slot])
                        except (TypeError, ValueError):
                            pass
            steps.append(SpecStep(verb, params))
        _auto_bind(steps)
        # Fix C — qwen must not force-fit a drop onto a static/fluid scene (no verb
        # for it). Decline → translate() falls through to clarify, not a fake drop.
        low = question.lower()
        if any(s.verb == "drop_object" for s in steps) and (
                any(k in low for k in _OUT_OF_SCOPE) or not _is_drop(low)):
            return None
        n = len(steps)
        return SimulationSpec(question, steps, source="qwen",
                              note=f"planned by qwen2.5:7b ({n} step{'s' * (n != 1)})")
    except Exception:
        return None


# ── Public API ──────────────────────────────────────────────────────────
def translate(question: str, use_qwen: bool = True) -> SimulationSpec:
    """Natural language → SimulationSpec (deterministic, else qwen, else clarify)."""
    verbs = _classify_verbs(question)
    if verbs:
        steps = [SpecStep(v, _params_for(v, question)) for v in verbs]
        _auto_bind(steps)
        return SimulationSpec(question, steps, source="deterministic")

    if use_qwen:
        spec = _qwen_plan(question)
        if spec and spec.is_runnable():
            return spec

    return SimulationSpec(question, [], source="clarify", note=_CLARIFY)


def answer(question: str, use_qwen: bool = True, ledger=None,
           verify: bool = False) -> str:
    """Translate, run, and narrate — the full babble→simulation→answer path.

    INFALLIBILITY GATE: a run whose results don't pass their own self-check (or
    produce no grounded value) is REFUSED, not reported — so a Materia answer is
    EXACT or ``[refused due to incompetence]``, never confidently wrong. Every
    refusal is logged (reason → the Phase-2 backlog)."""
    from . import groundedness as _g
    spec = translate(question, use_qwen=use_qwen)
    if not spec.is_runnable():
        return f"❓ {spec.note}"
    # GATE 2 for the LLM residual: a qwen-routed answer must survive a
    # route-consistency cross-check (deterministic routes are already reliable —
    # 100% on the corpus — so they skip it). An uncertain route → refuse.
    if verify and spec.source == "qwen":
        if not _route_consistent(question, use_qwen=use_qwen):
            v = _g.Verdict(False, _g.CROSS_CHECK_FAILED,
                           "router not self-consistent — uncertain which tool")
            (ledger or _g.DEFAULT_LEDGER).record(question, v)
            return _g.refuse_text(v)
    results = run_spec(spec)
    verdict = _g.gate_results(results)
    if not verdict.grounded:
        (ledger or _g.DEFAULT_LEDGER).record(question, verdict)
        return _g.refuse_text(verdict)
    routes = "; ".join(
        f"{s.verb}(" + ", ".join(f"{k}={v}" for k, v in s.params.items()) + ")"
        for s in spec.steps)
    header = f"[routed → {routes}  via {spec.source}]"
    return header + "\n\n" + "\n\n".join(r.render() for r in results)
