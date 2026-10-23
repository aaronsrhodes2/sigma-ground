"""What a Materia verb needs, and what to SUGGEST when the question leaves it out.

The Captain's rule (2026-10-23): Mentat never fills in a value by itself. A verb's
inputs are its manifest slots plus the physical parameters of its scenario
function. When the question (or a chain binding, or the caller's ``given``) does
not supply one, Mentat computes NOTHING: it answers with a clarification that
names the value it would use, its units and why, and the caller asks again with
those values given explicitly once the person agrees ("yes", "sure", "go ahead").

This replaced the silent fill-in (2026-06 → 2026-10): ``run_spec`` called each
scenario with only the parameters the translator found, so every other input
took the Python default without a word. That is how "a baseball at 100 miles
per hour" became 30 m/s: ``projectile_motion`` declared no slots at all, so the
question's speed and angle never reached it and its defaults (30 m/s, 45°) ran.

The old default values are kept, as the SUGGESTIONS. Nothing is applied until
the caller sends it back in ``given``.
"""
from __future__ import annotations

import inspect
import re

from .manifest import VERB_MANIFEST

# Numerical settings of a solver, not physical assumptions about the question:
# never suggested, never asked for.
SOLVER_SETTINGS = {"tolerance"}

# Inputs that are alternatives to each other: any one of a group given means the
# others are not missing. When none is given, the first member with a default is
# the one suggested.
ALTERNATIVES = {
    "orbital_velocity": [("orbiter", "period_s", "semimajor_axis_au", "altitude_m")],
    "mass_energy": [("energy_j", "particle", "mass_kg")],
    "black_hole": [("mass_kg", "solar_masses")],
    "relativistic_motion": [("particle", "rest_mass_kg")],
}

# Units of an input, from its name (the scenario functions take SI, named by
# suffix). Checked in order; the first match wins.
_UNIT_RULES = (
    (r"(^|_)m_s$|^v[12]$|velocity_m_s$", "m/s"),
    (r"rad_s$", "rad/s"),
    (r"kg_m3$", "kg/m^3"),
    (r"_m3$", "1/m^3"),
    (r"_m2$", "m^2"),
    (r"L_per_m$", "H/m"),
    (r"_kg$|^m[12]$", "kg"),
    (r"_um$", "micrometer"),
    (r"_nm$", "nanometer"),
    (r"_m$", "m"),
    (r"_deg$", "degree"),
    (r"_eV$", "eV"),
    (r"(_k|_K)$|^T0?$|^t_(hot|cold|c)$|^T_hot$", "K"),
    (r"_s$", "s"),
    (r"_t$", "tesla"),
    (r"^charge_c$", "coulomb"),
    (r"_pa$", "Pa"),
    (r"_atm$", "atm"),
    (r"_j$", "J"),
    (r"_ohm$", "ohm"),
    (r"_F$", "F"),
    (r"_H$", "H"),
    (r"^volts$", "V"),
    (r"^solar_masses$", "solar_mass"),
)

# Why each suggestion is what it is, for the common inputs. Anything else says
# plainly that it is the scenario's standing illustrative value.
_WHY = {
    ("projectile_motion", "speed_m_s"): "a hard throw by hand, about 67 mph",
    ("projectile_motion", "angle_deg"): "the angle that goes farthest without air",
    "material_key": "a common metal",
    "bell_material": "cast metal, as most bells are",
    "radius_m": "a fist-sized ball",
    "drop_altitude_m": "the height of a cruising airliner",
    "start_altitude_m": "a stratospheric jump",
    "T": "the standard atmosphere at sea level, 15 °C",
    "T0": "the standard atmosphere at sea level, 15 °C",
    "T0_K": "room temperature",
    "temperature_k": "room temperature",
    "central_body": "the planet we are on",
    "altitude_m": "the International Space Station's orbit",
    "mass_kg": "a round one-kilogram mass",
    "launch_speed_m_s": "a fast launch, a rifle-bullet speed",
    "launch_mach": "a supersonic rifle round",
    "particle": "the lightest charged particle",
    "velocity_fraction_c": "a fast particle, where relativity matters",
    "solar_masses": "a typical stellar black hole",
    "duration_s": "one year of exposure",
    "body": "the planet we are on",
}


# What to call an input in a sentence. Anything else: its name without the unit suffix.
_LABEL = {
    "speed_m_s": "launch speed", "angle_deg": "launch angle", "T": "air temperature", "T0": "starting temperature",
    "T0_K": "starting temperature", "temperature_k": "temperature", "material_key": "material",
    "radius_m": "ball radius", "drop_altitude_m": "drop height", "start_altitude_m": "starting altitude",
    "altitude_m": "orbit altitude", "central_body": "body it orbits", "launch_speed_m_s": "launch speed",
    "launch_mach": "launch speed (Mach number)", "mass_kg": "mass", "particle": "particle",
    "velocity_fraction_c": "speed (fraction of light speed)", "solar_masses": "mass (in Suns)",
    "duration_s": "exposure time", "body": "body",
}
_SUFFIX = re.compile(r"_(m_s|rad_s|kg_m3|m3|m2|kg|um|nm|m|deg|eV|k|K|s|t|c|pa|atm|j|ohm|F|H|key)$")


def label(name: str) -> str:
    return _LABEL.get(name) or _SUFFIX.sub("", name).replace("_", " ")


# Manifest unit labels that are not units pint knows: a count, a ratio, a label.
_NOT_UNITS = {"material", "object", "body", "environment", "particle", "mach", "Z", "c", "count", "preset"}
_PINT_NAME = {"Msun": "solar_mass", "degree": "degree"}


def _units_of(name: str, slot: dict | None) -> str:
    unit = (slot or {}).get("unit")
    if unit in _NOT_UNITS:
        return ""
    if unit:
        return _PINT_NAME.get(unit, unit)
    for pattern, u in _UNIT_RULES:
        if re.search(pattern, name):
            return u
    return ""


def _why(verb: str, name: str) -> str:
    return (_WHY.get((verb, name)) or _WHY.get(name)
            or "the scenario's standing illustrative value, not a measured one")


def verb_inputs(verb: str) -> dict:
    """{input name: {"default", "units", "kind"}} for a verb: its manifest slots
    and its scenario function's parameters, minus solver settings. ``kind`` is
    "number" or "text". An input whose default is None is optional (the verb
    works it out, or it is one of a group of alternatives)."""
    from .scenarios import SCENARIOS
    slots = VERB_MANIFEST.get(verb, {}).get("slots", {}) or {}
    out = {}
    fn = SCENARIOS.get(verb)
    params = inspect.signature(fn).parameters if fn else {}
    names = list(slots) + [n for n in params if n not in slots]
    for name in names:
        if name in SOLVER_SETTINGS:
            continue
        slot = slots.get(name)
        if name in params and params[name].default is not inspect.Parameter.empty:
            default = params[name].default
        elif slot is not None:
            default = slot.get("default")
        else:
            continue                      # a required argument: the verb can't run without it
        if default == "":
            default = None                # an empty label: the verb's "not given", nothing assumed
        units = _units_of(name, slot)
        textual = slot is not None and slot.get("unit") in ("material", "object", "body", "environment",
                                                            "particle", "preset")
        if default is None:
            kind = "text" if textual or not (units or (slot or {}).get("unit") in ("c", "Z", "count", "mach")) else "number"
        else:
            kind = "number" if isinstance(default, (int, float)) and not isinstance(default, bool) else "text"
        if textual:
            kind = "text"
        out[name] = {"default": default, "units": units, "kind": kind}
    return out


def _covered(verb: str, name: str, have: set) -> bool:
    for group in ALTERNATIVES.get(verb, []):
        if name in group:
            return any(n in have for n in group)
    return name in have


def missing_inputs(spec) -> dict:
    """The inputs a runnable spec does not supply: {name: suggestion}, in step
    order, each suggestion {"value", "units", "why", "verb"}. A name shared by two
    steps (a launch and its descent both take a material) is suggested once; the
    caller's ``given`` value then fills every step that takes it."""
    missing: dict = {}
    for step in spec.steps:
        have = set(step.params) | set(step.bindings)
        inputs = verb_inputs(step.verb)
        alt_members = {n for g in ALTERNATIVES.get(step.verb, []) for n in g}
        for name, meta in inputs.items():
            if _covered(step.verb, name, have) or name in missing:
                continue
            if meta["default"] is None:
                continue                  # optional, or an alternative with no default of its own
            if name in alt_members:
                group = next(g for g in ALTERNATIVES[step.verb] if name in g)
                first = next((n for n in group if inputs.get(n, {}).get("default") is not None), None)
                if first != name:
                    continue              # one suggestion per group of alternatives
            missing[name] = {"value": meta["default"], "units": meta["units"],
                             "why": _why(step.verb, name), "verb": step.verb}
    return missing


def _fmt(value, units: str) -> str:
    if isinstance(value, float):
        txt = f"{value:g}"
    else:
        txt = str(value).replace("_", " ")
    if units == "degree":
        return f"{txt}°"
    return f"{txt} {units}".strip()


def suggestion_sentence(missing: dict) -> str:
    """One plain sentence naming every suggested value, ending in a question."""
    parts = [f"{_fmt(s['value'], s['units'])} for the {label(name)} ({s['why']})"
             for name, s in missing.items()]
    if len(parts) == 1:
        body = parts[0]
    else:
        body = "; ".join(parts[:-1]) + "; and " + parts[-1]
    return f"Your question doesn't give everything I need. I'd assume {body}. Use {'that' if len(parts) == 1 else 'those'}?"


def clarification(question: str, missing: dict, given: dict | None = None) -> dict:
    """The machine-readable suggestion: what is missing, what Mentat would use,
    and the ready-to-send follow-up for a "yes" (the same question, with every
    suggested value given explicitly; Mentat is stateless, so the caller carries it)."""
    suggested = {n: {**{k: s[k] for k in ("value", "units", "why")}, "label": label(n)} for n, s in missing.items()}
    follow = dict(given or {})
    follow.update({n: s["value"] for n, s in missing.items()})
    return {"needs": list(missing), "suggested": suggested,
            "sentence": suggestion_sentence(missing),
            "follow_up": {"question": question, "given": follow}}


def apply_given(spec, given: dict | None) -> dict:
    """Put the caller's explicit values into every step that takes them. A value
    may be a number in the input's own units, or a quantity string ("100 mph",
    "60 km/h") converted with Mentat's unit tool. Returns {name: problem} for
    values that could not be used (an unknown input, or a unit that won't convert);
    those inputs then stay missing."""
    problems = {}
    if not given:
        return problems
    known = set()
    for step in spec.steps:
        inputs = verb_inputs(step.verb)
        for name, raw in given.items():
            if name not in inputs:
                continue
            known.add(name)
            meta = inputs[name]
            value, err = coerce(raw, meta)
            if err:
                problems[name] = err
                continue
            step.params[name] = value
            step.bindings.pop(name, None)
    for name in given:
        if name not in known:
            problems[name] = "none of the steps for this question takes that input"
    return problems


def coerce(raw, meta: dict):
    """(value, None) or (None, why it can't be used)."""
    if meta["kind"] == "text":
        txt = str(raw).strip()
        return (txt, None) if txt else (None, "empty")
    if isinstance(raw, bool):
        return None, "a number is needed"
    if isinstance(raw, (int, float)):
        return float(raw), None
    txt = str(raw).strip()
    try:
        return float(txt), None
    except ValueError:
        pass
    from .quantities import to_units
    v = to_units(txt, meta["units"])
    if v is None:
        return None, f"couldn't read {txt!r} as {meta['units'] or 'a number'}"
    return v, None
