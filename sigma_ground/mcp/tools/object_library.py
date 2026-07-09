"""Object resolution for the playground 'make' verb: guess when you reasonably
can, ASK when a missing variable is decisive.

Two everyday situations:
  - "make a brick"            -> a brick has a standard size; fill it in.
  - "make a ball of plutonium" -> a sphere's behavior is DECISIVE in its size
                                  (a pellet is inert; a hand-sized sphere is at/
                                  above critical mass). Guessing is not OK -> ask.

This module holds the by-the-book knowledge (canonical object dimensions, a few
densities the surface DB lacks, and bare-sphere critical masses) and the
resolve() decision. It does NOT build scenes -- playground.py consumes it.
"""
from __future__ import annotations

import math
import re
from typing import Any

# Densities the surface MATERIALS DB lacks but we need for 'make' (kg/m^3).
_EXTRA_DENSITY = {
    "plutonium": 19816.0,        # alpha-phase Pu-239
    "uranium_235": 18900.0,      # highly-enriched U metal
    "fired_clay": 1920.0,        # red building brick
    "wood": 700.0,
    "ice": 917.0,
    "stone": 2600.0,
}

# Bare-sphere critical masses (kg) -- APPROXIMATE, unreflected. The point is the
# order of magnitude: a hand-sized fissile sphere is near/above these.
_CRITICAL_MASS_KG = {
    "plutonium": 10.0,           # Pu-239, bare ~10 kg (with reflector ~5-6)
    "uranium_235": 52.0,         # U-235, bare ~52 kg
}

# Materials whose behavior is DECISIVE in size -> never silently guess a size.
_SIZE_DECISIVE = {
    "plutonium": ("A plutonium sphere's behavior is dominated by its size: a "
                  "small pellet is essentially inert, but a hand-sized sphere "
                  "approaches/exceeds the bare critical mass (~10 kg, ~5 cm "
                  "radius) and would be a supercritical (weapons) assembly. "
                  "Guessing the size could silently imply either -- so I need it."),
    "uranium_235": ("Enriched-uranium sphere behavior is size-decisive: below "
                    "the bare critical mass (~52 kg) it is subcritical; above it, "
                    "supercritical. I won't guess the size."),
}

# Everyday named objects with standard dimensions (so we DON'T have to ask).
# material is a surface.MATERIALS key or an _EXTRA_DENSITY key; shape sphere|box.
CANONICAL_OBJECTS: dict[str, dict[str, Any]] = {
    "brick":        {"material": "fired_clay", "shape": "box",
                     "dims_m": (0.215, 0.1025, 0.065),
                     "note": "standard UK red clay building brick (215x102.5x65 mm)"},
    "marble":       {"material": "glass", "shape": "sphere", "radius_m": 0.008,
                     "note": "a toy glass marble (16 mm diameter)"},
    "cannonball":   {"material": "iron", "shape": "sphere", "radius_m": 0.05,
                     "note": "a 6-pounder iron cannonball (~10 cm diameter)"},
    "ball_bearing": {"material": "steel_mild", "shape": "sphere", "radius_m": 0.005,
                     "note": "a 10 mm steel ball bearing"},
    "golf_ball":    {"material": "plastic_abs", "shape": "sphere", "radius_m": 0.0214,
                     "note": "a regulation golf ball (42.7 mm diameter)"},
    "bowling_ball": {"material": "plastic_abs", "shape": "sphere", "radius_m": 0.1088,
                     "note": "a 10-pin bowling ball (~21.8 cm diameter)"},
    "die":          {"material": "plastic_abs", "shape": "box",
                     "dims_m": (0.016, 0.016, 0.016),
                     "note": "a standard 16 mm gaming die"},
    "coin":         {"material": "copper", "shape": "box",
                     "dims_m": (0.0195, 0.0195, 0.00152),
                     "note": "a coin (~19.5 mm diameter, ~1.5 mm thick)"},
}

_SHAPE_WORDS = {"ball": "sphere", "sphere": "sphere", "orb": "sphere",
                "cube": "box", "block": "box", "brick-shaped": "box",
                "chunk": "sphere", "lump": "sphere", "pellet": "sphere"}

_DEFAULT_RADIUS_M = 0.05         # fist-sized, for non-decisive raw materials
_DEFAULT_NOTE = "assumed a fist-sized ~5 cm sphere (size wasn't decisive here)"


def _known_materials() -> dict[str, float]:
    """material key -> density (kg/m^3), merging the surface DB and our extras."""
    out = dict(_EXTRA_DENSITY)
    try:
        from sigma_ground.field.interface import surface as S
        for k, v in S.MATERIALS.items():
            if isinstance(v, dict) and v.get("density_kg_m3"):
                out.setdefault(k, float(v["density_kg_m3"]))
    except Exception:
        pass
    return out


def _match_material(text: str, mats: dict[str, float]) -> str | None:
    t = text.lower()
    # direct/alias hits first (longer keys win, so 'uranium_235' before 'uranium')
    aliases = {"plutonium": "plutonium", "pu": "plutonium",
               "u-235": "uranium_235", "u235": "uranium_235",
               "enriched uranium": "uranium_235", "heu": "uranium_235",
               "uranium": "depleted_uranium",   # bare 'uranium' -> the DB's DU
               "clay": "fired_clay", "steel": "steel_mild"}
    for alias, key in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
        if alias in t:
            return key
    for k in sorted(mats, key=lambda s: -len(s)):
        if k.replace("_", " ") in t or k in t:
            return k
    return None


def _criticality(material: str, mass_kg: float) -> dict[str, Any] | None:
    cm = _CRITICAL_MASS_KG.get(material)
    if cm is None or mass_kg is None:
        return None
    ratio = mass_kg / cm
    if ratio >= 1.0:
        verdict = "SUPERCRITICAL (above bare critical mass — a runaway/weapons assembly)"
    elif ratio >= 0.7:
        verdict = "near-critical (approaching the bare critical mass)"
    else:
        verdict = "subcritical (below the bare critical mass)"
    return {"bare_critical_mass_kg": cm, "mass_over_critical": round(ratio, 3),
            "verdict": verdict}


def _parse_size(size: Any) -> float | None:
    """Coerce a size to metres. Accepts a number, or a string like '3 cm',
    '5mm', '0.05 m', '0.07'. Returns None if it can't be parsed."""
    if size is None:
        return None
    if isinstance(size, (int, float)):
        return float(size)
    s = str(size).strip().lower()
    m = re.match(r"([0-9]*\.?[0-9]+)\s*(mm|cm|m|km)?", s)
    if not m:
        return None
    val = float(m.group(1))
    return val * {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "km": 1e3,
                  None: 1.0, "": 1.0}.get(m.group(2), 1.0)


def resolve(object_desc: str, size_m: float | None = None) -> dict[str, Any]:
    """Resolve a 'make X' request to a concrete object spec, OR a clarification.

    Returns one of:
      {"status": "made", material, shape, radius_m|dims_m, mass_kg, note, criticality?}
      {"status": "clarify", variable, question, reason, suggestions}
      {"status": "unknown", note}
    """
    desc = (object_desc or "").strip().lower()
    size_m = _parse_size(size_m)
    mats = _known_materials()

    # 1) Canonical named object -> fill in its standard size.
    for name, spec in CANONICAL_OBJECTS.items():
        if name.replace("_", " ") in desc or name in desc:
            return _build(spec["material"], spec["shape"], mats,
                          radius_m=spec.get("radius_m"), dims_m=spec.get("dims_m"),
                          note=f"assumed {spec['note']}")

    # 2) "<shape> of <material>" (or just a material name).
    material = _match_material(desc, mats)
    shape = next((sh for w, sh in _SHAPE_WORDS.items() if w in desc), "sphere")

    if material is None:
        return {"status": "clarify", "variable": "material",
                "question": f"What material should the {shape} be made of?",
                "reason": "No known material was named in the request.",
                "suggestions": ["copper", "lead", "iron", "glass", "plutonium"]}

    # 3) Size decision.
    if size_m is None:
        if material in _SIZE_DECISIVE:
            return {"status": "clarify", "variable": "size",
                    "question": (f"How big is the {material.replace('_',' ')} "
                                 f"{shape}? (give a radius, e.g. '3 cm' or '0.05 m')"),
                    "reason": _SIZE_DECISIVE[material],
                    "suggestions": ["0.01 m (pellet)", "0.05 m (fist-sized)",
                                    "0.1 m (large)"]}
        radius_m = _DEFAULT_RADIUS_M
        note = _DEFAULT_NOTE
    else:
        radius_m = size_m
        note = f"using your size (radius {radius_m:.3g} m)"

    return _build(material, shape, mats, radius_m=radius_m, note=note)


def _build(material: str, shape: str, mats: dict[str, float],
           radius_m: float | None = None, dims_m: tuple | None = None,
           note: str = "") -> dict[str, Any]:
    rho = mats.get(material)
    if shape == "box":
        if dims_m:
            d = tuple(dims_m)
        else:  # a "cube of X" with no given size -> a cube of side = 2*default r
            side = 2.0 * (radius_m if radius_m is not None else _DEFAULT_RADIUS_M)
            d = (side, side, side)
        vol = d[0] * d[1] * d[2]
        size = {"dims_m": [round(x, 5) for x in d]}
        char = vol ** (1.0 / 3.0)
    else:
        r = radius_m if radius_m is not None else _DEFAULT_RADIUS_M
        vol = (4.0 / 3.0) * math.pi * r ** 3
        size = {"radius_m": r}
        char = r
    mass = (rho * vol) if rho else None
    out: dict[str, Any] = {"status": "made", "material": material, "shape": shape,
                           "mass_kg": (round(mass, 6) if mass is not None else None),
                           "density_kg_m3": rho, "char_size_m": round(char, 5),
                           "note": note, **size}
    crit = _criticality(material, mass) if mass is not None else None
    if crit:
        out["criticality"] = crit
    if rho is None:
        out["note"] = (note + " — no density on file for this material, so mass "
                       "is unknown.").strip(" —")
    return out
