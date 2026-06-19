"""Per-site physics response — a lattice node answers a ray from real, cited
physics, not a stored colour.

Given a material key and its current temperature, this returns the emergent
appearance: the cold reflectance colour (Drude/Fresnel from measured n+k), the
Planck thermal glow (colour from Planck × Kirchhoff emissivity, brightness a
tonemap of the cited εσT⁴ power, gated at the Draper point), and the specular
fraction (Rayleigh, from thermal roughness). Every value traces to a
``sigma_ground.field.interface`` function — none is invented.

This is the heart of "direct physics to pixel": the renderer asks the matter
"what colour are you at this temperature?" and the matter answers from physics.
This is the deterministic-MEAN render core; the per-photon stochastic version
("sometimes the atom answers") is a later slice.
"""
from __future__ import annotations

import math

from ...field.interface import optics, thermal, thermal_emission, texture

# A DISPLAY tonemap of emitted power (W/m²) → [0,1] glow brightness. The glow
# COLOUR is physics (Planck × emissivity); this only maps the cited intensity to a
# screen value — no display-referred radiometry is "correct" without a full
# tonemapper, so this constant is flagged as a tonemap, not a physics claim.
_GLOW_P0 = 3000.0

# Memoise: a whole object's thousands of nodes share one (material, T), so the
# field functions run once per material+temperature, not once per node.
_CACHE: dict = {}


def site_response(material_key: str, temperature_K: float) -> dict:
    """Emergent per-site appearance for ``(material_key, temperature_K)``.

    Returns ``{cold_rgb, glow_rgb, glow_power, glow_level, glowing,
    specular_fraction, sources}``. ``cold_rgb`` is None when the material isn't a
    metal we can ground from measured n+k (the caller then keeps its own colour —
    e.g. a dielectric on the legacy Beer–Lambert path).
    """
    key = (material_key, round(float(temperature_K), 1))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    T = float(temperature_K)

    # COLD — the material's true colour from measured n+k (Drude/Fresnel), emergent,
    # never chosen. None if not a groundable metal → caller keeps its colour.
    cold_rgb = None
    src_cold = None
    try:
        cold_rgb = tuple(optics.metal_rgb(material_key))
        src_cold = "optics.metal_rgb (Drude/Fresnel, measured n+k Palik/JC72)"
    except Exception:
        cold_rgb = None

    # GLOW — Planck × Kirchhoff emissivity for the colour, gated at the Draper
    # point (~798 K); brightness a tonemap of the cited εσT⁴ power.
    glowing = bool(thermal_emission.is_visibly_glowing(T))
    glow_rgb = (0.0, 0.0, 0.0)
    glow_power = 0.0
    glow_level = 0.0
    if glowing:
        glow_rgb = tuple(thermal_emission.thermal_emission_rgb(material_key, T))
        try:
            glow_power = float(thermal.thermal_emission_power(material_key, T))
        except Exception:
            glow_power = 0.0
        glow_level = 1.0 - math.exp(-max(0.0, glow_power) / _GLOW_P0)

    # SPECULAR fraction — Rayleigh criterion on thermal roughness (hotter → rougher
    # → more diffuse). Exposed for the shade; the view-dependent highlight is a
    # later refinement (illuminate_node has no view vector yet).
    try:
        specular_fraction = float(texture.specular_fraction(material_key, T))
    except Exception:
        specular_fraction = None

    out = {
        "cold_rgb": cold_rgb,
        "glow_rgb": glow_rgb,
        "glow_power": glow_power,
        "glow_level": glow_level,
        "glowing": glowing,
        "specular_fraction": specular_fraction,
        "sources": {
            "cold_rgb": src_cold,
            "glow": "thermal_emission.thermal_emission_rgb (Planck×Kirchhoff) + "
                    "thermal.thermal_emission_power (εσT⁴); Draper gate",
            "specular_fraction":
                "texture.specular_fraction (Rayleigh / thermal roughness)",
        },
    }
    _CACHE[key] = out
    return out


__all__ = ["site_response"]
