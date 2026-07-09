"""Shading — color comes from material physics, never from a texture file.

For metals, the albedo is *derived*: `optics.get_material_color('metal', key)`
runs the Drude/Fresnel response of the metal's electrons. That's the whole
point — the color is emergent from what the thing is made of.

Honest v1 scope:
  - Metals: emergent color (real).
  - Dielectrics (ceramic, glass, water…): a flagged neutral STUB. Their true
    color comes from molecular absorption/band-gap, which is the roadmap rung
    "color from molecular content" — not yet wired. We do not fake it with a
    paint chip; we return a labeled grey so the render is honest about what it
    does and doesn't yet know.
  - Lighting is Lambert + ambient — a v1 shading approximation. The *color* is
    physics; the *light transport* (specular, shadows, refraction) is v2.
"""
from __future__ import annotations

from ..dynamics.vec import Vec3

_DIELECTRIC_STUB = Vec3(0.72, 0.72, 0.72)   # [v1 stub — awaiting molecular color]

# Incandescence constants — the EXACT literals viewer.js bakes into GLSL
# (tests/test_shade_thermal.py greps the JS for them, so the two renderers
# cannot drift apart silently).
_DRAPER_K = 700.0            # below ~700 K emission is entirely IR (Draper point)
_C2_WIEN = 1.4388e-2         # c2 = h·c/k_B (m·K), Planck's second radiation constant
_EMISSION_SCALE = 2400000000.0
_LAMBDA_RGB = (650e-9, 550e-9, 450e-9)      # the shader's three sample wavelengths


def incandescence(T_k: float, emissivity_rgb) -> Vec3:
    """Planck's law × Kirchhoff emissivity — nature's glow, no colour table.

    The twin of viewer.js ``incandescence()``: spectral radiance at (650, 550,
    450) nm, ε(λ)-weighted, zero below the Draper point. NON-DERIVED (audit):
    the Planck × Kirchhoff spectrum IS physics; the Reinhard tone-map + 1.7
    gain (and _EMISSION_SCALE) is a camera/exposure choice — the error term
    between the real glow and the pixel, not the glow.
    """
    import math
    if T_k < _DRAPER_K:
        return Vec3(0.0, 0.0, 0.0)
    e = []
    for lam, eps in zip(_LAMBDA_RGB, emissivity_rgb):
        x = _C2_WIEN / (lam * T_k)
        L = eps / ((lam / 650e-9) ** 5 * (math.exp(x) - 1.0))   # ε(λ)·B(λ,T)
        e.append(L * _EMISSION_SCALE)
    peak = max(e)
    scale = 1.7 / (1.0 + peak)               # tone-map: compress brightness, keep the Planck hue
    return Vec3(e[0] * scale, e[1] * scale, e[2] * scale)


def material_albedo(material_key: str) -> Vec3:
    """Base reflectance color for a material — emergent for metals."""
    try:
        from ..field.interface.surface import MATERIALS
        from ..field.interface.optics import get_material_color
        mat = MATERIALS.get(material_key, {})
        if mat.get("material_type") == "metal":
            r, g, b = get_material_color("metal", material_key)
            return Vec3(r, g, b)
    except Exception:
        pass
    return _DIELECTRIC_STUB


def shade(scene, point: Vec3, normal: Vec3, view_dir: Vec3) -> Vec3:
    """Lambert diffuse + ambient, tinted by the emergent material albedo."""
    # The ray halts a hair OUTSIDE the surface (sdf ≈ +eps), where no leaf is
    # strictly inside and material_at would miss → grey fallback. Sample the
    # material just INSIDE, along −normal (the GPU shader does the identical hop).
    label = scene.material_at(point - normal * 1.0e-3)
    # Prefer a scene-supplied albedo (e.g. BAKED emergent colors from a
    # SceneSpec); otherwise derive it from the material library.
    if getattr(scene, "albedo", None) is not None:
        albedo = scene.albedo(label)
    else:
        albedo = material_albedo(label)
    to_light = -scene.light_dir                      # surface is lit from -travel
    diffuse = max(0.0, normal.dot(to_light))
    intensity = scene.ambient + (1.0 - scene.ambient) * diffuse
    col = albedo * (scene.light_color * intensity)
    # Thermal hooks (scene_from_spec wires them): each point glows at ITS
    # temperature — Planck × Kirchhoff, sampled at the same inside point the
    # material was. Hooks absent → identical pre-thermal output.
    t_at = getattr(scene, "temperature_at", None)
    e_of = getattr(scene, "emissivity_of", None)
    if t_at is not None and e_of is not None:
        inside = point - normal * 1.0e-3
        col = col + incandescence(t_at(inside), e_of(label))
    return col.clamp(0.0, 1.0)
