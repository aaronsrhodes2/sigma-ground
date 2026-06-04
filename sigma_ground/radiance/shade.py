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
    return (albedo * (scene.light_color * intensity)).clamp(0.0, 1.0)
