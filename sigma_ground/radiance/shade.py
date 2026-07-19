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


def _cosine_hemisphere_dirs(n: int):
    """n DETERMINISTIC cosine-weighted directions on the +z hemisphere
    (Fibonacci spiral in azimuth, sqrt-stratified in altitude so sample
    density ∝ cos θ — a plain average of per-ray visibility then
    approximates the cosine-weighted occlusion integral). Deterministic on
    purpose: no RNG means renders and tests are exactly reproducible,
    matching this renderer's precompute-once-replay doctrine."""
    import math
    ga = math.pi * (3.0 - math.sqrt(5.0))            # golden angle
    dirs = []
    for i in range(n):
        u = (i + 0.5) / n
        st = math.sqrt(u)                             # sin(theta)
        ct = math.sqrt(1.0 - u)                       # cos(theta)
        phi = ga * i
        dirs.append((st * math.cos(phi), st * math.sin(phi), ct))
    return dirs


def ambient_occlusion(sdf, point: Vec3, normal: Vec3, rays: int,
                      reach: float) -> float:
    """Teardown-style ambient occlusion: march a few cosine-weighted
    secondary rays from the hit point; each ray's hit DISTANCE sets its
    darkening (near hit = strongly occluded, hit at the reach limit = not
    occluded at all) — the verified real mechanism behind Teardown's
    lighting, which has no global illumination either (research finding,
    2026-07-18: ~2 cosine-weighted rays/pixel, hit distance sets AO).

    Returns a visibility factor in [0, 1]: 1 = fully open hemisphere.
    A ray that runs out of march steps without a verdict counts as open —
    a v1 simplification, noted here rather than hidden."""
    from .raymarch import march

    # tangent basis around the normal (same deterministic construction
    # dynamics/joints.py uses for its constraint tangents)
    ref = Vec3(1.0, 0.0, 0.0) if abs(normal.x) < 0.9 else Vec3(0.0, 1.0, 0.0)
    t1 = normal.cross(ref).normalized()
    t2 = normal.cross(t1)

    origin = point + normal * 4.0e-4      # hop off the surface (> march eps)
    total = 0.0
    for (dx, dy, dz) in _cosine_hemisphere_dirs(rays):
        d = (t1 * dx + t2 * dy + normal * dz).normalized()
        t = march(sdf, origin, d, max_dist=reach, max_steps=32)
        total += 1.0 if t is None else min(1.0, t / reach)
    return total / rays


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
    ambient = scene.ambient
    # Opt-in secondary-ray AO (scene.ao_rays > 0): occlusion attenuates the
    # ISOTROPIC ambient term only — ambient is this v1 pipeline's stand-in
    # for indirect sky/bounce light, which is exactly what nearby geometry
    # blocks. The directional term is left alone: shadowing the light
    # itself is a separate (future) transport rung, not conflated here.
    ao_rays = getattr(scene, "ao_rays", 0)
    if ao_rays:
        reach = getattr(scene, "ao_reach", None) or 0.1 * scene.max_dist
        ambient = ambient * ambient_occlusion(scene.sdf, point, normal,
                                              ao_rays, reach)
    intensity = ambient + (1.0 - scene.ambient) * diffuse
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
