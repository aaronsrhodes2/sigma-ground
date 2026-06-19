"""Bridge — a Materia render-handle (a moving sphere) → entangler still frames.

The sphere-family render kinds (``sphere`` / ``launch_arc`` / ``descent`` /
``horizontal``) are all ONE sphere of a known material and radius. The entangler
renders that sphere natively and *from physics*: cold reflectance from measured
n+k, plus Planck glow when the matter is hot. This bridge reads the scenario's
OWN computed temperature (``peak_T_K`` from the drag-heating energy budget) and
renders the sphere at that temperature — so a slug the physics says reaches
2000 K actually glows on screen. No new physics; the keystone applied to a real
scenario.

Output is browser-viewable PNG via ``radiance.image.write_png`` (pure-stdlib).
Named (non-sphere) objects are out of scope here — they need the Deckard→entangler
CSG converter; this bridge covers the ballistic sphere family, which is the bulk
of the renderable motion catalog.
"""
from __future__ import annotations

import os

from .image import write_png
from .entangler.vec import Vec3
from .entangler.shapes import EntanglerSphere
from .entangler.projection import PushCamera
from .entangler.illumination import PushLight
from .entangler.engine import entangle
from .materials.material import Material

# Scenario material keys → an entangler key that grounds BOTH cold n+k colour and
# the Planck glow (field.interface.optics + thermal_emission). Aliases map the
# real substance to its closest grounded metal (steel≈iron, etc.) — honest proxy.
_GLOW_ALIAS = {
    "steel": "iron", "steel_mild": "iron", "mild_steel": "iron",
    "stainless": "iron", "stainless_steel": "iron", "carbon_steel": "iron",
    "aluminium": "aluminum", "al": "aluminum",
    "brass": "gold", "bronze": "copper", "zinc": "silver",
    "cu": "copper", "fe": "iron", "w": "tungsten", "au": "gold",
    "ag": "silver", "pb": "lead", "ni": "nickel", "ti": "titanium",
    "pt": "platinum",
}
_GROUNDED = {"copper", "iron", "tungsten", "aluminum", "gold", "silver",
             "lead", "nickel", "titanium", "platinum"}

_AMBIENT_K = 288.15
_DRAPER_K = 700.0          # visible-glow threshold (field.thermal_emission)


def entangler_key(material_key):
    """Return the grounded entangler key for a scenario material, or None.

    None means the entangler keeps a neutral fallback colour (no emergent cold
    colour, no glow) — used only for materials outside the grounded metal set.
    """
    k = (material_key or "").strip().lower()
    k = _GLOW_ALIAS.get(k, k)
    return k if k in _GROUNDED else None


def _fallback_color(material_key):
    """A neutral stand-in colour for ungrounded materials (kept static)."""
    return Vec3(0.62, 0.62, 0.64)


def render_sphere_still(material_key, temperature_K, out_path, *,
                        px=220, density=130, label=None):
    """Render one framed portrait of a sphere of ``material_key`` at
    ``temperature_K`` to ``out_path`` (PNG). Returns a small metadata dict.

    The display sphere is a fixed size (the real radius rides in the caption, not
    the pixels); only the material and temperature drive appearance.
    """
    key = entangler_key(material_key)
    mat = Material(
        material_key or "material",
        _fallback_color(material_key),
        material_key=key,                       # None → fallback colour, no glow
        temperature_K=float(temperature_K),
    )
    sphere = EntanglerSphere(Vec3(0.0, 0.0, 0.0), 1.25, mat)
    cam = PushCamera(Vec3(0.0, 0.35, 5.0), Vec3(0.0, 0.0, 0.0), px, px, fov=46)
    light = PushLight(Vec3(3.5, 3.5, 5.0), intensity=1.1)
    bg = Vec3(0.05, 0.05, 0.06)                 # dark "far wall" so the glow pops

    pixels = entangle([sphere], cam, light, density=density, bg_color=bg,
                      jitter={"frame": 0})
    rgb = bytearray()
    for row in pixels:
        for p in row:
            rgb.extend(p.to_rgb())
    write_png(out_path, px, px, bytes(rgb))

    glowing = float(temperature_K) >= _DRAPER_K
    return {"path": out_path, "T_K": round(float(temperature_K), 1),
            "grounded": key is not None, "entangler_key": key,
            "glowing": glowing, "label": label}


def _keyframe_temps(outputs, inputs):
    """Pick (label, T) keyframes from a scenario's results.

    Always an in-flight frame at ambient. If the scenario computed a peak
    temperature above the Draper point, add a peak-heating frame (and, when very
    hot, a mid-heating frame) so the gallery shows the cold→glowing ramp the
    physics produced.
    """
    ambient = float((inputs or {}).get("T") or _AMBIENT_K)
    peak = outputs.get("peak_T_K")
    try:
        peak = float(peak) if peak is not None else None
    except (TypeError, ValueError):
        peak = None

    frames = [("in flight", ambient)]
    if peak is not None and peak >= _DRAPER_K:
        if peak >= 1500.0:
            frames.append(("heating", 0.5 * (ambient + peak)))
        frames.append(("peak heating", peak))
    return frames


def render_handle_gallery(handle, outputs, inputs, out_dir, slug, *, px=220):
    """Render the keyframe stills for one sphere-family render-handle.

    Returns {"slug", "material", "renderable", "stills":[...], "peak_T_K", "note"}.
    ``renderable`` is False (with a note) for non-sphere kinds — they need the
    Deckard CSG converter and are skipped here.
    """
    kind = (handle or {}).get("kind")
    sphere_kinds = {"sphere", "launch_arc", "descent", "horizontal"}
    material_key = (handle or {}).get("material_key") or \
        (handle or {}).get("label") or "iron"

    if kind not in sphere_kinds:
        return {"slug": slug, "material": material_key, "renderable": False,
                "stills": [], "peak_T_K": outputs.get("peak_T_K"),
                "note": f"kind={kind!r} needs the Deckard→entangler CSG converter "
                        f"(named-object family) — not rendered by this bridge"}

    os.makedirs(out_dir, exist_ok=True)
    stills = []
    for i, (lbl, T) in enumerate(_keyframe_temps(outputs, inputs)):
        fname = f"{slug}_{i}_{lbl.replace(' ', '-')}.png"
        path = os.path.join(out_dir, fname)
        meta = render_sphere_still(material_key, T, path, px=px, label=lbl)
        meta["file"] = fname
        stills.append(meta)

    return {"slug": slug, "material": material_key, "renderable": True,
            "stills": stills, "peak_T_K": outputs.get("peak_T_K"),
            "entangler_key": entangler_key(material_key), "note": None}


if __name__ == "__main__":  # quick self-test
    out = os.path.join(os.path.dirname(__file__), "entangler", "_bridge_selftest")
    os.makedirs(out, exist_ok=True)
    # cold copper vs a glowing tungsten reentry slug
    a = render_handle_gallery(
        {"kind": "sphere", "material_key": "copper", "radius_m": 0.05},
        {"impact_speed_m_s": 95.0}, {"T": 288.15}, out, "copper_drop")
    b = render_handle_gallery(
        {"kind": "descent", "material_key": "tungsten", "radius_m": 0.1},
        {"peak_T_K": 2200.0}, {"T": 288.15}, out, "tungsten_reentry")
    import json
    print(json.dumps([a, b], indent=2))
