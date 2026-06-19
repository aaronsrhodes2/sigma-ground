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

import math
import os

from .image import write_png
from .entangler.vec import Vec3
from .entangler.shapes import EntanglerSphere
from .entangler.surface_nodes import SurfaceNode
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
    "cast iron": "iron", "cast_iron": "iron", "wrought iron": "iron",
    "wrought_iron": "iron", "pig iron": "iron", "pig_iron": "iron",
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


# Plausible static colours for common non-metal construct substances, so a
# multi-material object (steel head + wood handle) reads correctly. Metals are
# NOT here — they ground to emergent n+k colour via entangler_key instead.
_SUBSTANCE_COLOR = {
    "wood": (0.55, 0.36, 0.20), "oak": (0.60, 0.40, 0.23),
    "pine": (0.78, 0.60, 0.38), "birch": (0.82, 0.68, 0.46),
    "mahogany": (0.40, 0.20, 0.14), "bamboo": (0.80, 0.70, 0.42),
    "plastic": (0.20, 0.22, 0.26), "abs": (0.18, 0.19, 0.22),
    "rubber": (0.10, 0.10, 0.11), "leather": (0.40, 0.26, 0.16),
    "glass": (0.78, 0.86, 0.92), "ceramic": (0.88, 0.86, 0.82),
    "concrete": (0.60, 0.60, 0.58), "stone": (0.52, 0.51, 0.49),
    "cardboard": (0.66, 0.52, 0.36), "paper": (0.90, 0.88, 0.84),
    "fabric": (0.45, 0.45, 0.55), "foam": (0.85, 0.84, 0.80),
}


def _substance_color(substance):
    """Static colour for an ungrounded substance: table lookup, else neutral."""
    s = (substance or "").strip().lower()
    if s in _SUBSTANCE_COLOR:
        return Vec3(*_SUBSTANCE_COLOR[s])
    # crude keyword fallback (e.g. 'red_oak' → wood-ish)
    for key, rgb in _SUBSTANCE_COLOR.items():
        if key in s:
            return Vec3(*rgb)
    return Vec3(0.62, 0.62, 0.64)


def _material_for_substance(substance, temperature_K, _cache):
    """One cached entangler Material per substance — grounded metals get emergent
    colour + glow; everything else gets a sensible static colour."""
    if substance not in _cache:
        key = entangler_key(substance)
        if key:
            mat = Material(substance or key, _fallback_color(substance),
                           material_key=key, temperature_K=float(temperature_K))
        else:
            mat = Material(substance or "material", _substance_color(substance))
        _cache[substance] = mat
    return _cache[substance]


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


# ── Deckard construct → entangler node cloud (the CSG converter) ─────────────

class _NodeCloud:
    """A pre-sampled cloud of SurfaceNodes that the entangler renders directly
    (``shape_type='nodes'``). This is how a Deckard construct — any CSG tree —
    enters the entangler: we sample its real SDF surface, so subtraction and
    clipping come for free (they're already in the final SDF)."""
    shape_type = "nodes"
    fill_volume = False

    def __init__(self, nodes):
        self.nodes = nodes
        self.material = None        # per-node materials; engine reads node.material


def _part_substance(construct):
    """Map a construct's part labels → substances (from its layer table)."""
    return {L.name: L.material for L in getattr(construct, "layers", [])}


def _dominant_substance(construct):
    """Densest non-air substance — the fallback for unlabelled surface points."""
    label_material = _part_substance(construct)
    cands = sorted(((rho, lbl) for lbl, rho in
                    (getattr(construct, "density_by_label", None) or {}).items()
                    if lbl != "air" and rho), reverse=True)
    if not cands:
        return "steel_mild"
    lbl = cands[0][1]
    return label_material.get(lbl, lbl)


def construct_to_node_cloud(construct, n_surface=6000, temperature_K=288.15):
    """Sample a Deckard construct's SDF surface into entangler SurfaceNodes.

    Each node carries: its surface position, the outward normal (normalised SDF
    gradient), and the entangler Material for the substance at that point (real
    per-part materials, so a multi-material object renders correctly). Arbitrary
    CSG is handled because we sample the COMPOSED SDF, not the individual leaves.
    """
    comp = construct.composed
    (x0, x1), (y0, y1), (z0, z1) = construct.bbox
    bounds = ((x0, y0, z0), (x1, y1, z1))
    pts = comp.sample_surface(n_surface, bounds=bounds)

    label_sub = _part_substance(construct)
    dom = _dominant_substance(construct)
    cache = {}
    size = max(x1 - x0, y1 - y0, z1 - z0, 1e-6)
    eps = max(1e-5, 1e-3 * size)        # gradient step, scaled to the object

    nodes = []
    for p in pts:
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        gx = comp.sdf(px + eps, py, pz) - comp.sdf(px - eps, py, pz)
        gy = comp.sdf(px, py + eps, pz) - comp.sdf(px, py - eps, pz)
        gz = comp.sdf(px, py, pz + eps) - comp.sdf(px, py, pz - eps)
        g = math.sqrt(gx * gx + gy * gy + gz * gz)
        normal = Vec3(gx / g, gy / g, gz / g) if g > 1e-12 else Vec3(0.0, 1.0, 0.0)
        lbl = comp.material_at(px, py, pz)
        sub = label_sub.get(lbl, dom) if lbl else dom
        mat = _material_for_substance(sub, temperature_K, cache)
        nodes.append(SurfaceNode(Vec3(px, py, pz), normal, mat))
    return _NodeCloud(nodes), {"substances": sorted(set(
        (label_sub.get(comp.material_at(float(p[0]), float(p[1]), float(p[2])), dom)
         if comp.material_at(float(p[0]), float(p[1]), float(p[2])) else dom)
        for p in pts[:400]))}


def _find_catalog_md(catdir, slug):
    """Match a (possibly short) object slug to a catalog .md stem.

    Tries exact / a_ / an_ prefixes first, then a fuzzy contains-match either way
    (so the router's short noun 'skillet' finds 'cast_iron_skillet.md'). Returns
    the file path or None. Offline — never touches the network.
    """
    if not os.path.isdir(catdir):
        return None
    stems = {f[:-3]: f for f in os.listdir(catdir) if f.endswith(".md")}
    for cand in (slug, f"a_{slug}", f"an_{slug}"):
        if cand in stems:
            return os.path.join(catdir, stems[cand])
    # fuzzy: a stem that contains the slug, or the slug contains a stem
    norm = lambda s: s.replace("a_", "").replace("an_", "")
    best = None
    for stem, fn in stems.items():
        ns = norm(stem)
        if slug and (slug in ns or ns in slug):
            # prefer the longest such stem (most specific match)
            if best is None or len(ns) > best[0]:
                best = (len(ns), os.path.join(catdir, fn))
    return best[1] if best else None


def _load_construct(object_name, allow_network=False):
    """Compile a construct from the LOCAL Deckard catalog (offline, deterministic).

    Returns the Construct, or None. Network research (deckard.identify) is OFF by
    default — this keeps the batch deterministic and prevents fetch hangs.
    """
    from sigma_ground import deckard                       # lazy: tier-2, downward
    catdir = os.path.join(os.path.dirname(__file__), "..", "deckard", "catalog")
    slug = (object_name or "").strip().lower().replace(" ", "_")
    path = _find_catalog_md(catdir, slug)
    if path:
        with open(path, encoding="utf-8") as f:
            return deckard.compile(deckard.parse_markdown(f.read()))
    if allow_network:
        try:
            return deckard.identify(object_name)
        except Exception:
            return None
    return None


def render_construct_still(construct, out_path, *, px=240, n_surface=6000):
    """Render one framed portrait of a Deckard construct to ``out_path`` (PNG),
    via the SDF-surface node cloud. Returns metadata incl. node count + substances."""
    cloud, info = construct_to_node_cloud(construct, n_surface)
    (x0, x1), (y0, y1), (z0, z1) = construct.bbox
    cx, cy, cz = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)
    size = max(x1 - x0, y1 - y0, z1 - z0, 1e-3)

    cam = PushCamera(Vec3(cx + 0.85 * size, cy + 0.45 * size, cz + 2.4 * size),
                     Vec3(cx, cy, cz), px, px, fov=45)
    light = PushLight(Vec3(cx + 2.0 * size, cy + 3.0 * size, cz + 3.0 * size),
                      intensity=1.4)
    bg = Vec3(0.05, 0.05, 0.06)
    # splat density ≈ nodes per unit surface area (keeps splats ~1-2 px, no gaps)
    surf_area = max(2.0 * ((x1 - x0) * (y1 - y0) + (y1 - y0) * (z1 - z0)
                           + (x1 - x0) * (z1 - z0)), 1e-3)
    density = max(40.0, n_surface / surf_area)

    pixels = entangle([cloud], cam, light, density=density, bg_color=bg)
    rgb = bytearray()
    for row in pixels:
        for p in row:
            rgb.extend(p.to_rgb())
    write_png(out_path, px, px, bytes(rgb))
    return {"path": out_path, "n_nodes": len(cloud.nodes),
            "substances": info["substances"]}


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

    # Named-object family: a Deckard construct. Sample its SDF surface and render
    # the real shape (multi-material), via the CSG converter.
    obj_name = (handle or {}).get("object_name")
    if kind not in sphere_kinds and obj_name:
        construct = _load_construct(obj_name)
        if construct is None:
            return {"slug": slug, "material": obj_name, "renderable": False,
                    "stills": [], "peak_T_K": outputs.get("peak_T_K"),
                    "note": f"could not load a Deckard construct for {obj_name!r}"}
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{slug}_0_object.png")
        meta = render_construct_still(construct, path, px=px)
        meta.update({"file": os.path.basename(path), "label": "object",
                     "T_K": round(_AMBIENT_K, 1), "grounded": True, "glowing": False})
        return {"slug": slug, "material": obj_name, "renderable": True,
                "stills": [meta], "peak_T_K": outputs.get("peak_T_K"),
                "entangler_key": None,
                "note": f"Deckard construct · {meta['n_nodes']} SDF-surface nodes · "
                        f"substances: {', '.join(meta['substances'][:4])}"}

    if kind not in sphere_kinds:
        return {"slug": slug, "material": material_key, "renderable": False,
                "stills": [], "peak_T_K": outputs.get("peak_T_K"),
                "note": f"kind={kind!r} carries no object_name — cannot render"}

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
