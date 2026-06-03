"""Scene export — Deckard Construct → serializable SceneSpec (and back).

The browser viewer (and the validation oracle) need the construct as DATA, not
Python closures. We serialize the CSG as the flat, ordered `composed._leaves`
list — each `(CSGLeaf{shape, material}, op)` — because replaying that list left-
to-right reproduces both `sdf()` and `material_at()` *exactly* (it is how the
kernel evaluates them). Per-material color is BAKED here in Python via the
optics library, so the color physics stays server-side and the browser only
shades.

`scene_spec_to_sdf` is the faithful inverse used to validate the round-trip and
to feed Radiance-core (the ground-truth renderer).
"""
from __future__ import annotations

from ..dynamics.vec import Vec3

# Dielectric color is a v1 STUB until the molecular-color rung lands — labeled,
# never a faked paint chip. Metals are emergent (below).
_LABEL_STUB = {
    "water":   (0.62, 0.74, 0.86),     # pale blue
    "glaze":   (0.90, 0.90, 0.93),     # glossy off-white
    "ceramic": (0.86, 0.82, 0.76),     # warm off-white
    "air":     (0.0, 0.0, 0.0),        # void — never a surface hit
}


# ── primitive ⇄ dict ────────────────────────────────────────────────────
def _shape_to_dict(shape) -> dict:
    t = type(shape).__name__
    cx, cy, cz = shape.center
    d = {"type": t, "center": [cx, cy, cz]}
    if t in ("Sphere", "HollowSphere"):
        d["radius"] = shape.radius
    elif t in ("Cylinder", "Cone"):
        d["radius"], d["height"] = shape.radius, shape.height
    elif t == "Box":
        d["x"], d["y"], d["z"] = shape.x, shape.y, shape.z
    elif t == "Torus":
        d["major_radius"], d["minor_radius"] = shape.major_radius, shape.minor_radius
    else:
        raise ValueError(f"scene_export: primitive {t!r} not serializable yet")
    return d


def _shape_from_dict(d):
    from ..shapes import Sphere, Cylinder, Box, Cone, Torus
    t, c = d["type"], tuple(d.get("center", [0, 0, 0]))
    if t in ("Sphere", "HollowSphere"):
        return Sphere(d["radius"], center=c)
    if t == "Cylinder":
        return Cylinder(d["radius"], d["height"], center=c)
    if t == "Cone":
        return Cone(d["radius"], d["height"], center=c)
    if t == "Box":
        return Box(d["x"], d["y"], d["z"], center=c)
    if t == "Torus":
        return Torus(d["major_radius"], d["minor_radius"], center=c)
    raise ValueError(f"scene_export: cannot rebuild primitive {t!r}")


# ── material color (emergent for metals, flagged stub otherwise) ─────────
def _bake_material(label: str, density=None) -> dict:
    try:
        from ..field.interface.surface import MATERIALS
        if label in MATERIALS and MATERIALS[label].get("material_type") == "metal":
            from .shade import material_albedo
            c = material_albedo(label)
            return {"color_rgb": [c.x, c.y, c.z], "emergent": True,
                    "density_kg_m3": density}
    except Exception:
        pass
    rgb = _LABEL_STUB.get(label, (0.72, 0.72, 0.72))
    return {"color_rgb": list(rgb), "emergent": False,
            "note": "v1 stub — awaiting molecular color", "density_kg_m3": density}


def _suggest_camera(bbox) -> dict:
    (x0, x1), (y0, y1), (z0, z1) = bbox
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    r = 0.5 * ((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) ** 0.5
    return {"target": [cx, cy, cz], "orbit_radius": 2.6 * r, "fov_deg": 40.0,
            "up": [0.0, 0.0, 1.0]}        # Deckard builds along +z


# ── Light sources (the sandbox's illumination) ─────────────────────────
def _light_color(temp_k, fallback):
    """Emergent light color from a color temperature (blackbody), hue-normalized."""
    try:
        from ..field.interface.thermal import blackbody_color
        c = list(blackbody_color(temp_k))
        if max(c) > 1.5:                       # 0..255 → 0..1
            c = [v / 255.0 for v in c]
        m = max(c) or 1.0                      # keep hue, normalize brightness
        return [round(v / m, 4) for v in c]
    except Exception:
        return fallback


def _perp_frame(up):
    import math
    n = math.sqrt(sum(v * v for v in up)) or 1.0
    U = [v / n for v in up]
    H = [1, 0, 0] if abs(U[0]) < 0.9 else [0, 0, 1]
    cx = [U[1]*H[2]-U[2]*H[1], U[2]*H[0]-U[0]*H[2], U[0]*H[1]-U[1]*H[0]]
    n1 = math.sqrt(sum(v * v for v in cx)) or 1.0
    S1 = [v / n1 for v in cx]
    S2 = [U[1]*S1[2]-U[2]*S1[1], U[2]*S1[0]-U[0]*S1[2], U[0]*S1[1]-U[1]*S1[0]]
    return U, S1, S2


def _default_lighting(up):
    """A key + fill directional rig, oriented relative to the scene's up axis.

    `dir` is the direction the light TRAVELS. The key is a warm ~5500 K daylight
    from above-and-to-one-side; the fill is a cooler, dimmer counter-light.
    Ambient is hemispheric (sky from +up, a dim ground bounce from −up) so the
    shadowed side of matter still reads — while empty space stays pure black.
    """
    import math
    U, S1, S2 = _perp_frame(up or [0.0, 0.0, 1.0])

    def comb(a, b, c):
        v = [a*S1[i] + b*S2[i] + c*U[i] for i in range(3)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [round(x / n, 4) for x in v]

    return {
        "lights": [
            {"dir": comb(0.45, 0.40, -0.85), "color": _light_color(5500, [1.0, 0.96, 0.90]),
             "intensity": 1.05, "temperature_k": 5500},
            {"dir": comb(-0.55, 0.20, -0.25), "color": _light_color(7600, [0.85, 0.90, 1.0]),
             "intensity": 0.40, "temperature_k": 7600},
        ],
        "ambient": {"sky": [0.10, 0.12, 0.16], "ground": [0.06, 0.05, 0.045], "up": U},
    }


# ── Construct → SceneSpec ────────────────────────────────────────────────
def construct_to_scene(construct) -> dict:
    """Serialize a Deckard Construct into a JSON-able SceneSpec."""
    leaves = construct.composed._leaves
    csg_leaves, materials = [], {}
    for leaf, op in leaves:
        csg_leaves.append({"op": op, "material": leaf.material,
                           "shape": _shape_to_dict(leaf.shape)})
        if leaf.material not in materials:
            materials[leaf.material] = _bake_material(
                leaf.material, construct.density_by_label.get(leaf.material))
    cam = _suggest_camera(construct.bbox)
    lighting = _default_lighting(cam["up"])
    return {
        "name": construct.name,
        "csg_leaves": csg_leaves,                 # flat, ordered = faithful to evaluation
        "materials": materials,
        "physics": {"mass_kg": construct.mass_kg, "com_m": list(construct.com_m),
                    "inertia_kgm2": list(construct.inertia_kgm2)},
        "bbox": [list(b) for b in construct.bbox],
        "camera": cam,
        "lights": lighting["lights"],
        "ambient": lighting["ambient"],
        "identified": construct.identified,
        "source": construct.source,
    }


# ── SceneSpec → SDF (faithful inverse) ──────────────────────────────────
def scene_spec_to_sdf(spec):
    """Rebuild (sdf, material_at) callables from a SceneSpec — replays the tree."""
    from ..csg import CSGLeaf, CSGBranch
    leaves = [(CSGLeaf(_shape_from_dict(n["shape"]), n["material"]), n["op"])
              for n in spec["csg_leaves"]]
    root = leaves[0][0]
    for leaf, op in leaves[1:]:
        root = CSGBranch(root, leaf, op)

    def sdf(p):
        return root.sdf(p.x, p.y, p.z)

    def material_at(p):
        for leaf, _ in reversed(leaves):
            if leaf.shape.surface_distance(p.x, p.y, p.z) < 0.0:
                return leaf.material
        return None

    return sdf, material_at


def sdf_samples(construct, n: int = 4) -> list:
    """Ground-truth (point, signed-distance) samples for the browser self-check.

    The web viewer rebuilds the SDF in JS/GLSL from the csg_tree; comparing its
    values to these Python-computed samples proves the browser draws the same
    geometry the physics weighs (not-faked, checkable in-page)."""
    (x0, x1), (y0, y1), (z0, z1) = construct.bbox
    out = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = x0 + (i + 0.5) / n * (x1 - x0)
                y = y0 + (j + 0.5) / n * (y1 - y0)
                z = z0 + (k + 0.5) / n * (z1 - z0)
                out.append({"p": [x, y, z], "d": construct.composed.sdf(x, y, z)})
    return out


def scene_from_spec(spec, **kw):
    """A RadianceScene that renders a SceneSpec with its BAKED emergent colors."""
    from .scene import RadianceScene
    sdf, material_at = scene_spec_to_sdf(spec)
    colors = {k: Vec3(*v["color_rgb"]) for k, v in spec["materials"].items()}
    albedo = lambda label: colors.get(label, Vec3(0.72, 0.72, 0.72))
    return RadianceScene(sdf, material_at, albedo=albedo,
                         max_dist=kw.pop("max_dist", 5.0), **kw)
