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

# ── Emergent-color routing tables ──────────────────────────────────────
# The renderer never stores a chosen color. Each material's color is COMPUTED
# from its physical optical constants by the mechanism that fits what it IS —
# the same discipline metals already follow (Drude/Fresnel). These tables say
# only WHICH physics applies and with WHICH physical parameters; the color
# itself still falls out of the model.

# Transparent dielectrics → (Cauchy-index key, absorption key, path length m).
# Color emerges from Fresnel(n(lambda)) + Beer-Lambert(absorption x depth).
_DIELECTRIC = {
    "water": ("water", "water_blue", 0.04),      # faint blue over a few cm of depth
    "glaze": ("fused_silica", "clear", 0.0015),  # clear glassy coat -> near-colorless
    "glass": ("crown_glass", "clear", 0.003),
    "ice":   ("ice", "clear", 0.01),
}
# Colored ceramics/glazes → the transition-metal ION that colors them,
# as (Z, oxidation_state, coordination). Color emerges from crystal-field d-d
# absorption (Tanabe-Sugano). Nobody picks "blue"; Co2+ has no other choice.
_CHROMOPHORE = {
    "cobalt_glaze":    (27, 2, "oxide_tet"),     # Co2+ tetrahedral -> cobalt blue
    "copper_glaze":    (29, 2, "carbonate_oct"), # Cu2+ -> green
    "chrome_glaze":    (24, 3, "oxide_oct"),     # Cr3+ in OXIDE  -> ruby RED
    "emerald_glaze":   (24, 3, "silicate_oct"),  # Cr3+ in SILICATE -> emerald GREEN (same ion!)
    "nickel_glaze":    (28, 2, "carbonate_oct"), # Ni2+ -> green
    "manganese_glaze": (25, 3, "silicate_oct"),  # Mn3+ -> pink-violet
    "titanium_glaze":  (22, 3, "oxide_oct"),     # Ti3+ (d1); model yields amber (textbook Ti:sapphire is pink-purple)
}
# Wide-gap insulators with no visible chromophore: no absorber -> diffuse white.
_WHITE_INSULATOR = {"ceramic", "ceramic_alumina", "porcelain", "stoneware",
                    "bone", "gypsum", "salt"}

# Phase-qualified fill labels — the shape researcher emits "liquid water" /
# "liquid mercury", but every optics table here is keyed by the BARE substance.
# Strip a leading phase word so the colour routes to real physics instead of the
# grey stub. No optical key begins with a phase word, so this only ever rescues a
# label that would otherwise fall through (it never mangles a resolvable one).
_PHASE_WORDS = ("liquid ", "molten ", "solid ", "frozen ", "gaseous ")


def _canonical_substance(label: str) -> str:
    s = (label or "").strip()
    # Drop a trailing descriptive qualifier the researcher appends:
    # "glaze (glassy)" → "glaze". No optical key contains parentheses, so this
    # only ever rescues a label that would otherwise fall through.
    if s.endswith(")") and "(" in s:
        s = s[:s.rindex("(")].strip()
    low = s.lower()
    for w in _PHASE_WORDS:
        if low.startswith(w):
            return s[len(w):].strip()
    return s


# ── primitive ⇄ dict ────────────────────────────────────────────────────
# Every shape type the exporter can emit — MUST mirror the viewer's coverage
# (viewer.js jsShapeSDF/glslShapeCall); tools/verify_artifacts.py imports this
# as the single source of truth.
SUPPORTED_SHAPE_TYPES = {"Sphere", "HollowSphere", "Cylinder", "Cone", "Box",
                         "Ellipsoid", "Torus", "Water", "Outline", "Voxel",
                         "Rotated", "Clipped", "Subtracted"}

# Voxel SDF grids are clamped to a narrow band of this many cells and quantized
# to signed int8 (R8_SNORM in the viewer): band/127 ≈ 0.05·voxel sub-cell
# resolution at the surface, big enough steps for cheap sphere-tracing far away.
_VOXEL_BAND_CELLS = 6.0


def _mat_to_quat(R):
    """Row-major 3x3 rotation -> unit quaternion (x, y, z, w), w >= 0.

    Shepperd's method (largest-pivot branch, numerically safe). Converts the
    MATRIX the SDF actually used — never re-derived from euler angles — so the
    browser rotates exactly the geometry the physics weighed.
    """
    import math
    t = R[0][0] + R[1][1] + R[2][2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = [(R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s,
             (R[1][0] - R[0][1]) / s, 0.25 * s]
    elif R[0][0] >= R[1][1] and R[0][0] >= R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0
        q = [0.25 * s, (R[0][1] + R[1][0]) / s,
             (R[0][2] + R[2][0]) / s, (R[2][1] - R[1][2]) / s]
    elif R[1][1] >= R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0
        q = [(R[0][1] + R[1][0]) / s, 0.25 * s,
             (R[1][2] + R[2][1]) / s, (R[0][2] - R[2][0]) / s]
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0
        q = [(R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s,
             0.25 * s, (R[1][0] - R[0][1]) / s]
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    q = [v / n for v in q]
    return [-v for v in q] if q[3] < 0.0 else q


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
    elif t == "Ellipsoid":
        d["rx"], d["ry"], d["rz"] = shape.rx, shape.ry, shape.rz
    elif t == "Torus":
        d["major_radius"], d["minor_radius"] = shape.major_radius, shape.minor_radius
    elif t == "Voxel":
        # A real voxelized solid: ship the signed-distance grid itself, narrow-band
        # int8-quantized + base64, so the WebGL2 viewer can raymarch it as a 3-D
        # texture ("just another SDF"). Heavy numpy import is lazy — only paid when
        # a Voxel is actually serialized.
        import base64
        import numpy as np
        g = np.asarray(shape._g, dtype=np.float32)
        vs = float(shape.voxel_size)
        band = _VOXEL_BAND_CELLS * vs
        q = np.clip(np.round(g * (127.0 / band)), -127, 127).astype(np.int8)
        # texImage3D wants x (width) fastest: payload[(k·ny+j)·nx+i] = grid[i,j,k]
        payload = np.ascontiguousarray(q.transpose(2, 1, 0)).tobytes()
        nx, ny, nz = shape._nx, shape._ny, shape._nz
        d["type"] = "Voxel"
        d["voxel_size"] = vs
        d["dims"] = [nx, ny, nz]                    # box spans center ± dims·vs/2
        d["band"] = band                            # decode: snorm[-1,1] · band = metres
        d["encoding"] = "int8-snorm-xfast"
        d["sdf_b64"] = base64.b64encode(payload).decode("ascii")
        if getattr(shape, "source", ""):
            d["source"] = shape.source
    elif t == "Outline":
        d["type"] = "Outline"
        d["mode"] = shape.mode
        d["profile"] = [[u, v] for u, v in shape.profile]
        if shape.mode == "extrude":
            d["thickness"] = shape.thickness
        if len(shape.profile) > 64:
            raise ValueError("scene_export: outline profile too long for the "
                             f"viewer shader ({len(shape.profile)} > 64 points)")
    elif t == "_Rotated":                          # deckard rotation wrapper
        d["type"] = "Rotated"
        d["rot"] = _mat_to_quat(shape._R)
        d["shape"] = _shape_to_dict(shape._base)
    elif t == "_Clipped":                          # deckard fill half-space
        d["type"] = "Clipped"
        d["axis"] = shape._axis
        d["sign"] = shape._sign
        d["level"] = shape._level
        d["shape"] = _shape_to_dict(shape._base)
    elif t == "_Subtracted":                       # deckard conforming solid
        d["type"] = "Subtracted"
        d["shape"] = _shape_to_dict(shape._base)
        d["cut"] = _shape_to_dict(shape._cut)
    else:
        raise ValueError(f"scene_export: primitive {t!r} not serializable yet")
    return d


def _qrot_inv(q, v):
    """Inverse-rotate v by quaternion q=(x,y,z,w) — same form as the viewer's
    qrotInv: v + 2w(u x v) + 2 u x (u x v) with u = -q.xyz."""
    ux, uy, uz = -q[0], -q[1], -q[2]
    tx = 2.0 * (uy * v[2] - uz * v[1])
    ty = 2.0 * (uz * v[0] - ux * v[2])
    tz = 2.0 * (ux * v[1] - uy * v[0])
    return (v[0] + q[3] * tx + uy * tz - uz * ty,
            v[1] + q[3] * ty + uz * tx - ux * tz,
            v[2] + q[3] * tz + ux * ty - uy * tx)


class _QuatRotatedShape:
    """Inverse of the exported "Rotated" node — applies the QUATERNION (the same
    math path the browser runs), not a rebuilt matrix, so the round-trip
    validation exercises the viewer's formula."""
    def __init__(self, inner, quat, center):
        self._inner, self._q = inner, quat
        self.center = tuple(center)

    def surface_distance(self, px, py, pz):
        c = self.center
        lx, ly, lz = _qrot_inv(self._q, (px - c[0], py - c[1], pz - c[2]))
        return self._inner.surface_distance(c[0] + lx, c[1] + ly, c[2] + lz)

    def volume(self):
        return self._inner.volume()

    def bounding_radius(self):
        return self._inner.bounding_radius()


class _ClippedShape:
    def __init__(self, inner, axis, sign, level):
        self._inner, self._axis, self._sign, self._level = inner, axis, sign, level
        self.center = inner.center

    def surface_distance(self, px, py, pz):
        d = self._inner.surface_distance(px, py, pz)
        half = self._sign * ((px, py, pz)[self._axis] - self._level)
        return half if half > d else d

    def volume(self):
        return 0.0                                  # validation never integrates this

    def bounding_radius(self):
        return self._inner.bounding_radius()


class _SubtractedShape:
    def __init__(self, base, cut):
        self._base, self._cut = base, cut
        self.center = base.center

    def surface_distance(self, px, py, pz):
        d = self._base.surface_distance(px, py, pz)
        dc = -self._cut.surface_distance(px, py, pz)
        return dc if dc > d else d

    def volume(self):
        return 0.0

    def bounding_radius(self):
        return self._base.bounding_radius()


def _shape_from_dict(d):
    from ..shapes import Sphere, Cylinder, Box, Cone, Torus, Ellipsoid
    t, c = d["type"], tuple(d.get("center", [0, 0, 0]))
    if t in ("Sphere", "HollowSphere"):
        return Sphere(d["radius"], center=c)
    if t == "Cylinder":
        return Cylinder(d["radius"], d["height"], center=c)
    if t == "Cone":
        return Cone(d["radius"], d["height"], center=c)
    if t == "Box":
        return Box(d["x"], d["y"], d["z"], center=c)
    if t == "Ellipsoid":
        return Ellipsoid(d["rx"], d["ry"], d["rz"], center=c)
    if t == "Torus":
        return Torus(d["major_radius"], d["minor_radius"], center=c)
    if t == "Voxel":
        import base64
        import numpy as np
        from ..kernel.voxel import Voxel
        nx, ny, nz = d["dims"]
        band = float(d["band"])
        raw = np.frombuffer(base64.b64decode(d["sdf_b64"]), dtype=np.int8)
        # inverse of the x-fastest pack: (nz,ny,nx) → (nx,ny,nz)
        g = (raw.reshape(nz, ny, nx).transpose(2, 1, 0).astype(np.float32)
             * (band / 127.0))
        g = np.ascontiguousarray(g)
        return Voxel(g, float(d["voxel_size"]), center=c, source=d.get("source", ""))
    if t == "Outline":
        from ..kernel.outline import Outline
        prof = [(u, v) for u, v in d["profile"]]
        if d.get("mode") == "revolve":
            o = Outline(prof, mode="revolve")
        else:
            o = Outline(prof, mode="extrude", thickness=d.get("thickness", 0.001))
        o.center = c
        return o
    if t == "Rotated":
        return _QuatRotatedShape(_shape_from_dict(d["shape"]), d["rot"], c)
    if t == "Clipped":
        return _ClippedShape(_shape_from_dict(d["shape"]), d["axis"],
                             d["sign"], d["level"])
    if t == "Subtracted":
        return _SubtractedShape(_shape_from_dict(d["shape"]),
                                _shape_from_dict(d["cut"]))
    raise ValueError(f"scene_export: cannot rebuild primitive {t!r}")


# ── material color (emergent for metals, flagged stub otherwise) ─────────
def _emergent_color(label: str) -> dict:
    """Pick the color mechanism from what the material IS; COMPUTE it, never store.

    Returns {color_rgb, emergent, metal, mechanism, ...}. `emergent` is True when
    a physics model converts the substance's optical constants into color;
    False only for the honest exceptions (measured-reflectance organics, or an
    unknown material with no model yet).
    """
    label = _canonical_substance(label)                  # "liquid water" → "water"
    if label == "air":                                   # void — never a surface hit
        return {"color_rgb": [0.0, 0.0, 0.0], "emergent": True, "metal": False,
                "mechanism": "void (no matter)"}
    # 1) Metals — free-electron Drude + Fresnel. Solid metals are tagged in
    #    MATERIALS; liquid metals (mercury) are mirrors too — a measured n+k in
    #    MEASURED_NK is the same Fresnel-mirror physics, even when MATERIALS does
    #    not carry them as a structural "metal".
    try:
        from ..field.interface.surface import MATERIALS
        if MATERIALS.get(label, {}).get("material_type") == "metal":
            from .shade import material_albedo
            c = material_albedo(label)
            return {"color_rgb": [round(c.x, 4), round(c.y, 4), round(c.z, 4)],
                    "emergent": True, "metal": True,
                    "mechanism": "Drude-Fresnel (free electrons)"}
        from ..field.interface.optics import MEASURED_NK, metal_rgb
        if label in MEASURED_NK:                         # liquid-metal mirror (Hg)
            r, g, b = metal_rgb(label)
            return {"color_rgb": [round(r, 4), round(g, 4), round(b, 4)],
                    "emergent": True, "metal": True,
                    "mechanism": "measured n+k Fresnel mirror (liquid metal)"}
    except Exception:
        pass
    # 2) Semiconductors — band-gap absorption.
    try:
        from ..field.interface.semiconductor_optics import (
            VARSHNI_PARAMS, semiconductor_rgb, band_gap_ev)
        if label in VARSHNI_PARAMS:
            c = list(semiconductor_rgb(label))
            return {"color_rgb": [round(v, 4) for v in c], "emergent": True,
                    "metal": False, "mechanism": "band-gap absorption",
                    "band_gap_ev": round(band_gap_ev(label), 2)}
    except Exception:
        pass
    # 3) Colored ceramic/glaze — transition-metal ion, crystal-field d-d.
    try:
        from ..field.interface.crystal_field import crystal_field_rgb
        spec = _CHROMOPHORE.get(label)
        if spec:
            Z, ox, coord = spec
            c = list(crystal_field_rgb(Z, ox, coord))
            return {"color_rgb": [round(v, 4) for v in c], "emergent": True,
                    "metal": False, "ion": [Z, ox, coord],
                    "mechanism": f"crystal-field d-d (Z{Z} ox+{ox}, {coord})"}
    except Exception:
        pass
    # 4) Transparent dielectric — Fresnel(n) + Beer-Lambert.
    try:
        from ..field.interface.optics import dielectric_color_rgb
        rec = _DIELECTRIC.get(label)
        if rec:
            c = list(dielectric_color_rgb(*rec))
            return {"color_rgb": [round(v, 4) for v in c], "emergent": True,
                    "metal": False, "mechanism": "dielectric Fresnel + Beer-Lambert"}
    except Exception:
        pass
    # 5) Wide-gap insulator, no visible chromophore -> diffuse white (no absorber).
    if label in _WHITE_INSULATOR:
        return {"color_rgb": [0.9, 0.9, 0.9], "emergent": True, "metal": False,
                "mechanism": "wide-gap insulator: no visible absorber -> white"}
    if label == "concrete":
        return {"color_rgb": [0.55, 0.54, 0.52], "emergent": True, "metal": False,
                "mechanism": "weakly-absorbing aggregate -> grey"}
    # 6) Organic — MEASURED reflectance (no derivation; honestly NOT emergent).
    #    Keratin family (feather/hair/horn/nail) is the SAME protein as wool, so we
    #    proxy to wool's measured spectrum rather than fall through to grey — flagged
    #    as a proxy, not a fabricated colour.
    try:
        from ..field.interface.optics import ORGANIC_SPECTRA, organic_rgb
        _KERATIN = {"keratin", "feather", "hair", "horn", "nail", "wool"}
        olabel = ("wool_natural" if label in _KERATIN and "wool_natural" in ORGANIC_SPECTRA
                  else label)
        if olabel in ORGANIC_SPECTRA:
            c = list(organic_rgb(olabel))
            proxied = olabel != label
            return {"color_rgb": [round(v, 4) for v in c], "emergent": False, "metal": False,
                    "mechanism": ("measured keratin-family reflectance (wool proxy)" if proxied
                                  else "measured organic reflectance (not derived)")}
    except Exception:
        pass
    # 7) Unknown — neutral default, flagged honestly.
    return {"color_rgb": [0.72, 0.72, 0.72], "emergent": False, "metal": False,
            "mechanism": "no model yet", "note": "awaiting composition"}


def _emergent_mechanics(label: str) -> dict:
    """A material's MECHANICAL signature, derived (not stored) from its cohesive
    energy by the existing field.interface physics:
        cohesive energy -> bulk/Young's modulus (mechanical.py)
                        -> speed of sound  (acoustics.py: sqrt((K+4G/3)/rho))
                        -> restitution + ring frequency (impact.py: Hertz/Johnson)
    The ring frequency IS the pitch of the clatter — steel rings ~15 kHz, lead
    thuds ~5 kHz, rubber boings ~30 Hz. Reference impact: 1 m/s onto a 2 cm body
    (restitution & ring are velocity-dependent; this is a representative value).
    Returns only the fields that compute — non-structural materials are skipped.
    """
    V_REF, R_REF, out = 1.0, 0.02, {}
    try:
        from ..field.interface.mechanical import youngs_modulus
        out["youngs_modulus_gpa"] = round(youngs_modulus(label) / 1e9, 1)
    except Exception:
        pass
    try:
        from ..field.interface.acoustics import longitudinal_wave_speed
        out["sound_speed_m_s"] = round(longitudinal_wave_speed(label), 0)
    except Exception:
        pass
    try:
        from ..field.interface.impact import (coefficient_of_restitution,
                                              impact_sound_frequency)
        out["restitution_ref"] = round(coefficient_of_restitution(
            label, velocity=V_REF, radius_m=R_REF), 3)
        out["ring_frequency_hz"] = round(impact_sound_frequency(
            label, velocity=V_REF, radius_m=R_REF), 0)
    except Exception:
        pass
    if out:
        out["mechanics_origin"] = "derived from cohesive energy (mechanical/acoustics/impact)"
    return out


def _bake_material(label: str, density=None) -> dict:
    label = _canonical_substance(label)                  # canonical optical key
    info = _emergent_color(label)
    # Density is emergent too: derive it from the crystal cell (see
    # inventory.density). If the caller supplied one (e.g. Deckard), keep it as
    # authoritative but attach the derivation as a cross-check; if not, use the
    # derived value directly.
    try:
        from ..inventory.density import density_of
        d = density_of(label)
    except Exception:
        d = {}
    derived = d.get("density_kg_m3")
    if density is None:
        info["density_kg_m3"] = derived
    else:
        info["density_kg_m3"] = density
        if derived:
            info["density_derived_kg_m3"] = derived
            info["density_pct_vs_supplied"] = round(100.0 * (derived - density) / density, 2)
    if d.get("method"):
        info["density_method"] = d["method"]
    info.update(_emergent_mechanics(label))           # E, sound speed, restitution, ring pitch
    # Reflective clear dielectrics (liquid water, glass, ice) carry a Fresnel R0
    # from their refractive index, so the renderer casts a real reflection ray —
    # near-transparent looking straight down, mirror-bright at grazing angles.
    # Clear glaze is a thin fused-silica-like coat — give it the same Cauchy index
    # so it's a glossy Fresnel dielectric (refracts to the body beneath), not matte.
    _CAUCHY = {"water": "water", "glass": "crown_glass", "ice": "ice",
               "glaze": "fused_silica"}
    if label in _CAUCHY:
        try:
            from ..field.interface.optics import cauchy_n, dielectric_surface_reflectance
            n = cauchy_n(_CAUCHY[label], 550e-9)
            info["reflect_r0"] = round(dielectric_surface_reflectance(n), 4)
            info["refractive_index"] = round(n, 4)
        except Exception:
            pass
    # Liquid-metal mirrors (mercury) reflect but aren't tagged structural metals;
    # give them the Fresnel R0 from their measured n+k so the renderer casts a
    # reflection ray at the DERIVED strength (~0.76 for Hg). Solid metals — which
    # already render via their albedo — are deliberately left untouched.
    if info.get("metal") and "reflect_r0" not in info:
        try:
            from ..field.interface.surface import MATERIALS
            from ..field.interface.optics import MEASURED_NK, metal_reflectance
            if label in MEASURED_NK and MATERIALS.get(label, {}).get("material_type") != "metal":
                info["reflect_r0"] = round(metal_reflectance(label, 550e-9), 4)
        except Exception:
            pass
    # Kirchhoff emissivity ε(λ)=1−R(λ) from the material's MEASURED n+k — the SAME
    # optical data that sets the cold colour also sets the hot glow. The renderer
    # multiplies this by Planck's law to make heated matter incandesce.
    try:
        from ..field.interface.thermal_emission import emissivity, THERMAL_EMISSION_MATERIALS
        if label in THERMAL_EMISSION_MATERIALS:
            info["emissivity_rgb"] = [round(emissivity(label, w), 3)
                                      for w in (650e-9, 550e-9, 450e-9)]
    except Exception:
        pass
    return info


def _bake_band_gap(semi_key: str) -> dict:
    """Color of a semiconductor/dielectric from its BAND GAP — emergent.

    semiconductor_rgb runs the 3-regime band-gap model (Beer-Lambert + Fresnel):
    a wide gap reflects all visible (neutral/pale), a mid gap eats the blue
    (yellow→orange as it shrinks), a narrow gap absorbs it all (dark). The hue
    is forced by the gap; nobody picks it. (Brightness is the true — dim —
    surface reflectance; the shader's sRGB gamma makes the hue legible.)
    """
    from ..field.interface.semiconductor_optics import semiconductor_rgb, band_gap_ev
    c = list(semiconductor_rgb(semi_key))
    if max(c) > 1.5:
        c = [v / 255.0 for v in c]
    return {"color_rgb": [round(v, 4) for v in c], "emergent": True, "metal": False,
            "band_gap_ev": round(band_gap_ev(semi_key), 2),
            "note": "emergent hue from band gap (surface reflectance)"}


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
    # A leaf's label may be a PART name (rachis/vane) or a material (ceramic);
    # the substance to derive COLOUR from lives on the layer. Map label→material
    # so an anatomically-named part (both feather parts are keratin) still colours
    # by what it IS, not falls through to grey.
    label_material = {L.name: L.material for L in getattr(construct, "layers", [])}
    csg_leaves, materials = [], {}
    for leaf, op in leaves:
        label = leaf.material                       # the leaf's identity label
        csg_leaves.append({"op": op, "material": label,
                           "shape": _shape_to_dict(leaf.shape)})
        if label not in materials:
            substance = label_material.get(label, label)   # what it's MADE of
            materials[label] = _bake_material(
                substance, construct.density_by_label.get(label))
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


def scene_from_construct(construct, **kw):
    """A RadianceScene that renders a Construct DIRECTLY from its SDF + per-cell
    material — no SceneSpec serialization needed. Works for ANY backend: a voxel
    Construct (real chair) or a primitive one. Each material is coloured by its
    physics-derived emergent colour (the same `_emergent_color` the web bake uses),
    so the render stays render-from-physics. This is R-vox's server-side path.
    """
    from .scene import RadianceScene
    sdf = lambda p: construct.sdf(p.x, p.y, p.z)
    mat = lambda p: construct.material_at(p.x, p.y, p.z)
    _cache = {}

    def albedo(label):
        if label not in _cache:
            if not label:
                _cache[label] = Vec3(0.0, 0.0, 0.0)
            else:
                c = _emergent_color(label).get("color_rgb", [0.72, 0.72, 0.72])
                _cache[label] = Vec3(*c)
        return _cache[label]

    (x0, x1), (y0, y1), (z0, z1) = construct.bbox
    span = max(x1 - x0, y1 - y0, z1 - z0, 1e-3)
    return RadianceScene(sdf, mat, albedo=albedo,
                         max_dist=kw.pop("max_dist", span * 10.0), **kw)


def bake_construct_png(construct, path, *, width=200, height=200,
                       azimuth_deg=35.0, elevation_deg=20.0, distance=2.6,
                       fov_deg=38.0):
    """Render a Construct to a PNG from a framing camera (z-up world). Returns the
    path. The server-side proof-of-life for a voxelized real object."""
    import math
    from .camera import Camera
    from .render import render_to_png
    scene = scene_from_construct(construct)
    cx, cy, cz = construct.com_m
    com = Vec3(cx, cy, cz)
    (x0, x1), (y0, y1), (z0, z1) = construct.bbox
    R = 0.5 * math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
    a = math.radians(azimuth_deg)
    e = math.radians(elevation_deg)
    eye = com + Vec3(distance * R * math.cos(e) * math.sin(a),
                     distance * R * math.cos(e) * math.cos(a),
                     distance * R * math.sin(e))
    cam = Camera(eye, com, up=Vec3(0, 0, 1), fov_deg=fov_deg,
                 width=width, height=height)
    return render_to_png(scene, cam, path)
