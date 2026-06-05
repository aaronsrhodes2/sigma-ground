"""Stage-A tests — the Python→browser contract is faithful and renderable.

The keystone is the round-trip: an SDF rebuilt FROM the serialized SceneSpec must
equal the construct's own SDF to machine precision. If that holds, the browser
shader (built from the same JSON) draws exactly the matter the physics weighs.
"""
import os
import sys

_CANON = r"D:\Aaron\development\sigma-ground"
if os.path.isdir(_CANON) and _CANON not in sys.path:
    sys.path.insert(0, _CANON)

from sigma_ground import deckard
from sigma_ground.dynamics.vec import Vec3
from sigma_ground.radiance import (construct_to_scene, scene_spec_to_sdf,
                                   scene_from_spec, record_fall, Camera)
from sigma_ground.radiance.render import render


def test_scene_spec_roundtrip_sdf():
    """SDF rebuilt from the SceneSpec matches the construct's SDF everywhere."""
    c = deckard.identify("coffee cup")
    sdf, _ = scene_spec_to_sdf(construct_to_scene(c))
    (x0, x1), (y0, y1), (z0, z1) = c.bbox
    maxdiff = 0.0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                x = x0 + (i + 0.5) / 6 * (x1 - x0)
                y = y0 + (j + 0.5) / 6 * (y1 - y0)
                z = z0 + (k + 0.5) / 6 * (z1 - z0)
                maxdiff = max(maxdiff, abs(sdf(Vec3(x, y, z)) - c.composed.sdf(x, y, z)))
    assert maxdiff < 1e-9, maxdiff


def test_scene_spec_structure():
    c = deckard.identify("coffee cup")
    spec = construct_to_scene(c)
    assert spec["csg_leaves"]
    for n in spec["csg_leaves"]:
        assert {"op", "material", "shape"} <= set(n)
        assert "type" in n["shape"] and "center" in n["shape"]
    for m in spec["materials"].values():
        assert len(m["color_rgb"]) == 3
    assert abs(spec["physics"]["mass_kg"] - c.mass_kg) < 1e-12
    assert spec["camera"]["orbit_radius"] > 0


def test_spec_renders_nonblack():
    """A scene rebuilt from the SceneSpec actually renders the cup (lit pixels)."""
    spec = construct_to_scene(deckard.identify("coffee cup"))
    scene = scene_from_spec(spec)
    target = Vec3(*spec["camera"]["target"])
    R = spec["camera"]["orbit_radius"]
    cam = Camera(target + Vec3(R * 0.7, -R * 0.7, R * 0.45), target,
                 up=Vec3(0, 0, 1), fov_deg=spec["camera"]["fov_deg"],
                 width=48, height=48)
    buf = render(scene, cam)
    assert any(buf[i] for i in range(len(buf)))


def test_deckard_feather_cone_ellipsoid_roundtrip():
    """Deckard's feather (Cone rachis + Ellipsoid vane) serializes, colours by its
    true substance (keratin, not the anatomical part label), and round-trips the
    SDF exactly -- the new Ellipsoid/Cone primitives Radiance was taught."""
    c = deckard.identify("feather", allow_llm=False)
    spec = construct_to_scene(c)
    kinds = sorted(n["shape"]["type"] for n in spec["csg_leaves"])
    assert kinds == ["Cone", "Ellipsoid"], kinds           # the two new primitives
    ell = next(n for n in spec["csg_leaves"] if n["shape"]["type"] == "Ellipsoid")
    assert {"rx", "ry", "rz"} <= set(ell["shape"])          # semi-axes for sdEllip
    # coloured by SUBSTANCE (keratin -> measured cream), not the grey "no model" stub
    for m in spec["materials"].values():
        assert m["color_rgb"] != [0.72, 0.72, 0.72], "feather fell through to grey"
        assert "keratin" in m["mechanism"]
    # SDF rebuilt from the spec matches the compiled construct everywhere
    sdf, _ = scene_spec_to_sdf(spec)
    (x0, x1), (y0, y1), (z0, z1) = c.bbox
    maxdiff = 0.0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                x = x0 + (i + 0.5) / 6 * (x1 - x0)
                y = y0 + (j + 0.5) / 6 * (y1 - y0)
                z = z0 + (k + 0.5) / 6 * (z1 - z0)
                maxdiff = max(maxdiff, abs(sdf(Vec3(x, y, z)) - c.composed.sdf(x, y, z)))
    assert maxdiff < 1e-9, maxdiff


def test_record_fall_trajectory():
    out = record_fall("copper", 0.05, 2000.0, frame_dt=0.5)
    tr = out["trajectory"]
    frames = tr["frames"]
    assert len(frames) >= 3
    ts = [f["t_sim"] for f in frames]
    assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))     # time increases
    ys = [f["bodies"][0]["pos"][1] for f in frames]
    assert ys[0] > ys[-1] and ys[-1] <= 0.5                        # it falls to ground
    assert tr["suggested_rate"] > 0
    assert out["scene"]["csg_leaves"][0]["shape"]["type"] == "Sphere"


def test_density_emerges_from_crystal_cell():
    """A crystalline element's density is forced by its unit cell — compute it
    from structure + lattice + atomic mass and it matches reality to <1%."""
    from sigma_ground.inventory.density import crystal_density, density_of
    u = 1.66053906660e-27
    cu = crystal_density("fcc", 3.615e-10, 64 * u)        # copper fcc
    assert abs(cu - 8960) / 8960 < 0.01                   # within 1% of real Cu
    d = density_of("iron")
    assert d["emergent"] is True and "crystallographic" in d["method"]
    assert abs(d["error_vs_tabulated_pct"]) < 1.0         # derived ~ tabulated


def test_density_compound_falls_back_not_faked():
    """A compound/amorphous solid has no elemental unit cell to derive from —
    it must fall back to the tabulated value, honestly flagged NOT emergent
    (never the garbage the elemental formula would produce for it)."""
    from sigma_ground.inventory.density import density_of
    for label in ("glass", "ceramic_alumina", "concrete"):
        d = density_of(label)
        assert d["emergent"] is False
        assert d["density_kg_m3"] and d["density_kg_m3"] > 1000   # sane, = tabulated


def test_bake_material_carries_density_provenance():
    """The renderer attaches the emergent derivation as a cross-check against the
    supplied (e.g. Deckard) density — visible provenance, no silent override."""
    from sigma_ground.radiance.scene_export import _bake_material
    b = _bake_material("copper", 8960.0)
    assert b["density_kg_m3"] == 8960.0                    # supplied value preserved
    assert "density_derived_kg_m3" in b
    assert abs(b["density_pct_vs_supplied"]) < 1.0         # derivation agrees


def test_water_wave_dispersion_and_gradient():
    """Ripples obey the real gravity-capillary dispersion — phase speed has a
    minimum (~0.23 m/s) at the crossover — and the surface gradient is exact."""
    import math
    from sigma_ground.radiance.water_waves import (dispersion_omega, wind_wave_components,
                                                  height_at, gradient_at)
    c = lambda lam: dispersion_omega(2 * math.pi / lam) / (2 * math.pi / lam)
    assert c(2.0) > c(0.017) and c(0.005) > c(0.017)   # min near the 1.7 cm crossover
    assert 0.20 < c(0.017) < 0.25                       # textbook min phase speed
    comps = wind_wave_components(wind_speed=5.0, n=6)
    assert len(comps) == 6 and all(x["omega"] > 0 for x in comps)
    gx, _ = gradient_at(comps, 0.37, 0.21, 1.5)         # analytic == finite difference
    e = 1e-5
    fx = (height_at(comps, 0.37 + e, 0.21, 1.5) - height_at(comps, 0.37 - e, 0.21, 1.5)) / (2 * e)
    assert abs(gx - fx) < 1e-3


def test_simlayer_temperature_default_stp_is_cold():
    """Sim-layer contract: an object's temperature defaults to STP — below the
    Draper point, so it shows its reflectance with NO glow. Raise it and Planck
    emission appears (a red-dominant forge glow). The renderer derives the colour;
    the sim layer only holds the temperature."""
    from sigma_ground.field.interface.thermal_emission import (thermal_emission_rgb,
                                                              is_visibly_glowing)
    assert not is_visibly_glowing(293.15)             # STP default → invisible (all IR)
    assert thermal_emission_rgb("iron", 293.15) == (0.0, 0.0, 0.0)
    assert is_visibly_glowing(1200.0)                 # forge heat → glowing
    r, g, b = thermal_emission_rgb("iron", 1200.0)
    assert r > 0 and r >= g >= b                       # red-dominant forge glow


def test_incandescence_emissivity_is_kirchhoff():
    """A heated material glows by Planck × Kirchhoff: ε(λ)=1−R(λ) from the SAME
    measured optical constants that set its cold colour. Copper reflects red, so
    its emissivity is LOW in red, higher in blue — its glow is the complement of
    its cold colour. Grey iron has flat emissivity → it glows like a blackbody."""
    from sigma_ground.radiance.scene_export import _bake_material
    cu = _bake_material("copper", 8960.0)["emissivity_rgb"]
    assert cu[0] < cu[2]                              # ε_red < ε_blue (reflects red, emits less red)
    fe = _bake_material("iron", 7874.0)["emissivity_rgb"]
    assert max(fe) - min(fe) < 0.1                    # iron ~grey → flat ε → blackbody-like glow


def test_water_reflectivity_is_fresnel_grazing():
    """Water carries a Fresnel R0 from its refractive index (~0.02); the Schlick
    angle term makes it nearly clear looking down but a mirror at grazing."""
    from sigma_ground.radiance.scene_export import _bake_material
    w = _bake_material("water", None)
    r0 = w["reflect_r0"]
    assert 0.015 < r0 < 0.03                             # water R0 ~ 0.02
    schlick = lambda cosi: r0 + (1 - r0) * (1 - cosi) ** 5
    assert schlick(1.0) < 0.05                           # straight down → clear
    assert schlick(0.05) > 0.6                           # grazing → mirror


def test_mechanical_signature_is_emergent_and_ordered():
    """Each material carries a mechanical signature DERIVED from cohesive energy
    (modulus, sound speed, restitution, ring pitch) — and it orders like reality:
    steel is far stiffer than lead, and rings at a far higher pitch."""
    from sigma_ground.radiance.scene_export import _emergent_mechanics
    steel = _emergent_mechanics("steel_mild")
    lead = _emergent_mechanics("lead")
    assert steel["youngs_modulus_gpa"] > 5 * lead["youngs_modulus_gpa"]   # steel ≫ lead
    assert steel["ring_frequency_hz"] > 2 * lead["ring_frequency_hz"]     # steel rings, lead thuds
    assert steel["sound_speed_m_s"] > lead["sound_speed_m_s"]


def test_clatter_bounce_envelope_is_emergent():
    """The bounce envelope comes from the material's derived restitution: a rubber
    sphere climbs back near its drop height; copper dies on first contact."""
    from sigma_ground.radiance.trajectory import bounce_heights
    rub = bounce_heights("rubber", radius_m=0.07, drop_height_m=0.45, t_total=3.0)
    cop = bounce_heights("copper", radius_m=0.07, drop_height_m=0.45, t_total=3.0)
    start = 0.45 + 0.07
    # after the first second, rubber is still bouncing high; copper has settled low
    late_rub = max(rub[60:])           # frames at dt=0.02 -> 60 = t>1.2s
    late_cop = max(cop[60:])
    assert late_rub > 0.4 * start      # rubber still boinging
    assert late_cop < 0.15 * start     # copper thudded flat


def test_record_fall_emits_rigid_body_schema():
    """The viewer's rigid-body contract: a `bodies` list with a pivot, each
    moving leaf tagged with its body index, and a quaternion in every frame
    (identity here — a sphere — but the slot the rotation pipeline reads)."""
    out = record_fall("copper", 0.05, 1500.0, frame_dt=0.5)
    sc = out["scene"]
    assert sc["bodies"] and "pivot" in sc["bodies"][0]
    assert len(sc["bodies"][0]["pivot"]) == 3
    assert sc["csg_leaves"][0]["body"] == 0
    for fr in out["trajectory"]["frames"]:
        assert fr["bodies"], "every frame poses at least one body"
        for bd in fr["bodies"]:
            assert len(bd["pos"]) == 3 and len(bd["quat"]) == 4


def test_record_object_fall_renders_real_shape_falling():
    """record_object_fall drops Deckard's ACTUAL construct — a feather's
    cone+ellipsoid, not a sphere: the real shapes are body 0, a static floor is
    untagged, the fall descends to the ground, and the body is laid flat by a
    constant (non-fluttering) orientation that points its thinnest axis up."""
    from sigma_ground import deckard
    from sigma_ground.radiance.trajectory import record_object_fall

    def _rotate(q, v):                       # rotate vector v by quaternion (x,y,z,w)
        x, y, z, w = q
        qv = (x, y, z)
        cross = lambda a, b: (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                              a[0]*b[1]-a[1]*b[0])
        t = cross(qv, v)
        t2 = cross(qv, t)
        return tuple(v[i] + 2*w*t[i] + 2*t2[i] for i in range(3))

    feather = deckard.identify("feather", allow_llm=False)   # catalogue hit, offline
    out = record_object_fall(feather, start_altitude_m=2.4384, target_watch_s=5.0)
    sc, tr = out["scene"], out["trajectory"]
    assert out["kind"] == "trajectory"
    # the REAL compiled shapes (not a Sphere) are the moving body 0
    moving = [l for l in sc["csg_leaves"] if l.get("body") == 0]
    types = {l["shape"]["type"] for l in moving}
    assert types and "Sphere" not in types and types <= {"Cone", "Ellipsoid"}
    # a static floor leaf (untagged) so the feather floats down TO something
    assert any("body" not in l for l in sc["csg_leaves"])
    assert sc["bodies"] and len(sc["bodies"][0]["pivot"]) == 3
    # the fall: starts at the release height, descends monotonically to the floor
    ys = [fr["bodies"][0]["pos"][1] for fr in tr["frames"]]
    assert len(ys) >= 3
    assert abs(ys[0] - 2.4384) < 0.05
    assert ys[-1] <= 0.5                                  # reaches the ground
    assert all(a >= b - 1e-6 for a, b in zip(ys, ys[1:]))  # never rises
    assert tr["suggested_rate"] > 0
    # laid flat by a CONSTANT orientation (same quat every frame), and that quat
    # really points the thinnest axis up (+y) — flat-plate rest pose, no flutter
    quats = {tuple(fr["bodies"][0]["quat"]) for fr in tr["frames"]}
    assert len(quats) == 1
    (x0, x1), (y0, y1), (z0, z1) = feather.bbox
    thin = min((x1-x0, "x"), (y1-y0, "y"), (z1-z0, "z"))[1]
    thin_axis = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0),
                 "z": (0.0, 0.0, 1.0)}[thin]
    assert _rotate(next(iter(quats)), thin_axis)[1] > 0.99   # thin axis → +y


def test_metal_flag_split_from_emergent():
    """`metal` drives chrome-vs-matte shading; it is NOT a synonym for emergent.
    Copper is both; an emergent dielectric (ceramic) is emergent but matte; an
    unknown material with no model is neither."""
    from sigma_ground.radiance.scene_export import _bake_material, _emergent_color
    cu = _bake_material("copper", 8960.0)
    assert cu["emergent"] is True and cu["metal"] is True
    ceramic = _emergent_color("ceramic")                 # now emergent, but matte
    assert ceramic["emergent"] is True and ceramic["metal"] is False
    unknown = _emergent_color("unobtanium_xyz")          # no model -> neither
    assert unknown["emergent"] is False and unknown["metal"] is False


def test_dielectric_color_no_longer_a_stub():
    """The cup's glaze/ceramic/water now earn an emergent color from a physics
    model (white insulator, or Fresnel + Beer-Lambert) — not the old grey stub."""
    from sigma_ground.radiance.scene_export import _emergent_color
    for label in ("ceramic", "glaze", "water"):
        m = _emergent_color(label)
        assert m["emergent"] is True and m["metal"] is False
        assert "stub" not in m.get("mechanism", "").lower()
        assert all(0.0 <= v <= 1.0 for v in m["color_rgb"])


def test_chromophore_color_emerges_from_crystal_field():
    """Glaze color comes from the metal ion's crystal-field d-d absorption.
    Cobalt is blue; and the SAME ion (Cr3+) is red in an oxide host (ruby) but
    green in a silicate host (emerald) — the crystal field sets it, not the element."""
    from sigma_ground.radiance.scene_export import _emergent_color
    cobalt = _emergent_color("cobalt_glaze")["color_rgb"]
    assert cobalt[2] > cobalt[0]                            # blue: B > R
    red = _emergent_color("chrome_glaze")["color_rgb"]      # Cr3+ in oxide
    green = _emergent_color("emerald_glaze")["color_rgb"]   # Cr3+ in silicate
    assert red[0] > red[1] and green[1] > green[0]          # ruby reddish, emerald greenish
    assert red[0] > green[0] and green[1] > red[1]          # same ion, color swaps with host


def test_band_gap_color_is_emergent_yellow():
    """A semiconductor's color is FORCED by its band gap — nobody picks it.
    CdS (Eg≈2.4 eV) absorbs blue, so it must come out yellow (R,G > B). It is
    emergent yet dielectric (metal=False) — band-gap matter, not chrome."""
    from sigma_ground.radiance.scene_export import _bake_band_gap
    cds = _bake_band_gap("cadmium_sulfide")
    assert cds["emergent"] is True and cds["metal"] is False
    assert cds.get("band_gap_ev", 0) > 0
    r, g, b = cds["color_rgb"]
    assert all(0.0 <= v <= 1.0 for v in (r, g, b))
    assert r > b and g > b                              # blue eaten by the 2.4 eV gap → yellow
