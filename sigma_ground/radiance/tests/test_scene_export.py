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


def test_metal_flag_split_from_emergent():
    """`metal` drives chrome-vs-matte shading; it is NOT a synonym for emergent.
    A metal is both; the dielectric stub is neither."""
    from sigma_ground.radiance.scene_export import _bake_material
    cu = _bake_material("copper", 8960.0)
    assert cu["emergent"] is True and cu["metal"] is True
    glaze = _bake_material("glaze")
    assert glaze["emergent"] is False and glaze["metal"] is False


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
