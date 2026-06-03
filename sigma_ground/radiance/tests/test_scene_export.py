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
