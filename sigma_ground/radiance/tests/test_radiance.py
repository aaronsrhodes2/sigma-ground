"""Radiance smoke tests — fast, zero-dependency.

They pin the renderer's contract: a sphere casts a lit silhouette on a black
field, normals point outward, metal color is emergent (and material-specific),
and the PNG writer emits a valid file.
"""
import os
import sys

_CANON = r"D:\Aaron\development\sigma-ground"
if os.path.isdir(_CANON) and _CANON not in sys.path:
    sys.path.insert(0, _CANON)

from sigma_ground.shapes import Sphere
from sigma_ground.dynamics.vec import Vec3
from sigma_ground.radiance import (RadianceScene, Camera, orbit_eye,
                                   surface_normal, material_albedo, render,
                                   render_to_png)


def _sphere_scene():
    return RadianceScene.from_shape(Sphere(0.06), "copper")


def _cam(w=50, h=40):
    return Camera(orbit_eye(Vec3(0, 0, 0), 0.26, 35.0, 18.0), Vec3(0, 0, 0),
                  fov_deg=42.0, width=w, height=h)


def test_sphere_silhouette_on_black():
    """Center pixel hits the lit sphere; a corner pixel is black background."""
    scene, cam = _sphere_scene(), _cam()
    buf = render(scene, cam)
    w, h = cam.width, cam.height
    ci = (h // 2 * w + w // 2) * 3
    assert buf[ci] + buf[ci + 1] + buf[ci + 2] > 0      # center: lit matter
    assert buf[0] == 0 and buf[1] == 0 and buf[2] == 0  # top-left: empty space


def test_normal_points_outward():
    """At the +x pole of a sphere, the SDF gradient points +x."""
    s = Sphere(0.06)
    sdf = lambda p: s.surface_distance(p.x, p.y, p.z)
    n = surface_normal(sdf, Vec3(0.06, 0, 0))
    assert n.x > 0.99 and abs(n.y) < 0.05 and abs(n.z) < 0.05


def test_metal_color_is_emergent_and_specific():
    """Metals get a derived color; different metals differ; dielectric is the stub."""
    copper = material_albedo("copper")
    iron = material_albedo("iron")
    stub = material_albedo("water_ice")          # dielectric → v1 stub
    for c in (copper, iron):
        for ch in (c.x, c.y, c.z):
            assert 0.0 <= ch <= 1.0
    # copper read its real (reddish) response, not the neutral grey stub
    assert (abs(copper.x - stub.x) + abs(copper.y - stub.y) +
            abs(copper.z - stub.z)) > 0.01
    # and the two metals are not identical
    assert (abs(copper.x - iron.x) + abs(copper.y - iron.y) +
            abs(copper.z - iron.z)) > 0.01


def test_png_written(tmp_path):
    path = render_to_png(_sphere_scene(), _cam(40, 32), str(tmp_path / "s.png"))
    with open(path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"
