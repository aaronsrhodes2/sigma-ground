"""R-vox: a voxel Construct renders DIRECTLY through the Python path-tracer.

scene_from_construct raymarches the voxel `.sdf()` and colours each material by its
physics-derived emergent colour — so a real voxelized object becomes a picture with
no SceneSpec serialization. We render a small image of a voxel iron ball and confirm
it produces a real lit object (not an all-black frame, not a flooded one).
numpy/trimesh/scipy are the opt-in [shapes] deps.
"""
import math

import pytest

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")

from sigma_ground.deckard.voxelize import voxelize, construct_from_field
from sigma_ground.radiance.scene_export import scene_from_construct
from sigma_ground.radiance.camera import Camera
from sigma_ground.radiance.render import render
from sigma_ground.dynamics.vec import Vec3


def _iron_ball():
    sph = trimesh.creation.icosphere(subdivisions=3, radius=0.10)
    field = voxelize([(sph, "iron")], pitch=0.008, density_of=lambda n: 7874.0)
    return construct_from_field("iron ball", field)


def test_voxel_construct_renders_a_real_lit_object():
    c = _iron_ball()
    scene = scene_from_construct(c)
    com = Vec3(*c.com_m)
    (x0, x1), (y0, y1), (z0, z1) = c.bbox
    R = 0.5 * math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
    eye = com + Vec3(2.6 * R, 2.6 * R, 1.2 * R)
    cam = Camera(eye, com, up=Vec3(0, 0, 1), fov_deg=40, width=64, height=64)
    buf = render(scene, cam)

    total = 64 * 64
    lit = sum(1 for i in range(0, len(buf), 3) if buf[i] or buf[i + 1] or buf[i + 2])
    assert lit > 0.05 * total          # the ball fills a real chunk of the frame
    assert lit < 0.95 * total          # but it's an object, not a flooded frame
    # the lit pixels carry a real (non-black) colour
    assert max(buf) > 0
