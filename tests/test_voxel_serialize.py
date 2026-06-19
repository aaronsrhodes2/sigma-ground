"""RR1: a Voxel survives the SceneSpec round-trip the web viewer reads.

`_shape_to_dict` ships the signed-distance grid narrow-band int8-quantized +
base64 (the bytes the WebGL2 viewer uploads as a 3-D texture); `_shape_from_dict`
rebuilds a kernel Voxel from exactly those bytes. The rebuilt grid must reproduce
the surface within the int8 band quantum — proving the encode/decode and the
`uvw=(p-center)/extent+0.5` indexing the viewer's GLSL will use. A voxel Construct
also serializes through `construct_to_scene` as one real Voxel leaf (not a box
pile). numpy/trimesh/scipy are the opt-in [shapes] deps.
"""
import math

import pytest

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")

from sigma_ground.deckard.voxelize import voxelize, construct_from_field
from sigma_ground.radiance.scene_export import (
    _shape_to_dict, _shape_from_dict, construct_to_scene, SUPPORTED_SHAPE_TYPES,
)


def _ball_field(pitch=0.01):
    sph = trimesh.creation.icosphere(subdivisions=3, radius=0.10)
    return voxelize([(sph, "iron")], pitch=pitch, density_of=lambda n: 7874.0)


def test_voxel_is_a_supported_shape_type():
    assert "Voxel" in SUPPORTED_SHAPE_TYPES


def test_voxel_round_trips_within_the_band_quantum():
    field = _ball_field()
    vox = field.to_voxel()
    d = _shape_to_dict(vox)
    assert d["type"] == "Voxel"
    assert d["dims"] == [vox._nx, vox._ny, vox._nz]
    assert d["encoding"] == "int8-snorm-xfast"
    assert isinstance(d["sdf_b64"], str) and len(d["sdf_b64"]) > 0

    rebuilt = _shape_from_dict(d)
    # the quantization step: one int8 level of the band (the worst-case error)
    band = d["band"]
    quantum = band / 127.0
    # sample a line through the centre; near-surface distances must match closely
    cx, cy, cz = vox.center
    for t in np.linspace(-0.13, 0.13, 27):
        a = vox.surface_distance(cx + t, cy, cz)
        b = rebuilt.surface_distance(cx + t, cy, cz)
        # both clamp to ±band; only compare where the original is inside the band
        if abs(a) < band * 0.9:
            assert abs(a - b) <= 2.5 * quantum


def test_rebuilt_voxel_keeps_the_sphere_sign():
    field = _ball_field()
    vox = field.to_voxel()
    rebuilt = _shape_from_dict(_shape_to_dict(vox))
    # centre is deep inside (negative), a far corner is outside (positive)
    cx, cy, cz = vox.center
    assert rebuilt.surface_distance(cx, cy, cz) < 0.0
    assert rebuilt.surface_distance(cx + 0.3, cy, cz) > 0.0


def test_voxel_construct_serializes_as_one_voxel_leaf():
    field = _ball_field()
    c = construct_from_field("iron ball", field)
    spec = construct_to_scene(c)
    leaves = spec["csg_leaves"]
    # ONE real voxel leaf — not a pile of primitive boxes
    assert len(leaves) == 1
    assert leaves[0]["shape"]["type"] == "Voxel"
    assert leaves[0]["material"] == "iron"
    # the leaf's material is registered with a baked colour for the viewer
    assert "iron" in spec["materials"]
    assert "color_rgb" in spec["materials"]["iron"]
