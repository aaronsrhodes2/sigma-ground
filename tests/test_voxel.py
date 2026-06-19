"""kernel.Voxel — an SDF-grid shape that honours the Shape contract.

A Voxel trilinearly interpolates a sampled signed-distance grid, so the renderer
and integrator treat it like any analytic primitive. We build a known sphere SDF
grid and confirm the interpolation reproduces the analytic sphere distance (inside,
on the surface, and outside the grid box) and the volume is right. numpy is the
opt-in [shapes] dep; skip cleanly if absent.
"""
import math

import pytest

np = pytest.importorskip("numpy")

from sigma_ground.kernel.voxel import Voxel
from sigma_ground.kernel.shapes import Shape


def _sphere_voxel(R=0.15, N=41, vs=0.01):
    ax = (np.arange(N) - (N - 1) / 2.0) * vs        # cell-centre coords about origin
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    sdf = np.sqrt(X * X + Y * Y + Z * Z) - R         # signed distance to the sphere
    vol = float((sdf < 0).sum()) * vs ** 3
    return Voxel(sdf, vs, center=(0.0, 0.0, 0.0), volume_m3=vol, source="unit sphere"), R, vs


def test_voxel_is_a_shape():
    v, R, vs = _sphere_voxel()
    assert isinstance(v, Shape)
    assert v.is_volumetric and v.dimensionality == 3
    assert "Voxel" in repr(v)


def test_surface_distance_matches_analytic_sphere():
    v, R, vs = _sphere_voxel()
    # interior centre: deepest, sdf ≈ -R
    assert abs(v.surface_distance(0, 0, 0) - (-R)) < vs
    # on the surface: sdf ≈ 0
    assert abs(v.surface_distance(R, 0, 0)) < vs
    assert abs(v.surface_distance(0, R, 0)) < vs
    # inside the box, outside the sphere: true sdf = dist - R
    for p in (0.18, 0.05, 0.10):
        assert abs(v.surface_distance(p, 0, 0) - (p - R)) < vs
    # OUTSIDE the grid box (0.25 > half-extent 0.205): the box+extra formula must
    # still recover the true sphere distance 0.25 - R
    assert abs(v.surface_distance(0.25, 0, 0) - (0.25 - R)) < 1.5 * vs
    # a diagonal point well outside stays positive and monotone
    far = v.surface_distance(0.4, 0.4, 0.4)
    assert far > v.surface_distance(0.2, 0.2, 0.2) > 0


def test_voxel_volume_and_bounds():
    v, R, vs = _sphere_voxel()
    true_vol = (4.0 / 3.0) * math.pi * R ** 3
    assert abs(v.volume() - true_vol) / true_vol < 0.05      # discretization < 5%
    assert v.bounding_radius() > R                            # box encloses the sphere
    assert v.cross_section("z") < math.pi * R * R * 1.3       # ~projected area, fill-scaled


def test_voxel_does_not_import_numpy_at_module_scope():
    # kernel must stay import-clean: the module imports only stdlib + kernel.
    import sigma_ground.kernel.voxel as vox
    src = open(vox.__file__, encoding="utf-8").read()
    assert "import numpy" not in src.split("def _count_inside")[0]   # only lazy, inside the fallback


def test_label_grid_material_lookup():
    v, R, vs = _sphere_voxel()
    assert v.material_at(0, 0, 0) is None                     # unlabeled → None
    labels = np.zeros((5, 5, 5), dtype=int)
    labels[2, 2, 2] = 7
    sdf = np.full((5, 5, 5), -1.0)
    lv = Voxel(sdf, 0.01, center=(0, 0, 0), volume_m3=1.0, label_grid=labels)
    assert lv.material_at(0, 0, 0) == 7                       # centre cell label
