"""deckard.voxelize — real mesh → labeled voxel field, exact mass props.

A watertight icosphere is the tight check (convex → fills exactly): the voxelized
volume + mass + CoM match the analytic sphere, and the field builds a kernel Voxel
whose SDF reproduces the sphere. A smoke test voxelizes the real chair 44164 if the
242 GB PartNet data is present. numpy/trimesh/scipy are the opt-in [shapes] deps.
"""
import math

import pytest

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")

from sigma_ground.deckard.voxelize import voxelize, VoxelField, construct_from_field
from sigma_ground.deckard.sources import mesh as meshsrc
from sigma_ground.kernel.voxel import Voxel

_DENS = {"iron": 7874.0, "wood": 700.0}


def _density(name):
    return _DENS.get(name, 1000.0)


def test_icosphere_voxelizes_to_exact_mass():
    R = 0.10
    sph = trimesh.creation.icosphere(subdivisions=3, radius=R)
    field = voxelize([(sph, "iron")], pitch=0.006, density_of=_density)
    assert isinstance(field, VoxelField)
    true_vol = (4.0 / 3.0) * math.pi * R ** 3
    assert abs(field.volume_m3 - true_vol) / true_vol < 0.12          # discretization
    assert abs(field.mass_kg - 7874.0 * true_vol) / (7874.0 * true_vol) < 0.12
    # centre of mass at the origin (within a voxel)
    assert max(abs(c) for c in field.com_m) < 0.006
    # a solid sphere is watertight → high confidence
    assert field.watertight_frac > 0.9 and field.confidence > 0.9
    # the field exposes the iron free surface (sphere ↔ void)
    assert field.free_surfaces.get("iron", 0) > 0
    # builds a kernel Voxel whose SDF reads the sphere centre as deep inside
    v = field.to_voxel()
    assert isinstance(v, Voxel)
    assert v.surface_distance(*field.com_m) < -0.5 * R


def test_inertia_is_sane_for_sphere():
    R = 0.10
    sph = trimesh.creation.icosphere(subdivisions=3, radius=R)
    field = voxelize([(sph, "iron")], pitch=0.006, density_of=_density)
    Ixx, Iyy, Izz = field.inertia_kgm2
    # solid sphere: I = 2/5 m R²; symmetric → the three axes agree
    I_true = 0.4 * field.mass_kg * R ** 2
    for I in (Ixx, Iyy, Izz):
        assert abs(I - I_true) / I_true < 0.15
    assert abs(Ixx - Izz) / Ixx < 0.05                                # symmetry


def test_voxel_construct_is_a_drop_in():
    R = 0.10
    sph = trimesh.creation.icosphere(subdivisions=3, radius=R)
    field = voxelize([(sph, "iron")], pitch=0.006, density_of=_density)
    c = construct_from_field("iron ball", field, source="unit test")
    # exact mass props pass straight through (no bbox-mass bug)
    assert c.mass_kg == field.mass_kg
    assert c.com_m == field.com_m
    assert c.inertia_kgm2 == field.inertia_kgm2
    # the bbox encloses the sphere
    (x0, x1), (y0, y1), (z0, z1) = c.bbox
    assert x0 < -R and x1 > R and z0 < -R and z1 > R
    # the full Construct interface works (what physics queries)
    assert c.material_at(*c.com_m) == "iron"
    assert c.sdf(*c.com_m) < 0.0                       # CoM is inside
    assert c.density_at(*c.com_m) == pytest.approx(7874.0)
    assert c.material_at(0.5, 0.5, 0.5) is None        # far outside → void
    # one cited layer for iron; render() narrates it
    assert any(L.material == "iron" for L in c.layers)
    txt = c.render()
    assert "iron" in txt and "voxel" in c.validation["mode"]


@pytest.mark.skipif(not meshsrc.data_available("44164"),
                    reason="PartNet mesh data not on disk")
def test_real_chair_voxelizes_sane():
    parts = meshsrc.load_parts("44164", scale_to_m=0.9)
    assert parts, "chair 44164 parts should load"
    field = voxelize([(m, "wood") for m, _ in parts], pitch=0.01, density_of=_density)
    # per-part fill+union avoids the merged-shell over-solidification (was 0.16 m³);
    # a ~0.9 m chair lands in a physical range (inflated at 1cm, not 10× wrong)
    assert 0.005 < field.volume_m3 < 0.08
    assert field.mass_kg > 0
    assert isinstance(field.to_voxel(), Voxel)
    assert "wood" in field.free_surfaces
