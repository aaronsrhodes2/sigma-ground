"""Tests for deckard.sdf_voxelize -- the direct-SDF voxel bridge (Phase D2/D3)."""
import math
import random

import pytest

from sigma_ground.deckard.construct import compile as dcompile, Construct
from sigma_ground.deckard.schema import ConstructSpec, Part, Fact
from sigma_ground.deckard.sdf_voxelize import (
    adaptive_pitch, voxelize_construct_sdf, construct_to_voxel_fidelity,
    _grid_extent, _chunk_points,
)
from sigma_ground.kernel.csg import ComposedSDF

ESTIMATED = "estimated"


def _multi_leaf_spec():
    """A small multi-primitive composite: one big body + several small,
    off-center, differently-sized parts -- enough leaves for adaptive_pitch's
    min-vs-whole logic to actually be exercised."""
    body = Part("body", "sphere", {"radius_m": Fact(0.05, ESTIMATED, "", 0.5)},
                "steel", Fact(7850.0, ESTIMATED, "", 0.5))
    stud = Part("stud", "sphere", {"radius_m": Fact(0.003, ESTIMATED, "", 0.5)},
                "steel", Fact(7850.0, ESTIMATED, "", 0.5),
                center_m=(0.06, 0.0, 0.0))
    return ConstructSpec(name="body-with-stud", kind="composite", identified=True,
                         parts=[body, stud])


def test_adaptive_pitch_rejects_mesh_sourced_composed():
    class _FakeMeshComposed:
        _leaves = []
    c = dcompile(_multi_leaf_spec(), resolution=24)
    fake = Construct(**{**c.__dict__, "composed": _FakeMeshComposed()})
    with pytest.raises(TypeError):
        adaptive_pitch(fake)


def test_adaptive_pitch_feature_vs_whole():
    c = dcompile(_multi_leaf_spec(), resolution=24)
    pitch, report = adaptive_pitch(c)
    assert report["pitch_feature_m"] < report["pitch_whole_m"]   # stud is binding
    assert pitch == pytest.approx(report["pitch_target_m"])
    assert not report["budget_clamped"]


def test_adaptive_pitch_uses_leaf_shape_radius_not_csgleaf_radius():
    # a leaf far from the origin must NOT report a huge "feature size" --
    # this is exactly the bug the design's warning guards against.
    c = dcompile(_multi_leaf_spec(), resolution=24)
    pitch, report = adaptive_pitch(c)
    # the stud has radius 3mm; if CSGLeaf.bounding_radius() (which folds in
    # the 60mm offset) were used instead, min_leaf_radius would be ~63mm, not
    # ~3mm, and pitch_feature would be absurdly coarse (>1cm) instead of sub-mm
    assert report["min_leaf_radius_m"] < 0.004
    assert report["pitch_feature_m"] < 0.002


def test_voxelize_construct_sdf_matches_composed_sdf_away_from_surface():
    """Ground-truth parity: voxel.surface_distance() vs the exact
    composed.sdf_batch(), tolerance scaled to pitch (this compares an exact
    field to an intentional discretization -- NOT D1's 1e-9 parity bar),
    excluding a near-surface shell where sign disagreement across a
    discretized boundary is expected and harmless."""
    import numpy as np
    c = dcompile(_multi_leaf_spec(), resolution=24)
    voxel, materials, report = voxelize_construct_sdf(c)
    pitch = report["pitch_m"]

    rng = np.random.default_rng(2024)
    (x0, x1), (y0, y1), (z0, z1) = c.bbox
    lo, dims = _grid_extent(c.bbox, pitch, margin=4)
    hi = lo + dims * pitch
    pts = rng.uniform(lo, hi, size=(4000, 3))

    exact = c.composed.sdf_batch(pts)
    approx = np.array([voxel.surface_distance(*p) for p in pts])
    diff = np.abs(exact - approx)

    shell = np.abs(exact) < 2.0 * pitch
    tol = 1.5 * pitch
    outside = diff[~shell]
    assert outside.size > 0
    assert outside.max() < tol, f"max diff {outside.max()} exceeds tol {tol}"


def test_voxelize_construct_sdf_rejects_mesh_sourced_composed_with_explicit_pitch():
    class _FakeMeshComposed:
        _leaves = []
    c = dcompile(_multi_leaf_spec(), resolution=24)
    fake = Construct(**{**c.__dict__, "composed": _FakeMeshComposed()})
    with pytest.raises(TypeError):
        voxelize_construct_sdf(fake, pitch=0.001)   # explicit pitch skips adaptive_pitch's guard


def test_voxelize_construct_sdf_zero_occupancy_raises():
    c = dcompile(_multi_leaf_spec(), resolution=24)
    with pytest.raises(ValueError):
        voxelize_construct_sdf(c, pitch=10.0)        # absurdly coarse: whole object < 1 cell


def test_construct_to_voxel_fidelity_preserves_physics_verbatim():
    c = dcompile(_multi_leaf_spec(), resolution=24)
    wrapped = construct_to_voxel_fidelity(c)
    assert wrapped.mass_kg == c.mass_kg
    assert wrapped.com_m == c.com_m
    assert wrapped.inertia_kgm2 == c.inertia_kgm2
    assert wrapped.bbox == c.bbox
    assert wrapped.layers == c.layers
    assert wrapped.density_by_label == c.density_by_label
    assert wrapped.articulation == c.articulation
    # only the query surface + provenance changed
    assert wrapped.composed is not c.composed
    assert "voxel_bridge" in wrapped.validation
    assert wrapped.validation["passed"] == c.validation["passed"]   # original verdict kept


def test_construct_to_voxel_fidelity_renders_through_the_unmodified_scene_export_path():
    """Zero shader/exporter changes needed -- construct_to_scene()/scene_spec_to_sdf()
    round-trip a voxel-bridged construct exactly like a mesh-voxel one."""
    from sigma_ground.radiance.scene_export import construct_to_scene, scene_spec_to_sdf
    from sigma_ground.dynamics.vec import Vec3
    c = dcompile(_multi_leaf_spec(), resolution=24)
    wrapped = construct_to_voxel_fidelity(c)
    spec = construct_to_scene(wrapped)
    assert len(spec["csg_leaves"]) == 1
    assert spec["csg_leaves"][0]["shape"]["type"] == "Voxel"
    sdf, material_at = scene_spec_to_sdf(spec)
    cx, cy, cz = wrapped.com_m
    assert sdf(Vec3(cx, cy, cz)) < 0.0   # CoM is inside the reconstructed solid


def test_grid_extent_and_chunk_points_are_consistent_with_voxel_center():
    """Hand-derived algebraic identity check: lo+(idx+0.5)*pitch must equal
    kernel.voxel.Voxel's own center+(i-(n-1)/2)*pitch convention."""
    import numpy as np
    c = dcompile(_multi_leaf_spec(), resolution=24)
    pitch, _ = adaptive_pitch(c)
    lo, dims = _grid_extent(c.bbox, pitch, margin=4)
    voxel, _, report = voxelize_construct_sdf(c, pitch=pitch)
    nx, ny, nz = dims.tolist()
    i, j, k = nx // 2, ny // 2, nz // 2
    grid_pt = lo + (np.array([i, j, k]) + 0.5) * pitch
    cx, cy, cz = voxel.center
    voxel_conv_pt = (cx + (i - (nx - 1) * 0.5) * pitch,
                     cy + (j - (ny - 1) * 0.5) * pitch,
                     cz + (k - (nz - 1) * 0.5) * pitch)
    for a, b in zip(grid_pt, voxel_conv_pt):
        assert a == pytest.approx(b, abs=1e-9)


def test_identify_voxel_sdf_sentinel_reaches_the_bridge():
    """The public opt-in: identify(fidelity="voxel_sdf") compiles at primitive
    fidelity then wraps through the bridge -- composed becomes a _VoxelComposed,
    the bridge report lands in validation, and mass rides over verbatim."""
    from sigma_ground import deckard
    plain = deckard.identify("coffee cup", allow_llm=False)
    bridged = deckard.identify("coffee cup", allow_llm=False, fidelity="voxel_sdf")
    assert type(bridged.composed).__name__ == "_VoxelComposed"
    assert "voxel_bridge" in bridged.validation
    assert bridged.mass_kg == plain.mass_kg          # physics never recomputed


def test_identify_default_fidelity_unaffected_by_new_module():
    """The sdf_voxelize module and the new 'voxel_sdf' sentinel must be fully
    inert unless explicitly requested -- identify()'s default ('primitive')
    path is unchanged: still returns a bare ComposedSDF, not wrapped."""
    from sigma_ground import deckard
    c1 = deckard.identify("doughnut", allow_llm=False)   # frozen catalog entry, offline
    c2 = deckard.identify("doughnut", allow_llm=False)
    assert c1.mass_kg == pytest.approx(c2.mass_kg)
    assert c1.bbox == c2.bbox
    assert c1.com_m == c2.com_m
    assert isinstance(c1.composed, ComposedSDF)           # never wrapped -- branch never fired
    assert "voxel_bridge" not in c1.validation
