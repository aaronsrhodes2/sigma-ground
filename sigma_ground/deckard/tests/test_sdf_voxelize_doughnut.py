"""Phase D3(b) 'real demo' regression: the direct-SDF voxel bridge against the
actual doughnut-with-24-rainbow-sprinkles reference construct.

Uses the CORRECTED per-leaf resolution metric (see the D2/D3 design's §3 flaw
writeup): connected-component blob-counting on the merged sprinkle-color mask
undercounts (21, not 24) because several sprinkle placements in this exact
seeded scatter (tests/test_scatter.py, seed=7) genuinely overlap each other in
the ANALYTIC geometry already -- independent of voxelization. The right
per-leaf question is "did each of the 24 sprinkle LEAVES individually resolve
with enough of its own geometry", which is well-defined regardless of
neighbor overlap.

Leaf-filter note (a real bug caught by the independent verify pass before
this file was ever written): scattered leaves carry per-INSTANCE labels on
`CSGLeaf.material` ("sprinkle_0".."sprinkle_23") -- the rainbow COLOR names
("sprinkle_red"...) live on `Layer.material`, a separate table (see
tests/test_scatter.py:242). Filtering leaves by membership in _RAINBOW finds
zero leaves; the correct filter is the "sprinkle_" label prefix.
"""
import numpy as np
import pytest

from sigma_ground.deckard.construct import compile as dcompile
from sigma_ground.deckard.sdf_voxelize import (
    voxelize_construct_sdf, construct_to_voxel_fidelity, _grid_extent, _chunk_points,
)
from tests.test_scatter import _build_doughnut


@pytest.fixture(scope="module")
def doughnut_construct():
    spec = _build_doughnut(sprinkle_count=24)
    return dcompile(spec, resolution=40)


def test_doughnut_compiles_with_26_leaves(doughnut_construct):
    assert len(doughnut_construct.composed._leaves) == 26


def test_doughnut_adaptive_pitch_resolves_smallest_sprinkle(doughnut_construct):
    from sigma_ground.deckard.sdf_voxelize import adaptive_pitch
    pitch, report = adaptive_pitch(doughnut_construct)
    # sanity band around the live-measured value (0.4273mm); not pinned exactly
    # so an unrelated upstream dimension tweak doesn't spuriously break this
    assert 0.0003 < pitch < 0.0007
    assert not report["budget_clamped"]
    old_floor_mm = 3.5
    smallest_diam_mm = 2 * report["min_leaf_radius_m"] * 1000
    assert smallest_diam_mm / old_floor_mm < 1.0     # sub-cell at the OLD floor
    assert smallest_diam_mm / (pitch * 1000) >= 4.0  # well-resolved at the NEW pitch


def test_all_24_sprinkle_leaves_individually_resolved(doughnut_construct):
    """The corrected D3(b) metric: per-leaf exact-geometry cell count on the
    SAME lattice voxelize_construct_sdf builds internally -- not connected
    components on the merged occupancy mask (see module docstring)."""
    voxel, materials, report = voxelize_construct_sdf(doughnut_construct)
    pitch = report["pitch_m"]
    lo, dims = _grid_extent(doughnut_construct.bbox, pitch, margin=4)
    total = int(dims[0] * dims[1] * dims[2])
    pts = _chunk_points(lo, dims, pitch, 0, total)

    sprinkle_leaves = [(leaf.material, leaf.shape)
                      for leaf, _op in doughnut_construct.composed._leaves
                      if leaf.material.startswith("sprinkle_")]
    assert len(sprinkle_leaves) == 24

    counts = []
    for name, shape in sprinkle_leaves:
        n = int((shape.surface_distance_batch(pts) < 0.0).sum())
        counts.append(n)
    counts = np.array(counts)

    assert (counts > 0).all(), "at least one sprinkle leaf vanished entirely"
    assert counts.min() >= 4, f"weakest sprinkle only got {counts.min()} cells"


def test_doughnut_voxel_bridge_correctness_vs_exact_sdf(doughnut_construct):
    voxel, materials, report = voxelize_construct_sdf(doughnut_construct)
    pitch = report["pitch_m"]
    lo, dims = _grid_extent(doughnut_construct.bbox, pitch, margin=4)
    hi = lo + dims * pitch

    rng = np.random.default_rng(99)
    pts = rng.uniform(lo, hi, size=(20000, 3))
    exact = doughnut_construct.composed.sdf_batch(pts)
    approx = np.array([voxel.surface_distance(*p) for p in pts])
    diff = np.abs(exact - approx)

    shell = np.abs(exact) < 2.0 * pitch
    tol = 1.5 * pitch
    outside = diff[~shell]
    assert outside.max() < tol


def test_doughnut_voxel_bridge_preserves_mass_and_bbox(doughnut_construct):
    wrapped = construct_to_voxel_fidelity(doughnut_construct)
    assert wrapped.mass_kg == doughnut_construct.mass_kg
    assert wrapped.bbox == doughnut_construct.bbox
    assert wrapped.com_m == doughnut_construct.com_m
    report = wrapped.validation["voxel_bridge"]
    assert report["grid_dims"][0] * report["grid_dims"][1] * report["grid_dims"][2] < 6_000_001
