"""sdf_voxelize.py — Direct-SDF voxel bridge for compiled Deckard Constructs.

Turns an already-compiled, PRIMITIVE-fit Construct (composed = a real
kernel.csg.ComposedSDF, built by construct.py's _compile_parts/_compile_vessel)
into a drop-in voxel-fidelity Construct by SAMPLING the existing analytic SDF
on a regular grid via ComposedSDF.sdf_batch() (Phase D1's vectorized query
path) -- not by re-deriving geometry, not by re-integrating physics. Mass /
CoM / inertia / layers / density_by_label / articulation all ride forward
UNCHANGED from the source construct; only `.composed` (the renderer/physics
query surface) is replaced.

Scope: ONLY constructs whose `.composed` is a `kernel.csg.ComposedSDF` (the
"primitive" fidelity path). Mesh-voxelized constructs (fidelity="voxel"/
"auto") already carry a `deckard.voxelize._VoxelComposed`, which has no
`sdf_batch` -- feeding one in here raises TypeError rather than silently
doing the wrong thing.

Kernel/core-library convention (matches kernel/csg.py's *_batch functions and
deckard/voxelize.py): numpy is imported lazily inside function bodies, so
importing this module -- or `deckard` itself -- never requires numpy. The
core sigma_ground library stays zero-dependency (pyproject.toml
`dependencies = []`); numpy is only paid for when this bridge is actually
invoked.
"""
from __future__ import annotations

import dataclasses


def _grid_extent(bbox, pitch, margin):
    """(lo, dims) for a padded grid covering `bbox` at `pitch`, `margin` cells
    of empty padding on every side (matches deckard/voxelize.py's own
    lo/hi/dims convention so cell (i,j,k) sits at lo+(idx+0.5)*pitch, the same
    lattice kernel.voxel.Voxel's center convention reproduces exactly --
    center = lo + dims*pitch/2 makes Voxel's `center+(i-(n-1)/2)*pitch`
    algebraically identical to `lo+(i+0.5)*pitch`, verified by hand)."""
    import numpy as np
    (x0, x1), (y0, y1), (z0, z1) = bbox
    lo = np.array([x0, y0, z0], dtype=np.float64) - margin * pitch
    hi = np.array([x1, y1, z1], dtype=np.float64) + margin * pitch
    dims = np.maximum(np.ceil((hi - lo) / pitch).astype(np.int64), 1)
    return lo, dims


def _chunk_points(lo, dims, pitch, start, end):
    """World-space sample points for flat cell indices [start,end)."""
    import numpy as np
    nx, ny, nz = (int(d) for d in dims)
    idx = np.arange(start, end)
    ii, jj, kk = np.unravel_index(idx, (nx, ny, nz))
    return lo + (np.stack([ii, jj, kk], axis=1).astype(np.float64) + 0.5) * pitch


def adaptive_pitch(construct, *, cells_across_feature=6, cells_across_whole=45,
                   max_grid_cells=6_000_000, margin=4):
    """Pick a voxel pitch (m) that resolves the construct's smallest REALIZED
    leaf without exploding the grid for a large-extent, tiny-feature object.

    ``pitch_feature`` uses ``leaf.shape.bounding_radius()`` -- the PRIMITIVE's
    own size -- never ``CSGLeaf.bounding_radius()``, which folds in distance
    from the world origin and would make an off-center small part look huge.
    Uses the REALIZED post-compile leaves (construct.composed._leaves), not
    spec templates, so scatter's size_jitter is honoured.

    ``pitch_whole`` reuses deckard/__init__.py's existing "45 cells across the
    longest axis" constant for a recognizable-shape floor on simple objects.

    ``pitch_target = min(pitch_feature, pitch_whole)`` -- whichever is
    binding. A huge but SIMPLE object (one leaf) never explodes the grid,
    because its one leaf sets both quantities equal.

    ``pitch_budget`` is the real safety valve for the genuine risk case (an
    extreme extent-to-smallest-feature ratio): the pitch that keeps the
    padded grid under ``max_grid_cells``. ``report["budget_clamped"]`` is
    True when the budget -- not the target -- won, so callers can see when
    a feature had to be under-resolved to stay within budget.

    Raises ``TypeError`` if ``construct.composed`` isn't a
    ``kernel.csg.ComposedSDF`` -- the scope guard against mesh-sourced input
    (fidelity="voxel"/"auto", whose composed is a `_VoxelComposed` with no
    `sdf_batch`).
    """
    from ..kernel.csg import ComposedSDF

    composed = construct.composed
    if not isinstance(composed, ComposedSDF):
        raise TypeError(
            f"adaptive_pitch: construct.composed is {type(composed).__name__}, "
            "not a kernel.csg.ComposedSDF -- this bridge only accepts primitive-"
            "compiled constructs (fidelity='primitive'), not mesh-voxelized ones "
            "(fidelity='voxel'/'auto', composed=_VoxelComposed, no sdf_batch).")
    leaves = composed._leaves
    if not leaves:
        raise ValueError("adaptive_pitch: construct.composed has no leaves")

    min_leaf_radius = min(leaf.shape.bounding_radius() for leaf, _op in leaves)

    (x0, x1), (y0, y1), (z0, z1) = construct.bbox
    longest_bbox_extent = max(x1 - x0, y1 - y0, z1 - z0)
    bbox_volume = (x1 - x0) * (y1 - y0) * (z1 - z0)

    pitch_feature = 2.0 * min_leaf_radius / cells_across_feature
    pitch_whole = longest_bbox_extent / cells_across_whole
    pitch_target = min(pitch_feature, pitch_whole)
    pitch_budget = ((bbox_volume / max_grid_cells) ** (1.0 / 3.0)
                    if max_grid_cells > 0 and bbox_volume > 0 else 0.0)

    pitch = max(pitch_target, pitch_budget, 5e-5)
    budget_clamped = pitch_budget > pitch_target

    report = {
        "min_leaf_radius_m": min_leaf_radius,
        "longest_bbox_extent_m": longest_bbox_extent,
        "bbox_volume_m3": bbox_volume,
        "pitch_feature_m": pitch_feature,
        "pitch_whole_m": pitch_whole,
        "pitch_target_m": pitch_target,
        "pitch_budget_m": pitch_budget,
        "budget_clamped": budget_clamped,
        "pitch_m": pitch,
        "n_leaves": len(leaves),
        "cells_across_feature": cells_across_feature,
        "cells_across_whole": cells_across_whole,
        "max_grid_cells": max_grid_cells,
        "margin": margin,
    }
    return pitch, report


def voxelize_construct_sdf(construct, *, pitch=None, margin=4, with_materials=True,
                           chunk_points=500_000):
    """Sample construct.composed.sdf_batch() on a regular grid -> a real
    kernel.voxel.Voxel (+ optional per-cell material label grid).

    Builds grid points per chunk via np.unravel_index (bounds peak memory --
    no full (N,3) index array materialized for the whole grid at once), calls
    composed.sdf_batch(pts) per chunk -- the vectorized geometry sampling
    Phase D1 built for exactly this.

    Optional (default-on) materials pass: for each leaf, in the SAME order
    construct.composed._leaves iterates them, `leaf.shape.surface_distance_
    batch(pts) < 0` overwrites a label array -- forward iteration + last-
    write-wins is exactly equivalent to `ComposedSDF.material_at`'s
    documented "reverse iterate, first match" rule (the highest-index leaf
    containing a point wins either way), so the bridge is bug-compatible
    with `material_at` (including its documented ignore-CSG-ops quirk on raw
    per-leaf containment). The result is then gated by the AUTHORITATIVE
    `sdf_grid < 0` occupancy (which, unlike material_at, DOES respect
    subtract/intersect ops via sdf_batch), so a cell truly carved away by a
    subtract op is correctly void even if some leaf's raw, un-subtracted
    shape would have claimed it.

    Raises ValueError if the sampled grid has zero occupied cells (an
    absurdly coarse caller-supplied `pitch`) -- mirrors deckard/voxelize.py's
    `_finalize`'s existing empty-occupancy guard on the mesh path, so this
    bridge never silently hands back a physically-empty Voxel.

    Returns (Voxel, materials_list_or_None, report).
    """
    import time
    import numpy as np
    from ..kernel.voxel import Voxel
    from ..kernel.csg import ComposedSDF

    composed = construct.composed
    pitch_report = None
    if pitch is None:
        pitch, pitch_report = adaptive_pitch(construct, margin=margin)
    elif not isinstance(composed, ComposedSDF):
        raise TypeError(
            f"voxelize_construct_sdf: construct.composed is "
            f"{type(composed).__name__}, not a kernel.csg.ComposedSDF")

    lo, dims = _grid_extent(construct.bbox, pitch, margin)
    nx, ny, nz = (int(d) for d in dims)
    total_cells = nx * ny * nz

    sdf_flat = np.empty(total_cells, dtype=np.float64)
    leaves = composed._leaves if with_materials else None
    label_flat = np.zeros(total_cells, dtype=np.int32) if with_materials else None
    materials = ["air"] if with_materials else None
    mat_index = {} if with_materials else None

    t0 = time.time()
    for start in range(0, total_cells, chunk_points):
        end = min(start + chunk_points, total_cells)
        pts = _chunk_points(lo, dims, pitch, start, end)
        sdf_flat[start:end] = composed.sdf_batch(pts)
        if with_materials:
            for leaf, _op in leaves:
                m = leaf.material
                if m not in mat_index:
                    mat_index[m] = len(materials)
                    materials.append(m)
                inside = leaf.shape.surface_distance_batch(pts) < 0.0
                if inside.any():
                    label_flat[start:end][inside] = mat_index[m]
    sample_time_s = time.time() - t0

    sdf_grid = sdf_flat.reshape(nx, ny, nz)
    occ = sdf_grid < 0.0
    occupied_cells = int(occ.sum())
    if occupied_cells == 0:
        raise ValueError(
            f"voxelize_construct_sdf: zero occupied cells at pitch={pitch*1000:.4f}mm "
            f"over a {nx}x{ny}x{nz} grid -- pitch is too coarse for this construct's "
            f"bbox {construct.bbox}; pass an explicit finer `pitch` or omit it to use "
            f"adaptive_pitch()")

    dominant_material = None
    label_grid = None
    if with_materials:
        label_grid = label_flat.reshape(nx, ny, nz)
        label_grid[~occ] = 0                     # authoritative occupancy gate
        if len(materials) > 1:
            counts = np.bincount(label_grid.ravel(), minlength=len(materials))
            counts[0] = 0                          # exclude void/air
            if counts.any():
                dominant_material = materials[int(counts.argmax())]

    center = tuple((lo + dims.astype(np.float64) * pitch * 0.5).tolist())
    voxel = Voxel(sdf_grid, pitch, center=center,
                  volume_m3=occupied_cells * pitch ** 3,
                  label_grid=label_grid,
                  source=f"SDF-sampled from '{construct.name}' @ {pitch*1000:.4f}mm "
                         f"(direct-SDF voxel bridge, {nx}x{ny}x{nz})")

    report = {
        "pitch_m": pitch, "margin_cells": margin,
        "grid_dims": (nx, ny, nz), "total_cells": total_cells,
        "occupied_cells": occupied_cells,
        "occupied_fraction": occupied_cells / total_cells if total_cells else 0.0,
        "sample_time_s": sample_time_s,
        "with_materials": with_materials,
        "n_materials": (len(materials) - 1) if materials else 0,
        "dominant_material": dominant_material,
        "source_construct": construct.name,
    }
    if pitch_report is not None:
        report["adaptive_pitch"] = pitch_report
    return voxel, materials, report


def construct_to_voxel_fidelity(construct, *, pitch=None, margin=4,
                                with_materials=True, chunk_points=500_000):
    """Wrap `construct` in a voxel-sampled `.composed`; everything else
    (mass/CoM/inertia/layers/density_by_label/articulation) copied verbatim
    -- physics is NEVER recomputed from the grid, only the render/geometry-
    query surface changes.
    """
    from .voxelize import _VoxelComposed

    voxel, materials, report = voxelize_construct_sdf(
        construct, pitch=pitch, margin=margin, with_materials=with_materials,
        chunk_points=chunk_points)

    dominant = report.get("dominant_material")
    if dominant is None:
        # unlabeled grid (with_materials=False), or a labeled grid that
        # happened to end up all-void (shouldn't happen post the zero-
        # occupancy guard, but stay honest): fall back to the analytic
        # construct's largest-volume layer so the single render leaf still
        # gets a sensible material identity.
        dominant = (max(construct.layers, key=lambda L: L.volume_m3).material
                    if construct.layers else "default")
    if materials is None:
        materials = ["air", dominant]

    wrapped = _VoxelComposed(voxel, materials, dominant=dominant, part_of_label=None)

    gd = report["grid_dims"]
    note = (f"{construct.source}; voxel-SDF bridge: {gd[0]}x{gd[1]}x{gd[2]} @ "
            f"{report['pitch_m']*1000:.4f}mm ({report['occupied_cells']:,} occupied "
            f"/ {report['total_cells']:,} cells)")

    return dataclasses.replace(
        construct,
        composed=wrapped,
        source=note,
        validation={**construct.validation, "voxel_bridge": report},
    )


__all__ = ["adaptive_pitch", "voxelize_construct_sdf", "construct_to_voxel_fidelity"]
