"""Dodoxel far-field SDF gates -- Teardown Phase 4 (part 1): the empty-
space skip, plus the validity property that motivated rebuilding the far
band in the first place.

The original _sdf_grid_from_sites capped every no-candidate cell at a flat
4*pitch -- an OVERESTIMATE wherever the nearest site sat just outside the
+-2-index search window (true surface distance there can be as low as
2g ~= 1.41*pitch), i.e. a latent sphere-tracing tunneling bug for thin
dodoxel walls. The rebuilt two-band construction (exact-clamped near band +
EDT-derived far band) must satisfy the ONE property sphere tracing actually
requires -- grid value <= true distance, everywhere -- while making far-
field values GROW with distance (the Teardown-style skip) instead of
flat-capping, so empty-space marching takes big steps.
"""
import math

import pytest

from sigma_ground.deckard.dodoxelize import dodoxelize_region
from sigma_ground.kernel.rhombic_dodecahedron import RhombicDodecahedron
from sigma_ground.dynamics.vec import Vec3
from sigma_ground.radiance.raymarch import march

PITCH = 0.01
DENSITY = 2700.0


def _density_of(name):
    return DENSITY


def _small_ball(radius=0.015, margin=8):
    def inside(x, y, z):
        return x * x + y * y + z * z <= radius * radius
    lo = (-radius - PITCH, -radius - PITCH, -radius - PITCH)
    hi = (radius + PITCH, radius + PITCH, radius + PITCH)
    return dodoxelize_region(inside, PITCH, lo, hi, "aluminum", _density_of,
                             sdf_margin_cells=margin)


def _brute_force_true_distance(field, x, y, z):
    """Independent ground truth: min over ALL occupied sites of the exact
    per-cell closed-form SDF -- no windows, no caps, no EDT."""
    import numpy as np
    g = field.cell_spacing
    nx, ny, nz = field.occ_grid.shape
    cell = RhombicDodecahedron(field.pitch)
    best = float("inf")
    for (i, j, k) in np.argwhere(field.occ_grid):
        cx = field.center[0] + (int(i) - (nx - 1) * 0.5) * g
        cy = field.center[1] + (int(j) - (ny - 1) * 0.5) * g
        cz = field.center[2] + (int(k) - (nz - 1) * 0.5) * g
        cell.center = (cx, cy, cz)
        d = cell.surface_distance(x, y, z)
        if d < best:
            best = d
    return best


def _brute_force_center_distance(field, x, y, z):
    """min over ALL occupied sites of the Euclidean distance to the site
    CENTER -- the independent cross-check for the EDT-derived far band."""
    import numpy as np
    g = field.cell_spacing
    nx, ny, nz = field.occ_grid.shape
    best = float("inf")
    for (i, j, k) in np.argwhere(field.occ_grid):
        cx = field.center[0] + (int(i) - (nx - 1) * 0.5) * g
        cy = field.center[1] + (int(j) - (ny - 1) * 0.5) * g
        cz = field.center[2] + (int(k) - (nz - 1) * 0.5) * g
        d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        if d < best:
            best = d
    return best


def test_grid_bands_match_their_own_certified_constructions():
    """Validity is certified band by band (an earlier draft of this test
    compared the whole grid against the min-over-cells ANALYTIC SDF as
    "ground truth" -- wrong oracle: that SDF is itself conservative near
    cell corners (the documented Phase 0 property), so the EDT far band,
    which is valid against ACTUAL geometry by the triangle inequality,
    legitimately exceeds it. The correct empirical checks are the two
    below; the far band's validity vs. actual geometry is a theorem
    (|p-c| - R <= true distance for every cell), so what needs testing is
    the IMPLEMENTATION, not the inequality):

      NEAR BAND: wherever the stored value is strictly below the 2g clamp,
        the achieving site is provably within the +-2 search window (its
        center must lie within d + R < 3g), so the stored value must EQUAL
        the brute-force min over ALL sites of the analytic SDF -- exact.

      FAR BAND: wherever no window candidate exists, the stored value must
        EQUAL (brute-force min center distance) - R -- validating the EDT
        call's axis conventions and scaling against a from-scratch loop.
    """
    field = _small_ball()
    import numpy as np
    g = field.cell_spacing
    R = field.pitch / math.sqrt(2.0)
    near_cap = 3.0 * g - R
    nx, ny, nz = field.sdf_grid.shape
    occ_set = {tuple(int(v) for v in idx) for idx in np.argwhere(field.occ_grid)}

    def has_window_candidate(i, j, k):
        for di in range(-2, 3):
            for dj in range(-2, 3):
                for dk in range(-2, 3):
                    if (i + di, j + dj, k + dk) in occ_set:
                        return True
        return False

    checked_near = checked_far = 0
    for i in range(0, nx, 2):                      # stride 2: keep runtime sane
        for j in range(0, ny, 2):
            for k in range(0, nz, 2):
                x = field.center[0] + (i - (nx - 1) * 0.5) * g
                y = field.center[1] + (j - (ny - 1) * 0.5) * g
                z = field.center[2] + (k - (nz - 1) * 0.5) * g
                v = field.sdf_grid[i, j, k]
                if has_window_candidate(i, j, k):
                    brute = _brute_force_true_distance(field, x, y, z)
                    if v < near_cap - 1e-9:
                        assert v == pytest.approx(brute, abs=1e-9), (
                            f"near-band mismatch at ({i},{j},{k})")
                        checked_near += 1
                    else:
                        assert v == pytest.approx(near_cap, abs=1e-9)
                else:
                    center_d = _brute_force_center_distance(field, x, y, z)
                    assert v == pytest.approx(center_d - R, abs=1e-9), (
                        f"far-band mismatch at ({i},{j},{k})")
                    checked_far += 1
    assert checked_near > 25
    assert checked_far > 50


def test_far_field_grows_with_distance_not_flat_capped():
    """The empty-space skip itself: values in the padded margin must grow
    roughly linearly toward the corners, not sit at a flat cap."""
    field = _small_ball(margin=8)
    nx, ny, nz = field.sdf_grid.shape
    corner = field.sdf_grid[0, 0, 0]
    nearer = field.sdf_grid[nx // 4, ny // 4, nz // 4]
    old_cap = 4.0 * field.pitch
    assert corner > nearer > 0.0                   # monotone toward the solid
    assert corner > old_cap                        # genuinely beyond the old cap


def test_march_from_far_away_hits_same_point_with_fewer_steps():
    """Identical hit point, fewer steps -- the plan's own gate, comparing
    the new grid against an emulation of the old behavior (clamping the
    valid grid at the old 4*pitch cap is itself still valid, just slower,
    so the comparison is apples-to-apples on correctness)."""
    import numpy as np
    field = _small_ball(margin=10)
    voxel = field.to_voxel()

    calls = {"n": 0}

    def sdf_new(p):
        calls["n"] += 1
        return voxel.surface_distance(p.x, p.y, p.z)

    clamped_grid = np.minimum(field.sdf_grid, 4.0 * field.pitch)
    from sigma_ground.kernel.voxel import Voxel
    voxel_old = Voxel(clamped_grid, field.cell_spacing, center=field.center,
                      volume_m3=field.volume_m3)

    def sdf_old(p):
        calls["n"] += 1
        return voxel_old.surface_distance(p.x, p.y, p.z)

    origin = Vec3(field.center[0] - 30.0 * field.pitch,
                 field.center[1], field.center[2])
    direction = Vec3(1.0, 0.0, 0.0)

    calls["n"] = 0
    t_new = march(sdf_new, origin, direction, max_dist=1.0, max_steps=256)
    steps_new = calls["n"]

    calls["n"] = 0
    t_old = march(sdf_old, origin, direction, max_dist=1.0, max_steps=256)
    steps_old = calls["n"]

    assert t_new is not None and t_old is not None
    assert t_new == pytest.approx(t_old, abs=1e-3)   # same hit point
    assert steps_new < steps_old                      # genuinely fewer steps
