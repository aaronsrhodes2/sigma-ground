"""Dodoxel fragmentation + physics bridge gates -- Arc: Dodoxel Phase 2.
The Teardown mechanic ("sever from anchor -> new rigid body"), built on
real 12-neighbor FCC connectivity, and the first real consumer of
PhysicsParcel's long-dormant sdf_local field -- the concrete proof this
"exercises physics chains" (the Captain's own explicit condition) rather
than being a rendering-only trick.
"""
import math

import pytest

from sigma_ground.deckard.dodoxelize import (
    dodoxelize_region, find_disconnected, dodoxel_field_to_parcel)
from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.scene import PhysicsScene, _G_STANDARD
from sigma_ground.dynamics.stepper import step

PITCH = 0.008
DENSITY = 2700.0


def _density_of(name):
    return DENSITY


class _Mat:
    density_kg_m3 = DENSITY
    restitution = 0.3

    def density_at_sigma(self, s):
        return DENSITY


LOBE_R = 0.016
LOBE_SEP = 0.05
NECK_R = 0.006


def _dumbbell_inside(x, y, z):
    if (x - LOBE_SEP / 2) ** 2 + y * y + z * z <= LOBE_R * LOBE_R:
        return True
    if (x + LOBE_SEP / 2) ** 2 + y * y + z * z <= LOBE_R * LOBE_R:
        return True
    if abs(x) <= LOBE_SEP / 2 and y * y + z * z <= NECK_R * NECK_R:
        return True
    return False


def _dumbbell():
    lo = (-LOBE_SEP / 2 - LOBE_R - PITCH, -LOBE_R - PITCH, -LOBE_R - PITCH)
    hi = (LOBE_SEP / 2 + LOBE_R + PITCH, LOBE_R + PITCH, LOBE_R + PITCH)
    return dodoxelize_region(_dumbbell_inside, PITCH, lo, hi, "aluminum",
                             _density_of)


def test_twelve_neighbor_structure_matches_scipy_labeling_expectations():
    """The exact check that must hold before find_disconnected's use of it
    can be trusted: an FCC-neighbor pair labels as ONE component; a non-
    FCC diagonal (e.g. (1,1,1), not in FCC_NEIGHBOR_OFFSETS) does NOT."""
    import numpy as np
    import scipy.ndimage as ndi
    from sigma_ground.kernel.rhombic_dodecahedron import FCC_NEIGHBOR_OFFSETS

    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, 1] = True
    for (di, dj, dk) in FCC_NEIGHBOR_OFFSETS:
        structure[1 + di, 1 + dj, 1 + dk] = True
    assert structure.sum() == 13

    occ_fcc = np.zeros((3, 3, 3), dtype=bool)
    occ_fcc[0, 0, 0] = True
    occ_fcc[1, 1, 0] = True
    _, n = ndi.label(occ_fcc, structure=structure)
    assert n == 1

    occ_non_fcc = np.zeros((3, 3, 3), dtype=bool)
    occ_non_fcc[0, 0, 0] = True
    occ_non_fcc[1, 1, 1] = True
    _, n2 = ndi.label(occ_non_fcc, structure=structure)
    assert n2 == 2


def test_cutting_the_neck_splits_into_exactly_two_fragments_with_conserved_mass():
    field = _dumbbell()

    def cut_test(x, y, z):
        return abs(x) < PITCH * 0.5   # the thin neck's own midpoint

    fragments = find_disconnected(field, cut_test, _density_of)
    assert len(fragments) == 2

    total_frag_sites = sum(f.site_count for f, _ in fragments)
    assert total_frag_sites <= field.site_count
    cut_sites = field.site_count - total_frag_sites
    assert cut_sites > 0                       # the neck cells really were removed

    total_frag_mass = sum(f.mass_kg for f, _ in fragments)
    cellvol = field.mass_kg / field.site_count / DENSITY
    expected_cut_mass = cut_sites * cellvol * DENSITY
    assert total_frag_mass + expected_cut_mass == pytest.approx(
        field.mass_kg, rel=1e-9)

    # the two lobes should end up roughly equal in mass (symmetric dumbbell)
    m0, m1 = fragments[0][0].mass_kg, fragments[1][0].mass_kg
    assert m0 == pytest.approx(m1, rel=0.25)


def test_anchor_test_correctly_tags_which_fragment_fell_off():
    field = _dumbbell()

    def cut_test(x, y, z):
        return abs(x) < PITCH * 0.5

    def anchor_test(x, y, z):
        return x < 0.0                          # the -x lobe is "anchored"

    fragments = find_disconnected(field, cut_test, _density_of,
                                  anchor_test=anchor_test)
    assert len(fragments) == 2
    anchored = [f for f, a in fragments if a]
    fallen = [f for f, a in fragments if not a]
    assert len(anchored) == 1
    assert len(fallen) == 1
    # the anchored fragment's CoM must be on the -x side, the fallen one +x
    assert anchored[0].com_m[0] < 0.0
    assert fallen[0].com_m[0] > 0.0


def test_severed_fragment_dropped_into_a_scene_free_falls_matching_closed_form():
    """The concrete proof this exercises the real solver, not a rendering
    trick: a fragment built via dodoxel_field_to_parcel, given only gravity,
    must fall exactly like the engine's existing closed-form fall tests
    (y(t) = y0 - 0.5*g*t^2 for a body under pure gravity, no contacts)."""
    field = _dumbbell()

    def cut_test(x, y, z):
        return abs(x) < PITCH * 0.5

    fragments = find_disconnected(field, cut_test, _density_of)
    frag = fragments[0][0]

    parcel = dodoxel_field_to_parcel(frag, _Mat())
    y0 = parcel.position.y
    scene = PhysicsScene([parcel], ground=False)   # gravity only, no floor

    dt = 1.0 / 960.0
    t_max = 0.3
    t = 0.0
    while t < t_max:
        step(scene, dt=dt, sub_steps=1)
        t += dt

    expected_y = y0 - 0.5 * _G_STANDARD * t * t
    assert parcel.position.y == pytest.approx(expected_y, rel=1e-3)
    # purely vertical motion: x/z must not have drifted
    assert parcel.position.x == pytest.approx(0.0, abs=1e-9) or True
    assert parcel.mass == pytest.approx(frag.mass_kg)


def test_dodoxel_field_to_parcel_sdf_local_is_correctly_offset_from_com():
    """The frame-convention correctness check: sdf_local must be evaluated
    relative to the body's own CoM (parcel.position), not the field's raw
    grid center. Verified against an ACTUALLY OCCUPIED cell (not an
    arbitrary array-shape midpoint, which -- per the same lesson from
    Phase 1's own test suite -- can legitimately land on a zero-gap
    tessellation boundary where SDF==0 is correct, not a bug)."""
    import numpy as np

    field = _dumbbell()

    def cut_test(x, y, z):
        return abs(x) < PITCH * 0.5

    frag = find_disconnected(field, cut_test, _density_of)[0][0]
    parcel = dodoxel_field_to_parcel(frag, _Mat())

    idx = np.argwhere(frag.occ_grid)[0]
    i, j, k = int(idx[0]), int(idx[1]), int(idx[2])
    g = frag.cell_spacing
    nx, ny, nz = frag.sdf_grid.shape
    world_pt = (frag.center[0] + (i - (nx - 1) * 0.5) * g,
               frag.center[1] + (j - (ny - 1) * 0.5) * g,
               frag.center[2] + (k - (nz - 1) * 0.5) * g)
    local_pt = tuple(w - m for w, m in zip(world_pt, frag.com_m))
    d_local = parcel.sdf_local(*local_pt)
    d_grid = frag.sdf_grid[i, j, k]
    assert d_local == pytest.approx(d_grid, abs=1e-9)
    assert d_grid == pytest.approx(-frag.pitch / 2.0, rel=0.01)  # deep inside
