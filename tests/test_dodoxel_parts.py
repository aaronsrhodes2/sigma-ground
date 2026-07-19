"""Multi-part DodoxelField gates -- hinge arc Phase A. The dodoxel twin of
voxelize.py's parts/part_of_label tables: per-part mass/CoM/inertia computed
from the SAME sums as the totals, partitioned, and asserted to reproduce
them exactly (the _finalize mass_check discipline).
"""
import math

import pytest

from sigma_ground.deckard.dodoxelize import dodoxelize_parts, dodoxelize_region

PITCH = 0.008
DENSITIES = {"iron": 7870.0, "copper": 8960.0}


def _density_of(name):
    return DENSITIES[name]


R = 0.014
SEP = 0.04


def _two_ball_field():
    def left(x, y, z):
        return (x + SEP / 2) ** 2 + y * y + z * z <= R * R

    def right(x, y, z):
        return (x - SEP / 2) ** 2 + y * y + z * z <= R * R

    lo = (-SEP / 2 - R - PITCH, -R - PITCH, -R - PITCH)
    hi = (SEP / 2 + R + PITCH, R + PITCH, R + PITCH)
    return dodoxelize_parts(
        [("left_ball", "iron", left), ("right_ball", "copper", right)],
        PITCH, lo, hi, _density_of)


def test_two_parts_with_correct_names_materials_and_sides():
    field = _two_ball_field()
    assert field.parts is not None and len(field.parts) == 2
    pa, pb = field.parts
    assert pa["name"] == "left_ball" and pa["material"] == "iron"
    assert pb["name"] == "right_ball" and pb["material"] == "copper"
    assert pa["com_m"][0] < 0.0 < pb["com_m"][0]      # correct sides
    assert pa["site_count"] > 0 and pb["site_count"] > 0


def test_part_masses_sum_exactly_to_the_total():
    field = _two_ball_field()
    assert sum(p["mass_kg"] for p in field.parts) == pytest.approx(
        field.mass_kg, rel=1e-12)
    # different densities must actually show: copper ball heavier than iron
    # ball of (approximately) the same site count
    pa, pb = field.parts
    if pa["site_count"] == pb["site_count"]:
        assert pb["mass_kg"] > pa["mass_kg"]
    # per-part mass == density * part volume, exactly
    for p in field.parts:
        assert p["mass_kg"] == pytest.approx(
            DENSITIES[p["material"]] * p["volume_m3"], rel=1e-12)


def test_part_grid_is_consistent_with_occupancy():
    import numpy as np
    field = _two_ball_field()
    assert field.part_grid is not None
    assert field.part_grid.shape == field.occ_grid.shape
    assert bool(np.all((field.part_grid >= 0) == field.occ_grid))
    for p in field.parts:
        assert int((field.part_grid == p["part_id"]).sum()) == p["site_count"]


def test_overlap_resolves_by_priority_first_wins():
    def blob(x, y, z):
        return x * x + y * y + z * z <= R * R

    lo = (-R - PITCH,) * 3
    hi = (R + PITCH,) * 3
    field = dodoxelize_parts(
        [("first", "iron", blob), ("second", "copper", blob)],
        PITCH, lo, hi, _density_of)
    assert field.parts[0]["site_count"] > 0
    assert field.parts[1]["site_count"] == 0           # fully shadowed
    assert field.parts[1]["mass_kg"] == 0.0


def test_single_part_path_is_unchanged():
    def blob(x, y, z):
        return x * x + y * y + z * z <= R * R

    lo = (-R - PITCH,) * 3
    hi = (R + PITCH,) * 3
    field = dodoxelize_region(blob, PITCH, lo, hi, "iron",
                              lambda n: DENSITIES["iron"])
    assert field.parts is None
    assert field.part_grid is None
