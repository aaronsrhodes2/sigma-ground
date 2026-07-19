"""Dodoxel joint-inference gates -- hinge arc Phase C. Geometry-driven
typing (line patch -> revolute, broad patch -> weld) with lattice-snapped
hinge axes and the verified {60,90,120,180} degree FCC angle vocabulary as
machine-readable metadata.
"""
import math

import pytest

from sigma_ground.deckard.dodoxelize import dodoxelize_parts
from sigma_ground.deckard.dodoxel_articulate import (
    infer_dodoxel_joints, FCC_ANGLE_VOCABULARY_DEG)
from sigma_ground.kernel.rhombic_dodecahedron import FCC_NEIGHBOR_UNIT_DIRECTIONS

_SQRT2 = math.sqrt(2.0)
PITCH = 0.008
G = PITCH / _SQRT2


def _density_of(name):
    return 5000.0


def test_angle_vocabulary_is_the_verified_fcc_set():
    """Re-verify the vocabulary against the direction set itself, so the
    module constant can never silently drift from the geometry."""
    import itertools
    angles = set()
    for a, b in itertools.combinations(FCC_NEIGHBOR_UNIT_DIRECTIONS, 2):
        dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
        # round to 3 decimals: acos near +-1 loses precision (180 degrees
        # computes as 179.999999...), and the vocabulary is exact integers
        angles.add(round(math.degrees(math.acos(dot)), 3))
    assert angles == set(FCC_ANGLE_VOCABULARY_DEG)


def _line_pair_field():
    def line_a(x, y, z):
        return abs(x - 2 * G) < 0.35 * G and abs(y - 2 * G) < 0.35 * G

    def line_b(x, y, z):
        return abs(x - 3 * G) < 0.35 * G and abs(y - 3 * G) < 0.35 * G

    return dodoxelize_parts(
        [("a", "iron", line_a), ("b", "iron", line_b)],
        PITCH, (0.0, 0.0, 0.0), (5 * G, 5 * G, 14 * G), _density_of)


def _slab_pair_field():
    split = 3.5 * G

    def left(x, y, z):
        return x < split

    def right(x, y, z):
        return x >= split

    span = 8 * G
    return dodoxelize_parts(
        [("left", "iron", left), ("right", "iron", right)],
        PITCH, (0.0, 0.0, 0.0), (span, span, span), _density_of)


def test_line_interface_infers_revolute_with_snapped_z_axis():
    out = infer_dodoxel_joints(_line_pair_field())
    (j,) = out["joints"]
    assert j["type"] == "revolute"
    assert j["axis_family"] == "axis"                  # the (0,0,+-2) chain family
    assert abs(j["axis"][2]) == pytest.approx(1.0)     # hinge line along z
    assert j["snap_error_deg"] < 1.0
    assert j["angle_vocabulary_deg"] == list(FCC_ANGLE_VOCABULARY_DEG)
    assert "verified" in j["angle_vocabulary_note"]
    assert j["confidence"] == 0.5


def test_slab_interface_infers_weld():
    out = infer_dodoxel_joints(_slab_pair_field())
    (j,) = out["joints"]
    assert j["type"] == "weld"
    assert j["confidence"] == 0.8
    assert "axis" not in j                             # no hinge data on a weld


def test_threshold_is_a_recorded_choice():
    """The elongation threshold is an assistant-invented parameter -- the
    Choice doctrine applies to inference knobs exactly as it does to
    simulation variables."""
    out = infer_dodoxel_joints(_line_pair_field())
    assert len(out["choices"]) == 1
    c = out["choices"][0]
    assert "elongation threshold" in c["description"]
    assert c["value"] == 8.0
    assert c["adjustable"]
    # and it genuinely is adjustable — proven by PROMOTION rather than
    # demotion: a perfect single-file line has zero secondary variance, so
    # its elongation is literally infinite and correctly beats any finite
    # threshold (demoting it is geometrically impossible, not a bug). The
    # SLAB's finite elongation (~2-4), though, flips to revolute when the
    # threshold drops below it — the knob is live.
    loose = infer_dodoxel_joints(_slab_pair_field(),
                                 elongation_threshold=0.5)
    assert loose["joints"][0]["type"] == "revolute"
    assert loose["choices"][0]["value"] == 0.5
