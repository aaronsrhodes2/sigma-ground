"""Dodoxel joint inference -- the FCC lattice's contact geometry turned
into typed joint proposals, the dodoxel twin of ``articulate.py``'s
``infer_joints`` with one decisive upgrade: where the cubic scan infers a
hinge axis by PCA alone and joint TYPE by a part-NAME rule table, the
dodoxel scan has REAL geometry to reason from --

  - each contact's normal is an exact discrete FCC direction (Phase B's
    scan, grounded in Phase 0's Voronoi-cell tessellation gate), and
  - the angle between ANY two of the 12 neighbor directions is exactly one
    of {60, 90, 120, 180} degrees (verified numerically 2026-07-18 over
    all 66 direction pairs; a cubic 6-neighbor grid has only {90, 180}).

That discrete angle vocabulary is the Captain's own insight (2026-07-18,
the cold-welding / "rolly hinge" observation): a chain of dodoxel cells
can only re-orient between a small set of exact surface-flush angles --
surfaces meet face-to-face at one of these angles or not at all, which is
both why the lattice supports clean re-welding and why a hinge inferred
here carries an honest CANDIDATE ANGLE SET instead of a made-up limit
range. This module attaches that vocabulary as machine-readable metadata
on every inferred joint; actually enforcing vocabulary-snapped angle
LIMITS on a live RevoluteJoint is deliberately NOT done yet (a physical
claim needing its own gated phase, not a metadata note).
"""
from __future__ import annotations

import math

from ..kernel.rhombic_dodecahedron import FCC_NEIGHBOR_UNIT_DIRECTIONS
from ..dynamics.mechanisms.choice import Choice

# Verified exact pairwise angle set of the 12 FCC neighbor directions.
FCC_ANGLE_VOCABULARY_DEG = (60.0, 90.0, 120.0, 180.0)
_ANGLE_PROVENANCE = ("exact pairwise angles between the 12 FCC neighbor "
                     "directions (verified over all 66 pairs, 2026-07-18); "
                     "a cubic 6-neighbor grid offers only {90, 180}")

# Snap candidates for an inferred hinge line: nearest-neighbor chains run
# along the 12 FCC directions, and same-sublattice colinear runs (the
# (0,0,+-2)-type second-nearest chains, e.g. Phase B's line-interface test
# fixture) run along the 6 cube axes -- both are real lattice line
# families, so both are legal snap targets, tagged by family.
_AXIS_DIRECTIONS = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
                    (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
_SNAP_CANDIDATES = ([(d, "fcc") for d in FCC_NEIGHBOR_UNIT_DIRECTIONS]
                    + [(d, "axis") for d in _AXIS_DIRECTIONS])


def _snap_to_lattice_direction(raw_dir):
    """Nearest lattice line direction (sign-insensitive -- a hinge line has
    no orientation). Returns (snapped_unit_dir, family, error_deg)."""
    rx, ry, rz = raw_dir
    mag = math.sqrt(rx * rx + ry * ry + rz * rz)
    if mag < 1e-12:
        return None, None, None
    rx, ry, rz = rx / mag, ry / mag, rz / mag
    best, best_family, best_dot = None, None, -1.0
    for (dx, dy, dz), family in _SNAP_CANDIDATES:
        dot = abs(rx * dx + ry * dy + rz * dz)
        if dot > best_dot:
            best_dot, best, best_family = dot, (dx, dy, dz), family
    err_deg = math.degrees(math.acos(min(1.0, best_dot)))
    return best, best_family, err_deg


def infer_dodoxel_joints(field, elongation_threshold=None):
    """Infer a typed joint proposal for every touching part pair of a
    multi-part DodoxelField (Phase A/B tables required).

    Returns {"joints": [...], "choices": [...]}. Each joint dict:
      part_a/part_b, type ("weld"|"revolute"), confidence, anchor_m
      (contact centroid), area_m2, and for revolute: axis_raw, axis
      (lattice-snapped), axis_family ("fcc"|"axis"), snap_error_deg,
      angle_vocabulary_deg + angle_vocabulary_note (the FCC set, as
      metadata -- see module docstring for why limits are NOT set from it
      yet).

    Type decision, mirroring articulate.py's own conservatism: weld by
    default (confidence 0.8 -- the same value its cubic twin assigns an
    unmatched pair); revolute only when the contact patch is genuinely
    line-like (elongation >= threshold AND >= 3 contacts), confidence 0.5
    (again matching the cubic rule-table's mover confidence).
    """
    if field.part_interfaces is None:
        raise ValueError("infer_dodoxel_joints needs a multi-part field "
                         "(build it with dodoxelize_parts)")

    threshold_choice = Choice(
        "hinge-inference elongation threshold (contact-patch lambda1/lambda2 "
        "above which a part pair reads as a hinge line rather than a weld "
        "patch)",
        elongation_threshold if elongation_threshold is not None else 8.0,
        checked="deckard/articulate.py (the cubic twin types joints by a "
        "part-NAME rule table, not geometry — no geometric threshold "
        "precedent exists to cite)",
        reason="picked so Phase B's own line fixture (elongation > 10) "
              "reads as a hinge and its slab fixture (< 4) reads as a "
              "weld, with margin both ways; nothing physical pins the "
              "exact value")
    threshold = float(threshold_choice.value)

    joints = []
    for (pa, pb), iface in sorted(field.part_interfaces.items()):
        base = {
            "part_a": field.parts[pa]["name"],
            "part_b": field.parts[pb]["name"],
            "anchor_m": iface["centroid_m"],
            "area_m2": iface["area_m2"],
            "n_contacts": iface["n_contacts"],
        }
        is_line = (iface["elongation"] >= threshold
                   and iface["n_contacts"] >= 3)
        if is_line:
            snapped, family, err = _snap_to_lattice_direction(
                iface["principal_dir"])
            base.update({
                "type": "revolute",
                "confidence": 0.5,
                "axis_raw": iface["principal_dir"],
                "axis": snapped,
                "axis_family": family,
                "snap_error_deg": round(err, 3),
                "angle_vocabulary_deg": list(FCC_ANGLE_VOCABULARY_DEG),
                "angle_vocabulary_note": _ANGLE_PROVENANCE,
            })
        else:
            base.update({"type": "weld", "confidence": 0.8})
        joints.append(base)

    return {"joints": joints, "choices": [threshold_choice.to_dict()]}


__all__ = ["infer_dodoxel_joints", "FCC_ANGLE_VOCABULARY_DEG"]
