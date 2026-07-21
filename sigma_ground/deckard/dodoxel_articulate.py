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
range. This module derives the actual limit window from that vocabulary
(see _hinge_period_deg): the FCC lattice's own rotational-symmetry period
about the SPECIFIC hinge axis -- how far the joint can swing from its
flush construction pose before the two parts would cold-weld flush again
on the other side. dodoxelize.py wires that window onto the live
RevoluteJoint's `limits`, so the physical claim is enforced by the
solver, not just noted as metadata.
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


def _rotate_about_axis(v, axis, theta_rad):
    """Rodrigues' rotation formula: v rotated by theta_rad about the unit
    vector axis (both plain (x,y,z) tuples)."""
    vx, vy, vz = v
    ax, ay, az = axis
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    dot = ax * vx + ay * vy + az * vz
    cx = ay * vz - az * vy
    cy = az * vx - ax * vz
    cz = ax * vy - ay * vx
    return (vx * c + cx * s + ax * dot * (1.0 - c),
           vy * c + cy * s + ay * dot * (1.0 - c),
           vz * c + cz * s + az * dot * (1.0 - c))


def _maps_neighbor_star_onto_itself(axis, theta_deg, tol=1e-6):
    """True iff rotating the full 12-direction FCC neighbor star about
    ``axis`` (unit vector) by theta_deg maps the star back onto itself AS A
    SET -- the exact condition for theta_deg to be a rotational-symmetry
    period of the FCC lattice about this specific axis (verified for both
    axis families this module snaps to: 90deg for a <100> cube axis, 180deg
    for a <110> FCC-neighbor axis -- standard cuboctahedral/point-group-O
    facts, reconfirmed numerically here rather than assumed)."""
    theta = math.radians(theta_deg)
    for v in FCC_NEIGHBOR_UNIT_DIRECTIONS:
        rv = _rotate_about_axis(v, axis, theta)
        if not any(math.sqrt(sum((p - q) ** 2 for p, q in zip(rv, ov))) < tol
                  for ov in FCC_NEIGHBOR_UNIT_DIRECTIONS):
            return False
    return True


def _hinge_period_deg(axis):
    """The smallest FCC_ANGLE_VOCABULARY_DEG angle that is a rotational-
    symmetry period of the lattice about ``axis`` -- the angular spacing
    between consecutive mutually-flush (re-weldable) registrations of two
    dodoxel parts hinged about this axis. Starting from the construction
    pose (a flush registration, angle()==0), the two NEAREST other flush
    registrations reachable by pivoting about axis sit at +-this period,
    so a joint window of (-period, +period) lets the hinge swing freely
    through the one open interval where the parts are never flush, and
    stops it (inelastically, via RevoluteJoint's existing brake-only
    limit rows) exactly where they would cold-weld flush again.

    Computed fresh from the lattice geometry every call (not hardcoded by
    axis_family) so it stays correct if a third axis family (e.g. a <111>
    3-fold axis) is ever added to _SNAP_CANDIDATES. Every axis this module
    ever passes in comes verbatim from _SNAP_CANDIDATES (FCC_NEIGHBOR_
    UNIT_DIRECTIONS or _AXIS_DIRECTIONS), both of which are, by
    construction, exact symmetry axes of the cubic point group -- so the
    ValueError below is defensive/unreachable in current usage, not a
    real runtime path.
    """
    for theta_deg in FCC_ANGLE_VOCABULARY_DEG:   # ascending: smallest first
        if _maps_neighbor_star_onto_itself(axis, theta_deg):
            return theta_deg
    raise ValueError(
        f"axis {axis} is not a rotational symmetry axis of the FCC "
        "lattice at any FCC_ANGLE_VOCABULARY_DEG angle -- "
        "_snap_to_lattice_direction should never produce this")


def infer_dodoxel_joints(field, elongation_threshold=None):
    """Infer a typed joint proposal for every touching part pair of a
    multi-part DodoxelField (Phase A/B tables required).

    Returns {"joints": [...], "choices": [...]}. Each joint dict:
      part_a/part_b, type ("weld"|"revolute"), confidence, anchor_m
      (contact centroid), area_m2, and for revolute: axis_raw, axis
      (lattice-snapped), axis_family ("fcc"|"axis"), snap_error_deg,
      angle_vocabulary_deg + angle_vocabulary_note (the FCC set, as
      metadata), hinge_period_deg (the ONE vocabulary angle that is this
      axis's actual lattice rotational-symmetry period) and limits_rad
      (the (-period, +period) radian window derived from it -- see
      module docstring for the physical claim this encodes).

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
            if snapped is not None:
                period_deg = _hinge_period_deg(snapped)
                limit_rad = math.radians(period_deg)
                limits_rad = (-limit_rad, limit_rad)
            else:
                period_deg = None
                limits_rad = None
            base.update({
                "type": "revolute",
                "confidence": 0.5,
                "axis_raw": iface["principal_dir"],
                "axis": snapped,
                "axis_family": family,
                "snap_error_deg": round(err, 3),
                "angle_vocabulary_deg": list(FCC_ANGLE_VOCABULARY_DEG),
                "angle_vocabulary_note": _ANGLE_PROVENANCE,
                "hinge_period_deg": period_deg,
                "limits_rad": limits_rad,
            })
        else:
            base.update({"type": "weld", "confidence": 0.8})
        joints.append(base)

    return {"joints": joints, "choices": [threshold_choice.to_dict()]}


__all__ = ["infer_dodoxel_joints", "FCC_ANGLE_VOCABULARY_DEG"]
