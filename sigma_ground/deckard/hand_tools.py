"""Two-jaw pivoting hand tools (pliers, tongs, scissors, tweezers,
nutcracker) as a small, cited/flagged procedural mechanism catalog -- the
same tier as the clock/windmill's Kelly-1944-style catalog entries, but
with NO real blueprint on file for any specific product (unlike the clock's
cited going-train). Every dimension is a ``Choice``, same discipline as
``record_motor_spin``'s "placeholder disc, real gear geometry arrives with
blueprint extraction" precedent -- this exists to demonstrate the pivot
GEOMETRY and the joint-INFERENCE pipeline honestly, not to reproduce a
specific manufacturer's tool.

Mechanically each tool is two STRAIGHT bars crossing at one pivot point
(scissors/slip-joint pliers topology) -- jaw on one side of the pivot,
handle on the other, both collinear through it. This is deliberate, not a
simplification of a "real" bent-lever pliers shape: a straight-bar cross
gives a single relative-angle DOF where squeezing the handles together
rotates the jaws together by the SAME angle (opposite linear direction,
same angular sense) -- exactly the motion "open and closed over time"
describes, with no separate linkage needed.

The pivot region is built NARROW in-plane and DEEP along the rotation axis
(Z) on purpose, so the dodoxel interface scan finds a genuinely elongated
contact patch and ``infer_dodoxel_joints`` reads it as a revolute from real
geometry -- nothing here declares the joint type; it is discovered.
"""
from __future__ import annotations

import math

from .dodoxelize import dodoxelize_parts
from .dodoxel_articulate import infer_dodoxel_joints
from ..dynamics.mechanisms.choice import Choice

# name -> (jaw_len_m, handle_len_m, rest_half_angle_deg, bar_width_m,
#          depth_m, material, pitch_m). All Choice-flagged in
# build_hand_tool_field's return, not individually cited here -- no
# specific product is being reproduced; see module docstring. The
# half_angle/bar_width/depth ratios (bar_width ~ 0.125x jaw_len, depth ~
# 1.5-1.7x jaw_len, half_angle=20 deg) are EMPIRICALLY tuned so the pivot's
# overlap patch elongates along Z well past infer_dodoxel_joints' default
# threshold (8.0) -- verified per-entry, not assumed to generalize from one.
_TOOL_CATALOG: dict[str, tuple] = {
    "pliers":       (0.012, 0.028, 20.0, 0.0015, 0.020, "steel_mild", 0.0008),
    "tongs":        (0.010, 0.045, 20.0, 0.0013, 0.017, "steel_mild", 0.0007),
    "scissors":     (0.030, 0.020, 20.0, 0.0038, 0.050, "steel_mild", 0.0018),
    "tweezers":     (0.015, 0.015, 20.0, 0.0019, 0.025, "aluminum", 0.0009),
    "nutcracker":   (0.018, 0.040, 20.0, 0.0023, 0.030, "iron", 0.0011),
    "shears":       (0.035, 0.030, 20.0, 0.0044, 0.058, "steel_mild", 0.0021),
    "tongs_bbq":    (0.014, 0.070, 20.0, 0.0018, 0.023, "steel_mild", 0.0009),
    "wire_cutters": (0.012, 0.032, 20.0, 0.0015, 0.020, "steel_mild", 0.0008),
}

# text-noun -> catalog key, so translator.py can match on the word people
# actually say ("a pair of pliers", "the tongs", "kitchen shears").
TOOL_ALIASES: dict[str, str] = {
    "pliers": "pliers", "plier": "pliers", "needle-nose pliers": "pliers",
    "needle nose pliers": "pliers", "slip-joint pliers": "pliers",
    "tongs": "tongs", "bbq tongs": "tongs_bbq", "barbecue tongs": "tongs_bbq",
    "grill tongs": "tongs_bbq", "kitchen tongs": "tongs",
    "scissors": "scissors", "shears": "shears", "kitchen shears": "shears",
    "tweezers": "tweezers", "forceps": "tweezers",
    "nutcracker": "nutcracker", "nut cracker": "nutcracker",
    "wire cutters": "wire_cutters", "wire cutter": "wire_cutters",
}


def list_hand_tools() -> list[str]:
    return sorted(_TOOL_CATALOG)


def _default_density(name):
    """Tier-1-only density lookup (never materia -- deckard/voxelize.py's
    own ``_default_density`` is the precedent this mirrors)."""
    from ..field.interface.resolve import material_profile
    d = material_profile(name).get("density")
    return float(d.value) if d is not None else 1000.0


def _lever_inside_test(direction_xy, pivot, jaw_len, handle_len, half_w, half_z):
    dx, dy = direction_xy
    px, py, pz = pivot

    def inside(x, y, z):
        rx, ry = x - px, y - py
        s = rx * dx + ry * dy                    # coordinate along lever axis
        if not (-handle_len <= s <= jaw_len):
            return False
        perp = abs(-dy * rx + dx * ry)            # perpendicular in-plane offset
        if perp > half_w:
            return False
        return abs(z - pz) <= half_z

    return inside


def build_hand_tool_field(tool: str = "pliers", pitch_m: float | None = None,
                          rest_half_angle_deg: float | None = None,
                          density_of=None):
    """Build the two-lever DodoxelField for ``tool`` (a key or alias) and run
    joint inference on it. Returns (field, joint_info, meta) where
    ``joint_info`` is the single discovered interface's joint dict from
    ``infer_dodoxel_joints`` (raises if the pivot wasn't discovered as a
    revolute -- a genuine geometry failure, not something to paper over).
    """
    key = TOOL_ALIASES.get((tool or "").strip().lower(),
                           (tool or "").strip().lower())
    if key not in _TOOL_CATALOG:
        raise ValueError(f"unknown hand tool {tool!r}. known: "
                         f"{list_hand_tools()} (aliases: {sorted(TOOL_ALIASES)})")
    (jaw_len, handle_len, cat_half_angle, bar_w, depth, material,
     cat_pitch) = _TOOL_CATALOG[key]
    half_angle = math.radians(rest_half_angle_deg if rest_half_angle_deg is not None
                              else cat_half_angle)
    pitch = pitch_m if pitch_m is not None else cat_pitch

    geometry_choice = Choice(
        f"{key} dimensions (jaw/handle length, pivot half-angle, bar "
        "width, depth) -- no blueprint mechanism pins a real product",
        {"jaw_len_m": jaw_len, "handle_len_m": handle_len,
         "rest_half_angle_deg": math.degrees(half_angle),
         "bar_width_m": bar_w, "depth_m": depth, "material": material},
        checked="sigma_ground/blueprint/catalog/ (no MechanismSpec on file "
        f"for {key!r})",
        reason="dims chosen for a legible render AND a genuinely elongated "
              "pivot contact patch (narrow in-plane, deep along the "
              "rotation axis) so infer_dodoxel_joints discovers the "
              "revolute from real geometry rather than a name rule.")

    half_z = depth / 2.0
    pivot = (0.0, 0.0, 0.0)
    dir_a = (math.cos(half_angle), math.sin(half_angle))
    dir_b = (math.cos(half_angle), -math.sin(half_angle))
    inside_a = _lever_inside_test(dir_a, pivot, jaw_len, handle_len, bar_w / 2.0, half_z)
    inside_b = _lever_inside_test(dir_b, pivot, jaw_len, handle_len, bar_w / 2.0, half_z)

    # swept bounds: both levers' full length at the widest possible angle
    # (rest + generous margin) plus a cell of slack for the SDF margin.
    reach = max(jaw_len, handle_len)
    margin = 3.0 * pitch
    x_lo = -handle_len * math.cos(half_angle) - margin
    x_hi = jaw_len * math.cos(half_angle) + margin
    y_span = reach * math.sin(half_angle) + margin
    bounds_lo = (x_lo, -y_span, -half_z - margin)
    bounds_hi = (x_hi, y_span, half_z + margin)

    if density_of is None:
        density_of = _default_density   # tier-1 only: never import materia here

    field = dodoxelize_parts(
        [("lever_a", material, inside_a), ("lever_b", material, inside_b)],
        pitch, bounds_lo, bounds_hi, density_of)

    if not field.part_interfaces:
        raise ValueError(f"{key!r} geometry produced no lever-lever contact "
                         "-- widen bar_width or shrink the crossing angle")
    inference = infer_dodoxel_joints(field)
    if len(inference["joints"]) != 1:
        raise ValueError(f"{key!r} expected exactly one pivot interface, "
                         f"got {len(inference['joints'])}")
    joint_info = inference["joints"][0]
    if joint_info["type"] != "revolute":
        raise ValueError(f"{key!r} pivot geometry read as "
                         f"{joint_info['type']!r}, not revolute -- narrow "
                         "the bar width or deepen the tool to elongate the "
                         "contact patch along the rotation axis")

    meta = {
        "tool": key, "material": material, "half_angle_rad": half_angle,
        "jaw_len_m": jaw_len, "handle_len_m": handle_len,
        "choices": [geometry_choice.to_dict()] + inference["choices"],
    }
    return field, joint_info, meta


__all__ = ["list_hand_tools", "build_hand_tool_field", "TOOL_ALIASES"]
