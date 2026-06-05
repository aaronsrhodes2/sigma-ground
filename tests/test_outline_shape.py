"""The Outline primitive — a 2D researched profile swept into 3D matter.

Known-answer checks against the analytic primitives: an extruded rectangle is a
box; a revolved constant-radius profile is a cylinder. Plus sign correctness of
the SDF (negative inside, positive outside), which is what Deckard's integrator
relies on for mass / material_at.
"""
import math

from sigma_ground.kernel.outline import Outline
from sigma_ground.kernel.shapes import Box, Cylinder


def test_extruded_rectangle_is_a_box():
    w, h, t = 0.04, 0.02, 0.006
    rect = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    o = Outline(rect, mode="extrude", thickness=t)
    assert abs(o.volume() - w * h * t) < 1e-12                 # area x thickness
    assert abs(o.volume() - Box(w, h, t).volume()) < 1e-12
    # sign correctness vs the box it represents
    assert o.surface_distance(0.0, 0.0, 0.0) < 0.0             # centre is inside
    assert o.surface_distance(w, 0.0, 0.0) > 0.0               # outside in x
    assert o.surface_distance(0.0, h, 0.0) > 0.0               # outside in y
    assert o.surface_distance(0.0, 0.0, t) > 0.0               # outside in z (slab)


def test_revolved_constant_profile_is_a_cylinder():
    R, H = 0.02, 0.10
    profile = [(-H / 2, R), (H / 2, R)]            # r(z)=R over z in [-H/2, H/2]
    o = Outline(profile, mode="revolve")
    assert abs(o.volume() - math.pi * R * R * H) < 1e-9
    assert abs(o.volume() - Cylinder(R, H).volume()) < 1e-9
    assert o.surface_distance(0.0, 0.0, 0.0) < 0.0             # on axis, inside
    assert o.surface_distance(R * 0.5, 0.0, 0.0) < 0.0         # inside radially
    assert o.surface_distance(R * 1.5, 0.0, 0.0) > 0.0         # outside radially
    assert o.surface_distance(0.0, 0.0, H) > 0.0               # above the top cap


def test_revolved_cone_profile_matches_cone_volume():
    # r grows linearly 0 -> R over height H: a cone, V = 1/3 pi R^2 H
    R, H = 0.03, 0.09
    profile = [(0.0, 0.0), (H, R)]
    o = Outline(profile, mode="revolve")
    assert abs(o.volume() - (1.0 / 3.0) * math.pi * R * R * H) < 1e-9


def test_extruded_outline_holds_an_organic_shape():
    # a non-convex leaf-ish polygon (a point at each end, bulge in the middle):
    # primitives can't express it, Outline can, and it integrates to real matter.
    leaf = [(-0.05, 0.0), (-0.01, 0.012), (0.0, 0.013), (0.02, 0.010),
            (0.05, 0.0), (0.02, -0.010), (0.0, -0.013), (-0.01, -0.012)]
    o = Outline(leaf, mode="extrude", thickness=0.0006)
    assert o.volume() > 0.0
    assert o.surface_distance(0.0, 0.0, 0.0) < 0.0             # mid-leaf is solid
    assert o.surface_distance(0.0, 0.05, 0.0) > 0.0            # outside the rim
    assert o.is_volumetric                                     # a real 3D body
