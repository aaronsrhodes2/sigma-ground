"""The Outline primitive — a 2D researched profile swept into 3D matter.

Known-answer checks against the analytic primitives: an extruded rectangle is a
box; a revolved constant-radius profile is a cylinder. Plus sign correctness of
the SDF (negative inside, positive outside), which is what Deckard's integrator
relies on for mass / material_at.
"""
import json
import math

from sigma_ground.kernel.outline import Outline
from sigma_ground.kernel.shapes import Box, Cylinder
from sigma_ground.deckard import compile
from sigma_ground.deckard.researcher import research_spec
from sigma_ground.deckard.sources import outline_of


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


def test_outline_of_loads_the_distilled_quickdraw_feather():
    got = outline_of("a feather")          # word-match -> the distilled feather medoid
    assert got is not None
    profile, source, lic = got
    assert len(profile) >= 12              # a real outline, many points
    assert "Quick" in source and "BY" in lic   # cited, CC BY 4.0


def test_distilled_outlines_all_form_valid_closed_matter():
    # every shipped Quick Draw outline loads and sweeps to real, closed matter.
    for noun in ("feather", "leaf", "key", "fish", "tree", "flower"):
        got = outline_of(noun)
        assert got is not None, noun
        prof = [(u * 0.1, v * 0.1) for u, v in got[0]]
        o = Outline(prof, mode="extrude", thickness=0.001)
        assert o.volume() > 0.0, noun
        assert o.surface_distance(0.0, 0.0, 0.0) < 0.0, noun     # centre is inside (closed)


def test_researcher_grounds_an_organic_outline_from_quickdraw():
    # the model declares a part shape:"outline"; Deckard supplies the RESEARCHED
    # Quick Draw profile, scales it to length_m, cites it — no hand-drawn shape,
    # no cloud LLM. (Stubbed ask + local material -> fully offline.)
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "vane", "shape": "outline",
         "dims": {"length_m": 0.12, "thickness_m": 0.0006}, "material": "oak"},
        {"name": "shaft", "shape": "cylinder",
         "dims": {"radius_m": 0.0009, "height_m": 0.12}, "material": "oak",
         "euler_deg": [0, 90, 0]}]})
    spec = research_spec("feather", ask=lambda n: payload, model="stub")
    assert spec is not None and spec.parts[0].shape == "outline"
    assert len(spec.parts[0].outline["profile"]) >= 12          # grounded Quick Draw outline
    assert any("Quick, Draw" in s.get("name", "") for s in spec.sources)   # cited CC BY
    c = compile(spec, resolution=72)
    assert c.validation["passed"] and c.mass_kg > 0.0
