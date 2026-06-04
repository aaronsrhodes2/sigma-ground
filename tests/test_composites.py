"""Deckard multi-part composites + extended primitives (deterministic, offline).

Disjoint composites (hammer): exact analytic mass with a tight per-part
self-check, mode='disjoint'. Overlapping parts: the SDF integrator's union mass
is canonical (not the over-counting Σρ·V), mode='overlapping'. Torus & ellipsoid
match their closed-form volumes. Plus the researcher emitting a multi-part spec.
"""
import json
import math

from sigma_ground.deckard import compile
from sigma_ground.deckard.schema import ConstructSpec, Part, Fact, emit_markdown, parse_markdown
from sigma_ground.deckard.sources import density_of
from sigma_ground.deckard.researcher import research_spec


def _part(name, shape, dims, material, center=(0.0, 0.0, 0.0)):
    return Part(name, shape, {k: Fact(v, "t", "", 0.5) for k, v in dims.items()},
                material, density_of(material) or Fact(700.0, "estimated"), center)


def test_disjoint_composite_uses_exact_analytic_mass():
    hammer = ConstructSpec(name="hammer", kind="composite", parts=[
        _part("handle", "cylinder", {"radius_m": 0.012, "height_m": 0.30}, "oak", (0, 0, 0.15)),
        _part("head", "box", {"x_m": 0.10, "y_m": 0.03, "z_m": 0.03}, "steel", (0, 0, 0.315))])
    c = compile(hammer, resolution=64)
    assert c.validation["mode"] == "disjoint"
    assert c.validation["passed"] and c.validation["mass_residual"] < 0.02
    m_hand = density_of("oak").value * math.pi * 0.012 ** 2 * 0.30
    m_head = density_of("steel").value * 0.10 * 0.03 * 0.03
    assert abs(c.mass_kg - (m_hand + m_head)) < 1e-9       # exact analytic sum


def test_overlapping_parts_use_integrator_union_mass():
    blob = ConstructSpec(name="blob", kind="composite", parts=[
        _part("a", "sphere", {"radius_m": 0.05}, "steel", (0, 0, 0)),
        _part("b", "sphere", {"radius_m": 0.05}, "steel", (0.03, 0, 0))])
    c = compile(blob, resolution=48)
    assert c.validation["mode"] == "overlapping"
    assert c.mass_kg < c.validation["mass_analytic_sum_kg"]   # not double-counted
    assert c.validation["passed"]                            # 2-resolution stable
    # union ≈ 2·sphere − lens(overlap), within a few %
    rho, r, d = density_of("steel").value, 0.05, 0.03
    lens = math.pi * (4 * r + d) * (2 * r - d) ** 2 / 12.0
    union = 2 * (4 / 3 * math.pi * r ** 3) - lens
    assert abs(c.mass_kg - rho * union) / (rho * union) < 0.03


def test_torus_and_ellipsoid_match_closed_form():
    ring = ConstructSpec(name="ring", kind="composite", parts=[
        _part("r", "torus", {"major_radius_m": 0.02, "minor_radius_m": 0.005}, "gold")])
    c = compile(ring, resolution=48)
    assert abs(c.mass_kg - density_of("gold").value * 2 * math.pi ** 2 * 0.02 * 0.005 ** 2) < 1e-9
    assert c.validation["passed"]
    egg = ConstructSpec(name="egg", kind="composite", parts=[
        _part("e", "ellipsoid", {"rx_m": 0.02, "ry_m": 0.02, "rz_m": 0.03}, "water")])
    c2 = compile(egg, resolution=48)
    assert abs(c2.mass_kg - density_of("water").value * 4 / 3 * math.pi * 0.02 * 0.02 * 0.03) < 1e-9


def test_researcher_emits_multipart_composite():
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "handle", "shape": "cylinder", "dims": {"radius_m": 0.012, "height_m": 0.30},
         "material": "oak", "center_m": [0, 0, 0.15]},
        {"name": "head", "shape": "box", "dims": {"x_m": 0.10, "y_m": 0.03, "z_m": 0.03},
         "material": "steel", "center_m": [0, 0, 0.315]}]})
    spec = research_spec("hammer", ask=lambda n: payload, model="stub")
    assert spec is not None and len(spec.parts) == 2
    assert all(all(f.estimated for f in p.dims.values()) for p in spec.parts)   # dims flagged
    assert not any(p.density.estimated for p in spec.parts)                     # grounded
    assert compile(spec, resolution=56).validation["passed"]


def test_rotation_reorients_a_part_and_preserves_mass():
    spec = ConstructSpec(name="rod-x", kind="composite", parts=[
        Part("rod", "cylinder", {"radius_m": Fact(0.01), "height_m": Fact(0.20)},
             "steel", density_of("steel"), (0, 0, 0), (0, 90, 0))])
    c = compile(spec, resolution=56)
    assert abs(c.mass_kg - density_of("steel").value * math.pi * 0.01 ** 2 * 0.20) < 1e-9
    assert c.material_at(0.09, 0.0, 0.0) == "rod"     # along the new (x) axis
    assert c.material_at(0.0, 0.0, 0.09) is None      # the old (z) axis is now empty


def test_euler_round_trips_through_markdown():
    spec = ConstructSpec(name="p", kind="composite", parts=[
        Part("c", "cylinder", {"radius_m": Fact(0.01), "height_m": Fact(0.10)},
             "steel", density_of("steel"), (0, 0, 0), (90, 0, 0))])
    assert tuple(parse_markdown(emit_markdown(spec)).parts[0].euler_deg) == (90.0, 0.0, 0.0)


def test_hollow_part_carves_a_cavity():
    pipe = ConstructSpec(name="pipe", kind="composite", parts=[
        Part("wall", "cylinder", {"radius_m": Fact(0.02), "height_m": Fact(0.30)},
             "steel", density_of("steel")),
        Part("bore", "cylinder", {"radius_m": Fact(0.016), "height_m": Fact(0.32)},
             "air", Fact(0.0), op="subtract")])
    c = compile(pipe, resolution=64)
    assert c.validation["mode"] == "hollow" and c.validation["passed"]
    rho = density_of("steel").value
    wall = rho * math.pi * (0.02 ** 2 - 0.016 ** 2) * 0.30
    solid = rho * math.pi * 0.02 ** 2 * 0.30
    assert c.mass_kg < 0.5 * solid                      # hollow, not the full solid
    assert abs(c.mass_kg - wall) / wall < 0.03          # ≈ the wall mass
    assert c.material_at(0.018, 0.0, 0.0) == "wall"     # steel wall
    assert c.density_at(0.0, 0.0, 0.0) == 0.0           # empty bore
    assert c.sdf(0.0, 0.0, 0.0) > 0.0                   # bore is carved (outside the solid)
