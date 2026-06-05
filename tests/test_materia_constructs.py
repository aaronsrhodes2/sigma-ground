"""Materia pulls Deckard's catalogued shapes as simulation bodies (tier 3 → 2).

Materia is aware of the shapes Deckard has catalogued, can load one as validated
matter, and can drop it through air — mass and frontal area sourced from the
Construct, drag from Materia. Deterministic (catalog hit, no network).
"""
import math

from sigma_ground.materia import constructs as MC


def test_materia_is_aware_of_catalogued_shapes_including_feather():
    shapes = MC.available_shapes()
    assert "feather" in shapes                 # the shape we froze for it
    assert "coffee_cup" in shapes              # ...alongside the rest
    assert MC.has_shape("feather")
    assert not MC.has_shape("nonexistent_widget_xyz")


def test_materia_loads_the_feather_as_validated_matter():
    c = MC.load_shape("feather", resolution=56)
    assert c.validation["passed"]
    assert 0.0005 < c.mass_kg < 0.005          # ~1 g — a feather, not a stone
    area, axis = MC.broadside_area_m2(c, n=40)
    assert area > 5e-4                         # a real broadside silhouette (> 5 cm²)


def test_materia_drops_a_feather_and_it_flutters():
    r = MC.drop("feather", from_altitude_m=5.0, resolution=48)
    assert r.validation["passed"], r.validation["note"]     # drag energy balances
    vt = r.outputs["terminal_velocity_m_s"]
    assert 1.0 < vt < 6.0                      # drag-limited flutter, not free fall
    # a dense stone in vacuum would hit at √(2gh) ≈ 9.9 m/s from 5 m; the feather
    # is far slower because drag caps it at terminal
    free_fall = math.sqrt(2 * 9.80665 * 5.0)
    assert r.outputs["impact_speed_m_s"] < 0.5 * free_fall
