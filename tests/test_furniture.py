"""Legged-furniture archetype: a chair is BUILT like a chair, not sampled.

Marginal part statistics can't express the joints (legs to the floor at the
seat's corners, a back standing at the rear edge). The archetype assembles the
structure; dimensions stay grounded (ShapeNetSem size, PartNet leg count). These
checks assert the STRUCTURE, offline.
"""
from sigma_ground.deckard import research, compile
from sigma_ground.deckard.research import _furniture_spec, _furniture_archetype


def _legs(spec):
    return [p for p in spec.parts if p.name.startswith("leg")]


def test_chair_has_seat_four_corner_legs_and_a_vertical_back():
    spec = _furniture_spec("a wooden chair")
    assert spec is not None and spec.identified
    names = {p.name for p in spec.parts}
    assert "seat" in names and "back" in names
    legs = _legs(spec)
    assert len(legs) == 4

    seat = next(p for p in spec.parts if p.name == "seat")
    back = next(p for p in spec.parts if p.name == "back")
    sz = seat.center_m[2]

    # legs sit BELOW the seat and reach the floor (z spans 0 → seat underside)
    for L in legs:
        half_h = L.dims["z_m"].value / 2
        assert abs(L.center_m[2] - half_h) < 2e-3           # bottom at z≈0 (floor)
        assert L.center_m[2] + half_h <= sz + 2e-3          # top no higher than the seat
    # four DISTINCT corners (not stacked at the centre)
    xy = {(round(L.center_m[0], 3), round(L.center_m[1], 3)) for L in legs}
    assert len(xy) == 4
    assert all(abs(x) > 0.05 and abs(y) > 0.05 for x, y in xy)

    # the back is a vertical panel: thin in depth (y), tall in z, ABOVE the seat,
    # at the rear (−y) edge
    assert back.dims["y_m"].value < back.dims["z_m"].value   # thin & tall
    assert back.center_m[2] > sz                             # rises above the seat
    assert back.center_m[1] < 0                              # at the rear edge

    # the geometry is exact (the SDF is what we assert); the mass integration of
    # thin slabs on a ~1 m object is resolution-noisy (same limitation as hollow
    # shells), so we check the shape, not the convergence gate.
    c = compile(spec, resolution=64)
    assert c.mass_kg > 0
    (x0, x1), (y0, y1), (z0, z1) = c.bbox
    assert (z1 - z0) > (y1 - y0)                  # a chair stands taller than it is deep
    assert z0 < 0.01                              # it sits on the floor


def test_table_has_a_top_and_legs_but_no_back():
    spec = _furniture_spec("a dining table")
    names = {p.name for p in spec.parts}
    assert "top" in names and "back" not in names
    assert len(_legs(spec)) == 4
    top = next(p for p in spec.parts if p.name == "top")
    # the top sits high (legs span nearly the full height beneath it)
    assert top.center_m[2] > 0.6 * compile(spec, resolution=48).bbox[2][1]


def test_dimensions_are_grounded_not_fabricated():
    spec = _furniture_spec("a wooden chair")
    # every dim is flagged a structural-archetype estimate (never "measured");
    # the overall SIZE is cited to ShapeNetSem in the sources
    assert all(f.estimated for p in spec.parts for f in p.dims.values())
    assert any("ShapeNetSem" in s.get("name", "") for s in spec.sources)
    assert any("archetype" in s.get("name", "") for s in spec.sources)


def test_non_furniture_is_not_captured():
    assert _furniture_spec("a hammer") is None
    assert _furniture_spec("a bookcase") is None            # shelves, not legged
    assert _furniture_spec("a coffee mug") is None
    assert _furniture_archetype("a chair")[1] is True       # has_back
    assert _furniture_archetype("a table")[1] is False      # no back


def test_research_routes_structured_furniture_before_the_llm():
    # deterministic: a chair name resolves to a structured, identified build with
    # no LLM — via the PartNet exemplar (real layout) or the archetype fallback
    spec = research("a wooden chair", allow_llm=False)
    assert spec.identified and len(spec.parts) >= 5     # seat + legs (+ back)
    # it stands on the floor, taller than deep
    (x0, x1), (y0, y1), (z0, z1) = compile(spec, resolution=48).bbox
    assert z0 < 0.02 and (z1 - z0) > 0.5 * (y1 - y0)
