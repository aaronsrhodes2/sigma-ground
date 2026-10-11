"""Occupancy of a bounding-box part: the bulk density of a PartNet leaf.

A PartNet leaf is only the axis-aligned box around a surface mesh. Giving it its material's FULL
density made a refrigerator 1 m3 of steel (7750 kg), a chair 123 kg, a floor lamp 179 kg. A part
is stock (board, sheet, casting) of limited wall thickness T, so it holds no more than a thin-wall
shell of its box; a plank or rod stays solid. Every T is an ESTIMATE and is stored as one.
Offline: no compile except where noted.
"""
import importlib

import pytest

R = importlib.import_module("sigma_ground.deckard.research")     # the module (the package re-exports the function)
from sigma_ground.deckard.researcher import _is_solid_class
from sigma_ground.deckard.schema import Fact, Part


def _box(name, x, y, z, material="steel", rho=7850.0, op="add", cited=True):
    f = Fact(rho, "surface.MATERIALS", "", 0.9) if cited else Fact(rho)
    return Part(name, "box", {"x_m": Fact(x), "y_m": Fact(y), "z_m": Fact(z)}, material, f, op=op)


# ── the arithmetic ──────────────────────────────────────────────────────────────────────────
def test_a_plank_is_solid():
    assert R._occupancy((0.9, 0.3, 0.018), 0.018) == 1.0          # some side <= 2T
    assert R._occupancy((0.9, 0.3, 0.036), 0.018) == 1.0


def test_a_slender_post_is_nearly_solid():
    assert R._occupancy((0.04, 0.04, 0.45), 0.018) >= 0.98


def test_a_refrigerator_frame_is_a_thin_shell():
    occ = R._occupancy((0.82, 0.69, 1.59), R._SHEET_METAL_T_M)
    assert 0.005 < occ < 0.02                                      # ~1% of the box is steel: 78 kg/m3, not 7850


def test_occupancy_grows_with_wall_thickness_and_is_a_fraction():
    ext = (0.5, 0.4, 0.3)
    last = 0.0
    for T in (0.0005, 0.001, 0.0015, 0.0025, 0.005, 0.018, 0.05, 1.0):
        occ = R._occupancy(ext, T)
        assert last <= occ <= 1.0
        last = occ
    assert R._occupancy(ext, 1.0) == 1.0


def test_it_is_the_exact_shell_fraction_of_a_cylinder_and_a_sphere():
    import math
    r, h, t = 0.1, 0.4, 0.004
    shell = math.pi * (r ** 2 * h - (r - t) ** 2 * (h - 2 * t))
    assert R._occupancy((2 * r, 2 * r, h), t) == pytest.approx(shell / (math.pi * r ** 2 * h), rel=1e-9)
    sph = 4 / 3 * math.pi * (r ** 3 - (r - t) ** 3)
    assert R._occupancy((2 * r,) * 3, t) == pytest.approx(sph / (4 / 3 * math.pi * r ** 3), rel=1e-9)


def test_stock_thickness_is_by_material_class():
    assert R._stock_thickness("wood") == R._stock_thickness("plywood") == R._BOARD_T_M
    assert R._stock_thickness("steel") == R._stock_thickness("stainless steel") == R._SHEET_METAL_T_M
    assert R._stock_thickness("metal") == R._stock_thickness("aluminum") == R._SHEET_METAL_T_M
    assert R._stock_thickness("cast iron") == R._CAST_METAL_T_M          # cast beats the 'iron' sheet rule
    assert R._stock_thickness("brass") == R._stock_thickness("bronze") == R._CAST_METAL_T_M
    assert R._stock_thickness("glass") == R._GLASS_T_M
    assert R._stock_thickness("ceramic") == R._CERAMIC_T_M
    assert R._stock_thickness("plastic") == R._PLASTIC_T_M
    assert R._stock_thickness("foliage") == R._stock_thickness(None) == R._STOCK_T_DEFAULT_M
    # the conventions are named constants, in a plausible range
    assert 0.001 <= R._SHEET_METAL_T_M < R._CAST_METAL_T_M < R._BOARD_T_M <= 0.025
    assert 20.0 <= R._UPHOLSTERY_KG_M3 <= 100.0


# ── applying it to parts ────────────────────────────────────────────────────────────────────
def test_solid_stock_keeps_its_cited_density():
    plank = _box("shelf", 0.9, 0.3, 0.018, "wood", 700.0)
    post = _box("leg", 0.04, 0.04, 0.45, "wood", 700.0)             # 99% occupied: solid for practical purposes
    assert R._apply_occupancy([plank, post]) == 0
    for p in (plank, post):
        assert p.density.value == 700.0 and not p.density.estimated


def test_a_box_part_becomes_an_estimated_bulk_density():
    frame = _box("frame", 0.82, 0.69, 1.59)
    n = R._apply_occupancy([frame])
    assert n == 1 and frame.density.estimated
    assert frame.density.value == pytest.approx(7850.0 * R._occupancy((0.82, 0.69, 1.59), R._SHEET_METAL_T_M), rel=1e-3)
    assert frame.material == "steel"                                  # the label stays: only the mass model changed


def test_soft_parts_take_the_upholstery_bulk_not_the_fibre_density():
    seat = _box("seat_single_surface", 0.5, 0.5, 0.05, "fabric", 1540.0)
    assert R._apply_occupancy([seat]) == 1
    assert seat.density.value == R._UPHOLSTERY_KG_M3 and seat.density.estimated and seat.material == "fabric"
    mattress = _box("mattress_0", 1.8, 1.1, 0.5, "wood", 700.0)       # a composition with no upholstery material
    assert R._apply_occupancy([mattress]) == 1
    assert mattress.density.value == R._UPHOLSTERY_KG_M3 and mattress.material == "foam"
    foam = _box("cushion", 0.5, 0.5, 0.1, "foam", 40.0)               # already lighter than the cap: untouched, still cited
    assert R._apply_occupancy([foam]) == 0 and foam.density.value == 40.0 and not foam.density.estimated


def test_subtract_skip_and_unmodelled_shapes_are_untouched():
    cav = _box("interior", 0.4, 0.4, 0.4, "air", 1.2, op="subtract")
    body = _box("body", 0.5, 0.5, 0.5, "ceramic", 2400.0)
    disc = Part("lens", "outline", {}, "glass", Fact(2500.0, "x", "", 0.9))
    ring = Part("ring", "torus", {"major_radius_m": Fact(0.1), "minor_radius_m": Fact(0.01)}, "steel", Fact(7850.0, "x", "", 0.9))
    bad = Part("odd", "box", {"x_m": Fact(0.5)}, "steel", Fact(7850.0, "x", "", 0.9))          # missing dims
    assert R._apply_occupancy([cav, body, disc, ring, bad], skip={"body"}) == 0
    assert body.density.value == 2400.0 and not body.density.estimated
    assert disc.density.value == 2500.0 and ring.density.value == 7850.0 and bad.density.value == 7850.0


def test_a_solid_brass_fixture_part_does_not_collapse_to_sheet_metal():
    """Decision for cast solids (the faucet class): a compact brass body is a casting, whose wall is
    >= 3 mm, not 1.5 mm sheet. The same box as sheet steel would lose 4/5 of its mass."""
    spout = lambda mat, rho: Part("spout", "cylinder", {"radius_m": Fact(0.015), "height_m": Fact(0.15)}, mat,
                                  Fact(rho, "CRC", "", 0.9))
    brass, steel = spout("brass", 8500.0), spout("steel", 7850.0)
    R._apply_occupancy([brass, steel])
    assert brass.density.value >= 0.5 * 8500.0                         # 59% of solid brass (a bored casting)
    assert steel.density.value < 0.25 * 7850.0                         # 21%: a 1.5 mm sheet tube
    # and it never goes below what the minimum casting wall allows
    floor = 8500.0 * R._occupancy((0.03, 0.03, 0.15), R._CAST_METAL_T_M)
    assert brass.density.value == pytest.approx(floor, rel=1e-3)


# ── the solid-class guard ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["an anvil", "a hammer", "a claw hammer", "a dumbbell", "two dumbbells",
                                  "a knife", "a key", "a ball", "a brass bar", "a die", "dice"])
def test_cast_solid_kinds_are_solid_class(name):
    assert _is_solid_class(name)


@pytest.mark.parametrize("name", ["a bar fridge", "a stone vase", "a hammer drill", "a refrigerator",
                                  "a keyboard", "a ball chair", "a block table", "", "a chair"])
def test_head_noun_not_modifier_decides_solid_class(name):
    assert not _is_solid_class(name)


def test_a_knife_keeps_its_cited_densities():
    spec = R._exemplar_spec("a knife")
    assert spec is not None
    assert all(not p.density.estimated for p in spec.parts)
    assert not any("modeled bulk density" in s.get("name", "") for s in spec.sources)


def test_the_scaffold_shells_a_toaster_but_not_a_dumbbell(monkeypatch):
    """The enriched scaffold gets the same rule, except for cast-solid kinds."""
    from sigma_ground.deckard import sources
    census = [{"name": "body", "shape": "box", "count": 1, "size_frac": [1.0, 1.0, 1.0], "z_frac": 0.0, "r_frac": 0.0}]
    monkeypatch.setattr(sources, "composition_of", lambda name, *a, **k: (census, "census", "lic"))
    monkeypatch.setattr(sources.shapenetsem, "dims_of", lambda name: ([0.3, 0.3, 0.3], 9, "S", "L"))
    monkeypatch.setattr(sources.shapenetsem, "materials_of", lambda name: ({"steel": 1.0}, "S", "L"))
    toaster = R._scaffold_from_composition("a toaster")
    dumbbell = R._scaffold_from_composition("a dumbbell")
    assert toaster.parts[0].density.estimated and toaster.parts[0].density.value < 0.1 * 7850.0
    assert any("modeled bulk density" in s["name"] for s in toaster.sources)
    assert not dumbbell.parts[0].density.estimated and dumbbell.parts[0].density.value > 7000.0
    assert not any("modeled bulk density" in s["name"] for s in dumbbell.sources)
    assert toaster.identified is False and dumbbell.identified is False      # still flagged placeholders


# ── what it does to real exemplars (spec level: no compile) ─────────────────────────────────
def test_a_refrigerator_is_not_a_block_of_steel():
    spec = R._exemplar_spec("a refrigerator")
    frame = next(p for p in spec.parts if p.name.startswith("frame"))
    assert frame.density.estimated and frame.density.value < 0.05 * 7850.0          # was 7850
    assert any("modeled bulk density" in s["name"] and "ESTIMATED" in s["name"] for s in spec.sources)
    full = sum(R._part_extent(p)[0] * R._part_extent(p)[1] * R._part_extent(p)[2] * 7850.0 for p in spec.parts)
    assert full > 4000.0                                                             # the old (buggy) total, for scale


def test_a_chair_is_not_123_kg():
    from sigma_ground.deckard.construct import _shape_from
    spec = R._exemplar_spec("a chair")
    m = sum(_shape_from(p).volume() * p.density.value for p in spec.parts if p.op == "add")
    assert 3.0 < m < 90.0
    seat = next(p for p in spec.parts if p.name.startswith("seat_single"))
    assert seat.density.value <= R._UPHOLSTERY_KG_M3


def test_a_lamp_shade_is_cloth_not_glass():
    """A shade is fabric/paper: the role convention gives it the upholstery material when the category
    composition has one (ShapeNetSem lamps: 35% fabric), and the soft rule then caps it at the bulk."""
    assert R._role_material("shade_2", "glass", "fabric") == "fabric"
    assert R._role_material("shade_2", "glass", None) == "glass"              # no upholstery in the census: unchanged
    spec = R._exemplar_spec("a floor lamp")
    shade = next(p for p in spec.parts if p.name.startswith("shade"))
    assert shade.material == "fabric" and shade.density.value <= R._UPHOLSTERY_KG_M3
    body = next(p for p in spec.parts if p.name.startswith("body"))
    assert body.material == "glass" and body.density.estimated                # the pole keeps the frame material, shelled


def test_a_carved_container_body_is_not_hollowed_twice():
    for q in ("a bowl", "a mug", "a vase", "a bottle"):
        spec = R._exemplar_spec(q)
        carved = next(p for p in spec.parts if p.name == "interior")
        assert carved.op == "subtract"
        walls = [p for p in spec.parts if p.op == "add" and p.shape == "cylinder"
                 and p.center_m[:2] == carved.center_m[:2]
                 and p.dims["radius_m"].value > carved.dims["radius_m"].value]
        assert walls, q
        assert all(not p.density.estimated for p in walls), q      # the wall keeps its cited ceramic/glass density


def test_a_mass_cannot_come_from_the_generic_cup_for_a_non_cup():
    for q in ("a clothes cupboard", "a wardrobe", "a closet", "a bookcase", "a bookshelf", "a shelf",
              "a tv", "a television", "a plant", "a potted plant", "a houseplant"):
        spec = R.research(q, allow_llm=False)
        assert spec.kind == "composite" and spec.identified, q


# ── the densities this leans on, against Mentat (PLATINUM section 21) ───────────────────────
# Recorded 2026-10-10 from `tools/mentat_call.sh density '{"material": ...}'` (CRC Handbook values). The test does not
# call Mentat (offline); it pins the resolver to what Mentat said. Mentat has no steel, brass, bronze, ceramic, fabric
# or foam: those densities rest on the resolver's own cited tables.
_MENTAT_KG_M3 = {"glass": 2500.0, "aluminum": 2700.0, "iron": 7874.0, "copper": 8960.0,
                 "wood": 770.0,       # Mentat: wood_oak (white oak nominal); the resolver's 'wood' is 700
                 "plastic": 1100.0}   # Mentat: typical thermoplastic nominal; the resolver's is 1040


@pytest.mark.parametrize("material", sorted(_MENTAT_KG_M3))
def test_resolver_densities_agree_with_mentat(material):
    ours = R._density_fact(material).value
    assert ours == pytest.approx(_MENTAT_KG_M3[material], rel=0.12)
