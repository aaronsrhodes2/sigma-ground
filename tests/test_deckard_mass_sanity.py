"""Deckard's compiled masses for everyday things agree with the repo's own SOURCED mass ranges.

The holodeck audit (misc/HOLODECK_PHYSICS_AUDIT.md) caught Deckard compiling a refrigerator at
7750 kg (a 1 m3 block of steel), a microwave at 404 kg, a chair at 123 kg, a bed at 856 kg, a floor
lamp at 179 kg, a wardrobe at 0.6 kg (a coffee cup), a bookcase at 4 kg and a potted plant at
1457 kg. The first repair pinned these with bands 5-25x wider than the truth (a 50 kg chair passed).
The bands below are now DERIVED from stage/gmod/holodeck/data/fidelity_specs.json, whose families carry
mass ranges read from product listings and spec sheets (`low_kg`, `high_kg`, the evidence is in that
file); a case passes when Deckard lands within ``_SLACK`` of the sourced range. ``_SLACK`` = 1.5x on
each end because a PartNet exemplar is one real model at a ShapeNetSem median size, not the listed
product (the 'chair' exemplar has arms and a 0.78 x 0.73 m footprint, the listed side chair 0.45 x 0.52).

Cases that cannot honestly agree are `xfail(strict=True)` with the reason, not blessed with a band:
a test that starts passing fails loudly so the reason is removed.

The grid-resolution half of the same audit finding is `test_the_mass_does_not_depend_on_the_grid`:
a thin-board part is integrated on its own grid, so 32/48/64/96 cells give the same kilograms.
"""
import importlib
import json
import pathlib

import pytest

from sigma_ground.deckard import compile, research
from sigma_ground.deckard.construct import _shape_from
from sigma_ground.deckard.sources import exemplar as _exemplar

R = importlib.import_module("sigma_ground.deckard.research")

_SPECS = (pathlib.Path(__file__).resolve().parents[1] / "stage" / "gmod" / "holodeck" / "data"
          / "fidelity_specs.json")
_SLACK = 1.5


def _families():
    try:
        return json.loads(_SPECS.read_text(encoding="utf-8"))["families"]
    except Exception:
        return {}


_FAM = _families()
pytestmark = pytest.mark.skipif(not _FAM, reason="stage/gmod/holodeck/data/fidelity_specs.json not present")


def _range(*families):
    """(low, high) kg over the sourced families; several families SUM when an object is made of
    several (a bed = a frame + a mattress)."""
    return (sum(_FAM[f]["low_kg"] for f in families), sum(_FAM[f]["high_kg"] for f in families))


# name -> sourced families whose mass ranges are added
_SOURCED = {
    "a refrigerator":     ("fridge_cabinet",),
    "a fridge":           ("fridge_cabinet",),
    "a microwave":        ("microwave",),
    "a chair":            ("chair",),
    "an office chair":    ("chair",),                  # the same PartNet chair exemplar; see the office_chair note below
    "a bed":              ("bed_frame", "mattress_single"),
    "a floor lamp":       ("floor_lamp",),
    "a desk lamp":        ("desk_lamp",),
    "a clothes cupboard": ("wardrobe",),
    "a wardrobe":         ("wardrobe",),
    "a closet":           ("wardrobe",),
    "a bookcase":         ("bookshelf",),
    "a bookshelf":        ("bookshelf",),
    "a trash can":        ("trash_can_plastic",),
    "a desk":             ("desk",),
}

# name -> (families, why Deckard cannot agree with the sourced range)
_XFAIL = {
    "a bunk bed": (("bunk_steel",),
                   "the sourced range is an IKEA steel bunk FRAME (39-45 kg, no mattresses); Deckard has no bunk "
                   "exemplar and builds a wooden bed with a foam mattress, and the ShapeNetSem axis mapping "
                   "(research._exemplar_spec sorts the longest edge to height) makes that bed 2.3 m tall: "
                   "mattress 0.5 m thick, headboard 2.2 m. Fix with the axis change, which moves pinned scenes"),
    "a potted plant": (("potted_plant",),
                       "the sourced plant is a 24 in plant in a 10 in nursery pot (3.2-4.8 kg, mostly soil); Deckard "
                       "scales the pot to the ShapeNetSem 'plant' median (1.1 m, a floor plant: a 0.41 x 0.36 x 0.46 m "
                       "ceramic pot, 14 kg) and has no soil"),
    "a tv": (("tv_crt",),
             "the only sourced TV in the repo is a 27 in CRT set (21-47 kg); the 'tv' exemplar is a flat-panel "
             "display (5.5 kg at 0.6 m), a different object. No sourced flat-panel range exists to compare with"),
}


@pytest.fixture(scope="module")
def compiled():
    cache = {}

    def get(name, resolution=48):
        if (name, resolution) not in cache:
            spec = research(name, allow_llm=False)
            cache[(name, resolution)] = (spec, compile(spec, resolution=resolution))
        return cache[(name, resolution)]
    return get


@pytest.mark.parametrize("name", sorted(_SOURCED))
def test_everyday_masses_agree_with_the_sourced_ranges(compiled, name):
    lo, hi = _range(*_SOURCED[name])
    spec, c = compiled(name)
    assert spec.identified, f"{name}: should be grounded by a real exemplar, not the generic vessel"
    assert spec.kind == "composite", f"{name}: the generic vessel is a layered_vessel of 0.6 kg"
    assert lo / _SLACK <= c.mass_kg <= hi * _SLACK, (
        f"{name}: {c.mass_kg:.1f} kg is outside the sourced {lo:g}-{hi:g} kg (x{_SLACK:g} slack)")
    assert c.validation["passed"], f"{name}: self-check failed ({c.validation['mass_residual']:.1%})"


@pytest.mark.parametrize("name", sorted(_XFAIL))
def test_known_disagreements_with_the_sourced_ranges(compiled, name):
    families, why = _XFAIL[name]
    lo, hi = _range(*families)
    _spec, c = compiled(name)
    if not (lo / _SLACK <= c.mass_kg <= hi * _SLACK):
        pytest.xfail(f"{c.mass_kg:.1f} kg vs sourced {lo:g}-{hi:g} kg: {why}")
    pytest.fail(f"{name} now agrees ({c.mass_kg:.1f} kg in {lo:g}-{hi:g}): move it to _SOURCED and drop the reason")


def test_the_bands_are_the_sourced_ones_not_the_old_wide_ones():
    """A 50 kg chair passed the old 3-80 kg band; the sourced 2.7-12 kg range (x1.5) refuses it."""
    lo, hi = _range("chair")
    assert (lo, hi) == (2.7, 12)
    assert hi * _SLACK < 20.0
    assert _range("floor_lamp")[1] * _SLACK < 16.0 and _range("desk_lamp")[1] * _SLACK < 8.0
    assert _range("bed_frame", "mattress_single")[1] * _SLACK < 140.0


@pytest.mark.parametrize("name", ["a clothes cupboard", "a wardrobe", "a bookcase", "a tv", "a desk lamp"])
def test_sizes_are_those_of_the_named_thing(compiled, name):
    _spec, c = compiled(name)
    ext = sorted(abs(b[1] - b[0]) for b in c.bbox)
    assert ext[2] > 0.4, f"{name}: {ext[2]:.2f} m is smaller than a mug"           # the cup was 0.095 m
    assert ext[2] < 2.2                                                              # nothing here is taller than a door


def test_a_desk_lamp_is_not_a_desk_sized_object(compiled):
    _spec, c = compiled("a desk lamp")
    assert max(abs(b[1] - b[0]) for b in c.bbox) < 1.0                              # was 1.33 m


def test_no_non_cup_comes_back_as_the_generic_cup():
    for name in ("a clothes cupboard", "a wardrobe", "a closet", "a bookcase", "a bookshelf", "a shelf", "a tv",
                 "a television", "a plant", "a refrigerator", "a lamp"):
        spec = research(name, allow_llm=False)
        assert spec.kind != "layered_vessel", name
        assert "coffee" not in " ".join(s.get("name", "") for s in spec.sources).lower(), name


# ── the grid-resolution half: the mass must not depend on how finely it is integrated ───────────
_CONVERGED = ["a bed", "a bunk bed", "a bookcase", "a table", "a monitor", "a microwave", "a wardrobe",
              "a refrigerator", "a chair", "a floor lamp", "a desk lamp", "a potted plant"]
_RESOLUTIONS = (32, 48, 64, 96)
_CONVERGENCE = 0.03                       # max relative spread of the kilograms over 32/48/64/96 cells


@pytest.mark.parametrize("name", _CONVERGED)
def test_the_mass_does_not_depend_on_the_grid(name):
    """Before: 'a bed' was 76 / 71 / 151 / 138 kg at 32 / 48 / 64 / 96 cells (exact 123), validation failed at every
    one, because a 16 mm board is 0 or 1 cells thick on a global grid. Now every part is integrated on its own
    grid with >= 6 cells across its thin axis (construct._integrate_items): the spread is under 3%."""
    spec = research(name, allow_llm=False)
    got = {r: compile(spec, resolution=r) for r in _RESOLUTIONS}
    masses = [c.mass_kg for c in got.values()]
    spread = (max(masses) - min(masses)) / min(masses)
    assert spread < _CONVERGENCE, f"{name}: {dict(zip(_RESOLUTIONS, (round(m, 2) for m in masses)))} ({spread:.1%})"
    for r, c in got.items():
        assert c.validation["passed"], f"{name} at {r}: self-check failed ({c.validation['mass_residual']:.1%})"


def test_the_default_resolution_passes_the_self_check_and_respects_the_analytic_bound():
    """At the library default (what audit and consumers see) the mass is at most the sum of rho*V of the parts."""
    for name in ("a bed", "a bookcase", "a table", "a monitor"):
        spec = research(name, allow_llm=False)
        c = compile(spec)                                                    # default resolution
        ub = sum(_shape_from(p).volume() * p.density.value for p in spec.parts if p.op == "add")
        assert c.validation["passed"], name
        assert c.mass_kg <= ub * 1.005, f"{name}: {c.mass_kg:.1f} kg > the analytic upper bound {ub:.1f} kg"


def test_an_overlap_is_counted_once_and_the_last_part_wins():
    """Two boxes sharing a quarter of their volume: mass = rho_a*(V_a - V_overlap) + rho_b*V_b, at any grid."""
    from sigma_ground.deckard.schema import ConstructSpec, Fact, Part
    a = Part("a", "box", {"x_m": Fact(1.0), "y_m": Fact(1.0), "z_m": Fact(0.02)}, "wood", Fact(700.0), (0.0, 0.0, 0.0))
    b = Part("b", "box", {"x_m": Fact(0.5), "y_m": Fact(1.0), "z_m": Fact(0.5)}, "steel", Fact(7850.0), (0.25, 0.0, 0.0))
    spec = ConstructSpec(name="overlap", kind="composite", identified=True, parts=[a, b])
    exact = 700.0 * (1.0 - 0.5 * 1.0 * 0.02) * 0.02 / 0.02 * 0.02 + 7850.0 * 0.5 * 1.0 * 0.5     # = 700*(0.02-0.01)+7850*0.25
    exact = 700.0 * (1.0 * 1.0 * 0.02 - 0.5 * 1.0 * 0.02) + 7850.0 * 0.5 * 1.0 * 0.5
    for r in _RESOLUTIONS:
        assert compile(spec, resolution=r).mass_kg == pytest.approx(exact, rel=0.02)


def test_a_sixteen_mm_board_in_a_two_metre_object_is_exact_at_every_grid():
    """The failure itself: a 16 mm board in a 2 m object is 0 or 1 cells thick on a global grid."""
    from sigma_ground.deckard.schema import ConstructSpec, Fact, Part
    board = Part("board", "box", {"x_m": Fact(1.9), "y_m": Fact(0.016), "z_m": Fact(2.2)}, "wood", Fact(700.0), (0.0, 0.0, 1.1))
    post = Part("post", "box", {"x_m": Fact(0.05), "y_m": Fact(1.2), "z_m": Fact(0.05)}, "wood", Fact(700.0), (0.9, 0.6, 0.1))
    spec = ConstructSpec(name="thin", kind="composite", identified=True, parts=[board, post])
    exact = 700.0 * (1.9 * 0.016 * 2.2 + 0.05 * 1.2 * 0.05)
    for r in (16, 32, 48, 64, 96, 128):
        c = compile(spec, resolution=r)
        assert c.mass_kg == pytest.approx(exact, rel=0.01), r
        assert c.validation["passed"], r


# ── every shipped exemplar, every variant ───────────────────────────────────────────────────
def _variants():
    """Every shipped layout, except the thin steel tools (scissors, knife: dense plate stock, really)
    and the vessels (a cavity is a subtract part, so the analytic sum below would not apply)."""
    for p in sorted(_exemplar._DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["category"] in ("scissors", "knife") or R._is_container("a " + d["category"]):
            continue
        yield pytest.param(d, id=p.stem)


@pytest.mark.parametrize("doc", _variants())
def test_no_enclosure_or_furniture_is_denser_than_wood_on_average(doc, monkeypatch):
    """Mean density over the bounding box (analytic sum of the parts, an upper bound) of every
    shipped layout. Before the occupancy rule, compiled: refrigerator 6250 kg/m3, faucet 2690,
    microwave 1940, dishwasher 590, bag 660."""
    cat = doc["category"]
    layout, anno = doc["parts"], str(doc["source_anno_id"])
    monkeypatch.setattr(_exemplar, "exemplar_of", lambda name, exclude=None, hint=None, _aliased=False:
                        (layout, "src", "lic", anno))
    spec = R._exemplar_spec("a " + cat)
    mass, lo, hi = 0.0, [1e9] * 3, [-1e9] * 3
    for p in spec.parts:
        e = R._part_extent(p)
        if p.op == "add":
            mass += _shape_from(p).volume() * p.density.value
        for k in range(3):
            lo[k] = min(lo[k], p.center_m[k] - e[k] / 2.0)
            hi[k] = max(hi[k], p.center_m[k] + e[k] / 2.0)
    bbox_m3 = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])
    assert mass / bbox_m3 < 600.0, f"{cat} ({doc['source_anno_id']}): {mass / bbox_m3:.0f} kg/m3 over its bounding box"


def test_the_envelope_leaves_of_appliances_are_shells():
    """A leaf that spans most of the object (the refrigerator frame is 86% of its bbox, the microwave
    frame 91%) is an enclosure's envelope, not a block of its material."""
    # (name, share of the material's full density the envelope may keep): a steel fridge 5%, a sheet-metal
    # microwave 10%, a dishwasher (no ShapeNetSem material: wood boards by default) 25%
    for q, share in (("a refrigerator", 0.05), ("a microwave", 0.10), ("a dishwasher", 0.25)):
        spec = R._exemplar_spec(q)
        big = max((p for p in spec.parts if p.op == "add"),
                  key=lambda p: _shape_from(p).volume())
        assert big.density.estimated, q
        assert big.density.value < share * R._density_fact(big.material).value, q
