"""Deckard's compiled masses for everyday things are the right order of magnitude.

The holodeck audit (misc/HOLODECK_PHYSICS_AUDIT.md) caught Deckard compiling a refrigerator at
7750 kg (a 1 m3 block of steel), a microwave at 404 kg, a chair at 123 kg, a bed at 856 kg, a floor
lamp at 179 kg, a wardrobe at 0.6 kg (a coffee cup), a bookcase at 4 kg and a potted plant at
1457 kg. The bands below are WIDE plausibility bands, not references: the models are bounding-box
exemplars with estimated wall thickness (research._STOCK_T_M), so a factor of 2-3 either way is
expected; what must not happen is the factor of 10-100 the old numbers were out by. The "real"
column is a rough general-knowledge range, not a Mentat result.
"""
import importlib
import json

import pytest

from sigma_ground.deckard import compile, research
from sigma_ground.deckard.construct import _shape_from
from sigma_ground.deckard.sources import exemplar as _exemplar

R = importlib.import_module("sigma_ground.deckard.research")

# name -> (low kg, high kg, real-world rough range, for the reader)
_BANDS = {
    "a refrigerator":    (40.0, 250.0, "70-120"),
    "a fridge":          (40.0, 250.0, "70-120"),
    "a microwave":       (5.0, 60.0, "12-16"),
    "a chair":           (3.0, 80.0, "5-8"),
    "an office chair":   (3.0, 80.0, "10-20"),
    "a bed":             (30.0, 250.0, "60-110"),
    "a bunk bed":        (30.0, 250.0, "100+"),
    "a floor lamp":      (1.0, 40.0, "3-6"),
    "a desk lamp":       (0.3, 20.0, "1-3"),
    "a potted plant":    (2.0, 120.0, "5-15"),
    "a clothes cupboard": (30.0, 200.0, "60-100"),
    "a wardrobe":        (30.0, 250.0, "60-100"),
    "a closet":          (30.0, 250.0, "60-100"),
    "a bookcase":        (15.0, 120.0, "30-50"),
    "a bookshelf":       (15.0, 120.0, "30-50"),
    "a tv":              (5.0, 40.0, "10-25"),
    "a pot belly stove": (20.0, 1500.0, "50-150"),     # resolves to the flowerpot layout via the alias 'pot'
}
_MAY_BE_UNIDENTIFIED = {"a pot belly stove"}             # its identification is a separate question (alias on a modifier)


@pytest.fixture(scope="module")
def compiled():
    cache = {}

    def get(name):
        if name not in cache:
            spec = research(name, allow_llm=False)
            cache[name] = (spec, compile(spec, resolution=48))
        return cache[name]
    return get


@pytest.mark.parametrize("name", sorted(_BANDS))
def test_everyday_masses_are_plausible(compiled, name):
    lo, hi, _real = _BANDS[name]
    spec, c = compiled(name)
    if name not in _MAY_BE_UNIDENTIFIED:
        assert spec.identified, f"{name}: should be grounded by a real exemplar, not the generic vessel"
    assert spec.kind == "composite", f"{name}: the generic vessel is a layered_vessel of 0.6 kg"
    assert lo < c.mass_kg < hi, f"{name}: {c.mass_kg:.1f} kg is outside the plausibility band {lo}-{hi} kg"


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
    # (name, share of the material's full density the envelope may keep): a steel fridge 5%, a glass
    # microwave 10%, a dishwasher (no ShapeNetSem material: wood boards by default) 25%
    for q, share in (("a refrigerator", 0.05), ("a microwave", 0.10), ("a dishwasher", 0.25)):
        spec = R._exemplar_spec(q)
        big = max((p for p in spec.parts if p.op == "add"),
                  key=lambda p: _shape_from(p).volume())
        assert big.density.estimated, q
        assert big.density.value < share * R._density_fact(big.material).value, q
