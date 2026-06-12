"""PartNet shaping priors in the researcher — offline, stubbed qwen + sources.

The enriched composition table (size_frac/z_frac/r_frac/freq) corrects
proportions, reseats degenerate placement, and replicates under-counted parts —
while NEVER touching cited dims, attach-anchored parts, or the estimated flag
(the absolute size pass must stay active downstream). Unknown object names keep
every other pass quiet, so each behavior is observed in isolation.
"""
import json

from sigma_ground.deckard import compile
from sigma_ground.deckard.researcher import research_spec
from sigma_ground.deckard import researcher as R


_PRIORS = [
    {"name": "leg", "shape": "cylinder", "count": 4, "freq": 0.9,
     "size_frac": [0.08, 0.08, 0.45], "z_frac": -0.27, "r_frac": 0.8},
    {"name": "seat", "shape": "box", "count": 1, "freq": 0.97,
     "size_frac": [0.95, 0.9, 0.1], "z_frac": 0.0, "r_frac": 0.0},
]


def _with_priors(monkeypatch):
    monkeypatch.setattr(R._sources, "composition_of",
                        lambda n: (_PRIORS, "PartNet (Mo et al. 2019) test", "test")
                        if "zz" in n else None)


def _spec(payload):
    return research_spec("zz contraption", ask=lambda n: json.dumps(payload),
                         model="stub")


def test_grossly_wrong_part_proportions_are_corrected(monkeypatch):
    _with_priors(monkeypatch)
    # qwen makes the seat a near-cube slab (way off the 0.1 z-fraction census)
    spec = _spec({"kind": "composite", "parts": [
        {"name": "seat", "shape": "box",
         "dims": {"x_m": 0.5, "y_m": 0.5, "z_m": 0.5}, "material": "oak",
         "center_m": [0, 0, 0.25]},
        {"name": "leg", "shape": "cylinder",
         "dims": {"radius_m": 0.02, "height_m": 0.25}, "material": "oak",
         "center_m": [0.2, 0.2, -0.125]}]})
    seat = next(p for p in spec.parts if p.name == "seat")
    assert seat.dims["z_m"].value < 0.2 * seat.dims["x_m"].value   # a slab again
    assert seat.dims["z_m"].estimated                              # census != citation
    assert any("PartNet" in s.get("name", "") for s in spec.sources)
    assert compile(spec, resolution=48).mass_kg > 0


def test_under_counted_ring_parts_are_replicated(monkeypatch):
    _with_priors(monkeypatch)
    spec = _spec({"kind": "composite", "parts": [
        {"name": "seat", "shape": "box",
         "dims": {"x_m": 0.5, "y_m": 0.5, "z_m": 0.06}, "material": "oak",
         "center_m": [0, 0, 0.0]},
        {"name": "leg", "shape": "cylinder",
         "dims": {"radius_m": 0.02, "height_m": 0.25}, "material": "oak",
         "center_m": [0.2, 0.2, -0.15]}]})
    legs = [p for p in spec.parts if "leg" in p.name]
    assert len(legs) == 4                                          # census count
    centers = {tuple(round(c, 4) for c in p.center_m) for p in legs}
    assert len(centers) == 4                                       # distinct ring seats
    zs = {round(p.center_m[2], 4) for p in legs}
    assert len(zs) == 1                                            # same height


def test_degenerate_all_at_origin_placement_is_reseated(monkeypatch):
    _with_priors(monkeypatch)
    spec = _spec({"kind": "composite", "parts": [
        {"name": "seat", "shape": "box",
         "dims": {"x_m": 0.5, "y_m": 0.5, "z_m": 0.06}, "material": "oak"},
        {"name": "leg", "shape": "cylinder",
         "dims": {"radius_m": 0.02, "height_m": 0.25}, "material": "oak"}]})
    seat = next(p for p in spec.parts if p.name == "seat")
    leg = next(p for p in spec.parts if p.name == "leg")
    assert leg.center_m[2] < seat.center_m[2]                      # legs reseated LOW
    ext = max(abs(c) for p in spec.parts for c in p.center_m)
    assert ext < 2.0                                               # nothing flies away


def test_attach_anchored_parts_are_never_moved_or_cloned(monkeypatch):
    _with_priors(monkeypatch)
    spec = _spec({"kind": "composite", "parts": [
        {"name": "seat", "shape": "box",
         "dims": {"x_m": 0.5, "y_m": 0.5, "z_m": 0.06}, "material": "oak",
         "center_m": [0, 0, 0]},
        {"name": "leg", "shape": "cylinder",
         "dims": {"radius_m": 0.02, "height_m": 0.25}, "material": "oak",
         "center_m": [0, 0, 0],
         "attach": {"to": "seat", "face": "-z"}}]})
    legs = [p for p in spec.parts if "leg" in p.name]
    assert len(legs) == 1                                          # attach => no clones
    assert legs[0].attach is not None                              # mating keeps authority


def test_cited_dims_are_never_touched(monkeypatch):
    # invariant at the pass level: a part carrying ANY cited dim is exempt from
    # proportion correction (citations always outrank census medians)
    _with_priors(monkeypatch)
    from sigma_ground.deckard.schema import Part
    seat = Part("seat", "box", {"x_m": R.Fact(0.5, "estimated", "", 0.5),
                                "y_m": R.Fact(0.5, "estimated", "", 0.5),
                                "z_m": R.Fact(0.5, "estimated", "", 0.5)},
                "oak", R.Fact(700.0), (0, 0, 0.25))
    leg = Part("leg", "cylinder", {"radius_m": R.Fact(0.5, "Cited Source", "CC0", 0.9),
                                   "height_m": R.Fact(0.25, "estimated", "", 0.5)},
               "oak", R.Fact(700.0), (0.2, 0.2, -0.125))
    parts, sources, seen = [seat, leg], [], set()
    R._apply_composition_priors("zz contraption", parts, sources, seen)
    assert leg.dims["radius_m"].value == 0.5                       # cited -> untouched
    assert leg.dims["radius_m"].source == "Cited Source"


def test_multipart_specs_never_brand_parts_with_object_level_citations(monkeypatch):
    # the chair-slab bug: Wikidata's "chair height" must not land on a SEAT.
    _with_priors(monkeypatch)
    called = []
    monkeypatch.setattr(R, "_ground_dims",
                        lambda *a, **k: called.append(a[0]))
    _spec({"kind": "composite", "parts": [
        {"name": "seat", "shape": "box",
         "dims": {"x_m": 0.5, "y_m": 0.5, "z_m": 0.1}, "material": "oak"},
        {"name": "leg", "shape": "cylinder",
         "dims": {"radius_m": 0.02, "height_m": 0.4}, "material": "oak"}]})
    assert called == []                            # multi-part: no per-part grounding
    _spec({"kind": "composite", "parts": [
        {"name": "ball", "shape": "sphere",
         "dims": {"radius_m": 0.04}, "material": "oak"}]})
    assert len(called) == 1                        # single-part: object dims apply
