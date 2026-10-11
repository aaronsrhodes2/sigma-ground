"""Tests for stage/skippyverse: object cards, the licence gate, local ingest, source adapters, the FTS5 index and the resolver.

Offline and network-free by construction: one test replaces socket.socket with a function that raises and runs every adapter and the
resolver under it. Fixture formats for the source adapters are from memory and UNVERIFIED (stage/skippyverse/data/source_fixtures.json).
No physics is computed here or in the package; masses are the fidelity layer's, or the design's labelled-`guessed` prior.
"""
from __future__ import annotations

import copy
import json
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "stage"
sys.path.insert(0, str(STAGE))

from skippyverse import cards as C  # noqa: E402
from skippyverse import ingest_local, index as IX, metrics, resolve as R  # noqa: E402
from skippyverse.sources import cli as src_cli  # noqa: E402
from skippyverse.sources import common as SC  # noqa: E402
from skippyverse.sources import gazebo_fuel, kenney, nasa3d, polyhaven, smithsonian  # noqa: E402
from skippyverse.taxonomy import category_closure  # noqa: E402
from skippyverse.text import damerau, norm, parse_query  # noqa: E402

CATALOG = ROOT / "stage" / "gmod" / "holodeck" / "data" / "catalog.json"
FIX = json.loads((STAGE / "skippyverse" / "data" / "source_fixtures.json").read_text(encoding="utf-8"))
HU = 0.01905


# ----------------------------------------------------------------------------------------------------------------- helpers
def good_card(**over):
    kw = dict(id="t.1", name="widget", aliases=["gizmo"], category="misc",
              size={"w": 1.0, "d": 0.5, "h": 0.8, "basis": "sourced", "source": "a test"},
              front={"yaw_deg": 0.0, "source": "curated-by-eye"}, symmetric=False,
              mass={"kg": 10.0, "low": 5.0, "high": 20.0, "basis": "estimated", "derived_from_size": True, "source": "t"},
              visual={"class": "found", "style": "low-poly", "blocky": False},
              provenance={"source": "Poly Haven", "license": "CC0", "author": "A", "url": "https://example.invalid/a"})
    kw.update(over)
    return C.make_card(**kw)


def tiny_cards():
    def sz(w, d, h, basis="measured"):
        return {"w": w, "d": d, "h": h, "basis": basis, "source": "test"}

    def mk(i, name, size, cls="curated", aliases=(), approx=(), category="seating", lic="CC0", author="A", url="https://e.invalid/x",
           flags=(), local=None, tags=()):
        vis = {"class": cls, "style": "low-poly", "blocky": False}
        if local:
            vis["local_model"] = local
        prov = ({"source": "Half-Life 2 (Valve)", "license": "Valve-SSA", "author": "Valve"} if local
                else {"source": "Poly Haven", "license": lic, "author": author, "url": url})
        return C.make_card(id=i, name=name, aliases=aliases, approx_names=approx, category=category, size=size, symmetric=True,
                           front={"yaw_deg": 0.0, "source": "curated-by-eye"}, visual=vis, provenance=prov, flags=flags, tags=tags,
                           mass=C.prior_mass(size) if size else None)

    cs = [mk(f"c.couch.{n}", "couch", sz(*dims), aliases=["sofa", "settee"], local=f"models/c{n}.mdl")
          for n, dims in enumerate([(1.5, 0.8, 0.8), (1.8, 0.9, 0.85), (2.0, 0.9, 0.9), (2.2, 0.95, 0.9), (2.4, 1.0, 0.9), (2.6, 1.0, 0.9),
                                    (2.8, 1.0, 0.9), (3.0, 1.1, 0.9), (3.3, 1.1, 0.95), (3.5, 1.2, 1.0)])]
    cs += [
        mk("c.stove", "wood stove", sz(0.6, 0.5, 1.0), approx=["fireplace"], category="appliance", local="models/stove.mdl"),
        mk("c.floor", "floor lamp", sz(0.4, 0.4, 1.6), aliases=["standing lamp"], category="lighting", local="models/fl.mdl"),
        mk("c.table", "table lamp", sz(0.3, 0.3, 0.5), category="lighting", local="models/tl.mdl"),
        mk("f.hydrant", "fire hydrant", None, cls="found", category="misc"),                      # found, size unknown
        mk("f.jet", "jumbo jet", sz(64.44, 70.66, 19.4, "sourced"), cls="found", aliases=["747"], category="vehicle/aircraft"),
        mk("p.jet", "toy jet", sz(0.27, 0.25, 0.08, "estimated"), cls="placeholder", category="vehicle/aircraft",
           lic="ShapeNet Terms of Use (non-commercial)", author="ShapeNet"),
        mk("f.piano", "piano", sz(1.5, 0.6, 1.2, "sourced"), cls="found", category="misc"),
        mk("f.bench", "park bench", sz(1.8, 0.6, 0.8), cls="found", category="seating", aliases=["bench"]),
        mk("f.tbench", "workbench", sz(1.8, 0.6, 0.9), cls="found", category="table", aliases=["bench"]),
    ]
    return cs


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_index")
    IX.build_index(tiny_cards(), d, synonym_groups=[["settee", "lounge"], ["automobile", "motorcar", "car"]])
    return IX.load_index(d)


@pytest.fixture(scope="module")
def local_cards():
    return ingest_local.ingest_all()


@pytest.fixture(scope="module")
def full(tmp_path_factory, local_cards):
    d = tmp_path_factory.mktemp("full_index")
    IX.build_index(local_cards[0], d)
    return IX.load_index(d)


# ============================================================================================================ 1. licence gate
@pytest.mark.parametrize("raw,canon", [
    ("CC0", "CC0"), ("cc0-1.0", "CC0"), ("Creative Commons - Public Domain", "CC0"), ("CC BY 4.0", "CC-BY-4.0"), ("CC-BY-3.0", "CC-BY-3.0"),
    ("cc-by-4", "CC-BY-4.0"), ("NASA", "PD-USGOV"), ("CC BY-NC 4.0", "CC-NC"), ("CC-BY-SA-4.0", "CC-SA"), ("CC BY-ND 4.0", "CC-ND"),
    ("", "UNKNOWN"), ("Valve-SSA", "Valve-SSA")])
def test_normalize_licence(raw, canon):
    assert C.normalize_licence(raw) == canon


def test_allowed_set_is_exactly_the_four():
    assert C.ALLOWED_LICENCES == {"CC0", "CC-BY-3.0", "CC-BY-4.0", "PD-USGOV"}
    assert not C.licence_allowed("CC BY 2.0")                       # a CC-BY version outside the set
    assert not C.licence_allowed("CC BY-NC 4.0") and not C.licence_allowed("something else")


def test_allowed_licence_card_is_redistributable():
    assert good_card()["redistributable"] is True


@pytest.mark.parametrize("lic", ["CC BY-NC 4.0", "CC-BY-SA-4.0", "CC BY-ND 4.0", "Valve-SSA", "unverified", ""])
def test_disallowed_licence_card_exists_but_is_flagged(lic):
    c = good_card(provenance={"source": "Somewhere", "license": lic or "x", "author": "A", "url": "https://e.invalid"})
    if not lic:
        c["provenance"]["license"] = ""
        c["redistributable"], _ = C.compute_redistributable(c)
        c["redistributable_reason"] = ""
        c["flags"].append("not_redistributable")
    assert c["redistributable"] is False and "not_redistributable" in c["flags"]
    if lic:
        assert C.validate_card(c) == []                          # allowed to exist, just not servable


@pytest.mark.parametrize("source", ["Half-Life 2 / Counter-Strike: Source (Valve)", "ShapeNetSem", "Sketchfab", "PartNet exemplar", "Quaternius pack"])
def test_never_servable_sources_override_a_claimed_cc0(source):
    c = good_card(provenance={"source": source, "license": "CC0", "author": "A", "url": "https://e.invalid"})
    assert c["redistributable"] is False
    assert "never-servable" in c["redistributable_reason"]


def test_validator_rejects_a_forged_redistributable_flag():
    c = good_card(provenance={"source": "Somewhere", "license": "CC BY-NC 4.0", "author": "A", "url": "https://e.invalid"})
    c["redistributable"] = True
    assert any("LICENCE GATE" in p for p in C.validate_card(c))
    v = good_card(provenance={"source": "Valve", "license": "CC0", "author": "A", "url": "u"})
    v["redistributable"] = True
    assert any("LICENCE GATE" in p for p in C.validate_card(v))


def test_servable_found_asset_needs_author_and_url():
    c = good_card()
    c["provenance"].pop("author")
    assert any("author" in p for p in C.validate_card(c))
    c = good_card()
    c["provenance"].pop("url")
    assert any("url" in p for p in C.validate_card(c))


def test_not_redistributable_requires_the_flag():
    c = good_card(provenance={"source": "x", "license": "CC BY-NC 4.0", "author": "A", "url": "u"})
    c["flags"].remove("not_redistributable")
    assert any("flagged" in p for p in C.validate_card(c))


# ================================================================================================================ 2. the card
def test_good_card_validates():
    assert C.validate_card(good_card()) == []


def test_validator_catches_each_schema_problem():
    base = good_card()
    cases = {
        "card_v": lambda c: c.update(card_v=2),
        "size positive": lambda c: c["size_m"].update(w=-1),
        "size basis": lambda c: c["size_m"].update(basis="certain"),
        "size source": lambda c: c["size_m"].update(source=""),
        "mass range": lambda c: c["mass"].update(kg=100),
        "visual class": lambda c: c["visual"].update({"class": "magic"}),
        "front source": lambda c: c["front"].update(source="vibes"),
        "provenance": lambda c: c["provenance"].pop("source"),
        "mobility": lambda c: c.update(mobility="flying"),
    }
    for name, mut in cases.items():
        c = copy.deepcopy(base)
        mut(c)
        assert C.validate_card(c), f"validator missed: {name}"


def test_unknown_front_must_be_symmetric():
    c = good_card(front={"yaw_deg": 0.0, "source": "unknown"}, symmetric=False)
    assert any("symmetric" in p for p in C.validate_card(c))
    assert C.validate_card(good_card(front={"yaw_deg": 0.0, "source": "unknown"}, symmetric=True)) == []


def test_size_none_needs_the_size_unknown_flag():
    c = good_card(size=None, mass=None)
    assert "size_unknown" in c["flags"] and C.validate_card(c) == []
    c["flags"].remove("size_unknown")
    assert C.validate_card(c)


def test_derived_mass_is_never_stronger_than_its_size():
    c = good_card(size={"w": 1, "d": 1, "h": 1, "basis": "guessed", "source": "s"},
                  mass={"kg": 5, "basis": "sourced", "derived_from_size": True, "source": "s"})
    assert any("stronger" in p for p in C.validate_card(c))
    assert C.weakest_basis("measured", "guessed", "sourced") == "guessed"
    assert C.weakest_basis("measured", "sourced") == "sourced"


def test_prior_mass_is_the_designs_formula_and_always_guessed():
    m = C.prior_mass({"w": 1.0, "d": 1.0, "h": 1.0})
    assert m["basis"] == "guessed" and abs(m["kg"] - 41.9) < 1e-9
    assert m["low"] < m["kg"] < m["high"]
    big = C.prior_mass({"w": 2.0, "d": 1.0, "h": 1.0})
    assert abs(big["kg"] - 41.9 * 2 ** 0.72) < 1e-9


def test_mobility_static_over_8m_or_2000kg():
    assert good_card(size={"w": 9.0, "d": 2, "h": 2, "basis": "sourced", "source": "s"}, mass=None)["mobility"] == "static"
    assert good_card(mass={"kg": 2500, "basis": "sourced", "source": "s"})["mobility"] == "static"
    assert good_card()["mobility"] == "dynamic"


def test_honesty_clause_comes_from_class_and_basis_only():
    assert "game library" in good_card(visual={"class": "curated"}, provenance={"source": "Half-Life 2", "license": "x"})["say"]
    s = good_card()["say"]
    assert "low-poly model by A, CC0" in s and "size from a published source" in s
    assert "weight" not in s.replace("weight is a guess", "")            # estimated mass says nothing
    g = good_card(mass={"kg": 3, "basis": "guessed", "source": "s"})["say"]
    assert "weight is a guess" in g
    assert "nly a box" in good_card(visual={"class": "placeholder"})["say"]
    assert "blocky" in good_card(visual={"class": "deckard", "blocky": True})["say"]
    assert "AI-made guess" in good_card(visual={"class": "generated"})["say"]
    assert "size unknown" in good_card(size=None, mass=None)["say"]


# ================================================================================================= planner converter (HU)
def test_converter_roundtrips_every_catalog_variant():
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    cards, _ = ingest_local.ingest_all()
    cur = [c for c in cards if c["visual"]["class"] == "curated"]
    assert len(cur) == sum(len(k["variants"]) for k in cat["kinds"].values()) == 126
    by_model = {v["model"]: v for k in cat["kinds"].values() for v in k["variants"]}
    for c in cur:
        fp = C.to_planner_footprint(c)
        v = by_model[c["visual"]["local_model"]]
        for key in ("w", "d", "h", "cx", "cy"):
            assert abs(fp[key] - v[key]) < 1e-6, (c["id"], key)
        assert fp["front"] == pytest.approx(v["front"]) and fp["symmetric"] == v["symmetric"]
        assert fp["mins"] == v["mins"] and fp["maxs"] == v["maxs"]
        assert (fp["top"] is None) == (v["top"] is None)
        if v["top"]:
            for k in ("z", "x0", "x1", "y0", "y1"):
                assert fp["top"][k] == pytest.approx(v["top"][k], abs=1e-6)


def test_converter_uses_hammer_units_and_sets_the_floor_origin():
    c = good_card(size={"w": 1.905, "d": 0.9525, "h": 0.381, "basis": "sourced", "source": "s"}, mass=None)
    fp = C.to_planner_footprint(c)
    assert (fp["w"], fp["d"], fp["h"]) == pytest.approx((100.0, 50.0, 20.0))
    assert fp["mins"][2] == 0.0 and fp["maxs"][2] == pytest.approx(20.0)      # origin on the floor
    assert fp["cx"] == fp["cy"] == 0.0
    c2 = good_card(front={"yaw_deg": 90.0, "source": "curated-by-eye"}, size={"w": 1.905, "d": 0.9525, "h": 0.381, "basis": "sourced", "source": "s"})
    f2 = C.to_planner_footprint(c2)
    assert f2["maxs"][0] == pytest.approx(50.0) and f2["maxs"][1] == pytest.approx(25.0)   # facing along y: width runs along x


def test_converter_refuses_a_card_with_no_size_and_passes_the_asset_id():
    with pytest.raises(ValueError):
        C.to_planner_footprint(good_card(size=None, mass=None))
    assert C.to_planner_footprint(good_card())["model"] == "t.1"


def test_to_planner_kind_matches_catalog_kind_fields():
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    cards, _ = ingest_local.ingest_all()
    couches = [c for c in cards if c["kind"] == "couch" and c["visual"]["class"] == "curated"]
    k = C.to_planner_kind(couches)
    ck = cat["kinds"]["couch"]
    assert k["kind"] == "couch" and k["names"][0] == "couch" and set(ck["names"]) <= set(k["names"])
    assert k["approx"] == ck["approx"] and k["category"] == ck["category"] and k["wall"] == ck["wall"] and k["seat"] is True
    assert len(k["variants"]) == len(ck["variants"])


def test_strip_for_web_removes_game_paths_and_asset_data():
    cards, _ = ingest_local.ingest_all()
    cur = next(c for c in cards if c["visual"]["class"] == "curated")
    w = C.strip_for_web(cur)
    assert "local_model" not in w["visual"] and "model_frame_hu" not in w["visual"]
    assert set(w["provenance"]) <= {"source", "license"}
    assert "local_model" in cur["visual"]                                 # the original is untouched
    assert "models/" not in json.dumps(w)


# ============================================================================================================== 3. local ingest
def test_ingest_counts_and_validity(local_cards):
    cards, rep = local_cards
    assert rep["by_class"]["curated"] == 126 and rep["curated_kinds"] == 46
    assert rep["by_class"]["deckard"] >= 10 and rep["by_class"]["placeholder"] > 300
    assert rep["total"] == len(cards) == len({c["id"] for c in cards})
    for c in cards:
        assert C.validate_card(c) == [], c["id"]


def test_valve_cards_are_curated_measured_and_not_redistributable(local_cards):
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    for c in (x for x in local_cards[0] if x["visual"]["class"] == "curated"):
        assert c["redistributable"] is False and "not_redistributable" in c["flags"]
        assert c["size_m"]["basis"] == "measured" and "Valve" in c["provenance"]["author"]
        assert c["visual"]["local_model"].startswith("models/")
    model = {v["model"]: v for k in cat["kinds"].values() for v in k["variants"]}
    for c in (x for x in local_cards[0] if x["visual"]["class"] == "curated"):
        v = model[c["visual"]["local_model"]]
        assert c["mass"]["kg"] == v["mass"]
        assert c["mass"]["basis"] == ("estimated" if v["mass_status"] == "audited" else "guessed")


def test_unaudited_engine_mass_is_labelled_guessed(local_cards):
    g = [c for c in local_cards[0] if c["visual"]["class"] == "curated" and c["mass"]["basis"] == "guessed"]
    assert len(g) == 6 and all("gameplay mass" in c["mass"]["source"] for c in g)


def test_shapenet_derived_cards_are_never_servable_and_never_measured(local_cards):
    sn = [c for c in local_cards[0] if "shapenet_derived" in c["flags"]]
    assert len(sn) > 300
    for c in sn:
        assert c["redistributable"] is False
        assert c["size_m"] is None or c["size_m"]["basis"] in ("estimated", "sourced")
        assert c["mass"] is None or c["mass"]["basis"] == "guessed"


def test_shapenet_axis_order_is_x_yup_z(local_cards):
    sn = json.loads((ROOT / "sigma_ground/inventory/data/shapenetsem_sizes.json").read_text(encoding="utf-8"))["categories"]
    c = next(x for x in local_cards[0] if x["id"] == "placeholder.shapenetsem.piano")
    x, y, z = sn["piano"]["dims_m"]
    assert (c["size_m"]["w"], c["size_m"]["h"], c["size_m"]["d"]) == (x, y, z)


def test_names_already_held_are_not_duplicated_and_crosschecked(local_cards):
    cards, rep = local_cards
    names = [c["name"].lower() for c in cards]
    assert "sofa" not in names and "floorlamp" not in names and "coffeetable" not in names      # curated owns them
    assert rep["shapenet_names_already_held"] > 100
    car = next(c for c in cards if c["id"] == "seed.car")
    assert car["crosscheck"][0]["verdict"].startswith("ShapeNetSem value is the outlier")
    assert "size_crosscheck_disagrees" not in car["flags"]


def test_seed_facts_are_labelled_second_hand_and_unservable(local_cards):
    jet = next(c for c in local_cards[0] if c["id"] == "seed.jumbo-jet")
    assert "unverified_source" in jet["flags"] and jet["redistributable"] is False
    assert "NOT re-checked" in jet["size_m"]["source"]
    assert jet["mobility"] == "static" and jet["size_m"]["w"] == 64.44


def test_scalar_sizes_are_guessed(local_cards):
    sc = [c for c in local_cards[0] if "scalar_size_only" in c["flags"]]
    assert sc and all(c["size_m"]["basis"] == "guessed" for c in sc)


def test_deckard_cards_carry_parts_and_prefer_published_standard_sizes(local_cards):
    door = next(c for c in local_cards[0] if c["id"] == "deckard.door")
    assert door["visual"]["class"] == "deckard" and door["visual"]["parts"]
    assert door["size_m"]["basis"] == "sourced" and "ANSI" in door["size_m"]["source"]


# ====================================================================================================== 4. source adapters
def test_polyhaven_converts_mm_and_axes_and_keeps_cc0():
    cards, rep = polyhaven.parse_listing_with_report(FIX["polyhaven"]["raw"])
    sofa = next(c for c in cards if c["name"] == "sofa")
    assert (sofa["size_m"]["w"], sofa["size_m"]["d"], sofa["size_m"]["h"]) == pytest.approx((2.0, 0.95, 0.85))
    assert sofa["size_m"]["basis"] == "sourced" and sofa["redistributable"] and sofa["aliases"] == ["sofa 01"]
    assert sofa["provenance"]["url"].endswith("/a/sofa_01") and "source_format_unverified" in sofa["flags"]
    assert next(c for c in cards if "boulder" in c["name"])["size_m"] is None
    assert rep["problems"] and len(cards) == 3
    y_up = polyhaven.parse_listing(FIX["polyhaven"]["raw"], up="y")
    assert next(c for c in y_up if c["name"] == "sofa")["size_m"]["h"] == pytest.approx(0.95)


def test_kenney_sizes_are_guessed_and_scale_flagged():
    cards = kenney.parse_listing(FIX["kenney"]["raw"])
    sedan = next(c for c in cards if c["name"] == "sedan")
    assert sedan["size_m"]["basis"] == "guessed" and "scale_arbitrary" in sedan["flags"]
    assert (sedan["size_m"]["w"], sedan["size_m"]["d"], sedan["size_m"]["h"]) == (1.5, 2.55, 1.45)      # glTF y-up
    assert sedan["redistributable"] and sedan["provenance"]["author"] == "Kenney"
    assert next(c for c in cards if "wheel" in c["name"])["size_m"] is None


def test_kenney_pack_with_nc_licence_is_skipped_unless_asked():
    assert kenney.parse_listing(FIX["kenney"]["raw_nc"]) == []
    kept = kenney.parse_listing(FIX["kenney"]["raw_nc"], include_disallowed=True)
    assert len(kept) == 1 and kept[0]["redistributable"] is False and C.validate_card(kept[0]) == []


def test_fuel_keeps_only_cc0_and_ccby4():
    cards, rep = gazebo_fuel.parse_listing_with_report(FIX["fuel"]["raw"])
    assert {c["name"] for c in cards} == {"fire hydrant", "bicycle", "grand piano"}
    assert len(rep["skipped"]) == 2 and len(rep["problems"]) == 1                  # NC teapot, SA cabinet, one nonsense record
    bike = next(c for c in cards if c["name"] == "bicycle")
    assert bike["provenance"]["license"] == "CC-BY-4.0" and "licence_version_assumed" in bike["flags"]
    assert "licence_version_assumed" not in next(c for c in cards if c["name"] == "fire hydrant")["flags"]
    allc = gazebo_fuel.parse_listing(FIX["fuel"]["raw"], include_disallowed=True)
    assert len(allc) == 5 and sum(not c["redistributable"] for c in allc) == 2
    assert next(c for c in cards if c["name"] == "grand piano")["size_m"] is None


def test_nasa_real_size_is_sourced_and_arbitrary_bbox_is_guessed():
    cards = nasa3d.parse_listing(FIX["nasa"]["raw"])
    sat = next(c for c in cards if c["name"] == "saturn v")
    shu = next(c for c in cards if c["name"] == "space shuttle")
    assert sat["size_m"]["basis"] == "guessed" and "scale_arbitrary" in sat["flags"]
    assert shu["size_m"]["basis"] == "sourced" and shu["size_m"]["w"] == 23.8
    assert all(c["provenance"]["license"] == "NASA" and c["redistributable"] and "nasa_credit_no_endorsement" in c["flags"] for c in cards)


def test_smithsonian_cc0_only_and_dimension_label_order():
    cards, rep = smithsonian.parse_listing_with_report(FIX["smithsonian"]["raw"])
    assert len(cards) == 1 and len(rep["skipped"]) == 1 and len(rep["problems"]) == 1
    s = cards[0]["size_m"]
    assert (s["h"], s["w"], s["d"]) == pytest.approx((0.10, 0.20, 0.30))               # 'H x W x D: 10 x 20 x 30 cm'
    assert s["basis"] == "sourced"


def test_dimension_text_parsing():
    assert SC.parse_dims_text("H x W x D: 10 x 20 x 30 cm") == pytest.approx((0.1, 0.2, 0.3))
    assert SC.parse_dims_text("5 x 4 x 3 in") == pytest.approx((0.127, 0.1016, 0.0762))
    assert SC.parse_dims_text("10 x 20 cm") is None and SC.parse_dims_text("") is None
    assert SC.label_order_to_whd((1, 2, 3), "H x L x W") == (2, 3, 1)
    assert SC.label_order_to_whd((1, 2, 3), "H x W x D") == (2, 3, 1)
    assert SC.label_order_to_whd((1, 2, 3), "no label") == (1, 2, 3)


@pytest.mark.parametrize("mod,bad", [(polyhaven, []), (kenney, {}), (gazebo_fuel, {}), (nasa3d, {}), (smithsonian, {"x": 1})])
def test_adapters_reject_the_wrong_shape(mod, bad):
    with pytest.raises(SC.AdapterFormatError):
        mod.parse_listing(bad)


def test_every_adapter_declares_its_fixture_format_unverified():
    for mod in (polyhaven, kenney, gazebo_fuel, nasa3d, smithsonian):
        assert "unverified" in mod.FIXTURE_STATUS and "UNVERIFIED" in mod.__doc__
    for key in ("polyhaven", "kenney", "fuel", "nasa", "smithsonian"):
        assert "unverified" in FIX[key]["status"]


def test_no_network_anywhere(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    for name, mod, key in [("polyhaven", polyhaven, "polyhaven"), ("kenney", kenney, "kenney"), ("fuel", gazebo_fuel, "fuel"),
                           ("nasa", nasa3d, "nasa"), ("smithsonian", smithsonian, "smithsonian")]:
        assert mod.parse_listing(FIX[key]["raw"])
    idx_dir = tmp_path / "ix"
    IX.build_index(tiny_cards(), idx_dir, synonym_groups=[])
    assert R.resolve("dragon", IX.load_index(idx_dir), online=True)["status"] == "would_fetch"


def test_fetch_listing_is_a_stub():
    with pytest.raises(SC.NetworkNotAllowed):
        src_cli.fetch_listing("polyhaven")
    with pytest.raises(NotImplementedError):
        src_cli.fetch_listing("polyhaven", allow_network=True)
    assert src_cli.main(["--source", "polyhaven", "--allow-network"]) == 3


def test_adapter_sources_do_not_import_network_modules():
    for p in (STAGE / "skippyverse").rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        for bad in ("import urllib", "import requests", "import http", "import socket", "import httpx", "from urllib", "import ftplib"):
            assert bad not in text, f"{p.name} imports {bad}"


def test_cli_parses_a_raw_file(tmp_path, capsys):
    raw = tmp_path / "l.json"
    raw.write_text(json.dumps(FIX["fuel"]["raw"]), encoding="utf-8")
    out = tmp_path / "cards.json"
    assert src_cli.main(["--source", "fuel", "--raw", str(raw), "--out", str(out)]) == 0
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 3
    assert "2 skipped by the licence gate" in capsys.readouterr().out


# ================================================================================================================ 5. the index
def test_fts5_is_available():
    assert IX.fts5_available()


def test_version_json_is_written_last(tmp_path, monkeypatch):
    d = tmp_path / "ix"
    v = IX.build_index(tiny_cards(), d, synonym_groups=[])
    assert (d / "version.json").stat().st_mtime_ns >= (d / "index.sqlite").stat().st_mtime_ns
    assert v["n_cards"] == len(tiny_cards())
    # an interrupted rebuild must leave NO version.json, so a reader refuses the half-built index
    real_replace = Path.replace
    monkeypatch.setattr(Path, "replace", lambda self, tgt: (_ for _ in ()).throw(OSError("crash before the swap")))
    with pytest.raises(OSError):
        IX.build_index(tiny_cards(), d, synonym_groups=[])
    monkeypatch.setattr(Path, "replace", real_replace)
    assert not (d / "version.json").exists()
    with pytest.raises(IX.IndexError_):
        IX.load_index(d)


def test_load_refuses_a_tampered_database(tmp_path):
    d = tmp_path / "ix"
    IX.build_index(tiny_cards(), d, synonym_groups=[])
    with open(d / "index.sqlite", "ab") as f:
        f.write(b"x")
    with pytest.raises(IX.IndexError_, match="does not match"):
        IX.load_index(d)


def test_search_name_beats_tag_and_plural_matches(tiny):
    hits = [c["name"] for c, _ in tiny.search("couches")]
    assert hits and hits[0] == "couch"
    assert tiny.search("zzzqqq") == []


def test_category_closure_and_terms(tiny):
    assert category_closure("vehicle/aircraft") == ["vehicle/aircraft", "vehicle"]
    assert category_closure("seating") == ["seating", "furniture"]
    assert {c["name"] for c in tiny.category_cards("vehicle")} >= {"jumbo jet", "toy jet"}
    assert {c["kind"] for c in tiny.category_cards("furniture")} >= {"couch", "park bench"}


def test_compact_names_are_findable(tiny):
    assert tiny.by_term("woodstove") and tiny.by_term("wood stove")


def test_synonym_groups_load_from_the_repo(local_cards):
    groups = IX.load_synonym_groups()
    assert len(groups) == 162 and ["jumbojet", "jumbo jet"] in groups
    assert IX.load_synonym_groups(Path("/no/such/file.json")) == []


def test_size_sanity_flags_the_absurd_and_spares_the_curated(local_cards):
    found = IX.size_sanity(local_cards[0])
    assert "placeholder.shapenetsem.jet" in found and "0.271" in found["placeholder.shapenetsem.jet"][0]
    assert not any(i.startswith("curated.") for i in found)
    assert "seed.car" not in found and "seed.jumbo-jet" not in found
    huge = good_card(id="x", category="seating", size={"w": 50.0, "d": 1, "h": 1, "basis": "guessed", "source": "s"}, mass=None)
    assert "x" in IX.size_sanity([huge])
    dense = good_card(id="y", size={"w": 1, "d": 1, "h": 1, "basis": "guessed", "source": "s"}, mass={"kg": 1e6, "basis": "guessed", "source": "s"})
    assert "density" in IX.size_sanity([dense])["y"][0]


def test_flagged_cards_are_marked_in_the_index(full):
    assert full.size_ok("placeholder.shapenetsem.jet") is False
    assert "size_suspect" in full.card("placeholder.shapenetsem.jet")["flags"]
    assert full.size_ok("seed.car") is True


def test_web_index_strips_game_paths_but_keeps_the_card(tmp_path, local_cards):
    IX.build_index(local_cards[0], tmp_path / "w", web=True)
    ix = IX.load_index(tmp_path / "w")
    c = ix.card(next(i for i in ix.all_ids() if i.startswith("curated.couch")))
    assert "local_model" not in c["visual"] and c["redistributable"] is False and ix.version["web"] is True
    raw = ix.db.execute("SELECT group_concat(json) FROM cards").fetchone()[0]
    assert ("models/props" in raw) is False and ("furniturecouch" in raw) is False
    assert ix.n_cards() == len(local_cards[0])


def test_index_counts_match_ingest(full, local_cards):
    assert full.n_cards() == len(local_cards[0]) and full.version["n_cards"] == len(local_cards[0])
    assert full.version["n_synonym_groups"] == 162


# ================================================================================================================ text helpers
@pytest.mark.parametrize("raw,sing", [("couches", "couch"), ("shelves", "shelf"), ("boxes", "box"), ("batteries", "battery"), ("glass", "glass"),
                                       ("mattress", "mattress"), ("bus", "bus"), ("lamps", "lamp"), ("knives", "knife")])
def test_singular(raw, sing):
    from skippyverse.text import singular
    assert singular(raw) == sing


def test_edit_distance():
    assert damerau("couch", "cuoch") == 1 and damerau("couch", "couch") == 0 and damerau("dragon", "wagon") == 2
    assert norm("Coffee Tables") == "coffee table"


def test_parse_query():
    p = parse_query("Skippy, add three red couches next to the wall")
    assert p["phrase"] == "couch" and p["count"] == 3 and p["modifiers"] == ["red"]
    assert parse_query("I'd like a dozen barrels, please")["count"] == 12
    assert parse_query("a big couch")["size_pref"] == "up" and parse_query("a tiny lamp")["size_pref"] == "down"
    assert parse_query("can you give me a refrigerator?")["phrase"] == "refrigerator"
    assert parse_query("")["phrase"] == "" and parse_query("Skippy,")["phrase"] == ""
    assert parse_query("put a lamp in the corner")["phrase"] == "lamp"


# ================================================================================================================ 6. resolver
def test_exact_plural_and_alias_at_rung_0(tiny):
    r = R.resolve("Skippy, add three couches", tiny)
    assert (r["status"], r["rung"], r["kind"], r["count"]) == ("exact", "0", "couch", 3)
    assert "I have a couch" in r["sentence"]
    a = R.resolve("a sofa", tiny)
    assert a["status"] == "alias" and a["kind"] == "couch" and "sofa" in a["sentence"] and "filed as a couch" in a["sentence"]


def test_unknown_noun_is_refused_and_returns_no_object(tiny):
    for q in ("dragon", "a unicorn", "phoenix", "submarine"):
        r = R.resolve(q, tiny)
        assert r["status"] == "not_found" and r["card"] is None and r["kind"] is None and r["shortlist"] == []
        assert "I don't have" in r["sentence"] and "can't make new things" in r["sentence"]
        assert r["refusal"] == "unknown" and r["rung"] == "5"


def test_unknown_noun_plan_describes_but_never_executes_rungs_2_3_4(tiny):
    r = R.resolve("a helicopter", tiny)
    rungs = {p["rung"]: p for p in r["plan"]}
    assert set(rungs) == {2, 3, 4} and all(p["executed"] is False for p in r["plan"])
    assert any("approval" in n for n in rungs[2]["needs"]) and "CC-BY-4.0" in rungs[2]["would"]
    assert {p["rung"] for p in R.resolve("a dragon", tiny)["plan"]} == {2, 3}          # no Deckard template for a dragon
    on = R.resolve("dragon", tiny, online=True)
    assert on["status"] == "would_fetch" and on["rung"] == "2" and on["card"] is None and "(not done" in on["sentence"]


def test_approx_name_is_a_closest_never_silent(tiny):
    r = R.resolve("fireplace", tiny)
    assert r["status"] == "closest" and r["kind"] == "wood stove" and r["rung"] == "1b"
    assert "closest I have is a wood stove" in r["sentence"] and "fireplace" in r["sentence"]


def test_typo_is_announced(tiny):
    r = R.resolve("cuoch", tiny)
    assert r["status"] == "closest" and r["kind"] == "couch" and "I read 'cuoch' as 'couch'" in r["sentence"]
    assert R.resolve("sofaa", tiny)["kind"] == "couch"
    assert R.resolve("zzz", tiny)["kind"] is None                         # too short to be fuzzy-matched
    assert R.resolve("dragon", tiny)["status"] == "not_found"             # dragon is 2 edits from nothing in this index


def test_synonym_group_resolves_with_a_statement(tiny):
    r = R.resolve("lounge", tiny)
    assert r["status"] == "synonym" and r["kind"] == "couch" and "listed synonym" in r["sentence"]


def test_ambiguity_is_reported_not_resolved(tiny):
    r = R.resolve("a lamp", tiny)
    assert r["status"] == "ambiguous" and r["card"] is None and set(r["options"]) >= {"floor lamp", "table lamp"}
    assert "Which one" in r["sentence"]
    b = R.resolve("a bench", tiny)                                         # two kinds carry the alias 'bench'
    assert b["status"] == "ambiguous" and set(b["options"]) == {"park bench", "workbench"}


def test_category_word_is_not_an_object(tiny):
    r = R.resolve("furniture", tiny)
    assert r["status"] == "category" and r["card"] is None and "couch" in r["options"]
    assert "won't pick one" in r["sentence"]


def test_partial_text_needs_head_noun_and_coverage(tiny):
    r = R.resolve("antique brass lamp", tiny)
    assert r["status"] in ("not_found",) and r["card"] is None
    c = R.resolve("tall standing floor lamp", tiny)                       # modifier 'tall' dropped; 3 of 3 words are in one kind
    assert c["kind"] == "floor lamp"


def test_modifiers_are_dropped_only_when_they_are_modifiers(tiny):
    assert R.resolve("a big red couch", tiny)["kind"] == "couch"
    r = R.resolve("grand piano", tiny)                                    # 'grand' is not a modifier: not silently dropped
    assert r["status"] == "closest" and "closest I have is a piano" in r["sentence"]


def test_size_words_pick_the_extreme_variants(tiny):
    mid, big, small = (R.resolve(q, tiny)["card"]["size_m"]["w"] for q in ("a couch", "a huge couch", "a tiny couch"))
    assert small == 1.5 and big == 3.5 and small < mid < big


def test_shortlist_spans_the_size_range_with_at_most_seven(tiny):
    r = R.resolve("couch", tiny)
    ws = [c["size_m"]["w"] for c in r["shortlist"]]
    assert len(ws) == 7 and ws[0] == 1.5 and ws[-1] == 3.5 and ws == sorted(ws)
    assert len(R.shortlist([tiny.card("c.couch.0")])) == 1


def test_too_big_for_the_room_is_refused_with_the_size_and_a_suggestion(tiny):
    r = R.resolve("a jumbo jet", tiny, room={"w": 10.7, "d": 9.1, "h": 2.7})
    assert r["status"] == "too_big" and r["refusal"] == "too_big" and "(71 m)" in r["sentence"] and "Try it outdoors" in r["sentence"]
    out = R.resolve("a jumbo jet", tiny, room={"outdoor": True})
    assert out["status"] == "exact" and out["card"]["id"] == "f.jet"
    couch = R.resolve("a couch", tiny, room={"w": 2.0, "d": 1.0})          # small variants fit, big ones are never chosen
    assert couch["status"] == "exact" and couch["card"]["size_m"]["w"] <= 2.0
    huge = R.resolve("a huge couch", tiny, room={"w": 2.0, "d": 1.0})
    assert huge["card"]["size_m"]["w"] <= 2.0


def test_nothing_is_ever_silently_shrunk_to_fit(tiny):
    r = R.resolve("a jumbo jet", tiny, room={"w": 10, "d": 10})
    assert r["card"]["size_m"]["w"] == 64.44 and r["status"] == "too_big"


def test_suspect_size_is_not_trusted(tiny):
    r = R.resolve("a toy jet", tiny)
    assert r["status"] == "untrusted_size" and "looks wrong" in r["sentence"] and r["plan"][0]["executed"] is False


def test_unknown_size_is_not_guessed(tiny):
    r = R.resolve("a fire hydrant", tiny)
    assert r["status"] == "needs_size" and "not its size" in r["sentence"] and r["refusal"] == "no_size"


def test_web_audience_reports_but_does_not_serve_valve_models(tiny):
    r = R.resolve("a couch", tiny, audience="web")
    assert r["status"] == "exact" and r["servable"] is False
    assert "I can't share a model of it, so it would be a box" in r["sentence"]
    assert "local_model" not in r["card"]["visual"] and r["card"]["size_m"]["w"] > 0
    local = R.resolve("a couch", tiny, audience="local")
    assert "local_model" in local["card"]["visual"]


def test_web_audience_prefers_a_servable_card_of_the_same_kind(tmp_path):
    cs = tiny_cards()
    cs.append(C.make_card(id="f.couch", name="couch", aliases=["sofa"], category="seating",
                          size={"w": 2.0, "d": 0.9, "h": 0.9, "basis": "sourced", "source": "t"}, symmetric=True,
                          front={"yaw_deg": 0.0, "source": "curated-by-eye"}, visual={"class": "found", "style": "low-poly", "blocky": False},
                          provenance={"source": "Poly Haven", "license": "CC0", "author": "A", "url": "https://e.invalid"},
                          mass=C.prior_mass({"w": 2.0, "d": 0.9, "h": 0.9})))
    IX.build_index(cs, tmp_path / "ix", synonym_groups=[])
    ix = IX.load_index(tmp_path / "ix")
    w = R.resolve("a couch", ix, audience="web")
    assert w["card"]["id"] == "f.couch" and w["servable"] is True and "can't share" not in w["sentence"]
    assert R.resolve("a couch", ix, audience="local")["card"]["visual"]["class"] == "curated"


def test_sentence_is_generated_from_the_card(tiny):
    r = R.resolve("a piano", tiny)
    assert r["sentence"] == "I have a piano. " + tiny.card("f.piano")["say"]
    assert "low-poly model by A, CC0" in r["sentence"] and "size from a published source" in r["sentence"]


def test_empty_and_gibberish_queries(tiny):
    assert R.resolve("", tiny)["refusal"] == "empty"
    assert R.resolve("Skippy,", tiny)["status"] == "not_found"
    assert R.resolve("qwertyuiop asdfgh", tiny)["card"] is None


def test_a_an():
    assert R.a_an("couch") == "a couch" and R.a_an("armchair") == "an armchair" and R.a_an("unicorn") == "a unicorn"


# ----------------------------------------------------------------------------------------------- resolver on the real local index
def test_dragon_never_returns_a_wrong_object_on_the_real_index(full):
    for q in ("dragon", "a dragon", "red dragon", "Skippy, add three dragons", "dragons", "drago"):
        r = R.resolve(q, full)
        assert r["card"] is None or r["status"] == "closest", q
    assert R.resolve("dragon", full)["status"] == "not_found"


def test_real_index_known_things(full):
    for q, kind in [("a couch", "couch"), ("refrigerator", "fridge"), ("television", "tv"), ("bookcases", "bookshelf"),
                    ("a car", "car"), ("a jumbo jet", "jumbo jet"), ("laptop", "laptop")]:
        r = R.resolve(q, full)
        assert r["kind"] == kind, (q, r["kind"], r["status"])
    r = R.resolve("a jumbo jet", full, room={"w": 10.7, "d": 9.1, "h": 2.7})
    assert r["status"] == "too_big"
    assert R.resolve("a jumbo jet", full, room={"outdoor": True})["status"] == "exact"
    assert R.resolve("a car", full)["card"]["size_m"]["d"] == 4.5          # not ShapeNet's 0.18 m


def test_real_index_airplane_is_not_placed_at_toy_size(full):
    r = R.resolve("airplane", full)
    assert r["status"] == "untrusted_size"                                   # ShapeNetSem 'plane' is 0.23 m: flagged, never placed


def test_every_curated_kind_resolves_to_itself_by_its_own_name(full):
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    for kid, k in cat["kinds"].items():
        for n in k["names"]:
            r = R.resolve(f"a {n}", full)
            assert r["status"] in ("exact", "alias") and r["rung"] == "0", (n, r["status"], r["kind"])
            assert r["kind"] == norm(kid) or r["status"] == "ambiguous" or r["card"]["visual"]["class"] == "curated", (n, r["kind"])


# ============================================================================================================ 7. the seed metrics
def test_seed_query_set_shape():
    q = json.loads((STAGE / "skippyverse" / "data" / "queries_seed.json").read_text(encoding="utf-8"))["queries"]
    assert len(q) >= 120
    neg = [x for x in q if not x["expect"]]
    assert 28 <= len(neg) <= 40
    tags = {x["tag"] for x in q}
    assert {"natural", "plural", "synonym", "typo", "hard", "negative"} <= tags
    assert all(x["tag"] == "negative" for x in neg)


def test_seed_metrics_meet_the_design_thresholds(full):
    q = json.loads((STAGE / "skippyverse" / "data" / "queries_seed.json").read_text(encoding="utf-8"))["queries"]
    m = metrics.evaluate(full, q)
    assert m["labels_not_in_index"] == []
    assert m["top3"] >= 0.85                                                # design increment 2: top-3 at least 85%
    assert m["negatives"]["false_accept"] <= 2                              # design: at most 2 wrong objects per ~30 unknown nouns
    assert m["by_tag"]["hard"]["top3"] == 0.0                               # paraphrases are NOT solved; this documents the limit
    assert m["by_tag"]["typo"]["top1"] >= 0.9 and m["by_tag"]["plural"]["top1"] >= 0.95


def test_metrics_would_catch_a_bad_resolver(full, monkeypatch):
    q = [{"q": "dragon", "expect": None, "tag": "negative"}, {"q": "couch", "expect": "couch", "tag": "natural"}]
    monkeypatch.setattr(metrics, "resolve", lambda *a, **k: {"status": "exact", "kind": "chair", "options": [], "candidates": ["chair"],
                                                              "sentence": ""})
    m = metrics.evaluate(full, q)
    assert m["negatives"]["false_accept"] == 1 and m["top1"] == 0.0
