"""Tests for the holodeck fidelity layer: every library prop has a REAL mass and a documented makeup, honestly sourced.

The layer is `stage/gmod/holodeck/fidelity.py`: it turns the sourced reference table (data/fidelity_research.json), the family chosen for
each model (data/fidelity_specs.json) and the engine's own geometry (data/measured.json) into data/fidelity.json and
misc/HOLODECK_FIDELITY.md, and build_catalog.py carries the adopted mass into the catalog. These tests check the arithmetic, that every
adopted figure lies inside its sourced range, that nothing keeps the engine's gameplay mass without a recorded reason, that the files are
what the inputs give, and that the bottom-up cross-check never moves a mass. No game, no network, no Mentat call (the densities are
Mentat's, recorded in the specs; `fidelity.py --verify-densities` re-asks it by hand).
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HOLODECK = ROOT / "stage" / "gmod" / "holodeck"
sys.path.insert(0, str(HOLODECK))

import build_catalog  # noqa: E402
import fidelity  # noqa: E402

SPECS = fidelity.load_specs()
RESEARCH = fidelity.load_research()
MEASURED = fidelity.load_measured()
FID = json.loads((HOLODECK / "data" / "fidelity.json").read_text(encoding="utf-8"))
CATALOG = json.loads((HOLODECK / "data" / "catalog.json").read_text(encoding="utf-8"))
VARIANTS = FID["variants"]
EPS = 1e-9

# The densities Mentat returned for tools/mentat_call.sh density '{"material": ...}' on 2026-10-10. A change here must be a deliberate
# re-query, not an edit to make a cross-check come out.
MENTAT_DENSITIES = {"wood_oak": 770.0, "wood_pine": 530.0, "glass": 2500.0, "plastic": 1100.0, "rubber": 1100.0, "iron": 7874.0,
                    "aluminum": 2700.0, "concrete": 2400.0, "copper": 8960.0, "graphite": 2266.0, "water": 999.972}
NO_MENTAT_DENSITY = ("steel", "fabric", "foam", "ceramic", "cardboard")


def usable_models() -> list[tuple[str, str]]:
    return [(kind, path) for kind, path in fidelity.library_models()
            if MEASURED.get(path) and not MEASURED[path].get("missing") and not MEASURED[path].get("no_physics")]


# ------------------------------------------------------------------------------------------------ synthetic worlds
def synth_family(low=10.0, high=20.0, ref=15.0, size=(1.0, 1.0, 1.0), topology="solid", scale="bbox", evidence="opened", crosscheck=None):
    fam = {"label": "test family", "research": "item", "reference": {"item": "test item", "mass_kg": ref, "mass_basis": "test", "weight_kind": "test",
                                                                  "size_m": list(size) if scale == "bbox" else None, "size_basis": "test"},
           "low_kg": low, "high_kg": high, "bounds_basis": "test", "topology": topology, "scale": scale, "evidence": evidence,
           "real": {"pieces": [1, 2], "interior": "closed", "text": "test"}}
    if scale == "fixed":
        fam["fixed_reason"] = "test reason"
        fam["fixed_short"] = "test"
    if crosscheck:
        fam["crosscheck"] = crosscheck
    else:
        fam["crosscheck_unavailable"] = "test"
    return fam


def synth_research():
    return {"item": {"confidence": "sourced", "materials": ["m"], "construction": "c", "hollow": False, "wall_thickness_mm": "t", "sources": ["https://example.org/x"]}}


def synth_measured(path, size_m, mass=100.0, hull_volume_m3=None, hull_surface_m2=1.0):
    hu = fidelity.HU_M
    vol = size_m[0] * size_m[1] * size_m[2] if hull_volume_m3 is None else hull_volume_m3
    return {path: {"model": path, "mass": mass, "mins": [0, 0, 0], "maxs": [d / hu for d in size_m], "material": "wood",
                   "phys": {"volume": vol / hu ** 3, "surface_area": hull_surface_m2 / hu ** 2, "convexes": 1, "triangles": 12}}}


def synth_compute(families, variants, measured):
    specs = {"method": {"guard_factor": 2.0}, "densities": {"values": {"iron": {"kg_m3": 7874.0}}}, "families": families, "variants": variants}
    models = [("test", p) for p in variants]
    return fidelity.compute(specs, measured, synth_research(), models)


# ------------------------------------------------------------------------------------------------ the arithmetic
def test_shells_scale_with_area_and_solids_with_volume():
    """k is the exponent on the linear scale: V^1 for solids and frame-and-panel, V^(2/3) (surface area) for thin shells."""
    assert fidelity.K_BY_TOPOLOGY == {"solid": 3, "frame-and-panel": 3, "shell": 2}
    for topology, factor in (("solid", 8.0), ("frame-and-panel", 8.0), ("shell", 4.0)):
        fam = synth_family(low=0.001, high=1e9, ref=10.0, size=(1, 1, 1), topology=topology)
        ev = fidelity.evaluate(fam, 8.0)                      # a box of twice the length: volume x8
        assert ev["s"] == pytest.approx(2.0)
        assert ev["scaled_kg"] == pytest.approx(10.0 * factor), topology
        assert ev["clamped"] is None
    # the literal reading (volume ratio)^k would give 8^3 for a solid: far outside any sourced figure
    assert fidelity.evaluate(synth_family(low=0.001, high=1e9, ref=10.0), 8.0)["scaled_kg"] < 100.0


def test_a_reference_scaled_to_itself_is_unchanged():
    for topology in ("solid", "shell"):
        fam = synth_family(ref=15.0, size=(2, 1, 0.5), topology=topology)
        ev = fidelity.evaluate(fam, 1.0)
        assert ev["scaled_kg"] == pytest.approx(15.0)
        assert ev["adopted_kg"] == pytest.approx(15.0)


def test_the_clamp_holds_a_scaled_mass_inside_the_sourced_range():
    fam = synth_family(low=10.0, high=20.0, ref=15.0, size=(1, 1, 1))
    small = fidelity.evaluate(fam, 0.2)                       # x0.2 of a solid: 3 kg
    big = fidelity.evaluate(fam, 5.0)                         # x5: 75 kg
    mid = fidelity.evaluate(fam, 1.0)
    assert (small["clamped"], small["adopted_kg"]) == ("low", 10.0)
    assert (big["clamped"], big["adopted_kg"]) == ("high", 20.0)
    assert mid["clamped"] is None
    assert small["outside_factor"] == pytest.approx(10.0 / 3.0)
    assert big["outside_factor"] == pytest.approx(75.0 / 20.0)
    assert mid["outside_factor"] == 1.0


def test_fixed_scale_uses_the_reference_mass_as_is():
    fam = synth_family(low=1.0, high=3.0, ref=2.0, scale="fixed")
    for vol in (0.001, 1.0, 100.0):
        ev = fidelity.evaluate(fam, vol)
        assert ev["adopted_kg"] == 2.0 and ev["clamped"] is None and ev["ref_volume_m3"] is None
    # a variant may override its family's scale
    fam = synth_family(low=1.0, high=1e6, ref=2.0, size=(1, 1, 1))
    assert fidelity.evaluate(fam, 27.0, "fixed")["adopted_kg"] == 2.0
    assert fidelity.evaluate(fam, 27.0)["adopted_kg"] == pytest.approx(54.0)


@pytest.mark.parametrize("fid", sorted(SPECS["families"]))
def test_scaling_is_monotone_in_volume(fid):
    """A bigger box never gets a lighter prop of the same family, before or after the clamp or the rounding."""
    fam = SPECS["families"][fid]
    vols = [10 ** (e / 4.0) for e in range(-24, 17)]          # 1e-6 .. 1e4 m3
    scaled = [fidelity.evaluate(fam, v) for v in vols]
    for a, b in zip(scaled, scaled[1:]):
        assert b["scaled_kg"] >= a["scaled_kg"] - EPS, fid
        assert b["adopted_kg"] >= a["adopted_kg"] - EPS, fid
    for ev in scaled:
        assert fam["low_kg"] - EPS <= ev["adopted_kg"] <= fam["high_kg"] + EPS, fid


def test_catalog_masses_are_monotone_in_box_volume_within_a_family():
    by_family: dict[str, list[tuple[float, float, str]]] = {}
    for path, r in VARIANTS.items():
        if r["status"] == "audited" and r["family"] and r["scale_mode"] == "bbox":
            by_family.setdefault(r["family"], []).append((r["geometry"]["box_volume_m3"], r["mass_kg"], path))
    assert by_family
    for fid, rows in by_family.items():
        rows.sort()
        for (v1, m1, p1), (v2, m2, p2) in zip(rows, rows[1:]):
            assert m2 >= m1 - EPS, f"{fid}: {p2} ({v2:.3f} m3, {m2} kg) is lighter than the smaller {p1} ({v1:.3f} m3, {m1} kg)"


def test_confidence_steps_down_with_the_evidence_the_fit_and_the_clamp():
    assert fidelity.confidence("opened", "good", False) == "high"
    assert fidelity.confidence("opened", "partial", False) == "medium"
    assert fidelity.confidence("opened", "good", True) == "medium"
    assert fidelity.confidence("mixed", "good", False) == "medium"
    assert fidelity.confidence("mixed", "partial", False) == "low"
    assert fidelity.confidence("search-summary", "good", False) == "low"
    assert fidelity.confidence("single-source", "good", False) == "low"
    assert fidelity.confidence("opened", "partial", True) == "low"
    assert fidelity.confidence("opened", "good", False, True) == "medium"           # a fixed scale does not use the model's size
    assert fidelity.confidence("mixed", "good", False, True) == "low"


# ------------------------------------------------------------------------------------------------ compute() on synthetic worlds
def test_compute_audits_clamps_and_refuses_far_extrapolations():
    fams = {"f": synth_family(low=10.0, high=20.0, ref=15.0, size=(1, 1, 1))}
    sizes = {"inside": (1.0, 1.0, 1.0), "near": (0.7, 0.7, 0.7), "far": (0.4, 0.4, 0.4)}          # x15, x5.1 (clamped to 10), x0.96
    variants = {f"m/{k}.mdl": {"family": "f", "fit": "good"} for k in sizes}
    measured = {}
    for k, s in sizes.items():
        measured.update(synth_measured(f"m/{k}.mdl", s, mass=100.0))
    out = synth_compute(fams, variants, measured)["variants"]
    inside, near, far = out["m/inside.mdl"], out["m/near.mdl"], out["m/far.mdl"]
    assert inside["status"] == "audited" and inside["mass_kg"] == 15.0 and inside["clamped"] is None
    assert near["status"] == "audited" and near["clamped"] == "low" and near["mass_kg"] == 10.0     # 15 x 0.343 = 5.1 kg, 1.96x below
    assert far["status"] == "unaudited" and far["mass_kg"] == 100.0                                 # 15 x 0.064 = 0.96 kg, 10x below
    assert far["extrapolated_kg"] == pytest.approx(0.96, rel=1e-3)
    assert "not adopted" in far["unaudited_reason"] and far["mass_source"].startswith("unaudited")
    assert far["mass_low_kg"] is None and far["mass_high_kg"] is None and far["confidence"] is None


def test_compute_keeps_the_engine_mass_for_an_unaudited_variant_and_says_why():
    out = synth_compute({}, {"m/x.mdl": {"unaudited": "no family covers it", "hint": "about a kilo"}}, synth_measured("m/x.mdl", (1, 1, 1), mass=123.0))["variants"]["m/x.mdl"]
    assert out["status"] == "unaudited" and out["mass_kg"] == 123.0 and out["unaudited_reason"] == "no family covers it" and out["hint"] == "about a kilo"
    assert "123" in out["mass_source"] and "no family covers it" in out["mass_source"]


def test_compute_flags_a_model_nobody_decided_about():
    out = synth_compute({}, {}, {})["variants"]
    assert out == {}
    out = fidelity.compute({"method": {"guard_factor": 2.0}, "densities": {"values": {}}, "families": {}, "variants": {}},
                           synth_measured("m/new.mdl", (1, 1, 1), mass=7.0), synth_research(), [("test", "m/new.mdl")])["variants"]["m/new.mdl"]
    assert out["status"] == "unaudited" and out["mass_kg"] == 7.0 and "no decision" in out["unaudited_reason"]


def test_compute_sums_a_composite_and_its_bounds():
    fams = {"a": synth_family(low=1.0, high=2.0, ref=1.5, scale="fixed"), "b": synth_family(low=10.0, high=20.0, ref=15.0, size=(1, 1, 1))}
    variants = {"m/c.mdl": {"parts": [{"family": "a", "scale": "fixed"}, {"family": "b", "scale": "bbox"}], "fit": "partial"}}
    out = synth_compute(fams, variants, synth_measured("m/c.mdl", (1, 1, 1)))["variants"]["m/c.mdl"]
    assert out["status"] == "audited" and out["mass_kg"] == pytest.approx(16.5)
    assert (out["mass_low_kg"], out["mass_high_kg"]) == (11.0, 22.0)
    assert out["scale_mode"] == "composite" and out["parts"] == ["a", "b"]


def test_fixed_variant_override_ignores_the_box():
    fams = {"f": synth_family(low=1.0, high=1e6, ref=2.0, size=(1, 1, 1))}
    variants = {"m/p.mdl": {"family": "f", "fit": "good", "scale": "fixed", "scale_note": "a pipe fills the box", "scale_short": "pipe"}}
    out = synth_compute(fams, variants, synth_measured("m/p.mdl", (3, 3, 3)))["variants"]["m/p.mdl"]
    assert out["mass_kg"] == 2.0 and out["scale_mode"] == "fixed" and "pipe" in out["mass_source"]


def test_the_cross_check_is_reported_and_never_moves_the_mass():
    cc = {"mode": "shell", "layers": [{"material": "iron", "thickness_mm": 1.0, "thickness_source": "test", "thickness_evidence": "opened"}],
          "excluded": "nothing", "proxy": None}
    with_cc = {"f": synth_family(low=1.0, high=100.0, ref=10.0, size=(1, 1, 1), topology="shell", crosscheck=cc)}
    without = {"f": synth_family(low=1.0, high=100.0, ref=10.0, size=(1, 1, 1), topology="shell")}
    variants = {"m/s.mdl": {"family": "f", "fit": "good"}}
    measured = synth_measured("m/s.mdl", (1, 1, 1), hull_surface_m2=2.0)
    a = synth_compute(with_cc, variants, measured)["variants"]["m/s.mdl"]
    b = synth_compute(without, variants, measured)["variants"]["m/s.mdl"]
    assert a["mass_kg"] == b["mass_kg"] and a["mass_source"] == b["mass_source"]
    assert a["cross_check"]["available"] and a["cross_check"]["bottom_up_kg"] == pytest.approx(7874.0 * 2.0 * 0.001, rel=1e-5)
    assert a["cross_check"]["sourced_over_bottom_up"] == pytest.approx(10.0 / (7874.0 * 0.002), rel=1e-4)
    assert not b["cross_check"]["available"]


def test_bottom_up_shell_and_solid():
    dens = {"iron": {"kg_m3": 7874.0}, "water": {"kg_m3": 1000.0}}
    shell = fidelity.bottom_up({"mode": "shell", "layers": [{"material": "iron", "thickness_mm": 2.0}], "excluded": "x"}, 3.0, 99.0, dens)
    assert shell["bottom_up_kg"] == pytest.approx(7874.0 * 3.0 * 0.002)
    two = fidelity.bottom_up({"mode": "shell", "layers": [{"material": "iron", "thickness_mm": 1.0}, {"material": "water", "thickness_mm": 5.0}]}, 2.0, 99.0, dens)
    assert two["bottom_up_kg"] == pytest.approx(2.0 * (7874.0 * 0.001 + 1000.0 * 0.005))
    solid = fidelity.bottom_up({"mode": "solid", "layers": [{"material": "water"}]}, 99.0, 0.5, dens)
    assert solid["bottom_up_kg"] == pytest.approx(500.0)
    with pytest.raises(ValueError):
        fidelity.bottom_up({"mode": "wafer", "layers": [{"material": "iron"}]}, 1.0, 1.0, dens)


def test_complexity_verdicts():
    real = lambda lo, hi, interior: {"pieces": [lo, hi], "interior": interior}          # noqa: E731
    assert fidelity.complexity(real(6, 10, "open"), 1, 12)["verdict"] == "much simpler"          # one hull for a bookcase
    assert fidelity.complexity(real(1, 2, "closed"), 1, 12)["verdict"] == "comparable"           # one hull for a closed cabinet is fine
    assert fidelity.complexity(real(5, 9, "open"), 2, 24)["verdict"] == "much simpler"
    assert fidelity.complexity(real(5, 9, "open"), 4, 24)["verdict"] == "simpler"
    assert fidelity.complexity(real(5, 9, "open"), 7, 24)["verdict"] == "comparable"
    assert fidelity.complexity(real(1, 2, "closed"), 3, 24)["verdict"] == "more complex"
    assert fidelity.complexity(real(1, 2, "closed"), 7, 24)["verdict"] == "much more complex"
    dense = fidelity.complexity(real(1, 5, "closed"), 4, 400)
    assert "dense collision mesh" in dense["note"]
    assert fidelity.complexity(real(1, 5, "closed"), None, None)["verdict"] == "unknown"


# ------------------------------------------------------------------------------------------------ the committed data
def test_every_library_model_has_an_explicit_decision_and_an_entry():
    for kind, path in usable_models():
        assert path in SPECS["variants"], f"{path} has no decision in data/fidelity_specs.json (a family, or 'unaudited' with a reason)"
        assert path in VARIANTS, f"{path} is missing from data/fidelity.json (run fidelity.py)"
        assert VARIANTS[path]["kind"] == kind
    assert set(VARIANTS) == {p for _, p in usable_models()}, "fidelity.json holds a model the library does not"
    assert set(SPECS["variants"]) == {p for _, p in usable_models()}, "fidelity_specs.json decides about a model the library does not have"


def test_every_variant_has_a_mass_source_in_the_json_and_the_catalog():
    for path, r in VARIANTS.items():
        assert isinstance(r["mass_source"], str) and len(r["mass_source"]) > 40, path
        assert r["mass_source"].startswith(("fidelity/", "unaudited")), path
    for kid, k in CATALOG["kinds"].items():
        for v in k["variants"]:
            assert isinstance(v["mass_source"], str) and v["mass_source"] == VARIANTS[v["model"]]["mass_source"], v["model"]
            assert v["mass_status"] in ("audited", "unaudited")


def test_each_variant_records_the_engine_mass_and_the_ratio_to_it():
    for path, r in VARIANTS.items():
        assert r["engine_mass_kg"] == MEASURED[path]["mass"], path
        assert r["mass_over_engine"] == pytest.approx(r["mass_kg"] / r["engine_mass_kg"], rel=1e-5), path
        if r["status"] == "unaudited":
            assert r["mass_over_engine"] == 1.0


def test_families_name_the_materials_that_make_them_up():
    for fid, fam in SPECS["families"].items():
        assert fam["materials"] and all(isinstance(m, str) and m.strip() for m in fam["materials"]), fid
        assert FID["families"].get(fid, {"makeup": {"materials": fam["materials"]}})["makeup"]["materials"] == fam["materials"], fid
    for path, r in VARIANTS.items():
        if r["status"] == "audited":
            assert r["makeup"] and ";" in r["makeup"], path


def test_every_adopted_mass_lies_inside_its_sourced_range():
    n = 0
    for path, r in VARIANTS.items():
        if r["status"] != "audited":
            continue
        n += 1
        assert r["mass_low_kg"] - EPS <= r["mass_kg"] <= r["mass_high_kg"] + EPS, (path, r["mass_low_kg"], r["mass_kg"], r["mass_high_kg"])
        assert r["mass_low_kg"] > 0 and r["mass_kg"] > 0
        parts = r["parts"] or [r["family"]]
        assert r["mass_low_kg"] == pytest.approx(sum(SPECS["families"][f]["low_kg"] for f in parts))
        assert r["mass_high_kg"] == pytest.approx(sum(SPECS["families"][f]["high_kg"] for f in parts))
        assert r["confidence"] in ("high", "medium", "low") and r["fit"] in ("good", "partial")
    assert n >= 100, "the layer is meant to cover nearly the whole library"


def test_the_catalog_carries_the_fidelity_masses():
    for kid, k in CATALOG["kinds"].items():
        for v in k["variants"]:
            r = VARIANTS[v["model"]]
            assert v["mass"] == r["mass_kg"], v["model"]
            assert v["engine_mass"] == MEASURED[v["model"]]["mass"], v["model"]
            assert v["mass_status"] == r["status"]
            assert v["makeup"] == r["makeup"]
            if r["status"] == "audited":
                assert v["mass_low"] <= v["mass"] <= v["mass_high"]
                assert v["mass_confidence"] == r["confidence"]
            else:
                assert v["mass"] == v["engine_mass"] and v["mass_low"] is None and v["mass_high"] is None and v["mass_confidence"] is None
            # every field the layout code relies on is still there
            for key in ("model", "front", "symmetric", "front_source", "top", "w", "d", "h", "mins", "maxs", "cx", "cy", "material", "skins"):
                assert key in v, (v["model"], key)


def test_no_mass_is_the_engines_when_a_source_exists_unless_the_reason_is_recorded():
    for path, r in VARIANTS.items():
        same = abs(r["mass_kg"] - r["engine_mass_kg"]) <= EPS * max(1.0, abs(r["engine_mass_kg"]))
        if r["status"] == "audited":
            assert not same, f"{path}: the adopted mass is the engine's and nothing records why"
            assert r["family"] or r["parts"], path
        else:
            assert same, path
            assert r["unaudited_reason"] and len(r["unaudited_reason"]) > 20, f"{path}: unaudited with no reason"
            assert "engine mass" in r["mass_source"]
            if r.get("family") or r.get("parts"):
                # a family exists, so the reason must be the recorded extrapolation guard
                assert r["extrapolated_kg"] is not None and r["outside_factor"] > SPECS["method"]["guard_factor"], path
    # a model that is unaudited by decision names what would be needed
    for path, spec in SPECS["variants"].items():
        if "unaudited" in spec:
            assert spec["unaudited"].strip(), path


def test_the_engines_absurd_gameplay_masses_are_gone():
    """The engine's locker is 2000 kg, its stoves 1500 to 5000, its bathtub 900, its mug up to 4: the real ones are a few tens or hundreds."""
    def masses(kind):
        return [r["mass_kg"] for r in VARIANTS.values() if r["kind"] == kind]
    assert max(masses("locker")) < 40
    assert max(masses("stove")) < 90
    assert max(masses("fridge")) <= 119
    assert 100 < min(masses("bathtub")) and max(masses("bathtub")) < 170
    assert max(masses("mug")) < 0.5 and min(masses("mug")) > 0.2
    assert 10 < min(masses("mattress")) and max(masses("mattress")) < 30
    assert max(masses("chair")) <= 12 and max(masses("trash can")) <= 6.2 + EPS
    assert max(masses("couch")) <= 90 and max(masses("armchair")) <= 42
    # the other direction: a few engine masses were far too LIGHT
    bench = VARIANTS["models/props_trainstation/bench_indoor001a.mdl"]
    assert bench["engine_mass_kg"] == 1.0 and bench["mass_kg"] >= 40


def test_no_prop_is_denser_than_solid_iron_the_densest_thing_mentat_lists():
    """A physical bound from Mentat's own table: the adopted mass cannot exceed solid iron filling the engine's collision hull."""
    iron = SPECS["densities"]["values"]["iron"]["kg_m3"]
    for path, r in VARIANTS.items():
        if r["status"] == "audited":
            bound = iron * r["geometry"]["hull_volume_m3"]
            assert r["mass_kg"] <= bound, f"{path}: {r['mass_kg']} kg in a {r['geometry']['hull_volume_m3']} m3 hull is denser than iron"


@pytest.mark.parametrize("fid", sorted(SPECS["families"]))
def test_family_is_well_formed(fid):
    fam = SPECS["families"][fid]
    ref = fam["reference"]
    assert fam["research"] in RESEARCH, f"{fid}: no such research item"
    assert fam["topology"] in fidelity.K_BY_TOPOLOGY and fam["evidence"] in fidelity.EVIDENCE_STEPS and fam["scale"] in ("bbox", "fixed")
    assert 0 < fam["low_kg"] <= ref["mass_kg"] <= fam["high_kg"], f"{fid}: the reference mass is outside its own sourced range"
    assert ref["item"].strip() and ref["mass_basis"].strip() and ref["weight_kind"].strip() and fam["bounds_basis"].strip()
    if fam["scale"] == "bbox":
        assert len(ref["size_m"]) == 3 and all(d > 0 for d in ref["size_m"]), fid
    else:
        assert fam["fixed_reason"].strip() and fam["fixed_short"].strip(), f"{fid}: a fixed scale needs its reason"
    pieces = fam["real"]["pieces"]
    assert 1 <= pieces[0] <= pieces[1] and fam["real"]["interior"] in ("solid", "closed", "open")
    # a cross-check or the reason there is none
    assert bool(fam.get("crosscheck")) != bool(fam.get("crosscheck_unavailable")), fid
    # the sourced range sits inside what the research item says for the kind, or says why not (checked loosely: never absurdly outside)
    r = RESEARCH[fam["research"]]
    assert fam["low_kg"] >= 0.2 * r["mass_kg_low"] and fam["high_kg"] <= 5 * r["mass_kg_high"], f"{fid}: bounds far from the research item's"


def test_every_research_item_is_used_or_set_aside_with_a_reason():
    used = {f["research"] for f in SPECS["families"].values()}
    unused = set(RESEARCH) - used
    assert unused == set(SPECS["unused_research"]), (unused, set(SPECS["unused_research"]))
    for label, why in SPECS["unused_research"].items():
        assert why.strip()
    for label, item in RESEARCH.items():
        assert item["sources"] and all(s.startswith("http") for s in item["sources"]), label
        assert not any(".example" in s for s in item["sources"]), f"{label}: a placeholder source"
        assert item["mass_kg_low"] <= item["mass_kg_typical"] <= item["mass_kg_high"], label
    for fid, fam in SPECS["families"].items():
        used_in_a_variant = any(fid == e.get("family") or fid in [p["family"] for p in e.get("parts", [])] for e in SPECS["variants"].values())
        assert used_in_a_variant, f"family {fid} is used by no model"


def test_every_density_is_mentats_and_none_is_one_it_lacks():
    table = SPECS["densities"]["values"]
    assert {k: v["kg_m3"] for k, v in table.items()} == MENTAT_DENSITIES
    assert all(v["mentat_source"].startswith("sigma-ground") for v in table.values())
    assert set(SPECS["densities"]["no_mentat_density"]) == set(NO_MENTAT_DENSITY)
    for fid, fam in SPECS["families"].items():
        for layer in (fam.get("crosscheck") or {}).get("layers", []):
            assert layer["material"] in table, f"{fid}: {layer['material']} is not a Mentat density"
            assert layer["material"] not in NO_MENTAT_DENSITY
            assert layer["thickness_mm"] > 0 and layer["thickness_source"] and layer["thickness_evidence"] in ("opened", "search-summary")
        if fam.get("crosscheck") and "iron" in [l["material"] for l in fam["crosscheck"]["layers"]]:
            assert "iron" in (fam["crosscheck"].get("proxy") or "") or "steel" not in json.dumps(fam["crosscheck"]).replace("steel -> iron", ""), \
                f"{fid}: steel described but the proxy is not stated"


def test_cross_checks_in_the_json_are_the_densities_times_the_engine_hull():
    n = 0
    for path, r in VARIANTS.items():
        cc = r["cross_check"]
        if not cc or not cc["available"]:
            continue
        n += 1
        sa = r["geometry"]["hull_surface_m2"]
        expect = sum(l["density_kg_m3"] * sa * l["thickness_mm"] / 1000.0 for l in cc["layers"])
        assert cc["bottom_up_kg"] == pytest.approx(expect, rel=1e-4, abs=1e-5), path
        assert cc["sourced_over_bottom_up"] == pytest.approx(cc["compared_kg"] / cc["bottom_up_kg"], rel=1e-4)
        for l in cc["layers"]:
            assert l["density_kg_m3"] == MENTAT_DENSITIES[l["material"]]
        assert cc["compared_kg"] == (r["mass_kg"] if r["status"] == "audited" else r["extrapolated_kg"])
    assert n >= 15, "the families with a sourced thickness and a Mentat density should be cross-checked"


def test_the_cross_check_does_not_set_any_committed_mass():
    stripped = copy.deepcopy(SPECS)
    for fam in stripped["families"].values():
        if fam.get("crosscheck"):
            fam.pop("crosscheck")
            fam["crosscheck_unavailable"] = "removed by the test"
    out = fidelity.compute(stripped, MEASURED, RESEARCH)["variants"]
    for path, r in VARIANTS.items():
        assert out[path]["mass_kg"] == r["mass_kg"] and out[path]["status"] == r["status"], path


# ------------------------------------------------------------------------------------------------ reproducibility
def test_fidelity_json_is_what_the_inputs_give():
    fresh = json.loads(fidelity.render_json(fidelity.compute()))
    assert fresh == FID, "run: python stage/gmod/holodeck/fidelity.py"


def test_the_report_is_what_the_json_gives():
    report = (ROOT / "misc" / "HOLODECK_FIDELITY.md").read_text(encoding="utf-8")
    assert report == fidelity.render_report(FID), "run: python stage/gmod/holodeck/fidelity.py"
    assert report == fidelity.render_report(fidelity.compute())
    for path in VARIANTS:
        assert path.replace("models/", "") in report, f"{path} is missing from the report"
    for heading in ("## Unaudited", "## Held at a sourced bound", "## Bottom-up cross-check", "## Complexity audit", "## Surface materials",
                    "## Families", "## Honest limits"):
        assert heading in report


def test_fidelity_check_command_agrees(capsys):
    assert fidelity.main(["--check"]) == 0
    assert "current" in capsys.readouterr().out


def test_the_catalog_is_built_from_the_committed_fidelity_json():
    built = build_catalog.build()
    assert not built["problems"], built["problems"]
    for kid, k in built["kinds"].items():
        for v in k["variants"]:
            assert v["mass"] == VARIANTS[v["model"]]["mass_kg"]
    assert json.loads(json.dumps({k: v for k, v in built.items() if k != "problems"})) == CATALOG


def test_a_model_without_a_fidelity_entry_is_a_catalog_problem(monkeypatch):
    real = fidelity.load()
    cut = copy.deepcopy(real)
    gone = next(iter(cut["variants"]))
    del cut["variants"][gone]
    monkeypatch.setattr(fidelity, "load", lambda: cut)
    problems = build_catalog.build()["problems"]
    assert any(gone in p and "fidelity" in p for p in problems), problems


# ------------------------------------------------------------------------------------------------ the complexity audit
def test_the_complexity_audit_covers_every_variant_with_a_family():
    n = 0
    for path, r in VARIANTS.items():
        c = r["complexity"]
        if r["family"] or r["parts"]:
            n += 1
            assert c and c["verdict"] in ("comparable", "simpler", "much simpler", "more complex", "much more complex", "unknown"), path
            assert c["engine_convexes"] == MEASURED[path]["phys"]["convexes"] and c["engine_triangles"] == MEASURED[path]["phys"]["triangles"]
        else:
            assert c is None, path
    assert n >= 120
    # the headline findings the report states
    for model in ("cs_office/bookshelf1", "cs_office/bookshelf2", "cs_office/bookshelf3", "cs_havana/bookcase_small"):
        c = VARIANTS[f"models/props/{model}.mdl"]["complexity"]
        assert c["engine_convexes"] == 1 and c["verdict"] == "much simpler", model
    assert VARIANTS["models/props/de_tides/patio_table2.mdl"]["complexity"]["verdict"] == "much more complex"
    toilet = VARIANTS["models/props/cs_militia/toilet.mdl"]["complexity"]
    assert "dense collision mesh" in toilet["note"]


def test_the_engine_geometry_in_the_json_is_the_measured_geometry():
    hu = fidelity.HU_M
    for path, r in VARIANTS.items():
        m = MEASURED[path]
        g = r["geometry"]
        assert r["engine_mass_kg"] == m["mass"]
        assert g["hull_volume_m3"] == pytest.approx(m["phys"]["volume"] * hu ** 3, rel=1e-5, abs=1e-6)
        assert g["hull_surface_m2"] == pytest.approx(m["phys"]["surface_area"] * hu ** 2, rel=1e-5, abs=1e-6)
        ex = [(m["maxs"][i] - m["mins"][i]) * hu for i in range(3)]
        assert g["box_volume_m3"] == pytest.approx(ex[0] * ex[1] * ex[2], rel=1e-5, abs=1e-6)
        assert math.isclose(g["engine_density_kg_m3"], m["mass"] / (m["phys"]["volume"] * hu ** 3), rel_tol=1e-4)
