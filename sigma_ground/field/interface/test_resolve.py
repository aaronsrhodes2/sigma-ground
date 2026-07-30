"""material_profile(name) — names resolve to CITED/DERIVED properties, never
invented. Mercury must read as a mirror; glass must carry fracture data; an
unknown must come back flagged; every shipped catalog material must be cited.
"""
import json

import pytest

from sigma_ground.field.interface import resolve
from sigma_ground.field.interface.resolve import material_profile, MaterialFact, _JSON_PATH


def _is_fact(x):
    return isinstance(x, MaterialFact)


def test_mercury_is_a_reflective_liquid_metal():
    p = material_profile("mercury")
    assert p["identified"] and p["material_class"] == "liquid"
    # reflective, NOT the grey fallback — real Hg reflects ~0.74-0.79
    assert "reflectance" in p and _is_fact(p["reflectance"])
    assert p["reflectance"].value > 0.6
    assert not p["reflectance"].estimated          # derived from Fresnel n+ik
    # the liquid props travel with it
    assert p["density"].value > 13000 and "surface_tension" in p
    # alias resolves too
    assert material_profile("quicksilver")["canonical"] == "mercury"


def test_glass_carries_fracture_data_and_is_brittle():
    p = material_profile("glass")
    assert "K_Ic" in p and "sigma_UTS" in p
    assert p["is_ductile"].value is False
    assert "n_refractive" in p                      # refracts (transparent)
    # tempered glass is ~4x stronger than soda-lime, different fracture
    soda = material_profile("soda-lime glass")
    temp = material_profile("tempered glass")
    assert temp["sigma_UTS"].value > 3 * soda["sigma_UTS"].value


def test_unknown_name_is_flagged_not_fabricated():
    p = material_profile("flarbleglomp")
    assert p["identified"] is False
    assert p["density"].estimated                   # flagged, low confidence
    assert p["density"].confidence <= 0.3


def test_metal_resolves_fully_from_physics():
    p = material_profile("gold")
    for key in ("density", "E", "K", "sigma_UTS", "K_Ic", "reflectance", "color"):
        assert key in p, f"gold missing {key}"
    assert all(not p[k].estimated for k in ("density", "reflectance"))


def test_catalog_material_resolves_with_citation():
    # an everyday material the physics modules lack — comes from the cited JSON
    p = material_profile("balsa")
    assert p["identified"] and p["canonical"] == "wood_balsa"
    assert p["density"].value < 300 and not p["density"].estimated
    assert "youngs" not in p  # sanity: key is 'E', not a typo
    assert "E" in p


def test_every_catalog_material_is_cited_and_complete():
    doc = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
    _STRENGTH = ("youngs_modulus_pa", "sigma_UTS_pa", "K_Ic_pa_sqrtm")
    _OPTICAL = ("n_refractive", "reflectance")
    # structural classes must carry a strength/stiffness; soft textiles (whose
    # tensile strength is weave-dependent — we won't fabricate it) carry an
    # optical instead. Every material has density + >=1 cited extra property.
    _STRUCTURAL = ("wood", "plastic", "glass", "metal", "ceramic", "stone", "foam")
    for key, rec in doc["materials"].items():
        cls = rec.get("class")
        assert "density_kg_m3" in rec, f"{key}: no density"
        has_strength = any(s in rec for s in _STRENGTH)
        has_optical = any(o in rec for o in _OPTICAL)
        has_fluid = "viscosity_pa_s" in rec or "surface_tension_n_m" in rec
        assert has_strength or has_optical or has_fluid, f"{key}: no property beyond density"
        if cls in _STRUCTURAL:
            assert has_strength, f"{key}: structural material needs a strength/stiffness"
        if cls == "fabric":
            assert has_optical, f"{key}: fabric needs an optical (reflectance)"
        # every value cell is cited (real source code + confidence present)
        for prop, cell in rec.items():
            if isinstance(cell, dict) and "v" in cell:
                assert cell.get("src") and cell.get("src") != "estimated", f"{key}.{prop} uncited"
                assert "c" in cell, f"{key}.{prop} no confidence"


# ── TPSX tier (0a) ───────────────────────────────────────────────────────

def test_tpsx_tier_overrides_density_for_aliased_metals():
    p = material_profile("gold")
    assert p["density"].value == pytest.approx(19300)
    assert "TPSX" in p["density"].source
    assert not p["density"].estimated
    # thermal properties are new, TPSX-only additions
    assert p["specific_heat_j_kgk"].value == pytest.approx(128.0)
    assert p["thermal_conductivity_w_mk"].value == pytest.approx(318.0)


def test_materials_without_tpsx_entry_unchanged():
    # silicon is explicitly excluded from the TPSX alias table (the
    # "Silicone A/B" false-friend) -- density must still come from
    # surface.MATERIALS, exactly as before this feature existed.
    p = material_profile("silicon")
    assert p["density"].source == "surface.MATERIALS"
    assert "thermal_conductivity_w_mk" not in p
    assert "specific_heat_j_kgk" not in p


def test_tpsx_mechanical_properties_stay_inert_for_aliased_metals():
    # documented, verified fact: TPSX's E/G/sigma_UTS data belongs to
    # aerospace TPS composites, not our metal alias set -- gold's E must
    # still come from mechanical.py, not TPSX.
    p = material_profile("gold")
    assert "mechanical.youngs_modulus" in p["E"].source
    assert "TPSX" not in p["E"].source


# ── Materials Project tier (2b) ──────────────────────────────────────────

def test_mp_elastic_pa_derives_E_via_isotropic_identity():
    entry = {"bulk_modulus_vrh": 140.0, "shear_modulus_vrh": 48.0}
    out = resolve._mp_elastic_pa(entry)
    assert out["K"] == pytest.approx(140.0e9)
    assert out["G"] == pytest.approx(48.0e9)
    expected_E = 9.0 * 140e9 * 48e9 / (3.0 * 140e9 + 48e9)
    assert out["E"] == pytest.approx(expected_E)


def test_mp_elastic_pa_skips_missing_or_nonpositive_moduli():
    assert resolve._mp_elastic_pa({}) == {}
    # a negative K (the documented MP-artifact case, e.g. Zn's canonical
    # entry) is dropped, but a valid positive G alone is still kept -- E is
    # only derived when BOTH K and G are present and positive
    out = resolve._mp_elastic_pa({"bulk_modulus_vrh": -5.0, "shear_modulus_vrh": 10.0})
    assert out == {"G": 10.0e9}
    assert resolve._mp_elastic_pa({"bulk_modulus_vrh": 10.0, "shear_modulus_vrh": None}) == {"K": 10.0e9}


def test_mp_tier_does_not_override_mechanicals_own_E_for_covered_metals(monkeypatch):
    # Deliberate design correction (see resolve.py's 2b) comment): MP's
    # automatic lowest-energy-polymorph selection was live-verified to
    # return physically unrepresentative elastic tensors for several of our
    # metals (aluminum +68% vs textbook E, gold -53%). So for any canon
    # mechanical.py already covers, a synthetic MP cache entry must NOT
    # win -- mechanical.py's measured-Poisson-anchored E stays authoritative.
    fake_cache = {"copper": {"bulk_modulus_vrh": 999.0, "shear_modulus_vrh": 999.0,
                             "poisson_ratio": 0.999, "source": "fake-mp-entry"}}
    monkeypatch.setattr(resolve, "_MP_CACHE", fake_cache)
    p = material_profile("copper")
    assert "fake-mp-entry" not in p["E"].source
    assert p["poisson_ratio"].value != pytest.approx(0.999)
    assert "mechanical.MECHANICAL_DATA" in p["poisson_ratio"].source
    # sigma_UTS / K_Ic are untouched either way -- MP has no fracture data
    assert "stress.STRESS_DATA" in p["sigma_UTS"].source


def test_mp_tier_fills_a_genuine_gap_when_mechanical_data_lacks_the_material(monkeypatch):
    # For a canon NOT in MECHANICAL_DATA/MATERIALS, the MP tier is the ONLY
    # source of E/G/K/poisson_ratio -- prove it actually fires end-to-end.
    fake_cache = {"unobtainium_test_alloy": {
        "bulk_modulus_vrh": 140.0, "shear_modulus_vrh": 48.0, "poisson_ratio": 0.34,
        "source": "Materials Project (DFT-computed elastic tensor, mp-fake)"}}
    monkeypatch.setattr(resolve, "_MP_CACHE", fake_cache)
    p = material_profile("unobtainium_test_alloy")
    assert p["K"].value == pytest.approx(140.0e9)
    assert p["G"].value == pytest.approx(48.0e9)
    assert p["E"].value == pytest.approx(9.0 * 140e9 * 48e9 / (3.0 * 140e9 + 48e9))
    assert p["poisson_ratio"].value == pytest.approx(0.34)
    assert "Materials Project" in p["E"].source


def test_mp_cache_loaded_from_real_fetched_file_has_expected_shape():
    # Not a mock -- sigma_ground/inventory/data/materials_project_cache.json
    # is the output of a real fetch_materials_project.py run against the
    # live API. Confirms the loader + resolve._MP_CACHE actually see it.
    assert "copper" in resolve._MP_CACHE
    assert resolve._MP_CACHE["copper"]["bulk_modulus_vrh"] > 0
    assert resolve._MP_CACHE["copper"]["provenance_tag"] == "DFT-COMPUTED"


# ── byte-identical-additive guarantee (real regression check) ────────────
# Materials with no TPSX alias and no MP cache entry -- section 0a/2b are
# verified no-ops for these; only a possible new `poisson_ratio` key (a
# pure addition, sourced from mechanical.py's own already-measured table,
# unrelated to either new external database) may appear.
_UNTOUCHED_BY_NEW_TIERS = ("silicon", "steel_mild", "mercury", "glass",
                          "flarbleglomp", "balsa", "concrete", "kevlar")


@pytest.mark.parametrize("name", _UNTOUCHED_BY_NEW_TIERS)
def test_materials_outside_both_new_tiers_keep_every_existing_value(name):
    p = material_profile(name)
    for key, fact in p.items():
        if not isinstance(fact, MaterialFact):
            continue
        assert "TPSX" not in fact.source, f"{name}.{key} unexpectedly TPSX-sourced"
        assert "Materials Project" not in fact.source, f"{name}.{key} unexpectedly MP-sourced"
    # and no TPSX-only keys leaked in
    for tpsx_only_key in ("thermal_conductivity_w_mk", "specific_heat_j_kgk", "melt_temperature_k"):
        assert tpsx_only_key not in p, f"{name} unexpectedly gained {tpsx_only_key}"


def test_the_13_aliased_metals_are_exactly_the_ones_with_changed_density_source():
    # Locks in the design's own accounting: exactly 13 canon tokens should
    # have a TPSX-sourced density; everything else in MATERIALS keeps
    # surface.MATERIALS as the density source.
    from sigma_ground.field.interface.surface import MATERIALS
    from sigma_ground.field.interface.tpsx_lookup import _MATERIAL_ALIAS
    tpsx_sourced = []
    unchanged = []
    for canon in MATERIALS:
        p = material_profile(canon)
        if "TPSX" in p["density"].source:
            tpsx_sourced.append(canon)
        else:
            assert p["density"].source == "surface.MATERIALS"
            unchanged.append(canon)
    assert set(tpsx_sourced) == set(_MATERIAL_ALIAS.keys())
    assert len(tpsx_sourced) == 13
    assert len(unchanged) == len(MATERIALS) - 13
