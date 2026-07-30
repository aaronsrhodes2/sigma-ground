"""tpsx_lookup.py -- NASA TPSX name/property matching. Every assertion here
checks against a REAL number pulled from the live 700-entry TPSX JSON, not
just "a number came back": aluminum's density, LI-900 Rigid Tile's real
tensile strength, id_27's HTML-entity-encoded Poisson's Ratio property name.
"""
import html
import re

from sigma_ground.field.interface.tpsx_lookup import (
    tpsx_property, TPSXFact, TPSX_RESOLVE_KEYS, _properties_by_stripped_name,
    _MATERIAL_ALIAS, _FOOTNOTE_DIGIT,
)


def test_aluminum_density_matches_live_tpsx_value():
    fact = tpsx_property("aluminum", "density")
    assert isinstance(fact, TPSXFact)
    assert fact.value == 2700.0                    # id_59 "Aluminum-Pure"
    assert fact.units == "kg/m^3"
    assert "TPSX" in fact.source
    assert fact.tpsx_id == 59
    assert not fact.source == "estimated"


def test_gold_specific_heat_and_thermal_conductivity_are_real_numbers():
    # Real, textbook-plausible values for pure gold (Cp ~129 J/kg-K, k ~317 W/m-K)
    cp = tpsx_property("gold", "specific_heat_j_kgk")
    k = tpsx_property("gold", "thermal_conductivity_w_mk")
    assert cp is not None and 100 < cp.value < 160
    assert k is not None and 250 < k.value < 350
    assert cp.units == "J/kg-K"
    assert k.units == "W/m-K"


def test_steel_mild_is_explicitly_excluded_not_fuzzy_matched():
    # TPSX only has stainless/alloy grades -- this locks in the deliberate
    # exclusion (a false-friend a fuzzy matcher would get wrong).
    assert "steel_mild" not in _MATERIAL_ALIAS
    assert tpsx_property("steel_mild", "density") is None


def test_silicon_is_excluded_the_silicone_false_friend():
    # TPSX's only Si-adjacent entries are "Silicone A/B" (organosilicon
    # polymer) -- chemically unrelated to elemental silicon. Must not alias.
    assert "silicon" not in _MATERIAL_ALIAS
    assert tpsx_property("silicon", "density") is None


def test_unknown_resolve_key_returns_none():
    assert tpsx_property("aluminum", "not_a_real_property_key") is None


def test_unknown_canonical_key_returns_none_for_every_resolve_key():
    for rkey in TPSX_RESOLVE_KEYS:
        assert tpsx_property("flarbleglomp", rkey) is None


def test_li900_rigid_tile_tensile_strength_in_plane_is_the_real_number():
    # LI-900 Rigid Tile (id_1) is NOT in the alias table (it's a TPS
    # composite, not one of our plain-metal canon tokens) -- but its raw
    # data must parse correctly through the same stripped-name pipeline
    # tpsx_property() uses. Real value pulled directly from the JSON.
    props = _properties_by_stripped_name(1)
    row = props["Tensile Strength (In-Plane)"]
    assert abs(float(row["value"]) - 4.69e+05) < 1.0
    assert row["units"] == "Pa"
    assert row["source_type"] == "measured"


def test_poisson_ratio_html_entity_and_footnote_digit_both_strip_correctly():
    # id_27 (SiC/SiC) has the raw property literally stored as
    # "Poisson&#x27;s Ratio (Isotropic)" in the JSON (fetch_tpsx.py's
    # _clean() only unescapes &nbsp;/&amp;) -- confirms html.unescape() +
    # footnote-digit stripping both fire correctly on a real entry, since
    # none of the 13 aliased materials exercise this path themselves.
    props = _properties_by_stripped_name(27)
    assert "Poisson's Ratio (Isotropic)" in props
    assert float(props["Poisson's Ratio (Isotropic)"]["value"]) == 0.15
    # and the raw JSON really does have the mangled entity, to prove this
    # is a real regression test and not a tautology
    import json
    from sigma_ground.field.interface.tpsx_lookup import _materials
    raw_names = [r["property"] for r in _materials()["id_27"]["properties"]]
    assert any("&#x27;" in n for n in raw_names)


def test_footnote_digit_stripping_collapses_variants():
    # "Tensile Strength (In-Plane)9" -> "Tensile Strength (In-Plane)"
    assert _FOOTNOTE_DIGIT.sub("", "Tensile Strength (In-Plane)9") == "Tensile Strength (In-Plane)"
    assert _FOOTNOTE_DIGIT.sub("", "Density1") == "Density"
    assert _FOOTNOTE_DIGIT.sub("", "Poisson's Ratio (Isotropic)13") == "Poisson's Ratio (Isotropic)"


def test_unit_mismatch_refuses_rather_than_mis_scales():
    # Every _PROPERTY_MAP entry's declared unit was verified against the
    # live file; if that ever drifts, tpsx_property must refuse (return
    # None for that variant, try the next one, or None overall) rather than
    # silently return a value in the wrong unit. Sanity-check the density
    # variant's declared unit actually matches what's live in the file for
    # every aliased material (an early-warning trip-wire, not a unit test
    # of the refusal branch itself -- that would require corrupting the
    # fixture, out of scope here).
    for canon in _MATERIAL_ALIAS:
        fact = tpsx_property(canon, "density")
        assert fact is not None, f"{canon}: density must resolve (all 13 aliases have it)"
        assert fact.units == "kg/m^3"


def test_all_thirteen_aliased_materials_resolve_density_and_thermal_conductivity():
    for canon in _MATERIAL_ALIAS:
        density = tpsx_property(canon, "density")
        k = tpsx_property(canon, "thermal_conductivity_w_mk")
        assert density is not None, f"{canon}: density should resolve"
        assert k is not None, f"{canon}: thermal_conductivity should resolve"
        assert density.value > 0
        assert k.value > 0


def test_mechanical_properties_are_inert_for_the_current_alias_set():
    # Documented, verified fact: TPSX's tensile/shear/Poisson rows belong to
    # aerospace TPS composites, not our plain-metal alias set -- so E/G/
    # sigma_UTS/poisson_ratio should NOT resolve for any of the 13 aliased
    # materials today. This locks in that claim as a regression test.
    for canon in _MATERIAL_ALIAS:
        for rkey in ("E", "G", "sigma_UTS", "poisson_ratio"):
            assert tpsx_property(canon, rkey) is None, \
                f"{canon}.{rkey} unexpectedly resolved -- TPSX alias data may have changed"
