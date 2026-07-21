"""Arc B — the generative tree pipeline: cited allometry -> a compiled,
validated Construct.

`sigma_ground.deckard.__init__` re-exports the `research` FUNCTION under
the same name as the `research` submodule, which shadows plain
`import sigma_ground.deckard.research as x` (Python resolves `import a.b.c
as x` via attribute access on `a.b`, not sys.modules, so it picks up the
package's own rebound attribute) -- importlib.import_module bypasses that.
"""
import importlib
import math
import os
import sys

_CANON = r"D:\Aaron\development\sigma-ground"
if os.path.isdir(_CANON) and _CANON not in sys.path:
    sys.path.insert(0, _CANON)

from sigma_ground.deckard import identify, research
from sigma_ground.deckard import construct as _construct_pkg

research_mod = importlib.import_module("sigma_ground.deckard.research")
allometry = importlib.import_module("sigma_ground.deckard.sources.allometry")


def test_allometry_matches_the_cited_equations_exactly():
    """height_of/crown_radius_of reproduce the Chapman-Richards / linear
    models from the module's own hardcoded coefficients -- a transcription-
    drift guard, not a re-derivation of the papers."""
    d = allometry._SPECIES["loblolly pine"]["height"]
    dbh = 30.0
    expected_h = 1.37 + d["b1"] * (1.0 - math.exp(-d["b2"] * dbh)) ** d["b3"]
    h = allometry.height_of("loblolly pine", dbh)
    assert h is not None
    assert abs(h.value - round(expected_h, 3)) < 1e-6
    assert "Bragg 2008" in h.source

    c = allometry._SPECIES["loblolly pine"]["crown"]
    dbh_in = dbh / allometry.CM_PER_IN
    expected_lcw_ft = c["b0"] + c["b1"] * dbh_in
    expected_r = 0.5 * expected_lcw_ft * allometry.M_PER_FT
    r = allometry.crown_radius_of("loblolly pine", dbh)
    assert r is not None
    assert abs(r.value - round(expected_r, 4)) < 1e-6
    assert "Bechtold 2003" in r.source


def test_allometry_flags_extrapolation_outside_the_fitted_range():
    in_range = allometry.height_of("loblolly pine", 30.0)
    out_of_range = allometry.height_of("loblolly pine", 500.0)   # > 108.2cm max
    assert "outside the fitted" not in in_range.source
    assert "outside the fitted" in out_of_range.source
    assert out_of_range.confidence < in_range.confidence


def test_unknown_species_returns_none_not_a_guess():
    assert allometry.height_of("douglas fir", 30.0) is None
    assert allometry.crown_radius_of("douglas fir", 30.0) is None


def test_tree_spec_identifies_a_named_species():
    spec = research_mod._tree_spec("loblolly pine", dbh_cm=30.0)
    assert spec is not None
    assert spec.identified is True
    assert spec.kind == "composite"
    names = {p.name for p in spec.parts}
    assert names == {"trunk", "canopy"}


def test_tree_spec_flags_a_bare_trigger_word_as_unidentified():
    # "a tree" has no species match -> falls back to loblolly pine as a
    # flagged stand-in, per _tree_archetype's documented Deckard-Cain
    # discipline (partial ID allowed, confident fake not).
    spec = research_mod._tree_spec("a tall pine tree", dbh_cm=30.0)
    assert spec is not None
    assert spec.identified is False


def test_non_tree_name_returns_none():
    assert research_mod._tree_spec("coffee cup") is None
    assert research_mod._tree_archetype("coffee cup") is None


def test_default_dbh_is_a_disclosed_choice():
    spec = research_mod._tree_spec("loblolly pine")   # no dbh_cm -> Choice
    assert spec is not None
    choice_sources = [s["name"] for s in spec.sources if "DBH" in s["name"]
                      or "diameter-at-breast-height" in s["name"]]
    assert choice_sources, spec.sources   # the Choice must be disclosed, not silent


def test_tree_compiles_through_construct_with_no_render_code_changes():
    """The whole point of Arc B: a tree ConstructSpec compiles through the
    EXISTING, unmodified _compile_parts() path -- Cylinder trunk + Outline
    canopy, both already-wired primitives."""
    spec = research_mod._tree_spec("loblolly pine", dbh_cm=30.0)
    c = _construct_pkg.compile(spec)
    assert c is not None
    assert c.mass_kg > 0.0


def test_compiled_height_tracks_the_allometric_prediction():
    """The trunk's cited height_m matches height_of() for the same DBH --
    catches a wiring bug (stale DBH, unit slip) between allometry and the
    part builder, independent of construct.compile()'s own integration."""
    dbh = 45.0
    spec = research_mod._tree_spec("loblolly pine", dbh_cm=dbh)
    trunk = next(p for p in spec.parts if p.name == "trunk")
    expected = allometry.height_of("loblolly pine", dbh)
    assert abs(trunk.dims["height_m"].value - expected.value) < 1e-6


def test_larger_dbh_produces_a_taller_wider_tree():
    """Monotonicity sanity check across the one free parameter."""
    small = research_mod._tree_spec("loblolly pine", dbh_cm=10.0)
    large = research_mod._tree_spec("loblolly pine", dbh_cm=60.0)
    small_h = small.parts[0].dims["height_m"].value
    large_h = large.parts[0].dims["height_m"].value
    small_r = small.parts[0].dims["radius_m"].value
    large_r = large.parts[0].dims["radius_m"].value
    assert large_h > small_h
    assert large_r > small_r


def test_validator_passes_for_a_correctly_wired_construct():
    dbh, species = 30.0, "loblolly pine"
    spec = research_mod._tree_spec(species, dbh_cm=dbh)
    trunk = next(p for p in spec.parts if p.name == "trunk")
    height_m = trunk.dims["height_m"].value
    radius_m = trunk.dims["radius_m"].value
    wood_density = 420.0   # wood_pine, matches _density_fact in research.py
    # An honest UPPER bound (untapered cylinder), not the real trunk mass --
    # exercise the validator with something legitimately under that bound.
    trunk_mass_kg = 0.5 * wood_density * math.pi * radius_m ** 2 * height_m
    result = allometry.validate_tree_mass(species, dbh, height_m, trunk_mass_kg, wood_density)
    assert result["passed"], result["checks"]
    assert result["needs_independent_biomass_citation"] is True   # honest, not oversold


def test_validator_catches_a_stale_height_wiring_bug():
    dbh, species = 30.0, "loblolly pine"
    result = allometry.validate_tree_mass(species, dbh, height_m=999.0,
                                          trunk_mass_kg=1.0, wood_density_kg_m3=420.0)
    assert result["passed"] is False
    assert result["checks"][0]["name"] == "height self-consistency"
    assert result["checks"][0]["passed"] is False


def test_validator_catches_a_physically_impossible_trunk_mass():
    dbh, species = 30.0, "loblolly pine"
    spec = research_mod._tree_spec(species, dbh_cm=dbh)
    trunk = next(p for p in spec.parts if p.name == "trunk")
    height_m = trunk.dims["height_m"].value
    # A tapered real bole can never exceed the untapered-cylinder upper
    # bound -- ask for 10x it and the bound check must fail.
    radius_m = trunk.dims["radius_m"].value
    wood_density = 420.0
    upper_bound = wood_density * math.pi * radius_m ** 2 * height_m
    result = allometry.validate_tree_mass(species, dbh, height_m, upper_bound * 10, wood_density)
    assert result["passed"] is False
    assert result["checks"][1]["name"].startswith("trunk-mass upper bound")
    assert result["checks"][1]["passed"] is False


def test_tree_reachable_through_the_public_research_and_identify_api():
    """End-to-end through the SAME entry points every other object type
    uses -- research()'s dispatch chain and identify()'s compile step --
    not just the private _tree_spec helper directly."""
    spec = research("loblolly pine", allow_llm=False, dbh_cm=25.0)
    assert spec.identified is True
    assert spec.kind == "composite"

    construct = identify("loblolly pine", resolution=32, allow_llm=False)
    assert construct.mass_kg > 0.0
