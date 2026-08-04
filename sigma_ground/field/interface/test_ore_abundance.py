"""
Tests for ore_abundance.py -- asteroid spectral taxonomy -> real mineral
assemblage, extraction energy, and body-scale recoverable mass.

Plain pytest functions (not unittest.TestCase), matching
sigma_ground/field/interface/test_propulsion.py style within this repo's
own test suite.
"""
import pytest

from sigma_ground.dynamics.mechanisms.choice import Choice
from sigma_ground.field.interface.ore_abundance import (
    ASSEMBLAGES,
    CI_ASSEMBLAGE,
    CM_ASSEMBLAGE,
    M_ASSEMBLAGE,
    S_ASSEMBLAGE,
    MineralComponent,
    OreAssemblage,
    extraction_energy_per_kg,
    ore_grade_estimate,
    total_recoverable_mass,
)
from sigma_ground.field.interface.surface import MATERIALS as SURFACE_MATERIALS
from sigma_ground.inventory.data.loader import MaterialDB


# ═══════════════════════════════════════════════════════════════════════════
# § 1. Mass fractions sum to 1.0, every assemblage
# ═══════════════════════════════════════════════════════════════════════════

def test_every_assemblage_mass_fractions_sum_to_one():
    for key, assemblage in ASSEMBLAGES.items():
        total = sum(c.mass_fraction for c in assemblage.components)
        assert total == pytest.approx(1.0, abs=1e-6), key


def test_ore_grade_estimate_reports_mass_fraction_total_near_one():
    for spectral_class in ("C", "CM", "S", "M"):
        result = ore_grade_estimate(spectral_class)
        assert result["mass_fraction_total"] == pytest.approx(1.0, abs=1e-6)
        component_sum = sum(c["mass_fraction"] for c in result["components"])
        assert component_sum == pytest.approx(1.0, abs=1e-6)


def test_ore_assemblage_rejects_mass_fractions_not_summing_to_one():
    bad_component = MineralComponent(
        component_key="x", mineral_name="x", mass_fraction=0.5,
        provenance="fact", source="none",
        gap_note="test-only unresolved component")
    with pytest.raises(ValueError):
        OreAssemblage(
            assemblage_key="BAD", taxonomy_system="test",
            meteorite_analog="test", components=(bad_component,),
            source="none")


def test_ore_assemblage_rejects_duplicate_component_keys():
    c1 = MineralComponent(component_key="dup", mineral_name="a",
                          mass_fraction=0.5, provenance="fact",
                          source="none", gap_note="gap")
    c2 = MineralComponent(component_key="dup", mineral_name="b",
                          mass_fraction=0.5, provenance="fact",
                          source="none", gap_note="gap")
    with pytest.raises(ValueError):
        OreAssemblage(assemblage_key="BAD", taxonomy_system="test",
                      meteorite_analog="test", components=(c1, c2),
                      source="none")


# ═══════════════════════════════════════════════════════════════════════════
# § 2. Every component either resolves to a real database key, or is
#      explicitly flagged as a disclosed gap -- never neither, never both
#      silently
# ═══════════════════════════════════════════════════════════════════════════

def test_every_component_resolves_or_carries_a_disclosed_gap_note():
    inventory_names = {n.lower() for n in MaterialDB.get().names()}
    checked_any_resolved = False
    checked_any_gap = False
    for assemblage in ASSEMBLAGES.values():
        for c in assemblage.components:
            resolves_surface = (c.material_key_surface is not None
                                and c.material_key_surface in SURFACE_MATERIALS)
            resolves_inventory = (c.material_key_inventory is not None
                                  and c.material_key_inventory.lower()
                                  in inventory_names)
            if c.material_key_surface is not None:
                assert resolves_surface, (
                    f"{assemblage.assemblage_key}/{c.component_key}: "
                    f"material_key_surface {c.material_key_surface!r} does "
                    f"not resolve against surface.py MATERIALS")
            if c.material_key_inventory is not None:
                assert resolves_inventory, (
                    f"{assemblage.assemblage_key}/{c.component_key}: "
                    f"material_key_inventory "
                    f"{c.material_key_inventory!r} does not resolve "
                    f"against inventory/data/materials.json")
            if c.resolved:
                checked_any_resolved = True
            else:
                # unresolved -> MUST carry a real, non-empty gap note
                assert c.gap_note, (
                    f"{assemblage.assemblage_key}/{c.component_key} is "
                    f"unresolved but has no gap_note")
                assert len(c.gap_note) > 10
                checked_any_gap = True
    # sanity: the fixture actually exercises both branches
    assert checked_any_resolved
    assert checked_any_gap


def test_mineral_component_rejects_neither_resolved_nor_gap_flagged():
    with pytest.raises(ValueError):
        MineralComponent(component_key="x", mineral_name="x",
                         mass_fraction=0.1, provenance="fact", source="none")
        # no material_key_surface, no material_key_inventory, no gap_note


def test_mineral_component_rejects_bad_provenance():
    with pytest.raises(ValueError):
        MineralComponent(component_key="x", mineral_name="x",
                         mass_fraction=0.1, provenance="measured",
                         source="none", gap_note="gap")


def test_mineral_component_rejects_out_of_range_mass_fraction():
    with pytest.raises(ValueError):
        MineralComponent(component_key="x", mineral_name="x",
                         mass_fraction=1.5, provenance="fact",
                         source="none", gap_note="gap")
    with pytest.raises(ValueError):
        MineralComponent(component_key="x", mineral_name="x",
                         mass_fraction=-0.1, provenance="fact",
                         source="none", gap_note="gap")


def test_choice_provenance_components_carry_a_real_choice_object():
    # fe_metal/ni_metal are the two exceptions: their Choice holds an INPUT
    # parameter (bulk Fe-Ni fraction, Ni-in-metal fraction) that gets
    # multiplied to DERIVE mass_fraction, not mass_fraction itself.
    derived_from_a_choice_product = {"fe_metal", "ni_metal"}
    for assemblage in ASSEMBLAGES.values():
        for c in assemblage.components:
            if c.provenance == "choice":
                assert isinstance(c.choice, Choice), (
                    f"{assemblage.assemblage_key}/{c.component_key}")
                if c.component_key not in derived_from_a_choice_product:
                    assert c.choice.value == pytest.approx(c.mass_fraction)


def test_m_type_fe_and_ni_components_derive_from_their_choice_product():
    fe = next(c for c in M_ASSEMBLAGE.components
             if c.component_key == "fe_metal")
    ni = next(c for c in M_ASSEMBLAGE.components
             if c.component_key == "ni_metal")
    bulk = fe.choice.value            # bulk Fe-Ni fraction Choice
    ni_content = ni.choice.value      # Ni-in-metal fraction Choice
    assert fe.mass_fraction == pytest.approx(bulk * (1.0 - ni_content))
    assert ni.mass_fraction == pytest.approx(bulk * ni_content)
    assert fe.mass_fraction + ni.mass_fraction == pytest.approx(bulk)


def test_mineral_component_choice_provenance_requires_choice_object():
    with pytest.raises(ValueError):
        MineralComponent(component_key="x", mineral_name="x",
                         mass_fraction=0.1, provenance="choice",
                         source="none", gap_note="gap", choice=None)


def test_every_component_has_a_nonempty_source_citation():
    for assemblage in ASSEMBLAGES.values():
        for c in assemblage.components:
            assert c.source, f"{assemblage.assemblage_key}/{c.component_key}"


# ═══════════════════════════════════════════════════════════════════════════
# § 3. C-type is volatile/water-rich relative to S-type (quantitative)
# ═══════════════════════════════════════════════════════════════════════════

def test_ci_is_dominated_by_a_hydrous_phyllosilicate_phase():
    phyllosilicate = next(c for c in CI_ASSEMBLAGE.components
                          if c.component_key == "phyllosilicate")
    assert phyllosilicate.mass_fraction > 0.5   # dominant CI phase


def test_cm_has_measured_indigenous_water_content():
    water = next(c for c in CM_ASSEMBLAGE.components
                if c.component_key == "water")
    assert water.mass_fraction > 0.0
    assert water.provenance == "fact"


def test_s_type_has_no_water_or_phyllosilicate_component():
    keys = {c.component_key for c in S_ASSEMBLAGE.components}
    assert "water" not in keys
    assert "phyllosilicate" not in keys


def test_c_type_water_fraction_exceeds_s_type_water_fraction():
    # S-type has no explicit water component at all -> treat as 0.0
    cm_water = next(c for c in CM_ASSEMBLAGE.components
                    if c.component_key == "water").mass_fraction
    s_water = sum(c.mass_fraction for c in S_ASSEMBLAGE.components
                 if c.component_key == "water")
    assert cm_water > 0.05
    assert s_water == 0.0
    assert cm_water > s_water


def test_ci_hydrous_phyllosilicate_fraction_exceeds_any_s_type_hydrous_phase():
    ci_phyllosilicate = next(c for c in CI_ASSEMBLAGE.components
                             if c.component_key == "phyllosilicate").mass_fraction
    s_hydrous = sum(c.mass_fraction for c in S_ASSEMBLAGE.components
                    if c.component_key in ("water", "phyllosilicate"))
    assert ci_phyllosilicate > 0.8
    assert s_hydrous == 0.0
    assert ci_phyllosilicate > s_hydrous


# ═══════════════════════════════════════════════════════════════════════════
# § 4. M-type is iron-dominated relative to C-type and S-type (quantitative)
# ═══════════════════════════════════════════════════════════════════════════

def _metal_fraction(assemblage: OreAssemblage) -> float:
    metal_keys = {"free_metal", "metallic_fe_ni", "fe_metal", "ni_metal"}
    return sum(c.mass_fraction for c in assemblage.components
              if c.component_key in metal_keys)


def test_m_type_metal_fraction_exceeds_s_type_and_ci():
    m_metal = _metal_fraction(M_ASSEMBLAGE)
    s_metal = _metal_fraction(S_ASSEMBLAGE)
    ci_metal = _metal_fraction(CI_ASSEMBLAGE)
    assert m_metal > 0.85
    assert s_metal < 0.2
    assert ci_metal == pytest.approx(0.0)
    assert m_metal > s_metal > ci_metal


def test_m_type_note_flags_the_psyche_partial_metal_caveat():
    # the doctrine requires disclosing the analogy-vs-direct-measurement gap
    rest = next(c for c in M_ASSEMBLAGE.components
               if c.component_key == "rest")
    assert "Psyche" in rest.choice.reason


# ═══════════════════════════════════════════════════════════════════════════
# § 5. ore_grade_estimate -- spectral class resolution and caveats
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("spectral_class,expected_assemblage", [
    ("C", "CI"), ("c", "CI"), ("B", "CI"), ("Cb", "CI"), ("Cg", "CI"),
    ("CI", "CI"), ("Cgh", "CM"), ("Ch", "CM"), ("CM", "CM"),
    ("S", "S"), ("s", "S"),
    ("M", "M"), ("m", "M"), ("X", "M"), ("Xc", "M"), ("Xk", "M"),
])
def test_ore_grade_estimate_resolves_spectral_class_aliases(
        spectral_class, expected_assemblage):
    result = ore_grade_estimate(spectral_class)
    assert result["assemblage_key"] == expected_assemblage


def test_ore_grade_estimate_rejects_unknown_spectral_class():
    with pytest.raises(ValueError):
        ore_grade_estimate("Q")


def test_ore_grade_estimate_x_complex_carries_metallicity_ambiguity_caveat():
    result = ore_grade_estimate("X")
    assert result["caveats"]
    assert any("albedo" in c for c in result["caveats"])


def test_ore_grade_estimate_m_type_input_has_no_caveats_by_default():
    # "M" and "Xc"/"Xk" are treated as confirmed-metallic inputs
    result = ore_grade_estimate("Xc")
    assert result["caveats"] == []


def test_ore_grade_estimate_echoes_input_spectral_class():
    result = ore_grade_estimate("Cgh")
    assert result["input_spectral_class"] == "Cgh"


# ═══════════════════════════════════════════════════════════════════════════
# § 6. extraction_energy_per_kg
# ═══════════════════════════════════════════════════════════════════════════

def test_extraction_energy_per_kg_hand_check_iron():
    # E/kg = cohesive_energy_ev * EV_TO_J / (A * AMU_KG)
    from sigma_ground.field.constants import AMU_KG, EV_TO_J
    mat = SURFACE_MATERIALS["iron"]
    expected = (mat["cohesive_energy_ev"] * EV_TO_J
               / (mat["A"] * AMU_KG))
    result = extraction_energy_per_kg("iron")
    assert result["extraction_energy_j_per_kg"] == pytest.approx(
        expected, rel=1e-9)
    assert result["extraction_energy_mj_per_kg"] == pytest.approx(
        expected / 1.0e6, rel=1e-9)


def test_extraction_energy_per_kg_positive_for_every_surface_material():
    for key in SURFACE_MATERIALS:
        result = extraction_energy_per_kg(key)
        assert result["extraction_energy_j_per_kg"] > 0.0, key


def test_extraction_energy_per_kg_order_of_magnitude_matches_known_physics():
    # independently cross-checked figure from the sourced research: pure
    # aluminum cohesive energy implies ~12.1 MJ/kg (vs. real Hall-Heroult
    # electrolytic reduction at ~47-54 MJ/kg -- this function is a floor,
    # not a prediction, and says so in its own output)
    result = extraction_energy_per_kg("aluminum")
    assert result["extraction_energy_mj_per_kg"] == pytest.approx(
        12.1, abs=0.5)


def test_extraction_energy_per_kg_discloses_the_simplified_model_caveat():
    result = extraction_energy_per_kg("iron")
    assert result["simplified_model_caveat"]
    assert "Gibbs free energy" in result["simplified_model_caveat"]


def test_extraction_energy_per_kg_includes_self_adhesion_diagnostic():
    from sigma_ground.field.interface.adhesion import work_of_adhesion
    result = extraction_energy_per_kg("nickel")
    expected = work_of_adhesion("nickel", "nickel")
    assert result["self_adhesion_work_of_cohesion_j_m2"] == pytest.approx(
        expected, rel=1e-9)


def test_extraction_energy_per_kg_rejects_non_surface_material_key():
    # "Water" resolves in materials.json but not in surface.py MATERIALS
    with pytest.raises(KeyError):
        extraction_energy_per_kg("Water")
    with pytest.raises(KeyError):
        extraction_energy_per_kg("not_a_real_material")


# ═══════════════════════════════════════════════════════════════════════════
# § 7. total_recoverable_mass -- exact mass conservation
# ═══════════════════════════════════════════════════════════════════════════

def test_total_recoverable_mass_hand_check():
    body_mass = 1.0e10
    result = total_recoverable_mass(body_mass, "M", "iron")
    grade = ore_grade_estimate("M")
    fe = next(c for c in grade["components"] if c["component_key"] == "fe_metal")
    assert result["recoverable_mass_kg"] == pytest.approx(
        body_mass * fe["mass_fraction"], rel=1e-9)


@pytest.mark.parametrize("spectral_class", ["C", "CM", "S", "M"])
def test_total_recoverable_mass_conserves_mass_exactly(spectral_class):
    body_mass = 7.3e14   # an arbitrary large small-body mass, kg
    grade = ore_grade_estimate(spectral_class)
    any_component_key = grade["components"][0]["component_key"]
    result = total_recoverable_mass(body_mass, spectral_class,
                                    any_component_key)
    closure = result["mass_closure"]
    assert closure["total_accounted_kg"] == pytest.approx(
        body_mass, rel=1e-9)
    assert closure["residual_kg"] == pytest.approx(0.0, abs=body_mass * 1e-9)
    # every named component (resolved AND gap-flagged) is present in the
    # breakdown -- nothing dropped from the mass balance
    assert len(closure["components"]) == len(grade["components"])
    assert sum(c["mass_kg"] for c in closure["components"]) == pytest.approx(
        body_mass, rel=1e-9)


def test_total_recoverable_mass_matches_by_component_key_or_material_key():
    body_mass = 5.0e9
    by_component = total_recoverable_mass(body_mass, "M", "fe_metal")
    by_surface_key = total_recoverable_mass(body_mass, "M", "iron")
    by_inventory_key = total_recoverable_mass(body_mass, "M", "Iron")
    assert by_component["recoverable_mass_kg"] == pytest.approx(
        by_surface_key["recoverable_mass_kg"])
    assert by_component["recoverable_mass_kg"] == pytest.approx(
        by_inventory_key["recoverable_mass_kg"])


def test_total_recoverable_mass_works_for_gap_flagged_components():
    # a real mineral this codebase has no physics data for still has a
    # real, exactly-accounted mass in the body
    body_mass = 2.0e12
    result = total_recoverable_mass(body_mass, "C", "phyllosilicate")
    assert result["resolved"] is False
    assert result["gap_note"]
    assert result["recoverable_mass_kg"] == pytest.approx(
        body_mass * 0.825, rel=1e-9)
    assert result["mass_closure"]["residual_kg"] == pytest.approx(
        0.0, abs=body_mass * 1e-9)


def test_total_recoverable_mass_rejects_nonpositive_body_mass():
    with pytest.raises(ValueError):
        total_recoverable_mass(0.0, "M", "iron")
    with pytest.raises(ValueError):
        total_recoverable_mass(-1.0, "M", "iron")


def test_total_recoverable_mass_rejects_unknown_material_key():
    with pytest.raises(ValueError):
        total_recoverable_mass(1.0e10, "M", "unobtainium")


def test_total_recoverable_mass_zero_for_ci_free_metal():
    # CI is the uniquely metal-free chondrite group (Weisberg et al. 2006)
    result = total_recoverable_mass(1.0e10, "C", "free_metal")
    assert result["recoverable_mass_kg"] == pytest.approx(0.0, abs=1e-6)
