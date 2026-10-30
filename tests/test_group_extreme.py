"""atomic.extreme_in_group and its cascade cue (misc/STATIC_INTENTS_PLAN.md group E).

atom_004 "What's the highest noble gas first ionization energy?" is an argmax over a set; the router answered Ne. The function refuses
an extremum over a group whose table is incomplete, so the cue never guesses.
"""
import pytest

from sigma_ground.mcp.benchmark.group_extreme_classifier import classify_for_group_extreme
from sigma_ground.mcp.tools.atomic import ELEMENT_GROUPS, _FIRST_IONIZATION_EV, extreme_in_group


def test_highest_noble_gas_first_ionization_energy_is_helium():
    r = extreme_in_group("first ionization energy", "noble gas", "highest")
    assert r.value == pytest.approx(24.587) and r.units == "eV" and r.inputs["element"] == "He"
    assert r.notes.startswith("He has the highest first ionization energy of the noble gass".rstrip("s") + "s") or "He has the highest" in r.notes
    r = extreme_in_group("first ionization energy", "noble gases", "lowest")
    assert r.inputs["element"] == "Rn" and r.value == pytest.approx(10.7485)


def test_every_noble_gas_has_a_first_ionization_energy_in_the_table():
    # the extremum is only honest over a complete group (Og has no measured value and is not a member here)
    assert all(s in _FIRST_IONIZATION_EV for s in ELEMENT_GROUPS["noble gas"])


def test_an_incomplete_group_is_refused_not_guessed():
    r = extreme_in_group("first ionization energy", "halogen", "highest")      # At has no entry in the table
    assert r.value is None and "At" in r.notes and "cannot be stated" in r.notes
    assert extreme_in_group("density", "nonsense group", "highest").value is None


def test_other_properties_come_from_periodictable():
    r = extreme_in_group("atomic mass", "alkali metal", "highest")
    assert r.inputs["element"] == "Fr" and r.units == "amu"
    r = extreme_in_group("atomic number", "halogen", "lowest")
    assert r.inputs["element"] == "F" and r.value == 9.0


def test_the_cue_matches_the_question_shapes_and_nothing_else():
    for q in ("What's the highest noble gas first ionization energy?",
              "Which noble gas has the highest first ionization energy?",
              "highest ionization energy among the noble gases",
              "what is the largest first ionisation potential of any inert gas"):
        m = classify_for_group_extreme(q)
        assert m is not None and m.result_override == pytest.approx(24.587) and m.result_units == "eV", q
    m = classify_for_group_extreme("which alkali metal has the lowest atomic mass")
    assert m.args["which"] == "lowest" and m.result_override == pytest.approx(6.94, rel=1e-2)
    for q in ("What's the first ionization energy of neon?", "highest first ionization energy of any element",
              "the noble gases have high ionization energies", "what is the density of helium",
              "highest halogen first ionization energy"):            # the last: At missing -> the tool refuses -> no cue
        assert classify_for_group_extreme(q) is None, q
