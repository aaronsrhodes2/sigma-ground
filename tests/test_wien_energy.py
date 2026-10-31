"""The Wien cue answers the quantity asked: wavelength, photon energy (J or eV) or frequency (found by the distillation branch, day one)."""
import pytest

from sigma_ground.mcp.benchmark.thermo_classifier import classify_for_thermo, execute_thermo_match


def test_photon_energy_at_the_wien_peak_is_an_energy_not_a_wavelength():
    m = classify_for_thermo("Photon energy at the Wien peak for 3.5e3 K?")
    assert m is not None and m.tool == "peak_photon_energy"
    val, units, _ = execute_thermo_match(m)
    assert units == "J" and val == pytest.approx(2.3992779983134604e-19, rel=1e-6)      # h c T / b
    m = classify_for_thermo("What is the energy in eV of a photon at the peak emission of a 5778 K blackbody?")
    val, units, _ = execute_thermo_match(m)
    assert units == "eV" and val == pytest.approx(2.47, abs=0.01)
    m = classify_for_thermo("Peak emission frequency of a 300 K blackbody (Wien)?")
    val, units, _ = execute_thermo_match(m)
    assert units == "Hz" and val == pytest.approx(3.1e13, rel=0.01)


def test_the_wavelength_question_is_unchanged():
    m = classify_for_thermo("What is the peak emission wavelength of a 5778 K blackbody?")
    assert m.tool == "blackbody_peak_wavelength"
    val, units, _ = execute_thermo_match(m)
    assert units == "m" and val == pytest.approx(5.014e-7, rel=1e-3)
