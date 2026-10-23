"""Regression tests for the simulate() front door (audit 2026-10-10, A38-A42).

The deterministic router (use_llm=False) must use what the question says: the
body asked about, the black-hole mass, the speed and the particle, masses in
exponent notation; and the verbs must not floor or clamp what they were given.
Ground truth is independent of the code: NASA fact sheets, the geostationary
radius, CODATA (scipy) and closed forms.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

# Import sigma_ground from THIS tree (worktree-portable).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

pytest.importorskip("scipy")
import scipy.constants as sc  # noqa: E402

from sigma_ground.mcp.tools.simulation import simulate, run_simulation  # noqa: E402
from sigma_ground.materia.manifest import VERB_MANIFEST  # noqa: E402

DAY = 86400.0


def _chain_rows(tr):
    """{row label: value} of the last verb in a simulate() result."""
    chain = tr.inputs["chain"]
    return {s["label"]: s["value"] for s in chain[-1]["steps"]}


def _outputs(tr):
    return tr.inputs["chain"][-1]["outputs"]


# ── A38: orbit questions use the body asked about ──────────────────────────

def test_manifest_example_moon_orbit_is_the_moons_orbit():
    q = "how fast does the Moon orbit"
    assert q in VERB_MANIFEST["orbital_velocity"]["examples"]
    tr = simulate(q, use_llm=False)
    # NASA NSSDCA Moon fact sheet: mean orbital velocity 1.022 km/s,
    # sidereal period 27.3217 d (circular two-body estimate: within 0.5 %)
    assert tr.value == pytest.approx(1022.0, rel=5e-3, abs=0.0)
    assert _outputs(tr)["orbital_period_s"] == pytest.approx(27.3217 * DAY, rel=5e-3, abs=0.0)


def test_moon_orbit_the_earth():
    tr = simulate("how fast does the Moon orbit the Earth", use_llm=False)
    assert tr.value == pytest.approx(1022.0, rel=5e-3, abs=0.0)


def test_orbital_period_of_mars_is_heliocentric():
    tr = simulate("what is the orbital period of Mars", use_llm=False)
    # NASA NSSDCA Mars fact sheet: sidereal orbit period 686.980 d
    assert _outputs(tr)["orbital_period_s"] == pytest.approx(686.980 * DAY, rel=1e-3, abs=0.0)


def test_geostationary_orbit():
    tr = simulate("what is the speed of a geostationary orbit", use_llm=False)
    # geostationary radius 42,164 km, altitude 35,786 km above the equator;
    # speed = 2 pi r / sidereal day = 3.0747 km/s
    v_ref = 2.0 * math.pi * 42_164.17e3 / 86164.0905
    assert tr.value == pytest.approx(v_ref, rel=1e-4, abs=0.0)
    assert _outputs(tr)["altitude_m"] == pytest.approx(35_786e3, abs=2e3)


def test_satellite_phrasings_still_route_to_a_satellite_orbit():
    tr = simulate("orbital velocity at 400 km above Earth", use_llm=False)
    # v = sqrt(GM/r), GM_Earth = 3.986004418e14 m^3/s^2 (IERS), R = 6378.1 km
    v_ref = math.sqrt(3.986004418e14 / (6378.1e3 + 400e3))
    assert tr.value == pytest.approx(v_ref, rel=1e-4, abs=0.0)


def test_unknown_central_body_is_refused_not_earth():
    tr = run_simulation("orbital_velocity", {"central_body": "vulcan"})
    assert tr.value is None


# ── A39: black-hole mass slot (and no 1e-12 M_sun floor) ───────────────────

def _hawking_T(M):
    return sc.hbar * sc.c ** 3 / (8.0 * math.pi * sc.G * M * sc.k)


def test_black_hole_solar_mass_slot_is_used():
    tr = simulate("Hawking temperature of a 3 solar mass black hole", use_llm=False)
    M = _outputs(tr)["mass_kg"]
    assert M == pytest.approx(3 * 1.98847e30, rel=1e-3, abs=0.0)        # IAU M_sun
    # rows are 4-significant-figure strings; M_sun convention differs 3e-4
    assert float(_chain_rows(tr)["hawking temperature"]) == pytest.approx(
        _hawking_T(3 * 1.98847e30), rel=2e-3, abs=0.0)


def test_black_hole_mass_in_kg_exponent_notation():
    tr = simulate("Schwarzschild radius of a 1e11 kg black hole", use_llm=False)
    r_s = 2.0 * sc.G * 1e11 / sc.c ** 2                         # 1.485e-16 m
    assert tr.value == pytest.approx(r_s, rel=1e-6, abs=0.0)
    tr = simulate("how long until a 1e11 kg black hole evaporates", use_llm=False)
    assert float(_chain_rows(tr)["hawking temperature"]) == pytest.approx(
        _hawking_T(1e11), rel=1e-3, abs=0.0)                       # 4-sig-fig row


def test_black_hole_primordial_mass_not_floored():
    tr = run_simulation("black_hole", {"solar_masses": 1e-20})
    M = _outputs(tr)["mass_kg"]
    assert M == pytest.approx(1e-20 * 1.98847e30, rel=1e-3, abs=0.0)     # not 2e18 kg
    assert tr.value == pytest.approx(2.0 * sc.G * M / sc.c ** 2, rel=1e-6, abs=0.0)
    for bad in (0.0, -5.0):
        assert run_simulation("black_hole", {"solar_masses": bad}).value is None


# Review of the A39 fix: the hole's mass must be the HOLE's. Ground truth is
# r_s = 2 GM/c^2 with independent GM values: IAU 2015 nominal GM_sun =
# 1.3271244e20 m^3/s^2, IERS GM_earth = 3.986004418e14, JPL DE440 GM_moon =
# 4.9028e12 (the body table's masses agree to < 1e-3).
GM_SUN, GM_EARTH, GM_MOON = 1.3271244e20, 3.986004418e14, 4.9028e12


def _r_s(gm):
    return 2.0 * gm / sc.c ** 2


@pytest.mark.parametrize("q, n_sun", [
    ("tidal force on a 70 kg astronaut falling into a 10 solar mass black hole", 10.0),
    ("drop a 1 kg clock toward a 4 million solar mass black hole: Hawking temperature", 4e6),
])
def test_black_hole_unrelated_kg_mass_is_not_the_hole(q, n_sun):
    tr = simulate(q, use_llm=False)
    assert tr.value == pytest.approx(_r_s(n_sun * GM_SUN), rel=1e-3, abs=0.0)


@pytest.mark.parametrize("q", [
    "What's the event horizon radius for one solar mass?",
    "If the Sun collapsed into a black hole, what would its event horizon radius be?",  # weak_domains gr_001
    "How big is the Sun's event horizon?",                                           # robustness para_rs_01
    "Bekenstein-Hawking entropy of a solar-mass black hole",
])
def test_black_hole_one_solar_mass_in_words(q):
    tr = simulate(q, use_llm=False)
    assert tr.value == pytest.approx(_r_s(GM_SUN), rel=5e-3, abs=0.0)    # 2953 m


@pytest.mark.parametrize("q, gm", [
    ("Bekenstein-Hawking entropy of an Earth-mass black hole in units of k_B", GM_EARTH),
    ("Schwarzschild radius of a black hole with the mass of the Moon", GM_MOON),
])
def test_black_hole_body_mass(q, gm):
    tr = simulate(q, use_llm=False)
    assert tr.value == pytest.approx(_r_s(gm), rel=1e-3, abs=0.0)


def test_black_hole_kg_mass_tied_to_the_hole():
    for q, m in [("Hawking temperature of a primordial black hole of mass 1e12 kg", 1e12),
                 ("a 1e9-kg primordial black hole: lifetime", 1e9),
                 ("If a 1-kilometer-mass black hole existed (1e12 kg), what's its "
                  "event horizon size and Hawking temperature?", 1e12),
                 ("density of 1 kg of matter compressed to a black hole", 1.0)]:
        tr = simulate(q, use_llm=False)
        assert tr.value == pytest.approx(2.0 * sc.G * m / sc.c ** 2, rel=1e-6, abs=0.0)


def test_black_hole_two_disagreeing_masses_are_refused():
    assert run_simulation("black_hole", {"solar_masses": 10.0, "mass_kg": 70.0}).value is None


@pytest.mark.parametrize("q", [
    "how much energy does 1 kg of matter release in fusion",
    "what happens to 5 g of matter near a neutron star",
    "density of 1 kg of matter compressed to a black hole",
    "how much energy is released when 1 kg of matter falls into a black hole",
])
def test_kg_of_matter_is_not_a_mass_energy_conversion(q):
    from sigma_ground import materia
    steps = materia.translate(q, use_qwen=False).steps
    assert "mass_energy" not in [s.verb for s in steps]


# ── A40: relativistic_motion reads the speed and the particle ──────────────

@pytest.mark.parametrize("q, beta", [
    ("Lorentz factor at 0.99c", 0.99),
    ("time dilation at 99% of the speed of light", 0.99),
    ("time dilation at 0.5c", 0.5),
    ("how much does a clock slow at half the speed of light", 0.5),
])
def test_relativistic_speed_slot(q, beta):
    tr = simulate(q, use_llm=False)
    assert tr.value == pytest.approx(1.0 / math.sqrt(1.0 - beta ** 2), rel=1e-12, abs=0.0)


def test_relativistic_particle_slot_proton():
    tr = simulate("relativistic energy of a proton at 0.9c", use_llm=False)
    E = float(_chain_rows(tr)["relativistic energy"])            # 4-sig-fig row
    E_ref = sc.m_p * sc.c ** 2 / math.sqrt(1.0 - 0.9 ** 2)       # 3.449e-10 J
    assert E == pytest.approx(E_ref, rel=1e-3, abs=0.0)
    assert _outputs(tr)["rest_mass_kg"] == pytest.approx(sc.m_p, rel=1e-8, abs=0.0)


def test_relativistic_motion_refuses_superluminal_and_reads_speed():
    assert run_simulation("relativistic_motion",
                          {"velocity_fraction_c": 1.5}).value is None
    assert run_simulation("relativistic_motion",
                          {"velocity_fraction_c": 1.0}).value is None
    tr = run_simulation("relativistic_motion", {"velocity_fraction_c": -0.5})
    assert tr.value == pytest.approx(1.0 / math.sqrt(0.75), rel=1e-12, abs=0.0)


# ── A41: mass regex keeps the mantissa; grams are grams ────────────────────

@pytest.mark.parametrize("q, m", [
    ("how much energy is in 1e-3 kg of mass", 1e-3),
    ("how much energy is in 2.5e-3 kg of mass", 2.5e-3),
    ("how much energy is in 1 gram of mass", 1e-3),
    ("how much energy is in 5 kg of mass", 5.0),
])
def test_mass_slot_exponent_notation(q, m):
    tr = simulate(q, use_llm=False)
    assert tr.value == pytest.approx(m * sc.c ** 2, rel=1e-12, abs=0.0)


# ── A42: mass_energy does not floor the mass ───────────────────────────────

def test_mass_energy_electron_rest_energy():
    tr = run_simulation("mass_energy", {"mass_kg": sc.m_e})
    # CODATA 2022 m_e c^2 = 8.1871057880(26)e-14 J (NIST)
    assert tr.value == pytest.approx(8.1871057880e-14, rel=1e-9, abs=0.0)
    from sigma_ground.materia.scenarios import mass_energy
    assert mass_energy(mass_kg=1e-35).outputs["rest_energy_J"] == pytest.approx(
        1e-35 * sc.c ** 2, rel=1e-12, abs=0.0)
    for bad in (0.0, -5.0):
        assert mass_energy(mass_kg=bad).validation["passed"] is False


# ── audit medium: "rest energy of an electron" answered with the proton mass ─

def test_rest_energy_of_an_electron():
    tr = simulate("rest energy of an electron", use_llm=False)
    # CODATA 2022 m_e c^2 = 8.1871057880(26)e-14 J (NIST); the verb's CODATA
    # 2018 mass differs by 1.4e-9
    assert tr.value == pytest.approx(8.1871057880e-14, rel=1e-8, abs=0.0)


def test_rest_energy_of_a_muon_and_of_a_kg_mass():
    tr = simulate("what is the rest energy of a muon", use_llm=False)
    mu = sc.physical_constants["muon mass energy equivalent"][0]   # J
    assert tr.value == pytest.approx(mu, rel=1e-8, abs=0.0)
    tr = simulate("rest energy of 1e11 kg of mass", use_llm=False)
    assert tr.value == pytest.approx(1e11 * sc.c ** 2, rel=1e-12, abs=0.0)


def test_rest_energy_routing_keeps_nucleons_and_annihilation_apart():
    from sigma_ground import materia
    steps = materia.translate("rest energy of a neutron in MeV", use_qwen=False).steps
    assert [s.verb for s in steps] == ["nucleon_masses"]
    # a pair annihilation is 2 m c^2, so the particle slot must NOT be filled
    steps = materia.translate("how much energy when an electron and positron "
                              "annihilate", use_qwen=False).steps
    assert "particle" not in steps[0].params
