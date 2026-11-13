"""Tests for the cat-in-a-box model (schrodinger_cat.py): the sample, the environment that records the cat, and the quantum state."""
import math

import numpy as np
import pytest

from sigma_ground.field.interface import schrodinger_cat as sc


def test_the_sample_gives_even_odds_in_the_hour_from_the_measured_half_life():
    s = sc.sample()
    assert s["half_life_s"] == pytest.approx(1.663e8)                 # cobalt-60, measured (5.27 years)
    assert sc.decay_probability(3600.0, s) == pytest.approx(0.5)
    assert sc.decay_probability(0.0, s) == 0.0
    assert sc.decay_probability(7200.0, s) == pytest.approx(0.75)
    assert 4.0e4 < s["atoms"] < 5.0e4                                   # a few femtograms
    assert s["mass_kg"] == pytest.approx(s["atoms"] * 60 * 1.66054e-27, rel=1e-4)


def test_every_probe_is_far_below_the_separation_so_one_event_records_the_state():
    r = sc.regime()
    assert r["saturated"]
    assert r["air_de_broglie_nm"] == pytest.approx(0.019, rel=0.1)    # N2/O2 at room temperature
    assert 10 < r["photon_peak_um"] < 25                                # a skin-temperature body glows in the far infrared


def test_the_air_inside_the_box_records_the_cat_long_before_its_own_light_does():
    d = sc.decoherence_times()
    assert d["rate_air_per_s"] == pytest.approx(5.8e26, rel=0.1)
    assert d["rate_photons_per_s"] == pytest.approx(8.6e21, rel=0.1)
    assert d["air_s"] < 1e-3 * d["photons_s"]
    assert d["total_s"] == pytest.approx(1.0 / (d["rate_air_per_s"] + d["rate_photons_per_s"]))
    assert d["total_s"] < 1e-26                                           # far below anything a person (0.1 s) or a clock could notice
    vac = sc.decoherence_times(vacuum=True)
    assert vac["air_s"] == math.inf and vac["total_s"] == pytest.approx(d["photons_s"])


def test_a_pinhole_tells_the_outside_in_a_blink():
    assert 1.0 / sc.leak_rate() < 1e-15
    assert math.isinf(sc.timeline()["leak_tau_s"]) and sc.timeline(hole=True)["leak_tau_s"] == pytest.approx(1.0 / sc.leak_rate())


def test_the_chain_is_unitary_and_correlates_everything_without_collapsing_anything():
    psi = sc.box_state(0.3, 1.0, 1.0)
    assert np.linalg.norm(psi) == pytest.approx(1.0)
    pops = np.abs(psi.reshape((2,) * 7)) ** 2
    macro = pops.sum(axis=(5, 6))                                         # atom, counter, vial, cat, observer
    assert macro[0, 0, 0, 0, 0] == pytest.approx(0.7) and macro[1, 1, 1, 1, 1] == pytest.approx(0.3)
    assert macro.sum() - macro[0, 0, 0, 0, 0] - macro[1, 1, 1, 1, 1] == pytest.approx(0.0, abs=1e-12)   # no state with the parts disagreeing


def test_the_chains_branch_coherence_is_the_product_of_the_environments_overlaps():
    for p in (0.5, 0.2):
        for gi in (1.0, 0.6, 0.0):
            for go in (1.0, 0.4, 0.0):
                assert sc.branch_coherence(sc.box_state(p, gi, go)) == pytest.approx(gi * go, abs=1e-9)


def test_the_cat_alone_is_a_mixture_from_the_first_moment_it_is_entangled_with_the_counter():
    assert sc.branch_coherence(sc.box_state(0.5, 1.0, 1.0), (3,)) == pytest.approx(0.0, abs=1e-12)


def test_the_outside_test_tells_a_superposition_from_a_mixture():
    # P(yes) = 1/2 + sqrt(p (1-p)) * gamma_out when the experimenter can reverse everything inside
    for p in (0.5, 0.2):
        for go in (1.0, 0.5, 0.0):
            assert sc.outside_test(p, 1.0, go) == pytest.approx(0.5 + math.sqrt(p * (1 - p)) * go, abs=1e-9)
    # the sealed box keeps it in principle even with the air gone to "mixture" (she reverses the air too) ...
    assert sc.outside_test(0.5, 0.0, 1.0) == pytest.approx(1.0)
    # ... and not in practice: with only the macroscopic variables to measure, the air's record has destroyed it
    assert sc.outside_test(0.5, 0.0, 1.0, practical=True) == pytest.approx(0.5)
    assert sc.outside_test(0.5, 0.6, 0.5, practical=True) == pytest.approx(0.5 + 0.5 * 0.6 * 0.5)


def test_the_observer_inside_sees_one_outcome_with_the_born_probability():
    rng = np.random.default_rng(7)
    sees = [sc.observer_sees(0.3, rng) for _ in range(4000)]
    assert sees.count("dead") / 4000 == pytest.approx(0.3, abs=0.03)
    # and she is entangled with it: her state alone is a mixture of the two outcomes with those odds
    rho = sc.reduced(sc.box_state(0.3, 1.0, 1.0), (4,))
    assert rho[0, 0].real == pytest.approx(0.7) and rho[1, 1].real == pytest.approx(0.3) and abs(rho[0, 1]) < 1e-12


def test_gamma_falls_as_the_environment_records():
    assert sc.gamma_at(0.0, 1e-27) == 1.0
    assert sc.gamma_at(1e-27, 1e-27) == pytest.approx(math.exp(-1))
    assert sc.gamma_at(1e-20, 1e-27) == 0.0


def test_chain_coherence_in_practice_is_the_product_and_in_principle_only_the_leak():
    for p in (0.5, 0.2):
        for gi in (1.0, 0.6, 0.0):
            for go in (1.0, 0.4, 0.0):
                assert sc.chain_coherence(p, gi, go) == pytest.approx(gi * go, abs=1e-9)
                assert sc.chain_coherence(p, gi, go, in_principle=True) == pytest.approx(go, abs=1e-9)
    assert sc.chain_coherence(0.3, 0.5, 1.0, observer_looked=False, in_principle=True) == pytest.approx(1.0)


# ── a lump of inert matter ───────────────────────────────────────────────

def test_a_lump_of_osmium_is_95_grams_of_the_densest_metal():
    assert sc.lump_mass_kg() == pytest.approx(0.0946, rel=0.01)


def test_at_zero_kelvin_in_a_vacuum_nothing_records_the_lump():
    d = sc.lump_decoherence_times(0.0)
    assert math.isinf(d["total_s"]) and d["rate_own_per_s"] == 0.0 and d["rate_box_per_s"] == 0.0
    assert sc.resolving_fraction(0.01, 0.0) == 0.0 and sc.photon_flux_per_area(0.0) == 0.0


def test_a_warm_observer_in_the_box_records_it_whatever_the_temperature_of_the_box():
    d = sc.lump_decoherence_times(0.0, observer_inside=True)
    assert d["total_s"] == pytest.approx(5.4e-19, rel=0.15)             # half an attosecond: her own light
    assert d["rate_own_per_s"] == 0.0 and d["rate_box_per_s"] == 0.0
    assert sc.lump_decoherence_times(0.0, observer_inside=False)["total_s"] > 1e30


def test_a_cold_lump_holds_for_minutes_and_a_warm_one_for_no_time_at_all():
    mk = sc.lump_decoherence_times(1e-3)["total_s"]
    assert 30 < mk < 600                                                  # the wavelength (5 m) is far above the 1 cm separation: a weak record
    assert sc.lump_decoherence_times(1.0)["total_s"] < 1e-9               # at 1 K the wavelength is below it: each photon records it in full
    assert sc.lump_decoherence_times(300.0)["total_s"] < 1e-18
    # the colder, the longer: monotonic
    times = [sc.lump_decoherence_times(t)["total_s"] for t in (300.0, 1.0, 1e-3, 1e-5)]
    assert times == sorted(times)


def test_the_record_by_photons_is_capped_by_their_number_and_scaled_below_the_wavelength():
    assert sc.resolving_fraction(0.01, 300.0) == 1.0                      # 17 um photons against 1 cm
    assert sc.resolving_fraction(0.01, 1e-3) == pytest.approx((0.01 / sc.thermal_wavelength_m(1e-3)) ** 2)
    assert sc.thermal_wavelength_m(300.0) == pytest.approx(1.7e-5, rel=0.1)     # Wien's law, in photon-energy form: ~17 um at room temperature


def test_air_in_the_box_records_the_lump_at_once():
    d = sc.lump_decoherence_times(0.0, vacuum=False)
    assert d["gas_s"] < 1e-24 and d["total_s"] == pytest.approx(d["gas_s"])


def test_a_cold_lamp_in_the_box_records_the_lump_even_at_a_picowatt():
    # no heat, only photons: a picowatt of green light is 2.5 million photons a second, and 60 of them a second land on the lump
    d = sc.lump_decoherence_times(0.0, lamp_w=1e-12, lamp_wavelength_m=500e-9)
    assert d["rate_own_per_s"] == 0.0 and d["rate_box_per_s"] == 0.0
    assert d["rate_lamp_per_s"] == pytest.approx(59, rel=0.1) and d["total_s"] == pytest.approx(0.017, rel=0.1)
    # a femtowatt: a thousand times slower, still recorded in seconds
    assert sc.lump_decoherence_times(0.0, lamp_w=1e-15)["total_s"] == pytest.approx(17.0, rel=0.1)
    assert sc.lump_decoherence_times(0.0, lamp_w=0.0)["lamp_s"] == math.inf


def test_a_lamps_long_wavelength_light_records_it_less_per_photon_but_a_watt_of_it_still_does():
    green = sc.lump_decoherence_times(0.0, lamp_w=1e-12, lamp_wavelength_m=500e-9)["rate_lamp_per_s"]
    radio = sc.lump_decoherence_times(0.0, lamp_w=1e-12, lamp_wavelength_m=1.0)["rate_lamp_per_s"]
    assert radio > green                    # a picowatt of 1 m waves is a million times as many photons, each carrying a ten-thousandth of the answer
    assert sc.resolving_fraction_wavelength(0.01, 1.0) == pytest.approx(1e-4) and sc.resolving_fraction_wavelength(0.01, 5e-7) == 1.0
