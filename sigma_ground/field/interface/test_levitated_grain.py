"""Tests for the resting and floating osmium grain (levitated_grain.py)."""
import math

import pytest

from sigma_ground.field.interface import levitated_grain as lg
from sigma_ground.field.interface import schrodinger_cat as sc


def test_a_20_micron_grain_weighs_nanonewtons():
    assert lg.mass_kg() == pytest.approx(7.6e-10, rel=0.02)
    assert lg.weight_n() == pytest.approx(7.4e-9, rel=0.02)


def test_the_hertz_patch_is_tens_of_nanometres_and_carries_tens_of_megapascals():
    h = lg.hertz_contact()
    assert 5e-9 < h["patch_radius_m"] < 5e-8
    assert 5e6 < h["peak_pressure_pa"] < 1e8
    assert h["peak_pressure_pa"] == pytest.approx(1.5 * h["mean_pressure_pa"])                 # 3F / 2 pi a^2 against F / pi a^2
    assert h["mean_pressure_pa"] * math.pi * h["patch_radius_m"] ** 2 == pytest.approx(h["weight_n"])
    assert h["approach_m"] == pytest.approx(h["patch_radius_m"] ** 2 / lg.GRAIN_RADIUS_M)


def test_hertz_scales_as_the_cube_root_of_the_load():
    a1 = lg.hertz_contact(10e-6)["patch_radius_m"]
    a2 = lg.hertz_contact(20e-6)["patch_radius_m"]
    # W ~ R^3 and a ~ (W R)^(1/3) ~ R^(4/3)
    assert a2 / a1 == pytest.approx(2 ** (4 / 3), rel=1e-6)


def test_the_combined_modulus_is_dominated_by_the_softer_body():
    assert lg.combined_modulus() < lg.FLOOR_E_PA / (1 - lg.FLOOR_NU ** 2) * 1.01
    assert lg.combined_modulus(1e12, 0.25, 1e12, 0.25) == pytest.approx(1e12 / (2 * (1 - 0.0625)))


def test_osmiums_critical_field_is_a_few_millitesla_and_the_limit_a_few_tens_of_microns():
    bc = lg.critical_field_t()
    assert 3e-3 < bc < 9e-3                                           # the measured value is ~7 mT; the free-electron estimate is a lower bound
    assert lg.critical_field_t(0.66) == 0.0 and lg.critical_field_t(0.33) == pytest.approx(0.75 * bc)     # H_c(T) = H_c(0)(1 - (T/T_c)^2)
    assert 1e-5 < lg.levitation_limit_radius_m() < 1e-4
    # at the limit the field pressure over the projected area carries exactly the weight
    r = lg.levitation_limit_radius_m()
    assert lg.field_pressure_limit_pa() * math.pi * r ** 2 == pytest.approx(lg.weight_n(r), rel=1e-9)


def test_the_grain_floats_and_the_one_centimetre_lump_cannot():
    assert lg.can_float(lg.GRAIN_RADIUS_M)
    assert not lg.can_float(sc.LUMP_RADIUS_M)                          # 95 g of osmium is hundreds of times beyond what the field can carry
    assert not lg.can_float(lg.GRAIN_RADIUS_M, temperature_k=0.65)     # near T_c the critical field is nearly gone


def test_a_photon_changes_nothing_you_could_see_in_the_grain():
    e = lg.photon_effects(500e-9)
    assert e["momentum_kg_m_s"] == pytest.approx(1.33e-27, rel=0.01)
    assert e["recoil_m_s"] < 1e-16                                      # a sub-attometre-per-second kick
    assert e["temperature_rise_k"] < 1e-11
    assert lg.flight_time_s(1.0) == pytest.approx(3.34e-9, rel=0.01)


def test_the_grains_decoherence_by_a_lamp_matches_the_lump_models_scaling():
    # the same model with the grain's radius and separation: the photons that hit scale with the cross-section
    big = sc.lump_decoherence_times(0.0, lamp_w=1e-6)["rate_lamp_per_s"]
    small = sc.lump_decoherence_times(0.0, radius_m=lg.GRAIN_RADIUS_M, separation_m=lg.GRAIN_SEPARATION_M, lamp_w=1e-6)["rate_lamp_per_s"]
    assert small / big == pytest.approx((lg.GRAIN_RADIUS_M / sc.LUMP_RADIUS_M) ** 2, rel=1e-6)
    assert 1 / small == pytest.approx(4e-3, rel=0.3)                    # a microwatt of green light records it in a few milliseconds
