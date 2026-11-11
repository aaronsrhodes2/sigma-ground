"""Oblique Fresnel reflectance, Snell refraction and Beer-Lambert absorption (optics.py), against the textbook values and sigma-ground's own
normal-incidence and critical-angle functions."""
import math

import pytest

from sigma_ground.field.interface import optics, photonics
from sigma_ground.mcp.tools.optics import snells_law_refraction_angle


def test_normal_incidence_is_the_existing_dielectric_reflectance():
    for n in (1.33, 1.5, 2.42):
        R, cos_t = optics.fresnel_interface(1.0, n, 1.0)
        assert R == pytest.approx(optics.dielectric_surface_reflectance(n)) and cos_t == pytest.approx(1.0)


def test_grazing_incidence_reflects_everything():
    assert optics.fresnel_interface(1.0, 1.5, 0.0)[0] == pytest.approx(1.0)


def test_snell_matches_sigma_grounds_own_snell_tool():
    for deg in (10, 35, 60, 80):
        _, cos_t = optics.fresnel_interface(1.0, 1.33, math.cos(math.radians(deg)))
        theta_t = snells_law_refraction_angle(1.0, 1.33, deg).value
        assert math.degrees(math.acos(cos_t)) == pytest.approx(theta_t, abs=1e-9)


def test_brewster_angle_for_glass_reflects_half_of_s_only():
    n = 1.5
    brewster = math.atan(n)                                   # r_p = 0 there, so R = r_s^2 / 2
    R, _ = optics.fresnel_interface(1.0, n, math.cos(brewster))
    assert R == pytest.approx(0.0739, abs=2e-4)


def test_total_internal_reflection_starts_at_the_critical_angle():
    n_dense, n_rare = 1.5, 1.0
    theta_c = photonics.critical_angle(n_dense, n_rare)         # radians
    below = optics.fresnel_interface(n_dense, n_rare, math.cos(theta_c - 0.01))
    beyond = optics.fresnel_interface(n_dense, n_rare, math.cos(theta_c + 0.01))
    assert below[0] < 1.0 and beyond == (1.0, 0.0)


def test_the_table_is_the_function_sampled():
    t = optics.interface_table(1.0, 1.5, samples=9)
    assert len(t["cos_i"]) == 9 and t["cos_i"][0] == 0.0 and t["cos_i"][-1] == 1.0
    for c, r, ct in zip(t["cos_i"], t["reflectance"], t["cos_t"]):
        assert (r, ct) == optics.fresnel_interface(1.0, 1.5, c)


def test_beer_lambert_and_its_table():
    assert optics.beer_lambert_transmittance(0.0, 5.0) == 1.0
    assert optics.beer_lambert_transmittance(0.3, 2.0) == pytest.approx(math.exp(-0.6))
    tab = optics.transmittance_table((0.3, 0.07, 0.02), 0.5, 4)
    assert len(tab) == 3 and len(tab[0]) == 4 and tab[0][0] == 1.0
    assert tab[0][3] == pytest.approx(optics.beer_lambert_transmittance(0.3, 1.5))


def test_srgb_encoding_is_the_iec_61966_curve():
    assert optics.srgb_encode(0.0) == 0.0 and optics.srgb_encode(1.0) == pytest.approx(1.0)
    assert optics.srgb_encode(0.18) == pytest.approx(0.4614, abs=1e-3)              # middle grey
    assert optics.srgb_encode(0.002) == pytest.approx(12.92 * 0.002)                 # the linear toe
    assert optics.srgb_encode(2.0) == pytest.approx(1.0) and optics.srgb_encode(-1.0) == 0.0        # clamped to the displayable range
