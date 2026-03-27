"""
Tests for sigma_ground.field.interface.plasma

Reference values:
  NRL Plasma Formulary 2019
  Goldston & Rutherford, "Introduction to Plasma Physics"
  Solar wind / corona typical parameters from literature
"""

import math
import pytest

from sigma_ground.field.constants import E_CHARGE, EPS_0, MU_0, M_ELECTRON_KG, K_B
from sigma_ground.field.interface.plasma import (
    plasma_frequency,
    plasma_frequency_hz,
    debye_length,
    debye_number,
    alfven_speed,
    cyclotron_radius,
    plasma_beta,
    coulomb_logarithm,
    spitzer_resistivity,
    sigma_plasma_transition,
)


# ── Plasma frequency ───────────────────────────────────────────────────

def test_plasma_frequency_positive():
    """Plasma frequency is always positive."""
    assert plasma_frequency(1e18) > 0


def test_plasma_frequency_scales_sqrt_n():
    """ω_p ∝ √n_e: quadrupling n doubles ω_p."""
    wp1 = plasma_frequency(1e18)
    wp4 = plasma_frequency(4e18)
    assert wp4 == pytest.approx(2 * wp1, rel=1e-10)


def test_plasma_frequency_solar_corona():
    """Solar corona: n_e ≈ 1e14 m⁻³ → f_p ≈ 2.8 MHz (radio).

    Reference: Goldston & Rutherford, Table 1.2
    """
    n_e_corona = 1e14  # m⁻³
    f_p = plasma_frequency_hz(n_e_corona)
    # ω_p = √(1e14 × (1.6e-19)² / (8.85e-12 × 9.11e-31)) ≈ 1.78e7 rad/s
    # f_p ≈ 2.83 MHz
    assert 1e6 < f_p < 1e8  # between 1 MHz and 100 MHz


def test_plasma_frequency_hz_vs_angular():
    """f_p = ω_p / (2π)."""
    n_e = 1e18
    assert plasma_frequency_hz(n_e) == pytest.approx(
        plasma_frequency(n_e) / (2 * math.pi), rel=1e-10
    )


def test_plasma_frequency_zero_density_raises():
    with pytest.raises(ValueError):
        plasma_frequency(0.0)


# ── Debye length ───────────────────────────────────────────────────────

def test_debye_length_positive():
    """Debye length is always positive."""
    assert debye_length(1e18, 10000.0) > 0


def test_debye_length_increases_with_T():
    """Hotter plasma → longer Debye length."""
    lam1 = debye_length(1e18, 1000.0)
    lam2 = debye_length(1e18, 10000.0)
    assert lam2 > lam1


def test_debye_length_decreases_with_n():
    """Higher density → shorter Debye length."""
    lam1 = debye_length(1e18, 1000.0)
    lam2 = debye_length(1e20, 1000.0)
    assert lam2 < lam1


def test_debye_length_scales_sqrt_T():
    """λ_D ∝ √T: quadrupling T doubles λ_D."""
    lam1 = debye_length(1e18, 1000.0)
    lam4 = debye_length(1e18, 4000.0)
    assert lam4 == pytest.approx(2 * lam1, rel=1e-10)


def test_debye_length_scales_inverse_sqrt_n():
    """λ_D ∝ 1/√n_e."""
    lam1 = debye_length(1e18, 1000.0)
    lam4 = debye_length(4e18, 1000.0)
    assert lam4 == pytest.approx(lam1 / 2, rel=1e-10)


def test_debye_length_zero_density_raises():
    with pytest.raises(ValueError):
        debye_length(0.0, 1000.0)


def test_debye_length_zero_temperature_raises():
    with pytest.raises(ValueError):
        debye_length(1e18, 0.0)


# ── Debye number ───────────────────────────────────────────────────────

def test_debye_number_positive():
    assert debye_number(1e18, 10000.0) > 0


def test_debye_number_lab_plasma():
    """Lab plasma (n=1e18 m⁻³, T=10000 K) should have N_D > 1."""
    assert debye_number(1e18, 10000.0) > 1


# ── Alfvén speed ───────────────────────────────────────────────────────

def test_alfven_speed_solar_wind():
    """Solar wind: B ≈ 5 nT, ρ ≈ 5 mp/cm³.

    B = 5e-9 T, n_p = 5e6 m⁻³, rho = n_p × m_p ≈ 8.35e-21 kg/m³
    v_A ≈ 50 km/s (typical solar wind)
    """
    m_proton = 1.67262192369e-27  # kg
    B = 5e-9      # T
    n_p = 5e6     # m⁻³
    rho = n_p * m_proton
    v_A = alfven_speed(B, rho)
    assert 10e3 < v_A < 200e3   # between 10 and 200 km/s


def test_alfven_speed_scales_with_B():
    """v_A ∝ B."""
    rho = 1e-6
    v1 = alfven_speed(1.0, rho)
    v2 = alfven_speed(2.0, rho)
    assert v2 == pytest.approx(2 * v1, rel=1e-10)


def test_alfven_speed_scales_inverse_sqrt_rho():
    """v_A ∝ 1/√ρ."""
    B = 1e-3
    v1 = alfven_speed(B, 1.0)
    v4 = alfven_speed(B, 4.0)
    assert v4 == pytest.approx(v1 / 2, rel=1e-10)


def test_alfven_speed_zero_density_raises():
    with pytest.raises(ValueError):
        alfven_speed(1e-3, 0.0)


# ── Cyclotron radius ───────────────────────────────────────────────────

def test_cyclotron_radius_positive():
    assert cyclotron_radius(M_ELECTRON_KG, 1e6, 0.1) > 0


def test_cyclotron_radius_scales_with_v():
    """r_c ∝ v_⊥."""
    r1 = cyclotron_radius(M_ELECTRON_KG, 1e6, 0.1)
    r2 = cyclotron_radius(M_ELECTRON_KG, 2e6, 0.1)
    assert r2 == pytest.approx(2 * r1, rel=1e-10)


def test_cyclotron_radius_scales_inverse_B():
    """r_c ∝ 1/B."""
    r1 = cyclotron_radius(M_ELECTRON_KG, 1e6, 0.1)
    r2 = cyclotron_radius(M_ELECTRON_KG, 1e6, 0.2)
    assert r2 == pytest.approx(r1 / 2, rel=1e-10)


# ── Plasma beta ────────────────────────────────────────────────────────

def test_plasma_beta_positive():
    assert plasma_beta(1e18, 10000.0, 0.01) > 0


def test_plasma_beta_magnetically_dominated():
    """Strong B, low T: β ≪ 1."""
    beta = plasma_beta(1e15, 100.0, 10.0)
    assert beta < 0.01


def test_plasma_beta_thermally_dominated():
    """Weak B, high T: β ≫ 1."""
    beta = plasma_beta(1e22, 1e6, 1e-6)
    assert beta > 100


# ── Coulomb logarithm ──────────────────────────────────────────────────

def test_coulomb_logarithm_typical_range():
    """ln Λ is typically 10–20 for laboratory / astrophysical plasmas."""
    lnL = coulomb_logarithm(1e18, 10000.0)
    assert 5 < lnL < 30


def test_coulomb_logarithm_increases_with_T():
    """Hotter plasma → larger Debye sphere → larger ln Λ."""
    lnL_cold = coulomb_logarithm(1e18, 100.0)
    lnL_hot = coulomb_logarithm(1e18, 1e6)
    assert lnL_hot > lnL_cold


# ── Spitzer resistivity ────────────────────────────────────────────────

def test_spitzer_resistivity_positive():
    assert spitzer_resistivity(10000.0) > 0


def test_spitzer_resistivity_decreases_with_T():
    """Hotter plasma → lower resistivity (η ∝ T^{-3/2})."""
    eta_cold = spitzer_resistivity(1000.0)
    eta_hot = spitzer_resistivity(10000.0)
    assert eta_hot < eta_cold


# ── σ-connection ───────────────────────────────────────────────────────

def test_sigma_plasma_transition_zero():
    """At σ = 0: scale factor = 1."""
    scale = sigma_plasma_transition(0.0)
    assert scale == pytest.approx(1.0)


def test_sigma_plasma_transition_positive():
    """At σ > 0: scale factor > 1 (density enhancement)."""
    scale = sigma_plasma_transition(1.0)
    assert scale > 1.0
    assert scale == pytest.approx(math.exp(1.5), rel=1e-10)


def test_sigma_plasma_transition_effective_freq():
    """ω_p,eff = ω_p × sigma_plasma_transition(σ)."""
    n_e = 1e18
    sigma = 0.5
    wp_standard = plasma_frequency(n_e)
    scale = sigma_plasma_transition(sigma)
    wp_eff = wp_standard * scale
    assert wp_eff > wp_standard
