"""
Tests for sigma_ground.field.gr_basics

Reference values:
  Misner, Thorne & Wheeler "Gravitation" (Princeton, 1973)
  Hawking 1974, Comm. Math. Phys. 43, 199
  PDG Astrophysical Constants
  NIST CODATA 2018
"""

import math
import pytest

from sigma_ground.field.constants import G, C, HBAR, K_B, M_SUN_KG
from sigma_ground.field.gr_basics import (
    schwarzschild_radius,
    gravitational_redshift,
    time_dilation_gr,
    escape_velocity,
    isco_radius,
    photon_sphere_radius,
    hawking_temperature,
    hawking_luminosity,
    hawking_evaporation_time,
    page_time_sm,
    tidal_force,
)


# ── Schwarzschild radius ───────────────────────────────────────────────

def test_schwarzschild_radius_sun():
    """Solar Schwarzschild radius ≈ 2954 m ≈ 2.95 km.

    Reference: MTW §32.4
    r_s = 2 × 6.674e-11 × 1.989e30 / (3e8)² ≈ 2954 m
    """
    r_s = schwarzschild_radius(M_SUN_KG)
    assert r_s == pytest.approx(2953.0, rel=1e-2)


def test_schwarzschild_radius_earth():
    """Earth's Schwarzschild radius ≈ 8.87 mm.

    M_Earth = 5.972e24 kg
    """
    M_EARTH = 5.972e24  # kg
    r_s = schwarzschild_radius(M_EARTH)
    assert r_s == pytest.approx(8.87e-3, rel=1e-2)


def test_schwarzschild_radius_scales_linearly():
    """r_s ∝ M."""
    r1 = schwarzschild_radius(M_SUN_KG)
    r2 = schwarzschild_radius(2 * M_SUN_KG)
    assert r2 == pytest.approx(2 * r1, rel=1e-10)


def test_schwarzschild_radius_at_rs_equals_escape_velocity_c():
    """At r = r_s: escape velocity = c (defining property of the horizon)."""
    M = 10.0 * M_SUN_KG
    rs = schwarzschild_radius(M)
    v_esc = escape_velocity(M, rs)
    assert v_esc == pytest.approx(C, rel=1e-8)


# ── Gravitational redshift ─────────────────────────────────────────────

def test_gravitational_redshift_large_r():
    """Far from the BH: z → 0."""
    M = M_SUN_KG
    r = 1e15  # 1 petameter — essentially infinity
    z = gravitational_redshift(M, r)
    assert z < 1e-10


def test_gravitational_redshift_approaches_infinity_at_horizon():
    """Just outside the horizon: z is very large."""
    M = M_SUN_KG
    rs = schwarzschild_radius(M)
    r = rs * 1.0001  # 0.01% outside
    z = gravitational_redshift(M, r)
    assert z > 50.0


def test_gravitational_redshift_at_horizon_raises():
    """At r = r_s: z = ∞ → ValueError."""
    M = M_SUN_KG
    rs = schwarzschild_radius(M)
    with pytest.raises(ValueError):
        gravitational_redshift(M, rs)


def test_gravitational_redshift_inside_raises():
    """Inside the horizon: raises ValueError."""
    M = M_SUN_KG
    rs = schwarzschild_radius(M)
    with pytest.raises(ValueError):
        gravitational_redshift(M, rs * 0.5)


# ── GR time dilation ──────────────────────────────────────────────────

def test_time_dilation_gr_far_away():
    """Far from the BH: τ/t ≈ 1 (negligible dilation)."""
    M = M_SUN_KG
    r = 1e15
    factor = time_dilation_gr(M, r)
    assert factor == pytest.approx(1.0, rel=1e-10)


def test_time_dilation_gr_near_horizon():
    """Near horizon: τ/t approaches 0."""
    M = M_SUN_KG
    rs = schwarzschild_radius(M)
    r = rs * 1.01  # 1% outside
    factor = time_dilation_gr(M, r)
    assert factor < 0.15


def test_time_dilation_gr_between_zero_and_one():
    """GR time dilation factor ∈ (0, 1] outside horizon."""
    M = 10 * M_SUN_KG
    rs = schwarzschild_radius(M)
    for r_mult in [1.001, 1.1, 2.0, 10.0, 100.0]:
        factor = time_dilation_gr(M, rs * r_mult)
        assert 0 < factor <= 1


# ── Escape velocity ────────────────────────────────────────────────────

def test_escape_velocity_earth_surface():
    """Earth surface escape velocity ≈ 11.2 km/s."""
    M_EARTH = 5.972e24
    R_EARTH = 6.371e6
    v_esc = escape_velocity(M_EARTH, R_EARTH)
    assert v_esc == pytest.approx(11186.0, rel=1e-2)


def test_escape_velocity_decreases_with_r():
    """Escape velocity decreases at larger radii."""
    M = M_SUN_KG
    v1 = escape_velocity(M, 1e9)
    v2 = escape_velocity(M, 2e9)
    assert v2 < v1


def test_escape_velocity_zero_radius_raises():
    with pytest.raises(ValueError):
        escape_velocity(M_SUN_KG, 0.0)


# ── Special radii ──────────────────────────────────────────────────────

def test_isco_is_three_rs():
    """r_ISCO = 3 r_s for a Schwarzschild BH."""
    M = 10 * M_SUN_KG
    rs = schwarzschild_radius(M)
    assert isco_radius(M) == pytest.approx(3 * rs, rel=1e-10)


def test_photon_sphere_is_15_rs():
    """r_ph = 1.5 r_s for a Schwarzschild BH."""
    M = 10 * M_SUN_KG
    rs = schwarzschild_radius(M)
    assert photon_sphere_radius(M) == pytest.approx(1.5 * rs, rel=1e-10)


def test_radii_ordering():
    """r_s < r_ph < r_ISCO."""
    M = 10 * M_SUN_KG
    rs = schwarzschild_radius(M)
    r_ph = photon_sphere_radius(M)
    r_isco = isco_radius(M)
    assert rs < r_ph < r_isco


# ── Hawking radiation ──────────────────────────────────────────────────

def test_hawking_temperature_solar_mass():
    """Solar-mass BH: T_H ≈ 6.17e-8 K (imperceptibly cold).

    Reference: Hawking 1974
    T_H = ℏc³/(8πGMk_B)
    """
    T_H = hawking_temperature(M_SUN_KG)
    assert T_H == pytest.approx(6.17e-8, rel=1e-2)


def test_hawking_temperature_smaller_mass_is_hotter():
    """Smaller BH → hotter Hawking temperature."""
    T1 = hawking_temperature(M_SUN_KG)
    T2 = hawking_temperature(0.001 * M_SUN_KG)
    assert T2 > T1


def test_hawking_temperature_scales_inverse_M():
    """T_H ∝ 1/M."""
    T1 = hawking_temperature(M_SUN_KG)
    T2 = hawking_temperature(2 * M_SUN_KG)
    assert T2 == pytest.approx(T1 / 2, rel=1e-10)


def test_hawking_temperature_zero_mass_raises():
    with pytest.raises(ValueError):
        hawking_temperature(0.0)


def test_hawking_luminosity_positive():
    """Hawking luminosity is always positive."""
    assert hawking_luminosity(M_SUN_KG) > 0


def test_hawking_luminosity_scales_inverse_M_squared():
    """L_H ∝ 1/M²."""
    L1 = hawking_luminosity(M_SUN_KG)
    L2 = hawking_luminosity(2 * M_SUN_KG)
    assert L2 == pytest.approx(L1 / 4, rel=1e-10)


def test_hawking_evaporation_time_solar():
    """Solar-mass BH evaporation time ≫ age of universe (≫ 4e17 s)."""
    t = hawking_evaporation_time(M_SUN_KG)
    age_universe_s = 13.8e9 * 365.25 * 86400  # ≈ 4.35e17 s
    assert t > 1e50 * age_universe_s  # absurdly longer


def test_hawking_evaporation_time_scales_M_cubed():
    """t_evap ∝ M³."""
    t1 = hawking_evaporation_time(M_SUN_KG)
    t2 = hawking_evaporation_time(2 * M_SUN_KG)
    assert t2 == pytest.approx(8 * t1, rel=1e-10)


# ── Tidal forces ───────────────────────────────────────────────────────

def test_tidal_force_positive():
    """Tidal force is positive for positive dr."""
    assert tidal_force(M_SUN_KG, 1e9, 1.0) > 0


def test_tidal_force_decreases_with_r():
    """Tidal force ∝ 1/r³: stronger close to the mass."""
    F1 = tidal_force(M_SUN_KG, 1e9, 1.0)
    F2 = tidal_force(M_SUN_KG, 2e9, 1.0)
    assert F2 == pytest.approx(F1 / 8, rel=1e-10)


def test_tidal_force_zero_radius_raises():
    with pytest.raises(ValueError):
        tidal_force(M_SUN_KG, 0.0, 1.0)


def test_page_time_sm_is_fraction_of_evaporation():
    """SM Page time must be a positive fraction of total evaporation time."""
    from sigma_ground.field.gr_basics import hawking_evaporation_time
    M = 1.0e12  # primordial BH scale (kg) — Page-time is astronomically long
    t_sm = page_time_sm(M)
    t_evap = hawking_evaporation_time(M)
    assert 0.0 < t_sm < t_evap
    assert t_sm == pytest.approx(0.5384 * t_evap, rel=1e-10)


def test_page_time_sm_scales_with_M_cubed():
    """Page time inherits the M^3 scaling of Hawking evaporation time."""
    ratio = page_time_sm(2.0e12) / page_time_sm(1.0e12)
    assert ratio == pytest.approx(8.0, rel=1e-10)


def test_page_time_sm_rejects_nonpositive_mass():
    """page_time_sm rejects M <= 0."""
    with pytest.raises(ValueError):
        page_time_sm(0.0)
    with pytest.raises(ValueError):
        page_time_sm(-1.0)


def test_bekenstein_hawking_entropy_positive():
    from sigma_ground.field.gr_basics import bekenstein_hawking_entropy
    s = bekenstein_hawking_entropy(M_SUN_KG)
    assert s > 0


def test_bekenstein_hawking_entropy_scales_M_squared():
    """S_BH ∝ M²."""
    from sigma_ground.field.gr_basics import bekenstein_hawking_entropy
    s1 = bekenstein_hawking_entropy(1e30)
    s2 = bekenstein_hawking_entropy(2e30)
    assert s2 / s1 == pytest.approx(4.0, rel=1e-10)


def test_bekenstein_hawking_solar_mass_value():
    """S_BH(M_sun) ≈ 1.5×10⁵⁴ J/K (canonical textbook value)."""
    from sigma_ground.field.gr_basics import bekenstein_hawking_entropy
    s = bekenstein_hawking_entropy(M_SUN_KG)
    # Textbook: S_BH(M_sun) ≈ 1.05 × 10⁷⁷ in units of k_B
    # In SI: S ~ 1.45 × 10⁵⁴ J/K
    assert 1e54 < s < 2e54


def test_bekenstein_hawking_rejects_nonpositive_mass():
    from sigma_ground.field.gr_basics import bekenstein_hawking_entropy
    with pytest.raises(ValueError):
        bekenstein_hawking_entropy(0.0)
    with pytest.raises(ValueError):
        bekenstein_hawking_entropy(-1.0)
