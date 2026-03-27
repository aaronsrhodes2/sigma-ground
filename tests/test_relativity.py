"""
Tests for sigma_ground.field.relativity

Reference values from:
  - NIST CODATA 2018
  - Jackson "Classical Electrodynamics" 3rd ed.
  - Einstein 1905 (Ann. Physik 17, 891)
"""

import math
import pytest

from sigma_ground.field.constants import C, M_ELECTRON_KG, E_CHARGE
from sigma_ground.field.relativity import (
    lorentz_factor,
    beta,
    rest_energy,
    relativistic_energy,
    kinetic_energy_rel,
    momentum_rel,
    energy_momentum_invariant,
    velocity_addition,
    length_contraction,
    time_dilation,
    doppler_factor,
    sigma_time_dilation,
)


# ── Lorentz factor ─────────────────────────────────────────────────────

def test_lorentz_factor_zero():
    """At rest: γ = 1."""
    assert lorentz_factor(0) == pytest.approx(1.0)


def test_lorentz_factor_classical():
    """At v = 0.1c: γ ≈ 1.005 (nearly classical)."""
    gamma = lorentz_factor(0.1 * C)
    assert gamma == pytest.approx(1.0 / math.sqrt(1 - 0.01), rel=1e-6)


def test_lorentz_factor_relativistic():
    """At v = 0.99c: γ ≈ 7.089."""
    gamma = lorentz_factor(0.99 * C)
    expected = 1.0 / math.sqrt(1 - 0.99**2)
    assert gamma == pytest.approx(expected, rel=1e-6)


def test_lorentz_factor_negative_v():
    """Sign of v is irrelevant — γ depends on |v|."""
    assert lorentz_factor(-0.5 * C) == pytest.approx(lorentz_factor(0.5 * C))


def test_lorentz_factor_at_c_raises():
    """At v = c: ValueError (γ diverges)."""
    with pytest.raises(ValueError):
        lorentz_factor(C)


def test_lorentz_factor_above_c_raises():
    """Above c: ValueError."""
    with pytest.raises(ValueError):
        lorentz_factor(1.01 * C)


def test_lorentz_factor_high():
    """At v = 0.9999c: γ ≈ 70.7."""
    gamma = lorentz_factor(0.9999 * C)
    assert gamma > 70.0


def test_beta_range():
    """β is between 0 and 1 for any physical speed."""
    b = beta(0.5 * C)
    assert 0 < b < 1
    assert b == pytest.approx(0.5)


# ── Rest energy ────────────────────────────────────────────────────────

def test_rest_energy_electron():
    """Electron rest energy = 0.51099895 MeV (NIST CODATA 2018)."""
    E_j = rest_energy(M_ELECTRON_KG)
    E_mev = E_j / (E_CHARGE * 1e6)   # J → MeV
    assert E_mev == pytest.approx(0.51099895, rel=1e-4)


def test_rest_energy_proton():
    """Proton rest energy = 938.272 MeV."""
    from sigma_ground.field.constants import PROTON_TOTAL_MEV
    M_PROTON_KG = PROTON_TOTAL_MEV * 1e6 * E_CHARGE / C**2
    E_mev = rest_energy(M_PROTON_KG) / (E_CHARGE * 1e6)
    assert E_mev == pytest.approx(PROTON_TOTAL_MEV, rel=1e-4)


# ── Relativistic energy & momentum ────────────────────────────────────

def test_relativistic_energy_low_v():
    """At v ≪ c: E ≈ m₀c² + ½m₀v² (rest + classical KE)."""
    v = 1000.0  # 1 km/s ≪ c
    E = relativistic_energy(M_ELECTRON_KG, v)
    E0 = rest_energy(M_ELECTRON_KG)
    KE_classical = 0.5 * M_ELECTRON_KG * v**2
    assert E == pytest.approx(E0 + KE_classical, rel=1e-8)


def test_kinetic_energy_rel_zero():
    """At v = 0: kinetic energy is zero."""
    assert kinetic_energy_rel(1.0, 0.0) == pytest.approx(0.0, abs=1e-30)


def test_kinetic_energy_rel_classical_limit():
    """At v ≪ c: K_rel ≈ K_classical = ½mv².

    Uses v = 1e5 m/s (0.033% of c) where:
      - relativistic correction  ≈ 3β²/4 ≈ 8e-8 (negligible)
      - float precision in (γ-1) is adequate (β² ≈ 1.1e-7, no bad cancellation)
    """
    v = 1e5   # 100 km/s — still ≪ c (β ≈ 3.3e-4)
    m = 1.0
    K_rel = kinetic_energy_rel(m, v)
    K_classical = 0.5 * m * v**2
    assert K_rel == pytest.approx(K_classical, rel=1e-6)


def test_momentum_rel_zero():
    """At v = 0: momentum = 0."""
    assert momentum_rel(1.0, 0.0) == pytest.approx(0.0, abs=1e-30)


def test_momentum_rel_classical_limit():
    """At low v: p ≈ m₀v."""
    v = 100.0
    m = 2.0
    p = momentum_rel(m, v)
    assert p == pytest.approx(m * v, rel=1e-10)


def test_energy_momentum_invariant():
    """E² − (pc)² = (m₀c²)² must hold at any speed."""
    m0 = M_ELECTRON_KG
    v = 0.8 * C
    E = relativistic_energy(m0, v)
    p = momentum_rel(m0, v)
    lhs = E**2 - (p * C)**2
    rhs = energy_momentum_invariant(m0)
    assert lhs == pytest.approx(rhs, rel=1e-10)


# ── Kinematics ─────────────────────────────────────────────────────────

def test_velocity_addition_classical_limit():
    """At low v: addition is classical."""
    u, v = 1000.0, 2000.0
    assert velocity_addition(u, v) == pytest.approx(u + v, rel=1e-10)


def test_velocity_addition_speed_of_light():
    """u = v = 0.6c: result must be < c."""
    w = velocity_addition(0.6 * C, 0.6 * C)
    assert w < C
    assert w == pytest.approx(0.6 * C * 2 / (1 + 0.36), rel=1e-10)


def test_velocity_addition_antisymmetric():
    """Adding v and -v gives zero."""
    w = velocity_addition(0.5 * C, -0.5 * C)
    assert w == pytest.approx(0.0, abs=1e-6)


def test_length_contraction():
    """At v = 0.6c: L = L₀ × 0.8 (γ = 5/4, contraction = 4/5)."""
    # γ at 0.6c = 1/√(1−0.36) = 1/√0.64 = 1/0.8 = 1.25
    L0 = 10.0
    L = length_contraction(L0, 0.6 * C)
    assert L == pytest.approx(L0 / (1.0 / 0.8), rel=1e-6)


def test_time_dilation_at_rest():
    """At v = 0: no dilation, t = t₀."""
    assert time_dilation(5.0, 0.0) == pytest.approx(5.0)


def test_time_dilation_grows_with_v():
    """Higher speed → more dilation."""
    t1 = time_dilation(1.0, 0.5 * C)
    t2 = time_dilation(1.0, 0.9 * C)
    assert t2 > t1 > 1.0


def test_doppler_head_on_blueshift():
    """Head-on approach (cos θ = 1): observed frequency > emitted."""
    D = doppler_factor(0.5 * C, cos_theta=1.0)
    assert D > 1.0


def test_doppler_receding_redshift():
    """Receding source (cos θ = −1): observed frequency < emitted."""
    D = doppler_factor(0.5 * C, cos_theta=-1.0)
    assert D < 1.0


def test_doppler_transverse_redshift():
    """Transverse motion (cos θ = 0): purely time-dilation redshift."""
    v = 0.5 * C
    D = doppler_factor(v, cos_theta=0.0)
    gamma = lorentz_factor(v)
    assert D == pytest.approx(1.0 / gamma, rel=1e-10)


# ── σ-connection ───────────────────────────────────────────────────────

def test_sigma_time_dilation_zero():
    """At σ = 0: no dilation, t_coord = t₀."""
    assert sigma_time_dilation(0.0, 1.0) == pytest.approx(1.0)


def test_sigma_time_dilation_positive():
    """At σ > 0: coordinate time is longer than proper time."""
    t_coord = sigma_time_dilation(1.0, 1.0)
    assert t_coord == pytest.approx(math.e, rel=1e-10)


def test_sigma_time_dilation_at_conv():
    """At σ_conv ≈ 1.849: dilation factor ≈ 1/ξ = 1/0.1582 ≈ 6.32."""
    from sigma_ground.field.constants import SIGMA_CONV, XI
    dilation = sigma_time_dilation(SIGMA_CONV, 1.0)
    assert dilation == pytest.approx(1.0 / XI, rel=1e-4)


def test_sigma_time_dilation_linearity():
    """Dilation is linear in t₀."""
    t0 = 7.5
    sigma = 0.5
    result = sigma_time_dilation(sigma, t0)
    assert result == pytest.approx(t0 * math.exp(sigma), rel=1e-10)
