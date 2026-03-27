"""
Tests for sigma_ground.field.decay

Reference values:
  PDG Review: Radioactive nuclides (pdg.lbl.gov)
  Krane, "Introductory Nuclear Physics" (Wiley, 1988)
  NNDC Nuclear Data — nndc.bnl.gov
"""

import math
import pytest

from sigma_ground.field.decay import (
    decay_constant,
    half_life,
    activity,
    remaining_nuclei,
    q_value_mev,
    q_value_alpha,
    q_value_beta_minus,
    q_value_beta_plus,
    gamow_factor,
    alpha_decay_rate_geiger_nuttall,
    sigma_decay_shift,
)


# ── Decay constant / half-life round-trips ────────────────────────────

def test_decay_constant_carbon14():
    """C-14 half-life = 5730 years (NNDC).

    λ = ln(2) / (5730 × 365.25 × 86400)
    """
    YEAR_S = 365.25 * 86400.0
    t_half = 5730.0 * YEAR_S
    lam = decay_constant(t_half)
    expected = math.log(2) / t_half
    assert lam == pytest.approx(expected, rel=1e-10)


def test_half_life_round_trip():
    """decay_constant and half_life are inverses."""
    t = 1234.567  # seconds
    assert half_life(decay_constant(t)) == pytest.approx(t, rel=1e-10)


def test_decay_constant_zero_raises():
    with pytest.raises(ValueError):
        decay_constant(0.0)


def test_decay_constant_negative_raises():
    with pytest.raises(ValueError):
        decay_constant(-100.0)


def test_half_life_zero_lambda_raises():
    with pytest.raises(ValueError):
        half_life(0.0)


# ── Activity ──────────────────────────────────────────────────────────

def test_activity_positive():
    """Activity is always positive for N > 0."""
    A = activity(1e10, 1000.0)
    assert A > 0


def test_activity_proportional_to_N():
    """Activity scales linearly with N."""
    A1 = activity(1e10, 1000.0)
    A2 = activity(2e10, 1000.0)
    assert A2 == pytest.approx(2 * A1, rel=1e-10)


def test_activity_decreases_with_longer_half_life():
    """Longer-lived nuclides are less active per nucleus."""
    A_short = activity(1e10, 1.0)      # 1 second half-life
    A_long = activity(1e10, 1e9)       # billion second half-life
    assert A_short > A_long


# ── Remaining nuclei ──────────────────────────────────────────────────

def test_remaining_nuclei_at_t0():
    """At t = 0: N = N₀."""
    assert remaining_nuclei(1e12, 0.0, 100.0) == pytest.approx(1e12, rel=1e-10)


def test_remaining_nuclei_at_one_half_life():
    """After 1 half-life: N = N₀/2."""
    N0, t_half = 1e12, 5730.0
    N = remaining_nuclei(N0, t_half, t_half)
    assert N == pytest.approx(N0 / 2.0, rel=1e-10)


def test_remaining_nuclei_at_two_half_lives():
    """After 2 half-lives: N = N₀/4."""
    N0, t_half = 1e12, 100.0
    N = remaining_nuclei(N0, 2 * t_half, t_half)
    assert N == pytest.approx(N0 / 4.0, rel=1e-10)


def test_remaining_nuclei_exponential_decay():
    """N(t) = N₀ e^(−λt)."""
    N0, t_half, t = 1e10, 100.0, 250.0
    N = remaining_nuclei(N0, t, t_half)
    lam = math.log(2) / t_half
    assert N == pytest.approx(N0 * math.exp(-lam * t), rel=1e-10)


# ── Q-values ──────────────────────────────────────────────────────────

def test_q_value_generic():
    """Q = M_parent − sum(M_products)."""
    Q = q_value_mev(100.0, [60.0, 38.0])
    assert Q == pytest.approx(2.0, rel=1e-10)


def test_q_value_alpha_u238():
    """U-238 → Th-234 + α:  Q ≈ 4.270 MeV (PDG/NNDC).

    Atomic masses (AME2020):
      U-238: 221742.931 MeV/c²  (≈ 238 × 931.494 − binding)
      Th-234: 217873.726 MeV/c²
      He-4:     3728.401 MeV/c²  (including electron masses for atomic)

    Use approximate values to test the formula, not the masses.
    """
    # From AME2020 atomic mass excesses (Δ = M − A·u in keV):
    # U-238:  Δ = 47307.1 keV → M = 238 × 931494.102 + 47307.1 keV
    # Th-234: Δ = 40614.1 keV → M = 234 × 931494.102 + 40614.1 keV
    # He-4:   Δ =  2424.9 keV → M = 4 × 931494.102 + 2424.9 keV
    u_mev = 931.494102  # MeV/c² per amu
    M_U238 = 238 * u_mev + 47.307   # MeV
    M_Th234 = 234 * u_mev + 40.614  # MeV
    M_He4 = 4 * u_mev + 2.425       # MeV
    Q = q_value_alpha(M_U238, M_Th234, M_He4)
    assert Q == pytest.approx(4.268, rel=1e-2)  # ~1% tolerance on approximate masses


def test_q_value_alpha_positive_means_spontaneous():
    """Positive Q-value: decay is energetically allowed."""
    Q = q_value_alpha(100.0, 95.0, 3.0)  # 100 − 95 − 3 = 2 MeV
    assert Q > 0


def test_q_value_alpha_negative_means_forbidden():
    """Negative Q-value: decay is energetically forbidden at rest."""
    Q = q_value_alpha(100.0, 98.0, 3.0)  # 100 − 98 − 3 = −1 MeV
    assert Q < 0


def test_q_value_beta_minus_round_trip():
    """Beta minus Q = M_parent − M_daughter in MeV."""
    Q = q_value_beta_minus(100.0, 99.5)
    assert Q == pytest.approx(0.5, rel=1e-10)


def test_q_value_beta_plus_costs_2me():
    """Beta+ Q is reduced by 2m_e relative to beta- with same masses."""
    M_p, M_d = 100.0, 98.0
    Q_minus = q_value_beta_minus(M_p, M_d)   # = 2.0
    Q_plus = q_value_beta_plus(M_p, M_d)     # = 2.0 − 2×0.511 ≈ 0.978
    assert Q_minus - Q_plus == pytest.approx(2 * 0.511, rel=1e-4)


# ── Gamow factor ──────────────────────────────────────────────────────

def test_gamow_factor_positive():
    """Gamow factor is always positive."""
    G = gamow_factor(90, 2, 4.27, 234)
    assert G > 0


def test_gamow_factor_decreases_with_Q():
    """Higher Q (faster alpha) → lower barrier → smaller Gamow factor."""
    G_low_Q = gamow_factor(90, 2, 1.0, 234)
    G_high_Q = gamow_factor(90, 2, 10.0, 234)
    assert G_high_Q < G_low_Q


def test_gamow_factor_increases_with_Z():
    """Higher Z daughter (higher charge barrier) → larger Gamow factor."""
    G_low_Z = gamow_factor(40, 2, 4.0, 100)
    G_high_Z = gamow_factor(80, 2, 4.0, 200)
    assert G_high_Z > G_low_Z


def test_gamow_factor_zero_Q_raises():
    with pytest.raises(ValueError):
        gamow_factor(90, 2, 0.0, 234)


def test_gamow_factor_negative_Q_raises():
    with pytest.raises(ValueError):
        gamow_factor(90, 2, -1.0, 234)


# ── Geiger-Nuttall rate ────────────────────────────────────────────────

def test_alpha_decay_rate_positive():
    """Decay rate is always positive."""
    lam = alpha_decay_rate_geiger_nuttall(92, 238, 4.27)
    assert lam > 0


def test_alpha_decay_rate_increases_with_Q():
    """Higher Q → shorter half-life (Geiger-Nuttall law)."""
    lam_low_Q = alpha_decay_rate_geiger_nuttall(92, 238, 2.0)
    lam_high_Q = alpha_decay_rate_geiger_nuttall(92, 238, 8.0)
    assert lam_high_Q > lam_low_Q


# ── σ-connection ───────────────────────────────────────────────────────

def test_sigma_decay_shift_zero():
    """At σ = 0: λ_eff = λ₀."""
    lam = 1e-5
    assert sigma_decay_shift(0.0, lam) == pytest.approx(lam, rel=1e-12)


def test_sigma_decay_shift_increases():
    """At σ > 0: λ_eff > λ₀ (faster decay in compressed spacetime)."""
    lam = 1e-5
    assert sigma_decay_shift(1.0, lam) > lam


def test_sigma_decay_shift_at_conv():
    """At σ_conv: λ_eff ≈ λ₀/ξ."""
    from sigma_ground.field.constants import SIGMA_CONV, XI
    lam = 1e-5
    lam_eff = sigma_decay_shift(SIGMA_CONV, lam)
    assert lam_eff == pytest.approx(lam / XI, rel=1e-3)


def test_sigma_decay_shift_exponential():
    """λ_eff = λ₀ × e^σ."""
    lam, sigma = 2e-3, 0.75
    assert sigma_decay_shift(sigma, lam) == pytest.approx(lam * math.exp(sigma), rel=1e-10)
