"""
Tests for sigma_ground.field.interface.statistical

Reference values:
  Reif, "Fundamentals of Statistical and Thermal Physics"
  NIST: nitrogen molecular mass, standard conditions
"""

import math
import pytest

from sigma_ground.field.constants import K_B
from sigma_ground.field.interface.statistical import (
    boltzmann_factor,
    partition_function,
    mean_energy,
    entropy_from_partition,
    boltzmann_entropy,
    fermi_dirac,
    bose_einstein,
    maxwell_speed_dist,
    rms_speed,
    mean_speed,
    most_probable_speed,
    heat_capacity_equipartition,
    sigma_partition_shift,
)


# ── Boltzmann factor ───────────────────────────────────────────────────

def test_boltzmann_factor_zero_energy():
    """At E = 0: factor = 1 regardless of T."""
    assert boltzmann_factor(0.0, 300.0) == pytest.approx(1.0)


def test_boltzmann_factor_high_energy():
    """High energy relative to kT: factor → 0."""
    f = boltzmann_factor(1e-18, 300.0)  # E ≫ kT ≈ 4.1e-21 J
    assert f < 1e-100


def test_boltzmann_factor_negative_T_raises():
    with pytest.raises(ValueError):
        boltzmann_factor(1.0, -10.0)


def test_boltzmann_factor_zero_T_raises():
    with pytest.raises(ValueError):
        boltzmann_factor(1.0, 0.0)


def test_boltzmann_factor_ratio():
    """Ratio of factors for two energies = e^(−ΔE/kT)."""
    E1, E2 = 1e-21, 2e-21
    T = 300.0
    f1 = boltzmann_factor(E1, T)
    f2 = boltzmann_factor(E2, T)
    expected_ratio = math.exp(-(E2 - E1) / (K_B * T))
    assert f2 / f1 == pytest.approx(expected_ratio, rel=1e-10)


# ── Partition function ─────────────────────────────────────────────────

def test_partition_function_single_state():
    """Single state at E = 0: Z = 1."""
    Z = partition_function([0.0], 300.0)
    assert Z == pytest.approx(1.0)


def test_partition_function_two_degenerate_states():
    """Two states at E = 0: Z = 2."""
    Z = partition_function([0.0, 0.0], 300.0)
    assert Z == pytest.approx(2.0)


def test_partition_function_high_T_limit():
    """At T → ∞: all Boltzmann factors → 1, Z → number of states."""
    energies = [0.0, K_B * 1.0, K_B * 2.0]  # spacings ~ 1 K
    Z_high = partition_function(energies, 1e10)  # very high T
    assert Z_high == pytest.approx(3.0, rel=1e-6)


def test_partition_function_negative_T_raises():
    with pytest.raises(ValueError):
        partition_function([1e-21], -10.0)


# ── Mean energy ────────────────────────────────────────────────────────

def test_mean_energy_single_state():
    """Single state at energy E: ⟨E⟩ = E."""
    E0 = 1e-21
    assert mean_energy([E0], 300.0) == pytest.approx(E0, rel=1e-10)


def test_mean_energy_two_equal_states_low_T():
    """Two states; ground state dominates at low T: ⟨E⟩ → 0."""
    E_low, E_high = 0.0, 1e-18  # E_high ≫ kT
    mu = mean_energy([E_low, E_high], 300.0)
    assert mu < 1e-40  # essentially zero


def test_mean_energy_two_equal_states_high_T():
    """Two states at 0 and ΔE; at T → ∞: ⟨E⟩ → ΔE/2."""
    dE = 1e-21
    mu = mean_energy([0.0, dE], 1e8)
    assert mu == pytest.approx(dE / 2, rel=1e-3)


# ── Entropy ────────────────────────────────────────────────────────────

def test_boltzmann_entropy_one_state():
    """W = 1: S = 0 (only one microstate)."""
    assert boltzmann_entropy(1) == pytest.approx(0.0, abs=1e-40)


def test_boltzmann_entropy_increases_with_W():
    """More microstates = more entropy."""
    S1 = boltzmann_entropy(10)
    S2 = boltzmann_entropy(100)
    assert S2 > S1


def test_boltzmann_entropy_additive():
    """S(W₁ × W₂) = S(W₁) + S(W₂)."""
    W1, W2 = 10, 20
    S_combined = boltzmann_entropy(W1 * W2)
    S_sum = boltzmann_entropy(W1) + boltzmann_entropy(W2)
    assert S_combined == pytest.approx(S_sum, rel=1e-10)


def test_boltzmann_entropy_zero_W_raises():
    with pytest.raises(ValueError):
        boltzmann_entropy(0)


# ── Fermi-Dirac ────────────────────────────────────────────────────────

def test_fermi_dirac_at_fermi_energy():
    """At E = E_F: occupation = 0.5 for any T > 0."""
    E_f = 5e-19  # ~ 3 eV
    f = fermi_dirac(E_f, E_f, 300.0)
    assert f == pytest.approx(0.5, rel=1e-10)


def test_fermi_dirac_deep_below():
    """Far below E_F: occupation ≈ 1."""
    E_f = 5e-19
    E = 1e-21  # much less than E_F
    f = fermi_dirac(E, E_f, 300.0)
    assert f > 0.9999


def test_fermi_dirac_far_above():
    """Far above E_F: occupation ≈ 0."""
    E_f = 5e-21
    E = 5e-19  # much greater than E_F
    f = fermi_dirac(E, E_f, 300.0)
    assert f < 1e-10


def test_fermi_dirac_t_zero_step():
    """At T = 0: exact step function."""
    E_f = 5e-19
    assert fermi_dirac(E_f - 1e-25, E_f, 0) == 1.0
    assert fermi_dirac(E_f + 1e-25, E_f, 0) == 0.0
    assert fermi_dirac(E_f, E_f, 0) == 0.5


# ── Bose-Einstein ──────────────────────────────────────────────────────

def test_bose_einstein_positive():
    """Occupation number must be positive for E > mu."""
    E_j = 2e-21
    mu = 1e-21
    n = bose_einstein(E_j, mu, 300.0)
    assert n > 0


def test_bose_einstein_increases_as_e_approaches_mu():
    """As E → μ from above, n → ∞ (condensate onset)."""
    mu = 1e-21
    T = 300.0
    n1 = bose_einstein(mu + 1e-21, mu, T)
    n2 = bose_einstein(mu + 1e-24, mu, T)
    assert n2 > n1


def test_bose_einstein_e_equal_mu_raises():
    with pytest.raises(ValueError):
        bose_einstein(1e-21, 1e-21, 300.0)


def test_bose_einstein_e_less_than_mu_raises():
    with pytest.raises(ValueError):
        bose_einstein(0.5e-21, 1e-21, 300.0)


# ── Maxwell-Boltzmann speed distribution ──────────────────────────────

def test_rms_speed_nitrogen_300K():
    """N₂ at 300 K: v_rms = √(3kT/m) ≈ 517 m/s.

    M_N2 = 28 g/mol → m = 28e-3 / 6.022e23 kg
    """
    M_N2 = 28.014e-3 / 6.02214076e23  # kg per molecule
    v = rms_speed(M_N2, 300.0)
    assert v == pytest.approx(517.0, rel=1e-2)


def test_rms_speed_increases_with_T():
    """Hotter gas → higher RMS speed."""
    m = 1e-26  # kg
    v1 = rms_speed(m, 300.0)
    v2 = rms_speed(m, 1200.0)
    assert v2 == pytest.approx(v1 * 2.0, rel=1e-6)  # v ∝ √T


def test_speed_hierarchy():
    """v_p < ⟨v⟩ < v_rms for Maxwell-Boltzmann."""
    m, T = 1e-26, 300.0
    vp = most_probable_speed(m, T)
    vm = mean_speed(m, T)
    vr = rms_speed(m, T)
    assert vp < vm < vr


def test_maxwell_distribution_zero_speed():
    """f(0) = 0: no particles at exactly zero speed."""
    m, T = 1e-26, 300.0
    assert maxwell_speed_dist(m, 0.0, T) == pytest.approx(0.0, abs=1e-50)


def test_maxwell_distribution_peak_near_vp():
    """Distribution peaks near most-probable speed v_p."""
    m, T = 1e-26, 300.0
    vp = most_probable_speed(m, T)
    dv = vp * 0.01
    f_at_vp = maxwell_speed_dist(m, vp, T)
    f_above = maxwell_speed_dist(m, vp + dv, T)
    f_below = maxwell_speed_dist(m, vp - dv, T)
    assert f_at_vp > f_above
    assert f_at_vp > f_below


# ── Equipartition ──────────────────────────────────────────────────────

def test_equipartition_monatomic_gas():
    """Monatomic ideal gas: C = (3/2)k_B per particle."""
    C = heat_capacity_equipartition(3, n=1)
    assert C == pytest.approx(1.5 * K_B, rel=1e-10)


def test_equipartition_diatomic_gas():
    """Diatomic gas (classical): C = (5/2)k_B per particle."""
    C = heat_capacity_equipartition(5, n=1)
    assert C == pytest.approx(2.5 * K_B, rel=1e-10)


def test_equipartition_scales_with_n():
    """Total capacity = n × per-particle capacity."""
    C1 = heat_capacity_equipartition(3, n=1)
    C100 = heat_capacity_equipartition(3, n=100)
    assert C100 == pytest.approx(100 * C1, rel=1e-10)


# ── σ-connection ───────────────────────────────────────────────────────

def test_sigma_partition_shift_zero():
    """At σ = 0: T_eff = T."""
    assert sigma_partition_shift(0.0, 300.0) == pytest.approx(300.0)


def test_sigma_partition_shift_increases():
    """At σ > 0: T_eff > T (hotter relative to QCD scale)."""
    T_eff = sigma_partition_shift(1.0, 300.0)
    assert T_eff > 300.0
    assert T_eff == pytest.approx(300.0 * math.e, rel=1e-10)


def test_sigma_partition_shift_at_conv():
    """At σ_conv: T_eff ≈ T/ξ."""
    from sigma_ground.field.constants import SIGMA_CONV, XI
    T = 1000.0
    T_eff = sigma_partition_shift(SIGMA_CONV, T)
    assert T_eff == pytest.approx(T / XI, rel=1e-3)
