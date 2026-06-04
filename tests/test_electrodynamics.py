"""
Tests for sigma_ground.field.electrodynamics

Reference values:
  Jackson, "Classical Electrodynamics" 3rd ed.
  Griffiths, "Introduction to Electrodynamics" 4th ed.
  NIST CODATA 2018
"""

import math
import pytest

from sigma_ground.field.constants import E_CHARGE, EPS_0, C, HBAR, ALPHA
from sigma_ground.field.electrodynamics import (
    coulomb_force,
    electric_field_point,
    electric_potential,
    magnetic_force,
    lorentz_force,
    radiation_power_larmor,
    em_wave_energy_density,
    em_wave_intensity,
    cyclotron_frequency,
    skin_depth,
    fine_structure_constant,
    muon_anomalous_moment_experimental,
    muon_anomalous_moment_sm,
    muon_g2_tension_sigmas,
    muon_g2_consistency_status,
)


# ── Coulomb force ──────────────────────────────────────────────────────

def test_coulomb_like_charges_repel():
    """Two protons 1 Å apart: force is positive (repulsive)."""
    F = coulomb_force(E_CHARGE, E_CHARGE, 1e-10)
    assert F > 0


def test_coulomb_opposite_charges_attract():
    """Proton and electron 1 Å apart: force is negative (attractive)."""
    F = coulomb_force(E_CHARGE, -E_CHARGE, 1e-10)
    assert F < 0


def test_coulomb_bohr_radius():
    """Hydrogen atom at Bohr radius a₀ = 5.292e-11 m.

    Force = e²/(4πε₀ a₀²) ≈ 8.24×10⁻⁸ N.
    """
    a0 = 5.29177210903e-11  # Bohr radius (m)
    F = coulomb_force(E_CHARGE, -E_CHARGE, a0)
    # F should be ≈ -8.24e-8 N (attractive)
    assert abs(F) == pytest.approx(8.238e-8, rel=1e-3)


def test_coulomb_inverse_square():
    """Force ∝ 1/r²: double the distance → quarter the force."""
    F1 = coulomb_force(E_CHARGE, E_CHARGE, 1.0)
    F2 = coulomb_force(E_CHARGE, E_CHARGE, 2.0)
    assert F2 == pytest.approx(F1 / 4.0, rel=1e-10)


def test_coulomb_zero_separation_raises():
    with pytest.raises(ValueError):
        coulomb_force(1.0, 1.0, 0.0)


def test_electric_field_point_charge():
    """E field 1 m from a +1 C charge ≈ 8.988×10⁹ V/m."""
    E = electric_field_point(1.0, 1.0)
    k_e = 1.0 / (4 * math.pi * EPS_0)
    assert E == pytest.approx(k_e, rel=1e-6)


def test_electric_potential_consistency():
    """V = k_e q/r, and E = −dV/dr = k_e q/r²: check consistency."""
    q, r = E_CHARGE, 1e-10
    V = electric_potential(q, r)
    E = electric_field_point(q, r)
    # E = -dV/dr = k_e q / r²; V = k_e q / r  → E = V / r
    assert E == pytest.approx(V / r, rel=1e-10)


# ── Lorentz force ──────────────────────────────────────────────────────

def test_magnetic_force_perpendicular():
    """v in x, B in z: force is in y direction."""
    q = E_CHARGE
    v_vec = (1e6, 0.0, 0.0)   # m/s in x
    B_vec = (0.0, 0.0, 1.0)   # T in z
    F = magnetic_force(q, v_vec, B_vec)
    # v × B = (1,0,0) × (0,0,1) = (0·1−0·0, 0·0−1·1, 1·0−0·0) = (0,-1,0)
    assert F[0] == pytest.approx(0.0, abs=1e-30)
    assert F[1] == pytest.approx(-q * 1e6 * 1.0, rel=1e-10)
    assert F[2] == pytest.approx(0.0, abs=1e-30)


def test_magnetic_force_parallel_zero():
    """v ∥ B: no magnetic force."""
    q = E_CHARGE
    v_vec = (0.0, 0.0, 1e6)
    B_vec = (0.0, 0.0, 1.0)
    F = magnetic_force(q, v_vec, B_vec)
    assert all(abs(fi) < 1e-30 for fi in F)


def test_lorentz_force_electric_only():
    """B = 0: Lorentz force = qE."""
    q = E_CHARGE
    E_vec = (1000.0, 0.0, 0.0)   # V/m in x
    v_vec = (1e6, 0.0, 0.0)
    B_vec = (0.0, 0.0, 0.0)
    F = lorentz_force(q, E_vec, v_vec, B_vec)
    assert F[0] == pytest.approx(q * 1000.0, rel=1e-10)
    assert F[1] == pytest.approx(0.0, abs=1e-30)


# ── Radiation ──────────────────────────────────────────────────────────

def test_larmor_power_positive():
    """Any non-zero acceleration radiates positive power."""
    P = radiation_power_larmor(E_CHARGE, 1e10)
    assert P > 0


def test_larmor_power_scales_q_squared():
    """Larmor power ∝ q²."""
    P1 = radiation_power_larmor(E_CHARGE, 1e10)
    P2 = radiation_power_larmor(2 * E_CHARGE, 1e10)
    assert P2 == pytest.approx(4 * P1, rel=1e-10)


def test_larmor_power_scales_a_squared():
    """Larmor power ∝ a²."""
    P1 = radiation_power_larmor(E_CHARGE, 1e10)
    P2 = radiation_power_larmor(E_CHARGE, 2e10)
    assert P2 == pytest.approx(4 * P1, rel=1e-10)


def test_larmor_zero_acceleration():
    """No acceleration → no radiation."""
    P = radiation_power_larmor(E_CHARGE, 0.0)
    assert P == pytest.approx(0.0, abs=1e-50)


# ── EM wave energetics ─────────────────────────────────────────────────

def test_em_wave_energy_density_time_averaged():
    """Time-averaged energy density = ½ε₀E₀²."""
    E0 = 1000.0  # V/m
    u = em_wave_energy_density(E0, time_average=True)
    assert u == pytest.approx(0.5 * EPS_0 * E0**2, rel=1e-10)


def test_em_wave_energy_density_instantaneous():
    """Instantaneous energy density = ε₀E²."""
    E0 = 1000.0
    u = em_wave_energy_density(E0, time_average=False)
    assert u == pytest.approx(EPS_0 * E0**2, rel=1e-10)


def test_em_wave_intensity_sunlight():
    """Solar constant ≈ 1361 W/m²; check at E₀ ≈ 1013 V/m.

    I = ½ε₀c E₀²  with E₀ = √(2I/(ε₀c))
    Verify round-trip.
    """
    I_solar = 1361.0  # W/m²
    E0 = math.sqrt(2 * I_solar / (EPS_0 * C))
    I_calc = em_wave_intensity(E0)
    assert I_calc == pytest.approx(I_solar, rel=1e-6)


# ── Cyclotron ──────────────────────────────────────────────────────────

def test_cyclotron_frequency_electron_in_1T():
    """Electron cyclotron freq in B = 1 T ≈ 1.759×10¹¹ rad/s."""
    from sigma_ground.field.constants import M_ELECTRON_KG
    omega_c = cyclotron_frequency(E_CHARGE, M_ELECTRON_KG, 1.0)
    assert omega_c == pytest.approx(1.7588e11, rel=1e-3)


def test_cyclotron_frequency_scales_with_B():
    """ω_c ∝ B."""
    from sigma_ground.field.constants import M_ELECTRON_KG
    w1 = cyclotron_frequency(E_CHARGE, M_ELECTRON_KG, 1.0)
    w2 = cyclotron_frequency(E_CHARGE, M_ELECTRON_KG, 2.0)
    assert w2 == pytest.approx(2 * w1, rel=1e-10)


# ── Fine structure constant ────────────────────────────────────────────

def test_fine_structure_constant_value():
    """α ≈ 7.2973525693e-3 (NIST CODATA 2018)."""
    alpha = fine_structure_constant()
    assert alpha == pytest.approx(7.2973525693e-3, rel=1e-6)


def test_fine_structure_constant_inverse():
    """1/α ≈ 137.036 (NIST)."""
    alpha = fine_structure_constant()
    assert 1.0 / alpha == pytest.approx(137.036, rel=1e-4)


def test_fine_structure_constant_matches_module_constant():
    """fine_structure_constant() == ALPHA constant in constants.py."""
    assert fine_structure_constant() == pytest.approx(ALPHA, rel=1e-12)


# ── Phase XII.c.1: Muon g−2 Theory Initiative 2025 ────────────────────

def test_muon_g2_experimental_value():
    """Fermilab final 2025: a_μ × 10¹¹ = 116 592 061 ± 22 (127 ppb)."""
    a, s = muon_anomalous_moment_experimental()
    assert a == pytest.approx(116_592_061.0, rel=1e-12)
    assert s == pytest.approx(22.0, rel=1e-12)


def test_muon_g2_sm_prediction_value():
    """MTI 2025 WP: a_μ,SM × 10¹¹ = 116 591 810 ± 43 (lattice-HVP)."""
    a, s = muon_anomalous_moment_sm()
    assert a == pytest.approx(116_591_810.0, rel=1e-12)
    assert s == pytest.approx(43.0, rel=1e-12)


def test_muon_g2_tension_sigmas_in_expected_band():
    """Residual CMD-3-vs-lattice tension is ~5σ but is a data-choice artefact."""
    t = muon_g2_tension_sigmas()
    # Not a physics tension per MTI 2025 — but the number itself should match.
    assert 4.5 < t < 6.0


def test_muon_g2_consistency_status_keys():
    r = muon_g2_consistency_status()
    for key in ('a_mu_exp_x1e11', 'a_mu_sm_x1e11', 'tension_sigmas',
                'tension_resolved', 'lambda_qcd_mev', 'note'):
        assert key in r


def test_muon_g2_tension_resolved_flag():
    """Per MTI 2025 WP, anomaly dissolved with lattice-HVP adopted."""
    r = muon_g2_consistency_status()
    assert r['tension_resolved'] is True


def test_muon_g2_engine_lambda_qcd_matches_pdg():
    """Sigma-ground's Λ_QCD = 217 MeV must equal the PDG value used by MTI 2025."""
    from sigma_ground.field.constants import LAMBDA_QCD_MEV
    r = muon_g2_consistency_status()
    assert r['lambda_qcd_mev'] == LAMBDA_QCD_MEV
    assert LAMBDA_QCD_MEV == 217.0
