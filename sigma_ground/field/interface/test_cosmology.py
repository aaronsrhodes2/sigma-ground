"""
Tests for sigma_ground.field.interface.cosmology — Phase XII.b Batch-3 integrations.

Reference values (after 2026-05-15 ETA empirical-input rework):
  ETA = ETA_HDE_UNION3 = 0.642² ≈ 0.4122   (DESI DR2 Union3 c², adopted)
  ETA_FORMULA = None                         (REJECTED -- formula-search numerology)
  DESI DR2 Union3: c = 0.642 ± 0.028 → c² ∈ [0.377, 0.449] (arXiv:2411.08639)
"""

import math
import pytest

from sigma_ground.field.constants import (
    ETA,
    ETA_FORMULA,
    ETA_HDE_UNION3,
    ETA_HDE_DESY5,
)
from sigma_ground.field.interface.cosmology import (
    eta_candidates,
    eta_coincidence_report,
    eta_in_desi_union3_band,
)


# ── eta_candidates ─────────────────────────────────────────────────────

def test_eta_candidates_returns_working():
    cands = eta_candidates()
    assert 'working' in cands
    value, source = cands['working']
    assert value == ETA
    # Source now references DESI / empirical input, not ρ_DE matching.
    assert 'DESI' in source or 'empirical' in source.lower()


def test_eta_candidates_no_longer_returns_formula():
    """ETA_FORMULA was rejected (2026-05-15 audit) and removed from candidates."""
    cands = eta_candidates()
    assert 'formula' not in cands


def test_eta_formula_constant_is_none():
    """The constant itself is now None so any callsite that still expects
    a numeric value fails loudly."""
    assert ETA_FORMULA is None


def test_eta_candidates_omits_pending_dr3_values():
    """DR3 placeholders are None — they must NOT appear in the candidates dict."""
    cands = eta_candidates()
    assert 'hde_union3_dr3' not in cands
    assert 'hde_desy5_dr3' not in cands


# ── eta_coincidence_report ─────────────────────────────────────────────

def test_coincidence_report_union3_agrees_by_construction():
    """ETA is now anchored at ETA_HDE_UNION3, so Union3 must agree exactly."""
    r = eta_coincidence_report(tolerance_pct=1.0)
    assert r['agrees']['hde_union3_dr2'] is True
    assert r['deviations_pct']['hde_union3_dr2'] == pytest.approx(0.0, abs=1e-12)


def test_coincidence_report_desy5_does_not_agree():
    """DESY5 central value (~0.491) differs from Union3 (~0.412) by ~19%."""
    r = eta_coincidence_report(tolerance_pct=1.0)
    assert r['agrees']['hde_desy5_dr2'] is False
    assert r['deviations_pct']['hde_desy5_dr2'] > 10.0


def test_coincidence_live_flag():
    """One non-working candidate (Union3) agrees → coincidence_live remains True
    by the new 'single external corroboration' rule."""
    r = eta_coincidence_report(tolerance_pct=1.0)
    assert r['coincidence_live'] is True


def test_coincidence_no_longer_reports_formula():
    """The Phase XI formula candidate is gone -- ensure it's not in deviations."""
    r = eta_coincidence_report(tolerance_pct=1.0)
    assert 'formula' not in r['deviations_pct']
    assert 'formula' not in r['agrees']


def test_coincidence_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        eta_coincidence_report(tolerance_pct=-1.0)


# ── eta_in_desi_union3_band ────────────────────────────────────────────

def test_eta_in_desi_union3_band_dr2():
    r = eta_in_desi_union3_band('dr2')
    assert r['in_band'] is True
    assert r['eta'] == ETA
    assert r['low'] < ETA < r['high']


def test_eta_in_desi_union3_band_dr3_pending():
    """DR3 numerics not yet lifted — function reports 'pending' rather than guessing."""
    r = eta_in_desi_union3_band('dr3')
    assert r['in_band'] is False
    assert 'pending' in r['note'].lower() or 'not yet' in r['note'].lower()


def test_eta_in_desi_band_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        eta_in_desi_union3_band('dr7')


# ── Phase XII.b.2: HDE ρ_DE formula (arXiv:2511.09467) ─────────────────

def test_hubble_radius_positive():
    from sigma_ground.field.interface.cosmology import hubble_radius
    R_H = hubble_radius()
    assert R_H > 0
    # Sanity: Hubble radius ~ 10²⁶ m
    assert 1e25 < R_H < 1e27


def test_hde_rho_de_positive():
    from sigma_ground.field.interface.cosmology import hde_rho_de
    rho = hde_rho_de()  # default c²=ETA, L=R_H
    assert rho > 0


def test_hde_roundtrip_c_squared():
    """Inverting ρ_DE(c²) at a known c² must recover the input."""
    from sigma_ground.field.interface.cosmology import (
        hde_rho_de,
        hde_c_squared_given_rho_de,
    )
    c_sq_in = 0.5
    rho = hde_rho_de(c_squared=c_sq_in)
    c_sq_out = hde_c_squared_given_rho_de(rho)
    assert c_sq_out == pytest.approx(c_sq_in, rel=1e-10)


def test_hde_rho_de_scales_with_c_squared():
    """ρ_DE ∝ c² at fixed L."""
    from sigma_ground.field.interface.cosmology import hde_rho_de
    r1 = hde_rho_de(c_squared=0.1)
    r2 = hde_rho_de(c_squared=0.2)
    assert r2 / r1 == pytest.approx(2.0, rel=1e-10)


def test_hde_rho_de_scales_inverse_L_squared():
    """ρ_DE ∝ 1/L² at fixed c²."""
    from sigma_ground.field.interface.cosmology import hde_rho_de, hubble_radius
    L1 = hubble_radius()
    L2 = 2.0 * L1
    r1 = hde_rho_de(L=L1)
    r2 = hde_rho_de(L=L2)
    assert r1 / r2 == pytest.approx(4.0, rel=1e-10)


def test_hde_rejects_negative_inputs():
    from sigma_ground.field.interface.cosmology import (
        hde_rho_de,
        hde_c_squared_given_rho_de,
    )
    with pytest.raises(ValueError):
        hde_rho_de(c_squared=-0.1)
    with pytest.raises(ValueError):
        hde_rho_de(L=0.0)
    with pytest.raises(ValueError):
        hde_c_squared_given_rho_de(rho_de=-1.0)


def test_hde_rho_de_at_eta_matches_observed_rho_de_to_factor_eta_over_omega_de():
    """ρ_DE (at c²=η, L=R_H) should reproduce observed ρ_DE up to the η/Ω_DE ratio.

    Observed ρ_DE ≈ 6×10⁻¹⁰ J/m³ (Planck 2018 Ω_DE × ρ_crit, consistent with
    H₀ = 67.4 km/s/Mpc). With c² ≡ η = 0.4153 and Ω_DE ≈ 0.6889 from Planck,
    the predicted ρ_DE should be ~η/Ω_DE ≈ 0.603 of observed.
    """
    from sigma_ground.field.interface.cosmology import hde_rho_de
    OBSERVED_RHO_DE = 6.0e-10         # J/m³, Planck 2018 nominal
    OMEGA_DE_PLANCK = 0.6889
    predicted = hde_rho_de()
    ratio = predicted / OBSERVED_RHO_DE
    expected = ETA / OMEGA_DE_PLANCK
    # Allow 20% tolerance — absolute scale depends on H0 choice (late vs early)
    assert ratio == pytest.approx(expected, rel=0.20)


# ── Phase XII.b.3: EDE+IDE values (arXiv:2505.23382 Table 2) ──────────
# Values lifted 2026-05-15 from arXiv:2505.23382 Table 2 (mixed EDE+iDEDM
# model, joint P18+DESI+DES+PP+H0 analysis). Both parameters come out as
# 95% C.L. UPPER BOUNDS, not central values with symmetric uncertainties:
#   f_EDE      < 0.113   at 95% C.L.
#   ξ_iDEDM    < 0.071   at 95% C.L.

def test_ede_ide_now_available():
    """Values lifted from arXiv:2505.23382 -- predictor data is in place."""
    from sigma_ground.field.interface.cosmology import ede_ide_available
    assert ede_ide_available() is True


def test_ede_ide_prediction_refuses_to_fabricate():
    """Predictor still raises because the (f_ede,ξ) → (ΔH₀,ΔS₈) mapping
    function hasn't been implemented -- only the upper-bound values
    themselves are lifted, not the predictive coefficients."""
    from sigma_ground.field.interface.cosmology import ede_ide_h0_s8_prediction
    with pytest.raises(NotImplementedError):
        ede_ide_h0_s8_prediction()


def test_ede_ide_upper_bounds_match_paper_table2():
    """The EDE+iDEDM upper bounds from arXiv:2505.23382 Table 2 (mixed
    model, P18+DESI+DES+PP+H0): f_EDE < 0.113, ξ_iDEDM < 0.071, both
    at 95% C.L.
    """
    from sigma_ground.field.constants import (
        F_EDE_UPPER_95CL, XI_IDE_UPPER_95CL,
        F_EDE_CENTRAL, XI_IDE_CENTRAL,  # legacy aliases
    )
    assert F_EDE_UPPER_95CL == pytest.approx(0.113, abs=1e-12)
    assert XI_IDE_UPPER_95CL == pytest.approx(0.071, abs=1e-12)
    # Legacy names alias the upper bounds for backward compatibility.
    assert F_EDE_CENTRAL == F_EDE_UPPER_95CL
    assert XI_IDE_CENTRAL == XI_IDE_UPPER_95CL


# ── Phase XII.d.2: Entropic / MOND (arXiv:2511.05632) ────────────────

def test_newtonian_regime_earth_gravity():
    """g = 9.8 m/s² is unambiguously Newtonian."""
    from sigma_ground.field.interface.cosmology import newtonian_regime
    assert newtonian_regime(9.8) == 'newtonian'


def test_newtonian_regime_galaxy_outskirts():
    """a ~ 10⁻¹¹ m/s² is MOND-regime."""
    from sigma_ground.field.interface.cosmology import newtonian_regime
    assert newtonian_regime(1e-11) == 'mond'


def test_newtonian_regime_transition_band():
    """a near a₀ falls in the transition zone."""
    from sigma_ground.field.interface.cosmology import newtonian_regime
    from sigma_ground.field.constants import A0_MOND_MS2
    assert newtonian_regime(A0_MOND_MS2) == 'transition'


def test_newtonian_regime_rejects_negative():
    from sigma_ground.field.interface.cosmology import newtonian_regime
    with pytest.raises(ValueError):
        newtonian_regime(-1.0)


def test_newtonian_regime_rejects_small_safety_factor():
    from sigma_ground.field.interface.cosmology import newtonian_regime
    with pytest.raises(ValueError):
        newtonian_regime(1.0, safety_factor=0.5)


def test_milgrom_coincidence_within_factor_three():
    """Milgrom's prediction a₀ ≈ c·H₀/(2π) must sit within 3× measured a₀."""
    from sigma_ground.field.interface.cosmology import a0_mond_from_cosmological_scale
    r = a0_mond_from_cosmological_scale()
    assert r['coincidence_within_factor_3'] is True
    # Tighter check: within 30% of measured
    assert 0.7 < r['ratio'] < 1.3


def test_a0_mond_value_matches_sparc():
    """Milgrom's empirical a₀ = 1.2×10⁻¹⁰ m/s² (SPARC 2016 confirmation)."""
    from sigma_ground.field.constants import A0_MOND_MS2
    assert A0_MOND_MS2 == pytest.approx(1.2e-10, rel=1e-10)
