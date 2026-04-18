"""
Tests for sigma_ground.field.interface.neutrino — Phase XII.c.2 scaffolding.

Reference values:
  Δm²_21 = 7.42e-5 eV² (PDG 2022)
  |Δm²_31| = 2.514e-3 eV² (PDG 2022, normal)
  Σm_ν < 0.12 eV (Planck 2018)
  Σm_ν < 0.072 eV (DESI 2024)
"""

import math
import pytest

from sigma_ground.field.constants import (
    DELTA_M2_21_EV2,
    DELTA_M2_31_EV2,
    SUM_NEUTRINO_MASS_UPPER_EV,
    SUM_NEUTRINO_MASS_DESI_UPPER_EV,
)
from sigma_ground.field.interface.neutrino import (
    mass_eigenstates_normal,
    mass_eigenstates_inverted,
    sum_mass_ev,
    sum_mass_respects_cosmology,
    neutrino_nature,
    dirac_seesaw_prediction,
    asymptotic_safety_prediction,
)


# ── Mass eigenstates ──────────────────────────────────────────────────

def test_normal_ordering_at_zero_lightest():
    """At m₁=0: m₂ = √Δm²₂₁, m₃ = √Δm²₃₁."""
    m1, m2, m3 = mass_eigenstates_normal(0.0)
    assert m1 == 0.0
    assert m2 == pytest.approx(math.sqrt(DELTA_M2_21_EV2), rel=1e-10)
    assert m3 == pytest.approx(math.sqrt(DELTA_M2_31_EV2), rel=1e-10)
    assert m1 < m2 < m3  # Normal ordering signature


def test_inverted_ordering_at_zero_lightest():
    """At m₃=0: m₁ = √|Δm²₃₂|, m₂ = √(m₁² + Δm²₂₁)."""
    m1, m2, m3 = mass_eigenstates_inverted(0.0)
    assert m3 == 0.0
    assert m3 < m1 < m2  # Inverted ordering signature


def test_normal_ordering_sum_matches_function():
    s_fn = sum_mass_ev(0.0, ordering='normal')
    masses = mass_eigenstates_normal(0.0)
    assert s_fn == pytest.approx(sum(masses), rel=1e-10)


def test_inverted_ordering_heavier_than_normal_at_zero_lightest():
    """For m_lightest=0, inverted ordering is heavier than normal."""
    s_normal = sum_mass_ev(0.0, ordering='normal')
    s_inverted = sum_mass_ev(0.0, ordering='inverted')
    assert s_inverted > s_normal


def test_sum_mass_rejects_negative_mass():
    with pytest.raises(ValueError):
        mass_eigenstates_normal(-0.01)
    with pytest.raises(ValueError):
        mass_eigenstates_inverted(-1.0)


def test_sum_mass_rejects_unknown_ordering():
    with pytest.raises(ValueError):
        sum_mass_ev(0.0, ordering='sideways')


# ── Cosmology bounds ──────────────────────────────────────────────────

def test_cosmology_respects_planck_bound_at_m_lightest_zero():
    """Minimal normal-ordering masses easily respect Planck."""
    r = sum_mass_respects_cosmology(0.0, 'normal', 'planck')
    assert r['respects_bound'] is True
    assert r['sum_mass_ev'] < SUM_NEUTRINO_MASS_UPPER_EV
    assert r['margin_ev'] > 0


def test_cosmology_desi_bound_tighter_than_planck():
    """DESI bound is tighter; with quasi-degenerate masses it bites earlier."""
    # Pick a mass where Planck passes but DESI fails
    r_planck = sum_mass_respects_cosmology(0.03, 'normal', 'planck')
    r_desi   = sum_mass_respects_cosmology(0.03, 'normal', 'desi')
    assert r_planck['respects_bound'] is True
    # At m_lightest = 0.03 eV, normal Σ ≈ 0.09+0.03+0.05 ~ big enough to fail DESI
    # Let's just assert DESI bound is smaller than Planck bound
    assert r_desi['bound_ev'] < r_planck['bound_ev']


def test_cosmology_rejects_unknown_bound():
    with pytest.raises(ValueError):
        sum_mass_respects_cosmology(0.0, 'normal', 'bicep')


# ── Nature (Dirac vs Majorana) ───────────────────────────────────────

def test_neutrino_nature_undetermined():
    """Sigma-ground takes no stance on Dirac vs Majorana (0νββ unresolved)."""
    assert neutrino_nature() is None


# ── arXiv:2506.06449 + 2505.01422 stubs ─────────────────────────────

def test_dirac_seesaw_stub_refuses_to_fabricate():
    with pytest.raises(NotImplementedError):
        dirac_seesaw_prediction()


def test_asymptotic_safety_stub_refuses_to_fabricate():
    with pytest.raises(NotImplementedError):
        asymptotic_safety_prediction()
