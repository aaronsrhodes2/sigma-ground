"""
Neutrino sector — σ-grounded scaffolding for Tier-2 arXiv integrations.

Added in Phase XII.c.2. Houses basic neutrino-mass machinery so that
arXiv:2506.06449 (Dirac Type-I seesaw) and arXiv:2505.01422 (neutrino mass
from asymptotic-safety gravity) have a place to plug in.

Confidence tags:
  - PDG mass-squared splittings are [VERIFIED].
  - Cosmological Σm_ν bounds are [VERIFIED] from Planck/DESI.
  - Per-eigenstate masses require a choice of ordering + lightest mass;
    these are functions returning predictions, not stored constants.
  - The two paper-specific mechanisms (Dirac Type-I seesaw, asymptotic-safety)
    are scaffolded as stubs that refuse to fabricate numerics.

References:
  PDG Review of Particle Physics (Neutrino Mixing and Mass)
  Planck 2018 results VI (cosmological parameters)
  DESI 2024 DR1 (arXiv:2404.03002)
  arXiv:2506.06449 — Dirac Type-I seesaw cosmology (May 2025)
  arXiv:2505.01422 — Neutrino mass from asymptotic-safety gravity (May 2025)
"""

import math

from ..constants import (
    DELTA_M2_21_EV2,
    DELTA_M2_31_EV2,
    DELTA_M2_32_EV2,
    SUM_NEUTRINO_MASS_UPPER_EV,
    SUM_NEUTRINO_MASS_DESI_UPPER_EV,
    NEUTRINO_NATURE,
)


__all__ = [
    'mass_eigenstates_normal',
    'mass_eigenstates_inverted',
    'sum_mass_ev',
    'sum_mass_respects_cosmology',
    'neutrino_nature',
    'dirac_seesaw_prediction',
    'asymptotic_safety_prediction',
]


# ── Mass eigenstates ──────────────────────────────────────────────────

def mass_eigenstates_normal(m_lightest_ev):
    """Return (m₁, m₂, m₃) in eV for normal ordering (m₁ < m₂ < m₃).

    Normal ordering: m₂² = m₁² + Δm²₂₁, m₃² = m₁² + Δm²₃₁.

    Args:
        m_lightest_ev: mass of the lightest eigenstate m₁ (eV), ≥ 0.

    Returns:
        (m1, m2, m3) tuple of floats in eV.
    """
    if m_lightest_ev < 0:
        raise ValueError(f"m_lightest_ev={m_lightest_ev} < 0")
    m1 = m_lightest_ev
    m2 = math.sqrt(m1 * m1 + DELTA_M2_21_EV2)
    m3 = math.sqrt(m1 * m1 + DELTA_M2_31_EV2)
    return (m1, m2, m3)


def mass_eigenstates_inverted(m_lightest_ev):
    """Return (m₁, m₂, m₃) in eV for inverted ordering (m₃ < m₁ < m₂).

    Inverted ordering: m₁² = m₃² + |Δm²₃₂|, m₂² = m₁² + Δm²₂₁.

    Args:
        m_lightest_ev: mass of the lightest eigenstate m₃ (eV), ≥ 0.

    Returns:
        (m1, m2, m3) tuple of floats in eV.
    """
    if m_lightest_ev < 0:
        raise ValueError(f"m_lightest_ev={m_lightest_ev} < 0")
    m3 = m_lightest_ev
    m1 = math.sqrt(m3 * m3 + abs(DELTA_M2_32_EV2))
    m2 = math.sqrt(m1 * m1 + DELTA_M2_21_EV2)
    return (m1, m2, m3)


def sum_mass_ev(m_lightest_ev, ordering='normal'):
    """Sum of neutrino masses Σm_ν for a given ordering and lightest mass.

    Args:
        m_lightest_ev: lightest eigenstate mass (eV), ≥ 0.
        ordering:      'normal' (default) or 'inverted'.

    Returns:
        Σm_ν in eV.
    """
    if ordering == 'normal':
        masses = mass_eigenstates_normal(m_lightest_ev)
    elif ordering == 'inverted':
        masses = mass_eigenstates_inverted(m_lightest_ev)
    else:
        raise ValueError(f"Unknown ordering: {ordering!r}")
    return sum(masses)


def sum_mass_respects_cosmology(m_lightest_ev, ordering='normal',
                                 bound='planck'):
    """Check whether Σm_ν at the given configuration respects a cosmology bound.

    Args:
        m_lightest_ev: lightest eigenstate mass (eV).
        ordering:      'normal' | 'inverted'.
        bound:         'planck' (<0.12 eV) or 'desi' (<0.072 eV).

    Returns:
        dict with 'sum_mass_ev', 'bound_ev', 'respects_bound', 'margin_ev'.
    """
    if bound == 'planck':
        upper = SUM_NEUTRINO_MASS_UPPER_EV
    elif bound == 'desi':
        upper = SUM_NEUTRINO_MASS_DESI_UPPER_EV
    else:
        raise ValueError(f"Unknown bound: {bound!r}")
    s = sum_mass_ev(m_lightest_ev, ordering)
    return {
        'sum_mass_ev': s,
        'bound_ev': upper,
        'respects_bound': s <= upper,
        'margin_ev': upper - s,
    }


def neutrino_nature():
    """Return the currently-stored neutrino nature ('dirac'/'majorana'/None).

    Sigma-ground takes no stance until 0νββ experiments resolve it.
    """
    return NEUTRINO_NATURE


# ── arXiv:2506.06449 — Dirac Type-I seesaw (stub) ────────────────────

def dirac_seesaw_prediction(m_D_ev=None, M_R_ev=None):
    """Dirac Type-I seesaw: m_ν ≈ m_D² / M_R (schematic).

    Paper-specific coefficients (right-handed neutrino number, Yukawa
    couplings, charged-lepton flavour mixing) have NOT yet been lifted
    from arXiv:2506.06449. This stub therefore refuses to run — it is
    safer to raise than to ship an incomplete formula.

    Args:
        m_D_ev: Dirac mass m_D (eV), optional.
        M_R_ev: right-handed Majorana mass M_R (eV), optional.

    Raises:
        NotImplementedError: until the paper's Table / equations are
        lifted into constants.py and this function.
    """
    _ = m_D_ev
    _ = M_R_ev
    raise NotImplementedError(
        "Dirac Type-I seesaw coefficients from arXiv:2506.06449 not yet "
        "lifted. Pending: right-handed-neutrino count, Yukawa couplings, "
        "flavour-mixing angles."
    )


# ── arXiv:2505.01422 — Neutrino mass from asymptotic-safety ──────────

def asymptotic_safety_prediction():
    """Predict neutrino-mass hierarchy from asymptotic-safety RG flow.

    Per UP-005 (see misc/unpublished_predictions.md): if sigma-ground's
    XI = 0.1572 (Kobakhidze N=8) coincides with the RG fixed-point
    coupling at the neutrino mass scale, we have a consistency check
    between two independently-motivated frameworks. The numerics
    require solving the asymptotic-safety RG equations, which are NOT
    yet implemented here.

    Raises:
        NotImplementedError: until the RG flow solver is implemented.
    """
    raise NotImplementedError(
        "Asymptotic-safety RG flow from arXiv:2505.01422 not yet "
        "implemented. Pending: β-function coefficients, flow integration, "
        "UV fixed-point identification."
    )
