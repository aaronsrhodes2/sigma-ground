"""
Hidden-sector physics — axions, dark photons, dark-axion portal.

Added in Phase XII.d.3 for arXiv:2410.12634 (dark axion portal at Z factories)
and paired with Kobakhidze-Liang N=8 composite-axion work (already in constants
as XI_KOBAKHIDZE). Also scaffolds arXiv:2408.14245 (GALPs — composite heavy ALPs).

The module is deliberately thin: it reports on parameter-space bounds and
provides a regime classifier, but does not compute paper-specific sensitivity
reaches until those Tables are lifted.

References:
  arXiv:2410.12634 — Dark axion portal at Z factories (Jodłowski 2024)
  arXiv:2408.14245 — GALPs: Composite heavy axion-like Dark Matter
  CAST 2017 (Nature Physics 13, 584) — helioscope g_aγγ bound
"""

from ..constants import (
    AXION_MASS_ULDM_MIN_EV,
    AXION_MASS_ULDM_MAX_EV,
    ALP_PHOTON_COUPLING_MAX_INVGEV,
    DARK_AXION_PORTAL_Z_SENSITIVITY,
    XI_KOBAKHIDZE,
)


__all__ = [
    'axion_mass_regime',
    'alp_coupling_allowed',
    'dark_axion_portal_available',
    'kobakhidze_gauge_group_ratio',
]


def axion_mass_regime(m_a_ev):
    """Classify an axion mass into its canonical regime.

    Args:
        m_a_ev: axion mass in eV, > 0.

    Returns:
        one of: 'sub-ultralight', 'ultralight-DM', 'QCD-axion-window',
                'heavy', or raises for m ≤ 0.
    """
    if m_a_ev <= 0:
        raise ValueError(f"m_a_ev={m_a_ev} must be positive")
    if m_a_ev < AXION_MASS_ULDM_MIN_EV:
        return 'sub-ultralight'  # fuzzy DM, below ULDM floor
    if m_a_ev <= AXION_MASS_ULDM_MAX_EV:
        # Spans ULDM through QCD-axion window — report the finer slot
        if m_a_ev < 1e-10:
            return 'ultralight-DM'
        return 'QCD-axion-window'
    return 'heavy'


def alp_coupling_allowed(g_agg_invgev):
    """Is an ALP-photon coupling below the CAST 2017 helioscope bound?

    Args:
        g_agg_invgev: ALP-photon coupling g_aγγ in GeV⁻¹, ≥ 0.

    Returns:
        bool.
    """
    if g_agg_invgev < 0:
        raise ValueError(f"g_agg_invgev={g_agg_invgev} < 0")
    return g_agg_invgev < ALP_PHOTON_COUPLING_MAX_INVGEV


def dark_axion_portal_available():
    """Whether arXiv:2410.12634 Z-factory sensitivity values are lifted.

    Returns:
        bool — False until Table values are filled in constants.py.
    """
    return DARK_AXION_PORTAL_Z_SENSITIVITY is not None


def kobakhidze_gauge_group_ratio(N=8):
    """Dark-matter to baryon ratio from Kobakhidze N-gauge-group composite axion.

    Reference: arXiv:2410.12634 (paired with arXiv:2410.22412).
    For N=8: Ω_DM/Ω_B = 5.36 → ξ = 1 / (1 + 5.36) = 0.1572.

    Sigma-ground's working XI = 0.1582 and Kobakhidze XI = 0.1572
    agree at 0.6% — the N=8 identification is the hidden-sector hook
    that, if confirmed, promotes XI from [EMPIRICAL-INPUT] to [DERIVED].

    Args:
        N: gauge-group integer. Only N=8 is currently supported.

    Returns:
        Ω_DM/Ω_B ratio (float).
    """
    if N == 8:
        return (1.0 - XI_KOBAKHIDZE) / XI_KOBAKHIDZE
    raise NotImplementedError(
        f"Kobakhidze QCD-β-function ratio for N={N} not implemented. "
        f"arXiv:2410.22412 only derives the N=8 case explicitly."
    )
