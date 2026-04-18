"""
Cosmology interface — σ-grounded dark-energy, HDE, and tension-resolution models.

Adds in Phase XII.b (Batch 3 arXiv integrations):
- η-coincidence status: compares the three independent candidate derivations
  of η (working ρ_DE fit, Phase XI formula, DESI HDE c²) against each other.
- HDE Hubble-tension helpers (arXiv:2511.09467).
- EDE + interacting dark-sector helpers (arXiv:2505.23382).

Confidence tags follow the Phase XII audit convention:
  [SPECULATIVE], [SPECULATIVE-PENDING], [THEORETICAL], [VERIFIED], [DERIVED].

References:
  Phase XI η candidate analysis: misc/bh_phase_xi_eta_candidates_results.md
  Unpublished-predictions ledger: misc/unpublished_predictions.md (UP-001)
"""

import math

from ..constants import (
    C,
    G,
    H0,
    HBAR,
    M_PLANCK_KG,
    ETA,
    ETA_FORMULA,
    ETA_HDE_UNION3,
    ETA_HDE_DESY5,
    ETA_HDE_UNION3_DR3,
    ETA_HDE_DESY5_DR3,
    A0_MOND_MS2,
)


__all__ = [
    'eta_candidates',
    'eta_coincidence_report',
    'eta_in_desi_union3_band',
    'hubble_radius',
    'hde_rho_de',
    'hde_c_squared_given_rho_de',
    'ede_ide_available',
    'ede_ide_h0_s8_prediction',
    'newtonian_regime',
    'a0_mond_from_cosmological_scale',
]


def eta_candidates():
    """Return the available candidate values for η with their sources.

    Any candidate whose source value is not yet lifted into constants.py
    (SPECULATIVE-PENDING placeholders) is omitted from the returned dict.

    Returns:
        dict mapping short-name → (value, source_label).
    """
    cands = {
        'working':       (ETA,              'ρ_DE match (hardcoded working value)'),
        'formula':       (ETA_FORMULA,      'exp(−φ/σ_conv), Phase XI candidate'),
        'hde_union3_dr2': (ETA_HDE_UNION3,  'DESI DR2 Union3 HDE c² (arXiv:2411.08639)'),
        'hde_desy5_dr2':  (ETA_HDE_DESY5,   'DESI DR2 DESY5 HDE c² (arXiv:2411.08639)'),
    }
    if ETA_HDE_UNION3_DR3 is not None:
        cands['hde_union3_dr3'] = (ETA_HDE_UNION3_DR3, 'DESI DR3 Union3 HDE c² (arXiv:2512.07281)')
    if ETA_HDE_DESY5_DR3 is not None:
        cands['hde_desy5_dr3'] = (ETA_HDE_DESY5_DR3, 'DESI DR3 DESY5 HDE c² (arXiv:2512.07281)')
    return cands


def eta_coincidence_report(tolerance_pct=1.0):
    """Return a dict summarizing the pairwise agreement between η candidates.

    The central working value is ETA. For each other candidate, the relative
    deviation (|cand − ETA| / ETA) is reported in percent. A candidate is
    flagged as 'agrees' when the deviation is within `tolerance_pct` percent.

    UP-001 (triple coincidence): we predict that 'working', 'formula', and
    'hde_union3_dr2' all agree within ~1% of one another. DR3 updates are
    expected to tighten this further; a falsifier would be a DR3 value
    outside [0.410, 0.420].

    Args:
        tolerance_pct: agreement threshold in percent (default 1.0 for
                       UP-001 triple-coincidence). Non-negative.

    Returns:
        dict with keys:
          'working':        central ETA value
          'deviations_pct': dict {name → deviation %}
          'agrees':         dict {name → bool}
          'tolerance_pct':  echoed threshold
          'coincidence_live': bool — True when ≥2 candidates agree within
                              `tolerance_pct`, indicating the pattern is
                              not yet dissolved.
    """
    if tolerance_pct < 0:
        raise ValueError(f"tolerance_pct={tolerance_pct} must be ≥ 0")
    cands = eta_candidates()
    working = cands['working'][0]
    deviations = {}
    agrees = {}
    for name, (val, _src) in cands.items():
        if name == 'working':
            continue
        dev = abs(val - working) / working * 100.0
        deviations[name] = dev
        agrees[name] = dev <= tolerance_pct
    coincidence_live = sum(agrees.values()) >= 2
    return {
        'working': working,
        'deviations_pct': deviations,
        'agrees': agrees,
        'tolerance_pct': tolerance_pct,
        'coincidence_live': coincidence_live,
    }


def eta_in_desi_union3_band(dataset='dr2'):
    """Report whether ETA falls within the DESI Union3 HDE c² 1-σ band.

    Because the DESI paper publishes c = 0.642 ± 0.028 (DR2), the 1-σ band
    on c² is asymmetric. The 1-σ bounds on η under the identification c² ≡ η:
      low  = (0.642 − 0.028)² = 0.614² ≈ 0.3770
      high = (0.642 + 0.028)² = 0.670² ≈ 0.4489

    Args:
        dataset: 'dr2' (default) or 'dr3'. DR3 returns False with a
                 'pending' note until numerics are lifted.

    Returns:
        dict with 'in_band' (bool), 'low', 'high', 'eta', 'note'.
    """
    if dataset == 'dr2':
        c_mean = 0.642
        c_sigma = 0.028
        low = (c_mean - c_sigma) ** 2
        high = (c_mean + c_sigma) ** 2
        return {
            'in_band': low <= ETA <= high,
            'low': low,
            'high': high,
            'eta': ETA,
            'note': 'DESI DR2 Union3 (arXiv:2411.08639)',
        }
    elif dataset == 'dr3':
        if ETA_HDE_UNION3_DR3 is None:
            return {
                'in_band': False,
                'low': None,
                'high': None,
                'eta': ETA,
                'note': 'DR3 numerics not yet lifted from arXiv:2512.07281',
            }
        # Placeholder path — not used until DR3 numbers are filled in.
        c_sq_mean = ETA_HDE_UNION3_DR3
        return {
            'in_band': abs(ETA - c_sq_mean) <= 0.01,
            'low': c_sq_mean - 0.01,
            'high': c_sq_mean + 0.01,
            'eta': ETA,
            'note': 'DESI DR3 reconstruction (arXiv:2512.07281)',
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


# ── Holographic Dark Energy (HDE) — arXiv:2511.09467 ──────────────────
# ρ_DE = 3 c² M_Pl² / L²  (Li 2004; revisited in arXiv:2511.09467 under
# DESI + Hubble-tension constraints). In sigma-ground we identify c² ≡ η
# and take L = Hubble radius = c / H₀ as the IR cutoff.
#
# Phase XII.b.2 deliverable: implement the HDE relation as a callable so
# the identification c² ≡ η is testable against future DR3 data with one
# function call.

def hubble_radius():
    """Current-epoch Hubble radius R_H = c / H₀ in meters.

    [DERIVED] from the [EMPIRICAL-INPUT] H0 and [VERIFIED] C.

    Returns:
        R_H (m).
    """
    return C / H0


def hde_rho_de(c_squared=None, L=None):
    """Holographic dark-energy density in SI units.

    Natural-units form (Li 2004): ρ_DE = 3 c_HDE² M_Pl,red² / L²
    where M_Pl,red = 1/√(8πG) in natural units.

    SI equivalent (dimensionally verified):
        ρ_DE [J/m³] = 3 c_HDE² × c⁴ / (8π G L²)

    This is the canonical Friedmann form with ρ_critical × Ω_DE
    identified as c_HDE². At L = R_H = c/H₀ it reduces to
    ρ_DE = 3 c_HDE² c² H₀² / (8π G), the familiar form.

    Reference: Li 2004 (Phys. Lett. B 603, 1); arXiv:2511.09467 for
    Hubble-tension-era HDE constraints. In sigma-ground, c_HDE² ≡ η
    by the UP-001 identification.

    Args:
        c_squared: HDE c² parameter. Defaults to sigma-ground's ETA.
        L:          IR cutoff length (m). Defaults to Hubble radius.

    Returns:
        ρ_DE in J/m³.
    """
    if c_squared is None:
        c_squared = ETA
    if L is None:
        L = hubble_radius()
    if L <= 0:
        raise ValueError(f"L={L} ≤ 0: cutoff length must be positive")
    if c_squared < 0:
        raise ValueError(f"c_squared={c_squared} < 0: must be non-negative")
    return 3.0 * c_squared * C ** 4 / (8.0 * math.pi * G * L * L)


def hde_c_squared_given_rho_de(rho_de, L=None):
    """Invert ρ_DE = 3 c² c⁴ / (8π G L²) for c².

    Useful for testing: given an observed ρ_DE and a cutoff length, what
    HDE c² does that imply? If c² ≡ η, this inverted value should equal ETA.

    Args:
        rho_de: dark-energy density (J/m³), must be ≥ 0.
        L:      IR cutoff length (m). Defaults to Hubble radius.

    Returns:
        c² (dimensionless).
    """
    if rho_de < 0:
        raise ValueError(f"rho_de={rho_de} < 0: must be non-negative")
    if L is None:
        L = hubble_radius()
    if L <= 0:
        raise ValueError(f"L={L} ≤ 0")
    prefactor = 3.0 * C ** 4 / (8.0 * math.pi * G * L * L)
    return rho_de / prefactor


# ── EDE + Interacting Dark Sector (arXiv:2505.23382) ─────────────────
# Placeholder integration. The paper fits two parameters (f_EDE, ξ_IDE) to
# simultaneously reduce H₀ and S₈ tensions to ≤2σ. The coefficients that
# translate (f_EDE, ξ_IDE) → (ΔH₀, ΔS₈) require reading the paper's Table;
# until those are lifted, the prediction function returns a sentinel None.

def ede_ide_available():
    """Whether the arXiv:2505.23382 central values have been lifted.

    Until F_EDE_CENTRAL and XI_IDE_CENTRAL are filled in constants.py,
    ede_ide_h0_s8_prediction() refuses to run — safer than fabricating
    numbers.

    Returns:
        bool.
    """
    from ..constants import F_EDE_CENTRAL, XI_IDE_CENTRAL
    return F_EDE_CENTRAL is not None and XI_IDE_CENTRAL is not None


def ede_ide_h0_s8_prediction(f_ede=None, xi_ide=None):
    """Predict (H₀_shift, S₈_shift) from EDE fraction and IDE coupling.

    NOT YET IMPLEMENTED numerically — the coefficients that turn
    (f_ede, xi_ide) into concrete H₀ and S₈ shifts live in the paper's
    Table 2, which has not yet been read in detail. Raises
    NotImplementedError until then.

    Args:
        f_ede:  EDE fraction (dimensionless). Defaults to F_EDE_CENTRAL.
        xi_ide: IDE coupling (dimensionless). Defaults to XI_IDE_CENTRAL.

    Raises:
        NotImplementedError: always, until the paper's coefficients are
        lifted into constants.py (F_EDE_CENTRAL, XI_IDE_CENTRAL, and the
        coefficients that map them to ΔH₀, ΔS₈).
    """
    raise NotImplementedError(
        "EDE+IDE numerics from arXiv:2505.23382 not yet lifted. "
        "Lift F_EDE_CENTRAL, XI_IDE_CENTRAL and the ΔH₀/ΔS₈ coefficients "
        "from the paper's Table 2 before calling this function."
    )


# ── Entropic / MOND (arXiv:2511.05632) ────────────────────────────────
# Relativistic MOND from modified entropic gravity. The key observable
# prediction is the characteristic acceleration a₀ ≈ 1.2×10⁻¹⁰ m/s² at
# which Newton transitions to modified dynamics.

def newtonian_regime(a_ms2, safety_factor=10.0):
    """Classify an acceleration as Newtonian, transition, or MOND.

    Args:
        a_ms2:        acceleration magnitude in m/s².
        safety_factor: the boundary half-width multiplier. Accelerations
            above `safety_factor × a₀` are unambiguously Newtonian; below
            `a₀ / safety_factor` are MOND-regime; between is "transition".

    Returns:
        one of: 'newtonian', 'transition', 'mond'.
    """
    if a_ms2 < 0:
        raise ValueError(f"a_ms2={a_ms2} < 0: acceleration magnitude")
    if safety_factor <= 1.0:
        raise ValueError(f"safety_factor={safety_factor} must be > 1")
    if a_ms2 >= safety_factor * A0_MOND_MS2:
        return 'newtonian'
    if a_ms2 <= A0_MOND_MS2 / safety_factor:
        return 'mond'
    return 'transition'


def a0_mond_from_cosmological_scale():
    """Milgrom's numerical coincidence: a₀ ≈ c × H₀ / (2π).

    Milgrom 1983 noted that a₀ is empirically close to c × H₀. The
    entropic-gravity derivation of arXiv:2511.05632 argues this is not
    coincidence but a prediction: a₀ emerges as a cosmological-scale
    acceleration set by the current Hubble rate.

    Returns:
        dict with 'a0_predicted' (from c·H₀/2π), 'a0_measured' (Milgrom),
        'ratio', and 'coincidence_within_factor' (True if within 3x).
    """
    predicted = C * H0 / (2.0 * math.pi)
    return {
        'a0_predicted': predicted,
        'a0_measured': A0_MOND_MS2,
        'ratio': predicted / A0_MOND_MS2,
        'coincidence_within_factor_3': 1.0 / 3.0 < predicted / A0_MOND_MS2 < 3.0,
    }
