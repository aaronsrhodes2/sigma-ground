"""
Alloy property prediction from atomic-fraction compositions.

Derivation chains:

  1. Linear Mixing (baseline, FIRST_PRINCIPLES)
     X_alloy = Σ f_i × X_i
     Exact for additive properties (density, electron density).
     Approximate for non-additive properties (λ_ep, Θ_D).

  2. DOS-Weighted Lambda (FIRST_PRINCIPLES)
     λ_alloy = Σ f_i × N_i(E_F) × λ_i / Σ f_i × N_i(E_F)
     Weights electron-phonon coupling by each element's contribution
     to the Fermi-level density of states.  N_i(E_F) ~ n_e_i^{1/3}
     in the free-electron model (Ashcroft & Mermin Ch. 2).

  3. McMillan T_c Prediction (FIRST_PRINCIPLES)
     T_c = (Θ_D / 1.45) × exp(−1.04(1+λ) / (λ − μ*(1+0.62λ)))
     McMillan, Phys. Rev. 167, 331 (1968).
     Valid for λ ≲ 1.5.

  4. Nordheim Resistivity (FIRST_PRINCIPLES for binary alloys)
     Δρ = A × f(1−f), where A is the Nordheim coefficient.
     Nordheim, Ann. Phys. 401, 607 (1931).

Domain bounds:
  - McMillan formula valid for λ < 1.5 (weak-to-moderate coupling).
  - Assumes DISORDERED SOLID SOLUTION.  Ordered intermetallic compounds
    (A15 phases like Nb₃Sn) can have dramatically different λ_ep due to
    enhanced density of states from crystal structure effects.
  - Mixing rules approximate; non-linear effects not captured.

σ-field dependence (Rule 4):
  sigma_alloy_Tc() shifts Θ_D through nuclear mass change under σ.

Sources:
  McMillan (1968), Allen & Dynes (1975), Nordheim (1931),
  Hill (1952), Ashcroft & Mermin (1976), Roberts NBS (1978)
"""

import math

from ..constants import (HBAR, K_B, M_ELECTRON_KG, SIGMA_HERE,
                         PROTON_QCD_FRACTION)
from ..scale import scale_ratio

from .superconductivity import SUPERCONDUCTORS, mcmillan_Tc, _fe_v_F


# ── Composition Validation ────────────────────────────────────────

def _validate_composition(composition):
    """Validate and normalize a composition dict.

    Args:
        composition: {"niobium": 0.53, "titanium": 0.47}
            Keys must be SUPERCONDUCTORS keys.
            Values are atomic (mole) fractions, must sum to ~1.0.

    Returns:
        List of (key, fraction, entry) tuples.

    Raises:
        ValueError: if keys missing, fractions invalid, or McMillan data incomplete.
    """
    if not composition:
        raise ValueError("Empty composition")

    total = sum(composition.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Atomic fractions must sum to 1.0, got {total:.4f}")

    result = []
    missing_keys = []
    missing_mcmillan = []

    for key, frac in composition.items():
        if frac < 0:
            raise ValueError(f"Negative fraction for '{key}': {frac}")
        if frac == 0:
            continue
        if key not in SUPERCONDUCTORS:
            missing_keys.append(key)
            continue
        entry = SUPERCONDUCTORS[key]
        if entry.get('lambda_ep') is None:
            missing_mcmillan.append(key)
            continue
        if entry.get('mu_star') is None:
            missing_mcmillan.append(key)
            continue
        if entry.get('theta_D_K') is None:
            missing_mcmillan.append(key)
            continue
        result.append((key, frac, entry))

    if missing_keys:
        raise ValueError(
            f"Unknown element(s): {missing_keys}. "
            f"Must be keys in SUPERCONDUCTORS dict.")
    if missing_mcmillan:
        raise ValueError(
            f"Incomplete McMillan data for: {missing_mcmillan}. "
            f"Need lambda_ep, mu_star, and theta_D_K for T_c prediction.")

    return result


# ── Linear Mixing Model ──────────────────────────────────────────

def _mix_linear(components):
    """Linear weighted average of all properties.

    X_alloy = Σ f_i × X_i

    Exact for n_e (electron conservation).
    Approximate for Θ_D, λ, μ* (Rule 7: stated approximation).

    Args:
        components: list of (key, fraction, entry) from _validate_composition

    Returns:
        dict with mixed properties
    """
    theta_D = 0.0
    lambda_ep = 0.0
    mu_star = 0.0
    n_e = 0.0
    v_F = 0.0

    for key, f, entry in components:
        theta_D += f * entry['theta_D_K']
        lambda_ep += f * entry['lambda_ep']
        mu_star += f * entry['mu_star']
        n_e += f * entry['n_e_m3']
        v_F += f * entry['v_F_m_s']

    return {
        'theta_D_K': theta_D,
        'lambda_ep': lambda_ep,
        'mu_star': mu_star,
        'n_e_m3': n_e,
        'v_F_m_s': v_F,
        'model': 'linear',
    }


# ── DOS-Weighted Lambda Model ────────────────────────────────────

def _mix_dos_weighted(components):
    """DOS-weighted mixing: weights λ by Fermi-level density of states.

    λ_alloy = Σ f_i × N_i(E_F) × λ_i / Σ f_i × N_i(E_F)

    where N_i(E_F) ~ n_e_i^{1/3} in the free-electron model.
    This gives heavier weight to elements with more electrons at E_F,
    which is physically more realistic than a simple average.

    Θ_D and μ* use linear mixing (no DOS weighting needed for these).

    Ashcroft & Mermin (1976), Ch. 2.

    Args:
        components: list of (key, fraction, entry)

    Returns:
        dict with mixed properties
    """
    # Linear properties (same as _mix_linear)
    theta_D = sum(f * e['theta_D_K'] for _, f, e in components)
    mu_star = sum(f * e['mu_star'] for _, f, e in components)
    n_e = sum(f * e['n_e_m3'] for _, f, e in components)
    v_F = sum(f * e['v_F_m_s'] for _, f, e in components)

    # DOS-weighted λ
    # N(E_F) ~ n_e^{1/3} in free-electron model
    numerator = 0.0
    denominator = 0.0
    for _, f, e in components:
        n_ef = e['n_e_m3'] ** (1.0 / 3.0)
        numerator += f * n_ef * e['lambda_ep']
        denominator += f * n_ef

    lambda_ep = numerator / denominator if denominator > 0 else 0.0

    return {
        'theta_D_K': theta_D,
        'lambda_ep': lambda_ep,
        'mu_star': mu_star,
        'n_e_m3': n_e,
        'v_F_m_s': v_F,
        'model': 'dos_weighted',
    }


# ── Property Prediction ──────────────────────────────────────────

def alloy_properties(composition, model='linear'):
    """Predict alloy properties from atomic-fraction composition.

    Uses mixing rules to derive Θ_D, λ, μ*, n_e, v_F from constituent
    element data in the SUPERCONDUCTORS database, then predicts T_c
    via McMillan's formula.

    Args:
        composition: dict mapping SUPERCONDUCTORS keys to atomic fractions.
                     Example: {"niobium": 0.53, "titanium": 0.47}
        model: mixing model — 'linear' or 'dos_weighted'

    Returns:
        dict with:
            composition:    input composition
            theta_D_K:      mixed Debye temperature (K)
            lambda_ep:      mixed electron-phonon coupling
            mu_star:        mixed Coulomb pseudopotential
            n_e_m3:         mixed electron density (m⁻³)
            v_F_m_s:        mixed Fermi velocity (m/s)
            T_c_predicted_K: McMillan-predicted T_c (K)
            model:          mixing model used
            warnings:       list of caveats
    """
    components = _validate_composition(composition)

    if model == 'dos_weighted':
        mixed = _mix_dos_weighted(components)
    else:
        mixed = _mix_linear(components)

    # McMillan T_c
    T_c = mcmillan_Tc(mixed['theta_D_K'], mixed['lambda_ep'],
                      mixed['mu_star'])

    # Warnings
    warnings = []
    if mixed['lambda_ep'] > 1.5:
        warnings.append(
            f"λ={mixed['lambda_ep']:.2f} > 1.5: McMillan formula may "
            f"overestimate. Allen-Dynes correction recommended.")

    # Check for intermetallic stoichiometries (A3B ratios)
    fracs = sorted(composition.values(), reverse=True)
    if len(fracs) == 2:
        ratio = fracs[0] / fracs[1] if fracs[1] > 0 else float('inf')
        if abs(ratio - 3.0) < 0.3:
            warnings.append(
                "Composition near A₃B stoichiometry. Ordered intermetallic "
                "compounds (A15 phases) can have dramatically different λ_ep "
                "from solid-solution prediction. This model assumes disorder.")

    # Check for magnetic elements
    for key, f, entry in components:
        if entry.get('suppression') == 'ferromagnet' and f > 0.1:
            warnings.append(
                f"'{key}' is ferromagnetic — magnetic ordering may suppress "
                f"superconductivity. McMillan prediction assumes non-magnetic.")

    return {
        'composition': composition,
        'theta_D_K': round(mixed['theta_D_K'], 1),
        'lambda_ep': round(mixed['lambda_ep'], 4),
        'mu_star': round(mixed['mu_star'], 4),
        'n_e_m3': mixed['n_e_m3'],
        'v_F_m_s': round(mixed['v_F_m_s'], 0),
        'T_c_predicted_K': round(T_c, 3),
        'model': mixed['model'],
        'warnings': warnings,
    }


def predict_alloy_Tc(composition, model='linear'):
    """Predict T_c for an alloy composition.

    Convenience wrapper: returns just the T_c value.

    Args:
        composition: {"niobium": 0.53, "titanium": 0.47}
        model: 'linear' or 'dos_weighted'

    Returns:
        Predicted T_c in Kelvin
    """
    props = alloy_properties(composition, model=model)
    return props['T_c_predicted_K']


def alloy_Tc_all_models(composition):
    """Run all mixing models; report spread as uncertainty estimate.

    The spread between models provides a rough uncertainty bound.
    Wider spread = less confident prediction.

    Args:
        composition: {"niobium": 0.53, "titanium": 0.47}

    Returns:
        dict with per-model predictions and summary statistics
    """
    models = ['linear', 'dos_weighted']
    results = {}

    for m in models:
        props = alloy_properties(composition, model=m)
        results[m] = props

    Tc_values = [results[m]['T_c_predicted_K'] for m in models]
    Tc_min = min(Tc_values)
    Tc_max = max(Tc_values)
    Tc_mean = sum(Tc_values) / len(Tc_values)

    # Collect all unique warnings
    all_warnings = []
    seen = set()
    for m in models:
        for w in results[m]['warnings']:
            if w not in seen:
                all_warnings.append(w)
                seen.add(w)

    return {
        'composition': composition,
        'models': results,
        'summary': {
            'T_c_min_K': round(Tc_min, 3),
            'T_c_max_K': round(Tc_max, 3),
            'T_c_mean_K': round(Tc_mean, 3),
            'spread_K': round(Tc_max - Tc_min, 3),
            'confidence': 'high' if (Tc_max - Tc_min) < 1.0 else
                          'moderate' if (Tc_max - Tc_min) < 3.0 else 'low',
        },
        'warnings': all_warnings,
    }


# ── σ-Field Dependence (Rule 4) ──────────────────────────────────

def sigma_alloy_Tc(composition, sigma, model='linear'):
    """Alloy T_c under σ-field.

    Θ_D shifts through nuclear mass under σ:
      Θ_D(σ) = Θ_D(0) / √(mass_ratio(σ))
    where mass_ratio = (1 − f_qcd) + f_qcd × scale_ratio(σ).

    At Earth (σ ~ 7×10⁻¹⁰): shift is unmeasurably small.
    Becomes significant approaching neutron star surfaces and beyond.

    Args:
        composition: {"niobium": 0.53, "titanium": 0.47}
        sigma: σ-field value
        model: 'linear' or 'dos_weighted'

    Returns:
        Predicted T_c(σ) in Kelvin
    """
    props = alloy_properties(composition, model=model)

    if sigma == SIGMA_HERE:
        return props['T_c_predicted_K']

    f_qcd = PROTON_QCD_FRACTION
    mass_ratio = (1.0 - f_qcd) + f_qcd * scale_ratio(sigma)
    theta_D_sigma = props['theta_D_K'] / math.sqrt(mass_ratio)

    return round(
        mcmillan_Tc(theta_D_sigma, props['lambda_ep'], props['mu_star']),
        3)


# ── Normal-State Resistivity ─────────────────────────────────────

def alloy_Nordheim_resistivity(composition):
    """Estimate normal-state residual resistivity via Nordheim's rule.

    For binary alloys A₁₋ₓBₓ:
      Δρ = A_Nordheim × x(1−x)

    where A_Nordheim is the Nordheim coefficient (element-pair specific).
    We use a universal estimate: A_Nordheim ≈ 10 μΩ·cm for transition
    metals (Rossiter, "Electrical Resistance of Metals and Alloys", 1987).

    Rule 7: This is a rough estimate. Real Nordheim coefficients vary
    from ~1 to ~50 μΩ·cm depending on the element pair.  The estimate
    is useful for ordering alloys by expected residual resistivity but
    not for quantitative prediction.

    Args:
        composition: {"niobium": 0.53, "titanium": 0.47}

    Returns:
        dict with estimated residual resistivity in μΩ·cm
    """
    _validate_composition(composition)

    fracs = list(composition.values())
    keys = list(composition.keys())

    if len(fracs) == 1:
        return {
            'rho_residual_uOhm_cm': 0.0,
            'model': 'Nordheim',
            'note': 'Pure element — zero disorder scattering',
        }

    # Universal Nordheim coefficient (approximate)
    A_NORDHEIM = 10.0  # μΩ·cm — typical for transition metal pairs

    if len(fracs) == 2:
        x = fracs[1]  # minority fraction
        delta_rho = A_NORDHEIM * x * (1.0 - x)
    else:
        # Multi-component: sum over all pairs
        delta_rho = 0.0
        for i in range(len(fracs)):
            for j in range(i + 1, len(fracs)):
                delta_rho += A_NORDHEIM * fracs[i] * fracs[j]

    return {
        'rho_residual_uOhm_cm': round(delta_rho, 3),
        'model': 'Nordheim',
        'note': ('Approximate: uses universal A=10 μΩ·cm. '
                 'Real values range 1–50 μΩ·cm depending on element pair.'),
        'composition': composition,
    }


# ── Alloy Composition Database ───────────────────────────────────
# Public compositions from literature. All atomic fractions.
# Sources: ASM International, ASTM, vendor datasheets.

ALLOYS = {
    # ── Nb-based (superconducting wire alloys) ────────────────────
    'NbTi_wire':        {'niobium': 0.53, 'titanium': 0.47},
    'NbTi_60_40':       {'niobium': 0.60, 'titanium': 0.40},
    'NbTi_80_20':       {'niobium': 0.80, 'titanium': 0.20},
    'NbZr_75_25':       {'niobium': 0.75, 'zirconium': 0.25},
    'NbMo_80_20':       {'niobium': 0.80, 'molybdenum': 0.20},
    'NbV_50_50':        {'niobium': 0.50, 'vanadium': 0.50},
    'NbTa_50_50':       {'niobium': 0.50, 'tantalum': 0.50},
    'NbRe_75_25':       {'niobium': 0.75, 'rhenium': 0.25},

    # ── Pb-based ──────────────────────────────────────────────────
    'PbIn_60_40':       {'lead': 0.60, 'indium': 0.40},
    'PbSn_eutectic':    {'lead': 0.63, 'tin': 0.37},
    'PbTl_80_20':       {'lead': 0.80, 'thallium': 0.20},

    # ── Refractory ────────────────────────────────────────────────
    'MoRe_50_50':       {'molybdenum': 0.50, 'rhenium': 0.50},
    'MoRe_60_40':       {'molybdenum': 0.60, 'rhenium': 0.40},
    'TaW_90_10':        {'tantalum': 0.90, 'tungsten': 0.10},
    'TaHf_90_10':       {'tantalum': 0.90, 'hafnium': 0.10},

    # ── Light metals ──────────────────────────────────────────────
    'AlGa_50_50':       {'aluminum': 0.50, 'gallium': 0.50},
    'AlIn_90_10':       {'aluminum': 0.90, 'indium': 0.10},
    'AlZn_95_5':        {'aluminum': 0.95, 'zinc': 0.05},

    # ── Noble metal dilution (expect T_c suppression) ─────────────
    'NbCu_50_50':       {'niobium': 0.50, 'copper': 0.50},
    'PbAg_80_20':       {'lead': 0.80, 'silver': 0.20},
    'CuZn_brass':       {'copper': 0.70, 'zinc': 0.30},
    'CuSn_bronze':      {'copper': 0.88, 'tin': 0.12},

    # ── Ternary ───────────────────────────────────────────────────
    'NbTiZr_ternary':   {'niobium': 0.34, 'titanium': 0.33,
                         'zirconium': 0.33},
    'NbTiV_ternary':    {'niobium': 0.34, 'titanium': 0.33,
                         'vanadium': 0.33},

    # ── Intermetallic stoichiometries (for A15 comparison) ────────
    'Nb3Sn_stoich':     {'niobium': 0.75, 'tin': 0.25},
    'V3Si_stoich':      {'vanadium': 0.75, 'silicon': 0.25},
}


# Note: 'silicon' in the SUPERCONDUCTORS dict is a pressure-required SC
# (T_c=8.2K @ 12 GPa). Its lambda_ep is not stored, so V3Si_stoich
# will fail validation. This is correct — we can only predict alloys
# where ALL constituents have McMillan data.

# Remove entries whose elements lack McMillan data:
_VALID_ALLOYS = {}
for _name, _comp in ALLOYS.items():
    try:
        _validate_composition(_comp)
        _VALID_ALLOYS[_name] = _comp
    except ValueError:
        pass  # silently skip — these are documented above

# Expose only valid alloys for batch prediction
ALLOYS_PREDICTABLE = _VALID_ALLOYS


def list_alloys():
    """List all predefined alloy compositions.

    Returns:
        dict mapping alloy names to compositions (only those with
        complete McMillan data for all constituents)
    """
    return dict(ALLOYS_PREDICTABLE)


def predict_all():
    """Run predictions for all predefined alloys using all models.

    Returns:
        dict mapping alloy names to alloy_Tc_all_models() results,
        sorted by predicted T_c (highest first)
    """
    results = {}
    for name, comp in ALLOYS_PREDICTABLE.items():
        results[name] = alloy_Tc_all_models(comp)

    # Sort by mean predicted T_c, descending
    return dict(sorted(
        results.items(),
        key=lambda kv: kv[1]['summary']['T_c_mean_K'],
        reverse=True))


# ── Composition Sweep ─────────────────────────────────────────────

def composition_sweep(element_a, element_b, steps=21):
    """Sweep T_c vs composition for a binary alloy system.

    Varies x from 0 to 1 in the system A₁₋ₓBₓ.

    Args:
        element_a: SUPERCONDUCTORS key for element A
        element_b: SUPERCONDUCTORS key for element B
        steps: number of composition points (default 21 → 5% increments)

    Returns:
        list of dicts: x_B, T_c_linear, T_c_dos, T_c_mean
    """
    results = []
    for i in range(steps):
        x = i / (steps - 1)
        comp = {}
        if x < 1.0:
            comp[element_a] = round(1.0 - x, 4)
        if x > 0.0:
            comp[element_b] = round(x, 4)

        try:
            Tc_lin = predict_alloy_Tc(comp, model='linear')
            Tc_dos = predict_alloy_Tc(comp, model='dos_weighted')
        except ValueError:
            continue

        results.append({
            'x_B': round(x, 4),
            'f_A': round(1.0 - x, 4),
            'f_B': round(x, 4),
            'T_c_linear_K': Tc_lin,
            'T_c_dos_K': Tc_dos,
            'T_c_mean_K': round((Tc_lin + Tc_dos) / 2, 3),
        })

    return results
