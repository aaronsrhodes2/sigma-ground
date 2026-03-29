#!/usr/bin/env python3
"""
Multi-variable regression to predict λ_ep from derivable quantities.

Goal: replace McMillan's measured λ_ep with a derived λ_ep predicted from
quantities we can compute from Z alone (the cascade). If successful, this
enables T_c prediction for ANY material without measured coupling data.

Physics basis (Eliashberg theory):
    λ = N(E_F) × <I²> / (M × <ω²>)

We approximate:
    N(E_F) ~ n_e^(2/3)           (free-electron DOS)
    <ω²>   ~ θ_D²                (Debye model)
    M      = atomic mass          (known exactly)
    <I²>   ~ f(E_coh, d_count)   (the unknown — we fit this)

Output: prints correlation matrix, regression results, and T_c predictions.

Usage:
    cd /path/to/sigma-ground
    python scripts/lambda_regression.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sigma_ground.field.interface.superconductivity import (
    SUPERCONDUCTORS, mcmillan_Tc,
)
from sigma_ground.field.interface.element import (
    free_electron_count, d_electron_count, d_row,
    cohesive_energy_eV, predict_density_kg_m3,
    predict_crystal_structure, atomic_mass_kg, slater_zeff,
    slater_radius_m, element_properties,
)

# Load periodic table data for ionization energies, electronegativity, etc.
import json
_ELEMENTS_JSON = Path(__file__).resolve().parent.parent / 'sigma_ground' / 'inventory' / 'data' / 'elements.json'
with open(_ELEMENTS_JSON, encoding='utf-8') as _f:
    _PERIODIC_TABLE = {e['atomic_number']: e for e in json.load(_f)}

# Resistivity data (room temperature, Ohm*m) for elements where available
_RESISTIVITY_300K = {
    13: 2.65e-8,   # Al
    22: 4.20e-7,   # Ti
    23: 1.97e-7,   # V  (literature: 19.7 uOhm.cm)
    26: 9.7e-8,    # Fe
    27: 6.24e-8,   # Co (literature: 6.24 uOhm.cm)
    28: 6.99e-8,   # Ni
    29: 1.68e-8,   # Cu
    30: 5.90e-8,   # Zn (literature: 5.90 uOhm.cm)
    31: 1.40e-7,   # Ga (literature: 14.0 uOhm.cm)
    40: 4.21e-7,   # Zr (literature: 42.1 uOhm.cm)
    41: 1.52e-7,   # Nb (literature: 15.2 uOhm.cm)
    42: 5.34e-8,   # Mo (literature: 5.34 uOhm.cm)
    43: 2.26e-7,   # Tc (literature: 22.6 uOhm.cm)
    44: 7.10e-8,   # Ru (literature: 7.10 uOhm.cm)
    46: 1.05e-7,   # Pd (literature: 10.5 uOhm.cm)
    47: 1.59e-8,   # Ag
    48: 6.83e-8,   # Cd (literature: 6.83 uOhm.cm)
    49: 8.37e-8,   # In (literature: 8.37 uOhm.cm)
    50: 1.15e-7,   # Sn (literature: 11.5 uOhm.cm)
    57: 6.15e-7,   # La (literature: 61.5 uOhm.cm)
    72: 3.31e-7,   # Hf (literature: 33.1 uOhm.cm)
    73: 1.35e-7,   # Ta (literature: 13.5 uOhm.cm)
    74: 5.28e-8,   # W
    75: 1.93e-7,   # Re (literature: 19.3 uOhm.cm)
    76: 8.12e-8,   # Os (literature: 8.12 uOhm.cm)
    77: 4.71e-8,   # Ir (literature: 4.71 uOhm.cm)
    78: 1.06e-7,   # Pt
    79: 2.44e-8,   # Au
    80: 9.61e-7,   # Hg (literature: 96.1 uOhm.cm — liquid metal!)
    81: 1.80e-7,   # Tl (literature: 18.0 uOhm.cm)
    82: 2.07e-7,   # Pb
    90: 1.47e-7,   # Th (literature: 14.7 uOhm.cm)
}


# ── Map SC element keys to atomic number Z ────────────────────────
# Only ambient-pressure elements with measured λ_ep
_SC_KEY_TO_Z = {
    'aluminum': 13,
    'titanium': 22,
    'vanadium': 23,
    'zinc': 30,
    'gallium': 31,
    'zirconium': 40,
    'niobium': 41,
    'molybdenum': 42,
    'technetium': 43,
    'ruthenium': 44,
    'cadmium': 48,
    'indium': 49,
    'tin': 50,
    'lanthanum': 57,
    'hafnium': 72,
    'tantalum': 73,
    'tungsten': 74,
    'rhenium': 75,
    'osmium': 76,
    'iridium': 77,
    'mercury': 80,
    'thallium': 81,
    'lead': 82,
    'thorium': 90,
    # Non-SC metals with measured λ_ep (T_c = 0)
    'copper': 29,
    'silver': 47,
    'gold': 79,
    'platinum': 78,
    'palladium': 46,
    'iron_ambient': 26,
    'cobalt': 27,
    'nickel': 28,
}


def _coordination(struct):
    """Coordination number from crystal structure."""
    return {'fcc': 12, 'bcc': 8, 'hcp': 12, 'diamond': 4,
            'diamond_cubic': 4}.get(struct, 8)


def build_feature_matrix():
    """Build feature matrix for all elements with measured λ_ep.

    Returns list of dicts, one per element, with all derivable features
    and the measured λ_ep / T_c as targets.
    """
    rows = []

    for key, Z in _SC_KEY_TO_Z.items():
        sc = SUPERCONDUCTORS[key]
        lam = sc.get('lambda_ep')
        if lam is None:
            continue

        mu_star = sc.get('mu_star', 0.12)
        theta_D = sc.get('theta_D_K')
        T_c = sc['T_c_K']
        n_e = sc['n_e_m3']
        v_F = sc['v_F_m_s']

        if theta_D is None:
            continue

        # Derivable features from Z
        free = free_electron_count(Z)
        d_count = d_electron_count(Z)
        d_r = d_row(Z)
        A_kg = atomic_mass_kg(Z)
        A_amu = A_kg / 1.66054e-27
        E_coh = cohesive_energy_eV(Z)
        rho_pred = predict_density_kg_m3(Z)
        struct = predict_crystal_structure(Z)
        z_coord = _coordination(struct)
        Z_eff = slater_zeff(Z)
        r_slater = slater_radius_m(Z)

        # Periodic table data
        pt = _PERIODIC_TABLE.get(Z, {})
        IE1 = pt.get('ionization_energy_1', 7.0)  # eV
        EN = pt.get('electronegativity', 1.5)
        EA = pt.get('electron_affinity_ev', 0.5)
        r_atom = pt.get('atomic_radius', 150)  # pm

        # Resistivity at 300K (if available)
        rho_300 = _RESISTIVITY_300K.get(Z)

        # Derived combinations (physics-motivated)
        # N(E_F) proxy: n_e^(2/3) from free-electron model
        N_EF_proxy = n_e ** (2.0 / 3.0)

        # Eliashberg-motivated: N(E_F) / (M × θ_D²)
        eliashberg_proxy = N_EF_proxy / (A_amu * theta_D ** 2)

        # free / θ_D (previous best single variable)
        free_over_thetaD = free / theta_D

        # E_coh / θ_D² — bond strength relative to phonon energy
        ecoh_over_thetaD2 = E_coh / theta_D ** 2

        # d-band filling fraction (0-1 scale, 0 for sp metals)
        d_filling = d_count / 10.0 if d_count > 0 else 0.0

        # Hopping parameter proxy: t_d = E_coh / (2z)
        t_d = E_coh / (2.0 * z_coord) if z_coord > 0 else 0.0

        # 1/v_F — inversely related to bandwidth (narrow band → large m* → large λ)
        inv_vF = 1.0 / v_F if v_F > 0 else 0.0

        # Is this a d-block element? (binary feature)
        is_d_block = 1 if d_count > 0 else 0

        # d-electron count adjusted for half-filling peak
        # λ_ep peaks near half-filled d-band (d⁵)
        d_half_filling = abs(d_count - 5) if d_count > 0 else 0

        # atomic mass in amu
        mass_amu = A_amu

        # ── NEW: <I²> proxy candidates ──────────────────────────

        # Wigner-Seitz radius (from density + mass)
        N_A = 6.02214076e23
        m_atom_kg = A_amu * 1.66054e-27
        omega_atom = m_atom_kg / (sc['n_e_m3'] / free if free > 0 else rho_pred)
        # Actually compute from known density
        rho_actual = SUPERCONDUCTORS[key].get('_density', rho_pred)
        # Use density from _sc call (it's rho argument)
        # We stored it: n_e = Z_val * N_A * rho / M_g * 1000
        # So rho = n_e * M_g / (Z_val * N_A * 1000)
        # But let's just use n_e / free for number density
        if free > 0:
            n_atom = n_e / free  # atoms per m³
        else:
            n_atom = rho_pred / m_atom_kg
        r_ws = (3.0 / (4.0 * math.pi * n_atom)) ** (1.0 / 3.0)

        # Core radius / WS radius — fraction of cell that is "core"
        # Larger core fraction = stronger scattering potential
        core_fraction = r_slater / r_ws if r_ws > 0 else 0.0

        # Scattering potential proxy: Z_eff / r_ws (Coulomb at WS boundary)
        scattering_strength = Z_eff / (r_ws * 1e10)  # Z_eff / r_ws_Angstrom

        # Pseudopotential proxy: IE1 × free / theta_D
        # IE1 measures potential depth, free/theta_D is our best single var
        IE1_free_thetaD = IE1 * free / theta_D

        # Resistivity-based lambda proxy (Bloch-Gruneisen relation)
        # At high T: rho ~ lambda * T * (omega_p^2 / n_e)
        # So rho * n_e should scale with lambda
        if rho_300 is not None:
            rho_n_e = rho_300 * n_e  # should correlate with lambda
            rho_thetaD2 = rho_300 * theta_D ** 2  # <I²> proxy
        else:
            rho_n_e = None
            rho_thetaD2 = None

        # Electronegativity / theta_D
        EN_over_thetaD = EN / theta_D if EN else 0.0

        # Z_eff * free / theta_D — combines scattering strength with our best predictor
        Zeff_free_thetaD = Z_eff * free / theta_D

        # Derived IE1 from Slater: IE1_derived = 13.6 * Z_eff^2 / n*^2
        # where n* is effective principal quantum number of outermost shell
        _n_star_map = {1: 1, 2: 2, 3: 3, 4: 3.7, 5: 4.0, 6: 4.2, 7: 4.4}
        period = pt.get('period', 4)
        n_star = _n_star_map.get(period, 4.0)
        IE1_derived = 13.6 * Z_eff ** 2 / n_star ** 2
        IE1d_free_thetaD = IE1_derived * free / theta_D

        # Z_eff^2 * free / (n*^2 * theta_D) — fully derived proxy for IE1*free/tD
        Zeff2_n2_free_thetaD = Z_eff ** 2 * free / (n_star ** 2 * theta_D)

        row = {
            'key': key,
            'Z': Z,
            'lambda_ep': lam,
            'mu_star': mu_star,
            'theta_D_K': theta_D,
            'T_c_K': T_c,
            'n_e_m3': n_e,
            'v_F_m_s': v_F,
            'is_sc': sc['is_superconductor'],
            # Raw features
            'free_electrons': free,
            'd_electron_count': d_count,
            'd_row': d_r,
            'A_amu': mass_amu,
            'E_coh_eV': E_coh,
            'rho_pred': rho_pred,
            'z_coord': z_coord,
            'Z_eff': Z_eff,
            # Derived combinations
            'N_EF_proxy': N_EF_proxy,
            'eliashberg_proxy': eliashberg_proxy,
            'free_over_thetaD': free_over_thetaD,
            'ecoh_over_thetaD2': ecoh_over_thetaD2,
            'd_filling': d_filling,
            't_d_eV': t_d,
            'inv_vF': inv_vF,
            'is_d_block': is_d_block,
            'd_half_filling': d_half_filling,
            # NEW: <I²> proxy candidates
            'IE1_eV': IE1,
            'EN': EN,
            'EA_eV': EA,
            'r_atom_pm': r_atom,
            'r_ws_A': r_ws * 1e10,
            'core_fraction': core_fraction,
            'scattering_strength': scattering_strength,
            'IE1_free_thetaD': IE1_free_thetaD,
            'rho_300': rho_300,
            'rho_n_e': rho_n_e,
            'rho_thetaD2': rho_thetaD2,
            'EN_over_thetaD': EN_over_thetaD,
            'Zeff_free_thetaD': Zeff_free_thetaD,
            'IE1_derived': IE1_derived,
            'IE1d_free_thetaD': IE1d_free_thetaD,
            'Zeff2_n2_free_thetaD': Zeff2_n2_free_thetaD,
            'period': period,
            'n_star': n_star,
        }
        rows.append(row)

    return rows


def pearson_r(xs, ys):
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    return cov / (sx * sy)


def linear_fit(xs, ys):
    """Simple linear regression: y = a*x + b. Returns (a, b, r²)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    ss_xx = sum((x - mx) ** 2 for x in xs)
    ss_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if ss_xx == 0:
        return 0, my, 0
    a = ss_xy / ss_xx
    b = my - a * mx
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return a, b, r2


def multi_linear_fit(X, y):
    """Multi-variable linear regression using normal equations.

    X: list of lists (n_samples × n_features)
    y: list of floats (n_samples)

    Returns: coefficients (list), intercept (float), r² (float)

    Uses Gaussian elimination — no numpy needed.
    """
    n = len(y)
    p = len(X[0])

    # Build augmented matrix [X'X | X'y] with intercept
    # Add column of 1s for intercept
    X_aug = [[1.0] + row for row in X]
    p_aug = p + 1

    # X'X
    XtX = [[0.0] * p_aug for _ in range(p_aug)]
    Xty = [0.0] * p_aug

    for i in range(n):
        for j in range(p_aug):
            for k in range(p_aug):
                XtX[j][k] += X_aug[i][j] * X_aug[i][k]
            Xty[j] += X_aug[i][j] * y[i]

    # Gaussian elimination with partial pivoting
    aug = [XtX[i][:] + [Xty[i]] for i in range(p_aug)]

    for col in range(p_aug):
        # Find pivot
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, p_aug):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        if abs(aug[col][col]) < 1e-15:
            continue

        pivot = aug[col][col]
        for j in range(p_aug + 1):
            aug[col][j] /= pivot

        for row in range(p_aug):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(p_aug + 1):
                aug[row][j] -= factor * aug[col][j]

    # Extract solution
    beta = [aug[i][p_aug] for i in range(p_aug)]
    intercept = beta[0]
    coeffs = beta[1:]

    # Compute R²
    y_mean = sum(y) / n
    y_pred = [intercept + sum(c * x for c, x in zip(coeffs, X[i])) for i in range(n)]
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return coeffs, intercept, r2


def main():
    rows = build_feature_matrix()
    print(f"Elements with measured λ_ep: {len(rows)}")
    print()

    # ── 1. Single-variable correlations with λ_ep ────────────────
    features = [
        'free_electrons', 'd_electron_count', 'A_amu', 'theta_D_K',
        'E_coh_eV', 'n_e_m3', 'v_F_m_s', 'Z_eff', 'z_coord',
        'N_EF_proxy', 'eliashberg_proxy', 'free_over_thetaD',
        'ecoh_over_thetaD2', 'd_filling', 't_d_eV', 'inv_vF',
        'd_half_filling',
        # NEW: <I²> proxy candidates
        'IE1_eV', 'EN', 'EA_eV', 'r_atom_pm', 'r_ws_A',
        'core_fraction', 'scattering_strength',
        'IE1_free_thetaD', 'EN_over_thetaD', 'Zeff_free_thetaD',
        'rho_300', 'rho_n_e', 'rho_thetaD2',
        'IE1_derived', 'IE1d_free_thetaD', 'Zeff2_n2_free_thetaD',
    ]

    lambdas = [r['lambda_ep'] for r in rows]

    print("=" * 65)
    print("SINGLE-VARIABLE CORRELATIONS WITH λ_ep")
    print("=" * 65)
    print(f"{'Feature':<25} {'r(λ_ep)':>8} {'r²':>8}")
    print("-" * 45)

    corrs = []
    for feat in features:
        # Skip features with None values
        valid = [(r[feat], r['lambda_ep']) for r in rows if r[feat] is not None]
        if len(valid) < 5:
            continue
        vals, lam_valid = zip(*valid)
        r = pearson_r(list(vals), list(lam_valid))
        corrs.append((feat, r, r ** 2, len(valid)))

    corrs.sort(key=lambda x: -abs(x[1]))
    for feat, r, r2, n in corrs:
        print(f"  {feat:<23} {r:+8.3f} {r2:8.3f}  (n={n})")

    # ── 2. Best single-variable regressions ──────────────────────
    print()
    print("=" * 65)
    print("BEST SINGLE-VARIABLE REGRESSIONS")
    print("=" * 65)

    for feat, r_val, r2, n in corrs[:5]:
        valid = [(r_[feat], r_['lambda_ep']) for r_ in rows if r_[feat] is not None]
        vals, lam_v = zip(*valid)
        vals, lam_v = list(vals), list(lam_v)
        a, b, r2_fit = linear_fit(vals, lam_v)
        print(f"\n  λ_pred = {a:.6f} × {feat} + {b:.4f}")
        print(f"  R² = {r2_fit:.4f}")

        # Show predictions vs actual for this fit
        errors = []
        for row in rows:
            lam_pred = a * row[feat] + b
            err = lam_pred - row['lambda_ep']
            errors.append((row['key'], row['lambda_ep'], lam_pred, err))

        mae = sum(abs(e[3]) for e in errors) / len(errors)
        rmse = math.sqrt(sum(e[3] ** 2 for e in errors) / len(errors))
        print(f"  MAE = {mae:.4f}, RMSE = {rmse:.4f}")

    # ── 3. Multi-variable regressions ────────────────────────────
    print()
    print("=" * 65)
    print("MULTI-VARIABLE REGRESSIONS")
    print("=" * 65)

    # Regression set A: physics-motivated (Eliashberg-inspired)
    # λ ~ f(free/θ_D, d_filling, E_coh/θ_D²)
    combo_sets = {
        'A: free/tD + d_fill': ['free_over_thetaD', 'd_filling'],
        'B: free/tD + E_coh/tD2': ['free_over_thetaD', 'ecoh_over_thetaD2'],
        'C: inv_vF + d_fill + free/tD + E_coh': ['inv_vF', 'd_filling', 'free_over_thetaD', 'E_coh_eV'],
        # NEW: <I2> proxy combos
        'D: IE1*free/tD': ['IE1_free_thetaD'],
        'E: IE1*free/tD + d_fill': ['IE1_free_thetaD', 'd_filling'],
        'F: Zeff*free/tD': ['Zeff_free_thetaD'],
        'G: Zeff*free/tD + d_fill': ['Zeff_free_thetaD', 'd_filling'],
        'H: core_frac + free/tD': ['core_fraction', 'free_over_thetaD'],
        'I: scatter + free/tD': ['scattering_strength', 'free_over_thetaD'],
        'J: scatter + d_fill': ['scattering_strength', 'd_filling'],
        'K: EN/tD + d_fill': ['EN_over_thetaD', 'd_filling'],
        'L: EN/tD + free/tD': ['EN_over_thetaD', 'free_over_thetaD'],
        'M: IE1*free/tD + core_frac': ['IE1_free_thetaD', 'core_fraction'],
        'N: IE1 + free/tD + d_fill': ['IE1_eV', 'free_over_thetaD', 'd_filling'],
        'O: EN + free/tD + d_fill': ['EN', 'free_over_thetaD', 'd_filling'],
        'P: scatter + free/tD + d_fill': ['scattering_strength', 'free_over_thetaD', 'd_filling'],
        'Q: IE1*free/tD + d_fill + A_amu': ['IE1_free_thetaD', 'd_filling', 'A_amu'],
        'R: core_frac + free/tD + d_fill': ['core_fraction', 'free_over_thetaD', 'd_filling'],
        'S: EN/tD + core_frac + d_fill': ['EN_over_thetaD', 'core_fraction', 'd_filling'],
        'T: scatter + EN/tD + free/tD': ['scattering_strength', 'EN_over_thetaD', 'free_over_thetaD'],
    }

    best_r2 = 0
    best_name = ''
    best_coeffs = None
    best_intercept = None
    best_features = None

    for name, feat_names in combo_sets.items():
        X = [[row[f] for f in feat_names] for row in rows]
        y = lambdas
        coeffs, intercept, r2 = multi_linear_fit(X, y)

        # Adjusted R²
        n = len(y)
        p = len(feat_names)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

        print(f"\n  {name}")
        print(f"    R² = {r2:.4f}  (adj R² = {r2_adj:.4f})")
        eq_parts = [f"{c:+.6f}×{f}" for c, f in zip(coeffs, feat_names)]
        print(f"    λ = {intercept:.4f} {' '.join(eq_parts)}")

        if r2_adj > best_r2:
            best_r2 = r2_adj
            best_name = name
            best_coeffs = coeffs
            best_intercept = intercept
            best_features = feat_names

    # ── 4. Best model: detailed predictions ──────────────────────
    print()
    print("=" * 65)
    print(f"BEST MODEL: {best_name}")
    print(f"Adjusted R² = {best_r2:.4f}")
    print("=" * 65)

    print(f"\n{'Element':<16} {'λ_meas':>7} {'λ_pred':>7} {'Δλ':>7} "
          f"{'Tc_meas':>8} {'Tc_pred':>8} {'ΔTc':>8}")
    print("-" * 73)

    tc_errors = []
    lam_errors = []

    for row in rows:
        X_i = [row[f] for f in best_features]
        lam_pred = best_intercept + sum(c * x for c, x in zip(best_coeffs, X_i))
        lam_pred = max(0.0, lam_pred)  # λ can't be negative

        lam_meas = row['lambda_ep']
        lam_err = lam_pred - lam_meas

        # Predict T_c using derived λ
        mu_star = row['mu_star']
        theta_D = row['theta_D_K']

        if lam_pred > mu_star * (1 + 0.62 * lam_pred):
            Tc_pred = mcmillan_Tc(theta_D, lam_pred, mu_star)
        else:
            Tc_pred = 0.0

        Tc_meas = row['T_c_K']
        Tc_err = Tc_pred - Tc_meas

        tc_errors.append({'key': row['key'], 'Tc_meas': Tc_meas, 'Tc_pred': Tc_pred,
                          'Tc_err': Tc_err, 'lam_meas': lam_meas, 'lam_pred': lam_pred})
        lam_errors.append(abs(lam_err))

        print(f"  {row['key']:<14} {lam_meas:7.3f} {lam_pred:7.3f} {lam_err:+7.3f} "
              f"{Tc_meas:8.3f} {Tc_pred:8.3f} {Tc_err:+8.3f}")

    print()
    lam_mae = sum(lam_errors) / len(lam_errors)
    lam_rmse = math.sqrt(sum(e ** 2 for e in lam_errors) / len(lam_errors))
    print(f"λ_ep:  MAE = {lam_mae:.4f}, RMSE = {lam_rmse:.4f}")

    # T_c errors (SC elements only)
    sc_tc = [e for e in tc_errors if e['Tc_meas'] > 0]
    if sc_tc:
        tc_mae = sum(abs(e['Tc_err']) for e in sc_tc) / len(sc_tc)
        tc_rmse = math.sqrt(sum(e['Tc_err'] ** 2 for e in sc_tc) / len(sc_tc))
        print(f"T_c (SC only): MAE = {tc_mae:.2f} K, RMSE = {tc_rmse:.2f} K")

    # Check non-SC predictions
    non_sc = [e for e in tc_errors if e['Tc_meas'] == 0]
    print(f"\nNon-SC metals ({len(non_sc)}):")
    for e in non_sc:
        status = "✓ correct" if e['Tc_pred'] < 0.5 else "✗ FALSE POSITIVE"
        print(f"  {e['key']:<16} Tc_pred = {e['Tc_pred']:.3f} K  {status}")

    # Check ranking
    sc_ranked = sorted(sc_tc, key=lambda e: -e['Tc_meas'])
    pred_ranked = sorted(sc_tc, key=lambda e: -e['Tc_pred'])
    print(f"\nRanking check (top 5 by measured T_c):")
    for i in range(min(5, len(sc_ranked))):
        m = sc_ranked[i]
        p = pred_ranked[i]
        print(f"  #{i+1} measured: {m['key']:<14} ({m['Tc_meas']:.2f} K)  "
              f"predicted: {p['key']:<14} ({p['Tc_pred']:.2f} K)")

    # ── 5. Leave-one-out cross-validation ────────────────────────
    print()
    print("=" * 65)
    print("LEAVE-ONE-OUT CROSS-VALIDATION")
    print("=" * 65)

    loo_errors = []
    for i in range(len(rows)):
        # Train on all except i
        train = rows[:i] + rows[i+1:]
        test = rows[i]

        X_train = [[r[f] for f in best_features] for r in train]
        y_train = [r['lambda_ep'] for r in train]

        coeffs_loo, intercept_loo, _ = multi_linear_fit(X_train, y_train)

        X_test = [test[f] for f in best_features]
        lam_pred_loo = intercept_loo + sum(c * x for c, x in zip(coeffs_loo, X_test))
        lam_pred_loo = max(0.0, lam_pred_loo)

        err = lam_pred_loo - test['lambda_ep']
        loo_errors.append((test['key'], test['lambda_ep'], lam_pred_loo, err))

    loo_mae = sum(abs(e[3]) for e in loo_errors) / len(loo_errors)
    loo_rmse = math.sqrt(sum(e[3] ** 2 for e in loo_errors) / len(loo_errors))
    print(f"LOO MAE  = {loo_mae:.4f}")
    print(f"LOO RMSE = {loo_rmse:.4f}")

    # Worst outliers
    loo_errors.sort(key=lambda e: -abs(e[3]))
    print(f"\nWorst outliers:")
    for key, lam_m, lam_p, err in loo_errors[:5]:
        print(f"  {key:<16} λ_meas={lam_m:.3f}  λ_pred={lam_p:.3f}  err={err:+.3f}")

    # ── 6. Physics-motivated non-linear models ───────────────────
    print()
    print("=" * 65)
    print("PHYSICS-MOTIVATED NON-LINEAR MODELS")
    print("=" * 65)

    # Model P1: λ = a × (free/θ_D) × (1 + c × d_filling) + b
    # This separates sp contribution from d enhancement
    print("\n  P1: λ = a × (free/θ_D) × (1 + c × d_filling) + b")
    # Grid search over c
    best_p1_r2 = 0
    best_p1 = None
    for c_int in range(-20, 80):
        c = c_int * 0.1
        # Construct effective feature: (free/θ_D) × (1 + c × d_filling)
        xs = [r['free_over_thetaD'] * (1 + c * r['d_filling']) for r in rows]
        a, b, r2 = linear_fit(xs, lambdas)
        if r2 > best_p1_r2:
            best_p1_r2 = r2
            best_p1 = (a, b, c, r2)

    if best_p1:
        a, b, c, r2 = best_p1
        print(f"    Best: a={a:.4f}, b={b:.4f}, c={c:.2f}")
        print(f"    R² = {r2:.4f}")
        print(f"    λ = {a:.4f} × (free/θ_D) × (1 + {c:.2f} × d_filling) + {b:.4f}")

    # Model P2: λ = a × (1/v_F) × (1 + c × d_filling) + b
    print("\n  P2: λ = a × (1/v_F) × (1 + c × d_filling) + b")
    best_p2_r2 = 0
    best_p2 = None
    for c_int in range(-20, 80):
        c = c_int * 0.1
        xs = [r['inv_vF'] * (1 + c * r['d_filling']) for r in rows]
        a, b, r2 = linear_fit(xs, lambdas)
        if r2 > best_p2_r2:
            best_p2_r2 = r2
            best_p2 = (a, b, c, r2)

    if best_p2:
        a, b, c, r2 = best_p2
        print(f"    Best: a={a:.4f}, b={b:.4f}, c={c:.2f}")
        print(f"    R² = {r2:.4f}")

    # Model P3: λ = a × E_coh / (A × θ_D²) × (1 + c × d_count) + b
    # Direct Eliashberg form with d-electron enhancement
    print("\n  P3: λ = a × E_coh/(A×θ_D²) × (1 + c × d_count) + b")
    best_p3_r2 = 0
    best_p3 = None
    for c_int in range(0, 50):
        c = c_int * 0.05
        xs = [r['ecoh_over_thetaD2'] / r['A_amu'] * (1 + c * r['d_electron_count'])
              for r in rows]
        a, b, r2 = linear_fit(xs, lambdas)
        if r2 > best_p3_r2:
            best_p3_r2 = r2
            best_p3 = (a, b, c, r2)

    if best_p3:
        a, b, c, r2 = best_p3
        print(f"    Best: a={a:.6f}, b={b:.4f}, c={c:.2f}")
        print(f"    R² = {r2:.4f}")

    # Model P4: resistivity-derived lambda (Bloch-Gruneisen relation)
    # rho(300K) is proportional to lambda_tr * T at T > theta_D
    # rho_300 * n_e is proportional to lambda * <omega^2> / omega_p^2
    # This uses measured rho_300 for VALIDATION only — we can derive rho from BG model
    print("\n  P4: Resistivity-lambda correlation (VALIDATION)")
    rho_rows = [r for r in rows if r['rho_300'] is not None]
    print(f"    Elements with rho_300: {len(rho_rows)}")

    if len(rho_rows) > 5:
        rho_lam = [r['lambda_ep'] for r in rho_rows]

        # rho_300 alone
        rho_vals = [r['rho_300'] for r in rho_rows]
        r_rho = pearson_r(rho_vals, rho_lam)
        print(f"    r(rho_300, lambda) = {r_rho:.3f}")

        # rho * n_e  (should remove electron density dependence)
        rho_ne_vals = [r['rho_n_e'] for r in rho_rows]
        r_rho_ne = pearson_r(rho_ne_vals, rho_lam)
        print(f"    r(rho*n_e, lambda) = {r_rho_ne:.3f}")

        # rho * theta_D^2 (proxy for <I^2>)
        rho_td2_vals = [r['rho_thetaD2'] for r in rho_rows]
        r_rho_td2 = pearson_r(rho_td2_vals, rho_lam)
        a_rt, b_rt, r2_rt = linear_fit(rho_td2_vals, rho_lam)
        print(f"    r(rho*thetaD2, lambda) = {r_rho_td2:.3f}, R2 = {r2_rt:.4f}")

        # rho / (free * A) — normalize by valence and mass
        rho_norm = [r['rho_300'] / (r['free_electrons'] * r['A_amu'])
                    for r in rho_rows]
        r_rho_norm = pearson_r(rho_norm, rho_lam)
        print(f"    r(rho/(free*A), lambda) = {r_rho_norm:.3f}")

        # Show best resistivity predictor per-element
        if abs(r_rho) > 0.5 or abs(r_rho_ne) > 0.5:
            print(f"\n    {'Element':<14} {'rho_300':>10} {'lambda':>7}")
            for r in sorted(rho_rows, key=lambda x: x['rho_300']):
                print(f"      {r['key']:<12} {r['rho_300']:10.2e} {r['lambda_ep']:7.3f}")

    # ── 7. Log-space regression ────────────────────────────────────
    # lambda varies over an order of magnitude (0.12 to 1.60)
    # Log transform may improve linearity
    print()
    print("=" * 65)
    print("LOG-SPACE ANALYSIS")
    print("=" * 65)

    # ln(lambda) vs various features
    log_lam = [math.log(r['lambda_ep']) for r in rows]

    log_corrs = []
    for feat in features:
        valid = [(r[feat], math.log(r['lambda_ep'])) for r in rows
                 if r[feat] is not None]
        if len(valid) < 5:
            continue
        vals, ll = zip(*valid)
        r = pearson_r(list(vals), list(ll))
        log_corrs.append((feat, r, r ** 2))

    log_corrs.sort(key=lambda x: -abs(x[1]))
    print(f"{'Feature':<25} {'r(ln lam)':>9} {'r2':>8}")
    print("-" * 45)
    for feat, r, r2 in log_corrs[:10]:
        print(f"  {feat:<23} {r:+9.3f} {r2:8.3f}")

    # Try ln(lambda) = a*ln(free/theta_D) + b
    print("\n  Log-log: ln(lam) = a*ln(free/theta_D) + b")
    log_fot = [math.log(r['free_over_thetaD']) if r['free_over_thetaD'] > 0
               else -10 for r in rows]
    a_ll, b_ll, r2_ll = linear_fit(log_fot, log_lam)
    print(f"    a={a_ll:.4f}, b={b_ll:.4f}, R2={r2_ll:.4f}")
    print(f"    => lam = exp({b_ll:.4f}) * (free/theta_D)^{a_ll:.4f}")

    # Multi-var in log space
    print("\n  Multi-var in log space:")
    log_combos = {
        'ln: free/tD + E_coh + 1/vF': ['free_over_thetaD', 'E_coh_eV', 'inv_vF'],
        'ln: free/tD + A_amu': ['free_over_thetaD', 'A_amu'],
        'ln: free/tD + E_coh + A_amu': ['free_over_thetaD', 'E_coh_eV', 'A_amu'],
        'ln: free/tD + E_coh/tD2 + A_amu': ['free_over_thetaD', 'ecoh_over_thetaD2', 'A_amu'],
    }

    for name, feat_names in log_combos.items():
        X = [[row[f] for f in feat_names] for row in rows]
        coeffs, intercept, r2 = multi_linear_fit(X, log_lam)
        n = len(log_lam)
        p = len(feat_names)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        print(f"    {name}: R2={r2:.4f} (adj={r2_adj:.4f})")

    # ── 8. Direct T_c prediction (bypass McMillan entirely) ──────
    print()
    print("=" * 65)
    print("DIRECT T_c PREDICTION (bypass McMillan)")
    print("=" * 65)

    # Use only SC elements for this
    sc_rows = [r for r in rows if r['T_c_K'] > 0]
    non_sc_rows = [r for r in rows if r['T_c_K'] == 0]

    print(f"  SC elements: {len(sc_rows)}, Non-SC: {len(non_sc_rows)}")

    # Target: ln(T_c) for SC elements
    log_Tc = [math.log(r['T_c_K']) for r in sc_rows]

    print(f"\n  Single-variable correlations with ln(T_c):")
    tc_corrs = []
    for feat in features:
        valid = [(r[feat], math.log(r['T_c_K'])) for r in sc_rows
                 if r[feat] is not None]
        if len(valid) < 5:
            continue
        vals, lt = zip(*valid)
        r = pearson_r(list(vals), list(lt))
        tc_corrs.append((feat, r, r ** 2))

    tc_corrs.sort(key=lambda x: -abs(x[1]))
    print(f"  {'Feature':<25} {'r(lnTc)':>8} {'r2':>8}")
    print("  " + "-" * 43)
    for feat, r, r2 in tc_corrs[:10]:
        print(f"    {feat:<23} {r:+8.3f} {r2:8.3f}")

    # Multi-var direct T_c prediction
    print(f"\n  Multi-variable ln(T_c) regressions:")
    tc_combos = {
        'free/tD': ['free_over_thetaD'],
        'free/tD + E_coh/tD2': ['free_over_thetaD', 'ecoh_over_thetaD2'],
        'free/tD + E_coh/tD2 + 1/vF': ['free_over_thetaD', 'ecoh_over_thetaD2', 'inv_vF'],
        'free/tD + E_coh + 1/vF + A': ['free_over_thetaD', 'E_coh_eV', 'inv_vF', 'A_amu'],
        'free/tD + theta_D': ['free_over_thetaD', 'theta_D_K'],
        'free/tD + E_coh + theta_D': ['free_over_thetaD', 'E_coh_eV', 'theta_D_K'],
    }

    best_tc_r2 = 0
    best_tc_name = ''
    best_tc_coeffs = None
    best_tc_intercept = None
    best_tc_features = None

    for name, feat_names in tc_combos.items():
        X = [[row[f] for f in feat_names] for row in sc_rows]
        coeffs, intercept, r2 = multi_linear_fit(X, log_Tc)
        n = len(log_Tc)
        p = len(feat_names)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        print(f"    {name}: R2={r2:.4f} (adj={r2_adj:.4f})")

        if r2_adj > best_tc_r2:
            best_tc_r2 = r2_adj
            best_tc_name = name
            best_tc_coeffs = coeffs
            best_tc_intercept = intercept
            best_tc_features = feat_names

    if best_tc_features:
        print(f"\n  Best direct model: {best_tc_name} (adj R2={best_tc_r2:.4f})")
        print(f"\n  {'Element':<16} {'Tc_meas':>8} {'Tc_pred':>8} {'ratio':>8}")
        print("  " + "-" * 43)

        for row in sc_rows:
            X_i = [row[f] for f in best_tc_features]
            ln_tc_pred = best_tc_intercept + sum(c * x for c, x in zip(best_tc_coeffs, X_i))
            tc_pred = math.exp(ln_tc_pred)
            ratio = tc_pred / row['T_c_K'] if row['T_c_K'] > 0 else 0
            print(f"    {row['key']:<14} {row['T_c_K']:8.3f} {tc_pred:8.3f} {ratio:8.2f}x")

        # Check non-SC elements
        print(f"\n  Non-SC predictions (should be near 0):")
        for row in non_sc_rows:
            X_i = [row[f] for f in best_tc_features]
            ln_tc_pred = best_tc_intercept + sum(c * x for c, x in zip(best_tc_coeffs, X_i))
            tc_pred = math.exp(ln_tc_pred)
            status = "OK" if tc_pred < 0.5 else "HIGH"
            print(f"    {row['key']:<14} Tc_pred = {tc_pred:8.3f} K  {status}")

    # ── 9. Power-law fit: lambda = A * IE1^a * free^b * theta_D^c ─
    print()
    print("=" * 65)
    print("POWER-LAW FIT (ln-space multi-linear)")
    print("=" * 65)

    # ln(lambda) = ln(A) + a*ln(IE1) + b*ln(free) + c*ln(theta_D) + ...
    # Filter elements where all values are > 0
    pl_rows = [r for r in rows if r['IE1_eV'] > 0 and r['free_electrons'] > 0
               and r['theta_D_K'] > 0]

    ln_lam = [math.log(r['lambda_ep']) for r in pl_rows]
    ln_IE1 = [math.log(r['IE1_eV']) for r in pl_rows]
    ln_free = [math.log(r['free_electrons']) for r in pl_rows]
    ln_thetaD = [math.log(r['theta_D_K']) for r in pl_rows]
    ln_A = [math.log(r['A_amu']) for r in pl_rows]
    ln_Ecoh = [math.log(max(r['E_coh_eV'], 0.01)) for r in pl_rows]

    # Power-law combos
    pl_combos = {
        'PL1: IE1^a * free^b * tD^c': [ln_IE1, ln_free, ln_thetaD],
        'PL2: IE1^a * free^b * tD^c * A^d': [ln_IE1, ln_free, ln_thetaD, ln_A],
        'PL3: IE1^a * tD^c': [ln_IE1, ln_thetaD],
        'PL4: free^b * tD^c': [ln_free, ln_thetaD],
        'PL5: IE1^a * free^b * tD^c * Ecoh^e': [ln_IE1, ln_free, ln_thetaD, ln_Ecoh],
    }

    for name, feat_cols in pl_combos.items():
        X = list(zip(*feat_cols))
        X = [list(row) for row in X]
        coeffs, intercept, r2 = multi_linear_fit(X, ln_lam)
        n = len(ln_lam)
        p = len(feat_cols)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        print(f"  {name}")
        print(f"    R2 = {r2:.4f} (adj = {r2_adj:.4f})")
        print(f"    ln(lam) = {intercept:.4f}", end='')
        labels = name.split(': ')[1].split(' * ')
        for c, lab in zip(coeffs, labels):
            print(f" {c:+.4f}*ln({lab})", end='')
        print(f"\n    => lam = {math.exp(intercept):.4f}", end='')
        for c, lab in zip(coeffs, labels):
            print(f" * {lab.split('^')[0]}^{c:.3f}", end='')
        print()

    # ── 10. d-Row splitting ──────────────────────────────────────
    print()
    print("=" * 65)
    print("d-ROW SPLITTING (3d vs 4d vs 5d vs sp)")
    print("=" * 65)

    row_groups = {}
    for r in rows:
        dr = r['d_row'] if r['d_row'] is not None else 0
        if dr not in row_groups:
            row_groups[dr] = []
        row_groups[dr].append(r)

    for dr in sorted(row_groups.keys()):
        grp = row_groups[dr]
        label = {0: 'sp metals', 3: '3d metals', 4: '4d metals', 5: '5d metals'}.get(dr, f'row {dr}')
        print(f"\n  {label} ({len(grp)} elements):")
        if len(grp) < 3:
            for r in grp:
                print(f"    {r['key']:<14} lam={r['lambda_ep']:.3f}")
            continue

        grp_lam = [r['lambda_ep'] for r in grp]
        grp_fot = [r['free_over_thetaD'] for r in grp]
        grp_ie1fot = [r['IE1_free_thetaD'] for r in grp]

        # free/thetaD correlation within this row
        r_fot = pearson_r(grp_fot, grp_lam)
        a_fot, b_fot, r2_fot = linear_fit(grp_fot, grp_lam)

        # IE1*free/thetaD correlation within this row
        r_ie1 = pearson_r(grp_ie1fot, grp_lam)
        a_ie1, b_ie1, r2_ie1 = linear_fit(grp_ie1fot, grp_lam)

        print(f"    free/tD:      r={r_fot:+.3f}, R2={r2_fot:.4f}")
        print(f"    IE1*free/tD:  r={r_ie1:+.3f}, R2={r2_ie1:.4f}")

        for r in sorted(grp, key=lambda x: -x['lambda_ep']):
            print(f"    {r['key']:<14} lam={r['lambda_ep']:.3f}  "
                  f"IE1={r['IE1_eV']:.2f}  free={r['free_electrons']}  "
                  f"tD={r['theta_D_K']:.0f}")

    # ── 11. Row-aware model ────────────────────────────────────────
    print()
    print("=" * 65)
    print("ROW-AWARE MODEL")
    print("=" * 65)

    # Fit separate coefficients per d-row for IE1*free/thetaD
    # Then combine into a single formula using row indicator
    print("\n  Per-row fits for IE1*free/thetaD:")
    row_fits = {}
    for dr in sorted(row_groups.keys()):
        grp = row_groups[dr]
        if len(grp) < 3:
            continue
        label = {0: 'sp', 3: '3d', 4: '4d', 5: '5d'}.get(dr, f'row{dr}')
        grp_lam = [r['lambda_ep'] for r in grp]
        grp_ie1fot = [r['IE1_free_thetaD'] for r in grp]
        a, b, r2 = linear_fit(grp_ie1fot, grp_lam)
        row_fits[dr] = (a, b, r2, len(grp))
        print(f"    {label}: lam = {a:.4f} * IE1*free/tD + {b:.4f}  (R2={r2:.4f}, n={len(grp)})")

    # Apply per-row fit to all elements
    print(f"\n  Row-aware predictions:")
    print(f"  {'Element':<16} {'row':>4} {'lam_m':>7} {'lam_p':>7} {'err':>7} "
          f"{'Tc_m':>8} {'Tc_p':>8}")
    print("  " + "-" * 60)

    ra_tc_errors_sc = []
    ra_lam_errors = []

    for r in rows:
        dr = r['d_row'] if r['d_row'] is not None else 0
        if dr in row_fits:
            a, b, _, _ = row_fits[dr]
        elif 5 in row_fits:
            # Fallback to 5d for actinides (row 6d/7d treated as 5d)
            a, b, _, _ = row_fits[5]
        else:
            continue

        lam_pred = max(0.0, a * r['IE1_free_thetaD'] + b)
        lam_err = lam_pred - r['lambda_ep']
        ra_lam_errors.append(abs(lam_err))

        # T_c prediction
        mu = r['mu_star']
        tD = r['theta_D_K']
        if lam_pred > mu * (1 + 0.62 * lam_pred):
            tc_pred = mcmillan_Tc(tD, lam_pred, mu)
        else:
            tc_pred = 0.0

        tc_err = tc_pred - r['T_c_K']
        label = {0: 'sp', 3: '3d', 4: '4d', 5: '5d'}.get(dr, str(dr))

        print(f"  {r['key']:<14} {label:>4} {r['lambda_ep']:7.3f} {lam_pred:7.3f} "
              f"{lam_err:+7.3f} {r['T_c_K']:8.3f} {tc_pred:8.3f}")

        if r['T_c_K'] > 0:
            ra_tc_errors_sc.append(abs(tc_err))

    if ra_lam_errors:
        ra_lam_mae = sum(ra_lam_errors) / len(ra_lam_errors)
        ra_lam_rmse = math.sqrt(sum(e ** 2 for e in ra_lam_errors) / len(ra_lam_errors))
        print(f"\n  Row-aware lambda: MAE = {ra_lam_mae:.4f}, RMSE = {ra_lam_rmse:.4f}")

    if ra_tc_errors_sc:
        ra_tc_mae = sum(ra_tc_errors_sc) / len(ra_tc_errors_sc)
        ra_tc_rmse = math.sqrt(sum(e ** 2 for e in ra_tc_errors_sc) / len(ra_tc_errors_sc))
        print(f"  Row-aware T_c (SC): MAE = {ra_tc_mae:.2f} K, RMSE = {ra_tc_rmse:.2f} K")

    # Compute overall R2 for row-aware model
    lam_all = [r['lambda_ep'] for r in rows]
    lam_mean = sum(lam_all) / len(lam_all)

    lam_preds_ra = []
    for r in rows:
        dr = r['d_row'] if r['d_row'] is not None else 0
        if dr in row_fits:
            a, b, _, _ = row_fits[dr]
        elif 5 in row_fits:
            a, b, _, _ = row_fits[5]
        else:
            lam_preds_ra.append(lam_mean)
            continue
        lam_preds_ra.append(max(0.0, a * r['IE1_free_thetaD'] + b))

    ss_res_ra = sum((m - p) ** 2 for m, p in zip(lam_all, lam_preds_ra))
    ss_tot_ra = sum((m - lam_mean) ** 2 for m in lam_all)
    r2_ra = 1 - ss_res_ra / ss_tot_ra if ss_tot_ra > 0 else 0

    # Adjusted R2: k = number of parameters (2 per row * 3 rows = 6)
    k_ra = sum(2 for dr in row_fits)
    n_ra = len(rows)
    r2_adj_ra = 1 - (1 - r2_ra) * (n_ra - 1) / (n_ra - k_ra - 1) if n_ra > k_ra + 1 else r2_ra

    print(f"\n  Overall R2 = {r2_ra:.4f} (adj = {r2_adj_ra:.4f})")

    # ── 12. Summary ──────────────────────────────────────────────
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Best lambda model: {best_name}")
    print(f"  Adjusted R2:       {best_r2:.4f}")
    print(f"  LOO RMSE:          {loo_rmse:.4f}")
    print(f"  lambda MAE:        {lam_mae:.4f}")
    if sc_tc:
        print(f"  T_c MAE (SC):      {tc_mae:.2f} K")
        print(f"  T_c RMSE (SC):     {tc_rmse:.2f} K")

    print(f"\n  Best direct T_c:   {best_tc_name}")
    print(f"  Direct adj R2:     {best_tc_r2:.4f}")

    # Print the formula
    print(f"\n  Lambda formula:")
    print(f"    lambda_pred = {best_intercept:.4f}", end='')
    for c, f in zip(best_coeffs, best_features):
        print(f" {c:+.6f} * {f}", end='')
    print()

    if best_tc_features:
        print(f"\n  Direct T_c formula (SC elements only):")
        print(f"    ln(T_c) = {best_tc_intercept:.4f}", end='')
        for c, f in zip(best_tc_coeffs, best_tc_features):
            print(f" {c:+.6f} * {f}", end='')
        print()


if __name__ == '__main__':
    main()
