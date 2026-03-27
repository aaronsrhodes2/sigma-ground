"""
Superconductivity — BCS theory, critical fields, London penetration, Meissner effect.

Derivation chains:

  1. BCS Energy Gap (Bardeen-Cooper-Schrieffer 1957, FIRST_PRINCIPLES)
     Δ(0) = 1.764 × k_B × T_c

     At T=0, the superconducting gap is proportional to T_c.
     The factor 1.764 is the weak-coupling BCS result (exact in the
     weak-coupling limit).

     Temperature dependence (BCS approximation):
       Δ(T) ≈ Δ(0) × tanh(1.74 √(T_c/T − 1))   for T < T_c

     This is a convenient analytic approximation to the self-consistent
     BCS gap equation, accurate to ±2%.

  2. Critical Temperature T_c (MEASURED)
     T_c is the fundamental measured quantity for each superconductor.
     BCS theory predicts T_c from the electron-phonon coupling constant
     and Debye temperature, but the coupling constant is not independently
     measurable with sufficient accuracy.

  3. London Penetration Depth (FIRST_PRINCIPLES)
     λ_L = √(m_e / (μ₀ n_s e²))

     Where n_s = superconducting electron density ≈ n_e (conduction electrons).
     London & London (1935). The characteristic depth to which a magnetic
     field penetrates into a superconductor.

     Temperature dependence (Gorter-Casimir two-fluid model):
       λ_L(T) = λ_L(0) / √(1 − (T/T_c)⁴)

  4. Coherence Length — BCS/Pippard (FIRST_PRINCIPLES)
     ξ₀ = ℏ v_F / (π Δ(0))

     Where v_F = Fermi velocity. The size of a Cooper pair.
     Pippard (1953): in dirty superconductors, ξ_eff < ξ₀.

  5. Ginzburg-Landau Parameter (FIRST_PRINCIPLES)
     κ = λ_L / ξ

     Type I: κ < 1/√2 (complete Meissner effect, sharp transition)
     Type II: κ > 1/√2 (mixed state with vortices)

  6. Critical Magnetic Fields (FIRST_PRINCIPLES: thermodynamics)
     Type I:
       H_c(T) = H_c(0) × (1 − (T/T_c)²)
       H_c(0) = Δ(0) / (μ₀ μ_B √(2π))  (thermodynamic critical field)

     Type II (Abrikosov):
       H_c1 = H_c × (ln κ) / (√2 κ)   (lower critical field, vortex entry)
       H_c2 = √2 κ H_c                  (upper critical field, bulk normal)

     Abrikosov (1957): vortex lattice between H_c1 and H_c2.

  7. Critical Current Density — Depairing (FIRST_PRINCIPLES)
     J_c = Φ₀ / (3√3 π μ₀ λ_L² ξ)

     Where Φ₀ = h/(2e) = 2.068×10⁻¹⁵ Wb (flux quantum).
     This is the theoretical maximum (depairing current).
     Real J_c is lower due to vortex pinning and defects.

  8. Specific Heat Jump (BCS, FIRST_PRINCIPLES)
     ΔC / γT_c = 1.43

     The normalized specific heat discontinuity at T_c.
     γ = electronic specific heat coefficient (Sommerfeld).
     The factor 1.43 is the weak-coupling BCS universal ratio.

σ-dependence:
  T_c depends on the Debye temperature Θ_D and electron-phonon coupling g:
    T_c ∝ Θ_D × exp(−1/g)   (McMillan 1968)

  Under σ, Θ_D shifts through nuclear mass (heavier nuclei → softer phonons
  → lower Θ_D). The coupling g also shifts because phonon frequencies enter
  the coupling integral. Net effect:
    - T_c decreases slightly with increasing σ (heavier lattice)
    - λ_L and ξ₀ shift accordingly
    - H_c fields shift through Δ(0) ∝ T_c

  At Earth (σ ~ 7×10⁻¹⁰): < 10⁻⁹ change.
  At neutron star surface (σ ~ 0.1): T_c shift ~0.5% — potentially
  observable in neutron star crust superconductivity.

Origin tags:
  - BCS gap: FIRST_PRINCIPLES (weak-coupling BCS)
  - London penetration: FIRST_PRINCIPLES (London equations)
  - Coherence length: FIRST_PRINCIPLES (Pippard/BCS)
  - GL parameter: FIRST_PRINCIPLES (Ginzburg-Landau)
  - Critical fields: FIRST_PRINCIPLES (thermodynamics + Abrikosov)
  - T_c values: MEASURED
  - σ-dependence: CORE (through Θ_D shift, □σ = −ξR)
"""

import math
from ..constants import HBAR, C, K_B, E_CHARGE, MU_0, M_ELECTRON_KG, EPS_0

# ── Fundamental superconducting constants ─────────────────────────
# Flux quantum: Φ₀ = h/(2e) = π×ℏ/e
PHI_0 = math.pi * HBAR / E_CHARGE  # ≈ 2.068×10⁻¹⁵ Wb

# Bohr magneton for field conversions
_MU_BOHR = E_CHARGE * HBAR / (2.0 * M_ELECTRON_KG)  # ≈ 9.274×10⁻²⁴ J/T


# ── Derivation Helpers ──────────────────────────────────────────
# For Rule 9 compliance: every entry has every field.
# Elements derive n_e and v_F from free-electron model when not measured.
# Measured κ stored where available; otherwise derived from clean-limit BCS.

_N_A = 6.02214076e23  # Avogadro (mol⁻¹) — local (not yet in constants.py)


def _fe_n_e(rho, M_g, Z_val):
    """Free-electron density: n_e = Z_val × N_A × ρ / M.
    FIRST_PRINCIPLES: each atom contributes Z_val conduction electrons."""
    return Z_val * _N_A * rho / (M_g * 1e-3)


def _fe_v_F(n_e):
    """Sommerfeld Fermi velocity: v_F = ℏ(3π²n_e)^{1/3} / m_e.
    FIRST_PRINCIPLES: free-electron Fermi energy."""
    return HBAR * (3.0 * math.pi ** 2 * n_e) ** (1.0 / 3.0) / M_ELECTRON_KG


def _bcs_kappa(n_e, v_F, T_c):
    """Clean-limit BCS GL parameter: κ = λ_L / ξ₀.
    FIRST_PRINCIPLES: ratio of London depth to Pippard coherence length.
    NOTE: underestimates κ for dirty superconductors."""
    lam = math.sqrt(M_ELECTRON_KG / (MU_0 * n_e * E_CHARGE ** 2))
    delta_0 = 1.764 * K_B * T_c
    xi = HBAR * v_F / (math.pi * delta_0)
    return lam / xi


_K_THRESH = 1.0 / math.sqrt(2.0)  # ≈ 0.707


def _sc(name, T_c, rho, M_g, Z_val,
        n_e=None, v_F=None, kappa=None, pressure=None,
        lambda_ep=None, mu_star=None, theta_D_K=None,
        is_superconductor=True, suppression=None):
    """Build a complete superconductor/metal entry (Rule 9: every field populated).

    Args:
        name: human-readable name
        T_c: critical temperature (K), MEASURED (0.0 for non-SC metals)
        rho: density (kg/m³) — for n_e derivation
        M_g: molar mass (g/mol) — for n_e derivation
        Z_val: conduction electrons per atom — for n_e derivation
        n_e: override electron density (m⁻³) if measured
        v_F: override Fermi velocity (m/s) if measured
        kappa: override GL parameter if measured
        pressure: minimum pressure (GPa) for SC, None = ambient
        lambda_ep: electron-phonon coupling (MEASURED), None if unknown
        mu_star: Coulomb pseudopotential (MEASURED), None if unknown
        theta_D_K: Debye temperature (K, MEASURED), None if unknown
        is_superconductor: True for SC, False for normal metals
        suppression: None, or why T_c=0 despite λ ('ferromagnet',
                     'weak_coupling', 'spin_fluctuations')
    """
    _n_e = n_e if n_e is not None else _fe_n_e(rho, M_g, Z_val)
    _v_F = v_F if v_F is not None else _fe_v_F(_n_e)
    # For non-SC metals with T_c=0, kappa derivation would divide by zero.
    if T_c > 0:
        _kap = kappa if kappa is not None else _bcs_kappa(_n_e, _v_F, T_c)
    else:
        _kap = kappa if kappa is not None else 0.0
    return {
        'name': name,
        'T_c_K': T_c,
        'n_e_m3': _n_e,
        'v_F_m_s': _v_F,
        'kappa': _kap,
        'type': 'II' if _kap > _K_THRESH else 'I',
        'pressure_GPa': pressure,
        'kappa_source': 'measured' if kappa is not None else 'derived',
        'lambda_ep': lambda_ep,
        'mu_star': mu_star,
        'theta_D_K': theta_D_K,
        'is_superconductor': is_superconductor,
        'suppression': suppression,
    }


# ── Superconductor Database ──────────────────────────────────────
# Rule 9 — If One, Then All: every known superconductor, every field.
#
# Every entry has: name, T_c_K, n_e_m3, v_F_m_s, kappa, type,
#                  pressure_GPa, kappa_source
#
# T_c:    MEASURED (literature)
# n_e:    MEASURED or DERIVED (free-electron model)
# v_F:    MEASURED or DERIVED (Sommerfeld model)
# kappa:  MEASURED or DERIVED (clean-limit BCS — underestimates for dirty SC)
# type:   from κ > 1/√2 threshold
#
# Sources: Ashcroft & Mermin (1976), Poole et al. "Superconductivity" (2007),
#          CRC Handbook (2023), Buzea & Robbie (2005), NIST,
#          Roberts "Superconducting Materials List" (NBS 1978)
#
# Valence convention: s/p metals use group valence; d-block use effective
# conduction electron count (1–2, calibrated to reproduce known n_e).
# Actinides: 3 effective.

SUPERCONDUCTORS = {

    # ── Elemental — Ambient Pressure (ordered by Z) ────────────
    # McMillan data: λ from Allen & Dynes (1975), Carbotte (1990)
    #                μ* from McMillan (1968), typically 0.10-0.13
    #                Θ_D from CRC Handbook / Kittel

    'lithium': _sc('Lithium', 0.0004, 534, 6.941, 1,
                    theta_D_K=344),
    # Z=3, T_c = 0.4 mK (Webb et al. 1969). Extreme Type I.

    'beryllium': _sc('Beryllium', 0.026, 1850, 9.012, 2,
                      theta_D_K=1440),
    # Z=4, T_c = 26 mK. Type I.

    'aluminum': _sc('Aluminum', 1.175, 2700, 26.982, 3,
                     n_e=1.81e29, v_F=2.03e6, kappa=0.01,
                     lambda_ep=0.43, mu_star=0.10, theta_D_K=428),
    # Z=13, MEASURED n_e/v_F/κ/λ. Ashcroft & Mermin. Deep Type I.

    'titanium': _sc('Titanium', 0.40, 4507, 47.867, 2,
                     lambda_ep=0.38, mu_star=0.12, theta_D_K=420),
    # Z=22, T_c = 0.40 K. Type I.

    'vanadium': _sc('Vanadium', 5.40, 6110, 50.942, 2, kappa=0.85,
                     lambda_ep=0.80, mu_star=0.13, theta_D_K=380),
    # Z=23, MEASURED κ = 0.85 (Type II, dirty limit).

    'zinc': _sc('Zinc', 0.85, 7140, 65.38, 2, kappa=0.085,
                 lambda_ep=0.38, mu_star=0.10, theta_D_K=327),
    # Z=30, MEASURED κ. Type I.

    'gallium': _sc('Gallium', 1.083, 5904, 69.723, 3,
                    lambda_ep=0.40, mu_star=0.10, theta_D_K=325),
    # Z=31, T_c = 1.083 K. Type I.

    'zirconium': _sc('Zirconium', 0.61, 6506, 91.224, 2,
                      lambda_ep=0.41, mu_star=0.12, theta_D_K=291),
    # Z=40, T_c = 0.61 K. Type I.

    'niobium': _sc('Niobium', 9.25, 8570, 92.906, 1,
                    n_e=5.56e28, v_F=1.37e6, kappa=1.05,
                    lambda_ep=1.26, mu_star=0.13, theta_D_K=275),
    # Z=41, MEASURED n_e/v_F/κ/λ. Highest elemental T_c. Barely Type II.

    'molybdenum': _sc('Molybdenum', 0.915, 10220, 95.95, 1,
                       lambda_ep=0.41, mu_star=0.12, theta_D_K=450),
    # Z=42, T_c = 0.915 K. Type I.

    'technetium': _sc('Technetium', 7.8, 11500, 98.0, 1,
                       lambda_ep=0.82, mu_star=0.13, theta_D_K=411),
    # Z=43, T_c = 7.8 K. Radioactive. Second-highest elemental T_c.

    'ruthenium': _sc('Ruthenium', 0.49, 12370, 101.07, 1,
                      lambda_ep=0.38, mu_star=0.12, theta_D_K=600),
    # Z=44, T_c = 0.49 K. Type I.

    'rhodium': _sc('Rhodium', 0.000325, 12450, 102.906, 1,
                    theta_D_K=480),
    # Z=45, T_c = 0.325 mK. Extreme Type I. λ not well characterized.

    'cadmium': _sc('Cadmium', 0.517, 8650, 112.414, 2, kappa=0.14,
                    lambda_ep=0.38, mu_star=0.10, theta_D_K=209),
    # Z=48, MEASURED κ. Type I.

    'indium': _sc('Indium', 3.41, 7310, 114.818, 3, kappa=0.11,
                   lambda_ep=0.81, mu_star=0.10, theta_D_K=108),
    # Z=49, MEASURED κ/λ. Type I.

    'tin': _sc('Tin', 3.722, 7310, 118.71, 4,
               n_e=1.48e29, v_F=1.89e6, kappa=0.15,
               lambda_ep=0.72, mu_star=0.10, theta_D_K=200),
    # Z=50, MEASURED n_e/v_F/κ/λ. White tin (β-Sn). Type I.

    'lanthanum': _sc('Lanthanum', 6.0, 6146, 138.906, 3,
                      lambda_ep=0.77, mu_star=0.12, theta_D_K=152),
    # Z=57, T_c = 6.0 K (fcc β-La). Third-highest elemental T_c.

    'lutetium': _sc('Lutetium', 0.1, 9841, 174.967, 3,
                     theta_D_K=210),
    # Z=71, T_c ≈ 0.1 K. Type I. λ not well characterized.

    'hafnium': _sc('Hafnium', 0.128, 13310, 178.49, 2,
                    lambda_ep=0.26, mu_star=0.12, theta_D_K=252),
    # Z=72, T_c = 0.128 K. Type I.

    'tantalum': _sc('Tantalum', 4.47, 16650, 180.948, 2, kappa=0.45,
                     lambda_ep=0.69, mu_star=0.12, theta_D_K=258),
    # Z=73, MEASURED κ/λ. Type I.

    'tungsten': _sc('Tungsten', 0.0154, 19250, 183.84, 2,
                     lambda_ep=0.28, mu_star=0.12, theta_D_K=400),
    # Z=74, T_c = 15.4 mK (α-W). Extreme Type I.

    'rhenium': _sc('Rhenium', 1.697, 21020, 186.207, 2,
                    lambda_ep=0.46, mu_star=0.12, theta_D_K=430),
    # Z=75, T_c = 1.697 K. Type I.

    'osmium': _sc('Osmium', 0.66, 22610, 190.23, 2,
                   lambda_ep=0.38, mu_star=0.12, theta_D_K=500),
    # Z=76, T_c = 0.66 K. Type I.

    'iridium': _sc('Iridium', 0.1125, 22650, 192.217, 2,
                    lambda_ep=0.34, mu_star=0.12, theta_D_K=420),
    # Z=77, T_c = 0.1125 K. Type I.

    'mercury': _sc('Mercury', 4.153, 13534, 200.592, 2,
                    n_e=8.52e28, v_F=1.37e6, kappa=0.42,
                    lambda_ep=1.60, mu_star=0.13, theta_D_K=72),
    # Z=80, MEASURED n_e/v_F/κ/λ. First superconductor (Onnes 1911). Type I.

    'thallium': _sc('Thallium', 2.38, 11850, 204.38, 3, kappa=0.18,
                     lambda_ep=0.79, mu_star=0.10, theta_D_K=79),
    # Z=81, MEASURED κ/λ. Type I.

    'lead': _sc('Lead', 7.193, 11340, 207.2, 4,
                 n_e=1.32e29, v_F=1.83e6, kappa=0.48,
                 lambda_ep=1.55, mu_star=0.13, theta_D_K=105),
    # Z=82, MEASURED n_e/v_F/κ/λ. Type I.

    'thorium': _sc('Thorium', 1.38, 11720, 232.038, 2,
                    lambda_ep=0.56, mu_star=0.12, theta_D_K=170),
    # Z=90, T_c = 1.38 K. Only naturally occurring actinide SC.

    'protactinium': _sc('Protactinium', 1.4, 15370, 231.036, 2,
                         theta_D_K=185),
    # Z=91, T_c = 1.4 K. λ not well characterized.

    'uranium': _sc('Uranium', 0.2, 19050, 238.029, 3,
                    theta_D_K=207),
    # Z=92, T_c ≈ 0.2 K (α-U). λ not well characterized.

    'americium': _sc('Americium', 1.0, 13670, 243.0, 3),
    # Z=95, T_c ≈ 1.0 K. Radioactive actinide. λ/Θ_D not characterized.

    # ── Elemental — Pressure-Required (ordered by Z) ───────────
    # These elements require high pressure for superconductivity.
    # Density and n_e are for the ambient-pressure phase (approximation —
    # high-pressure phases have different crystal structure/density).
    # McMillan parameters not well-defined for high-pressure phases.

    'silicon': _sc('Silicon', 8.2, 2330, 28.086, 4, pressure=12.0,
                    theta_D_K=645),
    # Z=14, T_c = 8.2 K at 12 GPa (metallic Si-II phase).

    'phosphorus': _sc('Phosphorus', 6.0, 1820, 30.974, 3, pressure=10.0),
    # Z=15, T_c ≈ 6 K at 10 GPa.

    'sulfur': _sc('Sulfur', 17.0, 2070, 32.06, 2, pressure=160.0),
    # Z=16, T_c = 17 K at 160 GPa. Highest elemental T_c under pressure.

    'calcium': _sc('Calcium', 25.0, 1550, 40.078, 2, pressure=161.0,
                    theta_D_K=230),
    # Z=20, T_c = 25 K at 161 GPa.

    'iron': _sc('Iron', 2.0, 7874, 55.845, 2, pressure=15.0,
                 theta_D_K=470),
    # Z=26, T_c ≈ 2 K at 15 GPa (ε-Fe, hcp phase). Shimizu et al. 2001.

    'germanium': _sc('Germanium', 5.35, 5323, 72.63, 4, pressure=11.5,
                      theta_D_K=374),
    # Z=32, T_c = 5.35 K at 11.5 GPa.

    'arsenic': _sc('Arsenic', 2.7, 5727, 74.922, 3, pressure=25.0),
    # Z=33, T_c ≈ 2.7 K at 25 GPa.

    'selenium': _sc('Selenium', 6.9, 4819, 78.971, 2, pressure=13.0),
    # Z=34, T_c = 6.9 K at 13 GPa.

    'strontium': _sc('Strontium', 4.0, 2640, 87.62, 2, pressure=3.5,
                      theta_D_K=147),
    # Z=38, T_c ≈ 4 K at 3.5 GPa.

    'yttrium': _sc('Yttrium', 20.0, 4472, 88.906, 3, pressure=115.0,
                    theta_D_K=280),
    # Z=39, T_c ≈ 20 K at 115 GPa.

    'antimony': _sc('Antimony', 3.55, 6697, 121.76, 3, pressure=8.5),
    # Z=51, T_c = 3.55 K at 8.5 GPa.

    'tellurium': _sc('Tellurium', 7.4, 6240, 127.60, 2, pressure=4.0),
    # Z=52, T_c = 7.4 K at 4 GPa.

    'barium': _sc('Barium', 5.0, 3510, 137.327, 2, pressure=18.0,
                   theta_D_K=110),
    # Z=56, T_c ≈ 5 K at 18 GPa.

    'bismuth': _sc('Bismuth', 8.0, 9780, 208.98, 5, pressure=2.55,
                    theta_D_K=119),
    # Z=83, T_c ≈ 8 K at 2.55 GPa.

    # ── Compounds & Alloys ─────────────────────────────────────
    # n_e and v_F stored directly (free-electron model not applicable).
    # All have MEASURED κ. McMillan not applicable to cuprates (d-wave).

    'NbTi': _sc('Niobium-Titanium', 10.0, 0, 1, 0,
                 n_e=4.2e28, v_F=1.2e6, kappa=80.0),
    # Most common SC wire material. Extreme Type II (dirty alloy).

    'Nb3Sn': _sc('Niobium-Tin (A15)', 18.3, 0, 1, 0,
                   n_e=5.0e28, v_F=1.0e6, kappa=23.0,
                   lambda_ep=1.80, mu_star=0.13, theta_D_K=228),
    # A15 compound. High-field magnets.

    'Nb3Ge': _sc('Niobium-Germanium (A15)', 23.2, 0, 1, 0,
                   n_e=4.8e28, v_F=0.9e6, kappa=20.0),
    # A15 compound. Highest A15 T_c.

    'V3Si': _sc('Vanadium-Silicon (A15)', 17.1, 0, 1, 0,
                  n_e=6.0e28, v_F=1.1e6, kappa=15.0,
                  lambda_ep=1.10, mu_star=0.13, theta_D_K=334),
    # A15 compound.

    'MgB2': _sc('Magnesium Diboride', 39.0, 0, 1, 0,
                 n_e=6.7e28, v_F=4.8e5, kappa=26.0,
                 lambda_ep=0.87, mu_star=0.10, theta_D_K=750),
    # Nagamatsu et al. (2001). Two-gap superconductor.

    'PbMo6S8': _sc('Lead Molybdenum Sulfide (Chevrel)', 15.3, 0, 1, 0,
                     n_e=3.5e28, v_F=5.0e5, kappa=30.0),
    # Chevrel phase. High critical field.

    'YBCO': _sc('YBa₂Cu₃O₇ (YBCO)', 92.0, 0, 1, 0,
                 n_e=5.9e27, v_F=2.5e5, kappa=95.0),
    # First cuprate above 77 K (liquid nitrogen). Wu et al. 1987.
    # McMillan NOT applicable — d-wave pairing, not phonon-mediated.

    'BSCCO_2212': _sc('Bi₂Sr₂CaCu₂O₈ (BSCCO-2212)', 85.0, 0, 1, 0,
                        n_e=5.0e27, v_F=2.0e5, kappa=140.0),
    # Bi-based cuprate. Highly anisotropic. McMillan NOT applicable.

    'BSCCO_2223': _sc('Bi₂Sr₂Ca₂Cu₃O₁₀ (BSCCO-2223)', 110.0, 0, 1, 0,
                        n_e=4.5e27, v_F=2.0e5, kappa=150.0),
    # Three-layer Bi cuprate. Highest BSCCO T_c.

    # ── Non-Superconducting Metals ─────────────────────────────
    # These have measurable λ and Θ_D but do NOT superconduct at ambient.
    # McMillan predicts T_c ≈ 0 for weak-coupling metals (Cu, Ag, Au, Pt).
    # For ferromagnets (Fe, Co, Ni), McMillan predicts nonzero T_c but
    # magnetic ordering destroys Cooper pairing — flagged via 'suppression'.
    # For Pd, spin fluctuations suppress SC despite adequate coupling.

    'copper': _sc('Copper', 0.0, 8960, 63.546, 1,
                   is_superconductor=False, suppression='weak_coupling',
                   lambda_ep=0.13, mu_star=0.10, theta_D_K=343),
    # Z=29. Noble metal. λ too weak to overcome μ*.

    'silver': _sc('Silver', 0.0, 10490, 107.868, 1,
                   is_superconductor=False, suppression='weak_coupling',
                   lambda_ep=0.12, mu_star=0.10, theta_D_K=225),
    # Z=47. Noble metal. Weakest electron-phonon coupling of common metals.

    'gold': _sc('Gold', 0.0, 19300, 196.967, 1,
                 is_superconductor=False, suppression='weak_coupling',
                 lambda_ep=0.15, mu_star=0.10, theta_D_K=165),
    # Z=79. Noble metal. Excellent conductor but no SC.

    'platinum': _sc('Platinum', 0.0, 21450, 195.084, 1,
                     is_superconductor=False, suppression='weak_coupling',
                     lambda_ep=0.13, mu_star=0.10, theta_D_K=240),
    # Z=78. λ too weak.

    'palladium': _sc('Palladium', 0.0, 12020, 106.42, 1,
                      is_superconductor=False, suppression='spin_fluctuations',
                      lambda_ep=0.47, mu_star=0.13, theta_D_K=274),
    # Z=46. Enhanced paramagnet. λ adequate but spin fluctuations kill SC.

    'iron_ambient': _sc('Iron (ambient)', 0.0, 7874, 55.845, 2,
                          is_superconductor=False, suppression='ferromagnet',
                          lambda_ep=0.47, mu_star=0.12, theta_D_K=470),
    # Z=26. Ferromagnet at ambient pressure. McMillan predicts SC but
    # magnetic ordering destroys Cooper pairs. SC at 15 GPa (see 'iron').

    'cobalt': _sc('Cobalt', 0.0, 8900, 58.933, 2,
                   is_superconductor=False, suppression='ferromagnet',
                   lambda_ep=0.35, mu_star=0.12, theta_D_K=445),
    # Z=27. Ferromagnet. No SC observed.

    'nickel': _sc('Nickel', 0.0, 8908, 58.693, 2,
                   is_superconductor=False, suppression='ferromagnet',
                   lambda_ep=0.33, mu_star=0.12, theta_D_K=450),
    # Z=28. Ferromagnet. No SC observed.
}


# ── BCS Energy Gap ───────────────────────────────────────────────

def bcs_gap_zero(T_c):
    """BCS energy gap at T=0 (Joules).

    Δ(0) = 1.764 × k_B × T_c

    FIRST_PRINCIPLES: weak-coupling BCS result. The factor 1.764
    = π × exp(−γ_E) where γ_E is the Euler-Mascheroni constant.
    Exact in the BCS weak-coupling limit.

    Args:
        T_c: critical temperature (K)

    Returns:
        Energy gap Δ(0) in Joules
    """
    return 1.764 * K_B * T_c


def bcs_gap_temperature(T_c, T):
    """BCS energy gap at temperature T (Joules).

    Δ(T) = Δ(0) × tanh(1.74 √(T_c/T − 1))   for T < T_c
    Δ(T) = 0                                     for T ≥ T_c

    APPROXIMATION: Mühlschlegel (1959) fit to self-consistent BCS gap.
    Accuracy: ±2% over full temperature range.

    Args:
        T_c: critical temperature (K)
        T: temperature (K)

    Returns:
        Energy gap Δ(T) in Joules
    """
    if T >= T_c:
        return 0.0
    if T <= 0:
        return bcs_gap_zero(T_c)

    delta_0 = bcs_gap_zero(T_c)
    return delta_0 * math.tanh(1.74 * math.sqrt(T_c / T - 1.0))


def gap_frequency(T_c):
    """Gap frequency ν_gap = 2Δ(0)/h (Hz).

    The minimum photon frequency that can break a Cooper pair.

    FIRST_PRINCIPLES: energy conservation.

    Args:
        T_c: critical temperature (K)

    Returns:
        Gap frequency in Hz
    """
    delta = bcs_gap_zero(T_c)
    return 2.0 * delta / (2.0 * math.pi * HBAR)


# ── London Penetration Depth ─────────────────────────────────────

def london_penetration_depth(n_e):
    """London penetration depth at T=0 (m).

    λ_L = √(m_e / (μ₀ n_s e²))

    FIRST_PRINCIPLES: London equation. The depth to which magnetic
    field penetrates a superconductor. Set by the superfluid density.

    Args:
        n_e: superconducting electron density (m⁻³)

    Returns:
        Penetration depth in metres
    """
    return math.sqrt(M_ELECTRON_KG / (MU_0 * n_e * E_CHARGE ** 2))


def london_penetration_at_T(n_e, T_c, T):
    """London penetration depth at temperature T (m).

    λ_L(T) = λ_L(0) / √(1 − (T/T_c)⁴)

    FIRST_PRINCIPLES: Gorter-Casimir two-fluid model (1934).
    As T → T_c, penetration diverges (more normal electrons).

    Args:
        n_e: electron density (m⁻³)
        T_c: critical temperature (K)
        T: temperature (K)

    Returns:
        Penetration depth in metres
    """
    if T >= T_c:
        return float('inf')

    lam_0 = london_penetration_depth(n_e)
    return lam_0 / math.sqrt(1.0 - (T / T_c) ** 4)


# ── Coherence Length ─────────────────────────────────────────────

def bcs_coherence_length(v_F, T_c):
    """BCS coherence length ξ₀ (m).

    ξ₀ = ℏ v_F / (π Δ(0))

    FIRST_PRINCIPLES: Pippard (1953) / BCS. The spatial extent
    of a Cooper pair — the characteristic correlation length.

    Args:
        v_F: Fermi velocity (m/s)
        T_c: critical temperature (K)

    Returns:
        Coherence length in metres
    """
    delta_0 = bcs_gap_zero(T_c)
    return HBAR * v_F / (math.pi * delta_0)


# ── Ginzburg-Landau Parameter ────────────────────────────────────

def gl_parameter(n_e, v_F, T_c):
    """Ginzburg-Landau parameter κ = λ_L/ξ₀ (clean-limit BCS estimate).

    κ < 1/√2: Type I superconductor
    κ > 1/√2: Type II superconductor

    FIRST_PRINCIPLES: ratio of two fundamental lengths.
    NOTE: This is the clean-limit estimate. For dirty superconductors
    (Nb, alloys), mean free path effects increase κ significantly.
    Use gl_parameter_effective() with a database key for measured values.

    Args:
        n_e: electron density (m⁻³)
        v_F: Fermi velocity (m/s)
        T_c: critical temperature (K)

    Returns:
        κ (dimensionless)
    """
    lam = london_penetration_depth(n_e)
    xi = bcs_coherence_length(v_F, T_c)
    return lam / xi


def gl_parameter_effective(sc_key):
    """Effective GL parameter κ using MEASURED value when available.

    The clean-limit BCS formula underestimates κ for dirty superconductors.
    We prefer measured κ from the database.

    Args:
        sc_key: key into SUPERCONDUCTORS dict

    Returns:
        κ (dimensionless)
    """
    data = SUPERCONDUCTORS[sc_key]
    if 'kappa' in data:
        return data['kappa']
    return gl_parameter(data['n_e_m3'], data['v_F_m_s'], data['T_c_K'])


def is_type_II(n_e, v_F, T_c, sc_key=None):
    """Determine if superconductor is Type II.

    Type II when κ > 1/√2 ≈ 0.707

    If sc_key is provided, uses MEASURED κ from database.
    Otherwise falls back to clean-limit BCS estimate.

    Args:
        n_e, v_F, T_c: superconductor parameters
        sc_key: optional database key for measured κ

    Returns:
        True if Type II
    """
    if sc_key is not None:
        kappa = gl_parameter_effective(sc_key)
    else:
        kappa = gl_parameter(n_e, v_F, T_c)
    return kappa > 1.0 / math.sqrt(2.0)


# ── Critical Fields ──────────────────────────────────────────────

def thermodynamic_critical_field(n_e, T_c, T=0.0):
    """Thermodynamic critical field H_c (A/m).

    H_c(0) = Δ(0) × √(n_e) / (μ₀^(1/2) × √(k_B T_c))
    Simplified: H_c(0) = Δ(0) / (μ₀ × λ_L × √2)

    H_c(T) = H_c(0) × (1 − (T/T_c)²)

    FIRST_PRINCIPLES: condensation energy = (1/2)μ₀ H_c².

    Args:
        n_e: electron density (m⁻³)
        T_c: critical temperature (K)
        T: temperature (K)

    Returns:
        H_c in A/m
    """
    if T >= T_c:
        return 0.0

    delta_0 = bcs_gap_zero(T_c)
    lam = london_penetration_depth(n_e)

    H_c0 = delta_0 / (MU_0 * lam * math.sqrt(2.0))
    return H_c0 * (1.0 - (T / T_c) ** 2)


def lower_critical_field(n_e, v_F, T_c, T=0.0, sc_key=None):
    """Lower critical field H_c1 for Type II superconductors (A/m).

    H_c1 = H_c × ln(κ) / (√2 κ)

    FIRST_PRINCIPLES: energy balance for a single vortex entry.
    Below H_c1: complete Meissner effect.
    Above H_c1: vortices begin to enter.

    Args:
        n_e, v_F, T_c: superconductor parameters
        T: temperature (K)
        sc_key: optional database key for measured κ

    Returns:
        H_c1 in A/m
    """
    H_c = thermodynamic_critical_field(n_e, T_c, T)
    if sc_key is not None:
        kappa = gl_parameter_effective(sc_key)
    else:
        kappa = gl_parameter(n_e, v_F, T_c)

    if kappa <= 1:
        return H_c

    return H_c * math.log(kappa) / (math.sqrt(2.0) * kappa)


def upper_critical_field(n_e, v_F, T_c, T=0.0, sc_key=None):
    """Upper critical field H_c2 for Type II superconductors (A/m).

    H_c2 = √2 κ H_c

    FIRST_PRINCIPLES: Abrikosov (1957). Bulk superconductivity
    destroyed when vortex cores overlap.
    Above H_c2: fully normal state.

    Args:
        n_e, v_F, T_c: superconductor parameters
        T: temperature (K)
        sc_key: optional database key for measured κ

    Returns:
        H_c2 in A/m
    """
    H_c = thermodynamic_critical_field(n_e, T_c, T)
    if sc_key is not None:
        kappa = gl_parameter_effective(sc_key)
    else:
        kappa = gl_parameter(n_e, v_F, T_c)
    return math.sqrt(2.0) * kappa * H_c


# ── Critical Current ─────────────────────────────────────────────

def depairing_current_density(n_e, v_F, T_c):
    """Depairing (maximum theoretical) critical current density (A/m²).

    J_c = Φ₀ / (3√3 π μ₀ λ_L² ξ₀)

    FIRST_PRINCIPLES: Ginzburg-Landau depairing limit.
    The current density that destroys superconductivity by breaking
    Cooper pairs. Real J_c is always lower (vortex motion, defects).

    Args:
        n_e, v_F, T_c: superconductor parameters

    Returns:
        J_c in A/m²
    """
    lam = london_penetration_depth(n_e)
    xi = bcs_coherence_length(v_F, T_c)

    return PHI_0 / (3.0 * math.sqrt(3.0) * math.pi * MU_0 * lam ** 2 * xi)


# ── Specific Heat Jump ───────────────────────────────────────────

def specific_heat_jump_ratio():
    """BCS specific heat discontinuity ratio.

    ΔC / (γ T_c) = 1.43

    FIRST_PRINCIPLES: universal BCS weak-coupling prediction.
    γ = Sommerfeld electronic specific heat coefficient.

    Returns:
        1.43 (dimensionless, universal)
    """
    return 1.43


# ── Meissner Fraction ────────────────────────────────────────────

def meissner_fraction(T_c, T):
    """Fraction of superconducting electrons (two-fluid model).

    n_s/n = 1 − (T/T_c)⁴

    FIRST_PRINCIPLES: Gorter-Casimir two-fluid model.
    At T=0: all electrons superconducting.
    At T=T_c: n_s → 0 (normal state).

    Args:
        T_c: critical temperature (K)
        T: temperature (K)

    Returns:
        Superconducting fraction (0 to 1)
    """
    if T >= T_c:
        return 0.0
    if T <= 0:
        return 1.0
    return 1.0 - (T / T_c) ** 4


# ── σ-Dependence ─────────────────────────────────────────────────

def sigma_Tc_shift(T_c_0, sigma):
    """Critical temperature under σ-field.

    T_c depends on Debye temperature Θ_D:
      T_c ∝ Θ_D × exp(−1/g)   (McMillan 1968)

    Under σ, Θ_D shifts through nuclear mass:
      Θ_D(σ) = Θ_D(0) / √(mass_ratio)

    The coupling g also shifts, but the dominant effect is through Θ_D.

    CORE: σ-dependence through □σ = −ξR → nuclear mass → Debye temp.

    Args:
        T_c_0: critical temperature at σ=0 (K)
        sigma: σ-field value

    Returns:
        T_c(σ) in Kelvin
    """
    from ..scale import scale_ratio
    from ..constants import PROTON_QCD_FRACTION

    if sigma == 0.0:
        return T_c_0

    f_qcd = PROTON_QCD_FRACTION
    mass_ratio = (1.0 - f_qcd) + f_qcd * scale_ratio(sigma)

    # Θ_D ∝ 1/√M → T_c ∝ Θ_D ∝ 1/√M (dominant effect)
    return T_c_0 / math.sqrt(mass_ratio)


def sigma_gap_shift(T_c_0, sigma, T=0.0):
    """BCS gap under σ-field.

    Δ(0, σ) = 1.764 × k_B × T_c(σ)

    CORE: through T_c(σ).

    Args:
        T_c_0: critical temperature at σ=0 (K)
        sigma: σ-field value
        T: temperature (K)

    Returns:
        Gap Δ(T, σ) in Joules
    """
    T_c_sigma = sigma_Tc_shift(T_c_0, sigma)
    return bcs_gap_temperature(T_c_sigma, T)


# ── McMillan Formula ────────────────────────────────────────────

def mcmillan_Tc(theta_D, lambda_ep, mu_star):
    """Predicted critical temperature from McMillan formula (1968).

    T_c = (Θ_D / 1.45) × exp(−1.04(1+λ) / (λ − μ*(1+0.62λ)))

    McMillan, Phys. Rev. 167, 331 (1968).

    FIRST_PRINCIPLES: electron-phonon theory of superconductivity.
    Valid for λ ≲ 1.5. For stronger coupling, Allen-Dynes (1975)
    modification gives better results.

    Guard: if denominator ≤ 0, returns 0.0 (coupling too weak to
    overcome Coulomb repulsion — no superconductivity).

    Args:
        theta_D: Debye temperature (K)
        lambda_ep: electron-phonon coupling constant (dimensionless)
        mu_star: Coulomb pseudopotential (dimensionless, typically 0.10–0.13)

    Returns:
        Predicted T_c in Kelvin (0.0 if coupling insufficient)
    """
    denom = lambda_ep - mu_star * (1.0 + 0.62 * lambda_ep)
    if denom <= 0.0:
        return 0.0
    exponent = -1.04 * (1.0 + lambda_ep) / denom
    return (theta_D / 1.45) * math.exp(exponent)


def mcmillan_Tc_for(sc_key):
    """McMillan-predicted T_c for a database entry.

    Returns None if λ or Θ_D data is missing for this entry.

    Args:
        sc_key: key into SUPERCONDUCTORS dict

    Returns:
        Predicted T_c in Kelvin, or None
    """
    data = SUPERCONDUCTORS[sc_key]
    lam = data.get('lambda_ep')
    mu = data.get('mu_star')
    theta = data.get('theta_D_K')
    if lam is None or mu is None or theta is None:
        return None
    return mcmillan_Tc(theta, lam, mu)


def sigma_mcmillan_Tc(theta_D, lambda_ep, mu_star, sigma):
    """McMillan T_c under σ-field.

    CORE: Θ_D shifts through nuclear mass under σ.
    Θ_D(σ) = Θ_D(0) / √(mass_ratio(σ))

    λ also shifts (phonon frequencies enter coupling integral),
    but the dominant effect is through Θ_D.

    Args:
        theta_D: Debye temperature at σ=0 (K)
        lambda_ep: electron-phonon coupling constant
        mu_star: Coulomb pseudopotential
        sigma: σ-field value

    Returns:
        Predicted T_c(σ) in Kelvin
    """
    from ..scale import scale_ratio
    from ..constants import PROTON_QCD_FRACTION

    if sigma == 0.0:
        return mcmillan_Tc(theta_D, lambda_ep, mu_star)

    f_qcd = PROTON_QCD_FRACTION
    mass_ratio = (1.0 - f_qcd) + f_qcd * scale_ratio(sigma)
    theta_D_sigma = theta_D / math.sqrt(mass_ratio)
    return mcmillan_Tc(theta_D_sigma, lambda_ep, mu_star)


def debye_comparison():
    """Compare derived Θ_D (from thermal.py matter model) vs measured Θ_D.

    For the materials where thermal.py can derive Θ_D from bulk modulus
    and density, compare against measured values stored in SUPERCONDUCTORS.

    Returns:
        List of dicts: material, derived_theta_D, measured_theta_D, percent_error
    """
    from .thermal import debye_temperature as derive_theta_D

    # Mapping: thermal.py MATERIALS key → SUPERCONDUCTORS key
    _OVERLAP = {
        'aluminum': 'aluminum',
        'titanium': 'titanium',
        'copper': 'copper',
        'gold': 'gold',
        'iron': 'iron_ambient',
        'nickel': 'nickel',
        'tungsten': 'tungsten',
        'silicon': 'silicon',
    }
    results = []
    for mat_key, sc_key in _OVERLAP.items():
        if sc_key not in SUPERCONDUCTORS:
            continue
        entry = SUPERCONDUCTORS[sc_key]
        measured = entry.get('theta_D_K')
        if measured is None:
            continue
        derived = derive_theta_D(mat_key)
        pct = 100.0 * (derived - measured) / measured
        results.append({
            'material': mat_key,
            'sc_key': sc_key,
            'derived_theta_D': derived,
            'measured_theta_D': measured,
            'percent_error': pct,
        })
    return results


# ── Nagatha Integration ──────────────────────────────────────────

def superconductor_properties(sc_key, T=0.0, sigma=0.0):
    """Export superconductor properties in Nagatha-compatible format.

    Args:
        sc_key: key into SUPERCONDUCTORS dict
        T: temperature (K)
        sigma: σ-field value

    Returns:
        Dict of superconducting properties
    """
    data = SUPERCONDUCTORS[sc_key]
    T_c = data['T_c_K']
    n_e = data['n_e_m3']
    v_F = data['v_F_m_s']
    sc_type = data['type']
    is_sc = data.get('is_superconductor', True)

    result = {
        'name': data['name'],
        'is_superconductor': is_sc,
        'suppression': data.get('suppression'),
        'sigma': sigma,
        'lambda_ep': data.get('lambda_ep'),
        'mu_star': data.get('mu_star'),
        'theta_D_K': data.get('theta_D_K'),
        'mcmillan_Tc_K': mcmillan_Tc_for(sc_key),
    }

    if not is_sc or T_c <= 0:
        # Non-superconducting metal — no BCS properties
        result['T_c_K'] = 0.0
        result['origin_tag'] = (
            "NON-SUPERCONDUCTOR: " + (data.get('suppression') or 'unknown') + ". "
            "MEASURED: λ_ep, μ*, Θ_D. "
            "McMillan prediction included for validation."
        )
        return result

    T_c_eff = sigma_Tc_shift(T_c, sigma) if sigma != 0.0 else T_c
    delta = bcs_gap_temperature(T_c_eff, T)
    lam_depth = london_penetration_at_T(n_e, T_c_eff, T) if T < T_c_eff else float('inf')
    xi = bcs_coherence_length(v_F, T_c_eff)
    kappa = gl_parameter_effective(sc_key)

    result.update({
        'type': sc_type,
        'T_c_K': T_c_eff,
        'gap_J': delta,
        'gap_meV': delta / (E_CHARGE * 1e-3) if delta > 0 else 0.0,
        'london_depth_m': lam_depth,
        'coherence_length_m': xi,
        'gl_parameter': kappa,
        'kappa_source': data.get('kappa_source', 'measured'),
        'pressure_GPa': data.get('pressure_GPa'),
        'meissner_fraction': meissner_fraction(T_c_eff, T),
    })

    H_c = thermodynamic_critical_field(n_e, T_c_eff, T)
    result['H_c_A_m'] = H_c

    if sc_type == 'II' or is_type_II(n_e, v_F, T_c_eff, sc_key=sc_key):
        result['H_c1_A_m'] = lower_critical_field(n_e, v_F, T_c_eff, T, sc_key=sc_key)
        result['H_c2_A_m'] = upper_critical_field(n_e, v_F, T_c_eff, T, sc_key=sc_key)

    result['depairing_J_c_A_m2'] = depairing_current_density(n_e, v_F, T_c_eff)

    result['origin_tag'] = (
        "FIRST_PRINCIPLES: BCS gap (weak-coupling, Δ=1.764kT_c). "
        "FIRST_PRINCIPLES: London penetration depth (London 1935). "
        "FIRST_PRINCIPLES: Pippard coherence length. "
        "FIRST_PRINCIPLES: Abrikosov critical fields (Type II). "
        "MEASURED: T_c, λ_ep, μ*, Θ_D. "
        "CORE: σ-dependence through Θ_D → T_c shift."
    )
    return result


# ── Block Cooling Simulation ──────────────────────────────────────

def block_cooling_profile(sc_key, T_start, T_end, steps,
                          rho_normal=1.0e-7, sigma=0.0):
    """Simulate cooling a block of superconductor from T_start to T_end.

    At each temperature step, computes:
      - resistivity: rho_normal above T_c, exactly 0.0 below
      - bcs_gap: Δ(T) in Joules (opens at T_c)
      - meissner_fraction: superconducting electron fraction
      - london_depth: penetration depth (inf above T_c, finite below)
      - H_c: thermodynamic critical field (0 above T_c)

    The superconducting transition is SHARP at T_c for DC resistance:
    any superconducting condensate fraction creates a zero-resistance
    channel in parallel with the normal electrons. Total DC resistance
    drops to exactly zero. This is not an approximation — it is the
    defining property of a superconductor.

    FIRST_PRINCIPLES: BCS + London + Gorter-Casimir two-fluid model.

    Args:
        sc_key: key into SUPERCONDUCTORS dict
        T_start: starting temperature (K), should be > T_c
        T_end: ending temperature (K), should be ≥ 0
        steps: number of temperature steps
        rho_normal: normal-state resistivity (Ω·m), default 1e-7
        sigma: σ-field value (shifts T_c)

    Returns:
        List of dicts, one per temperature step, with keys:
        T_K, resistivity, bcs_gap_J, meissner_frac, london_depth_m, H_c_A_m
    """
    data = SUPERCONDUCTORS[sc_key]
    T_c = sigma_Tc_shift(data['T_c_K'], sigma) if sigma != 0.0 else data['T_c_K']
    n_e = data['n_e_m3']
    v_F = data['v_F_m_s']

    profile = []
    for i in range(steps + 1):
        T = T_start - (T_start - T_end) * i / steps
        if T < 0:
            T = 0.0

        is_below_Tc = T < T_c

        rho = 0.0 if is_below_Tc else rho_normal
        gap = bcs_gap_temperature(T_c, T)
        frac = meissner_fraction(T_c, T)

        if is_below_Tc and T_c > 0:
            lam = london_penetration_at_T(n_e, T_c, T)
            H_c = thermodynamic_critical_field(n_e, T_c, T)
        else:
            lam = float('inf')
            H_c = 0.0

        profile.append({
            'T_K': T,
            'resistivity': rho,
            'bcs_gap_J': gap,
            'meissner_frac': frac,
            'london_depth_m': lam,
            'H_c_A_m': H_c,
        })

    return profile
