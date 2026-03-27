"""
Phase transition thermodynamics from measured and first-principles data.

Derivation chain:
  σ → nuclear mass → Debye temperature → melting point (Lindemann criterion)
  Clausius-Clapeyron: dT/dP = T_m × ΔV / L_fus (FIRST_PRINCIPLES)
  Lindemann melting: T_m ≈ C × M × Θ_D² × a² / k_B (FIRST_PRINCIPLES estimate)
  Richard's rule: ΔS_fus = L_fus / T_m ≈ R for metals (MEASURED trend)

The core physics:
  1. Melting points, latent heats, and volume changes are MEASURED.
     All values from CRC Handbook of Chemistry and Physics.

  2. Clausius-Clapeyron slope
     dT/dP = T_m × ΔV_fus / L_fus
     FIRST_PRINCIPLES: exact thermodynamic identity from the
     Gibbs free energy condition for coexistence (dG_solid = dG_liquid).
     At the melting point, ΔG = 0, and dT/dP = ΔV/ΔS = T_m ΔV/L_fus.
     Pressure dependence is linear in the first approximation.

  3. Lindemann melting criterion (1910)
     When the root-mean-square atomic displacement exceeds ~10% of
     the lattice parameter, the lattice becomes unstable and melts.
     At the Debye temperature scale:
       <u²> ≈ 3 k_B T / (M ω_D²) = 3 k_B T a² / (M v_s²)
     Setting <u²>^(1/2) = c_L × a at T = T_m:
       T_m = C × M × ω_D² × a² / (3 k_B)
     where C ≈ 0.0032 is the Lindemann constant (empirical).
     This gives estimates within a factor of 2 for most metals.
     FIRST_PRINCIPLES: harmonic lattice theory. APPROXIMATION: C is empirical.

  4. Richard's rule
     For metals, the entropy of fusion ΔS = L_fus / T_m ≈ R ≈ 8.314 J/(mol·K).
     This empirical law (Richard 1897) holds because melting destroys
     approximately one degree of orientational freedom per atom.
     MEASURED trend; exact values deviate ±50% for real materials.

  5. σ-field melting shift
     The Debye temperature Θ_D shifts with nuclear mass (through bulk modulus),
     and Lindemann predicts T_m ∝ Θ_D².
     In thermal.py, Θ_D depends on bulk_modulus(σ) and FIXED density.
     Higher σ → heavier nuclei → slightly higher K (via E_coh QCD fraction)
     → Θ_D increases slightly → T_m increases with σ.
     The effect is tiny: at σ=1, T_m shifts by ~0.1%.
     CORE: through □σ = −ξR → nuclear mass → E_coh → K → Θ_D → T_m.

σ-dependence:
  σ → m_nucleus → K(E_coh) → v_s → Θ_D → T_m(σ)
  In thermal.py, K shifts through E_coh (small QCD fraction ~1%) while
  density is held fixed. Result: Θ_D increases slightly with σ, so T_m
  increases slightly. The shift is tiny: < 0.1% per unit σ.

Origin tags:
  - PHASE_DATA: MEASURED (CRC Handbook, 103rd edition)
  - clausius_clapeyron_slope: FIRST_PRINCIPLES (Gibbs coexistence condition)
  - melting_point_at_pressure: FIRST_PRINCIPLES (linear Clausius-Clapeyron)
  - lindemann_melting_estimate: FIRST_PRINCIPLES (harmonic lattice) +
    APPROXIMATION (empirical Lindemann constant C ≈ 0.0032)
  - latent_heat_ratio: MEASURED validation
  - entropy_of_fusion: FIRST_PRINCIPLES (ΔS = Q_rev/T)
  - sigma_melting_shift: CORE (□σ = −ξR through Debye temperature)
  - phase_transition_properties: Nagatha export
"""

import math
from .surface import MATERIALS
from .thermal import debye_temperature
from .mechanical import _number_density

# ── Constants ─────────────────────────────────────────────────────
_K_BOLTZMANN = 1.380649e-23     # J/K (exact, 2019 SI)
_HBAR = 1.054571817e-34         # J·s (exact, 2019 SI)
_EV_TO_JOULE = 1.602176634e-19  # exact
_AMU_KG = 1.66053906660e-27     # atomic mass unit in kg
_R_GAS = 8.314462618            # J/(mol·K) (CODATA 2018)

# Lindemann constant for the formula T_m = C × M × ω_D² / (k_B × n^(2/3)).
# This form uses the atomic volume length scale n^(-1/3) (atoms/m³ → m per atom)
# rather than the conventional lattice parameter. It is more accurate because
# the Lindemann displacement threshold is properly normalised by the
# inter-atomic spacing regardless of crystal structure.
#
# Calibration: best fit over all 8 materials in PHASE_DATA.
# For metals, estimates are within ±44% of measured T_m.
# Silicon (covalent) is ~46% low — Lindemann is less accurate for non-metals.
# APPROXIMATION: C is empirical (Gilvarry 1956 analysis with our Θ_D values).
_LINDEMANN_C = 0.00075


# ── Phase Transition Data ──────────────────────────────────────────
# All values MEASURED from experiment.
# Source: CRC Handbook of Chemistry and Physics, 103rd edition.
#
# T_melt_K         : melting point at 1 atm (K)
# T_boil_K         : boiling point at 1 atm (K)
# L_fus_J_mol      : latent heat of fusion (J/mol)
# L_vap_J_mol      : latent heat of vaporization (J/mol)
# delta_V_fus      : fractional volume change on melting ΔV/V_solid
#                    (dimensionless; positive = expands on melting)
# dT_dP_melt_K_GPa : Clausius-Clapeyron slope, MEASURED (K/GPa)
#                    where available; None means use computed value.
#
# Notes on delta_V_fus:
#   Most metals expand on melting (positive). The fractional change
#   is typically 2-6%. Bismuth contracts (anomalous), but it is not
#   in our set. Silicon also contracts but is listed here as positive
#   based on density measurements for liquid Si.
#
# Notes on dT_dP_melt_K_GPa:
#   We include measured slopes where reliably known.
#   These can be cross-checked against clausius_clapeyron_slope().

PHASE_DATA = {
    'iron': {
        # BCC α-iron → liquid at 1811 K
        'T_melt_K':         1811.0,
        'T_boil_K':         3134.0,
        'L_fus_J_mol':      13810.0,   # 13.81 kJ/mol
        'L_vap_J_mol':      340000.0,  # 340 kJ/mol
        'delta_V_fus':      0.0336,    # ~3.4% volume increase
        'dT_dP_melt_K_GPa': 3.4,       # MEASURED: ~3.4 K/GPa at low P
    },
    'copper': {
        'T_melt_K':         1357.77,
        'T_boil_K':         2835.0,
        'L_fus_J_mol':      13050.0,   # 13.05 kJ/mol
        'L_vap_J_mol':      300400.0,  # 300.4 kJ/mol
        'delta_V_fus':      0.0424,    # ~4.2% volume increase
        'dT_dP_melt_K_GPa': 4.4,       # MEASURED
    },
    'aluminum': {
        'T_melt_K':         933.47,
        'T_boil_K':         2792.0,
        'L_fus_J_mol':      10710.0,   # 10.71 kJ/mol
        'L_vap_J_mol':      284000.0,  # 284 kJ/mol
        'delta_V_fus':      0.0620,    # ~6.2% volume increase
        'dT_dP_melt_K_GPa': 6.8,       # MEASURED
    },
    'gold': {
        'T_melt_K':         1337.33,
        'T_boil_K':         3129.0,
        'L_fus_J_mol':      12550.0,   # 12.55 kJ/mol
        'L_vap_J_mol':      330000.0,  # 330 kJ/mol
        'delta_V_fus':      0.0507,    # ~5.1% volume increase
        'dT_dP_melt_K_GPa': 5.8,       # MEASURED
    },
    'silicon': {
        # Silicon is anomalous: liquid Si is metallic and DENSER than solid Si.
        # rho_solid(Si, at T_m) ~ 2330 kg/m3, rho_liquid(Si) ~ 2570 kg/m3
        # delta_V/V = rho_solid/rho_liquid - 1 = 2330/2570 - 1 ≈ -0.093
        # This negative ΔV gives a negative Clausius-Clapeyron slope:
        # pressure LOWERS the melting point (anomalous, like water or Bi).
        'T_melt_K':         1687.0,
        'T_boil_K':         3538.0,
        'L_fus_J_mol':      50210.0,   # 50.21 kJ/mol (covalent lattice, very high)
        'L_vap_J_mol':      359000.0,  # 359 kJ/mol
        'delta_V_fus':      -0.093,    # MEASURED: ~-9.3% (contracts on melting)
        'dT_dP_melt_K_GPa': -45.0,     # MEASURED: anomalous negative slope
    },
    'tungsten': {
        'T_melt_K':         3695.0,
        'T_boil_K':         5828.0,
        'L_fus_J_mol':      52310.0,   # 52.31 kJ/mol (highest of pure metals)
        'L_vap_J_mol':      774000.0,  # 774 kJ/mol
        'delta_V_fus':      0.0291,    # ~2.9% volume increase
        'dT_dP_melt_K_GPa': 5.1,       # MEASURED/ESTIMATED
    },
    'nickel': {
        'T_melt_K':         1728.0,
        'T_boil_K':         3186.0,
        'L_fus_J_mol':      17150.0,   # 17.15 kJ/mol
        'L_vap_J_mol':      370400.0,  # 370.4 kJ/mol
        'delta_V_fus':      0.0359,    # ~3.6% volume increase
        'dT_dP_melt_K_GPa': 3.9,       # MEASURED
    },
    'titanium': {
        'T_melt_K':         1941.0,
        'T_boil_K':         3560.0,
        'L_fus_J_mol':      14150.0,   # 14.15 kJ/mol
        'L_vap_J_mol':      421000.0,  # 421 kJ/mol
        'delta_V_fus':      0.0285,    # ~2.9% volume increase
        'dT_dP_melt_K_GPa': 3.3,       # ESTIMATED from Clausius-Clapeyron
    },
}


# ── Clausius-Clapeyron Slope ──────────────────────────────────────

def clausius_clapeyron_slope(material_key):
    """Clausius-Clapeyron slope dT/dP at the melting point (K/Pa).

    dT/dP = T_m × ΔV_fus / L_fus

    FIRST_PRINCIPLES: exact thermodynamic identity from Gibbs coexistence.
    At the solid-liquid boundary, ΔG = 0 and d(ΔG)/dP = 0 simultaneously.
    This gives dP/dT = ΔS/ΔV = L_fus / (T_m × ΔV_fus).

    ΔV_fus = delta_V_fus × V_molar (fractional × molar volume)
    V_molar = M / ρ  (molar volume of solid)

    Units:
      T_m in K, ΔV in m³/mol, L_fus in J/mol
      → dT/dP in K/Pa

    For metals, typical values are 1-10 K/GPa = 1-10 × 10⁻⁹ K/Pa.

    Args:
        material_key: key into PHASE_DATA

    Returns:
        dT/dP in K/Pa (scalar; negative if volume contracts on melting).
    """
    data = PHASE_DATA[material_key]
    mat = MATERIALS[material_key]

    T_m = data['T_melt_K']
    L_fus = data['L_fus_J_mol']           # J/mol
    delta_V_frac = data['delta_V_fus']     # dimensionless

    # Molar volume of solid: V_molar = M × _AMU_KG × N_Avogadro / ρ
    # = (A g/mol × 10⁻³ kg/g) / ρ = A × 10⁻³ / ρ  (m³/mol)
    A = mat['A']                           # atomic mass (amu → essentially g/mol)
    rho = mat['density_kg_m3']
    V_molar = (A * 1e-3) / rho            # m³/mol

    # Absolute volume change on melting
    delta_V = delta_V_frac * V_molar      # m³/mol

    # Clausius-Clapeyron: dT/dP = T_m × ΔV / L_fus
    if L_fus == 0:
        return 0.0
    return T_m * delta_V / L_fus          # K/Pa


# ── Melting Point at Pressure ─────────────────────────────────────

def melting_point_at_pressure(material_key, pressure_Pa):
    """Melting temperature as a function of pressure (K).

    T_m(P) = T_m(0) + (dT/dP) × P

    FIRST_PRINCIPLES: linear Clausius-Clapeyron integration.
    Valid for moderate pressures (P << phase-transition boundary).
    For most metals, the approximation holds to ~10-20 GPa.
    Above that, high-pressure phases (bcc→hcp etc.) require corrections.

    Args:
        material_key: key into PHASE_DATA
        pressure_Pa: applied pressure in Pa

    Returns:
        Melting temperature in K.
    """
    data = PHASE_DATA[material_key]
    T_m0 = data['T_melt_K']
    slope = clausius_clapeyron_slope(material_key)   # K/Pa
    return T_m0 + slope * pressure_Pa


# ── Lindemann Melting Estimate ────────────────────────────────────

def lindemann_melting_estimate(material_key, sigma=0.0):
    """Estimate melting temperature via the Lindemann criterion (K).

    T_m ≈ C × M × ω_D² / (k_B × n^(2/3))

    Where:
      C = 0.00075 (Lindemann constant, APPROXIMATION, calibrated for these materials)
      M = atomic mass in kg
      ω_D = Debye angular frequency = k_B Θ_D / ℏ  (rad/s)
      Θ_D = Debye temperature from thermal.py (K)
      n = atomic number density (atoms/m³) from mechanical.py
      k_B = Boltzmann constant

    FIRST_PRINCIPLES derivation:
      In the harmonic lattice, the mean-square displacement is
        <u²> = 3 k_B T / (M ω_D²)
      The Lindemann criterion states melting occurs when
        <u²>^(1/2) = f_L × n^(-1/3)  (f_L ≈ 0.047, inter-atomic spacing)
      Setting T = T_m:
        T_m = f_L² × M × ω_D² / (3 k_B × n^(2/3))
      Grouping constants: C = f_L² / 3 ≈ 0.00075 (empirical for our Θ_D values).

    The length scale n^(-1/3) (atomic volume length) is preferred over the
    conventional lattice parameter because it is structure-independent.

    APPROXIMATION: C is calibrated empirically to our computed Θ_D values
    (which are ~30-65% higher than literature values for most metals due to
    the harmonic-approximation K in mechanical.py). Estimates are within
    a factor of 2 for all materials; silicon (covalent) is at the lower edge.

    σ-dependence: enters through Θ_D(σ) from thermal.py.
    Higher σ → heavier nuclei → Θ_D decreases → T_m decreases.

    Args:
        material_key: key into MATERIALS
        sigma: σ-field value

    Returns:
        Estimated melting temperature in K.
    """
    mat = MATERIALS[material_key]
    A = mat['A']                                    # atomic mass in amu
    M_kg = A * _AMU_KG                              # kg per atom

    theta_D = debye_temperature(material_key, sigma)  # K

    # Debye angular frequency: ω_D = k_B × Θ_D / ℏ  (rad/s)
    omega_D = _K_BOLTZMANN * theta_D / _HBAR

    # Number density: n = ρ × N_A / M_molar = ρ / (A × m_amu) (atoms/m³)
    n = _number_density(material_key)

    # Length scale: n^(-1/3) = inter-atomic spacing from atomic volume.
    # This is more structure-independent than the conventional lattice parameter.
    # The Lindemann displacement threshold is then <u>^(1/2) = f_L × n^(-1/3).

    # Lindemann formula: T_m = C × M × ω_D² / (k_B × n^(2/3))
    # From: <u²> = 3 k_B T / (M ω_D²) at T = T_m
    # with <u>^(1/2) = f_L × n^(-1/3) and C = f_L² / 3 ≈ 0.00075
    T_m_est = _LINDEMANN_C * M_kg * omega_D**2 / (_K_BOLTZMANN * n**(2.0 / 3.0))
    return T_m_est


# ── Latent Heat Ratio ─────────────────────────────────────────────

def latent_heat_ratio(material_key):
    """Ratio L_vap / L_fus (dimensionless).

    For metals, this ratio is typically 10-30. This large ratio
    reflects that vaporization requires complete separation of atoms
    (breaking all bonds) while melting only destroys long-range order.

    MEASURED validation check: if the ratio is outside 5-50, the
    data or physics is suspect.

    Trouton's rule (for liquids): L_vap / T_boil ≈ 10.5 R (Trouton 1884).
    Richard's rule (for melting): L_fus / T_m ≈ R.
    Ratio: L_vap / L_fus ≈ (10.5 × T_boil) / (1 × T_m) ≈ 10-30.

    Args:
        material_key: key into PHASE_DATA

    Returns:
        L_vap / L_fus (dimensionless).
    """
    data = PHASE_DATA[material_key]
    L_fus = data['L_fus_J_mol']
    L_vap = data['L_vap_J_mol']
    if L_fus == 0:
        return float('inf')
    return L_vap / L_fus


# ── Entropy of Fusion ─────────────────────────────────────────────

def entropy_of_fusion(material_key):
    """Molar entropy of fusion ΔS_fus = L_fus / T_m (J/(mol·K)).

    FIRST_PRINCIPLES: entropy change at a reversible phase transition
    equals the latent heat divided by the transition temperature.
    ΔS = Q_rev / T = L_fus / T_m

    Richard's rule (1897): for simple metals, ΔS ≈ R ≈ 8.314 J/(mol·K).
    This is because melting destroys ~1 vibrational degree of freedom
    per atom, contributing k_B per atom = R per mole.

    The dimensionless ratio ΔS/R should be ~1 for simple metals,
    up to ~2 for complex structures (silicon, etc.).

    Args:
        material_key: key into PHASE_DATA

    Returns:
        ΔS in J/(mol·K).
    """
    data = PHASE_DATA[material_key]
    L_fus = data['L_fus_J_mol']
    T_m = data['T_melt_K']
    if T_m == 0:
        return 0.0
    return L_fus / T_m


# ── σ-field Melting Shift ─────────────────────────────────────────

def sigma_melting_shift(material_key, sigma):
    """Melting point shift due to σ-field, relative to σ=0.

    The Lindemann criterion predicts T_m ∝ Θ_D².
    Θ_D shifts with nuclear mass through the Debye formula in thermal.py.

    T_m(σ) / T_m(0) ≈ [Θ_D(σ) / Θ_D(0)]²

    We use this ratio to scale the MEASURED melting point.
    This is more reliable than the absolute Lindemann estimate because
    it cancels the empirical constant C.

    CORE: σ → nuclear mass → K(E_coh) → v_s → Θ_D → T_m.
    In thermal.py, Θ_D uses bulk_modulus(σ) with fixed density.
    K increases slightly with σ through the E_coh QCD scaling fraction,
    so Θ_D and T_m increase slightly with σ.

    At Earth σ (~7×10⁻¹⁰): shift is negligible (<10⁻⁹ fractional).
    At σ=1: T_m increases by ~0.1% (small correction).

    Args:
        material_key: key into PHASE_DATA / MATERIALS
        sigma: σ-field value

    Returns:
        T_m(σ) in K — the σ-shifted melting temperature.
    """
    data = PHASE_DATA[material_key]
    T_m0 = data['T_melt_K']

    theta_0 = debye_temperature(material_key, sigma=0.0)
    theta_s = debye_temperature(material_key, sigma=sigma)

    if theta_0 == 0:
        return T_m0

    # Ratio of Debye temperatures squared
    ratio = (theta_s / theta_0) ** 2
    return T_m0 * ratio


# ── Nagatha Export ────────────────────────────────────────────────

def phase_transition_properties(material_key, P=0.0, sigma=0.0):
    """Export phase transition properties in Nagatha-compatible format.

    Returns a dict with all phase transition quantities at given pressure
    and σ-field value. Suitable for merging into Nagatha's material database.

    Args:
        material_key: key into PHASE_DATA
        P: pressure in Pa (default 0 = 1 atm, i.e., gauge pressure = 0)
        sigma: σ-field value

    Returns:
        Dict with all phase transition properties and origin tags.
    """
    data = PHASE_DATA[material_key]

    T_m_base = data['T_melt_K']
    T_m_P = melting_point_at_pressure(material_key, P)
    T_m_sigma = sigma_melting_shift(material_key, sigma)

    slope_K_Pa = clausius_clapeyron_slope(material_key)
    slope_K_GPa = slope_K_Pa * 1e9

    T_m_lindemann = lindemann_melting_estimate(material_key, sigma)
    L_ratio = latent_heat_ratio(material_key)
    delta_S = entropy_of_fusion(material_key)
    delta_S_over_R = delta_S / _R_GAS

    theta_D = debye_temperature(material_key, sigma)

    return {
        'material':                     material_key,
        'pressure_Pa':                  P,
        'sigma':                        sigma,
        # Measured reference values
        'T_melt_K':                     T_m_base,
        'T_boil_K':                     data['T_boil_K'],
        'L_fus_J_mol':                  data['L_fus_J_mol'],
        'L_vap_J_mol':                  data['L_vap_J_mol'],
        'delta_V_fus':                  data['delta_V_fus'],
        'dT_dP_melt_K_GPa_measured':    data['dT_dP_melt_K_GPa'],
        # Derived quantities
        'dT_dP_melt_K_Pa':              slope_K_Pa,
        'dT_dP_melt_K_GPa_derived':     slope_K_GPa,
        'T_melt_at_P_K':                T_m_P,
        'T_melt_lindemann_K':           T_m_lindemann,
        'T_melt_sigma_K':               T_m_sigma,
        'latent_heat_ratio':            L_ratio,
        'entropy_of_fusion_J_molK':     delta_S,
        'entropy_of_fusion_over_R':     delta_S_over_R,
        'debye_temperature_K':          theta_D,
        'origin': (
            "T_melt_K, T_boil_K, L_fus_J_mol, L_vap_J_mol, "
            "delta_V_fus: MEASURED (CRC Handbook, 103rd edition). "
            "clausius_clapeyron_slope: FIRST_PRINCIPLES (Gibbs coexistence dT/dP = TΔV/L). "
            "melting_point_at_pressure: FIRST_PRINCIPLES (linear Clausius-Clapeyron). "
            "lindemann_melting_estimate: FIRST_PRINCIPLES (harmonic lattice <u²>=f_L²a²) + "
            "APPROXIMATION (Lindemann constant C=0.0032 is empirical). "
            "entropy_of_fusion: FIRST_PRINCIPLES (ΔS=L/T, reversible transition). "
            "sigma_melting_shift: CORE (□σ=−ξR → mass → Debye temp → T_m ∝ Θ_D²)."
        ),
    }
