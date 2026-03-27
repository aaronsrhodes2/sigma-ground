"""
Thermal expansion from the Grüneisen framework.

Derivation chain:
  Grüneisen parameter γ (MEASURED) + bulk modulus K (from mechanical.py)
  + molar heat capacity C_v (Debye model or MEASURED)
  + molar volume V_m (from density and atomic mass)
  → linear thermal expansion coefficient α (1/K) via:

      α = γ C_v / (3 V_m K)   [Grüneisen relation, 1926]

  where:
    γ  = dimensionless Grüneisen parameter (MEASURED)
    C_v = molar heat capacity (J/(mol·K))
    V_m = molar volume (m³/mol)
    K  = bulk modulus (Pa)

  Factor of 3: because α is LINEAR while the Grüneisen relation is
  thermodynamically for VOLUMETRIC expansion β = 3α for isotropic solids.

  The Grüneisen relation connects:
    - how phonon frequencies shift under compression (anharmonicity → γ)
    - how fast the lattice vibrates (C_v)
    - how stiff the lattice is (K)

  Together these determine how much the equilibrium lattice spacing grows
  with temperature.

Temperature dependence:
  α(T) ∝ C_v(T)/C_v(300) × α(300)

  At low T, C_v → 0 as T³ (Debye law), so α → 0 as well.
  At high T, C_v saturates at 3R (Dulong-Petit), so α saturates too.

  The Debye integral for molar heat capacity:
    C_v = 9R (T/Θ_D)³ ∫₀^{Θ_D/T} x⁴ eˣ/(eˣ-1)² dx

  Computed here with Simpson's rule (200 steps). Integrand guarded for
  large x (eˣ overflow).

σ-field dependence:
  Θ_D(σ) shifts through nuclear mass → phonon frequency changes.
  K(σ)   shifts through cohesive energy → lattice stiffness changes.
  α(σ) = γ C_v(Θ_D(σ), T) / (3 V_m K(σ))

  At Earth (σ ~ 7×10⁻¹⁰): negligible shift (<10⁻⁹).
  At σ > 0.1: measurable shifts in expansion coefficients.

Origin tags:
  - α MEASURED values: CRC Handbook of Chemistry and Physics, 97th ed.
  - Grüneisen parameters: MEASURED (CRC Handbook)
  - Molar heat capacity at 300K: MEASURED (CRC Handbook)
  - Grüneisen relation: FIRST_PRINCIPLES (thermodynamic identity, 1926)
  - β = 3α: FIRST_PRINCIPLES (isotropic solid geometry)
  - Temperature dependence: FIRST_PRINCIPLES (Debye model)
  - σ-dependence: CORE (through □σ = −ξR via K and Θ_D)
"""

import math
from .surface import MATERIALS
from .mechanical import bulk_modulus, youngs_modulus
from .mechanical import MECHANICAL_DATA
from .thermal import debye_temperature

# ── Physical constants ────────────────────────────────────────────
_R_GAS = 8.314462618       # J/(mol·K) — universal gas constant (exact)
_AMU_KG = 1.66053906660e-27  # kg/amu


# ── Expansion Data ────────────────────────────────────────────────
# All values MEASURED at 300 K.
# Sources: CRC Handbook of Chemistry and Physics, 97th edition.
#
# alpha_linear_per_K: linear thermal expansion coefficient (1/K)
# gruneisen_gamma: Grüneisen parameter γ (dimensionless)
# C_v_J_mol_K: molar heat capacity at constant volume (J/(mol·K))
#
# Rule 9: every field present for every material.

EXPANSION_DATA = {
    'iron': {
        'alpha_linear_per_K': 11.8e-6,   # MEASURED (CRC)
        'gruneisen_gamma':     1.7,       # MEASURED (CRC)
        'C_v_J_mol_K':         25.1,      # MEASURED (CRC)
    },
    'copper': {
        'alpha_linear_per_K': 16.5e-6,   # MEASURED (CRC)
        'gruneisen_gamma':     2.0,       # MEASURED (CRC)
        'C_v_J_mol_K':         24.4,      # MEASURED (CRC)
    },
    'aluminum': {
        'alpha_linear_per_K': 23.1e-6,   # MEASURED (CRC)
        'gruneisen_gamma':     2.2,       # MEASURED (CRC)
        'C_v_J_mol_K':         24.2,      # MEASURED (CRC)
    },
    'gold': {
        'alpha_linear_per_K': 14.2e-6,   # MEASURED (CRC)
        'gruneisen_gamma':     3.0,       # MEASURED (CRC)
        'C_v_J_mol_K':         25.4,      # MEASURED (CRC)
    },
    'silicon': {
        'alpha_linear_per_K': 2.6e-6,    # MEASURED (CRC)
        'gruneisen_gamma':     0.56,      # MEASURED (CRC) — anomalously low
        'C_v_J_mol_K':         20.0,      # MEASURED (CRC)
    },
    'tungsten': {
        'alpha_linear_per_K': 4.5e-6,    # MEASURED (CRC)
        'gruneisen_gamma':     1.6,       # MEASURED (CRC)
        'C_v_J_mol_K':         24.3,      # MEASURED (CRC)
    },
    'nickel': {
        'alpha_linear_per_K': 13.4e-6,   # MEASURED (CRC)
        'gruneisen_gamma':     1.9,       # MEASURED (CRC)
        'C_v_J_mol_K':         26.1,      # MEASURED (CRC)
    },
    'titanium': {
        'alpha_linear_per_K': 8.6e-6,    # MEASURED (CRC)
        'gruneisen_gamma':     1.2,       # MEASURED (CRC)
        'C_v_J_mol_K':         25.1,      # MEASURED (CRC)
    },
}


# ── Debye integral (Simpson's rule) ──────────────────────────────

def _debye_integrand(x):
    """Integrand for the Debye heat capacity: x⁴ eˣ / (eˣ - 1)².

    Guarded against overflow for large x.
    For x > 500: eˣ overflows; integrand → 0 exponentially.
    Use identity: x⁴ eˣ/(eˣ-1)² = x⁴/(2 sinh(x/2))² × (x/2)² / (x/2)²
    Simpler guard: if x > 500, return 0.0 directly.
    """
    if x <= 0.0:
        return 0.0
    if x > 500.0:
        return 0.0
    ex = math.exp(x)
    denom = (ex - 1.0) ** 2
    if denom == 0.0:
        return 0.0
    return (x ** 4) * ex / denom


def _debye_cv_molar(T, theta_D, n_steps=200):
    """Molar heat capacity C_v (J/(mol·K)) from the Debye model.

    C_v = 9R (T/Θ_D)³ ∫₀^{Θ_D/T} x⁴ eˣ/(eˣ-1)² dx

    FIRST_PRINCIPLES: quantum statistical mechanics of a phonon gas
    with a Debye density of states.

    The integral is evaluated numerically using Simpson's rule
    with n_steps intervals. n_steps must be even.

    Limits:
      High T (T >> Θ_D): C_v → 3R  (Dulong-Petit)
      Low  T (T << Θ_D): C_v → 12π⁴R/5 × (T/Θ_D)³

    Args:
        T:       temperature in Kelvin
        theta_D: Debye temperature in Kelvin
        n_steps: number of Simpson integration steps (must be even)

    Returns:
        Molar heat capacity in J/(mol·K).
    """
    if T <= 0.0:
        return 0.0
    if theta_D <= 0.0:
        return 3.0 * _R_GAS  # fallback: Dulong-Petit

    upper = theta_D / T  # upper limit of integration

    # Ensure n_steps is even for Simpson's rule
    if n_steps % 2 != 0:
        n_steps += 1

    h = upper / n_steps

    # Simpson's rule: (h/3)[f(x0) + 4f(x1) + 2f(x2) + 4f(x3) + ... + f(xN)]
    integral = _debye_integrand(0.0) + _debye_integrand(upper)
    for i in range(1, n_steps):
        x = i * h
        coeff = 4.0 if (i % 2 == 1) else 2.0
        integral += coeff * _debye_integrand(x)
    integral *= h / 3.0

    prefactor = 9.0 * _R_GAS * (T / theta_D) ** 3
    return prefactor * integral


def _molar_volume(material_key):
    """Molar volume V_m (m³/mol) from atomic mass and density.

    V_m = (A × 1e-3) / ρ

    where A is atomic mass in g/mol (numerically equal to mass number
    for our purposes) and ρ is density in kg/m³.

    FIRST_PRINCIPLES: mass balance.

    Args:
        material_key: key into MATERIALS dict

    Returns:
        Molar volume in m³/mol.
    """
    mat = MATERIALS[material_key]
    A = mat['A']            # atomic mass number (≈ g/mol)
    rho = mat['density_kg_m3']
    return (A * 1e-3) / rho  # (g/mol × 1e-3 kg/g) / (kg/m³) = m³/mol


# ── Public functions ──────────────────────────────────────────────

def linear_expansion_coefficient(material_key):
    """Linear thermal expansion coefficient α (1/K) at 300 K.

    MEASURED: from CRC Handbook of Chemistry and Physics.

    α is the fractional change in length per degree Kelvin:
      ΔL/L = α × ΔT

    Args:
        material_key: key into EXPANSION_DATA

    Returns:
        α in 1/K (order of magnitude: 1e-6 to 25e-6 for metals)
    """
    return EXPANSION_DATA[material_key]['alpha_linear_per_K']


def volumetric_expansion_coefficient(material_key):
    """Volumetric (cubic) thermal expansion coefficient β (1/K) at 300 K.

    β = 3α

    FIRST_PRINCIPLES: for an isotropic solid that expands equally in
    all three directions, the volumetric strain is:
      ΔV/V = (1 + α ΔT)³ - 1 ≈ 3α ΔT  (for α ΔT << 1)

    Hence β = 3α exactly in the isotropic, small-strain limit.

    Args:
        material_key: key into EXPANSION_DATA

    Returns:
        β in 1/K
    """
    return 3.0 * linear_expansion_coefficient(material_key)


def gruneisen_parameter(material_key):
    """Grüneisen parameter γ (dimensionless).

    MEASURED: from CRC Handbook of Chemistry and Physics.

    γ quantifies anharmonicity of the interatomic potential. It
    describes how phonon frequencies shift under compression:
      γ = -∂ ln ω / ∂ ln V

    Typical range: 0.5 (silicon, unusual) to 3.0 (gold).
    Most metals: 1.5–2.5. Higher γ → faster expansion with temperature.

    Args:
        material_key: key into EXPANSION_DATA

    Returns:
        γ (dimensionless)
    """
    return EXPANSION_DATA[material_key]['gruneisen_gamma']


def gruneisen_relation(material_key, sigma=0.0):
    """Derive α from the Grüneisen relation (thermodynamic identity).

    α = γ C_v / (3 V_m K)

    FIRST_PRINCIPLES: Grüneisen (1926). This is an exact thermodynamic
    identity relating:
      - anharmonicity of the potential (γ)
      - thermal energy storage per degree (C_v)
      - stiffness of the lattice (K)
      - volume per mole of atoms (V_m)

    The factor of 3 converts volumetric → linear coefficient.

    Note: this derived α uses measured γ and C_v but the K from our
    harmonic-approximation bulk modulus (±50%). Expect the result to be
    within a factor of ~2 of the measured α.

    Args:
        material_key: key into EXPANSION_DATA / MATERIALS
        sigma:        σ-field value

    Returns:
        Derived α in 1/K
    """
    gamma = gruneisen_parameter(material_key)
    C_v = EXPANSION_DATA[material_key]['C_v_J_mol_K']   # J/(mol·K)
    V_m = _molar_volume(material_key)                    # m³/mol
    K = bulk_modulus(material_key, sigma)                # Pa = J/m³

    # α = γ C_v / (3 V_m K)
    # Units: [dimensionless × J/(mol·K)] / [m³/mol × J/m³]
    #      = J/(mol·K) / (J/mol) = 1/K  ✓
    return gamma * C_v / (3.0 * V_m * K)


def thermal_strain(material_key, delta_T):
    """Linear thermal strain ε = α × ΔT.

    FIRST_PRINCIPLES: definition of linear thermal expansion.
    For a rod constrained at neither end (free expansion):
      ε = ΔL/L = α × ΔT

    This is a dimensionless strain (positive for heating, negative
    for cooling).

    Args:
        material_key: key into EXPANSION_DATA
        delta_T:      temperature change ΔT in Kelvin (positive = heating)

    Returns:
        Linear strain ε (dimensionless)
    """
    alpha = linear_expansion_coefficient(material_key)
    return alpha * delta_T


def thermal_stress(material_key, delta_T, sigma=0.0):
    """Thermal stress σ_th (Pa) in a fully constrained body.

    σ_th = E × α × ΔT / (1 - 2ν)

    FIRST_PRINCIPLES: When a body is prevented from expanding (fully
    constrained in all three principal directions), the suppressed
    thermal strain must be balanced by an elastic stress. From
    three-dimensional Hooke's law for isotropic materials:

      σ_thermal = -E × ε_th / (1 - 2ν)
                = -E × α × ΔT / (1 - 2ν)

    The 1/(1-2ν) factor accounts for the lateral stress contributions
    (Poisson effect). For ν ≈ 0.3 (typical metal), 1-2ν ≈ 0.4.

    Sign convention: positive ΔT → constrained body wants to expand
    but cannot → compressive stress (returned as positive value,
    representing the magnitude of compression). Negative ΔT → tensile.

    Note: for a uniaxial constraint (one free direction), the formula
    is simply σ = E × α × ΔT. The three-dimensional form used here
    gives higher stress because all three directions are locked.

    Args:
        material_key: key into EXPANSION_DATA / MECHANICAL_DATA
        delta_T:      temperature change in Kelvin
        sigma:        σ-field value

    Returns:
        Thermal stress magnitude in Pa (positive = compressive for heating)
    """
    E = youngs_modulus(material_key, sigma)              # Pa
    alpha = linear_expansion_coefficient(material_key)   # 1/K
    nu = MECHANICAL_DATA[material_key]['poisson_ratio']  # dimensionless

    denom = 1.0 - 2.0 * nu
    if abs(denom) < 1e-12:
        # Pathological: ν = 0.5 (incompressible). Return uniaxial stress.
        return E * alpha * delta_T

    return E * alpha * delta_T / denom


def length_change(material_key, length_m, delta_T):
    """Change in length ΔL (m) for a bar heated by ΔT.

    ΔL = L × α × ΔT

    FIRST_PRINCIPLES: direct application of the definition of linear
    thermal expansion coefficient.

    Example: 1 m bar of aluminum (α = 23.1e-6 /K) heated 100 K:
      ΔL = 1.0 × 23.1e-6 × 100 = 2.31 mm

    Args:
        material_key: key into EXPANSION_DATA
        length_m:     original length in meters
        delta_T:      temperature change in Kelvin

    Returns:
        ΔL in meters (positive for heating, negative for cooling)
    """
    alpha = linear_expansion_coefficient(material_key)
    return length_m * alpha * delta_T


def volume_change(material_key, volume_m3, delta_T):
    """Change in volume ΔV (m³) for a body heated by ΔT.

    ΔV = V × β × ΔT  where β = 3α

    FIRST_PRINCIPLES: volumetric expansion from isotropic linear
    expansion. For small strains (α ΔT << 1):
      V(T + ΔT) = V₀ (1 + α ΔT)³ ≈ V₀ (1 + 3α ΔT)
      ΔV = V₀ × 3α × ΔT = V × β × ΔT

    Args:
        material_key: key into EXPANSION_DATA
        volume_m3:    original volume in m³
        delta_T:      temperature change in Kelvin

    Returns:
        ΔV in m³
    """
    beta = volumetric_expansion_coefficient(material_key)
    return volume_m3 * beta * delta_T


def expansion_coefficient_at_T(material_key, T, sigma=0.0):
    """Temperature-dependent linear expansion coefficient α(T) (1/K).

    α(T) = α(300) × C_v(T) / C_v(300)

    FIRST_PRINCIPLES: from the Grüneisen relation α = γ C_v / (3 V_m K),
    γ, V_m, and K are approximately temperature-independent near 300 K,
    so α scales with C_v(T). The heat capacity is computed from the
    full Debye model using numerical integration (Simpson's rule,
    200 steps).

    Debye model for C_v(T):
      C_v = 9R(T/Θ_D)³ ∫₀^{Θ_D/T} x⁴ eˣ/(eˣ-1)² dx

    Limits:
      T → 0:    C_v ∝ T³ → α(T) → 0 (third law of thermodynamics)
      T >> Θ_D: C_v → 3R → α(T) → α(300) (approximately, once
                C_v has saturated)

    The Debye temperature Θ_D is computed from the σ-corrected bulk
    modulus and number density (via thermal.py).

    Args:
        material_key: key into EXPANSION_DATA
        T:            temperature in Kelvin
        sigma:        σ-field value

    Returns:
        α(T) in 1/K
    """
    if T <= 0.0:
        return 0.0

    theta_D = debye_temperature(material_key, sigma)

    cv_T = _debye_cv_molar(T, theta_D)
    cv_300 = _debye_cv_molar(300.0, theta_D)

    if cv_300 <= 0.0:
        return linear_expansion_coefficient(material_key)

    alpha_300 = linear_expansion_coefficient(material_key)
    return alpha_300 * (cv_T / cv_300)


def sigma_expansion_shift(material_key, sigma):
    """Thermal expansion coefficient α shifted by σ-field.

    α(σ) = γ C_v(Θ_D(σ), 300) / (3 V_m K(σ))

    CORE: the σ-field enters through two channels:
      1. K(σ): bulk modulus stiffens as nuclear mass increases
         (QCD mass fraction of nucleon mass scales as e^σ,
          stiffer lattice → lower α).
      2. Θ_D(σ): Debye temperature shifts through v_sound = √(K/ρ),
         changing C_v(300K) relative to the high-T limit.

    At Earth (σ ~ 7×10⁻¹⁰): shift < 10⁻⁹, negligible.
    At σ = 0: returns the Grüneisen-derived α (not the MEASURED α).

    Derivation chain:
      σ → nuclear mass ratio → K(σ) [from mechanical.py]
        → Θ_D(σ) [from thermal.py]
        → C_v(300K, Θ_D(σ)) [Debye model, this file]
        → α(σ) [Grüneisen relation]

    Args:
        material_key: key into EXPANSION_DATA
        sigma:        σ-field value (dimensionless)

    Returns:
        α(σ) in 1/K derived via Grüneisen + Debye with σ-shifted K and Θ_D
    """
    gamma = gruneisen_parameter(material_key)
    V_m = _molar_volume(material_key)
    K_sigma = bulk_modulus(material_key, sigma)
    theta_D_sigma = debye_temperature(material_key, sigma)

    cv_300_sigma = _debye_cv_molar(300.0, theta_D_sigma)

    return gamma * cv_300_sigma / (3.0 * V_m * K_sigma)


def thermal_expansion_properties(material_key, delta_T=0.0, T=300.0, sigma=0.0):
    """Nagatha export: all thermal expansion properties for a material.

    Returns a dict compatible with Nagatha's material database.
    Covers all functions in this module.

    Args:
        material_key: key into EXPANSION_DATA
        delta_T:      temperature change ΔT in Kelvin (for stress/strain)
        T:            absolute temperature in Kelvin
        sigma:        σ-field value

    Returns:
        Dict with all thermal expansion quantities and origin metadata.
    """
    alpha = linear_expansion_coefficient(material_key)
    beta = volumetric_expansion_coefficient(material_key)
    gamma = gruneisen_parameter(material_key)
    alpha_derived = gruneisen_relation(material_key, sigma)
    strain = thermal_strain(material_key, delta_T)
    stress = thermal_stress(material_key, delta_T, sigma)
    dl = length_change(material_key, 1.0, delta_T)
    dv = volume_change(material_key, 1.0, delta_T)
    alpha_T = expansion_coefficient_at_T(material_key, T, sigma)
    alpha_sigma = sigma_expansion_shift(material_key, sigma)

    return {
        'material': material_key,
        'temperature_K': T,
        'delta_T_K': delta_T,
        'sigma': sigma,
        # MEASURED
        'alpha_linear_per_K': alpha,
        'gruneisen_gamma': gamma,
        'C_v_J_mol_K': EXPANSION_DATA[material_key]['C_v_J_mol_K'],
        # FIRST_PRINCIPLES
        'beta_volumetric_per_K': beta,
        'alpha_derived_gruneisen': alpha_derived,
        'thermal_strain': strain,
        'thermal_stress_pa': stress,
        'length_change_per_meter_m': dl,
        'volume_change_per_m3_m3': dv,
        'alpha_at_T': alpha_T,
        'alpha_sigma_shifted': alpha_sigma,
        'origin': (
            "alpha_linear: MEASURED (CRC Handbook, 300 K). "
            "beta = 3*alpha: FIRST_PRINCIPLES (isotropic solid geometry). "
            "gruneisen_gamma: MEASURED (CRC Handbook). "
            "Grüneisen relation α = γCv/(3VmK): FIRST_PRINCIPLES (1926). "
            "thermal_strain = α ΔT: FIRST_PRINCIPLES (definition). "
            "thermal_stress = E α ΔT/(1-2ν): FIRST_PRINCIPLES (3D Hooke's law). "
            "expansion_coefficient_at_T: FIRST_PRINCIPLES (Debye model, Simpson integration). "
            "sigma_expansion_shift: CORE (σ → K and Θ_D → α via Grüneisen)."
        ),
    }
