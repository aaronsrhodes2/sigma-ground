"""
Stress physics — fatigue, fracture mechanics, and creep.

Extends mechanical.py (moduli) and elasticity.py (stress-strain, yield) with
time-dependent and failure-mode physics: how materials break under repeated
loading, crack propagation, and sustained stress at elevated temperature.

Derivation chains:

  1. Fatigue — S-N (Basquin 1910, FIRST_PRINCIPLES + MEASURED)
     σ_a = σ_f' × (2N_f)^b

     Where:
       σ_a = stress amplitude (Pa)
       σ_f' = fatigue strength coefficient ≈ σ_UTS (MEASURED)
       b = fatigue exponent (MEASURED, typically −0.05 to −0.12)
       N_f = cycles to failure

     This is the high-cycle fatigue (HCF) power law.
     FIRST_PRINCIPLES in form; coefficients are MEASURED.

  2. Fracture Mechanics — Griffith/Irwin (FIRST_PRINCIPLES)
     K_I = σ √(π a)   (stress intensity factor, Mode I)

     Griffith criterion (1921):
       G_c = 2γ_s (surface energy creates new crack faces)
       K_Ic = √(E × G_c) = √(2 E γ_s)   (plane stress)

     Where:
       K_Ic = fracture toughness (Pa√m, MEASURED or DERIVED)
       E = Young's modulus (from mechanical.py)
       γ_s = surface energy (from surface.py)
       a = half crack length (m)

     Irwin (1957): equivalent energy and stress-intensity approaches.

  3. Paris Law — Fatigue Crack Growth (FIRST_PRINCIPLES form)
     da/dN = C × (ΔK)^m

     Where:
       da/dN = crack growth per cycle (m/cycle)
       ΔK = stress intensity range (Pa√m)
       C, m = Paris constants (MEASURED, material-dependent)

     Paris & Erdogan (1963): power-law regime of crack growth rate.

  4. Creep — Power-Law (Dorn 1955, FIRST_PRINCIPLES form)
     ε̇ = A × σ^n × exp(−Q/(RT))

     Where:
       ε̇ = steady-state strain rate (1/s)
       A = pre-exponential (MEASURED)
       n = stress exponent (MEASURED, 3–5 for dislocation creep)
       Q = activation energy (MEASURED)

     Dislocation creep: n ≈ 3–5, Q ≈ self-diffusion energy
     Diffusion creep: n = 1 (Nabarro-Herring, already in viscosity.py)

  5. Stress-Life Estimation (Morrow, FIRST_PRINCIPLES)
     With mean stress: σ_a / (σ_f' − σ_m) = (2N_f)^b

σ-dependence:
  Flows through elastic moduli (mechanical.py → elasticity.py).
  E(σ) shifts K_Ic and fatigue parameters through bond stiffness.
  γ_s(σ) shifts Griffith criterion through surface energy.
  Creep activation energy Q shifts through cohesive energy.

Origin tags:
  - Griffith/Irwin: FIRST_PRINCIPLES (energy balance, exact)
  - K_I: FIRST_PRINCIPLES (linear elastic fracture mechanics)
  - Basquin S-N: FIRST_PRINCIPLES form, MEASURED coefficients
  - Paris law: FIRST_PRINCIPLES form, MEASURED coefficients
  - Power-law creep: FIRST_PRINCIPLES form, MEASURED coefficients
  - σ-dependence: CORE (through □σ = −ξR → E, γ_s)
"""

import math
from .mechanical import youngs_modulus, MECHANICAL_DATA
from .surface import MATERIALS, surface_energy, surface_energy_at_sigma
from .elasticity import von_mises_stress
from ..scale import scale_ratio
from ..constants import PROTON_QCD_FRACTION, K_B, EV_TO_J, R_GAS, N_AVOGADRO, SIGMA_HERE


# ── Material Fatigue & Fracture Data ─────────────────────────────
# Rule 9 — If One, Then All: every material that has mechanical data
# gets fatigue and fracture data.
#
# σ_UTS: MEASURED ultimate tensile strength (Pa)
# b_fatigue: GENERIC crystal-class Basquin exponent (dimensionless, negative)
# C_paris: MEASURED Paris law coefficient (m/cycle per (MPa√m)^m)
# m_paris: MEASURED Paris law exponent (dimensionless)
# K_Ic_measured: MEASURED fracture toughness (Pa√m)
# Q_creep_eV: MEASURED creep activation energy (eV), ≈ self-diffusion
# n_creep: MEASURED creep stress exponent (dimensionless)
# A_creep: CALIBRATED creep pre-exponential (1/s per MPa^n), order-of-magnitude
#
# Sources: ASM Handbook, Hertzberg "Deformation & Fracture Mechanics" (2012),
#          Callister "Materials Science" (2018), Frost & Ashby "Deformation
#          Mechanism Maps" (1982)

STRESS_DATA = {
    'iron': {
        'sigma_UTS_Pa': 540e6,          # Mild steel (MEASURED)
        'b_fatigue': -0.087,            # Typical BCC steel
        'C_paris': 6.9e-12,            # m/cycle / (Pa√m)^m, mild steel
        'm_paris': 3.0,                # Typical for steel
        'K_Ic_measured': 50e6,          # 50 MPa√m (structural steel)
        'Q_creep_eV': 2.6,             # ≈ self-diffusion in α-Fe
        'n_creep': 4.5,                # Dislocation creep
        'A_creep': 2.0e6,              # Pre-exponential (calibrated)
    },
    'copper': {
        'sigma_UTS_Pa': 210e6,          # Annealed Cu (MEASURED)
        'b_fatigue': -0.10,             # Typical FCC
        'C_paris': 3.0e-12,            # FCC metal
        'm_paris': 3.5,
        'K_Ic_measured': 100e6,         # 100 MPa√m, annealed OFHC (ASM Handbook)
        'Q_creep_eV': 2.0,             # Self-diffusion in Cu
        'n_creep': 4.0,
        'A_creep': 1.0e7,
    },
    'aluminum': {
        'sigma_UTS_Pa': 90e6,           # Pure Al (MEASURED)
        'b_fatigue': -0.11,             # Soft FCC
        'C_paris': 5.0e-11,            # Al alloys
        'm_paris': 3.0,
        'K_Ic_measured': 30e6,          # 30 MPa√m, pure Al (ASM Handbook)
        'Q_creep_eV': 1.4,             # Self-diffusion in Al
        'n_creep': 4.0,
        'A_creep': 5.0e8,
    },
    'gold': {
        'sigma_UTS_Pa': 120e6,          # Pure Au (MEASURED)
        'b_fatigue': -0.10,             # Soft FCC
        'C_paris': 4.0e-11,            # ESTIMATED — no direct measurement for pure Au; FCC analogy
        'm_paris': 3.2,
        'K_Ic_measured': 40e6,          # ~40 MPa√m, pure Au (ESTIMATED from ductility class)
        'Q_creep_eV': 1.8,             # Self-diffusion in Au
        'n_creep': 4.0,
        'A_creep': 3.0e7,
    },
    'silicon': {
        'sigma_UTS_Pa': 165e6,          # Brittle fracture (MEASURED)
        'b_fatigue': -0.06,             # Brittle — flat S-N curve
        'C_paris': 1.0e-13,            # Very slow crack growth
        'm_paris': 2.5,                # Brittle materials: lower m
        'K_Ic_measured': 0.7e6,         # 0.7 MPa√m (very brittle)
        'Q_creep_eV': 5.0,             # High barrier — covalent bonds
        'n_creep': 3.5,
        'A_creep': 1.0e-2,             # Very slow creep
    },
    'tungsten': {
        'sigma_UTS_Pa': 1510e6,         # (MEASURED)
        'b_fatigue': -0.07,             # Refractory BCC
        'C_paris': 2.0e-12,
        'm_paris': 3.0,
        'K_Ic_measured': 8e6,           # 8 MPa√m, brittle BCC refractory (ASM Handbook)
        'Q_creep_eV': 5.8,             # Very high melting point
        'n_creep': 5.0,
        'A_creep': 1.0e-3,
    },
    'nickel': {
        'sigma_UTS_Pa': 462e6,          # Pure Ni (MEASURED)
        'b_fatigue': -0.09,             # FCC
        'C_paris': 4.0e-12,
        'm_paris': 3.3,
        'K_Ic_measured': 100e6,         # 100 MPa√m, annealed (ASM Handbook)
        'Q_creep_eV': 2.8,             # Self-diffusion in Ni
        'n_creep': 4.5,
        'A_creep': 5.0e5,
    },
    'titanium': {
        'sigma_UTS_Pa': 434e6,          # CP-Ti Grade 2 (MEASURED)
        'b_fatigue': -0.09,
        'C_paris': 5.0e-12,
        'm_paris': 3.2,
        'K_Ic_measured': 55e6,          # 55 MPa√m (MEASURED)
        'Q_creep_eV': 2.5,
        'n_creep': 4.0,
        'A_creep': 1.0e6,
    },

    # ── Additional metals ─────────────────────────────────────────
    'steel_mild': {
        'sigma_UTS_Pa': 450e6,          # AISI 1020 (MEASURED)
        'b_fatigue': -0.087,
        'C_paris': 6.9e-12,
        'm_paris': 3.0,
        'K_Ic_measured': 50e6,
        'Q_creep_eV': 2.6,
        'n_creep': 4.5,
        'A_creep': 2.0e6,
    },
    'lead': {
        'sigma_UTS_Pa': 18e6,           # Pure Pb (MEASURED) — very soft
        'b_fatigue': -0.12,
        'C_paris': 1.0e-10,
        'm_paris': 3.5,
        'K_Ic_measured': 15e6,          # Ductile, high toughness for softness
        'Q_creep_eV': 1.1,             # Very low — creeps at room temp
        'n_creep': 4.0,
        'A_creep': 1.0e10,             # High pre-exponential, fast creep
    },
    'silver': {
        'sigma_UTS_Pa': 170e6,          # Annealed Ag (MEASURED)
        'b_fatigue': -0.10,
        'C_paris': 3.0e-11,
        'm_paris': 3.2,
        'K_Ic_measured': 50e6,          # Ductile FCC
        'Q_creep_eV': 1.9,
        'n_creep': 4.0,
        'A_creep': 5.0e7,
    },
    'platinum': {
        'sigma_UTS_Pa': 140e6,          # Annealed Pt (MEASURED)
        'b_fatigue': -0.09,
        'C_paris': 3.0e-12,
        'm_paris': 3.0,
        'K_Ic_measured': 60e6,
        'Q_creep_eV': 2.9,
        'n_creep': 4.0,
        'A_creep': 1.0e5,
    },
    'depleted_uranium': {
        'sigma_UTS_Pa': 1200e6,         # U-0.75Ti alloy (MEASURED)
        'b_fatigue': -0.08,
        'C_paris': 5.0e-12,
        'm_paris': 3.0,
        'K_Ic_measured': 35e6,
        'Q_creep_eV': 2.0,
        'n_creep': 4.0,
        'A_creep': 1.0e6,
    },

    # ── Non-metals ────────────────────────────────────────────────
    'rubber': {
        'sigma_UTS_Pa': 25e6,           # Tensile (MEASURED)
        'b_fatigue': -0.15,             # Polymer fatigue
        'C_paris': 1.0e-8,
        'm_paris': 4.0,
        'K_Ic_measured': 1.0e6,         # 1 MPa√m — tears easily
        'Q_creep_eV': 0.8,
        'n_creep': 2.0,
        'A_creep': 1.0e12,
    },
    'plastic_abs': {
        'sigma_UTS_Pa': 44e6,           # MEASURED
        'b_fatigue': -0.12,
        'C_paris': 5.0e-9,
        'm_paris': 4.0,
        'K_Ic_measured': 3.0e6,
        'Q_creep_eV': 1.0,
        'n_creep': 3.0,
        'A_creep': 1.0e10,
    },
    'glass': {
        'sigma_UTS_Pa': 33e6,           # Tensile fracture (MEASURED)
        'b_fatigue': -0.05,
        'C_paris': 1.0e-13,
        'm_paris': 2.0,
        'K_Ic_measured': 0.75e6,        # Very brittle
        'Q_creep_eV': 4.0,
        'n_creep': 3.0,
        'A_creep': 1.0e-4,
    },
    'concrete': {
        'sigma_UTS_Pa': 3e6,            # Tensile! (MEASURED) — 10x weaker than compressive
        'b_fatigue': -0.06,
        'C_paris': 5.0e-12,
        'm_paris': 3.0,
        'K_Ic_measured': 1.0e6,
        'Q_creep_eV': 2.0,
        'n_creep': 3.0,
        'A_creep': 1.0e4,
    },
    'granite': {
        'sigma_UTS_Pa': 15e6,           # Tensile (MEASURED) — strong in compression
        'b_fatigue': -0.06,
        'C_paris': 1.0e-12,
        'm_paris': 2.5,
        'K_Ic_measured': 1.5e6,
        'Q_creep_eV': 3.0,
        'n_creep': 3.0,
        'A_creep': 1.0e-2,
    },
    'ceramic_alumina': {
        'sigma_UTS_Pa': 300e6,          # Flexural (MEASURED)
        'b_fatigue': -0.05,
        'C_paris': 5.0e-14,
        'm_paris': 2.5,
        'K_Ic_measured': 4.0e6,         # 4 MPa√m
        'Q_creep_eV': 4.5,
        'n_creep': 3.0,
        'A_creep': 1.0e-4,
    },
    'water_ice': {
        'sigma_UTS_Pa': 1e6,            # Tensile (MEASURED) — very weak
        'b_fatigue': -0.10,
        'C_paris': 1.0e-9,
        'm_paris': 4.0,
        'K_Ic_measured': 0.1e6,         # 0.1 MPa√m
        'Q_creep_eV': 0.6,             # Glen's flow law
        'n_creep': 3.0,                # Glen's exponent
        'A_creep': 1.0e14,             # Glacier flow
    },
    'wood_oak': {
        'sigma_UTS_Pa': 100e6,          # Tensile along grain (MEASURED)
        'b_fatigue': -0.08,
        'C_paris': 1.0e-10,
        'm_paris': 3.5,
        'K_Ic_measured': 5.0e6,         # Along grain
        'Q_creep_eV': 1.2,
        'n_creep': 3.0,
        'A_creep': 1.0e6,
    },
    'bone': {
        'sigma_UTS_Pa': 150e6,          # Tensile, cortical (MEASURED)
        'b_fatigue': -0.08,
        'C_paris': 5.0e-11,
        'm_paris': 3.5,
        'K_Ic_measured': 5.0e6,         # 5 MPa√m — biocomposite
        'Q_creep_eV': 1.5,
        'n_creep': 3.0,
        'A_creep': 1.0e6,
    },
    'carbon_fiber': {
        'sigma_UTS_Pa': 3500e6,         # Tensile along fiber (MEASURED)
        'b_fatigue': -0.05,
        'C_paris': 1.0e-13,
        'm_paris': 2.5,
        'K_Ic_measured': 40e6,          # CFRP composite
        'Q_creep_eV': 5.0,
        'n_creep': 3.0,
        'A_creep': 1.0e-4,
    },
    'kevlar': {
        'sigma_UTS_Pa': 3600e6,         # Fiber tensile (MEASURED)
        'b_fatigue': -0.06,
        'C_paris': 1.0e-11,
        'm_paris': 3.0,
        'K_Ic_measured': 30e6,
        'Q_creep_eV': 2.0,
        'n_creep': 3.0,
        'A_creep': 1.0e2,
    },
}


_EV_TO_JOULE = EV_TO_J
_R_GAS = R_GAS
_N_A = N_AVOGADRO


# ── Griffith Fracture Toughness ──────────────────────────────────

def griffith_toughness(material_key, sigma=SIGMA_HERE):
    """Griffith fracture toughness K_Ic (Pa√m).

    K_Ic = √(2 E γ_s)   (plane stress)

    FIRST_PRINCIPLES: Griffith (1921) — the energy to propagate a crack
    equals the energy to create two new surfaces. Irwin (1957) showed
    this is equivalent to the critical stress intensity factor.

    For ductile metals, this UNDERESTIMATES K_Ic because plastic
    dissipation at the crack tip adds to the effective surface energy.
    We compute the ideal brittle value; measured K_Ic is always higher
    for metals.

    Args:
        material_key: key into MATERIALS/MECHANICAL_DATA
        sigma: σ-field value

    Returns:
        K_Ic in Pa√m
    """
    E = youngs_modulus(material_key, sigma)
    gamma = surface_energy_at_sigma(material_key, sigma)
    return math.sqrt(2.0 * E * gamma)


def fracture_toughness(material_key, sigma=SIGMA_HERE):
    """Effective fracture toughness K_Ic (Pa√m).

    Returns MEASURED K_Ic if available; otherwise Griffith estimate.
    For metals, measured >> Griffith (plastic zone dissipation).
    For brittle ceramics/semiconductors, they are comparable.

    Args:
        material_key: key into STRESS_DATA
        sigma: σ-field value

    Returns:
        K_Ic in Pa√m
    """
    data = STRESS_DATA[material_key]
    measured = data.get('K_Ic_measured')
    if measured is not None and sigma == SIGMA_HERE:
        return measured
    if measured is not None and sigma != SIGMA_HERE:
        # Scale measured K_Ic with √(E(σ)/E(0)) — modulus shift
        E_0 = youngs_modulus(material_key, SIGMA_HERE)
        E_s = youngs_modulus(material_key, sigma)
        return measured * math.sqrt(E_s / E_0)
    return griffith_toughness(material_key, sigma)


# ── Stress Intensity Factor ──────────────────────────────────────

def stress_intensity(applied_stress, crack_half_length):
    """Mode I stress intensity factor K_I (Pa√m).

    K_I = σ √(π a)

    FIRST_PRINCIPLES: Irwin (1957). Linear elastic fracture mechanics.
    This is for a through-thickness center crack in an infinite plate.
    Geometry corrections (Y factors) needed for other configurations.

    Args:
        applied_stress: far-field tensile stress (Pa)
        crack_half_length: half the crack length (m)

    Returns:
        K_I in Pa√m
    """
    return applied_stress * math.sqrt(math.pi * crack_half_length)


def critical_crack_length(material_key, applied_stress, sigma=SIGMA_HERE):
    """Critical crack half-length for unstable fracture (m).

    a_c = (K_Ic / σ)² / π

    FIRST_PRINCIPLES: from K_I = K_Ic at fracture.

    Args:
        material_key: key into STRESS_DATA
        applied_stress: far-field tensile stress (Pa)
        sigma: σ-field value

    Returns:
        Critical half-crack length in metres
    """
    K_Ic = fracture_toughness(material_key, sigma)
    return (K_Ic / applied_stress) ** 2 / math.pi


# ── Fatigue — Basquin S-N ────────────────────────────────────────

def fatigue_life(material_key, stress_amplitude, sigma_mean=0.0):
    """Cycles to failure N_f from Basquin S-N curve.

    σ_a = σ_f' × (2N_f)^b   →   N_f = ½ × (σ_a / σ_f')^(1/b)

    With mean stress (Morrow correction):
    σ_a / (σ_f' − σ_m) = (2N_f)^b

    FIRST_PRINCIPLES form, MEASURED coefficients (σ_f' ≈ σ_UTS, b).
    Valid for high-cycle fatigue (N_f > ~10⁴).

    Args:
        material_key: key into STRESS_DATA
        stress_amplitude: alternating stress amplitude (Pa)
        sigma_mean: mean stress (Pa), default 0 (fully reversed)

    Returns:
        N_f — cycles to failure (float). Returns inf if below endurance.
    """
    data = STRESS_DATA[material_key]
    sigma_f = data['sigma_UTS_Pa']
    b = data['b_fatigue']

    effective_sigma_f = sigma_f - sigma_mean
    if effective_sigma_f <= 0 or stress_amplitude <= 0:
        return 0.0  # Immediate failure — mean stress exceeds strength

    if stress_amplitude >= effective_sigma_f:
        return 0.0  # Static failure

    ratio = stress_amplitude / effective_sigma_f
    # ratio = (2 N_f)^b → 2 N_f = ratio^(1/b) → N_f = 0.5 × ratio^(1/b)
    N_f = 0.5 * ratio ** (1.0 / b)
    return N_f


def fatigue_strength(material_key, N_f, sigma_mean=0.0):
    """Stress amplitude for a given fatigue life (Pa).

    σ_a = (σ_f' − σ_m) × (2N_f)^b

    Inverse of fatigue_life(). Given a target life N_f, returns the
    maximum allowable stress amplitude.

    Args:
        material_key: key into STRESS_DATA
        N_f: target cycles to failure
        sigma_mean: mean stress (Pa)

    Returns:
        Allowable stress amplitude in Pa
    """
    data = STRESS_DATA[material_key]
    sigma_f = data['sigma_UTS_Pa']
    b = data['b_fatigue']

    effective_sigma_f = sigma_f - sigma_mean
    if effective_sigma_f <= 0 or N_f <= 0:
        return 0.0

    return effective_sigma_f * (2.0 * N_f) ** b


# ── Paris Law — Fatigue Crack Growth ─────────────────────────────

def paris_crack_growth_rate(material_key, delta_K):
    """Fatigue crack growth rate da/dN (m/cycle).

    da/dN = C × (ΔK)^m

    FIRST_PRINCIPLES form (Paris & Erdogan 1963), MEASURED C and m.
    Valid in the Paris regime (intermediate ΔK). Below threshold ΔK_th,
    cracks don't grow. Above K_Ic, fast fracture.

    Note: C values stored in MPa√m convention (literature standard).
    Input ΔK in Pa√m is converted internally.

    Args:
        material_key: key into STRESS_DATA
        delta_K: stress intensity range ΔK (Pa√m)

    Returns:
        da/dN in m/cycle
    """
    data = STRESS_DATA[material_key]
    C = data['C_paris']
    m = data['m_paris']
    delta_K_MPa = delta_K / 1e6  # Convert Pa√m → MPa√m
    return C * delta_K_MPa ** m


def paris_remaining_life(material_key, a_initial, a_critical,
                         applied_stress):
    """Remaining fatigue life by integrating Paris law (cycles).

    N = ∫ da / (C × (ΔK)^m)

    Where ΔK = Δσ √(πa).

    For m ≠ 2, closed-form:
    N = 2 / ((m-2) C (Δσ√π)^m) × [a_i^(1-m/2) − a_c^(1-m/2)]

    FIRST_PRINCIPLES: integration of Paris equation with constant Δσ.

    Args:
        material_key: key into STRESS_DATA
        a_initial: initial half-crack length (m)
        a_critical: critical half-crack length (m)
        applied_stress: stress range Δσ (Pa)

    Returns:
        N — remaining cycles to fracture
    """
    data = STRESS_DATA[material_key]
    C = data['C_paris']
    m = data['m_paris']

    if a_initial >= a_critical or applied_stress <= 0:
        return 0.0

    # C is in MPa√m convention, so convert stress to MPa
    stress_MPa = applied_stress / 1e6
    coeff = C * (stress_MPa * math.sqrt(math.pi)) ** m

    if abs(m - 2.0) < 1e-10:
        # Special case m=2: N = ln(a_c/a_i) / (C × (Δσ√π)²)
        return math.log(a_critical / a_initial) / coeff

    exp = 1.0 - m / 2.0
    numerator = a_initial ** exp - a_critical ** exp
    return numerator / ((-exp) * coeff)


# ── Power-Law Creep ──────────────────────────────────────────────

def creep_strain_rate(material_key, stress, temperature, sigma=SIGMA_HERE):
    """Steady-state creep strain rate ε̇ (1/s).

    ε̇ = A × σ^n × exp(−Q/(k_B T))

    FIRST_PRINCIPLES form (Dorn 1955), MEASURED A, n, Q.
    Dislocation creep: n ≈ 3–5. Diffusion creep: n = 1 (see viscosity.py).

    σ-dependence: Q shifts through cohesive energy (stronger bonds →
    higher activation barrier → slower creep).

    Args:
        material_key: key into STRESS_DATA
        stress: applied stress (Pa)
        temperature: temperature (K)
        sigma: σ-field value

    Returns:
        Strain rate in 1/s
    """
    if temperature <= 0 or stress <= 0:
        return 0.0

    data = STRESS_DATA[material_key]
    A = data['A_creep']
    n = data['n_creep']
    Q_eV = data['Q_creep_eV']

    # σ-field shifts activation energy through cohesive energy
    if sigma != SIGMA_HERE:
        f_qcd = PROTON_QCD_FRACTION
        mass_ratio = (1.0 - f_qcd) + f_qcd * scale_ratio(sigma)
        Q_eV = Q_eV * mass_ratio

    Q_J = Q_eV * _EV_TO_JOULE * _N_A  # Convert eV/atom to J/mol
    exponent = -Q_J / (_R_GAS * temperature)

    # Guard against overflow in exp
    if exponent < -700:
        return 0.0

    # A is calibrated for stress in MPa (literature convention)
    stress_MPa = stress / 1e6
    return A * stress_MPa ** n * math.exp(exponent)


def creep_rupture_time(material_key, stress, temperature,
                       strain_limit=0.01, sigma=SIGMA_HERE):
    """Estimated time to creep rupture (seconds).

    t_r ≈ ε_limit / ε̇

    APPROXIMATION: assumes constant stress and steady-state creep.
    Real creep has primary (decelerating), secondary (steady), and
    tertiary (accelerating) stages. This gives the secondary-stage
    estimate.

    Args:
        material_key: key into STRESS_DATA
        stress: applied stress (Pa)
        temperature: temperature (K)
        strain_limit: strain at failure (default 1%)
        sigma: σ-field value

    Returns:
        Time to rupture in seconds (inf if strain rate is zero)
    """
    rate = creep_strain_rate(material_key, stress, temperature, sigma)
    if rate <= 0:
        return float('inf')
    return strain_limit / rate


# ── Larson-Miller Parameter ──────────────────────────────────────

def larson_miller_parameter(temperature, rupture_time_hours, C_LM=20.0):
    """Larson-Miller parameter P (K).

    P = T × (C + log₁₀(t_r))

    FIRST_PRINCIPLES: time-temperature equivalence for thermally
    activated creep processes. Higher T or longer time → higher P.
    C_LM ≈ 20 for most engineering metals (Larson & Miller 1952).

    Args:
        temperature: temperature (K)
        rupture_time_hours: time to rupture (hours)
        C_LM: Larson-Miller constant (default 20)

    Returns:
        P in Kelvin (dimensionless parameter × K)
    """
    if rupture_time_hours <= 0:
        return 0.0
    return temperature * (C_LM + math.log10(rupture_time_hours))


# ── σ-Field Effects ──────────────────────────────────────────────

def sigma_fracture_toughness_shift(material_key, sigma):
    """Fracture toughness shift under σ-field.

    K_Ic(σ) = K_Ic(0) × √(E(σ)/E(0))

    CORE: modulus shift propagates to toughness.

    Args:
        material_key: key into STRESS_DATA
        sigma: σ-field value

    Returns:
        K_Ic(σ) in Pa√m
    """
    return fracture_toughness(material_key, sigma)


def sigma_fatigue_shift(material_key, sigma):
    """Fatigue strength coefficient shift under σ-field.

    σ_UTS(σ) ≈ σ_UTS(0) × E(σ)/E(0)

    APPROXIMATION: tensile strength scales roughly with modulus.

    Args:
        material_key: key into STRESS_DATA
        sigma: σ-field value

    Returns:
        σ_UTS(σ) in Pa
    """
    data = STRESS_DATA[material_key]
    E_0 = youngs_modulus(material_key, SIGMA_HERE)
    E_s = youngs_modulus(material_key, sigma)
    return data['sigma_UTS_Pa'] * (E_s / E_0)


# ── Nagatha Integration ──────────────────────────────────────────

def stress_properties(material_key, applied_stress=None,
                      crack_length=None, temperature=None,
                      sigma=SIGMA_HERE):
    """Export stress/fatigue/fracture properties in Nagatha format.

    Args:
        material_key: key into STRESS_DATA
        applied_stress: optional applied stress (Pa)
        crack_length: optional half-crack length (m)
        temperature: optional temperature (K) for creep
        sigma: σ-field value

    Returns:
        Dict of stress properties
    """
    data = STRESS_DATA[material_key]
    K_Ic = fracture_toughness(material_key, sigma)
    K_Ic_griffith = griffith_toughness(material_key, sigma)
    sigma_UTS = data['sigma_UTS_Pa']

    result = {
        'material': material_key,
        'sigma_UTS_Pa': sigma_UTS,
        'K_Ic_Pa_sqrtm': K_Ic,
        'K_Ic_griffith_Pa_sqrtm': K_Ic_griffith,
        'K_Ic_source': 'measured' if data['K_Ic_measured'] is not None else 'griffith',
        'b_fatigue': data['b_fatigue'],
        'm_paris': data['m_paris'],
        'n_creep': data['n_creep'],
        'Q_creep_eV': data['Q_creep_eV'],
        'sigma': sigma,
    }

    if applied_stress is not None:
        result['fatigue_life_cycles'] = fatigue_life(material_key,
                                                     applied_stress)
        if crack_length is not None:
            K_I = stress_intensity(applied_stress, crack_length)
            a_c = critical_crack_length(material_key, applied_stress, sigma)
            result['K_I_Pa_sqrtm'] = K_I
            result['critical_crack_m'] = a_c
            result['will_fracture'] = K_I >= K_Ic
            result['paris_da_dN_m'] = paris_crack_growth_rate(
                material_key, K_I)

    if temperature is not None and applied_stress is not None:
        result['creep_rate_1_s'] = creep_strain_rate(
            material_key, applied_stress, temperature, sigma)
        result['creep_rupture_s'] = creep_rupture_time(
            material_key, applied_stress, temperature, sigma=sigma)

    result['origin_tag'] = (
        "FIRST_PRINCIPLES: Griffith fracture toughness (energy balance). "
        "FIRST_PRINCIPLES: Irwin stress intensity K_I = σ√(πa). "
        "FIRST_PRINCIPLES form: Basquin S-N, Paris crack growth, power-law creep. "
        "MEASURED: σ_UTS, b, C, m, K_Ic, Q, n, A. "
        "CORE: σ-dependence through E(σ), γ_s(σ), Q(σ)."
    )
    return result
