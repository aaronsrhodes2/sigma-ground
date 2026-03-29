"""
Plasticity — post-yield deformation, strain hardening, and flow stress.

Fills the gap between elastic yield (elasticity.py) and failure (stress.py):
what happens to a material AFTER it yields but BEFORE it breaks.

Derivation chains:

  1. Yield Stress (MEASURED)
     σ_y = measured uniaxial yield strength.
     Tabulated for each material. This is the threshold where plastic
     deformation begins (0.2% offset convention for metals).

     Relation to other modules:
       friction.py: H ≈ 3σ_y (Tabor)
       stress.py: σ_UTS > σ_y (UTS is at necking instability)

  2. Ramberg-Osgood (1943, FIRST_PRINCIPLES form)
     ε = σ/E + (σ/K')^(1/n')

     Where:
       ε = total strain (elastic + plastic)
       σ = applied stress (Pa)
       E = Young's modulus (from mechanical.py)
       K' = strength coefficient (Pa, MEASURED)
       n' = strain hardening exponent (dimensionless, MEASURED)

     At σ = σ_y: plastic strain ≈ 0.002 (definition of yield)
     At σ = σ_UTS: Considère criterion dσ/dε = σ (necking)

     K' and n' are derived from σ_y and σ_UTS:
       σ_y = K' × (0.002)^n'   → K' = σ_y / (0.002)^n'
       σ_UTS ≈ K' × n'^n'      → n' from σ_UTS/σ_y ratio

  3. Hollomon Power Law (1945, FIRST_PRINCIPLES form)
     σ = K' × ε_p^n'

     Plastic region only. The true stress-true strain relation for
     work hardening. n' is the slope on a log-log plot.

     n' ranges:
       0.05-0.10 — low hardening (steel, Ti)
       0.15-0.30 — medium hardening (Al, Cu)
       0.30-0.50 — high hardening (austenitic stainless, brass)

  4. Johnson-Cook (1983, FIRST_PRINCIPLES form + MEASURED coefficients)
     σ = (A + B ε_p^n)(1 + C ln(ε̇/ε̇_0))(1 − T*^m)

     Where:
       A = yield stress (Pa)
       B = hardening coefficient (Pa)
       n = hardening exponent
       C = strain rate sensitivity
       ε̇_0 = reference strain rate (1/s)
       T* = (T − T_room)/(T_melt − T_room), homologous temperature
       m = thermal softening exponent

     This is THE constitutive model for dynamic loading (impact, machining,
     ballistics). Separates strain hardening, strain-rate hardening, and
     thermal softening into multiplicative factors.

  5. Ludwik Hardening (1909)
     σ = σ_y + K × ε_p^n

     Oldest hardening law. Linear offset + power law.
     Special case of Hollomon with explicit yield point.

  6. Necking (Considère Criterion, FIRST_PRINCIPLES)
     Necking initiates when dσ_true/dε_true = σ_true.
     For Hollomon: ε_neck = n' (necking strain = hardening exponent).
     For engineering stress: σ_UTS = K' × n'^n' × exp(-n').

  7. Uniform Elongation & Ductility
     ε_uniform = n' (from Considère)
     ε_fracture ≈ ln(1/(1 − RA)) where RA = reduction in area

σ-dependence:
  σ → E_coh → E → elastic contribution changes
  σ → E_coh → σ_y (yield shifts with bond strength)
  σ → E_coh → K', σ_UTS shift proportionally

Origin tags:
  - Yield stress: MEASURED
  - Ramberg-Osgood: FIRST_PRINCIPLES form, MEASURED K', n'
  - Hollomon: FIRST_PRINCIPLES form, MEASURED coefficients
  - Johnson-Cook: FIRST_PRINCIPLES form, MEASURED A, B, C, n, m
  - Considère: FIRST_PRINCIPLES (force equilibrium at neck)
  - σ-dependence: CORE (through □σ = −ξR → E_coh → σ_y)
"""

import math
from .mechanical import youngs_modulus, MECHANICAL_DATA
from .surface import MATERIALS
from ..scale import scale_ratio
from ..constants import PROTON_QCD_FRACTION, SIGMA_HERE


# ── Plasticity Data ──────────────────────────────────────────────
# Rule 9 — If One, Then All: every material gets plasticity data.
#
# sigma_y_Pa: MEASURED 0.2% offset yield stress (Pa)
# n_hardening: MEASURED strain hardening exponent (Hollomon)
# elongation_pct: MEASURED elongation to fracture (%)
# is_ductile: True if material deforms plastically before fracture
#
# Johnson-Cook parameters (for dynamic loading):
#   jc_A: yield stress at reference conditions (Pa)
#   jc_B: hardening coefficient (Pa)
#   jc_n: hardening exponent
#   jc_C: strain rate sensitivity
#   jc_m: thermal softening exponent
#   jc_T_melt: melting temperature (K)
#   jc_edot_0: reference strain rate (1/s)
#
# Sources: ASM Handbook, Dieter "Mechanical Metallurgy" (1986),
#          Meyers "Dynamic Behavior of Materials" (1994),
#          Johnson & Cook, Eng. Fract. Mech. 21 (1985) 31-48.

PLASTICITY_DATA = {
    'iron': {
        'sigma_y_Pa': 250e6,           # Mild steel, annealed (MEASURED)
        'n_hardening': 0.22,           # Moderate hardening
        'elongation_pct': 25.0,        # Ductile
        'is_ductile': True,
        'jc_A': 350e6,                 # AISI 1006 steel
        'jc_B': 275e6,
        'jc_n': 0.36,
        'jc_C': 0.022,
        'jc_m': 1.0,
        'jc_T_melt': 1811.0,
        'jc_edot_0': 1.0,
    },
    'copper': {
        'sigma_y_Pa': 70e6,            # Annealed OFHC Cu (MEASURED)
        'n_hardening': 0.44,           # High hardening (FCC, many slip systems)
        'elongation_pct': 45.0,        # Very ductile
        'is_ductile': True,
        'jc_A': 90e6,                  # OFHC copper
        'jc_B': 292e6,
        'jc_n': 0.31,
        'jc_C': 0.025,
        'jc_m': 1.09,
        'jc_T_melt': 1358.0,
        'jc_edot_0': 1.0,
    },
    'aluminum': {
        'sigma_y_Pa': 35e6,            # Pure Al, annealed (MEASURED)
        'n_hardening': 0.24,           # Moderate
        'elongation_pct': 40.0,        # Very ductile
        'is_ductile': True,
        'jc_A': 148.4e6,              # Al 6061-T6 (alloy, for JC reference)
        'jc_B': 345.5e6,
        'jc_n': 0.183,
        'jc_C': 0.001,                # Low strain rate sensitivity
        'jc_m': 0.895,
        'jc_T_melt': 933.0,
        'jc_edot_0': 1.0,
    },
    'gold': {
        'sigma_y_Pa': 30e6,            # Pure Au, annealed (MEASURED)
        'n_hardening': 0.42,           # High hardening (FCC)
        'elongation_pct': 30.0,        # Ductile
        'is_ductile': True,
        'jc_A': 30e6,                  # Estimated from pure Au
        'jc_B': 150e6,
        'jc_n': 0.40,
        'jc_C': 0.020,
        'jc_m': 1.0,
        'jc_T_melt': 1337.0,
        'jc_edot_0': 1.0,
    },
    'silicon': {
        'sigma_y_Pa': 165e6,           # Fracture stress (brittle, no yield)
        'n_hardening': 0.0,            # Brittle — no strain hardening
        'elongation_pct': 0.0,         # Zero ductility
        'is_ductile': False,
        'jc_A': 165e6,                 # Fracture stress
        'jc_B': 0.0,                   # No hardening
        'jc_n': 1.0,                   # Irrelevant (B=0)
        'jc_C': 0.0,                   # No rate sensitivity
        'jc_m': 1.0,
        'jc_T_melt': 1687.0,
        'jc_edot_0': 1.0,
    },
    'tungsten': {
        'sigma_y_Pa': 750e6,           # Polycrystalline W (MEASURED)
        'n_hardening': 0.12,           # Low hardening (BCC, limited slip)
        'elongation_pct': 2.0,         # Low ductility
        'is_ductile': True,            # Marginally — depends on temperature
        'jc_A': 1506e6,               # Tungsten (elevated T data)
        'jc_B': 177e6,
        'jc_n': 0.12,
        'jc_C': 0.016,
        'jc_m': 1.0,
        'jc_T_melt': 3695.0,
        'jc_edot_0': 1.0,
    },
    'nickel': {
        'sigma_y_Pa': 148e6,           # Pure Ni, annealed (MEASURED)
        'n_hardening': 0.36,           # Moderate-high (FCC)
        'elongation_pct': 30.0,        # Ductile
        'is_ductile': True,
        'jc_A': 163e6,                 # Pure Ni
        'jc_B': 648e6,
        'jc_n': 0.33,
        'jc_C': 0.006,
        'jc_m': 1.44,
        'jc_T_melt': 1728.0,
        'jc_edot_0': 1.0,
    },
    'titanium': {
        'sigma_y_Pa': 275e6,           # CP-Ti Grade 2 (MEASURED)
        'n_hardening': 0.10,           # Low (HCP, limited slip systems)
        'elongation_pct': 20.0,        # Moderate ductility
        'is_ductile': True,
        'jc_A': 862.5e6,              # Ti-6Al-4V (alloy, for JC reference)
        'jc_B': 331.2e6,
        'jc_n': 0.34,
        'jc_C': 0.012,
        'jc_m': 0.8,
        'jc_T_melt': 1941.0,
        'jc_edot_0': 1.0,
    },

    # ── Additional metals ─────────────────────────────────────────
    'steel_mild': {
        'sigma_y_Pa': 250e6,           # AISI 1020 (MEASURED)
        'n_hardening': 0.22,           # Moderate
        'elongation_pct': 25.0,
        'is_ductile': True,
        'jc_A': 350e6,
        'jc_B': 275e6,
        'jc_n': 0.36,
        'jc_C': 0.022,
        'jc_m': 1.0,
        'jc_T_melt': 1793.0,
        'jc_edot_0': 1.0,
    },
    'lead': {
        'sigma_y_Pa': 11e6,            # Pure Pb, annealed (MEASURED)
        'n_hardening': 0.40,           # High hardening (FCC)
        'elongation_pct': 50.0,        # Very ductile — soft metal
        'is_ductile': True,
        'jc_A': 24e6,
        'jc_B': 300e6,
        'jc_n': 1.0,
        'jc_C': 0.01,
        'jc_m': 1.0,
        'jc_T_melt': 600.6,
        'jc_edot_0': 1.0,
    },
    'silver': {
        'sigma_y_Pa': 45e6,            # Pure Ag, annealed (MEASURED)
        'n_hardening': 0.44,           # High (FCC)
        'elongation_pct': 40.0,
        'is_ductile': True,
        'jc_A': 45e6,
        'jc_B': 200e6,
        'jc_n': 0.40,
        'jc_C': 0.015,
        'jc_m': 1.0,
        'jc_T_melt': 1234.9,
        'jc_edot_0': 1.0,
    },
    'platinum': {
        'sigma_y_Pa': 50e6,            # Pure Pt, annealed (MEASURED)
        'n_hardening': 0.38,
        'elongation_pct': 35.0,
        'is_ductile': True,
        'jc_A': 50e6,
        'jc_B': 250e6,
        'jc_n': 0.38,
        'jc_C': 0.015,
        'jc_m': 1.0,
        'jc_T_melt': 2041.0,
        'jc_edot_0': 1.0,
    },
    'depleted_uranium': {
        'sigma_y_Pa': 790e6,           # U-0.75Ti alloy (MEASURED)
        'n_hardening': 0.10,           # Low hardening
        'elongation_pct': 10.0,        # Moderate ductility
        'is_ductile': True,
        'jc_A': 1079e6,               # DU penetrator alloy
        'jc_B': 1120e6,
        'jc_n': 0.25,
        'jc_C': 0.007,
        'jc_m': 1.0,
        'jc_T_melt': 1405.0,
        'jc_edot_0': 1.0,
    },

    # ── Non-metals ────────────────────────────────────────────────
    'rubber': {
        'sigma_y_Pa': 15e6,            # Tensile strength (MEASURED)
        'n_hardening': 0.80,           # High — entropic elasticity
        'elongation_pct': 500.0,       # Hyper-elastic
        'is_ductile': True,
        'jc_A': 15e6,
        'jc_B': 5e6,
        'jc_n': 0.80,
        'jc_C': 0.01,
        'jc_m': 1.0,
        'jc_T_melt': 473.0,           # Degrades, doesn't truly melt
        'jc_edot_0': 1.0,
    },
    'plastic_abs': {
        'sigma_y_Pa': 40e6,            # MEASURED
        'n_hardening': 0.30,
        'elongation_pct': 20.0,
        'is_ductile': True,
        'jc_A': 40e6,
        'jc_B': 50e6,
        'jc_n': 0.30,
        'jc_C': 0.01,
        'jc_m': 1.0,
        'jc_T_melt': 473.0,           # Glass transition, not true melt
        'jc_edot_0': 1.0,
    },
    'glass': {
        'sigma_y_Pa': 33e6,            # Compressive fracture (MEASURED)
        'n_hardening': 0.0,            # Brittle — no strain hardening
        'elongation_pct': 0.0,
        'is_ductile': False,
        'jc_A': 33e6,
        'jc_B': 0.0,
        'jc_n': 1.0,
        'jc_C': 0.0,
        'jc_m': 1.0,
        'jc_T_melt': 1273.0,          # Softening point
        'jc_edot_0': 1.0,
    },
    'concrete': {
        'sigma_y_Pa': 30e6,            # Compressive strength (MEASURED)
        'n_hardening': 0.0,
        'elongation_pct': 0.0,
        'is_ductile': False,
        'jc_A': 30e6,
        'jc_B': 0.0,
        'jc_n': 1.0,
        'jc_C': 0.0,
        'jc_m': 1.0,
        'jc_T_melt': 1773.0,          # Calcium silicates decompose
        'jc_edot_0': 1.0,
    },
    'granite': {
        'sigma_y_Pa': 130e6,           # Compressive strength (MEASURED)
        'n_hardening': 0.0,
        'elongation_pct': 0.0,
        'is_ductile': False,
        'jc_A': 130e6,
        'jc_B': 0.0,
        'jc_n': 1.0,
        'jc_C': 0.0,
        'jc_m': 1.0,
        'jc_T_melt': 1473.0,          # Melting onset
        'jc_edot_0': 1.0,
    },
    'ceramic_alumina': {
        'sigma_y_Pa': 2000e6,          # Compressive (MEASURED) — very hard
        'n_hardening': 0.0,
        'elongation_pct': 0.0,
        'is_ductile': False,
        'jc_A': 2000e6,
        'jc_B': 0.0,
        'jc_n': 1.0,
        'jc_C': 0.0,
        'jc_m': 1.0,
        'jc_T_melt': 2345.0,
        'jc_edot_0': 1.0,
    },
    'water_ice': {
        'sigma_y_Pa': 5e6,             # Compressive (MEASURED)
        'n_hardening': 0.0,
        'elongation_pct': 0.0,
        'is_ductile': False,
        'jc_A': 5e6,
        'jc_B': 0.0,
        'jc_n': 1.0,
        'jc_C': 0.0,
        'jc_m': 1.0,
        'jc_T_melt': 273.15,
        'jc_edot_0': 1.0,
    },
    'wood_oak': {
        'sigma_y_Pa': 40e6,            # Compressive parallel to grain (MEASURED)
        'n_hardening': 0.05,
        'elongation_pct': 2.0,
        'is_ductile': False,
        'jc_A': 40e6,
        'jc_B': 10e6,
        'jc_n': 0.05,
        'jc_C': 0.0,
        'jc_m': 1.0,
        'jc_T_melt': 573.0,           # Ignition, not melting
        'jc_edot_0': 1.0,
    },
    'bone': {
        'sigma_y_Pa': 130e6,           # Compressive, cortical (MEASURED)
        'n_hardening': 0.05,
        'elongation_pct': 2.0,
        'is_ductile': False,
        'jc_A': 130e6,
        'jc_B': 30e6,
        'jc_n': 0.05,
        'jc_C': 0.01,
        'jc_m': 1.0,
        'jc_T_melt': 1673.0,          # Hydroxyapatite decomposition
        'jc_edot_0': 1.0,
    },
    'carbon_fiber': {
        'sigma_y_Pa': 600e6,           # Tensile along fiber (MEASURED)
        'n_hardening': 0.02,           # Nearly linear to fracture
        'elongation_pct': 1.5,
        'is_ductile': False,
        'jc_A': 600e6,
        'jc_B': 50e6,
        'jc_n': 0.02,
        'jc_C': 0.0,
        'jc_m': 1.0,
        'jc_T_melt': 3773.0,          # Carbon sublimation
        'jc_edot_0': 1.0,
    },
    'kevlar': {
        'sigma_y_Pa': 350e6,           # Tensile (MEASURED)
        'n_hardening': 0.05,
        'elongation_pct': 3.6,
        'is_ductile': False,           # Fails by fiber breakage
        'jc_A': 350e6,
        'jc_B': 50e6,
        'jc_n': 0.05,
        'jc_C': 0.0,
        'jc_m': 1.0,
        'jc_T_melt': 773.0,           # Decomposes
        'jc_edot_0': 1.0,
    },
}


# ── Yield Stress ─────────────────────────────────────────────────

def yield_stress(material_key, sigma=SIGMA_HERE):
    """Yield stress σ_y (Pa) at given σ-field.

    MEASURED at σ=0. Scales with bond strength at σ≠0.

    Args:
        material_key: key into PLASTICITY_DATA
        sigma: σ-field value

    Returns:
        Yield stress in Pa.
    """
    sy = PLASTICITY_DATA[material_key]['sigma_y_Pa']
    if sigma == SIGMA_HERE:
        return sy
    r = scale_ratio(sigma)
    return sy * r


# ── Strength Coefficient ────────────────────────────────────────

def strength_coefficient(material_key, sigma=SIGMA_HERE):
    """Hollomon/Ramberg-Osgood strength coefficient K' (Pa).

    Derived from yield stress and hardening exponent:
      σ_y = K' × (0.002)^n'
      K' = σ_y / (0.002)^n'

    The 0.002 is the plastic strain at the 0.2% offset yield point.

    Args:
        material_key: key into PLASTICITY_DATA
        sigma: σ-field value

    Returns:
        Strength coefficient in Pa.
    """
    data = PLASTICITY_DATA[material_key]
    sy = yield_stress(material_key, sigma)
    n = data['n_hardening']

    if n <= 0:
        # Brittle material or perfectly plastic: K' = σ_y
        return sy

    return sy / (0.002 ** n)


# ── Hollomon Flow Stress ─────────────────────────────────────────

def hollomon_stress(material_key, plastic_strain, sigma=SIGMA_HERE):
    """True stress from Hollomon power-law hardening (Pa).

    σ = K' × ε_p^n'

    Valid for plastic_strain > 0 (post-yield).
    For plastic_strain = 0, returns yield stress.

    Args:
        material_key: key into PLASTICITY_DATA
        plastic_strain: true plastic strain (dimensionless, > 0)
        sigma: σ-field value

    Returns:
        True stress in Pa.
    """
    if plastic_strain <= 0:
        return yield_stress(material_key, sigma)

    data = PLASTICITY_DATA[material_key]
    n = data['n_hardening']
    Kp = strength_coefficient(material_key, sigma)

    if n <= 0:
        # Perfectly plastic (no hardening)
        return yield_stress(material_key, sigma)

    return Kp * (plastic_strain ** n)


# ── Ramberg-Osgood Total Strain ──────────────────────────────────

def ramberg_osgood_strain(material_key, stress_pa, sigma=SIGMA_HERE):
    """Total strain (elastic + plastic) from Ramberg-Osgood equation.

    ε = σ/E + (σ/K')^(1/n')

    Smooth transition from elastic to plastic without a sharp yield point.

    Args:
        material_key: key into PLASTICITY_DATA
        stress_pa: applied true stress (Pa)
        sigma: σ-field value

    Returns:
        Total true strain (dimensionless).
    """
    if stress_pa <= 0:
        return 0.0

    E = youngs_modulus(material_key, sigma)
    Kp = strength_coefficient(material_key, sigma)
    n = PLASTICITY_DATA[material_key]['n_hardening']

    elastic = stress_pa / E

    if n <= 0 or Kp <= 0:
        # Brittle: only elastic strain until fracture
        return elastic

    plastic = (stress_pa / Kp) ** (1.0 / n)
    return elastic + plastic


# ── Ludwik Hardening ─────────────────────────────────────────────

def ludwik_stress(material_key, plastic_strain, sigma=SIGMA_HERE):
    """True stress from Ludwik hardening law (Pa).

    σ = σ_y + K × ε_p^n

    Explicit yield offset plus power-law hardening.
    K is derived so that at ε_p = 0.2 (20% strain), σ ≈ σ_UTS.

    Args:
        material_key: key into PLASTICITY_DATA
        plastic_strain: true plastic strain
        sigma: σ-field value

    Returns:
        True stress in Pa.
    """
    sy = yield_stress(material_key, sigma)
    if plastic_strain <= 0:
        return sy

    data = PLASTICITY_DATA[material_key]
    n = data['n_hardening']

    if n <= 0:
        return sy

    # Derive Ludwik K from the difference between UTS and yield
    from .stress import STRESS_DATA
    if material_key in STRESS_DATA:
        uts = STRESS_DATA[material_key]['sigma_UTS_Pa']
        if sigma != SIGMA_HERE:
            uts *= scale_ratio(sigma)
        K_ludwik = (uts - sy) / (0.2 ** n) if 0.2 ** n > 0 else 0.0
    else:
        # Fallback: K_ludwik from strength coefficient
        Kp = strength_coefficient(material_key, sigma)
        K_ludwik = Kp - sy

    return sy + K_ludwik * (plastic_strain ** n)


# ── Johnson-Cook Flow Stress ─────────────────────────────────────

def johnson_cook_stress(material_key, plastic_strain, strain_rate=1.0,
                        temperature_K=293.0, sigma=SIGMA_HERE):
    """Johnson-Cook dynamic flow stress (Pa).

    σ = (A + B ε_p^n)(1 + C ln(ε̇/ε̇_0))(1 − T*^m)

    Three multiplicative factors:
      1. Strain hardening: A + B ε_p^n
      2. Strain rate hardening: 1 + C ln(ε̇/ε̇_0)
      3. Thermal softening: 1 − T*^m

    T* = (T − T_room)/(T_melt − T_room), clamped to [0, 1].

    Args:
        material_key: key into PLASTICITY_DATA
        plastic_strain: equivalent plastic strain (dimensionless)
        strain_rate: strain rate (1/s), default 1.0 (quasi-static)
        temperature_K: temperature in Kelvin, default 293 (room temp)
        sigma: σ-field value

    Returns:
        Flow stress in Pa.
    """
    data = PLASTICITY_DATA[material_key]

    A = data['jc_A']
    B = data['jc_B']
    n = data['jc_n']
    C = data['jc_C']
    m = data['jc_m']
    T_melt = data['jc_T_melt']
    edot_0 = data['jc_edot_0']
    T_room = 293.0

    # σ-field scaling: A and B scale with bond strength
    if sigma != SIGMA_HERE:
        r = scale_ratio(sigma)
        A *= r
        B *= r

    # 1. Strain hardening
    eps_p = max(plastic_strain, 0.0)
    f_hardening = A + B * (eps_p ** n) if n > 0 and B > 0 else A

    # 2. Strain rate hardening
    edot = max(strain_rate, 1e-30)  # Guard against zero
    ratio = edot / edot_0
    f_rate = 1.0 + C * math.log(ratio) if ratio > 1.0 else 1.0

    # 3. Thermal softening
    if T_melt > T_room and temperature_K > T_room:
        T_star = (temperature_K - T_room) / (T_melt - T_room)
        T_star = min(max(T_star, 0.0), 1.0)
        f_thermal = 1.0 - (T_star ** m)
    else:
        f_thermal = 1.0

    return f_hardening * f_rate * f_thermal


# ── Necking (Considère Criterion) ────────────────────────────────

def necking_strain(material_key):
    """True strain at onset of necking (dimensionless).

    From Considère criterion: dσ/dε = σ at the necking point.
    For Hollomon hardening (σ = K' ε^n), this gives ε_neck = n.

    This is also the uniform elongation — the maximum strain before
    localization begins.

    Args:
        material_key: key into PLASTICITY_DATA

    Returns:
        True strain at necking.
    """
    n = PLASTICITY_DATA[material_key]['n_hardening']
    return max(n, 0.0)


def necking_stress(material_key, sigma=SIGMA_HERE):
    """True stress at onset of necking (Pa).

    σ_neck = K' × n'^n'

    This equals the engineering UTS (approximately).

    Args:
        material_key: key into PLASTICITY_DATA
        sigma: σ-field value

    Returns:
        True stress at necking in Pa.
    """
    data = PLASTICITY_DATA[material_key]
    n = data['n_hardening']

    if n <= 0:
        return yield_stress(material_key, sigma)

    Kp = strength_coefficient(material_key, sigma)
    return Kp * (n ** n)


# ── Ductility ───────────────────────────────────────────────────

def uniform_elongation(material_key):
    """Uniform elongation before necking (engineering strain, fraction).

    ε_eng = exp(ε_true) - 1 = exp(n) - 1

    Args:
        material_key: key into PLASTICITY_DATA

    Returns:
        Engineering strain at uniform elongation (dimensionless).
    """
    n = PLASTICITY_DATA[material_key]['n_hardening']
    if n <= 0:
        return 0.0
    return math.exp(n) - 1.0


def total_elongation(material_key):
    """Total elongation to fracture (engineering strain, fraction).

    MEASURED value, converted from percentage.

    Args:
        material_key: key into PLASTICITY_DATA

    Returns:
        Elongation to fracture as fraction (e.g. 0.25 for 25%).
    """
    return PLASTICITY_DATA[material_key]['elongation_pct'] / 100.0


def reduction_in_area(material_key):
    """Estimated reduction in area at fracture (fraction).

    APPROXIMATION: RA ≈ 1 - exp(-elongation_pct/100 × 1.5)
    This empirical relation works reasonably for metals.
    True RA requires tensile test measurement.

    Args:
        material_key: key into PLASTICITY_DATA

    Returns:
        Reduction in area (dimensionless, 0 to 1).
    """
    elong = total_elongation(material_key)
    if elong <= 0:
        return 0.0
    return 1.0 - math.exp(-elong * 1.5)


# ── True Stress-Strain Curve ─────────────────────────────────────

def stress_strain_curve(material_key, max_strain=None, steps=100, sigma=SIGMA_HERE):
    """Generate a full true stress-strain curve.

    Elastic region: σ = E × ε
    Plastic region: Hollomon σ = K' × ε_p^n'

    Returns points up to necking (or max_strain if specified).

    Args:
        material_key: key into PLASTICITY_DATA
        max_strain: maximum strain to compute (default: necking strain × 1.2)
        steps: number of points
        sigma: σ-field value

    Returns:
        List of dicts with strain, stress_pa, is_plastic, is_necking.
    """
    data = PLASTICITY_DATA[material_key]
    E = youngs_modulus(material_key, sigma)
    sy = yield_stress(material_key, sigma)
    n = data['n_hardening']
    eps_y = sy / E  # yield strain

    if max_strain is None:
        eps_neck = necking_strain(material_key)
        max_strain = max(eps_neck * 1.2, eps_y * 5.0)

    if steps < 2:
        steps = 2

    curve = []
    for i in range(steps + 1):
        eps = max_strain * i / steps
        eps_p = max(eps - eps_y, 0.0)

        if eps <= eps_y or n <= 0:
            # Elastic region (or brittle material)
            stress = E * eps
            is_plastic = False
        else:
            # Plastic: Hollomon
            stress = hollomon_stress(material_key, eps_p, sigma)
            is_plastic = True

        curve.append({
            'strain': eps,
            'stress_pa': stress,
            'is_plastic': is_plastic,
            'is_necking': is_plastic and n > 0 and eps_p >= n,
        })

    return curve


# ── Work Hardening Rate ──────────────────────────────────────────

def work_hardening_rate(material_key, plastic_strain, sigma=SIGMA_HERE):
    """Instantaneous work hardening rate dσ/dε_p (Pa).

    For Hollomon: dσ/dε_p = K' × n × ε_p^(n-1) = n × σ / ε_p

    This decreases with strain — hardening saturates as dislocations
    accumulate and annihilation balances multiplication.

    Args:
        material_key: key into PLASTICITY_DATA
        plastic_strain: current plastic strain
        sigma: σ-field value

    Returns:
        Work hardening rate in Pa.
    """
    data = PLASTICITY_DATA[material_key]
    n = data['n_hardening']

    if n <= 0 or plastic_strain <= 0:
        return 0.0

    Kp = strength_coefficient(material_key, sigma)
    return Kp * n * (plastic_strain ** (n - 1.0))


# ── Plastic Work / Energy Absorption ─────────────────────────────

def plastic_work_density(material_key, plastic_strain, sigma=SIGMA_HERE):
    """Plastic work per unit volume up to given strain (J/m³).

    W = ∫₀^{ε_p} σ dε_p = ∫₀^{ε_p} K' ε^n dε = K' ε_p^(n+1) / (n+1)

    This is the energy absorbed by the material during plastic deformation.
    Related to toughness — the area under the stress-strain curve.

    Args:
        material_key: key into PLASTICITY_DATA
        plastic_strain: plastic strain reached
        sigma: σ-field value

    Returns:
        Plastic work density in J/m³.
    """
    if plastic_strain <= 0:
        return 0.0

    data = PLASTICITY_DATA[material_key]
    n = data['n_hardening']

    if n <= 0:
        # Perfectly plastic: W = σ_y × ε_p
        return yield_stress(material_key, sigma) * plastic_strain

    Kp = strength_coefficient(material_key, sigma)
    return Kp * (plastic_strain ** (n + 1.0)) / (n + 1.0)


def toughness_estimate(material_key, sigma=SIGMA_HERE):
    """Approximate material toughness — area under stress-strain curve (J/m³).

    Integrates from 0 to fracture strain.
    Toughness = elastic energy + plastic work.

    APPROXIMATION: uses Hollomon model up to elongation_pct.

    Args:
        material_key: key into PLASTICITY_DATA
        sigma: σ-field value

    Returns:
        Toughness in J/m³.
    """
    E = youngs_modulus(material_key, sigma)
    sy = yield_stress(material_key, sigma)

    # Elastic energy density
    elastic = sy * sy / (2.0 * E)

    # Plastic energy density (to fracture)
    elong = total_elongation(material_key)
    if elong <= 0:
        return elastic

    # Convert engineering strain to true strain for plastic part
    eps_y = sy / E
    eps_p_fracture = max(elong - eps_y, 0.0)

    plastic = plastic_work_density(material_key, eps_p_fracture, sigma)
    return elastic + plastic


# ── σ-field Coupling ─────────────────────────────────────────────

def sigma_yield_shift(material_key, sigma):
    """Ratio of yield stress at σ to yield stress at σ=0.

    σ_y(σ) / σ_y(0) = scale_ratio(σ)

    Yield stress scales with bond strength.

    Args:
        material_key: material key
        sigma: σ-field value

    Returns:
        Ratio (dimensionless). > 1 means stronger.
    """
    if sigma == SIGMA_HERE:
        return 1.0
    return scale_ratio(sigma)


# ── Nagatha Export ───────────────────────────────────────────────

def plasticity_properties(material_key, sigma=SIGMA_HERE):
    """Export plasticity properties in Nagatha-compatible format.

    Args:
        material_key: key into PLASTICITY_DATA
        sigma: σ-field value

    Returns:
        Dict with plasticity properties and origin tags.
    """
    data = PLASTICITY_DATA[material_key]
    sy = yield_stress(material_key, sigma)
    Kp = strength_coefficient(material_key, sigma)
    eps_neck = necking_strain(material_key)
    sig_neck = necking_stress(material_key, sigma)
    elong_u = uniform_elongation(material_key)
    elong_t = total_elongation(material_key)
    ra = reduction_in_area(material_key)
    tough = toughness_estimate(material_key, sigma)
    shift = sigma_yield_shift(material_key, sigma)

    return {
        'material': material_key,
        'sigma': sigma,
        'yield_stress_pa': sy,
        'strength_coefficient_pa': Kp,
        'n_hardening': data['n_hardening'],
        'necking_strain': eps_neck,
        'necking_stress_pa': sig_neck,
        'uniform_elongation': elong_u,
        'total_elongation': elong_t,
        'reduction_in_area': ra,
        'is_ductile': data['is_ductile'],
        'toughness_J_m3': tough,
        'sigma_yield_ratio': shift,
        'jc_A_pa': data['jc_A'],
        'jc_B_pa': data['jc_B'],
        'jc_n': data['jc_n'],
        'jc_C': data['jc_C'],
        'jc_m': data['jc_m'],
        'origin': (
            "Yield stress: MEASURED (0.2% offset, ASM Handbook). "
            "Hollomon: σ = K'ε^n (FIRST_PRINCIPLES form, MEASURED K', n). "
            "Ramberg-Osgood: ε = σ/E + (σ/K')^(1/n) (FIRST_PRINCIPLES). "
            "Johnson-Cook: (A+Bε^n)(1+Clnε̇)(1-T*^m) (MEASURED A,B,C,n,m). "
            "Considère necking: ε_neck = n (FIRST_PRINCIPLES). "
            "σ-coupling: CORE (□σ = −ξR → E_coh → σ_y, K')."
        ),
    }
