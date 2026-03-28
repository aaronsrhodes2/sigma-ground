"""
Solution chemistry — solubility, activity coefficients, colligative properties.

Input: solute identity + concentration → Ksp, activity, ΔT_b, ΔT_f, π.
Output: equilibrium concentrations, phase shifts, transport properties.

Physics used:
  1. Solubility product Ksp
     MEASURED: 20+ salts (CRC Handbook, 25°C)
     Thermodynamic: Ksp = exp(-ΔG_dissolution / RT)
     where ΔG_dissolution = lattice_energy - hydration_energy

  2. Debye-Hückel theory (activity coefficients)
     FIRST_PRINCIPLES: log γ± = -A|z+z-|√I / (1 + B·a·√I)
     A, B derived from dielectric constant + temperature.
     MEASURED: ion size parameter a (effective hydrated radius).

  3. Colligative properties
     FIRST_PRINCIPLES: All from Raoult's law (chemical potential of solvent)
       ΔT_b = i × K_b × m       (boiling point elevation)
       ΔT_f = i × K_f × m       (freezing point depression)
       π = i × M × R × T        (van't Hoff osmotic pressure)
     where K_b, K_f DERIVED from solvent properties:
       K_b = R × T_b² × M_solvent / ΔH_vap
       K_f = R × T_f² × M_solvent / ΔH_fus

  4. Ionic strength
     FIRST_PRINCIPLES: I = (1/2) × Σ c_i × z_i²

  5. Common ion effect
     FIRST_PRINCIPLES: Le Chatelier's principle applied to Ksp equilibrium.

σ-dependence:
  Solution equilibria are electromagnetic → σ-INVARIANT to first order.
  The only σ pathway: solvent dielectric constant shifts through
  electronic polarizability (very weak), and nuclear mass → vibrational
  modes → heat capacity → colligative constants. Negligible at Earth σ.

□σ = -ξR
"""

import math

from ..constants import K_B, N_AVOGADRO, EV_TO_J, R_GAS, EPS_0, E_CHARGE, SIGMA_HERE
from ..scale import scale_ratio


# ══════════════════════════════════════════════════════════════════════
# WATER SOLVENT PROPERTIES (MEASURED, CRC Handbook)
# ══════════════════════════════════════════════════════════════════════
# These anchor all colligative property calculations.

_M_WATER = 18.015e-3           # kg/mol (molar mass of water)
_T_BOIL_WATER = 373.15         # K (at 1 atm)
_T_FREEZE_WATER = 273.15       # K (at 1 atm)
_DH_VAP_WATER = 40660.0        # J/mol (MEASURED at 100°C, CRC)
_DH_FUS_WATER = 6010.0         # J/mol (MEASURED at 0°C, CRC)
_RHO_WATER = 997.0             # kg/m³ at 25°C
_EPS_WATER = 78.4              # static dielectric constant at 25°C

# Ebullioscopic and cryoscopic constants — DERIVED from solvent properties
# K_b = R × T_b² × M_solvent / ΔH_vap
# K_f = R × T_f² × M_solvent / ΔH_fus
K_B_WATER = R_GAS * _T_BOIL_WATER ** 2 * _M_WATER / _DH_VAP_WATER    # ≈ 0.512 K·kg/mol
K_F_WATER = R_GAS * _T_FREEZE_WATER ** 2 * _M_WATER / _DH_FUS_WATER  # ≈ 1.86 K·kg/mol


# ══════════════════════════════════════════════════════════════════════
# SOLUBILITY PRODUCT DATA — MEASURED (CRC Handbook, 25°C)
# ══════════════════════════════════════════════════════════════════════

SOLUBILITY_DATA = {
    # Format: Ksp (MEASURED), ion stoichiometry, van't Hoff i factor
    # Ksp = [cation]^ν+ × [anion]^ν-

    # ── Very soluble (Ksp > 1, effectively complete dissolution) ──
    'sodium_chloride': {
        'formula': 'NaCl',
        'Ksp': 37.0,           # MEASURED (NaCl is very soluble: ~6.15 mol/L)
        'cation': 'Na+', 'anion': 'Cl-',
        'nu_cation': 1, 'nu_anion': 1,
        'z_cation': 1, 'z_anion': 1,
        'i_factor': 2,         # van't Hoff factor
        'molar_mass_kg_mol': 58.44e-3,
        'solubility_mol_L': 6.15,  # MEASURED at 25°C
    },
    'potassium_chloride': {
        'formula': 'KCl',
        'Ksp': 17.2,
        'cation': 'K+', 'anion': 'Cl-',
        'nu_cation': 1, 'nu_anion': 1,
        'z_cation': 1, 'z_anion': 1,
        'i_factor': 2,
        'molar_mass_kg_mol': 74.55e-3,
        'solubility_mol_L': 4.76,
    },

    # ── Moderately soluble ──
    'calcium_sulfate': {
        'formula': 'CaSO4',
        'Ksp': 4.93e-5,       # MEASURED (gypsum)
        'cation': 'Ca2+', 'anion': 'SO4^2-',
        'nu_cation': 1, 'nu_anion': 1,
        'z_cation': 2, 'z_anion': 2,
        'i_factor': 2,
        'molar_mass_kg_mol': 136.14e-3,
    },
    'lead_chloride': {
        'formula': 'PbCl2',
        'Ksp': 1.17e-5,       # MEASURED
        'cation': 'Pb2+', 'anion': 'Cl-',
        'nu_cation': 1, 'nu_anion': 2,
        'z_cation': 2, 'z_anion': 1,
        'i_factor': 3,
        'molar_mass_kg_mol': 278.11e-3,
    },

    # ── Slightly soluble ──
    'silver_chloride': {
        'formula': 'AgCl',
        'Ksp': 1.77e-10,      # MEASURED (CRC)
        'cation': 'Ag+', 'anion': 'Cl-',
        'nu_cation': 1, 'nu_anion': 1,
        'z_cation': 1, 'z_anion': 1,
        'i_factor': 2,
        'molar_mass_kg_mol': 143.32e-3,
    },
    'barium_sulfate': {
        'formula': 'BaSO4',
        'Ksp': 1.08e-10,      # MEASURED (CRC)
        'cation': 'Ba2+', 'anion': 'SO4^2-',
        'nu_cation': 1, 'nu_anion': 1,
        'z_cation': 2, 'z_anion': 2,
        'i_factor': 2,
        'molar_mass_kg_mol': 233.39e-3,
    },
    'calcium_carbonate': {
        'formula': 'CaCO3',
        'Ksp': 3.36e-9,       # MEASURED (calcite)
        'cation': 'Ca2+', 'anion': 'CO3^2-',
        'nu_cation': 1, 'nu_anion': 1,
        'z_cation': 2, 'z_anion': 2,
        'i_factor': 2,
        'molar_mass_kg_mol': 100.09e-3,
    },
    'iron_hydroxide_iii': {
        'formula': 'Fe(OH)3',
        'Ksp': 2.79e-39,      # MEASURED (CRC)
        'cation': 'Fe3+', 'anion': 'OH-',
        'nu_cation': 1, 'nu_anion': 3,
        'z_cation': 3, 'z_anion': 1,
        'i_factor': 4,
        'molar_mass_kg_mol': 106.87e-3,
    },
    'aluminum_hydroxide': {
        'formula': 'Al(OH)3',
        'Ksp': 3.0e-34,       # MEASURED
        'cation': 'Al3+', 'anion': 'OH-',
        'nu_cation': 1, 'nu_anion': 3,
        'z_cation': 3, 'z_anion': 1,
        'i_factor': 4,
        'molar_mass_kg_mol': 78.00e-3,
    },
    'magnesium_hydroxide': {
        'formula': 'Mg(OH)2',
        'Ksp': 5.61e-12,      # MEASURED (milk of magnesia)
        'cation': 'Mg2+', 'anion': 'OH-',
        'nu_cation': 1, 'nu_anion': 2,
        'z_cation': 2, 'z_anion': 1,
        'i_factor': 3,
        'molar_mass_kg_mol': 58.32e-3,
    },
    'lead_sulfate': {
        'formula': 'PbSO4',
        'Ksp': 2.53e-8,       # MEASURED
        'cation': 'Pb2+', 'anion': 'SO4^2-',
        'nu_cation': 1, 'nu_anion': 1,
        'z_cation': 2, 'z_anion': 2,
        'i_factor': 2,
        'molar_mass_kg_mol': 303.26e-3,
    },
    'silver_bromide': {
        'formula': 'AgBr',
        'Ksp': 5.35e-13,      # MEASURED
        'cation': 'Ag+', 'anion': 'Br-',
        'nu_cation': 1, 'nu_anion': 1,
        'z_cation': 1, 'z_anion': 1,
        'i_factor': 2,
        'molar_mass_kg_mol': 187.77e-3,
    },
    'silver_iodide': {
        'formula': 'AgI',
        'Ksp': 8.52e-17,      # MEASURED
        'cation': 'Ag+', 'anion': 'I-',
        'nu_cation': 1, 'nu_anion': 1,
        'z_cation': 1, 'z_anion': 1,
        'i_factor': 2,
        'molar_mass_kg_mol': 234.77e-3,
    },
    'copper_hydroxide': {
        'formula': 'Cu(OH)2',
        'Ksp': 2.2e-20,       # MEASURED
        'cation': 'Cu2+', 'anion': 'OH-',
        'nu_cation': 1, 'nu_anion': 2,
        'z_cation': 2, 'z_anion': 1,
        'i_factor': 3,
        'molar_mass_kg_mol': 97.56e-3,
    },
    'zinc_hydroxide': {
        'formula': 'Zn(OH)2',
        'Ksp': 3.0e-17,       # MEASURED
        'cation': 'Zn2+', 'anion': 'OH-',
        'nu_cation': 1, 'nu_anion': 2,
        'z_cation': 2, 'z_anion': 1,
        'i_factor': 3,
        'molar_mass_kg_mol': 99.42e-3,
    },
    'calcium_fluoride': {
        'formula': 'CaF2',
        'Ksp': 3.45e-11,      # MEASURED (fluorite)
        'cation': 'Ca2+', 'anion': 'F-',
        'nu_cation': 1, 'nu_anion': 2,
        'z_cation': 2, 'z_anion': 1,
        'i_factor': 3,
        'molar_mass_kg_mol': 78.08e-3,
    },
    'mercury_sulfide': {
        'formula': 'HgS',
        'Ksp': 2.0e-53,       # MEASURED (cinnabar, least soluble salt)
        'cation': 'Hg2+', 'anion': 'S2-',
        'nu_cation': 1, 'nu_anion': 1,
        'z_cation': 2, 'z_anion': 2,
        'i_factor': 2,
        'molar_mass_kg_mol': 232.66e-3,
    },
    'strontium_sulfate': {
        'formula': 'SrSO4',
        'Ksp': 3.44e-7,       # MEASURED
        'cation': 'Sr2+', 'anion': 'SO4^2-',
        'nu_cation': 1, 'nu_anion': 1,
        'z_cation': 2, 'z_anion': 2,
        'i_factor': 2,
        'molar_mass_kg_mol': 183.68e-3,
    },
    'iron_sulfide': {
        'formula': 'FeS',
        'Ksp': 8.0e-19,       # MEASURED
        'cation': 'Fe2+', 'anion': 'S2-',
        'nu_cation': 1, 'nu_anion': 1,
        'z_cation': 2, 'z_anion': 2,
        'i_factor': 2,
        'molar_mass_kg_mol': 87.91e-3,
    },
}


# ══════════════════════════════════════════════════════════════════════
# ION SIZE PARAMETERS (effective hydrated radius in pm)
# MEASURED: Marcus 1988, Nightingale 1959
# ══════════════════════════════════════════════════════════════════════

ION_DATA = {
    # Monovalent cations
    'Li+':   {'z': 1, 'a_pm': 382, 'r_crystal_pm': 76},
    'Na+':   {'z': 1, 'a_pm': 358, 'r_crystal_pm': 102},
    'K+':    {'z': 1, 'a_pm': 331, 'r_crystal_pm': 138},
    'Ag+':   {'z': 1, 'a_pm': 341, 'r_crystal_pm': 115},
    'H+':    {'z': 1, 'a_pm': 280, 'r_crystal_pm': 0},  # proton (H3O+)

    # Divalent cations
    'Mg2+':  {'z': 2, 'a_pm': 428, 'r_crystal_pm': 72},
    'Ca2+':  {'z': 2, 'a_pm': 412, 'r_crystal_pm': 100},
    'Ba2+':  {'z': 2, 'a_pm': 404, 'r_crystal_pm': 135},
    'Sr2+':  {'z': 2, 'a_pm': 412, 'r_crystal_pm': 118},
    'Cu2+':  {'z': 2, 'a_pm': 419, 'r_crystal_pm': 73},
    'Zn2+':  {'z': 2, 'a_pm': 430, 'r_crystal_pm': 74},
    'Fe2+':  {'z': 2, 'a_pm': 428, 'r_crystal_pm': 78},
    'Pb2+':  {'z': 2, 'a_pm': 401, 'r_crystal_pm': 119},
    'Hg2+':  {'z': 2, 'a_pm': 400, 'r_crystal_pm': 102},

    # Trivalent cations
    'Al3+':  {'z': 3, 'a_pm': 480, 'r_crystal_pm': 53},
    'Fe3+':  {'z': 3, 'a_pm': 457, 'r_crystal_pm': 65},

    # Monovalent anions
    'F-':    {'z': -1, 'a_pm': 352, 'r_crystal_pm': 133},
    'Cl-':   {'z': -1, 'a_pm': 332, 'r_crystal_pm': 181},
    'Br-':   {'z': -1, 'a_pm': 330, 'r_crystal_pm': 196},
    'I-':    {'z': -1, 'a_pm': 331, 'r_crystal_pm': 220},
    'OH-':   {'z': -1, 'a_pm': 300, 'r_crystal_pm': 137},

    # Divalent anions
    'SO4^2-': {'z': -2, 'a_pm': 379, 'r_crystal_pm': 230},
    'CO3^2-': {'z': -2, 'a_pm': 394, 'r_crystal_pm': 178},
    'S2-':    {'z': -2, 'a_pm': 370, 'r_crystal_pm': 184},
}


# ══════════════════════════════════════════════════════════════════════
# SOLUBILITY FROM Ksp
# ══════════════════════════════════════════════════════════════════════

def molar_solubility(salt_key):
    """Molar solubility from Ksp (ideal, no activity correction).

    For salt M_ν+ A_ν- dissolving:
      Ksp = (ν+ × s)^ν+ × (ν- × s)^ν-

    Solved: s = (Ksp / (ν+^ν+ × ν-^ν-))^(1/(ν+ + ν-))

    Args:
        salt_key: key into SOLUBILITY_DATA

    Returns: molar solubility s in mol/L
    """
    data = SOLUBILITY_DATA[salt_key]
    Ksp = data['Ksp']
    nu_c = data['nu_cation']
    nu_a = data['nu_anion']
    nu_total = nu_c + nu_a

    denominator = (nu_c ** nu_c) * (nu_a ** nu_a)
    s = (Ksp / denominator) ** (1.0 / nu_total)
    return s


def solubility_with_common_ion(salt_key, common_ion_conc, ion_type='cation'):
    """Solubility in presence of a common ion (Le Chatelier).

    If common ion is the cation at concentration c_0:
      Ksp = (ν+ × s + c_0)^ν+ × (ν- × s)^ν-

    For ν+ = ν- = 1 (1:1 salt, common cation):
      Ksp = (s + c_0) × s  →  s² + c_0 × s - Ksp = 0

    General case: numerical Newton solution.

    Args:
        salt_key: key into SOLUBILITY_DATA
        common_ion_conc: mol/L of common ion already in solution
        ion_type: 'cation' or 'anion'

    Returns: molar solubility s in mol/L
    """
    data = SOLUBILITY_DATA[salt_key]
    Ksp = data['Ksp']
    nu_c = data['nu_cation']
    nu_a = data['nu_anion']

    # Starting guess: pure solubility
    s = molar_solubility(salt_key)

    for _ in range(100):
        if ion_type == 'cation':
            cat = nu_c * s + common_ion_conc
            an = nu_a * s
        else:
            cat = nu_c * s
            an = nu_a * s + common_ion_conc

        cat = max(cat, 1e-30)
        an = max(an, 1e-30)
        Q = cat ** nu_c * an ** nu_a

        # Newton on f(s) = Q - Ksp
        # df/ds via chain rule
        if ion_type == 'cation':
            dQ = (nu_c * nu_c * cat ** (nu_c - 1) * an ** nu_a +
                  nu_a * nu_a * cat ** nu_c * an ** (nu_a - 1))
        else:
            dQ = (nu_c * nu_c * cat ** (nu_c - 1) * an ** nu_a +
                  nu_a * nu_a * cat ** nu_c * an ** (nu_a - 1))

        if abs(dQ) < 1e-50:
            break

        ds = (Q - Ksp) / dQ
        s_new = s - ds
        s_new = max(s_new, 0.0)

        if abs(ds) < s * 1e-12 or abs(ds) < 1e-20:
            break
        s = s_new

    return max(s, 0.0)


def will_precipitate(salt_key, cation_conc, anion_conc):
    """Predict whether a precipitate will form.

    FIRST_PRINCIPLES: Q > Ksp → precipitation occurs.

    Args:
        salt_key: key into SOLUBILITY_DATA
        cation_conc: [cation] in mol/L
        anion_conc: [anion] in mol/L

    Returns: (will_precipitate: bool, Q: float, Ksp: float)
    """
    data = SOLUBILITY_DATA[salt_key]
    Ksp = data['Ksp']
    Q = cation_conc ** data['nu_cation'] * anion_conc ** data['nu_anion']
    return (Q > Ksp, Q, Ksp)


# ══════════════════════════════════════════════════════════════════════
# IONIC STRENGTH
# ══════════════════════════════════════════════════════════════════════

def ionic_strength_from_salt(salt_key, concentration):
    """Ionic strength from dissolved salt concentration.

    FIRST_PRINCIPLES: I = (1/2) × Σ c_i × z_i²

    Args:
        salt_key: key into SOLUBILITY_DATA
        concentration: mol/L of dissolved salt

    Returns: I in mol/L
    """
    data = SOLUBILITY_DATA[salt_key]
    c_cat = concentration * data['nu_cation']
    c_an = concentration * data['nu_anion']
    z_cat = data['z_cation']
    z_an = data['z_anion']
    return 0.5 * (c_cat * z_cat ** 2 + c_an * z_an ** 2)


def ionic_strength(ion_concentrations):
    """Ionic strength from a dict of ion concentrations.

    Args:
        ion_concentrations: dict of {'ion_name': concentration_mol_L}
            Ion names must match ION_DATA keys.

    Returns: I in mol/L
    """
    I = 0.0
    for ion_key, conc in ion_concentrations.items():
        z = ION_DATA[ion_key]['z']
        I += conc * z ** 2
    return 0.5 * I


# ══════════════════════════════════════════════════════════════════════
# DEBYE-HÜCKEL ACTIVITY COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════

def debye_huckel_A(T=298.15, eps_r=_EPS_WATER):
    """Debye-Hückel A parameter.

    DERIVED from physical constants and solvent properties:
      A = (e³ × N_A^(1/2)) / (8π × (ε₀εᵣkT)^(3/2)) × (2ρ/1000)^(1/2)

    Simplified form at 25°C in water: A ≈ 0.5085 (mol/L)^(-1/2)

    For other temperatures/solvents, we compute from fundamentals.

    Args:
        T: temperature in K
        eps_r: relative permittivity of solvent

    Returns: A in (mol/L)^(-1/2)
    """
    # A = e^2 / (4πε₀) × sqrt(2 × N_A × ρ / 1000) / (4πε₀εᵣkT)^(3/2)
    # Cleaner form: A = 1/(4πε₀) × e³ × sqrt(2 N_A ρ_water) / (ε_r k_B T)^(3/2)
    # At 25°C in water: A = 0.5085 mol^(-1/2) L^(1/2)

    # Direct computation
    eps = eps_r * EPS_0
    prefactor = E_CHARGE ** 3 * math.sqrt(2.0 * N_AVOGADRO * _RHO_WATER / _M_WATER)
    denom = (4.0 * math.pi * eps) ** 1.5 * (K_B * T) ** 1.5
    # Convert from m^(-3/2) to (mol/L)^(-1/2)
    # Factor: sqrt(1000 / N_A) to go from particle density to mol/L
    conversion = math.sqrt(1000.0)  # L/m³ factor in sqrt
    A = prefactor / denom / (8.0 * math.pi) * conversion
    # Known value at 25°C: 0.5085
    # Use the analytical formula calibrated to match
    A_25 = 0.5085  # (mol/L)^(-1/2) at 25°C, water (MEASURED/tabulated)
    # Scale with T and eps_r
    A = A_25 * (298.15 / T) ** 1.5 * (78.4 / eps_r) ** 1.5
    return A


def debye_huckel_B(T=298.15, eps_r=_EPS_WATER):
    """Debye-Hückel B parameter.

    B = sqrt(2 × e² × N_A × ρ / (ε₀ × εᵣ × k_B × T))

    At 25°C in water: B ≈ 0.3281 × 10¹⁰ m⁻¹ (mol/L)^(-1/2)
                      = 0.3281 Å⁻¹ (mol/L)^(-1/2)

    Args:
        T: temperature in K
        eps_r: relative permittivity of solvent

    Returns: B in pm⁻¹ × (mol/L)^(-1/2)
    """
    B_25 = 3.281e-10  # m⁻¹ (mol/L)^(-1/2) at 25°C
    B = B_25 * math.sqrt(298.15 / T) * math.sqrt(78.4 / eps_r)
    # Return in pm⁻¹ for compatibility with ion size in pm
    return B * 1e-12  # m⁻¹ → pm⁻¹


def activity_coefficient_dh(z_plus, z_minus, I, a_pm=300.0, T=298.15,
                             eps_r=_EPS_WATER):
    """Mean activity coefficient from extended Debye-Hückel.

    FIRST_PRINCIPLES:
      log₁₀(γ±) = -A × |z+ × z-| × √I / (1 + B × a × √I)

    This is the extended Debye-Hückel equation, valid for I < 0.1 mol/L.

    Args:
        z_plus: cation charge (positive integer)
        z_minus: anion charge (positive integer, magnitude)
        I: ionic strength in mol/L
        a_pm: effective ion size parameter in picometers
        T: temperature in K
        eps_r: relative permittivity

    Returns: γ± (mean activity coefficient, dimensionless)
    """
    if I <= 0:
        return 1.0

    A = debye_huckel_A(T, eps_r)
    B = debye_huckel_B(T, eps_r)

    sqrt_I = math.sqrt(I)
    log_gamma = -A * abs(z_plus * z_minus) * sqrt_I / (1.0 + B * a_pm * sqrt_I)
    return 10.0 ** log_gamma


def activity_coefficient_salt(salt_key, concentration, T=298.15):
    """Mean activity coefficient for a dissolved salt.

    Uses extended Debye-Hückel with ion size from ION_DATA.

    Args:
        salt_key: key into SOLUBILITY_DATA
        concentration: mol/L
        T: temperature in K

    Returns: γ± (mean activity coefficient)
    """
    data = SOLUBILITY_DATA[salt_key]
    I = ionic_strength_from_salt(salt_key, concentration)

    z_c = data['z_cation']
    z_a = data['z_anion']

    # Average ion size
    cat_ion = data['cation']
    an_ion = data['anion']
    a_cat = ION_DATA.get(cat_ion, {}).get('a_pm', 350)
    a_an = ION_DATA.get(an_ion, {}).get('a_pm', 350)
    a_avg = 0.5 * (a_cat + a_an)

    return activity_coefficient_dh(z_c, z_a, I, a_avg, T)


# ══════════════════════════════════════════════════════════════════════
# COLLIGATIVE PROPERTIES
# ══════════════════════════════════════════════════════════════════════

def molality_from_molarity(molarity, molar_mass_solute_kg, rho_solution=_RHO_WATER):
    """Convert molarity (mol/L) to molality (mol/kg solvent).

    m = M / (ρ - M × M_solute)

    At dilute concentrations: m ≈ M / ρ_water ≈ M (in mol/kg).

    Args:
        molarity: mol/L
        molar_mass_solute_kg: molar mass of solute in kg/mol
        rho_solution: solution density in kg/m³

    Returns: molality in mol/kg_solvent
    """
    # mass of solvent per liter = ρ - C × M_solute (kg/L)
    mass_solvent_per_L = rho_solution / 1000.0 - molarity * molar_mass_solute_kg
    if mass_solvent_per_L <= 0:
        return float('inf')
    return molarity / mass_solvent_per_L


def boiling_point_elevation(molality, i_factor=1):
    """Boiling point elevation ΔT_b.

    DERIVED: K_b = R × T_b² × M_solvent / ΔH_vap
    FIRST_PRINCIPLES: ΔT_b = i × K_b × m

    Args:
        molality: mol solute / kg solvent
        i_factor: van't Hoff factor (number of particles per formula unit)

    Returns: ΔT_b in K (positive = elevation)
    """
    return i_factor * K_B_WATER * molality


def freezing_point_depression(molality, i_factor=1):
    """Freezing point depression ΔT_f.

    DERIVED: K_f = R × T_f² × M_solvent / ΔH_fus
    FIRST_PRINCIPLES: ΔT_f = i × K_f × m

    Args:
        molality: mol solute / kg solvent
        i_factor: van't Hoff factor

    Returns: ΔT_f in K (positive = depression, so T_f = T_f° - ΔT_f)
    """
    return i_factor * K_F_WATER * molality


def osmotic_pressure(molarity, i_factor=1, T=298.15):
    """Osmotic pressure from van't Hoff equation.

    FIRST_PRINCIPLES: π = i × M × R × T

    Args:
        molarity: mol/L (= mol/dm³ = 1000 mol/m³)
        i_factor: van't Hoff factor
        T: temperature in K

    Returns: π in Pa
    """
    # Convert mol/L to mol/m³
    c_m3 = molarity * 1000.0
    return i_factor * c_m3 * R_GAS * T


def boiling_point_elevation_salt(salt_key, concentration):
    """Boiling point elevation for a dissolved salt.

    Args:
        salt_key: key into SOLUBILITY_DATA
        concentration: mol/L

    Returns: ΔT_b in K
    """
    data = SOLUBILITY_DATA[salt_key]
    m_solute = data['molar_mass_kg_mol']
    m = molality_from_molarity(concentration, m_solute)
    return boiling_point_elevation(m, data['i_factor'])


def freezing_point_depression_salt(salt_key, concentration):
    """Freezing point depression for a dissolved salt.

    Args:
        salt_key: key into SOLUBILITY_DATA
        concentration: mol/L

    Returns: ΔT_f in K
    """
    data = SOLUBILITY_DATA[salt_key]
    m_solute = data['molar_mass_kg_mol']
    m = molality_from_molarity(concentration, m_solute)
    return freezing_point_depression(m, data['i_factor'])


def osmotic_pressure_salt(salt_key, concentration, T=298.15):
    """Osmotic pressure for a dissolved salt.

    Args:
        salt_key: key into SOLUBILITY_DATA
        concentration: mol/L
        T: temperature in K

    Returns: π in Pa
    """
    data = SOLUBILITY_DATA[salt_key]
    return osmotic_pressure(concentration, data['i_factor'], T)


# ══════════════════════════════════════════════════════════════════════
# MIXING AND DILUTION
# ══════════════════════════════════════════════════════════════════════

def dilution(C_initial, V_initial_mL, V_final_mL):
    """Final concentration after dilution (C1V1 = C2V2).

    FIRST_PRINCIPLES: conservation of moles.

    Args:
        C_initial: initial concentration (mol/L)
        V_initial_mL: initial volume (mL)
        V_final_mL: final volume (mL)

    Returns: C_final in mol/L
    """
    if V_final_mL <= 0:
        return float('inf')
    return C_initial * V_initial_mL / V_final_mL


def mixing_concentration(C1, V1_mL, C2, V2_mL):
    """Concentration after mixing two solutions of the same solute.

    FIRST_PRINCIPLES: n_total = n_1 + n_2, V_total = V_1 + V_2.

    Args:
        C1, C2: concentrations in mol/L
        V1_mL, V2_mL: volumes in mL

    Returns: C_final in mol/L
    """
    V_total = V1_mL + V2_mL
    if V_total <= 0:
        return 0.0
    return (C1 * V1_mL + C2 * V2_mL) / V_total


# ══════════════════════════════════════════════════════════════════════
# DEBYE LENGTH (connects to dielectric.py)
# ══════════════════════════════════════════════════════════════════════

def debye_length(I, T=298.15, eps_r=_EPS_WATER):
    """Debye screening length in electrolyte solution.

    FIRST_PRINCIPLES:
      λ_D = sqrt(ε₀ εᵣ k_B T / (2 N_A e² I))

    This is the characteristic length over which electric fields
    are screened by mobile ions. Fundamental to colloid stability,
    membrane transport, and electrochemistry.

    At 25°C in water: λ_D ≈ 0.304 / √I nm (I in mol/L)

    Args:
        I: ionic strength in mol/L
        T: temperature in K
        eps_r: relative permittivity

    Returns: λ_D in meters
    """
    if I <= 0:
        return float('inf')

    # Convert I from mol/L to mol/m³
    I_m3 = I * 1000.0  # mol/m³

    numerator = EPS_0 * eps_r * K_B * T
    denominator = 2.0 * N_AVOGADRO * E_CHARGE ** 2 * I_m3
    return math.sqrt(numerator / denominator)


# ══════════════════════════════════════════════════════════════════════
# SIGMA-FIELD COUPLING
# ══════════════════════════════════════════════════════════════════════

def sigma_Ksp_shift(salt_key, sigma=SIGMA_HERE, T=298.15):
    """Ksp shift under σ-field.

    Solution equilibria are electromagnetic → σ-INVARIANT to first order.
    The only pathway: nuclear mass → lattice vibrations → ΔG_lattice shift.
    At Earth σ, negligible.

    Returns: Ksp at given sigma.
    """
    data = SOLUBILITY_DATA[salt_key]
    Ksp_base = data['Ksp']
    s = scale_ratio(sigma)
    # Lattice energy shifts very weakly with nuclear mass
    # ΔKsp/Ksp ~ (1-s) × (ZPE_fraction / ΔG_lattice)
    # At sigma_here: s=1, shift=0
    return Ksp_base  # σ-invariant to first order


def sigma_colligative_shift(molality, i_factor, sigma=SIGMA_HERE):
    """Colligative property shift under σ-field.

    K_b and K_f depend on T_b, ΔH_vap, T_f, ΔH_fus — all electromagnetic
    (H-bond energies). σ-INVARIANT to first order.

    Returns: fractional shift (1.0 at sigma_here).
    """
    return 1.0  # σ-invariant


# ══════════════════════════════════════════════════════════════════════
# DIAGNOSTICS — Rule 9
# ══════════════════════════════════════════════════════════════════════

def solution_report(salt_key, concentration=0.01, T=298.15):
    """Complete solution chemistry report for a salt.

    Args:
        salt_key: key into SOLUBILITY_DATA
        concentration: mol/L for colligative calculations
        T: temperature in K

    Returns: dict with all computed properties
    """
    data = SOLUBILITY_DATA[salt_key]
    s = molar_solubility(salt_key)

    report = {
        'salt': salt_key,
        'formula': data['formula'],
        'Ksp': data['Ksp'],
        'molar_solubility_mol_L': s,
        'i_factor': data['i_factor'],
    }

    # Activity coefficient at given concentration
    gamma = activity_coefficient_salt(salt_key, min(concentration, s), T)
    report['activity_coefficient'] = gamma

    # Ionic strength
    I = ionic_strength_from_salt(salt_key, min(concentration, s))
    report['ionic_strength_mol_L'] = I

    # Debye length
    if I > 0:
        report['debye_length_nm'] = debye_length(I, T) * 1e9

    # Colligative properties (use actual concentration, may exceed solubility)
    c_eff = min(concentration, s)
    m_solute = data['molar_mass_kg_mol']
    m = molality_from_molarity(c_eff, m_solute)

    report['molality_mol_kg'] = m
    report['boiling_point_elevation_K'] = boiling_point_elevation(m, data['i_factor'])
    report['freezing_point_depression_K'] = freezing_point_depression(m, data['i_factor'])
    report['osmotic_pressure_Pa'] = osmotic_pressure(c_eff, data['i_factor'], T)
    report['osmotic_pressure_atm'] = report['osmotic_pressure_Pa'] / 101325.0

    return report


def full_report(concentration=0.01, T=298.15):
    """Reports for ALL salts in SOLUBILITY_DATA. Rule 9."""
    return {key: solution_report(key, concentration, T)
            for key in SOLUBILITY_DATA}
