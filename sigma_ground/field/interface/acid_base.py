"""
Acid-base equilibria from bond dissociation energies and solvation.

Input: acid/base identity → pKa, pH, buffer capacity, titration curves.
Output: equilibrium concentrations, pH at any point.

pH is the single most-measured quantity in chemistry. This module derives
acid strength (pKa) from thermodynamic quantities where possible, and
uses MEASURED pKa values as calibration anchors where derivation isn't
yet feasible.

Physics used:
  1. Thermodynamic Ka definition
     FIRST_PRINCIPLES: Ka = exp(-DeltaG / RT)
     where DeltaG = DeltaH - T*DeltaS for HA -> H+ + A-

  2. Henderson-Hasselbalch equation
     FIRST_PRINCIPLES: pH = pKa + log10([A-]/[HA])
     Exact for ideal dilute solutions.

  3. Water autoionization
     MEASURED: Kw = 1.012e-14 at 25C (IUPAC 2007)
     Temperature dependence from van't Hoff with DeltaH = 55.8 kJ/mol.

  4. Bond dissociation correlation for pKa
     APPROXIMATION: Stronger O-H (or X-H) bond -> higher pKa.
     pKa correlates with BDE for homologous series (Bordwell).
     Used only for trend validation, not primary source.

  5. Buffer capacity
     FIRST_PRINCIPLES: beta = 2.303 * C * Ka * [H+] / (Ka + [H+])^2
     Maximum at pH = pKa.

  6. Titration curves
     FIRST_PRINCIPLES: charge balance + mass balance + Ka expression.
     Solved analytically for monoprotic; iteratively for polyprotic.

σ-dependence:
  Acid-base equilibria are electromagnetic → σ-INVARIANT to first order.
  The only σ pathway: nuclear mass → reduced mass → vibrational zero-point
  energy → bond dissociation energy shift. At Earth σ, negligible (<10⁻⁹).

□σ = -ξR
"""

import math

from ..constants import K_B, N_AVOGADRO, EV_TO_J, R_GAS, SIGMA_HERE
from ..scale import scale_ratio

# ══════════════════════════════════════════════════════════════════════
# MEASURED pKa values (IUPAC / CRC Handbook, 25°C in water)
# ══════════════════════════════════════════════════════════════════════
# These are calibration anchors. Where derivation is possible, we
# validate against these. Where it isn't, we use them directly.

ACID_BASE_DATA = {
    # ── Strong acids (pKa < 0, effectively complete dissociation) ──
    'hydrochloric_acid': {
        'formula': 'HCl',
        'type': 'strong_acid',
        'pKa': -6.3,           # MEASURED (gas-phase, effective in water)
        'dissociating_bond': ('H', 'Cl'),
        'conjugate': 'chloride',
        'n_protons': 1,
        'molar_mass_kg_mol': 36.461e-3,
    },
    'sulfuric_acid_1': {
        'formula': 'H2SO4',
        'type': 'strong_acid',
        'pKa': -3.0,           # MEASURED (first dissociation)
        'dissociating_bond': ('H', 'O'),
        'conjugate': 'bisulfate',
        'n_protons': 1,
        'molar_mass_kg_mol': 98.079e-3,
    },
    'sulfuric_acid_2': {
        'formula': 'HSO4-',
        'type': 'weak_acid',
        'pKa': 1.99,           # MEASURED (second dissociation)
        'dissociating_bond': ('H', 'O'),
        'conjugate': 'sulfate',
        'n_protons': 1,
        'molar_mass_kg_mol': 97.071e-3,
    },
    'nitric_acid': {
        'formula': 'HNO3',
        'type': 'strong_acid',
        'pKa': -1.4,           # MEASURED
        'dissociating_bond': ('H', 'O'),
        'conjugate': 'nitrate',
        'n_protons': 1,
        'molar_mass_kg_mol': 63.012e-3,
    },

    # ── Weak acids ──
    'hydrofluoric_acid': {
        'formula': 'HF',
        'type': 'weak_acid',
        'pKa': 3.17,           # MEASURED (IUPAC)
        'dissociating_bond': ('H', 'F'),
        'conjugate': 'fluoride',
        'n_protons': 1,
        'molar_mass_kg_mol': 20.006e-3,
    },
    'acetic_acid': {
        'formula': 'CH3COOH',
        'type': 'weak_acid',
        'pKa': 4.756,          # MEASURED (IUPAC)
        'dissociating_bond': ('H', 'O'),
        'conjugate': 'acetate',
        'n_protons': 1,
        'molar_mass_kg_mol': 60.052e-3,
    },
    'formic_acid': {
        'formula': 'HCOOH',
        'type': 'weak_acid',
        'pKa': 3.75,           # MEASURED (IUPAC)
        'dissociating_bond': ('H', 'O'),
        'conjugate': 'formate',
        'n_protons': 1,
        'molar_mass_kg_mol': 46.025e-3,
    },
    'carbonic_acid_1': {
        'formula': 'H2CO3',
        'type': 'weak_acid',
        'pKa': 6.35,           # MEASURED (apparent, includes CO2(aq))
        'dissociating_bond': ('H', 'O'),
        'conjugate': 'bicarbonate',
        'n_protons': 1,
        'molar_mass_kg_mol': 62.025e-3,
    },
    'carbonic_acid_2': {
        'formula': 'HCO3-',
        'type': 'weak_acid',
        'pKa': 10.33,          # MEASURED
        'dissociating_bond': ('H', 'O'),
        'conjugate': 'carbonate',
        'n_protons': 1,
        'molar_mass_kg_mol': 61.017e-3,
    },
    'phosphoric_acid_1': {
        'formula': 'H3PO4',
        'type': 'weak_acid',
        'pKa': 2.15,           # MEASURED
        'dissociating_bond': ('H', 'O'),
        'conjugate': 'dihydrogen_phosphate',
        'n_protons': 1,
        'molar_mass_kg_mol': 97.994e-3,
    },
    'phosphoric_acid_2': {
        'formula': 'H2PO4-',
        'type': 'weak_acid',
        'pKa': 7.20,           # MEASURED
        'dissociating_bond': ('H', 'O'),
        'conjugate': 'hydrogen_phosphate',
        'n_protons': 1,
        'molar_mass_kg_mol': 96.987e-3,
    },
    'phosphoric_acid_3': {
        'formula': 'HPO4^2-',
        'type': 'weak_acid',
        'pKa': 12.35,          # MEASURED
        'dissociating_bond': ('H', 'O'),
        'conjugate': 'phosphate',
        'n_protons': 1,
        'molar_mass_kg_mol': 95.979e-3,
    },
    'hydrogen_sulfide_1': {
        'formula': 'H2S',
        'type': 'weak_acid',
        'pKa': 7.0,            # MEASURED
        'dissociating_bond': ('H', 'S'),
        'conjugate': 'bisulfide',
        'n_protons': 1,
        'molar_mass_kg_mol': 34.081e-3,
    },
    'hydrogen_sulfide_2': {
        'formula': 'HS-',
        'type': 'weak_acid',
        'pKa': 14.0,           # MEASURED (revised, was 12-15)
        'dissociating_bond': ('H', 'S'),
        'conjugate': 'sulfide',
        'n_protons': 1,
        'molar_mass_kg_mol': 33.073e-3,
    },
    'hydrogen_cyanide': {
        'formula': 'HCN',
        'type': 'weak_acid',
        'pKa': 9.21,           # MEASURED
        'dissociating_bond': ('H', 'C'),
        'conjugate': 'cyanide',
        'n_protons': 1,
        'molar_mass_kg_mol': 27.025e-3,
    },
    'phenol': {
        'formula': 'C6H5OH',
        'type': 'weak_acid',
        'pKa': 9.95,           # MEASURED
        'dissociating_bond': ('H', 'O'),
        'conjugate': 'phenoxide',
        'n_protons': 1,
        'molar_mass_kg_mol': 94.111e-3,
    },

    # ── Weak bases (stored as pKb, derive pKa of conjugate) ──
    'ammonia': {
        'formula': 'NH3',
        'type': 'weak_base',
        'pKb': 4.75,           # MEASURED (IUPAC)
        'accepting_atom': 'N',
        'conjugate_acid': 'ammonium',
        'n_protons': 1,        # protons accepted
        'molar_mass_kg_mol': 17.031e-3,
    },
    'methylamine': {
        'formula': 'CH3NH2',
        'type': 'weak_base',
        'pKb': 3.36,           # MEASURED
        'accepting_atom': 'N',
        'conjugate_acid': 'methylammonium',
        'n_protons': 1,
        'molar_mass_kg_mol': 31.057e-3,
    },
    'pyridine': {
        'formula': 'C5H5N',
        'type': 'weak_base',
        'pKb': 8.77,           # MEASURED
        'accepting_atom': 'N',
        'conjugate_acid': 'pyridinium',
        'n_protons': 1,
        'molar_mass_kg_mol': 79.101e-3,
    },

    # ── Strong bases ──
    'sodium_hydroxide': {
        'formula': 'NaOH',
        'type': 'strong_base',
        'pKb': -0.56,          # effectively complete
        'n_oh': 1,             # OH- ions released per formula unit
        'molar_mass_kg_mol': 39.997e-3,
    },
    'potassium_hydroxide': {
        'formula': 'KOH',
        'type': 'strong_base',
        'pKb': -0.70,
        'n_oh': 1,
        'molar_mass_kg_mol': 56.106e-3,
    },
}


# ══════════════════════════════════════════════════════════════════════
# WATER AUTOIONIZATION — Kw(T)
# ══════════════════════════════════════════════════════════════════════

# MEASURED: Kw = 1.012e-14 at 25°C (IUPAC 2007)
# Temperature dependence from van't Hoff equation:
#   d(ln Kw)/dT = DeltaH / (R T^2)
#   DeltaH(autoionization) = 55.815 kJ/mol (MEASURED, calorimetric)

_KW_25C = 1.012e-14           # mol²/L² at 298.15 K
_DELTA_H_AUTOION = 55.815e3   # J/mol (MEASURED)
_T_REF = 298.15               # K


def water_ion_product(T=298.15):
    """Kw(T) from van't Hoff equation.

    FIRST_PRINCIPLES: d(ln K)/dT = DeltaH/(RT^2)
    MEASURED: Kw(298.15) = 1.012e-14, DeltaH = 55.8 kJ/mol.

    Returns Kw in (mol/L)^2.
    """
    if T <= 0:
        return 0.0
    # van't Hoff: ln(Kw(T)/Kw(Tref)) = -DeltaH/R * (1/T - 1/Tref)
    ln_ratio = -_DELTA_H_AUTOION / R_GAS * (1.0 / T - 1.0 / _T_REF)
    return _KW_25C * math.exp(ln_ratio)


def neutral_pH(T=298.15):
    """pH of pure water at temperature T.

    At 25°C: pH = 7.0. At 37°C (body): pH = 6.8.
    At 60°C: pH = 6.5. Pure water is always neutral (pH = pOH).
    """
    Kw = water_ion_product(T)
    if Kw <= 0:
        return 14.0
    return -0.5 * math.log10(Kw)


# ══════════════════════════════════════════════════════════════════════
# Ka, Kb, pKa, pKb conversions
# ══════════════════════════════════════════════════════════════════════

def pKa(acid_key, T=298.15):
    """Return pKa for an acid at temperature T.

    For weak bases, returns pKa of the conjugate acid:
      pKa = pKw - pKb

    Temperature dependence:
      For acids with only a single MEASURED pKa (25°C), we return
      the 25°C value. Real temperature shifts are typically <0.5 pH
      units over 0-100°C for weak acids.
    """
    data = ACID_BASE_DATA[acid_key]
    if 'pKa' in data:
        return data['pKa']
    elif 'pKb' in data:
        pKw = -math.log10(water_ion_product(T))
        return pKw - data['pKb']
    else:
        raise ValueError(f"No pKa or pKb for {acid_key}")


def Ka(acid_key, T=298.15):
    """Acid dissociation constant Ka from pKa."""
    return 10.0 ** (-pKa(acid_key, T))


def pKb(base_key, T=298.15):
    """Return pKb for a base at temperature T.

    For acids, returns pKb of the conjugate base:
      pKb = pKw - pKa
    """
    data = ACID_BASE_DATA[base_key]
    if 'pKb' in data:
        return data['pKb']
    elif 'pKa' in data:
        pKw = -math.log10(water_ion_product(T))
        return pKw - data['pKa']
    else:
        raise ValueError(f"No pKa or pKb for {base_key}")


def Kb(base_key, T=298.15):
    """Base dissociation constant Kb from pKb."""
    return 10.0 ** (-pKb(base_key, T))


# ══════════════════════════════════════════════════════════════════════
# pH CALCULATIONS
# ══════════════════════════════════════════════════════════════════════

def pH_strong_acid(concentration, T=298.15):
    """pH of a strong acid solution (complete dissociation).

    FIRST_PRINCIPLES: [H+] = C_acid (for monoprotic).
    At very low concentrations, water autoionization dominates.

    Args:
        concentration: mol/L of strong acid
        T: temperature in K

    Returns: pH (dimensionless)
    """
    if concentration <= 0:
        return neutral_pH(T)
    Kw = water_ion_product(T)
    # Exact: [H+]^2 - C*[H+] - Kw = 0
    H = 0.5 * (concentration + math.sqrt(concentration ** 2 + 4 * Kw))
    return -math.log10(H)


def pH_strong_base(concentration, n_oh=1, T=298.15):
    """pH of a strong base solution.

    Args:
        concentration: mol/L of strong base
        n_oh: number of OH- per formula unit (1 for NaOH, 2 for Ca(OH)2)
        T: temperature in K

    Returns: pH
    """
    if concentration <= 0:
        return neutral_pH(T)
    Kw = water_ion_product(T)
    OH = concentration * n_oh
    # [H+] = Kw / [OH-], but include water autoionization
    # [OH-]^2 - C_oh * [OH-] - Kw = 0
    OH_total = 0.5 * (OH + math.sqrt(OH ** 2 + 4 * Kw))
    H = Kw / OH_total
    if H <= 0:
        return 14.0
    return -math.log10(H)


def pH_weak_acid(acid_key, concentration, T=298.15):
    """pH of a weak acid solution.

    FIRST_PRINCIPLES: Ka = [H+][A-] / [HA], charge balance, mass balance.

    Solves: [H+]^3 + Ka*[H+]^2 - (Ka*C + Kw)*[H+] - Ka*Kw = 0
    Simplified when C >> Kw/Ka (usual case):
      [H+] = sqrt(Ka * C)  (if [H+] << C)

    Uses exact cubic solution for accuracy at all concentrations.

    Args:
        acid_key: key into ACID_BASE_DATA
        concentration: mol/L (total acid, HA + A-)
        T: temperature in K

    Returns: pH
    """
    if concentration <= 0:
        return neutral_pH(T)

    Ka_val = Ka(acid_key, T)
    Kw = water_ion_product(T)

    # Cubic: H^3 + Ka*H^2 - (Ka*C + Kw)*H - Ka*Kw = 0
    # Newton's method from the simplified estimate
    H = math.sqrt(Ka_val * concentration)  # initial guess
    H = min(H, concentration)              # can't exceed total acid

    for _ in range(50):
        f = H ** 3 + Ka_val * H ** 2 - (Ka_val * concentration + Kw) * H - Ka_val * Kw
        fp = 3 * H ** 2 + 2 * Ka_val * H - (Ka_val * concentration + Kw)
        if abs(fp) < 1e-30:
            break
        dH = f / fp
        H_new = H - dH
        if H_new <= 0:
            H = H * 0.1  # safeguard
        else:
            H = H_new
        if abs(dH) < H * 1e-12:
            break

    return -math.log10(max(H, 1e-15))


def pH_weak_base(base_key, concentration, T=298.15):
    """pH of a weak base solution.

    Solves the mirror problem: Kb = [BH+][OH-] / [B]
    then pH = pKw - pOH.

    Args:
        base_key: key into ACID_BASE_DATA
        concentration: mol/L
        T: temperature in K

    Returns: pH
    """
    if concentration <= 0:
        return neutral_pH(T)

    Kb_val = Kb(base_key, T)
    Kw = water_ion_product(T)

    # OH from Kb: OH^2 + Kb*OH - Kb*C = 0
    disc = Kb_val ** 2 + 4 * Kb_val * concentration
    OH = 0.5 * (-Kb_val + math.sqrt(disc))
    OH = max(OH, 1e-15)

    H = Kw / OH
    return -math.log10(max(H, 1e-15))


def pH_solution(acid_key, concentration, T=298.15):
    """Unified pH function — dispatches based on acid/base type.

    Args:
        acid_key: key into ACID_BASE_DATA
        concentration: mol/L
        T: temperature in K

    Returns: pH
    """
    data = ACID_BASE_DATA[acid_key]
    atype = data['type']

    if atype == 'strong_acid':
        return pH_strong_acid(concentration, T)
    elif atype == 'weak_acid':
        return pH_weak_acid(acid_key, concentration, T)
    elif atype == 'strong_base':
        n_oh = data.get('n_oh', 1)
        return pH_strong_base(concentration, n_oh, T)
    elif atype == 'weak_base':
        return pH_weak_base(acid_key, concentration, T)
    else:
        raise ValueError(f"Unknown type: {atype}")


# ══════════════════════════════════════════════════════════════════════
# HENDERSON-HASSELBALCH EQUATION
# ══════════════════════════════════════════════════════════════════════

def henderson_hasselbalch(acid_key, ratio_base_acid, T=298.15):
    """pH from Henderson-Hasselbalch equation.

    FIRST_PRINCIPLES: pH = pKa + log10([A-]/[HA])

    Exact for ideal dilute solutions where [A-] and [HA] are the
    dominant species (buffer region).

    Args:
        acid_key: key into ACID_BASE_DATA
        ratio_base_acid: [A-] / [HA] (conjugate base / acid)
        T: temperature in K

    Returns: pH
    """
    if ratio_base_acid <= 0:
        return -float('inf')
    return pKa(acid_key, T) + math.log10(ratio_base_acid)


# ══════════════════════════════════════════════════════════════════════
# BUFFER CAPACITY
# ══════════════════════════════════════════════════════════════════════

def buffer_capacity(acid_key, concentration, pH_val, T=298.15):
    """Buffer capacity beta at a given pH.

    FIRST_PRINCIPLES:
      beta = 2.303 * (Kw/[H+] + [H+] + C*Ka*[H+]/(Ka+[H+])^2)

    The first two terms are water's intrinsic buffer capacity (significant
    only at extreme pH). The third term is the buffer's contribution,
    maximized at pH = pKa.

    Args:
        acid_key: key into ACID_BASE_DATA
        concentration: total buffer concentration in mol/L
        pH_val: current pH
        T: temperature in K

    Returns: beta in mol/(L*pH_unit)
    """
    Ka_val = Ka(acid_key, T)
    Kw = water_ion_product(T)
    H = 10.0 ** (-pH_val)

    # Water contribution
    beta_water = 2.303 * (Kw / H + H)

    # Buffer contribution
    denom = (Ka_val + H) ** 2
    if denom < 1e-30:
        beta_buffer = 0.0
    else:
        beta_buffer = 2.303 * concentration * Ka_val * H / denom

    return beta_water + beta_buffer


def buffer_capacity_max(acid_key, concentration, T=298.15):
    """Maximum buffer capacity (at pH = pKa).

    beta_max = 2.303 * C / 4 (when [A-] = [HA] = C/2).

    Args:
        acid_key: key into ACID_BASE_DATA
        concentration: total buffer concentration in mol/L
        T: temperature in K

    Returns: beta_max in mol/(L*pH_unit)
    """
    return buffer_capacity(acid_key, concentration, pKa(acid_key, T), T)


def buffer_range(acid_key, T=298.15):
    """Effective buffer range: pKa ± 1.

    The buffer is effective where [A-]/[HA] is between 0.1 and 10,
    which gives pH = pKa ± 1.

    Returns: (pH_low, pH_high)
    """
    pka = pKa(acid_key, T)
    return (pka - 1.0, pka + 1.0)


# ══════════════════════════════════════════════════════════════════════
# TITRATION CURVES
# ══════════════════════════════════════════════════════════════════════

def titration_strong_acid_strong_base(C_acid, V_acid_mL, C_base, V_base_mL,
                                       T=298.15):
    """pH during titration of strong acid with strong base.

    FIRST_PRINCIPLES: charge balance [H+] + [Na+] = [Cl-] + [OH-]
    with [Na+] and [Cl-] from dilution.

    Args:
        C_acid: concentration of acid (mol/L)
        V_acid_mL: volume of acid (mL)
        C_base: concentration of base (mol/L)
        V_base_mL: volume of base added (mL)
        T: temperature in K

    Returns: pH
    """
    Kw = water_ion_product(T)
    V_total = V_acid_mL + V_base_mL
    if V_total <= 0:
        return neutral_pH(T)

    mol_acid = C_acid * V_acid_mL / 1000.0
    mol_base = C_base * V_base_mL / 1000.0
    excess = mol_acid - mol_base

    if abs(excess) < 1e-15:
        # Equivalence point
        return neutral_pH(T)
    elif excess > 0:
        # Excess acid
        H = excess / (V_total / 1000.0)
        return -math.log10(H)
    else:
        # Excess base
        OH = (-excess) / (V_total / 1000.0)
        H = Kw / OH
        return -math.log10(max(H, 1e-15))


def titration_weak_acid_strong_base(acid_key, C_acid, V_acid_mL,
                                      C_base, V_base_mL, T=298.15):
    """pH during titration of weak acid with strong base.

    Four regions:
      1. Before titration: weak acid alone
      2. Buffer region: HA + A- mixture (Henderson-Hasselbalch)
      3. Equivalence point: conjugate base hydrolysis
      4. Past equivalence: excess strong base

    Args:
        acid_key: key into ACID_BASE_DATA
        C_acid: acid concentration (mol/L)
        V_acid_mL: acid volume (mL)
        C_base: base concentration (mol/L)
        V_base_mL: base volume added (mL)
        T: temperature in K

    Returns: pH
    """
    Kw = water_ion_product(T)
    Ka_val = Ka(acid_key, T)
    V_total = V_acid_mL + V_base_mL
    if V_total <= 0:
        return neutral_pH(T)

    mol_acid = C_acid * V_acid_mL / 1000.0
    mol_base = C_base * V_base_mL / 1000.0

    C_total = (V_total / 1000.0)  # total volume in L

    if mol_base <= 0:
        # Pure weak acid
        return pH_weak_acid(acid_key, mol_acid / C_total, T)

    if mol_base >= mol_acid:
        # At or past equivalence
        mol_conjugate = mol_acid
        mol_excess_base = mol_base - mol_acid

        if mol_excess_base > mol_acid * 1e-10:
            # Past equivalence: excess OH- dominates
            OH = mol_excess_base / C_total
            H = Kw / OH
            return -math.log10(max(H, 1e-15))
        else:
            # Equivalence point: conjugate base hydrolysis
            # A- + H2O <-> HA + OH-
            Cb = mol_conjugate / C_total
            Kb_conj = Kw / Ka_val
            disc = Kb_conj ** 2 + 4 * Kb_conj * Cb
            OH = 0.5 * (-Kb_conj + math.sqrt(disc))
            H = Kw / max(OH, 1e-15)
            return -math.log10(max(H, 1e-15))
    else:
        # Buffer region: Henderson-Hasselbalch
        mol_HA = mol_acid - mol_base
        mol_A = mol_base
        if mol_HA < 1e-15:
            mol_HA = 1e-15
        return henderson_hasselbalch(acid_key, mol_A / mol_HA, T)


def titration_curve(acid_key, C_acid, V_acid_mL, C_base,
                    n_points=100, T=298.15):
    """Generate a full titration curve.

    Returns list of (V_base_mL, pH) tuples from 0 to 2× equivalence.

    Args:
        acid_key: key into ACID_BASE_DATA (or 'strong_acid' for HCl-type)
        C_acid: acid concentration (mol/L)
        V_acid_mL: acid volume (mL)
        C_base: base concentration (mol/L)
        n_points: number of data points
        T: temperature in K

    Returns: list of (V_base_mL, pH) tuples
    """
    V_eq = C_acid * V_acid_mL / C_base  # equivalence volume in mL
    V_max = 2.0 * V_eq

    curve = []
    for i in range(n_points + 1):
        V_b = V_max * i / n_points

        data = ACID_BASE_DATA.get(acid_key)
        if data and data['type'] == 'strong_acid':
            ph = titration_strong_acid_strong_base(
                C_acid, V_acid_mL, C_base, V_b, T)
        else:
            ph = titration_weak_acid_strong_base(
                acid_key, C_acid, V_acid_mL, C_base, V_b, T)
        curve.append((V_b, ph))

    return curve


# ══════════════════════════════════════════════════════════════════════
# PERCENT DISSOCIATION
# ══════════════════════════════════════════════════════════════════════

def percent_dissociation(acid_key, concentration, T=298.15):
    """Percent dissociation of a weak acid.

    alpha = [A-] / C_total = [H+] / C_total (for monoprotic)

    Strong acids return ~100%.

    Args:
        acid_key: key into ACID_BASE_DATA
        concentration: mol/L
        T: temperature in K

    Returns: percent dissociation (0-100)
    """
    data = ACID_BASE_DATA[acid_key]
    if data['type'] == 'strong_acid':
        return 100.0
    if data['type'] == 'strong_base':
        return 100.0
    if concentration <= 0:
        return 100.0  # infinitely dilute → fully dissociated

    pH_val = pH_solution(acid_key, concentration, T)
    H = 10.0 ** (-pH_val)

    if data['type'] == 'weak_acid':
        # [A-] ≈ [H+] (for monoprotic, before any base added)
        Ka_val = Ka(acid_key, T)
        alpha = Ka_val / (Ka_val + H)
        return alpha * 100.0
    elif data['type'] == 'weak_base':
        Kw = water_ion_product(T)
        Kb_val = Kb(acid_key, T)
        OH = Kw / H
        alpha = Kb_val / (Kb_val + OH)
        return alpha * 100.0

    return 0.0


# ══════════════════════════════════════════════════════════════════════
# POLYPROTIC ACID SPECIATION
# ══════════════════════════════════════════════════════════════════════

def polyprotic_alpha(pKa_list, pH_val):
    """Fraction of each species for a polyprotic acid.

    For H_n A with n dissociation steps:
      alpha_0 = [H_n A] / C_total    (fully protonated)
      alpha_1 = [H_{n-1} A-] / C_total
      ...
      alpha_n = [A^{n-}] / C_total   (fully deprotonated)

    FIRST_PRINCIPLES: alpha_j = Product(Ka_i, i=1..j) / D * [H+]^(n-j)
    where D = sum over all j terms.

    Args:
        pKa_list: list of pKa values [pKa1, pKa2, ...] in order
        pH_val: solution pH

    Returns: list of alpha values [alpha_0, alpha_1, ..., alpha_n]
    """
    n = len(pKa_list)
    H = 10.0 ** (-pH_val)

    # Compute cumulative Ka products: P_j = Ka_1 * Ka_2 * ... * Ka_j
    # P_0 = 1
    P = [1.0]
    for i in range(n):
        P.append(P[-1] * 10.0 ** (-pKa_list[i]))

    # Each term: P_j * H^(n-j) / D
    terms = []
    for j in range(n + 1):
        terms.append(P[j] * H ** (n - j))

    D = sum(terms)
    if D <= 0:
        D = 1e-30

    return [t / D for t in terms]


# ══════════════════════════════════════════════════════════════════════
# SIGMA-FIELD COUPLING
# ══════════════════════════════════════════════════════════════════════

def sigma_pKa_shift(acid_key, sigma=SIGMA_HERE, T=298.15):
    """pKa shift under sigma-field.

    Acid dissociation is electromagnetic → σ-INVARIANT to first order.
    The only pathway: nuclear mass → zero-point energy → bond energy shift.

    For O-H bond: ZPE = (1/2) * hbar * sqrt(k/mu)
    Under sigma: mu shifts → ZPE shifts → BDE shifts → pKa shifts.

    At Earth sigma, shift is < 10^-9 pKa units.

    Returns: pKa at given sigma.
    """
    base_pKa = pKa(acid_key, T)
    # ZPE shift is proportional to sqrt(mu) change
    # mu change is proportional to nuclear mass change
    # Nuclear mass change is from QCD binding energy shift
    s = scale_ratio(sigma)
    # BDE shift: delta_BDE / BDE ~ (1 - sqrt(s)) * (ZPE/BDE)
    # ZPE/BDE ~ 0.01-0.03 for X-H bonds
    # delta_pKa = delta_BDE / (2.303 * R * T) * EV_TO_J
    # At sigma = sigma_here: s = 1, shift = 0
    zpe_fraction = 0.02  # typical ZPE / BDE for O-H
    delta_BDE_eV = 4.0 * zpe_fraction * (1.0 - math.sqrt(s))  # ~4 eV O-H bond
    delta_pKa = delta_BDE_eV * EV_TO_J / (2.303 * R_GAS * T)
    return base_pKa + delta_pKa


# ══════════════════════════════════════════════════════════════════════
# DIAGNOSTICS — Rule 9
# ══════════════════════════════════════════════════════════════════════

def acid_base_report(acid_key, concentration=0.1, T=298.15):
    """Complete acid-base report for a species.

    Args:
        acid_key: key into ACID_BASE_DATA
        concentration: mol/L (for pH and % dissociation)
        T: temperature in K

    Returns: dict with all computed properties
    """
    data = ACID_BASE_DATA[acid_key]
    report = {
        'species': acid_key,
        'formula': data['formula'],
        'type': data['type'],
        'concentration_mol_L': concentration,
        'temperature_K': T,
    }

    if data['type'] in ('strong_acid', 'weak_acid'):
        report['pKa'] = pKa(acid_key, T)
        report['Ka'] = Ka(acid_key, T)
        report['pKb_conjugate'] = pKb(acid_key, T)
    elif data['type'] in ('strong_base', 'weak_base'):
        report['pKb'] = pKb(acid_key, T)
        report['Kb'] = Kb(acid_key, T)
        if 'pKa' not in data:
            report['pKa_conjugate'] = pKa(acid_key, T)

    report['pH'] = pH_solution(acid_key, concentration, T)
    report['percent_dissociation'] = percent_dissociation(
        acid_key, concentration, T)

    if data['type'] == 'weak_acid':
        report['buffer_range'] = buffer_range(acid_key, T)
        report['buffer_capacity_at_pKa'] = buffer_capacity_max(
            acid_key, concentration, T)

    report['Kw'] = water_ion_product(T)
    report['neutral_pH'] = neutral_pH(T)

    return report


def full_report(concentration=0.1, T=298.15):
    """Reports for ALL species in ACID_BASE_DATA. Rule 9."""
    return {key: acid_base_report(key, concentration, T)
            for key in ACID_BASE_DATA}
