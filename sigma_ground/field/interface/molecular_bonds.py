"""
Molecular bond physics — deriving covalent bond properties from atomic data.

The foundation of the matter information cascade for organic chemistry.
From ~42 measured atomic properties (7 atoms × 6 each) plus 7 homonuclear
bond energies, this module derives ALL heteronuclear bond energies, lengths,
angles, polarities, and vibrational frequencies.

Derivation chains:

  1. Pauling Bond Energy (1932, FIRST_PRINCIPLES + MEASURED seeds)
     D(A-B) = ½(D(A-A) + D(B-B)) + (χ_A − χ_B)²

     Where:
       D(A-A), D(B-B) = homonuclear bond dissociation energies (MEASURED)
       χ_A, χ_B = Pauling electronegativities (MEASURED, from elements.json)

     The (Δχ)² term is the ionic resonance energy — extra stability from
     charge transfer in polar bonds. Pauling (1932) showed this predicts
     heteronuclear bond energies to ~10%.

     Bond energy is electromagnetic → σ-INVARIANT to first order.

  2. Schomaker-Stevenson Bond Length (1941, FIRST_PRINCIPLES)
     r(A-B) = r_A + r_B − c|χ_A − χ_B|

     Where:
       r_A, r_B = covalent radii (MEASURED)
       c = 9 pm (MEASURED, empirical constant)

     Electronegativity difference contracts the bond (charge transfer
     pulls atoms closer). Corrected for bond order:
       double ≈ 0.86 × single
       triple ≈ 0.78 × single

  3. VSEPR Molecular Geometry (Gillespie-Nyholm 1957, FIRST_PRINCIPLES)
     Electron domains (bonding pairs + lone pairs) arrange to minimize
     repulsion:
       2 domains → 180° (linear)
       3 domains → 120° (trigonal planar)
       4 domains → 109.5° (tetrahedral)

     Lone pairs compress bond angles by ~2.5° each (they occupy more
     angular space than bonding pairs).

     H₂O: 4 domains, 2 lone pairs → 109.5 − 2×2.5 = 104.5°

  4. Bond Polarity (Pauling 1960, FIRST_PRINCIPLES)
     δ = 1 − exp(−0.25 × (χ_A − χ_B)²)

     Fractional ionic character from electronegativity difference.
     δ = 0: pure covalent (Δχ = 0)
     δ → 1: ionic (Δχ > 3)

  5. Molecular Dipole Moment (FIRST_PRINCIPLES: vector addition)
     μ_mol = Σ μ_bond_i (vector sum)

     Each bond dipole: μ = q × d where q ∝ δ and d = bond length.
     Molecular geometry determines cancellation.
     CO₂: two C=O dipoles cancel (linear) → μ = 0
     H₂O: two O-H dipoles don't cancel (bent) → μ = 1.85 D

  6. Badger's Rule for Force Constants (1934, FIRST_PRINCIPLES form)
     k = C_ij / (r_e − d_ij)³

     Where:
       r_e = equilibrium bond length (from Schomaker-Stevenson)
       C_ij, d_ij = row-pair constants (MEASURED)

     Shorter bonds are stiffer. Combined with reduced mass:
       ν = (1/2π) × √(k/μ)  →  vibrational frequency

σ-dependence:
  Bond energies are EM → σ-invariant (no nuclear mass in Coulomb law).
  Vibrational frequencies shift through nuclear mass → reduced mass:
    μ(σ) = μ(0) × scale_ratio(σ)
    ν(σ) = ν(0) / √(scale_ratio(σ))

  "Measure the O-H stretch frequency — you can read off σ."

Origin tags:
  - Pauling bond energy: FIRST_PRINCIPLES + MEASURED (D(A-A), χ)
  - Schomaker-Stevenson: FIRST_PRINCIPLES + MEASURED (r_cov, c=9pm)
  - VSEPR angles: FIRST_PRINCIPLES (electrostatic repulsion geometry)
  - Bond polarity: FIRST_PRINCIPLES (Pauling ionic character formula)
  - Badger's rule: FIRST_PRINCIPLES form + MEASURED (C_ij, d_ij)
  - σ-coupling: CORE (through □σ = −ξR → nuclear mass → reduced mass)
"""

import math
from ..scale import scale_ratio
from ..constants import AMU_KG, EV_TO_J, HBAR, E_CHARGE, SIGMA_HERE


# ── Atomic Data ──────────────────────────────────────────────────
# Extracted from elements.json + Pauling covalent radii.
# Rule 9: every atom that participates in organic chemistry.
#
# Z: atomic number
# mass_amu: MEASURED atomic mass (u)
# chi: Pauling electronegativity (MEASURED)
# r_cov_pm: Pauling covalent radius (pm, MEASURED)
#   NOTE: elements.json stores smaller values (Cordero et al. 2008);
#   we use Pauling (1960) radii here because the Schomaker-Stevenson
#   constant c=9pm was calibrated against them.
# r_vdw_pm: van der Waals radius (pm, MEASURED, Bondi 1964)
# IE1_eV: first ionization energy (eV, MEASURED)
# valence_e: number of valence electrons
# lone_pairs: typical lone pairs when fully bonded
#
# Sources: Pauling "Nature of the Chemical Bond" (1960),
#          CRC Handbook 101st ed., NIST Atomic Spectra Database

ATOMS = {
    'H':  {'Z': 1,  'mass_amu': 1.008,  'chi': 2.20, 'r_cov_pm': 37,
            'r_vdw_pm': 120, 'IE1_eV': 13.598, 'valence_e': 1, 'lone_pairs': 0},
    'C':  {'Z': 6,  'mass_amu': 12.011, 'chi': 2.55, 'r_cov_pm': 77,
            'r_vdw_pm': 170, 'IE1_eV': 11.260, 'valence_e': 4, 'lone_pairs': 0},
    'N':  {'Z': 7,  'mass_amu': 14.007, 'chi': 3.04, 'r_cov_pm': 75,
            'r_vdw_pm': 155, 'IE1_eV': 14.534, 'valence_e': 5, 'lone_pairs': 1},
    'O':  {'Z': 8,  'mass_amu': 15.999, 'chi': 3.44, 'r_cov_pm': 66,
            'r_vdw_pm': 152, 'IE1_eV': 13.618, 'valence_e': 6, 'lone_pairs': 2},
    'F':  {'Z': 9,  'mass_amu': 18.998, 'chi': 3.98, 'r_cov_pm': 64,
            'r_vdw_pm': 147, 'IE1_eV': 17.423, 'valence_e': 7, 'lone_pairs': 3},
    'S':  {'Z': 16, 'mass_amu': 32.06,  'chi': 2.58, 'r_cov_pm': 105,
            'r_vdw_pm': 180, 'IE1_eV': 10.360, 'valence_e': 6, 'lone_pairs': 2},
    'Cl': {'Z': 17, 'mass_amu': 35.45,  'chi': 3.16, 'r_cov_pm': 99,
            'r_vdw_pm': 175, 'IE1_eV': 12.968, 'valence_e': 7, 'lone_pairs': 3},
}


# ── Homonuclear Bond Dissociation Energies ───────────────────────
# MEASURED seeds for the Pauling equation.
# These are the inputs the derivation CANNOT produce — they must be
# measured. Everything else in this module derives from these + ATOMS.
#
# Sources: CRC Handbook, Herzberg "Spectra of Diatomic Molecules" (1950),
#          NIST-JANAF Thermochemical Tables

HOMONUCLEAR_BONDS_EV = {
    'H':  4.52,    # H-H, Herzberg (1970)
    'C':  3.61,    # C-C single bond, ethane dissociation (CRC)
    'N':  1.59,    # N-N single bond, hydrazine (CRC)
    'O':  1.49,    # O-O single bond, hydrogen peroxide (CRC)
    'F':  1.64,    # F-F, CRC Handbook
    'S':  2.69,    # S-S, CRC Handbook
    'Cl': 2.51,    # Cl-Cl, CRC Handbook
}


# ── Constants ────────────────────────────────────────────────────

# Schomaker-Stevenson empirical constant (pm)
_SS_C_PM = 9.0  # MEASURED: Schomaker & Stevenson (1941)

# Bond order length correction factors (MEASURED from crystallography)
# double bonds are ~14% shorter than single, triple ~22% shorter
_BOND_ORDER_LENGTH = {1: 1.0, 2: 0.86, 3: 0.78}

# VSEPR base angles by number of electron domains
_VSEPR_BASE_ANGLE = {
    2: 180.0,    # linear (sp)
    3: 120.0,    # trigonal planar (sp²)
    4: 109.47,   # tetrahedral (sp³) — exact: arccos(-1/3)
    5: 90.0,     # trigonal bipyramidal (equatorial-axial angle)
    6: 90.0,     # octahedral
}

# Lone pair compression: each lone pair reduces bond angle by this amount
_LONE_PAIR_COMPRESSION_DEG = 2.5  # APPROXIMATION: empirical average

# Badger's rule constants by row pair (MEASURED)
# k = C_ij / (r_e - d_ij)^3, with k in N/m, r_e in pm
# Row pair (1,1): H-H; (1,2): H-C/N/O/F; (2,2): C-C/N/O/F pairs
_BADGER_C = {
    # Standard Badger C values (mdyn/Å·Å³) converted for r in pm, k in N/m:
    # k(N/m) = C_std × 10⁸ / (r_pm − d_pm)³
    (1, 1): 1.86e8,    # H-H type
    (1, 2): 2.35e8,    # H with 2nd row (C, N, O, F)
    (2, 2): 2.46e8,    # 2nd row pairs
    (2, 3): 2.60e8,    # 2nd row with 3rd row (S, Cl)
    (3, 3): 3.00e8,    # 3rd row pairs
}

_BADGER_D = {
    (1, 1): 68.0,      # pm
    (1, 2): 34.0,      # pm
    (2, 2): 42.0,      # pm
    (2, 3): 45.0,      # pm
    (3, 3): 55.0,      # pm
}

# Physical constants
_AMU_KG = AMU_KG
_EV_J = EV_TO_J
_HBAR = HBAR
_PM_M = 1e-12                # m per pm
_DEBYE_CM = 3.33564e-30      # C·m per Debye
_E_CHARGE = E_CHARGE


def _row(atom_key):
    """Periodic table row from atomic number."""
    Z = ATOMS[atom_key]['Z']
    if Z <= 2:
        return 1
    if Z <= 10:
        return 2
    return 3


def _row_pair(atom_A, atom_B):
    """Sorted row pair for Badger's rule lookup."""
    r_A, r_B = _row(atom_A), _row(atom_B)
    return (min(r_A, r_B), max(r_A, r_B))


# ── Pauling Bond Energy ──────────────────────────────────────────

def pauling_bond_energy(atom_A, atom_B):
    """Heteronuclear bond dissociation energy (eV) via Pauling equation.

    D(A-B) = ½(D(A-A) + D(B-B)) + (χ_A − χ_B)²

    The (Δχ)² term is in eV when electronegativities are on Pauling's
    scale (calibrated so that 1 eV ≈ 96.5 kJ/mol = 23.06 kcal/mol).

    Bond energy is electromagnetic → σ-INVARIANT.

    For homonuclear bonds (A == B), returns the measured D(A-A).

    Args:
        atom_A, atom_B: keys into ATOMS dict (e.g. 'O', 'H')

    Returns:
        Bond dissociation energy in eV.
    """
    if atom_A == atom_B:
        return HOMONUCLEAR_BONDS_EV[atom_A]

    D_AA = HOMONUCLEAR_BONDS_EV[atom_A]
    D_BB = HOMONUCLEAR_BONDS_EV[atom_B]
    chi_A = ATOMS[atom_A]['chi']
    chi_B = ATOMS[atom_B]['chi']

    arithmetic_mean = 0.5 * (D_AA + D_BB)
    ionic_resonance = (chi_A - chi_B) ** 2

    return arithmetic_mean + ionic_resonance


# ── Schomaker-Stevenson Bond Length ──────────────────────────────

def schomaker_stevenson_length(atom_A, atom_B, bond_order=1):
    """Covalent bond length (pm) from atomic radii and electronegativity.

    r(A-B) = r_A + r_B − c|χ_A − χ_B|

    Corrected for bond order:
      single: r × 1.00
      double: r × 0.86
      triple: r × 0.78

    Args:
        atom_A, atom_B: keys into ATOMS dict
        bond_order: 1, 2, or 3

    Returns:
        Bond length in picometers.
    """
    r_A = ATOMS[atom_A]['r_cov_pm']
    r_B = ATOMS[atom_B]['r_cov_pm']
    chi_A = ATOMS[atom_A]['chi']
    chi_B = ATOMS[atom_B]['chi']

    r_single = r_A + r_B - _SS_C_PM * abs(chi_A - chi_B)

    order_factor = _BOND_ORDER_LENGTH.get(bond_order, 1.0)
    return r_single * order_factor


# ── VSEPR Bond Angle ─────────────────────────────────────────────

def vsepr_bond_angle(n_electron_domains, n_lone_pairs=0):
    """Predicted bond angle (degrees) from VSEPR theory.

    Starting from the ideal geometry for n_electron_domains, each lone
    pair compresses the bond angle by ~2.5°.

    Examples:
      CH₄: 4 domains, 0 lone → 109.5°
      NH₃: 4 domains, 1 lone → 107.0°
      H₂O: 4 domains, 2 lone → 104.5°
      BF₃: 3 domains, 0 lone → 120.0°
      CO₂: 2 domains, 0 lone → 180.0°

    FIRST_PRINCIPLES: electrostatic repulsion of electron pairs.
    APPROXIMATION: constant compression per lone pair (~2.5°).

    Args:
        n_electron_domains: total electron domains (bonding + lone)
        n_lone_pairs: number of lone pairs

    Returns:
        Bond angle in degrees.
    """
    if n_electron_domains < 2:
        return 0.0

    base = _VSEPR_BASE_ANGLE.get(n_electron_domains, 109.47)
    return base - n_lone_pairs * _LONE_PAIR_COMPRESSION_DEG


# ── Bond Polarity ────────────────────────────────────────────────

def bond_polarity(atom_A, atom_B):
    """Fractional ionic character of the A-B bond (dimensionless).

    δ = 1 − exp(−0.25 × (χ_A − χ_B)²)

    Pauling (1960): empirical fit to dipole moment data.
    δ = 0 for homonuclear (pure covalent), δ → 1 for large Δχ (ionic).

    Args:
        atom_A, atom_B: keys into ATOMS dict

    Returns:
        Tuple (delta, negative_atom) where delta is the fractional ionic
        character and negative_atom is the more electronegative atom.
    """
    chi_A = ATOMS[atom_A]['chi']
    chi_B = ATOMS[atom_B]['chi']

    delta_chi = chi_A - chi_B
    delta = 1.0 - math.exp(-0.25 * delta_chi ** 2)

    negative = atom_A if chi_A >= chi_B else atom_B
    return (delta, negative)


# ── Molecular Dipole Moment ──────────────────────────────────────

def bond_dipole_debye(atom_A, atom_B, bond_order=1):
    """Dipole moment of a single bond (Debye).

    μ = δ × e × d

    Where:
      δ = fractional ionic character (from bond_polarity)
      e = elementary charge
      d = bond length (from schomaker_stevenson)

    One Debye = 3.336e-30 C·m.

    Args:
        atom_A, atom_B: keys into ATOMS dict
        bond_order: 1, 2, or 3

    Returns:
        Bond dipole moment in Debye (positive, direction is A→B if B is
        more electronegative).
    """
    delta, _ = bond_polarity(atom_A, atom_B)
    d_pm = schomaker_stevenson_length(atom_A, atom_B, bond_order)
    d_m = d_pm * _PM_M

    mu_Cm = delta * _E_CHARGE * d_m
    return mu_Cm / _DEBYE_CM


def molecular_dipole_moment(bond_dipoles_debye, angles_deg):
    """Molecular dipole moment from vector sum of bond dipoles (Debye).

    For a molecule with N bonds in a plane, each bond dipole is projected
    onto x and y axes using the bond angle, then summed.

    This handles the key cases:
      CO₂ (linear, 180°): dipoles cancel → μ = 0
      H₂O (bent, 104.5°): dipoles partially add → μ ≈ 1.85 D

    FIRST_PRINCIPLES: vector addition of electrostatic dipoles.

    Args:
        bond_dipoles_debye: list of individual bond dipole magnitudes
        angles_deg: list of angles from a reference direction for each bond

    Returns:
        Net molecular dipole moment in Debye.
    """
    mu_x = 0.0
    mu_y = 0.0

    for mu, angle in zip(bond_dipoles_debye, angles_deg):
        rad = math.radians(angle)
        mu_x += mu * math.cos(rad)
        mu_y += mu * math.sin(rad)

    return math.sqrt(mu_x ** 2 + mu_y ** 2)


# ── Force Constant (Badger's Rule) ───────────────────────────────

def badger_force_constant(atom_A, atom_B, bond_order=1):
    """Bond stretching force constant (N/m) via Badger's rule.

    k = C_ij / (r_e − d_ij)³

    Where C_ij and d_ij are empirical constants that depend on the
    periodic table rows of atoms A and B.

    FIRST_PRINCIPLES form: shorter bonds are stiffer (deeper, narrower
    potential well). Badger (1934) showed the (r − d)^−3 dependence.
    C_ij and d_ij are MEASURED from IR spectroscopy data.

    Args:
        atom_A, atom_B: keys into ATOMS dict
        bond_order: 1, 2, or 3

    Returns:
        Force constant in N/m.
    """
    r_e = schomaker_stevenson_length(atom_A, atom_B, bond_order)
    rp = _row_pair(atom_A, atom_B)

    C = _BADGER_C.get(rp, _BADGER_C[(2, 2)])
    d = _BADGER_D.get(rp, _BADGER_D[(2, 2)])

    denom = r_e - d
    if denom <= 0:
        # Bond shorter than Badger offset — use minimum safe value
        denom = 1.0

    return C / (denom ** 3)


# ── Vibrational Frequency ────────────────────────────────────────

def reduced_mass_kg(atom_A, atom_B, sigma=SIGMA_HERE):
    """Reduced mass of A-B pair (kg).

    μ = m_A × m_B / (m_A + m_B)

    σ-dependence: nuclear masses scale with scale_ratio(σ).

    Args:
        atom_A, atom_B: keys into ATOMS dict
        sigma: σ-field value

    Returns:
        Reduced mass in kg.
    """
    m_A = ATOMS[atom_A]['mass_amu'] * _AMU_KG
    m_B = ATOMS[atom_B]['mass_amu'] * _AMU_KG

    if sigma != SIGMA_HERE:
        r = scale_ratio(sigma)
        m_A *= r
        m_B *= r

    return m_A * m_B / (m_A + m_B)


def vibrational_frequency(atom_A, atom_B, bond_order=1, sigma=SIGMA_HERE):
    """Bond stretching frequency (Hz) from derived force constant.

    ν = (1/2π) × √(k/μ)

    Force constant k from Badger's rule (derived from bond length).
    Reduced mass μ from atomic masses (shifted by σ).

    Args:
        atom_A, atom_B: keys into ATOMS dict
        bond_order: 1, 2, or 3
        sigma: σ-field value

    Returns:
        Vibrational frequency in Hz.
    """
    k = badger_force_constant(atom_A, atom_B, bond_order)
    mu = reduced_mass_kg(atom_A, atom_B, sigma)

    if mu <= 0:
        return 0.0

    return (1.0 / (2.0 * math.pi)) * math.sqrt(k / mu)


def vibrational_wavenumber(atom_A, atom_B, bond_order=1, sigma=SIGMA_HERE):
    """Bond stretching wavenumber (cm⁻¹) — standard IR spectroscopy unit.

    ν̃ = ν / c  (in cm⁻¹)

    Args:
        atom_A, atom_B: keys into ATOMS dict
        bond_order: 1, 2, or 3
        sigma: σ-field value

    Returns:
        Wavenumber in cm⁻¹.
    """
    freq = vibrational_frequency(atom_A, atom_B, bond_order, sigma)
    c_cm_s = 2.998e10  # speed of light in cm/s
    return freq / c_cm_s


# ── Hybridization ────────────────────────────────────────────────

def hybridization(n_bonds, n_lone_pairs):
    """Predict hybridization from steric number.

    steric_number = n_bonds + n_lone_pairs
      2 → sp
      3 → sp²
      4 → sp³

    Args:
        n_bonds: number of σ bonds
        n_lone_pairs: number of lone pairs

    Returns:
        String: 'sp', 'sp2', 'sp3', or 'other'.
    """
    sn = n_bonds + n_lone_pairs
    if sn == 2:
        return 'sp'
    elif sn == 3:
        return 'sp2'
    elif sn == 4:
        return 'sp3'
    else:
        return 'other'


# ── Nagatha Export ───────────────────────────────────────────────

def bond_properties(atom_A, atom_B, bond_order=1, sigma=SIGMA_HERE):
    """Export all derived bond properties in Nagatha-compatible format.

    Args:
        atom_A, atom_B: keys into ATOMS dict
        bond_order: 1, 2, or 3
        sigma: σ-field value

    Returns:
        Dict with all bond properties and origin tags.
    """
    energy = pauling_bond_energy(atom_A, atom_B)
    length = schomaker_stevenson_length(atom_A, atom_B, bond_order)
    delta, neg = bond_polarity(atom_A, atom_B)
    mu_bond = bond_dipole_debye(atom_A, atom_B, bond_order)
    k = badger_force_constant(atom_A, atom_B, bond_order)
    freq = vibrational_frequency(atom_A, atom_B, bond_order, sigma)
    wn = vibrational_wavenumber(atom_A, atom_B, bond_order, sigma)

    return {
        'atom_A': atom_A,
        'atom_B': atom_B,
        'bond_order': bond_order,
        'sigma': sigma,
        'dissociation_energy_eV': energy,
        'bond_length_pm': length,
        'fractional_ionic_character': delta,
        'negative_end': neg,
        'bond_dipole_debye': mu_bond,
        'force_constant_N_m': k,
        'vibrational_frequency_Hz': freq,
        'vibrational_wavenumber_cm-1': wn,
        'origin': (
            "Pauling bond energy (1932): D(A-B) = ½(D(A-A)+D(B-B)) + (Δχ)². "
            "FIRST_PRINCIPLES + MEASURED (D(A-A), χ). "
            "Schomaker-Stevenson length (1941): r = r_A+r_B − 9|Δχ|. "
            "FIRST_PRINCIPLES + MEASURED (r_cov, c=9pm). "
            "Badger force constant (1934): k = C/(r−d)³. "
            "FIRST_PRINCIPLES form + MEASURED (C, d by row pair). "
            "σ-coupling: nuclear mass → reduced mass → ν. "
            "Bond energy is EM → σ-invariant."
        ),
    }
