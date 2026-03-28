"""
Hydrogen bonding and intermolecular forces — deriving bulk behavior from bonds.

The second stage of the matter information cascade: molecular bonds (Module 1)
→ intermolecular forces → bulk properties (Module 3).

From the bond-level properties derived in molecular_bonds.py, this module
computes the three intermolecular forces that govern condensed-phase behavior:

  1. Hydrogen bonding (dominant for water, alcohols, HF, ammonia)
  2. London dispersion (universal, from induced dipole fluctuations)
  3. Keesom orientation (permanent dipole–dipole interaction)

Together these predict which molecules are liquids at room temperature,
their boiling points, and the strength of their intermolecular cohesion.

Derivation chains:

  1. Hydrogen Bond Energy (electrostatic model, calibrated)
     E_HB = E_0 × δ_donor × δ_acceptor × (r_0 / r)²

     Where:
       δ_donor = bond polarity of D-H bond (from molecular_bonds)
       δ_acceptor = lone pair availability ∝ χ_acceptor / χ_max
       r_0 = calibration distance (280 pm, O···O in water)
       E_0 = calibration energy (MEASURED: 0.23 eV for water O-H···O)

     The H-bond is primarily electrostatic: partial positive charge on H
     interacts with lone pair electrons on the acceptor. The Pauling
     bond polarity (from Module 1) gives the charge separation.

  2. London Dispersion Energy (1930, FIRST_PRINCIPLES)
     E_London = −(3/4) × (α_A × α_B) / r⁶ × (IE_A × IE_B)/(IE_A + IE_B)

     Where:
       α = polarizability ≈ 4πε₀ × r_vdW³ (APPROXIMATION)
       IE = first ionization energy (MEASURED)
       r = intermolecular distance

     Every molecule has London dispersion. It dominates for nonpolar
     molecules (methane, noble gases).

  3. Keesom Orientation Energy (1921, FIRST_PRINCIPLES)
     E_Keesom = −(2/3) × μ_A² × μ_B² / ((4πε₀)² × k_B × T × r⁶)

     Permanent dipoles tend to align. Temperature fights alignment.
     Dominant for polar molecules without H-bonding capability.

Boiling point estimation:
  Trouton's rule: ΔS_vap ≈ 85 J/(mol·K) for "normal" liquids.
  ΔH_vap ≈ n_HB × E_HB / 2 + E_dispersion + E_dipole (per molecule)
  T_boil ≈ ΔH_vap × N_A / ΔS_vap

σ-dependence:
  H-bond energy: electrostatic → σ-INVARIANT (to first order).
  London dispersion: depends on polarizability (EM) → σ-INVARIANT.
  Keesom: depends on dipole moment (EM) → σ-INVARIANT.
  Boiling point: shifts only through σ-dependent corrections to molecular
  geometry (negligible at Earth σ).

Origin tags:
  - H-bond energy: FIRST_PRINCIPLES (electrostatics) + MEASURED (E_0 calibration)
  - London dispersion: FIRST_PRINCIPLES (QM perturbation theory)
  - Keesom: FIRST_PRINCIPLES (classical statistical mechanics)
  - Boiling point: FIRST_PRINCIPLES (Trouton) + APPROXIMATION
"""

import math

from .molecular_bonds import ATOMS, bond_polarity, bond_dipole_debye
from ..constants import EV_TO_J, K_B, N_AVOGADRO, EPS_0, SIGMA_HERE


# ── Physical Constants ──────────────────────────────────────────
_EV_J = EV_TO_J
_K_B_EV = 8.617333262e-5      # eV/K (Boltzmann) — derived, not in constants
_K_B_J = K_B
_N_A = N_AVOGADRO
_PM_M = 1e-12                 # m per pm
_EPS_0 = EPS_0
_DEBYE_CM = 3.33564e-30       # C·m per Debye
_TROUTON_ENTROPY = 85.0       # J/(mol·K), Trouton's rule for ΔS_vap


# ── Hydrogen Bond Calibration ──────────────────────────────────
# Calibrated to water O-H···O: E_HB = 0.23 eV at r = 280 pm
# MEASURED: from ice sublimation enthalpy / 2 H-bonds per molecule
_HB_E0_EV = 0.23              # eV, water O-H···O H-bond energy
_HB_R0_PM = 280.0             # pm, O···O distance in liquid water
_HB_DELTA_DONOR_WATER = None  # computed at import from bond_polarity('O','H')
_HB_DELTA_ACCEPTOR_WATER = None

# Compute water calibration values at module load
_delta_OH, _ = bond_polarity('O', 'H')
_chi_O = ATOMS['O']['chi']
_chi_max = ATOMS['F']['chi']  # F is most electronegative
_acceptor_O = _chi_O / _chi_max
_HB_CALIBRATION = _HB_E0_EV / (_delta_OH * _acceptor_O)


# ── Molecule Database ──────────────────────────────────────────
# Rule 9: every molecule gets every field.
#
# donor_bond: (donor_heavy_atom, 'H') — the D-H bond providing partial + on H
# acceptor_atom: atom with lone pairs accepting the H-bond
# n_donor_bonds: number of D-H bonds that can donate
# n_acceptor_lps: number of lone pairs that can accept
# n_hb_liquid: MEASURED average H-bonds per molecule in liquid (neutron diffraction)
# hb_energy_ev: MEASURED H-bond energy (calorimetry / sublimation)
# molecular_mass_amu: sum of atomic masses (DERIVED)
# dipole_debye: MEASURED molecular dipole moment
# polarizability_A3: MEASURED molecular polarizability in ų (= 10⁻³⁰ m³)
#   Sources: CRC Handbook, Maryott & Buckley (1953), NIST
#   From refractive index or dielectric constant measurements.
# IE_mol_eV: MEASURED first ionization energy of the molecule (eV)
#   Sources: NIST Chemistry WebBook
# r_intermol_pm: MEASURED nearest-neighbor distance in liquid
# n_neighbors: coordination number in liquid (MEASURED / MD simulation)

MOLECULES = {
    'water': {
        'formula': 'H₂O',
        'atoms': {'O': 1, 'H': 2},
        'donor_bond': ('O', 'H'),
        'acceptor_atom': 'O',
        'n_donor_bonds': 2,          # two O-H bonds can donate
        'n_acceptor_lps': 2,         # two lone pairs on O
        'n_hb_liquid': 3.5,          # MEASURED: neutron diffraction (Soper 2000)
        'hb_energy_ev': 0.23,        # MEASURED: ice sublimation / structure
        'molecular_mass_amu': 18.015, # DERIVED: 2×1.008 + 15.999
        'dipole_debye': 1.85,        # MEASURED: Clough et al. (1973)
        'polarizability_A3': 1.45,   # MEASURED: CRC Handbook (10⁻³⁰ m³)
        'IE_mol_eV': 12.62,          # MEASURED: NIST
        'r_intermol_pm': 280,        # MEASURED: O···O in liquid water
        'n_neighbors': 4.4,          # MEASURED: tetrahedral + interstitial
        'T_boil_K': 373.15,          # MEASURED: reference for validation
    },
    'methanol': {
        'formula': 'CH₃OH',
        'atoms': {'C': 1, 'O': 1, 'H': 4},
        'donor_bond': ('O', 'H'),
        'acceptor_atom': 'O',
        'n_donor_bonds': 1,          # one O-H
        'n_acceptor_lps': 2,         # two lone pairs on O
        'n_hb_liquid': 2.0,          # MEASURED: neutron diffraction
        'hb_energy_ev': 0.20,        # MEASURED: sublimation calorimetry
        'molecular_mass_amu': 32.042, # DERIVED: 12.011 + 15.999 + 4×1.008
        'dipole_debye': 1.70,        # MEASURED: CRC Handbook
        'polarizability_A3': 3.29,   # MEASURED: CRC Handbook
        'IE_mol_eV': 10.84,          # MEASURED: NIST
        'r_intermol_pm': 290,        # MEASURED: O···O in liquid methanol
        'n_neighbors': 4.2,          # MEASURED: MD simulation
        'T_boil_K': 337.7,           # MEASURED
    },
    'ammonia': {
        'formula': 'NH₃',
        'atoms': {'N': 1, 'H': 3},
        'donor_bond': ('N', 'H'),
        'acceptor_atom': 'N',
        'n_donor_bonds': 3,          # three N-H bonds
        'n_acceptor_lps': 1,         # one lone pair on N
        'n_hb_liquid': 2.0,          # MEASURED: neutron diffraction
        'hb_energy_ev': 0.16,        # MEASURED: sublimation
        'molecular_mass_amu': 17.031, # DERIVED: 14.007 + 3×1.008
        'dipole_debye': 1.47,        # MEASURED
        'polarizability_A3': 2.26,   # MEASURED: CRC Handbook
        'IE_mol_eV': 10.07,          # MEASURED: NIST
        'r_intermol_pm': 340,        # MEASURED: N···N
        'n_neighbors': 5.8,          # MEASURED: liquid ammonia
        'T_boil_K': 239.8,           # MEASURED
    },
    'hydrogen_fluoride': {
        'formula': 'HF',
        'atoms': {'H': 1, 'F': 1},
        'donor_bond': ('F', 'H'),
        'acceptor_atom': 'F',
        'n_donor_bonds': 1,          # one H-F bond
        'n_acceptor_lps': 3,         # three lone pairs on F
        'n_hb_liquid': 2.0,          # MEASURED: forms zigzag chains
        'hb_energy_ev': 0.29,        # MEASURED: strongest common H-bond
        'molecular_mass_amu': 20.006, # DERIVED: 1.008 + 18.998
        'dipole_debye': 1.83,        # MEASURED
        'polarizability_A3': 0.83,   # MEASURED: CRC Handbook
        'IE_mol_eV': 16.06,          # MEASURED: NIST
        'r_intermol_pm': 250,        # MEASURED: F···F in liquid HF
        'n_neighbors': 3.5,          # MEASURED: chain + branching
        'T_boil_K': 292.7,           # MEASURED
    },
    'ethanol': {
        'formula': 'C₂H₅OH',
        'atoms': {'C': 2, 'O': 1, 'H': 6},
        'donor_bond': ('O', 'H'),
        'acceptor_atom': 'O',
        'n_donor_bonds': 1,          # one O-H
        'n_acceptor_lps': 2,         # two lone pairs on O
        'n_hb_liquid': 1.8,          # MEASURED: neutron diffraction
        'hb_energy_ev': 0.20,        # MEASURED: similar to methanol
        'molecular_mass_amu': 46.068, # DERIVED: 2×12.011 + 15.999 + 6×1.008
        'dipole_debye': 1.69,        # MEASURED
        'polarizability_A3': 5.11,   # MEASURED: CRC Handbook
        'IE_mol_eV': 10.48,          # MEASURED: NIST
        'r_intermol_pm': 290,        # MEASURED: O···O
        'n_neighbors': 4.0,          # MEASURED
        'T_boil_K': 351.4,           # MEASURED
    },
    'methane': {
        'formula': 'CH₄',
        'atoms': {'C': 1, 'H': 4},
        'donor_bond': None,          # C-H too nonpolar to H-bond
        'acceptor_atom': None,       # no lone pairs on C
        'n_donor_bonds': 0,
        'n_acceptor_lps': 0,
        'n_hb_liquid': 0,            # ZERO: no H-bonding
        'hb_energy_ev': 0.0,         # ZERO: no H-bonding
        'molecular_mass_amu': 16.043, # DERIVED: 12.011 + 4×1.008
        'dipole_debye': 0.0,         # ZERO by symmetry (tetrahedral)
        'polarizability_A3': 2.59,   # MEASURED: CRC Handbook
        'IE_mol_eV': 12.51,          # MEASURED: NIST
        'r_intermol_pm': 410,        # MEASURED: C···C in liquid methane
        'n_neighbors': 12.0,         # MEASURED: close-packed
        'T_boil_K': 111.7,           # MEASURED
    },
}


# ── Hydrogen Bond Energy ───────────────────────────────────────

def hydrogen_bond_energy(donor_atom, acceptor_atom, r_pm=None):
    """Hydrogen bond energy (eV) from donor/acceptor electronegativity.

    E_HB = E_cal × δ_donor × (χ_acceptor / χ_max) × (r_0 / r)²

    Where E_cal is calibrated so that O-H···O at 280 pm = 0.23 eV.

    The key physics: H-bonds are electrostatic. Stronger D-H polarity
    (larger δ) and more electronegative acceptors (more lone pair density)
    give stronger H-bonds.

    FIRST_PRINCIPLES (electrostatic model) + MEASURED (E_0 calibration).

    Args:
        donor_atom: atom bonded to H in D-H···A (e.g. 'O' in O-H···O)
        acceptor_atom: atom with lone pairs (e.g. 'O', 'N', 'F')
        r_pm: D···A distance in pm (default: use calibration distance)

    Returns:
        H-bond energy in eV. Returns 0 if donor or acceptor is None.
    """
    if donor_atom is None or acceptor_atom is None:
        return 0.0

    # Donor D-H bond polarity
    delta_donor, _ = bond_polarity(donor_atom, 'H')

    # Acceptor lone pair availability ∝ electronegativity
    chi_acc = ATOMS[acceptor_atom]['chi']
    delta_acceptor = chi_acc / _chi_max

    # Distance dependence (1/r² — intermediate between Coulomb 1/r and dipole 1/r³)
    if r_pm is None:
        r_factor = 1.0
    else:
        r_factor = (_HB_R0_PM / r_pm) ** 2

    return _HB_CALIBRATION * delta_donor * delta_acceptor * r_factor


def hydrogen_bond_energy_molecule(mol_key):
    """H-bond energy for a molecule from the MOLECULES database (eV).

    Uses the molecule's donor bond and acceptor atom.

    Args:
        mol_key: key into MOLECULES dict

    Returns:
        Predicted H-bond energy in eV.
    """
    mol = MOLECULES[mol_key]
    donor_bond = mol['donor_bond']
    acceptor = mol['acceptor_atom']

    if donor_bond is None:
        return 0.0

    donor_heavy = donor_bond[0]
    r = mol['r_intermol_pm']
    return hydrogen_bond_energy(donor_heavy, acceptor, r)


# ── London Dispersion Energy ──────────────────────────────────

def _polarizability_SI(mol_key):
    """Molecular polarizability in SI units (F·m²) from MEASURED data.

    α(SI) = 4πε₀ × α(ų) × 10⁻³⁰

    Polarizability is MEASURED from refractive index or dielectric constant.
    Cannot be accurately derived from atomic radii alone (overlap effects).

    Args:
        mol_key: key into MOLECULES dict

    Returns:
        Polarizability in SI units (F·m² = C²·s²/(kg·m³)).
    """
    alpha_A3 = MOLECULES[mol_key]['polarizability_A3']  # in 10⁻³⁰ m³
    return 4.0 * math.pi * _EPS_0 * alpha_A3 * 1e-30


def london_dispersion_energy(mol_A_key, mol_B_key=None, r_pm=None):
    """London dispersion interaction energy (eV) between two molecules.

    E_London = −(3/4) × (α_A × α_B) / ((4πε₀)² × r⁶)
               × (IE_A × IE_B) / (IE_A + IE_B)

    This is always attractive (negative). Returns magnitude (positive).

    Uses MEASURED molecular polarizabilities and ionization energies.
    FIRST_PRINCIPLES: London (1930), quantum mechanical perturbation theory.

    Args:
        mol_A_key: key into MOLECULES dict
        mol_B_key: second molecule (default: same as A — self-interaction)
        r_pm: intermolecular distance in pm (default: from molecule data)

    Returns:
        |E_London| in eV (positive).
    """
    if mol_B_key is None:
        mol_B_key = mol_A_key

    alpha_A = _polarizability_SI(mol_A_key)
    alpha_B = _polarizability_SI(mol_B_key)
    IE_A = MOLECULES[mol_A_key]['IE_mol_eV'] * _EV_J
    IE_B = MOLECULES[mol_B_key]['IE_mol_eV'] * _EV_J

    if r_pm is None:
        r_pm = MOLECULES[mol_A_key]['r_intermol_pm']

    r_m = r_pm * _PM_M

    if r_m <= 0:
        return 0.0

    # London formula (SI units → Joules → eV)
    numerator = 0.75 * alpha_A * alpha_B * IE_A * IE_B / (IE_A + IE_B)
    denominator = (4.0 * math.pi * _EPS_0) ** 2 * r_m ** 6

    E_J = numerator / denominator
    return E_J / _EV_J


def keesom_dipole_energy(mol_A_key, mol_B_key=None, r_pm=None, T_K=300.0):
    """Keesom orientation energy (eV) between two polar molecules.

    E_Keesom = −(2/3) × μ_A² × μ_B² / ((4πε₀)² × k_B × T × r⁶)

    Temperature-dependent: thermal motion disrupts alignment.
    Returns magnitude (positive).

    FIRST_PRINCIPLES: Keesom (1921), classical statistical mechanics.

    Args:
        mol_A_key: key into MOLECULES
        mol_B_key: second molecule (default: same as A)
        r_pm: intermolecular distance in pm
        T_K: temperature in Kelvin

    Returns:
        |E_Keesom| in eV (positive).
    """
    if mol_B_key is None:
        mol_B_key = mol_A_key

    mu_A = MOLECULES[mol_A_key]['dipole_debye'] * _DEBYE_CM  # C·m
    mu_B = MOLECULES[mol_B_key]['dipole_debye'] * _DEBYE_CM

    if mu_A == 0 or mu_B == 0:
        return 0.0

    if r_pm is None:
        r_pm = MOLECULES[mol_A_key]['r_intermol_pm']

    r_m = r_pm * _PM_M

    if r_m <= 0 or T_K <= 0:
        return 0.0

    numerator = (2.0 / 3.0) * mu_A ** 2 * mu_B ** 2
    denominator = (4.0 * math.pi * _EPS_0) ** 2 * _K_B_J * T_K * r_m ** 6

    E_J = numerator / denominator
    return E_J / _EV_J


# ── Total Intermolecular Energy ────────────────────────────────

def total_intermolecular_energy(mol_key, T_K=300.0):
    """Total intermolecular interaction energy per molecule (eV).

    Sum of:
      - H-bond contribution: n_HB × E_HB / 2 (÷2 for double counting)
      - London dispersion × n_neighbors / 2
      - Keesom dipole-dipole × (n_neighbors − n_HB) / 2

    The Keesom term applies ONLY to non-H-bonded neighbors because
    H-bonds already include the electrostatic dipole-dipole component.
    A hydrogen bond IS a directed dipole-dipole interaction — adding
    Keesom on top would double-count the electrostatic contribution.

    FIRST_PRINCIPLES sum + APPROXIMATION (pairwise additivity).

    Args:
        mol_key: key into MOLECULES
        T_K: temperature in Kelvin

    Returns:
        Total intermolecular energy in eV per molecule (positive = cohesive).
    """
    mol = MOLECULES[mol_key]

    # H-bond contribution (already per-molecule average)
    n_hb = mol['n_hb_liquid']
    E_hb = hydrogen_bond_energy_molecule(mol_key)
    hb_total = n_hb * E_hb / 2.0  # divide by 2: each H-bond shared

    # Dispersion: applies to ALL neighbors (London is distinct from Coulomb)
    n_nn = mol['n_neighbors']
    E_london = london_dispersion_energy(mol_key)
    disp_total = n_nn * E_london / 2.0

    # Keesom: only non-H-bonded neighbors (H-bond subsumes the dipole term)
    n_non_hb = max(0.0, n_nn - n_hb)
    E_keesom = keesom_dipole_energy(mol_key, T_K=T_K)
    keesom_total = n_non_hb * E_keesom / 2.0

    return hb_total + disp_total + keesom_total


def intermolecular_breakdown(mol_key, T_K=300.0):
    """Breakdown of intermolecular forces for a molecule.

    Returns dict with individual contributions and total.

    Args:
        mol_key: key into MOLECULES
        T_K: temperature

    Returns:
        Dict with hb_ev, london_ev, keesom_ev, total_ev, dominant.
    """
    mol = MOLECULES[mol_key]

    n_hb = mol['n_hb_liquid']
    E_hb = hydrogen_bond_energy_molecule(mol_key)
    hb_total = n_hb * E_hb / 2.0

    n_nn = mol['n_neighbors']
    E_london = london_dispersion_energy(mol_key)
    disp_total = n_nn * E_london / 2.0

    # Keesom only for non-H-bonded neighbors
    n_non_hb = max(0.0, n_nn - n_hb)
    E_keesom = keesom_dipole_energy(mol_key, T_K=T_K)
    keesom_total = n_non_hb * E_keesom / 2.0

    total = hb_total + disp_total + keesom_total

    # Identify dominant force
    contributions = {
        'hydrogen_bond': hb_total,
        'london_dispersion': disp_total,
        'keesom_dipole': keesom_total,
    }
    dominant = max(contributions, key=contributions.get)

    return {
        'hb_ev': hb_total,
        'london_ev': disp_total,
        'keesom_ev': keesom_total,
        'total_ev': total,
        'dominant': dominant,
    }


# ── Boiling Point Estimation ──────────────────────────────────

def estimated_vaporization_enthalpy(mol_key, T_K=300.0):
    """Enthalpy of vaporization (J/mol) from intermolecular energy.

    ΔH_vap ≈ E_intermol × N_A

    FIRST_PRINCIPLES: energy to separate one mole of molecules from liquid.
    APPROXIMATION: pairwise sum, no many-body corrections.

    Args:
        mol_key: key into MOLECULES
        T_K: temperature

    Returns:
        ΔH_vap in J/mol.
    """
    E_ev = total_intermolecular_energy(mol_key, T_K)
    E_J_per_mol = E_ev * _EV_J * _N_A
    return E_J_per_mol


def estimated_boiling_point(mol_key, T_K=300.0):
    """Estimated boiling point (K) from Trouton's rule.

    T_boil ≈ ΔH_vap / ΔS_vap

    Where ΔS_vap ≈ 85 J/(mol·K) (Trouton's rule, 1884).
    MEASURED empirical constant — works for "normal" liquids.

    APPROXIMATION: Trouton's rule breaks for strongly H-bonded liquids
    (water: ΔS_vap = 109 J/(mol·K) — higher because H-bonds impose
    more order in liquid than typical van der Waals liquids).

    Args:
        mol_key: key into MOLECULES
        T_K: reference temperature for energy calculation

    Returns:
        Estimated boiling point in Kelvin.
    """
    dH_vap = estimated_vaporization_enthalpy(mol_key, T_K)

    # For strongly H-bonded liquids, use higher ΔS_vap
    mol = MOLECULES[mol_key]
    if mol['n_hb_liquid'] >= 3.0:
        # Water-like: ΔS_vap ≈ 109 J/(mol·K)
        dS_vap = 109.0
    elif mol['n_hb_liquid'] >= 1.0:
        # Moderate H-bonding: ΔS_vap ≈ 95 J/(mol·K)
        dS_vap = 95.0
    else:
        # Normal liquid (Trouton)
        dS_vap = _TROUTON_ENTROPY

    if dS_vap <= 0:
        return 0.0

    return dH_vap / dS_vap


# ── Ordering Functions ─────────────────────────────────────────

def hb_energy_ordering():
    """Return molecules sorted by H-bond energy (descending).

    Expected: HF > H₂O > methanol ≈ ethanol > NH₃ > methane = 0.
    """
    pairs = [(k, hydrogen_bond_energy_molecule(k)) for k in MOLECULES]
    pairs.sort(key=lambda x: -x[1])
    return pairs


def boiling_point_ordering():
    """Return molecules sorted by estimated boiling point (descending).

    Expected: H₂O > ethanol > methanol > HF > NH₃ > methane.
    """
    pairs = [(k, estimated_boiling_point(k)) for k in MOLECULES]
    pairs.sort(key=lambda x: -x[1])
    return pairs


# ── Nagatha Export ─────────────────────────────────────────────

def intermolecular_properties(mol_key, T_K=300.0, sigma=SIGMA_HERE):
    """Export all intermolecular properties in Nagatha-compatible format.

    Args:
        mol_key: key into MOLECULES
        T_K: temperature
        sigma: σ-field value (reserved for future use)

    Returns:
        Dict with all properties and origin tags.
    """
    mol = MOLECULES[mol_key]
    breakdown = intermolecular_breakdown(mol_key, T_K)
    T_boil = estimated_boiling_point(mol_key, T_K)
    dH_vap = estimated_vaporization_enthalpy(mol_key, T_K)

    return {
        'molecule': mol_key,
        'formula': mol['formula'],
        'molecular_mass_amu': mol['molecular_mass_amu'],
        'dipole_debye': mol['dipole_debye'],
        'n_hb_liquid': mol['n_hb_liquid'],
        'hb_energy_ev': breakdown['hb_ev'],
        'london_energy_ev': breakdown['london_ev'],
        'keesom_energy_ev': breakdown['keesom_ev'],
        'total_intermolecular_ev': breakdown['total_ev'],
        'dominant_force': breakdown['dominant'],
        'estimated_dH_vap_J_mol': dH_vap,
        'estimated_T_boil_K': T_boil,
        'measured_T_boil_K': mol['T_boil_K'],
        'sigma': sigma,
        'origin': (
            "H-bond: FIRST_PRINCIPLES (electrostatic) + MEASURED (E₀=0.23eV). "
            "London: FIRST_PRINCIPLES (QM perturbation, London 1930). "
            "Keesom: FIRST_PRINCIPLES (Boltzmann-weighted dipole orientation). "
            "Boiling point: Trouton's rule (MEASURED ΔS_vap ≈ 85 J/(mol·K)). "
            "σ-invariant to first order (all EM interactions)."
        ),
    }
