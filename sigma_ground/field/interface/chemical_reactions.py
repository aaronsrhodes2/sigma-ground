"""
Chemical reactions — thermodynamics and kinetics from bond physics.

Fourth stage of the matter information cascade:
  molecular_bonds.py (bond energies, lengths, force constants)
  → organic_materials.py (combustion enthalpy, material properties)
  → chemical_reactions.py (general reactions, rates, equilibria)

Derivation chains:

  1. General Reaction Enthalpy (Hess's Law, FIRST_PRINCIPLES)
     ΔH = Σ(bonds broken) − Σ(bonds formed)

     The same Hess's law used for combustion, generalized to ANY reaction
     with known bond inventories. Bond energies from Pauling equation
     (molecular_bonds.py).

     Accuracy: ±15% for simple reactions (same-order bonds).
     The Pauling equation gives AVERAGE bond energies — real molecules
     have strain, resonance, hyperconjugation. Honest about this.

  2. Evans-Polanyi Activation Energy (1938, FIRST_PRINCIPLES + MEASURED)
     E_a = E_0 + α × ΔH_rxn

     A linear free-energy relationship: more exothermic reactions have
     lower barriers (Bell-Evans-Polanyi principle).

     Where:
       E_0 = intrinsic barrier for reaction family (MEASURED)
       α = Brønsted coefficient, 0 < α < 1 (MEASURED)
       ΔH_rxn = reaction enthalpy (DERIVED from bonds)

     E_0 and α are per-family constants. Within each family (e.g. all
     H-abstractions by radicals), the correlation is tight (R² > 0.9).

  3. Arrhenius Rate Constant (1889, FIRST_PRINCIPLES)
     k(T) = A × exp(−E_a / k_B T)

     Temperature dependence of rate from transition state theory.
     Exact in the harmonic-transition-state approximation.

     The pre-exponential A is derived from collision theory (kinetic
     theory of gases) or estimated from transition state theory.

  4. Collision Theory Pre-exponential (FIRST_PRINCIPLES)
     A = N_A × σ_coll × √(8 k_B T / (π μ)) × p

     Where:
       σ_coll = π (r_A + r_B)²  [collision cross-section]
       μ = m_A × m_B / (m_A + m_B)  [reduced mass]
       p = steric factor (fraction of collisions with correct orientation)

     FIRST_PRINCIPLES: kinetic theory of gases + geometric probability.
     The steric factor p is the only empirical piece (~0.01-1.0).

  5. Equilibrium Constant (FIRST_PRINCIPLES: thermodynamics)
     K = exp(−ΔG° / RT)

     Where:
       ΔG° = ΔH° − TΔS°
       ΔH° from Hess's law (bonds)
       ΔS° estimated from molecular degrees of freedom

     At equilibrium, ΔG = 0 and the ratio of products to reactants
     is fixed by K. This is exact thermodynamics.

  6. Entropy Estimation (FIRST_PRINCIPLES: statistical mechanics)
     ΔS ≈ ΔS_trans + ΔS_rot + ΔS_vib

     For gas-phase reactions, the dominant contribution is translational
     entropy change (number of moles of gas changes).

     Rough rule: ΔS ≈ Δn_gas × S_trans_per_mol
     where S_trans_per_mol ≈ 150 J/(mol·K) for a typical small molecule
     at 298 K (Sackur-Tetrode).

σ-dependence:
  Bond energies: EM → σ-INVARIANT.
  Activation energies: EM barrier → σ-INVARIANT.
  Rate constants: k = A exp(-Ea/kBT). Ea is σ-invariant, but
    the pre-exponential A depends on reduced mass μ(σ) through
    collision frequency. Heavier nuclei → slower collisions → smaller A.

  "Same chemistry, slower clock."

Origin tags:
  - Hess's law: FIRST_PRINCIPLES (thermodynamic identity)
  - Evans-Polanyi: FIRST_PRINCIPLES + MEASURED (E_0, α per family)
  - Arrhenius: FIRST_PRINCIPLES (transition state theory)
  - Collision theory: FIRST_PRINCIPLES (kinetic theory of gases)
  - Equilibrium constant: FIRST_PRINCIPLES (thermodynamics)
  - Steric factors: MEASURED (0.01-1.0 per reaction type)
"""

import math
from .molecular_bonds import pauling_bond_energy, ATOMS, AMU_KG
from ..constants import N_AVOGADRO, K_B, EV_TO_J, HBAR


# ── Bond Energies ──────────────────────────────────────────────────
# Reuse the bond energy lookup from organic_materials.py logic.
# MEASURED double/triple bond energies; single bonds from Pauling.

_BOND_ENERGIES_EV = {
    # Single bonds: None = derive from Pauling equation
    'C-H': None,
    'C-C': None,
    'C-O': None,
    'C-N': None,
    'C-S': None,
    'C-F': None,
    'C-Cl': None,
    'O-H': None,
    'N-H': None,
    'S-H': None,
    'H-F': None,
    'H-Cl': None,
    'N-N': None,
    'O-O': None,
    'S-S': None,
    'F-F': None,
    'Cl-Cl': None,
    # Double/triple bonds: MEASURED (π bonds ≠ scaled σ bonds)
    'O=O': 4.98,     # MEASURED: O₂ (CRC Handbook)
    'C=O': 7.71,     # MEASURED: CO₂ bond energy (CRC)
    'C=C': 6.27,     # MEASURED: ethylene (CRC)
    'C≡C': 8.49,     # MEASURED: acetylene (CRC)
    'C=N': 6.37,     # MEASURED: imines (CRC)
    'C≡N': 9.26,     # MEASURED: HCN/nitriles (CRC)
    'N=N': 4.19,     # MEASURED: diazene (CRC)
    'N≡N': 9.79,     # MEASURED: N₂ triple bond (CRC)
    'C-O_alcohol': None,   # same as C-O single, for clarity
    'C=O_aldehyde': 7.43,  # MEASURED: formaldehyde C=O (CRC)
    'S=O': 5.36,     # MEASURED: SO₂ (CRC)
}


def _bond_energy_ev(bond_type):
    """Get bond energy in eV, deriving from Pauling where possible."""
    # Check explicit table first
    if bond_type in _BOND_ENERGIES_EV:
        val = _BOND_ENERGIES_EV[bond_type]
        if val is not None:
            return val
    # Parse "A-B" format for Pauling derivation
    parts = bond_type.split('-')
    if len(parts) == 2:
        return pauling_bond_energy(parts[0], parts[1])
    raise KeyError(f"Unknown bond type: {bond_type}")


# ── Evans-Polanyi Parameters ──────────────────────────────────────
# MEASURED parameters for reaction families.
# E_0 = intrinsic barrier (eV), α = Brønsted coefficient (dimensionless).
#
# Sources:
#   Evans & Polanyi (1938), Trans. Faraday Soc.
#   Marcus (1968), J. Phys. Chem. (extended Marcus theory)
#   Roberts & Steel (1994), J. Phys. Chem. (H-abstraction compilation)
#   Blowers & Masel (2000), AIChE J. (comprehensive review)

EVANS_POLANYI_FAMILIES = {
    'hydrogen_abstraction': {
        # R-H + X• → R• + H-X
        # E_0 ≈ 0.5 eV (48 kJ/mol), typical for C-H abstraction
        'E_0_eV': 0.50,
        'alpha': 0.4,
        'description': 'R-H + X• → R• + H-X (radical H-transfer)',
    },
    'addition_to_alkene': {
        # X• + C=C → X-C-C•
        # E_0 ≈ 0.35 eV (34 kJ/mol)
        'E_0_eV': 0.35,
        'alpha': 0.35,
        'description': 'X• + C=C → X-C-C• (radical addition)',
    },
    'sn2_displacement': {
        # Y⁻ + R-X → Y-R + X⁻
        # E_0 ≈ 0.65 eV (63 kJ/mol), Marcus theory estimate
        'E_0_eV': 0.65,
        'alpha': 0.5,
        'description': 'Y⁻ + R-X → Y-R + X⁻ (nucleophilic substitution)',
    },
    'ester_hydrolysis': {
        # R-COOR' + H₂O → R-COOH + R'OH
        # E_0 ≈ 0.85 eV (82 kJ/mol), tetrahedral intermediate
        'E_0_eV': 0.85,
        'alpha': 0.3,
        'description': 'Ester + H₂O → Acid + Alcohol',
    },
    'dehydration': {
        # R-CH(OH)-CH₂R' → R-CH=CHR' + H₂O
        # E_0 ≈ 0.70 eV (68 kJ/mol), E1/E2 mechanism
        'E_0_eV': 0.70,
        'alpha': 0.45,
        'description': 'Alcohol → Alkene + H₂O (elimination)',
    },
    'halogenation': {
        # R-H + X₂ → R-X + HX
        # E_0 ≈ 0.15 eV (14 kJ/mol), Cl₂ propagation step
        'E_0_eV': 0.15,
        'alpha': 0.45,
        'description': 'R-H + X₂ → R-X + H-X (free radical chain)',
    },
    'combustion_initiation': {
        # R-H + O₂ → R• + HO₂•
        # E_0 ≈ 1.5 eV (145 kJ/mol), rate-limiting initiation step
        'E_0_eV': 1.50,
        'alpha': 0.5,
        'description': 'R-H + O₂ → R• + HO₂• (radical initiation)',
    },
}


# ── Steric Factors ─────────────────────────────────────────────────
# MEASURED from fitting pre-exponential factors to Arrhenius plots.
# p = fraction of collisions with correct orientation for reaction.
#
# Sources: Laidler "Chemical Kinetics" 3rd ed. (1987), Table 3.6

_STERIC_FACTORS = {
    'atom_atom': 1.0,           # spherically symmetric
    'atom_diatomic': 0.3,       # one orientation axis
    'atom_polyatomic': 0.1,     # need to hit correct end
    'diatomic_diatomic': 0.1,   # two orientation axes
    'small_polyatomic': 0.01,   # both need correct orientation
    'large_polyatomic': 0.001,  # many wrong orientations
}


# ── Known Reactions Database ───────────────────────────────────────
# Bond inventories for common organic reactions.
# bonds_broken/bonds_formed are dicts: {bond_type: count}
#
# These encode the stoichiometry at the BOND level — the fundamental
# accounting unit for Hess's law.

REACTIONS = {
    'methane_combustion': {
        'name': 'Methane combustion',
        'equation': 'CH₄ + 2 O₂ → CO₂ + 2 H₂O',
        'bonds_broken': {'C-H': 4, 'O=O': 2},
        'bonds_formed': {'C=O': 2, 'O-H': 4},
        'delta_n_gas': -1,  # 3 mol gas → 3 mol gas (liquid H₂O: -1)
        'family': 'combustion_initiation',
        'measured_dH_kJ_mol': -890.4,
    },
    'ethanol_combustion': {
        'name': 'Ethanol combustion',
        'equation': 'C₂H₅OH + 3 O₂ → 2 CO₂ + 3 H₂O',
        'bonds_broken': {'C-C': 1, 'C-H': 5, 'C-O': 1, 'O-H': 1, 'O=O': 3},
        'bonds_formed': {'C=O': 4, 'O-H': 6},
        'delta_n_gas': 0,
        'family': 'combustion_initiation',
        'measured_dH_kJ_mol': -1367.0,
    },
    'hydrogenation_ethylene': {
        'name': 'Ethylene hydrogenation',
        'equation': 'C₂H₄ + H₂ → C₂H₆',
        'bonds_broken': {'C=C': 1, 'H-H': 1},
        'bonds_formed': {'C-C': 1, 'C-H': 2},
        'delta_n_gas': -1,
        'family': 'addition_to_alkene',
        'measured_dH_kJ_mol': -137.0,
    },
    'hydrogenation_acetylene': {
        'name': 'Acetylene hydrogenation to ethylene',
        'equation': 'C₂H₂ + H₂ → C₂H₄',
        'bonds_broken': {'C≡C': 1, 'H-H': 1},
        'bonds_formed': {'C=C': 1, 'C-H': 2},
        'delta_n_gas': -1,
        'family': 'addition_to_alkene',
        'measured_dH_kJ_mol': -175.0,
    },
    'haber_process': {
        'name': 'Haber process (ammonia synthesis)',
        'equation': 'N₂ + 3 H₂ → 2 NH₃',
        'bonds_broken': {'N≡N': 1, 'H-H': 3},
        'bonds_formed': {'N-H': 6},
        'delta_n_gas': -2,
        'family': None,  # catalyzed, no simple EP family
        'measured_dH_kJ_mol': -92.2,
        # NOTE: Pauling gives WRONG SIGN for this reaction.
        # The N-N single bond in hydrazine (1.59 eV) is anomalously weak
        # due to lone pair repulsion, making the arithmetic mean for N-H
        # too low (3.76 vs 4.0 eV measured). This 6% error per bond
        # accumulates over 6 N-H bonds to flip the sign.
        # Known Pauling limitation — honest about it.
        'pauling_limitation': True,
    },
    'hydrogen_chloride': {
        'name': 'HCl formation',
        'equation': 'H₂ + Cl₂ → 2 HCl',
        'bonds_broken': {'H-H': 1, 'Cl-Cl': 1},
        'bonds_formed': {'H-Cl': 2},
        'delta_n_gas': 0,
        'family': 'halogenation',
        'measured_dH_kJ_mol': -184.6,
    },
    'hydrogen_fluoride': {
        'name': 'HF formation',
        'equation': 'H₂ + F₂ → 2 HF',
        'bonds_broken': {'H-H': 1, 'F-F': 1},
        'bonds_formed': {'H-F': 2},
        'delta_n_gas': 0,
        'family': 'halogenation',
        'measured_dH_kJ_mol': -542.2,
    },
    'water_formation': {
        'name': 'Water formation',
        'equation': '2 H₂ + O₂ → 2 H₂O',
        'bonds_broken': {'H-H': 2, 'O=O': 1},
        'bonds_formed': {'O-H': 4},
        'delta_n_gas': -1,
        'family': 'combustion_initiation',
        'measured_dH_kJ_mol': -483.6,
    },
    'methane_chlorination': {
        'name': 'Methane chlorination (propagation)',
        'equation': 'CH₄ + Cl• → CH₃• + HCl (then CH₃• + Cl₂ → CH₃Cl + Cl•)',
        'bonds_broken': {'C-H': 1, 'Cl-Cl': 1},
        'bonds_formed': {'C-Cl': 1, 'H-Cl': 1},
        'delta_n_gas': 0,
        'family': 'halogenation',
        'measured_dH_kJ_mol': -99.8,
    },
}

# Add H-H bond energy (needed for hydrogenation, Haber, etc.)
_BOND_ENERGIES_EV['H-H'] = None  # Pauling: homonuclear


# ═══════════════════════════════════════════════════════════════════
# THERMODYNAMICS
# ═══════════════════════════════════════════════════════════════════

def reaction_enthalpy_kJ_mol(bonds_broken, bonds_formed):
    """Reaction enthalpy from bond inventory (kJ/mol) via Hess's law.

    FIRST_PRINCIPLES: energy conservation (thermodynamic identity).

    ΔH = Σ(energy to break reactant bonds) − Σ(energy released forming
    product bonds). Negative ΔH = exothermic.

    Bond energies from Pauling equation (molecular_bonds.py).

    Accuracy: ±15% for reactions involving same-order bonds.
    Worse for resonance-stabilized molecules (benzene, CO₂).

    Args:
        bonds_broken: dict {bond_type: count} for all bonds broken
        bonds_formed: dict {bond_type: count} for all bonds formed

    Returns:
        Reaction enthalpy in kJ/mol (negative = exothermic).
    """
    E_broken = sum(count * _bond_energy_ev(bt)
                   for bt, count in bonds_broken.items())
    E_formed = sum(count * _bond_energy_ev(bt)
                   for bt, count in bonds_formed.items())

    # ΔH = broken - formed (breaking costs energy, forming releases it)
    delta_H_eV = E_broken - E_formed
    # Convert eV/molecule → kJ/mol
    return delta_H_eV * EV_TO_J * N_AVOGADRO / 1000.0


def reaction_enthalpy_by_key(reaction_key):
    """Reaction enthalpy for a named reaction (kJ/mol).

    Convenience wrapper around reaction_enthalpy_kJ_mol().

    Args:
        reaction_key: key into REACTIONS dict

    Returns:
        Reaction enthalpy in kJ/mol (negative = exothermic).
    """
    rxn = REACTIONS[reaction_key]
    return reaction_enthalpy_kJ_mol(rxn['bonds_broken'], rxn['bonds_formed'])


def entropy_change_estimate(delta_n_gas, T=298.15):
    """Estimate reaction entropy change (J/(mol·K)).

    FIRST_PRINCIPLES: Sackur-Tetrode equation for translational entropy.

    For gas-phase reactions, the dominant entropy contribution comes from
    the change in number of gas-phase moles. Each mole of gas contributes
    ~150 J/(mol·K) of translational entropy at 298 K (from Sackur-Tetrode
    for a typical small molecule of ~30 amu at 1 atm).

    This is a rough estimate (±30%). Vibrational and rotational entropy
    changes are smaller and partially cancel between reactants and products.

    Args:
        delta_n_gas: change in moles of gas (products − reactants)
        T: temperature in K (used for T-dependence of S_trans)

    Returns:
        Estimated ΔS in J/(mol·K).
    """
    # Sackur-Tetrode: S_trans ∝ (5/2)R + R ln(V/N) + (3/2)R ln(mT)
    # At 298K, 1 atm, m~30 amu: S_trans ≈ 150 J/(mol·K)
    # Temperature correction: S_trans scales as (3/2)R ln(T/T_ref)
    R = K_B * N_AVOGADRO  # J/(mol·K)
    T_ref = 298.15
    S_per_mol = 150.0 + 1.5 * R * math.log(T / T_ref)

    return delta_n_gas * S_per_mol


def gibbs_energy_kJ_mol(delta_H_kJ, delta_n_gas, T=298.15):
    """Gibbs free energy change (kJ/mol).

    FIRST_PRINCIPLES: ΔG = ΔH − TΔS

    Args:
        delta_H_kJ: reaction enthalpy in kJ/mol
        delta_n_gas: change in moles of gas (products − reactants)
        T: temperature in K

    Returns:
        ΔG in kJ/mol (negative = spontaneous).
    """
    delta_S = entropy_change_estimate(delta_n_gas, T)  # J/(mol·K)
    return delta_H_kJ - T * delta_S / 1000.0  # kJ/mol


def equilibrium_constant(delta_H_kJ, delta_n_gas, T=298.15):
    """Equilibrium constant K from thermodynamics.

    FIRST_PRINCIPLES: K = exp(−ΔG° / RT)

    This is exact thermodynamics. The approximation enters only through
    our estimates of ΔH (Pauling bonds, ±15%) and ΔS (Sackur-Tetrode, ±30%).

    Because K = exp(−ΔG/RT), even modest errors in ΔG give large errors
    in K. This function is honest: it gives the RIGHT TREND (exothermic +
    entropy-favorable → large K) but the magnitude can be off by orders
    of magnitude. This is inherent to the exponential sensitivity.

    Args:
        delta_H_kJ: reaction enthalpy in kJ/mol
        delta_n_gas: change in moles of gas
        T: temperature in K

    Returns:
        Equilibrium constant K (dimensionless).
    """
    R = K_B * N_AVOGADRO  # J/(mol·K)
    delta_G_J = gibbs_energy_kJ_mol(delta_H_kJ, delta_n_gas, T) * 1000.0

    exponent = -delta_G_J / (R * T)
    # Clamp to prevent overflow — K > 1e300 means "goes to completion"
    exponent = max(-700, min(700, exponent))
    return math.exp(exponent)


# ═══════════════════════════════════════════════════════════════════
# KINETICS
# ═══════════════════════════════════════════════════════════════════

def evans_polanyi_activation_energy(delta_H_kJ, family_key):
    """Activation energy from Evans-Polanyi relation (eV).

    FIRST_PRINCIPLES + MEASURED:
      E_a = E_0 + α × ΔH_rxn

    The Bell-Evans-Polanyi principle: within a family of related reactions,
    the activation barrier correlates linearly with the reaction enthalpy.
    More exothermic → lower barrier (late transition state is product-like).

    The Evans-Polanyi relation breaks down for:
    - Very exothermic reactions (E_a can't go negative; clamped to 0)
    - Reactions crossing families (different E_0, α)
    - Reactions with unusual transition states (pericyclic, etc.)

    Args:
        delta_H_kJ: reaction enthalpy in kJ/mol (negative = exothermic)
        family_key: key into EVANS_POLANYI_FAMILIES

    Returns:
        Activation energy in eV.
    """
    family = EVANS_POLANYI_FAMILIES[family_key]
    E_0 = family['E_0_eV']
    alpha = family['alpha']

    # Convert ΔH to eV/molecule for consistent units
    delta_H_eV = delta_H_kJ * 1000.0 / (EV_TO_J * N_AVOGADRO)

    E_a = E_0 + alpha * delta_H_eV

    # Activation energy can't be negative — clamp to zero
    # (barrierless reactions exist but EP can't predict them)
    return max(0.0, E_a)


def arrhenius_rate(A, E_a_eV, T):
    """Arrhenius rate constant k(T) = A × exp(−E_a / k_B T).

    FIRST_PRINCIPLES: transition state theory in the high-barrier limit.

    This is exact when:
    - The barrier is high compared to k_BT (E_a >> k_BT)
    - Quantum tunneling is negligible (fails for light atoms like H)
    - The transition state is well-defined (single saddle point)

    Args:
        A: pre-exponential factor (units depend on reaction order,
           typically L/(mol·s) for bimolecular)
        E_a_eV: activation energy in eV
        T: temperature in K

    Returns:
        Rate constant k in same units as A.
    """
    if T <= 0:
        return 0.0
    exponent = -E_a_eV * EV_TO_J / (K_B * T)
    # Clamp to prevent underflow
    exponent = max(-700, exponent)
    return A * math.exp(exponent)


def collision_prefactor(m_A_amu, m_B_amu, r_A_pm, r_B_pm, T=298.15,
                        steric_factor=0.1):
    """Bimolecular collision theory pre-exponential (L/(mol·s)).

    FIRST_PRINCIPLES: kinetic theory of gases.

    A = N_A × σ_coll × v_rel × p

    Where:
      σ_coll = π(r_A + r_B)² — collision cross-section
      v_rel = √(8 k_B T / (π μ)) — mean relative speed (Maxwell-Boltzmann)
      p = steric factor (MEASURED, reaction-dependent)

    This gives A in L/(mol·s) for gas-phase bimolecular reactions.
    The temperature dependence is weak (√T), so A is approximately
    constant over modest temperature ranges.

    Args:
        m_A_amu: molecular mass of reactant A in amu
        m_B_amu: molecular mass of reactant B in amu
        r_A_pm: effective collision radius of A in pm
        r_B_pm: effective collision radius of B in pm
        T: temperature in K
        steric_factor: p, fraction of collisions with correct orientation

    Returns:
        Pre-exponential factor A in L/(mol·s).
    """
    # Reduced mass in kg
    mu = (m_A_amu * m_B_amu) / (m_A_amu + m_B_amu) * AMU_KG

    # Collision cross-section in m²
    r_sum = (r_A_pm + r_B_pm) * 1e-12  # pm → m
    sigma_coll = math.pi * r_sum ** 2

    # Mean relative speed (Maxwell-Boltzmann)
    v_rel = math.sqrt(8.0 * K_B * T / (math.pi * mu))

    # Rate per unit volume (m³/(molecule·s))
    z_per_pair = sigma_coll * v_rel * steric_factor

    # Convert to L/(mol·s): multiply by N_A and by 1000 (m³→L)
    A = z_per_pair * N_AVOGADRO * 1000.0

    return A


def half_life(k):
    """Half-life from first-order rate constant.

    t_½ = ln(2) / k

    FIRST_PRINCIPLES: integration of first-order rate law.

    Args:
        k: first-order rate constant (1/s)

    Returns:
        Half-life in seconds.
    """
    if k <= 0:
        return float('inf')
    return math.log(2.0) / k


def temperature_for_rate(k_target, A, E_a_eV):
    """Temperature needed to achieve a target rate constant.

    Inverted Arrhenius: T = E_a / (k_B × ln(A / k_target))

    Useful for: "At what temperature does this reaction become fast?"

    Args:
        k_target: desired rate constant (same units as A)
        A: pre-exponential factor
        E_a_eV: activation energy in eV

    Returns:
        Temperature in K, or inf if impossible.
    """
    if k_target <= 0 or A <= 0 or k_target >= A:
        return float('inf')
    ratio = A / k_target
    if ratio <= 1.0:
        return float('inf')
    return E_a_eV * EV_TO_J / (K_B * math.log(ratio))


# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════

def reaction_report(reaction_key, T=298.15):
    """Complete thermodynamic and kinetic report for a named reaction.

    Args:
        reaction_key: key into REACTIONS dict
        T: temperature in K

    Returns:
        dict with all derived properties.
    """
    rxn = REACTIONS[reaction_key]

    # Thermodynamics
    dH = reaction_enthalpy_kJ_mol(rxn['bonds_broken'], rxn['bonds_formed'])
    dS = entropy_change_estimate(rxn['delta_n_gas'], T)
    dG = gibbs_energy_kJ_mol(dH, rxn['delta_n_gas'], T)
    K = equilibrium_constant(dH, rxn['delta_n_gas'], T)

    report = {
        'name': rxn['name'],
        'equation': rxn['equation'],
        'T_K': T,
        'delta_H_kJ_mol': dH,
        'delta_S_J_mol_K': dS,
        'delta_G_kJ_mol': dG,
        'K_eq': K,
        'spontaneous': dG < 0,
        'exothermic': dH < 0,
    }

    # Kinetics (only if family is known)
    if rxn.get('family') and rxn['family'] in EVANS_POLANYI_FAMILIES:
        E_a = evans_polanyi_activation_energy(dH, rxn['family'])
        # Estimate rate at T using typical collision prefactor
        # Use 1e10 L/(mol·s) as a reasonable gas-phase bimolecular A
        A_typical = 1e10  # L/(mol·s), order-of-magnitude for bimolecular
        k = arrhenius_rate(A_typical, E_a, T)

        report['E_a_eV'] = E_a
        report['E_a_kJ_mol'] = E_a * EV_TO_J * N_AVOGADRO / 1000.0
        report['k_at_T'] = k
        report['A_prefactor'] = A_typical

    # Comparison with measured value
    if 'measured_dH_kJ_mol' in rxn:
        measured = rxn['measured_dH_kJ_mol']
        report['measured_dH_kJ_mol'] = measured
        if abs(measured) > 0:
            report['enthalpy_error_pct'] = (
                abs(dH - measured) / abs(measured) * 100.0
            )

    return report


def full_report(T=298.15):
    """Reports for ALL known reactions.

    Returns:
        dict: {reaction_key: report_dict}
    """
    return {key: reaction_report(key, T) for key in REACTIONS}
