"""
Electrochemistry — electrode potentials, Nernst equation, Faraday's laws.

Derivation chains:

  1. Standard Electrode Potential (MEASURED)
     E° = tabulated half-cell potential vs SHE (Standard Hydrogen Electrode).
     Source: Bard, Parsons & Jordan "Standard Potentials in Aqueous Solution"
     (1985), IUPAC recommendation.

     These are MEASURED thermodynamic quantities. The ordering (activity series)
     follows directly from reduction Gibbs energies.

  2. Nernst Equation (FIRST_PRINCIPLES: thermodynamics)
     E = E° − (RT / nF) × ln(Q)

     Where:
       R = gas constant = N_A × k_B
       T = temperature (K)
       n = electrons transferred
       F = Faraday constant = N_A × e
       Q = reaction quotient [products]/[reactants]

     Derived from ΔG = ΔG° + RT ln(Q) and ΔG = −nFE.
     This is exact thermodynamics — no approximations.

  3. Faraday's Laws of Electrolysis (FIRST_PRINCIPLES)
     First law:  m = (M × I × t) / (n × F)
     Second law: m₁/m₂ = (M₁/n₁) / (M₂/n₂)

     Where:
       m = mass deposited (kg)
       M = molar mass (kg/mol)
       I = current (A)
       t = time (s)
       n = electrons per ion

     FIRST_PRINCIPLES: conservation of charge + quantization of charge.
     Every electron transferred deposits exactly one ion equivalent.

  4. Cell Potential (FIRST_PRINCIPLES: thermodynamics)
     E_cell = E_cathode − E_anode
     ΔG = −nFE_cell

     Spontaneous if E_cell > 0 (ΔG < 0).

  5. Tafel Equation (FIRST_PRINCIPLES: Butler-Volmer limit)
     η = a + b × log₁₀(j)

     Overpotential as a function of current density j.
     High-overpotential limit of Butler-Volmer kinetics.
     b = 2.303 RT / (α n F)  (Tafel slope)

     FIRST_PRINCIPLES: activated rate theory applied to charge transfer.
     α = transfer coefficient (MEASURED, typically 0.3-0.7).

  6. Conductivity of Electrolyte Solutions (Kohlrausch, FIRST_PRINCIPLES)
     κ = Σ cᵢ λᵢ

     Where cᵢ = concentration of ion i, λᵢ = molar conductivity.
     At infinite dilution: λ° is an additive property of individual ions.
     Kohlrausch's law of independent migration (1876).

σ-dependence:
  Electrode potentials are electromagnetic (electron energy levels in the
  metallic lattice + solvation energy). To first order, σ-INVARIANT.

  However, the Nernst equation contains temperature, and thermal equilibrium
  shifts through nuclear mass (heavier nuclei → shifted Debye temperature →
  shifted heat capacity → shifted T at equilibrium).

  Faraday's laws are exact and σ-invariant (charge quantization doesn't change).

  The σ-bridge: we provide sigma_nernst_shift() which shows how the
  effective temperature seen by the reaction shifts under σ, modifying
  the equilibrium potential.

Origin tags:
  - Electrode potentials: MEASURED (vs SHE, IUPAC)
  - Nernst equation: FIRST_PRINCIPLES (thermodynamics)
  - Faraday's laws: FIRST_PRINCIPLES (charge conservation)
  - Tafel equation: FIRST_PRINCIPLES (Butler-Volmer limit)
  - Transfer coefficients: MEASURED
  - σ-dependence: CORE (through thermal equilibrium shift)
"""

import math
from ..constants import K_B, E_CHARGE

# ── Fundamental electrochemical constants ─────────────────────────
_N_AVOGADRO = 6.02214076e23     # /mol (exact, 2019 SI)
_R_GAS = _N_AVOGADRO * K_B     # J/(mol·K) = 8.31446...
FARADAY = _N_AVOGADRO * E_CHARGE  # C/mol = 96485.33...


# ── Standard Electrode Potentials ─────────────────────────────────
# Half-cell reduction potentials vs SHE at 25°C, 1 atm, unit activity.
# MEASURED: Bard, Parsons & Jordan (1985), CRC Handbook.
#
# Format: element → (E° in volts, n electrons, half-reaction description)
#
# Convention: reduction reaction (Mⁿ⁺ + ne⁻ → M)

STANDARD_POTENTIALS = {
    'lithium':    {'E0_V': -3.040, 'n': 1, 'reaction': 'Li⁺ + e⁻ → Li'},
    'potassium':  {'E0_V': -2.924, 'n': 1, 'reaction': 'K⁺ + e⁻ → K'},
    'calcium':    {'E0_V': -2.868, 'n': 2, 'reaction': 'Ca²⁺ + 2e⁻ → Ca'},
    'sodium':     {'E0_V': -2.714, 'n': 1, 'reaction': 'Na⁺ + e⁻ → Na'},
    'magnesium':  {'E0_V': -2.372, 'n': 2, 'reaction': 'Mg²⁺ + 2e⁻ → Mg'},
    'aluminum':   {'E0_V': -1.662, 'n': 3, 'reaction': 'Al³⁺ + 3e⁻ → Al'},
    'titanium':   {'E0_V': -1.630, 'n': 2, 'reaction': 'Ti²⁺ + 2e⁻ → Ti'},
    'zinc':       {'E0_V': -0.762, 'n': 2, 'reaction': 'Zn²⁺ + 2e⁻ → Zn'},
    'iron':       {'E0_V': -0.447, 'n': 2, 'reaction': 'Fe²⁺ + 2e⁻ → Fe'},
    'nickel':     {'E0_V': -0.257, 'n': 2, 'reaction': 'Ni²⁺ + 2e⁻ → Ni'},
    'tin':        {'E0_V': -0.138, 'n': 2, 'reaction': 'Sn²⁺ + 2e⁻ → Sn'},
    'hydrogen':   {'E0_V':  0.000, 'n': 2, 'reaction': '2H⁺ + 2e⁻ → H₂'},
    'copper':     {'E0_V': +0.342, 'n': 2, 'reaction': 'Cu²⁺ + 2e⁻ → Cu'},
    'silver':     {'E0_V': +0.800, 'n': 1, 'reaction': 'Ag⁺ + e⁻ → Ag'},
    'platinum':   {'E0_V': +1.188, 'n': 2, 'reaction': 'Pt²⁺ + 2e⁻ → Pt'},
    'gold':       {'E0_V': +1.498, 'n': 3, 'reaction': 'Au³⁺ + 3e⁻ → Au'},
}


# ── Nernst Equation ──────────────────────────────────────────────

def nernst_potential(E0, n, Q, T=298.15):
    """Electrode potential from the Nernst equation (V).

    E = E° − (RT / nF) × ln(Q)

    FIRST_PRINCIPLES: from ΔG = ΔG° + RT ln(Q) and ΔG = −nFE.
    Exact thermodynamics — no approximations.

    Args:
        E0: standard electrode potential E° (V)
        n: number of electrons transferred
        Q: reaction quotient [products]/[reactants]
        T: temperature in Kelvin (default 298.15 = 25°C)

    Returns:
        Electrode potential in Volts
    """
    if Q <= 0:
        raise ValueError(f"Q={Q}: reaction quotient must be positive")
    if n <= 0:
        raise ValueError(f"n={n}: electrons transferred must be positive")

    return E0 - (_R_GAS * T / (n * FARADAY)) * math.log(Q)


def cell_potential(cathode_key, anode_key, Q=1.0, T=298.15):
    """Cell potential for a galvanic cell (V).

    E_cell = E_cathode − E_anode

    At standard conditions (Q=1): E_cell = E°_cathode − E°_anode.
    With non-standard concentrations: applies Nernst to each half-cell.

    Spontaneous (galvanic) if E_cell > 0.

    Args:
        cathode_key: key into STANDARD_POTENTIALS (reduction occurs here)
        anode_key: key into STANDARD_POTENTIALS (oxidation occurs here)
        Q: overall reaction quotient (default 1.0 = standard)
        T: temperature in Kelvin

    Returns:
        Cell potential in Volts
    """
    E_cathode = STANDARD_POTENTIALS[cathode_key]['E0_V']
    E_anode = STANDARD_POTENTIALS[anode_key]['E0_V']
    n_cathode = STANDARD_POTENTIALS[cathode_key]['n']

    E_cell_std = E_cathode - E_anode

    if Q == 1.0:
        return E_cell_std

    # Apply Nernst to overall cell
    return E_cell_std - (_R_GAS * T / (n_cathode * FARADAY)) * math.log(Q)


def gibbs_energy_cell(cathode_key, anode_key, Q=1.0, T=298.15):
    """Gibbs free energy of cell reaction (J/mol).

    ΔG = −nFE_cell

    FIRST_PRINCIPLES: thermodynamic identity.
    Negative ΔG → spontaneous reaction.

    Args:
        cathode_key: key into STANDARD_POTENTIALS
        anode_key: key into STANDARD_POTENTIALS
        Q: reaction quotient
        T: temperature (K)

    Returns:
        ΔG in J/mol (negative = spontaneous)
    """
    E = cell_potential(cathode_key, anode_key, Q, T)
    n = STANDARD_POTENTIALS[cathode_key]['n']
    return -n * FARADAY * E


def is_spontaneous(cathode_key, anode_key, Q=1.0, T=298.15):
    """Check if cell reaction is spontaneous (ΔG < 0, E_cell > 0).

    Args:
        cathode_key, anode_key: keys into STANDARD_POTENTIALS
        Q: reaction quotient
        T: temperature (K)

    Returns:
        True if spontaneous
    """
    return cell_potential(cathode_key, anode_key, Q, T) > 0


# ── Activity Series ──────────────────────────────────────────────

def activity_series():
    """Return elements ordered by standard potential (most reactive first).

    MEASURED: the electrochemical activity series.
    More negative E° → more easily oxidized → more reactive metal.

    Returns:
        List of (element, E°) tuples, sorted from most to least reactive
    """
    return sorted(
        [(k, v['E0_V']) for k, v in STANDARD_POTENTIALS.items()],
        key=lambda x: x[1]
    )


def can_displace(metal_active, metal_passive):
    """Check if active metal can displace passive metal from solution.

    Metal with more negative E° displaces metal with less negative E°.
    E.g., iron displaces copper: Fe + Cu²⁺ → Fe²⁺ + Cu

    FIRST_PRINCIPLES: thermodynamic spontaneity.

    Args:
        metal_active: proposed reducing agent (key)
        metal_passive: metal ion in solution (key)

    Returns:
        True if displacement is thermodynamically favorable
    """
    E_active = STANDARD_POTENTIALS[metal_active]['E0_V']
    E_passive = STANDARD_POTENTIALS[metal_passive]['E0_V']
    return E_active < E_passive


# ── Faraday's Laws ───────────────────────────────────────────────

def faraday_mass_deposited(molar_mass_kg, current, time, n_electrons):
    """Mass deposited by electrolysis (kg).

    m = M × I × t / (n × F)

    FIRST_PRINCIPLES: Faraday's first law. Each mole of electrons
    (F coulombs) deposits M/n kg of material.

    Args:
        molar_mass_kg: molar mass M (kg/mol)
        current: current I (A)
        time: time t (s)
        n_electrons: electrons per ion

    Returns:
        Mass deposited in kg
    """
    return molar_mass_kg * current * time / (n_electrons * FARADAY)


def faraday_charge_required(molar_mass_kg, mass_kg, n_electrons):
    """Total charge required to deposit given mass (C).

    q = n × F × m / M

    FIRST_PRINCIPLES: inversion of Faraday's first law.

    Args:
        molar_mass_kg: molar mass M (kg/mol)
        mass_kg: target mass (kg)
        n_electrons: electrons per ion

    Returns:
        Charge in Coulombs
    """
    return n_electrons * FARADAY * mass_kg / molar_mass_kg


def faraday_time_required(molar_mass_kg, mass_kg, current, n_electrons):
    """Time required to deposit given mass at given current (s).

    t = n × F × m / (M × I)

    Args:
        molar_mass_kg: molar mass M (kg/mol)
        mass_kg: target mass (kg)
        current: current I (A)
        n_electrons: electrons per ion

    Returns:
        Time in seconds
    """
    q = faraday_charge_required(molar_mass_kg, mass_kg, n_electrons)
    return q / current


# ── Tafel Kinetics ───────────────────────────────────────────────

def tafel_slope(T=298.15, alpha=0.5, n=1):
    """Tafel slope b (V/decade).

    b = 2.303 RT / (α n F)

    FIRST_PRINCIPLES: high-overpotential limit of Butler-Volmer equation.
    α is the transfer coefficient (MEASURED, typically 0.3-0.7).

    Args:
        T: temperature (K)
        alpha: transfer coefficient (dimensionless)
        n: electrons in rate-determining step

    Returns:
        Tafel slope in V/decade
    """
    return 2.303 * _R_GAS * T / (alpha * n * FARADAY)


def tafel_overpotential(j, j0, T=298.15, alpha=0.5, n=1):
    """Tafel overpotential (V).

    η = b × log₁₀(j / j₀)

    Where b = Tafel slope, j₀ = exchange current density.

    Valid only for |η| > ~50 mV (high overpotential regime).
    Below that, Butler-Volmer linearizes to η = RT j / (nFj₀).

    Args:
        j: current density (A/m²)
        j0: exchange current density (A/m²)
        T: temperature (K)
        alpha: transfer coefficient
        n: electrons in RDS

    Returns:
        Overpotential in Volts
    """
    if j <= 0 or j0 <= 0:
        raise ValueError("Current densities must be positive")
    b = tafel_slope(T, alpha, n)
    return b * math.log10(j / j0)


# ── Ionic Conductivity ──────────────────────────────────────────

def molar_conductivity_dilute(lambda_cation, lambda_anion):
    """Molar conductivity at infinite dilution (S·m²/mol).

    Λ° = λ°₊ + λ°₋

    FIRST_PRINCIPLES: Kohlrausch's law of independent migration (1876).
    At infinite dilution, ions do not interact — conductivities are additive.

    Args:
        lambda_cation: limiting molar conductivity of cation (S·m²/mol)
        lambda_anion: limiting molar conductivity of anion (S·m²/mol)

    Returns:
        Molar conductivity in S·m²/mol
    """
    return lambda_cation + lambda_anion


def solution_conductivity(concentration, molar_conductivity):
    """Conductivity of electrolyte solution (S/m).

    κ = c × Λ

    Args:
        concentration: molar concentration c (mol/m³)
        molar_conductivity: Λ (S·m²/mol)

    Returns:
        Conductivity in S/m
    """
    return concentration * molar_conductivity


# ── σ-Dependence ─────────────────────────────────────────────────

def sigma_nernst_shift(E0, n, Q, T, sigma):
    """Nernst potential shift under σ-field.

    The σ-field modifies effective temperature through nuclear mass
    shift → Debye temperature shift → thermal equilibrium shift.

    At the electrochemical level, the dominant effect is the RT/nF
    thermal voltage term. Under σ, the effective thermal voltage shifts
    because the heat capacity of the electrode-solution system changes.

    For small σ (Earth-like): the shift is negligible (<10⁻⁹ V).
    For large σ (neutron star surface): measurable mV-scale shifts.

    CORE: σ-dependence through thermal equilibrium (□σ = −ξR).

    Args:
        E0: standard potential (V)
        n: electrons transferred
        Q: reaction quotient
        T: temperature (K)
        sigma: σ-field value

    Returns:
        Modified Nernst potential in Volts
    """
    from ..scale import scale_ratio
    from ..constants import PROTON_QCD_FRACTION

    if sigma == 0.0:
        return nernst_potential(E0, n, Q, T)

    # σ shifts the effective temperature through Debye temperature
    # Θ_D ∝ √(K/M) — heavier nuclei lower Θ_D, shifting thermal equilibrium
    f_qcd = PROTON_QCD_FRACTION
    mass_ratio = (1.0 - f_qcd) + f_qcd * scale_ratio(sigma)

    # Effective temperature shift: T_eff = T × √(1/mass_ratio)
    # (phonon softening from heavier nuclei)
    T_eff = T / math.sqrt(mass_ratio)

    return nernst_potential(E0, n, Q, T_eff)


# ── Nagatha Integration ──────────────────────────────────────────

def material_electrochemical_properties(element_key, T=298.15, sigma=0.0):
    """Export electrochemical properties in Nagatha-compatible format.

    Returns a dict for elements in the STANDARD_POTENTIALS table.
    """
    if element_key not in STANDARD_POTENTIALS:
        return {'error': f'{element_key} not in electrode potential table'}

    data = STANDARD_POTENTIALS[element_key]
    E0 = data['E0_V']
    n = data['n']

    E_sigma = sigma_nernst_shift(E0, n, 1.0, T, sigma)
    series = activity_series()
    rank = next(i for i, (k, _) in enumerate(series) if k == element_key)

    return {
        'standard_potential_V': E0,
        'n_electrons': n,
        'reaction': data['reaction'],
        'nernst_at_sigma_V': E_sigma,
        'activity_rank': rank,
        'temperature_K': T,
        'sigma': sigma,
        'tafel_slope_V_decade': tafel_slope(T),
        'origin_tag': (
            "MEASURED: standard electrode potentials (Bard et al. 1985, IUPAC). "
            "FIRST_PRINCIPLES: Nernst equation (thermodynamics). "
            "FIRST_PRINCIPLES: Faraday's laws (charge conservation). "
            "FIRST_PRINCIPLES: Tafel kinetics (Butler-Volmer limit). "
            "CORE: σ-dependence through thermal equilibrium shift."
        ),
    }
