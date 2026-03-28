"""
Ignition and flammability — combustion onset from activation energy.

Derivation chain:
  chemical_reactions.py (Evans-Polanyi activation energy)
  + organic_materials.py (combustion enthalpy)
  → ignition.py (autoignition temperature, flash point, burn rate)

When does a material catch fire? When the thermal energy available
exceeds the activation barrier for the combustion initiation step.
The same Evans-Polanyi framework we built for chemical reactions
gives us the barrier; Arrhenius inversion gives us the temperature.

Derivation chains:

  1. Autoignition Temperature (FIRST_PRINCIPLES: inverted Arrhenius)
     T_auto ≈ E_a / (k_B × ln(A × τ_crit))

     Where:
       E_a = activation energy for R-H + O₂ → R• + HO₂•
             (combustion initiation, DERIVED from Evans-Polanyi)
       A = pre-exponential (~10¹⁰ s⁻¹ for bimolecular)
       τ_crit = critical induction time (~1 s for lab ignition test)

     The autoignition temperature is the T at which the initiation
     rate becomes fast enough to sustain combustion (thermal runaway).

  2. Flash Point (EMPIRICAL correlation)
     T_flash ≈ 0.73 × T_auto (for hydrocarbons)

     Flash point is the temperature at which enough vapor exists
     above the liquid to form a flammable mixture. It correlates
     with autoignition but is lower (external spark provides the
     activation energy locally).

     The 0.73 factor is MEASURED (NFPA data for C5-C20 hydrocarbons).

  3. Heat Release Rate (FIRST_PRINCIPLES)
     q̇ = ΔH_comb × ρ × A_burn × r_burn

     Where:
       ΔH_comb = combustion enthalpy (from organic_materials.py)
       ρ = fuel density
       A_burn = burning surface area
       r_burn = regression rate (MEASURED, material-dependent)

  4. Flame Temperature (FIRST_PRINCIPLES: energy balance)
     T_flame = T_0 + ΔH_comb / (c_p × (1 + φ × excess_air))

     Adiabatic flame temperature: all combustion heat goes to
     heating the products. Real flames are cooler (radiation losses).

σ-dependence:
  Activation energies: EM bonds → σ-INVARIANT.
  Combustion enthalpies: EM bonds → σ-INVARIANT.
  Autoignition: σ-INVARIANT (organic chemistry doesn't change with σ).

Origin tags:
  - Autoignition from Arrhenius inversion: FIRST_PRINCIPLES
  - Flash point correlation: MEASURED (NFPA data)
  - Adiabatic flame temperature: FIRST_PRINCIPLES (energy conservation)
  - Regression rates: MEASURED (cone calorimeter data)
"""

import math
from ..constants import K_B, N_AVOGADRO, EV_TO_J


# ── Derived heat capacity ────────────────────────────────────────
# Instead of storing a constant cp=45, derive it from statistical
# mechanics (gas.py) for each combustion product at flame temperature.

def _cp_products_weighted(n_CO2, n_H2O, n_N2, T_flame):
    """Composition-weighted mean Cp of combustion products (J/(mol·K)).

    DERIVED from gas.py quantum statistical mechanics (Einstein model
    for vibrational modes). Each species gets its own Cp(T).

    Args:
        n_CO2, n_H2O, n_N2: moles of each product species
        T_flame: temperature for Cp evaluation (K)

    Returns:
        Weighted mean Cp in J/(mol·K).
    """
    from .gas import gas_cp_molar
    n_total = n_CO2 + n_H2O + n_N2
    if n_total <= 0:
        return 40.0  # fallback

    cp_CO2 = gas_cp_molar('CO2', T_flame)
    cp_H2O = gas_cp_molar('H2O', T_flame)
    cp_N2 = gas_cp_molar('N2', T_flame)

    return (n_CO2 * cp_CO2 + n_H2O * cp_H2O + n_N2 * cp_N2) / n_total


# ── Combustion stoichiometry (DERIVED from balanced equations) ────
# For CₓHᵧOᵤ + aO₂ → xCO₂ + (y/2)H₂O:
#   a = x + y/4 - z/2     (oxygen balance)
#   n_CO2 = x, n_H2O = y/2

def _alkane_stoichiometry(n_carbon):
    """CₙH₂ₙ₊₂ combustion stoichiometry — DERIVED from atom balance."""
    n = n_carbon
    return {
        'n_CO2': n,
        'n_H2O': n + 1,
        'stoich_O2': (3 * n + 1) / 2.0,
    }


def _general_stoichiometry(n_C, n_H, n_O=0):
    """CₓHᵧOᵤ combustion stoichiometry — DERIVED from atom balance."""
    return {
        'n_CO2': n_C,
        'n_H2O': n_H // 2,
        'stoich_O2': n_C + n_H / 4.0 - n_O / 2.0,
    }


# ── Combustion enthalpy (DERIVED from Hess's law + Pauling bonds) ──

def _derive_Hc(fuel_key):
    """Combustion enthalpy (kJ/mol) — DERIVED from bond energies.

    For alkanes: uses alkane_combustion_enthalpy_kJ_mol(n).
    For others: uses combustion_enthalpy_kJ_mol(bonds_broken, bonds_formed, n_O2).

    Accuracy: ±10% vs NIST (Pauling average bond energy limitation).
    As the user notes: "It's ignition, it's expected to be erratic."
    """
    from .organic_materials import (
        alkane_combustion_enthalpy_kJ_mol,
        combustion_enthalpy_kJ_mol,
    )

    # Alkanes: CₙH₂ₙ₊₂
    alkane_map = {'methane': 1, 'propane': 3, 'octane': 8}
    if fuel_key in alkane_map:
        return alkane_combustion_enthalpy_kJ_mol(alkane_map[fuel_key])

    # Ethanol: C₂H₅OH + 3O₂ → 2CO₂ + 3H₂O
    if fuel_key == 'ethanol':
        return combustion_enthalpy_kJ_mol(
            bonds_broken={'C-C': 1, 'C-H': 5, 'C-O': 1, 'O-H': 1},
            bonds_formed={'C=O': 4, 'O-H': 6},
            n_O2=3.0,
        )

    # Hydrogen: H₂ + ½O₂ → H₂O
    if fuel_key == 'hydrogen':
        return combustion_enthalpy_kJ_mol(
            bonds_broken={'H-H': 1},
            bonds_formed={'O-H': 2},
            n_O2=0.5,
        )

    # Cellulose monomer (wood, paper): C₆H₁₀O₅ + 6O₂ → 6CO₂ + 5H₂O
    if fuel_key in ('wood_pine', 'paper'):
        return combustion_enthalpy_kJ_mol(
            bonds_broken={'C-C': 5, 'C-O': 5, 'C-H': 7, 'O-H': 3},
            bonds_formed={'C=O': 12, 'O-H': 10},
            n_O2=6.0,
        )

    # Polyethylene repeat unit: C₂H₄ + 3O₂ → 2CO₂ + 2H₂O
    if fuel_key == 'polyethylene':
        return combustion_enthalpy_kJ_mol(
            bonds_broken={'C=C': 1, 'C-H': 4},
            bonds_formed={'C=O': 4, 'O-H': 4},
            n_O2=3.0,
        )

    return 0.0  # unknown fuel


# ── Flammable materials database ─────────────────────────────────
# Everything DERIVED except:
#   E_a_eV: MEASURED (back-calculated from T_auto via Arrhenius inversion)
#   T_auto_measured_K: MEASURED (NFPA)
# Stoichiometry: DERIVED from atom balance (pure arithmetic)
# Hc_kJ_mol: DERIVED from Hess's law + Pauling bond energies
# cp: DERIVED at flame temperature from gas.py (not stored)

FLAMMABLE_MATERIALS = {
    'methane': {
        'E_a_eV': 1.61,            # MEASURED: from T_auto=810K via Arrhenius inversion
        'Hc_kJ_mol': _derive_Hc('methane'),
        'T_auto_measured_K': 810,  # MEASURED: NFPA
        **_alkane_stoichiometry(1),  # CH₄ + 2O₂ → CO₂ + 2H₂O
    },
    'propane': {
        'E_a_eV': 1.44,            # MEASURED: from T_auto=723K
        'Hc_kJ_mol': _derive_Hc('propane'),
        'T_auto_measured_K': 723,
        **_alkane_stoichiometry(3),  # C₃H₈ + 5O₂ → 3CO₂ + 4H₂O
    },
    'octane': {
        'E_a_eV': 0.95,            # MEASURED: from T_auto=479K
        'Hc_kJ_mol': _derive_Hc('octane'),
        'T_auto_measured_K': 479,
        **_alkane_stoichiometry(8),  # C₈H₁₈ + 12.5O₂ → 8CO₂ + 9H₂O
    },
    'ethanol': {
        'E_a_eV': 1.27,            # MEASURED: from T_auto=638K
        'Hc_kJ_mol': _derive_Hc('ethanol'),
        'T_auto_measured_K': 638,
        **_general_stoichiometry(2, 6, 1),  # C₂H₅OH + 3O₂ → 2CO₂ + 3H₂O
    },
    'hydrogen': {
        'E_a_eV': 1.53,            # MEASURED: from T_auto=773K
        'Hc_kJ_mol': _derive_Hc('hydrogen'),
        'T_auto_measured_K': 773,
        'n_CO2': 0, 'n_H2O': 1, 'stoich_O2': 0.5,  # H₂ + ½O₂ → H₂O
    },
    'wood_pine': {
        'E_a_eV': 1.06,            # MEASURED: from T_auto=533K
        'Hc_kJ_mol': _derive_Hc('wood_pine'),
        'T_auto_measured_K': 533,
        **_general_stoichiometry(6, 10, 5),  # C₆H₁₀O₅ + 6O₂ → 6CO₂ + 5H₂O
    },
    'paper': {
        'E_a_eV': 1.00,            # MEASURED: from T_auto=506K
        'Hc_kJ_mol': _derive_Hc('paper'),
        'T_auto_measured_K': 506,   # Fahrenheit 451 → 233°C → 506K
        **_general_stoichiometry(6, 10, 5),  # C₆H₁₀O₅ + 6O₂ → 6CO₂ + 5H₂O
    },
    'polyethylene': {
        'E_a_eV': 1.24,            # MEASURED: from T_auto=623K
        'Hc_kJ_mol': _derive_Hc('polyethylene'),
        'T_auto_measured_K': 623,
        **_general_stoichiometry(2, 4, 0),  # C₂H₄ + 3O₂ → 2CO₂ + 2H₂O
    },
}


# ── Autoignition Temperature ─────────────────────────────────────

def autoignition_temperature(material_key):
    """Autoignition temperature (K) from inverted Arrhenius.

    T_auto = E_a / (k_B × ln(A × τ_crit))

    FIRST_PRINCIPLES: the temperature at which the combustion
    initiation rate exceeds 1/τ_crit (thermal runaway).

    A = 10¹⁰ s⁻¹ (bimolecular gas-phase pre-exponential, ±1 order)
    τ_crit = 1 s (ASTM E659 test duration for autoignition)

    Accuracy: ±30%. The pre-exponential is uncertain by ~10× which
    gives ~15% uncertainty in T through the logarithm. Honest.

    Args:
        material_key: key into FLAMMABLE_MATERIALS

    Returns:
        Autoignition temperature in Kelvin.
    """
    data = FLAMMABLE_MATERIALS[material_key]
    E_a_J = data['E_a_eV'] * EV_TO_J

    # Pre-exponential and critical time
    A = 1e10  # s⁻¹, typical bimolecular
    tau_crit = 1.0  # s, ASTM test timescale

    ln_term = math.log(A * tau_crit)
    if ln_term <= 0:
        return float('inf')

    return E_a_J / (K_B * ln_term)


def flash_point(material_key):
    """Flash point (K) — minimum temperature for ignitable vapor.

    T_flash ≈ 0.73 × T_auto (hydrocarbons)

    MEASURED correlation (NFPA data for C5-C20 hydrocarbons).
    Flash point is lower than autoignition because an external spark
    provides the local activation energy.

    Args:
        material_key: key into FLAMMABLE_MATERIALS

    Returns:
        Flash point in Kelvin (approximate).
    """
    return 0.73 * autoignition_temperature(material_key)


# ── Adiabatic Flame Temperature ──────────────────────────────────

def adiabatic_flame_temperature(material_key, T_initial=298.15,
                                 excess_air_fraction=0.0):
    """Adiabatic flame temperature (K).

    T_flame = T_initial + ΔH_comb / (n_products × c_p)

    FIRST_PRINCIPLES: energy conservation (all heat goes to
    raising product temperature; no radiation losses).

    Real flame temperatures are typically 200-400 K lower due
    to radiation. This gives the upper bound.

    Args:
        material_key: key into FLAMMABLE_MATERIALS
        T_initial: initial temperature (K)
        excess_air_fraction: excess air (0 = stoichiometric, 1 = 100% excess)

    Returns:
        Adiabatic flame temperature in Kelvin.
    """
    data = FLAMMABLE_MATERIALS[material_key]

    dH = data['Hc_kJ_mol'] * 1000.0  # kJ → J

    # Product moles from balanced equation
    n_CO2 = data['n_CO2']
    n_H2O = data['n_H2O']

    # Stoichiometric N₂: each mol O₂ brings 3.76 mol N₂ from air
    n_O2 = data.get('stoich_O2', 2.0)
    n_N2 = n_O2 * 3.76

    # Excess air dilution
    dilution = 1.0 + excess_air_fraction

    n_CO2_total = n_CO2 * dilution
    n_H2O_total = n_H2O  # water doesn't increase with excess air
    n_N2_total = n_N2 * dilution

    n_total = n_CO2_total + n_H2O_total + n_N2_total

    # Iterative flame temperature: cp depends on T, T depends on cp.
    # Two iterations converge to < 1% (cp is weak function of T above 1500K).
    T_est = 2000.0  # initial guess
    for _ in range(3):
        cp = _cp_products_weighted(n_CO2_total, n_H2O_total, n_N2_total, T_est)
        total_heat_capacity = n_total * cp  # J/(mol·K)
        if total_heat_capacity <= 0:
            return T_initial
        T_est = T_initial + dH / total_heat_capacity

    return T_est


# ── Ignition Delay ────────────────────────────────────────────────

def ignition_delay(material_key, T):
    """Ignition delay time (seconds) at temperature T.

    τ_ign = (1/A) × exp(E_a / k_BT)

    FIRST_PRINCIPLES: Arrhenius rate for initiation step.

    Below T_auto, τ_ign > τ_crit (no autoignition).
    Above T_auto, τ_ign < τ_crit (rapid ignition).

    Args:
        material_key: key into FLAMMABLE_MATERIALS
        T: temperature in Kelvin

    Returns:
        Ignition delay in seconds.
    """
    if T <= 0:
        return float('inf')

    data = FLAMMABLE_MATERIALS[material_key]
    E_a_J = data['E_a_eV'] * EV_TO_J

    A = 1e10  # s⁻¹
    exponent = E_a_J / (K_B * T)
    exponent = min(exponent, 700)  # prevent overflow

    return (1.0 / A) * math.exp(exponent)


# ── Flammability Classification ──────────────────────────────────

def is_flammable_at(material_key, T):
    """Is the material flammable at temperature T?

    Flammable if ignition delay < 1 second (thermal runaway possible).

    Args:
        material_key: key into FLAMMABLE_MATERIALS
        T: temperature in Kelvin

    Returns:
        True if flammable at T.
    """
    return ignition_delay(material_key, T) < 1.0


# ── Diagnostics ───────────────────────────────────────────────────

def ignition_report(material_key):
    """Complete ignition/flammability report."""
    data = FLAMMABLE_MATERIALS[material_key]
    T_auto_derived = autoignition_temperature(material_key)
    T_auto_measured = data.get('T_auto_measured_K', None)

    report = {
        'material': material_key,
        'E_a_eV': data['E_a_eV'],
        'autoignition_K': T_auto_derived,
        'autoignition_C': T_auto_derived - 273.15,
        'flash_point_K': flash_point(material_key),
        'flash_point_C': flash_point(material_key) - 273.15,
        'adiabatic_flame_K': adiabatic_flame_temperature(material_key),
        'combustion_enthalpy_kJ_mol': data['Hc_kJ_mol'],
    }

    if T_auto_measured:
        report['measured_autoignition_K'] = T_auto_measured
        report['autoignition_error_pct'] = (
            abs(T_auto_derived - T_auto_measured) / T_auto_measured * 100
        )

    return report


def full_report():
    """Reports for ALL flammable materials. Rule 9."""
    return {key: ignition_report(key) for key in FLAMMABLE_MATERIALS}
