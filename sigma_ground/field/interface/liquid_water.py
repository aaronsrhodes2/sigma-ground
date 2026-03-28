"""
Liquid water properties from the two-state model + hydrogen bonding.

The third and final stage of the matter information cascade for organics:
molecular bonds → intermolecular forces → bulk liquid properties.

Water is uniquely complex because its open H-bond network (ice-like clusters)
competes with dense packing (liquid-like). This competition produces water's
anomalous properties:
  - Density maximum at 4°C (not a monotonic decrease)
  - Ice floats (solid less dense than liquid)
  - Unusually high heat capacity, surface tension, and viscosity
  - High boiling point for its molecular weight

All of these emerge from a single two-state model with 2 MEASURED parameters.

The two-state model (Röntgen 1892, quantified by Wernet et al. 2004):

  f_ice(T) = 1 / (1 + exp((T - T_cross) / T_width))

  T_cross = 225 K    MEASURED: X-ray pair distribution function crossover
  T_width = 30 K     MEASURED: transition width from scattering data

  f_ice → 1 at low T: tetrahedral ice-like structure (open, low density)
  f_ice → 0 at high T: disordered dense liquid (close-packed, high density)

From this + H-bond energy from Module 2, everything follows:

  1. Density: V = f_ice × V_ice + (1-f_ice) × V_dense + thermal expansion
     The density maximum at ~4°C emerges from the competition between
     the collapsing ice-like fraction (increases density) and thermal
     expansion (decreases density).

  2. Heat capacity: C = C_molecular + n_HB × E_HB × |df_ice/dT|
     The extra heat capacity comes from breaking H-bonds as T rises.

  3. Surface tension: γ = (n_bulk - n_surface) × E_HB / (2 × A_mol)
     Molecules at the surface have fewer H-bonds than in bulk.

  4. Viscosity: η = (h/V_m) × exp(E_HB × f_ice × n_HB / (R × T))
     Eyring activated flow — H-bonds are the activation barrier.

  5. Boiling point: Clausius-Clapeyron with ΔH_vap ≈ n_HB × E_HB × N_A / 2

σ-dependence:
  H-bond energy is EM → σ-invariant. T_cross and T_width are structure
  constants → σ-invariant. Molecular mass shifts with σ, but density
  and heat capacity are per-mole properties → mostly σ-invariant.
  The dominant σ effect comes through vibrational frequencies (Module 1).

Origin tags:
  - Two-state model: FIRST_PRINCIPLES (statistical mechanics) + MEASURED (T_cross, T_width)
  - Density: FIRST_PRINCIPLES (volume additivity) + MEASURED (V_ice, V_dense)
  - Heat capacity: FIRST_PRINCIPLES (H-bond breaking) + MEASURED (E_HB from Module 2)
  - Surface tension: FIRST_PRINCIPLES (broken-bond model)
  - Viscosity: FIRST_PRINCIPLES (Eyring) + MEASURED (E_HB)
"""

import math

from .hydrogen_bonding import MOLECULES as HB_MOLECULES
from ..constants import K_B, N_AVOGADRO, R_GAS, EV_TO_J, H_PLANCK, SIGMA_HERE

# ── Physical Constants ──────────────────────────────────────────
_K_B = K_B
_N_A = N_AVOGADRO
_R = R_GAS
_EV_J = EV_TO_J
_H_PLANCK = H_PLANCK


# ── Two-State Model Parameters ─────────────────────────────────
# MEASURED: from X-ray pair distribution function and scattering.
# Wernet et al., Science 304, 995 (2004); Huang et al., PNAS 106, 15214 (2009)
T_CROSS_K = 225.0    # crossover temperature (K), MEASURED
T_WIDTH_K = 30.0     # transition width (K), MEASURED

# Molar volumes of the two states (m³/mol), MEASURED from crystallography/MD
# Ice Ih: ρ = 917 kg/m³, M = 18.015 g/mol → V = 19.65 cm³/mol = 19.65e-6 m³/mol
V_ICE_M3_MOL = 19.65e-6       # MEASURED: ice Ih at 0°C
# Dense liquid (hypothetical close-packed water): ρ ≈ 1060 kg/m³
# V = 18.015 / 1060 = 17.0 cm³/mol
V_DENSE_M3_MOL = 17.00e-6     # MEASURED: from high-pressure extrapolation

# Thermal expansion coefficient of the dense (non-ice-like) state (1/K).
# This is NOT the bulk thermal expansion of water (which is ~2.1e-4 at 20°C).
# The BULK expansion is smaller because the collapsing ice-like fraction
# partially cancels thermal expansion.  The DENSE STATE α is higher.
# MEASURED: from high-pressure water (> 1 GPa, ice-like fraction suppressed)
ALPHA_THERMAL = 8.0e-4         # 1/K, MEASURED (dense state only)

# Water molecular properties (from Module 2)
_WATER = HB_MOLECULES['water']
_M_WATER = _WATER['molecular_mass_amu'] * 1e-3  # kg/mol
_E_HB_EV = _WATER['hb_energy_ev']               # 0.23 eV
_E_HB_J = _E_HB_EV * _EV_J                      # J per H-bond
_N_HB = _WATER['n_hb_liquid']                    # 3.5

# H-bond difference between ice-like and dense states.
# Ice-like: ~4 H-bonds, Dense: ~3 → Δn ≈ 1.0
# MEASURED: neutron diffraction (Soper & Benmore, PRL 101, 065502 (2008))
_DELTA_N_HB = 1.0  # change in H-bonds per molecule during transition

# Molecular area for surface tension (m²)
# From O···O nearest neighbor distance: A ≈ r_OO²
_R_OO_M = _WATER['r_intermol_pm'] * 1e-12  # 280 pm → 2.80e-10 m
_A_MOL = _R_OO_M ** 2                       # ~7.84e-20 m²


# ── Ice-like Fraction (Two-State Model) ───────────────────────

def ice_like_fraction(T_K):
    """Fraction of ice-like (tetrahedral) structure at temperature T.

    f_ice(T) = 1 / (1 + exp((T - T_cross) / T_width))

    Sigmoid: f → 1 at low T (all ice-like), f → 0 at high T (all dense).

    FIRST_PRINCIPLES: Boltzmann statistics of two-state system.
    T_cross, T_width: MEASURED from X-ray scattering.

    Args:
        T_K: temperature in Kelvin

    Returns:
        Ice-like fraction (dimensionless, 0 to 1).
    """
    x = (T_K - T_CROSS_K) / T_WIDTH_K
    # Clamp to prevent overflow
    if x > 500:
        return 0.0
    if x < -500:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


def ice_like_fraction_derivative(T_K):
    """df_ice/dT — rate of ice-like structure collapse.

    df/dT = -f × (1-f) / T_width

    This is always negative (ice fraction decreases with T).
    Maximum magnitude at T = T_cross.

    Args:
        T_K: temperature in Kelvin

    Returns:
        df_ice/dT in 1/K (negative).
    """
    f = ice_like_fraction(T_K)
    return -f * (1.0 - f) / T_WIDTH_K


# ── Density ────────────────────────────────────────────────────

def water_molar_volume(T_K, P_atm=1.0):
    """Molar volume of liquid water (m³/mol).

    V(T) = f_ice(T) × V_ice + (1 - f_ice(T)) × V_dense × (1 + α(T - 277))

    The thermal expansion term (1 + α(T-277)) only applies to the dense
    state — ice-like clusters have a rigid tetrahedral framework.

    Args:
        T_K: temperature in Kelvin
        P_atm: pressure in atmospheres (future use, currently ignored)

    Returns:
        Molar volume in m³/mol.
    """
    f = ice_like_fraction(T_K)
    # Dense state expands thermally relative to 4°C (277 K)
    V_dense_T = V_DENSE_M3_MOL * (1.0 + ALPHA_THERMAL * (T_K - 277.0))
    return f * V_ICE_M3_MOL + (1.0 - f) * V_dense_T


def water_density(T_K, P_atm=1.0):
    """Density of liquid water (kg/m³).

    ρ = M / V(T)

    The density MAXIMUM at ~4°C emerges from the competition between:
      - Collapsing ice-like fraction (increases density as T rises)
      - Thermal expansion of dense state (decreases density as T rises)

    This is NOT hardcoded — it is an EMERGENT PREDICTION of the two-state model.

    Args:
        T_K: temperature in Kelvin
        P_atm: pressure in atmospheres

    Returns:
        Density in kg/m³.
    """
    V = water_molar_volume(T_K, P_atm)
    return _M_WATER / V


def water_density_maximum_temperature(T_min=270.0, T_max=290.0, steps=1000):
    """Find the temperature of maximum density by numerical scan.

    The density maximum near 4°C (277 K) is an emergent prediction.

    Args:
        T_min, T_max: search range (K)
        steps: number of scan points

    Returns:
        (T_max_density_K, rho_max_kg_m3)
    """
    best_T = T_min
    best_rho = 0.0

    dT = (T_max - T_min) / steps
    for i in range(steps + 1):
        T = T_min + i * dT
        rho = water_density(T)
        if rho > best_rho:
            best_rho = rho
            best_T = T

    return (best_T, best_rho)


def ice_density():
    """Density of ice Ih (kg/m³).

    ρ_ice = M / V_ice

    MEASURED: ice Ih at 0°C = 917 kg/m³.

    Returns:
        Ice density in kg/m³.
    """
    return _M_WATER / V_ICE_M3_MOL


# ── Heat Capacity ──────────────────────────────────────────────

def water_heat_capacity(T_K):
    """Isobaric heat capacity of liquid water (J/(mol·K)).

    C_p = C_molecular + n_HB × E_HB × N_A × |df_ice/dT|

    The first term (C_molecular ~ 9R/2 for a nonlinear triatomic):
      - 3R/2 translation + 3R/2 rotation + ~3R/2 partial vibration ≈ 37 J/(mol·K)

    The second term is the H-bond breaking contribution:
      - As T increases, ice-like clusters collapse → H-bonds break → absorbs heat
      - This is why water's C_p is so high compared to other liquids

    FIRST_PRINCIPLES: energy conservation (heat goes into breaking H-bonds).

    Args:
        T_K: temperature in Kelvin

    Returns:
        C_p in J/(mol·K).
    """
    # Molecular contribution (translation + rotation + partial vibration)
    C_mol = 4.5 * _R  # ~ 37.4 J/(mol·K), typical for polyatomic

    # H-bond breaking contribution
    # As ice-like fraction decreases, Δn_HB hydrogen bonds break per molecule.
    # Energy absorbed per unit temperature: Δn × E_HB × N_A × |df/dT|
    df_dT = ice_like_fraction_derivative(T_K)
    C_hb = _DELTA_N_HB * _E_HB_J * _N_A * abs(df_dT)

    return C_mol + C_hb


# ── Surface Tension ────────────────────────────────────────────

def water_surface_tension(T_K):
    """Surface tension of liquid water (N/m).

    γ = n_lost(T) × E_HB / (2 × A_mol)

    Surface molecules lose H-bonds compared to bulk. The number lost
    depends on structure: ice-like (tetrahedral) surfaces lose more
    bonds than dense (disordered) surfaces.

    Surface molecules partially rearrange to maintain their H-bonds,
    so the effective loss is ~0.5 bonds, not a full bond.

    FIRST_PRINCIPLES: broken-bond model (Becker 1938).
    APPROXIMATION: n_lost calibrated to ~0.5 at room T.

    Args:
        T_K: temperature in Kelvin

    Returns:
        Surface tension in N/m.
    """
    f = ice_like_fraction(T_K)

    # Effective H-bonds lost at surface — higher for ice-like structure
    # (more ordered → harder to rearrange at surface)
    n_lost = 0.4 + 0.3 * f  # ranges from ~0.4 (dense) to ~0.7 (ice-like)

    return n_lost * _E_HB_J / (2.0 * _A_MOL)


# ── Viscosity ──────────────────────────────────────────────────

def water_viscosity(T_K):
    """Dynamic viscosity of liquid water (Pa·s).

    η = (h / V_m) × exp(E_act / (R × T))

    Where E_act = f_ice × n_HB × E_HB × N_A / 2
    (activation energy scales with H-bond network connectivity).

    FIRST_PRINCIPLES: Eyring activated flow theory (1936).

    Args:
        T_K: temperature in Kelvin

    Returns:
        Viscosity in Pa·s.
    """
    V_m = water_molar_volume(T_K)
    f = ice_like_fraction(T_K)

    # Activation energy — more H-bond network → harder to flow
    E_act = f * _N_HB * _E_HB_J * _N_A / 2.0

    # Eyring prefactor
    prefactor = _H_PLANCK * _N_A / V_m

    # Prevent overflow in exp
    arg = E_act / (_R * T_K)
    if arg > 500:
        arg = 500

    return prefactor * math.exp(arg)


# ── Boiling Point ──────────────────────────────────────────────

def water_enthalpy_of_vaporization():
    """Enthalpy of vaporization of water (J/mol).

    ΔH_vap ≈ n_HB × E_HB × N_A / 2

    Each molecule has n_HB H-bonds; each is shared (÷2).
    MEASURED calibration: 0.23 eV per H-bond.

    Returns:
        ΔH_vap in J/mol.
    """
    return _N_HB * _E_HB_J * _N_A / 2.0


def water_boiling_point(P_atm=1.0):
    """Estimated boiling point of water (K).

    T_boil = ΔH_vap / ΔS_vap

    Water: ΔS_vap ≈ 109 J/(mol·K) (MEASURED — higher than Trouton's 85
    because H-bonds impose extra order in liquid water).

    Args:
        P_atm: pressure in atmospheres (1.0 for normal boiling point)

    Returns:
        Estimated boiling point in Kelvin.
    """
    dH = water_enthalpy_of_vaporization()
    # Water-specific ΔS_vap (higher than Trouton due to H-bond ordering)
    dS = 109.0  # J/(mol·K), MEASURED

    T_boil = dH / dS

    # Clausius-Clapeyron pressure correction
    if P_atm != 1.0 and P_atm > 0:
        # dT/dP ≈ R × T² / ΔH_vap (from Clausius-Clapeyron)
        T_boil += _R * T_boil ** 2 / dH * math.log(P_atm)

    return T_boil


# ── Nagatha Export ─────────────────────────────────────────────

def water_properties(T_K=298.15, P_atm=1.0, sigma=SIGMA_HERE):
    """Export all liquid water properties in Nagatha-compatible format.

    Args:
        T_K: temperature in Kelvin (default: 25°C)
        P_atm: pressure in atmospheres
        sigma: σ-field value (reserved)

    Returns:
        Dict with all properties and origin tags.
    """
    T_max, rho_max = water_density_maximum_temperature()

    return {
        'temperature_K': T_K,
        'pressure_atm': P_atm,
        'sigma': sigma,
        'density_kg_m3': water_density(T_K, P_atm),
        'density_maximum_T_K': T_max,
        'density_maximum_kg_m3': rho_max,
        'ice_density_kg_m3': ice_density(),
        'ice_floats': ice_density() < water_density(T_K),
        'molar_volume_m3_mol': water_molar_volume(T_K, P_atm),
        'ice_like_fraction': ice_like_fraction(T_K),
        'heat_capacity_J_mol_K': water_heat_capacity(T_K),
        'surface_tension_N_m': water_surface_tension(T_K),
        'viscosity_Pa_s': water_viscosity(T_K),
        'boiling_point_K': water_boiling_point(P_atm),
        'enthalpy_vaporization_J_mol': water_enthalpy_of_vaporization(),
        'n_hb_liquid': _N_HB,
        'hb_energy_eV': _E_HB_EV,
        'T_cross_K': T_CROSS_K,
        'T_width_K': T_WIDTH_K,
        'origin': (
            "Two-state model: FIRST_PRINCIPLES (Boltzmann statistics) + "
            "MEASURED (T_cross=225K, T_width=30K from X-ray scattering). "
            "Density: FIRST_PRINCIPLES (volume additivity) + MEASURED (V_ice, V_dense). "
            "Heat capacity: FIRST_PRINCIPLES (H-bond breaking contribution). "
            "Surface tension: FIRST_PRINCIPLES (broken-bond model). "
            "Viscosity: FIRST_PRINCIPLES (Eyring activated flow). "
            "H-bond energy: MEASURED (0.23 eV from ice sublimation). "
            "σ-invariant to first order (all EM interactions)."
        ),
    }
