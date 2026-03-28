"""
Atmosphere — thermodynamics of a planetary gas envelope.

Derivation chain:
  gas.py (cp, gamma, viscosity, thermal conductivity for N2/O2/CO2/H2O/CH4)
  + liquid_water.py (enthalpy of vaporization from H-bond model)
  → atmosphere.py (lapse rate, vapor pressure, humidity, scale height)

What a climate scientist needs from the material properties cascade:

  1. Air as a mixture: mean molecular weight, effective cp, gamma, kappa
     → DERIVED from gas.py per-species properties, composition-weighted

  2. Barometric formula P(z) = P0 exp(-Mgz/RT)
     → FIRST_PRINCIPLES: hydrostatic equilibrium + ideal gas

  3. Dry adiabatic lapse rate Gamma_d = g/cp
     → FIRST_PRINCIPLES: adiabatic ascent of a parcel

  4. Saturation vapor pressure P_sat(T)
     → FIRST_PRINCIPLES: Clausius-Clapeyron (liquid-vapor)
     → MEASURED: L_vap from liquid_water.py H-bond model

  5. Humidity: relative, absolute, dew point, mixing ratio
     → DERIVED from P_sat

  6. Moist adiabatic lapse rate
     → FIRST_PRINCIPLES: adiabatic + condensation heating

  7. Speed of sound in air
     → FIRST_PRINCIPLES: v = sqrt(gamma R T / M)

sigma-dependence:
  Molecular masses shift with sigma → M_air shifts → scale height,
  lapse rate, sound speed all change. Heavier nuclei at higher sigma
  give denser atmosphere, lower scale height, slower lapse rate change.
  The greenhouse effect itself is sigma-INVARIANT (EM absorption bands),
  but atmospheric structure depends on sigma through gravity coupling.

Origin tags:
  - Air composition: MEASURED (NOAA, present-day dry air)
  - Barometric formula: FIRST_PRINCIPLES (hydrostatic + ideal gas)
  - Clausius-Clapeyron: FIRST_PRINCIPLES (thermodynamic identity)
  - Lapse rates: FIRST_PRINCIPLES (energy conservation)
  - L_vap for water: DERIVED from H-bond model (liquid_water.py)
"""

import math
from ..constants import K_B, R_GAS, N_AVOGADRO, SIGMA_HERE

# Standard gravity (MEASURED, exact by definition)
_G_STANDARD = 9.80665  # m/s²

# Standard atmosphere (MEASURED, exact by definition)
_P_STANDARD = 101325.0  # Pa
_T_STANDARD = 288.15    # K (15°C, ISA sea level)


# ═══════════════════════════════════════════════════════════════════
# DRY AIR COMPOSITION
# ═══════════════════════════════════════════════════════════════════
# MEASURED: NOAA Global Monitoring Laboratory, present-day values.
# Volume fractions (= mole fractions for ideal gas).

DRY_AIR_COMPOSITION = {
    'N2':  0.7808,    # 78.08%
    'O2':  0.2095,    # 20.95%
    'Ar':  0.00934,   # 0.934% (noble gas — no vibrational modes)
    'CO2': 0.000420,  # 420 ppm (2024 value, rising ~2.5 ppm/yr)
    'CH4': 0.0000019, # 1.9 ppm
    'CO':  0.0000001, # ~100 ppb
}

# Argon properties (not in gas.py MOLECULES — monatomic noble gas)
_AR_MASS_KG = 39.948 / (N_AVOGADRO * 1000.0)  # kg per atom
_AR_CP = 2.5 * R_GAS  # monatomic: cp = 5/2 R (exact)


# ═══════════════════════════════════════════════════════════════════
# AIR MIXTURE PROPERTIES
# ═══════════════════════════════════════════════════════════════════

def mean_molecular_mass_kg(sigma=SIGMA_HERE):
    """Mean molecular mass of dry air (kg/molecule).

    M_air = sum(x_i × m_i) — DERIVED from composition + gas.py masses.

    Args:
        sigma: sigma-field value

    Returns:
        Mean molecular mass in kg.
    """
    from .gas import molecular_mass_kg, MOLECULES

    total = 0.0
    for species, frac in DRY_AIR_COMPOSITION.items():
        if species == 'Ar':
            total += frac * _AR_MASS_KG
        elif species in MOLECULES:
            total += frac * molecular_mass_kg(species, sigma)
    return total


def mean_molar_mass_kg_mol(sigma=SIGMA_HERE):
    """Mean molar mass of dry air (kg/mol).

    Returns:
        M in kg/mol (≈ 0.02897 for Earth air).
    """
    return mean_molecular_mass_kg(sigma) * N_AVOGADRO


def air_cp_molar(T=288.15, sigma=SIGMA_HERE):
    """Molar heat capacity Cp of dry air (J/(mol·K)).

    DERIVED: composition-weighted from gas.py per-species Cp(T).

    Args:
        T: temperature in Kelvin
        sigma: sigma-field value

    Returns:
        Cp in J/(mol·K).
    """
    from .gas import gas_cp_molar, MOLECULES

    total = 0.0
    for species, frac in DRY_AIR_COMPOSITION.items():
        if species == 'Ar':
            total += frac * _AR_CP
        elif species in MOLECULES:
            total += frac * gas_cp_molar(species, T, sigma)
    return total


def air_cp_mass(T=288.15, sigma=SIGMA_HERE):
    """Specific heat capacity of dry air (J/(kg·K)).

    cp = Cp_molar / M_air

    Args:
        T: temperature in Kelvin
        sigma: sigma-field value

    Returns:
        cp in J/(kg·K) (≈ 1005 for Earth air at 288K).
    """
    M = mean_molar_mass_kg_mol(sigma)
    if M <= 0:
        return 0.0
    return air_cp_molar(T, sigma) / M


def air_gamma(T=288.15, sigma=SIGMA_HERE):
    """Adiabatic index gamma = Cp/Cv for dry air.

    DERIVED from composition-weighted Cp and Cv.

    Args:
        T: temperature in Kelvin
        sigma: sigma-field value

    Returns:
        gamma (≈ 1.400 for Earth air at 288K).
    """
    from .gas import gas_cv_molar, gas_cp_molar, MOLECULES

    cp_total = 0.0
    cv_total = 0.0
    for species, frac in DRY_AIR_COMPOSITION.items():
        if species == 'Ar':
            cp_total += frac * _AR_CP
            cv_total += frac * 1.5 * R_GAS  # monatomic cv = 3/2 R
        elif species in MOLECULES:
            cp_total += frac * gas_cp_molar(species, T, sigma)
            cv_total += frac * gas_cv_molar(species, T, sigma)

    if cv_total <= 0:
        return 1.4
    return cp_total / cv_total


def air_density(T=288.15, P=_P_STANDARD, sigma=SIGMA_HERE):
    """Density of dry air (kg/m³).

    rho = P M / (R T) — FIRST_PRINCIPLES: ideal gas law.

    Args:
        T: temperature (K)
        P: pressure (Pa)
        sigma: sigma-field value

    Returns:
        Density in kg/m³ (≈ 1.225 at sea level).
    """
    if T <= 0:
        return float('inf')
    M = mean_molar_mass_kg_mol(sigma)
    return P * M / (R_GAS * T)


# ═══════════════════════════════════════════════════════════════════
# SPEED OF SOUND
# ═══════════════════════════════════════════════════════════════════

def speed_of_sound(T=288.15, sigma=SIGMA_HERE):
    """Speed of sound in dry air (m/s).

    v = sqrt(gamma × R × T / M)

    FIRST_PRINCIPLES: Newton-Laplace equation for an ideal gas.
    The adiabatic bulk modulus K_s = gamma × P, and v = sqrt(K_s/rho).

    Args:
        T: temperature (K)
        sigma: sigma-field value

    Returns:
        Speed of sound in m/s (≈ 340 at 288K).
    """
    if T <= 0:
        return 0.0
    gamma = air_gamma(T, sigma)
    M = mean_molar_mass_kg_mol(sigma)
    if M <= 0:
        return 0.0
    return math.sqrt(gamma * R_GAS * T / M)


# ═══════════════════════════════════════════════════════════════════
# BAROMETRIC FORMULA
# ═══════════════════════════════════════════════════════════════════

def scale_height(T=288.15, sigma=SIGMA_HERE):
    """Atmospheric scale height H (metres).

    H = R T / (M g)

    FIRST_PRINCIPLES: the e-folding height for pressure in an
    isothermal atmosphere. Pressure drops by 1/e every H metres.

    Args:
        T: temperature (K)
        sigma: sigma-field value

    Returns:
        Scale height in metres (≈ 8500 m for Earth).
    """
    M = mean_molar_mass_kg_mol(sigma)
    if M <= 0:
        return float('inf')
    return R_GAS * T / (M * _G_STANDARD)


def pressure_at_altitude(z_m, T=288.15, P0=_P_STANDARD, sigma=SIGMA_HERE):
    """Atmospheric pressure at altitude z (Pa).

    P(z) = P0 × exp(-z/H)

    FIRST_PRINCIPLES: hydrostatic equilibrium + ideal gas law
    in an isothermal atmosphere.

    For the real atmosphere, use with T = T(z) from lapse rate,
    or integrate numerically. This isothermal form is the standard
    barometric formula — exact for an isothermal column.

    Args:
        z_m: altitude in metres
        T: temperature (K) — assumed constant (isothermal approximation)
        P0: surface pressure (Pa)
        sigma: sigma-field value

    Returns:
        Pressure in Pa.
    """
    H = scale_height(T, sigma)
    if H <= 0:
        return 0.0
    return P0 * math.exp(-z_m / H)


def density_at_altitude(z_m, T=288.15, P0=_P_STANDARD, sigma=SIGMA_HERE):
    """Air density at altitude z (kg/m³).

    Same exponential decay as pressure in isothermal atmosphere.

    Args:
        z_m: altitude (m)
        T: temperature (K)
        P0: surface pressure (Pa)
        sigma: sigma-field value

    Returns:
        Density in kg/m³.
    """
    P = pressure_at_altitude(z_m, T, P0, sigma)
    return air_density(T, P, sigma)


def altitude_for_pressure(P, T=288.15, P0=_P_STANDARD, sigma=SIGMA_HERE):
    """Altitude at which pressure equals P (metres).

    z = -H × ln(P/P0) — inverse barometric formula.

    Args:
        P: target pressure (Pa)
        T: temperature (K)
        P0: surface pressure (Pa)
        sigma: sigma-field value

    Returns:
        Altitude in metres.
    """
    if P <= 0 or P0 <= 0:
        return float('inf')
    H = scale_height(T, sigma)
    return -H * math.log(P / P0)


# ═══════════════════════════════════════════════════════════════════
# LAPSE RATE
# ═══════════════════════════════════════════════════════════════════

def dry_adiabatic_lapse_rate(T=288.15, sigma=SIGMA_HERE):
    """Dry adiabatic lapse rate Gamma_d (K/m).

    Gamma_d = g / cp

    FIRST_PRINCIPLES: when a parcel of dry air rises adiabatically,
    it expands and cools at this rate. No heat exchange with
    surroundings, no condensation.

    Args:
        T: temperature (K) — affects cp slightly
        sigma: sigma-field value

    Returns:
        Lapse rate in K/m (≈ 0.00976 K/m = 9.76 K/km for Earth).
    """
    cp = air_cp_mass(T, sigma)
    if cp <= 0:
        return 0.0
    return _G_STANDARD / cp


def temperature_at_altitude(z_m, T_surface=288.15, sigma=SIGMA_HERE):
    """Temperature at altitude using dry adiabatic lapse rate (K).

    T(z) = T_surface - Gamma_d × z

    Valid in the troposphere for unsaturated air.
    The tropopause (where T stops decreasing) is at ~11 km.

    Args:
        z_m: altitude (m)
        T_surface: surface temperature (K)
        sigma: sigma-field value

    Returns:
        Temperature in K. Capped at 0 K minimum.
    """
    gamma_d = dry_adiabatic_lapse_rate(T_surface, sigma)
    T = T_surface - gamma_d * z_m
    return max(T, 0.0)


# ═══════════════════════════════════════════════════════════════════
# WATER VAPOR — SATURATION AND HUMIDITY
# ═══════════════════════════════════════════════════════════════════

# Water phase data (MEASURED + DERIVED)
_T_BOIL_WATER = 373.15   # K, at 1 atm (MEASURED, exact by definition)
_T_MELT_WATER = 273.15   # K, at 1 atm (MEASURED, exact by definition)
_P_SAT_100C = 101325.0   # Pa, at boiling point (by definition)
_M_WATER = 18.015e-3     # kg/mol (MEASURED)


def _water_L_vap_J_mol():
    """Latent heat of vaporization of water (J/mol).

    DERIVED from H-bond model in liquid_water.py:
      L_vap = n_HB × E_HB × N_A / 2

    Falls back to MEASURED 40,700 J/mol if import fails.
    """
    try:
        from .liquid_water import water_enthalpy_of_vaporization
        return water_enthalpy_of_vaporization()
    except (ImportError, AttributeError):
        return 40700.0  # MEASURED fallback


def saturation_vapor_pressure(T):
    """Saturation vapor pressure of water (Pa).

    FIRST_PRINCIPLES: Clausius-Clapeyron equation (liquid-vapor).

    P_sat(T) = P_ref × exp(-L_vap/R × (1/T - 1/T_ref))

    Where:
      P_ref = 101325 Pa at T_ref = 373.15 K (boiling point)
      L_vap = DERIVED from H-bond model (liquid_water.py)

    Accuracy: ±5% over 250-380 K range. The Clausius-Clapeyron
    approximation assumes constant L_vap; in reality L_vap decreases
    slightly with T. For better accuracy, use the Antoine equation
    (MEASURED fit).

    Args:
        T: temperature in Kelvin

    Returns:
        Saturation vapor pressure in Pa.
    """
    if T <= 0:
        return 0.0

    L_vap = _water_L_vap_J_mol()

    exponent = -L_vap / R_GAS * (1.0 / T - 1.0 / _T_BOIL_WATER)
    exponent = min(exponent, 700)  # prevent overflow
    exponent = max(exponent, -700)

    return _P_SAT_100C * math.exp(exponent)


def dew_point(T, relative_humidity):
    """Dew point temperature (K).

    The temperature at which air becomes saturated (RH = 100%).

    FIRST_PRINCIPLES: invert Clausius-Clapeyron.
      P_actual = RH × P_sat(T)
      T_dew = T such that P_sat(T_dew) = P_actual

    From CC inversion:
      1/T_dew = 1/T_ref - (R/L_vap) × ln(P_actual/P_ref)

    Args:
        T: air temperature (K)
        relative_humidity: RH as fraction (0-1)

    Returns:
        Dew point in Kelvin.
    """
    if relative_humidity <= 0 or T <= 0:
        return 0.0

    L_vap = _water_L_vap_J_mol()
    P_actual = relative_humidity * saturation_vapor_pressure(T)

    if P_actual <= 0:
        return 0.0

    # Invert CC: 1/T_dew = 1/T_boil - (R/L) ln(P_actual/P_boil)
    log_ratio = math.log(P_actual / _P_SAT_100C)
    inv_T = 1.0 / _T_BOIL_WATER - R_GAS / L_vap * log_ratio

    if inv_T <= 0:
        return float('inf')

    return 1.0 / inv_T


def mixing_ratio(T, P=_P_STANDARD, relative_humidity=1.0):
    """Water vapor mixing ratio (kg water / kg dry air).

    w = (M_water / M_air) × (P_vapor / (P - P_vapor))

    FIRST_PRINCIPLES: from Dalton's law of partial pressures.

    Args:
        T: temperature (K)
        P: total pressure (Pa)
        relative_humidity: RH as fraction (0-1)

    Returns:
        Mixing ratio (dimensionless, typically ~0.001-0.02).
    """
    P_vapor = relative_humidity * saturation_vapor_pressure(T)
    P_dry = P - P_vapor
    if P_dry <= 0:
        return float('inf')  # all vapor, no dry air

    epsilon = _M_WATER / mean_molar_mass_kg_mol()  # ≈ 0.622
    return epsilon * P_vapor / P_dry


def absolute_humidity(T, P=_P_STANDARD, relative_humidity=1.0):
    """Absolute humidity (kg/m³ of water vapor).

    rho_v = P_vapor × M_water / (R × T)

    FIRST_PRINCIPLES: ideal gas law for the vapor component.

    Args:
        T: temperature (K)
        P: total pressure (Pa)
        relative_humidity: RH as fraction

    Returns:
        Water vapor density in kg/m³.
    """
    if T <= 0:
        return 0.0
    P_vapor = relative_humidity * saturation_vapor_pressure(T)
    return P_vapor * _M_WATER / (R_GAS * T)


def specific_humidity(T, P=_P_STANDARD, relative_humidity=1.0):
    """Specific humidity (kg water / kg moist air).

    q = w / (1 + w) ≈ w for small w.

    Args:
        T, P, relative_humidity: as in mixing_ratio

    Returns:
        Specific humidity (dimensionless).
    """
    w = mixing_ratio(T, P, relative_humidity)
    if w == float('inf'):
        return 1.0
    return w / (1.0 + w)


# ═══════════════════════════════════════════════════════════════════
# MOIST ADIABATIC LAPSE RATE
# ═══════════════════════════════════════════════════════════════════

def moist_adiabatic_lapse_rate(T, P=_P_STANDARD, sigma=SIGMA_HERE):
    """Moist (saturated) adiabatic lapse rate (K/m).

    Gamma_m = Gamma_d × [1 + L_v w_s / (R_d T)]
                       / [1 + L_v² w_s / (cp R_v T²)]

    FIRST_PRINCIPLES: adiabatic ascent WITH condensation.
    Released latent heat partially offsets the adiabatic cooling.

    Gamma_m < Gamma_d always (condensation warms the parcel).
    Typical: ~5-6 K/km at sea level (vs 9.8 K/km dry).

    Args:
        T: temperature (K)
        P: pressure (Pa)
        sigma: sigma-field value

    Returns:
        Moist lapse rate in K/m.
    """
    gamma_d = dry_adiabatic_lapse_rate(T, sigma)
    L_vap = _water_L_vap_J_mol()
    cp = air_cp_mass(T, sigma)
    M_air = mean_molar_mass_kg_mol(sigma)

    # Per-mass latent heat
    L_v = L_vap / _M_WATER  # J/kg

    # Saturation mixing ratio
    w_s = mixing_ratio(T, P, 1.0)
    if w_s == float('inf') or w_s <= 0:
        return gamma_d

    # Gas constants
    R_d = R_GAS / M_air      # dry air specific gas constant (J/(kg·K))
    R_v = R_GAS / _M_WATER   # water vapor gas constant (J/(kg·K))

    numerator = 1.0 + L_v * w_s / (R_d * T)
    denominator = 1.0 + L_v * L_v * w_s / (cp * R_v * T * T)

    if denominator <= 0:
        return gamma_d

    return gamma_d * numerator / denominator


# ═══════════════════════════════════════════════════════════════════
# COLUMN PROPERTIES
# ═══════════════════════════════════════════════════════════════════

def column_number_density(species_fraction, T=288.15, P0=_P_STANDARD,
                           sigma=SIGMA_HERE):
    """Column number density of a gas species (molecules/m²).

    N_col = n_surface × H × x_i

    Where n_surface = P/(k_B T) and H is scale height.
    This is the total number of molecules of species i above
    one square metre of surface — needed for optical depth.

    FIRST_PRINCIPLES: integrate n(z)dz over isothermal atmosphere.

    Args:
        species_fraction: volume/mole fraction (e.g., 420e-6 for CO₂)
        T: temperature (K)
        P0: surface pressure (Pa)
        sigma: sigma-field value

    Returns:
        Column number density in molecules/m².
    """
    if T <= 0:
        return 0.0
    n_surface = P0 / (K_B * T)  # total molecules/m³ at surface
    H = scale_height(T, sigma)
    return n_surface * H * species_fraction


# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════

def atmosphere_report(T_surface=288.15, P_surface=_P_STANDARD,
                       RH=0.50, sigma=SIGMA_HERE):
    """Complete atmospheric state report."""
    return {
        'T_surface_K': T_surface,
        'P_surface_Pa': P_surface,
        'relative_humidity': RH,
        'M_air_kg_mol': mean_molar_mass_kg_mol(sigma),
        'air_density_kg_m3': air_density(T_surface, P_surface, sigma),
        'cp_air_J_kg_K': air_cp_mass(T_surface, sigma),
        'gamma': air_gamma(T_surface, sigma),
        'speed_of_sound_m_s': speed_of_sound(T_surface, sigma),
        'scale_height_m': scale_height(T_surface, sigma),
        'dry_lapse_rate_K_km': dry_adiabatic_lapse_rate(T_surface, sigma) * 1000,
        'moist_lapse_rate_K_km': moist_adiabatic_lapse_rate(
            T_surface, P_surface, sigma) * 1000,
        'P_sat_Pa': saturation_vapor_pressure(T_surface),
        'dew_point_K': dew_point(T_surface, RH),
        'dew_point_C': dew_point(T_surface, RH) - 273.15,
        'mixing_ratio_g_kg': mixing_ratio(T_surface, P_surface, RH) * 1000,
        'absolute_humidity_g_m3': absolute_humidity(
            T_surface, P_surface, RH) * 1000,
        'CO2_column_molecules_m2': column_number_density(420e-6, T_surface,
                                                          P_surface, sigma),
        'tropopause_T_K': temperature_at_altitude(11000, T_surface, sigma),
    }


def full_report(sigma=SIGMA_HERE):
    """Standard atmosphere report. Rule 9: multiple conditions."""
    conditions = {
        'standard_ISA': {'T_surface': 288.15, 'P_surface': _P_STANDARD, 'RH': 0.50},
        'tropical': {'T_surface': 303.15, 'P_surface': _P_STANDARD, 'RH': 0.80},
        'arctic': {'T_surface': 253.15, 'P_surface': _P_STANDARD, 'RH': 0.60},
        'desert': {'T_surface': 323.15, 'P_surface': _P_STANDARD, 'RH': 0.10},
    }
    return {name: atmosphere_report(**params, sigma=sigma)
            for name, params in conditions.items()}
