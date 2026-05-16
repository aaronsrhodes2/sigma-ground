"""Thermodynamics tools: gas laws, blackbody radiation, heat engines.

Pure formulas using sigma_ground constants. The ideal gas / blackbody
/ Wien / Stefan-Boltzmann family covers ~80% of intro thermodynamics
questions.

Phase-transition data (melting points, latent heats of common materials)
lives in tools/materials.py.
"""

from __future__ import annotations

import math

from sigma_ground.mcp.provenance import ToolResult


# Ideal gas constant R = N_A * k_B = 8.314462618 J/(mol K), CODATA 2018 exact.
_R_GAS = 8.314462618


def ideal_gas_pressure(n_moles: float, temperature_k: float,
                        volume_m3: float) -> ToolResult:
    """P = n R T / V. Solves the ideal gas law for pressure."""
    if n_moles <= 0 or temperature_k <= 0 or volume_m3 <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"n_moles": n_moles,
                                   "temperature_k": temperature_k,
                                   "volume_m3": volume_m3})
    P = n_moles * _R_GAS * temperature_k / volume_m3
    return ToolResult(
        value=P,
        units="Pa",
        source="sigma-ground via CODATA 2018 R = 8.314462618 J/(mol K)",
        formula="P = n R T / V",
        inputs={"n_moles": n_moles, "temperature_k": temperature_k,
                "volume_m3": volume_m3},
        notes=("Ideal gas law. Valid when intermolecular forces are "
                "negligible (low pressure, high temperature). For real "
                "gases at high pressure / low T, use Van der Waals."),
    )


def ideal_gas_volume(n_moles: float, temperature_k: float,
                      pressure_pa: float) -> ToolResult:
    """V = n R T / P. Solves the ideal gas law for volume."""
    if n_moles <= 0 or temperature_k <= 0 or pressure_pa <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"n_moles": n_moles,
                                   "temperature_k": temperature_k,
                                   "pressure_pa": pressure_pa})
    V = n_moles * _R_GAS * temperature_k / pressure_pa
    return ToolResult(
        value=V,
        units="m^3",
        source="sigma-ground via CODATA 2018 R",
        formula="V = n R T / P",
        inputs={"n_moles": n_moles, "temperature_k": temperature_k,
                "pressure_pa": pressure_pa},
        notes=("Molar volume at STP (T=273.15 K, P=100 kPa) is "
                "22.711 L; at NTP (T=293.15 K, P=101.325 kPa) is 24.055 L."),
    )


def blackbody_peak_wavelength(temperature_k: float) -> ToolResult:
    """Wien's displacement law: lambda_max = b / T.

    Wien constant b = 2.897771955e-3 m K (CODATA 2018 exact).
    """
    if temperature_k <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes="temperature_k must be positive",
                           inputs={"temperature_k": temperature_k})
    b = 2.897771955e-3
    lam = b / temperature_k
    return ToolResult(
        value=lam,
        units="m",
        source="sigma-ground via Wien's displacement law (CODATA 2018 b)",
        formula="lambda_max = b / T, b = 2.897771955e-3 m K",
        inputs={"temperature_k": temperature_k},
        notes=(f"Wavelength of peak emission for blackbody at T={temperature_k} K. "
                f"Sun's surface T=5778 K -> 501 nm (visible green/yellow). "
                f"Human body T=310 K -> 9.35 um (far IR)."),
    )


def blackbody_total_power(temperature_k: float, area_m2: float = 1.0,
                            emissivity: float = 1.0) -> ToolResult:
    """Stefan-Boltzmann: P = epsilon sigma A T^4.

    Stefan-Boltzmann constant sigma = 5.670374419e-8 W/(m^2 K^4) (CODATA 2018).
    """
    if temperature_k < 0 or area_m2 < 0 or not 0 <= emissivity <= 1:
        return ToolResult(value=None, source="invalid input",
                           inputs={"temperature_k": temperature_k,
                                   "area_m2": area_m2,
                                   "emissivity": emissivity})
    sigma_sb = 5.670374419e-8
    P = emissivity * sigma_sb * area_m2 * temperature_k ** 4
    return ToolResult(
        value=P,
        units="W",
        source="sigma-ground via Stefan-Boltzmann (CODATA 2018 sigma_SB)",
        formula="P = epsilon sigma_SB A T^4",
        inputs={"temperature_k": temperature_k, "area_m2": area_m2,
                "emissivity": emissivity},
        notes=("Total radiated power. Real surfaces have emissivity<1; "
                "ideal blackbody emits at emissivity=1. Sun (T=5778 K, "
                "A=4 pi R_sun^2): 3.85e26 W = solar luminosity."),
    )


def carnot_efficiency(t_hot_k: float, t_cold_k: float) -> ToolResult:
    """Maximum-efficiency heat engine bound: eta = 1 - T_cold / T_hot."""
    if t_hot_k <= 0 or t_cold_k < 0 or t_cold_k >= t_hot_k:
        return ToolResult(value=None, source="invalid input",
                           notes="0 <= t_cold_k < t_hot_k required",
                           inputs={"t_hot_k": t_hot_k, "t_cold_k": t_cold_k})
    eta = 1.0 - t_cold_k / t_hot_k
    return ToolResult(
        value=eta,
        units="dimensionless",
        source="sigma-ground (Carnot 1824, 2nd law of thermodynamics)",
        formula="eta = 1 - T_cold / T_hot",
        inputs={"t_hot_k": t_hot_k, "t_cold_k": t_cold_k},
        notes=("Upper bound on efficiency of ANY heat engine operating "
                "between these temperatures. Real engines achieve less. "
                "Coal plant (T_hot=800 K, T_cold=300 K) Carnot = 0.625; "
                "real efficiency ~0.35-0.40."),
    )


def thermal_energy_per_molecule(temperature_k: float,
                                  degrees_of_freedom: int = 3) -> ToolResult:
    """Average thermal energy per molecule: E = (f/2) k_B T.

    For a monatomic ideal gas f=3 (translation only).
    Diatomic at room T: f=5 (3 translation + 2 rotation).
    Diatomic at high T: f=7 (add 2 vibration).
    """
    if temperature_k < 0 or degrees_of_freedom < 1:
        return ToolResult(value=None, source="invalid input",
                           inputs={"temperature_k": temperature_k,
                                   "degrees_of_freedom": degrees_of_freedom})
    from sigma_ground.field.constants import K_B
    E = 0.5 * degrees_of_freedom * K_B * temperature_k
    return ToolResult(
        value=E,
        units="J",
        source="sigma-ground via equipartition theorem (CODATA k_B)",
        formula="E = (f/2) k_B T",
        inputs={"temperature_k": temperature_k,
                "degrees_of_freedom": degrees_of_freedom},
        notes=(f"At room T=300 K and f=3: E = {0.5 * 3 * 1.380649e-23 * 300:.3e} J "
                f"= ~26 meV per particle. This sets typical kinetic energies "
                f"in gases."),
    )


def speed_of_sound_in_ideal_gas(temperature_k: float, gamma: float = 1.4,
                                  molar_mass_kg_per_mol: float = 0.029) -> ToolResult:
    """Speed of sound in an ideal gas: v = sqrt(gamma R T / M).

    Defaults gamma=1.4 (diatomic) and M=0.029 kg/mol (dry air at sea level).

    Parameters
    ----------
    temperature_k : float
        Absolute temperature in kelvin.
    gamma : float
        Heat capacity ratio C_p/C_v. Default 1.4 (diatomic, air).
        Monatomic (He, Ar): 5/3 = 1.667.
    molar_mass_kg_per_mol : float
        Molar mass in kg/mol. Default 0.029 (dry air).
    """
    if temperature_k <= 0 or gamma <= 1 or molar_mass_kg_per_mol <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"temperature_k": temperature_k,
                                   "gamma": gamma,
                                   "molar_mass_kg_per_mol":
                                   molar_mass_kg_per_mol})
    v = math.sqrt(gamma * _R_GAS * temperature_k / molar_mass_kg_per_mol)
    return ToolResult(
        value=v,
        units="m/s",
        source="sigma-ground (ideal-gas adiabatic sound speed)",
        formula="v = sqrt(gamma R T / M)",
        inputs={"temperature_k": temperature_k, "gamma": gamma,
                "molar_mass_kg_per_mol": molar_mass_kg_per_mol},
        notes=("Air at T=293 K (20 C): ~343 m/s. Helium at same T: ~1007 m/s "
                "(lighter, smaller M). Sound speed is independent of pressure "
                "for an ideal gas."),
    )


def maxwell_boltzmann_most_probable_speed(temperature_k: float,
                                            molecular_mass_kg: float) -> ToolResult:
    """v_p = sqrt(2 k_B T / m). Peak of the Maxwell-Boltzmann distribution."""
    if temperature_k <= 0 or molecular_mass_kg <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"temperature_k": temperature_k,
                                   "molecular_mass_kg": molecular_mass_kg})
    from sigma_ground.field.constants import K_B
    v_p = math.sqrt(2.0 * K_B * temperature_k / molecular_mass_kg)
    return ToolResult(
        value=v_p,
        units="m/s",
        source="sigma-ground (Maxwell-Boltzmann distribution)",
        formula="v_p = sqrt(2 k_B T / m)",
        inputs={"temperature_k": temperature_k,
                "molecular_mass_kg": molecular_mass_kg},
        notes=("Most probable speed. RMS speed = sqrt(3 k_B T / m), and "
                "mean speed = sqrt(8 k_B T / (pi m)). All scale as sqrt(T/m)."),
    )


def temperature_celsius_to_kelvin(t_celsius: float) -> ToolResult:
    """T_K = T_C + 273.15. Defined exactly by SI."""
    return ToolResult(
        value=t_celsius + 273.15,
        units="K",
        source="sigma-ground via SI temperature scale definition",
        formula="T_K = T_C + 273.15",
        inputs={"t_celsius": t_celsius},
    )


def temperature_kelvin_to_celsius(t_kelvin: float) -> ToolResult:
    """T_C = T_K - 273.15."""
    return ToolResult(
        value=t_kelvin - 273.15,
        units="C",
        source="sigma-ground via SI temperature scale definition",
        formula="T_C = T_K - 273.15",
        inputs={"t_kelvin": t_kelvin},
    )
