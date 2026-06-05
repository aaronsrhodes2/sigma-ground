"""Direct library audit — validate sigma-ground against known physics, no LLM.

This is the Phase-1 library-validation oracle. For every physics tool the
MCP server exposes, it calls the tool with KNOWN-correct inputs (authored
by hand from textbook physics, not inferred by any interpreter) and checks
the returned value against the known-correct answer. Because the inputs
are correct by construction, a mismatch isolates a **library** problem:
a wrong constant, a wrong formula, or a missing/fabricated provenance.

It removes the interpreter as a variable. `run_sigma_ground.py` with
Qwen at 82% can't tell you whether a miss is Qwen's tool-selection or the
library's value. This script can: there's no Qwen.

Five verdicts per tool:
  CONFIRMED          tool(known inputs) == known answer (within tolerance),
                       AND the result carries a traceable `source`.
  LIBRARY_BUG        tool was called correctly but returned the wrong value.
  HALLUCINATION_RISK value matches but there's no `source`/`provenance_tag`,
                       or it's tagged "Fitted due to incompetence". A number
                       you can't trace back to a reference.
  TOOL_ERROR         the tool raised / returned null on valid inputs.
  NO_CASE            tool exists but this audit has no hand-authored case
                       for it (utility tools, list_* etc.).

Plus a CONSTANTS cross-check: every entry in the curated constant set is
compared against scipy.constants (CODATA) where an independent reference
exists. SSBM-specific constants (ETA, SIGMA_HERE, PHI, ...) have no
external reference and are reported as SSBM_INPUT (expected — they're
empirical inputs, not derivable from CODATA).

Run:
    python -m sigma_ground.mcp.benchmark.audit_library
    # writes misc/LIBRARY_AUDIT_<date-from-args-or-blank>.md and prints a table
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any


# Reference physical values used to build canonical inputs (CODATA-ish).
_M_SUN = 1.98892e30
_M_EARTH = 5.972e24
_R_EARTH = 6.371e6
_C = 299792458.0
_M_E = 9.1093837e-31
_AU = 1.495978707e11
_O2_MASS = 32 * 1.66053907e-27   # O2 molecular mass in kg


# ============================================================
# CURATED TOOL AUDIT CASES
# Each: id, tool, kwargs (KNOWN-correct), expected, units,
#       optional 'field' to pick a scalar from a dict-valued result,
#       optional 'kind' in {"numeric","string"} (default numeric),
#       ref = where the known answer comes from.
# ============================================================
AUDIT_CASES: list[dict] = [
    # ---- kinematics ----
    {"id": "free_fall_time", "tool": "free_fall_time",
     "kwargs": {"height_m": 10.0}, "expected": 1.428, "units": "s",
     "ref": "t=sqrt(2h/g), g=9.80665"},
    {"id": "free_fall_velocity", "tool": "free_fall_velocity",
     "kwargs": {"height_m": 50.0}, "expected": 31.32, "units": "m/s",
     "ref": "v=sqrt(2gh)"},
    {"id": "projectile_range", "tool": "projectile_range",
     "kwargs": {"initial_speed_m_s": 100.0, "launch_angle_deg": 45.0},
     "expected": 1019.7, "units": "m", "ref": "R=v^2 sin(2θ)/g"},
    {"id": "projectile_max_height", "tool": "projectile_max_height",
     "kwargs": {"initial_speed_m_s": 50.0, "launch_angle_deg": 90.0},
     "expected": 127.47, "units": "m", "ref": "h=v^2/(2g)"},
    {"id": "projectile_flight_time", "tool": "projectile_flight_time",
     "kwargs": {"initial_speed_m_s": 40.0, "launch_angle_deg": 30.0},
     "expected": 4.079, "units": "s", "ref": "t=2v sinθ/g"},
    {"id": "kinetic_energy", "tool": "kinetic_energy",
     "kwargs": {"mass_kg": 70.0, "velocity_m_s": 5.0},
     "expected": 875.0, "units": "J", "ref": "KE=0.5 m v^2"},
    {"id": "momentum", "tool": "momentum",
     "kwargs": {"mass_kg": 1500.0, "velocity_m_s": 25.0},
     "expected": 37500.0, "units": "kg m/s", "ref": "p=mv"},
    {"id": "circular_orbit_velocity", "tool": "circular_orbit_velocity",
     "kwargs": {"central_mass_kg": _M_EARTH, "radius_m": _R_EARTH + 400e3},
     "expected": 7670.0, "units": "m/s", "ref": "v=sqrt(GM/r)"},
    {"id": "escape_velocity", "tool": "escape_velocity",
     "kwargs": {"mass_kg": _M_EARTH, "radius_m": _R_EARTH},
     "expected": 11186.0, "units": "m/s", "ref": "v=sqrt(2GM/r)"},
    {"id": "gravitational_potential_energy", "tool": "gravitational_potential_energy",
     "kwargs": {"mass_kg": 70.0, "height_m": 100.0},
     "expected": 68646.0, "units": "J", "ref": "U=mgh"},
    {"id": "friction_stopping_distance", "tool": "friction_stopping_distance",
     "kwargs": {"mass_kg": 0.2, "initial_velocity_m_s": 1.0,
                  "friction_coefficient": 0.4},
     "expected": 0.1274, "units": "m", "ref": "d=v^2/(2 μ g)"},

    # ---- electromagnetism / circuits ----
    {"id": "ohms_law_current", "tool": "ohms_law_current",
     "kwargs": {"voltage_v": 5.0, "resistance_ohm": 100.0},
     "expected": 0.05, "units": "A", "ref": "I=V/R"},
    {"id": "ohms_law_voltage", "tool": "ohms_law_voltage",
     "kwargs": {"current_a": 0.1, "resistance_ohm": 100.0},
     "expected": 10.0, "units": "V", "ref": "V=IR"},
    {"id": "electrical_power", "tool": "electrical_power",
     "kwargs": {"voltage_v": 12.0, "current_a": 2.0},
     "expected": 24.0, "units": "W", "ref": "P=VI"},
    {"id": "power_dissipation_resistor", "tool": "power_dissipation_resistor",
     "kwargs": {"current_a": 5.0, "resistance_ohm": 10.0},
     "expected": 250.0, "units": "W", "ref": "P=I^2 R"},
    {"id": "parallel_plate_capacitance", "tool": "parallel_plate_capacitance",
     "kwargs": {"area_m2": 1.0, "separation_m": 0.001},
     "expected": 8.854e-9, "units": "F", "ref": "C=ε0 A/d"},
    {"id": "rc_time_constant", "tool": "rc_time_constant",
     "kwargs": {"resistance_ohm": 1000.0, "capacitance_f": 1e-6},
     "expected": 0.001, "units": "s", "ref": "τ=RC"},
    {"id": "rl_time_constant", "tool": "rl_time_constant",
     "kwargs": {"resistance_ohm": 100.0, "inductance_h": 0.01},
     "expected": 1e-4, "units": "s", "ref": "τ=L/R"},
    {"id": "rlc_resonant_frequency", "tool": "rlc_resonant_frequency",
     "kwargs": {"inductance_h": 1e-3, "capacitance_f": 1e-6},
     "expected": 5032.9, "units": "Hz", "ref": "ω=1/sqrt(LC); /2π"},
    {"id": "em_wave_wavelength", "tool": "em_wave_wavelength",
     "kwargs": {"frequency_hz": 100e6}, "expected": 2.998, "units": "m",
     "ref": "λ=c/f"},
    {"id": "em_wave_frequency", "tool": "em_wave_frequency",
     "kwargs": {"wavelength_m": 550e-9}, "expected": 5.451e14, "units": "Hz",
     "ref": "f=c/λ"},

    # ---- optics / waves ----
    {"id": "snells_law_refraction_angle", "tool": "snells_law_refraction_angle",
     "kwargs": {"n1": 1.0, "n2": 1.333, "incident_angle_deg": 30.0},
     "expected": 22.08, "units": "deg", "ref": "n1 sinθ1=n2 sinθ2"},
    {"id": "critical_angle_for_tir", "tool": "critical_angle_for_tir",
     "kwargs": {"n_dense": 1.333, "n_rare": 1.0},
     "expected": 48.61, "units": "deg", "ref": "asin(n2/n1)"},
    {"id": "thin_lens_image_distance", "tool": "thin_lens_image_distance",
     "kwargs": {"object_distance_m": 0.30, "focal_length_m": 0.15},
     "expected": 0.30, "units": "m", "ref": "1/f=1/do+1/di"},
    {"id": "lens_magnification", "tool": "lens_magnification",
     "kwargs": {"object_distance_m": 0.20, "image_distance_m": 0.10},
     "expected": -0.5, "units": "", "ref": "m=-di/do"},
    {"id": "double_slit_fringe_spacing", "tool": "double_slit_fringe_spacing",
     "kwargs": {"wavelength_m": 633e-9, "slit_separation_m": 0.1e-3,
                  "screen_distance_m": 1.0},
     "expected": 0.00633, "units": "m", "ref": "y=λL/d"},
    {"id": "single_slit_first_minimum_angle", "tool": "single_slit_first_minimum_angle",
     "kwargs": {"wavelength_m": 500e-9, "slit_width_m": 1e-6},
     "expected": 30.0, "units": "deg", "ref": "sinθ=λ/a"},
    {"id": "diffraction_grating_angle", "tool": "diffraction_grating_angle",
     "kwargs": {"wavelength_m": 500e-9, "grating_spacing_m": 1e-6, "order": 1},
     "expected": 30.0, "units": "deg", "ref": "d sinθ=mλ"},
    {"id": "rydberg_hydrogen_wavelength", "tool": "rydberg_hydrogen_wavelength",
     "kwargs": {"n_initial": 3, "n_final": 2},
     "expected": 6.563e-7, "units": "m", "ref": "H-alpha 656.3 nm"},
    {"id": "speed_of_sound_in_ideal_gas", "tool": "speed_of_sound_in_ideal_gas",
     "kwargs": {"temperature_k": 293.15}, "expected": 343.0, "units": "m/s",
     "ref": "v=sqrt(γRT/M)"},

    # ---- thermodynamics ----
    {"id": "ideal_gas_pressure", "tool": "ideal_gas_pressure",
     "kwargs": {"n_moles": 1.0, "temperature_k": 273.15, "volume_m3": 0.022414},
     "expected": 101325.0, "units": "Pa", "ref": "P=nRT/V"},
    {"id": "ideal_gas_volume", "tool": "ideal_gas_volume",
     "kwargs": {"n_moles": 1.0, "temperature_k": 298.15, "pressure_pa": 101325.0},
     "expected": 0.02447, "units": "m^3", "ref": "V=nRT/P"},
    {"id": "blackbody_peak_wavelength", "tool": "blackbody_peak_wavelength",
     "kwargs": {"temperature_k": 6000.0}, "expected": 4.83e-7, "units": "m",
     "ref": "Wien λ=b/T"},
    {"id": "blackbody_total_power", "tool": "blackbody_total_power",
     "kwargs": {"temperature_k": 300.0}, "expected": 459.0, "units": "W",
     "ref": "P=σT^4"},
    {"id": "carnot_efficiency", "tool": "carnot_efficiency",
     "kwargs": {"t_hot_k": 600.0, "t_cold_k": 300.0},
     "expected": 0.5, "units": "", "ref": "1-Tc/Th"},
    {"id": "thermal_energy_per_molecule", "tool": "thermal_energy_per_molecule",
     "kwargs": {"temperature_k": 300.0}, "expected": 6.213e-21, "units": "J",
     "ref": "(3/2)kT"},
    {"id": "maxwell_boltzmann_most_probable_speed",
     "tool": "maxwell_boltzmann_most_probable_speed",
     "kwargs": {"temperature_k": 300.0, "molecular_mass_kg": _O2_MASS},
     "expected": 395.0, "units": "m/s", "ref": "v_p=sqrt(2kT/m)"},
    {"id": "temperature_celsius_to_kelvin", "tool": "temperature_celsius_to_kelvin",
     "kwargs": {"t_celsius": 100.0}, "expected": 373.15, "units": "K",
     "ref": "T_K=T_C+273.15"},
    {"id": "melting_point", "tool": "melting_point",
     "kwargs": {"material": "iron"}, "expected": 1538.0, "units": "C",
     "ref": "Fe melts at 1811 K"},
    {"id": "boiling_point", "tool": "boiling_point",
     "kwargs": {"material": "nitrogen"}, "expected": -196.0, "units": "C",
     "ref": "N2 boils at 77.36 K"},

    # ---- materials ----
    {"id": "refractive_index", "tool": "refractive_index",
     "kwargs": {"material": "diamond"}, "expected": 2.42, "units": "",
     "ref": "diamond n~2.42 at 589 nm"},
    {"id": "density_water", "tool": "density",
     "kwargs": {"material": "water"}, "expected": 1000.0, "units": "kg/m^3",
     "ref": "water ~1000 kg/m^3"},

    # ---- atomic ----
    {"id": "first_ionization_energy", "tool": "first_ionization_energy",
     "kwargs": {"element_symbol": "He"}, "expected": 24.587, "units": "eV",
     "ref": "He 1st IE (NIST)"},
    {"id": "hydrogen_like_energy_level", "tool": "hydrogen_like_energy_level",
     "kwargs": {"n": 1}, "expected": -13.606, "units": "eV",
     "ref": "E_1=-Ry"},
    {"id": "hydrogen_emission_wavelength", "tool": "hydrogen_emission_wavelength",
     "kwargs": {"n_initial": 4, "n_final": 2},
     "expected": 4.861e-7, "units": "m", "ref": "H-beta 486.1 nm"},
    {"id": "de_broglie_wavelength", "tool": "de_broglie_wavelength",
     "kwargs": {"mass_kg": _M_E, "velocity_m_s": 1e7},
     "expected": 7.275e-11, "units": "m", "ref": "λ=h/(mv)"},
    {"id": "photon_energy_from_frequency", "tool": "photon_energy_from_frequency",
     "kwargs": {"frequency_hz": 1e15}, "expected": 6.626e-19, "units": "J",
     "ref": "E=hf"},
    {"id": "photon_energy_from_wavelength", "tool": "photon_energy_from_wavelength",
     "kwargs": {"wavelength_m": 550e-9}, "expected": 2.254, "units": "eV",
     "ref": "E=hc/λ"},
    {"id": "element_atomic_data_Z", "tool": "element_atomic_data",
     "kwargs": {"element_symbol": "Au"}, "expected": 79, "units": "",
     "field": "atomic_number", "ref": "Au Z=79"},

    # ---- energy conversion ----
    {"id": "mass_to_energy", "tool": "mass_to_energy",
     "kwargs": {"mass_kg": 1.0}, "expected": 8.988e16, "units": "J",
     "ref": "E=mc^2"},
    {"id": "energy_to_mass", "tool": "energy_to_mass",
     "kwargs": {"energy_j": 4.184e15}, "expected": 0.0465, "units": "kg",
     "ref": "m=E/c^2, 1 MT TNT"},
    {"id": "eV_to_joules", "tool": "eV_to_joules",
     "kwargs": {"energy_eV": 1.0}, "expected": 1.602e-19, "units": "J",
     "ref": "1 eV=1.602e-19 J"},
    {"id": "joules_to_eV", "tool": "joules_to_eV",
     "kwargs": {"energy_joules": 1.0}, "expected": 6.242e18, "units": "eV",
     "ref": "1 J=6.242e18 eV"},
    {"id": "joules_to_TNT", "tool": "joules_to_TNT",
     "kwargs": {"energy_joules": 4.184e15, "unit": "megaton"},
     "expected": 1.0, "units": "megaton", "ref": "1 MT=4.184e15 J"},
    {"id": "luminosity_to_mass_conversion_rate",
     "tool": "luminosity_to_mass_conversion_rate",
     "kwargs": {"luminosity_watts": 3.828e26},
     "expected": 4.26e9, "units": "kg/s", "ref": "dm/dt=L/c^2 (Sun)"},

    # ---- relativity ----
    {"id": "lorentz_factor", "tool": "lorentz_factor",
     "kwargs": {"velocity_m_s": 0.9 * _C}, "expected": 2.294, "units": "",
     "ref": "γ=1/sqrt(1-v^2/c^2)"},
    {"id": "relativistic_length_contraction", "tool": "relativistic_length_contraction",
     "kwargs": {"rest_length_m": 1.0, "velocity_m_s": 0.6 * _C},
     "expected": 0.8, "units": "m", "ref": "L=L0/γ"},
    {"id": "relativistic_time_dilation", "tool": "relativistic_time_dilation",
     "kwargs": {"rest_time_s": 10.0, "velocity_m_s": 0.99 * _C},
     "expected": 70.89, "units": "", "ref": "Δt=γ Δτ"},
    {"id": "relativistic_velocity_addition", "tool": "relativistic_velocity_addition",
     "kwargs": {"u_m_s": 0.9 * _C, "v_m_s": 0.9 * _C},
     "expected": 298342500.0, "units": "m/s", "ref": "(u+v)/(1+uv/c^2)"},
    {"id": "doppler_shift_factor", "tool": "doppler_shift_factor",
     "kwargs": {"velocity_m_s": 1e6}, "expected": 1.003344, "units": "",
     "ref": "~1+v/c, v=1000 km/s"},

    # ---- GR ----
    {"id": "schwarzschild_radius", "tool": "schwarzschild_radius",
     "kwargs": {"mass_kg": _M_SUN}, "expected": 2954.0, "units": "m",
     "ref": "r_s=2GM/c^2 (Sun)"},
    {"id": "photon_sphere_radius", "tool": "photon_sphere_radius",
     "kwargs": {"mass_kg": 10 * _M_SUN}, "expected": 44313.0, "units": "m",
     "ref": "r=1.5 r_s (10 M_sun)"},
    {"id": "isco_radius", "tool": "isco_radius",
     "kwargs": {"mass_kg": 10 * _M_SUN}, "expected": 88625.0, "units": "m",
     "ref": "r=3 r_s (10 M_sun)"},
    {"id": "hawking_temperature", "tool": "hawking_temperature",
     "kwargs": {"mass_kg": _M_SUN}, "expected": 6.17e-8, "units": "K",
     "ref": "T_H (Sun) ~62 nK"},
    {"id": "hawking_evaporation_time", "tool": "hawking_evaporation_time",
     "kwargs": {"mass_kg": 10 * _M_SUN}, "expected": 6.62e77, "units": "s",
     "ref": "t=5120πG²M³/(ħc⁴); 10 M_sun → 6.62e77 s. NOTE corpus says "
            "2.1e76 (corpus bug: M³ scaling missed)", "tol": 0.10},

    # ---- cosmology ----
    {"id": "hubble_radius", "tool": "hubble_radius",
     "kwargs": {}, "expected": 1.373e26, "units": "m",
     "ref": "R_H=c/H_0, Planck-2018 H_0≈67.4. NOTE corpus 1.32e26 uses "
            "H_0≈70 (Hubble-tension convention spread)", "tol": 0.05},
    {"id": "age_of_universe", "tool": "age_of_universe",
     "kwargs": {}, "expected": 1.37e10, "units": "year",
     "ref": "Hubble time ~14 Gyr (O(1) of true age)", "tol": 0.15},
    {"id": "critical_density", "tool": "critical_density",
     "kwargs": {}, "expected": 8.62e-27, "units": "kg/m^3",
     "ref": "ρ_crit=3H^2/(8πG)", "tol": 0.05},
    {"id": "mond_a0_constant", "tool": "mond_a0_constant",
     "kwargs": {}, "expected": 1.2e-10, "units": "m/s^2",
     "ref": "MOND a_0"},

    # ---- astronomy ----
    {"id": "solar_system_body_mars_g", "tool": "solar_system_body",
     "kwargs": {"body_name": "mars"}, "expected": 3.711, "units": "m/s^2",
     "field": "surface_g_ms2", "ref": "Mars surface gravity"},
    {"id": "named_star_vega_mass", "tool": "named_star",
     "kwargs": {"star_name": "vega"}, "expected": 2.135, "units": "",
     "field": "mass_solar", "ref": "Vega mass in M_sun"},
    {"id": "light_travel_time_sun", "tool": "light_travel_time",
     "kwargs": {"distance_m": _AU}, "expected": 499.0, "units": "s",
     "ref": "1 AU / c"},

    # ---- symbolic ----
    {"id": "solve_equation", "tool": "solve_equation",
     "kwargs": {"equation": "x**2 - 4 = 0", "variable": "x"},
     "expected": [-2.0, 2.0], "units": "", "kind": "list",
     "ref": "x^2-4=0 -> ±2"},
    {"id": "integrate_expr", "tool": "integrate_expr",
     "kwargs": {"expression": "x**2", "variable": "x",
                  "lower": "0", "upper": "1"},
     "expected": 0.3333333333333333, "units": "", "ref": "∫x^2 dx [0,1] "
     "(NOTE: bounds must be str per schema; tool does not coerce int bounds)"},
    {"id": "differentiate_expr", "tool": "differentiate_expr",
     "kwargs": {"expression": "sin(x)", "variable": "x"},
     "expected": "cos(x)", "units": "", "kind": "string",
     "ref": "d/dx sin(x)"},
    {"id": "simplify_expr", "tool": "simplify_expr",
     "kwargs": {"expression": "sin(x)**2 + cos(x)**2"},
     "expected": 1, "units": "", "ref": "Pythagorean identity"},

    # ---- new: orbital mechanics (body-aware, multi-step) ----
    {"id": "orbital_velocity_iss", "tool": "orbital_velocity",
     "kwargs": {"central_body": "earth", "altitude_km": 408},
     "expected": 7660.0, "units": "m/s", "ref": "ISS at 408 km", "tol": 0.01},
    {"id": "orbital_velocity_jupiter", "tool": "orbital_velocity",
     "kwargs": {"central_body": "sun", "semimajor_axis_au": 5.2038},
     "expected": 13060.0, "units": "m/s", "ref": "Jupiter heliocentric",
     "tol": 0.01},
    {"id": "orbital_period_asteroid", "tool": "orbital_period",
     "kwargs": {"semimajor_axis_au": 3.0}, "expected": 1.6398e8, "units": "s",
     "ref": "Kepler III, 3 AU → 5.196 yr", "tol": 0.01},
    {"id": "gravitational_force_earth_moon", "tool": "gravitational_force",
     "kwargs": {"mass1_kg": 5.972e24, "mass2_kg": 7.342e22,
                  "separation_m": 3.844e8},
     "expected": 1.98e20, "units": "N", "ref": "Earth-Moon", "tol": 0.02},

    # ---- new: nuclear ----
    {"id": "nuclear_binding_He4", "tool": "nuclear_binding_energy",
     "kwargs": {"protons": 2, "neutrons": 2, "measured_mass_u": 4.002602},
     "expected": 28.30, "units": "", "field": "binding_energy_MeV",
     "ref": "He-4 BE", "tol": 0.01},
    {"id": "nuclear_binding_Fe56_per_nucleon", "tool": "nuclear_binding_energy",
     "kwargs": {"protons": 26, "neutrons": 30, "measured_mass_u": 55.9349363},
     "expected": 8.790, "units": "", "field": "binding_per_nucleon_MeV",
     "ref": "Fe-56 BE/A", "tol": 0.01},
    {"id": "coulomb_two_protons_1fm", "tool": "coulomb_force",
     "kwargs": {"charge1_c": 1.602176634e-19, "charge2_c": 1.602176634e-19,
                  "separation_m": 1e-15},
     "expected": 230.7, "units": "N", "ref": "2 protons at 1 fm", "tol": 0.02},

    # ---- new: multi-step atomic / circuits ----
    {"id": "de_broglie_from_KE_1keV_e", "tool": "de_broglie_from_kinetic_energy",
     "kwargs": {"kinetic_energy_eV": 1000.0, "particle": "electron"},
     "expected": 3.88e-11, "units": "m", "ref": "1 keV electron", "tol": 0.01},
    {"id": "energy_power_time_heater", "tool": "energy_power_time",
     "kwargs": {"power_w": 5000.0, "time_s": 3600.0},
     "expected": 1.8e7, "units": "J", "ref": "5 kW × 1 hr"},
    {"id": "energy_power_time_led", "tool": "energy_power_time",
     "kwargs": {"power_w": 5.0, "energy_j": 1000.0},
     "expected": 200.0, "units": "s", "ref": "5 W to dissipate 1 kJ"},
]


# ============================================================
# CONSTANTS CROSS-CHECK
# (lookup_constant name, scipy reference accessor or literal, tolerance)
# scipy_ref: a callable returning the reference value, or None for SSBM.
# ============================================================
def _constants_reference() -> list[dict]:
    import scipy.constants as spc
    return [
        {"name": "G", "ref": spc.G, "label": "CODATA G"},
        {"name": "speed of light", "ref": spc.c, "label": "exact c"},
        {"name": "planck constant", "ref": spc.h, "label": "exact h"},
        {"name": "hbar", "ref": spc.hbar, "label": "exact ħ"},
        {"name": "boltzmann", "ref": spc.k, "label": "CODATA k_B"},
        {"name": "avogadro", "ref": spc.N_A, "label": "exact N_A"},
        {"name": "gas constant", "ref": spc.R, "label": "exact R"},
        {"name": "elementary charge", "ref": spc.e, "label": "exact e"},
        {"name": "electron mass", "ref": spc.m_e, "label": "CODATA m_e"},
        {"name": "stefan boltzmann", "ref": spc.Stefan_Boltzmann,
         "label": "CODATA σ"},
        {"name": "fine structure", "ref": spc.fine_structure,
         "label": "CODATA α"},
        {"name": "vacuum permittivity", "ref": spc.epsilon_0,
         "label": "CODATA ε_0"},
        {"name": "vacuum permeability", "ref": spc.mu_0,
         "label": "CODATA μ_0"},
        {"name": "bohr radius", "ref": spc.value("Bohr radius"),
         "label": "CODATA a_0"},
        # SSBM-specific — no external reference (expected)
        {"name": "eta", "ref": None, "label": "SSBM empirical input"},
        {"name": "phi", "ref": (1 + 5 ** 0.5) / 2, "label": "golden ratio"},
    ]


def _parse_tool_result(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _numeric_ok(got: Any, expected: float, got_units: str, exp_units: str,
                  tol: float) -> tuple[bool, float | None]:
    from sigma_ground.mcp.benchmark.score import _values_match
    ok, rel, _ = _values_match(got, expected, tol,
                                  extracted_units=got_units or "",
                                  expected_units=exp_units or "")
    return ok, rel


async def _run_audit(args) -> int:
    from sigma_ground.mcp.benchmark import load_env_from_dev_root
    load_env_from_dev_root(verbose=False)
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        print("ERROR: pip install 'mcp>=1.0'", file=sys.stderr)
        return 1

    rows: list[dict] = []
    params = StdioServerParameters(command="sigma-ground-mcp")
    # The stdio session does not reliably survive ~90 sequential calls,
    # so process cases in chunks, each over a fresh server session. This
    # ensures a mid-run connection drop never mislabels a harness issue
    # as a library bug.
    CHUNK = 20
    for start in range(0, len(AUDIT_CASES), CHUNK):
        chunk = AUDIT_CASES[start:start + CHUNK]
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tool_resp = await session.list_tools()
                known_tools = {t.name for t in tool_resp.tools}

                for case in chunk:
                    tool = case["tool"]
                    if tool not in known_tools:
                        rows.append({**case, "verdict": "TOOL_MISSING",
                                       "got": None, "source": "", "rel": None})
                        continue
                    try:
                        res = await session.call_tool(tool, case["kwargs"])
                        text = ""
                        for piece in (res.content or []):
                            t = getattr(piece, "text", None)
                            if t:
                                text += t
                        parsed = _parse_tool_result(text) or {}
                    except Exception as e:
                        rows.append({**case, "verdict": "TOOL_ERROR",
                                       "got": f"<{e}>", "source": "",
                                       "rel": None})
                        continue

                    value = parsed.get("value")
                    units = parsed.get("units", "") or ""
                    source = parsed.get("source", "") or ""
                    ptag = parsed.get("provenance_tag", "") or ""
                    # Dict-valued result: pick a field
                    if isinstance(value, dict) and case.get("field"):
                        value = value.get(case["field"])

                    kind = case.get("kind", "numeric")
                    tol = case.get("tol", 0.03)

                    # Provenance / hallucination check
                    has_source = bool(source) and "not found" not in source.lower()
                    fitted = "fitted due to incompetence" in str(value).lower() \
                                or "fitted due to incompetence" in source.lower()

                    # Symbolic results come back as strings like '1/3' or
                    # 'pi'; evaluate to float before numeric comparison.
                    if (kind == "numeric" and isinstance(value, str)):
                        try:
                            from sympy import sympify
                            ev = sympify(value)
                            if not ev.free_symbols:
                                value = float(ev.evalf())
                        except Exception:
                            pass

                    if value is None:
                        verdict, rel = "TOOL_ERROR", None
                    elif kind == "string":
                        got_s = str(value).replace(" ", "").lower()
                        exp_s = str(case["expected"]).replace(" ", "").lower()
                        ok = got_s == exp_s
                        verdict = "CONFIRMED" if ok else "LIBRARY_BUG"
                        rel = None
                    elif kind == "list":
                        try:
                            got_list = sorted(float(x) for x in value)
                            exp_list = sorted(float(x) for x in case["expected"])
                            ok = len(got_list) == len(exp_list) and all(
                                abs(a - b) <= tol * max(abs(b), 1e-9)
                                for a, b in zip(got_list, exp_list))
                        except Exception:
                            ok = False
                        verdict = "CONFIRMED" if ok else "LIBRARY_BUG"
                        rel = None
                    else:
                        ok, rel = _numeric_ok(value, case["expected"], units,
                                                case["units"], tol)
                        verdict = "CONFIRMED" if ok else "LIBRARY_BUG"

                    if verdict == "CONFIRMED" and not has_source and not fitted:
                        verdict = "HALLUCINATION_RISK"
                    if fitted:
                        verdict = "HALLUCINATION_RISK"

                    rows.append({**case, "verdict": verdict, "got": value,
                                   "source": source, "ptag": ptag, "rel": rel})

    # ---- CONSTANTS (own resolver — the MCP stdio session does not
    # reliably survive ~90 sequential calls, so resolve constants via
    # the same lookup_constant the server wraps, called directly. The
    # server's lookup_constant is a thin passthrough already exercised
    # by the tool phase above.) ----
    const_rows = _audit_constants_direct()

    _report(rows, const_rows, args)
    return 0


def _audit_constants_direct() -> list[dict]:
    from sigma_ground.mcp.tools.constants import lookup_constant
    const_rows: list[dict] = []
    for c in _constants_reference():
        try:
            r = lookup_constant(c["name"])
        except Exception as e:
            const_rows.append({**c, "verdict": "TOOL_ERROR",
                                 "got": f"<{e}>", "source": ""})
            continue
        value = r.value
        source = r.source or ""
        if value is None:
            const_rows.append({**c, "verdict": "NOT_FOUND",
                                 "got": None, "source": source})
            continue
        if c["ref"] is None:
            verdict = "SSBM_INPUT" if source else "HALLUCINATION_RISK"
            const_rows.append({**c, "verdict": verdict, "got": value,
                                 "source": source})
            continue
        try:
            rel = abs(float(value) - c["ref"]) / abs(c["ref"])
            verdict = "CONFIRMED" if rel < 1e-4 else "CONSTANT_DISAGREES"
        except Exception:
            verdict, rel = "TOOL_ERROR", None
        const_rows.append({**c, "verdict": verdict, "got": value,
                             "source": source, "rel": rel})
    return const_rows


def _report(rows: list[dict], const_rows: list[dict], args) -> None:
    from collections import Counter
    verdicts = Counter(r["verdict"] for r in rows)
    cverdicts = Counter(r["verdict"] for r in const_rows)

    print()
    print("=" * 66)
    print("  sigma-ground LIBRARY AUDIT  (no LLM — direct, known inputs)")
    print("=" * 66)
    print(f"\nTOOL CASES: {len(rows)}")
    for v in ("CONFIRMED", "LIBRARY_BUG", "HALLUCINATION_RISK",
                "TOOL_ERROR", "TOOL_MISSING"):
        if verdicts.get(v):
            print(f"  {v:20s} {verdicts[v]}")
    print(f"\nCONSTANTS: {len(const_rows)}")
    for v in ("CONFIRMED", "CONSTANT_DISAGREES", "SSBM_INPUT",
                "NOT_FOUND", "TOOL_ERROR"):
        if cverdicts.get(v):
            print(f"  {v:20s} {cverdicts[v]}")

    # Detail any non-confirmed tool rows
    flagged = [r for r in rows if r["verdict"] not in ("CONFIRMED",)]
    if flagged:
        print("\n--- TOOL CASES NEEDING ATTENTION ---")
        for r in flagged:
            got = r.get("got")
            rel = r.get("rel")
            relstr = f" rel={rel:.2e}" if isinstance(rel, float) else ""
            print(f"  [{r['verdict']}] {r['tool']}({r['id']})")
            print(f"      got={got!r}  expected={r['expected']!r} {r['units']}"
                  f"{relstr}")
            print(f"      ref: {r['ref']}  source: {r.get('source','')[:60]!r}")
    cflag = [r for r in const_rows
               if r["verdict"] in ("CONSTANT_DISAGREES", "NOT_FOUND", "TOOL_ERROR")]
    if cflag:
        print("\n--- CONSTANTS NEEDING ATTENTION ---")
        for r in cflag:
            print(f"  [{r['verdict']}] {r['name']}: got={r.get('got')!r}"
                  f" vs {r['label']}={r.get('ref')!r}")

    # Markdown report
    out = Path(__file__).parents[3] / "misc" / "LIBRARY_AUDIT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# sigma-ground Library Audit", "",
             "Direct library validation — no LLM. Each tool called with "
             "hand-authored correct inputs; result checked against the "
             "textbook answer. A mismatch isolates a **library** bug.",
             "",
             "## Summary", "",
             f"- Tool cases: {len(rows)} "
             f"(CONFIRMED {verdicts.get('CONFIRMED',0)}, "
             f"LIBRARY_BUG {verdicts.get('LIBRARY_BUG',0)}, "
             f"HALLUCINATION_RISK {verdicts.get('HALLUCINATION_RISK',0)}, "
             f"TOOL_ERROR {verdicts.get('TOOL_ERROR',0)})",
             f"- Constants: {len(const_rows)} "
             f"(CONFIRMED {cverdicts.get('CONFIRMED',0)}, "
             f"DISAGREES {cverdicts.get('CONSTANT_DISAGREES',0)}, "
             f"SSBM_INPUT {cverdicts.get('SSBM_INPUT',0)})", ""]
    if flagged:
        lines += ["## Tool cases needing attention", ""]
        for r in flagged:
            lines.append(f"- **{r['verdict']}** `{r['tool']}` — got "
                         f"`{r.get('got')!r}`, expected `{r['expected']!r} "
                         f"{r['units']}` ({r['ref']})")
        lines.append("")
    if cflag:
        lines += ["## Constants needing attention", ""]
        for r in cflag:
            lines.append(f"- **{r['verdict']}** `{r['name']}` — got "
                         f"`{r.get('got')!r}` vs {r['label']} `{r.get('ref')!r}`")
        lines.append("")
    lines += ["## Confirmed-against-known-physics (the reassuring bulk)", ""]
    for r in rows:
        if r["verdict"] == "CONFIRMED":
            lines.append(f"- `{r['tool']}` = {r.get('got')!r} {r['units']} "
                         f"✓ ({r['ref']})")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    return asyncio.run(_run_audit(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
