"""FastMCP server entry point.

Run:
    sigma-ground-mcp          # stdio transport (standard MCP)
    python -m sigma_ground.mcp.server

Connects via the Model Context Protocol so any MCP-aware client (Claude
Desktop, Cline, Continue, Aider, etc., or a custom LLM-front-end
talking to Qwen/Llama/Gemini) can invoke physics tools.

Each tool returns a ToolResult dict. The LLM is instructed via system
prompt to faithfully cite the `source` and `provenance_tag` fields.
"""

from __future__ import annotations

import sys
from typing import Any

from sigma_ground.mcp.provenance import ToolResult


SYSTEM_INSTRUCTIONS = """\
You are a physics assistant backed by sigma-ground, a curated physics
library with rigorous provenance, plus wrapped externals (scipy.constants,
pint, sympy, astropy, periodictable).

ABSOLUTE RULES (these override convenience):

1. For every numerical value you put in an answer, you MUST do one of:
   (a) Call a sigma-ground MCP tool. Faithfully report the `source`
       and (when present) `provenance_tag` and `uncertainty` from the
       returned ToolResult. Phrase attribution like:
           "value units (sigma-ground via <source>)"
       Example: "2954 m (sigma-ground via gr_basics, standard Schwarzschild)"
   (b) If no tool can supply the value, state the value followed by:
           "[SOURCE: Fitted due to incompetence -- sigma-ground library
            lacks <specific physics description>. This value is my best
            estimate; please verify.]"
       Example: "Approximately 124 degrees C [SOURCE: Fitted due to
       incompetence -- sigma-ground library lacks melting-point data for
       trans-stilbene; this value is my best estimate; please verify.]"

   You are FORBIDDEN to silently insert numerical values from memory
   without one of these two attributions. If you find yourself wanting
   to state a number, STOP, and either call a tool or use the fitted tag.

2. Default to standard physics. Use SI units internally. Use
   convert_units() if the user asks for the result in non-SI units.

3. Always call get_manifest() at session start (or when uncertain) to
   know what tools exist. Do not hallucinate tool names.

PROVENANCE TAGS (in ToolResult.provenance_tag):
- VERIFIED            -- measured from CODATA/PDG/IAU/peer-reviewed
- DERIVED             -- computed from other library constants
- EMPIRICAL-INPUT     -- free parameter set by observation (e.g. HDE c^2)
- SPECULATIVE-PENDING -- placeholder; flag prominently
- REJECTED            -- former candidate, now disproven (value None)

SCOPE:
Mentat answers standard, observation-anchored physics and chemistry. Do
not introduce speculative or non-standard frameworks. If a question falls
outside the curated tools, decline or use the fitted-due-to-incompetence
tag rather than inventing a mechanism.

CONVERSATIONAL STYLE:
- Plain language, no equations unless asked.
- Include the tool-call citations naturally.
- If a value is "Fitted due to incompetence", say so prominently --
  the user MUST know the value isn't grounded.
"""


def main() -> int:
    """MCP server entry point. Returns process exit code."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("ERROR: mcp package not installed. "
              "Install with: pip install sigma-ground[mcp]",
              file=sys.stderr)
        return 1

    from sigma_ground.mcp.tools import constants as t_const
    from sigma_ground.mcp.tools import units as t_units
    from sigma_ground.mcp.tools import symbolic as t_sym
    from sigma_ground.mcp.tools import gr as t_gr
    from sigma_ground.mcp.tools import kinematics as t_kin
    from sigma_ground.mcp.tools import energy_conversion as t_econv
    from sigma_ground.mcp.tools import relativity as t_rel
    from sigma_ground.mcp.tools import cosmology as t_cos
    from sigma_ground.mcp.tools import thermodynamics as t_thermo
    from sigma_ground.mcp.tools import optics as t_opt
    from sigma_ground.mcp.tools import materials as t_mat
    from sigma_ground.mcp.tools import circuits as t_circ
    from sigma_ground.mcp.tools import atomic as t_atom
    from sigma_ground.mcp.tools import astronomy as t_astr
    from sigma_ground.mcp.tools import orbital as t_orb
    from sigma_ground.mcp.tools import nuclear as t_nuc
    from sigma_ground.mcp.tools import simulation as t_sim
    from sigma_ground.mcp.tools import chemistry as t_chem
    from sigma_ground.mcp.tools import electronics as t_elec
    from sigma_ground.mcp.tools import mathx as t_mathx
    from sigma_ground.mcp.tools import frontier as t_front
    from sigma_ground.mcp.tools import mechanics as t_mech
    from sigma_ground.mcp.tools import transport as t_trans
    from sigma_ground.mcp.tools import rotation as t_rot
    from sigma_ground.mcp.tools import materials_strength as t_matstr
    from sigma_ground.mcp import procedures as t_proc
    from sigma_ground.mcp import manifest as t_manifest

    server = FastMCP("mentat")

    # ── manifest ──────────────────────────────────────────────────────
    @server.tool()
    def get_manifest() -> dict[str, Any]:
        """List all available physics tools, grouped by domain and tier.

        Call this first to learn the tool surface. The LLM should plan
        multi-step answers using the listed tools.
        """
        return t_manifest.get_manifest().to_dict()

    # ── constants ────────────────────────────────────────────────────
    @server.tool()
    def lookup_constant(name: str) -> dict[str, Any]:
        """Look up a physical constant by name with units and provenance.

        Tries sigma_ground curated constants first, then CODATA via
        scipy.constants. Returns value, units, source, uncertainty, and
        (when curated) a provenance_tag like VERIFIED or DERIVED.

        Args:
            name: Constant name. Case-insensitive. Common synonyms accepted
                  (e.g. "G", "speed_of_light", "electron_mass").
        """
        return t_const.lookup_constant(name).to_dict()

    @server.tool()
    def list_constants(category: str | None = None,
                        contains: str | None = None,
                        limit: int = 50) -> dict[str, Any]:
        """List available constants, optionally filtered.

        Args:
            category: 'sigma_ground' | 'codata' | 'all' (default 'all').
            contains: Substring filter on the constant name.
            limit: Maximum number of results.
        """
        return t_const.list_constants(category, contains, limit).to_dict()

    # ── units ────────────────────────────────────────────────────────
    @server.tool()
    def convert_units(value: float, from_units: str,
                       to_units: str) -> dict[str, Any]:
        """Convert a value between unit systems (pint).

        Examples:
            convert_units(5, "eV", "joule")
            convert_units(1, "solar_mass", "kg")
            convert_units(1.496e11, "m", "AU")
        """
        return t_units.convert(value, from_units, to_units).to_dict()

    @server.tool()
    def parse_quantity(quantity_string: str) -> dict[str, Any]:
        """Parse '5.6 light_year' into magnitude + units."""
        return t_units.parse_quantity(quantity_string).to_dict()

    # ── symbolic math ────────────────────────────────────────────────
    @server.tool()
    def solve_equation(equation: str, variable: str) -> dict[str, Any]:
        """Symbolic solve via sympy. Accepts 'expr' (= 0) or 'lhs = rhs'.

        Args:
            equation: Expression set to zero, or 'lhs = rhs' form.
            variable: Symbol name to solve for.
        """
        return t_sym.solve_equation(equation, variable).to_dict()

    @server.tool()
    def integrate_expr(expression: str, variable: str,
                        lower: str | None = None,
                        upper: str | None = None) -> dict[str, Any]:
        """Symbolic integration. Definite if lower+upper given.

        Args:
            expression: The integrand.
            variable: Variable of integration.
            lower, upper: Optional integration bounds (sympy-parseable).
        """
        return t_sym.integrate_expr(expression, variable,
                                       lower, upper).to_dict()

    @server.tool()
    def differentiate_expr(expression: str, variable: str,
                            order: int = 1) -> dict[str, Any]:
        """Symbolic differentiation."""
        return t_sym.differentiate_expr(expression, variable, order).to_dict()

    @server.tool()
    def simplify_expr(expression: str) -> dict[str, Any]:
        """Apply sympy.simplify to an expression."""
        return t_sym.simplify_expr(expression).to_dict()

    # ── general relativity ──────────────────────────────────────────
    @server.tool()
    def schwarzschild_radius(mass_kg: float) -> dict[str, Any]:
        """Schwarzschild radius r_s = 2 G M / c^2 (meters)."""
        return t_gr.schwarzschild_radius(mass_kg).to_dict()

    @server.tool()
    def isco_radius(mass_kg: float) -> dict[str, Any]:
        """Innermost stable circular orbit r_ISCO = 6 G M / c^2."""
        return t_gr.isco_radius(mass_kg).to_dict()

    @server.tool()
    def photon_sphere_radius(mass_kg: float) -> dict[str, Any]:
        """Photon sphere r_ph = 3 G M / c^2."""
        return t_gr.photon_sphere_radius(mass_kg).to_dict()

    @server.tool()
    def hawking_temperature(mass_kg: float) -> dict[str, Any]:
        """Hawking temperature T_H = hbar c^3 / (8 pi G M k_B)."""
        return t_gr.hawking_temperature(mass_kg).to_dict()

    @server.tool()
    def hawking_evaporation_time(mass_kg: float) -> dict[str, Any]:
        """Black-hole evaporation timescale in seconds."""
        return t_gr.hawking_evaporation_time(mass_kg).to_dict()

    @server.tool()
    def gravitational_redshift(mass_kg: float,
                                radius_m: float) -> dict[str, Any]:
        """Schwarzschild gravitational redshift z = (1 - r_s/r)^(-1/2) - 1."""
        return t_gr.gravitational_redshift(mass_kg, radius_m).to_dict()

    @server.tool()
    def gravitational_time_dilation(mass_kg: float,
                                      radius_m: float) -> dict[str, Any]:
        """Clock rate at r relative to infinity: sqrt(1 - r_s/r)."""
        return t_gr.gravitational_time_dilation(mass_kg, radius_m).to_dict()

    # ── kinematics ──────────────────────────────────────────────────
    @server.tool()
    def free_fall_time(height_m: float, g_m_s2: float = 9.80665) -> dict[str, Any]:
        """Time to fall from rest: t = sqrt(2h/g). Vacuum approximation."""
        return t_kin.free_fall_time(height_m, g_m_s2).to_dict()

    @server.tool()
    def free_fall_velocity(height_m: float, g_m_s2: float = 9.80665) -> dict[str, Any]:
        """Impact speed: v = sqrt(2gh). Vacuum."""
        return t_kin.free_fall_velocity(height_m, g_m_s2).to_dict()

    @server.tool()
    def projectile_range(initial_speed_m_s: float, launch_angle_deg: float,
                          g_m_s2: float = 9.80665) -> dict[str, Any]:
        """Range R = v^2 sin(2 theta) / g (level ground, vacuum)."""
        return t_kin.projectile_range(initial_speed_m_s, launch_angle_deg,
                                         g_m_s2).to_dict()

    @server.tool()
    def projectile_max_height(initial_speed_m_s: float,
                                launch_angle_deg: float,
                                g_m_s2: float = 9.80665) -> dict[str, Any]:
        """Max height = (v sin theta)^2 / (2g)."""
        return t_kin.projectile_max_height(initial_speed_m_s,
                                              launch_angle_deg,
                                              g_m_s2).to_dict()

    @server.tool()
    def projectile_flight_time(initial_speed_m_s: float,
                                 launch_angle_deg: float,
                                 g_m_s2: float = 9.80665) -> dict[str, Any]:
        """Flight time = 2 v sin(theta) / g."""
        return t_kin.projectile_flight_time(initial_speed_m_s,
                                               launch_angle_deg,
                                               g_m_s2).to_dict()

    @server.tool()
    def kinetic_energy(mass_kg: float, velocity_m_s: float) -> dict[str, Any]:
        """Non-relativistic KE = 0.5 m v^2."""
        return t_kin.kinetic_energy(mass_kg, velocity_m_s).to_dict()

    @server.tool()
    def momentum(mass_kg: float, velocity_m_s: float) -> dict[str, Any]:
        """Non-relativistic momentum p = m v."""
        return t_kin.momentum(mass_kg, velocity_m_s).to_dict()

    @server.tool()
    def gravitational_potential_energy(mass_kg: float, height_m: float,
                                         g_m_s2: float = 9.80665) -> dict[str, Any]:
        """Uniform gravity U = m g h."""
        return t_kin.gravitational_potential_energy(mass_kg, height_m,
                                                       g_m_s2).to_dict()

    @server.tool()
    def friction_stopping_distance(mass_kg: float, initial_velocity_m_s: float,
                                     friction_coefficient: float,
                                     g_m_s2: float = 9.80665) -> dict[str, Any]:
        """Sliding stopping distance d = v^2 / (2 mu g)."""
        return t_kin.friction_stopping_distance(mass_kg, initial_velocity_m_s,
                                                   friction_coefficient,
                                                   g_m_s2).to_dict()

    @server.tool()
    def circular_orbit_velocity(central_mass_kg: float,
                                  radius_m: float) -> dict[str, Any]:
        """Keplerian circular orbit v = sqrt(G M / r)."""
        return t_kin.circular_orbit_velocity(central_mass_kg, radius_m).to_dict()

    @server.tool()
    def escape_velocity(mass_kg: float, radius_m: float) -> dict[str, Any]:
        """Classical escape velocity v_esc = sqrt(2 G M / r)."""
        return t_kin.escape_velocity_classical(mass_kg, radius_m).to_dict()

    # ── energy conversion ───────────────────────────────────────────
    @server.tool()
    def mass_to_energy(mass_kg: float) -> dict[str, Any]:
        """E = m c^2."""
        return t_econv.mass_to_energy(mass_kg).to_dict()

    @server.tool()
    def energy_to_mass(energy_j: float) -> dict[str, Any]:
        """m = E / c^2."""
        return t_econv.energy_to_mass(energy_j).to_dict()

    @server.tool()
    def luminosity_to_mass_conversion_rate(luminosity_watts: float) -> dict[str, Any]:
        """dm/dt = L / c^2 (e.g. Sun converts ~4.26e9 kg/s to energy)."""
        return t_econv.luminosity_to_mass_conversion_rate(luminosity_watts).to_dict()

    @server.tool()
    def joules_to_eV(energy_joules: float) -> dict[str, Any]:
        """Convert J to eV (1 eV = 1.602176634e-19 J)."""
        return t_econv.joules_to_eV(energy_joules).to_dict()

    @server.tool()
    def eV_to_joules(energy_eV: float) -> dict[str, Any]:
        """Convert eV to J."""
        return t_econv.eV_to_joules(energy_eV).to_dict()

    @server.tool()
    def joules_to_TNT(energy_joules: float, unit: str = "ton") -> dict[str, Any]:
        """Convert joules to TNT equivalent. unit: 'ton', 'kt', 'MT'."""
        return t_econv.joules_to_TNT(energy_joules, unit).to_dict()

    # ── special relativity ─────────────────────────────────────────
    @server.tool()
    def lorentz_factor(velocity_m_s: float) -> dict[str, Any]:
        """gamma = 1 / sqrt(1 - v^2/c^2)."""
        return t_rel.lorentz_factor(velocity_m_s).to_dict()

    @server.tool()
    def relativistic_time_dilation(rest_time_s: float,
                                     velocity_m_s: float) -> dict[str, Any]:
        """t = gamma t0 (moving clock ticks slower)."""
        return t_rel.relativistic_time_dilation(rest_time_s, velocity_m_s).to_dict()

    @server.tool()
    def relativistic_length_contraction(rest_length_m: float,
                                          velocity_m_s: float) -> dict[str, Any]:
        """L = L0 / gamma."""
        return t_rel.relativistic_length_contraction(rest_length_m,
                                                        velocity_m_s).to_dict()

    @server.tool()
    def relativistic_energy(rest_mass_kg: float,
                              velocity_m_s: float) -> dict[str, Any]:
        """Total energy E = gamma m c^2."""
        return t_rel.relativistic_energy(rest_mass_kg, velocity_m_s).to_dict()

    @server.tool()
    def relativistic_momentum(rest_mass_kg: float,
                                velocity_m_s: float) -> dict[str, Any]:
        """p = gamma m v."""
        return t_rel.relativistic_momentum(rest_mass_kg, velocity_m_s).to_dict()

    @server.tool()
    def relativistic_velocity_addition(u_m_s: float, v_m_s: float) -> dict[str, Any]:
        """Einstein velocity addition w = (u+v)/(1+uv/c^2)."""
        return t_rel.relativistic_velocity_addition(u_m_s, v_m_s).to_dict()

    @server.tool()
    def doppler_shift_factor(velocity_m_s: float,
                               angle_to_los_deg: float = 0.0) -> dict[str, Any]:
        """Relativistic Doppler factor lambda_obs / lambda_emit."""
        return t_rel.doppler_shift_factor(velocity_m_s, angle_to_los_deg).to_dict()

    # ── cosmology ───────────────────────────────────────────────────
    @server.tool()
    def hubble_radius() -> dict[str, Any]:
        """R_H = c / H_0 (~14 Gpc)."""
        return t_cos.hubble_radius().to_dict()

    @server.tool()
    def hde_dark_energy_density(c_squared: float | None = None,
                                  L_meters: float | None = None) -> dict[str, Any]:
        """Holographic dark energy density. Defaults c^2=DESI Union3 fit, L=R_H."""
        return t_cos.hde_dark_energy_density(c_squared, L_meters).to_dict()

    @server.tool()
    def eta_desi_band_check(dataset: str = "dr2") -> dict[str, Any]:
        """Is the adopted HDE c^2 within the DESI Union3 1-sigma band?"""
        return t_cos.eta_desi_band_check(dataset).to_dict()

    @server.tool()
    def mond_regime_classifier(acceleration_m_s2: float) -> dict[str, Any]:
        """Classify acceleration as newtonian / mond / transition."""
        return t_cos.mond_regime_classifier(acceleration_m_s2).to_dict()

    @server.tool()
    def mond_a0_constant() -> dict[str, Any]:
        """Milgrom's a_0 ~ 1.2e-10 m/s^2."""
        return t_cos.mond_a0_constant().to_dict()

    @server.tool()
    def critical_density() -> dict[str, Any]:
        """Cosmological critical density 3 H_0^2 / (8 pi G)."""
        return t_cos.critical_density().to_dict()

    @server.tool()
    def age_of_universe() -> dict[str, Any]:
        """Hubble time t_H = 1/H_0 (approx 13.8 Gyr in LambdaCDM)."""
        return t_cos.age_of_universe().to_dict()

    @server.tool()
    def eta_value_report() -> dict[str, Any]:
        """Adopted HDE c^2 = DESI Union3 fit ~ 0.4122 (with provenance)."""
        return t_cos.eta_value_report().to_dict()

    # ── thermodynamics ─────────────────────────────────────────────
    @server.tool()
    def ideal_gas_pressure(n_moles: float, temperature_k: float,
                            volume_m3: float) -> dict[str, Any]:
        """P = n R T / V."""
        return t_thermo.ideal_gas_pressure(n_moles, temperature_k,
                                              volume_m3).to_dict()

    @server.tool()
    def ideal_gas_volume(n_moles: float, temperature_k: float,
                          pressure_pa: float) -> dict[str, Any]:
        """V = n R T / P."""
        return t_thermo.ideal_gas_volume(n_moles, temperature_k,
                                            pressure_pa).to_dict()

    @server.tool()
    def blackbody_peak_wavelength(temperature_k: float) -> dict[str, Any]:
        """Wien's law: lambda_max = b / T."""
        return t_thermo.blackbody_peak_wavelength(temperature_k).to_dict()

    @server.tool()
    def blackbody_total_power(temperature_k: float, area_m2: float = 1.0,
                                emissivity: float = 1.0) -> dict[str, Any]:
        """Stefan-Boltzmann: P = epsilon sigma A T^4."""
        return t_thermo.blackbody_total_power(temperature_k, area_m2,
                                                 emissivity).to_dict()

    @server.tool()
    def carnot_efficiency(t_hot_k: float, t_cold_k: float) -> dict[str, Any]:
        """eta = 1 - T_cold/T_hot."""
        return t_thermo.carnot_efficiency(t_hot_k, t_cold_k).to_dict()

    @server.tool()
    def thermal_energy_per_molecule(temperature_k: float,
                                      degrees_of_freedom: int = 3) -> dict[str, Any]:
        """E = (f/2) k_B T per molecule via equipartition."""
        return t_thermo.thermal_energy_per_molecule(temperature_k,
                                                       degrees_of_freedom).to_dict()

    @server.tool()
    def speed_of_sound_in_ideal_gas(temperature_k: float, gamma: float = 1.4,
                                      molar_mass_kg_per_mol: float = 0.029
                                      ) -> dict[str, Any]:
        """v_sound = sqrt(gamma R T / M). Defaults to air."""
        return t_thermo.speed_of_sound_in_ideal_gas(
            temperature_k, gamma, molar_mass_kg_per_mol).to_dict()

    @server.tool()
    def maxwell_boltzmann_most_probable_speed(temperature_k: float,
                                                 molecular_mass_kg: float
                                                 ) -> dict[str, Any]:
        """v_p = sqrt(2 k_B T / m)."""
        return t_thermo.maxwell_boltzmann_most_probable_speed(
            temperature_k, molecular_mass_kg).to_dict()

    @server.tool()
    def temperature_celsius_to_kelvin(t_celsius: float) -> dict[str, Any]:
        """T_K = T_C + 273.15."""
        return t_thermo.temperature_celsius_to_kelvin(t_celsius).to_dict()

    @server.tool()
    def temperature_kelvin_to_celsius(t_kelvin: float) -> dict[str, Any]:
        """T_C = T_K - 273.15."""
        return t_thermo.temperature_kelvin_to_celsius(t_kelvin).to_dict()

    # ── optics ──────────────────────────────────────────────────────
    @server.tool()
    def snells_law_refraction_angle(n1: float, n2: float,
                                      incident_angle_deg: float) -> dict[str, Any]:
        """n1 sin(theta1) = n2 sin(theta2)."""
        return t_opt.snells_law_refraction_angle(n1, n2,
                                                    incident_angle_deg).to_dict()

    @server.tool()
    def critical_angle_for_tir(n_dense: float, n_rare: float) -> dict[str, Any]:
        """Total internal reflection critical angle."""
        return t_opt.critical_angle_for_tir(n_dense, n_rare).to_dict()

    @server.tool()
    def thin_lens_focal_length(object_distance_m: float,
                                 image_distance_m: float) -> dict[str, Any]:
        """1/f = 1/d_o + 1/d_i."""
        return t_opt.thin_lens_focal_length(object_distance_m,
                                                image_distance_m).to_dict()

    @server.tool()
    def thin_lens_image_distance(object_distance_m: float,
                                   focal_length_m: float) -> dict[str, Any]:
        """d_i = 1/(1/f - 1/d_o)."""
        return t_opt.thin_lens_image_distance(object_distance_m,
                                                  focal_length_m).to_dict()

    @server.tool()
    def lens_magnification(object_distance_m: float,
                             image_distance_m: float) -> dict[str, Any]:
        """m = -d_i / d_o."""
        return t_opt.lens_magnification(object_distance_m,
                                            image_distance_m).to_dict()

    @server.tool()
    def rydberg_hydrogen_wavelength(n_initial: int, n_final: int) -> dict[str, Any]:
        """Rydberg formula for hydrogen transitions."""
        return t_opt.rydberg_hydrogen_wavelength(n_initial, n_final).to_dict()

    @server.tool()
    def double_slit_fringe_spacing(wavelength_m: float, slit_separation_m: float,
                                     screen_distance_m: float) -> dict[str, Any]:
        """Young's double-slit y = lambda L / d."""
        return t_opt.double_slit_fringe_spacing(wavelength_m, slit_separation_m,
                                                   screen_distance_m).to_dict()

    @server.tool()
    def single_slit_first_minimum_angle(wavelength_m: float,
                                          slit_width_m: float) -> dict[str, Any]:
        """sin(theta) = lambda / a."""
        return t_opt.single_slit_first_minimum_angle(wavelength_m,
                                                        slit_width_m).to_dict()

    @server.tool()
    def diffraction_grating_angle(wavelength_m: float, grating_spacing_m: float,
                                    order: int = 1) -> dict[str, Any]:
        """d sin(theta) = m lambda."""
        return t_opt.diffraction_grating_angle(wavelength_m, grating_spacing_m,
                                                  order).to_dict()

    # ── materials ───────────────────────────────────────────────────
    @server.tool()
    def density(material: str) -> dict[str, Any]:
        """Density of a material in kg/m^3 (e.g. 'water', 'copper')."""
        return t_mat.density(material).to_dict()

    @server.tool()
    def refractive_index(material: str) -> dict[str, Any]:
        """Refractive index at 589 nm (e.g. 'water'=1.333, 'diamond'=2.42)."""
        return t_mat.refractive_index(material).to_dict()

    @server.tool()
    def melting_point(material: str) -> dict[str, Any]:
        """Melting point at 1 atm in K."""
        return t_mat.melting_point(material).to_dict()

    @server.tool()
    def boiling_point(material: str) -> dict[str, Any]:
        """Boiling point at 1 atm in K."""
        return t_mat.boiling_point(material).to_dict()

    @server.tool()
    def youngs_modulus(material: str) -> dict[str, Any]:
        """Young's modulus in Pa."""
        return t_mat.youngs_modulus(material).to_dict()

    @server.tool()
    def band_gap_ev(material: str) -> dict[str, Any]:
        """Semiconductor band gap at 300 K in eV."""
        return t_mat.band_gap_ev(material).to_dict()

    @server.tool()
    def list_materials() -> dict[str, Any]:
        """List all materials available in lookup tables."""
        return t_mat.list_materials().to_dict()

    @server.tool()
    def element_atomic_data(element_symbol: str) -> dict[str, Any]:
        """Atomic Z, name, mass via periodictable (e.g. 'Au')."""
        return t_mat.element_atomic_data(element_symbol).to_dict()

    # ── electrical circuits ────────────────────────────────────────
    @server.tool()
    def ohms_law_voltage(current_a: float,
                           resistance_ohm: float) -> dict[str, Any]:
        """V = I R."""
        return t_circ.ohms_law_voltage(current_a, resistance_ohm).to_dict()

    @server.tool()
    def ohms_law_current(voltage_v: float,
                           resistance_ohm: float) -> dict[str, Any]:
        """I = V / R."""
        return t_circ.ohms_law_current(voltage_v, resistance_ohm).to_dict()

    @server.tool()
    def electrical_power(voltage_v: float, current_a: float) -> dict[str, Any]:
        """P = V I."""
        return t_circ.electrical_power(voltage_v, current_a).to_dict()

    @server.tool()
    def power_dissipation_resistor(current_a: float,
                                     resistance_ohm: float) -> dict[str, Any]:
        """P = I^2 R."""
        return t_circ.power_dissipation_resistor(current_a, resistance_ohm).to_dict()

    @server.tool()
    def parallel_plate_capacitance(area_m2: float, separation_m: float,
                                     dielectric_constant: float = 1.0
                                     ) -> dict[str, Any]:
        """C = eps_0 eps_r A / d."""
        return t_circ.parallel_plate_capacitance(area_m2, separation_m,
                                                    dielectric_constant).to_dict()

    @server.tool()
    def rc_time_constant(resistance_ohm: float,
                           capacitance_f: float) -> dict[str, Any]:
        """tau = R C."""
        return t_circ.rc_time_constant(resistance_ohm, capacitance_f).to_dict()

    @server.tool()
    def rl_time_constant(resistance_ohm: float,
                           inductance_h: float) -> dict[str, Any]:
        """tau = L / R."""
        return t_circ.rl_time_constant(resistance_ohm, inductance_h).to_dict()

    @server.tool()
    def rlc_resonant_frequency(inductance_h: float,
                                 capacitance_f: float) -> dict[str, Any]:
        """omega_0 = 1 / sqrt(L C). Returns angular frequency in rad/s."""
        return t_circ.rlc_resonant_frequency(inductance_h, capacitance_f).to_dict()

    @server.tool()
    def em_wave_wavelength(frequency_hz: float,
                             refractive_index: float = 1.0) -> dict[str, Any]:
        """lambda = c / (n f)."""
        return t_circ.em_wave_wavelength(frequency_hz, refractive_index).to_dict()

    @server.tool()
    def em_wave_frequency(wavelength_m: float,
                            refractive_index: float = 1.0) -> dict[str, Any]:
        """f = c / (n lambda)."""
        return t_circ.em_wave_frequency(wavelength_m, refractive_index).to_dict()

    # ── atomic physics ─────────────────────────────────────────────
    # (element_atomic_data is registered in the materials section above
    # using periodictable lib; it covers Z + atomic mass for all elements.)

    @server.tool()
    def first_ionization_energy(element_symbol: str) -> dict[str, Any]:
        """First IE in eV (e.g. H=13.6, Na=5.1)."""
        return t_atom.first_ionization_energy(element_symbol).to_dict()

    @server.tool()
    def hydrogen_like_energy_level(n: int,
                                     atomic_number: int = 1) -> dict[str, Any]:
        """E_n = -13.606 Z^2 / n^2 eV."""
        return t_atom.hydrogen_like_energy_level(n, atomic_number).to_dict()

    @server.tool()
    def hydrogen_emission_wavelength(n_initial: int, n_final: int,
                                       atomic_number: int = 1) -> dict[str, Any]:
        """Rydberg wavelength for hydrogen-like transitions."""
        return t_atom.hydrogen_emission_wavelength(n_initial, n_final,
                                                      atomic_number).to_dict()

    @server.tool()
    def de_broglie_wavelength(mass_kg: float,
                                velocity_m_s: float) -> dict[str, Any]:
        """lambda = h / (m v)."""
        return t_atom.de_broglie_wavelength(mass_kg, velocity_m_s).to_dict()

    @server.tool()
    def photon_energy_from_wavelength(wavelength_m: float) -> dict[str, Any]:
        """E = h c / lambda."""
        return t_atom.photon_energy_from_wavelength(wavelength_m).to_dict()

    @server.tool()
    def photon_energy_from_frequency(frequency_hz: float) -> dict[str, Any]:
        """E = h f."""
        return t_atom.photon_energy_from_frequency(frequency_hz).to_dict()

    # ── astronomy ──────────────────────────────────────────────────
    @server.tool()
    def solar_system_body(body_name: str) -> dict[str, Any]:
        """Look up planet/moon/sun parameters (e.g. 'earth', 'mars')."""
        return t_astr.solar_system_body(body_name).to_dict()

    @server.tool()
    def named_star(star_name: str) -> dict[str, Any]:
        """Named bright star data (e.g. 'sirius_a', 'vega')."""
        return t_astr.named_star(star_name).to_dict()

    @server.tool()
    def light_travel_time(distance_m: float) -> dict[str, Any]:
        """t = d / c."""
        return t_astr.light_travel_time(distance_m).to_dict()

    @server.tool()
    def list_bodies() -> dict[str, Any]:
        """List all solar-system bodies and named stars available."""
        return t_astr.list_bodies().to_dict()

    # ── orbital mechanics (body-aware, multi-step) ─────────────────
    @server.tool()
    def orbital_velocity(central_body: str | None = None,
                           central_mass_kg: float | None = None,
                           altitude_km: float | None = None,
                           orbital_radius_m: float | None = None,
                           semimajor_axis_au: float | None = None
                           ) -> dict[str, Any]:
        """Circular orbital speed, body-aware. Give central_body (name) OR
        central_mass_kg, plus altitude_km (above surface) OR orbital_radius_m
        (from center) OR semimajor_axis_au. E.g. orbital_velocity('earth',
        altitude_km=408) for the ISS; orbital_velocity('sun',
        semimajor_axis_au=5.2) for Jupiter."""
        return t_orb.orbital_velocity(central_body, central_mass_kg,
                                        altitude_km, orbital_radius_m,
                                        semimajor_axis_au).to_dict()

    @server.tool()
    def orbital_period(semimajor_axis_au: float | None = None,
                         semimajor_axis_m: float | None = None,
                         central_body: str | None = "sun",
                         central_mass_kg: float | None = None
                         ) -> dict[str, Any]:
        """Kepler's third law T = 2π sqrt(a³/GM). Defaults to Sun; an
        asteroid at 3 AU is orbital_period(semimajor_axis_au=3). Returns
        seconds."""
        return t_orb.orbital_period(semimajor_axis_au, semimajor_axis_m,
                                      central_body, central_mass_kg).to_dict()

    @server.tool()
    def gravitational_force(mass1_kg: float, mass2_kg: float,
                              separation_m: float) -> dict[str, Any]:
        """Newton's law F = G m1 m2 / r²."""
        return t_orb.gravitational_force(mass1_kg, mass2_kg,
                                           separation_m).to_dict()

    @server.tool()
    def orbital_raise_energy(mass_kg: float,
                               central_body: str | None = "earth",
                               central_mass_kg: float | None = None,
                               from_altitude_km: float | None = 0.0,
                               to_altitude_km: float | None = None,
                               from_radius_m: float | None = None,
                               to_radius_m: float | None = None
                               ) -> dict[str, Any]:
        """Gravitational PE to raise a mass between two orbits:
        ΔU = G M m (1/r1 − 1/r2). Body-aware (altitudes above surface)."""
        return t_orb.orbital_raise_energy(mass_kg, central_body,
                                            central_mass_kg, from_altitude_km,
                                            to_altitude_km, from_radius_m,
                                            to_radius_m).to_dict()

    # ── nuclear physics ────────────────────────────────────────────
    @server.tool()
    def nuclear_binding_energy(protons: int, neutrons: int,
                                 measured_mass_u: float | None = None
                                 ) -> dict[str, Any]:
        """Nuclear binding energy + mass defect. Exact with measured_mass_u
        (atomic mass units), else SEMF estimate. Returns binding_energy_MeV,
        binding_per_nucleon_MeV, mass_defect_u, mass_defect_fraction —
        the last is how much lighter the bound nucleus is than its free
        baryons (peaks ~0.9% near iron)."""
        return t_nuc.nuclear_binding_energy(protons, neutrons,
                                              measured_mass_u).to_dict()

    @server.tool()
    def coulomb_force(charge1_c: float, charge2_c: float,
                        separation_m: float) -> dict[str, Any]:
        """Coulomb's law F = q1 q2 / (4π ε0 r²). +repulsive / −attractive."""
        return t_nuc.coulomb_force(charge1_c, charge2_c,
                                     separation_m).to_dict()

    # ── extra atomic / circuits multi-step ─────────────────────────
    @server.tool()
    def de_broglie_from_kinetic_energy(kinetic_energy_eV: float,
                                         particle: str = "electron"
                                         ) -> dict[str, Any]:
        """de Broglie wavelength from KINETIC ENERGY (relativistically exact).
        de_broglie_from_kinetic_energy(1000, 'electron') for a 1 keV electron.
        Particles: electron, proton, neutron, alpha, muon."""
        return t_atom.de_broglie_from_kinetic_energy(kinetic_energy_eV,
                                                       particle).to_dict()

    @server.tool()
    def energy_power_time(power_w: float | None = None,
                            time_s: float | None = None,
                            energy_j: float | None = None) -> dict[str, Any]:
        """Solve E = P·t for the missing one. Provide exactly two.
        energy_power_time(power_w=5000, time_s=3600) -> energy in joules."""
        return t_circ.energy_power_time(power_w, time_s, energy_j).to_dict()

    # ── procedures: canonical multi-step scientific routines ───────
    @server.tool()
    def procedure_black_hole_profile(mass_kg: float) -> dict[str, Any]:
        """Full black-hole thermodynamics cascade: Schwarzschild radius →
        Hawking temperature → Bekenstein-Hawking entropy → evaporation time.
        One mass in, the whole profile out, in the only correct order."""
        return t_proc.black_hole_profile(mass_kg).to_dict()

    @server.tool()
    def procedure_photon_spectrum(wavelength_m: float) -> dict[str, Any]:
        """Full photon cascade from one wavelength: frequency → energy (J) →
        energy (eV) → momentum. Each value derived from the previous, in code."""
        return t_proc.photon_spectrum(wavelength_m).to_dict()

    @server.tool()
    def procedure_relativistic_particle(kinetic_energy_eV: float,
                                          particle: str = "electron"
                                          ) -> dict[str, Any]:
        """Relativistic-kinematics cascade: KE → Lorentz factor γ → total
        energy → momentum → de Broglie wavelength. Particles: electron,
        proton, neutron, muon."""
        return t_proc.relativistic_particle(kinetic_energy_eV,
                                             particle).to_dict()

    @server.tool()
    def procedure_projectile_trajectory(initial_speed_m_s: float,
                                          launch_angle_deg: float
                                          ) -> dict[str, Any]:
        """Intro-mechanics cascade (no air resistance): time of flight →
        range → max height. One launch, three outputs."""
        return t_proc.projectile_trajectory(initial_speed_m_s,
                                            launch_angle_deg).to_dict()

    @server.tool()
    def procedure_stellar_blackbody(temperature_k: float) -> dict[str, Any]:
        """Blackbody/star-surface cascade: Wien peak wavelength →
        Stefan-Boltzmann surface flux → peak photon energy."""
        return t_proc.stellar_blackbody(temperature_k).to_dict()

    # ── simulation: the Materia time-evolution engine (NL front door) ──
    @server.tool()
    def simulate(scenario: str) -> dict[str, Any]:
        """Run a natural-language physics what-if through the Materia
        simulator. Returns a worked, self-validated answer, or a clarification
        (value null) if the scenario is not modeled — never a fabricated
        number. e.g. "how fast does a 5 cm copper ball hit the ground from
        10 km?"."""
        return t_sim.simulate(scenario).to_dict()

    @server.tool()
    def run_simulation(verb: str,
                       params: dict[str, Any] | None = None
                       ) -> dict[str, Any]:
        """Run one Materia simulation verb directly with explicit parameters
        (deterministic, no LLM). Use list_simulation_scenarios() to see the
        verbs and their input slots."""
        return t_sim.run_simulation(verb, params).to_dict()

    @server.tool()
    def list_simulation_scenarios() -> dict[str, Any]:
        """List the Materia simulation verbs with their inputs and named
        outputs — discovery for the simulator."""
        return t_sim.list_simulation_scenarios().to_dict()

    # ── chemistry: bonds, reactions, acids/bases, electrochem, solutions ──
    @server.tool()
    def bond_energy(atom_a: str, atom_b: str) -> dict[str, Any]:
        """Bond dissociation energy (kJ/mol) for a diatomic bond, e.g.
        bond_energy('H', 'H') or bond_energy('C', 'O')."""
        return t_chem.bond_energy(atom_a, atom_b)

    @server.tool()
    def bond_angle(electron_domains: int,
                   lone_pairs: int = 0) -> dict[str, Any]:
        """VSEPR bond angle (degrees) from steric number, e.g. bond_angle(4, 0)
        → 109.5° (tetrahedral), bond_angle(4, 2) → ~104.5° (water)."""
        return t_chem.bond_angle(electron_domains, lone_pairs)

    @server.tool()
    def reaction_enthalpy(reaction_key: str) -> dict[str, Any]:
        """Standard reaction enthalpy ΔH° (kJ/mol) from formation enthalpies,
        e.g. reaction_enthalpy('methane_combustion')."""
        return t_chem.reaction_enthalpy(reaction_key)

    @server.tool()
    def weak_acid_ph(acid_key: str,
                     concentration_mol_l: float) -> dict[str, Any]:
        """pH of a weak-acid solution from its Ka, e.g.
        weak_acid_ph('acetic_acid', 0.1) → ~2.88."""
        return t_chem.weak_acid_ph(acid_key, concentration_mol_l)

    @server.tool()
    def buffer_ph(acid_key: str,
                  ratio_base_over_acid: float = 1.0) -> dict[str, Any]:
        """Buffer pH (Henderson-Hasselbalch): pH = pKa + log([base]/[acid])."""
        return t_chem.buffer_ph(acid_key, ratio_base_over_acid)

    @server.tool()
    def cell_potential(cathode: str, anode: str,
                       reaction_quotient: float = 1.0) -> dict[str, Any]:
        """Galvanic cell EMF (V), Nernst-corrected, e.g.
        cell_potential('copper', 'zinc') → ~1.10 V (Daniell cell)."""
        return t_chem.cell_potential(cathode, anode, reaction_quotient)

    @server.tool()
    def electrolysis_mass(molar_mass_kg: float, current_a: float,
                          time_s: float, electrons: int) -> dict[str, Any]:
        """Mass deposited by electrolysis (kg) — Faraday's first law:
        m = M·I·t / (n·F)."""
        return t_chem.electrolysis_mass(molar_mass_kg, current_a,
                                        time_s, electrons)

    @server.tool()
    def boiling_point_elevation(molality_mol_kg: float,
                                van_t_hoff_i: float = 1.0) -> dict[str, Any]:
        """Boiling-point elevation ΔTb (K) = i·Kb·molality (colligative)."""
        return t_chem.boiling_point_elevation(molality_mol_kg, van_t_hoff_i)

    @server.tool()
    def freezing_point_depression(molality_mol_kg: float,
                                  van_t_hoff_i: float = 1.0
                                  ) -> dict[str, Any]:
        """Freezing-point depression ΔTf (K) = i·Kf·molality (colligative)."""
        return t_chem.freezing_point_depression(molality_mol_kg, van_t_hoff_i)

    @server.tool()
    def osmotic_pressure(molarity_mol_l: float,
                         van_t_hoff_i: float = 1.0) -> dict[str, Any]:
        """Osmotic pressure π (Pa) = i·M·R·T (van't Hoff)."""
        return t_chem.osmotic_pressure(molarity_mol_l, van_t_hoff_i)

    @server.tool()
    def molar_solubility(salt_key: str) -> dict[str, Any]:
        """Molar solubility (mol/L) of a sparingly-soluble salt from its Ksp,
        e.g. molar_solubility('silver_chloride') → ~1.33e-5."""
        return t_chem.molar_solubility(salt_key)

    # ── electronics: transport, semiconductors, junctions, capacitance ──
    @server.tool()
    def electrical_resistivity(metal_key: str,
                               temperature_k: float = 300.0
                               ) -> dict[str, Any]:
        """Electrical resistivity (Ω·m) of a metal, e.g.
        electrical_resistivity('copper') → ~1.68e-8."""
        return t_elec.electrical_resistivity(metal_key, temperature_k)

    @server.tool()
    def carrier_mobility(metal_key: str,
                         temperature_k: float = 300.0) -> dict[str, Any]:
        """Electron drift mobility (m²/V·s) of a metal (metals only)."""
        return t_elec.carrier_mobility(metal_key, temperature_k)

    @server.tool()
    def hall_coefficient(metal_key: str) -> dict[str, Any]:
        """Hall coefficient (m³/C) of a metal: R_H = −1/(n·e)."""
        return t_elec.hall_coefficient(metal_key)

    @server.tool()
    def electron_mean_free_path(metal_key: str,
                                temperature_k: float = 300.0
                                ) -> dict[str, Any]:
        """Electron mean free path (m) in a metal, e.g.
        electron_mean_free_path('copper') → ~39 nm."""
        return t_elec.electron_mean_free_path(metal_key, temperature_k)

    @server.tool()
    def free_electron_density(metal_key: str) -> dict[str, Any]:
        """Free-electron (conduction) number density (m⁻³) of a metal."""
        return t_elec.free_electron_density(metal_key)

    @server.tool()
    def semiconductor_band_gap(semiconductor_key: str,
                               temperature_k: float = 300.0
                               ) -> dict[str, Any]:
        """Temperature-dependent band gap (eV) of a semiconductor (Varshni),
        e.g. semiconductor_band_gap('silicon') → ~1.12 eV at 300 K."""
        return t_elec.semiconductor_band_gap(semiconductor_key, temperature_k)

    @server.tool()
    def intrinsic_carrier_density(semiconductor_key: str,
                                  temperature_k: float = 300.0
                                  ) -> dict[str, Any]:
        """Intrinsic carrier density n_i (m⁻³) of a semiconductor."""
        return t_elec.intrinsic_carrier_density(semiconductor_key,
                                                temperature_k)

    @server.tool()
    def pn_built_in_voltage(semiconductor_key: str, donor_density_m3: float,
                            acceptor_density_m3: float,
                            temperature_k: float = 300.0) -> dict[str, Any]:
        """Built-in voltage of a p-n junction (V):
        V_bi = (kT/e) ln(N_A N_D / n_i²)."""
        return t_elec.pn_built_in_voltage(semiconductor_key, donor_density_m3,
                                          acceptor_density_m3, temperature_k)

    @server.tool()
    def depletion_width(semiconductor_key: str, donor_density_m3: float,
                        acceptor_density_m3: float,
                        applied_voltage_v: float = 0.0,
                        temperature_k: float = 300.0) -> dict[str, Any]:
        """Depletion-region width of a p-n junction (m)."""
        return t_elec.depletion_width(semiconductor_key, donor_density_m3,
                                      acceptor_density_m3, applied_voltage_v,
                                      temperature_k)

    @server.tool()
    def diode_current(saturation_current_a: float, voltage_v: float,
                      temperature_k: float = 300.0) -> dict[str, Any]:
        """Shockley diode current (A): I = I₀(exp(eV/kT) − 1)."""
        return t_elec.diode_current(saturation_current_a, voltage_v,
                                    temperature_k)

    # (parallel_plate_capacitance is already exposed by the circuits group;
    #  electronics.parallel_plate_capacitance is the same physics — not
    #  re-registered here to avoid a duplicate tool name.)

    # ── extended math (sympy): linear algebra, transforms, calculus ──
    @server.tool()
    def matrix_determinant(matrix: str) -> dict[str, Any]:
        """Determinant of a square matrix. matrix='[[1,2],[3,4]]' -> -2."""
        return t_mathx.matrix_determinant(matrix).to_dict()

    @server.tool()
    def matrix_eigenvalues(matrix: str) -> dict[str, Any]:
        """Eigenvalues of a square matrix (with multiplicity)."""
        return t_mathx.matrix_eigenvalues(matrix).to_dict()

    @server.tool()
    def matrix_inverse(matrix: str) -> dict[str, Any]:
        """Inverse of a square matrix. matrix='[[1,2],[3,4]]'."""
        return t_mathx.matrix_inverse(matrix).to_dict()

    @server.tool()
    def matrix_multiply(matrix_a: str, matrix_b: str) -> dict[str, Any]:
        """Matrix product A·B (nested-list strings)."""
        return t_mathx.matrix_multiply(matrix_a, matrix_b).to_dict()

    @server.tool()
    def solve_linear_system(matrix_a: str, vector_b: str) -> dict[str, Any]:
        """Solve A x = b. matrix_a='[[2,1],[1,3]]', vector_b='[1,2]'."""
        return t_mathx.solve_linear_system(matrix_a, vector_b).to_dict()

    @server.tool()
    def compute_limit(expression: str, variable: str = "x",
                      point: str = "0", direction: str = "+-"
                      ) -> dict[str, Any]:
        """Limit of expression as variable -> point. direction '+','-','+-'.
        compute_limit('sin(x)/x') -> 1."""
        return t_mathx.compute_limit(expression, variable, point,
                                     direction).to_dict()

    @server.tool()
    def series_expansion(expression: str, variable: str = "x",
                         point: str = "0", order: int = 6) -> dict[str, Any]:
        """Taylor/Maclaurin series of expression about point, to given order."""
        return t_mathx.series_expansion(expression, variable, point,
                                        order).to_dict()

    @server.tool()
    def summation(expression: str, variable: str = "n",
                  lower: str = "1", upper: str = "oo") -> dict[str, Any]:
        """Symbolic sum over variable from lower to upper ('oo' = infinity).
        summation('1/n**2', upper='oo') -> pi**2/6."""
        return t_mathx.summation(expression, variable, lower, upper).to_dict()

    @server.tool()
    def laplace_transform(expression: str, t_var: str = "t",
                          s_var: str = "s") -> dict[str, Any]:
        """Laplace transform F(s) = L{f(t)}. laplace_transform('exp(-2*t)')
        -> 1/(s+2)."""
        return t_mathx.laplace_transform(expression, t_var, s_var).to_dict()

    @server.tool()
    def fourier_transform(expression: str, x_var: str = "x",
                          k_var: str = "k") -> dict[str, Any]:
        """Fourier transform of expression."""
        return t_mathx.fourier_transform(expression, x_var, k_var).to_dict()

    @server.tool()
    def factor_expression(expression: str) -> dict[str, Any]:
        """Factor a polynomial / expression. factor_expression('x**2-1')
        -> (x-1)(x+1)."""
        return t_mathx.factor_expression(expression).to_dict()

    @server.tool()
    def expand_expression(expression: str) -> dict[str, Any]:
        """Expand an expression. expand_expression('(x+1)**2') -> x**2+2*x+1."""
        return t_mathx.expand_expression(expression).to_dict()

    @server.tool()
    def solve_ode(equation: str, func: str = "y",
                  variable: str = "x") -> dict[str, Any]:
        """Solve an ODE; use y, y', y'' for the function and derivatives.
        solve_ode("y'' + y") -> C1*sin(x)+C2*cos(x)."""
        return t_mathx.solve_ode(equation, func, variable).to_dict()

    @server.tool()
    def gradient(scalar_field: str,
                 variables: str = "x,y,z") -> dict[str, Any]:
        """Gradient ∇f of a scalar field. variables comma-separated."""
        return t_mathx.gradient(scalar_field, variables).to_dict()

    @server.tool()
    def divergence(vector_field: str,
                   variables: str = "x,y,z") -> dict[str, Any]:
        """Divergence ∇·F. vector_field comma-separated components."""
        return t_mathx.divergence(vector_field, variables).to_dict()

    @server.tool()
    def curl(vector_field: str, variables: str = "x,y,z") -> dict[str, Any]:
        """Curl ∇×F for a 3-component field. curl('-y,x,0') -> [0,0,2]."""
        return t_mathx.curl(vector_field, variables).to_dict()

    @server.tool()
    def percent_of(percent: float, value: float) -> dict[str, Any]:
        """X percent of a value: (percent/100)*value. percent_of(2,60) -> 1.2."""
        return t_mathx.percent_of(percent, value).to_dict()

    # ── frontier physics (standard): BH thermo, Unruh, entanglement ──
    @server.tool()
    def bekenstein_hawking_entropy(mass_kg: float) -> dict[str, Any]:
        """Black-hole entropy, temperature, horizon area & Schwarzschild radius
        from mass. Returns {entropy_k_B, horizon_area_m2,
        schwarzschild_radius_m, hawking_temperature_K}; solar mass ~1e77 k_B."""
        return t_front.bekenstein_hawking_entropy(mass_kg).to_dict()

    @server.tool()
    def gravitational_binding_energy(mass_kg: float,
                                     radius_m: float) -> dict[str, Any]:
        """Self-gravity binding energy of a uniform sphere: U = (3/5) G M²/R.
        Earth ≈ 2.24e32 J."""
        return t_front.gravitational_binding_energy(mass_kg,
                                                    radius_m).to_dict()

    @server.tool()
    def unruh_temperature(acceleration_m_s2: float) -> dict[str, Any]:
        """Unruh temperature seen by an accelerated observer:
        T = ħ a / (2π c k_B)."""
        return t_front.unruh_temperature(acceleration_m_s2).to_dict()

    @server.tool()
    def entanglement_channel(scenario: str = "") -> dict[str, Any]:
        """What an entangled pair can / cannot do as a channel. Answers FTL
        signaling (NO — no-communication theorem), QKD (a shared secret key),
        and the CHSH / Tsirelson bound (2√2). Pass the user's question as
        scenario so the verdict is tailored."""
        return t_front.entanglement_channel(scenario).to_dict()

    # ── mechanics: collisions, work/energy, projectile, incline ──
    @server.tool()
    def collision_analysis(mass1_kg: float, velocity1_m_s: float,
                           mass2_kg: float, velocity2_m_s: float,
                           restitution: float = 0.8) -> dict[str, Any]:
        """1D two-body collision: elastic final velocities, the inelastic
        outcome, and the kinetic energy lost. collision_analysis(2, 3, 1, -1)."""
        return t_mech.collision_analysis(mass1_kg, velocity1_m_s, mass2_kg,
                                         velocity2_m_s, restitution)

    @server.tool()
    def work_energy_analysis(mass_kg: float, velocity_m_s: float,
                             height_m: float = 10.0, force_n: float = 20.0,
                             distance_m: float = 5.0) -> dict[str, Any]:
        """Work, mechanical power, friction loss, gravitational PE, rotational
        KE, impulse, and total mechanical energy of a moving body."""
        return t_mech.work_energy_analysis(mass_kg, velocity_m_s, height_m,
                                           force_n, distance_m)

    @server.tool()
    def projectile_analysis(speed_m_s: float, angle_deg: float,
                            mass_kg: float = 1.0,
                            drag_coefficient: float = 0.47,
                            area_m2: float = 0.01) -> dict[str, Any]:
        """Projectile from a launch speed + angle (degrees): vacuum range, apex
        height, flight time, drag force, terminal velocity, drag-corrected
        range. projectile_analysis(50, 45)."""
        return t_mech.projectile_analysis(speed_m_s, angle_deg, mass_kg,
                                          drag_coefficient, area_m2)

    @server.tool()
    def incline_analysis(angle_deg: float, friction_coefficient: float = 0.3,
                         speed_m_s: float = 10.0,
                         height_m: float = 5.0) -> dict[str, Any]:
        """Inclined plane: critical sliding angle, distance slid up before
        stopping, and slide speed at the bottom."""
        return t_mech.incline_analysis(angle_deg, friction_coefficient,
                                       speed_m_s, height_m)

    # ── transport & statistical mechanics ──
    @server.tool()
    def viscous_flow_analysis(velocity_m_s: float, radius_m: float,
                              viscosity_pa_s: float = 1.0e-3,
                              fluid_density_kg_m3: float = 1000.0) -> dict[str, Any]:
        """Viscous flow: Reynolds number, Stokes drag + terminal velocity, drag
        coefficient, Poiseuille pipe flow, boundary layer, wall shear, viscous
        heating. viscous_flow_analysis(2, 0.005)."""
        return t_trans.viscous_flow_analysis(velocity_m_s, radius_m,
                                             viscosity_pa_s, fluid_density_kg_m3)

    @server.tool()
    def diffusion_analysis(temperature_k: float,
                           diffusivity_m2_s: float = 1.0e-9) -> dict[str, Any]:
        """Diffusion: Einstein-Stokes diffusivity, Fick's first & second laws,
        penetration time, Darken interdiffusion."""
        return t_trans.diffusion_analysis(temperature_k, diffusivity_m2_s)

    @server.tool()
    def statistical_distribution(temperature_k: float,
                                 energy_ev: float = 0.5) -> dict[str, Any]:
        """Statistical mechanics: Fermi-Dirac & Bose-Einstein occupation,
        partition function, mean energy, entropy, equipartition heat capacity.
        statistical_distribution(300, 0.5)."""
        return t_trans.statistical_distribution(temperature_k, energy_ev)

    # ── rotational dynamics + atomic angular momentum ──
    @server.tool()
    def rotational_dynamics(mass_kg: float, radius_m: float,
                            angle_deg: float = 30.0,
                            angular_velocity_rad_s: float = 10.0) -> dict[str, Any]:
        """Rotational dynamics: moment of inertia (rod + shape geometry),
        parallel-axis, angular momentum, torque, angular acceleration, and
        rolling down a ramp (speed/distance/time). rotational_dynamics(2, 0.5)."""
        return t_rot.rotational_dynamics(mass_kg, radius_m, angle_deg,
                                         angular_velocity_rad_s)

    @server.tool()
    def atomic_angular_momentum(total_j: float = 1.5,
                                spin_orbit_constant_ev: float = 0.05) -> dict[str, Any]:
        """Atomic angular momentum: |J| magnitude, allowed m_j values, spin-orbit
        coupling energy/splitting, and the Lande interval (L=2, S=1/2)."""
        return t_rot.atomic_angular_momentum(total_j, spin_orbit_constant_ev)

    # ── materials strength + composites ──
    @server.tool()
    def elastic_analysis(material_key: str, strain: float = 0.001) -> dict[str, Any]:
        """Elastic response: uniaxial/shear/hydrostatic stress, strain-energy
        densities, transverse strain, volume change, von Mises yield, and
        moduli from Lame parameters. elastic_analysis('iron', 0.001)."""
        return t_matstr.elastic_analysis(material_key, strain)

    @server.tool()
    def stress_failure_analysis(material_key: str,
                                applied_stress: float = 1.0e8) -> dict[str, Any]:
        """Fracture/fatigue/creep: stress-intensity factor, critical crack
        length, fatigue life, Paris remaining life, creep rate + rupture time."""
        return t_matstr.stress_failure_analysis(material_key, applied_stress)

    @server.tool()
    def plasticity_analysis(material_key: str,
                            plastic_strain: float = 0.05) -> dict[str, Any]:
        """Plastic flow stress: Johnson-Cook, Ludwik hardening, and the
        work-hardening rate. plasticity_analysis('aluminum', 0.05)."""
        return t_matstr.plasticity_analysis(material_key, plastic_strain)

    @server.tool()
    def composite_bounds_analysis(bulk_modulus1_pa: float = 100.0e9,
                                  bulk_modulus2_pa: float = 200.0e9,
                                  fraction1: float = 0.5) -> dict[str, Any]:
        """Two-phase composite property bounds: Voigt-Reuss-Hill average,
        Hashin-Shtrikman bounds, thermal-conductivity bounds, Gibson-Ashby
        foam strength."""
        return t_matstr.composite_bounds_analysis(bulk_modulus1_pa,
                                                  bulk_modulus2_pa, fraction1)

    # Run via stdio transport (standard MCP).
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
