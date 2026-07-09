"""Offline end-to-end tests for the MCP tool functions.

These exercise each tool directly (no MCP transport, no LLM) so we can
verify the wrappers behave correctly and the ToolResult fields are
populated. The MCP server.py just thinly wraps these.

Skipped automatically if the [mcp] optional deps aren't installed.
"""

from __future__ import annotations

import math

import pytest

# Skip the whole module if optional deps aren't available.
pytest.importorskip("scipy", reason="install with: pip install sigma-ground[mcp]")
pytest.importorskip("pint")
pytest.importorskip("sympy")

from sigma_ground.mcp.provenance import ToolResult
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
from sigma_ground.mcp import manifest as t_manifest


# ── provenance helpers ─────────────────────────────────────────────────

def _is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


# ── constants ──────────────────────────────────────────────────────────

class TestConstantsLookup:

    def test_g_from_sigma_ground(self):
        r = t_const.lookup_constant("G")
        assert _is_finite_number(r.value)
        assert abs(r.value - 6.6743e-11) < 1e-13
        assert "sigma_ground" in r.source
        assert r.units == "m^3 / (kg s^2)"

    def test_speed_of_light_from_codata(self):
        # 'speed_of_light' is the canonical scipy name; sigma_ground has 'C'.
        r = t_const.lookup_constant("speed_of_light")
        assert _is_finite_number(r.value)
        assert abs(r.value - 299792458.0) < 1.0

    def test_planck_constant(self):
        r = t_const.lookup_constant("HBAR")
        assert _is_finite_number(r.value)
        assert abs(r.value - 1.054571817e-34) < 1e-44

    def test_eta_has_empirical_input_tag(self):
        r = t_const.lookup_constant("ETA")
        assert _is_finite_number(r.value)
        # After 2026-05-15 ETA is anchored at DESI Union3 c^2 ~ 0.4122
        assert abs(r.value - 0.412164) < 1e-5
        # Tag should be EMPIRICAL-INPUT after the rework
        assert r.provenance_tag == "EMPIRICAL-INPUT"

    def test_unknown_constant_returns_not_found(self):
        r = t_const.lookup_constant("nonexistent_quantity_xyz")
        assert r.value is None
        assert "not found" in r.source.lower()

    def test_list_constants_finds_sigma_ground(self):
        r = t_const.list_constants(category="sigma_ground", limit=200)
        assert isinstance(r.value, list)
        assert any("ETA" in n for n in r.value)

    def test_list_constants_contains_filter(self):
        r = t_const.list_constants(contains="planck", limit=20)
        assert isinstance(r.value, list)
        assert all("planck" in n.lower() for n in r.value)


# ── units ──────────────────────────────────────────────────────────────

class TestUnitConversion:

    def test_ev_to_joule(self):
        r = t_units.convert(1.0, "eV", "joule")
        assert _is_finite_number(r.value)
        assert abs(r.value - 1.602176634e-19) < 1e-25

    def test_solar_mass_to_kg(self):
        r = t_units.convert(1.0, "solar_mass", "kg")
        assert _is_finite_number(r.value)
        # 1 solar mass ~ 1.989e30 kg
        assert 1.98e30 < r.value < 1.99e30

    def test_dimensional_mismatch_returns_error(self):
        r = t_units.convert(5.0, "kg", "second")
        assert r.value is None
        assert "dimensional" in r.source.lower() or "dimensional" in r.notes.lower()

    def test_parse_quantity(self):
        r = t_units.parse_quantity("5.6 light_year")
        assert _is_finite_number(r.value)
        assert abs(r.value - 5.6) < 1e-12
        assert "light_year" in r.units


# ── symbolic ──────────────────────────────────────────────────────────

class TestSymbolicMath:

    def test_solve_quadratic(self):
        r = t_sym.solve_equation("x**2 - 4", "x")
        assert isinstance(r.value, list)
        sols = sorted(r.value)
        assert sols == ["-2", "2"]

    def test_solve_lhs_eq_rhs_form(self):
        r = t_sym.solve_equation("x + 3 = 7", "x")
        assert "4" in r.value

    def test_integrate_indefinite(self):
        r = t_sym.integrate_expr("x**2", "x")
        assert "x**3/3" in r.value.replace(" ", "")

    def test_integrate_definite(self):
        r = t_sym.integrate_expr("x", "x", "0", "1")
        # integral_0^1 x dx = 1/2
        assert r.value in ("1/2", "0.5")

    def test_differentiate(self):
        r = t_sym.differentiate_expr("sin(x)", "x")
        assert "cos(x)" in r.value

    def test_simplify(self):
        r = t_sym.simplify_expr("sin(x)**2 + cos(x)**2")
        assert r.value == "1"


# ── GR ─────────────────────────────────────────────────────────────────

class TestGeneralRelativity:

    M_SUN = 1.989e30

    def test_schwarzschild_radius_solar_mass(self):
        r = t_gr.schwarzschild_radius(self.M_SUN)
        # r_s(M_sun) ~ 2.95 km
        assert _is_finite_number(r.value)
        assert 2940 < r.value < 2960

    def test_schwarzschild_invalid_mass_returns_none(self):
        r = t_gr.schwarzschild_radius(-1.0)
        assert r.value is None

    def test_isco_is_three_schwarzschild_radii(self):
        rs = t_gr.schwarzschild_radius(self.M_SUN).value
        isco = t_gr.isco_radius(self.M_SUN).value
        assert abs(isco / rs - 3.0) < 1e-10

    def test_photon_sphere_is_1_5_schwarzschild_radii(self):
        rs = t_gr.schwarzschild_radius(self.M_SUN).value
        ph = t_gr.photon_sphere_radius(self.M_SUN).value
        assert abs(ph / rs - 1.5) < 1e-10

    def test_hawking_temperature_solar_mass(self):
        # T_H(M_sun) ~ 6.17e-8 K
        r = t_gr.hawking_temperature(self.M_SUN)
        assert _is_finite_number(r.value)
        assert 5e-8 < r.value < 7e-8

    def test_hawking_evaporation_solar_mass(self):
        # tau(M_sun) ~ 2.1e67 years = 6.6e74 s
        r = t_gr.hawking_evaporation_time(self.M_SUN)
        assert _is_finite_number(r.value)
        assert 5e74 < r.value < 1e75

    def test_redshift_at_2rs_is_finite(self):
        rs = t_gr.schwarzschild_radius(self.M_SUN).value
        r = t_gr.gravitational_redshift(self.M_SUN, 2.0 * rs)
        # At r = 2 r_s: 1 - r_s/r = 0.5; (0.5)^(-1/2) = sqrt(2); z = sqrt(2)-1
        assert abs(r.value - (math.sqrt(2.0) - 1.0)) < 1e-10

    def test_redshift_at_horizon_diverges(self):
        rs = t_gr.schwarzschild_radius(self.M_SUN).value
        r = t_gr.gravitational_redshift(self.M_SUN, rs)
        assert r.value == float("inf")

    def test_time_dilation_at_2rs(self):
        rs = t_gr.schwarzschild_radius(self.M_SUN).value
        r = t_gr.gravitational_time_dilation(self.M_SUN, 2.0 * rs)
        # sqrt(1 - 0.5) = sqrt(0.5)
        assert abs(r.value - math.sqrt(0.5)) < 1e-10


# ── manifest ──────────────────────────────────────────────────────────

class TestManifest:

    def test_manifest_lists_primary_tools(self):
        r = t_manifest.get_manifest()
        assert isinstance(r.value, dict)
        primary = r.value["primary"]
        names = {t["name"] for t in primary}
        # Every tool registered in server.py should appear here.
        for expected in {
            "lookup_constant", "list_constants",
            "convert_units", "parse_quantity",
            "solve_equation", "integrate_expr", "differentiate_expr",
            "simplify_expr",
            "schwarzschild_radius", "isco_radius", "photon_sphere_radius",
            "hawking_temperature", "hawking_evaporation_time",
            "gravitational_redshift", "gravitational_time_dilation",
        }:
            assert expected in names, f"manifest missing {expected}"

    def test_manifest_includes_positioning_guidance(self):
        r = t_manifest.get_manifest()
        positioning = r.value["positioning"]
        assert "SSBM" in positioning
        assert "PRIMARY" in positioning
        # The positioning should explicitly say "do not volunteer SSBM"
        assert "Do NOT volunteer" in positioning or "do not volunteer" in positioning


# ── ToolResult dataclass ──────────────────────────────────────────────

class TestToolResult:

    def test_format_for_llm_includes_source(self):
        r = ToolResult(value=6.674e-11, units="m^3/(kg s^2)",
                        source="CODATA 2018",
                        provenance_tag="VERIFIED")
        text = r.format_for_llm()
        assert "CODATA 2018" in text
        assert "VERIFIED" in text

    def test_to_dict_is_serializable(self):
        import json
        r = ToolResult(value=42.0, units="kg", source="test")
        d = r.to_dict()
        # Round-trip through JSON
        s = json.dumps(d)
        assert "42" in s
        assert "kg" in s

    def test_format_for_llm_includes_library_attribution(self):
        """Every formatted result must carry the sigma-ground library tag."""
        r = ToolResult(value=1.0, units="m", source="test")
        text = r.format_for_llm()
        assert "sigma-ground" in text


# ── Kinematics ─────────────────────────────────────────────────────────

class TestKinematics:

    def test_free_fall_time_10m_earth(self):
        r = t_kin.free_fall_time(10.0)
        assert abs(r.value - 1.4278) < 1e-3
        assert r.units == "s"

    def test_free_fall_time_moon(self):
        r = t_kin.free_fall_time(10.0, g_m_s2=1.625)
        # t = sqrt(20/1.625) ~ 3.51 s
        assert 3.49 < r.value < 3.52

    def test_free_fall_velocity(self):
        r = t_kin.free_fall_velocity(10.0)
        # v = sqrt(2*9.80665*10) ~ 14.01 m/s
        assert 13.9 < r.value < 14.1

    def test_projectile_range_45deg(self):
        # At 45 deg, range is maximum: R = v^2/g
        r = t_kin.projectile_range(20.0, 45.0)
        # Expected: 400/9.80665 ~ 40.79 m
        assert 40.5 < r.value < 41.1

    def test_kinetic_energy(self):
        r = t_kin.kinetic_energy(2.0, 3.0)
        # KE = 0.5 * 2 * 9 = 9 J
        assert abs(r.value - 9.0) < 1e-9

    def test_momentum(self):
        r = t_kin.momentum(2.0, 3.0)
        assert abs(r.value - 6.0) < 1e-9

    def test_friction_stopping_distance(self):
        # v=1 m/s, mu=0.4, g=9.81 -> d ~ 1/(2*0.4*9.81) = 0.1274 m
        r = t_kin.friction_stopping_distance(0.2, 1.0, 0.4)
        assert abs(r.value - 0.12740) < 1e-3

    def test_circular_orbit_velocity_earth_leo(self):
        # 400 km LEO: r=6.771e6 m, M_earth=5.972e24 kg -> v ~ 7670 m/s
        r = t_kin.circular_orbit_velocity(5.972e24, 6.771e6)
        assert 7600 < r.value < 7700

    def test_escape_velocity_earth(self):
        # Surface: r=6.371e6 m -> v_esc ~ 11186 m/s
        r = t_kin.escape_velocity_classical(5.972e24, 6.371e6)
        assert 11100 < r.value < 11250

    def test_free_fall_negative_height_rejected(self):
        r = t_kin.free_fall_time(-5.0)
        assert r.value is None


# ── Energy conversion ──────────────────────────────────────────────────

class TestEnergyConversion:

    def test_mass_to_energy_1kg(self):
        r = t_econv.mass_to_energy(1.0)
        # E = c^2 ~ 8.988e16 J
        assert 8.987e16 < r.value < 8.989e16

    def test_solar_mass_conversion_rate(self):
        """The 'one millionth of a gram' failure case -- catch it here."""
        L_sun = 3.828e26
        r = t_econv.luminosity_to_mass_conversion_rate(L_sun)
        # Expect ~4.26e9 kg/s, NOT a millionth of a gram
        assert 4.0e9 < r.value < 4.5e9
        assert r.units == "kg/s"

    def test_joules_to_eV(self):
        r = t_econv.joules_to_eV(1.602176634e-19)
        assert abs(r.value - 1.0) < 1e-9

    def test_eV_to_joules(self):
        r = t_econv.eV_to_joules(1.0)
        assert abs(r.value - 1.602176634e-19) < 1e-25

    def test_joules_to_TNT_1MT(self):
        # 1 MT = 4.184e15 J
        r = t_econv.joules_to_TNT(4.184e15, unit="MT")
        assert abs(r.value - 1.0) < 1e-9


# ── Special relativity ─────────────────────────────────────────────────

class TestRelativity:

    def test_gamma_at_half_c(self):
        r = t_rel.lorentz_factor(0.5 * 2.998e8)
        # gamma at v=0.5c is 1/sqrt(0.75) = 1.1547
        assert 1.154 < r.value < 1.156

    def test_time_dilation_v_zero(self):
        r = t_rel.relativistic_time_dilation(1.0, 0.0)
        assert abs(r.value - 1.0) < 1e-9

    def test_length_contraction(self):
        r = t_rel.relativistic_length_contraction(1.0, 0.5 * 2.998e8)
        # L = L0 / gamma = 1 / 1.1547 ~ 0.866
        assert 0.865 < r.value < 0.867

    def test_velocity_addition_classical_limit(self):
        # Low velocities: should be approximately u+v
        r = t_rel.relativistic_velocity_addition(100.0, 200.0)
        assert 299.99 < r.value < 300.01

    def test_velocity_addition_high(self):
        # Both 0.9c, classical would give 1.8c; relativistic ~0.994c
        c = 2.998e8
        r = t_rel.relativistic_velocity_addition(0.9 * c, 0.9 * c)
        assert r.value < c

    def test_superluminal_rejected(self):
        r = t_rel.lorentz_factor(4e8)
        assert r.value is None


# ── Cosmology ──────────────────────────────────────────────────────────

class TestCosmologyTools:

    def test_hubble_radius_positive(self):
        r = t_cos.hubble_radius()
        # ~10^26 m
        assert 1e26 < r.value < 1.5e26

    def test_eta_value(self):
        r = t_cos.eta_value_report()
        # ETA = 0.412164 (DESI Union3 anchor)
        assert abs(r.value - 0.412164) < 1e-5
        assert r.provenance_tag == "EMPIRICAL-INPUT"

    def test_mond_regime_strong(self):
        r = t_cos.mond_regime_classifier(9.8)
        assert r.value == "newtonian"

    def test_mond_regime_weak(self):
        r = t_cos.mond_regime_classifier(1e-12)
        assert r.value == "mond"


# ── Thermodynamics ─────────────────────────────────────────────────────

class TestThermodynamics:

    def test_ideal_gas_pressure(self):
        # 1 mol gas at 273.15 K in 22.414 L should be ~101325 Pa
        r = t_thermo.ideal_gas_pressure(1.0, 273.15, 22.414e-3)
        assert 101000 < r.value < 101400

    def test_blackbody_peak_sun(self):
        r = t_thermo.blackbody_peak_wavelength(5778.0)
        # Wien: 2.898e-3 / 5778 ~ 502 nm
        assert 500e-9 < r.value < 504e-9

    def test_stefan_boltzmann_sun_surface(self):
        # T=5778 K, A=4 pi R_sun^2 ~ 6.087e18 m^2, emissivity=1
        # L_sun expected ~3.828e26 W
        r = t_thermo.blackbody_total_power(5778.0, 6.087e18, 1.0)
        assert 3.7e26 < r.value < 3.9e26

    def test_carnot_basic(self):
        # T_h=400, T_c=300 -> eta = 0.25
        r = t_thermo.carnot_efficiency(400.0, 300.0)
        assert abs(r.value - 0.25) < 1e-9

    def test_celsius_to_kelvin(self):
        r = t_thermo.temperature_celsius_to_kelvin(0.0)
        assert abs(r.value - 273.15) < 1e-9

    def test_sound_speed_air_20c(self):
        # Air at 293.15 K, gamma=1.4, M=0.029 -> ~343 m/s
        r = t_thermo.speed_of_sound_in_ideal_gas(293.15)
        assert 340 < r.value < 346


# ── Optics ─────────────────────────────────────────────────────────────

class TestOpticsTools:

    def test_snell_water_to_air(self):
        # n1=1.333 water, n2=1.0 air, theta1=30 -> theta2=41.7 deg
        r = t_opt.snells_law_refraction_angle(1.333, 1.0, 30.0)
        assert 41.6 < r.value < 41.8

    def test_critical_angle_water(self):
        r = t_opt.critical_angle_for_tir(1.333, 1.0)
        # arcsin(1/1.333) ~ 48.6 deg
        assert 48.5 < r.value < 48.8

    def test_rydberg_h_alpha(self):
        # H-alpha n=3->2 should be ~656.3 nm
        r = t_opt.rydberg_hydrogen_wavelength(3, 2)
        assert abs(r.value - 656.3e-9) / 656.3e-9 < 0.01

    def test_rydberg_lyman_alpha(self):
        # Ly-alpha n=2->1 should be 121.6 nm
        r = t_opt.rydberg_hydrogen_wavelength(2, 1)
        assert abs(r.value - 121.6e-9) / 121.6e-9 < 0.01

    def test_thin_lens_focal_length(self):
        # d_o = 2f, d_i = 2f gives f from 1/f = 1/(2f) + 1/(2f) = 1/f
        # Use d_o=20, d_i=20 -> f = 10
        r = t_opt.thin_lens_focal_length(20.0, 20.0)
        assert abs(r.value - 10.0) < 1e-9

    def test_single_slit_diffraction(self):
        # 500 nm light, 1 micron slit: sin theta = 0.5 -> theta = 30
        r = t_opt.single_slit_first_minimum_angle(500e-9, 1e-6)
        assert abs(r.value - 30.0) < 0.1


# ── Materials ──────────────────────────────────────────────────────────

class TestMaterials:

    def test_water_density(self):
        r = t_mat.density("water")
        assert 999 < r.value < 1001

    def test_copper_density(self):
        r = t_mat.density("copper")
        assert 8959 < r.value < 8961

    def test_diamond_refractive_index(self):
        r = t_mat.refractive_index("diamond")
        assert 2.41 < r.value < 2.43

    def test_water_melting_point(self):
        r = t_mat.melting_point("water")
        assert abs(r.value - 273.15) < 0.01

    def test_water_boiling_point(self):
        r = t_mat.boiling_point("water")
        assert abs(r.value - 373.15) < 0.01

    def test_silicon_band_gap(self):
        r = t_mat.band_gap_ev("silicon")
        assert 1.11 < r.value < 1.13

    def test_unknown_material_returns_none(self):
        r = t_mat.density("unobtainium")
        assert r.value is None

    def test_list_materials_includes_water(self):
        r = t_mat.list_materials()
        assert "water" in r.value["densities"]


# ── Circuits ───────────────────────────────────────────────────────────

class TestCircuits:

    def test_ohms_law_v(self):
        # I=2A, R=5 ohm -> V=10
        r = t_circ.ohms_law_voltage(2.0, 5.0)
        assert abs(r.value - 10.0) < 1e-9

    def test_ohms_law_i(self):
        r = t_circ.ohms_law_current(10.0, 5.0)
        assert abs(r.value - 2.0) < 1e-9

    def test_power_dissipation(self):
        # I=2A, R=5 -> P = 20 W
        r = t_circ.power_dissipation_resistor(2.0, 5.0)
        assert abs(r.value - 20.0) < 1e-9

    def test_parallel_plate_cap(self):
        # 1 m^2 plates, 1 mm gap, vacuum -> C = eps_0 * 1 / 0.001 ~ 8.854 nF
        r = t_circ.parallel_plate_capacitance(1.0, 1e-3)
        assert 8.8e-9 < r.value < 8.9e-9

    def test_rc_tau(self):
        r = t_circ.rc_time_constant(1000.0, 1e-6)
        # 1 ms
        assert abs(r.value - 1e-3) < 1e-12

    def test_em_wave_wavelength_visible(self):
        # 600 nm corresponds to 5e14 Hz
        r = t_circ.em_wave_frequency(600e-9)
        assert 4.9e14 < r.value < 5.1e14


# ── Atomic ─────────────────────────────────────────────────────────────

class TestAtomic:

    def test_hydrogen_ionization(self):
        r = t_atom.first_ionization_energy("H")
        # 13.598 eV
        assert abs(r.value - 13.598) < 0.01

    def test_helium_ionization(self):
        r = t_atom.first_ionization_energy("He")
        assert abs(r.value - 24.587) < 0.01

    def test_hydrogen_ground_state(self):
        # n=1, Z=1 -> E = -13.606 eV
        r = t_atom.hydrogen_like_energy_level(1, 1)
        assert abs(r.value - (-13.606)) < 0.01

    def test_he_plus_first_level(self):
        # n=1, Z=2 -> E = -54.4 eV (hydrogen times 4)
        r = t_atom.hydrogen_like_energy_level(1, 2)
        assert abs(r.value - (-54.42)) < 0.1

    def test_h_alpha_emission(self):
        r = t_atom.hydrogen_emission_wavelength(3, 2)
        # 656.3 nm
        assert abs(r.value - 656.3e-9) / 656.3e-9 < 0.01

    def test_photon_energy_at_550nm(self):
        # 550 nm -> ~2.25 eV
        r = t_atom.photon_energy_from_wavelength(550e-9)
        eV = r.value / 1.602176634e-19
        assert 2.2 < eV < 2.3


# ── Astronomy ──────────────────────────────────────────────────────────

class TestAstronomy:

    def test_earth_lookup(self):
        r = t_astr.solar_system_body("earth")
        assert r.value["mass_kg"] == 5.97219e24

    def test_moon_lookup(self):
        r = t_astr.solar_system_body("moon")
        assert "surface_g_ms2" in r.value
        assert abs(r.value["surface_g_ms2"] - 1.625) < 0.01

    def test_named_star_vega(self):
        r = t_astr.named_star("vega")
        assert r.value["spectral_type"] == "A0V"

    def test_proxima_centauri_closest(self):
        r = t_astr.named_star("proxima_centauri")
        # 1.301 pc
        assert 1.2 < r.value["distance_pc"] < 1.4

    def test_light_travel_1au(self):
        # 1 AU in seconds: should be ~499 s (8 min 19 s)
        r = t_astr.light_travel_time(1.496e11)
        assert 498 < r.value < 500

    def test_unknown_body_returns_none(self):
        r = t_astr.solar_system_body("planet_x")
        assert r.value is None
