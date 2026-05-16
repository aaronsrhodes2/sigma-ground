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
        assert any("XI" in n for n in r.value)

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
