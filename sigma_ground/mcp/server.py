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
pint, sympy, astropy).

DEFAULTS:
- Standard physics for all queries unless the user explicitly invokes a
  theoretical framework.
- Every numerical answer must include its source (CODATA, PDG, DESI, etc.)
  and uncertainty when available. Read these from the ToolResult fields
  `source` and `uncertainty`.
- Use SI units internally. Use convert_units() if the user asks for
  different units in the output.

WORKFLOW:
1. Call get_manifest() at session start (or when uncertain about
   available tools) to see what's callable.
2. Plan multi-step answers using the listed tools. Do not invent tools.
3. For each numeric tool call, faithfully report the returned
   `source` and `provenance_tag` fields in your answer.

SSBM POSITIONING:
The library contains an SSBM (Scale-Shifted Baryonic Matter) theoretical
layer. It is NOT in the PRIMARY tier of tools. Only mention SSBM when:
- The user asks about black hole interior structure, dark matter
  mechanism, or cosmic origin hypothesis.
- The user explicitly invokes SSBM, sigma-field, eta entanglement
  fraction, or related concepts.
- A comparison between standard physics and an alternative framework
  is genuinely useful.
Do NOT volunteer SSBM framing for ordinary physics queries.

PROVENANCE TAGS (when present in ToolResult.provenance_tag):
- VERIFIED            -- measured value from CODATA/PDG/IAU/peer-reviewed
- DERIVED             -- computed from other library constants
- EMPIRICAL-INPUT     -- a free parameter set by observation (XI, ETA)
- SPECULATIVE-PENDING -- placeholder, awaits derivation or measurement
- REJECTED            -- former candidate, now disproven (will be `None`)

If a ToolResult has provenance_tag=SPECULATIVE-PENDING or REJECTED,
note this prominently in your answer.
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
    from sigma_ground.mcp import manifest as t_manifest

    server = FastMCP("sigma-ground")

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

    # Run via stdio transport (standard MCP).
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
