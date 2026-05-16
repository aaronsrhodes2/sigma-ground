"""Capabilities manifest -- what the LLM can call, in what tier.

The MCP server exposes this as a tool so the LLM can plan multi-step
answers. Without it, the LLM hallucinates tools that don't exist and
calls return 'tool not found' errors.

Two tiers:
  PRIMARY    -- standard physics, default-foregrounded
  EXTENDED   -- SSBM theoretical layer, only invoked on explicit request

See memory/project_mcp_server_positioning.md for the positioning rationale.
"""

from __future__ import annotations

from sigma_ground.mcp.provenance import ToolResult


_PRIMARY_TOOLS = [
    {
        "name": "lookup_constant",
        "tier": "PRIMARY",
        "domain": "constants",
        "summary": "Look up a physical constant by name. Tries sigma_ground curated, then scipy.constants CODATA.",
        "inputs": {"name": "str (e.g. 'G', 'speed_of_light', 'electron_mass')"},
        "returns": "value + units + source + uncertainty + provenance_tag",
    },
    {
        "name": "list_constants",
        "tier": "PRIMARY",
        "domain": "constants",
        "summary": "List constants available, with optional filter.",
        "inputs": {"category": "str|None: 'sigma_ground'|'codata'|'all'",
                    "contains": "str|None: substring filter",
                    "limit": "int (default 50)"},
        "returns": "list of constant names",
    },
    {
        "name": "convert_units",
        "tier": "PRIMARY",
        "domain": "units",
        "summary": "Convert a value between unit systems via pint.",
        "inputs": {"value": "float", "from_units": "str", "to_units": "str"},
        "returns": "converted value + target units",
    },
    {
        "name": "parse_quantity",
        "tier": "PRIMARY",
        "domain": "units",
        "summary": "Parse '5.6 light_years' into magnitude + units.",
        "inputs": {"quantity_string": "str"},
        "returns": "magnitude + units",
    },
    {
        "name": "solve_equation",
        "tier": "PRIMARY",
        "domain": "symbolic",
        "summary": "Symbolic solve via sympy. Accepts 'expr=0' or 'lhs=rhs'.",
        "inputs": {"equation": "str", "variable": "str"},
        "returns": "list of symbolic solutions",
    },
    {
        "name": "integrate_expr",
        "tier": "PRIMARY",
        "domain": "symbolic",
        "summary": "Symbolic integration. Definite if bounds given.",
        "inputs": {"expression": "str", "variable": "str",
                    "lower": "str|None", "upper": "str|None"},
        "returns": "symbolic result",
    },
    {
        "name": "differentiate_expr",
        "tier": "PRIMARY",
        "domain": "symbolic",
        "summary": "Symbolic differentiation.",
        "inputs": {"expression": "str", "variable": "str", "order": "int"},
        "returns": "symbolic derivative",
    },
    {
        "name": "simplify_expr",
        "tier": "PRIMARY",
        "domain": "symbolic",
        "summary": "Apply sympy.simplify.",
        "inputs": {"expression": "str"},
        "returns": "simplified form",
    },
    {
        "name": "schwarzschild_radius",
        "tier": "PRIMARY",
        "domain": "gr",
        "summary": "Schwarzschild radius r_s = 2 G M / c^2 in meters.",
        "inputs": {"mass_kg": "float (mass in kg)"},
        "returns": "r_s in meters",
    },
    {
        "name": "isco_radius",
        "tier": "PRIMARY",
        "domain": "gr",
        "summary": "Innermost stable circular orbit r_ISCO = 6 G M / c^2.",
        "inputs": {"mass_kg": "float"},
        "returns": "r_ISCO in meters",
    },
    {
        "name": "photon_sphere_radius",
        "tier": "PRIMARY",
        "domain": "gr",
        "summary": "Photon sphere r_ph = 3 G M / c^2.",
        "inputs": {"mass_kg": "float"},
        "returns": "r_ph in meters",
    },
    {
        "name": "hawking_temperature",
        "tier": "PRIMARY",
        "domain": "gr",
        "summary": "Hawking temperature T_H = hbar c^3 / (8 pi G M k_B) in K.",
        "inputs": {"mass_kg": "float"},
        "returns": "T_H in kelvin",
    },
    {
        "name": "hawking_evaporation_time",
        "tier": "PRIMARY",
        "domain": "gr",
        "summary": "Black-hole evaporation timescale in seconds.",
        "inputs": {"mass_kg": "float"},
        "returns": "tau in seconds",
    },
    {
        "name": "gravitational_redshift",
        "tier": "PRIMARY",
        "domain": "gr",
        "summary": "Schwarzschild gravitational redshift factor at radius r.",
        "inputs": {"mass_kg": "float", "radius_m": "float"},
        "returns": "z = (lambda_obs / lambda_emit) - 1",
    },
    {
        "name": "gravitational_time_dilation",
        "tier": "PRIMARY",
        "domain": "gr",
        "summary": "Clock-rate factor at radius r relative to infinity.",
        "inputs": {"mass_kg": "float", "radius_m": "float"},
        "returns": "dt_r / dt_inf = sqrt(1 - r_s/r)",
    },
]


_EXTENDED_TOOLS: list[dict] = [
    # SSBM-specific tools will be wired here later. They exist in the
    # sigma_ground library already (entanglement.py, sigma-page-time,
    # sigma-bounds) but are deliberately not foregrounded in PRIMARY.
    # See memory/project_mcp_server_positioning.md.
]


def get_manifest() -> ToolResult:
    """Return the full capabilities manifest.

    The LLM calls this first to know what tools exist. Without it,
    function-calling regimes will hallucinate tools.

    Returns
    -------
    ToolResult whose `value` is a dict with 'primary' and 'extended'
    tool listings.
    """
    return ToolResult(
        value={
            "primary": _PRIMARY_TOOLS,
            "extended": _EXTENDED_TOOLS,
            "positioning": (
                "PRIMARY tier is standard physics with rigorous provenance. "
                "Foreground these for routine queries. EXTENDED tier is the "
                "SSBM (Scale-Shifted Baryonic Matter) theoretical layer; "
                "invoke only when the user explicitly asks about black hole "
                "interior structure, cosmic-origin-as-BH hypothesis, sigma-"
                "field dynamics, or wants to compare standard physics to an "
                "alternative theoretical framework. Do NOT volunteer SSBM "
                "framing for ordinary physics questions."
            ),
            "response_style": (
                "Every answer should faithfully report the `source` and "
                "`provenance_tag` from each ToolResult, in addition to the "
                "value and units. This is the tool's distinguishing feature "
                "vs other physics assistants -- transparent provenance."
            ),
        },
        source="sigma_ground.mcp.manifest",
        notes=(
            f"{len(_PRIMARY_TOOLS)} primary tool(s), "
            f"{len(_EXTENDED_TOOLS)} extended tool(s). "
            "Use lookup_constant() or domain-specific tools by name. "
            "Pass inputs as the field types listed."
        ),
    )
