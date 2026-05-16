"""Unit conversion via pint.

The MCP server stays SI-internal for all physics calculations. This
tool handles user-facing conversions ("convert 5 eV to nm via E = hc/lambda")
and dimensional reasoning for the LLM.

The default pint registry is extended with astronomy/astrophysics units
that users frequently ask about (solar_mass, jupiter_mass, etc.) since
pint's default tables don't include them.
"""

from __future__ import annotations

from sigma_ground.mcp.provenance import ToolResult


# Singleton registry, lazily constructed. Extending pint's defaults with
# astrophysics units that aren't in the default registry but are commonly
# requested. Magnitudes match the values in sigma_ground.field.constants.
_REGISTRY = None


def _get_registry():
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    try:
        import pint
    except ImportError:
        return None
    ureg = pint.UnitRegistry()
    # Add astronomy / astrophysics units missing from the default registry.
    # Pint already has: light_year, parsec, astronomical_unit ("au"), atomic_mass_unit, eV.
    # We add solar_*, jupiter_*, earth_*. Values from NASA fact sheets.
    extra_defs = [
        "solar_mass = 1.98892e30 * kilogram = M_sun",
        "solar_radius = 6.957e8 * meter = R_sun",
        "solar_luminosity = 3.828e26 * watt = L_sun",
        "jupiter_mass = 1.89813e27 * kilogram = M_jup",
        "jupiter_radius = 7.1492e7 * meter = R_jup",
        "earth_mass = 5.97219e24 * kilogram = M_earth",
        "earth_radius = 6.3781e6 * meter = R_earth",
    ]
    for defn in extra_defs:
        try:
            ureg.define(defn)
        except (pint.errors.RedefinitionError, pint.errors.DefinitionSyntaxError):
            pass  # already defined or unparseable; skip
    _REGISTRY = ureg
    return ureg


def convert(value: float, from_units: str, to_units: str) -> ToolResult:
    """Convert a numeric value between unit systems.

    Parameters
    ----------
    value : float
        Numeric value to convert.
    from_units : str
        Source units. Pint syntax: "kg", "m/s", "eV", "solar_mass",
        "parsec", "year", "MeV/c^2", "atmospheres", etc.
    to_units : str
        Target units. Same syntax.

    Returns
    -------
    ToolResult with the converted value and the target units.

    Examples
    --------
        convert(5, "eV", "joule")              -> 8.011e-19 J
        convert(1, "solar_mass", "kg")         -> 1.989e30 kg
        convert(1.496e11, "m", "AU")           -> 1.0 AU
    """
    ureg = _get_registry()
    if ureg is None:
        return ToolResult(
            value=None, source="pint not installed",
            notes="pip install sigma-ground[mcp] to enable unit conversion",
            inputs={"value": value, "from_units": from_units,
                    "to_units": to_units},
        )

    import pint
    try:
        qty = value * ureg(from_units)
        result = qty.to(to_units)
    except pint.errors.DimensionalityError as e:
        return ToolResult(
            value=None, source="pint dimensional error",
            notes=str(e),
            inputs={"value": value, "from_units": from_units,
                    "to_units": to_units},
        )
    except pint.errors.UndefinedUnitError as e:
        return ToolResult(
            value=None, source="pint undefined unit",
            notes=str(e),
            inputs={"value": value, "from_units": from_units,
                    "to_units": to_units},
        )

    return ToolResult(
        value=float(result.magnitude),
        units=str(result.units),
        source="pint UnitRegistry",
        formula=f"{value} {from_units} -> {result.magnitude:.6g} {result.units}",
        inputs={"value": value, "from_units": from_units,
                "to_units": to_units},
    )


def parse_quantity(quantity_string: str) -> ToolResult:
    """Parse a unit-bearing string into a numeric value + units pair.

    Useful when the LLM receives raw user input like "5.6 light-years"
    and needs the magnitude and unit separated.

    Parameters
    ----------
    quantity_string : str
        Pint-parseable quantity. Examples: "5.6 light_year", "300 K",
        "3.14e8 m/s".

    Returns
    -------
    ToolResult with `value=magnitude` and `units=unit_string`.
    """
    ureg = _get_registry()
    if ureg is None:
        return ToolResult(
            value=None, source="pint not installed",
            inputs={"quantity_string": quantity_string},
        )

    import pint
    try:
        qty = ureg(quantity_string)
    except (pint.errors.UndefinedUnitError, ValueError) as e:
        return ToolResult(
            value=None, source="pint parse error",
            notes=str(e),
            inputs={"quantity_string": quantity_string},
        )

    return ToolResult(
        value=float(qty.magnitude),
        units=str(qty.units),
        source="pint parse",
        inputs={"quantity_string": quantity_string},
    )
