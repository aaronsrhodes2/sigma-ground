"""Unit conversion via pint.

The MCP server stays SI-internal for all physics calculations. This
tool handles user-facing conversions ("convert 5 eV to nm via E = hc/lambda")
and dimensional reasoning for the LLM.

The default pint registry is extended with astronomy/astrophysics units
that users frequently ask about (solar_mass, jupiter_mass, etc.) since
pint's default tables don't include them.
"""

from __future__ import annotations

import re

from sigma_ground.mcp.provenance import ToolResult


def _get_registry():
    """The pint registry shared with Materia's question reader (sigma_ground/units_registry.py)."""
    from sigma_ground.units_registry import get_registry
    return get_registry()


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
        convert(1, "solar_mass", "kg")         -> 1.988e30 kg
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


# A pint expression is evaluated by pint's own parser (no Python eval), but its numbers are Python ints: "9**9**9" never returns. So an
# expression is length-capped, may hold a few powers, each with a small plain-number exponent, and never a tower.
_EVAL_MAX_LEN = 200
_EVAL_MAX_POWERS = 3
_EVAL_MAX_EXPONENT = 400
_EVAL_CHARS = re.compile(r"^[A-Za-z0-9_\s.+\-*/^()%,]*$")
_EVAL_POWER = re.compile(r"(\*\*|\^)\s*(\(?\s*[-+]?\s*[\d.]+(?:[eE][-+]?\d+)?\s*\)?)?")
_EVAL_TOWER = re.compile(r"(\*\*|\^)\s*\(?\s*[-+]?[\d.]+\)?\s*(\*\*|\^)")


def _eval_guard(expression: str) -> str | None:
    """Why this expression is refused, or None."""
    if len(expression) > _EVAL_MAX_LEN:
        return f"longer than {_EVAL_MAX_LEN} characters"
    if not _EVAL_CHARS.match(expression):
        return "only numbers, unit names and + - * / ^ ( ) are allowed"
    powers = list(_EVAL_POWER.finditer(expression))
    if len(powers) > _EVAL_MAX_POWERS or _EVAL_TOWER.search(expression):
        return "powers of powers (and more than three powers) are not evaluated"
    for m in powers:
        try:
            exponent = float(re.sub(r"[()\s]", "", m.group(2) or ""))
        except ValueError:
            return "an exponent must be a plain number"
        if abs(exponent) > _EVAL_MAX_EXPONENT:
            return f"an exponent beyond {_EVAL_MAX_EXPONENT} is not evaluated"
    return None


def evaluate_quantity(expression: str, to_units: str | None = None) -> ToolResult:
    """Evaluate arithmetic on quantities with units, and cancel what cancels.

    Wolfram-style quantity arithmetic: "80 year * 70 / minute" is 2.945e9 (the years cancel against the minutes), "100 km / (60 km/h +
    40 km/h)" is 1 hour, "13.787e9 year / nanosecond" is 4.351e26. The result is reduced to one unit per dimension (a ratio of like units
    becomes a plain number); ``to_units`` asks for a particular unit instead.

    Parameters
    ----------
    expression : str
        Numbers and pint unit names joined by + - * / ^ and brackets. Only like units add. A COUNT word ("beats", "bananas") is not a
        unit: leave it out ("70 / minute" for 70 beats per minute). Constants are not looked up here (use lookup_constant first).
    to_units : str, optional
        A unit to express the result in, e.g. "hour", "J", "km/h".

    Returns
    -------
    ToolResult with the number, its units ("" when the units cancel) and ``interpreted_as`` (the expression as read).
    """
    inputs = {"expression": expression, "to_units": to_units}
    ureg = _get_registry()
    if ureg is None:
        return ToolResult(value=None, source="pint not installed", inputs=inputs)
    why = _eval_guard(expression)
    if why:
        return ToolResult(value=None, source="evaluate_quantity", notes=f"refused: {why}", inputs=inputs)

    import pint
    try:
        qty = ureg.parse_expression(expression)
        if not hasattr(qty, "to_reduced_units"):            # plain arithmetic with no unit in it
            qty = ureg.Quantity(qty, "dimensionless")
        qty = qty.to(to_units) if to_units else qty.to_reduced_units()
    except pint.errors.UndefinedUnitError as e:
        name = getattr(e, "unit_names", None) or str(e)
        return ToolResult(
            value=None, source="pint undefined unit", inputs=inputs,
            notes=f"{name} is not a unit pint knows. A count word (beats, bananas, people) is not a unit: leave it out and keep the rate "
                  f"(70 beats per minute -> 70 / minute).")
    except pint.errors.DimensionalityError as e:
        return ToolResult(value=None, source="pint dimensional error", notes=str(e), inputs=inputs)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, ArithmeticError, pint.errors.PintError, SyntaxError, AttributeError) as e:
        return ToolResult(value=None, source="pint parse error", notes=f"{type(e).__name__}: {str(e)[:160]}", inputs=inputs)

    units = "" if qty.dimensionless and not to_units else str(qty.units)
    if qty.dimensionless and not to_units:
        qty = qty.to("dimensionless")
    return ToolResult(
        value=float(qty.magnitude),
        units=units,
        source="pint quantity arithmetic",
        provenance_tag="DERIVED",
        formula=expression,
        interpreted_as=f"{expression} = {float(qty.magnitude):.6g} {units}".rstrip(),
        inputs=inputs,
    )
