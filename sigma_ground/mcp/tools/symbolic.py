"""Symbolic math via sympy: solve, integrate, differentiate, simplify.

When the user asks a math question that needs algebra rather than a
constant lookup, the LLM front-end calls these tools. Pure-text inputs
and outputs so the MCP wire protocol stays simple.
"""

from __future__ import annotations

from sigma_ground.mcp.provenance import ToolResult


def _sympify(expr: str):
    """Parse a string into a sympy expression, raising informatively."""
    import sympy as sp
    try:
        return sp.sympify(expr, convert_xor=True)
    except (sp.SympifyError, SyntaxError, TypeError) as e:
        raise ValueError(f"could not parse expression '{expr}': {e}") from e


def solve_equation(equation: str, variable: str) -> ToolResult:
    """Solve `equation = 0` for `variable`.

    Parameters
    ----------
    equation : str
        Expression (set equal to zero) or "lhs = rhs" form.
        Examples: "x^2 - 4", "v^2 / r - G*M/r^2", "E^2 - (p*c)^2 - (m*c^2)^2"
    variable : str
        The symbol to solve for. Example: "x", "v", "m".

    Returns
    -------
    ToolResult with `value` = list of symbolic solutions as strings.
    """
    try:
        import sympy as sp
    except ImportError:
        return ToolResult(
            value=None, source="sympy not installed",
            inputs={"equation": equation, "variable": variable},
        )

    try:
        # Allow either "expr" (= 0) or "lhs = rhs"
        if "=" in equation:
            lhs_str, rhs_str = equation.split("=", 1)
            expr = _sympify(lhs_str) - _sympify(rhs_str)
        else:
            expr = _sympify(equation)
        var = sp.Symbol(variable)
        solutions = sp.solve(expr, var)
    except (ValueError, NotImplementedError) as e:
        return ToolResult(
            value=None, source="sympy solve error",
            notes=str(e),
            inputs={"equation": equation, "variable": variable},
        )

    return ToolResult(
        value=[str(s) for s in solutions],
        source="sympy.solve",
        formula=f"solve({equation}, {variable})",
        inputs={"equation": equation, "variable": variable},
        notes=f"{len(solutions)} solution(s)" if solutions else "no solutions",
    )


def integrate_expr(expression: str, variable: str,
                    lower: str | None = None,
                    upper: str | None = None) -> ToolResult:
    """Symbolic integration. If lower+upper given, definite; else indefinite.

    Parameters
    ----------
    expression : str
        The integrand. Examples: "x^2", "sin(x)*exp(-x)", "1/(1+x^2)"
    variable : str
        The variable of integration.
    lower, upper : str | None
        Optional integration bounds (sympy-parseable strings like "0",
        "oo", "pi/2").
    """
    try:
        import sympy as sp
    except ImportError:
        return ToolResult(
            value=None, source="sympy not installed",
            inputs={"expression": expression, "variable": variable},
        )

    try:
        expr = _sympify(expression)
        var = sp.Symbol(variable)
        if lower is None and upper is None:
            result = sp.integrate(expr, var)
        else:
            a = _sympify(lower) if lower is not None else sp.S.NegativeInfinity
            b = _sympify(upper) if upper is not None else sp.S.Infinity
            result = sp.integrate(expr, (var, a, b))
    except (ValueError, NotImplementedError) as e:
        return ToolResult(
            value=None, source="sympy integrate error",
            notes=str(e),
            inputs={"expression": expression, "variable": variable,
                    "lower": lower, "upper": upper},
        )

    return ToolResult(
        value=str(result),
        source="sympy.integrate",
        formula=(f"integrate({expression}, ({variable}, {lower}, {upper}))"
                  if lower is not None
                  else f"integrate({expression}, {variable})"),
        inputs={"expression": expression, "variable": variable,
                "lower": lower, "upper": upper},
    )


def differentiate_expr(expression: str, variable: str,
                        order: int = 1) -> ToolResult:
    """Symbolic differentiation.

    Parameters
    ----------
    expression : str
        The expression to differentiate.
    variable : str
        The variable to differentiate with respect to.
    order : int
        Order of differentiation (default 1).
    """
    try:
        import sympy as sp
    except ImportError:
        return ToolResult(
            value=None, source="sympy not installed",
            inputs={"expression": expression, "variable": variable},
        )
    try:
        expr = _sympify(expression)
        var = sp.Symbol(variable)
        result = sp.diff(expr, var, order)
    except (ValueError, NotImplementedError) as e:
        return ToolResult(
            value=None, source="sympy diff error",
            notes=str(e),
            inputs={"expression": expression, "variable": variable,
                    "order": order},
        )
    return ToolResult(
        value=str(result),
        source="sympy.diff",
        formula=f"d^{order}/d{variable}^{order}({expression})",
        inputs={"expression": expression, "variable": variable, "order": order},
    )


def simplify_expr(expression: str) -> ToolResult:
    """Apply sympy.simplify."""
    try:
        import sympy as sp
    except ImportError:
        return ToolResult(value=None, source="sympy not installed",
                           inputs={"expression": expression})
    try:
        expr = _sympify(expression)
        result = sp.simplify(expr)
    except (ValueError, NotImplementedError) as e:
        return ToolResult(value=None, source="sympy simplify error",
                           notes=str(e),
                           inputs={"expression": expression})
    return ToolResult(
        value=str(result),
        source="sympy.simplify",
        formula=f"simplify({expression})",
        inputs={"expression": expression},
    )
