"""Symbolic math via sympy: solve, integrate, differentiate, simplify.

When the user asks a math question that needs algebra rather than a
constant lookup, the LLM front-end calls these tools. Pure-text inputs
and outputs so the MCP wire protocol stays simple.
"""

from __future__ import annotations

from sigma_ground.mcp.provenance import ToolResult


import re as _re

# Bare names that sympy's parser binds to objects, not symbols.
_SYMPY_RESERVED = {
    "E": "Euler's number e", "I": "the imaginary unit i",
    "N": "sympy's numeric-evaluation function N", "S": "sympy's singleton registry S",
    "O": "sympy's big-O order symbol", "Q": "sympy's assumptions object Q",
}
_RESERVED_RE = _re.compile(r"(?<![\w.])([EINSOQ])(?![\w])")


def _sympify(expr: str, variables=()):
    """Parse a string into a sympy expression, raising informatively.

    Every name in ``variables`` (the tool's named variable(s)) is parsed as a
    plain sympy Symbol, so a physicist's E, I, N, S, O or Q is the variable,
    not sympy's constant/function of that name.
    FIXED 2026-10-10: the variable was created after parsing, so
    differentiate_expr('E**2', 'E') returned '0' and solve_equation('E**2 - 4',
    'E') returned [] (A20).
    """
    import sympy as sp
    local = {v: sp.Symbol(v) for v in variables if v}
    try:
        return sp.sympify(expr, locals=local, convert_xor=True)
    except (sp.SympifyError, SyntaxError, TypeError) as e:
        raise ValueError(f"could not parse expression '{expr}': {e}") from e


def _reserved_note(texts, variables=()) -> str:
    """Disclose any OTHER bare E, I, N, S, O or Q in the inputs: sympy reads
    those as its own objects (E = Euler's number, I = imaginary unit, ...)."""
    hits = set()
    for t in texts:
        if t:
            hits.update(m.group(1) for m in _RESERVED_RE.finditer(str(t)))
    hits -= set(variables)
    if not hits:
        return ""
    return ("sympy read " + ", ".join(f"'{h}' as {_SYMPY_RESERVED[h]}" for h in sorted(hits))
            + " (only the named variable is parsed as a symbol)")


def _join_notes(*parts) -> str:
    return "; ".join(p for p in parts if p)


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
            expr = _sympify(lhs_str, [variable]) - _sympify(rhs_str, [variable])
        else:
            expr = _sympify(equation, [variable])
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
        notes=_join_notes(f"{len(solutions)} solution(s)" if solutions else "no solutions",
                          _reserved_note([equation], [variable])),
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
        expr = _sympify(expression, [variable])
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
        notes=_reserved_note([expression, lower, upper], [variable]),
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
        expr = _sympify(expression, [variable])
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
        notes=_reserved_note([expression], [variable]),
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
