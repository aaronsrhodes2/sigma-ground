"""Symbolic math via sympy: solve, integrate, differentiate, simplify.

When the user asks a math question that needs algebra rather than a
constant lookup, the LLM front-end calls these tools. Pure-text inputs
and outputs so the MCP wire protocol stays simple.

Every expression goes through ONE parser, ``mathparse.parse`` (implicit multiplication, ``x²``, ``√x``,
``π``, ``e^x``, ``|x|``, ``sin x``, ``∞`` ...). The result says how the input was read (``interpreted_as``);
a reading the parser will not guess comes back as ``value=None`` with the reason and a corrected example.
``variable`` may be left out when the expression has one free symbol (Wolfram does the same); the choice is
disclosed in ``notes``.
"""

from __future__ import annotations

import re

from sigma_ground.mcp.provenance import ToolResult
from sigma_ground.mcp.tools.mathparse import MathParseError, infer_variable, parse


def _join_notes(*parts) -> str:
    return "; ".join(p for p in parts if p)


def refusal(source: str, error: Exception, inputs: dict) -> ToolResult:
    """A tool that could not answer: value None, the reason (and what to write instead) in notes."""
    hint = getattr(error, "hint", "")
    return ToolResult(value=None, source=source, notes=_join_notes(str(error), f"try: {hint}" if hint else ""), inputs=inputs)


def _no_sympy(inputs: dict) -> ToolResult:
    return ToolResult(value=None, source="sympy not installed", inputs=inputs)


def _variable_for(variable, *texts):
    """(name, disclosure): the caller's variable, else the one the expressions have (several: x, t, ... and say so)."""
    if variable:
        return str(variable), ""
    return infer_variable([parse(t).expr for t in texts if t])


def _split_equation(equation: str):
    """'lhs = rhs' -> (lhs, rhs); an expression alone is set to zero -> (expr, None). '==' counts as '='."""
    eq = equation.replace("==", "=")
    parts = re.split(r"(?<![<>!=])=(?!=)", eq, maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return equation, None


def solve_equation(equation: str, variable: str | None = None) -> ToolResult:
    """Solve `equation = 0` (or 'lhs = rhs') for `variable`.

    Parameters
    ----------
    equation : str
        Expression (set equal to zero) or "lhs = rhs" form.
        Examples: "x^2 - 4", "v^2 / r - G*M/r^2", "E^2 - (p*c)^2 - (m*c^2)^2"
    variable : str, optional
        The symbol to solve for. Example: "x", "v", "m". If omitted and the equation has one free symbol, that one.

    Returns
    -------
    ToolResult with `value` = list of symbolic solutions as strings.
    """
    inputs = {"equation": equation, "variable": variable}
    try:
        import sympy as sp
    except ImportError:
        return _no_sympy(inputs)
    try:
        lhs, rhs = _split_equation(equation)
        name, vnote = _variable_for(variable, lhs, rhs)
        pl = parse(lhs, [name])
        pr = parse(rhs, [name]) if rhs is not None else None
        expr = pl.expr - pr.expr if pr is not None else pl.expr
        var = sp.Symbol(name)
        solutions = sp.solve(expr, var)
    except (MathParseError, ValueError, NotImplementedError) as e:
        return refusal("sympy solve error", e, inputs)
    shown = f"{sp.sstr(pl.expr)} = {sp.sstr(pr.expr)}" if pr is not None else f"{sp.sstr(pl.expr)} = 0"
    return ToolResult(
        value=[str(s) for s in solutions],
        source="sympy.solve",
        formula=f"solve({equation}, {name})",
        inputs={"equation": equation, "variable": name},
        interpreted_as=f"solve {shown} for {name}",
        notes=_join_notes(f"{len(solutions)} solution(s)" if solutions else "no solutions",
                          vnote, pl.notes, pr.notes if pr is not None else ""),
    )


def integrate_expr(expression: str, variable: str | None = None,
                    lower: str | None = None,
                    upper: str | None = None) -> ToolResult:
    """Symbolic integration. If lower+upper given, definite; else indefinite.

    Parameters
    ----------
    expression : str
        The integrand. Examples: "x^2", "sin(x)*exp(-x)", "1/(1+x^2)", "x² e^(-x)"
    variable : str, optional
        The variable of integration (inferred when the integrand has one free symbol).
    lower, upper : str | None
        Optional integration bounds ("0", "oo" / "infinity" / "∞", "pi/2"). One bound alone means the other is ±infinity.
    """
    inputs = {"expression": expression, "variable": variable, "lower": lower, "upper": upper}
    try:
        import sympy as sp
    except ImportError:
        return _no_sympy(inputs)
    try:
        name, vnote = _variable_for(variable, expression)
        p = parse(expression, [name])
        var = sp.Symbol(name)
        bnotes = ""
        if lower is None and upper is None:
            result = sp.integrate(p.expr, var)
            shown = f"∫ {sp.sstr(p.expr)} d{name}"
            formula = f"integrate({expression}, {name})"
        else:
            pa = parse(lower, [name]) if lower is not None else None
            pb = parse(upper, [name]) if upper is not None else None
            a = pa.expr if pa is not None else sp.S.NegativeInfinity
            b = pb.expr if pb is not None else sp.S.Infinity
            result = sp.integrate(p.expr, (var, a, b))
            shown = f"∫ from {sp.sstr(a)} to {sp.sstr(b)} of {sp.sstr(p.expr)} d{name}"
            formula = f"integrate({expression}, ({name}, {lower}, {upper}))"
            bnotes = _join_notes(pa.notes if pa else "", pb.notes if pb else "")
        unevaluated = result.has(sp.Integral)
    except (MathParseError, ValueError, NotImplementedError) as e:
        return refusal("sympy integrate error", e, inputs)
    return ToolResult(
        value=str(result),
        source="sympy.integrate",
        formula=formula,
        inputs={"expression": expression, "variable": name, "lower": lower, "upper": upper},
        interpreted_as=shown,
        notes=_join_notes(vnote, p.notes, bnotes,
                          "sympy found no closed form: the integral is returned unevaluated" if unevaluated else "",
                          "indefinite integral: one antiderivative, the constant of integration (+ C) is omitted" if lower is None and upper is None else ""),
    )


def differentiate_expr(expression: str, variable: str | None = None,
                        order: int = 1) -> ToolResult:
    """Symbolic differentiation.

    Parameters
    ----------
    expression : str
        The expression to differentiate. Examples: "3x² + 2x", "sin(x)/x", "e^(-x²)", "√x".
    variable : str, optional
        The variable to differentiate with respect to (inferred when the expression has one free symbol).
    order : int
        Order of differentiation (default 1).
    """
    inputs = {"expression": expression, "variable": variable, "order": order}
    try:
        import sympy as sp
    except ImportError:
        return _no_sympy(inputs)
    try:
        name, vnote = _variable_for(variable, expression)
        p = parse(expression, [name])
        var = sp.Symbol(name)
        real_note = ""
        if p.expr.has(sp.Abs, sp.sign, sp.re, sp.im, sp.conjugate, sp.arg):
            # |x|, sign, Re, Im are not complex-differentiable: the derivative a student means is the real one
            xr = sp.Symbol(name, real=True)
            result = sp.diff(p.expr.subs(var, xr), xr, int(order)).subs(xr, var)
            real_note = f"took {name} as real (the derivative of |{name}|, sign, Re, Im is a real-variable derivative)"
        else:
            result = sp.diff(p.expr, var, int(order))
    except (MathParseError, ValueError, NotImplementedError, TypeError) as e:
        return refusal("sympy diff error", e, inputs)
    return ToolResult(
        value=str(result),
        source="sympy.diff",
        formula=f"d^{order}/d{name}^{order}({expression})",
        inputs={"expression": expression, "variable": name, "order": order},
        interpreted_as=f"d^{order}/d{name}^{order} [{sp.sstr(p.expr)}]" if int(order) != 1 else f"d/d{name} [{sp.sstr(p.expr)}]",
        notes=_join_notes(vnote, p.notes, real_note),
    )


def simplify_expr(expression: str) -> ToolResult:
    """Apply sympy.simplify."""
    inputs = {"expression": expression}
    try:
        import sympy as sp
    except ImportError:
        return _no_sympy(inputs)
    try:
        p = parse(expression)
        result = sp.simplify(p.expr)
    except (MathParseError, ValueError, NotImplementedError) as e:
        return refusal("sympy simplify error", e, inputs)
    return ToolResult(
        value=str(result),
        source="sympy.simplify",
        formula=f"simplify({expression})",
        inputs=inputs,
        interpreted_as=f"simplify [{sp.sstr(p.expr)}]",
        notes=p.notes,
    )
