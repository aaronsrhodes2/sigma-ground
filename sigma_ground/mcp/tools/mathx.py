"""Extended symbolic math via sympy — linear algebra, transforms, series,
limits, ODEs, vector calculus.

Companion to symbolic.py (solve/integrate/differentiate/simplify). These
exist so the LLM front-end NEVER has to self-solve math: every operation
has an exact, provenance-tagged tool. sympy is the single source of truth;
the LLM's only job is to recognize the operation and call the tool.

All inputs/outputs are strings (MCP wire stays simple). Matrices are
passed as nested-list strings, e.g. "[[1,2],[3,4]]".

Every expression goes through the one parser, ``mathparse.parse`` (see symbolic.py): ``x²``, ``2x``, ``√x``,
``π``, ``e^x``, ``sin x``, ``∞`` are read the way a textbook reads them, the result says how (``interpreted_as``),
and a variable left out is inferred when the expression has one free symbol (disclosed in ``notes``).
"""

from __future__ import annotations

import re

from sigma_ground.mcp.provenance import ToolResult
from sigma_ground.mcp.tools.mathparse import MathParseError, infer_variable, parse
from sigma_ground.mcp.tools.symbolic import _join_notes, refusal


def _names(variables: str):
    return [v.strip() for v in variables.split(",") if v.strip()]


def _split_top(text: str):
    """Split on commas that are not inside brackets: 'x*y, atan2(y,x), z' -> three components."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return parts


def _matrix(s: str):
    import sympy as sp
    p = parse(s)
    if not isinstance(p.expr, (list, tuple)):
        raise MathParseError("a matrix is written as a nested list", "e.g. [[1,2],[3,4]]")
    return sp.Matrix(p.expr), p.notes


def _ok(value, source, formula, inputs, notes="", interpreted_as=None):
    return ToolResult(value=value, source=source, formula=formula, inputs=inputs, notes=notes,
                       interpreted_as=interpreted_as)


def _variable_for(variable, *texts):
    if variable:
        return str(variable), ""
    return infer_variable([parse(t).expr for t in texts if t])


# ── Linear algebra ──────────────────────────────────────────────────────

def matrix_determinant(matrix: str) -> ToolResult:
    """Determinant of a square matrix. matrix='[[1,2],[3,4]]' -> -2."""
    try:
        M, notes = _matrix(matrix)
        return _ok(str(M.det()), "sympy.Matrix.det", f"det({matrix})", {"matrix": matrix}, notes, f"det {M.tolist()}")
    except Exception as e:
        return refusal("sympy error", e, {"matrix": matrix})


def matrix_eigenvalues(matrix: str) -> ToolResult:
    """Eigenvalues of a square matrix (with multiplicity)."""
    try:
        M, notes = _matrix(matrix)
        vals = []
        for val, mult in M.eigenvals().items():
            vals.extend([str(val)] * int(mult))
        return _ok(vals, "sympy.Matrix.eigenvals", f"eigenvals({matrix})", {"matrix": matrix},
                   _join_notes(f"{len(vals)} eigenvalue(s)", notes), f"eigenvalues of {M.tolist()}")
    except Exception as e:
        return refusal("sympy error", e, {"matrix": matrix})


def matrix_inverse(matrix: str) -> ToolResult:
    """Inverse of a square matrix."""
    try:
        M, notes = _matrix(matrix)
        return _ok(str(M.inv().tolist()), "sympy.Matrix.inv", f"inv({matrix})", {"matrix": matrix}, notes,
                   f"inverse of {M.tolist()}")
    except Exception as e:
        return refusal("sympy error", e, {"matrix": matrix})


def matrix_multiply(matrix_a: str, matrix_b: str) -> ToolResult:
    """Matrix product A·B."""
    inputs = {"matrix_a": matrix_a, "matrix_b": matrix_b}
    try:
        A, na = _matrix(matrix_a)
        B, nb = _matrix(matrix_b)
        return _ok(str((A * B).tolist()), "sympy.Matrix.__mul__", f"({matrix_a})*({matrix_b})", inputs,
                   _join_notes(na, nb), f"{A.tolist()} · {B.tolist()}")
    except Exception as e:
        return refusal("sympy error", e, inputs)


def solve_linear_system(matrix_a: str, vector_b: str) -> ToolResult:
    """Solve A x = b. matrix_a='[[2,1],[1,3]]', vector_b='[1,2]'."""
    inputs = {"matrix_a": matrix_a, "vector_b": vector_b}
    try:
        import sympy as sp
        A, na = _matrix(matrix_a)
        pb = parse(vector_b)
        b = sp.Matrix(pb.expr)
        x = A.LUsolve(b)
        return _ok([str(v) for v in x], "sympy.Matrix.LUsolve", f"solve {matrix_a} x = {vector_b}", inputs,
                   _join_notes(na, pb.notes), f"solve {A.tolist()} x = {list(b)}")
    except Exception as e:
        return refusal("sympy error", e, inputs)


# ── Limits, series, sums ────────────────────────────────────────────────

def compute_limit(expression: str, variable: str | None = None,
                   point: str = "0", direction: str = "+-") -> ToolResult:
    """Limit of expression as variable -> point. direction '+', '-', or '+-'.
    point accepts numbers, expressions and infinity ("oo", "infinity", "∞", "-infinity")."""
    inputs = {"expression": expression, "variable": variable, "point": point, "direction": direction}
    try:
        import sympy as sp
        name, vnote = _variable_for(variable, expression)
        p = parse(expression, [name])
        pt = parse(point, [name])
        var = sp.Symbol(name)
        d = direction if direction in ("+", "-") else "+-"
        lim = sp.limit(p.expr, var, pt.expr, dir=d) if d in ("+", "-") else sp.limit(p.expr, var, pt.expr)
        arrow = {"+": "⁺", "-": "⁻"}.get(d, "")
        return _ok(str(lim), "sympy.limit", f"limit({expression}, {name}->{point})",
                   {"expression": expression, "variable": name, "point": point},
                   _join_notes(vnote, p.notes, pt.notes), f"lim {name}→{sp.sstr(pt.expr)}{arrow} of {sp.sstr(p.expr)}")
    except Exception as e:
        return refusal("sympy error", e, {"expression": expression, "variable": variable})


_SERIES_CONVENTIONS = {"exclusive": {"exclusive", "default", "sympy", "strict", "below", "o(x^n)"},
                       "inclusive": {"inclusive", "including", "through", "up to and including", "upto", "wolfram"}}


def _canonical_convention(value, table: dict, what: str) -> str:
    """The convention's canonical name for what the caller wrote ('Angular frequency' -> 'angular'); a clear refusal listing the options otherwise."""
    text = re.sub(r"[\s_-]+", " ", str(value or "").strip().lower())
    for canonical, names in table.items():
        if text == canonical or text in {re.sub(r"[\s_-]+", " ", n) for n in names}:
            return canonical
    raise MathParseError(f"unknown {what} convention {value!r}", f"one of {', '.join(table)}")


def series_expansion(expression: str, variable: str | None = None,
                      point: str = "0", order: int = 6, convention: str = "exclusive") -> ToolResult:
    """Taylor/Maclaurin series of expression about point, to given order.

    convention: "exclusive" (default, sympy's): terms BELOW (x - point)^order, so order 4 stops at x^3.
    "inclusive": terms THROUGH (x - point)^order, so order 4 includes x^4 (the usual reading of "up to order 4").
    """
    inputs = {"expression": expression, "variable": variable, "point": point, "order": order, "convention": convention}
    try:
        import sympy as sp
        conv = _canonical_convention(convention, _SERIES_CONVENTIONS, "series order")
        name, vnote = _variable_for(variable, expression)
        p = parse(expression, [name])
        pt = parse(point, [name])
        n = int(order) + (1 if conv == "inclusive" else 0)
        s = sp.series(p.expr, sp.Symbol(name), pt.expr, n).removeO()
        kept = f"through ({name} - {point})^{order}" if conv == "inclusive" else f"below ({name} - {point})^{order}, i.e. up to power {int(order) - 1}"
        return _ok(str(s), "sympy.series", f"series({expression}, {name}, {point}, {n})",
                   {"expression": expression, "variable": name, "point": point, "order": order, "convention": conv},
                   _join_notes(vnote, p.notes, pt.notes, f"order convention: {conv}, keeps the terms {kept} (the other is convention=\"{'exclusive' if conv == 'inclusive' else 'inclusive'}\")"),
                   f"series of {sp.sstr(p.expr)} about {name}={sp.sstr(pt.expr)} to order {order} ({conv})")
    except Exception as e:
        return refusal("sympy error", e, {"expression": expression, "variable": variable})


def summation(expression: str, variable: str | None = None,
              lower: str = "1", upper: str = "oo") -> ToolResult:
    """Symbolic sum of expression over variable from lower to upper."""
    try:
        import sympy as sp
        name, vnote = _variable_for(variable, expression)
        p = parse(expression, [name])
        lo, hi = parse(lower, [name]), parse(upper, [name])
        total = sp.summation(p.expr, (sp.Symbol(name), lo.expr, hi.expr))
        return _ok(str(total), "sympy.summation", f"Sum({expression}, ({name}, {lower}, {upper}))",
                   {"expression": expression, "variable": name, "lower": lower, "upper": upper},
                   _join_notes(vnote, p.notes, lo.notes, hi.notes),
                   f"Σ {name}={sp.sstr(lo.expr)}..{sp.sstr(hi.expr)} of {sp.sstr(p.expr)}")
    except Exception as e:
        return refusal("sympy error", e, {"expression": expression, "variable": variable})


# ── Transforms ──────────────────────────────────────────────────────────

def laplace_transform(expression: str, t_var: str = "t",
                       s_var: str = "s") -> ToolResult:
    """Laplace transform F(s) = L{f(t)}."""
    try:
        import sympy as sp
        t, s = sp.Symbol(t_var, positive=True), sp.Symbol(s_var)
        p = parse(expression, [t_var])
        expr = p.expr.subs(sp.Symbol(t_var), t)          # the transform integrates over t > 0
        F = sp.laplace_transform(expr, t, s, noconds=True)
        return _ok(str(F), "sympy.laplace_transform", f"L{{{expression}}}(s)", {"expression": expression},
                   p.notes, f"L{{{sp.sstr(p.expr)}}}({s_var})")
    except Exception as e:
        return refusal("sympy error", e, {"expression": expression})


_FOURIER_CONVENTIONS = {
    "ordinary": {"ordinary", "ordinary frequency", "frequency", "default", "sympy", "cycles", "hertz"},
    "angular": {"angular", "angular frequency", "omega", "ω", "radians", "radian frequency"},
    "unitary": {"unitary", "symmetric", "unitary angular", "unitary angular frequency", "1/sqrt(2pi)"},
}
_FOURIER_DEFINITION = {
    "ordinary": "F(k) = ∫ f(x) e^(-2πikx) dx (ordinary frequency; sympy's default)",
    "angular": "F(k) = ∫ f(x) e^(-ikx) dx (angular frequency, no prefactor)",
    "unitary": "F(k) = (1/√(2π)) ∫ f(x) e^(-ikx) dx (unitary, angular frequency)",
}


def fourier_transform(expression: str, x_var: str = "x",
                       k_var: str = "k", convention: str = "ordinary") -> ToolResult:
    """Fourier transform of expression.

    convention: "ordinary" (default, sympy's) F(k) = ∫ f(x) e^(-2πikx) dx; "angular" F(k) = ∫ f(x) e^(-ikx) dx (angular frequency);
    "unitary" F(k) = (1/√(2π)) ∫ f(x) e^(-ikx) dx. The result says which one it used.
    """
    inputs = {"expression": expression, "x_var": x_var, "k_var": k_var, "convention": convention}
    try:
        import sympy as sp
        conv = _canonical_convention(convention, _FOURIER_CONVENTIONS, "Fourier")
        x, k = sp.Symbol(x_var), sp.Symbol(k_var)
        p = parse(expression, [x_var])
        F = sp.fourier_transform(p.expr, x, k)                  # ordinary frequency
        if conv in ("angular", "unitary"):                      # G(ω) = F(ω / 2π); unitary also divides by √(2π)
            F = F.subs(k, k / (2 * sp.pi))
            if conv == "unitary":
                F = F / sp.sqrt(2 * sp.pi)
            F = sp.simplify(F)
        others = ", ".join(c for c in _FOURIER_CONVENTIONS if c != conv)
        return _ok(str(F), "sympy.fourier_transform", f"F{{{expression}}}(k)", {**inputs, "convention": conv},
                   _join_notes(p.notes, f"convention: {conv}, {_FOURIER_DEFINITION[conv]}; also available: {others}"),
                   f"F{{{sp.sstr(p.expr)}}}({k_var}) [{conv}]")
    except Exception as e:
        return refusal("sympy error", e, {"expression": expression})


# ── Algebra ─────────────────────────────────────────────────────────────

def factor_expression(expression: str) -> ToolResult:
    """Factor a polynomial / expression."""
    try:
        import sympy as sp
        p = parse(expression)
        return _ok(str(sp.factor(p.expr)), "sympy.factor", f"factor({expression})", {"expression": expression},
                   p.notes, f"factor [{sp.sstr(p.expr)}]")
    except Exception as e:
        return refusal("sympy error", e, {"expression": expression})


def expand_expression(expression: str) -> ToolResult:
    """Expand an expression."""
    try:
        import sympy as sp
        p = parse(expression)
        return _ok(str(sp.expand(p.expr)), "sympy.expand", f"expand({expression})", {"expression": expression},
                   p.notes, f"expand [{sp.sstr(p.expr)}]")
    except Exception as e:
        return refusal("sympy error", e, {"expression": expression})


# ── Differential equations ──────────────────────────────────────────────

def solve_ode(equation: str, func: str = "y",
              variable: str = "x") -> ToolResult:
    """Solve an ODE. Use y, y', y'' for the function and its derivatives.

    Example: "y'' + y" (= 0) -> general solution C1*sin(x)+C2*cos(x).
    """
    inputs = {"equation": equation, "func": func, "variable": variable}
    try:
        import sympy as sp
        x = sp.Symbol(variable)
        y = sp.Function(func)
        s = equation
        # y'' , y' , y  ->  Derivative(y(x),x,2), Derivative(y(x),x), y(x)
        s = s.replace(f"{func}''", f"Derivative({func}({variable}),{variable},2)")
        s = s.replace(f"{func}'", f"Derivative({func}({variable}),{variable})")
        # bare func not already followed by '(' -> func(variable)
        s = re.sub(rf"\b{func}\b(?!\s*\()", f"{func}({variable})", s)
        funcs = {func: y, "Derivative": sp.Derivative}
        if "=" in s:
            lhs, rhs = s.split("=", 1)
            pl, pr = parse(lhs, [variable], functions=funcs), parse(rhs, [variable], functions=funcs)
            eq, notes = sp.Eq(pl.expr, pr.expr), _join_notes(pl.notes, pr.notes)
        else:
            pl = parse(s, [variable], functions=funcs)
            eq, notes = sp.Eq(pl.expr, 0), pl.notes
        sol = sp.dsolve(eq, y(x))
        return _ok(str(sol), "sympy.dsolve", f"dsolve({equation})", inputs,
                   _join_notes(notes, "C1, C2, ... are the arbitrary constants of integration; for a given initial condition, solve for them"),
                   f"solve {sp.sstr(eq)}")
    except Exception as e:
        return refusal("sympy error", e, {"equation": equation})


# ── Vector calculus ─────────────────────────────────────────────────────

_AXES = ("x", "y", "z", "t", "u", "v", "w", "r", "s")


def _field_variables(variables, texts, n=None):
    """(names, disclosure): the caller's variables, or the free symbols of the field in the order x, y, z, ... .
    For a field with n components the list is padded with the next unused axis names to length n (curl of '-y, x, 0' is in x, y, z);
    more free symbols than components is refused with an example, never guessed."""
    if variables:
        return _names(variables), ""
    free = sorted({str(s) for t in texts for s in parse(t).expr.free_symbols},
                  key=lambda v: (_AXES.index(v) if v in _AXES else len(_AXES), v))
    if n is not None:
        if len(free) > n:
            raise MathParseError(f"the field has {len(free)} variables ({', '.join(free)}) but {n} components",
                                 f'give variables explicitly, e.g. variables="{",".join(free[:n])}"')
        free += [a for a in _AXES if a not in free][: n - len(free)]
        free.sort(key=lambda v: (_AXES.index(v) if v in _AXES else len(_AXES), v))      # x, z padded with y is x, y, z, not x, z, y
    elif not free:
        free = [_AXES[0]]
    return free, f"variables not given: used {','.join(free)}"


def gradient(scalar_field: str, variables: str | None = None) -> ToolResult:
    """Gradient ∇f of a scalar field. variables comma-separated (default: the field's own variables, in the order x, y, z)."""
    try:
        import sympy as sp
        names, vnote = _field_variables(variables, [scalar_field])
        vs = [sp.Symbol(v) for v in names]
        p = parse(scalar_field, names)
        grad = [str(sp.diff(p.expr, v)) for v in vs]
        return _ok(grad, "sympy.diff (gradient)", f"grad({scalar_field})",
                   {"scalar_field": scalar_field, "variables": ",".join(names)}, _join_notes(vnote, p.notes), f"∇ [{sp.sstr(p.expr)}]")
    except Exception as e:
        return refusal("sympy error", e, {"scalar_field": scalar_field})


def divergence(vector_field: str, variables: str | None = None) -> ToolResult:
    """Divergence ∇·F. vector_field as comma-separated components, e.g. 'x,y,z'. variables default to the field's own, in the order x, y, z."""
    try:
        import sympy as sp
        comps_text = _split_top(vector_field)
        names, vnote = _field_variables(variables, comps_text, n=len(comps_text))
        vs = [sp.Symbol(v) for v in names]
        parsed = [parse(c, names) for c in comps_text]
        comps = [q.expr for q in parsed]
        if len(comps) != len(vs):
            raise ValueError("component count must match variable count")
        div = sum(sp.diff(comps[i], vs[i]) for i in range(len(vs)))
        return _ok(str(sp.simplify(div)), "sympy.diff (divergence)", f"div({vector_field})",
                   {"vector_field": vector_field, "variables": ",".join(names)},
                   _join_notes(vnote, *[q.notes for q in parsed]), f"∇· ({', '.join(sp.sstr(c) for c in comps)})")
    except Exception as e:
        return refusal("sympy error", e, {"vector_field": vector_field})


def curl(vector_field: str, variables: str | None = None) -> ToolResult:
    """Curl ∇×F for a 3-component field. components comma-separated. variables default to x, y, z (the field's own, padded to three)."""
    try:
        import sympy as sp
        comps_text = _split_top(vector_field)
        names, vnote = _field_variables(variables, comps_text, n=3)
        vs = [sp.Symbol(v) for v in names]
        parsed = [parse(c, names) for c in comps_text]
        F = [q.expr for q in parsed]
        if len(F) != 3 or len(vs) != 3:
            raise ValueError("curl requires 3 components and 3 variables")
        cx = sp.diff(F[2], vs[1]) - sp.diff(F[1], vs[2])
        cy = sp.diff(F[0], vs[2]) - sp.diff(F[2], vs[0])
        cz = sp.diff(F[1], vs[0]) - sp.diff(F[0], vs[1])
        return _ok([str(sp.simplify(c)) for c in (cx, cy, cz)], "sympy.diff (curl)", f"curl({vector_field})",
                   {"vector_field": vector_field, "variables": ",".join(names)},
                   _join_notes(vnote, *[q.notes for q in parsed]), f"∇× ({', '.join(sp.sstr(c) for c in F)})")
    except Exception as e:
        return refusal("sympy error", e, {"vector_field": vector_field})


def percent_of(percent: float, value: float) -> ToolResult:
    """X percent of a value: result = (percent/100) * value.

    Example: percent_of(2, 60) -> 1.2  (2% of 60 watts).
    """
    try:
        r = (float(percent) / 100.0) * float(value)
        return ToolResult(value=r, units="",
                           source="sigma-ground (arithmetic)",
                           formula="(percent/100) * value",
                           inputs={"percent": percent, "value": value})
    except Exception as e:
        return refusal("sigma-ground (arithmetic) error", e, {"percent": percent, "value": value})
