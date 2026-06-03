"""Extended symbolic math via sympy — linear algebra, transforms, series,
limits, ODEs, vector calculus.

Companion to symbolic.py (solve/integrate/differentiate/simplify). These
exist so the LLM front-end NEVER has to self-solve math: every operation
has an exact, provenance-tagged tool. sympy is the single source of truth;
the LLM's only job is to recognize the operation and call the tool.

All inputs/outputs are strings (MCP wire stays simple). Matrices are
passed as nested-list strings, e.g. "[[1,2],[3,4]]".
"""

from __future__ import annotations

from sigma_ground.mcp.provenance import ToolResult


def _sympify(expr: str):
    import sympy as sp
    try:
        return sp.sympify(expr, convert_xor=True)
    except (sp.SympifyError, SyntaxError, TypeError) as e:
        raise ValueError(f"could not parse '{expr}': {e}") from e


def _matrix(s: str):
    import sympy as sp
    return sp.Matrix(_sympify(s))


def _ok(value, source, formula, inputs, notes=""):
    return ToolResult(value=value, source=source, formula=formula,
                       inputs=inputs, notes=notes)


def _err(msg, inputs):
    return ToolResult(value=None, source="sympy error", notes=str(msg),
                       inputs=inputs)


# ── Linear algebra ──────────────────────────────────────────────────────

def matrix_determinant(matrix: str) -> ToolResult:
    """Determinant of a square matrix. matrix='[[1,2],[3,4]]' -> -2."""
    try:
        d = _matrix(matrix).det()
        return _ok(str(d), "sympy.Matrix.det", f"det({matrix})",
                    {"matrix": matrix})
    except Exception as e:
        return _err(e, {"matrix": matrix})


def matrix_eigenvalues(matrix: str) -> ToolResult:
    """Eigenvalues of a square matrix (with multiplicity)."""
    try:
        ev = _matrix(matrix).eigenvals()
        vals = []
        for val, mult in ev.items():
            vals.extend([str(val)] * int(mult))
        return _ok(vals, "sympy.Matrix.eigenvals", f"eigenvals({matrix})",
                    {"matrix": matrix}, notes=f"{len(vals)} eigenvalue(s)")
    except Exception as e:
        return _err(e, {"matrix": matrix})


def matrix_inverse(matrix: str) -> ToolResult:
    """Inverse of a square matrix."""
    try:
        inv = _matrix(matrix).inv()
        return _ok(str(inv.tolist()), "sympy.Matrix.inv", f"inv({matrix})",
                    {"matrix": matrix})
    except Exception as e:
        return _err(e, {"matrix": matrix})


def matrix_multiply(matrix_a: str, matrix_b: str) -> ToolResult:
    """Matrix product A·B."""
    try:
        prod = _matrix(matrix_a) * _matrix(matrix_b)
        return _ok(str(prod.tolist()), "sympy.Matrix.__mul__",
                    f"({matrix_a})*({matrix_b})",
                    {"matrix_a": matrix_a, "matrix_b": matrix_b})
    except Exception as e:
        return _err(e, {"matrix_a": matrix_a, "matrix_b": matrix_b})


def solve_linear_system(matrix_a: str, vector_b: str) -> ToolResult:
    """Solve A x = b. matrix_a='[[2,1],[1,3]]', vector_b='[1,2]'."""
    try:
        import sympy as sp
        A = _matrix(matrix_a)
        b = sp.Matrix(_sympify(vector_b))
        x = A.LUsolve(b)
        return _ok([str(v) for v in x], "sympy.Matrix.LUsolve",
                    f"solve {matrix_a} x = {vector_b}",
                    {"matrix_a": matrix_a, "vector_b": vector_b})
    except Exception as e:
        return _err(e, {"matrix_a": matrix_a, "vector_b": vector_b})


# ── Limits, series, sums ────────────────────────────────────────────────

def compute_limit(expression: str, variable: str = "x",
                   point: str = "0", direction: str = "+-") -> ToolResult:
    """Limit of expression as variable -> point. direction '+', '-', or '+-'."""
    try:
        import sympy as sp
        expr = _sympify(expression)
        var = sp.Symbol(variable)
        pt = _sympify(point)
        d = direction if direction in ("+", "-") else "+-"
        lim = sp.limit(expr, var, pt, dir=d) if d in ("+", "-") \
            else sp.limit(expr, var, pt)
        return _ok(str(lim), "sympy.limit",
                    f"limit({expression}, {variable}->{point})",
                    {"expression": expression, "variable": variable,
                     "point": point})
    except Exception as e:
        return _err(e, {"expression": expression, "variable": variable})


def series_expansion(expression: str, variable: str = "x",
                      point: str = "0", order: int = 6) -> ToolResult:
    """Taylor/Maclaurin series of expression about point, to given order."""
    try:
        import sympy as sp
        expr = _sympify(expression)
        var = sp.Symbol(variable)
        s = sp.series(expr, var, _sympify(point), int(order)).removeO()
        return _ok(str(s), "sympy.series",
                    f"series({expression}, {variable}, {point}, {order})",
                    {"expression": expression, "variable": variable,
                     "point": point, "order": order})
    except Exception as e:
        return _err(e, {"expression": expression, "variable": variable})


def summation(expression: str, variable: str = "n",
              lower: str = "1", upper: str = "oo") -> ToolResult:
    """Symbolic sum of expression over variable from lower to upper."""
    try:
        import sympy as sp
        expr = _sympify(expression)
        var = sp.Symbol(variable)
        total = sp.summation(expr, (var, _sympify(lower), _sympify(upper)))
        return _ok(str(total), "sympy.summation",
                    f"Sum({expression}, ({variable}, {lower}, {upper}))",
                    {"expression": expression, "variable": variable,
                     "lower": lower, "upper": upper})
    except Exception as e:
        return _err(e, {"expression": expression})


# ── Transforms ──────────────────────────────────────────────────────────

def laplace_transform(expression: str, t_var: str = "t",
                       s_var: str = "s") -> ToolResult:
    """Laplace transform F(s) = L{f(t)}."""
    try:
        import sympy as sp
        t, s = sp.Symbol(t_var, positive=True), sp.Symbol(s_var)
        # parse with the SAME t symbol (positive) so the transform integrates
        expr = sp.sympify(expression, locals={t_var: t}, convert_xor=True)
        F = sp.laplace_transform(expr, t, s, noconds=True)
        return _ok(str(F), "sympy.laplace_transform",
                    f"L{{{expression}}}(s)",
                    {"expression": expression})
    except Exception as e:
        return _err(e, {"expression": expression})


def fourier_transform(expression: str, x_var: str = "x",
                       k_var: str = "k") -> ToolResult:
    """Fourier transform of expression."""
    try:
        import sympy as sp
        x, k = sp.Symbol(x_var), sp.Symbol(k_var)
        F = sp.fourier_transform(_sympify(expression), x, k)
        return _ok(str(F), "sympy.fourier_transform",
                    f"F{{{expression}}}(k)", {"expression": expression})
    except Exception as e:
        return _err(e, {"expression": expression})


# ── Algebra ─────────────────────────────────────────────────────────────

def factor_expression(expression: str) -> ToolResult:
    """Factor a polynomial / expression."""
    try:
        import sympy as sp
        return _ok(str(sp.factor(_sympify(expression))), "sympy.factor",
                    f"factor({expression})", {"expression": expression})
    except Exception as e:
        return _err(e, {"expression": expression})


def expand_expression(expression: str) -> ToolResult:
    """Expand an expression."""
    try:
        import sympy as sp
        return _ok(str(sp.expand(_sympify(expression))), "sympy.expand",
                    f"expand({expression})", {"expression": expression})
    except Exception as e:
        return _err(e, {"expression": expression})


# ── Differential equations ──────────────────────────────────────────────

def solve_ode(equation: str, func: str = "y",
              variable: str = "x") -> ToolResult:
    """Solve an ODE. Use y, y', y'' for the function and its derivatives.

    Example: "y'' + y" (= 0) -> general solution C1*sin(x)+C2*cos(x).
    """
    try:
        import sympy as sp
        x = sp.Symbol(variable)
        y = sp.Function(func)
        s = equation
        # y'' , y' , y  ->  Derivative(y(x),x,2), Derivative(y(x),x), y(x)
        s = s.replace(f"{func}''", f"Derivative({func}({variable}),{variable},2)")
        s = s.replace(f"{func}'", f"Derivative({func}({variable}),{variable})")
        # bare func not already followed by '(' -> func(variable)
        import re as _re
        s = _re.sub(rf"\b{func}\b(?!\s*\()", f"{func}({variable})", s)
        if "=" in s:
            lhs, rhs = s.split("=", 1)
            eq = sp.Eq(_sympify(lhs), _sympify(rhs))
        else:
            eq = sp.Eq(_sympify(s), 0)
        sol = sp.dsolve(eq, y(x))
        return _ok(str(sol), "sympy.dsolve", f"dsolve({equation})",
                    {"equation": equation, "func": func, "variable": variable})
    except Exception as e:
        return _err(e, {"equation": equation})


# ── Vector calculus ─────────────────────────────────────────────────────

def gradient(scalar_field: str, variables: str = "x,y,z") -> ToolResult:
    """Gradient ∇f of a scalar field. variables comma-separated."""
    try:
        import sympy as sp
        vs = [sp.Symbol(v.strip()) for v in variables.split(",")]
        f = _sympify(scalar_field)
        grad = [str(sp.diff(f, v)) for v in vs]
        return _ok(grad, "sympy.diff (gradient)", f"grad({scalar_field})",
                    {"scalar_field": scalar_field, "variables": variables})
    except Exception as e:
        return _err(e, {"scalar_field": scalar_field})


def divergence(vector_field: str, variables: str = "x,y,z") -> ToolResult:
    """Divergence ∇·F. vector_field as comma-separated components, e.g. 'x,y,z'."""
    try:
        import sympy as sp
        vs = [sp.Symbol(v.strip()) for v in variables.split(",")]
        comps = [_sympify(c) for c in vector_field.split(",")]
        if len(comps) != len(vs):
            raise ValueError("component count must match variable count")
        div = sum(sp.diff(comps[i], vs[i]) for i in range(len(vs)))
        return _ok(str(sp.simplify(div)), "sympy.diff (divergence)",
                    f"div({vector_field})",
                    {"vector_field": vector_field, "variables": variables})
    except Exception as e:
        return _err(e, {"vector_field": vector_field})


def curl(vector_field: str, variables: str = "x,y,z") -> ToolResult:
    """Curl ∇×F for a 3-component field. components comma-separated."""
    try:
        import sympy as sp
        vs = [sp.Symbol(v.strip()) for v in variables.split(",")]
        F = [_sympify(c) for c in vector_field.split(",")]
        if len(F) != 3 or len(vs) != 3:
            raise ValueError("curl requires 3 components and 3 variables")
        cx = sp.diff(F[2], vs[1]) - sp.diff(F[1], vs[2])
        cy = sp.diff(F[0], vs[2]) - sp.diff(F[2], vs[0])
        cz = sp.diff(F[1], vs[0]) - sp.diff(F[0], vs[1])
        return _ok([str(sp.simplify(c)) for c in (cx, cy, cz)],
                    "sympy.diff (curl)", f"curl({vector_field})",
                    {"vector_field": vector_field, "variables": variables})
    except Exception as e:
        return _err(e, {"vector_field": vector_field})


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
        return _err(e, {"percent": percent, "value": value})
