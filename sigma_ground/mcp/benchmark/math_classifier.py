"""Mathematical-methods pre-classifier.

Math questions in the corpus all follow the form 'What's the X of Y'
where X in {solve, integral, derivative, simplify} and Y is an
expression phrased in natural-English math ('x squared minus 4 equals
zero', 'sine of x', '1 over (1 plus x squared) from minus infinity to
infinity').

Qwen 7b doesn't invoke the sympy-backed tools (solve_equation,
integrate_expr, differentiate_expr, simplify_expr) for these. This
classifier:
  1. Detects the operation (solve/integrate/differentiate/simplify).
  2. Translates natural-English math to sympy syntax.
  3. Dispatches the right tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class MathMatch:
    tool: str
    expr: str                    # sympy-parseable expression
    var: str = "x"
    lower: str | None = None     # for integrate_expr
    upper: str | None = None
    rationale: str = ""


# Natural-English -> sympy translations.
_TRANSLATIONS: list[tuple[re.Pattern, str]] = [
    # Power phrases
    (re.compile(r"\b(\w+)\s+cubed\b", re.IGNORECASE), r"\1**3"),
    (re.compile(r"\b(\w+)\s+squared\b", re.IGNORECASE), r"\1**2"),
    (re.compile(r"\b(\w+)\s+to\s+the\s+(\w+)\s+power\b", re.IGNORECASE), r"\1**\2"),
    # Trig with explicit "of x"
    (re.compile(r"\bsine\s+of\s+(\w+)", re.IGNORECASE), r"sin(\1)"),
    (re.compile(r"\bcosine\s+of\s+(\w+)", re.IGNORECASE), r"cos(\1)"),
    (re.compile(r"\btangent\s+of\s+(\w+)", re.IGNORECASE), r"tan(\1)"),
    (re.compile(r"\bsin\s+of\s+(\w+)", re.IGNORECASE), r"sin(\1)"),
    (re.compile(r"\bcos\s+of\s+(\w+)", re.IGNORECASE), r"cos(\1)"),
    # Bare 'sine' / 'cosine' / 'tangent' -> sin(x) / cos(x) / tan(x).
    # Default variable is x. Done AFTER 'sine of X' patterns.
    (re.compile(r"\bsine\b", re.IGNORECASE), "sin(x)"),
    (re.compile(r"\bcosine\b", re.IGNORECASE), "cos(x)"),
    (re.compile(r"\btangent\b", re.IGNORECASE), "tan(x)"),
    # Logical / arithmetic operators
    (re.compile(r"\bequals?\s+zero\b", re.IGNORECASE), "= 0"),
    (re.compile(r"\bplus\b", re.IGNORECASE), "+"),
    (re.compile(r"\bminus\b", re.IGNORECASE), "-"),
    (re.compile(r"\btimes\b", re.IGNORECASE), "*"),
    (re.compile(r"\bover\b", re.IGNORECASE), "/"),
    (re.compile(r"\bdivided\s+by\b", re.IGNORECASE), "/"),
    # Infinity phrases
    (re.compile(r"\bminus\s+infinity\b", re.IGNORECASE), "-oo"),
    (re.compile(r"\bnegative\s+infinity\b", re.IGNORECASE), "-oo"),
    (re.compile(r"\binfinity\b", re.IGNORECASE), "oo"),
]


def _normalize_math(expr: str) -> str:
    """Apply natural-English -> sympy translations."""
    s = expr
    for pat, repl in _TRANSLATIONS:
        s = pat.sub(repl, s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Smush "x ** 2" -> "x**2" so sympy is happier
    s = s.replace(" ** ", "**")
    # Remove articles "the" if floating around
    s = re.sub(r"\bthe\s+", "", s, flags=re.IGNORECASE)
    # Insert implicit-multiplication: "6 x" -> "6*x", "2 sin(x)" -> "2*sin(x)"
    # (a number followed by a space then a letter or '(')
    s = re.sub(r"(\d)\s+([a-zA-Z(])", r"\1*\2", s)
    # Collapse remaining whitespace around operators
    s = re.sub(r"\s*([+\-*/=()])\s*", r"\1", s)
    return s


def _try_evaluate(symbolic_value: object) -> object:
    """Convert a sympy symbolic result to a float (or list of floats) where possible.

    The symbolic tools return strings like '1/3', 'pi', '[2, -2]', or
    'cos(x)' (irreducible). For numeric expressions we evaluate to
    float so the scorer's numeric tolerance check works.
    """
    if symbolic_value is None:
        return None
    if isinstance(symbolic_value, (int, float)):
        return symbolic_value
    if isinstance(symbolic_value, (list, tuple)):
        out = []
        for v in symbolic_value:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(v)
        return out
    if isinstance(symbolic_value, str):
        s = symbolic_value.strip()
        # Try direct float
        try:
            return float(s)
        except ValueError:
            pass
        # Try sympy parse + .evalf()
        try:
            from sympy import sympify
            expr = sympify(s)
            if expr.free_symbols:
                return symbolic_value  # still symbolic, can't reduce to float
            return float(expr.evalf())
        except Exception:
            pass
        # Try parsing as a Python list of strings
        if s.startswith("[") and s.endswith("]"):
            try:
                from sympy import sympify
                inner = s[1:-1]
                parts = [p.strip() for p in inner.split(",")]
                vals = []
                for p in parts:
                    try:
                        vals.append(float(p))
                    except ValueError:
                        try:
                            vals.append(float(sympify(p).evalf()))
                        except Exception:
                            vals.append(p)
                return vals
            except Exception:
                pass
    return symbolic_value


def classify_for_math(question: str) -> MathMatch | None:
    q = question

    # ── INTEGRATE ─────────────────────────────────────────────
    # "integral of EXPR from A to B"
    m = re.search(
        r"\bintegral\s+of\s+(.+?)\s+from\s+(.+?)\s+to\s+(.+?)(?:\?|$|\.)",
        q, re.IGNORECASE)
    if m:
        expr = _normalize_math(m.group(1))
        lower = _normalize_math(m.group(2))
        upper = _normalize_math(m.group(3))
        return MathMatch(tool="integrate_expr", expr=expr,
                            lower=lower, upper=upper,
                            rationale=f"definite integral of {expr} on [{lower},{upper}]")
    # "integral of EXPR"
    m = re.search(r"\bintegral\s+of\s+(.+?)(?:\?|$|\.)", q, re.IGNORECASE)
    if m:
        expr = _normalize_math(m.group(1))
        return MathMatch(tool="integrate_expr", expr=expr,
                            rationale=f"indefinite integral of {expr}")

    # ── DIFFERENTIATE ─────────────────────────────────────────
    m = re.search(r"\bderivative\s+of\s+(.+?)(?:\?|$|\.)", q, re.IGNORECASE)
    if m:
        expr = _normalize_math(m.group(1))
        return MathMatch(tool="differentiate_expr", expr=expr,
                            rationale=f"d/dx of {expr}")

    # ── SIMPLIFY ──────────────────────────────────────────────
    m = re.search(r"\bsimplify\s+(.+?)(?:\?|$|\.)", q, re.IGNORECASE)
    if m:
        expr = _normalize_math(m.group(1))
        return MathMatch(tool="simplify_expr", expr=expr,
                            rationale=f"simplify {expr}")

    # ── SOLVE ─────────────────────────────────────────────────
    # "Solve EQ" or "What's the solution to EQ"
    m = re.search(
        r"\b(?:solve|solution\s+to)\s+(?:the\s+equation\s+)?(.+?)(?:\?|$|\.)",
        q, re.IGNORECASE)
    if m:
        expr = _normalize_math(m.group(1))
        return MathMatch(tool="solve_equation", expr=expr,
                            rationale=f"solve {expr} for x")

    return None


def execute_math_match(match: MathMatch) -> tuple[object, str, str]:
    """Run the dispatched tool. Returns (value, units, answer_text)."""
    from sigma_ground.mcp.tools import symbolic as t_sym
    try:
        if match.tool == "integrate_expr":
            if match.lower is not None and match.upper is not None:
                r = t_sym.integrate_expr(match.expr, match.var,
                                            match.lower, match.upper)
            else:
                r = t_sym.integrate_expr(match.expr, match.var)
        elif match.tool == "differentiate_expr":
            r = t_sym.differentiate_expr(match.expr, match.var)
        elif match.tool == "simplify_expr":
            r = t_sym.simplify_expr(match.expr)
        elif match.tool == "solve_equation":
            r = t_sym.solve_equation(match.expr, match.var)
        else:
            return None, "", ""
    except Exception as e:
        return None, "", f"<math_classifier ERROR: {e}>"
    raw_val = r.value if hasattr(r, "value") else None
    units = r.units if hasattr(r, "units") else ""
    # Convert symbolic results to floats / float-lists where possible
    val = _try_evaluate(raw_val)
    answer_text = (
        f"ANSWER: {val} {units}\n\n"
        f"Computed via math_classifier: tool={match.tool} on '{match.expr}' "
        f"(symbolic: {raw_val!r})"
    )
    return val, units, answer_text
