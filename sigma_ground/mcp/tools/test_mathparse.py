"""Tests for the one math parser (mcp/tools/mathparse.py) and the tools that use it.

The oracle is plain ``sympy.sympify`` of a CANONICAL spelling (explicit ``*``, ``**``, function-call brackets),
never the parser under test: ``written`` must mean what ``meant`` says. The table is the notation audit of
2026-10-13 (misc/MENTAT_AUDIT_2026-10-13.md F1) grown to every way a student or a model writes the same thing.
"""
from __future__ import annotations

import random

import pytest
import sympy as sp

from sigma_ground.mcp.tools import mathx as MX
from sigma_ground.mcp.tools import symbolic as SY
from sigma_ground.mcp.tools.mathparse import MathParseError, infer_variable, parse, parse_point, resolve_variable

X, Y, T = sp.symbols("x y t")

# (written, declared variables, canonical meaning)
NOTATION = [
    # powers
    ("x**2", "x", "x**2"), ("x^2", "x", "x**2"), ("x²", "x", "x**2"), ("x³ + 1", "x", "x**3 + 1"), ("x⁴", "x", "x**4"),
    ("x⁻¹", "x", "1/x"), ("x^-2", "x", "x**(-2)"), ("x^(-2)", "x", "x**(-2)"), ("2^x", "x", "2**x"), ("x^0.5", "x", "x**0.5"),
    ("x^(1/2)", "x", "sqrt(x)"), ("(x+1)^2", "x", "(x+1)**2"), ("(x+1)²", "x", "(x+1)**2"), ("x^2^3", "x", "x**(2**3)"),
    ("-x^2", "x", "-x**2"), ("(-x)^2", "x", "x**2"), ("x**-1", "x", "1/x"), ("1/x^2", "x", "x**(-2)"),
    # juxtaposition
    ("2x", "x", "2*x"), ("3x^2", "x", "3*x**2"), ("3x^2 + 2x", "x", "3*x**2 + 2*x"), ("3x^2+2x-5", "x", "3*x**2 + 2*x - 5"),
    ("(x+1)(x-1)", "x", "(x+1)*(x-1)"), ("x(x+1)", "x", "x*(x+1)"), ("2(x+1)", "x", "2*(x+1)"), ("(x+1)2", "x", "2*(x+1)"),
    ("x^2y", "x,y", "x**2*y"), ("xy", "x,y", "x*y"), ("x y", "x,y", "x*y"), ("2xy", "x,y", "2*x*y"), ("xyz", "x,y,z", "x*y*z"),
    ("x*y", "x,y", "x*y"), ("3x/2", "x", "3*x/2"), ("1/2x", "x", "x/2"), ("1/(2x)", "x", "1/(2*x)"), ("x/2y", "x,y", "x*y/2"),
    ("2.5x", "x", "2.5*x"), ("x 2", "x", "2*x"), ("a x", "x", "a*x"), ("ax", "x", "a*x"), ("xy", "x", "x*y"), ("abc", "", "a*b*c"), ("mass", "", "mass_"),
    # roots, absolute value, factorial
    ("sqrt(x)", "x", "sqrt(x)"), ("√x", "x", "sqrt(x)"), ("√(x+1)", "x", "sqrt(x+1)"), ("√2", "", "sqrt(2)"), ("√2x", "x", "sqrt(2)*x"),
    ("√(x²+1)", "x", "sqrt(x**2+1)"), ("|x|", "x", "Abs(x)"), ("|x-1|", "x", "Abs(x-1)"), ("||x|-1|", "x", "Abs(Abs(x)-1)"),
    ("abs(x)", "x", "Abs(x)"), ("Abs(x)", "x", "Abs(x)"), ("2|x|", "x", "2*Abs(x)"), ("|x|+|y|", "x,y", "Abs(x)+Abs(y)"),
    ("x!", "x", "factorial(x)"), ("(x+1)!", "x", "factorial(x+1)"), ("5!", "", "120"), ("n!/(n-2)!", "n", "factorial(n)/factorial(n-2)"),
    # functions
    ("sin(x)", "x", "sin(x)"), ("sin x", "x", "sin(x)"), ("sinx", "x", "sin(x)"), ("sin 2x", "x", "sin(2*x)"), ("sin(2x)", "x", "sin(2*x)"),
    ("sin^2 x", "x", "sin(x)**2"), ("sin^2(x)", "x", "sin(x)**2"), ("sin(x)^2", "x", "sin(x)**2"), ("sin²(x)", "x", "sin(x)**2"),
    ("cos x", "x", "cos(x)"), ("tan x / x", "x", "tan(x)/x"), ("sin x cos x", "x", "sin(x)*cos(x)"), ("sin(x)cos(x)", "x", "sin(x)*cos(x)"),
    ("x sin x", "x", "x*sin(x)"), ("x sin(x)", "x", "x*sin(x)"), ("ln x", "x", "log(x)"), ("ln(x)", "x", "log(x)"), ("log(x)", "x", "log(x)"),
    ("log(x)/x", "x", "log(x)/x"), ("log_2(8)", "", "3"), ("log2(8)", "", "3"), ("log10(1000)", "", "3"), ("exp(x)", "x", "exp(x)"),
    ("exp(-x²)", "x", "exp(-x**2)"), ("arcsin x", "x", "asin(x)"), ("arctan(x)", "x", "atan(x)"), ("asin(x)", "x", "asin(x)"),
    ("sinh x", "x", "sinh(x)"), ("cosh(x)", "x", "cosh(x)"), ("tanh x", "x", "tanh(x)"), ("sec x", "x", "sec(x)"), ("cot(x)", "x", "cot(x)"),
    ("gamma(x)", "x", "gamma(x)"), ("erf(x)", "x", "erf(x)"), ("binomial(5,2)", "", "10"), ("Heaviside(x)", "x", "Heaviside(x)"),
    ("atan2(y, x)", "x,y", "atan2(y, x)"), ("sign(x)", "x", "sign(x)"), ("floor(x)", "x", "floor(x)"),
    # constants and the e / i / E / I question
    ("e^x", "x", "exp(x)"), ("e**x", "x", "exp(x)"), ("e^(2x)", "x", "exp(2*x)"), ("e^(-x²)", "x", "exp(-x**2)"), ("exp(1)", "", "E"), ("e", "", "E"),
    ("π", "", "pi"), ("pi", "", "pi"), ("πx", "x", "pi*x"), ("2π", "", "2*pi"), ("2πx", "x", "2*pi*x"), ("2π x", "x", "2*pi*x"), ("cos(2πx)", "x", "cos(2*pi*x)"),
    ("sin(πx)", "x", "sin(pi*x)"), ("x/π", "x", "x/pi"), ("i", "", "I"), ("2i", "", "2*I"), ("(1+i)(1-i)", "", "2"), ("e^(iπ)", "", "-1"),
    ("E**2", "E", "E_**2"), ("E - m*c**2", "m", "E - m*c**2"), ("I*x", "I", "I_*x"), ("N**2", "N", "N_**2"), ("S+1", "S", "S_+1"),
    ("O*2", "O", "O_*2"), ("Q^2", "Q", "Q_**2"), ("e", "e", "e_"), ("i+1", "i", "i_+1"),
    # infinity and operators
    ("∞", "", "oo"), ("-∞", "", "-oo"), ("infinity", "", "oo"), ("-infinity", "", "-oo"), ("inf", "", "oo"), ("Infinity", "", "oo"), ("oo", "", "oo"),
    ("1/∞", "", "0"), ("x − 1", "x", "x - 1"), ("x × 2", "x", "2*x"), ("x · 2", "x", "2*x"), ("x ÷ 2", "x", "x/2"), ("−x", "x", "-x"),
    # symbols, subscripts, greek
    ("x₁ + x₂", "", "x_1 + x_2"), ("x_1 + x_2", "x_1", "x_1 + x_2"), ("θ²", "θ", "theta**2"), ("2θ", "theta", "2*theta"), ("sin θ", "θ", "sin(theta)"),
    ("ω t", "ω,t", "omega*t"), ("alpha*x", "x", "alpha*x"), ("gamma", "gamma", "gamma_"), ("gamma*x", "gamma,x", "gamma_*x"),
    ("a*x**2 + b*x + c", "x", "a*x**2 + b*x + c"), ("ax^2+bx+c", "x", "a*x**2 + b*x + c"), ("m*c^2", "m", "m*c**2"), ("G M / r^2", "r", "G*M/r**2"),
    # numbers
    ("0.5", "", "0.5"), ("1e3", "", "1000.0"), ("1/3", "", "1/3"), ("100", "", "100"), (" x + 1 ", "x", "x + 1"),
]


def _oracle(meant: str, declared: str):
    """sympify of the canonical spelling, with a trailing underscore marking a plain Symbol that would otherwise be a sympy object."""
    names = [n for n in declared.split(",") if n]
    local = {n: sp.Symbol(n) for n in names}
    for reserved in ("E_", "I_", "N_", "S_", "O_", "Q_", "e_", "i_", "gamma_", "mass_"):
        local[reserved] = sp.Symbol(reserved[:-1])
    return sp.sympify(meant, locals=local)


@pytest.mark.parametrize("written,declared,meant", NOTATION)
def test_notation_means_what_it_says(written, declared, meant):
    p = parse(written, [n for n in declared.split(",") if n])
    want = _oracle(meant, declared)
    if want.has(sp.oo, -sp.oo):                      # oo - oo is nan: compare the infinities themselves
        assert p.expr == want, f"{written!r} was read as {p.interpreted_as!r}, meant {meant!r}"
        return
    diff = sp.simplify(p.expr - want)
    assert diff == 0, f"{written!r} was read as {p.interpreted_as!r}, meant {meant!r}"


def test_the_table_is_wide_enough():
    assert len(NOTATION) >= 150


# ── the audit's 20 cases, by their original wrong outcomes ────────────────────────────────────────

@pytest.mark.parametrize("written,derivative", [
    ("x²", "2*x"), ("x³ + 1", "3*x**2"), ("√x", "1/(2*sqrt(x))"), ("πx", "pi"), ("e^x", "exp(x)"), ("e**x", "exp(x)"),
    ("2x", "2"), ("sin x", "cos(x)"), ("3x^2 + 2x", "6*x + 2"), ("(x+1)(x-1)", "2*x"),
])
def test_the_audits_silently_wrong_derivatives_are_right(written, derivative):
    r = SY.differentiate_expr(written, "x")
    assert r.value is not None, r.notes
    assert sp.simplify(sp.sympify(r.value) - sp.sympify(derivative)) == 0, (written, r.value)


def test_abs_derivative_is_sign_like():
    r = SY.differentiate_expr("|x|", "x")
    assert r.value is not None
    assert sp.sympify(r.value).subs(X, 3) == 1 and sp.sympify(r.value).subs(X, -3) == -1


@pytest.mark.parametrize("point", ["oo", "∞", "infinity", "inf", "Infinity", "+infinity"])
def test_limits_accept_every_spelling_of_infinity(point):
    r = MX.compute_limit("1/x", "x", point)
    assert r.value == "0", (point, r.value, r.notes)


def test_minus_infinity_bound():
    assert SY.integrate_expr("exp(x)", "x", "-infinity", "0").value == "1"


# ── disclosures: what is said, and what is not ────────────────────────────────────────────────────

def test_plain_input_says_nothing():
    for text in ("x**2", "x^2 + 3x", "sin(x)/x", "√x"):
        assert parse(text, ["x"]).notes == "", text


def test_euler_and_imaginary_readings_are_disclosed():
    assert "Euler" in parse("e^x", ["x"]).notes
    assert "Euler" in parse("E - m*c**2", ["m"]).notes
    assert "imaginary" in parse("2i", []).notes
    assert "imaginary" in parse("I*x", ["x"]).notes


def test_undeclared_symbols_are_called_constants():
    p = parse("a*x**2 + b*x + c", ["x"])
    assert p.free == ("a", "b", "c") and "treated a, b, c as constants" in p.notes
    assert "treated a as a constant" in parse("a*x", ["x"]).notes


@pytest.mark.parametrize("name", ["E", "I", "N", "S", "O", "Q", "e", "i", "x", "gamma", "theta", "θ"])
def test_a_declared_variable_is_always_a_symbol(name):
    p = parse(f"{name}**2 + 1", [name])
    assert p.notes == "" and sp.Symbol(name if name != "θ" else "theta") in p.expr.free_symbols


# ── refusals: a reading we will not guess, and what must never be evaluated ───────────────────────

@pytest.mark.parametrize("text", [
    "__import__('os').system('x')", "x.__class__", "open('f')", "eval('1')", "exec('1')", "lambda: 1", "x; y", "'a'", '"a"', "x.real",
    "foo(x)", "bar(1, 2)", "globals()", "getattr(x, 'y')", "x = 1", "x == 1", "import os", "9^9^9", "x^10000", "factorial(100000)", "10000!",
    "", "   ", "(x", "x)", "||x|", "√", "x" * 5000, "1" * 500, "x @ y", "x if y else z", "{x: 1}", "x[0]", "os.system", "print(x)",
])
def test_refuses_what_must_not_be_evaluated_or_guessed(text):
    with pytest.raises(MathParseError):
        parse(text, ["x"])


def test_a_refusal_carries_a_hint_where_one_helps():
    with pytest.raises(MathParseError) as e:
        parse("foo(x)", ["x"])
    assert "unknown function" in str(e.value) and "sin" in e.value.hint
    with pytest.raises(MathParseError) as e:
        parse("x = 1", ["x"])
    assert "solve_equation" in e.value.hint


def test_a_single_unknown_letter_before_a_bracket_multiplies():
    assert parse("a(x+1)", ["x"]).expr == sp.Symbol("a") * (X + 1)


def test_no_builtins_leak_into_evaluation():
    # the names the tokeniser lets through are symbols, never python objects
    p = parse("print + len + id", [])
    assert all(isinstance(s, sp.Symbol) for s in p.expr.free_symbols)


# ── matrices and lists ────────────────────────────────────────────────────────────────────────────

def test_matrix_text_parses_to_nested_lists():
    p = parse("[[1,2],[3,4]]")
    assert p.expr == [[1, 2], [3, 4]] and p.interpreted_as == "[[1, 2], [3, 4]]"
    assert sp.Matrix(p.expr).det() == -2
    assert parse("[[a, b], [c, d]]").expr[0][0] == sp.Symbol("a")
    assert parse("[1, 2]").expr == [1, 2]


# ── variable inference ───────────────────────────────────────────────────────────────────────────

def test_infer_variable():
    assert infer_variable(parse("x**2 + 1").expr) == ("x", "variable not given: used x")
    assert infer_variable(parse("t**2").expr)[0] == "t"
    name, note = infer_variable(parse("a*x + b").expr)
    assert name == "x" and "treated a, b as constants" in note
    assert infer_variable(parse("5").expr)[0] == "x"
    assert resolve_variable("y", "x**2") == ("y", "")
    assert resolve_variable(None, "7*u**2")[0] == "u"


# ── property: the same expression written four ways is one expression ────────────────────────────

def test_renderings_of_the_same_expression_agree():
    from sigma_ground.mcp.benchmark import router_math as RM
    rng = random.Random(20261013)
    checked = 0
    for seed in range(60):
        rng.seed(seed)
        v = rng.choice(["x", "t", "y", "u"])
        for name, e in RM.forms(v, rng).items():
            canonical = parse(e.s, [v]).expr
            for kind, text in (("caret", e.c), ("unicode", e.u)):
                if not text:
                    continue
                got = parse(text, [v]).expr
                assert sp.simplify(got - canonical) == 0, f"{name}/{kind}: {text!r} read as {got}, canonical {canonical}"
                checked += 1
    assert checked > 1000


def test_canonical_text_agrees_with_plain_sympify():
    """On the spelling the old parser accepted, the new one agrees with it (no regression for existing callers)."""
    for text in ["x**2 + 3*x + 2", "sin(x)*cos(x)", "exp(-x**2)", "1/(1 + x**2)", "sqrt(x**2 + 1)", "log(x)/x", "(2*x + 1)**3", "x*exp(x)"]:
        assert sp.simplify(parse(text, ["x"]).expr - sp.sympify(text)) == 0


# ── the tools ────────────────────────────────────────────────────────────────────────────────────

def test_tools_report_how_they_read_the_input():
    r = SY.differentiate_expr("3x² + 2x", "x")
    assert r.value == "6*x + 2" and "3*x**2 + 2*x" in (r.interpreted_as or "")


def test_variable_is_optional_when_there_is_only_one():
    r = SY.integrate_expr("x^2")
    assert r.value == "x**3/3" and "variable not given: used x" in r.notes
    assert SY.differentiate_expr("t^3").value == "3*t**2"
    assert SY.solve_equation("x^2 - 4").value == ["-2", "2"]
    assert MX.compute_limit("sin(x)/x", point="0").value == "1"
    assert sp.simplify(sp.sympify(MX.series_expansion("exp(x)", order=3).value) - (1 + X + X**2 / 2)) == 0
    assert MX.summation("1/n^2").value == "pi**2/6"


def test_ambiguity_is_disclosed_not_hidden():
    r = SY.differentiate_expr("a*x + b")
    assert r.value == "a" and "used x" in r.notes and "treated" in r.notes


def test_bad_input_is_a_clear_refusal_not_a_wrong_answer():
    r = SY.differentiate_expr("foo(x)", "x")
    assert r.value is None and "unknown function" in r.notes
    r = SY.integrate_expr("__import__('os')", "x")
    assert r.value is None


def test_matrix_tools_still_work():
    assert MX.matrix_determinant("[[1,2],[3,4]]").value == "-2"
    assert MX.matrix_inverse("[[1,2],[3,4]]").value == "[[-2, 1], [3/2, -1/2]]"
    assert MX.solve_linear_system("[[2,1],[1,3]]", "[1,2]").value == ["1/5", "3/5"]
    assert sorted(MX.matrix_eigenvalues("[[2,0],[0,3]]").value) == ["2", "3"]


def test_ode_and_transforms_still_work():
    assert "C1" in MX.solve_ode("y'' + y", "y", "x").value
    assert MX.laplace_transform("exp(-2t)", "t", "s").value == "1/(s + 2)"
    assert MX.fourier_transform("exp(-x**2)", "x", "k").value is not None
    assert MX.gradient("x^2 y", "x,y").value == ["2*x*y", "x**2"]
    assert MX.divergence("x, y, z", "x,y,z").value == "3"
    assert MX.curl("-y, x, 0", "x,y,z").value == ["0", "0", "2"]
    assert MX.factor_expression("x²-1").value == "(x - 1)*(x + 1)"
    assert MX.expand_expression("(x+1)(x-1)").value == "x**2 - 1"
    assert SY.simplify_expr("sin(x)^2 + cos(x)^2").value == "1"


def test_vector_calculus_infers_the_fields_own_variables():
    assert MX.gradient("log(x^2 + y^2)").value == ["2*x/(x**2 + y**2)", "2*y/(x**2 + y**2)"]        # two variables, not the old default x,y,z
    assert "variables not given: used x,y" in MX.gradient("x*y").notes
    assert MX.gradient("x^2 y z").value == ["2*x*y*z", "x**2*z", "x**2*y"]
    assert MX.curl("-y, x, 0").value == ["0", "0", "2"]                                           # x, y padded to x, y, z
    assert MX.curl("z, 4y, x^2").value == ["0", "1 - 2*x", "0"]
    assert MX.divergence("x, y, z").value == "3"


def test_a_field_with_more_variables_than_components_asks_rather_than_guesses():
    r = MX.divergence("x*y, y*z")
    assert r.value is None and "give variables explicitly" in r.notes
    assert MX.divergence("x*y, y*z", "x,y").value == "y + z"
    r = MX.curl("x*y, y*z, z*t, t")
    assert r.value is None


def test_padding_keeps_the_axis_order():
    # a field in x and z only: the missing axis is y, and the components belong to x, y, z in that order
    assert MX.divergence("x, z, x^2").value == "1"
    got = MX.curl("x*z^2, 0, x^2 - z").value
    assert [sp.simplify(sp.sympify(g) - sp.sympify(w)) for g, w in zip(got, ["0", "2*x*z - 2*x", "0"])] == [0, 0, 0]
    assert "used x,y,z" in MX.curl("x*z^2, 0, x^2 - z").notes
