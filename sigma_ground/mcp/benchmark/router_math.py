"""Router v3.1 math: every math tool, every parameter, in many surface forms, each one checked.

The 4B missed the benchmark's definite integrals because no training example had ever used
`integrate_expr.lower` / `upper` (tools/tool_census.py coverage: 99 of 545 parameters unseen,
every optional math parameter among them). Math is math: this module teaches all 20 math tools
with ALL their parameters, as questions people actually write.

  * a question is built from a spec in several renderings of the same expression (spoken
    "x squared plus 3 x", caret "x^2 + 3x", Python "x**2 + 3*x", unicode "x² + 3x"), so the
    model learns to turn any of them into the sympy text the tools take;
  * the gold plan is run on the real Mentat server (a plan the server rejects is dropped);
  * the result is checked by an INDEPENDENT method where one exists (a numeric integral against
    the antiderivative, finite differences against a derivative, residuals against roots,
    numpy against matrices, partial sums against series...). A mismatch is not trained on and is
    listed in oracle_report.json: either the tool or the spec is wrong, and the Captain decides;
  * the benchmark guard drops anything that is a benchmark instance, and a set of held-out
    questions (the math suite, below) is never generated as training data.

Two outputs:

    python -m sigma_ground.mcp.benchmark.router_math --out local-cache/router_math_v31 --per 150
    python -m sigma_ground.mcp.benchmark.router_math --suite local-cache/router_math_suite.json

The suite uses WORDINGS the training templates do not (a disjoint template set) plus hand-written
"wild" questions, and carries each question's expected tool and an oracle answer, so a trained
model can be scored on math it was not shown (tools/eval_math_suite.py).

Conventions the plans follow (verified against the tools): expressions are sympy text with **
and explicit *; bounds, points and variables are STRINGS; infinity is "oo" ("infinity" is parsed
as a symbol and gives a wrong answer); a parameter the question does not state is left out so the
tool's own default applies; a matrix is the string "[[1, 2], [3, 4]]".
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp

from sigma_ground.mcp.benchmark import router_dataset as D
from sigma_ground.mcp.benchmark import router_plan as RP

# ── one expression, four renderings ──────────────────────────────────────────────────────

ORD = {2: "squared", 3: "cubed", 4: "to the fourth", 5: "to the fifth", 6: "to the sixth"}
SUP = {2: "²", 3: "³"}


@dataclass
class E:
    s: str                      # sympy text: what the plan carries
    c: str                      # caret / ASCII math: x^2 + 3x
    w: str | None = None        # spoken: x squared plus 3 x
    u: str | None = None        # unicode superscripts: x² + 3x

    def render(self, rng: random.Random) -> str:
        options = [(self.c, 4), (self.s, 2)]
        if self.w:
            options.append((self.w, 3))
        if self.u:
            options.append((self.u, 1.5))
        total = sum(w for _, w in options)
        r = rng.random() * total
        for text, w in options:
            r -= w
            if r <= 0:
                return text
        return self.c


def poly(coefs: list[int], v: str) -> E:
    """coefs from the highest power down, e.g. [3, -5, 2] = 3v^2 - 5v + 2."""
    terms = [(c, d) for d, c in zip(range(len(coefs) - 1, -1, -1), coefs) if c]
    out = {"s": "", "c": "", "w": "", "u": ""}
    for i, (c, d) in enumerate(terms):
        a = abs(c)
        co = "" if (a == 1 and d > 0) else str(a)
        mon = {
            "s": str(a) if d == 0 else (f"{co}*" if co else "") + (v if d == 1 else f"{v}**{d}"),
            "c": str(a) if d == 0 else co + (v if d == 1 else f"{v}^{d}"),
            "u": str(a) if d == 0 else co + (v if d == 1 else (f"{v}{SUP[d]}" if d in SUP else f"{v}^{d}")),
            "w": str(a) if d == 0 else (co + " " if co else "") + (v if d == 1 else f"{v} {ORD.get(d, f'to the {d}th')}"),
        }
        for k in out:
            if i == 0:
                out[k] += ("-" if c < 0 and k != "w" else "minus " if c < 0 else "") + mon[k]
            else:
                sign = (" - " if c < 0 else " + ") if k != "w" else (" minus " if c < 0 else " plus ")
                out[k] += sign + mon[k]
    return E(out["s"] or "0", out["c"] or "0", out["w"] or "zero", out["u"] or "0")


def rand_poly(rng: random.Random, v: str, deg: int | None = None, lo: int = -9, hi: int = 9, terms: int | None = None) -> E:
    deg = deg or rng.randint(1, 4)
    while True:
        coefs = [rng.randint(lo, hi) for _ in range(deg + 1)]
        coefs[0] = rng.choice([c for c in range(1, hi + 1)] + [c for c in range(lo, 0)]) if rng.random() < 0.7 else rng.randint(1, 4)
        if coefs[0] == 0:
            continue
        if terms:                                   # keep only `terms` nonzero coefficients (the leading one stays)
            keep = {0} | set(rng.sample(range(1, deg + 1), min(terms - 1, deg)))
            coefs = [c if i in keep else 0 for i, c in enumerate(coefs)]
        if sum(1 for c in coefs if c) >= 2 or deg == 1:
            return poly(coefs, v)


def a_text(a: int) -> str:
    return "" if a == 1 else str(a)


def sp_text(a: int, fn: str, v: str) -> E:
    arg_s = f"{a}*{v}" if a != 1 else v
    arg_c = f"{a}{v}" if a != 1 else v
    names = {"sin": "sine", "cos": "cosine", "tan": "tangent"}
    return E(f"{fn}({arg_s})", f"{fn}({arg_c})", f"{names[fn]} of {a_text(a) + ' ' if a != 1 else ''}{v}")


def form_exp(a: int, v: str) -> E:
    neg = a < 0
    k = abs(a)
    s = f"exp({'-' if neg else ''}{k if k != 1 else ''}{'*' if k != 1 else ''}{v})"
    c = f"e^({'-' if neg else ''}{k if k != 1 else ''}{v})"
    w = f"e to the {'minus ' if neg else ''}{k if k != 1 else ''}{' ' if k != 1 else ''}{v}"
    return E(s, c, w)


def forms(v: str, rng: random.Random) -> dict[str, E]:
    """The named expression forms (called fresh for each problem, parameters drawn from rng)."""
    a, b = rng.choice([1, 2, 3]), rng.choice([2, 3, 4, 5])
    n = rng.choice([2, 3, 4])
    f = {
        "sin": sp_text(a, "sin", v), "cos": sp_text(a, "cos", v), "tan": sp_text(1, "tan", v),
        "exp": form_exp(a, v), "expneg": form_exp(-a, v),
        "log": E(f"log({v})", f"ln({v})", f"natural log of {v}"),
        "sqrt": E(f"sqrt({v})", f"sqrt({v})", f"square root of {v}", f"√{v}"),
        "recip": E(f"1/{v}", f"1/{v}", f"1 over {v}"),
        "xexp": E(f"{v}*exp({v})", f"{v}e^{v}", f"{v} times e to the {v}"),
        "xsin": E(f"{v}*sin({v})", f"{v} sin({v})", f"{v} times sine of {v}"),
        "xcos": E(f"{v}*cos({v})", f"{v} cos({v})", f"{v} times cosine of {v}"),
        "sin2": E(f"sin({v})**2", f"sin^2({v})", None, f"sin²({v})"),
        "sincos": E(f"sin({v})*cos({v})", f"sin({v})cos({v})", f"sine of {v} times cosine of {v}"),
        "arctan1": E(f"1/(1 + {v}**2)", f"1/(1 + {v}^2)", f"1 over (1 plus {v} squared)", f"1/(1 + {v}²)"),
        "lorentz": E(f"1/({v}**2 + {b**2})", f"1/({v}^2 + {b**2})", f"1 over ({v} squared plus {b**2})", f"1/({v}² + {b**2})"),
        "xover": E(f"{v}/({v}**2 + 1)", f"{v}/({v}^2 + 1)", f"{v} over ({v} squared plus 1)", f"{v}/({v}² + 1)"),
        "gauss": E(f"exp(-{v}**2)", f"e^(-{v}^2)", f"e to the minus {v} squared", f"e^(-{v}²)"),
        "x2expneg": E(f"{v}**2*exp(-{v})", f"{v}^2 e^(-{v})", f"{v} squared times e to the minus {v}", f"{v}² e^(-{v})"),
        "power": E(f"({b}*{v} + 1)**{n}", f"({b}{v} + 1)^{n}", None),
        "sqrtp": E(f"sqrt({v}**2 + 1)", f"sqrt({v}^2 + 1)", None, f"sqrt({v}² + 1)"),
        "expsin": E(f"exp({v})*sin({v})", f"e^{v} sin({v})", f"e to the {v} times sine of {v}"),
        "ratfac": E(f"({v}**2 - {b**2})/({v} - {b})", f"({v}^2 - {b**2})/({v} - {b})", None, f"({v}² - {b**2})/({v} - {b})"),
        "logx": E(f"log({v})/{v}", f"ln({v})/{v}", f"natural log of {v} over {v}"),
    }
    f["poly"] = rand_poly(rng, v)
    f["poly2"] = rand_poly(rng, v, deg=2, terms=3)
    f["cubic"] = rand_poly(rng, v, deg=3)
    return f


def rename(text, old, new):
    """Rename a variable as a whole word: the x in exp(x) is not the x in 'x squared'."""
    return re.sub(r"\b" + re.escape(old) + r"\b", new, text) if text else text


# every function name a plan may use; anything else is a bug in this generator, never a label
KNOWN_FUNCTIONS = {"sin", "cos", "tan", "exp", "log", "sqrt", "abs", "atan", "factorial", "asin", "acos", "sinh", "cosh", "tanh"}


def unknown_functions(text):
    """Names followed by '(' that are not a known function (an undefined function would be 'computed' by sympy)."""
    return {m for m in re.findall(r"([A-Za-z_]\w*)\(", text)} - KNOWN_FUNCTIONS


# ── wording ──────────────────────────────────────────────────────────────────────────────

BOUND_WORDS = {"oo": ["infinity", "∞", "positive infinity"], "-oo": ["minus infinity", "-∞", "negative infinity"],
               "pi": ["pi", "π"], "pi/2": ["pi/2", "π/2", "pi over 2"], "pi/4": ["pi/4", "π/4"], "2*pi": ["2 pi", "2π", "2*pi"],
               "-pi": ["minus pi", "-π", "-pi"]}


def bound_text(b: str, rng: random.Random) -> str:
    return rng.choice(BOUND_WORDS[b]) if b in BOUND_WORDS else b


def pick(rng: random.Random, options: list[str]) -> str:
    return rng.choice(options)


def finish(q: str, rng: random.Random) -> str:
    q = q.strip()
    r = rng.random()
    if r < 0.12:
        q = q.rstrip("?.") + "?"
    elif r < 0.2:
        q = q.rstrip("?.")
    elif r < 0.28:
        q = q[0].lower() + q[1:]
    return q


@dataclass
class P:
    """One problem: the question, the plan, how to check the result, and which recipe made it."""
    recipe: str
    tool: str
    args: dict
    question: str
    check: dict


def step(tool: str, args: dict) -> list:
    return [{"tool": tool, "args": args}]


# Each tool has TRAIN templates and HOLDOUT templates. The held-out math suite uses only the
# HOLDOUT ones (and hand-written wild questions), so it measures wordings never trained on.
TPL = {
    "diff": {
        "train": ["What's the derivative of {e}?", "Differentiate {e} with respect to {v}.", "Find d/d{v} of {e}.",
                  "Compute the derivative of f({v}) = {e}.", "Take the derivative of {e}.", "What is the slope function of {e}?",
                  "Find f'({v}) for f({v}) = {e}.", "Give me the derivative of {e} in {v}.", "{e}: differentiate it.",
                  "derivative of {e}"],
        "holdout": ["How does {e} change with {v}? Give the derivative.", "Work out d/d{v}({e}).", "I need the rate of change of {e} with respect to {v}."],
        "order": ["Find the {o} derivative of {e}.", "What is the {o} derivative of {e} with respect to {v}?", "Differentiate {e} {n} times with respect to {v}.",
                  "Compute d^{n}/d{v}^{n} of {e}."],
        "order_holdout": ["Take {e} and differentiate it {n} times in {v}.", "What do you get for the {o} derivative of f({v}) = {e}?"],
    },
    "int": {
        "train": ["Integrate {e}.", "What's the integral of {e} with respect to {v}?", "Find the antiderivative of {e}.", "∫ {e} d{v}",
                  "Compute the indefinite integral of {e}.", "Integrate {e} with respect to {v}.", "What is the integral of {e}?", "Find ∫{e} d{v}."],
        "holdout": ["Give me an antiderivative for {e}.", "I want to integrate {e} in {v}. What is it?"],
        "definite": ["Evaluate the integral of {e} from {a} to {b}.", "What is the definite integral of {e} between {a} and {b}?",
                     "Compute ∫ from {a} to {b} of {e} d{v}.", "Find the integral of {e} with respect to {v} from {v} = {a} to {v} = {b}.",
                     "Integrate {e} from {a} to {b}.", "What's the area under {e} from {a} to {b}?", "∫_{a}^{b} {e} d{v}",
                     "Calculate the integral from {a} to {b} of {e} d{v}."],
        "definite_holdout": ["Over the interval [{a}, {b}], what is the integral of {e}?", "Find the exact value of ∫({e}) d{v} for {v} running from {a} up to {b}."],
    },
    "solve": {
        "train": ["Solve {e} = 0 for {v}.", "Find the roots of {e}.", "What are the solutions of {e} = 0?", "Solve {e} = 0.",
                  "Find all {v} such that {e} = 0.", "What values of {v} make {e} equal zero?", "Solve the equation {e} = 0 for {v}."],
        "holdout": ["Where does {e} cross zero?", "Find every {v} that satisfies {e} = 0."],
        "rhs": ["Solve {lhs} = {rhs} for {v}.", "Find {v} if {lhs} = {rhs}.", "What is {v} when {lhs} = {rhs}?", "Solve {lhs} = {rhs}."],
        "rhs_holdout": ["If {lhs} = {rhs}, what is {v}?"],
    },
    "simplify": {
        "train": ["Simplify {e}.", "What does {e} simplify to?", "Simplify the expression {e}.", "Reduce {e}.", "{e}: simplify it."],
        "holdout": ["Put {e} in its simplest form.", "Can {e} be simplified? What is it?"],
    },
    "factor": {
        "train": ["Factor {e}.", "Factorise {e}.", "Factor the polynomial {e}.", "Write {e} as a product of factors.", "What is the factored form of {e}?"],
        "holdout": ["Break {e} into factors.", "Express {e} as a product."],
    },
    "expand": {
        "train": ["Expand {e}.", "Multiply out {e}.", "Expand the expression {e}.", "What is {e} expanded?", "Give the expanded form of {e}."],
        "holdout": ["Write {e} without brackets.", "Multiply out the brackets in {e}."],
    },
    "limit": {
        "train": ["What's the limit of {e} as {v} approaches {p}?", "Find the limit of {e} as {v} → {p}.", "Compute lim_({v}→{p}) {e}.",
                  "Evaluate the limit as {v} goes to {p} of {e}.", "What does {e} approach as {v} tends to {p}?", "limit of {e} as {v} approaches {p}"],
        "holdout": ["As {v} gets closer and closer to {p}, what value does {e} settle on?", "Determine lim {e} when {v} → {p}."],
        "side": ["What's the limit of {e} as {v} approaches {p} from the {side}?", "Find the {side}-hand limit of {e} as {v} → {p}.",
                 "Compute the limit of {e} as {v} tends to {p} from the {side}."],
        "side_holdout": ["Approaching {p} from the {side}, what does {e} tend to?"],
    },
    "series": {
        "train": ["Find the Taylor series of {e} about {p} to order {n}.", "Expand {e} as a series around {v} = {p} up to order {n}.",
                  "Give the series expansion of {e} at {p} to order {n}.", "What is the Taylor expansion of {e} about {p}, order {n}?"],
        "default": ["Find the Taylor series of {e}.", "Give the series expansion of {e}.", "Expand {e} as a Taylor series."],
        "holdout": ["Approximate {e} near {v} = {p} with a polynomial of order {n}.", "Series for {e} around {p}, to order {n}."],
        "through": ["Expand {e} around {p} up to and including the {v}^{k} term.", "Give the Taylor series of {e} about {p} through the {v}^{k} term."],
    },
    "sum": {
        "train": ["What is the sum of {e} for {v} from {a} to {b}?", "Compute Σ {e} from {v} = {a} to {b}.", "Evaluate the sum of {e} as {v} runs from {a} to {b}.",
                  "Find the sum from {v} = {a} to {b} of {e}.", "Add up {e} for {v} = {a} to {b}."],
        "default": ["What is the infinite sum of {e}?", "Sum the series {e}."],
        "holdout": ["Total up {e} with {v} going from {a} to {b}.", "What does Σ_({v}={a})^{b} {e} come to?"],
    },
    "laplace": {
        "train": ["Find the Laplace transform of {e}.", "What's the Laplace transform of {e}?", "Laplace transform {e}.", "Compute L{{{e}}}.",
                  "Take the Laplace transform of f({v}) = {e}."],
        "holdout": ["What does {e} become under the Laplace transform?", "Apply the Laplace transform to {e}."],
    },
    "fourier": {
        "train": ["Find the Fourier transform of {e}.", "What's the Fourier transform of {e}?", "Fourier transform {e}.", "Compute the Fourier transform of f({v}) = {e}."],
        "holdout": ["What is the frequency-domain form of {e}? Use the Fourier transform.", "Apply the Fourier transform to {e}."],
    },
    "grad": {
        "train": ["Find the gradient of {e}.", "What's the gradient of {e}?", "Compute ∇({e}).", "Take the gradient of f = {e}.", "grad of {e}"],
        "vars": ["Find the gradient of {e} with respect to {vs}.", "Compute the gradient of f({vc}) = {e}."],
        "holdout": ["Give ∇f for f = {e}.", "What vector points in the direction of steepest increase of {e}?"],
    },
    "div": {
        "train": ["Find the divergence of the vector field ({e}).", "What's the divergence of F = ({e})?", "Compute div F for F = ({e}).", "Divergence of ({e})?"],
        "holdout": ["How much does the field ({e}) spread out at a point? Give the divergence."],
    },
    "curl": {
        "train": ["Find the curl of the vector field ({e}).", "What's the curl of F = ({e})?", "Compute curl F for F = ({e}).", "Curl of ({e})?"],
        "holdout": ["How much does the field ({e}) rotate? Give the curl."],
    },
    "det": {
        "train": ["Find the determinant of {m}.", "What's the determinant of the matrix {m}?", "Compute det {m}.", "Determinant of {m}?"],
        "holdout": ["What is |A| for A = {m}?", "Work out the determinant of this matrix: {m}"],
    },
    "eig": {
        "train": ["Find the eigenvalues of {m}.", "What are the eigenvalues of the matrix {m}?", "Compute the eigenvalues of {m}.", "Eigenvalues of {m}?"],
        "holdout": ["Which numbers λ satisfy det(A - λI) = 0 for A = {m}?", "What are the characteristic values of {m}?"],
    },
    "inv": {
        "train": ["Find the inverse of {m}.", "What's the inverse of the matrix {m}?", "Invert the matrix {m}.", "Compute the inverse of {m}.", "Inverse of {m}?"],
        "holdout": ["What is A^-1 for A = {m}?", "Give me the matrix that undoes {m}."],
    },
    "mul": {
        "train": ["Multiply {m} by {m2}.", "Compute the matrix product {m} times {m2}.", "What is {m} × {m2}?", "Find A·B for A = {m} and B = {m2}.", "Multiply the matrices {m} and {m2}."],
        "holdout": ["What do you get when you multiply {m} on the right by {m2}?"],
    },
    "lin": {
        "train": ["Solve the linear system A x = b with A = {m} and b = {b}.", "Solve {m} x = {b}.", "Find x if A = {m} and b = {b}, with A x = b.",
                  "Solve the system of equations with coefficient matrix {m} and right-hand side {b}."],
        "holdout": ["Given A = {m} and b = {b}, what x solves A x = b?"],
    },
    "ode": {
        "train": ["Solve the differential equation {e} = 0.", "Find the general solution of {e} = 0.", "Solve the ODE {e} = 0 for y({v}).", "What function y satisfies {e} = 0?"],
        "holdout": ["Give me every function y({v}) with {e} = 0."],
    },
}


def q_for(kind: str, rng: random.Random, holdout: bool, key: str = "train", **kw) -> str:
    pool = TPL[kind][key + "_holdout" if holdout and (key + "_holdout") in TPL[kind] else key]
    if holdout and key == "train":
        pool = TPL[kind]["holdout"]
    return finish(rng.choice(pool).format(**kw), rng)


# ── the problems ─────────────────────────────────────────────────────────────────────────

ORDINAL = {2: "second", 3: "third", 4: "fourth"}


def var_choice(rng: random.Random, weights=(("x", 12), ("t", 2), ("y", 1), ("z", 1), ("u", 1))) -> str:
    return rng.choices([w for w, _ in weights], [k for _, k in weights])[0]


def p_diff(rng, holdout):
    v = var_choice(rng)
    f = forms(v, rng)
    name = rng.choice(["poly", "poly2", "cubic", "sin", "cos", "tan", "exp", "expneg", "log", "sqrt", "recip", "xexp", "xsin", "sin2", "sincos",
                       "arctan1", "lorentz", "xover", "gauss", "power", "sqrtp", "ratfac", "logx", "expsin"])
    e = f[name]
    order = 1
    if rng.random() < 0.3 and name in ("poly", "cubic", "sin", "cos", "exp", "expneg", "xexp", "poly2", "log", "recip"):
        order = rng.choice([2, 2, 3])
    ex = e.render(rng)
    if order == 1:
        q = q_for("diff", rng, holdout, e=ex, v=v)
        args = {"expression": e.s, "variable": v}
    else:
        key = "order"
        pool = TPL["diff"]["order_holdout" if holdout else "order"]
        q = finish(rng.choice(pool).format(e=ex, v=v, o=ORDINAL[order], n=order), rng)
        args = {"expression": e.s, "variable": v, "order": order}
    if v == "x" and rng.random() < 0.35:
        args.pop("variable")                       # a question that never names the variable: the default x applies
        q = q.replace(" with respect to x", "").replace(" in x", "")
    return P("math_diff", "differentiate_expr", args, q, {"kind": "diff", "expr": e.s, "var": v, "order": order})


def p_int_indef(rng, holdout):
    v = var_choice(rng)
    f = forms(v, rng)
    name = rng.choice(["poly", "poly2", "cubic", "sin", "cos", "exp", "expneg", "recip", "log", "sqrt", "xexp", "arctan1", "lorentz", "xover",
                       "xsin", "sincos", "sin2", "tan", "power", "expsin"])
    e = f[name]
    args = {"expression": e.s, "variable": v}
    q = q_for("int", rng, holdout, e=e.render(rng), v=v)
    if v == "x" and rng.random() < 0.3:
        args.pop("variable")
        q = q.replace(" with respect to x", "")
    return P("math_int_indef", "integrate_expr", args, q, {"kind": "int", "expr": e.s, "var": v})


FINITE = [("poly", ["0", "1", "2", "3", "-1"]), ("poly2", ["0", "1", "2", "-2"]), ("sin", ["0", "pi", "pi/2", "2*pi"]), ("cos", ["0", "pi/2", "pi", "pi/4"]),
          ("exp", ["0", "1", "2"]), ("sqrt", ["0", "1", "4", "9"]), ("xexp", ["0", "1"]), ("arctan1", ["0", "1"]), ("recip", ["1", "2", "3"]),
          ("cubic", ["0", "1", "2"]), ("log", ["1", "2", "3"]), ("sincos", ["0", "pi/2", "pi"]), ("xover", ["0", "1", "2"]), ("lorentz", ["0", "1", "2"])]
IMPROPER = [("expneg", "0", "oo"), ("arctan1", "-oo", "oo"), ("arctan1", "0", "oo"), ("gauss", "-oo", "oo"), ("x2expneg", "0", "oo"), ("lorentz", "-oo", "oo"),
            ("xexp_neg", "0", "oo")]


def p_int_def(rng, holdout):
    v = var_choice(rng, (("x", 14), ("t", 2), ("u", 1)))
    f = forms(v, rng)
    f["xexp_neg"] = E(f"{v}*exp(-{v})", f"{v}e^(-{v})", f"{v} times e to the minus {v}")
    if rng.random() < 0.28:
        name, lo, hi = rng.choice(IMPROPER)
    else:
        name, bounds = rng.choice(FINITE)
        lo, hi = sorted(rng.sample(bounds, 2), key=lambda b: float(sp.N(sp.sympify(b))))
    e = f[name]
    if name in ("expneg",):
        e = f["expneg"]
    q = q_for("int", rng, holdout, key="definite", e=e.render(rng), v=v, a=bound_text(lo, rng), b=bound_text(hi, rng))
    return P("math_int_definite", "integrate_expr", {"expression": e.s, "variable": v, "lower": lo, "upper": hi}, q,
             {"kind": "int_def", "expr": e.s, "var": v, "lo": lo, "hi": hi})


def p_solve(rng, holdout):
    v = var_choice(rng, (("x", 14), ("t", 2), ("y", 2), ("n", 1)))
    f = forms(v, rng)
    r = rng.random()
    if r < 0.45:                                   # polynomial = 0 (the question may or may not write "= 0")
        e = rng.choice([f["poly2"], f["cubic"], rand_poly(rng, v, deg=2, terms=2), poly([1, -rng.randint(1, 9) - rng.randint(0, 3), rng.randint(1, 12)], v)])
        ex = e.render(rng)
        q = q_for("solve", rng, holdout, e=ex, v=v)
        args = {"equation": e.s, "variable": v}
        spec = {"kind": "solve", "eq": e.s, "var": v}
    elif r < 0.8:                                  # linear or quadratic with a right-hand side
        a1, b1 = rng.randint(2, 9), rng.randint(1, 15)
        rhs = rng.randint(-5, 30)
        lhs = poly([a1, b1], v) if rng.random() < 0.6 else poly([rng.randint(1, 3), rng.randint(-4, 4), rng.randint(0, 6)], v)
        q = finish(rng.choice(TPL["solve"]["rhs_holdout" if holdout else "rhs"]).format(lhs=lhs.render(rng), rhs=rhs, v=v), rng)
        eq = f"{lhs.s} = {rhs}"
        args = {"equation": eq, "variable": v}
        spec = {"kind": "solve", "eq": eq, "var": v}
    else:                                          # transcendental
        k = rng.choice([("exp(x) = 5", "e to the x = 5", "e^x = 5"), ("exp(x) = 7", "e to the x = 7", "e^x = 7"), ("2**x = 8", "2 to the x = 8", "2^x = 8"),
                        ("log(x) = 2", "natural log of x = 2", "ln(x) = 2"), ("sin(x) = 1/2", "sine of x = 1/2", "sin(x) = 1/2"), ("cos(x) = 0", "cosine of x = 0", "cos(x) = 0")])
        s, w, c = k
        lhs_r = rng.choice([w, c, s.replace("exp(x)", "e**x")]) if False else rng.choice([w, c])
        q = finish(rng.choice(["Solve {t} for x.", "Find x if {t}.", "What is x when {t}?", "Solve {t}."]).format(t=lhs_r), rng)
        args = {"equation": s, "variable": "x"}
        spec = {"kind": "solve", "eq": s, "var": "x"}
    return P("math_solve", "solve_equation", args, q, spec)


def p_simplify(rng, holdout):
    v = var_choice(rng, (("x", 10), ("t", 1), ("y", 1)))
    a, b = rng.randint(1, 5), rng.randint(2, 6)
    cases = [
        (f"sin({v})**2 + cos({v})**2", f"sin^2({v}) + cos^2({v})", f"sin²({v}) + cos²({v})"),
        (f"({v}**2 - {b*b})/({v} - {b})", f"({v}^2 - {b*b})/({v} - {b})", f"({v}² - {b*b})/({v} - {b})"),
        (f"({v} + {a})**2 - ({v} - {a})**2", f"({v} + {a})^2 - ({v} - {a})^2", f"({v} + {a})² - ({v} - {a})²"),
        (f"exp({v})*exp({a}*{v})", f"e^{v} e^({a}{v})", f"e^{v}·e^({a}{v})"),
        (f"log({v}**{b}) - {b}*log({v})", f"ln({v}^{b}) - {b} ln({v})", f"ln({v}^{b}) - {b}ln({v})"),
        (f"({v}**3 - {v})/({v}*({v} + 1))", f"({v}^3 - {v})/({v}({v} + 1))", f"({v}³ - {v})/({v}({v} + 1))"),
        (f"sin(2*{v})/(2*sin({v}))", f"sin(2{v})/(2 sin({v}))", f"sin(2{v})/(2sin({v}))"),
        (f"({v}**2 + 2*{v} + 1)/({v} + 1)", f"({v}^2 + 2{v} + 1)/({v} + 1)", f"({v}² + 2{v} + 1)/({v} + 1)"),
        (f"{a}*{v} + {b}*{v} - {v}", f"{a}{v} + {b}{v} - {v}", f"{a}{v} + {b}{v} - {v}"),
        (f"1/({v}) + 1/({v}**2)", f"1/{v} + 1/{v}^2", f"1/{v} + 1/{v}²"),
        (f"cos({v})**2 - sin({v})**2", f"cos^2({v}) - sin^2({v})", f"cos²({v}) - sin²({v})"),
        (f"sqrt({v}**2)", f"sqrt({v}^2)", f"sqrt({v}²)"),
    ]
    s, c, u = rng.choice(cases)
    ex = rng.choice([c, s, u])
    return P("math_simplify", "simplify_expr", {"expression": s}, q_for("simplify", rng, holdout, e=ex), {"kind": "simplify", "expr": s, "var": v})


def p_factor(rng, holdout):
    v = var_choice(rng, (("x", 12), ("t", 1), ("y", 1)))
    a, b = rng.randint(1, 7), rng.randint(1, 7)
    r = rng.random()
    if r < 0.5:
        e = poly([1, -(a + b), a * b], v)                       # (x - a)(x - b)
    elif r < 0.7:
        e = poly([1, 0, -(a * a)], v)                           # difference of squares
    elif r < 0.85:
        e = poly([rng.randint(2, 4), (a + b) * 2, a * b], v)
    else:
        e = poly([1, 0, 0, -(a ** 3)], v)                       # difference of cubes
    return P("math_factor", "factor_expression", {"expression": e.s}, q_for("factor", rng, holdout, e=e.render(rng)), {"kind": "factor", "expr": e.s, "var": v})


def p_expand(rng, holdout):
    v = var_choice(rng, (("x", 12), ("t", 1), ("y", 1)))
    a, b, n = rng.randint(1, 6), rng.randint(1, 6), rng.choice([2, 3, 4])
    cases = [
        (f"({v} + {a})**{n}", f"({v} + {a})^{n}", f"({v} + {a}){SUP[n] if n in SUP else '^' + str(n)}"),
        (f"({v} - {a})**{n}", f"({v} - {a})^{n}", f"({v} - {a}){SUP[n] if n in SUP else '^' + str(n)}"),
        (f"({v} + {a})*({v} + {b})", f"({v} + {a})({v} + {b})", f"({v} + {a})({v} + {b})"),
        (f"({v} - {a})*({v} + {b})", f"({v} - {a})({v} + {b})", f"({v} - {a})({v} + {b})"),
        (f"({a}*{v} + {b})**2", f"({a}{v} + {b})^2", f"({a}{v} + {b})²"),
        (f"({v} + {a})*({v}**2 - {v} + {b})", f"({v} + {a})({v}^2 - {v} + {b})", f"({v} + {a})({v}² - {v} + {b})"),
    ]
    s, c, u = rng.choice(cases)
    return P("math_expand", "expand_expression", {"expression": s}, q_for("expand", rng, holdout, e=rng.choice([c, s, u])), {"kind": "expand", "expr": s, "var": v})


LIMITS = [  # (form name, point, direction or None for the default, spoken-point)
    ("sinc", "0", None), ("1mcos", "0", None), ("expm1", "0", None), ("tanx", "0", None), ("log1p", "0", None), ("ratfac_pt", "b", None),
    ("sqrt1p", "0", None), ("recip", "0", "+"), ("recip", "0", "-"), ("sinoverx_oo", "oo", None), ("ratio_oo", "oo", None), ("euler", "oo", None),
    ("x2ovexp", "oo", None), ("expneg", "oo", None), ("xlogx", "0", "+"), ("abs_ratio", "0", "+"), ("abs_ratio", "0", "-"), ("exp_ratio", "0", None),
]


def p_limit(rng, holdout):
    v = var_choice(rng, (("x", 14), ("t", 1), ("n", 1)))
    a = rng.choice([1, 2, 3])
    b = rng.choice([2, 3, 4])
    name, pt, side = rng.choice(LIMITS)
    table = {
        "sinc": (f"sin({a}*{v})/{v}" if a != 1 else f"sin({v})/{v}", f"sin({a}{v})/{v}" if a != 1 else f"sin({v})/{v}"),
        "1mcos": (f"(1 - cos({v}))/{v}**2", f"(1 - cos({v}))/{v}^2"),
        "expm1": (f"(exp({v}) - 1)/{v}", f"(e^{v} - 1)/{v}"),
        "tanx": (f"tan({v})/{v}", f"tan({v})/{v}"),
        "log1p": (f"log(1 + {v})/{v}", f"ln(1 + {v})/{v}"),
        "ratfac_pt": (f"({v}**2 - {b*b})/({v} - {b})", f"({v}^2 - {b*b})/({v} - {b})"),
        "sqrt1p": (f"(sqrt(1 + {v}) - 1)/{v}", f"(sqrt(1 + {v}) - 1)/{v}"),
        "recip": (f"1/{v}", f"1/{v}"),
        "sinoverx_oo": (f"sin({v})/{v}", f"sin({v})/{v}"),
        "ratio_oo": (f"({a+2}*{v}**2 + {v})/({v}**2 + {b})", f"({a+2}{v}^2 + {v})/({v}^2 + {b})"),
        "euler": (f"(1 + 1/{v})**{v}", f"(1 + 1/{v})^{v}"),
        "x2ovexp": (f"{v}**2/exp({v})", f"{v}^2/e^{v}"),
        "expneg": (f"exp(-{v})", f"e^(-{v})"),
        "xlogx": (f"{v}*log({v})", f"{v} ln({v})"),
        "abs_ratio": (f"abs({v})/{v}", f"|{v}|/{v}"),
        "exp_ratio": (f"(exp({a}*{v}) - 1)/{v}", f"(e^({a}{v}) - 1)/{v}"),
    }
    s, c = table[name]
    point = str(b) if pt == "b" else pt
    ex = rng.choice([c, s])
    ptext = bound_text(point, rng) if point in BOUND_WORDS else point
    args = {"expression": s, "variable": v, "point": point}
    if side:
        args["direction"] = side
        word = "right" if side == "+" else "left"
        q = finish(rng.choice(TPL["limit"]["side_holdout" if holdout else "side"]).format(e=ex, v=v, p=ptext, side=word), rng)
    else:
        q = q_for("limit", rng, holdout, e=ex, v=v, p=ptext)
    if v == "x" and rng.random() < 0.3:
        args.pop("variable")
    return P("math_limit", "compute_limit", args, q, {"kind": "limit", "expr": s, "var": v, "point": point, "dir": side or "+-"})


SERIES = [("exp", "0"), ("sin", "0"), ("cos", "0"), ("geo", "0"), ("log1p", "0"), ("recip1p", "0"), ("atan", "0"), ("sqrt1p", "0"), ("exp", "1"), ("cos", "pi/2"), ("sin", "pi")]


def p_series(rng, holdout):
    v = var_choice(rng, (("x", 14), ("t", 1)))
    name, pt = rng.choice(SERIES)
    table = {"exp": (f"exp({v})", f"e^{v}", f"e to the {v}"), "sin": (f"sin({v})", f"sin({v})", f"sine of {v}"),
             "cos": (f"cos({v})", f"cos({v})", f"cosine of {v}"), "geo": (f"1/(1 - {v})", f"1/(1 - {v})", f"1 over (1 minus {v})"),
             "log1p": (f"log(1 + {v})", f"ln(1 + {v})", f"natural log of (1 plus {v})"), "recip1p": (f"1/(1 + {v})", f"1/(1 + {v})", f"1 over (1 plus {v})"),
             "atan": (f"atan({v})", f"arctan({v})", f"arctangent of {v}"), "sqrt1p": (f"sqrt(1 + {v})", f"sqrt(1 + {v})", f"square root of (1 plus {v})")}
    s, c, w = table[name]
    order = rng.choice([3, 4, 5, 6])
    ex = rng.choice([c, w, s])
    ptext = bound_text(pt, rng) if pt in BOUND_WORDS else pt
    args = {"expression": s, "variable": v}
    r = rng.random()
    if pt == "0" and r < 0.2 and not holdout:
        q = finish(rng.choice(TPL["series"]["default"]).format(e=ex), rng)               # nothing stated: tool defaults (point 0, order 6)
        order = 6
        spec_pt = "0"
    elif r < 0.35 and not holdout:
        k = order - 1
        q = finish(rng.choice(TPL["series"]["through"]).format(e=ex, p=ptext, v=v, k=k), rng)
        args.update({"point": pt, "order": order})
        spec_pt = pt
    else:
        pool = TPL["series"]["holdout" if holdout else "train"]
        q = finish(rng.choice(pool).format(e=ex, p=ptext, v=v, n=order), rng)
        args.update({"point": pt, "order": order})
        spec_pt = pt
    if v == "x" and "variable" in args and rng.random() < 0.3:
        args.pop("variable")
    return P("math_series", "series_expansion", args, q, {"kind": "series", "expr": s, "var": v, "point": spec_pt, "order": order})


def p_sum(rng, holdout):
    v = var_choice(rng, (("n", 8), ("k", 6), ("i", 2)))
    r = rng.random()
    if r < 0.5:
        N = rng.choice([5, 10, 20, 100, 50])
        lo = rng.choice(["1", "1", "1", "0", "2"])
        name = rng.choice(["k", "k2", "odd", "k3", "inv", "const"])
        table = {"k": (f"{v}", f"{v}"), "k2": (f"{v}**2", f"{v}^2"), "odd": (f"2*{v} + 1", f"2{v} + 1"), "k3": (f"{v}**3", f"{v}^3"),
                 "inv": (f"1/({v}*({v} + 1))", f"1/({v}({v} + 1))"), "const": (f"{rng.randint(2, 5)}*{v}", None)}
        s, c = table[name]
        c = c or s
        if name == "inv" and lo == "0":
            lo = "1"
        ex = rng.choice([c, s])
        q = finish(rng.choice(TPL["sum"]["holdout" if holdout else "train"]).format(e=ex, v=v, a=lo, b=N), rng)
        args = {"expression": s, "variable": v, "lower": lo, "upper": str(N)}
        spec = {"kind": "sum", "expr": s, "var": v, "lo": lo, "hi": str(N)}
        return P("math_sum_finite", "summation", args, q, spec)
    cases = [(f"1/{v}**2", f"1/{v}^2", "1", "oo"), (f"1/{v}**4", f"1/{v}^4", "1", "oo"), (f"(1/2)**{v}", f"(1/2)^{v}", "0", "oo"), (f"(1/3)**{v}", f"(1/3)^{v}", "1", "oo"),
             (f"1/({v}*({v} + 1))", f"1/({v}({v} + 1))", "1", "oo"), (f"{v}/2**{v}", f"{v}/2^{v}", "1", "oo"), (f"1/factorial({v})", f"1/{v}!", "0", "oo"),
             (f"1/{v}**3", f"1/{v}^3", "1", "oo"), (f"(2/3)**{v}", f"(2/3)^{v}", "0", "oo")]
    s, c, lo, hi = rng.choice(cases)
    ex = rng.choice([c, s])
    if lo == "1" and rng.random() < 0.3 and not holdout:
        q = finish(rng.choice(TPL["sum"]["default"]).format(e=ex), rng)               # nothing stated: variable n, 1 to infinity
        if v != "n":
            s = s.replace(v, "n"); ex = ex.replace(v, "n"); q = q.replace(v, "n")
        args = {"expression": s}
        spec = {"kind": "sum", "expr": s, "var": "n", "lo": "1", "hi": "oo"}
    else:
        q = finish(rng.choice(TPL["sum"]["holdout" if holdout else "train"]).format(e=ex, v=v, a=lo, b=bound_text(hi, rng)), rng)
        args = {"expression": s, "variable": v, "lower": lo, "upper": hi}
        spec = {"kind": "sum", "expr": s, "var": v, "lo": lo, "hi": hi}
    return P("math_sum_infinite", "summation", args, q, spec)


def p_laplace(rng, holdout):
    v = rng.choice(["t"] * 7 + ["x"])
    a, b = rng.choice([1, 2, 3]), rng.choice([2, 3, 4])
    cases = [(f"{v}**{b}", f"{v}^{b}", f"{v} to the {b}"), (f"exp(-{a}*{v})", f"e^(-{a}{v})", f"e to the minus {a} {v}"), (f"sin({b}*{v})", f"sin({b}{v})", f"sine of {b} {v}"),
             (f"cos({b}*{v})", f"cos({b}{v})", f"cosine of {b} {v}"), (f"{v}*exp(-{a}*{v})", f"{v}e^(-{a}{v})", f"{v} times e to the minus {a} {v}"),
             (f"exp({a}*{v})*sin({b}*{v})", f"e^({a}{v}) sin({b}{v})", None), ("1", "1", None), (f"{v}", f"{v}", None), (f"{v}**2*exp(-{v})", f"{v}^2 e^(-{v})", None)]
    s, c, w = rng.choice(cases)
    ex = rng.choice([x for x in (c, s, w) if x])
    args = {"expression": s}
    if v != "t":
        args["t_var"] = v
    q = q_for("laplace", rng, holdout, e=ex, v=v)
    sv = "s"
    if rng.random() < 0.14:
        sv = rng.choice(["p", "z"])
        args["s_var"] = sv
        q = finish(q.rstrip("?.") + rng.choice([f", using {sv} as the transform variable", f" (in terms of {sv})", f", with the result in {sv}"]), rng)
    return P("math_laplace", "laplace_transform", args, q, {"kind": "laplace", "expr": s, "var": v, "svar": sv})


def p_fourier(rng, holdout):
    a = rng.choice([1, 2, 3])
    cases = [("exp(-x**2)", "e^(-x^2)", "e to the minus x squared"), (f"exp(-{a}*x**2)", f"e^(-{a}x^2)", f"e to the minus {a} x squared")]
    s, c, w = rng.choice(cases)
    xv, kv = rng.choice([("x", "k")] * 6 + [("t", "w"), ("t", "f"), ("x", "p")])
    s, c, w = rename(s, "x", xv), rename(c, "x", xv), rename(w, "x", xv)
    args = {"expression": s}
    q = q_for("fourier", rng, holdout, e=rng.choice([c, s, w]), v=xv)
    if (xv, kv) != ("x", "k"):
        if xv != "x":
            args["x_var"] = xv
        if kv != "k":
            args["k_var"] = kv
        q = finish(q.rstrip("?.") + rng.choice([f", from {xv} to {kv}", f" (transform variable {kv})", f", writing the result in terms of {kv}"]), rng)
    return P("math_fourier", "fourier_transform", args, q, {"kind": "fourier", "expr": s, "xvar": xv, "kvar": kv})


def vec_part(rng, names):
    a, b = rng.sample(names, 2)                      # two different variables: no x*x**2 or z + z
    return rng.choice([lambda: f"{a}*{b}", lambda: f"{a}**2", lambda: f"{rng.randint(2, 5)}*{a}", lambda: f"{a}", lambda: f"{a}*{b}**2",
                       lambda: f"sin({a})", lambda: "0", lambda: f"{a} + {b}", lambda: f"{a}**2 - {b}", lambda: f"{rng.randint(2, 4)}*{a}*{b}"])()


def p_grad(rng, holdout):
    three = rng.random() < 0.55
    names = ["x", "y", "z"] if three else ["x", "y"]
    fields = [lambda: f"{names[0]}**2 + {names[1]}**2" + (f" + {names[2]}**2" if three else ""), lambda: f"{names[0]}*{names[1]}" + (f"*{names[2]}" if three else ""),
              lambda: f"{names[0]}**2*{names[1]}" + (f" + {names[2]}" if three else ""), lambda: f"sin({names[0]})*cos({names[1]})", lambda: f"exp({names[0]}*{names[1]})",
              lambda: f"{names[0]}**3 + {rng.randint(2, 4)}*{names[1]}**2", lambda: f"log({names[0]}**2 + {names[1]}**2)"]
    s = rng.choice(fields)()
    c = s.replace("**", "^")
    vs = ",".join(names)
    args = {"scalar_field": s}
    if rng.random() < 0.45 or not three:
        args["variables"] = vs
        q = (finish(rng.choice(TPL["grad"]["vars"]).format(e=rng.choice([c, s]), vs=" and ".join(names) if len(names) == 2 else ", ".join(names), vc=", ".join(names)), rng)
             if not holdout else q_for("grad", rng, True, e=rng.choice([c, s])))
    else:
        q = q_for("grad", rng, holdout, e=rng.choice([c, s]))
    return P("math_gradient", "gradient", args, q, {"kind": "grad", "expr": s, "vars": args.get("variables", "x,y,z")})


def field_names(rng, two_d_ok=False):
    if two_d_ok and rng.random() < 0.15:
        return ["x", "y"]
    return rng.choice([["x", "y", "z"]] * 7 + [["u", "v", "w"], ["r", "s", "t"]])


def p_div(rng, holdout):
    names = field_names(rng, two_d_ok=True)
    comps = [vec_part(rng, names) for _ in names]
    s = ", ".join(comps)
    args = {"vector_field": s}
    q = q_for("div", rng, holdout, e=s.replace("**", "^") if rng.random() < 0.5 else s)
    if names != ["x", "y", "z"]:
        args["variables"] = ",".join(names)
        q = q.rstrip("?.") + rng.choice([f" in the variables {', '.join(names)}", f" (variables {', '.join(names)})",
                                         f", with respect to {' and '.join(names) if len(names) == 2 else ', '.join(names)}"])
    return P("math_divergence", "divergence", args, finish(q, rng), {"kind": "div", "comps": comps, "names": names})


def p_curl(rng, holdout):
    names = field_names(rng)
    comps = [vec_part(rng, names) for _ in names]
    s = ", ".join(comps)
    args = {"vector_field": s}
    q = q_for("curl", rng, holdout, e=s.replace("**", "^") if rng.random() < 0.5 else s)
    if names != ["x", "y", "z"]:
        args["variables"] = ",".join(names)
        q = q.rstrip("?.") + rng.choice([f" in the variables {', '.join(names)}", f" (variables {', '.join(names)})", f", with respect to {', '.join(names)}"])
    return P("math_curl", "curl", args, finish(q, rng), {"kind": "curl", "comps": comps, "names": names})


def mat_text(M: list[list[int]], rng: random.Random) -> str:
    style = rng.random()
    if style < 0.5:
        return str(M)                                            # [[1, 2], [3, 4]]
    if style < 0.7:
        return "[" + "; ".join(" ".join(str(x) for x in r) for r in M) + "]"
    if style < 0.85:
        return "the matrix with rows " + " and ".join("(" + ", ".join(str(x) for x in r) + ")" for r in M)
    return "[" + ", ".join("[" + ", ".join(str(x) for x in r) + "]" for r in M) + "]"


def mat_arg(M: list[list[int]]) -> str:
    return str(M)                                                # "[[1, 2], [3, 4]]": the canonical argument


def rand_mat(rng, n, lo=-5, hi=6, invertible=False, symmetric=False):
    while True:
        M = [[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)]
        if symmetric:
            for i in range(n):
                for j in range(i):
                    M[i][j] = M[j][i]
        if not invertible or round(np.linalg.det(np.array(M, dtype=float))) != 0:
            return M


def p_det(rng, holdout):
    n = rng.choice([2, 2, 2, 3])
    M = rand_mat(rng, n)
    return P("math_det", "matrix_determinant", {"matrix": mat_arg(M)}, q_for("det", rng, holdout, m=mat_text(M, rng)), {"kind": "det", "M": M})


def p_eig(rng, holdout):
    n = rng.choice([2, 2, 3])
    r = rng.random()
    if r < 0.4:
        M = [[0] * n for _ in range(n)]
        for i in range(n):
            M[i][i] = rng.randint(-4, 6)
        for i in range(n):
            for j in range(i + 1, n):
                M[i][j] = rng.randint(-3, 4)                  # upper triangular: eigenvalues on the diagonal
    elif r < 0.7:
        M = rand_mat(rng, n, -3, 4, symmetric=True)
    else:
        a, b = rng.randint(1, 5), rng.randint(1, 5)
        M = [[a + b, b], [0, a]] if n == 2 else [[a, 1, 0], [0, b, 1], [0, 0, a + b]]
    return P("math_eig", "matrix_eigenvalues", {"matrix": mat_arg(M)}, q_for("eig", rng, holdout, m=mat_text(M, rng)), {"kind": "eig", "M": M})


def p_inv(rng, holdout):
    n = rng.choice([2, 2, 2, 3])
    M = rand_mat(rng, n, -4, 5, invertible=True)
    return P("math_inv", "matrix_inverse", {"matrix": mat_arg(M)}, q_for("inv", rng, holdout, m=mat_text(M, rng)), {"kind": "inv", "M": M})


def p_mul(rng, holdout):
    n, k, m = rng.choice([(2, 2, 2), (2, 2, 2), (3, 3, 3), (2, 3, 2), (3, 2, 3)])
    A = [[rng.randint(-4, 5) for _ in range(k)] for _ in range(n)]
    B = [[rng.randint(-4, 5) for _ in range(m)] for _ in range(k)]
    return P("math_mul", "matrix_multiply", {"matrix_a": mat_arg(A), "matrix_b": mat_arg(B)}, q_for("mul", rng, holdout, m=mat_text(A, rng), m2=mat_text(B, rng)),
             {"kind": "mul", "A": A, "B": B})


def p_lin(rng, holdout):
    n = rng.choice([2, 2, 3])
    A = rand_mat(rng, n, -4, 6, invertible=True)
    b = [rng.randint(-8, 12) for _ in range(n)]
    bt = str(b) if rng.random() < 0.6 else "(" + ", ".join(map(str, b)) + ")"
    return P("math_linsys", "solve_linear_system", {"matrix_a": mat_arg(A), "vector_b": str(b)},
             q_for("lin", rng, holdout, m=mat_text(A, rng), b=bt), {"kind": "lin", "A": A, "b": b})


def p_ode(rng, holdout):
    k = rng.randint(1, 9)
    a, b = rng.randint(1, 5), rng.randint(1, 6)
    ks, kc = (f"{k}*" if k != 1 else ""), (str(k) if k != 1 else "")
    cases = [(f"y'' + {ks}y", f"y'' + {kc}y"), (f"y'' - {ks}y", f"y'' - {kc}y"), (f"y' + {ks}y", f"y' + {kc}y"), (f"y' - {ks}y", f"y' - {kc}y"),
             (f"y'' + {a + b}*y' + {a * b}*y", f"y'' + {a + b}y' + {a * b}y"), (f"y'' - {a + b}*y' + {a * b}*y", f"y'' - {a + b}y' + {a * b}y"), ("y'' + y", "y'' + y"), ("y' - y", "y' - y")]
    s, c = rng.choice(cases)
    fn, var = rng.choice([("y", "x")] * 6 + [("u", "t"), ("y", "t"), ("f", "x"), ("x", "t")])
    if fn != "y":
        s, c = s.replace("y", fn), c.replace("y", fn)
    args = {"equation": s}
    q = q_for("ode", rng, holdout, e=rng.choice([c, s]), v=var).replace("y(" + var + ")", fn + "(" + var + ")").replace("function y ", f"function {fn} ")
    if (fn, var) != ("y", "x"):
        if fn != "y":
            args["func"] = fn
        if var != "x":
            args["variable"] = var
        extra = rng.choice([f", for {fn}({var})", f" ({fn} as a function of {var})", f", where {fn} depends on {var}"])
        if f"{fn}({var})" not in q:
            q = finish(q.rstrip("?.") + extra, rng)
    return P("math_ode", "solve_ode", args, q, {"kind": "ode", "eq": s, "fn": fn, "var": var})


GENERATORS = {
    "math_diff": (p_diff, 1.6), "math_int_indef": (p_int_indef, 1.5), "math_int_definite": (p_int_def, 2.2), "math_solve": (p_solve, 1.5), "math_simplify": (p_simplify, 0.8),
    "math_factor": (p_factor, 0.6), "math_expand": (p_expand, 0.6), "math_limit": (p_limit, 1.6), "math_series": (p_series, 1.2), "math_sum": (None, 0),
    "math_sum_finite": (p_sum, 0.8), "math_laplace": (p_laplace, 0.7), "math_fourier": (p_fourier, 0.4), "math_gradient": (p_grad, 0.8), "math_divergence": (p_div, 0.6),
    "math_curl": (p_curl, 0.6), "math_det": (p_det, 0.7), "math_eig": (p_eig, 0.7), "math_inv": (p_inv, 0.7), "math_mul": (p_mul, 0.7), "math_linsys": (p_lin, 0.7), "math_ode": (p_ode, 0.6),
}
GENERATORS.pop("math_sum")
GENERATORS["math_sum_infinite"] = (p_sum, 0.8)          # p_sum itself picks finite or infinite; the weight keeps both well covered


# ── independent checks ───────────────────────────────────────────────────────────────────

def _sym(text: str, extra: dict | None = None):
    return sp.sympify(text.replace("^", "**"), locals=extra or {})


def _num(expr, **subs) -> float | None:
    try:
        v = complex(sp.N(expr.subs({sp.Symbol(k): val for k, val in subs.items()}), 15))
    except Exception:
        return None
    return None if abs(v.imag) > 1e-9 or not math.isfinite(v.real) else v.real


def _points(lo=0.3, hi=2.7, n=5, seed=0):
    r = random.Random(seed)
    return [round(r.uniform(lo, hi), 3) for _ in range(n)]


def _close(a, b, tol=1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def check(spec: dict, raw) -> bool | None:
    """True: the result agrees with an independent method. False: it does not. None: no independent check applies."""
    k = spec["kind"]
    try:
        if k == "diff":
            v = sp.Symbol(spec["var"])
            got = _sym(raw if isinstance(raw, str) else str(raw))
            expr = _sym(spec["expr"])
            n_ok = n = 0
            for p in _points(0.4, 2.2, 5):
                h = 1e-3
                # a numerical n-th derivative from the expression alone, by high-precision central differences
                f = sp.lambdify(v, expr, "mpmath")
                import mpmath
                want = mpmath.diff(f, p, spec["order"])
                g = _num(got, **{spec["var"]: p})
                if g is None:
                    continue
                n += 1
                n_ok += _close(float(want), g, 1e-5)
            return None if n == 0 else n_ok == n
        if k == "int":
            v = sp.Symbol(spec["var"])
            anti = _sym(raw)
            expr = _sym(spec["expr"])
            n = ok = 0
            for p in _points(0.4, 2.2, 5):
                d = _num(sp.diff(anti, v), **{spec["var"]: p})
                e = _num(expr, **{spec["var"]: p})
                if d is None or e is None:
                    continue
                n += 1
                ok += _close(d, e, 1e-6)
            return None if n == 0 else ok == n
        if k == "int_def":
            import scipy.integrate as si
            v = sp.Symbol(spec["var"])
            expr = sp.lambdify(v, _sym(spec["expr"]), "numpy")
            lo, hi = float(sp.N(_sym(spec["lo"]))) if "oo" not in spec["lo"] else (-np.inf), None
            lo = -np.inf if spec["lo"] == "-oo" else (np.inf if spec["lo"] == "oo" else float(sp.N(_sym(spec["lo"]))))
            hi = np.inf if spec["hi"] == "oo" else (-np.inf if spec["hi"] == "-oo" else float(sp.N(_sym(spec["hi"]))))
            want, _ = si.quad(lambda t: float(expr(t)), lo, hi, limit=200)
            got = _num(_sym(raw))
            return None if got is None else _close(want, got, 1e-5)
        if k == "solve":
            roots = raw if isinstance(raw, list) else [raw]
            v = sp.Symbol(spec["var"])
            eq = spec["eq"]
            lhs, rhs = (eq.split("=", 1) + ["0"])[:2] if "=" in eq else (eq, "0")
            resid = _sym(lhs) - _sym(rhs)
            if not roots:
                return None
            for r in roots:
                val = _num(resid, **{spec["var"]: sp.sympify(r)}) if _sym(r).free_symbols == set() else None
                if val is None:
                    return None
                if abs(val) > 1e-8:
                    return False
            return True
        if k == "simplify":
            a, b = _sym(spec["expr"]), _sym(raw)
            v = spec["var"]
            n = ok = 0
            for p in _points(0.3, 1.9, 5, seed=3):
                x1, x2 = _num(a, **{v: p}), _num(b, **{v: p})
                if x1 is None or x2 is None:
                    continue
                n += 1
                ok += _close(x1, x2, 1e-7)
            return None if n == 0 else ok == n
        if k in ("factor", "expand"):
            a, b = _sym(spec["expr"]), _sym(raw)
            v = spec["var"]
            vals = [(_num(a, **{v: p}), _num(b, **{v: p})) for p in _points(-2, 3, 5, seed=5)]
            vals = [t for t in vals if None not in t]
            if not vals or not all(_close(x, y, 1e-8) for x, y in vals):
                return False if vals else None
            text = raw if isinstance(raw, str) else ""
            return ("(" not in text) if k == "expand" and spec["expr"].count("(") else True
        if k == "limit":
            v = sp.Symbol(spec["var"])
            expr = _sym(spec["expr"])
            f = sp.lambdify(v, expr, "mpmath")
            import mpmath
            mpmath.mp.dps = 40
            got_raw = raw if isinstance(raw, str) else str(raw)
            if spec["point"] in ("oo", "-oo"):
                sample = [mpmath.mpf(10) ** 12 * (1 if spec["point"] == "oo" else -1)]
            else:
                p = mpmath.mpf(str(float(sp.N(_sym(spec["point"])))))
                eps = mpmath.mpf(10) ** -15
                sample = [p + eps] if spec["dir"] == "+" else [p - eps] if spec["dir"] == "-" else [p + eps, p - eps]
            vals = [f(s) for s in sample]
            got = _sym(got_raw)
            if got in (sp.oo, -sp.oo):
                return all(abs(v_) > 1e6 and (v_ > 0) == (got == sp.oo) for v_ in vals)
            g = float(sp.N(got))
            return all(abs(float(v_) - g) < 1e-5 * max(1, abs(g)) for v_ in vals)
        if k == "series":
            v = sp.Symbol(spec["var"])
            ser = _sym(raw)
            expr = _sym(spec["expr"])
            p0 = float(sp.N(_sym(spec["point"])))
            h = 0.02
            a, b = _num(expr, **{spec["var"]: p0 + h}), _num(ser, **{spec["var"]: p0 + h})
            if a is None or b is None:
                return None
            return abs(a - b) < 20 * h ** spec["order"] + 1e-9
        if k == "sum":
            v = sp.Symbol(spec["var"])
            got = float(sp.N(_sym(raw)))
            expr = sp.lambdify(v, _sym(spec["expr"]), "numpy")
            lo = int(spec["lo"])
            hi = 4000 if spec["hi"] == "oo" else int(spec["hi"])
            total = sum(float(sp.N(_sym(spec["expr"]).subs(v, i))) for i in range(lo, min(hi, 400) + 1)) if hi <= 400 else sum(float(expr(float(i))) for i in range(lo, hi + 1)) if spec["expr"].find("factorial") < 0 else sum(1.0 / math.factorial(i) for i in range(lo, 40))
            return _close(total, got, 2e-3 if spec["hi"] == "oo" else 1e-9)
        if k == "laplace":
            import mpmath
            mpmath.mp.dps = 30
            v = sp.Symbol(spec["var"])
            f = sp.lambdify(v, _sym(spec["expr"]), "mpmath")
            s_val = 9                                  # larger than any growth rate in the forms (exp(3t) at most)
            want = mpmath.quad(lambda t: f(t) * mpmath.e ** (-s_val * t), [0, 2, 8, 30])
            got = _num(_sym(raw), **{spec.get("svar", "s"): s_val})
            return None if got is None else _close(float(want), got, 1e-5)
        if k == "fourier":
            import scipy.integrate as si
            f = sp.lambdify(sp.Symbol(spec.get("xvar", "x")), _sym(spec["expr"]), "numpy")
            kk = 0.4
            want, _ = si.quad(lambda x: float(f(x)) * math.cos(2 * math.pi * kk * x), -np.inf, np.inf, limit=200)
            got = _num(_sym(raw), **{spec.get("kvar", "k"): kk})
            return None if got is None else _close(want, got, 1e-5)
        if k in ("grad", "div", "curl"):
            names = spec.get("vars", "x,y,z").split(",") if k == "grad" else list(spec["names"])
            pt = {n_: 0.7 + 0.3 * i for i, n_ in enumerate(names)}
            def fd(expr, var):
                h = 1e-5
                up = {**pt, var: pt[var] + h}
                dn = {**pt, var: pt[var] - h}
                return (_num(expr, **up) - _num(expr, **dn)) / (2 * h)
            comps = [_sym(c_) for c_ in spec["comps"]] if k != "grad" else None
            if k == "grad":
                f_ = _sym(spec["expr"])
                want = [fd(f_, n_) for n_ in names]
                got = [_num(_sym(r_), **pt) for r_ in raw]
                return len(got) >= len(want) and all(_close(a_, b_, 1e-5) for a_, b_ in zip(want, got))
            if k == "div":
                want = sum(fd(comps[i], n_) for i, n_ in enumerate(names))
                got = _num(_sym(raw), **pt)
                return None if got is None else _close(want, got, 1e-5)
            n0, n1, n2 = names
            want = [fd(comps[2], n1) - fd(comps[1], n2), fd(comps[0], n2) - fd(comps[2], n0), fd(comps[1], n0) - fd(comps[0], n1)]
            got = [_num(_sym(r_), **pt) for r_ in raw]
            return None if None in got else all(_close(a_, b_, 1e-5) for a_, b_ in zip(want, got))
        if k == "det":
            return _close(float(np.linalg.det(np.array(spec["M"], dtype=float))), float(sp.N(_sym(raw))), 1e-8)
        if k == "eig":
            want = sorted(np.linalg.eigvals(np.array(spec["M"], dtype=float)).real)
            got = sorted(float(sp.N(_sym(r_))) for r_ in raw)
            return len(got) <= len(want) and all(any(_close(g, w, 1e-6) for w in want) for g in got)
        if k == "inv":
            got = np.array(sp.Matrix(sp.sympify(raw)).evalf().tolist(), dtype=float)
            return bool(np.allclose(got @ np.array(spec["M"], dtype=float), np.eye(len(spec["M"])), atol=1e-8))
        if k == "mul":
            got = np.array(sp.Matrix(sp.sympify(raw)).evalf().tolist(), dtype=float)
            return bool(np.allclose(got, np.array(spec["A"], dtype=float) @ np.array(spec["B"], dtype=float), atol=1e-8))
        if k == "lin":
            got = np.array([float(sp.N(_sym(r_))) for r_ in raw])
            return bool(np.allclose(np.array(spec["A"], dtype=float) @ got, np.array(spec["b"], dtype=float), atol=1e-8))
        if k == "ode":
            return f"{spec.get('fn', 'y')}({spec.get('var', 'x')})" in raw
    except Exception as exc:                                  # an unexpected shape: no independent verdict
        return None
    return None


# ── generation ───────────────────────────────────────────────────────────────────────────

def seed_of(p: P, raw, verdict, i: int, seed: int) -> dict:
    steps = step(p.tool, p.args)
    return {"question": p.question, "plan": RP.make_plan(steps), "group": f"recipe:{p.recipe}:{i}:{seed}",
            "meta": {"source": "recipe", "family": "math", "recipe": p.recipe, "steps": 1, "classifier": "none", "no_rewrite": True,
                     "oracle": {True: "agrees", False: "DISAGREES", None: "unchecked"}[verdict],
                     "verified": [{"tool": p.tool, "value": raw if isinstance(raw, (str, int, float)) else json.dumps(raw)}]},
            "args_meta": []}


async def generate(session, per: int, seed: int, guard, holdout: bool = False, forbidden: set | None = None):
    """-> (seeds, report). `per` is the target per recipe weight unit; holdout=True uses only held-out wordings."""
    rng = random.Random(f"math:{seed}:{'h' if holdout else 't'}")
    seeds, seen = [], set(forbidden or ())
    report = {"made": {}, "drops": {}, "oracle_disagree": [], "unchecked": {}}
    weights = {name: w for name, (_, w) in GENERATORS.items()}
    total_weight = sum(weights.values())
    for name, (fn, w) in GENERATORS.items():
        want = max(8, int(per * w * len(GENERATORS) / total_weight)) if per else 0
        made = attempts = 0
        while made < want and attempts < want * 40:
            attempts += 1
            try:
                p = fn(rng, holdout)
            except Exception as exc:                          # a generator bug must not stop the run
                report["drops"][f"{name}: generator error {type(exc).__name__}"] = report["drops"].get(f"{name}: generator error {type(exc).__name__}", 0) + 1
                continue
            bad = unknown_functions(" ".join(str(v) for v in p.args.values()))
            if bad:
                report["drops"][f"generator bug (unknown function {sorted(bad)}): {p.recipe}"] = report["drops"].get(f"generator bug (unknown function {sorted(bad)}): {p.recipe}", 0) + 1
                continue
            key = D._norm(p.question)
            if key in seen:
                report["drops"]["duplicate"] = report["drops"].get("duplicate", 0) + 1
                continue
            res = await D.execute(session, p.tool, p.args, p.question, timeout_s=60, keep_raw=True)
            if res is None or res.get("raw") is None:
                k_ = f"server returned no value: {p.recipe}"
                report["drops"][k_] = report["drops"].get(k_, 0) + 1
                report.setdefault("no_value_examples", {}).setdefault(p.recipe, [])
                if len(report["no_value_examples"][p.recipe]) < 4:
                    report["no_value_examples"][p.recipe].append({"question": p.question, "args": p.args})
                continue
            raw = res["raw"]
            verdict = check(p.check, raw)
            if verdict is False:
                report["oracle_disagree"].append({"recipe": p.recipe, "question": p.question, "tool": p.tool, "args": p.args, "result": raw})
                report["drops"]["oracle disagrees"] = report["drops"].get("oracle disagrees", 0) + 1
                continue
            if guard is not None and not holdout:
                if guard.check(p.question, p.tool, raw if isinstance(raw, (int, float)) else None):
                    report["drops"]["benchmark overlap"] = report["drops"].get("benchmark overlap", 0) + 1
                    continue
            seen.add(key)
            made += 1
            if verdict is None:
                report["unchecked"][p.recipe] = report["unchecked"].get(p.recipe, 0) + 1
            seeds.append(seed_of(p, raw, verdict, len(seeds), seed))
            if holdout:
                seeds[-1]["expected"] = {"tool": p.tool, "args": p.args, "result": raw, "spec": p.check}
        report["made"][name] = made
        print(f"  {name:20s} {made:4d}/{want}", flush=True)
    return seeds, report


WILD = [  # hand-written, deliberately unlike the templates: (question, tool, args)
    ("what's d/dx of sin x times cos x", "differentiate_expr", {"expression": "sin(x)*cos(x)"}),
    ("how fast is x cubed changing at any x? i want the formula", "differentiate_expr", {"expression": "x**3"}),
    ("integral of e to the minus x from zero to infinity", "integrate_expr", {"expression": "exp(-x)", "lower": "0", "upper": "oo"}),
    ("area under the curve y = x squared between x = 1 and x = 3", "integrate_expr", {"expression": "x**2", "lower": "1", "upper": "3"}),
    ("find the antiderivative of 6x^2 + 4x", "integrate_expr", {"expression": "6*x**2 + 4*x"}),
    ("x^2 = 49, what is x", "solve_equation", {"equation": "x**2 = 49", "variable": "x"}),
    ("roots of x^2 + 2x - 8", "solve_equation", {"equation": "x**2 + 2*x - 8", "variable": "x"}),
    ("what do you get from simplifying (x^2 - 9)/(x - 3)", "simplify_expr", {"expression": "(x**2 - 9)/(x - 3)"}),
    ("expand (a+b)^2 for me", "expand_expression", {"expression": "(a + b)**2"}),
    ("factor x^2 + 7x + 12", "factor_expression", {"expression": "x**2 + 7*x + 12"}),
    ("limit of sin(3x)/x when x goes to 0", "compute_limit", {"expression": "sin(3*x)/x", "point": "0"}),
    ("lim x->infinity of 1/x", "compute_limit", {"expression": "1/x", "point": "oo"}),
    ("sum of the first 100 integers", "summation", {"expression": "k", "variable": "k", "lower": "1", "upper": "100"}),
    ("sum of 1/2^n from n = 1 to infinity", "summation", {"expression": "1/2**n", "variable": "n", "lower": "1", "upper": "oo"}),
    ("taylor series for cos x up to order 4", "series_expansion", {"expression": "cos(x)", "order": 4}),
    ("laplace transform of e^(-3t)", "laplace_transform", {"expression": "exp(-3*t)"}),
    ("det of [[3,1],[2,5]]", "matrix_determinant", {"matrix": "[[3, 1], [2, 5]]"}),
    ("eigenvalues of [[2,1],[1,2]]", "matrix_eigenvalues", {"matrix": "[[2, 1], [1, 2]]"}),
    ("invert [[1,2],[3,5]]", "matrix_inverse", {"matrix": "[[1, 2], [3, 5]]"}),
    ("[[1,0],[0,2]] times [[3,4],[5,6]]", "matrix_multiply", {"matrix_a": "[[1, 0], [0, 2]]", "matrix_b": "[[3, 4], [5, 6]]"}),
    ("solve for x and y: 2x + y = 5, x - y = 1", "solve_linear_system", {"matrix_a": "[[2, 1], [1, -1]]", "vector_b": "[5, 1]"}),
    ("gradient of x^2 y", "gradient", {"scalar_field": "x**2*y", "variables": "x,y"}),
    ("div of (x, y, z)", "divergence", {"vector_field": "x, y, z"}),
    ("curl of (0, 0, x*y)", "curl", {"vector_field": "0, 0, x*y"}),
    ("solve y'' = -4y", "solve_ode", {"equation": "y'' + 4*y"}),
]


def write_seeds(seeds: list, report: dict, out: Path, per: int, seed: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "seeds.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    (out / "oracle_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8", newline="\n")
    from collections import Counter
    manifest = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed, "per": per, "seeds": len(seeds),
                "by_recipe": report["made"], "tools": dict(Counter(s["plan"]["tool"] for s in seeds)),
                "oracle": dict(Counter(s["meta"]["oracle"] for s in seeds)), "drops": report["drops"],
                "oracle_disagreements": len(report["oracle_disagree"])}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    return manifest


async def _amain(a) -> int:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    guard = None if a.no_guard else D.BenchmarkGuard.from_repo()
    async with stdio_client(D.server_params(a.server_mem_gb)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await D.execute(session, "free_fall_time", {"height_m": 1.0}, "", timeout_s=120)
            if a.suite:
                suite, report = await generate(session, a.per, a.seed + 1000, None, holdout=True)
                items = []
                for s in suite:
                    items.append({"id": f"msuite_{len(items):04d}", "question": s["question"], "expected": s["expected"], "oracle": s["meta"]["oracle"],
                                  "recipe": s["meta"]["recipe"], "kind": "templated-holdout"})
                for q, tool, args in WILD:
                    res = await D.execute(session, tool, args, q, timeout_s=60, keep_raw=True)
                    items.append({"id": f"msuite_{len(items):04d}", "question": q, "expected": {"tool": tool, "args": args, "result": None if res is None else res.get("raw"), "spec": None},
                                  "oracle": "unchecked", "recipe": "wild", "kind": "wild"})
                a.suite.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "items": items, "report": {k: v for k, v in report.items() if k != "oracle_disagree"},
                                               "oracle_disagreements": report["oracle_disagree"]}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
                print(f"Wrote {a.suite}: {len(items)} suite questions ({len(suite)} templated held-out, {len(WILD)} wild); oracle disagreements: {len(report['oracle_disagree'])}")
                return 0
            forbidden = set()
            if a.holdout_file and a.holdout_file.exists():
                forbidden = {D._norm(i["question"]) for i in json.loads(a.holdout_file.read_text(encoding="utf-8"))["items"]}
            seeds, report = await generate(session, a.per, a.seed, guard, forbidden=forbidden)
    manifest = write_seeds(seeds, report, a.out, a.per, a.seed)
    print(f"\nWrote {a.out}: {manifest['seeds']} seeds; oracle {manifest['oracle']}; drops {manifest['drops']}")
    if report["oracle_disagree"]:
        print(f"!! {len(report['oracle_disagree'])} result(s) disagreed with the independent check: see oracle_report.json")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=D.REPO / "local-cache" / "router_math_v31")
    p.add_argument("--per", type=int, default=150, help="average problems per recipe")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suite", type=Path, help="write the held-out math suite here instead of training seeds")
    p.add_argument("--holdout-file", type=Path, default=D.REPO / "local-cache" / "router_math_suite.json",
                   help="questions of the held-out suite: never generated as training data")
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--server-mem-gb", type=float, default=6)
    return asyncio.run(_amain(p.parse_args()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
