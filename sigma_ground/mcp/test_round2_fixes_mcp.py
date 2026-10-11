"""Regression sentinels for the Mentat tool fixes of 2026-10-10 (fix queue track D).

Each test checks the tool against an independent ground truth (measured data, a
closed form, or the library function the tool must agree with):

  D2 / A19  nuclear_binding_energy's SEMF estimate is the repo's ONE SEMF
            (field.interface.element.semf_binding_energy_MeV) and reproduces the
            AME2020 binding energies to < 1 % rms for A >= 12.
  D3 / A20, A30  symbolic and math tools treat the named variable as a symbol even
            when it is E, I, N, S, O or Q (sympy's Euler number, imaginary unit,
            N(), S, O(), Q): checked against closed forms (d/dv v^2 = 2v, roots of
            v^2 - 4, the Basel sum pi^2/6, the Gaussian's Fourier transform, ...).
"""
from __future__ import annotations

import json
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

from sigma_ground.mcp.tools import nuclear as t_nuc
from sigma_ground.mcp.tools import symbolic as t_sym
from sigma_ground.mcp.tools import mathx as t_mx
from sigma_ground.field.interface.element import semf_binding_energy_MeV

_ISOTOPES = os.path.join(_ROOT, "sigma_ground", "inventory", "data", "isotopes.json")


def _ame2020_binding_MeV():
    with open(_ISOTOPES, encoding="utf-8") as f:
        data = json.load(f)
    return {(r["Z"], r["A"]): r["binding_energy_per_nucleon_kev"] * r["A"] / 1000.0
            for r in data if r["binding_energy_per_nucleon_kev"]}


# ── D2: one SEMF ─────────────────────────────────────────────────────────
def test_nuclear_tool_semf_is_the_element_semf():
    for (Z, A) in _ame2020_binding_MeV():
        if A < 12:
            continue
        v = t_nuc.nuclear_binding_energy(Z, A - Z).value["binding_energy_MeV"]
        assert v == semf_binding_energy_MeV(Z, A), (Z, A)


def test_nuclear_tool_semf_matches_ame2020():
    ame = _ame2020_binding_MeV()
    rel = [(t_nuc.nuclear_binding_energy(Z, A - Z).value["binding_energy_MeV"] - be) / be
           for (Z, A), be in ame.items() if A >= 12]
    rms = math.sqrt(sum(x * x for x in rel) / len(rel))
    assert len(rel) >= 130
    assert rms < 0.009, rms


def test_nuclear_tool_exact_path_unchanged():
    # He-4 from its measured atomic mass (AME2020 4.00260325413 u): 28.296 MeV
    v = t_nuc.nuclear_binding_energy(2, 2, 4.00260325413).value["binding_energy_MeV"]
    assert abs(v - 28.2957) < 1e-3


# ── D3: variables named like sympy objects ───────────────────────────────
_RESERVED = ["E", "I", "N", "S", "O", "Q"]


def _same(out, expected, name):
    import sympy as sp
    got = sp.sympify(out, locals={name: sp.Symbol(name)})
    return sp.simplify(got - expected) == 0


@pytest.mark.parametrize("name", _RESERVED + ["x"])
def test_symbolic_tools_take_the_named_variable(name):
    import sympy as sp
    v = sp.Symbol(name)
    d = t_sym.differentiate_expr(f"{name}**2", name).value
    assert _same(d, 2 * v, name), d
    roots = t_sym.solve_equation(f"{name}**2 - 4", name).value
    assert sorted(int(r) for r in roots) == [-2, 2], roots
    roots_eq = t_sym.solve_equation(f"{name}**2 = 9", name).value
    assert sorted(int(r) for r in roots_eq) == [-3, 3], roots_eq
    ind = t_sym.integrate_expr(name, name).value
    assert _same(ind, v ** 2 / 2, name), ind
    assert t_sym.integrate_expr(name, name, "0", "2").value == "2"


@pytest.mark.parametrize("name", _RESERVED + ["x"])
def test_math_tools_take_the_named_variable(name):
    import sympy as sp
    v = sp.Symbol(name)
    assert t_mx.compute_limit(f"{name}**2", name, "2").value == "4"
    assert t_mx.compute_limit(f"sin({name})/{name}", name, "0").value == "1"
    ser = t_mx.series_expansion(f"exp({name})", name, "0", 3).value
    assert _same(ser, 1 + v + v ** 2 / 2, name), ser
    assert _same(t_mx.summation(f"1/{name}**2", name, "1", "oo").value, sp.pi ** 2 / 6, name)
    grad = t_mx.gradient(f"{name}**2*y", f"{name},y").value
    assert _same(grad[0], 2 * v * sp.Symbol("y"), name) and _same(grad[1], v ** 2, name), grad
    ft = t_mx.fourier_transform(f"exp(-{name}**2)", name, "k").value
    k = sp.Symbol("k")
    assert sp.simplify(sp.sympify(ft) - sp.sqrt(sp.pi) * sp.exp(-sp.pi ** 2 * k ** 2)) == 0, ft


def test_other_reserved_names_are_disclosed():
    r = t_sym.solve_equation("E - m*c**2", "m")
    assert "Euler" in r.notes
    assert t_sym.differentiate_expr("x**2", "x").notes == ""
