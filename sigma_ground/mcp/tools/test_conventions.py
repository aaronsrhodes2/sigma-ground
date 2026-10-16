"""Tests for the explicit conventions of the math tools (2026-10-15): Fourier transform, series order, constants of integration.

The baselines (misc/BASELINES.md) showed every model losing 11-13 of 100 questions to the oracle's choices. The defaults below are the previous behaviour, so
every existing call is unchanged; the other readings are one argument away and the result always says which one it used.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import sympy as sp

from sigma_ground.mcp.tools import mathx as M

REPO = Path(__file__).resolve().parents[3]
k = sp.Symbol("k")


def _expr(result):
    assert result.value is not None, result.notes
    return sp.sympify(result.value, locals={"k": k})


def _same(a, b) -> bool:
    return sp.simplify(a - b) == 0


# ── Fourier ────────────────────────────────────────────────────────────────────────────────────────

def test_the_three_fourier_conventions_on_a_gaussian():
    ordinary = _expr(M.fourier_transform("e^(-x^2)", "x", "k", "ordinary"))
    angular = _expr(M.fourier_transform("e^(-x^2)", "x", "k", "angular"))
    unitary = _expr(M.fourier_transform("e^(-x^2)", "x", "k", "unitary"))
    assert _same(ordinary, sp.sqrt(sp.pi) * sp.exp(-sp.pi**2 * k**2))            # e^(-2 pi i k x): sympy's own
    assert _same(angular, sp.sqrt(sp.pi) * sp.exp(-k**2 / 4))                      # e^(-i k x): what every baseline model wrote
    assert _same(unitary, sp.sqrt(2) * sp.exp(-k**2 / 4) / 2)                      # the same over sqrt(2 pi)


def test_the_conventions_are_exactly_related():
    f = "e^(-2*x^2)"
    ordinary = _expr(M.fourier_transform(f, "x", "k", "ordinary"))
    angular = _expr(M.fourier_transform(f, "x", "k", "angular"))
    unitary = _expr(M.fourier_transform(f, "x", "k", "unitary"))
    for w in (0.3, 1.7, 4.0):
        assert float(angular.subs(k, w)) == pytest.approx(float(ordinary.subs(k, w / (2 * sp.pi))), rel=1e-12)
        assert float(unitary.subs(k, w)) == pytest.approx(float(angular.subs(k, w)) / float(sp.sqrt(2 * sp.pi)), rel=1e-12)


def test_the_default_is_the_previous_behaviour():
    direct = sp.fourier_transform(sp.exp(-sp.Symbol("x") ** 2), sp.Symbol("x"), k)
    assert _same(_expr(M.fourier_transform("e^(-x^2)")), direct)
    assert M.fourier_transform("e^(-x^2)").value == M.fourier_transform("e^(-x^2)", "x", "k", "ordinary").value


@pytest.mark.parametrize("written", ["angular", "Angular", "angular frequency", "Angular_Frequency", "omega", "ω"])
def test_the_angular_convention_can_be_written_several_ways(written):
    assert M.fourier_transform("e^(-x^2)", "x", "k", written).inputs["convention"] == "angular"


def test_the_result_says_which_convention_it_used():
    for conv, fragment in (("ordinary", "e^(-2πikx)"), ("angular", "e^(-ikx)"), ("unitary", "1/√(2π)")):
        r = M.fourier_transform("e^(-x^2)", "x", "k", conv)
        assert f"convention: {conv}" in r.notes and fragment in r.notes
        assert r.interpreted_as.endswith(f"[{conv}]")


def test_an_unknown_convention_is_refused_with_the_options():
    r = M.fourier_transform("e^(-x^2)", "x", "k", "bogus")
    assert r.value is None and "ordinary" in r.notes and "angular" in r.notes and "unitary" in r.notes


# ── series ─────────────────────────────────────────────────────────────────────────────────────────

def test_series_default_is_sympys_and_inclusive_keeps_the_last_term():
    x = sp.Symbol("x")
    assert _same(sp.sympify(M.series_expansion("cos(x)", "x", "0", 4).value), sp.series(sp.cos(x), x, 0, 4).removeO())     # unchanged
    assert _same(sp.sympify(M.series_expansion("cos(x)", "x", "0", 4, "inclusive").value), 1 - x**2 / 2 + x**4 / 24)
    assert _same(sp.sympify(M.series_expansion("cos(x)", "x", "0", 4).value), 1 - x**2 / 2)


@pytest.mark.parametrize("expression,point", [("exp(x)", "0"), ("1/(1+x)", "0"), ("sin(x)", "0"), ("ln(x)", "1")])
def test_inclusive_order_n_is_exclusive_order_n_plus_one(expression, point):
    for n in (2, 3, 5):
        a = sp.sympify(M.series_expansion(expression, "x", point, n, "inclusive").value)
        b = sp.sympify(M.series_expansion(expression, "x", point, n + 1, "exclusive").value)
        assert _same(a, b)


def test_series_says_which_convention_and_refuses_an_unknown_one():
    r = M.series_expansion("exp(x)", "x", "0", 3, "inclusive")
    assert "order convention: inclusive" in r.notes and r.inputs["convention"] == "inclusive"
    assert "order convention: exclusive" in M.series_expansion("exp(x)", "x", "0", 3).notes
    bad = M.series_expansion("exp(x)", "x", "0", 3, "nonsense")
    assert bad.value is None and "exclusive" in bad.notes and "inclusive" in bad.notes


# ── constants of integration ───────────────────────────────────────────────────────────────────────

def test_the_ode_and_the_indefinite_integral_state_their_constants():
    assert "arbitrary constants of integration" in M.solve_ode("y' - 6*y").notes
    from sigma_ground.mcp.tools import symbolic as S
    assert "constant of integration" in S.integrate_expr("x^2").notes
    assert "constant of integration" not in S.integrate_expr("x^2", None, "0", "1").notes          # a definite integral has none


# ── in the schema ──────────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server():
    pytest.importorskip("mcp")
    spec = importlib.util.spec_from_file_location("mentat_contract_census", REPO / "tools" / "mentat_contract_census.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mentat_contract_census"] = mod
    spec.loader.exec_module(mod)
    return mod, mod.build_server()


@pytest.mark.parametrize("tool", ["fourier_transform", "series_expansion"])
def test_the_convention_is_a_documented_optional_parameter(server, tool):
    mod, srv = server
    schema = srv._tool_manager.get_tool(tool).parameters
    assert "convention" in schema["properties"] and "convention" not in schema.get("required", [])
    assert "convention" in schema["properties"]["convention"].get("description", "")


def test_a_call_through_the_server_uses_the_convention(server):
    mod, srv = server
    r = mod.call(srv, "fourier_transform", {"expression": "e^(-x^2)", "convention": "angular"})["result"]
    assert sp.simplify(sp.sympify(r["value"], locals={"k": k}) - sp.sqrt(sp.pi) * sp.exp(-k**2 / 4)) == 0
    r = mod.call(srv, "series_expansion", {"expression": "cos(x)", "order": 4, "convention": "inclusive"})["result"]
    assert "x**4/24" in r["value"]
