"""Tests for the extended-math MCP tools (mcp/tools/mathx.py).

sympy is the oracle, so these pin exact symbolic results (determinants,
eigenvalues, limits, transforms, sums, ODEs, vector calculus) and the
graceful decline on malformed input. Values are strings on the wire.
"""
import os
import sys

# Import sigma_ground from THIS tree (worktree-portable): walk up from this file
# (…/sigma_ground/mcp/<this>) to the repo root, rather than a hardcoded path —
# so the test validates the worktree it lives in, never a shadowing sibling tree.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sigma_ground.mcp.tools import mathx as M


def _n(s):
    """Normalize sympy string output (drop spaces) for stable comparison."""
    return str(s).replace(" ", "")


# ── linear algebra ───────────────────────────────────────────────────────
def test_determinant():
    assert M.matrix_determinant("[[1,2],[3,4]]").to_dict()["value"] == "-2"


def test_eigenvalues():
    v = M.matrix_eigenvalues("[[2,0],[0,3]]").to_dict()["value"]
    assert set(v) == {"2", "3"}


def test_inverse():
    v = M.matrix_inverse("[[1,2],[3,4]]").to_dict()["value"]
    assert _n(v) == "[[-2,1],[3/2,-1/2]]"


def test_multiply_identity():
    v = M.matrix_multiply("[[1,0],[0,1]]", "[[5,6],[7,8]]").to_dict()["value"]
    assert _n(v) == "[[5,6],[7,8]]"


def test_solve_linear_system():
    v = M.solve_linear_system("[[2,1],[1,3]]", "[1,2]").to_dict()["value"]
    assert v == ["1/5", "3/5"]


# ── limits, series, sums ─────────────────────────────────────────────────
def test_limit_sinc():
    assert M.compute_limit("sin(x)/x").to_dict()["value"] == "1"


def test_series_exp():
    v = _n(M.series_expansion("exp(x)", order=4).to_dict()["value"])
    assert "x**3/6" in v and "x**2/2" in v and v.endswith("+x+1")


def test_summation_basel():
    assert _n(M.summation("1/n**2", upper="oo").to_dict()["value"]) == "pi**2/6"


# ── transforms / algebra ─────────────────────────────────────────────────
def test_laplace():
    assert _n(M.laplace_transform("exp(-2*t)").to_dict()["value"]) == "1/(s+2)"


def test_factor():
    assert _n(M.factor_expression("x**2-1").to_dict()["value"]) == "(x-1)*(x+1)"


def test_expand():
    assert _n(M.expand_expression("(x+1)**2").to_dict()["value"]) == "x**2+2*x+1"


def test_solve_ode_harmonic():
    v = M.solve_ode("y'' + y").to_dict()["value"]
    assert "sin(x)" in v and "cos(x)" in v


# ── vector calculus ──────────────────────────────────────────────────────
def test_gradient():
    assert M.gradient("x**2+y**2", "x,y").to_dict()["value"] == ["2*x", "2*y"]


def test_divergence():
    assert M.divergence("x,y,z", "x,y,z").to_dict()["value"] == "3"


def test_curl_rigid_rotation():
    assert M.curl("-y,x,0", "x,y,z").to_dict()["value"] == ["0", "0", "2"]


# ── arithmetic helper + decline ──────────────────────────────────────────
def test_percent_of():
    assert abs(M.percent_of(2, 60).to_dict()["value"] - 1.2) < 1e-9


def test_nonsquare_matrix_declines():
    # determinant of a non-square matrix is undefined → graceful None.
    assert M.matrix_determinant("[[1,2,3],[4,5,6]]").to_dict()["value"] is None
