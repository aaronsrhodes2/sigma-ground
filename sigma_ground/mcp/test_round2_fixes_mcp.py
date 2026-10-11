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
  D4 / A21  elastic_analysis's moduli_from_lame and yield check use the requested
            material: iron K, E, G against the literature (Wikipedia 'Iron': 170, 211,
            82 GPa); every material's Lame moduli obey the isotropic identities with
            the same call's hydrostatic, uniaxial and shear moduli; annealed copper
            (0.2 % yield <= 140 MPa, C10200 O-temper datasheets) yields at 200 MPa.
  D5 / A22  one solar mass: every place that carries M_sun gives G*M equal to the
            IAU 2015 Resolution B3 nominal (GM)sun = 1.3271244e20 m^3/s^2 (exact).
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
from sigma_ground.mcp.tools import materials_strength as t_ms
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


# ── D4: elastic_analysis per material ────────────────────────────────────
def test_elastic_analysis_moduli_are_the_materials():
    v = t_ms.elastic_analysis("iron", 0.001)["value"]["moduli_from_lame"]
    # Wikipedia 'Iron' infobox: K = 170, E = 211, G = 82 GPa, nu = 0.29
    assert abs(v["K_pa"] / 170e9 - 1) < 0.03, v
    assert abs(v["E_pa"] / 211e9 - 1) < 0.03, v
    assert abs(v["G_pa"] / 82e9 - 1) < 0.03, v
    assert abs(v["poisson_ratio"] - 0.29) < 0.01, v


@pytest.mark.parametrize("mat", ["iron", "copper", "aluminum", "lead", "tungsten"])
def test_elastic_analysis_lame_moduli_match_the_same_calls_moduli(mat):
    eps = 0.001
    v = t_ms.elastic_analysis(mat, eps)["value"]
    m = v["moduli_from_lame"]
    # isotropic linear elasticity: sigma_hyd = -K eps_v, sigma = E eps, tau = G gamma
    assert abs(m["K_pa"] / (-v["hydrostatic_stress_Pa"] / eps) - 1) < 1e-9
    assert abs(m["E_pa"] / (v["uniaxial_stress_Pa"] / eps) - 1) < 1e-9
    assert abs(m["G_pa"] / (v["shear_stress_Pa"] / eps) - 1) < 1e-9


def test_elastic_analysis_yield_check_uses_the_materials_yield_stress():
    cu = t_ms.elastic_analysis("copper")["value"]
    assert cu["is_yielded_at_200MPa"] is True          # annealed Cu yields below 140 MPa
    assert cu["yield_stress_Pa"] <= 140e6
    w = t_ms.elastic_analysis("tungsten")["value"]
    assert w["is_yielded_at_200MPa"] is False


# ── D5: one solar mass ───────────────────────────────────────────────────
def test_every_solar_mass_reproduces_the_iau_2015_gm():
    from sigma_ground.field import constants as K
    from sigma_ground.mcp.tools import astronomy, orbital, units
    from sigma_ground.materia import bodies
    gm_iau = 1.3271244e20          # IAU 2015 Resolution B3 nominal (GM)sun, exact
    masses = {
        "field.constants.M_SUN_KG": K.M_SUN_KG,
        "solar_system_body('sun')": astronomy.solar_system_body("sun").value["mass_kg"],
        "materia.bodies sun": bodies._BODIES["sun"].mass_kg,
        "mcp.tools.orbital": orbital._M_SUN,
        "convert(1 solar_mass, kg)": units.convert(1.0, "solar_mass", "kg").value,
    }
    for where, m in masses.items():
        assert abs(K.G * m / gm_iau - 1.0) < 1e-15, (where, m)
    assert len(set(masses.values())) == 1, masses
