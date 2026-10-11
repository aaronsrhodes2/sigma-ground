"""Regression sentinels for the Mentat tool fixes of 2026-10-10 (fix queue track D).

Each test checks the tool against an independent ground truth (measured data, a
closed form, or the library function the tool must agree with):

  D2 / A19  nuclear_binding_energy's SEMF estimate is the repo's ONE SEMF
            (field.interface.element.semf_binding_energy_MeV) and reproduces the
            AME2020 binding energies to < 1 % rms for A >= 12.
"""
from __future__ import annotations

import json
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sigma_ground.mcp.tools import nuclear as t_nuc
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
