"""Deckard regression tests — a name compiles to validated matter.

The cup is computed once (module scope) and reused; the SDF integrator's
agreement with the closed-form volumes is the self-check.
"""
import math
import os
import sys

_CANON = r"D:\Aaron\development\sigma-ground"
if os.path.isdir(_CANON) and _CANON not in sys.path:
    sys.path.insert(0, _CANON)

from sigma_ground.deckard import identify, research

CUP = identify("coffee cup", resolution=56)     # compiled once, reused


def test_cup_self_validates():
    """SDF integrator agrees with the closed-form ceramic+water mass & CoM."""
    assert CUP.validation["passed"], CUP.validation["note"]


def test_total_mass_plausible():
    assert 0.45 < CUP.mass_kg < 0.75, CUP.mass_kg


def test_com_on_axis_and_below_mid():
    cx, cy, cz = CUP.com_m
    assert abs(cx) < 1e-9 and abs(cy) < 1e-9          # on the symmetry axis
    assert 0.0 < cz < 0.0475                          # below mid-height


def test_material_at_resolves_every_layer():
    assert CUP.material_at(0.0398, 0.0, 0.05) == "glaze"     # outer skin
    assert CUP.material_at(0.037, 0.0, 0.05) == "ceramic"    # wall
    assert CUP.material_at(0.0, 0.0, 0.03) == "water"        # fill
    assert CUP.material_at(0.0, 0.0, 0.09) == "air"          # headspace
    assert CUP.material_at(0.0, 0.0, 0.003) == "ceramic"     # base


def test_water_volume_matches_closed_form():
    R_in = 0.040 - 0.0003 - 0.005
    h_fill = 0.80 * (0.095 - 0.007)
    water = next(L for L in CUP.layers if L.name == "water")
    assert abs(water.volume_m3 - math.pi * R_in**2 * h_fill) < 1e-9


def test_density_at_matches_layer():
    assert abs(CUP.density_at(0.0, 0.0, 0.03) - 998.0) < 1e-9   # water
    assert CUP.density_at(0.0, 0.0, 0.09) == 0.0                # air headspace


def test_unidentified_is_flagged_not_faked():
    # allow_llm=False exercises the deterministic fallback (no network).
    assert research("flux capacitor", allow_llm=False).identified is False
    assert identify("flux capacitor", resolution=24, allow_llm=False).identified is False
