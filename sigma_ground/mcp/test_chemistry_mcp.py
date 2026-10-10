"""Tests for the chemistry MCP tools (mcp/tools/chemistry.py).

Pins verified textbook values and the graceful-decline behavior (bad keys list
the valid set; never crash). All tool functions return a JSON-ready dict.
"""
import math
import os
import sys

# Import sigma_ground from THIS tree (worktree-portable): walk up from this file
# (…/sigma_ground/mcp/<this>) to the repo root, rather than a hardcoded path —
# so the test validates the worktree it lives in, never a shadowing sibling tree.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sigma_ground.mcp.tools import chemistry as C


def _close(a, b, rel=0.02):
    return abs(a - b) <= rel * abs(b)


def test_bond_energy():
    r = C.bond_energy("H", "Cl")
    assert _close(r["value"], 4.44, 0.05) and r["units"] == "eV"


def test_bond_angle_tetrahedral():
    assert abs(C.bond_angle(4, 0)["value"] - 109.47) < 0.1


def test_weak_acid_ph_matches_textbook():
    assert _close(C.weak_acid_ph("acetic_acid", 0.1)["value"], 2.87, 0.02)


def test_weak_acid_ph_handles_every_species_in_its_table():
    # the tool used to run bases through the weak-ACID equation: 1.45 mM KOH
    # came back as pH 15.0 and 0.1 M ammonia as an acidic 5.12
    for key, conc, want in [("potassium_hydroxide", 1.45e-3, 11.156),
                            ("sodium_hydroxide", 0.01, 11.995),
                            ("ammonia", 0.1, 11.117),
                            ("pyridine", 4.01e-3, 8.412),
                            ("hydrochloric_acid", 0.01, 2.000),
                            ("acetic_acid", 0.1, 2.881),
                            ("phenol", 1e-6, 6.995)]:
        r = C.weak_acid_ph(key, conc)
        assert abs(r["value"] - want) < 2e-3, (key, r["value"], want)
    r = C.weak_acid_ph("potassium_hydroxide", 1.45e-3)
    assert "strong base" in r["formula"]


def test_weak_acid_ph_declines_negative_concentration():
    r = C.weak_acid_ph("acetic_acid", -0.1)
    assert r["value"] is None


def test_buffer_ph_is_pka_at_unit_ratio():
    assert _close(C.buffer_ph("acetic_acid", 1.0)["value"], 4.76, 0.02)


def test_daniell_cell_potential():
    assert _close(C.cell_potential("copper", "zinc")["value"], 1.10, 0.02)


def test_electrolysis_faraday():
    g = C.electrolysis_mass(0.06355, 10.0, 3600.0, 2)["value"] * 1000
    assert _close(g, 11.86, 0.01)


def test_freezing_point_depression():
    assert _close(C.freezing_point_depression(1.0)["value"], 1.86, 0.02)


def test_molar_solubility_agcl():
    assert _close(C.molar_solubility("silver_chloride")["value"], 1.33e-5, 0.05)


# ── graceful declines (never crash on bad keys) ─────────────────────────
def test_bad_acid_declines_with_keys():
    r = C.weak_acid_ph("dragon_acid", 0.1)
    assert r["value"] is None and "acetic_acid" in r["notes"]


def test_bad_electrode_declines():
    r = C.cell_potential("copper", "unobtainium")
    assert r["value"] is None


def test_unsupported_atom_declines():
    r = C.bond_energy("U", "Pu")
    assert r["value"] is None
