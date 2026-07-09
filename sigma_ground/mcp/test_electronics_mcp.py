"""Tests for the electronics / solid-state MCP tools (mcp/tools/electronics.py).

Pins verified textbook values (copper ρ=1.68e-8 Ω·m, Cu mean free path ≈39 nm,
Si E_g≈1.12 eV, parallel-plate C=ε₀A/d exact) and — critically — the graceful
declines. 'silicon' lives in BOTH the metal-transport and semiconductor tables,
so the metal Drude / free-electron tools must REFUSE it (redirect) rather than
return a confidently-wrong number. All tool functions return a JSON-ready dict.
"""
import os
import sys

# Import sigma_ground from THIS tree (worktree-portable): walk up from this file
# (…/sigma_ground/mcp/<this>) to the repo root, rather than a hardcoded path —
# so the test validates the worktree it lives in, never a shadowing sibling tree.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sigma_ground.mcp.tools import electronics as E


def _close(a, b, rel=0.02):
    return abs(a - b) <= rel * abs(b)


# ── metal transport: pinned textbook values ─────────────────────────────
def test_copper_resistivity():
    r = E.electrical_resistivity("copper")
    assert _close(r["value"], 1.68e-8, 0.03)
    assert r["provenance_tag"] == "DERIVED"


def test_copper_mean_free_path_about_39nm():
    r = E.electron_mean_free_path("copper")
    assert _close(r["value"], 39e-9, 0.25)        # texts cite ~39 nm at 300 K


def test_copper_free_electron_density():
    r = E.free_electron_density("copper")
    assert _close(r["value"], 8.5e28, 0.03)       # ~8.5e28 m⁻³


def test_copper_hall_coefficient_negative():
    r = E.hall_coefficient("copper")
    assert r["value"] < 0                          # electrons → negative R_H
    assert _close(r["value"], -7.4e-11, 0.10)


def test_copper_mobility_is_real():
    r = E.carrier_mobility("copper")
    assert r["value"] is not None and r["value"] > 0


# ── semiconductors ───────────────────────────────────────────────────────
def test_silicon_band_gap_about_1_12eV():
    r = E.semiconductor_band_gap("silicon")
    assert _close(r["value"], 1.12, 0.02) and r["units"] == "eV"


def test_silicon_intrinsic_density_order_of_magnitude():
    r = E.intrinsic_carrier_density("silicon")
    assert 1e15 < r["value"] < 5e16                # n_i ~1e16 m⁻³ at 300 K


def test_silicon_pn_built_in_voltage():
    r = E.pn_built_in_voltage("silicon", 1e22, 1e22)
    assert _close(r["value"], 0.74, 0.08) and r["units"] == "V"


def test_silicon_depletion_width_submicron():
    r = E.depletion_width("silicon", 1e22, 1e22)
    assert 1e-7 < r["value"] < 1e-6


# ── junctions & capacitance ──────────────────────────────────────────────
def test_diode_current_shockley():
    r = E.diode_current(1e-12, 0.6)
    assert _close(r["value"], 0.012, 0.05) and r["units"] == "A"


def test_parallel_plate_capacitance_exact():
    # C = ε₀ A / d = 8.854e-12 · 1 / 1e-3 = 8.854e-9 F (vacuum)
    r = E.parallel_plate_capacitance(1.0, 1e-3, 1.0)
    assert _close(r["value"], 8.854e-9, 0.01) and r["units"] == "F"


# ── graceful declines (never confidently wrong) ──────────────────────────
def test_unknown_metal_declines():
    r = E.electrical_resistivity("unobtainium")
    assert r["value"] is None
    assert "metal" in r["notes"].lower()


def test_metal_on_semiconductor_tool_declines():
    # 'copper' is not a semiconductor → band-gap tool must refuse.
    assert E.semiconductor_band_gap("copper")["value"] is None


def test_semiconductor_refused_by_metal_transport_tools():
    # 'silicon' is in BOTH tables; metal-model tools must redirect, not answer.
    for fn in (E.electrical_resistivity, E.carrier_mobility,
               E.hall_coefficient, E.electron_mean_free_path,
               E.free_electron_density):
        r = fn("silicon")
        assert r["value"] is None, f"{fn.__name__} should decline silicon"
        assert "semiconductor" in r["notes"].lower()
