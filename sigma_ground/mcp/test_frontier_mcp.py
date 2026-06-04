"""Tests for the STANDARD-physics frontier MCP tools (mcp/tools/frontier.py).

Pins textbook values for the four tools exposed via the MCP — Bekenstein-
Hawking black-hole thermodynamics, uniform-sphere gravitational binding
energy, the Unruh temperature, and the entanglement no-communication /
QKD / CHSH facts — plus the invalid-input declines.

The SSBM-supporting members of frontier.py (entanglements_to_pop_bubble,
holographic_matching_mass, baryon_vs_disc) are deliberately NOT exposed as
MCP tools, so they are not tested here.
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

from sigma_ground.mcp.tools import frontier as F


def _close(a, b, rel=0.02):
    return abs(a - b) <= rel * abs(b)


# ── Bekenstein-Hawking black-hole thermodynamics ─────────────────────────
def test_bh_entropy_solar_mass():
    d = F.bekenstein_hawking_entropy(1.989e30).to_dict()["value"]
    assert _close(d["schwarzschild_radius_m"], 2954.0, 0.01)   # Sun r_s ≈ 2.95 km
    assert d["entropy_k_B"] > 1e76                              # ~1e77 k_B
    assert _close(d["hawking_temperature_K"], 6.17e-8, 0.05)


def test_bh_entropy_invalid_declines():
    assert F.bekenstein_hawking_entropy(-1.0).to_dict()["value"] is None


# ── uniform-sphere gravitational binding energy ──────────────────────────
def test_earth_binding_energy():
    r = F.gravitational_binding_energy(5.972e24, 6.371e6).to_dict()
    assert _close(r["value"], 2.24e32, 0.02) and r["units"] == "J"


def test_binding_energy_invalid_declines():
    assert F.gravitational_binding_energy(-1.0, 1.0).to_dict()["value"] is None


# ── Unruh temperature ────────────────────────────────────────────────────
def test_unruh_temperature():
    r = F.unruh_temperature(9.8).to_dict()
    assert _close(r["value"], 3.97e-20, 0.05) and r["units"] == "K"


def test_unruh_invalid_declines():
    assert F.unruh_temperature(-1.0).to_dict()["value"] is None


# ── entanglement-channel facts (no-communication / QKD / CHSH) ───────────
def test_entanglement_cannot_signal_ftl():
    v = F.entanglement_channel("can entangled particles communicate "
                               "faster than light?").to_dict()["value"]
    assert v["primary"] is False
    assert v["can_signal_faster_than_light"] is False
    assert "no-communication" in v["verdict"].lower()


def test_entanglement_qkd_shares_key():
    v = F.entanglement_channel("can entanglement establish a secret "
                               "key (QKD)?").to_dict()["value"]
    assert "key" in str(v["primary"]).lower()
    assert v["verdict"].lower().startswith("yes")


def test_entanglement_chsh_tsirelson_bound():
    v = F.entanglement_channel("what is the maximum CHSH / Tsirelson "
                               "value?").to_dict()["value"]
    assert _close(v["primary"], 2.0 * math.sqrt(2.0), 1e-3)   # 2√2 ≈ 2.828
