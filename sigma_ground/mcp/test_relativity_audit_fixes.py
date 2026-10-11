"""Regression tests for the 2026-10-10 relativity-tool audit fixes (track B).

Every expected value here comes from an INDEPENDENT ground truth (an explicit
Lorentz boost, a closed form evaluated in exact or extended arithmetic, or a
CODATA value), never from the formula the code under test uses.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")
import scipy.constants as sc  # noqa: E402  CODATA reference values

from sigma_ground.mcp.tools import relativity as t_rel  # noqa: E402


# ── A37: doppler_shift_factor off-axis ─────────────────────────────────────

def _boost_wavelength_ratio(beta_signed: float, theta_deg: float) -> float:
    """lambda_obs / lambda_emit from an explicit 4x4 Lorentz boost (c = 1).

    Observer frame S: the source sits on the +x axis, so the line of sight
    from the observer to the source is n = +x. The source moves with
    velocity beta_signed * (cos theta, sin theta, 0). The received photon
    travels from the source to the observer, k = E (1, -n). Boosting k into
    the source's rest frame gives the emitted energy E'; the wavelength ratio
    is E' / E.
    """
    th = math.radians(theta_deg)
    b = np.array([beta_signed * math.cos(th), beta_signed * math.sin(th), 0.0])
    b2 = float(b @ b)
    g = 1.0 / math.sqrt(1.0 - b2)
    L = np.eye(4)
    L[0, 0] = g
    L[0, 1:] = -g * b
    L[1:, 0] = -g * b
    if b2 > 0:
        L[1:, 1:] += (g - 1.0) * np.outer(b, b) / b2
    k_obs = np.array([1.0, -1.0, 0.0, 0.0])
    k_src = L @ k_obs
    # still a null vector after the boost (sanity of the reference itself)
    assert abs(k_src[0] ** 2 - k_src[1:] @ k_src[1:]) < 1e-12
    return float(k_src[0] / k_obs[0])


@pytest.mark.parametrize("beta", [0.1, 0.6, 0.9, -0.6])
@pytest.mark.parametrize("theta", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
def test_doppler_shift_factor_matches_lorentz_boost(beta, theta):
    tr = t_rel.doppler_shift_factor(beta * sc.c, theta)
    ref = _boost_wavelength_ratio(beta, theta)
    assert tr.value == pytest.approx(ref, rel=1e-12, abs=0.0), (beta, theta, tr.value, ref)


def test_doppler_transverse_is_a_redshift_equal_to_gamma():
    # beta = 0.6 -> gamma = 1.25 exactly; observer-frame transverse = gamma
    tr = t_rel.doppler_shift_factor(0.6 * sc.c, 90.0)
    assert tr.value == pytest.approx(1.25, rel=1e-12, abs=0.0)
    assert tr.value > 1.0                       # redshift, not a blueshift


def test_doppler_head_on_recession_closed_form():
    # theta = 0: lambda ratio = sqrt((1 + b) / (1 - b)); b = 0.6 -> 2
    tr = t_rel.doppler_shift_factor(0.6 * sc.c, 0.0)
    assert tr.value == pytest.approx(2.0, rel=1e-12, abs=0.0)
    tr = t_rel.doppler_shift_factor(0.6 * sc.c, 180.0)
    assert tr.value == pytest.approx(0.5, rel=1e-12, abs=0.0)


def test_doppler_rejects_non_finite_input():
    assert t_rel.doppler_shift_factor(float("nan"), 0.0).value is None
    assert t_rel.doppler_shift_factor(1.0e7, float("inf")).value is None
    assert t_rel.doppler_shift_factor(sc.c, 0.0).value is None


# ── A43: procedure relativistic_particle ───────────────────────────────────

def _cascade(ke_eV, particle):
    from sigma_ground.mcp.procedures import relativistic_particle
    r = relativistic_particle(ke_eV, particle)
    return {s.label: s.value for s in r.steps}, r


def _codata_rest_eV(name):
    """CODATA m c^2 in eV (scipy, independent of the repo's mass tables)."""
    return sc.physical_constants[f"{name} mass energy equivalent in MeV"][0] * 1e6


def _pc_invariant_eV(ke_eV, mc2_eV):
    """pc from the invariant E^2 - (mc^2)^2 in 50-digit arithmetic."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    E = mp.mpf(ke_eV) + mp.mpf(mc2_eV)
    return float(mp.sqrt(E * E - mp.mpf(mc2_eV) ** 2))


def test_relativistic_particle_alpha_uses_the_alpha_mass():
    rows, r = _cascade(5e6, "alpha")
    mc2 = _codata_rest_eV("alpha particle")                    # 3727.379 MeV
    assert rows["Lorentz factor γ"] == pytest.approx(1.0 + 5e6 / mc2, rel=1e-9, abs=0.0)
    pc_ref = _pc_invariant_eV(5e6, mc2)                         # 1.931e8 eV
    assert rows["Momentum (pc)"] == pytest.approx(pc_ref, rel=1e-8, abs=0.0)
    lam_ref = sc.h * sc.c / (pc_ref * sc.e)                     # 6.420e-15 m
    assert rows["de Broglie wavelength"] == pytest.approx(lam_ref, rel=1e-8, abs=0.0)
    # one mass for the whole cascade: (gamma^2 - 1) (mc^2)^2 = (pc)^2
    g, pc = rows["Lorentz factor γ"], rows["Momentum (pc)"]
    assert (g * g - 1.0) * mc2 ** 2 == pytest.approx(pc ** 2, rel=1e-7, abs=0.0)


@pytest.mark.parametrize("particle", ["pion", "tau", "", "kaon"])
def test_relativistic_particle_unknown_particle_raises(particle):
    with pytest.raises(ValueError, match="unknown particle"):
        _cascade(5e6, particle)


@pytest.mark.parametrize("ke", [0.0, -1.0, -1e6, float("nan"), float("inf")])
def test_relativistic_particle_rejects_bad_kinetic_energy(ke):
    with pytest.raises(ValueError, match="kinetic_energy_eV"):
        _cascade(ke, "electron")


@pytest.mark.parametrize("ke_eV", [1e-7, 3e-7, 1e-5])
def test_relativistic_particle_ultracold_neutron_no_cancellation(ke_eV):
    rows, _ = _cascade(ke_eV, "neutron")
    # non-relativistic de Broglie, exact to KE/(4 m c^2) ~ 1e-17 here
    lam_ref = sc.h / math.sqrt(2.0 * sc.m_n * ke_eV * sc.e)     # 100 neV: 9.0446e-8 m
    assert rows["de Broglie wavelength"] == pytest.approx(lam_ref, rel=1e-8, abs=0.0)
    pc_ref = _pc_invariant_eV(ke_eV, _codata_rest_eV("neutron"))
    assert rows["Momentum (pc)"] == pytest.approx(pc_ref, rel=1e-8, abs=0.0)


def test_relativistic_particle_electron_unchanged_regime():
    rows, _ = _cascade(1e6, "electron")
    mc2 = _codata_rest_eV("electron")
    assert rows["Lorentz factor γ"] == pytest.approx(1.0 + 1e6 / mc2, rel=1e-9, abs=0.0)
    assert rows["Momentum (pc)"] == pytest.approx(_pc_invariant_eV(1e6, mc2), rel=1e-8, abs=0.0)


# ── B4: cancellation in the MCP faces of (γ−1) m c² and the weak-field z ────

@pytest.mark.parametrize("m, v, K", [(1.0, 1.0, 0.5), (1500.0, 30.0, 6.75e5),
                                     (70.0, 10.0, 3500.0)])
def test_relativistic_energy_analysis_everyday_kinetic_energy(m, v, K):
    from sigma_ground.mcp.tools.relativity_spectra import relativistic_energy_analysis
    out = relativistic_energy_analysis(m, v)["value"]
    # classical 1/2 m v^2; the relativistic correction 3 beta^2/4 is < 1e-14 here
    assert out["kinetic_energy_J"] == pytest.approx(K, rel=1e-13, abs=0.0)


@pytest.mark.parametrize("h", [1.0, 0.01])
def test_mcp_gravitational_redshift_difference_at_earth_surface(h):
    from sigma_ground.mcp.tools import gr as t_gr
    M, R = 5.9722e24, 6.371e6
    dz = t_gr.gravitational_redshift(M, R).value - t_gr.gravitational_redshift(M, R + h).value
    # weak field: dz = G M h / (c^2 R (R + h)) to relative O(r_s/R) = 1e-9
    dz_ref = sc.G * M * h / (sc.c ** 2 * R * (R + h))
    assert dz == pytest.approx(dz_ref, rel=1e-6, abs=0.0)                 # 1.09e-16 per metre
