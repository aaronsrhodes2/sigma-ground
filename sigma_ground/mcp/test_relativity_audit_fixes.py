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
    assert tr.value == pytest.approx(ref, rel=1e-12), (beta, theta, tr.value, ref)


def test_doppler_transverse_is_a_redshift_equal_to_gamma():
    # beta = 0.6 -> gamma = 1.25 exactly; observer-frame transverse = gamma
    tr = t_rel.doppler_shift_factor(0.6 * sc.c, 90.0)
    assert tr.value == pytest.approx(1.25, rel=1e-12)
    assert tr.value > 1.0                       # redshift, not a blueshift


def test_doppler_head_on_recession_closed_form():
    # theta = 0: lambda ratio = sqrt((1 + b) / (1 - b)); b = 0.6 -> 2
    tr = t_rel.doppler_shift_factor(0.6 * sc.c, 0.0)
    assert tr.value == pytest.approx(2.0, rel=1e-12)
    tr = t_rel.doppler_shift_factor(0.6 * sc.c, 180.0)
    assert tr.value == pytest.approx(0.5, rel=1e-12)


def test_doppler_rejects_non_finite_input():
    assert t_rel.doppler_shift_factor(float("nan"), 0.0).value is None
    assert t_rel.doppler_shift_factor(1.0e7, float("inf")).value is None
    assert t_rel.doppler_shift_factor(sc.c, 0.0).value is None
