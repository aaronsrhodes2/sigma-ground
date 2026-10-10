"""MCP rotation / atomic angular-momentum tools (mcp/tools/rotation.py).

atomic_angular_momentum passed total_j straight to the library: j = 3e8
enumerated 6e8 m_j values until the server was OOM-killed at 13.7 GB, and the
spin-orbit energy was computed for any J although the tool's term is 2D
(L=2, S=1/2), whose only levels are J = 3/2 and 5/2.
"""
import json
import time

from sigma_ground.mcp.tools import rotation as R


def _strict(r):
    json.dumps(r, allow_nan=False)          # NaN/inf aren't JSON
    return r


def test_huge_j_answers_without_enumerating():
    t = time.time()
    r = _strict(R.atomic_angular_momentum(3e8))
    assert time.time() - t < 2.0
    v = r["value"]
    assert v["m_j_values"] is None and v["multiplicity"] == 600_000_001
    assert "not listed" in r["notes"]
    assert v["magnitude_J_s"] > 0


def test_non_physical_j_declines():
    for j in (1.3, -1.0, float("nan"), float("inf")):
        r = _strict(R.atomic_angular_momentum(j))
        assert r["value"] is None, j
        assert "half-integer" in r["notes"]


def test_non_finite_spin_orbit_constant_declines():
    r = _strict(R.atomic_angular_momentum(1.5, float("nan")))
    assert r["value"] is None


def test_levels_of_the_2D_term():
    # E_SO = (A/2)[J(J+1) - L(L+1) - S(S+1)] with L=2, S=1/2, A=0.05 eV
    low = _strict(R.atomic_angular_momentum(1.5, 0.05))["value"]
    high = _strict(R.atomic_angular_momentum(2.5, 0.05))["value"]
    assert abs(low["spin_orbit_energy_eV"] - (-0.075)) < 1e-12
    assert abs(high["spin_orbit_energy_eV"] - 0.05) < 1e-12
    assert low["m_j_values"] == [-1.5, -0.5, 0.5, 1.5]


def test_j_that_is_not_a_level_of_the_term():
    r = _strict(R.atomic_angular_momentum(4.5, 0.05))
    v = r["value"]
    assert len(v["m_j_values"]) == 10          # still a valid j
    assert v["spin_orbit_energy_eV"] is None
    assert "not a level" in r["notes"]
