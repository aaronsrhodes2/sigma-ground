"""The truth kernel (DE442) is distinct from the start kernel (DE440), and the two are not swapped: their inner planets agree to tens of metres, their Uranus does not (2026-11-02)."""
from __future__ import annotations

import numpy as np
import pytest

from sigma_ground.field import datasets

jplephem = pytest.importorskip("jplephem.spk")


def _hel(k, target: int, jd: float):
    return k[0, target].compute(jd) - k[0, 10].compute(jd)


@pytest.mark.skipif(datasets.de440_kernel() is None, reason="DE440 not installed here")
def test_truth_kernel_is_de442_and_not_de440():
    f, name = datasets.truth_kernel()
    assert f is not None
    assert name == "DE442", name
    assert f != datasets.de440_kernel()
    a = jplephem.SPK.open(str(f))
    b = jplephem.SPK.open(str(datasets.de440_kernel()))
    try:
        jd2015 = 2451545.0 + 365.25 * 15
        jd2026 = 2451545.0 + 365.25 * 26
        assert np.linalg.norm(_hel(a, 3, jd2015) - _hel(b, 3, jd2015)) < 0.05      # Earth-Moon barycentre: the inner system is known to tens of metres
        assert np.linalg.norm(_hel(a, 7, jd2026) - _hel(b, 7, jd2026)) > 500.0     # Uranus: the 2024 refit moved it by over 1000 km
    finally:
        a.close()
        b.close()


def test_env_override_wins(monkeypatch, tmp_path):
    p = tmp_path / "de999.bsp"
    p.write_bytes(b"x")
    monkeypatch.setenv("SIGMA_TRUTH_KERNEL", str(p))
    assert datasets.truth_kernel() == (p, "DE999")
    monkeypatch.setenv("SIGMA_TRUTH_KERNEL", str(tmp_path / "missing.bsp"))
    assert datasets.truth_kernel() == (None, "none")
