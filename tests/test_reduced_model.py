"""sigma_ground.field.interface.reduced_model reproduces what scripts/fair_jpl_contest.py (run_one) computed for the same start and layers.

Ground truth that is not the code under test: the DE440s kernel (the answer key both are scored against) and the older script's own numbers (the reference implementation the module was factored out of).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
KERNEL = Path("D:/datasets/jpl/de440s.bsp")
FIX = ROOT / "sigma_ground/field/interface/fixtures/de440_state_vectors_v2physics.json"


def _errors_vs_kernel(kernel, jd0, jd1, predictor, dt):
    from sigma_ground.field.interface import reduced_model as rm
    system, order = rm.build_system(kernel, jd0, predictor, dt_days=dt, horizon_days=jd1 - jd0 + 5.0)
    d = rm.propagate_dense(system, jd0, jd1, dt, every=10 ** 9)
    p = d["pos_km"][-1]
    sun = p[order.index("Sun")]
    out = {}
    for i, n in enumerate(order):
        if n == "Sun":
            continue
        truth = rm.kernel_state(kernel, n, d["jd"][-1])[0] - rm.kernel_state(kernel, "Sun", d["jd"][-1])[0]
        out[n] = float(np.linalg.norm((p[i] - sun) - truth))
    return out


@unittest.skipUnless(KERNEL.exists() and FIX.exists(), "DE440s kernel not available")
class TestReducedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jplephem.spk import SPK
        import fair_jpl_contest as fc
        cls.kernel = SPK.open(str(KERNEL))
        cls.fc = fc
        cls.jd0 = fc.jd_of(2010, 1, 1)

    def test_kernel_state_matches_the_scripts_helper(self):                                    # BY CONSTRUCTION: km/s here, km/day there
        from sigma_ground.field.interface import reduced_model as rm
        for n in ("Earth", "Moon", "Jupiter"):
            p, v = rm.kernel_state(self.kernel, n, self.jd0)
            p2, v2 = self.fc.state(self.kernel, n, self.jd0)
            self.assertTrue(np.array_equal(p, p2))
            self.assertTrue(np.allclose(v * 86400.0, v2, rtol=0, atol=1e-12))

    def test_masses_sum_the_moons_into_the_giants(self):                                       # BUG: the system barycentre carries the moons' mass
        import json
        from sigma_ground.field.interface import reduced_model as rm
        snap = json.loads(FIX.read_text())["snapshots"]["j2010"]
        m = rm.reduced_masses(snap)
        gm_j = next(b["gm_km3_s2"] for b in snap["bodies"] if b["name"] == "Jupiter") * 1e9
        self.assertGreater(m["Jupiter"], gm_j)
        self.assertEqual(m["Earth"], next(b["gm_km3_s2"] for b in snap["bodies"] if b["name"] == "Earth") * 1e9)

    def test_one_year_without_layers_equals_the_script(self):                                  # BUG if the factoring changed a number
        ours = _errors_vs_kernel(self.kernel, self.jd0, self.jd0 + 365.25, "de440_stack", 0.25)
        ref = self.fc.run_one(("x", None, "j2010", self.jd0, 0.25, [1], None, None, None, None, False, None, None, None, None))["bodies"]
        for n, e in ours.items():
            self.assertAlmostEqual(e, ref[n][1]["eO"], delta=1e-6 + 1e-9 * e, msg=n)

    def test_one_year_with_the_restated_layers_equals_the_script(self):                        # BUG: asteroids, ring, KBOs and the Moon's C22 in the same places
        from sigma_ground.field.interface import rolling_shootout as rs
        ours = _errors_vs_kernel(self.kernel, self.jd0, self.jd0 + 365.25, "restated_ast_ring_kbo_c22", 0.25)
        ref = self.fc.run_one(("x", None, "j2010", self.jd0, 0.25, [1], None, None, None, None, False, rs._DE440_C22M, None, None, ("asteroids", "ring", "kbo")))["bodies"]
        for n, e in ours.items():
            self.assertAlmostEqual(e, ref[n][1]["eO"], delta=1e-6 + 1e-9 * e, msg=n)

    def test_hermite_reproduces_a_smooth_orbit_between_samples(self):                          # BY CONSTRUCTION: cubic Hermite from position and velocity
        from sigma_ground.field.interface import reduced_model as rm
        t = np.linspace(0.0, 4.0, 17)                                                          # days, 0.25 d samples of a circle of 1e6 km radius, period 30 d
        w = 2 * np.pi / 30.0
        pos = np.stack([np.cos(w * t), np.sin(w * t), 0 * t], axis=1)[:, None, :] * 1e6
        vel = np.stack([-np.sin(w * t), np.cos(w * t), 0 * t], axis=1)[:, None, :] * 1e6 * w / 86400.0
        p, v = rm.hermite(1.3, t, pos, vel)
        self.assertLess(np.linalg.norm(p[0] - np.array([np.cos(w * 1.3), np.sin(w * 1.3), 0.0]) * 1e6), 1e-2)
        self.assertLess(np.linalg.norm(v[0] - np.array([-np.sin(w * 1.3), np.cos(w * 1.3), 0.0]) * 1e6 * w / 86400.0), 1e-6)


if __name__ == "__main__":
    unittest.main()
