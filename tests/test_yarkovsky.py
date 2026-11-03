"""Yarkovsky on the small bodies (C31): JPL's fitted A2 on the (1 au / r)^2 law, attached to the start and applied by the existing non-gravitational routine.

Ground truth not under test: JPL SBDB's own numbers for Apophis (A2 = -2.902e-14 au/d^2, law ALN 1, NK 0, NM 2, R0 1 au), and the circular-orbit along-track drift of a transverse
acceleration, ds = 1.5 A2 t^2 / a^2 (the size Mentat computed before the first run: 1.58 km at 455 d), which the propagated difference must match within the ledger's +-35 %.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HAVE_KERNEL = Path("D:/datasets/jpl/de442/de442.bsp").exists() or Path("D:/datasets/jpl/de440s.bsp").exists()


class TestFixtureAndLaw(unittest.TestCase):
    def test_apophis_carries_jpls_a2_and_the_law_is_inverse_square(self):
        from sigma_ground.field.interface import small_body_position as sbp
        from sigma_ground.field.interface.comet_orbits import marsden_g
        y = sbp.yarkovsky_for(99942)
        self.assertAlmostEqual(y["A_au_day2"][1], -2.902e-14, delta=1e-17)
        self.assertIn("JPL SBDB", y["source"])
        self.assertAlmostEqual(marsden_g(2.0, **y["law"]), 0.25, places=12)                     # (1 au / r)^2 at 2 au
        self.assertAlmostEqual(marsden_g(0.5, **y["law"]), 4.0, places=12)

    def test_bodies_without_a_solution_get_no_term(self):                                       # BY CONSTRUCTION: nothing invented for Eros, Bennu or the heavies
        from sigma_ground.field.interface import small_body_position as sbp
        for n in (433, 1, 4, 101955):
            self.assertIsNone(sbp.yarkovsky_for(n), n)
        self.assertIsNone(sbp.yarkovsky_for(None))
        meta = sbp._nongrav()[1]
        self.assertIn("101955 Bennu (1999 RQ36)", meta["checked_without_A2"])


@unittest.skipUnless(HAVE_KERNEL, "no JPL kernel on this machine")
class TestPropagation(unittest.TestCase):
    def setUp(self):
        from sigma_ground.field.interface import small_body_position as sbp, where_when as ww
        self.sbp, self.ww = sbp, ww
        self.k = ww._truth()[0]

    def test_apophis_start_carries_the_term_and_eros_does_not(self):
        a = self.sbp.resolve("Apophis", self.k)
        self.assertEqual((a.tier, a.law), ("horizons", self.sbp.YARKOVSKY_LAW))
        self.assertAlmostEqual(a.nongrav[1], -2.902e-14, delta=1e-17)
        self.assertEqual((a.nongrav[0], a.nongrav[2], a.nongrav_dt_days), (0.0, 0.0, 0.0))                       # A1 = 5e-13 +- 4.9e-13 is a 1-sigma number: dropped (C31 verdict), and said so
        self.assertTrue(any("A1" in s and "not a detection" in s for s in a.extra["yarkovsky"]["dropped"]))
        self.assertIsNone(self.sbp.resolve("Eros", self.k).nongrav)
        t = self.sbp.yarkovsky_for(4179)                                                           # Toutatis: A1 at 3.4 sigma stays
        self.assertNotEqual(t["A_au_day2"][0], 0.0)

    def test_the_drift_matches_the_pre_registered_size_and_runs_ahead(self):                     # C31 P1, P2: 1.58 km at 455 d, +-35 %, along the velocity
        import dataclasses
        from sigma_ground.field.interface import comet_orbits as co
        st = self.sbp.resolve("Apophis", self.k)
        jd = st.jd0 + 455.0
        src, w0, _ = self.sbp._sources_for(self.k, st.number, st.jd0, jd)
        with_y = co.propagate(list(st.y0), st.jd0, [jd], self.k, self.sbp.gm_planets(), gr=True, sources=src, sources_epoch_jd=w0, nongrav=list(st.nongrav), law=st.law)[jd]
        without = co.propagate(list(st.y0), st.jd0, [jd], self.k, self.sbp.gm_planets(), gr=True, sources=src, sources_epoch_jd=w0)[jd]
        d = np.array(with_y[:3]) - np.array(without[:3])
        size = float(np.linalg.norm(d))
        self.assertTrue(1.0 <= size <= 2.1, f"{size:.2f} km (predicted 1.58)")
        v = np.array(without[3:]); along = float(d @ v / np.linalg.norm(v))
        self.assertGreater(along, 0.8 * size)                                                     # ahead, along the velocity
        r = self.ww.position_at("Apophis", f"JD {jd}", "ssb", budget_s=120)
        self.assertEqual(r["receipt"]["non_gravitational"]["what"][:9], "Yarkovsky")
        self.assertIn("Yarkovsky", r["basis"])
        self.assertLess(np.linalg.norm(np.array(r["position_km"]) - np.array(with_y[:3])), 1e-3)   # body_position serves the WITH-term answer


if __name__ == "__main__":
    unittest.main()
