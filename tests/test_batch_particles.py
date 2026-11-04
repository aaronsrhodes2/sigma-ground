"""The whole-sky batch engine (C35) against the served per-body engine: the pre-registered S-A acceptance, the RK4 order, and the Hermite reader.

Ground truth not under test: comet_orbits.propagate (dopri5, rtol 1e-13), the served engine whose agreement with DE440/JPL is measured elsewhere; the batch engine must reproduce
it within 0.1 km over a year on the tracked bodies at the pre-registered steps (1 d main belt, 0.25 d near-Earth).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
HAVE_KERNEL = Path("D:/datasets/jpl/de440s.bsp").exists()


class TestReader(unittest.TestCase):
    def test_hermite_returns_the_nodes_at_node_times_and_a_cubic_between(self):
        from sigma_ground.field.interface.batch_particles import positions_at
        t = np.array([0.0, 1.0, 2.0])
        c = lambda tt: np.array([tt ** 3, 2 * tt, 1.0])                                        # a cubic is reproduced exactly
        v = lambda tt: np.array([3 * tt ** 2, 2.0, 0.0])
        nodes = np.array([[np.concatenate([c(tt), v(tt)])] for tt in t])                       # (3, 1, 6)
        q = np.array([0.0, 0.5, 1.0, 1.7, 2.0])
        got = positions_at(t, nodes, q, np.zeros(5, dtype=int), velocity=True)
        for i, tt in enumerate(q):
            np.testing.assert_allclose(got[i, :3], c(tt), atol=1e-12); np.testing.assert_allclose(got[i, 3:], v(tt), atol=1e-12)


@unittest.skipUnless(HAVE_KERNEL, "no JPL kernel on this machine")
class TestAgainstTheServedEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mpc_live_score as m
        from sigma_ground.field.interface import comet_orbits as co
        cls.m, cls.co = m, co
        cls.engine = m.Engine(lambda s: None)
        cls.starts = {}
        for n in (1, 4, 433, 99942, 137108):                                                     # Ceres, Vesta (both sources themselves), Eros, Apophis, 1999 AN10
            st = m.start_state(n, m.ROOT / "forecast_seal", lambda s: None)
            if st is not None:
                cls.starts[n] = st

    def _served(self, n, jd0, y0, jd):
        from sigma_ground.field.interface import small_body_position as sbp
        yk = sbp.yarkovsky_for(n)
        extra = {"nongrav": list(yk["A_au_day2"]), "law": yk["law"]} if yk else {}
        return self.co.propagate(list(y0), jd0, [jd], self.engine.kernel, self.engine.gmp, gr=True, sources=self.engine.sources(n), sources_epoch_jd=self.engine.w0, **extra)[jd]

    def _batch(self, n, jd0, y0, jd, step):
        from sigma_ground.field.interface import batch_particles as bp, small_body_position as sbp
        from sigma_ground.field.interface.prescribed_sources import PrescribedSources
        t = self.engine.pp.to_prescribed_sources(self.engine.nodes, self.engine.gm_all, 1.0)
        src = PrescribedSources(t.gm, t.t0, t.h, t.pos, t.vel)
        idx = np.nonzero(self.engine.numbers == n)[0]
        self_index = np.array([idx[0] if len(idx) else -1])
        yk = sbp.yarkovsky_for(n)
        A = np.array([yk["A_au_day2"]]) if yk else None
        tn, nodes = bp.propagate_batch(np.array([y0]), jd0, jd, self.engine.kernel, self.engine.gmp, sources=src, sources_epoch_jd=self.engine.w0, self_index=self_index,
                                       step_days=step, nongrav=A, law=yk["law"] if yk else None)
        return nodes[-1, 0]

    def test_S_A_one_year_within_a_tenth_of_a_kilometre(self):
        worst = {}
        for n, (jd0, y0, _) in self.starts.items():
            step = 0.25 if n in (433, 99942, 137108) else 1.0
            jd = jd0 + 365.25
            d = np.linalg.norm(self._batch(n, jd0, y0, jd, step)[:3] - self._served(n, jd0, y0, jd)[:3])
            worst[n] = float(d)
        self.assertTrue(all(v <= 0.1 for v in worst.values()), f"km after one year: {worst}")

    def test_rk4_falls_sixteen_fold_per_halving(self):
        n = 1; jd0, y0, _ = self.starts[n]; jd = jd0 + 120.0
        ref = self._served(n, jd0, y0, jd)[:3]
        e = [float(np.linalg.norm(self._batch(n, jd0, y0, jd, s)[:3] - ref)) for s in (4.0, 2.0)]
        self.assertGreater(e[0] / max(e[1], 1e-9), 8.0, e)


if __name__ == "__main__":
    unittest.main()
