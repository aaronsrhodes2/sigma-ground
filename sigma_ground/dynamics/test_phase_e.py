"""Phase E4 runner: structure, energy honesty, and residual sanity."""
import math
import unittest

from sigma_ground.dynamics.phase_e import diagnostics, run_comparison
from sigma_ground.dynamics.seeding import G_CODATA, seed_ring_annulus

RING = dict(central_mass_kg=5.0e15, central_radius_m=800.0,
            r_inner_m=6000.0, r_outer_m=12000.0, ring_mass_kg=1.0e9,
            particle_density_kg_m3=920.0)


class TestDiagnostics(unittest.TestCase):
    def test_two_body_hand_check(self):
        """KE, PE, radii and nearest-neighbor for a hand-computable pair."""
        d = diagnostics([2.0, 2.0],
                        [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)],
                        [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)],
                        G=1.0)
        self.assertAlmostEqual(d["ke_j"], 2.0)           # 2 * 0.5*2*1
        self.assertAlmostEqual(d["pe_j"], -1.0)          # -1*2*2/4
        self.assertAlmostEqual(d["e_total_j"], 1.0)
        self.assertAlmostEqual(d["r_min_m"], 2.0)        # about barycenter
        self.assertAlmostEqual(d["r_max_m"], 2.0)
        self.assertAlmostEqual(d["nn_mean_m"], 4.0)
        self.assertAlmostEqual(d["speed_std_m_s"], 0.0)  # equal speeds


class TestRunComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bodies = seed_ring_annulus(20, seed=5, **RING)
        r_mid = 9000.0
        T = 2 * math.pi * math.sqrt(r_mid ** 3
                                    / (G_CODATA * RING["central_mass_kg"]))
        v_kep = math.sqrt(G_CODATA * RING["central_mass_kg"] / r_mid)
        cls.report = run_comparison(
            bodies, dt=T / 200.0, n_steps=60, sample_every=20,
            material_key='iron', sound_speed_m_s=0.01 * v_kep,
            softening_m=50.0, label="test ring")

    def test_report_structure_and_provenance(self):
        r = self.report
        self.assertEqual(r["kind"], "phase_e_comparison")
        self.assertTrue(r["choices"] and r["facts"])
        descs = " | ".join(c["description"] for c in r["choices"])
        self.assertIn("softening", descs)
        self.assertIn("velocity-dispersion", descs)
        # samples at t=0, then every 20 steps
        self.assertEqual(len(r["samples"]), 4)
        self.assertEqual(r["samples"][0]["t_s"], 0.0)

    def test_engines_start_identical(self):
        s0 = self.report["samples"][0]
        for k, v in s0["residual"].items():
            if k.startswith("dfrac_"):
                self.assertLess(abs(v), 1e-9, k)

    def test_each_engine_conserves_its_own_energy(self):
        """Both integrators are symplectic; over a third of an orbit each
        engine's total energy must hold to well under 1%. (The SPH side's
        viscosity dissipates KE into unmodeled heat — at this dispersion
        and duration the effect is far below this tolerance.)"""
        for eng in ("nbody", "sph"):
            e0 = self.report["samples"][0][eng]["e_total_j"]
            e1 = self.report["samples"][-1][eng]["e_total_j"]
            self.assertLess(abs(e1 - e0), abs(e0) * 0.01, eng)

    def test_residual_stays_statistical_not_explosive(self):
        """Early-time ensemble residuals are small fractions — the two
        descriptions agree while gravity dominates. (Their eventual
        divergence is the experiment; their early agreement is the
        validity check.)"""
        last = self.report["samples"][-1]["residual"]
        self.assertLess(abs(last["dfrac_r_med_m"]), 0.05)
        self.assertLess(abs(last["dfrac_e_total_j"]), 0.05)
        for s in self.report["samples"]:
            for eng in ("nbody", "sph"):
                for v in s[eng].values():
                    self.assertTrue(math.isfinite(v))


if __name__ == "__main__":
    unittest.main()
