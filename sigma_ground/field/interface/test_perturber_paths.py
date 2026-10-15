"""perturber_paths: the vectorised RK4 test-particle integrator and the Kuiper-belt ring.

Ground truth that does not come from the code under test: (1) the two-body problem's conserved energy and angular momentum and its exact circular-orbit solution, (2) the
independent adaptive dopri5 integration of Ceres already stored in the asteroid states fixture (a different integrator over the same physics), (3) circular Kepler motion for the ring.
"""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from sigma_ground.field.interface import perturber_paths as pp
from sigma_ground.field.interface.asteroid_perturbers import AU_KM, DAY_S, Perturber

GM_SUN = 1.32712440041279419e11
KERNEL = Path("D:/datasets/jpl/de440s.bsp")
STATES = Path(__file__).parent / "fixtures" / "asteroid_perturber_states_v2physics.json"


class _Seg:
    def __init__(self, p):
        self.p = np.array(p, dtype=float)

    def compute(self, jds):
        jds = np.atleast_1d(jds)
        return np.tile(self.p[:, None], (1, len(jds)))

    def compute_and_differentiate(self, jds):
        jds = np.atleast_1d(jds)
        return np.tile(self.p[:, None], (1, len(jds))), np.zeros((3, len(jds)))


class _SunOnlyKernel:
    """A kernel whose Sun sits at the origin; every other segment is parked far away (their GM is zero in the tests)."""

    def __getitem__(self, key):
        return _Seg([0.0, 0.0, 0.0] if key == (0, 10) else [1e12, 0.0, 0.0])


GM_ONLY_SUN = np.array([GM_SUN] + [0.0] * 9)


class TestRK4(unittest.TestCase):
    def test_circular_orbit_matches_the_exact_solution(self):                       # BUG
        a = 2.8 * AU_KM
        n = math.sqrt(GM_SUN / a ** 3) * DAY_S                                      # rad / day
        y0 = np.array([[a, 0.0, 0.0, 0.0, a * n, 0.0]])
        t = 4.0 * 365.25
        want = np.array([a * math.cos(n * t), a * math.sin(n * t), 0.0])
        err = [np.linalg.norm(pp.rk4_segment(y0, 0.0, t, steps, _SunOnlyKernel(), GM_ONLY_SUN)[0, :3] - want) for steps in (1461, 2922)]
        self.assertLess(err[0], 0.03)                                               # km: 11 m measured at the 1-day step
        self.assertGreater(err[0] / err[1], 12.0)                                   # fourth order: halving the step cuts the error ~16-fold

    def test_eccentric_orbit_conserves_energy_and_angular_momentum(self):           # BUG
        a, e = 2.6 * AU_KM, 0.3
        n = math.sqrt(GM_SUN / a ** 3)
        rp = a * (1 - e)
        vp = math.sqrt(GM_SUN * (1 + e) / rp) * DAY_S
        y0 = np.array([[rp, 0.0, 0.0, 0.0, vp, 0.0]])

        def invariants(y):
            r, v = y[0, :3], y[0, 3:] / DAY_S
            return 0.5 * v @ v - GM_SUN / np.linalg.norm(r), np.linalg.norm(np.cross(r, v))
        e0, l0 = invariants(y0)
        y = pp.rk4_segment(y0, 0.0, 6 * 365.25, 6 * 366, _SunOnlyKernel(), GM_ONLY_SUN)
        e1, l1 = invariants(y)
        self.assertLess(abs(e1 / e0 - 1), 1e-9)
        self.assertLess(abs(l1 / l0 - 1), 1e-9)

    def test_recording_returns_every_node(self):                                    # BY CONSTRUCTION
        a = 2.8 * AU_KM
        n = math.sqrt(GM_SUN / a ** 3) * DAY_S
        y0 = np.array([[a, 0, 0, 0, a * n, 0], [0, a, 0, -a * n, 0, 0]])
        _, nodes = pp.rk4_segment(y0, 0.0, 10.0, 10, _SunOnlyKernel(), GM_ONLY_SUN, record=True)
        self.assertEqual(nodes.shape, (11, 2, 6))
        self.assertTrue(np.array_equal(nodes[0], y0))


class TestRing(unittest.TestCase):
    def test_ring_points_stay_on_the_circle_and_move_at_the_kepler_rate(self):      # BUG
        nodes = pp.ring_window_paths(_SunOnlyKernel(), GM_SUN, 2455197.5, 365.25, node_days=5.0)
        a = 44.0 * AU_KM
        r = np.linalg.norm(nodes[:, :, :3], axis=2)
        self.assertLess(np.abs(r / a - 1).max(), 1e-12)
        speed = np.linalg.norm(nodes[:, :, 3:], axis=2) / DAY_S
        self.assertLess(np.abs(speed / math.sqrt(GM_SUN / a) - 1).max(), 1e-12)
        h = np.cross(nodes[0, 0, :3], nodes[0, 0, 3:])
        h2 = np.cross(nodes[-1, 0, :3], nodes[-1, 0, 3:])
        self.assertLess(np.linalg.norm(h - h2) / np.linalg.norm(h), 1e-12)         # angular momentum direction and size are constant

    def test_equally_spaced_points_have_no_net_dipole(self):                         # BY CONSTRUCTION: 36 equal points on a circle sum to zero about the centre
        nodes = pp.ring_window_paths(_SunOnlyKernel(), GM_SUN, 2455197.5, 100.0, node_days=10.0)
        self.assertLess(np.linalg.norm(nodes[0, :, :3].sum(axis=0)) / (44.0 * AU_KM), 1e-12)


@unittest.skipUnless(KERNEL.exists() and STATES.exists(), "DE440s kernel or stored asteroid states not available")
class TestAgainstIndependentIntegration(unittest.TestCase):
    def test_ceres_matches_the_stored_dopri5_state(self):                            # BUG: different integrator, same physics
        from jplephem.spk import SPK
        k = SPK.open(str(KERNEL))
        doc = json.loads(STATES.read_text())
        ceres = doc["asteroids"]["Ceres"]
        sb = json.loads((Path(__file__).parent / "fixtures" / "sbdb_asteroids_54.json").read_text(encoding="utf-8"))
        b = next(x for x in sb["bodies"] if x.get("name") == "Ceres")
        p = Perturber("Ceres", float(b["GM_km3_s2"]), 0.5 * float(b["diameter_km"]), float(b["epoch_jd_tdb"]), float(b["a_au"]), float(b["e"]),
                      float(b["i_deg"]), float(b["om_deg"]), float(b["w_deg"]), float(b["ma_deg"]))
        epochs = doc["metadata"]["snapshot_epochs_jd_tdb"]
        fx = json.loads((Path(__file__).parent / "fixtures" / "de440_state_vectors_v2physics.json").read_text())["snapshots"]["j2010"]["bodies"]
        gm = {x["name"]: x["gm_km3_s2"] for x in fx}
        anchor = {x["name"]: x.get("gravitational_anchor") for x in fx}
        system = {n: gm[n] + sum(g for m, g in gm.items() if anchor.get(m) == n) for n in ("Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")}
        gms = np.array([gm["Sun"], system["Mercury"], system["Venus"], gm["Earth"] + gm["Moon"], system["Mars"], system["Jupiter"], system["Saturn"], system["Uranus"],
                        system["Neptune"], system["Pluto"]])
        nodes = pp.asteroid_window_paths([p], k, gms, epochs["j2008"], 30.0, 1.0)
        stored = np.array(ceres["states"]["j2008"])
        self.assertLess(np.linalg.norm(nodes[0, 0, :3] - stored[:3]), 5.0)             # km, 18 years from the element epoch
        self.assertLess(np.linalg.norm(nodes[0, 0, 3:] - stored[3:]), 0.05)            # km/day: a few km of position error times the mean motion (0.0037 /day) is ~0.02


if __name__ == "__main__":
    unittest.main()
