"""scripts/shift_giant_systems.py: a rigid shift of each planet system onto DE440's barycentre.

Ground truth: the DE440s kernel (the target barycentre) and the fixture itself (the relative states that must not change). The r3 attempt moved only the planet and broke the moons
(Titania 963 -> 676,199 km at +5 yr); these tests state the property r3 lacked.
"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
KERNEL = Path("D:/datasets/jpl/de440s.bsp")
FIXTURE = ROOT / "sigma_ground/field/interface/fixtures/de440_state_vectors_v2physics.json"


@unittest.skipUnless(KERNEL.exists(), "DE440s kernel not available")
class TestRigidShift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import shift_giant_systems as sg
        from jplephem.spk import SPK
        cls.sg = sg
        cls.kernel = SPK.open(str(KERNEL))
        cls.orig = json.loads(FIXTURE.read_text())["snapshots"]["j2010"]
        cls.snap = copy.deepcopy(cls.orig)
        cls.shifts = sg.shift_snapshot(cls.snap, cls.kernel)

    def _sv(self, snap, name):
        b = next(x for x in snap["bodies"] if x["name"] == name)
        return np.array([b["state_vector"][c] for c in self.sg.POS + self.sg.VEL])

    def test_relative_states_inside_each_system_are_unchanged(self):                  # BUG: r3 changed them and broke the moons
        for planet, moons in (("Uranus", ["Titania", "Oberon", "Ariel", "Umbriel", "Miranda"]), ("Neptune", ["Triton"]), ("Pluto", ["Charon"]), ("Jupiter", ["Io", "Ganymede"])):
            for m in moons:
                before = self._sv(self.orig, m) - self._sv(self.orig, planet)
                after = self._sv(self.snap, m) - self._sv(self.snap, planet)
                self.assertLess(np.abs(after[:3] - before[:3]).max(), 1e-3)              # km: round-off of a ~1e9 km number
                self.assertLess(np.abs(after[3:] - before[3:]).max(), 1e-12)             # km/s

    def test_the_system_barycentre_is_now_de440s(self):                                # BUG
        jd = self.snap["epoch"]["jd_tdb"]
        anchor = {b["name"]: b.get("gravitational_anchor") for b in self.snap["bodies"]}
        for planet, seg in self.sg.SEG.items():
            members = [planet] + [m for m, an in anchor.items() if an == planet]
            if len(members) == 1:
                continue
            gm = np.array([next(b for b in self.snap["bodies"] if b["name"] == n)["gm_km3_s2"] for n in members])
            sv = np.array([self._sv(self.snap, n) for n in members])
            bary = (gm[:, None] * sv).sum(0) / gm.sum()
            p, v = self.kernel[0, seg].compute_and_differentiate(jd)
            self.assertLess(np.linalg.norm(bary[:3] - np.asarray(p, dtype=float)), 1e-3, planet)                    # km
            self.assertLess(np.linalg.norm(bary[3:] - np.asarray(v, dtype=float) / 86400.0) * 1e6, 1e-3, planet)    # mm/s

    def test_the_shifts_are_the_measured_ones(self):                                  # BUG: Uranus 5 mm/s, Neptune 3 mm/s (the old 'floors')
        self.assertGreater(self.shifts["Uranus"][1], 4.0)
        self.assertGreater(self.shifts["Neptune"][1], 2.0)
        self.assertLess(self.shifts["Saturn"][1], 0.01)

    def test_bodies_outside_the_systems_are_untouched(self):                           # BY CONSTRUCTION
        for n in ("Sun", "Mercury", "Venus", "Earth", "Moon"):
            self.assertTrue(np.array_equal(self._sv(self.snap, n), self._sv(self.orig, n)), n)


if __name__ == "__main__":
    unittest.main()
