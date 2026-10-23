"""The r7 fixture: original bodies + 23 Horizons moons, each planet system (all members) moved rigidly onto DE440's barycentre.

Ground truth that is not the code under test: the DE440s kernel (system barycentres), the Horizons responses (moon states) and the original fixture (relative states that must survive the shift).
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
FIX = ROOT / "sigma_ground/field/interface/fixtures"
KERNEL = Path("D:/datasets/jpl/de440s.bsp")
R7 = FIX / "de440_state_vectors_v2physics_r7.json"
KEYS = ("x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s")


@unittest.skipUnless(R7.exists() and KERNEL.exists(), "r7 fixture or DE440s kernel not available")
class TestR7(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jplephem.spk import SPK
        cls.kernel = SPK.open(str(KERNEL))
        cls.orig = json.loads((FIX / "de440_state_vectors_v2physics.json").read_text())["snapshots"]
        cls.r7 = json.loads(R7.read_text())["snapshots"]

    def _sv(self, snap, name):
        b = next(x for x in snap["bodies"] if x["name"] == name)
        return np.array([b["state_vector"][c] for c in KEYS])

    def test_every_system_barycentre_including_the_added_moons_is_de440s(self):                  # BUG
        snap = self.r7["j2010"]
        anchor = {b["name"]: b.get("gravitational_anchor") for b in snap["bodies"]}
        for planet, seg in (("Jupiter", 5), ("Saturn", 6), ("Uranus", 7), ("Neptune", 8), ("Pluto", 9)):
            members = [planet] + [m for m, an in anchor.items() if an == planet]
            gm = np.array([next(b for b in snap["bodies"] if b["name"] == n)["gm_km3_s2"] for n in members])
            sv = np.array([self._sv(snap, n) for n in members])
            bary = (gm[:, None] * sv).sum(0) / gm.sum()
            p, v = self.kernel[0, seg].compute_and_differentiate(snap["epoch"]["jd_tdb"])
            self.assertLess(np.linalg.norm(bary[:3] - np.asarray(p, dtype=float)), 1e-3, planet)                 # km
            self.assertLess(np.linalg.norm(bary[3:] - np.asarray(v, dtype=float) / 86400.0) * 1e6, 1e-3, planet)  # mm/s

    def test_the_neptune_velocity_defect_shrinks_once_the_small_moons_are_in(self):               # BUG: 2.87 mm/s with Triton alone, 0.5 mm/s with the four small inner moons
        from shift_giant_systems import shift_snapshot
        for key in ("j2010", "j2015"):
            snap = copy.deepcopy(self.orig[key])
            before = shift_snapshot(snap, self.kernel)["Neptune"][1]
            self.assertGreater(before, 2.5)
            # the r7 shift for the same snapshot, from the stored r7: recompute by comparing r7 Neptune with the original
            d = self._sv(self.r7[key], "Neptune") - self._sv(self.orig[key], "Neptune")
            self.assertLess(np.linalg.norm(d[3:]) * 1e6, 1.0)                                              # mm/s

    def test_added_moons_keep_their_horizons_state_relative_to_the_original_planet(self):          # BUG
        from sigma_ground.field.interface.comet_truth import parse_horizons_comet
        keys = sorted((k for k in self.r7 if k.startswith("j")), key=lambda k: int(k[1:]))
        for moon, planet, fname in (("Proteus", "Neptune", "808_Proteus"), ("Nereid", "Neptune", "802_Nereid"), ("Helene", "Saturn", "612_Helene"), ("Amalthea", "Jupiter", "505_Amalthea")):
            t = parse_horizons_comet(FIX / "horizons_moons" / f"horizons_moon_{fname}.json")
            for i in (0, 10, 26):
                h = np.array(t["state_km_kmday"][i], dtype=float)
                h[3:] /= 86400.0
                want = h - self._sv(self.orig[keys[i]], planet)
                got = self._sv(self.r7[keys[i]], moon) - self._sv(self.r7[keys[i]], planet)
                self.assertLess(np.abs(got[:3] - want[:3]).max(), 1e-3, (moon, keys[i]))
                self.assertLess(np.abs(got[3:] - want[3:]).max(), 1e-11, (moon, keys[i]))

    def test_original_members_keep_their_relative_states(self):                                    # BY CONSTRUCTION
        for planet, moon in (("Neptune", "Triton"), ("Pluto", "Charon"), ("Uranus", "Titania"), ("Jupiter", "Io")):
            a = self._sv(self.orig["j2010"], moon) - self._sv(self.orig["j2010"], planet)
            b = self._sv(self.r7["j2010"], moon) - self._sv(self.r7["j2010"], planet)
            self.assertLess(np.abs(a[:3] - b[:3]).max(), 1e-3)
            self.assertLess(np.abs(a[3:] - b[3:]).max(), 1e-11)

    def test_all_23_moons_are_present(self):                                                       # BY CONSTRUCTION
        names = {b["name"] for b in self.r7["j2010"]["bodies"]}
        self.assertTrue({"Amalthea", "Thebe", "Himalia", "Janus", "Epimetheus", "Prometheus", "Pandora", "Atlas", "Pan", "Helene", "Puck", "Portia", "Juliet", "Belinda", "Proteus", "Nereid", "Larissa",
                         "Galatea", "Despina", "Nix", "Hydra", "Kerberos", "Styx"} <= names)


if __name__ == "__main__":
    unittest.main()
