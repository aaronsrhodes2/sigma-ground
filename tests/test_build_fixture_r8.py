"""The r8 fixture: r7 with Neptune and Triton from the 2026-10-23 Horizons fetch (same vintage as the 23 moons), each system moved rigidly onto DE440's barycentre.

Ground truth that is not the code under test: the DE440s kernel, the fresh Horizons responses, the original fixture (the Neptune answer key must not move) and the measured outcome (Proteus's mean motion).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
FIX = ROOT / "sigma_ground/field/interface/fixtures"
KERNEL = Path("D:/datasets/jpl/de440s.bsp")
R8 = FIX / "de440_state_vectors_v2physics_r8.json"
CENTRES = Path("D:/datasets/jpl/horizons_centres")
KEYS = ("x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s")


@unittest.skipUnless(R8.exists() and KERNEL.exists() and CENTRES.exists(), "r8 fixture, DE440s kernel or the fetched centres not available")
class TestR8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jplephem.spk import SPK
        cls.kernel = SPK.open(str(KERNEL))
        cls.orig = json.loads((FIX / "de440_state_vectors_v2physics.json").read_text())["snapshots"]
        cls.r8 = json.loads(R8.read_text())["snapshots"]

    def _sv(self, snap, name):
        b = next(x for x in snap["bodies"] if x["name"] == name)
        return np.array([b["state_vector"][c] for c in KEYS])

    def test_neptune_answer_key_does_not_move(self):                                            # BUG: the shift must take the 159 km vintage offset back out (0.02 km measured)
        for key in ("j2000", "j2010", "j2020"):
            d = self._sv(self.r8[key], "Neptune") - self._sv(self.orig[key], "Neptune")
            self.assertLess(np.linalg.norm(d[:3]), 0.5, key)                                    # km
            self.assertLess(np.linalg.norm(d[3:]) * 1e6, 1.5, key)                              # mm/s

    def test_every_system_barycentre_is_de440s(self):                                           # BUG
        snap = self.r8["j2010"]
        anchor = {b["name"]: b.get("gravitational_anchor") for b in snap["bodies"]}
        for planet, seg in (("Jupiter", 5), ("Saturn", 6), ("Uranus", 7), ("Neptune", 8), ("Pluto", 9)):
            members = [planet] + [m for m, an in anchor.items() if an == planet]
            gm = np.array([next(b for b in snap["bodies"] if b["name"] == n)["gm_km3_s2"] for n in members])
            sv = np.array([self._sv(snap, n) for n in members])
            bary = (gm[:, None] * sv).sum(0) / gm.sum()
            p, v = self.kernel[0, seg].compute_and_differentiate(snap["epoch"]["jd_tdb"])
            self.assertLess(np.linalg.norm(bary[:3] - np.asarray(p, dtype=float)), 1e-3, planet)
            self.assertLess(np.linalg.norm(bary[3:] - np.asarray(v, dtype=float) / 86400.0) * 1e6, 1e-3, planet)

    def test_neptune_system_relative_states_are_the_fresh_horizons_ones(self):                  # BUG: relative states of Triton and Proteus against Neptune are the 2026-10-23 vintage's
        from sigma_ground.field.interface.comet_truth import parse_horizons_comet
        keys = sorted((k for k in self.r8 if k.startswith("j")), key=lambda k: int(k[1:]))
        n899 = parse_horizons_comet(CENTRES / "horizons_centre_899_Neptune.json")
        for moon, path in (("Triton", CENTRES / "horizons_centre_801_Triton.json"), ("Proteus", FIX / "horizons_moons/horizons_moon_808_Proteus.json")):
            t = parse_horizons_comet(path)
            for i in (0, 10, 26):
                h = np.array(t["state_km_kmday"][i], dtype=float)
                h[3:] /= 86400.0
                n = np.array(n899["state_km_kmday"][i], dtype=float)
                n[3:] /= 86400.0
                want = h - n
                got = self._sv(self.r8[keys[i]], moon) - self._sv(self.r8[keys[i]], "Neptune")
                self.assertLess(np.abs(got[:3] - want[:3]).max(), 1e-3, (moon, keys[i]))
                self.assertLess(np.abs(got[3:] - want[3:]).max(), 1e-11, (moon, keys[i]))

    def test_saturn_and_uranus_systems_are_the_same_as_in_r7(self):                              # BY CONSTRUCTION: the fresh Saturn, Titan, Uranus, Titania tracks equal the fixture's to 0.000 km, so r8 = r7 there
        r7 = json.loads((FIX / "de440_state_vectors_v2physics_r7.json").read_text())["snapshots"]["j2010"]
        for name in ("Saturn", "Titan", "Uranus", "Titania", "Pan", "Puck"):
            a = self._sv(r7, name)
            b = self._sv(self.r8["j2010"], name)
            self.assertLess(np.linalg.norm(a[:3] - b[:3]), 1e-3, name)


if __name__ == "__main__":
    unittest.main()
