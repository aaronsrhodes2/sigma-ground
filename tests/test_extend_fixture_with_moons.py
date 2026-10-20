"""The r5 fixture: moons added from Horizons keep their state relative to their planet exactly, and the systems stay on DE440's barycentre offset (the r4 property).

Ground truth: the Horizons responses themselves (a moon's barycentric state minus its planet's ORIGINAL fixture state is the relative state that must survive the system shift).
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

FIX = Path(__file__).resolve().parent.parent / "sigma_ground/field/interface/fixtures"
KEYS = ("x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s")


@unittest.skipUnless((FIX / "de440_state_vectors_v2physics_r5.json").exists(), "r5 fixture not built")
class TestR5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = json.loads((FIX / "de440_state_vectors_v2physics.json").read_text())["snapshots"]
        cls.r5 = json.loads((FIX / "de440_state_vectors_v2physics_r5.json").read_text())
        cls.idx = sorted((k for k in cls.full if k.startswith("j")), key=lambda k: int(k[1:]))

    def _sv(self, snap, name):
        b = next(x for x in snap["bodies"] if x["name"] == name)
        return np.array([b["state_vector"][c] for c in KEYS])

    def test_added_moons_keep_their_horizons_relative_state_in_every_snapshot(self):       # BUG
        for moon, planet, fname in (("Himalia", "Jupiter", "506_Himalia"), ("Nereid", "Neptune", "802_Nereid"), ("Helene", "Saturn", "612_Helene")):
            res = json.loads((FIX / "horizons_moons" / f"horizons_moon_{fname}.json").read_text())["result"]
            rows = res.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
            for i in (0, 10, 20, 26):
                key = self.idx[i]
                h = np.array([float(c) for c in rows[i].split(",")[2:8]])
                h[3:] /= 86400.0
                want = h - self._sv(self.full[key], planet)
                got = self._sv(self.r5["snapshots"][key], moon) - self._sv(self.r5["snapshots"][key], planet)
                self.assertLess(np.abs(got[:3] - want[:3]).max(), 1e-3, (moon, key))             # km
                self.assertLess(np.abs(got[3:] - want[3:]).max(), 1e-11, (moon, key))            # km/s

    def test_existing_bodies_are_exactly_the_r4_ones(self):                                    # BY CONSTRUCTION
        r4 = json.loads((FIX / "de440_state_vectors_v2physics_r4.json").read_text())["snapshots"]["j2010"]["bodies"]
        r5 = self.r5["snapshots"]["j2010"]["bodies"]
        self.assertEqual(r5[:len(r4)], r4)
        self.assertEqual([b["name"] for b in r5[len(r4):]], ["Himalia", "Nereid", "Helene"])


if __name__ == "__main__":
    unittest.main()
