"""Fix the planet-system barycentres of the fixture by moving each system RIGIDLY (planet and all its moons) onto DE440's barycentre. Writes the r4 fixture; the older fixtures are never touched.

Why (2026-10-16/17). The fixture's planet-centre velocities disagree with DE440's system barycentre velocity by Uranus 5.3, Neptune 2.9, Pluto 0.29, Jupiter 0.021 mm/s (position: Uranus 519 km),
which over 5 yr is the old 'planet-centre floors'. The first repair (scripts/rederive_giant_states.py, the r3 fixture) moved only the PLANET onto the barycentre-implied centre and left the moons:
that fixed Neptune (478 -> 0.14 km at +5 yr) but broke the moons, whose relative states are fitted to the Horizons planet centre (Titania 963 -> 676,199 km, Ariel 859 -> 239,791 km, Charon 113 -> 414 km).

Repair used here: for each system (planet p and its moons i, GM from the fixture), the offset  D = x_bary(DE440s kernel) - sum(GM_j x_j)/sum(GM_j)  (position, and the same for velocity) is added to EVERY member.
Relative positions and velocities inside the system are unchanged to round-off, and the system barycentre equals DE440's.

usage: python scripts/shift_giant_systems.py [--out PATH]
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
FIX = _ROOT / "sigma_ground/field/interface/fixtures"
SEG = {"Mars": 4, "Jupiter": 5, "Saturn": 6, "Uranus": 7, "Neptune": 8, "Pluto": 9}
POS = ("x_km", "y_km", "z_km")
VEL = ("vx_km_s", "vy_km_s", "vz_km_s")


def shift_snapshot(snap: dict, kernel) -> dict:
    """Shift the systems of one snapshot in place; returns {planet: (position offset km, velocity offset mm/s)}."""
    jd = snap["epoch"]["jd_tdb"]
    by = {b["name"]: b for b in snap["bodies"]}
    anchor = {b["name"]: b.get("gravitational_anchor") for b in snap["bodies"]}
    out = {}
    for planet, seg in SEG.items():
        members = [planet] + [m for m, an in anchor.items() if an == planet]
        if len(members) == 1:
            continue
        gm = np.array([by[n]["gm_km3_s2"] for n in members])
        X = np.array([[by[n]["state_vector"][c] for c in POS] for n in members])
        V = np.array([[by[n]["state_vector"][c] for c in VEL] for n in members])
        pb, vb = kernel[0, seg].compute_and_differentiate(jd)
        dX = np.asarray(pb, dtype=float) - (gm[:, None] * X).sum(0) / gm.sum()
        dV = np.asarray(vb, dtype=float) / 86400.0 - (gm[:, None] * V).sum(0) / gm.sum()
        for n in members:
            sv = by[n]["state_vector"]
            for c, d in zip(POS, dX):
                sv[c] = float(sv[c] + d)
            for c, d in zip(VEL, dV):
                sv[c] = float(sv[c] + d)
        out[planet] = (float(np.linalg.norm(dX)), float(np.linalg.norm(dV) * 1e6))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=FIX / "de440_state_vectors_v2physics_r4.json")
    a = ap.parse_args()
    from jplephem.spk import SPK
    kernel = SPK.open("D:/datasets/jpl/de440s.bsp")
    new = copy.deepcopy(json.loads((FIX / "de440_state_vectors_v2physics.json").read_text()))
    shifts = {}
    for key, snap in new["snapshots"].items():
        if key.startswith("j"):
            for p, v in shift_snapshot(snap, kernel).items():
                shifts.setdefault(p, []).append(v)
    new.setdefault("metadata", {})["r4_rigid_shift"] = ("v2physics_r4_2026-10-17: each planet system (planet + its moons) moved rigidly onto DE440's system barycentre "
                                                        "(scripts/shift_giant_systems.py); relative states inside a system unchanged. r3 (planet-only) is superseded: it broke the moons.")
    a.out.write_text(json.dumps(new))
    for p, v in shifts.items():
        print(f"{p:8s} median rigid shift over {len(v)} snapshots: position {np.median([x[0] for x in v]):9.2f} km, velocity {np.median([x[1] for x in v]):8.3f} mm/s")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
