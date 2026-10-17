"""SUPERSEDED 2026-10-17 by scripts/shift_giant_systems.py (the r4 fixture). This planet-only repair (r3) fixed Neptune (478 -> 0.14 km at +5 yr) but left the moons at their old
states, and a moon's relative state is fitted to the Horizons planet centre: Titania 963 -> 676,199 km, Ariel 859 -> 239,791 km, Charon 113 -> 414 km. Kept as the record of the attempt.

Re-derive the planet-centre states of the planets that have modelled moons from DE440's own system barycentres (new fixture; the old one is never touched).

Problem found 2026-10-16: the fixture's planet-centre VELOCITIES (Horizons planet-centre ephemerides) are not consistent with DE440's system barycentre velocity and the fixture's
moon states: Uranus 5.3 mm/s, Neptune 2.9 mm/s, Pluto 0.29 mm/s, Jupiter 0.021 mm/s (Mars and Saturn < 0.002 mm/s). Over 5 yr that is 842, 453, 46 and 3.4 km, which are exactly the
'planet-centre floors' of the shoot-out. Started from the kernel's barycentres the same dynamics ends within 0.6-5 km (reduced and merged tests).

Definition used: planet centre = ( M_system * x_barycentre(DE440s kernel) - sum_i m_i x_i(moons, fixture) ) / m_planet, for position and velocity, with GM from the fixture; applied at EVERY snapshot,
so initial states and the later 'truth' are consistent. Moons, the Sun and the inner planets are copied unchanged.

usage: python scripts/rederive_giant_states.py [--out PATH]
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=FIX / "de440_state_vectors_v2physics_r3.json")
    a = ap.parse_args()
    from jplephem.spk import SPK
    kernel = SPK.open("D:/datasets/jpl/de440s.bsp")
    doc = json.loads((FIX / "de440_state_vectors_v2physics.json").read_text())
    new = copy.deepcopy(doc)
    shifts = {}
    for key, snap in new["snapshots"].items():
        if not key.startswith("j"):
            continue
        jd = snap["epoch"]["jd_tdb"]
        by = {b["name"]: b for b in snap["bodies"]}
        anchor = {b["name"]: b.get("gravitational_anchor") for b in snap["bodies"]}
        for planet, seg in SEG.items():
            moons = [m for m, an in anchor.items() if an == planet]
            if not moons:
                continue
            gmp = by[planet]["gm_km3_s2"]
            gms = np.array([by[m]["gm_km3_s2"] for m in moons])
            pb, vb = kernel[0, seg].compute_and_differentiate(jd)
            pb, vb = np.asarray(pb, dtype=float), np.asarray(vb, dtype=float) / 86400.0                 # km, km/s
            X = np.array([[by[m]["state_vector"][c] for c in ("x_km", "y_km", "z_km")] for m in moons])
            V = np.array([[by[m]["state_vector"][c] for c in ("vx_km_s", "vy_km_s", "vz_km_s")] for m in moons])
            Mtot = gmp + gms.sum()
            pos = (Mtot * pb - (gms[:, None] * X).sum(0)) / gmp
            vel = (Mtot * vb - (gms[:, None] * V).sum(0)) / gmp
            old = by[planet]["state_vector"]
            dv = np.linalg.norm(vel - np.array([old["vx_km_s"], old["vy_km_s"], old["vz_km_s"]])) * 1e6
            dp = np.linalg.norm(pos - np.array([old["x_km"], old["y_km"], old["z_km"]]))
            shifts.setdefault(planet, []).append((dp, dv))
            for c, v in zip(("x_km", "y_km", "z_km"), pos):
                old[c] = float(v)
            for c, v in zip(("vx_km_s", "vy_km_s", "vz_km_s"), vel):
                old[c] = float(v)
    new.setdefault("metadata", {})["r3_rederived"] = ("v2physics_r3_2026-10-16: planet-centre states of Mars, Jupiter, Saturn, Uranus, Neptune, Pluto re-derived from the DE440s system barycentres and the fixture's moon "
                                                        "states (scripts/rederive_giant_states.py); everything else unchanged. Past fixture de440_state_vectors_v2physics.json untouched.")
    a.out.write_text(json.dumps(new))
    for p, v in shifts.items():
        print(f"{p:8s} median change at the {len(v)} snapshots: position {np.median([x[0] for x in v]):9.2f} km, velocity {np.median([x[1] for x in v]):8.3f} mm/s")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
