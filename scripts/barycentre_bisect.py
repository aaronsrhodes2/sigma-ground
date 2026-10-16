"""Which force layer moves a giant planet's SYSTEM BARYCENTRE away from DE440's in the full model?

Background (2026-10-16): in the full 33-body shoot-out the Neptune system barycentre (planet + Triton) is 480 km off DE440 at +5 yr and grows ~90 km/yr, while the reduced model (the
system as one point mass) matches DE440 to 0.2 km, and the planet-to-moon geometry in the full model is right to 0.05-0.2 km. So an internal-force layer is moving the barycentre.
This runs a small system (the Sun, the planets, and only the named moons) from one DE440 window under several toggle sets and prints the barycentre error against the DE440s kernel.

usage: python scripts/barycentre_bisect.py --out DIR [--window j2010] [--years 5] [--dt-days 0.2] [--moons Triton]
       (worker mode: --worker VARIANT OUT)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
FIXTURE = _ROOT / "sigma_ground/field/interface/fixtures/de440_state_vectors_v2physics.json"
SEG = {"Jupiter": (0, 5), "Saturn": (0, 6), "Uranus": (0, 7), "Neptune": (0, 8), "Pluto": (0, 9)}
VARIANTS = ["full", "no_zonals", "no_tides", "no_srp", "newton_eih_only", "newton_only"]


def run(variant: str, window: str, years: float, dt_days: float, moons: list[str]) -> dict:
    from jplephem.spk import SPK
    import sigma_ground.field.interface.rolling_shootout as rs
    from sigma_ground.field.interface.nbody import NBodySystem, PhysicsToggles
    fx = json.loads(FIXTURE.read_text())
    pred = [p for p in rs.canonical_predictors(dt_days) if p.name == "de440_stack"][0]
    bodies, names, sun_idx, parent = rs._build_bodies_at_snapshot(window, fx, pred)
    planets = ["Sun", "Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
    keep = [i for i, n in enumerate(names) if n in planets or n in moons]
    sub = [bodies[i] for i in keep]
    snames = [names[i] for i in keep]
    remap = {old: new for new, old in enumerate(keep)}
    sparent = [remap.get(parent[i]) if parent[i] in remap else None for i in keep]
    t = pred.toggles
    tog = {"full": t, "no_zonals": dataclasses.replace(t, j2_zonal=False, j3_zonal=False, j4_zonal=False), "no_tides": dataclasses.replace(t, tidal_force=False),
           "no_srp": dataclasses.replace(t, srp=False), "newton_eih_only": PhysicsToggles(eih_cross=True), "newton_only": PhysicsToggles()}[variant]
    system = NBodySystem(sub, toggles=tog, solar_luminosity_W=rs._L_SUN_W if tog.srp else 0.0, parent_attractor_indices=sparent)
    kernel = SPK.open("D:/datasets/jpl/de440s.bsp")
    start = fx["snapshots"][window]["epoch"]["jd_tdb"]
    gm = np.array([b.mass_kg * 6.6743e-11 / 1e9 for b in sub])
    out, t0 = {}, time.time()
    n_per_year = int(round(365.25 / dt_days))
    dts = 365.25 * 86400.0 / n_per_year
    sun_i = snames.index("Sun")
    for yr in range(1, int(years) + 1):
        for _ in range(n_per_year):
            system.forest_ruth_step(dts)
        jd = start + 365.25 * yr
        sun_true = np.asarray(kernel[0, 10].compute(jd), dtype=float)
        row = {}
        for p, (c, tt) in SEG.items():
            group = [i for i, n in enumerate(snames) if n == p or (parent_name(snames, sparent, i) == p)]
            M = gm[group]
            pos = np.array([system.bodies[i].position_m / 1e3 for i in group])
            bary = (M[:, None] * pos).sum(0) / M.sum()
            true = np.asarray(kernel[c, tt].compute(jd), dtype=float)
            row[p] = float(np.linalg.norm(bary - true))
        row["Sun"] = float(np.linalg.norm(system.bodies[sun_i].position_m / 1e3 - sun_true))
        out[str(yr)] = row
        print(f"[{variant}] year {yr}: " + " ".join(f"{k} {v:.1f}" for k, v in row.items()) + f"   ({time.time() - t0:.0f}s)", flush=True)
    return {"variant": variant, "moons": moons, "dt_days": dt_days, "barycentre_error_km": out}


def parent_name(names, sparent, i):
    p = sparent[i]
    return names[p] if p is not None else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--window", default="j2010")
    ap.add_argument("--years", type=float, default=5)
    ap.add_argument("--dt-days", type=float, default=0.2)
    ap.add_argument("--moons", nargs="*", default=["Triton"])
    ap.add_argument("--worker", nargs=2, metavar=("VARIANT", "OUT"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        Path(a.worker[1]).write_text(json.dumps(run(a.worker[0], a.window, a.years, a.dt_days, a.moons)))
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    procs = {}
    for v in VARIANTS:
        f = a.out / f"bisect_{v}.json"
        if not f.exists():
            procs[v] = subprocess.Popen([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", v, str(f), "--window", a.window, "--years", str(a.years),
                                         "--dt-days", str(a.dt_days), "--moons", *a.moons], stdout=open(a.out / f"bisect_{v}.log", "w"), stderr=subprocess.STDOUT)
    for v, p in procs.items():
        if p.wait() != 0:
            print(f"variant {v} FAILED (see {a.out / f'bisect_{v}.log'})")
            return 1
    lines = [f"Barycentre error (km) at +{int(a.years)} yr from {a.window}, step {a.dt_days} d, moons modelled: {', '.join(a.moons)}", "", "| variant | " + " | ".join(SEG) + " | Sun |", "|---|" + "---|" * (len(SEG) + 1)]
    for v in VARIANTS:
        d = json.loads((a.out / f"bisect_{v}.json").read_text())["barycentre_error_km"][str(int(a.years))]
        lines.append(f"| {v} | " + " | ".join(f"{d[k]:.1f}" for k in list(SEG) + ["Sun"]) + " |")
    text = "\n".join(lines)
    (a.out / "BARYCENTRE_BISECT.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
