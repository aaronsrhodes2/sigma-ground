"""Score every moon of one planet system in a SYSTEM-LOCAL model: Sun + the planet + all its moons, with the shoot-out's own integrator and a step set by the fastest moon.

Why: in the full 33+ body system one global step (0.0125 d) leaves moons with periods under about 4 days integrator-limited (Io 1,047 km at +5 yr, Phobos, Deimos, Mimas, Enceladus, Miranda and
20 of the 23 moons fetched from Horizons). A planet system only needs the Sun's tide, which a Sun + planet + moons system carries. The small system reproduces the full run where both exist
(Nereid +1 yr 402 vs 412 km, Callisto +5 yr 229 vs 203, Charon +5 yr 96 vs 92, Helene +1 yr 872 vs 872; scripts/slow_moon_diagnosis.py, MOON_DIAGNOSIS.md), and the planet's own path does not enter
because moons are scored RELATIVE to the planet. Not a replacement for the full system: other planets' tides on the moons are absent (they are 1e-3 of the Sun's, and the checks above include that).

The step is chosen as (shortest osculating period in the system) / steps-per-orbit; a convergence pair (--steps-per-orbit 150 and 300) shows whether it was fine enough.

usage: python scripts/system_local_moons.py --system Jupiter --windows j2010 j2015 --years 5 [--steps-per-orbit 150] [--zonals] [--fixture PATH] [--out FILE.json]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
FIXTURE = _ROOT / "sigma_ground/field/interface/fixtures/de440_state_vectors_v2physics_r7.json"
KEYS = ("x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s")


def system_members(snap: dict, planet: str) -> list[str]:
    return [planet] + [b["name"] for b in snap["bodies"] if b.get("gravitational_anchor") == planet]


def periods_days(snap: dict, planet: str, moons: list[str]) -> dict[str, float]:
    """Osculating period of each moon about the planet (two-body, GM of planet + moon) from the fixture state, days."""
    by = {b["name"]: b for b in snap["bodies"]}
    out = {}
    for m in moons:
        d = np.array([by[m]["state_vector"][k] - by[planet]["state_vector"][k] for k in KEYS])
        gm = by[planet]["gm_km3_s2"] + by[m]["gm_km3_s2"]
        r, v2 = np.linalg.norm(d[:3]), float(d[3:] @ d[3:])
        a = 1.0 / (2.0 / r - v2 / gm)
        out[m] = 2 * math.pi * math.sqrt(a ** 3 / gm) / 86400.0 if a > 0 else float("nan")
    return out


def run_system(planet: str, window: str, years: int, steps_per_orbit: float, zonals: bool, fix: dict, dt_override: float | None = None) -> dict:
    from sigma_ground.field.interface import rolling_shootout as rs
    base = [p for p in rs.canonical_predictors(0.0125) if p.name == "de440_stack"][0]
    snap0 = fix["snapshots"][window]
    members = system_members(snap0, planet)
    moons = members[1:]
    per = periods_days(snap0, planet, moons)
    dt = dt_override or min(0.0125, min(v for v in per.values() if v == v) / steps_per_orbit)
    pred = dataclasses.replace(base, dt_days=dt, zonal_overrides=rs._MEASURED_ZONALS if zonals else ())
    bodies, names, _, _ = rs._build_bodies_at_snapshot(window, fix, pred)
    keep = {"Sun", *members}
    sel = [i for i, n in enumerate(names) if n in keep]
    bodies = [bodies[i] for i in sel]
    names = [names[i] for i in sel]
    keys = sorted((k for k in fix["snapshots"] if k.startswith("j")), key=lambda k: int(k[1:]))
    k0 = keys.index(window)
    snaps = [fix["snapshots"][k] for k in keys[k0:k0 + years + 1]]
    jds = [float(s["epoch"]["jd_tdb"]) for s in snaps]
    t0 = time.time()
    out = rs._integrate_nbody(pred, bodies, names, names.index("Sun"), jds[0], jds, None)
    res = {}
    for m in moons:
        e = []
        for s, pm, pp in zip(snaps, out[m], out[planet]):
            b = {x["name"]: x["state_vector"] for x in s["bodies"]}
            truth = np.array([b[m][c] - b[planet][c] for c in KEYS[:3]])
            e.append(float(np.linalg.norm((np.asarray(pm) - np.asarray(pp)) - truth)))
        res[m] = e
    return {"system": planet, "window": window, "dt_days": dt, "steps_per_orbit": steps_per_orbit, "zonals": zonals, "period_days": per, "err_km": res, "wall_s": time.time() - t0, "bodies": names}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=["Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"])
    ap.add_argument("--windows", nargs="+", default=["j2010", "j2015"])
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--steps-per-orbit", type=float, default=150.0)
    ap.add_argument("--dt", type=float, default=None, help="override the step (days)")
    ap.add_argument("--zonals", action="store_true")
    ap.add_argument("--fixture", type=Path, default=FIXTURE)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    fix = json.loads(a.fixture.read_text())
    runs = []
    for w in a.windows:
        r = run_system(a.system, w, a.years, a.steps_per_orbit, a.zonals, fix, a.dt)
        runs.append(r)
        print(f"{a.system} {w} dt={r['dt_days']:.5f} d, {r['wall_s']:.0f} s", flush=True)
        for m, e in r["err_km"].items():
            print(f"  {m:11s} P={r['period_days'][m]:8.3f} d  " + " ".join(f"{x:10.2f}" for x in e), flush=True)
        if a.out:
            a.out.write_text(json.dumps(runs, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
