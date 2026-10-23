"""Diagnose a slow moon's drift with a small system: Sun + the moon's planet + the moon (+ chosen neighbours), run with the shoot-out's own integrator and the r5 fixture.

The moon's position RELATIVE to its planet is compared with the fixture's (Horizons) relative position at yearly epochs. The planet's own path does not matter for this (a Neptune error of
0.1 km is negligible against thousands), so the other planets are left out. Variants switch one thing at a time against the headline toggles, so the one that moves the drift is the cause.

usage: python scripts/slow_moon_diagnosis.py --moon Nereid --planet Neptune --windows j2010 j2015 --years 5 --variants base no_zonals no_tide no_triton fine [--out FILE.json]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
FIXTURE = _ROOT / "sigma_ground/field/interface/fixtures/de440_state_vectors_v2physics_r5.json"


def run(moon: str, planet: str, window: str, years: int, variant: str, fix: dict, keep_extra: tuple[str, ...]) -> list[float]:
    from sigma_ground.field.interface import rolling_shootout as rs
    base = [p for p in rs.canonical_predictors(0.0125) if p.name == "de440_stack"][0]
    pred = base
    keep = {"Sun", planet, moon, *keep_extra}
    if variant == "no_zonals":
        pred = dataclasses.replace(base, toggles=dataclasses.replace(base.toggles, j2_zonal=False, j3_zonal=False, j4_zonal=False))
    elif variant == "no_tide":
        pred = dataclasses.replace(base, toggles=dataclasses.replace(base.toggles, tidal_force=False))
    elif variant == "no_srp":
        pred = dataclasses.replace(base, toggles=dataclasses.replace(base.toggles, srp=False))
    elif variant == "no_triton":
        keep.discard("Triton")
    elif variant == "fine":
        pred = dataclasses.replace(base, dt_days=0.00625)
    elif variant == "coarse":
        pred = dataclasses.replace(base, dt_days=0.025)
    elif variant != "base":
        raise ValueError(variant)
    bodies, names, sun_idx, parents = rs._build_bodies_at_snapshot(window, fix, pred)
    sel = [i for i, n in enumerate(names) if n in keep]
    bodies = [bodies[i] for i in sel]
    names = [names[i] for i in sel]
    keys = sorted((k for k in fix["snapshots"] if k.startswith("j")), key=lambda k: int(k[1:]))
    k0 = keys.index(window)
    snaps = [fix["snapshots"][k] for k in keys[k0:k0 + years + 1]]
    jds = [float(s["epoch"]["jd_tdb"]) for s in snaps]
    out = rs._integrate_nbody(pred, bodies, names, names.index("Sun"), jds[0], jds, None)
    errs = []
    for s, pm, pp in zip(snaps, out[moon], out[planet]):
        b = {x["name"]: x["state_vector"] for x in s["bodies"]}
        truth = np.array([b[moon][c] - b[planet][c] for c in ("x_km", "y_km", "z_km")])
        errs.append(float(np.linalg.norm((np.asarray(pm) - np.asarray(pp)) - truth)))
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moon", required=True)
    ap.add_argument("--planet", required=True)
    ap.add_argument("--windows", nargs="+", default=["j2010", "j2015"])
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--variants", nargs="+", default=["base"])
    ap.add_argument("--extra", nargs="*", default=[], help="other bodies to keep (e.g. Triton Dione)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    fix = json.loads(FIXTURE.read_text())
    res = []
    for w in a.windows:
        for v in a.variants:
            t = time.time()
            e = run(a.moon, a.planet, w, a.years, v, fix, tuple(a.extra))
            res.append({"moon": a.moon, "planet": a.planet, "window": w, "variant": v, "err_km": e})
            print(f"{a.moon:8s} {w} {v:10s} " + " ".join(f"{x:9.1f}" for x in e) + f"   [{time.time() - t:.0f}s]", flush=True)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
