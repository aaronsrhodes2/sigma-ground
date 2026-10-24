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
sys.path.insert(0, str(_ROOT / "scripts"))
FIXTURE = _ROOT / "sigma_ground/field/interface/fixtures/de440_state_vectors_v2physics_r5.json"


def run(moon: str, planet: str, window: str, years: int, variant: str, fix: dict, keep_extra: tuple[str, ...], dv_m_s=None, vec: bool = False, dt_days: float | None = None, zonals: bool = False, dx_m=None):
    from sigma_ground.field.interface import rolling_shootout as rs
    base = [p for p in rs.canonical_predictors(0.0125) if p.name == "de440_stack"][0]
    pred = dataclasses.replace(base, zonal_overrides=rs._MEASURED_ZONALS) if zonals else base
    keep = {"Sun", planet, moon, *keep_extra}
    if variant == "no_zonals":
        pred = dataclasses.replace(pred, toggles=dataclasses.replace(base.toggles, j2_zonal=False, j3_zonal=False, j4_zonal=False))
    elif variant == "no_tide":
        pred = dataclasses.replace(pred, toggles=dataclasses.replace(base.toggles, tidal_force=False))
    elif variant == "no_srp":
        pred = dataclasses.replace(pred, toggles=dataclasses.replace(base.toggles, srp=False))
    elif variant == "no_gr":
        pred = dataclasses.replace(pred, toggles=dataclasses.replace(base.toggles, eih_cross=False, gr_1pn=False))
    elif variant == "newtonian":
        pred = dataclasses.replace(pred, toggles=dataclasses.replace(base.toggles, eih_cross=False, gr_1pn=False, srp=False, tidal_force=False))
    elif variant == "no_triton":
        keep.discard("Triton")
    elif variant == "fine":
        pred = dataclasses.replace(pred, dt_days=0.00625)
    elif variant == "coarse":
        pred = dataclasses.replace(pred, dt_days=0.025)
    elif variant == "measured_zonals":
        pred = dataclasses.replace(base, zonal_overrides=rs._MEASURED_ZONALS)
    elif variant.startswith(("moon_gm=", "gm_of_", "j2_of_")):
        pass                                                                    # applied to the body list below
    elif variant not in ("base", "vfix", "sat_jacobson"):
        raise ValueError(variant)
    bodies, names, sun_idx, parents = rs._build_bodies_at_snapshot(window, fix, pred)
    sel = [i for i, n in enumerate(names) if n in keep]
    bodies = [bodies[i] for i in sel]
    names = [names[i] for i in sel]
    if variant.startswith("moon_gm="):                                          # the moon's GM in km^3/s^2 (the planet's is untouched)
        i = names.index(moon)
        bodies[i] = dataclasses.replace(bodies[i], mass_kg=float(variant.split("=")[1]) * 1e9 / rs._G)
    if variant.startswith("gm_of_"):                                            # gm_of_<Body>=<km^3/s^2>
        who, val = variant[len("gm_of_"):].split("=")
        i = names.index(who)
        bodies[i] = dataclasses.replace(bodies[i], mass_kg=float(val) * 1e9 / rs._G)
    if variant.startswith("j2_of_"):                                            # j2_of_<Body>=<J2>, the reference radius is kept
        who, val = variant[len("j2_of_"):].split("=")
        i = names.index(who)
        bodies[i] = dataclasses.replace(bodies[i], j2=float(val))
    if variant == "vfix":
        # undo the system offset that scripts/extend_fixture_with_moons.py added to a state that Horizons already gave in the true frame (only its velocity part is non-zero)
        import extend_fixture_with_moons as ex
        from jplephem.spk import SPK
        orig = json.loads((ex.FIX / "de440_state_vectors_v2physics.json").read_text())["snapshots"][window]
        seg = {"Jupiter": 5, "Saturn": 6, "Uranus": 7, "Neptune": 8, "Pluto": 9}[planet]
        dX, dV = ex.system_offset(orig, planet, SPK.open("D:/datasets/jpl/de440s.bsp"), seg)
        i = names.index(moon)
        bodies[i] = dataclasses.replace(bodies[i], velocity_m_s=bodies[i].velocity_m_s - dV * 1e3)
    if variant == "sat_jacobson":
        # Saturn's J2 on the reference radius of the satellite ephemerides (Jacobson et al. 2006): J2 = 16290.71e-6 at R = 60330 km, instead of 1.6298e-2 at 60268 km
        i = names.index(planet)
        bodies[i] = dataclasses.replace(bodies[i], j2=16290.71e-6, radius_m=60330e3)
    if dv_m_s is not None:                                                      # a trial change of the moon's initial velocity (used by scripts/slow_moon_fit.py)
        i = names.index(moon)
        bodies[i] = dataclasses.replace(bodies[i], velocity_m_s=bodies[i].velocity_m_s + np.asarray(dv_m_s, dtype=float))
    if dx_m is not None:                                                        # a trial change of the moon's initial position (a frame-origin test)
        i = names.index(moon)
        bodies[i] = dataclasses.replace(bodies[i], position_m=bodies[i].position_m + np.asarray(dx_m, dtype=float))
    if dt_days is not None:
        pred = dataclasses.replace(pred, dt_days=dt_days)
    keys = sorted((k for k in fix["snapshots"] if k.startswith("j")), key=lambda k: int(k[1:]))
    k0 = keys.index(window)
    snaps = [fix["snapshots"][k] for k in keys[k0:k0 + years + 1]]
    jds = [float(s["epoch"]["jd_tdb"]) for s in snaps]
    out = rs._integrate_nbody(pred, bodies, names, names.index("Sun"), jds[0], jds, None)
    errs = []
    for s, pm, pp in zip(snaps, out[moon], out[planet]):
        b = {x["name"]: x["state_vector"] for x in s["bodies"]}
        truth = np.array([b[moon][c] - b[planet][c] for c in ("x_km", "y_km", "z_km")])
        d = (np.asarray(pm) - np.asarray(pp)) - truth
        errs.append(d if vec else float(np.linalg.norm(d)))
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moon", required=True)
    ap.add_argument("--planet", required=True)
    ap.add_argument("--windows", nargs="+", default=["j2010", "j2015"])
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--variants", nargs="+", default=["base"])
    ap.add_argument("--extra", nargs="*", default=[], help="other bodies to keep (e.g. Triton Dione)")
    ap.add_argument("--fixture", type=Path, default=FIXTURE, help="fixture to use (default r5; r6 has all 23 Horizons moons)")
    ap.add_argument("--dt", type=float, default=None, help="step in days (overrides the variant's)")
    ap.add_argument("--zonals", action="store_true", help="use the measured zonals (rolling_shootout._MEASURED_ZONALS) in every variant")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    fix = json.loads(a.fixture.read_text())
    res = []
    for w in a.windows:
        for v in a.variants:
            t = time.time()
            e = run(a.moon, a.planet, w, a.years, v, fix, tuple(a.extra), zonals=a.zonals, dt_days=a.dt)
            res.append({"moon": a.moon, "planet": a.planet, "window": w, "variant": v, "err_km": e})
            print(f"{a.moon:8s} {w} {v:10s} " + " ".join(f"{x:9.1f}" for x in e) + f"   [{time.time() - t:.0f}s]", flush=True)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
