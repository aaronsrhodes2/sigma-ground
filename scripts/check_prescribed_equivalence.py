"""Real-case check that PrescribedSources is a faithful replacement for membership: the seven heaviest DE440 asteroids as full N-body members (A) versus the same seven,
same masses, same initial states, replayed as prescribed sources (B), versus no asteroids (C).

The asteroid initial states and the prescribed paths come from the same pre-pass (perturber_paths.asteroid_window_paths: test particles under DE440's planets). In A the
asteroids then feel OUR planets (full EIH, co-integrated); in B they follow the pre-pass path. If the layer is faithful, B - A is a small fraction of A - C (what the asteroids do).

Full 33-body system, DE440 headline stack, window J2010. Both arms share the step, so the step's own error cancels in every difference.

usage: python scripts/check_prescribed_equivalence.py --out DIR [--years 2] [--dt-days 0.2] [--workers 3]
       (worker mode: --worker ARM OUT)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

SEVEN = (1, 4, 2, 10, 704, 16, 22)                       # Ceres, Vesta, Pallas, Hygiea, Interamnia, Psyche, Kalliope
WINDOW = "j2010"
FIXTURE = _ROOT / "sigma_ground/field/interface/fixtures/de440_state_vectors_v2physics.json"


def run_arm(arm: str, years: float, dt_days: float) -> dict:
    import dataclasses
    from jplephem.spk import SPK
    import sigma_ground.field.interface.rolling_shootout as rs
    from sigma_ground.field.constants import G
    from sigma_ground.field.interface import perturber_paths as pp
    from sigma_ground.field.interface.asteroid_perturbers import Perturber
    from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem

    fx = json.loads(FIXTURE.read_text())
    pred = [p for p in rs.canonical_predictors(dt_days) if p.name == "de440_stack"][0]
    bodies, names, sun_idx, parent = rs._build_bodies_at_snapshot(WINDOW, fx, pred)
    start = fx["snapshots"][WINDOW]["epoch"]["jd_tdb"]
    horizon = years * 365.25
    kernel = SPK.open(str(rs._DE440_KERNEL))
    gm = [b.mass_kg * G / 1e9 for b in bodies]
    idx = {n: i for i, n in enumerate(names)}
    sysgm = lambda n: gm[idx[n]] + sum(gm[j] for j, par in enumerate(parent) if par == idx[n])
    gm_planets = np.array([gm[idx["Sun"]]] + [sysgm(n) for n in ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")])
    cat = {a["number"]: a for a in json.loads(rs._PERTURBERS_FILE.read_text())["asteroids"]}
    pert = [Perturber(str(n), cat[n]["gm_km3_s2"], 0.0, *[cat[n]["elements"][k] for k in ("epoch_jd_tdb", "a_au", "e", "i_deg", "om_deg", "w_deg", "ma_deg")]) for n in SEVEN]
    nodes = pp.asteroid_window_paths(pert, kernel, gm_planets, start, horizon, 1.0)       # (T, 7, 6) km, km/day
    toggles, kwargs, members = pred.toggles, {}, list(bodies)
    if arm == "members":
        for k, n in enumerate(SEVEN):
            members.append(CelestialBody(mass_kg=cat[n]["gm_km3_s2"] * 1e9 / G, position_m=nodes[0, k, :3] * 1e3, velocity_m_s=nodes[0, k, 3:] * 1e3 / 86400.0, radius_m=1e5,
                                         love_number_k2=0.0, area_m2=0.0, reflectivity=0.0, j2=0.0, j3=0.0, j4=0.0, pole_axis_unit=np.array([0.0, 0.0, 1.0])))
        parent = list(parent) + [None] * len(SEVEN)
    elif arm == "prescribed":
        toggles = dataclasses.replace(toggles, prescribed_sources=True)
        kwargs["prescribed_sources"] = pp.to_prescribed_sources(nodes, np.array([cat[n]["gm_km3_s2"] for n in SEVEN]), 1.0)
    elif arm != "none":
        raise ValueError(arm)
    system = NBodySystem(members, toggles=toggles, solar_luminosity_W=rs._L_SUN_W if toggles.srp else 0.0, parent_attractor_indices=parent, **kwargs)
    out, t0 = {}, time.time()
    n_steps = int(round(horizon / dt_days))
    dts = horizon * 86400.0 / n_steps
    marks = {int(round(k * 365.25 / dt_days)): k for k in range(1, int(years) + 1)}
    for s in range(1, n_steps + 1):
        system.forest_ruth_step(dts)
        if s in marks:
            out[str(marks[s])] = {n: (system.bodies[i].position_m / 1e3).tolist() for i, n in enumerate(names)}
            print(f"[{arm}] year {marks[s]} done after {time.time() - t0:.0f}s", flush=True)
    return {"arm": arm, "years": years, "dt_days": dt_days, "positions_km": out}


def compare(res: dict) -> str:
    lines = ["| body | year | what the asteroids do (A - C), km | what the replay gets wrong (B - A), km | ratio |", "|---|---|---|---|---|"]
    worst = 0.0
    for yr in sorted(res["members"]["positions_km"]):
        for body in ("Sun", "Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn"):
            A, B, C = (np.array(res[k]["positions_km"][yr][body]) for k in ("members", "prescribed", "none"))
            eff, miss = np.linalg.norm(A - C), np.linalg.norm(B - A)
            lines.append(f"| {body} | {yr} | {eff:.4f} | {miss:.5f} | {miss / eff:.4f} |" if eff > 0 else f"| {body} | {yr} | 0 | {miss:.5f} | - |")
            if eff > 0.01:                                   # ignore bodies the asteroids hardly touch
                worst = max(worst, miss / eff)
    lines.append("")
    lines.append(f"Largest ratio over bodies the asteroids move by more than 10 m: {worst:.4f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--dt-days", type=float, default=0.2)
    ap.add_argument("--worker", nargs=2, metavar=("ARM", "OUT"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        r = run_arm(a.worker[0], a.years, a.dt_days)
        Path(a.worker[1]).write_text(json.dumps(r))
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    procs = {}
    for arm in ("members", "prescribed", "none"):
        f = a.out / f"equiv_{arm}.json"
        if not f.exists():
            procs[arm] = subprocess.Popen([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", arm, str(f), "--years", str(a.years), "--dt-days", str(a.dt_days)],
                                          stdout=open(a.out / f"equiv_{arm}.log", "w"), stderr=subprocess.STDOUT)
    for arm, p in procs.items():
        if p.wait() != 0:
            print(f"arm {arm} FAILED (see {a.out / f'equiv_{arm}.log'})")
            return 1
    res = {arm: json.loads((a.out / f"equiv_{arm}.json").read_text()) for arm in ("members", "prescribed", "none")}
    text = compare(res)
    (a.out / "EQUIVALENCE.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
